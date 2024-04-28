import os
import logging
import contextlib

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from transformers import LlamaTokenizer, PreTrainedTokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from vxverse.common.dist_utils import download_cached_file
from vxverse.common.utils import get_abs_path, is_url
from vxverse.models.eva_vit import create_eva_vit_g
from vxverse.models.eva2_vit import create_model_and_transforms_for_eva
from vxverse.models.clip_vit import create_model_and_transforms_for_vit_2
from vxverse.models.modeling_llama import LlamaForCausalLM
from vxverse.models.modeling_xverse import XverseForCausalLM, new_xverse_forward

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['llama_proj', 'query_tokens', 'Qformer']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self, train_precision="fp16"):
        super().__init__()
        self.train_precision = train_precision

    @property
    def device(self):
        return list(self.parameters())[-1].device

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        print("Missing keys {}".format(msg.missing_keys))
        logging.info("Missing keys {}".format(msg.missing_keys))
        print("load checkpoint from %s" % url_or_filename)
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu") and self.train_precision == "fp16"

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_vision_encoder(
        cls, model_name, vit_path, img_size, drop_path_rate, use_grad_checkpoint, precision, freeze, train_precision="fp16",
    ):
        logging.info('Loading VIT')

        # visual_encoder = create_eva_vit_g(
        #     img_size, drop_path_rate, use_grad_checkpoint, precision
        # )
        # TODO
        # model_name = 'EVA01-CLIP-g-14'
        # pretrained = './eva01/eva_vit_g.pth'

        # model_name = 'EVA02-CLIP-bigE-14-224'
        # model_name = "EVA02-CLIP-bigE-14-336"
        # pretrained = './eva02/EVA02_CLIP_E_psz14_s4B.pt'

        if "eva" in model_name.lower():
            model, _, preprocess = create_model_and_transforms_for_eva(model_name, vit_path, force_custom_clip=True)
            visual_encoder = model.visual
            print('visual_encoder.num_features', visual_encoder.num_features) #
            logging.info('visual_encoder.num_features : {}'.format(visual_encoder.num_features))
            if train_precision == "bf16":
                ln_vision = nn.LayerNorm(visual_encoder.num_features)
            else:
                ln_vision = LayerNorm(visual_encoder.num_features)  # visual_encoder.num_features eva_clip_g(1408)  /  EVA02-CLIP-bigE-14(1792)
            if freeze:
                for name, param in visual_encoder.named_parameters():
                    param.requires_grad = False
                visual_encoder = visual_encoder.eval()
                visual_encoder.train = disabled_train
                for name, param in ln_vision.named_parameters():
                    param.requires_grad = False
                ln_vision = ln_vision.eval()
                ln_vision.train = disabled_train
                print("freeze vision encoder")
                logging.info("freeze vision encoder")
            print('Loading VIT Done')
            logging.info('Loading VIT Done')
            if train_precision == "bf16":
                visual_encoder.to(torch.bfloat16)
                ln_vision.to(torch.bfloat16)

            return visual_encoder, ln_vision
        elif "vit" in model_name.lower() and "eva" not in model_name.lower():
            visual_encoder = create_model_and_transforms_for_vit_2(model_name, vit_path)
            print('visual_encoder.num_features', visual_encoder.num_features)
            if train_precision == "bf16":
                ln_vision = nn.LayerNorm(visual_encoder.num_features)
            else:
                ln_vision = LayerNorm(visual_encoder.num_features)
            if freeze:
                for name, param in visual_encoder.named_parameters():
                    param.requires_grad = False
                visual_encoder = visual_encoder.eval()
                visual_encoder.train = disabled_train
                for name, param in ln_vision.named_parameters():
                    param.requires_grad = False
                ln_vision = ln_vision.eval()
                ln_vision.train = disabled_train
                print("freeze vision encoder")
                logging.info("freeze vision encoder")
            print('Loading VIT Done')
            logging.info('Loading VIT Done')
            if train_precision == "bf16":
                print(">>>>>>>> converting vit to bf16 ")
                visual_encoder.visual.to(torch.bfloat16)
                ln_vision.to(torch.bfloat16)

            return visual_encoder, ln_vision

    def init_llm(cls, llama_model_path, low_resource=False, freeze_llm=True, low_res_device=0, lora_r=0,
                 lora_target_modules=["q_proj","v_proj"], **lora_kargs):
        print('Loading LLAMA-like model from {}'.format(llama_model_path))
        logging.info('Loading LLAMA-like model from {}'.format(llama_model_path))

        # TODO
        if "xverse" in llama_model_path.lower(): # For XVERSE
            print("Seting pad_token_id to 1 for xverse...")
            ModelForCausalLM = XverseForCausalLM
            llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path)  # for xverse
            llama_tokenizer.pad_token = "<pad>"
            llama_tokenizer.pad_token_id = 1
            cls.llm_torch_dtype = torch.bfloat16

        else:
            ModelForCausalLM = LlamaForCausalLM
            llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path, use_fast=False)  # ori repo
            llama_tokenizer.pad_token = "$$"
            llama_tokenizer.add_tokens("<|endofknowledge|>")
            print(f">>>>>>>>>>>  Vicuna Tokenizer additional token:  <|endofknowledge|> id {llama_tokenizer.convert_tokens_to_ids(['<|endofknowledge|>'])}")
            cls.llm_torch_dtype = torch.float16

        if low_resource:
            print("low resource to load pretrained LLM...")
            logging.info("low resource to load pretrained LLM...")
            llama_model = ModelForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=cls.llm_torch_dtype,
                trust_remote_code=True,
                # low_cpu_mem_usage=True,
                device_map={'': low_res_device},
            )
            # if "xverse" in llama_model_path.lower():
            #     llama_model.forward = new_xverse_forward.__get__(llama_model, ModelForCausalLM)

        else:
            llama_model = ModelForCausalLM.from_pretrained(
                llama_model_path,
                trust_remote_code=True,
                torch_dtype=cls.llm_torch_dtype,
                # low_cpu_mem_usage=True,
                # device_map='auto'
            )
            # if "xverse" in llama_model_path:
            #     llama_model.forward = new_xverse_forward.__get__(llama_model, ModelForCausalLM)

        if lora_r > 0 and freeze_llm:
            print(f"Using Lora lora_r is {lora_r}...")
            # llama_model = prepare_model_for_int8_training(llama_model)
            for name, param in llama_model.named_parameters():
                param.requires_grad = False

            if lora_target_modules == "all_linear":
                print("Lora for all linear module")
                lora_target_modules = find_all_linear_names(llama_model)
                print("############ All linear modules ##############")
                print(lora_target_modules)


            loraconfig = LoraConfig(
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                **lora_kargs
            )
            llama_model = get_peft_model(llama_model, loraconfig)
            print("############   Lora parameters    #############")
            llama_model.print_trainable_parameters()
            print("###############################################")
        elif lora_r <= 0 and not freeze_llm:
            for name, param in llama_model.named_parameters():
                param.requires_grad = True
            print("Fully train LLM...")

        elif lora_r > 0 and not freeze_llm:
            print(f'#########  【Warning】 freeze_llm is set to {freeze_llm} and lora is enable ############# ')
            print(f'#########  【Warning】 It is disable lora and kepp full-finetune llm  ############# ')
            for name, param in llama_model.named_parameters():
                param.requires_grad = True
            print("Disable lora and fully train LLM...")

        else:
            for name, param in llama_model.named_parameters():
                param.requires_grad = False
            print("Freeze LLM...")

        print('Loading LLAMA-like model done...')
        logging.info('Loading LLAMA-like model done...')
        return llama_model, llama_tokenizer


    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        print("load checkpoint from %s" % url_or_filename)
        logging.info("load checkpoint from %s" % url_or_filename)
        msg = self.load_state_dict(state_dict, strict=False)
        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



