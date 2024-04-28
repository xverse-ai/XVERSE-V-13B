import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from vxverse.common.registry import registry
from vxverse.models.base_model import disabled_train
from vxverse.models.vxverse_base import VXVERSEBase
from vxverse.models.Qformer import BertConfig, BertLMHeadModel
from vxverse.common.utils import get_abs_path, is_url

@registry.register_model("vxverse")
class VXVERSE(VXVERSEBase):
    """
    VXVERSE model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_xverse7b-chat": "configs/models/vxverse_7bchat.yaml",
        "pretrain_xverse13b-chat": "configs/models/vxverse_13bchat.yaml",
        "pretrain_xverse65b-chat": "configs/models/vxverse_65bchat.yaml",
    }

    Q_Former_Structure_CONFIG_DICT = {
        "bert-base-uncased": "configs/Qformer/bert-base-uncased",
    }

    def __init__(
            self,
            vit_model="EVA02-CLIP-bigE-14-224",
            vit_path="./eva02/EVA02_CLIP_E_psz14_s4B.pt",
            # q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            q_former_model="",
            img_size=224,
            drop_path_rate=0,
            use_grad_checkpoint=False,
            vit_precision="fp16",
            train_precision="fp16",
            freeze_vit=True,
            freeze_llm=True,
            has_qformer=True,
            n_proj_layers=1,
            freeze_qformer=True,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=128,
            max_context_len=800,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            lora_r=0,  # lora_r means lora is not used
            lora_target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0.1
    ):
        super().__init__(
            vit_model=vit_model,
            vit_path=vit_path,
            img_size=img_size,
            train_precision=train_precision,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_llm=freeze_llm,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,  # lora_r means lora is not used
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        self.has_qformer = has_qformer
        self.n_proj_layers = n_proj_layers
        self.train_precision = train_precision
        if self.has_qformer:
            print('Loading Q-Former')
            logging.info('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features, freeze_qformer
            )
            # TODO
            # delete or not?
            if q_former_model != "" :
                print("Loading QFormer weight from pretrained weight...")
                self.load_from_pretrained(url_or_filename=q_former_model)  # load q-former weights here
            else:
                print("Initial QFormer weight randomly and do not load from pretrained when constructing it...")

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
            logging.info('Loading Q-Former Done')
            self.llama_proj = nn.Linear(
                img_f_dim, self.llama_model.config.hidden_size
            )
        else:
            print('Do not use Q-Former here.')
            logging.info('Do not use Q-Former here.')
            img_f_dim = self.visual_encoder.num_features

            llama_hidden_size = self.llama_model.config.hidden_size
            modules = [nn.Linear(img_f_dim, llama_hidden_size)]
            print(f">>>>> img_f_dim: {img_f_dim}, llama_hidden_size: {llama_hidden_size}")
            print(f">>>>> n_proj_layers: {self.n_proj_layers}")
            for _ in range(1, self.n_proj_layers):
                modules.append(nn.GELU())
                modules.append(nn.Linear(llama_hidden_size, llama_hidden_size))
            self.llama_proj = nn.Sequential(*modules)

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
            logging.info('Load {} training prompts'.format(len(self.prompt_list)))
            logging.info('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, freeze):

        # TODO
        qformer_struct_config = get_abs_path(cls.Q_Former_Structure_CONFIG_DICT['bert-base-uncased'])
        encoder_config = BertConfig.from_pretrained(qformer_struct_config)
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer = Qformer.eval()
            Qformer.train = disabled_train
            query_tokens.requires_grad = False
            print("freeze Qformer")
            logging.info("freeze Qformer")

        return Qformer, query_tokens

    def encode_img(self, image):
        if type(image) == torch.Tensor:
            if self.train_precision == "bf16":
                image = image.to(torch.bfloat16)
            device = image.device
            with self.maybe_autocast():
                image_embeds = self.visual_encoder(image)
                image_embeds = self.ln_vision(image_embeds).to(device)
                if self.has_qformer:
                    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                    query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )
                    inputs_llama = self.llama_proj(
                        query_output.last_hidden_state)
                else:
                    inputs_llama = self.llama_proj(image_embeds)
                atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
            return inputs_llama, atts_llama

        elif type(image) == list:
            inputs_llama_lists, atts_llama_lists = [], []
            for per_imgs in image:
                if self.train_precision=="bf16":
                    per_imgs = per_imgs.to(torch.bfloat16)
                device = per_imgs.device
                with self.maybe_autocast():
                    image_embeds = self.visual_encoder(per_imgs)
                    image_embeds = self.ln_vision(image_embeds).to(device)
                    if self.has_qformer:
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
                        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                        query_output = self.Qformer.bert(
                            query_embeds=query_tokens,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )
                        inputs_llama = self.llama_proj(query_output.last_hidden_state)
                    else:
                        inputs_llama = self.llama_proj(image_embeds)
                    atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
                    inputs_llama_lists.append(inputs_llama)
                    atts_llama_lists.append(atts_llama)
            return inputs_llama_lists, atts_llama_lists

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "EVA02-CLIP-bigE-14-224")
        vit_path = cfg.get("vit_path", "./eva02/EVA02_CLIP_E_psz14_s4B.pt")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        train_precision = cfg.get("train_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_llm = cfg.get("freeze_llm", True)
        has_qformer = cfg.get("has_qformer", True)
        n_proj_layers = cfg.get("n_proj_layers", 1)
        freeze_qformer = cfg.get("freeze_qformer", False)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        max_context_len = cfg.get("max_context_len", 800)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 0)
        lora_alpha = cfg.get("lora_alpha", 16)
        lora_dropout = cfg.get("lora_dropout", 0.01)
        lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "v_proj"])

        # 检查是否支持bf16
        if train_precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 is not supported on your GPU.")
        print(f"train_precision is : {train_precision}")

        model = cls(
            vit_model=vit_model,
            vit_path=vit_path,
            train_precision=train_precision,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_llm=freeze_llm,
            has_qformer=has_qformer,
            n_proj_layers=n_proj_layers,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of Vxverse
        if ckpt_path:
            print("Loading Visual-Xverse Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        if train_precision == 'bf16':
            model.to(torch.bfloat16)

        return model
