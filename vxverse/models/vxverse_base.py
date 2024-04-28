import logging
import random
from itertools import groupby
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from vxverse.common.registry import registry
from vxverse.models.base_model import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList

from vxverse.conversation.conversation import StoppingCriteriaSub

IGNORE_INDEX = -100
XVERSE_IMAGE_INDEX = 5
VICUNA_IMAGE_INDEX = 32000

class VXVERSEBase(BaseModel):
    """
    Base class for VXVERSE
    """

    def __init__(
        self,
        vit_model="EVA02-CLIP-bigE-14-224",
        vit_path="./eva02/EVA02_CLIP_E_psz14_s4B.pt",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        train_precision="fp16",
        freeze_vit=True,
        freeze_llm=True,
        llama_model="",
        max_txt_len=128,
        max_context_len=3800,
        prompt_template="",
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        lora_r=0,  # lora_r means lora is not used
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.1,
    ):
        super().__init__(train_precision=train_precision)
        self.train_precision = train_precision
        self.llama_model_path = llama_model
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, vit_path, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, freeze_vit, train_precision=self.train_precision,
        )
        self.llama_model, self.llama_tokenizer = self.init_llm(
            llama_model_path=llama_model,
            low_resource=low_resource,
            freeze_llm=freeze_llm,
            low_res_device=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        self.max_txt_len = max_txt_len
        self.max_context_len = max_context_len
        self.end_sym = end_sym

        self.prompt_template = prompt_template
        self.prompt_list = []

        self.print_prompt_once = True

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def get_context_emb(self, prompt, img_list, patches_per_image=None, device='cuda:0'):
        if img_list == None or len(img_list) == 0:  # TO support pure text input
            prompt_tokens = self.llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device).input_ids
            prompt_embs = self.embed_tokens(prompt_tokens)
            return prompt_embs
        else:
            device = img_list[0].device
            prompt_segs = prompt.split('<ImageHere>')
            if patches_per_image==None:
                assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
            else: # for high-definition
                assert len(prompt_segs) == len(patches_per_image) + 1, "Unmatched numbers of image placeholders and images."
            seg_tokens = [
                self.llama_tokenizer(
                    seg, return_tensors="pt", add_special_tokens=i==0).to(device).input_ids # only add bos to the first seg
                for i, seg in enumerate(prompt_segs)
            ]
            seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]
            if patches_per_image==None:
                mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
            else: # for high-definition
                mixed_embs = []
                pn = img_list.shape[1]
                img_list = img_list.view(-1, img_list.shape[-1])
                cur_patches = 0
                for seg_emb, patches in zip(seg_embs[:-1], patches_per_image):
                    mixed_embs.append(seg_emb)
                    mixed_embs.append(img_list[cur_patches * pn: (patches + cur_patches)* pn , :].unsqueeze(0))
                    cur_patches = patches
                mixed_embs.append(seg_embs[-1])
            mixed_embs = torch.cat(mixed_embs, dim=1)
            return mixed_embs

    def prompt_wrap_v2(self, img_embeds, input_ids, labels, attention_masks, texts=None, patches_per_images=None, total_images=None):
        # In this version img_embeds and input_ids must not be None simultaneously

        IMAGE_INDEX = XVERSE_IMAGE_INDEX if "xverse" in self.llama_model_path.lower() else VICUNA_IMAGE_INDEX

        if img_embeds == None:
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
            emb_tensor = self.embed_tokens(input_ids)
            labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
            attention_masks_tensor = torch.tensor(attention_masks, dtype=torch.long, device=self.device)
        else:
            emb_lists = []
            labels_list = []
            attention_masks_list = []
            # feature patches of each image
            pn = img_embeds[0].shape[-2]
            device = img_embeds[0].device
            max_imgs = 1
            if patches_per_images != None:
                # # TODO to be compatible with high definition
                max_imgs = max([sum(x) for x in patches_per_images])
            else:
                for each_img_embed in img_embeds:
                    max_imgs = max(max_imgs, each_img_embed.shape[0])
                patches_per_images = [[1]*x.shape[0] for x in img_embeds]

            for idx, (each_img_embed, each_input_ids, each_labels, each_attention_masks, patches) in enumerate(zip(img_embeds, input_ids, labels, attention_masks, patches_per_images)):

                if max_imgs > 1: # for multiple images per sample or multiple patches per image
                    # TO keep consistency in the dimensions of samples with different numbers of images
                    # TODO: This operation may cause the length of sequence beyond the setting (i.e 1024 in hyperparameter, but result in 1024+(num_img_per_sample-num_img)*pn)
                    pad_id = self.llama_tokenizer.pad_token_id

                    # TODO to be compatible with high definition
                    if total_images != None:  # Don't use (if patches_per_images != None:) because it may be convert to (patches_per_images = [[1]*x.shape[0] for x in img_embeds])
                        num_img = sum(patches)
                    else:
                        num_img = each_input_ids.count(IMAGE_INDEX)

                    if max_imgs-num_img > 0:
                        each_input_ids.append([pad_id]*(max_imgs-num_img)*pn)
                        labels.append([IGNORE_INDEX]*(max_imgs-num_img)*pn)
                        attention_masks.append([0]*(max_imgs-num_img)*pn)

                # for pure-text sample
                if IMAGE_INDEX not in each_input_ids:
                    each_input_ids = torch.tensor(each_input_ids, dtype=torch.long, device=self.device)
                    wrapped_emb = self.embed_tokens(each_input_ids)
                    emb_lists.append(wrapped_emb)
                    labels_list.append(torch.tensor(each_labels, dtype=torch.long, device=device))
                    attention_masks_list.append(each_attention_masks)
                # for multi-modal sample
                else:
                    each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])  # [n_images, tokens, dim]
                    p_segs = [list(group) for key, group in groupby(each_input_ids, lambda x: x == IMAGE_INDEX) if not key]
                    img_indices = [i for i, x in enumerate(each_input_ids) if x == IMAGE_INDEX]
                    labels_segs = [each_labels[:img_indices[0]]]
                    for i in range(len(img_indices)-1):
                        labels_segs.append(each_labels[img_indices[i]+1:img_indices[i+1]])
                    labels_segs.append(each_labels[img_indices[-1]+1:])
                    interleave_emb = []
                    interleave_labels = []
                    cur_patches = 0
                    for idx, seg in enumerate(p_segs[:-1]):
                        next_patches = patches[idx]

                        p_tokens = torch.tensor(seg, dtype=torch.long).to(device)
                        p_embed = self.embed_tokens(p_tokens)

                        # TODO to be compatible with high definition
                        interleave_emb.append(torch.cat([p_embed, each_img_embed[cur_patches * pn: (next_patches + cur_patches)* pn , :]], dim=0))

                        interleave_labels.extend(labels_segs[idx]+[IGNORE_INDEX]*next_patches*pn)
                        each_attention_masks = [1]*(next_patches*pn-1) + each_attention_masks
                        cur_patches = next_patches

                    wrapped_emb = torch.cat(interleave_emb, dim=0)
                    p_tokens = torch.tensor(p_segs[-1], dtype=torch.long).to(device)
                    p_embed = self.embed_tokens(p_tokens)
                    wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=0)
                    interleave_labels.extend(labels_segs[-1])
                    emb_lists.append(wrapped_emb)
                    labels_list.append(torch.tensor(interleave_labels, dtype=torch.long, device=device))
                    attention_masks_list.append(each_attention_masks)

            attention_masks_tensor = torch.tensor(attention_masks_list, dtype=torch.long, device=device)
            emb_tensor = torch.stack(emb_lists, dim=0)
            labels_tensor = torch.stack(labels_list, dim=0)
            assert labels_tensor.shape[0] == attention_masks_tensor.shape[0] == emb_tensor.shape[0] and \
                   labels_tensor.shape[1] == attention_masks_tensor.shape[1] == emb_tensor.shape[1], \
                f"shape not match: labels shape: {labels_tensor.shape}, attention shape: {attention_masks_tensor.shape}, emb shape: {emb_tensor.shape} \n\n batch texts: {texts}"

        return emb_tensor, labels_tensor, attention_masks_tensor

    def prompt_wrap(self, img_embeds, atts_img, prompts, lengths=None):
        if prompts is None or len(prompts) == 0:
            # prompts is not provided, just return the original image embedding
            return img_embeds, atts_img
        elif img_embeds is None:
            # prompt is provided but there is no image embedding. return the prompt embedding in right padding
            self.llama_tokenizer.padding_side = "right"
            prompt_tokens = self.llama_tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(self.device)
            prompt_tokens.input_ids = prompt_tokens.input_ids.to(torch.long)
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)
            atts_prompt = prompt_tokens.attention_mask
            return prompt_embeds, atts_prompt
        else:
            # return the multi-modal embedding in right padding
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)

            for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds, prompts)):
                pn = each_img_embed.shape[-2]
                if lengths is not None:
                    each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])
                    each_img_embed = each_img_embed[:lengths[idx] * pn]
                p_segs = each_prompt.split('<ImageHere>')
                interleave_emb = []
                for idx, seg in enumerate(p_segs[:-1]):
                    p_tokens = self.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                    p_tokens.input_ids = p_tokens.input_ids.to(torch.long)
                    p_embed = self.embed_tokens(p_tokens.input_ids)
                    interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, idx * pn:(idx + 1) * pn]], dim=1))
                wrapped_emb = torch.cat(interleave_emb, dim=1)
                p_tokens = self.llama_tokenizer(
                    p_segs[-1], return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
                p_tokens.input_ids = p_tokens.input_ids.to(torch.long)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
                emb_lists.append(wrapped_emb)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device, dtype=torch.long))

            max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len
            wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=img_embeds.device)
            
            for i, emb in enumerate(emb_lists):
                length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len
                wrapped_embs[i, :length] = emb[:, :length]
                wrapped_atts[i, :length] = 1
            return wrapped_embs, wrapped_atts

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        """
        Concatenate the batched input embedding and batched output embedding together.
        Both the input and the output embedding should be right padded.
        """
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def tokenize_conversation(self, conv_q, conv_a):
        """concatenate conversation and make sure the model is only trained to regress the answer"""

        to_regress_token_ids_list = []
        targets_list = []

        batch_size = len(conv_q)
        for batch_idx in range(batch_size):
            questions, answers = conv_q[batch_idx], conv_a[batch_idx]
            questions = [self.llama_tokenizer(self.llama_tokenizer.bos_token + q,
                                              return_tensors="pt",
                                              add_special_tokens=False).to(self.device) for q in questions[1:]]  # the first question is handled in the prompt wrap function, skip it
            answers = [self.llama_tokenizer(a + self.end_sym,
                                            return_tensors="pt",
                                            add_special_tokens=False).to(self.device) for a in answers]
            cur_id = []
            cur_target = []
            for i in range(len(questions)):
                cur_id.append(answers[i].input_ids)
                cur_target.append(answers[i].input_ids)
                cur_id.append(questions[i].input_ids)
                cur_target.append(torch.ones_like(questions[i].input_ids) * -100)

            cur_id.append(answers[-1].input_ids)
            cur_target.append(answers[-1].input_ids)

            cur_id = torch.cat(cur_id, dim=1)
            cur_target = torch.cat(cur_target, dim=1)
            to_regress_token_ids_list.append(cur_id)
            targets_list.append(cur_target)

        max_len = min(max([target.shape[1] for target in targets_list]), self.max_txt_len)
        to_regress_token_ids = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * self.llama_tokenizer.pad_token_id
        targets = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * -100
        for batch_idx in range(batch_size):
            cur_len = to_regress_token_ids_list[batch_idx].shape[1]
            to_regress_token_ids[batch_idx, :cur_len] = to_regress_token_ids_list[batch_idx][0, :max_len]
            targets[batch_idx, :cur_len] = targets_list[batch_idx][0, :max_len]

        to_regress_token_attn = (to_regress_token_ids != self.llama_tokenizer.pad_token_id).to(torch.int)

        return to_regress_token_ids, to_regress_token_attn, targets

    def preparing_embedding_v2(self, samples):
        if 'image' in samples:
            img_embeds, img_atts = self.encode_img(samples["image"])
        else:
            img_embeds = img_atts = None

        emb_tensor, labels_tensor, attention_masks_tensor = self.prompt_wrap_v2(img_embeds,
                                                                                samples["input_ids"], samples["labels"],
                                                                                samples["attention_masks"], samples["texts"],
                                                                                samples.get("patches_per_image", None), samples.get("total_images", None))
        if self.print_prompt_once:
            print("######## emb_tensor ##########")
            print(emb_tensor)
            print("######## emb_tensor shape ##########")
            print(emb_tensor.shape)
            print("######## labels_tensor ##########")
            print(labels_tensor)
            print("######## labels_tensor shape ##########")
            print(labels_tensor.shape)
            print("######## attention_masks_tensor ##########")
            print(attention_masks_tensor)
            print("######## attention_masks_tensor shape ##########")
            print(attention_masks_tensor.shape)
            self.print_prompt_once = False

        return emb_tensor, labels_tensor, attention_masks_tensor


    def preparing_embedding(self, samples):
        ### prepare input tokens
        if 'image' in samples:
            img_embeds, img_atts = self.encode_img(samples["image"])
        else:
            img_embeds = img_atts = None

        if 'conv_q' in samples:
            # handeling conversation datasets
            conv_q, conv_a = samples['conv_q'], samples['conv_a']

            connect_sym = samples['connect_sym'][0]
            conv_q = [q.split(connect_sym)for q in conv_q]
            conv_a = [a.split(connect_sym) for a in conv_a]

            conv_q = [[self.prompt_template.format(item) for item in items] for items in conv_q]

            cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, [q[0] for q in conv_q])
            regress_token_ids, regress_atts, part_targets = self.tokenize_conversation(conv_q, conv_a)

        else:
            if "instruction_input" in samples:
                instruction = samples["instruction_input"]

            elif self.prompt_list:
                instruction = random.choice(self.prompt_list)
            else:
                instruction = None

            if hasattr(self, 'chat_template') and self.chat_template:
                instruction = [self.prompt_template.format(instruct) for instruct in instruction]

            if 'length' in samples:
                # the input is a image train (like videos)
                bsz, pn, hs = img_embeds.shape
                img_embeds = img_embeds.reshape(len(samples['image']), -1, pn, hs)
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction, samples['length'])
            else:
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction)

            ### prepare target tokens
            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["answer"]]
            if self.print_prompt_once :
                print("######## text ##########")
                print(text)
                print("######## Prompt ########")
                print(instruction)
                print("######## self.end_sym ##########")
                print(self.end_sym)
                print("######## cond_embeds dtype ##########")
                print(f"》》》》》 Inference: preparing_embedding(): self.prompt_wrap(): cond_embeds.dtype {cond_embeds.dtype}") # torch.float16
                self.print_prompt_once = False

            regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(self.device)

            regress_token_ids = regress_tokens.input_ids
            regress_atts = regress_tokens.attention_mask
            part_targets = regress_token_ids.masked_fill(
                regress_token_ids == self.llama_tokenizer.pad_token_id, -100
            )
        regress_token_ids = regress_token_ids.to(torch.long)
        regress_embeds = self.embed_tokens(regress_token_ids)

        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets

    def forward(self, samples, reduction='mean'):

        if "input_ids" not in samples:
            # prepare the embedding to condition and the embedding to regress
            cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = \
                self.preparing_embedding(samples)
            # concat the embedding to condition and the embedding to regress
            inputs_embeds, attention_mask, input_lens = \
                self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)
            # get bos token embedding
            bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
            bos = bos.to(torch.long)
            bos_embeds = self.embed_tokens(bos)
            bos_atts = cond_atts[:, :1]

            # add bos token at the begining
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
            attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

            # ensemble the final targets
            targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                                 dtype=torch.long).to(self.device).fill_(-100)

            for i, target in enumerate(part_targets):
                targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos

        else:  # stage3_datasets_v2 will go into this branch
            inputs_embeds, targets, attention_mask = self.preparing_embedding_v2(samples)

        with self.maybe_autocast(self.llm_torch_dtype):   # llama_model hidden_states dtype: torch.float32
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                # reduction=reduction
            )
        loss = outputs.loss
        return {"loss": loss}

    def embed_tokens(self, token_ids):
        token_ids = token_ids.to(torch.long)
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    @torch.no_grad()
    def generate(
        self,
        images,
        texts,
        patches_per_images=None,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        top_k=30,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
    ):
        '''
            function for generate test use
        '''
        if type(images) == torch.Tensor:
            img_embeds, atts_img = self.encode_img(images.to(self.device))
            image_lists = [[image_emb.unsqueeze(0)] for image_emb in img_embeds]
        elif type(images) == list:
            images2 = []
            for image in images:
                images2.append(image.to(self.device))
            image_lists, atts_img = self.encode_img(images2)
        else: raise ValueError
        if patches_per_images == None:
            patches_per_images = [None]*len(texts)
        batch_embs = [self.get_context_emb(text, img_list, patches_per_image) for text, img_list, patches_per_image in zip(texts, image_lists, patches_per_images)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]   # batch inference left-padding
            attn_mask[i, -emb_len:] = 1
        with self.maybe_autocast(self.llm_torch_dtype):
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                # stopping_criteria=stopping_criteria,
            )

        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split("<|endoftext|>")[0].strip()  # remove the stop sign
            answers.append(output_texts)

        return answers

    @torch.no_grad()
    def multi_select(self, images, texts, answers, num_cand=None):
        all_losses = []
        for answer in answers:
            choice_samples = {
                'image': images,
                'instruction_input': texts,
                'answer': answer
            }
            loss = self.forward(choice_samples, reduction='none')['loss'].reshape(-1, 1)
            all_losses.append(loss)
            torch.cuda.empty_cache()
        all_losses = torch.cat(all_losses, dim=-1)
        if num_cand is not None:
            for i in range(all_losses.shape[0]):
                all_losses[i, num_cand[i]:] = 9999
        output_class_ranks = torch.argsort(all_losses, dim=-1)
        return output_class_ranks.tolist()
