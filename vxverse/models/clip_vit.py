import torch.nn as nn
import torch
from transformers import CLIPVisionConfig, CLIPVisionModel, CLIPImageProcessor


class CLIP_VIT_MODEL(nn.Module):
    def __init__(self, model_name, vit_path=None, trainable=False, select_layer=-2, select_feature='patch'):
        super().__init__()
        self.vit_path = vit_path
        # self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.visual = CLIPVisionModel.from_pretrained(vit_path)
        self.num_features = self.visual.vision_model.config.hidden_size
        self.model_name = model_name
        self.trainable = trainable
        if not self.trainable:
            self.visual.requires_grad_(False)
        self.select_layer = select_layer
        self.select_feature = select_feature

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.visual(image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                                                      output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.visual(images.to(device=self.device, dtype=self.dtype),
                                                   output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.visual.dtype

    @property
    def device(self):
        return self.visual.device

    @property
    def config(self):
        if self.is_loaded:
            return self.visual.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def create_model_and_transforms_for_vit_2(model_name, vit_path):

    return CLIP_VIT_MODEL(model_name, vit_path)


