
import re

import torch

from vxverse.common.registry import registry
from vxverse.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from PIL import Image
import os
from torchvision.transforms.functional import InterpolationMode
from transformers import CLIPImageProcessor, CLIPVisionConfig

class BlipImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

class CLIPBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)


@registry.register_processor("hd_image_train")
class HDImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)
        self.image_size = image_size
        assert self.image_size == 224, f"In high resolution mode, image size must bet set as 224, but got {self.image_size}"
        self.window_size = (self.image_size, self.image_size)
        self.window_stride = self.image_size
        self.size_ratio_set = ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9),
                               (2, 1), (2, 2), (2, 3), (2, 4),
                               (3, 1), (3, 2), (3, 3),
                               (4, 1), (4, 2),
                               (5, 1), (6, 1), (7, 1), (8, 1), (9, 1))
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size,image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def resize_img(self, img, height, width):

        resize_fn = transforms.Compose(
            [
                transforms.Resize(
                    (height, width),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )
        return resize_fn(img)


    def sliding_window(self, image,  height_ratio, width_ratio):
        local_images = []
        for i in range(height_ratio):
            windows_col = []
            for j in range(width_ratio):
                window = image[:, i * self.window_stride: i * self.window_stride + self.image_size, j * self.window_stride: j * self.window_stride + self.image_size]
                if not torch.all(window == 0): # filter the pad patches
                    windows_col.append(window)
            local_images.extend(windows_col)
        return local_images

    def custom_round(self, value):
        if value - int(value) > 0.5:
            return int(value) + 1
        else:
            return int(value)

    def determine_resolution_v1(self, image):
        height, width = image.size
        height_ratio, width_ratio = max(1, self.custom_round(height / self.image_size)), max(1, self.custom_round(width / self.image_size))
        if width_ratio * height_ratio > 6:
            if width_ratio==1:
                image = self.resize_img(image, self.image_size*6, self.image_size*1)
            elif height_ratio==1:
                image = self.resize_img(image, self.image_size*1, self.image_size*6)
            elif width_ratio==2:
                image = self.resize_img(image, self.image_size*3, self.image_size*2)
            elif height_ratio==2:
                image = self.resize_img(image, self.image_size*2, self.image_size*3)
            else:
                image = self.resize_img(image, self.image_size*2, self.image_size*2)
        else:
            image = self.resize_img(image, self.image_size * height_ratio, self.image_size * width_ratio)
        height, width,  = image.shape[1], image.shape[2]
        height_ratio, width_ratio = max(1, self.custom_round(height / self.image_size)), max(1, self.custom_round(width / self.image_size))
        return image, height_ratio, width_ratio


    def determine_resolution_v2(self, image):
        height, width = image.size
        height_ratio, width_ratio = max(1, self.custom_round(height / self.image_size)), max(1, self.custom_round(
            width / self.image_size))
        if (height_ratio, width_ratio) in self.size_ratio_set:
            image = self.resize_img(image, self.image_size * height_ratio, self.image_size * width_ratio)
        else:
            if height_ratio == 1 or width_ratio == 1:
                height_ratio = 9 if width_ratio == 1 else height_ratio
                width_ratio = 9 if height_ratio == 1 else width_ratio
            elif height_ratio == 2 or width_ratio == 2:
                height_ratio = 4 if width_ratio == 2 else height_ratio
                width_ratio = 4 if height_ratio == 2 else width_ratio
            else:
                height_ratio, width_ratio = 3, 3
            image = self.resize_img(image, self.image_size * height_ratio, self.image_size * width_ratio)

        height, width, = image.shape[1], image.shape[2]
        height_ratio, width_ratio = max(1, self.custom_round(height / self.image_size)), max(1, self.custom_round(width / self.image_size))
        return image, height_ratio, width_ratio


    def __call__(self, image):
        global_image = self.transform(image)
        ret_imgs = [global_image]
        image, height_ratio, width_ratio = self.determine_resolution_v2(image)
        # image, height_ratio, width_ratio = self.determine_resolution_v1(image)
        if width_ratio != 1 or height_ratio != 1:
            local_images = self.sliding_window(image, height_ratio, width_ratio)
            ret_imgs.extend(local_images)
        ret_imgs = torch.stack(ret_imgs, dim=0)
        return ret_imgs

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )




@registry.register_processor("base_text_process")
class BaseTextProcessor(BaseProcessor):
    def __init__(self):
        self.transform = lambda x: x
        return

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)

        return self.from_config(cfg)


@registry.register_processor("blip_caption")
class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([.!\"()*#:;~])",
            " ",
            caption.lower(),
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


@registry.register_processor("blip2_image_train")
class Blip2ImageTrainProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None, min_scale=0.5, max_scale=1.0):
        super().__init__(mean=mean, std=std)
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size,image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
        )


@registry.register_processor("blip2_image_eval")
class Blip2ImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):
        super().__init__(mean=mean, std=std)
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)


@registry.register_processor("clip_image_processor")
class CLIPImageTrainProcessor(CLIPBaseProcessor):
    def __init__(self, config_path=None,  from_dict=None, image_aspect_ratio="pad", image_size=224,):
        super().__init__()
        self.image_aspect_ratio = image_aspect_ratio
        self.image_size = image_size
        self.config_path = config_path
        if config_path != None:
            if os.path.exists(config_path):
                self.processor = CLIPImageProcessor.from_pretrained(config_path)
            else:
                self.processor = CLIPImageProcessor.from_dict(from_dict)
        else:
            self.processor = CLIPImageProcessor.from_dict(from_dict)

    def transform(self, image):
        if self.image_aspect_ratio == 'pad':
            image = self.expand2square(image, tuple(int(x * 255) for x in self.processor.image_mean))
            image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image = self.processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        return image

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        config_path = cfg.get("config_path", None)
        image_size = cfg.get("image_size", 336)  # crop_size
        do_center_crop = cfg.get("do_center_crop", True)
        do_normalize = cfg.get("do_normalize", True)
        do_resize = cfg.get("do_resize", True)
        feature_extractor_type = cfg.get("feature_extractor_type", "CLIPFeatureExtractor") # CLIPFeatureExtractor
        resample = cfg.get("resample", 3)
        size = cfg.get("size", image_size)
        mean = cfg.get("mean", [0.48145466, 0.4578275, 0.40821073])
        std = cfg.get("std", [0.26862954, 0.26130258, 0.27577711])
        image_aspect_ratio = cfg.get("image_aspect_ratio", "pad")

        from_dict = {
            "crop_size": image_size,
            "do_center_crop": do_center_crop,
            "do_normalize": do_normalize,
            "do_resize": do_resize,
            "feature_extractor_type": feature_extractor_type,
            "resample": resample,
            "size": size,
            "mean": mean,
            "std": std,
        }
        return cls(config_path=config_path, from_dict=from_dict, image_size=image_size, image_aspect_ratio=image_aspect_ratio)


