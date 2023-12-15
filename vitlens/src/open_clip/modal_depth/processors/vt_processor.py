import torch
import random
import numpy as np
import re

from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
import open_clip.modal_depth.processors.transforms_rgbd as rgbd_T

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

rgbd_conf = OmegaConf.create(
    {
        "max_depth": 75,
        "clamp_max_before_scale": True,
        "num_ops": 2,
        "magnitude": 9,
    }
)


class BaseProcessor:
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


class BlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        if caption is None:
            return None

        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 70)

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


class RGBD_Processor_Train(BaseProcessor):
    def __init__(
        self,
        args,
        img_mean=None,
        img_std=None,
        depth_mean=None,
        depth_std=None,
        update_conf=None,
    ):
        img_mean = img_mean if img_mean is not None else OPENAI_CLIP_MEAN
        img_std = img_std if img_std is not None else OPENAI_CLIP_STD
        depth_mean = depth_mean if depth_mean is not None else 0.0
        depth_std = depth_std if depth_std is not None else 1.0

        self.mean = list(img_mean) + [depth_mean]
        self.std = list(img_std) + [depth_std]

        if update_conf is not None:
            args.update(update_conf)

        self.img_to_tensor = transforms.ToTensor()

        self.rgbd_transform = transforms.Compose(
            [
                rgbd_T.DepthNorm(
                    max_depth=args.max_depth,
                    clamp_max_before_scale=args.clamp_max_before_scale,
                ),
                transforms.RandomResizedCrop(size=224, interpolation=2),
                transforms.RandomHorizontalFlip(),
                rgbd_T.RandAugment3d(
                    num_ops=args.num_ops,
                    magnitude=args.magnitude,
                    interpolation=2,
                ),
                rgbd_T.ColorJitter3d(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4
                ),
                transforms.RandomErasing(p=0.25),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, img_path, depth_path):
        # here depth refers to disparity, in torch savefile format
        # note use ToTensor to scale image to [0,1] first
        img = Image.open(img_path).convert("RGB")
        img = self.img_to_tensor(img)

        depth = torch.load(depth_path)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)

        rgbd = torch.cat([img, depth], dim=0)
        transform_rgbd = self.rgbd_transform(rgbd)
        img = transform_rgbd[0:3, ...]
        depth = transform_rgbd[3:4, ...]

        return img, depth

    def __repr__(self):
        repr = "(DataAugmentationForRGBD,\n"
        repr += "transform = %s,\n" % str(self.rgbd_transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", rgbd_conf)
        img_mean = cfg.get("img_mean", OPENAI_CLIP_MEAN)
        img_std = cfg.get("img_std", OPENAI_CLIP_STD)
        depth_mean = cfg.get("depth_mean", 0.0418)
        depth_std = cfg.get("depth_std", 0.0295)

        return cls(
            args=args,
            img_mean=img_mean,
            img_std=img_std,
            depth_mean=depth_mean,
            depth_std=depth_std,
            update_conf=update_conf,
        )

    def __repr__(self):
        repr = "(DataAugmentationForRGBD,\n"
        repr += "transform = %s,\n" % str(self.rgbd_transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", rgbd_conf)
        img_mean = cfg.get("img_mean", OPENAI_CLIP_MEAN)
        img_std = cfg.get("img_std", OPENAI_CLIP_STD)
        depth_mean = cfg.get("depth_mean", 0.0418)
        depth_std = cfg.get("depth_std", 0.0295)

        return cls(
            args=args,
            img_mean=img_mean,
            img_std=img_std,
            depth_mean=depth_mean,
            depth_std=depth_std,
            update_conf=update_conf,
        )


class RGBD_Processor_Eval(BaseProcessor):
    def __init__(
        self,
        args,
        img_mean=None,
        img_std=None,
        depth_mean=None,
        depth_std=None,
        update_conf=None,
    ):
        img_mean = img_mean if img_mean is not None else OPENAI_CLIP_MEAN
        img_std = img_std if img_std is not None else OPENAI_CLIP_STD
        depth_mean = depth_mean if depth_mean is not None else 0.0
        depth_std = depth_std if depth_std is not None else 1.0

        self.mean = list(img_mean) + [depth_mean]
        self.std = list(img_std) + [depth_std]

        if update_conf is not None:
            args.update(update_conf)

        self.img_to_tensor = transforms.ToTensor()

        self.rgbd_transform = transforms.Compose(
            [
                rgbd_T.DepthNorm(
                    max_depth=args.max_depth,
                    clamp_max_before_scale=args.clamp_max_before_scale,
                ),
                transforms.Resize(size=224, interpolation=3),
                transforms.CenterCrop(
                    size=224,
                ),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, img_path, depth_path):
        # here depth refers to disparity, in torch savefile format
        # note use ToTensor to scale image to [0,1] first
        img = Image.open(img_path).convert("RGB")
        img = self.img_to_tensor(img)

        depth = torch.load(depth_path)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)

        rgbd = torch.cat([img, depth], dim=0)
        transform_rgbd = self.rgbd_transform(rgbd)
        img = transform_rgbd[0:3, ...]
        depth = transform_rgbd[3:4, ...]

        return img, depth

    def __repr__(self):
        repr = "(DataAugmentationForRGBD,\n"
        repr += "transform = %s,\n" % str(self.rgbd_transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", rgbd_conf)
        img_mean = cfg.get("img_mean", OPENAI_CLIP_MEAN)
        img_std = cfg.get("img_std", OPENAI_CLIP_STD)
        depth_mean = cfg.get("depth_mean", 0.0418)
        depth_std = cfg.get("depth_std", 0.0295)

        return cls(
            args=args,
            img_mean=img_mean,
            img_std=img_std,
            depth_mean=depth_mean,
            depth_std=depth_std,
            update_conf=update_conf,
        )


class DepthProcessorEval(BaseProcessor):
    def __init__(
        self,
        img_mean=None,
        img_std=None,
        depth_mean=0.0418,
        depth_std=0.0295,
        max_depth=75,
        clamp_max_before_scale=True,
    ):
        img_mean = img_mean if img_mean is not None else OPENAI_CLIP_MEAN
        img_std = img_std if img_std is not None else OPENAI_CLIP_STD
        depth_mean = depth_mean if depth_mean is not None else 0.0
        depth_std = depth_std if depth_std is not None else 1.0

        self.mean = list(img_mean) + [depth_mean]
        self.std = list(img_std) + [depth_std]

        self.rgbd_transform = transforms.Compose(
            [
                rgbd_T.DepthNorm(
                    max_depth=max_depth,
                    clamp_max_before_scale=clamp_max_before_scale,
                ),
                transforms.Resize(size=224, interpolation=3),
                transforms.CenterCrop(
                    size=224,
                ),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, depth):
        # here depth refers to disparity, in torch savefile format
        # note use ToTensor to scale image to [0,1] first
        img = torch.randn((3, 224, 224))

        if depth.ndim == 2:
            depth = depth.unsqueeze(0)

        rgbd = torch.cat([img, depth], dim=0)
        transform_rgbd = self.rgbd_transform(rgbd)
        img = transform_rgbd[0:3, ...]
        depth = transform_rgbd[3:4, ...]

        return depth
