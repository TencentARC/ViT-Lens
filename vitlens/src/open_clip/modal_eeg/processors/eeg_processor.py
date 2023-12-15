import torch
import random
import numpy as np
from scipy.interpolate import interp1d
import re

from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms

from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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


image_transform_conf = OmegaConf.create(
    {
        "num_ops": 2,
        "magnitude": 9,
    }
)


class Image_Processor_Train(BaseProcessor):
    def __init__(self, args, img_mean=None, img_std=None, update_conf=None):
        self.mean = img_mean if img_mean is not None else OPENAI_CLIP_MEAN
        self.std = img_std if img_std is not None else OPENAI_CLIP_STD

        if update_conf is not None:
            args.update(update_conf)

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(
                    num_ops=args.num_ops,
                    magnitude=args.magnitude,
                    interpolation=2,
                ),
                transforms.ToTensor(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, img_path):
        img = Image.open(img_path).convert("RGB")
        transform_img = self.transform(img)
        return transform_img

    def __repr__(self):
        repr = "(DataAugmentationForRGBD,\n"
        repr += "transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", image_transform_conf)
        img_mean = cfg.get("img_mean", OPENAI_CLIP_MEAN)
        img_std = cfg.get("img_std", OPENAI_CLIP_STD)

        return cls(
            args=args, img_mean=img_mean, img_std=img_std, update_conf=update_conf
        )


class Image_Processor_Eval(BaseProcessor):
    def __init__(self, args, img_mean=None, img_std=None, update_conf=None):
        self.mean = img_mean if img_mean is not None else OPENAI_CLIP_MEAN
        self.std = img_std if img_std is not None else OPENAI_CLIP_STD

        if update_conf is not None:
            args.update(update_conf)

        self.transform = transforms.Compose(
            [
                transforms.Resize(size=256, interpolation=3),
                transforms.CenterCrop(
                    size=224,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, img_path):
        img = Image.open(img_path).convert("RGB")
        transform_img = self.transform(img)
        return transform_img

    def __repr__(self):
        repr = "(DataAugmentationForRGBD,\n"
        repr += "transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", image_transform_conf)
        img_mean = cfg.get("img_mean", OPENAI_CLIP_MEAN)
        img_std = cfg.get("img_std", OPENAI_CLIP_STD)

        return cls(
            args=args, img_mean=img_mean, img_std=img_std, update_conf=update_conf
        )


eeg_conf = OmegaConf.create(
    {
        "time_low": 20,
        "time_high": 460,
        "data_len": 512,
    }
)


class EEG_Processor(BaseProcessor):
    def __init__(self, args, update_conf=None):
        self.time_low = args.time_low
        self.time_high = args.time_high
        self.data_len = args.data_len

        if update_conf is not None:
            args.update(update_conf)

    def __call__(self, eeg):
        eeg = eeg.float().t()  # [channel x time] -> [time x channel]
        eeg = eeg[self.time_low : self.time_high, :]

        eeg = np.array(eeg.transpose(0, 1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()

        return eeg

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", eeg_conf)

        return cls(args=args, update_conf=update_conf)


class EEGProcessorEval(BaseProcessor):
    def __init__(self, time_low=20, time_high=460, data_len=512):
        self.time_low = time_low
        self.time_high = time_high
        self.data_len = data_len

    def __call__(self, eeg_path):
        eeg = torch.load(eeg_path)
        eeg = eeg.float().t()  # [channel x time] -> [time x channel]
        eeg = eeg[self.time_low : self.time_high, :]

        eeg = np.array(eeg.transpose(0, 1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()

        return eeg
