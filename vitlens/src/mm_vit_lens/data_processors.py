import os
import re
import logging
import torch
import numpy as np

from PIL import Image
from omegaconf import OmegaConf
from torchvision import transforms
from easydict import EasyDict as edict

from open_clip import get_tokenizer


def wrap_list(data):
    if isinstance(data, list):
        return data
    return [
        data,
    ]


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


class TextProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=70, cfg=None):
        self.prompt = prompt
        self.max_words = max_words
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg.model)

    def __call__(self, caption, device="cpu"):
        if caption is None:
            return None

        caption = wrap_list(caption)
        caption = [self.prompt + self.pre_caption(c) for c in caption]
        tokenized_caption = self.tokenizer(caption).to(device)

        return tokenized_caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create(dict(model="ViT-L-14"))

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 70)

        return cls(prompt=prompt, max_words=max_words, cfg=cfg)

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


class ImageProcessor(BaseProcessor):
    def __init__(self, image_size=224, image_mean=None, image_std=None, transform=None):
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        if transform:
            self.transform = transform
        else:
            from open_clip.factory import image_transform

            self.transform = image_transform(
                image_size=image_size, is_train=False, mean=image_mean, std=image_std
            )

    def set_image_transform(self, transform):
        self.transform = transform

    def __call__(self, image_paths, device="cpu"):
        if image_paths is None:
            return None

        image_paths = wrap_list(image_paths)
        image_outputs = []
        for image_path in image_paths:
            with open(image_path, "rb") as fopen:
                image = Image.open(fopen).convert("RGB")
            image = self.transform(image)
            image_outputs.append(image)

        image_outputs = torch.stack(image_outputs, dim=0).to(device)

        return image_outputs


class PointCloudProcessor(BaseProcessor):
    def __init__(self, n_sample_points=8192, uniform=True, idendity=False):
        self.n_sample_points = n_sample_points
        self.uniform = uniform

        from open_clip.modal_3d.processors.pc_processor import PCProcessorEval

        self.wrap_processor = PCProcessorEval(n_sample_points, uniform, idendity)

    def set_idendity(self, idendity_v):
        self.wrap_processor.set_attr(idendity=idendity_v)

    def __call__(self, pc_paths, device="cpu"):
        if pc_paths is None:
            return None

        pc_paths = wrap_list(pc_paths)
        pc_outputs = []
        for pc_path in pc_paths:
            pc = np.load(pc_path)
            pc = self.wrap_processor(pc)
            pc_outputs.append(pc)

        pc_outputs = torch.stack(pc_outputs, dim=0).to(device)

        return pc_outputs


class DepthProcessor(BaseProcessor):
    def __init__(
        self,
        depth_mean=0.0418,
        depth_std=0.0295,
        max_depth=75,
        clamp_max_before_scale=True,
    ):
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.max_depth = max_depth
        self.clamp_max_before_scale = clamp_max_before_scale

        from open_clip.modal_depth.processors.vt_processor import DepthProcessorEval

        self.wrap_processor = DepthProcessorEval(
            depth_mean=depth_mean,
            depth_std=depth_std,
            max_depth=max_depth,
            clamp_max_before_scale=clamp_max_before_scale,
        )

    def __call__(self, depth_paths, device="cpu"):
        if depth_paths is None:
            return None

        depth_paths = wrap_list(depth_paths)
        depth_outputs = []
        for depth_path in depth_paths:
            depth = torch.load(depth_path)
            depth = self.wrap_processor(depth)
            depth_outputs.append(depth)

        depth_outputs = torch.stack(depth_outputs, dim=0).to(device)

        return depth_outputs


class AudioProcessor(BaseProcessor):
    def __init__(
        self,
        sampling_rate=16000,
        clip_duration=5.0,
        n_clip=3,
        target_length=512,
        mel_bins=128,
        cfg=None,
    ):
        self.sampling_rate = sampling_rate
        self.clip_duration = clip_duration
        self.n_clip = n_clip
        self.target_length = target_length
        self.mel_bins = mel_bins
        self.cfg = cfg

        from open_clip.modal_audio.processors.at_processor import AudioASTProcessorEval

        self.wrap_processor = AudioASTProcessorEval(
            sampling_rate=sampling_rate,
            clip_duration=clip_duration,
            n_clip=n_clip,
            target_length=target_length,
            mel_bins=mel_bins,
        )

    def setter(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])
        self.wrap_processor = AudioASTProcessorEval(
            sampling_rate=self.sampling_rate,
            clip_duration=self.clip_duration,
            n_clip=self.n_clip,
            target_length=self.target_length,
            mel_bins=self.mel_bins,
        )

    def __call__(self, audio_paths, device="cpu"):
        if audio_paths is None:
            return None

        audio_paths = wrap_list(audio_paths)
        audio_outputs = []
        for audio_path in audio_paths:
            audio = self.wrap_processor(audio_path)
            audio_outputs.append(audio)

        audio_outputs = torch.stack(audio_outputs, dim=0).to(device)

        return audio_outputs


class TactileProcessor(BaseProcessor):
    def __init__(self, image_mean=None, image_std=None):
        self.image_mean = image_mean
        self.image_std = image_std

        from open_clip.modal_tactile.processors.tact_processor import (
            TactileRGBProcessorEval,
        )

        self.wrap_processor = TactileRGBProcessorEval(
            img_mean=image_mean, img_std=image_std
        )

    def __call__(self, tactile_flist, device="cpu"):
        if tactile_flist is None:
            return None

        tactile_flist = wrap_list(tactile_flist)
        tactile_outputs = []
        for tactile_fn in tactile_flist:
            tactile = self.wrap_processor(tactile_fn)
            tactile_outputs.append(tactile)

        tactile_outputs = torch.stack(tactile_outputs, dim=0).to(device)

        return tactile_outputs


class EEGProcessor(BaseProcessor):
    def __init__(self, time_low=20, time_high=460, data_len=512):
        self.time_low = time_low
        self.time_high = time_high
        self.data_len = data_len

        from open_clip.modal_eeg.processors.eeg_processor import EEGProcessorEval

        self.wrap_processor = EEGProcessorEval(
            time_low=time_low, time_high=time_high, data_len=data_len
        )

    def __call__(self, eeg_paths, device="cpu"):
        if eeg_paths is None:
            return None

        eeg_paths = wrap_list(eeg_paths)
        eeg_outputs = []
        for eeg_path in eeg_paths:
            eeg = self.wrap_processor(eeg_path)
            eeg_outputs.append(eeg)

        eeg_outputs = torch.stack(eeg_outputs, dim=0).to(device)

        return eeg_outputs


def vitlensL_processors():
    processors = dict(
        image=ImageProcessor(
            image_size=224, image_mean=None, image_std=None, transform=None
        ),  # create default image transform
        text=TextProcessor(cfg=edict(model="ViT-L-14")),
        pc=PointCloudProcessor(n_sample_points=8192, uniform=True),
        depth=DepthProcessor(),
        audio=AudioProcessor(),
        tactile=TactileProcessor(),
        eeg=EEGProcessor(),
    )

    return processors


def vitlensB_processors():
    return None


def get_vitlens_processors_cls():
    processor_cls = dict(
        vitlensL=vitlensL_processors,
        vitlensB=vitlensB_processors,
    )
    return processor_cls
