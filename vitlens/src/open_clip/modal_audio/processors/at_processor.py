import os
import threading
import time
import random
import torch
import cv2
import json
import numpy as np
import re
import logging
from functools import partial
from einops import rearrange
from copy import deepcopy

from PIL import Image
import torchaudio
import decord
from decord import cpu
from omegaconf import OmegaConf
from torchvision import transforms

import pytorchvideo.transforms as pv_transfrom
from pytorchvideo.data.clip_sampling import (
    ConstantClipsPerVideoSampler,
    RandomClipSampler,
)
from pytorchvideo.data.encoded_video import EncodedVideo

import open_clip.modal_video.processors.functional_video as F
import open_clip.modal_video.processors.transforms_video as lavis_transform
import open_clip.modal_video.processors.video_transform_aio as vt
from open_clip.modal_audio.processors.util_transforms import (
    SpatialCrop,
    AdjustableConstantClipsPerVideoSampler,
)


from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

AST_AS_MEAN = (-4.2677393,)
AST_AS_STD = (4.5689974,)

AS_VGGS_MEAN = (-5.081,)
AS_VGGS_STD = (4.485,)


def get_clip_timepoints(constant_clip_sampler, duration):
    # Read out all clips in this video
    all_clips_timepoints = []
    is_last_clip = False
    end = 0.0
    while not is_last_clip:
        start, end, _, _, is_last_clip = constant_clip_sampler(
            end, duration, annotation=None
        )
        all_clips_timepoints.append((start, end))
    return all_clips_timepoints


def ret_start_end_from_path(video_path):
    res = re.findall(r"&&start=\d+\.?\d*&end=\d+\.?\d*", video_path)
    if len(res) == 0:
        return video_path, None, None
    fetch_res = res[0]
    path = video_path.split(fetch_res)[0]
    times = re.findall("\d+\.?\d*", fetch_res)
    start, end = float(times[0]), float(times[1])
    return path, start, end


def sample_frames(
    num_frames, start_idx=None, end_idx=None, mode="rand", fix_start=None
):
    vlen = end_idx - start_idx
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=start_idx, stop=end_idx, num=acc_samples + 1).astype(
        int
    )
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if mode == "rand":
        frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
    elif mode == "headtail":
        indices_h = sorted(
            random.sample(range(start_idx, start_idx + vlen // 2), acc_samples // 2)
        )
        indices_t = sorted(
            random.sample(
                range(start_idx + vlen // 2, start_idx + vlen),
                acc_samples - acc_samples // 2,
            )
        )
        frame_indices = indices_h + indices_t
    elif fix_start is not None:
        frame_indices = [x[0] + fix_start for x in ranges]
    elif mode == "uniform":
        frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError
    return frame_indices


# Load videos as PIL frames
def read_pil_frames_from_decord(path, num_frames, is_train=True, fix_start=None):
    # path: xxx.mp4 w/o start and end time
    # path: xxx.mp4&&start=1.02&end=3.14 w/ start and end time
    if is_train:
        sample_mode = "rand"
    else:
        sample_mode = "uniform"

    video_path, start, end = ret_start_end_from_path(video_path=path)
    video_reader = decord.VideoReader(
        video_path, width=-1, height=-1, num_threads=1, ctx=cpu(0)
    )
    decord.bridge.set_bridge("torch")
    vlen = len(video_reader)
    start_idx, end_idx = 0, vlen

    # case with specific start time and end time (in sec)
    if (start is not None) and (end is not None):
        fps = video_reader.get_avg_fps()
        start_idx = max(0, int(start * fps))
        end_idx = min(int(end * fps) + 1, vlen)

    frame_indices = sample_frames(
        num_frames,
        start_idx=start_idx,
        end_idx=end_idx,
        mode=sample_mode,
        fix_start=fix_start,
    )
    videodata = video_reader.get_batch(frame_indices).numpy()  # N x H x W x C
    sampled_pil_frames_list = [
        Image.fromarray(videodata[vid, :, :, :]).convert("RGB")
        for vid, _ in enumerate(frame_indices)
    ]
    return sampled_pil_frames_list


def decord_read_video(video_path):
    video_reader = decord.VideoReader(
        video_path, width=-1, height=-1, num_threads=1, ctx=cpu(0)
    )
    decord.bridge.set_bridge("torch")
    duration = len(video_reader) / video_reader.get_avg_fps()  # duration in sec

    if duration < 0.5:
        raise ValueError("Video too short, skip.")

    return video_reader, duration


def decord_get_clip(video_reader, start=None, end=None, sampler=None):
    start_idx, end_idx = 0, len(video_reader)
    fps = video_reader.get_avg_fps()
    if start is not None:
        start_idx = max(0, int(start * fps))
    if end is not None:
        end_idx = min(int(end * fps) + 1, len(video_reader))

    all_frame_indices = torch.from_numpy(np.arange(start_idx, end_idx))
    frame_indices = sampler(all_frame_indices)

    video_data = video_reader.get_batch(frame_indices)  # Size: [T x H x W x C]

    video_data = video_data.to(torch.float)
    video_data = video_data.permute(3, 0, 1, 2)  # Size: [C x T x H x W], torch.float32

    return video_data


def init_video_io_func(args, is_train=True):
    nframes = args.get("nframes", None)
    func = partial(
        read_pil_frames_from_decord,
        num_frames=nframes,
        is_train=is_train,
        fix_start=None,
    )
    return func


def audio_get_clip(
    waveform, sampling_rate, target_duration, start=None, end=None, sub_mean=True
):
    assert waveform.ndim == 2
    wf = deepcopy(waveform)
    orig_duration = float(waveform.shape[1] / sampling_rate)

    if start is not None and end is not None:
        if (
            start < orig_duration and end <= orig_duration and end - start > 0.5
        ):  # set 0.5 second as threshold
            wf = wf[:, int(start * sampling_rate) : int(end * sampling_rate)]

    target_t = int(sampling_rate * target_duration)

    repeat_i = 0
    while wf.shape[1] < target_t and repeat_i <= 5:
        wf = torch.cat([wf, wf], dim=1)
        repeat_i += 1
    if repeat_i > 5:
        raise ValueError(f"Original duration {orig_duration} too short, please skip.")

    if wf.shape[1] > target_t:
        start_idx = random.randint(
            0, (wf.shape[1] - 1) - target_t
        )  # Note: random.randint(A, B) inclusive
        wf = wf[:, start_idx : start_idx + target_t]

    if sub_mean:
        wf = wf - wf.mean()

    return wf


def load_audio(path, sampling_rate=16000, target_duration=5.0, start=None, end=None):
    waveform, sr = torchaudio.load(path)  # [1, sr*duration]
    if sr != sampling_rate:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sampling_rate
        )
    waveform = audio_get_clip(waveform, sampling_rate, target_duration, start, end)
    return waveform


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


ast_conf = OmegaConf.create(
    dict(
        sampling_rate=16000,
        clip_duration=5.0,
        n_clip=3,
        target_length=512,
        mel_bins=128,
        freqm=48,
        timem=96,
        noise_aug=True,
    )
)


class ASTProcessorTrain(BaseProcessor):
    def __init__(self, args, mean, std, update_conf=None):
        self.mean = mean if mean is not None else AST_AS_MEAN
        self.std = std if std is not None else AST_AS_STD

        if update_conf is not None:
            args.update(update_conf)
        logging.info(f"[ASTTrainProcessor conf]: {args}")

        # set the following when instantiate Dataset
        self.sampling_rate = args.sampling_rate
        self.clip_duration = args.clip_duration  # target clip duration
        self.n_clip = args.n_clip  # n_clip audios, for training is 1.
        self.target_length = args.target_length
        self.mel_bins = args.mel_bins

        self.clip_sampler = partial(
            RandomClipSampler(clip_duration=self.clip_duration),
            last_clip_time=None,
            annotation=None,
        )

        # data transform
        transform_list = []
        transform_list.extend(
            [
                lambda x: torch.transpose(x, 0, 1),
                lambda x: x.unsqueeze(0),
            ]
        )
        if args.freqm > 0:
            transform_list.append(torchaudio.transforms.FrequencyMasking(args.freqm))
        if args.timem > 0:
            transform_list.append(torchaudio.transforms.TimeMasking(args.timem))
        transform_list.extend(
            [
                transforms.Normalize(mean=self.mean, std=self.std),
                lambda x: x.squeeze(0),
                lambda x: torch.transpose(x, 0, 1),
            ]
        )
        if args.noise_aug:
            transform_list.extend(
                [
                    lambda x: x
                    + torch.rand(x.shape[0], x.shape[1]) * np.random.rand() / 10.0,
                    lambda x: torch.roll(x, np.random.randint(-10, 10), 0),
                ]
            )
        self.transform = transforms.Compose(transform_list)

    def load_audio_clip(self, path, start=None, end=None):
        wav, sr = torchaudio.load(path)  # Size: (1, sr*duration)
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.sampling_rate
            )
        audio_duration = wav.shape[1] / self.sampling_rate
        if start is None and end is None:
            sample_clip_info = self.clip_sampler(video_duration=audio_duration)
            start = sample_clip_info[0]
            end = sample_clip_info[1]

        start = max(0.0, start)
        end = min(end, audio_duration)
        waveform = audio_get_clip(
            wav,
            sampling_rate=self.sampling_rate,
            target_duration=self.clip_duration,
            start=start,
            end=end,
        )

        return waveform

    def convert2fbank(self, waveform):
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.sampling_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.mel_bins,
            dither=0.0,
            frame_shift=10,
        )
        p = self.target_length - fbank.shape[0]
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0 : self.target_length, :]

        return fbank

    def __call__(self, wav, se=None, **kwargs):
        if se is not None:
            assert len(se) == 2
            st, end = se[0], se[1]
        else:
            st, end = None, None

        if not isinstance(wav, torch.Tensor):
            wav = self.load_audio_clip(wav, start=st, end=end)

        assert wav.shape[1] == int(self.sampling_rate * self.clip_duration)

        # convert to spectrogram
        fbank = self.convert2fbank(wav)

        # transform below
        transformed_fbank = self.transform(fbank)

        return transformed_fbank

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", ast_conf)
        mean = cfg.get("mean", AST_AS_MEAN)
        std = cfg.get("std", AST_AS_STD)
        return cls(args=args, mean=mean, std=std, update_conf=update_conf)


class ASTProcessorEval(BaseProcessor):
    def __init__(self, args, mean, std, update_conf=None):
        self.mean = mean if mean is not None else AST_AS_MEAN
        self.std = std if std is not None else AST_AS_STD

        if update_conf is not None:
            args.update(update_conf)
        logging.info(f"[ASTEvalProcessor conf]: {args}")

        # set the following when instantiate Dataset
        self.sampling_rate = args.sampling_rate
        self.clip_duration = args.clip_duration  # target clip duration
        self.n_clip = args.n_clip  # n_clip audios, for training is 1.
        self.target_length = args.target_length
        self.mel_bins = args.mel_bins

        self.clip_sampler = AdjustableConstantClipsPerVideoSampler(
            clip_duration=self.clip_duration, clips_per_video=self.n_clip
        )  # randomly sample one clip

        # data transform
        transform_list = [
            lambda x: x.unsqueeze(0),
            transforms.Normalize(mean=self.mean, std=self.std),
            lambda x: x.squeeze(0),
        ]
        self.transform = transforms.Compose(transform_list)

    def convert2fbank(self, waveform):
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.sampling_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.mel_bins,
            dither=0.0,
            frame_shift=10,
        )
        p = self.target_length - fbank.shape[0]
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0 : self.target_length, :]

        return fbank

    def __call__(
        self, path, **kwargs
    ):  # no need for start and end from video at eval/inference stage
        wav, sr = torchaudio.load(path)  # [1, sr*duration]
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.sampling_rate
            )
        audio_duration = wav.shape[1] / self.sampling_rate

        audio_list = []
        if audio_duration <= self.clip_duration:
            audio_list = [
                audio_get_clip(wav, self.sampling_rate, self.clip_duration),
            ] * self.n_clip
        else:
            time_pts = get_clip_timepoints(self.clip_sampler, audio_duration)
            for st, end in time_pts:
                audio_list.append(
                    audio_get_clip(
                        wav, self.sampling_rate, self.clip_duration, start=st, end=end
                    )
                )

        audio_list = [
            self.transform(self.convert2fbank(wf_ele)) for wf_ele in audio_list
        ]
        adata = torch.stack(audio_list)

        return adata

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", ast_conf)
        mean = cfg.get("mean", AST_AS_MEAN)
        std = cfg.get("std", AST_AS_STD)
        return cls(args=args, mean=mean, std=std, update_conf=update_conf)


class CLAPProcessorTrain(BaseProcessor):
    # Following CLAP paper and impl
    pass


class CLAPProcessprEval(BaseProcessor):
    # Following CLAP paper and impl
    pass


open_clip_transform_conf = OmegaConf.create(
    {
        "input_size": 224,
        "second_input_size": None,
        "centercrop_size": 512,
        "min_crop_scale": 0.7,
        "train_interpolation": "bicubic",
        "second_interpolation": None,
    }
)


class OpenClipVisProcessorTrain(BaseProcessor):
    def __init__(self, args, mean, std, update_conf=None):
        self.mean = mean if mean is not None else OPENAI_CLIP_MEAN
        self.std = std if std is not None else OPENAI_CLIP_STD

        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_video_io_func(args=args, is_train=True)

        self.common_transform = transforms.Compose(
            [
                vt.GroupRandomResizedCropAndInterpolationWithTwoClips(
                    size=args.input_size,
                    second_size=None,
                    scale=(args.min_crop_scale, 1.0),
                    interpolation=args.train_interpolation,
                    second_interpolation=None,
                ),  # disable second (aug)
            ]
        )
        self.patch_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
                vt.GroupNormalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        n_frames = len(vdata)
        for_patches = self.common_transform(vdata)
        tr_for_patches = self.patch_transform(for_patches)
        tr_for_patches = tr_for_patches.view((n_frames, 3) + tr_for_patches.size()[-2:])
        return tr_for_patches

    def __repr__(self):
        repr = "(DataAugmentationForOpenClip_Video,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", open_clip_transform_conf)
        mean = cfg.get("mean", OPENAI_CLIP_MEAN)
        std = cfg.get("std", OPENAI_CLIP_STD)

        return cls(args=args, mean=mean, std=std, update_conf=update_conf)


class OpenClipVisProcessorEval(BaseProcessor):
    def __init__(self, args, mean, std, update_conf=None):
        self.mean = mean if mean is not None else OPENAI_CLIP_MEAN
        self.std = std if std is not None else OPENAI_CLIP_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_video_io_func(args=args, is_train=False)

        self.common_transform = transforms.Compose(
            [
                vt.GroupCenterCropAndResizedWithTwoClips(
                    centercrop_size=args.centercrop_size,
                    size=args.input_size,
                    second_size=None,
                    interpolation=args.train_interpolation,
                    second_interpolation=None,
                ),  # disable second vis_aug
            ]
        )
        self.patch_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
                vt.GroupNormalize(mean=self.mean, std=self.std),
            ]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        n_frames = len(vdata)
        for_patches = self.common_transform(vdata)
        tr_for_patches = self.patch_transform(for_patches)
        tr_for_patches = tr_for_patches.view((n_frames, 3) + tr_for_patches.size()[-2:])
        return tr_for_patches

    def __repr__(self):
        repr = "(DataAugmentationForOpenClipVal,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", open_clip_transform_conf)

        mean = cfg.get("mean", OPENAI_CLIP_MEAN)
        std = cfg.get("std", OPENAI_CLIP_STD)

        return cls(args=args, mean=mean, std=std, update_conf=update_conf)


pv_transform_conf = OmegaConf.create(
    # make sure to set these param for specific datasets
    {
        "clip_duration": None,
        "n_clip": None,
        "nframes": None,
        "agg_eval": None,
    }
)


class PVProcessorTrain(BaseProcessor):
    def __init__(self, args, mean, std, update_conf=None):
        self.mean = mean if mean is not None else OPENAI_CLIP_MEAN
        self.std = std if std is not None else OPENAI_CLIP_STD

        if update_conf is not None:
            args.update(update_conf)

        # set when instantiate Dataset
        self.clip_duration = args.clip_duration
        self.n_clip = args.n_clip
        self.nframes = args.nframes
        self.clip_sampler = partial(
            RandomClipSampler(clip_duration=self.clip_duration),
            last_clip_time=None,
            annotation=None,
        )  # randomly sample one video clip
        self.frame_sampler = pv_transfrom.UniformTemporalSubsample(
            num_samples=self.nframes, temporal_dim=0
        )

        self.transform = transforms.Compose(
            [
                # pv_transfrom.UniformTemporalSubsample(num_samples=self.nframes, temporal_dim=1), # Frame Sampler: Sampling #n frames within a video clip
                lambda x: rearrange(x, "c t h w -> t c h w"),
                pv_transfrom.RandAugment(
                    magnitude=9,
                    num_layers=2,
                    prob=0.3,
                ),
                lambda x: rearrange(x, "t c h w -> c t h w"),
                pv_transfrom.Div255(),
                pv_transfrom.Normalize(mean=self.mean, std=self.std),
                pv_transfrom.RandomShortSideScale(min_size=256, max_size=340),
                lavis_transform.RandomCropVideo(224),
                lavis_transform.RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    def __call__(self, path):
        # obtain video clips
        vr, duration = decord_read_video(path)
        start, end = None, None
        if duration < self.clip_duration:
            start = 0
            end = duration
        else:
            clip_info = self.clip_sampler(video_duration=duration)
            start = clip_info[0]
            end = clip_info[1]

        vclip = decord_get_clip(
            vr, start=start, end=end, sampler=self.frame_sampler
        )  #  [C x t x h x w]
        vdata = self.transform(vclip)  # [C x T x H x W]

        return vdata, (float(start), float(end))

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", pv_transform_conf)
        mean = cfg.get("mean", OPENAI_CLIP_MEAN)
        std = cfg.get("std", OPENAI_CLIP_STD)

        return cls(args=args, mean=mean, std=std, update_conf=update_conf)


class PVProcessorEval(BaseProcessor):
    def __init__(self, args, mean, std, update_conf=None):
        self.mean = mean if mean is not None else OPENAI_CLIP_MEAN
        self.std = std if std is not None else OPENAI_CLIP_STD

        if update_conf is not None:
            args.update(update_conf)

        # set when instantiate Dataset
        self.clip_duration = args.clip_duration
        self.n_clip = args.n_clip
        self.nframes = args.nframes
        self.agg_eval = args.agg_eval
        self.clip_sampler = AdjustableConstantClipsPerVideoSampler(
            clip_duration=self.clip_duration, clips_per_video=self.n_clip
        )
        self.frame_sampler = pv_transfrom.UniformTemporalSubsample(
            num_samples=self.nframes, temporal_dim=0
        )

        transform_list = [
            pv_transfrom.Div255(),
            pv_transfrom.Normalize(mean=self.mean, std=self.std),
            pv_transfrom.ShortSideScale(224),
            # lavis_transform.CenterCropVideo(224),
        ]
        if self.agg_eval:
            transform_list.append(lavis_transform.CenterCropVideo(224))
        self.transform = transforms.Compose(transform_list)
        self.spatial_crop = SpatialCrop(crop_size=224, num_crops=3)

    def __call__(self, path):
        # obtain videp clips
        vr, video_duration = decord_read_video(path)
        if not self.agg_eval:
            vdata = decord_get_clip(vr, 0.0, video_duration, self.frame_sampler)
            vdata = self.transform(vdata)
            return vdata, (0.0, duration)

        else:  # agg_eval mode: temporal segments --> spatial crops
            _duration = (
                video_duration
                if video_duration < self.clip_duration
                else self.clip_duration
            )
            self.clip_sampler.set_clip_duration(_duration)
            video_clips = []
            se_list = []
            time_pts = get_clip_timepoints(self.clip_sampler, video_duration)
            for st, end in time_pts:
                video_clips.append(
                    self.transform(decord_get_clip(vr, st, end, self.frame_sampler))
                )
                for _ in range(3):  # 3 spatial crops
                    se_list.append((st, end))
            video_clips = self.spatial_crop(
                video_clips
            )  # List: (n_clip.temporal x 3.spatial)
            video_clips = torch.stack(
                video_clips
            )  # (n_clip.temporal x 3.spatial) x C x T x H x W

            return video_clips, se_list

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()
        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", pv_transform_conf)
        mean = cfg.get("mean", OPENAI_CLIP_MEAN)
        std = cfg.get("std", OPENAI_CLIP_STD)

        return cls(args=args, mean=mean, std=std, update_conf=update_conf)


# for inference
class AudioASTProcessorEval(BaseProcessor):
    def __init__(
        self,
        mean=AST_AS_MEAN,
        std=AST_AS_STD,
        sampling_rate=16000,
        clip_duration=5.0,
        n_clip=3,
        target_length=512,
        mel_bins=128,
    ):
        self.mean = mean if mean is not None else AST_AS_MEAN
        self.std = std if std is not None else AST_AS_STD
        self.sampling_rate = sampling_rate
        self.clip_duration = clip_duration  # target clip duration
        self.n_clip = n_clip  # n_clip audios, for training is 1.
        self.target_length = target_length
        self.mel_bins = mel_bins

        self.clip_sampler = AdjustableConstantClipsPerVideoSampler(
            clip_duration=self.clip_duration, clips_per_video=self.n_clip
        )

        # data transform
        transform_list = [
            lambda x: x.unsqueeze(0),
            transforms.Normalize(mean=self.mean, std=self.std),
            lambda x: x.squeeze(0),
        ]
        self.transform = transforms.Compose(transform_list)

    def convert2fbank(self, waveform):
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=self.sampling_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.mel_bins,
            dither=0.0,
            frame_shift=10,
        )
        p = self.target_length - fbank.shape[0]
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0 : self.target_length, :]

        return fbank

    def __call__(
        self, path, **kwargs
    ):  # no need for start and end from video at eval/inference stage
        wav, sr = torchaudio.load(path)  # [1, sr*duration]
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(
                wav, orig_freq=sr, new_freq=self.sampling_rate
            )
        audio_duration = wav.shape[1] / self.sampling_rate

        audio_list = []
        if audio_duration <= self.clip_duration:  # only one clip, repeat
            audio_list = [
                audio_get_clip(wav, self.sampling_rate, self.clip_duration),
            ] * self.n_clip
        else:
            time_pts = get_clip_timepoints(self.clip_sampler, audio_duration)
            for st, end in time_pts:
                audio_list.append(
                    audio_get_clip(
                        wav, self.sampling_rate, self.clip_duration, start=st, end=end
                    )
                )

        audio_list = [
            self.transform(self.convert2fbank(wf_ele)) for wf_ele in audio_list
        ]
        adata = torch.stack(audio_list, dim=0)

        return adata
