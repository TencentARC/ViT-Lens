import torch
import random
import numpy as np
import re
import PIL
import decord

from PIL import Image
from decord import cpu
from open_clip.modal_video.processors.randaugment import VideoRandomAugment
import open_clip.modal_video.processors.functional_video as F
import open_clip.modal_video.processors.transforms_video as lavis_transform
import open_clip.modal_video.processors.video_transform_aio as vt
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

beitv1_transform_conf = OmegaConf.create(
    {
        "input_size": 224,
        "second_input_size": 112,
        "centercrop_size": 384,
        "train_interpolation": "bicubic",
        "second_interpolation": "lanczos",
    }
)

beitv2_transform_conf = OmegaConf.create(
    {
        "input_size": 224,
        "second_input_size": 224,
        "centercrop_size": 384,
        "min_crop_scale": 0.2,
        "train_interpolation": "bicubic",
        "second_interpolation": "bicubic",
    }
)

eva_clip_transform_conf = OmegaConf.create(
    {
        "input_size": 224,
        "second_input_size": 224,
        "centercrop_size": 384,
        "min_crop_scale": 0.2,
        "train_interpolation": "bicubic",
        "second_interpolation": "bicubic",
    }
)

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

lavis_transform_conf = OmegaConf.create(
    {
        "input_size": 224,
        "min_scale": 0.5,
        "max_scale": 1.0,
        "interpolation": "bicubic",
        "n": 2,
        "m": 5,
        "aug_list": [
            "Identity",
            "AutoContrast",
            "Brightness",
            "Sharpness",
            "Equalize",
            "ShearX",
            "ShearY",
            "TranslateX",
            "TranslateY",
            "Rotate",
        ],
    }
)

aiov1_transform_conf = OmegaConf.create(
    {
        "input_size": 224,
        "interpolation": "nearest",
    }
)


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
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
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
        frame_idxs = indices_h + indices_t
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif mode == "uniform":
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError
    return frame_idxs


# load data from video
# TODO add a version with start and end time out of an intact video
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

    frame_idxs = sample_frames(
        num_frames,
        start_idx=start_idx,
        end_idx=end_idx,
        mode=sample_mode,
        fix_start=fix_start,
    )
    videodata = video_reader.get_batch(frame_idxs).numpy()  # N x H x W x C
    sampled_pil_frames_list = [
        Image.fromarray(videodata[vid, :, :, :]).convert("RGB")
        for vid, _ in enumerate(frame_idxs)
    ]
    return sampled_pil_frames_list


# load data from image
def read_image_as_pil_list(path):
    if isinstance(path, Image.Image):  # LAION case
        img = path
        return [img]
    img = Image.open(path).convert("RGB")
    if img is None:
        raise Exception("Invalid img!", path)
    else:
        return [img]


# dvae transform, refer BEIT V1
logit_laplace_eps: float = 0.1


def map_pixels(x: torch.Tensor):
    if x.dtype != torch.float:
        raise ValueError("expected input to have type float")

    return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps


def unmap_pixels(x: torch.Tensor):
    if len(x.shape) != 4:
        raise ValueError("expected input to be 4d")
    if x.dtype != torch.float:
        raise ValueError("expected input to have type float")

    return torch.clamp((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)


def init_io_func(args, is_train=True):
    data_type = args.get("data_type", None)
    if data_type in ["video", "videos"]:
        from functools import partial

        nframes = args.get("nframes", None)
        func = partial(
            read_pil_frames_from_decord,
            num_frames=nframes,
            is_train=is_train,
            fix_start=None,
        )
    elif data_type in ["image", "images"]:
        func = read_image_as_pil_list
    else:
        raise ValueError(f"{data_type} is not yet supported!")

    return func


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


# @registry.register_processor("aio_beitv1_train")
class AIOBeitv1VisProcessor(BaseProcessor):
    def __init__(self, args, update_conf=None):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=True)

        self.common_transform = transforms.Compose(
            [
                # vt.ColorJitter(0.4, 0.4, 0.4),
                vt.GroupRandomResizedCropAndInterpolationWithTwoClips(
                    size=args.input_size,
                    second_size=args.second_input_size,
                    interpolation=args.train_interpolation,
                    second_interpolation=args.second_interpolation,
                ),
            ]
        )
        self.patch_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
                vt.GroupNormalize(mean=mean, std=std),
            ]
        )
        self.visual_token_transform = transforms.Compose(
            [vt.Stack(roll=False), vt.ToTorchFormatTensor(), map_pixels]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        n_frames = len(vdata)
        for_patches, for_visual_tokens = self.common_transform(vdata)
        tr_for_patches, tr_for_visual_tokens = self.patch_transform(
            for_patches
        ), self.visual_token_transform(for_visual_tokens)
        tr_for_patches = tr_for_patches.view((n_frames, 3) + tr_for_patches.size()[-2:])
        tr_for_visual_tokens = tr_for_visual_tokens.view(
            (n_frames, 3) + tr_for_visual_tokens.size()[-2:]
        )
        return tr_for_patches, tr_for_visual_tokens

    def __repr__(self):
        repr = "(DataAugmentationForBEiTv1,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        # repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", beitv1_transform_conf)

        return cls(args=args, update_conf=update_conf)


# @registry.register_processor("aio_beitv1_eval")
class AIOBeitv1VisProcessorEval(BaseProcessor):
    def __init__(self, args, update_conf=None):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=False)

        self.common_transform = transforms.Compose(
            [
                vt.GroupCenterCropAndResizedWithTwoClips(
                    centercrop_size=args.centercrop_size,
                    size=args.input_size,
                    second_size=args.second_input_size,
                    interpolation=args.train_interpolation,
                    second_interpolation=args.second_interpolation,
                ),
            ]
        )
        self.patch_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
                vt.GroupNormalize(mean=mean, std=std),
            ]
        )
        self.visual_token_transform = transforms.Compose(
            [vt.Stack(roll=False), vt.ToTorchFormatTensor(), map_pixels]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        n_frames = len(vdata)
        for_patches, for_visual_tokens = self.common_transform(vdata)
        tr_for_patches, tr_for_visual_tokens = self.patch_transform(
            for_patches
        ), self.visual_token_transform(for_visual_tokens)
        tr_for_patches = tr_for_patches.view((n_frames, 3) + tr_for_patches.size()[-2:])
        tr_for_visual_tokens = tr_for_visual_tokens.view(
            (n_frames, 3) + tr_for_visual_tokens.size()[-2:]
        )
        return tr_for_patches, tr_for_visual_tokens

    def __repr__(self):
        repr = "(DataAugmentationForBEiTValv1,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", beitv1_transform_conf)

        return cls(args=args, update_conf=update_conf)


# @registry.register_processor("aio_beitv2_train")
class AIOBeitv2VisProcessor(BaseProcessor):
    def __init__(self, args, update_conf=None):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=True)

        self.common_transform = transforms.Compose(
            [
                # vt.ColorJitter(0.4, 0.4, 0.4),
                vt.GroupRandomResizedCropAndInterpolationWithTwoClips(
                    size=args.input_size,
                    second_size=args.second_input_size,
                    scale=(args.min_crop_scale, 1.0),
                    interpolation=args.train_interpolation,
                    second_interpolation=args.second_interpolation,
                ),
            ]
        )
        self.patch_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
                vt.GroupNormalize(mean=mean, std=std),
            ]
        )
        self.visual_token_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
            ]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        n_frames = len(vdata)
        for_patches, for_visual_tokens = self.common_transform(vdata)
        tr_for_patches, tr_for_visual_tokens = self.patch_transform(
            for_patches
        ), self.visual_token_transform(for_visual_tokens)
        tr_for_patches = tr_for_patches.view((n_frames, 3) + tr_for_patches.size()[-2:])
        tr_for_visual_tokens = tr_for_visual_tokens.view(
            (n_frames, 3) + tr_for_visual_tokens.size()[-2:]
        )
        return tr_for_patches, tr_for_visual_tokens

    def __repr__(self):
        repr = "(DataAugmentationForBEiTv2,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", beitv2_transform_conf)

        return cls(args=args, update_conf=update_conf)


# @registry.register_processor("aio_beitv2_eval")
class AIOBeitv2VisProcessorEval(BaseProcessor):
    def __init__(self, args, update_conf=None):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=False)

        self.common_transform = transforms.Compose(
            [
                # vt.ColorJitter(0.4, 0.4, 0.4),
                vt.GroupCenterCropAndResizedWithTwoClips(
                    centercrop_size=args.centercrop_size,
                    size=args.input_size,
                    second_size=args.second_input_size,
                    interpolation=args.train_interpolation,
                    second_interpolation=args.second_interpolation,
                ),
            ]
        )
        self.patch_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
                vt.GroupNormalize(mean=mean, std=std),
            ]
        )
        self.visual_token_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
            ]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        n_frames = len(vdata)
        for_patches, for_visual_tokens = self.common_transform(vdata)
        tr_for_patches, tr_for_visual_tokens = self.patch_transform(
            for_patches
        ), self.visual_token_transform(for_visual_tokens)
        tr_for_patches = tr_for_patches.view((n_frames, 3) + tr_for_patches.size()[-2:])
        tr_for_visual_tokens = tr_for_visual_tokens.view(
            (n_frames, 3) + tr_for_visual_tokens.size()[-2:]
        )
        return tr_for_patches, tr_for_visual_tokens

    def __repr__(self):
        repr = "(DataAugmentationForBEiTValv2,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", beitv2_transform_conf)

        return cls(args=args, update_conf=update_conf)


# @registry.register_processor("aio_evaclip_train")
class AIOEVAClipVisProcessor(BaseProcessor):
    def __init__(self, args, update_conf=None):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=True)

        self.common_transform = transforms.Compose(
            [
                vt.GroupRandomResizedCropAndInterpolationWithTwoClips(
                    size=args.input_size,
                    second_size=args.second_input_size,
                    scale=(args.min_crop_scale, 1.0),
                    interpolation=args.train_interpolation,
                    second_interpolation=args.second_interpolation,
                ),
            ]
        )
        self.patch_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
                vt.GroupNormalize(mean=mean, std=std),
            ]
        )
        self.visual_token_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
                vt.GroupNormalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        n_frames = len(vdata)
        for_patches, for_visual_tokens = self.common_transform(vdata)
        tr_for_patches, tr_for_visual_tokens = self.patch_transform(
            for_patches
        ), self.visual_token_transform(for_visual_tokens)
        tr_for_patches = tr_for_patches.view((n_frames, 3) + tr_for_patches.size()[-2:])
        tr_for_visual_tokens = tr_for_visual_tokens.view(
            (n_frames, 3) + tr_for_visual_tokens.size()[-2:]
        )
        return tr_for_patches, tr_for_visual_tokens

    def __repr__(self):
        repr = "(DataAugmentationForEVAClip,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", eva_clip_transform_conf)

        return cls(args=args, update_conf=update_conf)


# @registry.register_processor("aio_evaclip_eval")
class AIOEVAClipVisProcessorEval(BaseProcessor):
    def __init__(self, args, update_conf=None):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=False)

        self.common_transform = transforms.Compose(
            [
                vt.GroupCenterCropAndResizedWithTwoClips(
                    centercrop_size=args.centercrop_size,
                    size=args.input_size,
                    second_size=args.second_input_size,
                    interpolation=args.train_interpolation,
                    second_interpolation=args.second_interpolation,
                ),
            ]
        )
        self.patch_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
                vt.GroupNormalize(mean=mean, std=std),
            ]
        )
        self.visual_token_transform = transforms.Compose(
            [
                vt.Stack(roll=False),
                vt.ToTorchFormatTensor(),
                vt.GroupNormalize(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
            ]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        n_frames = len(vdata)
        for_patches, for_visual_tokens = self.common_transform(vdata)
        tr_for_patches, tr_for_visual_tokens = self.patch_transform(
            for_patches
        ), self.visual_token_transform(for_visual_tokens)
        tr_for_patches = tr_for_patches.view((n_frames, 3) + tr_for_patches.size()[-2:])
        tr_for_visual_tokens = tr_for_visual_tokens.view(
            (n_frames, 3) + tr_for_visual_tokens.size()[-2:]
        )
        return tr_for_patches, tr_for_visual_tokens

    def __repr__(self):
        repr = "(DataAugmentationForEVAClipVal,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += ")"
        return repr

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", eva_clip_transform_conf)

        return cls(args=args, update_conf=update_conf)


# @registry.register_processor("aiov1_train")
class AIOV1VisProcessor(BaseProcessor):
    def __init__(self, args, update_conf=None):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=True)

        self.transform = transforms.Compose(
            [
                vt.Resize(int(args.input_size * 1.2), interpolation=args.interpolation),
                vt.RandomCrop(args.input_size),
                vt.ClipToTensor(channel_nb=3),
                vt.Normalize(mean=mean, std=std),
                lavis_transform.ToTCHW(),
            ]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        assert isinstance(vdata, (Image.Image, list, tuple))
        if not isinstance(vdata, (list, tuple)):
            vdata = [
                vdata,
            ]
        videos = [np.asarray(img) for img in vdata]
        video_tensor, aug_video_tensor = self.transform(videos), None
        return video_tensor, aug_video_tensor

    def __repr__(self):  # -> str:
        return self.__class__.__name__

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", aiov1_transform_conf)

        return cls(args=args, update_conf=update_conf)


# @registry.register_processor("aiov1_eval")
class AIOV1VisProcessorEval(BaseProcessor):
    def __init__(self, args, update_conf=None):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=False)

        self.transform = transforms.Compose(
            [
                vt.Resize(int(args.input_size * 1.2), interpolation=args.interpolation),
                vt.CenterCrop(args.input_size),
                vt.ClipToTensor(channel_nb=3),
                vt.Normalize(mean=mean, std=std),
                lavis_transform.ToTCHW(),
            ]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        assert isinstance(vdata, (Image.Image, list, tuple))
        if not isinstance(vdata, (list, tuple)):
            vdata = [
                vdata,
            ]
        videos = [np.asarray(img) for img in vdata]
        video_tensor, aug_video_tensor = self.transform(videos), None
        return video_tensor, aug_video_tensor

    def __repr__(self):  # -> str:
        return self.__class__.__name__

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", aiov1_transform_conf)

        return cls(args=args, update_conf=update_conf)


# @registry.register_processor("aio_lavis_train")
class AIOLavisVisProcessor(BaseProcessor):
    def __init__(self, args, update_conf=None):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=True)

        self.transform = transforms.Compose(
            [
                lavis_transform.TOCTHWfloat(),
                lavis_transform.RandomResizedCropVideo(
                    args.input_size,
                    scale=(args.min_scale, args.max_scale),
                    interpolation_mode=args.interpolation,
                ),
                lavis_transform.RandomHorizontalFlipVideo(),
                lavis_transform.ToTHWC(),
                VideoRandomAugment(args.n, args.m, augs=args.aug_list),
                lavis_transform.ToUint8(),
                lavis_transform.ToTensorVideo(),
                lavis_transform.NormalizeVideo(mean=mean, std=std),
                lavis_transform.ToTCHW(),
            ]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        if not torch.is_tensor(vdata):
            # is a list of PIL image, impl for check later
            if not isinstance(vdata, (list, tuple)):
                vdata = [
                    vdata,
                ]
            videos = [np.asarray(img) for img in vdata]
            videos = np.stack(videos)
            video_tensor = torch.from_numpy(videos)  # t x h x w x c
            video_tensor = video_tensor.permute(0, 3, 1, 2)  # t x c x h x w
            video_tensor, aug_video_tensor = self.transform(video_tensor), None
        else:
            if vdata.ndim == 3:
                vdata = vdata.unsqueeze(0)
            video_tensor, aug_video_tensor = self.transform(vdata), None

        return video_tensor, aug_video_tensor

    def __repr__(self):  # -> str:
        return self.__class__.__name__

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", lavis_transform_conf)

        return cls(args=args, update_conf=update_conf)


# @registry.register_processor("aio_lavis_eval")
class AIOLavisVisProcessorEval(BaseProcessor):
    def __init__(self, args, update_conf=None):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=False)

        self.transform = transforms.Compose(
            [
                lavis_transform.TOCTHWfloat(),
                lavis_transform.ResizeVideo(
                    target_size=args.input_size, interpolation_mode=args.interpolation
                ),
                lavis_transform.ToUint8(),
                lavis_transform.ToTHWC(),
                lavis_transform.ToTensorVideo(),
                lavis_transform.NormalizeVideo(mean=mean, std=std),
                lavis_transform.ToTCHW(),
            ]
        )

    def __call__(self, path):
        vdata = self.io_func(path=path)
        if not torch.is_tensor(vdata):
            # is a list of PIL image, impl for check later
            if not isinstance(vdata, (list, tuple)):
                vdata = [
                    vdata,
                ]
            videos = [np.asarray(img) for img in vdata]
            videos = np.stack(videos)
            video_tensor = torch.from_numpy(videos)  # t x h x w x c
            video_tensor = video_tensor.permute(0, 3, 1, 2)  # t x c x h x w
            video_tensor, aug_video_tensor = self.transform(video_tensor), None
        else:
            if vdata.ndim == 3:
                vdata = vdata.unsqueeze(0)
            video_tensor, aug_video_tensor = self.transform(vdata), None

        return video_tensor, aug_video_tensor

    def __repr__(self):  # -> str:
        return self.__class__.__name__

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        update_conf = cfg.pop("update_conf", None)
        args = cfg.get("params", lavis_transform_conf)

        return cls(args=args, update_conf=update_conf)


# @registry.register_processor("blip_caption")
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


# @registry.register_processor("blip_question")
class BlipQuestionProcessor(BaseProcessor):
    def __init__(self, max_words=50):
        self.max_words = max_words

    def __call__(self, question):
        return self.pre_question(question)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_words = cfg.get("max_words", 50)

        return cls(max_words=max_words)

    def pre_question(self, question):
        question = re.sub(
            r"([.!\"()*#:;~])",
            "",
            question.lower(),
        )
        question = question.rstrip(" ")

        # truncate question
        question_words = question.split(" ")
        if len(question_words) > self.max_words:
            question = " ".join(question_words[: self.max_words])

        return question


class OpenClip_VisProcessor(BaseProcessor):
    def __init__(self, args, mean, std, update_conf=None):
        self.mean = mean if mean is not None else OPENAI_CLIP_MEAN
        self.std = std if std is not None else OPENAI_CLIP_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=True)

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
        return tr_for_patches, vdata

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


class OpenClip_VisProcessorEval(BaseProcessor):
    def __init__(self, args, mean, std, update_conf=None):
        self.mean = mean if mean is not None else OPENAI_CLIP_MEAN
        self.std = std if std is not None else OPENAI_CLIP_STD
        if update_conf is not None:
            args.update(update_conf)

        self.io_func = init_io_func(args=args, is_train=False)

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
        return tr_for_patches, vdata

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
