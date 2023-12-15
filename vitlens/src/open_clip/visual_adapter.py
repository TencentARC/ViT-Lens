import os
import math
import torch
import torch.nn as nn


def get_visual_adapter(cfg, **kwargs):
    vtype = cfg.visual_modality_type

    if vtype in ["3dpc", "pc", "pointcloud", "point_cloud", "point cloud"]:
        if cfg.pc_tokenizer == "pointbert":
            from open_clip.modal_3d.models.pointbert.point_encoder import PointTokenizer

            return PointTokenizer(config=cfg)
        elif cfg.pc_tokenizer == "pnsa":
            from open_clip.modal_3d.models.pointnet.pointnet_util import (
                PointNSATokenizer,
            )

            return PointNSATokenizer(config=cfg)
        else:
            NotImplementedError

    elif vtype == "3dpc_raw":
        return nn.Identity()

    elif vtype == "depth":
        from open_clip.modal_depth.models.DepthTokenizer import DepthTokenizer

        return DepthTokenizer(
            grid_size=kwargs["grid_size"],
            patch_size=kwargs["patch_size"],
            width=kwargs["width"],
            input_patchnorm=kwargs["input_patchnorm"],
        )

    elif vtype == "audio":
        from open_clip.modal_audio.models.AST_tokenizer import AST_tokenizer

        exp_args = kwargs["exp_args"]
        return AST_tokenizer(
            fstride=exp_args.audio_fstride,
            tstride=exp_args.audio_tstride,
            input_fdim=exp_args.audio_mel_bins,
            input_tdim=exp_args.audio_target_length,
            patch_size=kwargs["patch_size"],
            width=kwargs["width"],
        )

    elif vtype == "tactile":
        return None

    elif vtype == "eeg":
        from open_clip.modal_eeg.models.EEG_tokenizer import PatchEmbed1D

        exp_args = kwargs["exp_args"]
        return PatchEmbed1D(
            time_len=exp_args.eeg_time_len,
            in_chans=exp_args.eeg_chans,
            window_size=exp_args.eeg_window_size,
            stride=exp_args.eeg_stride,
            width=kwargs["width"],
        )

    elif vtype == "video":
        raise NotImplementedError

    else:
        raise NotImplementedError
