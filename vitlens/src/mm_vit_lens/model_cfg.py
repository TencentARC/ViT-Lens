import os
import json

from copy import deepcopy
from easydict import EasyDict as edict
from open_clip.constants import CKPT_CACHE_DIR


default_cfg = edict(
    audio_clip_duration=5.0,
    audio_fstride=10,
    audio_mel_bins=128,
    audio_sampling_rate=16000,
    audio_target_length=512,
    audio_tstride=10,
    aug_cfg={},
    cache_dir=CKPT_CACHE_DIR,
    dataset_type="image",
    device="cpu",
    disable_orig_pos=False,
    disable_pt_vit=False,
    disable_visual_adapter_pos=False,
    eeg_chans=128,
    eeg_stride=1,
    eeg_time_len=512,
    eeg_window_size=1,
    force_custom_text=False,
    force_image_size=None,
    force_patch_dropout=None,
    force_quick_gelu=False,
    image_mean=None,
    image_std=None,
    load_ckpt_strict=False,
    model="ViT-L-14",
    pc_encoder_dims=256,
    pc_group_size=32,
    pc_in_channel=3,
    pc_npoints=8192,
    pc_num_group=512,
    pc_radius=0.2,
    pc_tokenizer="pointbert",
    pc_trans_dim=384,
    perceiver_as_identity=False,
    perceiver_as_transformer=False,
    perceiver_attn_dropout=0.0,
    perceiver_cross_dim_head=64,
    perceiver_cross_heads=1,
    perceiver_depth=1,
    perceiver_ff_dropout=0.0,
    perceiver_fourier_encode_data=False,
    perceiver_input_axis=1,
    perceiver_input_chan=1024,
    perceiver_latent_dim=1024,
    perceiver_latent_dim_head=64,
    perceiver_latent_heads=16,
    perceiver_max_freq=10.0,
    perceiver_num_classes=1000,
    perceiver_num_freq_bands=32,
    perceiver_num_latents=256,
    perceiver_self_per_cross_attn=1,
    perceiver_weight_tie_layers=False,
    precision="fp32",
    pretrained="datacomp_xl_s13b_b90k",
    pretrained_image=False,
    skip_trans_first_n_layers=None,
    torchcompile=False,
    torchscript=False,
    trace=False,
    use_bn_sync=False,
    use_bnb_linear=None,
    use_eva_pt_lin=False,
    use_openclip_transform=False,
    use_perceiver=False,
    use_visual_adapter=False,
    v_key="image",
    visual_arch="perceiver_vit",
    visual_modality_type="image",
)

vitlens_model_cfg = edict(
    # config for ViT-Lens-L
    vitlensL=edict(
        model="ViT-L-14",
        pretrained="datacomp_xl_s13b_b90k",
        pc=edict(
            visual_modality_type="3dpc",
            v_key="pc",
            use_perceiver=True,
            use_visual_adapter=True,
            pc_encoder_dims=256,
            pc_group_size=32,
            pc_npoints=8192,
            pc_num_group=512,
            pc_trans_dim=384,
            perceiver_attn_dropout=0.0,
            perceiver_cross_dim_head=64,
            perceiver_cross_heads=1,
            perceiver_depth=4,
            perceiver_ff_dropout=0.0,
            perceiver_fourier_encode_data=False,
            perceiver_input_axis=1,
            perceiver_input_chan=384,
            perceiver_latent_dim=1024,
            perceiver_latent_dim_head=64,
            perceiver_latent_heads=16,
            perceiver_num_latents=256,
            perceiver_self_per_cross_attn=1,
            perceiver_weight_tie_layers=True,
            ckpt_pth="/PATH_TO/vitlensL_pc.pt",
        ),
        audio=edict(
            visual_modality_type="audio",
            v_key="audio",
            use_perceiver=True,
            use_visual_adapter=True,
            audio_clip_duration=5.0,
            audio_sampling_rate=16000,
            audio_fstride=10,
            audio_tstride=10,
            audio_mel_bins=128,
            audio_target_length=512,
            perceiver_attn_dropout=0.0,
            perceiver_cross_dim_head=64,
            perceiver_cross_heads=1,
            perceiver_depth=2,
            perceiver_ff_dropout=0.0,
            perceiver_fourier_encode_data=False,
            perceiver_input_axis=1,
            perceiver_input_chan=1024,
            perceiver_latent_dim=1024,
            perceiver_latent_dim_head=64,
            perceiver_latent_heads=16,
            perceiver_num_latents=256,
            perceiver_self_per_cross_attn=3,
            perceiver_weight_tie_layers=False,
            ckpt_pth="/PATH_TO/vitlensL_audio.pt",
        ),
        depth=edict(
            visual_modality_type="depth",
            v_key="depth",
            use_perceiver=True,
            use_visual_adapter=True,
            perceiver_as_identity=True,
            ckpt_pth="/PATH_TO/vitlensL_depth.pt",
        ),
        tactile=edict(
            visual_modality_type="tactile",
            v_key="tactile",
            use_perceiver=False,
            use_visual_adapter=False,
            ckpt_pth="/PATH_TO/vitlensL_tactile.pt",
        ),
        eeg=edict(
            visual_modality_type="eeg",
            v_key="eeg",
            use_perceiver=True,
            use_visual_adapter=True,
            eeg_chans=128,
            eeg_stride=1,
            eeg_time_len=512,
            eeg_window_size=1,
            perceiver_as_transformer=False,
            perceiver_attn_dropout=0.0,
            perceiver_cross_dim_head=64,
            perceiver_cross_heads=1,
            perceiver_depth=1,
            perceiver_ff_dropout=0.0,
            perceiver_fourier_encode_data=False,
            perceiver_input_axis=1,
            perceiver_input_chan=1024,
            perceiver_latent_dim=1024,
            perceiver_latent_dim_head=64,
            perceiver_latent_heads=16,
            perceiver_max_freq=10.0,
            perceiver_self_per_cross_attn=1,
            perceiver_weight_tie_layers=False,
            ckpt_pth="/PATH_TO/vitlensL_eeg.pt",
        ),
    ),
    # config for ViT-Lens-B
    vitlensB=None,
)


def fetch_model_cfg(
    model_keys=["model", "pretrained"], modality="pc", model_option="vitlensL"
):
    base_cfg = deepcopy(default_cfg)
    model_cfg = vitlens_model_cfg[model_option]
    for k in model_keys:
        setattr(base_cfg, k, model_cfg[k])

    if not modality in ["image", "video", "text"]:
        modality_module_cfg = model_cfg[modality]
        base_cfg.update((modality_module_cfg))

    return base_cfg


if __name__ == "__main__":
    from pprint import pprint

    pprint(fetch_model_cfg(modality="audio"))

