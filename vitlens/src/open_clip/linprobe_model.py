import logging
import torch
import torch.nn as nn
import fsspec
from .factory import tri_create_model_and_transforms


def pt_load(file_path, map_location=None):
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out


class ViTLensLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        model, _, _ = tri_create_model_and_transforms(
            args.model,
            args.pretrained,
            precision=args.precision,
            device="cpu",
            jit=False,
            force_quick_gelu=args.force_quick_gelu,
            force_custom_text=args.force_custom_text,
            force_patch_dropout=False,
            force_image_size=args.force_image_size,
            pretrained_image=args.pretrained_image,
            load_ckpt_strict=False,
            image_mean=None,
            image_std=None,
            aug_cfg=None,
            output_dict=True,
            cache_dir=args.cache_dir,
            args=args,
        )

        self.backbone = model.visual  # refer to open_clip transformer.py for forward

        enable_vit_proj = args.lp_enable_vit_proj
        lp_input_dim = None
        if enable_vit_proj:
            lp_input_dim = self.backbone.output_dim
        else:
            self.backbone.proj = (
                None  # disable proj, in transformer.py impl, directly return `pooled`
            )
            lp_input_dim = self.backbone.width

        self.lp_head = nn.Sequential(
            nn.Dropout(args.lp_dropout_rate),
            nn.BatchNorm1d(lp_input_dim, affine=False, eps=1e-6),
            nn.Linear(lp_input_dim, args.lp_num_classes),
        )

    def forward(self, x, **kwargs):
        x = self.backbone(x, **kwargs)
        x = self.lp_head(x)
        return x

    def lp_lock_parameters(self):
        for _, p in self.named_parameters():
            p.requires_grad = False

        for _, p in self.lp_head.named_parameters():
            p.requires_grad = True

    def load_vitlens_weights_from_ckpt(self, args):
        checkpoint = pt_load(args.lp_ckpt_path, map_location="cpu")
        sd = checkpoint["state_dict"]

        # deal with distributed ckpt during pt
        if next(iter(sd.items()))[0].startswith("module."):
            sd = {k[len("module.") :]: v for k, v in sd.items()}

        # only load `visual` tower
        sd = {k[len("visual.") :]: v for k, v in sd.items() if k.startswith("visual.")}

        msg = self.backbone.load_state_dict(sd, strict=False)
        logging.info(f"[Linear Probe load ViT-Lens Pretrained ckpt] : {msg}.")
