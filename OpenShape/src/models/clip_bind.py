import os
import logging
import torch
import torch.nn as nn
from open_clip import tri_create_model_and_transforms
from typing import Callable, Optional, Sequence, Tuple

class CLIPBindWrap(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args 
        model, _, _ = tri_create_model_and_transforms(
            args.clip_model,
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
            args=args
        )
        
        self.backbone = model.visual
        self.proj_layer = nn.Identity()
        
        if hasattr(self.backbone, "proj"):
            if isinstance(self.backbone.proj, nn.Parameter):
                if self.backbone.proj.shape[-1] != args.model.out_channel:
                    self.backbone.proj = None
                    self.proj_layer = nn.Linear(self.backbone.transformer.width, args.model.out_channel)
            
            elif isinstance(self.backbone.proj, nn.Linear):
                if self.backbone.proj.weight.shape[-1] != args.model.out_channel:
                    self.backbone.proj = None
                    self.proj_layer = nn.Linear(self.backbone.transformer.width, args.model.out_channel)
        elif hasattr(self.backbone, "eva_vit_proj"):
            self.backbone.eva_vit_proj = None
            self.proj_layer = nn.Linear(self.backbone.eva_vit.embed_dim, args.model.out_channel)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False, unlock_cls=False, unlock_trans_first_n_layers=None):
        self.backbone.lock(unlocked_groups, freeze_bn_stats, unlock_cls, unlock_trans_first_n_layers)
        
        # unlock some specific groups
        groups = []
        
        # append cls
        append_cls = False
        if self.args.skip_trans_first_n_layers is not None:
            if self.args.skip_trans_first_n_layers > 0:                 # if skip some layers, unlock cls embed for training
                append_cls = True
        if self.args.unlock_cls:
            append_cls = True
        
        if append_cls:
            for possible_attr in ["class_embedding", "cls_token"]:
                if hasattr(self.backbone, possible_attr):
                    groups.append(getattr(self.backbone, possible_attr))
                
        def _unlock(x):
            if isinstance(x, Sequence):
                for g in x:
                    _unlock(g)
            else:
                if isinstance(x, torch.nn.Parameter):
                    x.requires_grad = True
                else:
                    for p in x.parameters():
                        p.requires_grad = True 
        
        _unlock(groups)
        

    def forward(self, x, **kwargs):
        x = self.backbone(x, **kwargs)
        x = self.proj_layer(x)
        return x