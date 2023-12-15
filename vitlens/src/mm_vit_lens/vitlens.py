import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import fsspec
from functools import partial
from open_clip.factory import tri_create_model_and_transforms
from open_clip.constants import PROJECT_DIR
from mm_vit_lens.model_cfg import fetch_model_cfg


def load_ckpt(file_path, map_location=None):
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out


class ViTLens(nn.Module):
    def __init__(
        self,
        model_var="vitlensL",
        modality_loaded=["image", "text", "pc", "depth", "audio", "tactile", "eeg"],
        load_from_ckpt=os.path.join(PROJECT_DIR, "model_release"),
    ):
        super().__init__()
        self.model_var = model_var
        self.modality_loaded = modality_loaded

        self.processors = dict()
        self.vitlens = nn.ModuleDict()

        self.init_processors_and_model(load_from_ckpt=load_from_ckpt)

    def _init_modality_module(self, modality, load_from_pt_flag=False):
        cfg = fetch_model_cfg(modality=modality, model_option=self.model_var)
        model, _, image_process_val = tri_create_model_and_transforms(
            cfg.model,
            cfg.pretrained,
            precision=cfg.precision,
            device=cfg.device,
            jit=False,
            force_quick_gelu=cfg.force_quick_gelu,
            force_custom_text=cfg.force_custom_text,
            force_patch_dropout=False,
            force_image_size=cfg.force_image_size,
            pretrained_image=cfg.pretrained_image,
            load_ckpt_strict=False,
            image_mean=None,
            image_std=None,
            aug_cfg=None,
            output_dict=True,
            cache_dir=cfg.cache_dir,
            args=cfg,
        )

        if modality == "image":
            self.vitlens.add_module("image", model.image)
            self.processors["image"].set_image_transform(image_process_val)

        elif modality == "text":
            if hasattr(model, "text"):
                self.vitlens.add_module("text", model.text)

            else:
                self.vitlens.add_module("text", nn.ModuleDict())
                self.vitlens.text.add_module("transformer", model.transformer)
                self.vitlens.text.context_length = model.context_length
                self.vitlens.text.vocab_size = model.vocab_size
                self.vitlens.text.add_module("token_embedding", model.token_embedding)
                self.vitlens.text.positional_embedding = model.positional_embedding
                self.vitlens.text.add_module("ln_final", model.ln_final)
                self.vitlens.text.text_projection = model.text_projection
                self.vitlens.text.register_buffer(
                    "attn_mask", model.attn_mask, persistent=False
                )

                def encode_text(module, text):
                    cast_dtype = module.transformer.get_cast_dtype()
                    x = module.token_embedding(text).to(
                        cast_dtype
                    )  # [batch_size, n_ctx, d_model]
                    x = x + module.positional_embedding.to(cast_dtype)
                    x = x.permute(1, 0, 2)  # NLD -> LND
                    x = module.transformer(x, attn_mask=module.attn_mask)
                    x = x.permute(1, 0, 2)  # LND -> NLD
                    x = module.ln_final(x)  # [batch_size, n_ctx, transformer.width]
                    # take features from the eot embedding (eot_token is the highest number in each sequence)
                    x = (
                        x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
                        @ module.text_projection
                    )
                    return x

                self.vitlens.text.forward = partial(encode_text, self.vitlens.text)

        else:  # for modalities except image and text
            self.vitlens.add_module(modality, model.visual)

            if load_from_pt_flag:  # load from modality specific pt weights
                self.load_modality_from_pt_ckpt(
                    modality=modality, pt_ckpt_path=cfg.ckpt_pth
                )

        del model

    def init_processors_and_model(self, load_from_ckpt=None):
        # init processors
        from mm_vit_lens.data_processors import get_vitlens_processors_cls

        processors = get_vitlens_processors_cls()[self.model_var]()
        for m in self.modality_loaded:
            self.processors[m] = processors[m]

        # init model
        for m in self.modality_loaded:
            self._init_modality_module(m)

        if load_from_ckpt:
            ckpt_path = os.path.join(load_from_ckpt, f"{self.model_var}.pt")
            if not os.path.exists(ckpt_path):
                print(f"Downloading ViT-Lens weights to {ckpt_path} ...")
                os.makedirs(load_from_ckpt, exist_ok=True)
                torch.hub.download_url_to_file(
                    f"https://huggingface.co/TencentARC/ViT-Lens/resolve/main/{self.model_var}.pt",
                    ckpt_path,
                    progress=True,
                )
            ckpt = load_ckpt(ckpt_path)
            msg = self.load_state_dict(ckpt["state_dict"], strict=False)
            logging.info(msg)

    def load_modality_from_pt_ckpt(self, modality, pt_ckpt_path):
        """
        Load weights from pre-trained ckpt of each modality encoder
        Use this function to load your own checkpoint
        """
        checkpoint = load_ckpt(pt_ckpt_path, map_location="cpu")
        sd = checkpoint["state_dict"]

        # deal with distributed ckpt during pt
        if next(iter(sd.items()))[0].startswith("module."):
            sd = {k[len("module.") :]: v for k, v in sd.items()}

        # only load `visual` tower
        sd = {k[len("visual.") :]: v for k, v in sd.items() if k.startswith("visual.")}

        msg = self.vitlens[modality].load_state_dict(sd, strict=False)
        print(f"[Load ViT-Lens from `{modality}` Pretrained ckpt] : {msg}.")

    def export_checkpoint(self, save_path="model_release/vitlens.pt"):
        model_dict = dict(
            model_var=self.model_var,
            modality_loaded=self.modality_loaded,
            state_dict=self.state_dict(),
        )
        torch.save(model_dict, save_path)

    @property
    def device(self):
        return list(self.parameters())[0].device

    def reduce_list(self, modality):
        return modality in [
            "audio",
        ]

    def encode(self, input_dict, normalize=True):
        output_dict = dict()
        for m in input_dict:
            x = self.processors[m](input_dict[m], device=self.device)

            if self.reduce_list(m):
                B, S = x.shape[:2]
                x = einops.rearrange(x, "B S ... -> (B S) ...")

            features = self.vitlens[m](x)

            if self.reduce_list(m):
                features = einops.rearrange(features, "(B S) ... -> B S ...", B=B, S=S)
                features = features.mean(dim=1)

            features = F.normalize(features, dim=-1) if normalize else features

            output_dict[m] = features

        return output_dict
