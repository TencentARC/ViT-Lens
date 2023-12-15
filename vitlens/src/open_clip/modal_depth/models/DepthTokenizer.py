import torch
import torch.nn as nn

from open_clip.util.Sample import Sample


class DepthTokenizer(nn.Module):
    def __init__(self, grid_size, patch_size, width, input_patchnorm):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.width = width  # transformer dim
        self.input_patchnorm = input_patchnorm

        # tokenize component
        if self.input_patchnorm:
            patch_input_dim = patch_size[0] * patch_size[1] * 1
            self.patchnorm_pre_ln = LayerNorm(patch_input_dim)
            self.conv1 = nn.Linear(patch_input_dim, width)
        else:
            self.patchnorm_pre_ln = nn.Identity()
            self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=width,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            )

        scale = width**-0.5
        self.pos_emb = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1], width)
        )

    def forward(self, x):
        if self.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.grid_size[0],
                self.patch_size[0],
                self.grid_size[1],
                self.patch_size[1],
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(x.shape[0], self.grid_size[0] * self.grid_size[1], -1)
            x = self.patchnorm_pre_ln(x)
            x = self.conv1(x)
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        return Sample(
            {
                "x": x,
                "pos": self.pos_emb,
            }
        )
