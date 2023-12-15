import torch
import torch.nn as nn

from open_clip.util.Sample import Sample


class PatchEmbed1D(nn.Module):
    def __init__(self, time_len=512, in_chans=128, window_size=4, stride=2, width=768):
        super().__init__()

        self.time_len = time_len
        self.in_chans = in_chans
        self.width = width  # transformer dim

        # tokenize component
        self.proj = nn.Conv1d(
            in_channels=in_chans,
            out_channels=width,
            kernel_size=window_size,
            stride=stride,
        )

        self.num_patches = self.get_tokenized_dim()

        scale = width**-0.5
        self.pos_emb = nn.Parameter(scale * torch.randn(self.num_patches, width))

    def get_tokenized_dim(self):
        with torch.no_grad():
            test_inp = torch.randn(1, self.in_chans, self.time_len)
            test_outp = self.proj(test_inp)
            np = test_outp.shape[-1]
        return np

    def forward(self, x):
        x = self.proj(x).transpose(1, 2).contiguous()
        return Sample(
            {
                "x": x,
                "pos": self.pos_emb,
            }
        )
