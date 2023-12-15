import torch
import torch.nn as nn

from open_clip.util.Sample import Sample


class AST_tokenizer(nn.Module):
    def __init__(
        self, fstride, tstride, input_fdim, input_tdim, patch_size=(16, 16), width=768
    ):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.fstride = fstride
        self.tstride = tstride
        self.input_fdim = input_fdim
        self.input_tdim = input_tdim
        self.width = width  # transformer dim

        # tokenize component
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=width,
            kernel_size=patch_size,
            stride=(fstride, tstride),
            bias=False,
        )

        self.fdim, self.tdim = self.get_tokenized_dim()
        self.num_patches = self.fdim * self.tdim

        scale = width**-0.5
        self.pos_emb = nn.Parameter(scale * torch.randn(self.num_patches, width))

    def get_tokenized_dim(self):
        with torch.no_grad():
            test_inp = torch.randn(1, 1, self.input_fdim, self.input_tdim)
            test_outp = self.conv1(test_inp)
            fd = test_outp.shape[2]
            td = test_outp.shape[3]
        return fd, td

    def forward(self, x):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)
        x = self.conv1(x)  # shape = [*, width, fdim, tdim]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, fdim * tdim]
        x = x.permute(0, 2, 1)  # shape = [*, fdim * tdim, width]

        return Sample(
            {
                "x": x,
                "pos": self.pos_emb,
            }
        )
