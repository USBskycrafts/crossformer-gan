import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F


class ConvShuffle(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int,
                 stride: int, padding):
        super(ConvShuffle, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride

        self.sequential = nn.Sequential(
            nn.Conv2d(input_dim, output_dim *
                      stride ** 4, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.PixelShuffle(stride ** 2),
            nn.InstanceNorm2d(output_dim),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, output_size):
        x = self.sequential(x)
        # print("start----------------")
        # print(x.shape)
        bs, c, h, w = x.size()
        bs_, c_, h_, w_ = output_size
        assert bs == bs_ and c == c_
        diff_x = w_ - w
        diff_y = h_ - h
        padding_size = (
            diff_x // 2,
            diff_x - diff_x // 2,
            diff_y // 2,
            diff_y - diff_y // 2
        )
        # print(padding_size)
        x = F.pad(x, padding_size, mode='reflect')
        # print(x.shape)
        # print("end----------------")
        assert x.shape == output_size, f"{x.shape} != {output_size}"
        return x


class ConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int,
                 stride: int, padding: int):
        super(ConvBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.sequential = nn.Sequential(
            nn.Conv2d(input_dim, output_dim,
                      kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(output_dim),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.sequential(x)


class CrossScaleEmbedding(nn.Module):
    # Should be used in pairs
    def __init__(self, input_dim: int, output_dim: int,
                 kernel_size: List[int] = [2, 4],
                 stride: int = 2, reversed: bool = False):
        super(CrossScaleEmbedding, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = [k for k in sorted(kernel_size)]
        self.stride = stride
        self.reversed = reversed
        self.convs = nn.ModuleList()

        if not reversed:
            token_size = self.token_size(self.kernel_size, output_dim)
            self.dim_list = token_size
            for i, k in enumerate(self.kernel_size):
                self.convs.append(
                    ConvBlock(input_dim, token_size[i],
                              kernel_size=k, stride=stride, padding=self.padding_size(k, stride)))
        else:
            token_size = self.token_size(self.kernel_size, input_dim)
            self.dim_list = token_size
            for i, k in enumerate(self.kernel_size):
                self.convs.append(
                    # Warning: may cause error if H and W are not even
                    ConvShuffle(2 * token_size[i], output_dim,
                                kernel_size=k, stride=stride, padding=self.padding_size(k, stride)))
            self.compress = nn.Conv2d(output_dim * len(self.kernel_size),
                                      output_dim, kernel_size=1)

    def token_size(self, kernel_size, output_dim) -> List[int]:
        token_dim = []
        for i in range(1, len(kernel_size)):
            token_dim.append(output_dim // (2**i))
            # the largest token dim should equals to the
            # secondary largest token dim
        token_dim.append(output_dim // (2**(len(kernel_size) - 1)))
        return token_dim

    def padding_size(self, kernel_size, stride) -> int:
        """Calculate padding size for convolution

        Args:
            kernel_size (_type_): _description_
            stride (_type_): _description_

        Returns:
            _type_: _description_
        while dilation=1,
        y.shape = (x.shape + 2 * p.shape - k.shape) // stride + 1
        if we want y.shape = x.shape // stride
        then we get this function
        """
        if (kernel_size - stride) % 2 == True:
            return (kernel_size - stride) // 2
        else:
            return (kernel_size - stride + 1) // 2

    def forward(self, input: List):
        if not self.reversed:
            x = input
            # from [B, C, H, W] to [B, H // stride, W // stride, C * stride]
            tokens = torch.cat([conv(x)
                                for conv in self.convs], dim=1)
            # a recursion to the deep layers
            return tokens
        else:
            x, y, input_size = input
            bs, c, h, w = input_size
            c = self.output_dim
            assert isinstance(y, torch.Tensor)
            assert isinstance(input_size, torch.Size)
            assert x.shape == y.shape
            features = torch.concat([x, y], dim=1)
            offset = 0
            output = []
            for i, d in enumerate(self.dim_list):
                output.append(self.convs[i](features
                                            [:, offset:offset + 2 * d, :, :],
                                            output_size=(bs, c, h, w)))
                offset = offset + 2 * d
            output = self.compress(torch.cat(output, dim=1))
            return output
