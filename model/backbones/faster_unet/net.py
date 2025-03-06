"""
modified from: https://github.com/Project-MONAI/MONAI/blob/59a7211070538586369afd4a01eca0a7fe2e742e/monai/networks/nets/unet.py#L28-L301
"""

from __future__ import annotations

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from itertools import zip_longest

import warnings
from collections.abc import Sequence

import torch
import torch.nn as nn

from model.backbones.generic import Seg3D
from model.backbones.blocks import ConvUnit, Convolution
from monai.networks.layers.factories import Act, Norm, LayerFactory

from model.backbones.registry import backbone_registry


@backbone_registry.register
class FasterUNet(Seg3D):
    def __init__(
        self,
        input_shape: Sequence[int],
        output_shape: Sequence[int],
        channels: Sequence[int] = [32, 64, 128, 256, 320, 320],
        strides: Sequence[int] = [2, 2, 2, 2, 2],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 2,
        act: str | tuple | LayerFactory = Act.PRELU,
        norm: str | tuple | LayerFactory = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        loss_args={
            "name": "DiceCELossNNUNET",
        },
        # set it to true to have down-size factor 16. otherwise 32. which is too large for the proposed differentiable patch smapling.
    ):

        super().__init__(input_shape, output_shape, loss_args=loss_args)

        spatial_dims = self.spatial_dims
        in_channels = input_shape[0]
        out_channels = output_shape[0]

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError(
                "the length of `strides` should equal to `len(channels) - 1`."
            )
        if delta > 0:
            warnings.warn(
                f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used."
            )
        if isinstance(kernel_size, Sequence) and len(kernel_size) != spatial_dims:
            raise ValueError(
                "the length of `kernel_size` should equal to `dimensions`."
            )
        if isinstance(up_kernel_size, Sequence) and len(up_kernel_size) != spatial_dims:
            raise ValueError(
                "the length of `up_kernel_size` should equal to `dimensions`."
            )

        self.encoder = Encoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            encoder_dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )

        self.decoder = Decoder(
            spatial_dims=spatial_dims,
            out_channels=out_channels,
            channels=channels[::-1],  # going from large to small. hence reverse
            strides=strides,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            act=act,
            norm=norm,
            decoder_dropout=dropout,
            bias=bias,
            adn_ordering=adn_ordering,
        )


class Encoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        channels: Sequence[int] = [32, 64, 128, 256, 512],
        strides: Sequence[int] = [2, 2, 2, 2],
        kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 2,
        act: str = Act.PRELU,
        norm: str = Norm.INSTANCE,
        encoder_dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ):
        super().__init__()

        c_ins = [in_channels] + channels[:-1]
        c_outs = channels

        self.num_encoders: int = len(channels)  # including bottleneck

        for i, (c_in, c_out, s) in enumerate(
            zip_longest(c_ins, c_outs, strides, fillvalue=1)
        ):
            self.add_module(
                f"down_{i}",
                Down(
                    spatial_dims,
                    c_in,
                    c_out,
                    kernel_size,
                    s,
                    num_res_units=num_res_units,
                    act=act,
                    norm=norm,
                    dropout=encoder_dropout,
                    bias=bias,
                    adn_ordering=adn_ordering,
                ),
            )

    def forward(self, x):

        feas = []
        for i in range(self.num_encoders):
            x = eval(f"self.down_{i}")(x)
            feas.append(x)

        return feas


class Decoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        out_channels: int,
        channels: Sequence[int] = (512, 256, 128, 64, 32),
        strides: Sequence[int] = (2, 2, 2, 2),
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 2,
        act: str = Act.PRELU,
        norm: str = Norm.INSTANCE,
        decoder_dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ):

        super().__init__()

        c_ins = [channels[0] + channels[1]] + compose(list, map(mul(2)))(channels[2:])
        c_outs = channels[2:] + [out_channels]

        self.num_decoders: int = len(channels) - 1  # including bottleneck

        for i, (c_in, c_out, s) in enumerate(zip(c_ins, c_outs, strides)):
            self.add_module(
                f"up_{i}",
                Up(
                    spatial_dims,
                    c_in,
                    c_out,
                    s,
                    up_kernel_size,
                    num_res_units,
                    act,
                    norm,
                    dropout=decoder_dropout,
                    bias=bias,
                    adn_ordering=adn_ordering,
                    is_top=(i == self.num_decoders - 1),
                ),
            )

    def forward(self, features):

        x0, x1, x2, x3, x4, x5 = features
        u4 = self.up_0(x5, x4)
        u3 = self.up_1(u4, x3)
        u2 = self.up_2(u3, x2)
        u1 = self.up_3(u2, x1)
        logit = self.up_4(u1, x0)
        return [*features, u4, u3, u2, u1, logit]


class Down(nn.Sequential):
    """strided-conv + conv"""

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | list[int],
        strides: int | list[int] = 2,
        num_res_units: int = 2,
        act: str | tuple = "PRELU",
        norm: str | tuple = "INSTANCE",
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ):

        super().__init__()

        if num_res_units > 0:
            self.down: nn.Module = ConvUnit(
                dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=kernel_size,
                subunits=num_res_units,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
                adn_ordering=adn_ordering,
            )
        else:
            self.down: nn.Module = Convolution(
                dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=kernel_size,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
                adn_ordering=adn_ordering,
            )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    """
    transposed_conv + conv
    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        strides: int = 2,
        kernel_size: int = 3,
        num_res_units: int = 2,
        act: str = Act.PRELU,
        norm: str = Norm.INSTANCE,
        dropout: float = 0.0,
        bias=True,
        adn_ordering: str = "NDA",
        is_top=False,
    ):
        super().__init__()

        conv: nn.Module = Convolution(
            dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
            conv_only=is_top and num_res_units == 0,
            is_transposed=True,
            adn_ordering=adn_ordering,
        )

        if num_res_units > 0:
            ru: nn.Module = ConvUnit(
                dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                subunits=1,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
                last_conv_only=is_top,
                adn_ordering=adn_ordering,
            )
            self.up: nn.Sequential = nn.Sequential(conv, ru)

    def forward(self, x: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        return self.up(torch.cat([x, x_e], axis=1))


if __name__ == "__main__":

    from pprint import pprint

    TEST_BLOCK = False
    TEST_NET = True

    device = "cuda:5"

    if TEST_BLOCK:
        fea = torch.randn(1, 256, 8, 8, 8).to(device)
        down = Down(3, 256, 256, 3).to(device)
        up = Up(3, 256, 256, 2, 3).to(device)
        print(up(down(fea), fea).shape)

    if TEST_NET:
        net = FasterUNet(
            input_shape=[3, 128, 128, 128],
            output_shape=[5, 128, 128, 128],
        ).to(device)
        net.unit_test()
        pprint(net.feature_shapes)
