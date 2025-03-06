import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.data.meta_tensor import MetaTensor

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from utils.functional import list_op


class Interpolate(nn.Module):
    def __init__(
        self,
        mode: str = "trilinear",
        size: None | list[int] | tuple[int, ...] = None,
        scale_factor: None | list[float] | tuple[float, ...] = None,
    ):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):

        x_resized = self.interp(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
        )

        if isinstance(x, MetaTensor):
            resize_rate = list_op(lambda x, y: x / y, x_resized.shape, x.shape)[2:]
            # get meta
            if "crop_center" in x.meta.keys():
                crop_centers: list = [
                    (r_r * c_i).to(torch.long)
                    for r_r, c_i in zip(resize_rate, x.meta["crop_center"])
                ]
                x_resized.meta["crop_center"] = crop_centers
            if "slice" in x.meta.keys():
                slices = [
                    [slice(None)]
                    + [
                        slice(int(_s.start * r), int(_s.stop * r))
                        for r, _s in zip(resize_rate, eval(s)[1:])
                    ]
                    for s in x.meta["slice"]
                ]
                x_resized.meta["slice"] = list(map(str, slices))

            return x_resized

        else:
            return x_resized


class ConvBnReLU(nn.Sequential):
    def __init__(
        self,
        in_chns: int,
        out_chns: int,
        kernel_size=1,
        padding=0,
        strides=1,
        spatial_dims: int = 3,
        act: str | tuple = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: str | tuple = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: float | tuple = 0.0,
        repeat: int = 1,
    ):
        super().__init__()

        for i in range(repeat):
            conv = Convolution(
                spatial_dims,
                in_chns if i == 0 else out_chns,
                out_chns,
                kernel_size=kernel_size,
                padding=padding,
                strides=strides,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
            )
            self.add_module(f"conv_{i}", conv)


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from data.transforms.transform import toBatch, Patchify, UnPatchify
    from utils.visualzation import VisVolLab, ShapeGenerator

    TEST_INTERPOLATE = False

    image_size = [128, 128, 128]
    scale_factor = [2, 2, 2]
    patch_size = [64, 64, 64]
    overlap_r = 0.5
    batch_size = 1

    vis = VisVolLab()

    vol = ShapeGenerator(
        image_size=image_size,
        batch_size=batch_size,
        shape_type="checkerboard",
    ).generate()

    patchify = toBatch(Patchify)(
        keys=["temp"],
        patch_size=patch_size,
        overlap_r=overlap_r,
    )

    unpatchify = UnPatchify(
        keys=["temp"],
        overlap_r=overlap_r,
    )

    interpolate = Interpolate(
        scale_factor=scale_factor,
        mode="trilinear",
    )

    if TEST_INTERPOLATE:

        vol_patches = patchify({"temp": vol})["temp"]
        out_vol_patches = interpolate(vol_patches)
        new_returned_data = unpatchify(
            {"temp": out_vol_patches[: len(out_vol_patches) // batch_size]}
        )["temp"]

        plt.imshow(vis.vis(lab=vol))
        plt.show()
        plt.imshow(vis.vis(lab=new_returned_data))
        plt.show()
