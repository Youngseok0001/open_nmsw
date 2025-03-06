#
# TODO
# * ignore padded region before feeding into model


from typing import Callable, Any

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
import torch.nn as nn

from torchtyping import TensorType

from monai.transforms.transform import MapTransform
from monai.transforms import (
    ResizeD,
    CropForeground,
    SpatialPadd,
)

from model.backbones.generic import Seg3D
from model.registry import model_registry
from model.backbones.registry import backbone_registry
from data.transforms.transform import toBatch
from utils.visualzation import VisVolLab, ShapeGenerator
from utils.definitions import *

from matplotlib import pyplot as plt
from utils.system import load_weights


class SelectLastElement(nn.Module):
    def forward(self, x):
        return x[-1]


@model_registry.register
class GlobalSeg3D(nn.Module):

    # i need these later during model training/testing
    train_keys = [GLOBAL + "_"]
    valid_keys = train_keys
    test_keys = [""]

    def __init__(
        self,
        global_backbone_name: str,
        input_shape: list[int],  # before down-size
        output_shape: list[int],  # before down-size
        down_size_rate: list[float],
        ckpt_path: None | str = None,
        global_loss: dict[str, Any] = {
            "name": "DiceCELossMONAI",
        },
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        self.input_shape_before_ds = input_shape
        self.output_shape_before_ds = output_shape
        self.backbone_name = global_backbone_name
        self.down_size_rate = down_size_rate
        self.ckpt_path = ckpt_path

        self.input_shape_after_ds = input_shape[0:1] + list(
            (np.array(input_shape[1:]) // np.array(down_size_rate)).astype(int)
        )
        self.output_shape_after_ds = output_shape[0:1] + list(
            (np.array(output_shape[1:]) // np.array(down_size_rate)).astype(int)
        )

        self.backbone: Seg3D = nn.Sequential(
            backbone_registry[self.backbone_name](
                input_shape=input_shape,
                output_shape=output_shape,
                loss_args=global_loss,
            ),
            SelectLastElement(),
        )
        if self.ckpt_path != None:
            load_weights(self.backbone, self.ckpt_path, "net.backbone")

        self.down_size = toBatch(ResizeD)(
            keys=[VOL, LAB],
            spatial_size=self.input_shape_after_ds[1:],
            mode=("bilinear", "nearest"),
        )

        self.up_size = toBatch(ResizeD)(
            keys=[VOL, LOGIT],
            spatial_size=self.output_shape_before_ds[1:],
            mode=("bilinear", "bilinear"),
        )

        self._vis: VisVolLab = VisVolLab(num_classes=self.output_shape_after_ds[0])

        # crop padded region

    def train_step(self, input_d: dict[str, torch.tensor]) -> dict[str, Any]:
        # keys in input_d : VOL, LAB, CASE_ID

        global_input_d: dict[str, TensorType["N", "C", "H", "W", "D"]] = self.down_size(
            input_d
        )

        global_logit: TensorType["N", "C", "H", "W", "D"] = self.backbone(
            global_input_d[VOL]
        )

        loss: int = self.backbone[0].get_loss(global_logit, global_input_d[LAB])

        return {
            TOTAL_LOSS: loss,
            GLOBAL_LOGIT: global_logit,
            GLOBAL_VOL: get(VOL)(global_input_d),
            GLOBAL_LAB: get(LAB)(global_input_d),
            CASE_ID: get(CASE_ID)(global_input_d),
        }

    @torch.no_grad
    def test_step(self, input_d):

        # turn volume into patches
        global_input_d: dict[str, TensorType["N", "C", "H", "W", "D"]] = self.down_size(
            input_d
        )

        global_logit: TensorType["N", "C", "H", "W", "D"] = self.backbone(
            global_input_d[VOL]
        )

        up_size_d = self.up_size(
            {
                VOL: global_input_d[VOL],
                LOGIT: global_logit,
            }
        )

        return {
            VOL: up_size_d[VOL],
            LAB: input_d[LAB],
            LOGIT: up_size_d[LOGIT],
            CASE_ID: get(CASE_ID)(input_d),
        }

    def visualize_train(self, input_d: dict[str, TensorType["N", "C", "H", "W", "D"]]):

        vis_lab = self._vis.vis(
            vol=get(GLOBAL_VOL)(input_d),
            lab=get(GLOBAL_LAB)(input_d),
        )
        vis_pred = self._vis.vis(
            vol=get(GLOBAL_VOL)(input_d),
            lab=get(GLOBAL_LOGIT)(input_d).argmax(1, keepdim=True),
        )

        return {GLOBAL_LAB: vis_lab, GLOBAL_PRED: vis_pred}

    def visualize_test(self, input_d: dict[str, TensorType["N", "C", "H", "W", "D"]]):

        vis_lab = self._vis.vis(
            vol=get(VOL)(input_d),
            lab=get(LAB)(input_d),
        )
        vis_ped = self._vis.vis(
            vol=get(VOL)(input_d),
            lab=get(LOGIT)(input_d).argmax(1, keepdim=True),
        )

        return {LAB: vis_lab, PRED: vis_ped}

    valid_step = torch.no_grad(train_step)
    visualize_valid = visualize_train

    @classmethod
    def unit_test(cls, device="cuda"):

        global_backbone_name = "FasterUNet"
        shape_type = "checkerboard"
        input_shape = [1, 512, 512, 512]  # before donwsize
        output_shape = [10, 512, 512, 512]  # before downsize
        down_size_rate = [2, 2, 2]
        batch_size = 1

        shape_generator = ShapeGenerator(
            image_size=input_shape[1:],
            batch_size=batch_size,
            shape_type=shape_type,
        )

        global_seg_3d = cls(
            global_backbone_name=global_backbone_name,
            input_shape=input_shape,
            output_shape=output_shape,
            down_size_rate=down_size_rate,
        ).to(device)

        input_d = {
            VOL: torch.randn(batch_size, *input_shape).to(device),
            LAB: shape_generator.generate(
                num_classes=output_shape[0],
            ).to(device),
            CASE_ID: ["lol.nii"] * batch_size,
        }

        train_output = global_seg_3d.train_step(input_d)
        test_output = global_seg_3d.test_step(input_d)
        train_vis_d = global_seg_3d.visualize_train(train_output)
        test_vis_d = global_seg_3d.visualize_test(test_output)

        for name, img in train_vis_d.items():
            plt.title(TRAIN + "_" + name)
            plt.imshow(img.detach().cpu().numpy())
            plt.show()

        for name, img in test_vis_d.items():
            plt.title(TEST + "_" + name)
            plt.imshow(img.detach().cpu().numpy())
            plt.show()


if __name__ == "__main__":

    GlobalSeg3D.unit_test(device="cuda:1")
