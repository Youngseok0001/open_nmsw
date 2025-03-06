# self.ccp: MapTransform = toBatch(KeepLargestConnectedComponentD)(
#     keys=["temp"]
# )  # sorry toBatch assumes input to be of type MapTransform


from typing import Callable, Any

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
import torch.nn as nn

from torchtyping import TensorType

from monai.transforms import (
    RandCropByPosNegLabelD,
    Compose,
)
from monai.inferers.inferer import SlidingWindowInferer

from model.backbones.generic import Seg3D
from model.registry import model_registry
from model.backbones.registry import backbone_registry
from data.transforms.transform import toBatch
from utils.visualzation import VisVolLab, ShapeGenerator
from utils.definitions import *
from utils.system import load_weights

from matplotlib import pyplot as plt

from monai.utils import BlendMode


class SelectLastElement(nn.Module):
    def forward(self, x):
        return x[-1]


@model_registry.register
class LocalSeg3D(nn.Module):

    train_keys = [PATCH + "_"]
    valid_keys = train_keys
    test_keys = [""]

    def __init__(
        self,
        local_backbone_name: str,  #
        input_shape: list[int],
        output_shape: list[int],
        overlap_r: list[float],  # [0.5, 0.5, 0.5]
        patch_size: list[int],  # [128, 128, 182]
        pos: int = 2,
        neg: int = 1,
        num_patches: int = 5,
        sigma_scale: float = 0.125,
        ckpt_path: None | str = None,
        local_loss: dict[str, Any] = {
            "name": "DiceCELossMONAI",
        },
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.backbone_name = local_backbone_name
        self.overlap_r = overlap_r
        self.patch_size = patch_size
        self.pos = pos
        self.neg = neg
        self.num_patches = num_patches
        self.ckpt_path = ckpt_path
        self.sigma_scale = sigma_scale

        self.backbone: Seg3D = nn.Sequential(
            backbone_registry[self.backbone_name](
                input_shape=[self.input_shape[0]] + patch_size,
                output_shape=[self.output_shape[0]] + patch_size,
                loss_args=local_loss,
            ),
            SelectLastElement(),
        )

        if self.ckpt_path != None:
            load_weights(self.backbone, self.ckpt_path, "net.backbone")

        self.extract_rand_patch = toBatch(RandCropByPosNegLabelD)(
            keys=[VOL, LAB],
            label_key=LAB,
            spatial_size=patch_size,
            pos=self.pos,
            neg=self.neg,
            num_samples=self.num_patches,
            image_key=VOL,
            image_threshold=-0.5,  # something larger than -1 and smaller than 0
        )

        self.sw_infer = SlidingWindowInferer(
            roi_size=self.patch_size,
            overlap=self.overlap_r,
            sigma_scale=self.sigma_scale,
            mode=BlendMode.GAUSSIAN,
        )

        self._vis: VisVolLab = VisVolLab(num_classes=self.output_shape[0])

    def train_step(self, input_d: dict[str, torch.tensor]) -> dict[str, Any]:

        input_d = keyfilter(lambda k: PATCH not in k)(input_d)

        # turn volume into patches
        patches_d: dict[str, TensorType["B*N", "C", "Hp", "Wp", "Dp"]] = (
            self.extract_rand_patch(input_d)
        )

        logit_patches: TensorType["B*N", "C", "Hp", "Wp", "Dp"] = self.backbone(
            patches_d[VOL]
        )

        loss: int = self.backbone[0].get_loss(logit_patches, patches_d[LAB])

        return {
            TOTAL_LOSS: loss,
            PATCH_LOGIT: logit_patches,
            PATCH_VOL: patches_d[VOL],
            PATCH_LAB: patches_d[LAB],
            CASE_ID: patches_d[CASE_ID],
        }

    @torch.no_grad
    def test_step(self, input_d: dict[str, TensorType["B", "C", "H", "W", "D"]]):

        logit: TensorType["B", "C", "H", "W", "D"] = self.sw_infer(
            input_d[VOL], self.backbone
        )

        return {
            LOGIT: logit,
            VOL: input_d[VOL],
            LAB: input_d[LAB],
            CASE_ID: input_d[CASE_ID],
        }

    def visualize_train(self, input_d: dict[str, torch.tensor]):

        vis_lab = self._vis.vis(
            vol=input_d[PATCH_VOL],
            lab=input_d[PATCH_LAB],
        )
        vis_ped = self._vis.vis(
            vol=input_d[PATCH_VOL],
            lab=input_d[PATCH_LOGIT].argmax(1, keepdim=True),
        )

        return {PATCH_LAB: vis_lab, PATCH_PRED: vis_ped}

    def visualize_test(self, input_d: dict[str, torch.tensor]):

        vis_lab = self._vis.vis(
            vol=input_d[VOL],
            lab=input_d[LAB],
        )
        vis_pred = self._vis.vis(
            vol=input_d[VOL],
            lab=input_d[LOGIT].argmax(1, keepdim=True),
        )

        return {LAB: vis_lab, PRED: vis_pred}

    valid_step = torch.no_grad(train_step)
    visualize_valid = visualize_train

    @classmethod
    def unit_test(cls, device="cuda"):

        local_backbone_name = "FasterUNet"
        input_shape = [1, 256, 256, 256]
        output_shape = [10, 256, 256, 256]
        overlap_r = [0.5, 0.5, 0.5]
        patch_size = [128, 128, 128]
        batch_size = 2

        shape_generator = ShapeGenerator(
            image_size=input_shape[1:],
            batch_size=batch_size,
            shape_type="checkerboard",
        )

        local_seg_3d = cls(
            local_backbone_name=local_backbone_name,
            input_shape=input_shape,
            output_shape=output_shape,
            overlap_r=overlap_r,
            patch_size=patch_size,
        ).to(device)

        input_d = {
            VOL: torch.randn(batch_size, *input_shape).to(device),
            LAB: shape_generator.generate(
                num_classes=output_shape[0],
            ).to(device),
            CASE_ID: ["lol.txt"] * batch_size,
        }

        train_output = local_seg_3d.train_step(input_d)
        test_output = local_seg_3d.test_step(input_d)

        train_vis_d = local_seg_3d.visualize_train(train_output)
        test_vis_d = local_seg_3d.visualize_test(test_output)

        for name, img in train_vis_d.items():
            plt.title(TRAIN + "_" + name)
            plt.imshow(img.detach().cpu().numpy())
            plt.show()

        for name, img in test_vis_d.items():
            plt.title(TEST + "_" + name)
            plt.imshow(img.detach().cpu().numpy())
            plt.show()


if __name__ == "__main__":

    LocalSeg3D.unit_test(device="cuda:1")
