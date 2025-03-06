from torchtyping import TensorType

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from monai.transforms import ResizeD, RandCropByPosNegLabelD, Compose, SelectItemsd
from monai.data.utils import list_data_collate

from utils.definitions import *
from utils.misc import get_patch_dim
from utils.system import load_weights
from utils.functional import list_op, div
from data.transforms.transform import Patchify, toBatch

from model.no_more_sw.blocks.learnable_wt import LearnablePatchAgg

from model.no_more_sw.blocks.sample_top_k_patch import SampleTopKPatch
from model.no_more_sw.blocks.bread_n_butter import ConvBnReLU, Interpolate
from model.backbones.registry import backbone_registry
from model.registry import model_registry
from model.sampler.registry import loss_sampler_registry
from model.loss.registry import loss_registry

from utils.visualzation import VisVolLab, ShapeGenerator
from matplotlib import pyplot as plt


def freeze(model):

    for param in model.parameters():
        param.requires_grad = False

    return model


@model_registry.register
class NSWNet3D(nn.Module):

    # i need these later during model training/testing
    train_keys = ["", PATCH + "_", GLOBAL + "_"]
    valid_keys = train_keys
    test_keys = ["", GLOBAL + "_"]

    epsilon = np.finfo(np.float32).tiny

    def __init__(
        self,
        input_shape: list[int] | tuple[int, int, int, int],
        output_shape: list[int] | tuple[int, int, int, int],
        patch_size: list[int] | tuple[int, int, int],
        down_size_rate: list[float] | tuple[float, float, float],
        local_backbone_name: str = "FasterUNet",
        global_backbone_name: str = "FasterUNet",
        overlap_r: float = 0.5,
        num_train_topk_patches: int = 3,  # keep it small during training and increase during testing.
        num_train_random_patches: int = 2,
        num_infernce_patches: int = 5,  # patches used during test/validation
        add_aggregation_module: bool = True,
        tau: float = 2 / 3,
        loss_num_patches: int = 5,
        loss_patch_size: list[int] = [128, 128, 128],
        local_ckpt_path: str = "",
        global_ckpt_path: str = "",
        entropy_multiplier: float = 0.005,
        local_loss: dict = {"name": "DiceCELossMONAI"},
        global_loss: dict = {"name": "DiceCELossMONAI"},
        agg_loss: dict = {
            "name": "DiceCELossMONAI",
        },
        loss_sampler: dict = {
            "name": "ToPatchLossV2",
            "num_patches": 3,
            "patch_size": [128, 128, 128],
        },
        *args,
        **kwargs,
    ):
        super().__init__()

        # base attrs
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.patch_size = patch_size
        self.down_size_rate = down_size_rate
        self.local_backbone_name = local_backbone_name
        self.global_backbone_name = global_backbone_name
        self.overlap_r = overlap_r
        self.num_train_topk_patches = num_train_topk_patches
        self.num_train_random_patches = num_train_random_patches
        self.num_infernce_patches = num_infernce_patches
        self.add_aggregation_module = add_aggregation_module
        # self.tau = tau
        self.local_ckpt_path = local_ckpt_path
        self.global_ckpt_path = global_ckpt_path
        self.loss_num_patches = loss_num_patches
        self.loss_patch_size = loss_patch_size
        self.entropy_multiplier = entropy_multiplier

        # derived attrs
        self.local_input_shape = self.input_shape[0:1] + self.patch_size
        self.local_output_shape = self.output_shape[0:1] + self.patch_size

        self.global_input_shape = input_shape[0:1] + list_op(
            compose(int, div), input_shape[1:], down_size_rate
        )

        self.global_output_shape = output_shape[0:1] + list_op(
            compose(int, div), output_shape[1:], down_size_rate
        )
        self.patch_shape: list[int] = get_patch_dim(
            input_shape[1:], patch_size, overlap_r
        )

        # methods
        assert global_backbone_name not in [
            "UNETRVit",
            "UNETRSwin",
        ], f"Please use convolutional model instead."
        self.local_backbone = backbone_registry[self.local_backbone_name](
            input_shape=self.local_input_shape,
            output_shape=self.local_output_shape,
            loss_args=local_loss,
        )

        self.global_backbone = backbone_registry[self.global_backbone_name](
            input_shape=self.global_input_shape,
            output_shape=self.global_output_shape,
            loss_args=global_loss,
        )

        # 2nd last
        pre_score_feature_shape = self.global_backbone.feature_shapes[-2]

        self.feature_to_logit = nn.Sequential(
            Interpolate(
                size=self.patch_shape,
                mode="trilinear",
            ),
            ConvBnReLU(
                pre_score_feature_shape[0],
                pre_score_feature_shape[0] // 2,
                1,
                0,
                1,
                repeat=2,
                norm=None,
            ),
            ConvBnReLU(
                pre_score_feature_shape[0] // 2,
                1,
                1,
                0,
                1,
                repeat=2,
                act=None,
                norm=None,
            ),
        )

        # learnable sampling
        self.sample_topk_patch = SampleTopKPatch(
            keys=[VOL, LAB],
            patch_size=self.patch_size,
            overlap_r=self.overlap_r,
            tau=tau,
        )

        self.downsize_d = toBatch(ResizeD)(
            keys=[VOL, LAB],
            spatial_size=self.global_input_shape[1:],
            mode=("bilinear", "nearest"),
        )

        self.get_aggreated_logit = LearnablePatchAgg(
            vol_size=self.input_shape[1:],
            patch_size=self.patch_size,
            down_size_rate=self.down_size_rate,
            num_classes=self.output_shape[0],
            add_aggregation_module=self.add_aggregation_module,
        )

        self.vis = VisVolLab(num_classes=output_shape[0])

        self.agg_loss_fn = loss_registry[agg_loss["name"]](**agg_loss)

    @property
    def tau(self):
        return self.sample_topk_patch.tau

    @tau.setter
    def tau(self, value):
        self.sample_topk_patch.tau = value

    def train_step(self, input_d):
        return self.common_step(
            num_topk_patches=self.num_train_topk_patches,
            input_d=input_d,
            mode=TRAIN,
        )

    @torch.no_grad()
    def valid_step(self, input_d):
        output_d = self.common_step(
            num_topk_patches=self.num_infernce_patches,
            input_d=input_d,
            mode=VALID,
        )
        return output_d

    @torch.no_grad()
    def test_step(self, input_d):
        output_d = self.common_step(
            num_topk_patches=self.num_infernce_patches,
            input_d=input_d,
            mode=TEST,
        )
        return output_d

    def common_step(
        self,
        num_topk_patches: int,
        input_d: dict[str, torch.tensor],
        mode: str,
    ):

        raw_input_d = keyfilter(lambda x: x in [VOL, LAB])(input_d)
        patch_input_d = keyfilter(lambda x: x in [PATCH_VOL, PATCH_LAB])(input_d)

        ###############GLOBAL##################
        global_input_d: dict[str, torch.tensor] = self.downsize_d(raw_input_d)
        global_input_d_vol = global_input_d[VOL]

        global_features: list[torch.tensor] = self.global_backbone(global_input_d_vol)

        global_segmentation_logit: TensorType["B", "1", "Hg", "Wg", "Dg"] = (
            global_features[-1]
        )

        # 2nd last
        pre_score = global_features[-2]
        # print(f"pre_score:{pre_score.shape}")

        # print(last(global_features).shape)
        ###############SAMPLE topk##################
        objectness_logit: TensorType["B", "1", "Hz", "Wz", "Dz"] = (
            self.feature_to_logit(pre_score)
        )
        # DEBUG
        objectness_logit = torch.clip(objectness_logit, max=70)

        background_mask: TensorType["B", "1", "Hz", "Wz", "Dz"] = (
            self.get_background_mask(global_segmentation_logit)
        )
        sampled_topk_local_patches_d, sample_logit, slice_meta = self.sample_topk_patch(
            raw_input_d,
            objectness_logit,
            background_mask.to(torch.float32),
            num_topk_patches,
            mode=mode,
        )
        ###############LOCAL##################
        if mode == TRAIN:
            sampled_random_local_patches_d = {
                VOL: patch_input_d[PATCH_VOL],
                LAB: patch_input_d[PATCH_LAB],
            }

            # concat top-k and random for faster training
            sampled_patches_d = merge_with(torch.cat)(
                sampled_topk_local_patches_d,
                sampled_random_local_patches_d,
            )

            # predict
            sampled_patches_features: TensorType["BN", "C", "Hp", "Wp", "Dp"] = (
                self.local_backbone(sampled_patches_d[VOL])
            )
            sampled_patches_logits = sampled_patches_features[-1]

            sampled_topk_patch_logits = sampled_patches_logits[
                : self.num_train_topk_patches
            ]

        if mode == VALID:
            # simulate per-patch inference for fair comparison with sliding_window
            # also use less memory
            sampled_topk_patch_logits = []
            for sampled_topk_local_vol in sampled_topk_local_patches_d[VOL]:
                sampled_topk_patch_features: TensorType["BN", "C", "Hp", "Wp", "Dp"] = (
                    self.local_backbone(sampled_topk_local_vol[None])
                )
                sampled_topk_patch_logit = sampled_topk_patch_features[-1]
                sampled_topk_patch_logits.append(sampled_topk_patch_logit[0])
            sampled_topk_patch_logits = list_data_collate(sampled_topk_patch_logits)
            sampled_patches_logits = sampled_topk_patch_logits
            sampled_patches_d = sampled_topk_local_patches_d

        if mode == TEST:
            # simulate per-patch inference for fair comparison with sliding_window
            # also use less memory
            sampled_topk_patch_logits = (
                self.local_backbone(sampled_topk_local_vol[None])[-1]
                for sampled_topk_local_vol in sampled_topk_local_patches_d[VOL]
            )

        #############AGGREGATION###############
        aggregated_logit = self.get_aggreated_logit(
            sampled_topk_patch_logits,
            global_segmentation_logit.detach(),
            slice_meta,
        )

        output_d = {
            CASE_ID: input_d[CASE_ID],
            # full-res
            VOL: raw_input_d[VOL],
            LAB: raw_input_d[LAB],
            LOGIT: aggregated_logit,
            # patch
            PATCH_VOL: sampled_topk_local_patches_d[VOL],
            PATCH_LAB: sampled_topk_local_patches_d[LAB],
            PATCH_LOGIT: sampled_topk_patch_logits,
            # global (low-res)
            GLOBAL_VOL: global_input_d[VOL],
            GLOBAL_LAB: global_input_d[LAB],
            GLOBAL_LOGIT: global_segmentation_logit,
            # aux
            "score": sample_logit,
        }

        if mode == TEST:  # do not compute loss and save some infernce time
            return output_d

        else:

            global_loss: int = self.global_backbone.get_loss(
                global_segmentation_logit,
                global_input_d[LAB],
            )
            local_loss: int = self.local_backbone.get_loss(
                sampled_patches_logits,
                sampled_patches_d[LAB],
            )
            agg_loss: int = self.agg_loss_fn(
                aggregated_logit,
                input_d[LAB],
            )

            B, *CHWD = sample_logit.shape

            sample_probability_flatten = F.softmax(sample_logit.flatten(1), dim=-1)
            sample_probability = sample_probability_flatten.view(B, *CHWD)

            entropy = (
                (-torch.log(sample_probability + 1e-20) * (sample_probability + 1e-20))
                .sum((1, 2, 3, 4))
                .mean(0)
            )

            total_loss = (
                agg_loss
                + global_loss
                + local_loss
                - (entropy * self.entropy_multiplier)
            )

            # DEBUG
            return output_d | {
                TOTAL_LOSS: total_loss,
                LOCAL_LOSS: local_loss,
                GLOBAL_LOSS: global_loss,
                "agg_loss": agg_loss,
                "entropy_loss": entropy,
                # auxiliary info for visualization
                "objectness_loss": self.get_objectness_score(sampled_patches_d[LAB]),
            }

    def forward(self, input_d):
        return self.common_step(
            self.num_infernce_patches,
            input_d,
            mode=TEST,
        )

    @torch.no_grad()
    def get_background_mask(
        self,
        global_segmentation_logit: TensorType["B", "C", "Hg", "Wg", "Dg"],
    ):

        B = global_segmentation_logit.shape[0]

        background_mask = (global_segmentation_logit.argmax(1) == 0).to(torch.float32)
        background_mask = torch.nn.functional.interpolate(
            background_mask.unsqueeze(1), self.patch_shape, mode="nearest"
        ).to(torch.float32)

        return background_mask.view(B, 1, *self.patch_shape)

    @torch.no_grad()
    def visualize(self, input_d, mode):

        # vis global
        vis_global_lab = self.vis.vis(
            vol=input_d[GLOBAL_VOL],
            lab=input_d[GLOBAL_LAB],
        )
        vis_global_pred = self.vis.vis(
            vol=input_d[GLOBAL_VOL],
            lab=input_d[GLOBAL_LOGIT].argmax(1, keepdim=True),
        )

        if not mode == TEST:
            # vis local
            vis_local_lab = self.vis.vis(
                vol=input_d[PATCH_VOL],
                lab=input_d[PATCH_LAB],
            )
            vis_local_pred = self.vis.vis(
                vol=input_d[PATCH_VOL],
                lab=input_d[PATCH_LOGIT].argmax(1, keepdim=True),
            )

        # vis agg
        vis_lab = self.vis.vis(
            vol=input_d[VOL],
            lab=input_d[LAB],
        )
        vis_pred = self.vis.vis(
            vol=input_d[VOL],
            lab=input_d[LOGIT].argmax(1, keepdim=True),
        )

        # # vis one_hot
        num_patches = input_d[PATCH_VOL].shape[0]
        one_hot_vis: TensorType["KB" "1", "Hg", "Wg", "Dg"] = torch.zeros(
            num_patches, 1, *self.input_shape[1:]
        )  # this is closer to the actual patch sampling location.

        for i, patch in enumerate(input_d[PATCH_VOL]):
            slices: list[slice] = eval(patch.meta["slice"])
            one_hot_vis[[i] + slices] = 1

        one_hot_vis: TensorType["K*B", "1", "H", "W", "D"] = (
            torch.nn.functional.interpolate(
                one_hot_vis,
                self.global_input_shape[1:],
                mode="nearest",
            )
        )

        vis_one_hot = self.vis.vis(
            vol=torch.cat([input_d[GLOBAL_VOL]] * len(one_hot_vis)),
            lab=one_hot_vis,
        )

        # vis probability
        prob: TensorType["B", "1", "*patch_shape"] = input_d["score"]
        vis_prob = self.vis.vis(vol=prob)

        return merge(
            {
                LAB: vis_lab,
                PRED: vis_pred,
                #
                GLOBAL_LAB: vis_global_lab,
                GLOBAL_PRED: vis_global_pred,
                #
                "one_hots": vis_one_hot,
                "score": vis_prob,
            }
            | (
                {
                    PATCH_LAB: vis_local_lab,
                    PATCH_PRED: vis_local_pred,
                }
                if mode != TEST
                else {}
            )
        )

    def visualize_train(self, input_d):
        return self.visualize(input_d, mode=TRAIN)

    def visualize_valid(self, input_d):
        return self.visualize(input_d, mode=VALID)

    def visualize_test(self, input_d):
        return self.visualize(input_d, mode=TEST)

    @classmethod
    def unit_test(cls, device="cuda:6"):

        from monai.transforms import (
            Compose,
            RandCropByPosNegLabelD,
            CopyItemsD,
            SelectItemsd,
        )

        input_shape = [1, 480, 480, 384]
        output_shape = [14, 480, 480, 384]
        patch_size = [128, 128, 128]
        down_size_rate = [3, 3, 3]
        overlap_r = 0.5
        add_aggregation_module = False
        batch_size = 1  # dont touch this
        num_train_topk_patches = 5
        num_train_random_patches = 2
        num_infernce_patches = 5
        local_ckpt_path = ""
        global_ckpt_path = ""

        shape_generator = ShapeGenerator(
            image_size=input_shape[1:],
            batch_size=batch_size,
            shape_type="checkerboard",
        )

        transform = RandCropByPosNegLabelD(
            keys=[VOL, LAB],
            label_key=LAB,
            spatial_size=patch_size,
            pos=1,
            neg=1,
            num_samples=2,
        )

        vol = torch.randn(batch_size, *input_shape, requires_grad=True).to(device)
        lab = shape_generator.generate(num_classes=output_shape[0]).to(device)
        patch_d = compose(list_data_collate, transform)({VOL: vol[0], LAB: lab[0]})

        input_d = {
            VOL: vol,
            LAB: lab,
            PATCH_VOL: patch_d[VOL],
            PATCH_LAB: patch_d[LAB],
            CASE_ID: ["lol.txt"] * batch_size,
        }

        net = cls(
            input_shape=input_shape,
            output_shape=output_shape,
            patch_size=patch_size,
            down_size_rate=down_size_rate,
            local_backbone_name="FasterUNet",
            global_backbone_name="FasterUNet",
            overlap_r=overlap_r,
            num_train_topk_patches=num_train_topk_patches,
            num_train_random_patches=num_train_random_patches,
            num_infernce_patches=num_infernce_patches,
            add_aggregation_module=add_aggregation_module,
            local_ckpt_path=local_ckpt_path,
            global_ckpt_path=global_ckpt_path,
        ).to(device)

        output_d = net.test_step(input_d)
        test_vis_d = net.visualize_test(output_d)

        output_d = net.train_step(input_d)
        train_vis_d = net.visualize_train(output_d)

        for name, img in train_vis_d.items():
            plt.title(TRAIN + "_" + name)
            plt.imshow(img.detach().cpu().numpy())
            plt.show()

        for name, img in test_vis_d.items():
            plt.title(TEST + "_" + name)
            plt.imshow(img.detach().cpu().numpy())
            plt.show()

        return output_d

    @staticmethod
    @torch.no_grad()
    def get_objectness_score(sampled_lab_patches: TensorType["B*n,1,H,W,D"]):
        """
        check the proportion of objectness
        """
        return (sampled_lab_patches > 0).sum() / (sampled_lab_patches > 0).numel()


if __name__ == "__main__":

    output_d = NSWNet3D.unit_test("cuda:1")
