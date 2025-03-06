from time import perf_counter
from git import WorkTreeRepositoryUnsupported
from torchtyping import TensorType

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from itertools import starmap

import numpy as np
import torch
from torch import nn, sigmoid
from torch.nn import functional as F

from monai.data.utils import compute_importance_map
from monai.data.utils import list_data_collate
from monai.data.meta_tensor import MetaTensor

from model.no_more_sw.blocks.bread_n_butter import ConvBnReLU, Interpolate
from data.transforms.transform import RandomObjectPatchSampling, toBatch
from utils.definitions import *


class LearnablePatchAgg(nn.Module):

    def __init__(
        self,
        vol_size: list[int] | tuple[int, int, int],
        patch_size: list[int] | tuple[int, int, int],
        down_size_rate: list[int],
        num_classes: int,
        add_aggregation_module: bool = False,
    ):
        assert all(
            starmap(gt, zip(vol_size, patch_size))
        ), f"`vol_size`:{vol_size} must be bigger than {patch_size}"

        super().__init__()

        self.vol_size = vol_size
        self.patch_size = patch_size
        self.down_size_rate = down_size_rate
        self.num_classes = num_classes
        self.add_aggregation_module = add_aggregation_module

        # fixed params
        fixed_wt = compute_importance_map(
            self.patch_size,
            mode="gaussian",
            sigma_scale=0.125,
            dtype=torch.float,
        )[None, None]
        self.register_buffer("fixed_wt", fixed_wt)

        weight_map = (
            torch.zeros(1, 1, *self.vol_size)
            + 1e-20  # add small value prevent division by zero error
        )
        self.register_buffer(
            "weight_map", weight_map
        )  # just in-case this takes much time

        # operations
        self.sigmoid = nn.Sigmoid()
        self.upsize_wt = Interpolate(size=self.patch_size)
        self.upsize_global = Interpolate(scale_factor=self.down_size_rate)
        self.downsize_global = Interpolate(
            mode="nearest", scale_factor=list(1 / np.array(self.down_size_rate))
        )

        self.aggregate = nn.Sequential(
            ConvBnReLU(num_classes * 2, num_classes * 2, 3, 1, repeat=2, norm=None),
            ConvBnReLU(
                num_classes * 2, num_classes, 3, 1, repeat=2, norm=None, act=None
            ),
        )

    def forward(
        self,
        logit_patches: TensorType["BN", "C", "Hp", "Wp", "Dp"],
        global_logit: None | TensorType["B", "C", "Hg", "Wg", "Dg"],
        slice_meta,
    ):

        batch_size = 1

        # window weight
        wt = self.fixed_wt

        # detach
        weight_map = self.compute_weight_map(wt, slice_meta, batch_size)

        final_logit = torch.zeros(
            batch_size,
            self.num_classes,
            *self.vol_size,
            device=global_logit.device,
        )

        upsized_global_logit = self.upsize_global(global_logit)

        mask = torch.zeros(batch_size, 1, *self.vol_size, device=global_logit.device)
        slicess = []
        for i, logit_patch in enumerate(logit_patches):

            batch_index = i // (len(slice_meta) // batch_size)
            slices: list[slice] = [slice(batch_index, batch_index + 1)] + eval(
                logit_patch.meta["slice"]
            )
            if self.add_aggregation_module:
                aggreated_patch_logit = self.aggregate(
                    torch.cat(
                        [
                            (
                                logit_patch
                                if len(logit_patch.shape) == 5
                                else logit_patch[None]
                            ),
                            upsized_global_logit[slices],
                        ],
                        dim=1,
                    )
                )
            else:
                aggreated_patch_logit = logit_patch

            # paste on final logit
            final_logit[slices] = final_logit[slices] + aggreated_patch_logit * wt
            mask[slices] = 1

        # DEBUG
        final_logit = final_logit / weight_map  # account for any overlapping sections

        final_logit = final_logit + torch.where(mask == 1, 0, upsized_global_logit)

        return final_logit

    @torch.no_grad()
    def compute_weight_map(
        self,
        patch_weight: TensorType["1", "1", "Hp", "Wp", "Dp"],
        slice_meta,
        batch_size: int,
    ):

        batched_weight_map = self.weight_map.repeat(
            batch_size, 1, 1, 1, 1
        )  # assuming 5d

        for i, _slice in enumerate(slice_meta):
            batch_index = i // (len(slice_meta) // batch_size)
            slices: list[slice] = [slice(batch_index, batch_index + 1)] + eval(_slice)
            batched_weight_map[slices] += patch_weight

        return batched_weight_map


if __name__ == "__main__":

    from torch.nn.functional import one_hot
    from matplotlib import pyplot as plt
    from utils.visualzation import VisVolLab, ShapeGenerator
    from data.transforms.transform import Patchify, toBatch, UnPatchify
    from model.loss.registry import loss_registry

    TEST_TRAIN = True
    TEST_TEST = True

    vis = VisVolLab()

    device = "cuda:5"
    vol_size = [420, 420, 420]
    patch_size = [128, 128, 128]
    down_size_rate = [3, 3, 3]
    num_classes = 2

    learnable_agg = LearnablePatchAgg(
        vol_size=vol_size,
        patch_size=patch_size,
        down_size_rate=down_size_rate,
        num_classes=num_classes,
        add_aggregation_module=False,
    ).to(device)

    patchify = toBatch(Patchify)(
        keys=["logit"],
        patch_size=patch_size,
        overlap_r=0.5,
    )

    # unpatchify = UnPatchify(
    #     keys=["logit"],
    #     overlap_r=0.25,
    # )

    lab = (
        ShapeGenerator(
            image_size=vol_size,
            shape_type="checkerboard",
            batch_size=1,
        )
        .generate(num_classes=num_classes)
        .type(torch.long)
    )

    logit = one_hot(lab.squeeze(1)).permute(0, -1, 1, 2, 3).to(device).to(torch.float32)

    global_logit = torch.nn.functional.interpolate(
        logit, scale_factor=list(1 / np.array(down_size_rate)), mode="trilinear"
    )
    data_d = {"logit": logit.to(device)}

    logit_patches = patchify(data_d)["logit"] * 10

    if TEST_TRAIN:

        # logit_patches.requires_grad = True

        final_logit = learnable_agg(
            logit_patches,
            global_logit,
            # global_logit,
        )

        # output = final_logit.sum()
        # output.backward()

        loss_fn = loss_registry["DiceCELossMONAI"](lambda_ce=1)

        loss = loss_fn(final_logit, data_d["logit"].argmax(1, keepdim=True))
        print(loss)

        plt.imshow(vis.vis(lab=data_d["logit"].argmax(1, keepdim=True).detach().cpu()))
        plt.show()
        plt.imshow(vis.vis(lab=final_logit.argmax(1, keepdim=True).detach().cpu()))
        plt.show()

        # plt.imshow(
        #     vis.vis(
        #         vol=weight_map[
        #             :,
        #             0:1,
        #             ...,
        #         ]
        #         .detach()
        #         .cpu()
        #     )
        # )

        # final = unpatchify({"logit": final_logit})["logit"]
        # plt.imshow(vis.vis(lab=data_d["logit"].argmax(1, keepdim=True).detach().cpu()))
        # plt.show()
        # plt.imshow(vis.vis(lab=final_logit.argmax(1, keepdim=True).detach().cpu()))
        # plt.show()

    if TEST_TEST:
        final_logit = learnable_agg(
            logit_patches[:50],
            global_logit=None,
            # global_logit,
        )
        print(final_logit.max())
        plt.imshow(vis.vis(lab=data_d["logit"].argmax(1, keepdim=True).detach().cpu()))
        plt.show()
        plt.imshow(vis.vis(lab=final_logit.argmax(1, keepdim=True).detach().cpu()))
        plt.show()
