from time import perf_counter
from git import WorkTreeRepositoryUnsupported
from torchtyping import TensorType

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from itertools import starmap

import numpy as np
import torch
from torch import nn
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
        add_gaussian_wt: bool = True,
        add_learnable_wt: bool = False,
        learnable_wt_type: str = "affine",
    ):
        assert all(
            starmap(gt, zip(vol_size, patch_size))
        ), f"`vol_size`:{vol_size} must be bigger than {patch_size}"

        super().__init__()

        self.vol_size = vol_size
        self.patch_size = patch_size
        self.down_size_rate = down_size_rate
        self.num_classes = num_classes
        self.add_gaussian_wt = add_gaussian_wt
        self.add_learnable_wt = add_learnable_wt
        self.learnable_wt_type = learnable_wt_type

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

        if self.add_learnable_wt and learnable_wt_type == "raw":
            # learnable params
            learable_wt = torch.zeros([1] + list(np.array(self.patch_size) // 4))[None]
            self.register_parameter("learnable_wt", nn.Parameter(learable_wt))

        if self.add_learnable_wt and learnable_wt_type == "affine":
            affine_identitiy = torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
            self.register_buffer("affine_identity", affine_identitiy)

            learable_wt = torch.ones(1)
            self.register_parameter("learnable_wt", nn.Parameter(learable_wt))

        patch_weight = torch.zeros(1, self.num_classes, 1, 1, 1)
        self.register_parameter("class_weight", nn.Parameter(patch_weight))

        # operations
        self.sigmoid = nn.Sigmoid()
        self.upsize_wt = Interpolate(size=self.patch_size)
        self.upsize_global = Interpolate(scale_factor=self.down_size_rate)
        self.downsize_global = Interpolate(
            mode="nearest", scale_factor=list(1 / np.array(self.down_size_rate))
        )

    def forward(
        self,
        logit_patches: list[TensorType["1", "C", "Hp", "Wp", "Dp"]],
        global_logit: None | TensorType["B", "C", "Hg", "Wg", "Dg"],
        slice_meta: None,
    ):

        batch_size = 1
        channel_size = global_logit.shape[1]

        if [self.add_gaussian_wt, self.add_learnable_wt, self.learnable_wt_type] == [
            True,
            True,
            "raw",
        ]:
            wt = self.sigmoid(self.upsize_wt(self.learnable_wt)) + self.fixed_wt

        if [self.add_gaussian_wt, self.add_learnable_wt, self.learnable_wt_type] == [
            False,
            True,
            "raw",
        ]:
            wt = self.sigmoid(self.upsize_wt(self.learnable_wt))

        if [self.add_gaussian_wt, self.add_learnable_wt, self.learnable_wt_type] == [
            False,
            True,
            "affine",
        ]:

            affine_matrix = self.affine_identity.clone()
            affine_matrix[:3, :3] *= torch.tanh(self.learnable_wt)
            grid = F.affine_grid(
                affine_matrix[None].repeat(batch_size, 1, 1),
                size=[batch_size] + [channel_size] + self.patch_size,
                align_corners=True,
            )

            wt = F.grid_sample(
                self.fixed_wt,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )

        if [self.add_gaussian_wt, self.add_learnable_wt] == [True, False]:
            wt = self.fixed_wt

        if [self.add_gaussian_wt, self.add_learnable_wt, self.learnable_wt_type] == [
            True,
            True,
            "affine",
        ]:
            raise NotImplementedError

        if [self.add_gaussian_wt, self.add_learnable_wt] == [False, False]:
            raise NotImplementedError

        # detach
        weight_map = self.compute_weight_map(wt, logit_patches, slice_meta, batch_size)

        final_logit = MetaTensor(
            torch.zeros(
                batch_size,
                self.num_classes,
                *self.vol_size,
                device=global_logit.device,
            ),
            device=global_logit.device,
        )

        mask = torch.zeros(batch_size, 1, *self.vol_size, device=global_logit.device)
        for i, logit_patch in enumerate(logit_patches):
            # extract infos
            batch_index = i // (
                len(logit_patches if slice_meta == None else slice_meta) // batch_size
            )
            slices: list[slice] = [slice(batch_index, batch_index + 1)] + eval(
                logit_patch.meta["slice"]
            )
            # paste on final logit
            final_logit[slices] = final_logit.as_tensor()[slices] + (wt * logit_patch)
            mask[slices] = 1

        # DEBUG
        final_logit = final_logit / weight_map  # account for any overlapping sections

        if global_logit == None:  # do not weigted
            final_logit[:, 0][(mask != 1).squeeze(1)] = 10
        else:  # weight local and global
            class_weight = self.class_weight.clamp(-1, 10)
            c_w = torch.sigmoid(class_weight)
            # do not weight regions with no patch
            masked_c_w = torch.where(self.downsize_global(mask) == 0, 1, (1 - c_w))
            weighted_global_logit = global_logit * masked_c_w
            final_logit = c_w * final_logit + self.upsize_global(weighted_global_logit)

        return final_logit

    @torch.no_grad()
    def compute_weight_map(
        self,
        patch_weight: TensorType["1", "1", "Hp", "Wp", "Dp"],
        logit_patches: TensorType["BN", "1", "Hp", "Wp", "Dp"],
        slice_meta,
        batch_size: int,
    ):

        batched_weight_map = self.weight_map.repeat(
            batch_size, 1, 1, 1, 1
        )  # assuming 5d

        if slice_meta == None:
            for i, logit_patch in enumerate(logit_patches):
                batch_index = i // (len(logit_patches) // batch_size)
                slices: list[slice] = [slice(batch_index, batch_index + 1)] + eval(
                    logit_patch.meta["slice"]
                )
                batched_weight_map[slices] += patch_weight
        else:
            for i, _slice in enumerate(slice_meta):
                batch_index = i // (len(slice_meta) // batch_size)
                slices: list[slice] = [slice(batch_index, batch_index + 1)] + eval(
                    _slice
                )
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

    device = "cuda:0"
    vol_size = [420, 420, 420]
    patch_size = [128, 128, 128]
    down_size_rate = [3, 3, 3]
    num_classes = 2

    learnable_agg = LearnablePatchAgg(
        vol_size=vol_size,
        patch_size=patch_size,
        down_size_rate=down_size_rate,
        num_classes=num_classes,
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

    logit_patches = (patchify(data_d)["logit"][:50]) * 10
    slice_meta = logit_patches.meta["slice"]
    logit_patches = iter(map(lambda x: x.unsqueeze(0), list(logit_patches)))

    if TEST_TRAIN:

        # logit_patches.requires_grad = True

        final_logit = learnable_agg(
            logit_patches,
            torch.zeros_like(global_logit).to(device),
            # global_logit,
            slice_meta,
        )

        plt.imshow(vis.vis(lab=data_d["logit"].argmax(1, keepdim=True).detach().cpu()))
        plt.show()
        plt.imshow(vis.vis(lab=final_logit.argmax(1, keepdim=True).detach().cpu()))
        plt.show()

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
