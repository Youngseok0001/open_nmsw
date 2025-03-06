from math import e
from typing import Callable, Any

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
import torch.nn as nn

from torchtyping import TensorType
from monai.data.utils import list_data_collate

from model.baseline.global_net import GlobalSeg3D
from model.baseline.local_net import LocalSeg3D
from data.transforms.transform import toBatch
from utils.visualzation import VisVolLab, ShapeGenerator
from utils.misc import get_patch_dim
from utils.definitions import *
from utils.functional import list_op, div

from matplotlib import pyplot as plt
from model.registry import model_registry

# memory efficient version that works in iterables
# otherwise memory increase as num_patches to be extracted increase
from model.baseline.blocks.sample_top_k_patch_memory_efficient import SampleTopKPatch
from model.baseline.blocks.learnable_wt_memory_efficient import LearnablePatchAgg

from monai.transforms import ResizeD
from data.transforms.transform import Patchify


@model_registry.register
class GlobalLocalSeg3D(nn.Module):
    """
    This class takes pretrained global- and local-net and predicts with one of the following sampling stretegies
        * sliding window (do not need foreground)
        * random (do not need global)
        * random foreground (need global)
    """

    # i need these later during model training/testing
    train_keys = [""]
    valid_keys = [""]
    test_keys = ["", GLOBAL + "_"]

    def __init__(
        self,
        input_shape: list[int] | tuple[int, int, int, int],
        output_shape: list[int] | tuple[int, int, int, int],
        patch_size: list[int] | tuple[int, int, int] = [128, 128, 128],
        down_size_rate: list[float] | tuple[float, float, float] = [2, 2, 2],
        overlap_r: float = 0.25,
        num_infernce_patches: int = 5,
        sigma_scale: float = 0.125,
        patch_weight: float = 0.99,
        local_backbone_name: str = "FasterUNet",
        global_backbone_name: str = "FasterUNet",
        local_ckpt_path: str | None = None,
        global_ckpt_path: str | None = None,
        sampling_stretegy: str = ["random", "random_fg", "s_w"],
        *args,
        **kwargs,
    ) -> None:

        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.patch_size = patch_size
        self.down_size_rate = down_size_rate
        self.overlap_r = overlap_r
        self.num_infernce_patches = num_infernce_patches
        self.sigma_scale = sigma_scale
        self.patch_weight = patch_weight
        self.local_backbone_name = local_backbone_name
        self.global_backbone_name = global_backbone_name
        self.local_ckpt_path = local_ckpt_path
        self.global_ckpt_path = global_ckpt_path
        self.sampling_stretegy = sampling_stretegy

        # derived attributes
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
        )  # I need this for random sampling

        self.local_net = LocalSeg3D(
            local_backbone_name=self.local_backbone_name,
            input_shape=self.local_input_shape,
            output_shape=self.local_output_shape,
            overlap_r=self.overlap_r,
            patch_size=self.patch_size,
            sigma_scale=self.sigma_scale,
            ckpt_path=self.local_ckpt_path,
        )

        self.global_net = GlobalSeg3D(
            global_backbone_name=self.global_backbone_name,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            down_size_rate=self.down_size_rate,
            ckpt_path=self.global_ckpt_path,
        )

        if self.sampling_stretegy in ["random", "random_fg"]:

            self.patchify_global = toBatch(Patchify)(
                keys=["temp"],
                patch_size=list(
                    np.array(self.patch_size) // np.array(self.down_size_rate)
                ),
                overlap_r=overlap_r,
            )

            self.sample_topk_patch = SampleTopKPatch(
                keys=[VOL],
                patch_size=self.patch_size,
                overlap_r=self.overlap_r,
            )

            # a simple
            self.get_aggregated_logit = LearnablePatchAgg(
                vol_size=self.input_shape[1:],
                patch_size=self.patch_size,
                down_size_rate=self.down_size_rate,
                num_classes=self.output_shape[0],
                add_gaussian_wt=True,
                add_learnable_wt=False,
            )

            # class_weight is not learnable here.
            # put higher importance to local patches.
            self.get_aggregated_logit.class_weight = nn.Parameter(
                torch.ones(1, self.output_shape[0], 1, 1, 1) * 10
            )

        self.vis: VisVolLab = VisVolLab(num_classes=output_shape[0])

        # if self.sampling_stretegy == "random_fg":
        #     self.n_samples = self.sampling_kwargs["n_sample"]
        #     self.patch_weight = self.sampling_kwargs["patch_weight"]
        #     self.sample_patches = toBatch(RandomObjectPatchSampling)(
        #         keys=[LOGIT],
        #         w_key=PRED,
        #         spatial_size=self.local_net.patch_size,
        #         num_samples=self.n_samples,
        #     )
        #     self.inferer = PatchInferer(patch_weight=self.patch_weight)

    @torch.no_grad
    def test_step(self, input_d):

        assert (
            input_d[VOL].shape[0] == 1
        ), f"{self.__class__.__name__} assumes foward method to take input dict with batch size : 1"

        device = input_d[VOL].device

        ###############GLOBAL##################
        global_otuput_d = self.global_net.train_step(input_d)

        if self.sampling_stretegy == "s_w":
            return keyfilter(
                lambda key: key in [LOGIT, VOL, LAB, CASE_ID],
                self.local_net.test_step(input_d),
            ) | {
                GLOBAL_VOL: global_otuput_d[GLOBAL_VOL],
                GLOBAL_LAB: global_otuput_d[GLOBAL_LAB],
                GLOBAL_LOGIT: global_otuput_d[GLOBAL_LOGIT],
            }

        if self.sampling_stretegy == "zoom_in":

            global_logit = global_otuput_d[GLOBAL_LOGIT]
            B, C, _, _, _ = global_logit.shape

            # upsize before getting the bounding box
            upsized_global_logit = torch.nn.functional.interpolate(
                global_logit,
                size=self.output_shape[1:],
                mode="trilinear",
            )
            upsized_global_pred = torch.argmax(upsized_global_logit, dim=1)
            # upsized_global_pred = input_d[LAB].squeeze(1)

            # locate roi for each organ
            rois = [[None] * C for _ in range(B)]
            for b in range(B):
                for c in range(1, C):  # skip background
                    organ_mask = upsized_global_pred[b] == c
                    if organ_mask.any():
                        # Get the indices along each axis where the organ exists.
                        indicess = torch.nonzero(organ_mask, as_tuple=True)

                        # Prepare a list for slices for this organ.
                        rois[b][c] = []
                        # Loop over the three spatial dimensions (assumed to correspond to the last three dims of output_shape and patch_size)
                        for indices, out_dim, p_size in zip(
                            indicess, self.output_shape[-3:], self.patch_size
                        ):
                            # Compute the minimal and maximal indices along this axis
                            current_min = indices.min().item()
                            current_max = indices.max().item()

                            # Adjust the slice if needed so that its length equals at least p_size.
                            adjusted = adjust_slice(
                                current_min, current_max, p_size, out_dim
                            )

                            # Append the slice to the ROI list for this organ.
                            rois[b][c].append(adjusted)
                    else:
                        rois[b][c] = None  # No presence of the organ

            logit = torch.zeros(*([B] + self.output_shape)).to(device)
            logit[:, 0, ...] = 0.0001  # predict background if no crop exists
            for b in range(B):
                for c in range(1, C):

                    slice_info = rois[b][c]
                    if slice_info != None:

                        slice_info = [
                            slice(b, b + 1),
                            slice(None),
                        ] + slice_info  # attach batch slice

                        input_vol = input_d[VOL]
                        croped_vol = input_vol[slice_info]
                        original_crop_size = croped_vol.shape[-3:]
                        resized_crop = torch.nn.functional.interpolate(
                            croped_vol,
                            size=self.patch_size,
                            mode="trilinear",
                        )
                        resized_crop_logit = self.local_net.backbone(resized_crop)

                        crop_logit = torch.nn.functional.interpolate(
                            resized_crop_logit,
                            size=original_crop_size,
                            mode="trilinear",
                        )
                        masked_crop_logit = torch.where(
                            crop_logit.argmax(1) == c, crop_logit, 0
                        )
                        logit[slice_info] += masked_crop_logit

            return {
                CASE_ID: input_d[CASE_ID],
                LOGIT: logit,
                VOL: input_d[VOL],
                LAB: input_d[LAB],
                GLOBAL_VOL: global_otuput_d[GLOBAL_VOL],
                GLOBAL_LAB: global_otuput_d[GLOBAL_LAB],
                GLOBAL_LOGIT: global_otuput_d[GLOBAL_LOGIT],
            }

        if (
            self.sampling_stretegy == "random_fg"
        ):  # behave similar to nmsw but with fixed score

            background_mask: TensorType["B", "1", "Hz", "Wz", "Dz"] = (
                self.get_background_mask(global_otuput_d[GLOBAL_LOGIT])
            )

            # puts equal weightage to all
            sampling_logit = torch.zeros(1, 1, *self.patch_shape, device=device)

            sampled_patches_d, _, slice_meta = self.sample_topk_patch(
                input_d={VOL: input_d[VOL]},
                logit=sampling_logit.to(torch.float32),
                background_mask=background_mask.to(torch.float32),
                k=self.num_infernce_patches,
                mode="test",
            )

            sampled_patches = sampled_patches_d[VOL]

            # turn into iterator for memmory efficiency
            sampled_logits = (
                self.local_net.backbone(sampled_patch[None])[-1]
                for sampled_patch in sampled_patches
            )

            # iterator should not be collated
            # sampled_logits = list_data_collate(sampled_logits)

            #############AGGREGATION###############
            aggregated_logit = self.get_aggregated_logit(
                sampled_logits,
                global_otuput_d[GLOBAL_LOGIT],
                slice_meta,
            )
            return {
                CASE_ID: input_d[CASE_ID],
                # full-res
                VOL: input_d[VOL],
                LAB: input_d[LAB],
                LOGIT: aggregated_logit,
                GLOBAL_VOL: global_otuput_d[GLOBAL_VOL],
                GLOBAL_LAB: global_otuput_d[GLOBAL_LAB],
                GLOBAL_LOGIT: global_otuput_d[GLOBAL_LOGIT],
            }

    forward = test_step

    @torch.no_grad
    def train_step(self, input_d: dict[str, torch.tensor]):
        NotImplementedError

    @torch.no_grad
    def valid_step(self, input_d: dict[str, torch.tensor]):
        NotImplementedError

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

    def visualize_test(self, input_d: dict[str, torch.tensor]):

        vis_lab = self.vis.vis(
            vol=input_d[VOL],
            lab=input_d[LAB],
        )
        vis_pred = self.vis.vis(
            vol=input_d[VOL],
            lab=input_d[LOGIT].argmax(1, keepdim=True),
        )

        return {LAB: vis_lab, PRED: vis_pred}

    @classmethod
    def unit_test(cls, device="cuda"):

        from time import perf_counter

        input_shape = [1, 480, 480, 288]
        output_shape = [2, 480, 480, 288]
        down_size_rate = [3, 3, 3]
        overlap_r = [0.5, 0.5, 0.5]
        patch_size = [128, 128, 128]
        num_infernce_patches = 100
        sigma_scale = 0.125
        patch_weight = 0.7
        local_backbone_name = "UNETRSwin"
        global_backbone_name = "FasterUNet"
        local_ckpt_path = None
        global_ckpt_path = None
        shape_type = "circle"

        batch_size = 1  # dont touch this

        shape_generator = ShapeGenerator(
            image_size=input_shape[1:],
            batch_size=batch_size,
            shape_type=shape_type,
        )

        vol = torch.randn(batch_size, *input_shape).to(device)
        lab = shape_generator.generate().to(device)
        input_d = {
            VOL: vol,
            LAB: lab,
            CASE_ID: ["lol.txt"] * batch_size,
        }

        for sampling_stretegy in ["s_w", "random_fg", "random_fg"]:
            print(f"Testing {sampling_stretegy}")

            global_local_seg_3d = cls(
                input_shape=input_shape,
                output_shape=output_shape,
                patch_size=patch_size,
                down_size_rate=down_size_rate,
                overlap_r=overlap_r,
                num_infernce_patches=num_infernce_patches,
                sigma_scale=sigma_scale,
                patch_weight=patch_weight,
                local_backbone_name=local_backbone_name,
                global_backbone_name=global_backbone_name,
                local_ckpt_path=local_ckpt_path,
                global_ckpt_path=global_ckpt_path,
                sampling_stretegy=sampling_stretegy,
            ).to(device)

            test_output = global_local_seg_3d.test_step(input_d)
            test_vis_d = global_local_seg_3d.visualize_test(test_output)

            for name, img in test_vis_d.items():
                plt.title(TEST + "_" + name)
                plt.imshow(img.detach().cpu().numpy())
                plt.show()


def adjust_slice(
    current_min: int, current_max: int, patch_size: int, dim_size: int
) -> slice:
    """
    Adjust the slice defined by current_min and current_max so that its length is at least patch_size.
    The new slice will be centered as much as possible around the original region but will be shifted
    if needed so that it does not exceed the boundary of the dimension (0 to dim_size).

    Parameters:
        current_min (int): The starting index of the current region.
        current_max (int): The ending index (inclusive) of the current region.
        patch_size (int): The desired size of the patch.
        dim_size (int): The size of the dimension (i.e. the boundary).

    Returns:
        slice: The adjusted slice object.
    """
    # Current length of the region
    current_length = current_max - current_min + 1

    # If the current region is already large enough, return it as is.
    if current_length >= patch_size:
        return slice(current_min, current_max + 1)

    # Calculate the additional length needed
    missing = patch_size - current_length

    # Try to add half the missing length on each side.
    left_expand = missing // 2
    right_expand = missing - left_expand

    # Initial expansion
    new_min = current_min - left_expand
    new_max = current_max + right_expand

    # If new_min is less than 0, shift the window to the right.
    if new_min < 0:
        # How much are we below zero?
        shift = -new_min
        new_min = 0
        new_max = new_max + shift

    # If new_max exceeds the boundary, shift the window to the left.
    if new_max >= dim_size:
        shift = new_max - (dim_size - 1)
        new_max = dim_size - 1
        new_min = new_min - shift
        # In case shifting left pushes new_min below 0, clamp it.
        if new_min < 0:
            new_min = 0

    # Final check: ensure the slice covers exactly patch_size elements if possible.
    # It might be less than patch_size if the dim_size is smaller than patch_size.
    if new_max - new_min + 1 < patch_size and dim_size >= patch_size:
        # Adjust new_max to be new_min + patch_size - 1, but do not exceed dim_size.
        new_max = min(new_min + patch_size - 1, dim_size - 1)
        new_min = new_max - patch_size + 1

    return slice(new_min, new_max + 1)


if __name__ == "__main__":

    GlobalLocalSeg3D.unit_test(device="cuda:1")
