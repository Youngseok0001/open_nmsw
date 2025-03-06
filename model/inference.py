import copy

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

from torchtyping import TensorType

import numpy as np
import torch
from torch import nn

from monai.data.utils import compute_importance_map
from monai.data.utils import dense_patch_slices as _dense_patch_slices
from monai.data.utils import list_data_collate


def get_dense_patch_slices(image_size, patch_size, overlap_r):
    interval = get_scan_interval(image_size, patch_size, overlap_r)
    slices: list[list[slice]] = _dense_patch_slices(
        image_size, patch_size, interval, overlap_r
    )
    return slices


class PatchInferer(nn.Module):
    """
    Given predicted patches and crude global prediction, compute the final prediction.
    NOTE
        * make sure the patches and global predictions are pre-softmax values.
        * make sure the global prediction is upscaled back to its orginal scale.
    """

    def __init__(
        self,
        patch_weight: float = 0.5,
    ):
        super().__init__()

        self.patch_weight = patch_weight

    def __call__(
        self,
        patches: TensorType["BN", "C", "Hp", "Wp", "Dp"],
        vol: TensorType["B", "C", "H", "W", "D"],
    ):

        # identical patches may exist, especially when oversampling
        # prevent this by deleting duplictes
        patches = compose(
            list_data_collate,
            list,
            lambda x: x.values(),
            valmap(first),
            groupby(lambda x: x.meta["slice"]),
        )(patches)

        print(patches.shape)

        batch_size = vol.shape[0]
        num_patch_per_batch = len(patches) // batch_size
        patch_size = patches.shape[-3:]
        vol_size = vol.shape[-3:]
        assert (
            np.array(patch_size) <= np.array(vol_size)
        ).all(), (
            f"patch size : {patch_size} should be smaller than vol size : {vol_size}"
        )

        # gaussian weight to be multiplied
        w_t = compute_importance_map(
            patch_size,
            mode="gaussian",
            sigma_scale=0.125,
            device=vol.device,
            dtype=vol.dtype,
        )
        w_t: TensorType["1", "1", "Hp", "Wp", "Dp"] = w_t[None, None]

        # DEBUG
        w_t = torch.ones_like(w_t)

        # paste patch on vol
        new_vol: TensorType["B", "C", "H", "W", "D"] = copy.deepcopy(vol) * (
            1 - self.patch_weight
        )
        for i, patch in enumerate(patches):
            batch_index: int = i // num_patch_per_batch
            if "slice" not in patch.meta.keys():
                crop_center: TensorType["3"] = patch.meta["crop_center"]
                _slice = [slice(batch_index, batch_index + 1), slice(None)]
                for p_s, c_c in zip(patch_size, crop_center):
                    lower_bound = int(c_c - p_s // 2)
                    upper_bound = int(c_c + p_s // 2)
                    _slice.append(slice(lower_bound, upper_bound))
            else:
                _slice = [slice(batch_index, batch_index + 1)] + eval(
                    patch.meta["slice"]
                )

            new_vol[_slice] = new_vol[_slice] + (
                (w_t * patch[None]) * self.patch_weight
            )

        return new_vol


def get_scan_interval(
    image_size: list[int],
    roi_size: list[int],
    overlap: list[float],
) -> tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """

    scan_interval = []
    for i, o in zip(range(len(roi_size)), overlap):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - o))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


if __name__ == "__main__":

    from utils.visualzation import VisVolLab, ShapeGenerator
    from utils.definitions import *
    from data.transforms.transform import (
        toBatch,
        RandomObjectPatchSampling,
        RandomPatchSampling,
    )

    from matplotlib import pyplot as plt

    vol_size = [256, 256, 256]
    patch_size = [64, 64, 64]
    down_size_rate = [4, 4, 4]
    num_samples = 8

    patch_inferer = PatchInferer()
    vis = VisVolLab(num_classes=100)

    circle_generator = ShapeGenerator(
        image_size=vol_size,
        shape_type="circle",
    )

    checkerboard_generator = ShapeGenerator(
        image_size=vol_size,
        shape_type="checkerboard",
    )

    vol_batch = torch.cat(
        [
            circle_generator.generate(),
            checkerboard_generator.generate(),
        ]
    )
    logit_batch = torch.cat(
        [
            circle_generator.generate(),
            checkerboard_generator.generate(),
        ]
    )

    logit_batch = (
        torch.nn.functional.one_hot(logit_batch.to(torch.long))
        .squeeze(1)
        .permute(0, -1, 1, 2, 3)
    ).to(torch.float)

    input_d = {VOL: vol_batch, LOGIT: logit_batch}

    TEST_PATCH_INFETR = True

    if TEST_PATCH_INFETR:

        cropper = toBatch(RandomObjectPatchSampling)(
            keys=[VOL, LOGIT],
            w_key=LOGIT,
            spatial_size=patch_size,
            num_samples=num_samples,
        )

        output_d = cropper(input_d)

        pred = patch_inferer(
            patches=output_d[LOGIT],
            vol=torch.zeros_like(input_d[LOGIT]),
        )

        # pred = patch_inferer(
        #     patches=output_d[LOGIT],
        #     vol=input_d[LOGIT],
        # )

        plt.imshow(vis.vis(lab=logit_batch.argmax(1, keepdim=True)))
        plt.show()
        plt.imshow(vis.vis(lab=pred.argmax(1, keepdim=True)))
        plt.show()
