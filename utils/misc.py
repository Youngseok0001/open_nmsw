from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

from monai.utils import ensure_tuple_rep
import math


def get_patch_dim(
    image_size: list[int],
    patch_size: list[int],
    overlap_r: float | tuple[float],
):
    """Extracts patch dimension if a volume of size `image_size` is patchified with patch size of `patch_size` and overlapping ratio of `overlap_r`
    Args:
        image_size (list[int])
        patch_size (list[int])
        overlap_r (float | tuple[float])
    """

    def get_scan_interval(
        image_size: list[int],
        roi_size: list[int],
        overlap: list[float],
    ) -> tuple[int, ...]:

        scan_interval = []
        for i, o in zip(range(len(roi_size)), overlap):
            if roi_size[i] == image_size[i]:
                scan_interval.append(int(roi_size[i]))
            else:
                interval = int(roi_size[i] * (1 - o))
                scan_interval.append(interval if interval > 0 else 1)
        return tuple(scan_interval)

    num_spatial_dims = len(image_size)
    overlap_r = ensure_tuple_rep(overlap_r, num_spatial_dims)

    scan_interval = get_scan_interval(image_size, patch_size, overlap_r)

    scan_num = []
    for i in range(num_spatial_dims):
        if scan_interval[i] == 0:
            scan_num.append(1)
        else:
            num = int(math.ceil(float(image_size[i]) / scan_interval[i]))
            scan_dim = first(
                d
                for d in range(num)
                if d * scan_interval[i] + patch_size[i] >= image_size[i]
            )
            scan_num.append(scan_dim + 1 if scan_dim is not None else 1)

    return scan_num
