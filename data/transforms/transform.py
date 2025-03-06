from pprint import pprint

# TODO
# [] verify the scanning order (this is needed for patch sampling later on)

from copy import deepcopy
from typing import Mapping, Hashable, Any

from itertools import starmap
from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
from torchtyping import TensorType

from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.traits import LazyTrait, MultiSampleTrait
from monai.transforms import (
    SpatialPadD,
    MapTransform,
    RandWeightedCropd,
    RandCropByPosNegLabelD,
)
from monai.data.meta_tensor import MetaTensor
from monai.utils import (
    BlendMode,
    PytorchPadMode,
    convert_data_type,
    convert_to_dst_type,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
    optional_import,
    pytorch_after,
)

from monai.data.utils import (
    compute_importance_map,
    dense_patch_slices,
    get_valid_patch_size,
    list_data_collate,
)
from monai.data import decollate_batch

from utils.definitions import *
from utils.visualzation import VisVolLab, ShapeGenerator
from itertools import repeat


def dict_data_decollate(input_d: dict[str, Any]) -> list[dict[str, Any]]:

    # if the value in input_d one of [str, int, float], repeat it.
    # otherwise it will produce truncated data
    tupled_data = (
        repeat(value) if type(value) in [str, int, float] else value
        for value in input_d.values()
    )
    keys = input_d.keys()
    return [dict(zip(keys, data)) for data in zip(*tupled_data)]


def toBatch(Transform: MapTransform):
    """
    this function is to create a `BatchTransform`class which lifts a generic `MapTransform` class to work on batch data.
    * I am using this function majorly to extract patch from a batched volume.
    * Why work on batch input?
        * because I consider patch extraction stretegy as a part of model bias. so I include it into a model class which takes a batched data.
            * for example, a vanilla segmentation model takes raddom patches during traning but sliding window patches during testing.
                * this is a model bias
    """

    class BatchTransform(Transform):

        def __call__(
            self, batch_data: dict[Hashable, torch.Tensor], *args, **kwargs
        ) -> dict[Hashable, torch.Tensor]:

            # check if the values in dict have same lenth
            batch_size = len(batch_data[first(self.keys)])

            new_batch_data_list = []
            for b in range(batch_size):
                # print(batch_data.keys())
                data = valmap(get(b), batch_data)
                data_transformed = Transform.__call__(self, data, *args, **kwargs)
                new_batch_data_list.append(data_transformed)
            collated = list_data_collate(new_batch_data_list)
            return collated

    return BatchTransform


class Patchify(MapTransform, MultiSampleTrait):
    """
    An interface to create a list of patches.
    modified from : https://github.com/Project-MONAI/MONAI/blob/dev/monai/inferers/utils.py#L43-L322
    """

    def __init__(
        self,
        keys: KeysCollection,
        patch_size: int | tuple[int, int, int],
        overlap_r: float | tuple[float, float, float] = 0.25,
    ):

        self.keys = keys
        self.patch_size = patch_size
        self.num_spatial_dims = len(patch_size)
        self.overlap_r = ensure_tuple_rep(overlap_r, len(patch_size))

    def __call__(
        self, data: dict[Hashable, torch.Tensor]
    ) -> list[dict[Hashable, torch.Tensor]]:

        d = dict(data)
        for key in self.keys:
            d[key] = self.fn(d[key])

        return dict_data_decollate(d)

    def fn(
        self, data: TensorType["C", "H", "W", "D"]
    ) -> list[TensorType["C", "H_patch", "W_patch", "D_patch"]]:
        """
        assume data is already padded to be larger `self.patch_size`
        """

        # assign name to key infos
        c, *image_size = data.shape
        device = data.device

        # in case the input is not torch.tensor
        data = convert_data_type(data, torch.Tensor, wrap_sequence=True)[0]

        # get distance between patches
        interval: list[int] = self._get_scan_interval(
            image_size,
            self.patch_size,
            self.overlap_r,
        )

        # get list of slices
        # add channel dim
        slices: list[list[slice]] = dense_patch_slices(
            image_size, self.patch_size, interval, self.overlap_r
        )
        slices: list[list[slice]] = [[slice(None)] + list(s) for s in slices]

        # extract patches from `data` using `slices`
        patches = []
        for s in slices:
            patch = data[s]
            patch = MetaTensor(patch)  # to metatensor
            patch = self._add_meta(
                patch,
                {
                    "slice": str(s),
                    "crop_center": self._slice_to_crop_center(s),
                },
            )
            patches.append(patch)
        # patches = MetaTensor(
        #     torch.empty(len(slices), c, *self.patch_size, device=device)
        # )
        # for i, s in enumerate(slices):
        #     patch = data[s]
        #     patches[i] = patch
        # patches.meta["slice"] = [str(s) for s in slices]
        # patches.meta["crop_center"] = [self._slice_to_crop_center(s) for s in slices]
        # patches.is_batch = True
        return patches

    @staticmethod
    def _slice_to_crop_center(s: list[slice]):
        # NOTE
        # rounding error could occur if _s.stop + _s.start is not an even number
        # but this is not going to occur if patch-size is set to an even number.

        # NOTE
        # why do you even need crop_center?
        # monai's random patch sampling class produce roi-centre. So I follow the convension.

        # NOTE
        # why not include patch-size info also?
        # monai's random patch sampling class only spits out roi-centre. So I follow the convension.
        # patch-size can be extracted easily later on anyways

        centre: list[int] = [(_s.stop + _s.start) // 2 for _s in s[1:]]

        return centre

    @staticmethod
    def _add_meta(
        patch: list[TensorType["H_patch", "W_patch", "D_patch"]],
        d: dict[str | Any],
    ) -> list[TensorType["H_patch", "W_patch", "D_patch"]]:
        patch.meta = merge(patch.meta, d)
        return patch

    @staticmethod
    def _get_scan_interval(
        image_size: list[int],
        roi_size: list[int],
        overlap: list[float],
    ) -> tuple[int, ...]:
        """
        Compute scan interval according to the image size, roi size and overlap.
        Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
        use 1 instead to make sure sliding window works.

        """
        if len(image_size) != len(roi_size):
            raise ValueError(
                f"len(image_size) {len(image_size)} different from spatial dims {num_spatial_dims}."
            )
        if len(roi_size) != len(roi_size):
            raise ValueError(
                f"len(roi_size) {len(roi_size)} different from spatial dims {num_spatial_dims}."
            )

        scan_interval = []
        for i, o in zip(range(len(roi_size)), overlap):
            if roi_size[i] == image_size[i]:
                scan_interval.append(int(roi_size[i]))
            else:
                interval = int(roi_size[i] * (1 - o))
                scan_interval.append(interval if interval > 0 else 1)
        return tuple(scan_interval)


class RandomPatchSampling(RandWeightedCropd):
    """
    Uniformly sample a list of `num_samples` image patches.
    You may wonder why I dont use `RandSpatialCropSamplesd`. 'RandSpatialCropSamplesd' seems to have some bug.
    The metatensor output does not contain `crop_roi` info, which is necessary when combining patches with global volume.
    """

    def __init__(
        self,
        keys: list[str],
        # w_key: str,
        spatial_size: list[int] | int,
        num_samples: int = 1,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ):

        super().__init__(
            keys=keys,
            w_key=keys[
                0
            ],  # just dummy w_key. will not be used during uniform sampling.
            spatial_size=spatial_size,
            num_samples=num_samples,
            allow_missing_keys=allow_missing_keys,
            lazy=lazy,
        )

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> list[dict[Hashable, torch.Tensor]]:
        # output starts as empty list of dictionaries
        ret: list = [dict(data) for _ in range(self.cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(self.cropper.num_samples):
            for key in set(data.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(data[key])

        # self.randomize(weight_map=data[self.w_key])
        # uniform distribution.
        self.randomize(weight_map=torch.ones_like(data[self.w_key]))
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(data):
            for i, im in enumerate(
                self.cropper(data[key], randomize=False, lazy=lazy_)
            ):
                ret[i][key] = im
                # this module has a
                # a quick fix
                ret[i][key].meta["crop_center"] = (
                    ret[i][key].meta["crop_center"].type(torch.int).tolist()
                )
        return ret


class RandomObjectPatchSampling(RandWeightedCropd):
    """
    Uniformly sample a list of `num_samples` image patches around objects (labels).
    You may wonder why I dont use `RandSpatialCropSamplesd`. 'RandSpatialCropSamplesd' seems to have some bug.
    The metatensor output does not contain `crop_roi` info, which is necessary when combining patches with global volume.
    """

    def __init__(
        self,
        keys: list[str],
        w_key: str,
        spatial_size: list[int] | int,
        num_samples: int = 1,
        allow_missing_keys: bool = False,
        lazy: bool = False,
    ):
        self.spatial_size = spatial_size
        super().__init__(
            keys=keys,
            w_key=w_key,
            spatial_size=spatial_size,
            num_samples=num_samples,
            allow_missing_keys=allow_missing_keys,
            lazy=lazy,
        )

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor], lazy: bool | None = None
    ) -> list[dict[Hashable, torch.Tensor]]:
        # output starts as empty list of dictionaries
        ret: list = [dict(data) for _ in range(self.cropper.num_samples)]
        # deep copy all the unmodified data
        for i in range(self.cropper.num_samples):
            for key in set(data.keys()).difference(set(self.keys)):
                ret[i][key] = deepcopy(data[key])

        binary_map = (data[self.w_key] > 0).float()
        self.randomize(weight_map=binary_map)
        lazy_ = self.lazy if lazy is None else lazy
        for key in self.key_iterator(data):
            for i, im in enumerate(
                self.cropper(data[key], randomize=False, lazy=lazy_)
            ):
                ret[i][key] = im
                # this module has a bug
                # a quick fix
                ret[i][key].meta["crop_center"] = (
                    ret[i][key].meta["crop_center"].type(torch.int).tolist()
                )
                ret[i][key].meta["slice"] = str(
                    [slice(None)]
                    + [
                        slice(c_c - s_s // 2, c_c + s_s // 2)
                        for c_c, s_s in zip(
                            ret[i][key].meta["crop_center"], self.spatial_size
                        )
                    ]
                )
        return ret


# this code does not work together with toBatch
class UnPatchify(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        overlap_r: list[float] | float,
    ):

        self.keys = keys
        self.overlap_r = overlap_r

    def __call__(
        self, data: dict[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:

        d = dict(data)
        for key in self.keys:
            d[key] = self.fn(d[key])
        return d

    def fn(
        self,
        patches: TensorType["N", "C", "H_patch", "W_patch", "D_patch"],
    ) -> TensorType["C", "H", "W", "D"]:
        """
        assume patches are list of **monai's meta-tensor**
        """

        # get key infos
        one_patch = patches[0]
        device = one_patch.device
        temp_meta = MetaTensor([], device=device).copy_meta_from(
            one_patch, copy_attr=False
        )
        temp_meta = MetaTensor([], device=device)
        img = self.unpachify(patches)
        img = convert_to_dst_type(img, temp_meta, device=device)[0]  # to metatensor

        return img

    @staticmethod
    def _get_img_size(patches: list[TensorType["C", "Hp", "Wp", "Dp"]]) -> list[int]:

        patch_size: list[int] = patches[0].shape[-3:]

        # centre_min: list[int] = list(map(min, zip(*patches.meta["crop_center"])))
        # centre_max: list[int] = list(map(max, zip(*patches.meta["crop_center"])))

        centre_min: list[int] = list(map(min, patches.meta["crop_center"]))
        centre_max: list[int] = list(map(max, patches.meta["crop_center"]))

        _min: np.array = np.array(centre_min) - (np.array(patch_size) // 2)
        _max: np.array = np.array(centre_max) + (np.array(patch_size) // 2)

        return list(_max - _min)

    def unpachify(
        self, patches: list[TensorType["C", "Hp", "Wp", "Dp"]]
    ) -> TensorType["C", "H", "W", "D"]:

        channel_size: int = patches[0].shape[0]
        img_size: list[int] = self._get_img_size(patches)
        img = torch.zeros(*([channel_size] + img_size))
        for i, patch in enumerate(patches):
            patch_slice: list[str] = eval(patch.meta["slice"])
            img[patch_slice] = patch

        return img


class MergeLabelD(MapTransform):
    """
    An interface to clean-up the labels in segmentation mask by either removing un-specified labels or grouping multiple labels into one.
    """

    def __init__(
        self,
        keys: KeysCollection,
        old_label_rule: list[str],
        new_label_rule: dict[str, list[str]],
    ) -> None:
        super().__init__(keys)

        self.old_label_rule = old_label_rule
        self.new_label_rule = new_label_rule

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Mapping[Hashable, NdarrayOrTensor]:

        d = dict(data)
        for key in self.keys:
            d[key] = self.transform(d[key])
        return d

    def transform(self, lab):

        lab_copy = deepcopy(lab)

        if self.new_label_rule != None:
            lab_copy.array = np.zeros_like(lab)
            for new_index, labs in enumerate(self.new_label_rule.values()):
                for lab_idx in map(self.old_label_rule.index)(labs):
                    lab_copy[lab == lab_idx] = new_index
            del lab

        return lab_copy


if __name__ == "__main__":

    import torch
    from matplotlib import pyplot as plt
    import time

    TEST_MERGE_LABEL_D = False  # passed
    TEST_TO_BATCH = False  # passed
    TEST_PATCHIFY = True  # passsed
    TEST_UNPATCHIFY = False  # passed
    TEST_RANDOM_PATCH = False  # passed
    TEST_RANDOM_OBJECT_PATCH = False  # passed

    input_shape = [3, 512, 512, 512]
    patch_size = [128, 128, 128]
    num_classes = 10
    overlap_r = 0.25
    shape_type = "checkerboard"
    batch_size = 2
    num_samples = 4
    device = "cuda:6"

    shape_generator = ShapeGenerator(
        image_size=input_shape[1:],
        shape_type=shape_type,
        batch_size=batch_size,
    )

    vis = VisVolLab(num_classes=num_classes)

    batch_data_d = {
        VOL: torch.randn(batch_size, *input_shape).to(device),
        LAB: shape_generator.generate().to(device),
        CASE_ID: [f"000_{i}.py" for i in range(batch_size)],
    }

    data_d = {
        VOL: torch.randn(*input_shape).to(device),
        LAB: shape_generator.generate()[0].to(device),
        CASE_ID: f"000_0.py",
    }

    if TEST_MERGE_LABEL_D:

        num_class_group = 2

        label_groups = [f"lab_group_{i}" for i in range(num_class_group)]
        old_label_rule = [f"lab_{i}" for i in range(num_classes)]

        # merge into 3 groups
        new_label_rule = compose(
            dict,
            partial(zip, label_groups),
            list,
            map(list),
            partial(np.split, indices_or_sections=num_class_group),
            np.array,
        )(old_label_rule)

        transform = MergeLabelD(
            keys=["lab"],
            old_label_rule=old_label_rule,
            new_label_rule=new_label_rule,
        )

        new_data_d = transform(data_d)

        print(
            "works!"
            if len(new_data_d[LAB].unique()) == num_class_group
            else "does not work..."
        )

        plt.imshow(vis.vis(lab=new_data_d[LAB]))
        plt.show()

    if TEST_TO_BATCH:

        crop_d = toBatch(SpatialPadD)(
            keys=LAB, spatial_size=[500, 500, 100], mode="constant"
        )
        new_batch_data_d = crop_d(batch_data_d)

        plt.imshow(vis.vis(lab=new_batch_data_d[LAB]))
        print(new_batch_data_d.keys())

    if TEST_PATCHIFY:

        patchify = toBatch(Patchify)(
            keys=[VOL, LAB],
            patch_size=patch_size,
            overlap_r=overlap_r,
        )

        start = time.perf_counter()
        returned_data_d = patchify(batch_data_d)
        end = time.perf_counter()
        print(end - start)

        # pprint(returned_data_d[VOL].meta["crop_center"])
        # plt.imshow(vis.vis(lab=returned_data_d[LAB]))
        # plt.show()

    if TEST_UNPATCHIFY:

        unpatchify = UnPatchify(
            keys=[VOL, LAB],
            overlap_r=overlap_r,
        )
        temp = valmap(lambda data: data[: (len(data) // batch_size)], returned_data_d)
        new_returned_data_d = unpatchify(temp)
        plt.imshow(vis.vis(lab=new_returned_data_d["lab"][0:1]))
        plt.show()
        plt.imshow(vis.vis(lab=new_returned_data_d["lab"]))
        plt.show()

    if TEST_RANDOM_PATCH:

        patchify = toBatch(RandomPatchSampling)(
            keys=[VOL, LAB],
            spatial_size=patch_size,
            num_samples=num_samples,
        )

        new_batch_data_d = patchify(batch_data_d)

        for i in range(len(new_batch_data_d[LAB])):
            plt.imshow(vis.vis(lab=new_batch_data_d["lab"][i : i + 1]))
            plt.show()

    if TEST_RANDOM_OBJECT_PATCH:

        patchify = toBatch(RandomObjectPatchSampling)(
            keys=[VOL, LAB],
            w_key=LAB,
            spatial_size=patch_size,
            num_samples=num_samples,
        )

        returned_batch_data_d = patchify(batch_data_d)

        for i in range(len(returned_batch_data_d[LAB])):
            plt.imshow(vis.vis(lab=returned_batch_data_d[LAB][i : i + 1]))
            plt.show()
