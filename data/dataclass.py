from pathlib import Path
from typing import Callable, Any

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

from monai.data.dataset import (
    PersistentDataset,  # if not this, though slower its not constrained by RAM size.
)

from data.augmentation import aug
from data.registry import data_registry
from utils.definitions import *

from operator import methodcaller as mc
import json

__all__ = ["Word"]


@curry
def load_val_from_json(keys: str | list[str], json_path: str) -> Any:
    return compose(
        get(keys),
        json.load,
        open,
    )(json_path)


class DecathlonDataset(PersistentDataset):

    root_dir: Path = Path(__file__).resolve().parent / "datasets/processed/"
    dataset_json_filename: str = "dataset.json"
    cache_dir_name: str = "persistent_cache"
    train_vol_name: str = "imagesTr"
    train_lab_name: str = "labelsTr"
    valid_vol_name: str = "imagesVal"
    valid_lab_name: str = "labelsVal"
    test_vol_name: str = "imagesTs"
    test_lab_name: str = "labelsTs"
    vol_key: str = "vol"
    lab_key: str = "lab"
    case_id_key: str = "case_id"
    vols_to_exclude: list[str] = []

    def __init__(
        self,
        mode: str,
        dataset_name: str,
        transform: Callable[[Any], Any] = identity,
        fold_n: int = 0,
        fold_size: int = 5,  # 5 fold cross validation
        # cache_rate: float = 0.5,
        *args,
        **kwargs,
    ) -> None:

        assert (
            self.train_vol_name != None and self.train_lab_name != None
        ), f"please specify class attributes `train_vol_name` and `train_lab_name` in {DecathlonDataset.__name__}"

        assert mode in [
            TRAIN,
            VALID,
            TEST,
        ], f"`mode`(current val: {mode}) should be one of [TRAIN, VALID, TEST]."

        self.mode = mode
        self.dataset_name = dataset_name
        self.fold_n = fold_n
        self.fold_size = fold_size

        data: list[dict[str, str]] = self.parse()

        cache_dir = self.root_dir.parent / self.cache_dir_name / self.__class__.__name__

        super().__init__(
            data,
            transform,
            cache_dir=cache_dir,
            *args,
            **kwargs,
        )

    def parse(self) -> list[dict[str, str]]:

        if [
            self.valid_vol_name,
            self.valid_lab_name,
            self.test_vol_name,
            self.test_lab_name,
        ] == [None, None, None, None]:
            return self._rolling_validation_k_fold_split(
                self.parse_pair_fn(self.train_vol_name)
            )

        else:
            return self.parse_pair_fn(
                {
                    TRAIN: self.train_vol_name,
                    VALID: self.valid_vol_name,
                    TEST: self.test_vol_name,
                }[self.mode]
            )

    def parse_pair_fn(self, vol_name):
        vols = sorted(
            (self.root_dir / self.dataset_name / vol_name).glob("[!.]*.nii.gz")
        )  # sort in case when data order is different across server.
        return [
            {
                self.case_id_key: vol.name,
                self.vol_key: vol,
                self.lab_key: str(vol).replace("images", "labels"),
            }
            for vol in vols
            if vol.name not in self.vols_to_exclude
        ]

    def _rolling_validation_k_fold_split(self, datapoints: list[dict[str, Path]]):
        """
        example when  fold_size = 5, and fold_n = 2

        |------||------||------||------||------||------|
        <----TRAIN----><-VALID-><-TEST-><----TRAIN---->
        """

        concat: Callable[[list[list[Any]]], list[Any]] = reduce(add)

        partitions: list[list[dict[str, Path]]] = list(
            partition_all(len(datapoints) // self.fold_size + 1, datapoints)
        )

        return {
            TRAIN: concat(partitions[: self.fold_n] + partitions[self.fold_n + 2 :]),
            VALID: partitions[self.fold_n],
            TEST: partitions[self.fold_n + 1],
        }[self.mode]


class DecathlonNSWDataset(DecathlonDataset):

    dataset_name: str
    global_downsize_rate: list[int]
    local_roi_size: list[int]
    global_roi_size: list[int]
    global_divisible_k: list[int]
    intensity_scale: list[float]
    spacing: list[float]
    do_deform: bool
    new_label_rule: dict[str, list[str]] | None = None
    prior_path: None | Path = None
    samping_proportion: None | list[float] = None
    orientation_mode: str = "RAS"

    def __init__(
        self,
        mode: str,
        fold_n: int,
        fold_size: int = 5,  # 5 fold cross validation
        cache_rate: float = 0.5,
        rand_aug_type: str = ["none", "light", "heavy"][-1],
        *args,
        **kwargs,
    ) -> None:

        # derive some attributese
        self.old_label_rule = compose(list, mc("values"), load_val_from_json("labels"))(
            self.root_dir / self.dataset_name / self.dataset_json_filename
        )

        self.num_channel = compose(len, load_val_from_json("channel_names"))(
            self.root_dir / self.dataset_name / self.dataset_json_filename
        )
        self.labels = (
            list(self.new_label_rule.keys())
            if self.new_label_rule != None
            else self.old_label_rule
        )
        self.num_classes = len(self.labels)

        # make these attributes visible just in case
        self.rand_aug_type = rand_aug_type

        transform = aug(
            spacing=self.spacing,
            local_roi_size=self.local_roi_size,
            global_roi_size=self.global_roi_size,
            intensity_scale=self.intensity_scale,
            old_label_rule=self.old_label_rule,
            new_label_rule=self.new_label_rule,
            global_divisible_k=self.global_divisible_k,
            orientation_mode=self.orientation_mode,
            rand_aug_type=rand_aug_type if mode == TRAIN else "none",
            do_deform=self.do_deform,
        )

        DecathlonDataset.__init__(
            self,
            mode=mode,
            dataset_name=self.dataset_name,
            transform=transform,
            fold_n=fold_n,
            fold_size=fold_size,
            *args,
            **kwargs,
        )


@data_registry.register
class Word(DecathlonNSWDataset):
    dataset_name = "Word"
    intensity_scale = [-250, 500, 0.0, 1.0]
    spacing = [1.0, 1.0, 3.0]
    local_roi_size = [128, 128, 128]
    global_roi_size = [480, 480, 288]
    global_divisible_k = [96, 96, 96]
    global_downsize_rate = [3, 3, 3]
    do_deform = True


if __name__ == "__main__":

    from tqdm import tqdm
    from matplotlib import pyplot as plt
    from utils.visualzation import VisVolLab
    import numpy as np
    from monai.transforms import ResizeD

    task = "Word"
    mode = TEST
    rand_aug_type = "none"
    num_workers = 12

    dataset = data_registry[task](
        mode=mode,
        fold_n=0,
        rand_aug_type=rand_aug_type,
    )

    visualizer = VisVolLab(num_classes=dataset.num_classes)

    down_size = ResizeD(
        keys=[VOL, LAB],
        spatial_size=list(
            np.array(dataset.global_roi_size) / np.array(dataset.global_downsize_rate)
        ),
        mode=("bilinear", "nearest"),
    )

    shapes = []
    for i in tqdm(range(len(dataset))):
        data = dataset[i]
        plt.imshow(
            visualizer.vis(
                data[0][VOL],
                data[0][LAB],
            )
        )
        plt.show()
        print(data[0][VOL].shape)
