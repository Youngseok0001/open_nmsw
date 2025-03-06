from typing import Callable
from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from utils.definitions import *
from monai.transforms import (
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
    EnsureChannelFirstd,
    SelectItemsd,
    RandAffined,
    RandSimulateLowResolutiond,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianNoised,
    RandScaleIntensityd,
    IdentityD,
    DivisiblePadD,
    ResizeWithPadOrCropd,
    SpatialPadD,
    RandCropByPosNegLabelD,
    CopyItemsD,
    RandGridDistortiond,
)

from data.transforms.transform import MergeLabelD
from pathlib import Path
import torch


def aug(
    spacing: list[float],
    local_roi_size: list[int],
    global_roi_size: list[int],
    intensity_scale: list[float],
    old_label_rule: list[str],
    new_label_rule: dict[str, list[str]] | None = None,
    global_divisible_k: list[int] = [32, 32, 48],  # 16*2 for h,w and # 16*3 for global
    orientation_mode: str = "RAS",
    rand_aug_type=["none", "heavy"][-1],
    do_deform: bool = True,  # grid deformation if the dataset has little training sample
) -> Callable[[dict[str, Path]], dict[str, torch.Tensor]]:

    pre_aug = [
        LoadImaged(keys=[VOL, LAB]),
        EnsureChannelFirstd(keys=[VOL, LAB]),
        Orientationd(keys=[VOL, LAB], axcodes=orientation_mode),
        ScaleIntensityRanged(
            keys=VOL,
            a_min=intensity_scale[0],
            a_max=intensity_scale[1],
            b_min=intensity_scale[2],
            b_max=intensity_scale[3],
            clip=True,
        ),
        CropForegroundd(keys=[VOL, LAB], source_key=VOL),
        Spacingd(keys=[VOL, LAB], pixdim=spacing, mode=("bilinear", "nearest")),
        SelectItemsd([VOL, LAB, CASE_ID]),
        ############################################
        # PAD IF VOLUME IS SMALLLER THAN ROIs
        SpatialPadD(
            keys=[VOL],
            spatial_size=local_roi_size,
            mode=("constant",),
            value=-0.1,
        ),
        SpatialPadD(
            keys=[LAB],
            spatial_size=local_roi_size,
            mode=("constant",),
            value=0,
        ),
        ToTensord(keys=[VOL, LAB]),
    ]

    if rand_aug_type == "heavy":
        # Closely follows nnunet's augmentation rule
        random_aug = [
            RandShiftIntensityd(
                keys=[VOL],
                offsets=0.10,
                prob=0.20,
            ),
            RandAffined(
                keys=[VOL, LAB],
                prob=1.0,
                mode=["bilinear", "nearest"],
                padding_mode="zeros",
                translate_range=[50.0, 50.0, 50.0],
            ),  # this is needed for more variation exposure to nmsw-net
            RandAffined(
                keys=[VOL, LAB],
                prob=0.3,
                mode=["bilinear", "nearest"],
                padding_mode="zeros",
                rotate_range=[0.52, 0.52, 0.52],
                scale_range=[0.2, 0.2, 0.3],  # more zooming in axial direction
                translate_range=[10.0, 10.0, 50.0],
            ),
            RandGaussianNoised(prob=0.1, keys=[VOL]),
            RandGaussianSmoothd(
                prob=0.2,
                sigma_x=(0.5, 1.0),
                sigma_y=(0.5, 1.0),
                sigma_z=(0.5, 1.0),
                keys=[VOL],
            ),
            RandScaleIntensityd(
                prob=0.15,
                factors=(-0.25, 0.25),
                keys=[VOL],
            ),
            RandSimulateLowResolutiond(
                prob=0.25,
                zoom_range=(0.5, 1),
                keys=[VOL],
            ),
            RandAdjustContrastd(
                prob=0.1,
                gamma=(0.7, 1.5),
                invert_image=True,
                retain_stats=True,
                keys=[VOL],
            ),
            RandAdjustContrastd(
                prob=0.3,
                gamma=(0.7, 1.5),
                invert_image=False,
                retain_stats=True,
                keys=[VOL],
            ),
            (
                RandGridDistortiond(
                    prob=0.3,
                    num_cells=5,
                    distort_limit=(-0.05, 0.05),
                    keys=[VOL, LAB],
                    mode=["bilinear", "nearest"],
                )
                if do_deform
                else IdentityD(keys=[VOL, LAB])
            ),
        ]

    elif rand_aug_type == "none":
        random_aug = [
            # random to save before padding
            RandShiftIntensityd(
                keys=[VOL],
                offsets=0.0,
                prob=0.01,
            ),
            IdentityD(
                keys=[VOL, LAB],
            ),
        ]

    else:
        raise ValueError(
            f"`rand_aug_type`:{rand_aug_type} is not one of [heavy | light | none]"
        )

    post_aug = [
        {
            1: MergeLabelD(
                keys=LAB,
                old_label_rule=old_label_rule,
                new_label_rule=new_label_rule,
            ),
            0: IdentityD(keys=LAB),
        }[new_label_rule != None],
        ############################################
        # MAKE SURE THE VOL CAN BE DOWN_SAMPLED ATLEAST 4TIMES, AS REQUIRED BY MOST SEGMENTATION BACKBONE.
        ############################################
        DivisiblePadD(
            keys=[VOL, LAB],
            k=global_divisible_k,
            mode=["constant", "constant"],
        ),
        ############################################
        # UNIFY the volume size.
        ############################################
        ResizeWithPadOrCropd(
            keys=[VOL],
            spatial_size=global_roi_size,
            mode=("constant"),
            value=-0.1,
        ),  #
        ResizeWithPadOrCropd(
            keys=[LAB],
            spatial_size=global_roi_size,
            mode="constant",
            value=0,
        ),
        CopyItemsD(
            keys=[VOL, LAB],
            names=[PATCH_VOL, PATCH_LAB],
        ),
        RandCropByPosNegLabelD(
            keys=[PATCH_VOL, PATCH_LAB],
            label_key=PATCH_LAB,
            spatial_size=local_roi_size,
            pos=1,
            neg=1,
            num_samples=1,
        ),
        SelectItemsd(keys=[VOL, LAB, PATCH_VOL, PATCH_LAB, CASE_ID]),
    ]
    return Compose(pre_aug + random_aug + post_aug)


if __name__ == "__main__":

    from utils.visualzation import VisVolLab
    from matplotlib import pyplot as plt

    TEST_DATASET = True

    if TEST_DATASET:

        data = {
            "vol": "/hpc/home/jeon74/no-more-sw/data/datasets/processed/Word/imagesTr/word_0002.nii.gz",
            "lab": "/hpc/home/jeon74/no-more-sw/data/datasets/processed/Word/labelsTr/word_0002.nii.gz",
            "case_id": "shit",
        }

        mode = TRAIN
        rand_aug_type = "heavy"

        new_label_rule = None
        old_label_rule = [
            "background",
            "liver",
            "spleen",
            "left_kidney",
            "right_kidney",
            "stomach",
            "gallbladder",
            "esophagus",
            "pancreas",
            "duodenum",
            "colon",
            "intestine",
            "adrenal",
            "rectum",
            "bladder",
            "Head_of_femur_L",
            "Head_of_femur_R",
        ]

        augmentation = aug(
            spacing=[1, 1, 1],
            intensity_scale=(-250, 500, 0.0, 1.0),
            old_label_rule=old_label_rule,
            new_label_rule=new_label_rule,
            local_roi_size=(128, 128, 128),
            global_roi_size=(256, 256, 512),
            rand_aug_type=rand_aug_type,
        )

        data = augmentation(data)

        visualizer = VisVolLab(num_classes=20)

        print(data.keys())
        print(data[VOL].shape)
        print(data[LAB].shape)

        img = visualizer.vis(
            data[VOL].detach().cpu(),
            data[LAB].detach().cpu(),
        )
        plt.figure(figsize=(30, 10))
        plt.title(data["case_id"])
        plt.imshow(img)
        plt.show()
