from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import torch
from torchmetrics import Metric, Dice
from torchtyping import TensorType

from torch.nn import functional as F
from monai.metrics import compute_hausdorff_distance, DiceMetric, compute_surface_dice
from data.transforms.transform import Patchify, toBatch

from functools import reduce

__all__ = ["Dice", "HAUS", "HAUS_95", "NSD"]

def nanstd(o, dim):
    return torch.sqrt(
        torch.nanmean(
            torch.pow(torch.abs(o - torch.nanmean(o, dim=dim).unsqueeze(dim)), 2),
            dim=dim,
        )
    )

class Dice(Metric):
    def __init__(
        self,
        num_classes: int,
        average: str = ["none", "mean", "all"][0],
        include_background: bool = False,
        label_keys: list[str | None] = [None],
        do_crop_based: bool = False,  # this is not used during validation. to prevent OOM.
        **kwargs,
    ):
        super().__init__(**kwargs)

        if average in ["all", "none"]:
            assert label_keys != [None]

        self.num_classes = num_classes - 1 if include_background else num_classes
        self.average = average
        self.include_background = include_background
        self.label_keys = label_keys if include_background else label_keys[1:]
        self.monai_dice_metric = DiceMetric(
            include_background=include_background, reduction=average
        )
        self.do_crop_based = do_crop_based
        self.add_state("scores", default=[], dist_reduce_fx="cat")

        if self.do_crop_based:
            self.patchify = toBatch(Patchify)(
                keys=["pred", "target"],
                patch_size=[128, 128, 128],
                overlap_r=0,
            )

    def update(
        self,
        pred: TensorType["B", "N", "H", "W", "D", float],
        target: TensorType["B", 1, "H", "W", "D", int],
    ) -> None:
        if pred.shape[2:] != target.shape[2:]:
            raise ValueError("pred and target must have the same shape")

        if self.do_crop_based:
            pred_patches, target_patches = self.patchify(
                {"pred": pred, "target": target}
            ).values()
            patch_scores = []
            for pred_patch, target_patch in zip(pred_patches, target_patches):
                score_patch = self.monai_dice_metric(
                    F.one_hot(
                        pred_patch.unsqueeze(0).argmax(dim=1), self.num_classes
                    ).permute(0, -1, 1, 2, 3),
                    # target_patch.unsqueeze(0).squeeze(1).long(),
                    F.one_hot(
                        target_patch.unsqueeze(0).squeeze(1).long(), self.num_classes
                    ).permute(0, -1, 1, 2, 3),
                )
                patch_scores.append(score_patch)
            self.scores.append(torch.stack(patch_scores, dim=0).nanmean(0))
        else:
            score = self.monai_dice_metric(
                F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, -1, 1, 2, 3),
                # target.long(),
                F.one_hot(target.squeeze(1).long(), self.num_classes).permute(
                    0, -1, 1, 2, 3
                ),
            )
            self.scores.append(score)

    def compute(self) -> float:

        scores = torch.cat(self.scores)

        if self.average == "none":
            return (
                dict(zip(self.label_keys, scores.nanmean(0).tolist()))
                if self.label_keys != [None]
                else scores.nanmean(0)
            )
        if self.average == "mean":
            return {"mean": scores.nanmean()}

        if self.average == "all":

            class_score = dict(
                zip([f"mean_{o}" for o in self.label_keys], scores.nanmean(0).tolist())
            )
            class_std = dict(
                zip([f"std_{o}" for o in self.label_keys], nanstd(scores, 0).tolist())
            )
            mean_score = {"mean": scores.nanmean()}
            mean_std = {"std": nanstd(scores.nanmean(1), 0)}

            return merge(class_score, class_std, mean_score, mean_std)


class Haus(Metric):
    def __init__(
        self,
        num_classes: int,
        average: str = ["none", "mean", "all"][0],
        percentile: float = 95,
        include_background: bool = False,
        label_keys: list[str | None] = [None],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes - 1 if include_background else num_classes
        self.average = average
        self.include_background = include_background
        self.label_keys = label_keys if include_background else label_keys[1:]
        self.percentile = percentile
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(
        self,
        pred: TensorType["B", "N", "H", "W", "D", float],
        target: TensorType["B", 1, "H", "W", "D", int],
    ) -> None:
        if pred.shape[2:] != target.shape[2:]:
            raise ValueError("pred and target must have the same shape")

        score = compute_hausdorff_distance(
            F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, -1, 1, 2, 3),
            F.one_hot(target.squeeze(1).long(), self.num_classes).permute(
                0, -1, 1, 2, 3
            ),
            percentile=self.percentile,
            include_background=self.include_background,
        )

        self.scores.append(score)

    def compute(self) -> float:

        scores: TensorType["n_sample", "n_class", float] = torch.cat(self.scores)

        if self.average == "none":
            return (
                dict(zip(self.label_keys, scores.nanmean(0).tolist()))
                if self.label_keys != [None]
                else scores.nanmean(0).squeeze()
            )

        if self.average == "mean":
            return {"mean": scores.nanmean()}

        if self.average == "all":

            class_score = dict(
                zip([f"mean_{o}" for o in self.label_keys], scores.nanmean(0).tolist())
            )
            class_std = dict(
                zip([f"std_{o}" for o in self.label_keys], nanstd(scores, 0).tolist())
            )
            mean_score = {"mean": scores.nanmean()}
            mean_std = {"std": nanstd(scores.nanmean(1), 0)}

            return merge(class_score, class_std, mean_score, mean_std)


class NSD(Metric):
    def __init__(
        self,
        num_classes: int,
        average: str = ["none", "mean", "all"][0],
        class_thresholds=None,
        include_background: bool = False,
        label_keys: list[str | None] = [None],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes - 1 if include_background else num_classes
        self.average = average
        self.include_background = include_background
        self.label_keys = label_keys if include_background else label_keys[1:]
        self.class_thresholds = (
            [2] * num_classes if class_thresholds is None else class_thresholds
        )
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(
        self,
        pred: TensorType["B", "N", "H", "W", "D", float],
        target: TensorType["B", 1, "H", "W", "D", int],
    ) -> None:
        if pred.shape[2:] != target.shape[2:]:
            raise ValueError("pred and target must have the same shape")

        score = compute_surface_dice(
            F.one_hot(pred.argmax(dim=1), self.num_classes).permute(0, -1, 1, 2, 3),
            F.one_hot(target.squeeze(1).long(), self.num_classes).permute(
                0, -1, 1, 2, 3
            ),
            class_thresholds=(
                self.class_thresholds
                if self.include_background
                else self.class_thresholds[1:]
            ),
            include_background=self.include_background,
        )

        self.scores.append(score)

    def compute(self) -> float:
        scores: TensorType["n_sample", "n_class", float] = torch.cat(self.scores)
        if self.average == "none":
            return (
                dict(zip(self.label_keys, scores.nanmean(0).tolist()))
                if self.label_keys != [None]
                else scores.nanmean(0).squeeze()
            )

        if self.average == "mean":
            return {"mean": scores.nanmean()}

        if self.average == "all":

            class_score = dict(
                zip([f"mean_{o}" for o in self.label_keys], scores.nanmean(0).tolist())
            )
            class_std = dict(
                zip([f"std_{o}" for o in self.label_keys], nanstd(scores, 0).tolist())
            )
            mean_score = {"mean": scores.nanmean()}
            mean_std = {"std": nanstd(scores.nanmean(1), 0)}

            return merge(class_score, class_std, mean_score, mean_std)


if __name__ == "__main__":

    TEST_HAUS = True
    TEST_NSD = True
    TEST_DICE = True

    from utils.visualzation import ShapeGenerator

    x = ShapeGenerator(image_size=[512, 512, 512], shape_type="checkerboard").generate(
        num_classes=3
    )
    pred = F.one_hot(x.long()).squeeze(1).permute(0, -1, 1, 2, 3).float().to("cuda:4")

    if TEST_HAUS:
        for i in range(5):
            haus = Haus(
                percentile=None,
                average="all",
                num_classes=3,
                label_keys=range(3),
                include_background=False,
            )

            target = torch.zeros(x.shape).to("cuda:4")
            target[:, :, i:-i, i:-i, i:-i] = x[:, :, i:-i, i:-i, i:-i]
            haus.update(pred, target)
            print(haus.compute())
        print("\n")

    if TEST_NSD:
        for i in range(5):

            nsd = NSD(
                average="all",
                num_classes=3,
                label_keys=range(3),
                include_background=False,
            )

            target = torch.zeros(x.shape).to("cuda:4")
            target[:, :, i:-i, i:-i, i:-i] = x[:, :, i:-i, i:-i, i:-i]
            nsd.update(pred, target)

            print(nsd.compute())
        print("\n")

    if TEST_DICE:

        for i in range(5):

            dice = Dice(
                average="all",
                num_classes=3,
                label_keys=range(3),
                include_background=False,
            )

            target = torch.zeros(x.shape).to("cuda:4")
            target[:, :, i:-i, i:-i, i:-i] = x[:, :, i:-i, i:-i, i:-i]
            dice.update(pred, target)

            print(dice.compute())
