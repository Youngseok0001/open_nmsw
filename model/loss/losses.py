from __future__ import division, print_function
import torch
from monai.networks.utils import one_hot
from model.loss.modified_monai_dice_class import DiceCELoss as _DiceCELoss
from model.loss.registry import loss_registry
from utils.definitions import *

__all__ = ["DiceCELossMONAI"]


@loss_registry.register
class DiceCELossMONAI(_DiceCELoss):

    def __init__(
        self,
        to_onehot_y=True,
        softmax=True,
        squared_pred=False,
        batch=True,
        reduction="mean",
        lambda_dice=1,
        lambda_ce=1,
        ce_weight=None,
        dice_weight=None,
        include_background=False,
        *args,
        **kwargs,
    ):
        super().__init__(
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            squared_pred=squared_pred,
            batch=batch,
            reduction=reduction,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            include_background=include_background,
        )


if __name__ == "__main__":

    from utils.visualzation import ShapeGenerator
    from torch.nn.functional import one_hot

    TEST_DICE_MONAI = True

    vol_size = [256, 256, 256]

    lab = (
        ShapeGenerator(
            image_size=vol_size,
            shape_type="checkerboard",
            batch_size=1,
        )
        .generate(num_classes=2)
        .type(torch.long)
    ).to("cuda:7")

    prob = (
        one_hot(lab.squeeze(1)).permute(0, -1, 1, 2, 3).to("cuda:7").to(torch.float32)
    )
    logit = torch.log((prob + 1e-5) / (1 - prob + 1e-5))

    if TEST_DICE_MONAI:
        loss_fn = loss_registry["DiceCELossMONAI"]()
        val = loss_fn(logit, lab)
        print(val)
