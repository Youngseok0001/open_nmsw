# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from typing import List

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

from torch import nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler


def set_scheduler(optimizer, sch_kwargs):
    """
    polylr if optimizer == sgd else cosineAnnealingLR
    """

    opt_name: str = optimizer.__class__.__name__

    sch: _LRScheduler = {
        1: PolyLRSch,
        0: LinearWarmupCosineAnnealingLRSch,
    }[opt_name == "SGD"]

    sch_name: str = sch.__name__

    sch_kwargs.pop("name")
    sch_kwargs: dict = sch_kwargs

    return sch(optimizer, **sch_kwargs)


class PolyLRSch(_LRScheduler):
    """
    modified from :
        "https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/lr_scheduler/polylr.py#L4"
    """

    def __init__(
        self,
        optimizer: Optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(
            optimizer, current_step if current_step is not None else -1, False
        )

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class _LRSchedulerMONAI(_LRScheduler):
    """Base class for increasing the learning rate between two boundaries over a number
    of iterations"""

    def __init__(
        self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1
    ) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            end_lr: the final learning rate.
            num_iter: the number of iterations over which the test occurs.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.end_lr = end_lr
        self.num_iter = num_iter
        super(_LRSchedulerMONAI, self).__init__(optimizer, last_epoch)


class LinearLR(_LRSchedulerMONAI):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRSchedulerMONAI):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Based on https://huggingface.co/ implementation.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        t_total: int,
        cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            warmup_steps: number of warmup iterations.
            t_total: total number of training iterations.
            cycles: cosine cycles parameter.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )


class LinearWarmupCosineAnnealingLRSch(_LRScheduler):

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epoch_ratio: float,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """

        self.warmup_epochs = int(max_epochs * warmup_epoch_ratio)
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLRSch, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (
            2 * (self.max_epochs - self.warmup_epochs)
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min)
                * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs)))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + self.last_epoch
                * (base_lr - self.warmup_start_lr)
                / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            for base_lr in self.base_lrs
        ]


if __name__ == "__main__":

    import torch
    from torch.optim import SGD, Adam

    TEST_SET_SCH = True

    if TEST_SET_SCH:

        optmizer_names = ["SGD", "Adam"]
        net = torch.nn.Linear(10, 1)
        default_kwargs_d: dict = {
            "LinearWarmupCosineAnnealingLRSch": {
                "warmup_epoch_ratio": 0.1,
                "max_epochs": 100,
            },
            "PolyLRSch": {
                "initial_lr": 0.0001,
                "max_steps": 100,
            },
        }

        for optimizer_name in optmizer_names:

            print(f"testing {optimizer_name}...", end="")

            opt = eval(optimizer_name)(
                net.parameters(),
                lr=0.0001,
            )
            sch = set_scheduler(
                opt,
                default_kwargs_d[
                    (
                        "PolyLRSch"
                        if optimizer_name == "SGD"
                        else "LinearWarmupCosineAnnealingLRSch"
                    )
                ],
            )

            print("works!")
