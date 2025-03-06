from typing import Callable, Any

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
import torch.nn as nn

from model.loss.registry import loss_registry
from monai.networks.layers.factories import Conv

from utils.visualzation import VisVolLab, ShapeGenerator
from utils.definitions import *
from matplotlib import pyplot as plt


# @backbone_registry.register
class Seg3D(nn.Module):
    """
    A generic segmentation backbone which has the encoder decoder structure.
    This Wrapper elliviates the need to define the forward and loss methods, which are pretty much
    identical across models.
    """

    spatial_dims = 3

    def __init__(
        self,
        input_shape: list[int],
        output_shape: list[int],
        loss_args: dict[str, Any] = {
            "name": "DiceCELossMONAI",
        },
    ):
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss_args = loss_args

        # dummy encoder
        self.encoder: Callable[[torch.tensor], Any] = Conv["conv", self.spatial_dims](
            self.input_shape[0], 1, 1
        )
        # dummy decoder
        self.decoder: Callable[[Any], torch.tensor] = Conv["conv", self.spatial_dims](
            1, self.output_shape[0], 1
        )

        self.loss_fn: Callable[[torch.tensor, torch.tensor], float] = loss_registry[
            self.loss_args["name"]
        ](**self.loss_args)

    def forward(self, vol: torch.tensor) -> torch.tensor:
        return compose(self.decoder, self.encoder)(vol)

    def get_loss(self, logit: torch.tensor, lab: torch.tensor):
        return self.loss_fn(logit, lab)

    @property
    def feature_shapes(self):
        with torch.no_grad():
            x = (
                torch.Tensor(*self.input_shape)
                .unsqueeze(0)
                .to(next(self.parameters()).device)
            )
            features = self.decoder(self.encoder(x))
        del x
        return [feature.shape[1:] for feature in features]

    def size(self):
        def _size(model):
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            return sum([np.prod(p.size()) for p in model_parameters])

        return _size(self.encoder) + _size(self.decoder.size)

    def unit_test(self, batch_size=2):
        print(
            f"testing the case when input: {self.input_shape} output: {self.output_shape}..."
        )

        device = next(self.parameters()).device

        vol = torch.randn(batch_size, *self.input_shape).to(device)
        lab = (
            ShapeGenerator(
                self.output_shape[1:],
                batch_size=batch_size,
            )
            .generate()
            .to(device)
        )

        outputs = self(vol)
        logit = outputs[-1]
        print(f"score_size : {outputs[-2].shape}")
        loss = self.get_loss(logit, lab)
        print(f"loss: {loss}")
        loss.backward()

        visualizer = VisVolLab(num_classes=self.output_shape[0])

        plt.imshow(visualizer.vis(vol=vol).detach().cpu())
        plt.show()
        plt.imshow(visualizer.vis(lab=logit.argmax(1).unsqueeze(1).detach().cpu()))
        plt.show()

        plt.imshow(visualizer.vis(vol=lab.detach().cpu()))
        plt.show()
        print("works!")


if __name__ == "__main__":

    net = Seg3D(
        [3, 128, 128, 128],
        [5, 128, 128, 128],
    ).to("cuda:0")
    net.unit_test()
