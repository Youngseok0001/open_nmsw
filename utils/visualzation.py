import random
from torchtyping import TensorType

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from itertools import zip_longest

import numpy as np
import colorcet as cc
import seaborn as sns

import torch
from torchvision.utils import draw_segmentation_masks, make_grid
from torch.nn import functional as F


class VisVolLab:

    def __init__(
        self,
        depth_sample_rate: float = 0.1,
        num_classes: None | int = None,
        colors: list[tuple[float, float, float]] = cc.glasbey,
        alpha: float = 0.25,
    ):
        """_summary_

        Args:
            depth_sample_rate (float, optional): Slice sampling rate in depth dimension. Defaults to 0.1.
            num_classes (int, optional): maximum number of labels the mask input could potentially have. Defaults to None.
            colors (list[list[float]], optional): list of . Defaults to cc.glasbey.
            alpha (float, optional): _description_. Defaults to 0.25.
        """

        self.depth_sample_rate = depth_sample_rate
        self.num_classes = num_classes
        self.colors = colors
        self.alpha = alpha

    def _to_grid(
        self,
        vol: TensorType["C", "H", "W", "D", float],
        normalize=True,
    ) -> TensorType["C", "H'", "W'", float]:

        depth = vol.shape[-1]
        depth_interval = max(1, int(self.depth_sample_rate * depth))
        nrow = depth // depth_interval + 1
        sub_vol = vol[..., range(0, depth, depth_interval)]

        return make_grid(
            sub_vol.permute(-1, 0, 1, 2),  # bring depth to batch_dim
            normalize=normalize,
            nrow=nrow,
            padding=1,
        ).float()

    def _to_one_hot(
        self, lab: TensorType["H", "W", "D", int]
    ) -> TensorType["C", "H", "W", "D", int]:

        self.num_classes = (
            len(lab.unique()) if self.num_classes == None else self.num_classes
        )
        return F.one_hot(lab, num_classes=self.num_classes).permute(
            -1, 0, 1, 2
        )  # bring last channel to 2nd.

    def _to_grid_vol(
        self, vol: TensorType["C", "H", "W", "D", float]
    ) -> TensorType["C", "H'", "W'", float]:
        return (self._to_grid(vol) * 255).to(torch.uint8)

    def _to_grid_lab(
        self, lab: TensorType["H", "W", "D", int]
    ) -> TensorType["C", "H'", "W'", "D'", float]:
        return self._to_grid(self._to_one_hot(lab.long()), normalize=False).bool()

    def _vis(
        self,
        vol: None | TensorType["C", "H", "W", "D", float] = None,
        lab: None | TensorType["H", "1", "W", "D", int] = None,
    ):

        # gernerate synthetic data in the case when either img or lab is empty
        match (torch.is_tensor(vol), torch.is_tensor(lab)):
            case [True, True]:
                assert (
                    vol.shape[-3:] == lab.shape[-3:]
                ), f"image and lab must match in their size but image: {vol.shape[-3:]} and lab: {lab.shape[-3:]}"
            case [True, False]:
                lab = torch.zeros_like(vol)[0:1].long()  # first channel
            case [False, True]:
                vol = torch.zeros_like(lab).float()  # dummy channel dim
            case [False, False]:
                raise RuntimeError(f"Both args:`imgs` and args:`labs` are not tensors.")

        return draw_segmentation_masks(
            self._to_grid_vol(vol),
            self._to_grid_lab(lab.squeeze(0)),
            alpha=self.alpha,
            colors=["#000000"]  # set first label (most probably background) to black
            + sns.color_palette(self.colors, self.num_classes - 1, as_cmap=True),
        ).permute(
            1, 2, 0
        )  # channel-last for pyplot vis later

    def vis(
        self,
        vol: (
            list[None]
            | TensorType["C", "H", "W", "D", float]
            | TensorType["B", "C", "H", "W", "D", float]
        ) = [None],
        lab: (
            list[None]
            | TensorType["1", "H", "W", "D", int]
            | TensorType["B", "1", "H", "W", "D", int]
        ) = [None],
    ) -> TensorType["H'", "W'", "C", float]:

        def f(x):
            if x != [None]:  ## no batch
                if len(x.shape) == 4:
                    x = x.unsqueeze(0)  # add batch
            return x

        return torch.cat(
            [self._vis(v, l) for v, l in zip_longest(f(vol), f(lab), fillvalue=None)],
            dim=0,
        )


class ShapeGenerator:

    # constraints
    shape_types = ["checkerboard", "circle"]

    def __init__(
        self,
        image_size: tuple[int, int, int] = (256, 256, 256),
        batch_size: int = 1,
        shape_type: str = ["checkerboard", "circle"][-1],
    ):
        assert (
            shape_type in ShapeGenerator.shape_types
        ), f"argL`shape_type` must be one of {ShapeGenerator.shape_types}"

        self.image_size = image_size
        self.batch_size = batch_size
        self.shape_type = shape_type
        self._generator = {
            "checkerboard": self._generate_checkerboard,
            "circle": self._generate_circle,
        }[self.shape_type]

    def generate(self, **shape_args):
        return torch.stack(
            [self._generator(**shape_args) for _ in range(self.batch_size)]
        ).unsqueeze(
            1
        )  # add channel dim

    # alias
    __call__ = generate

    def _generate_checkerboard(
        self,
        patch_number: int = 4,
        num_classes: int = 10,
    ) -> TensorType["H", "W", "D", int]:

        checkerboard = torch.zeros(self.image_size)
        steps = [s // patch_number for s in self.image_size]

        for i in range(patch_number):
            for j in range(patch_number):
                for k in range(patch_number):
                    if (i + j + k) % 2 == 1:
                        random_value = random.randint(0, num_classes - 1)
                        checkerboard[
                            i * steps[0] : (i + 1) * steps[0],
                            j * steps[1] : (j + 1) * steps[1],
                            k * steps[2] : (k + 1) * steps[2],
                        ] = random_value

        return checkerboard

    def _generate_circle(
        self,
        radius: int = 32,
        at_center: bool = True,
    ) -> TensorType["H", "W", "D", int]:

        assert (
            radius <= min(self.image_size) / 2
        ), f"radius: {radius} must be smaller than half the size of image_size: {self.image_size}"

        circle = torch.zeros(*self.image_size)

        if at_center:
            pivot = [self.image_size[i] // 2 for i in range(2)]
        else:
            pivot = np.random.randint(
                radius,
                min(self.image_size) - radius,
                size=2,
            )

        for i in range(self.image_size[0]):
            for j in range(self.image_size[1]):
                if (i - pivot[0]) ** 2 + (j - pivot[1]) ** 2 <= radius**2:
                    circle[i, j] = 1

        return circle


if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from itertools import product

    TEST_SHAPE = False
    TEST_VIS = True

    image_sizes = [(128, 128, 128)]
    batch_sizes = [2]
    shape_types = ["checkerboard", "circle"]

    if TEST_SHAPE:

        for i_s, b_s, s_t in product(image_sizes, batch_sizes, shape_types):
            shape_generator = ShapeGenerator(i_s, b_s, s_t)
            shape = shape_generator.generate()
            plt.imshow(shape[0, 0, ..., 64])
            plt.show()

    if TEST_VIS:

        visualizer = VisVolLab()

        for i_s, b_s, s_t in product(image_sizes, batch_sizes, shape_types):

            shape_generator = ShapeGenerator(i_s, b_s, s_t)

            vol = torch.randn(b_s, 1, *i_s)
            lab = shape_generator.generate()

            plt.imshow(visualizer.vis(lab=lab))
            plt.show()
            plt.imshow(visualizer.vis(vol=vol))
            plt.show()
            plt.imshow(visualizer.vis(vol=vol, lab=lab))
            plt.show()
