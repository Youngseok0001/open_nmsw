# this loss should only be used for training the model.
# for evaulation, I need a new dice function that outputs class-wise dice
# I have to evaluation the model's performance for each label.
# I expect to see my model model performing particularly good at segmenting small objects.
from __future__ import division, print_function
import random

import numpy as np
import torch
from torch import nn
from torchtyping import TensorType
from model.sampler.registry import loss_sampler_registry
from utils.definitions import *

from monai.data.meta_tensor import MetaTensor


@loss_sampler_registry.register
def TORandPool(LossClass):

    class RandPoolLoss(LossClass):

        def __init__(
            self,
            sampler_args={},
            loss_args={},
        ):
            LossClass.__init__(self, **loss_args)
            self.rand_pool = RandPool(**sampler_args)

        def forward(
            self,
            logit: TensorType["B", "C", "H", "W", "D"],
            lab: TensorType["B", "1", "H", "W", "D"],
        ):
            logit_patches, lab_patches = self.rand_pool.rand_pool(logit, lab)
            return LossClass.forward(self, logit_patches, lab_patches)

    return RandPoolLoss


class RandPool(nn.Module):

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

    def rand_pool(self, x, y):
        # Ensure the input size is divisible by 2 in all three spatial dimensions
        assert (
            x.size(-1) % 2 == 0 and x.size(-2) % 2 == 0 and x.size(-3) % 2 == 0
        ), "Input dimensions (h, w, d) must be divisible by 2."

        rand_indicies = self.get_rand_indices(x)
        x = self.pool_sample(x, rand_indicies)
        y = self.pool_sample(y, rand_indicies)

        # z = torch.cat([x, y], dim=1)

        # # Split tensor into 2x2x2 cubes
        # z = (
        #     z.unfold(2, 4, 4).unfold(3, 4, 4).unfold(4, 4, 4)
        # )  # shape: (b, c, h//2, w//2, d//2, 2, 2, 2)
        # b, c, h, w, d, _, _, _ = z.shape

        # # Flatten the 2x2x2 cubes into 8 values for easier random selection
        # z = z.reshape(b, c, h, w, d, 64)  # shape: (b, c, h//2, w//2, d//2, 8)

        # # Generate random indices with smaller data type (if applicable)
        # rand_indices = torch.randint(0, 64, (b, c, h, w, d), device=z.device)
        # # Gather values with combined operations
        # z = z.gather(-1, rand_indices.unsqueeze(-1)).squeeze(-1)

        return x, y

    def pool_sample(self, x, rand_indicies):

        # Split tensor into 2x2x2 cubes
        x = (
            x.unfold(2, 2, 2).unfold(3, 2, 2).unfold(4, 2, 2)
        )  # shape: (b, c, h//2, w//2, d//2, 2, 2, 2)
        b, c, h, w, d, _, _, _ = x.shape

        # Flatten the 2x2x2 cubes into 8 values for easier random selection
        x = x.reshape(b, c, h, w, d, 8)  # shape: (b, c, h//2, w//2, d//2, 8)
        # print(rand_indicies.unsqueeze(1).repeat(1, c, 1, 1, 1).unsqueeze(-1).shape)
        # Gather values with combined operations
        x = x.gather(
            -1, rand_indicies.unsqueeze(1).repeat(1, c, 1, 1, 1).unsqueeze(-1)
        ).squeeze(-1)

        return x

    def get_rand_indices(self, x):

        x = (
            x.unfold(2, 2, 2).unfold(3, 2, 2).unfold(4, 2, 2)
        )  # shape: (b, c, h//2, w//2, d//2, 2, 2, 2)
        b, c, h, w, d, _, _, _ = x.shape
        rand_indices = torch.randint(0, 8, (b, h, w, d), device=x.device)
        return rand_indices


@loss_sampler_registry.register
def ToPatchLossV1(LossClass):

    class PatchLoss(LossClass, PatchSamplerV1):

        def __init__(
            self,
            sampler_args={
                "num_patches": 1,
                "patch_size": [128, 128, 128],
            },
            loss_args={},
        ):
            LossClass.__init__(self, **loss_args)
            PatchSamplerV1.__init__(self, **sampler_args)

        def forward(
            self,
            logit: TensorType["B", "C", "H", "W", "D"],
            lab: TensorType["B", "1", "H", "W", "D"],
        ):
            logit_patches, lab_patches = self.sample_patches(logit, lab)
            return LossClass.forward(self, logit_patches, lab_patches)

    return PatchLoss


@loss_sampler_registry.register
def ToPatchLossV2(LossClass):

    class PatchLoss(LossClass, PatchSamplerV2):

        def __init__(
            self,
            sampler_args={
                "num_patches": 1,
                "patch_size": [128, 128, 128],
            },
            loss_args={},
        ):
            LossClass.__init__(self, **loss_args)
            PatchSamplerV2.__init__(self, **sampler_args)

        def forward(
            self,
            logit: TensorType["B", "C", "H", "W", "D"],
            lab: TensorType["B", "1", "H", "W", "D"],
        ):
            logit_patches, lab_patches = self.sample_patches(logit, lab)
            return LossClass.forward(self, logit_patches, lab_patches)

    return PatchLoss


class PatchSamplerV1:
    def __init__(
        self,
        num_patches: int = 1,
        patch_size: list[int] = [128, 128, 128],
        *args,
        **kwargs,
    ):

        self.num_patches = num_patches
        self.patch_size = patch_size

    def sample_patches(self, logit, lab):
        B, C, H, W, D = logit.shape
        patch_H, patch_W, patch_D = self.patch_size

        selected_boxes = []
        for b in range(B):
            for _ in range(self.num_patches):
                fg_voxels = (lab.squeeze(1)[b] > 0).nonzero(as_tuple=False)
                if len(fg_voxels) != 0:
                    center = fg_voxels[random.randint(0, len(fg_voxels) - 1)]
                    center = torch.clamp(
                        center,
                        min=torch.tensor([patch_H // 2, patch_W // 2, patch_D // 2]).to(
                            logit.device
                        ),
                        max=torch.tensor(
                            [H - patch_H // 2, W - patch_W // 2, D - patch_D // 2]
                        ).to(logit.device),
                    )  # prevent patch exceeding max/min image coordinate
                else:
                    center = torch.tensor(
                        [patch_H // 2, patch_W // 2, patch_D // 2]
                    ).to(logit.device)

                box = [
                    [center[0] - patch_H // 2, center[0] + patch_H // 2],
                    [center[1] - patch_W // 2, center[1] + patch_W // 2],
                    [center[2] - patch_D // 2, center[2] + patch_D // 2],
                ]  # box around sample
                box = torch.tensor(box)
                selected_boxes.append(box)

        logit_patches = []
        lab_patches = []
        for box in selected_boxes:

            slices = [slice(b, b + 1), slice(None)] + [
                slice(box[dim][0], box[dim][1]) for dim in range(3)
            ]
            logit_patch = logit[slices]
            lab_patch = lab[slices]

            logit_patches.append(logit_patch)
            lab_patches.append(lab_patch)

        # Concatenate all patches into batch format
        logit_patches = torch.cat(logit_patches, dim=0)
        lab_patches = torch.cat(lab_patches, dim=0)

        # vis = VisVolLab()
        # plt.imshow(vis.vis(lab=logit_patches.argmax(1, keepdim=True).detach().cpu()))
        # plt.show()
        # plt.imshow(vis.vis(lab=lab_patches.detach().cpu()))
        # plt.show()

        return logit_patches, lab_patches

    def overlap_percentage(self, box1, box2):
        """
        Check the percentage overlap between two 3D bounding boxes.
        If box2 is None, return 0 (i.e., no overlap).
        Return the overlap percentage.
        """
        if box2 == None or box1 == None:
            return 0.0

        # Calculate volumes of both boxes
        # box1_volume = torch.prod([box1[dim][1] - box1[dim][0] for dim in range(3)])
        # box2_volume = torch.prod([box2[dim][1] - box2[dim][0] for dim in range(3)])

        box1_volume = torch.prod(box1[:, 1] - box1[:, 0])
        box2_volume = torch.prod(box2[:, 1] - box2[:, 0])
        intersection = torch.min(box1[:, 1], box2[:, 1]) - torch.max(
            box1[:, 0], box2[:, 0]
        )
        intersection_volume = (
            torch.prod(intersection)
            if torch.all(intersection > 0)
            else torch.tensor(0.0)
        )

        # Calculate overlap percentage based on the smaller volume
        min_volume = min(box1_volume, box2_volume)
        return intersection_volume / min_volume


class PatchSamplerV2:

    def __init__(
        self,
        num_patches: int = 1,
        patch_size: list[int] = [128, 128, 128],
        *args,
        **kwargs,
    ):
        self.num_patches = num_patches
        self.patch_size = patch_size

    def sample_patches(self, logit, lab):

        assert isinstance(logit, MetaTensor)
        assert "slice" in logit.meta.keys()
        assert (len(logit.shape)) == 5
        assert (len(lab.shape)) == 5
        assert logit.shape[0] == 1, f"currently only supports single batch instance"

        min_center, max_center = self.get_center_range(logit)

        logit_patches = []
        lab_patches = []
        for n in range(self.num_patches):
            center = self.get_random_center(min_center, max_center)
            patch_slices = [slice(None), slice(None)] + [
                slice(c - (p // 2), c + (p // 2))
                for (p, c) in zip(self.patch_size, center)
            ]
            logit_patch = logit[patch_slices]
            lab_patch = lab[patch_slices]

            logit_patches.append(logit_patch)
            lab_patches.append(lab_patch)

        return torch.cat(logit_patches), torch.cat(lab_patches)

    def get_center_range(self, logit):

        slicess = logit.meta["slice"]

        # Extract start and stop values with torch tensors
        starts = torch.tensor(
            [[s.start for s in eval(slices)[1:]] for slices in slicess]
        )
        stops = torch.tensor([[s.stop for s in eval(slices)[1:]] for slices in slicess])

        # Calculate the minimum and maximum bounding box values
        min_start = torch.min(starts, dim=0).values
        max_start = torch.max(stops, dim=0).values

        # prevent border sampling by add/subtracting self.patchsize
        min_center: TensorType[3] = min_start + torch.tensor(self.patch_size) / 2
        max_center: TensorType[3] = max_start - torch.tensor(self.patch_size) / 2

        return (min_center, max_center)

    def get_random_center(self, min_center, max_center):
        return torch.tensor(
            [
                torch.randint(
                    int(min_c),
                    int(max_c) if max_c != min_c else int(max_c + 1),
                    size=(1,),
                )
                for min_c, max_c in zip(min_center, max_center)
            ]
        )


if __name__ == "__main__":

    from utils.visualzation import ShapeGenerator
    from model.loss.registry import loss_registry
    from torch.nn.functional import one_hot

    TEST_SAMPLER1 = False
    TEST_SAMPLER2 = False
    TEST_SAMPLER3 = True

    vol_size = [512, 512, 512]
    device = "cuda:1"
    lab = (
        ShapeGenerator(
            image_size=vol_size,
            shape_type="checkerboard",
            batch_size=1,
        )
        .generate(num_classes=2)
        .type(torch.long)
    ).to(device)

    prob = one_hot(lab.squeeze(1)).permute(0, -1, 1, 2, 3).to(device).to(torch.float32)
    logit = torch.log((prob + 1e-5) / (1 - prob + 1e-5))

    if TEST_SAMPLER3:

        num_patches = 5
        patch_size = [128, 128, 128]

        loss_fn = loss_sampler_registry["TORandPool"](
            loss_registry["DiceCELossNNUNET"]  # "RCELoss"
        )()

        val = loss_fn(logit, lab)
        print(val)

    if TEST_SAMPLER1:

        num_patches = 5
        patch_size = [128, 128, 128]

        loss_fn = loss_sampler_registry["ToPatchLossV1"](
            loss_registry["DiceCELossNNUNET"]  # "RCELoss"
        )(
            num_patches=num_patches,
            patch_size=patch_size,
        )

        val = loss_fn(logit, lab)
        print(val)

    if TEST_SAMPLER2:

        slice_meta = [
            "[slice(None, None, None), slice(0, 128, None), slice(0, 128, None), slice(0, 128, None)]",
            "[slice(None, None, None), slice(0, 128, None), slice(0, 128, None), slice(96, 224, None)]",
            "[slice(None, None, None), slice(0, 128, None), slice(0, 128, None), slice(192, 320, None)]",
            "[slice(None, None, None), slice(0, 128, None), slice(0, 128, None), slice(288, 416, None)]",
            "[slice(None, None, None), slice(0, 128, None), slice(0, 128, None), slice(384, 512, None)]",
            "[slice(None, None, None), slice(0, 128, None), slice(96, 224, None), slice(0, 128, None)]",
            "[slice(None, None, None), slice(0, 128, None), slice(96, 224, None), slice(96, 224, None)]",
        ]

        logit = MetaTensor(logit, meta={"slice": slice_meta})

        num_patches = 1
        patch_size = [128, 128, 128]

        loss_fn = loss_sampler_registry["ToPatchLossV2"](
            loss_registry["RCELoss"]  # "RCELoss"
        )(
            patch_size=patch_size,
            num_patches=num_patches,
        )

        val = loss_fn(logit, lab)
        print(val)


# def ToPatchLoss(LossClass):

#     class PatchLoss(LossClass):

#         def forward(
#             self,
#             logit_patches: TensorType["KB", "1", "Hp", "Wp", "Dp"],
#             lab: TensorType["B", "1", "H", "W", "D"],
#         ):

#             assert isinstance(logit_patches, MetaTensor)
#             assert "slice" in logit_patches.meta.keys()
#             assert "crop_center" in logit_patches.meta.keys()

#             # derive some variables
#             batch_size = lab.shape[0]
#             num_patches_per_batch = logit_patches.shape[0] // batch_size
#             patch_size = logit_patches.shape[2:]

#             # extract patches from lab
#             lab_patches = []
#             for i, patch in enumerate(logit_patches):
#                 batch_index = i // num_patches_per_batch
#                 slices = eval(patch.meta["slice"])
#                 lab_patch = lab[[slice(batch_index, batch_index + 1)] + slices]
#                 lab_patches.append(lab_patch)
#                 # print(slices)
#                 # print(lab_patch.shape)
#                 # print("\n")
#             lab_patches = torch.concatenate(lab_patches, dim=0)
#             loss_patch = LossClass.forward(self, logit_patches, lab_patches)

#             return loss_patch

#     return PatchLoss
