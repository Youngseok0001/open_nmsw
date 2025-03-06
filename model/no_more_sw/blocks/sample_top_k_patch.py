from torchtyping import TensorType

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import torch
from torch import nn
import numpy as np

from monai.utils import ensure_tuple_rep
from monai.data.utils import list_data_collate
from monai.transforms import Compose, SelectItemsD
from monai.data.meta_tensor import MetaTensor

from utils.definitions import *
from data.transforms.transform import toBatch, Patchify
from model.no_more_sw.blocks.top_k import GumbelTopK


class SampleTopKPatch(nn.Module):

    def __init__(
        self,
        keys: list[str],
        patch_size: list[int],
        overlap_r: float | list[float],
        tau: float = 0.01,
    ):

        super().__init__()

        self.keys = keys
        self.patch_size = patch_size
        self.overlap_r = overlap_r

        self.patchify = Compose(
            [
                toBatch(Patchify)(
                    keys=self.keys,
                    patch_size=self.patch_size,
                    overlap_r=self.overlap_r,
                ),
                SelectItemsD(keys=self.keys),
            ]
        )

        self.get_top_k_sample = GumbelTopK(tau=tau)

        self.register_buffer(
            "background_epsilon",
            torch.tensor([self.get_top_k_sample.epsilon]),
        )

    @property
    def tau(self):
        return self.get_top_k_sample.tau

    @tau.setter
    def tau(self, value):
        self.get_top_k_sample.tau = value

    def forward(
        self,
        input_d: dict[str, torch.tensor],
        logit: TensorType["B", "1", "Hn", "Wn", "Dn"],  # span -inf to +inf
        background_mask: float | TensorType["B", "1", "Hn", "Wn", "Dn"] = torch.tensor(
            0
        ),
        k: int = 5,
        mode=TRAIN,
    ) -> TensorType["BK", "C", "Hp", "Wp", "Dp"]:

        # prepare variables to be used
        B, *CHWD = logit.shape

        # get all the patches from the full-res input
        # both input and label
        local_patches_d: TensorType["BHnWnDn", "C", "Hp", "Wp", "Dp"] = self.patchify(
            input_d
        )
        # background_mask is 1 if background else 0 (foreground)
        # log_mask is a large negative value if background_mask is 1 else 0
        log_mask = torch.log(torch.max(1.0 - background_mask, self.background_epsilon))

        # paste logit with log_mask to prevent sampling from backgroud region
        masked_logit: TensorType["B", "1", "Hn", "Wn", "Dn"] = logit + log_mask

        # sample one-hots from logit
        one_hots_flatten, soft_hots_flatten = self.get_top_k_sample(
            masked_logit.flatten(1),
            k=k,
            mode=mode,
        )
        one_hots_flatten: TensorType["K", "BHnWnDn"] = one_hots_flatten.flatten(1)
        soft_hots_flatten: TensorType["K", "BHnWnDn"] = soft_hots_flatten.flatten(1)

        # keep this meta info
        patch_indexes: TensorType["K", int] = (
            one_hots_flatten.detach().cpu().numpy().argmax(-1)
        )

        # samples top-k patches
        slice_meta = [local_patches_d[VOL].meta["slice"][i] for i in patch_indexes]

        output_d = {}
        for key, local_patches in local_patches_d.items():
            # too much mememory consumption
            patches: TensorType["K", "C", "Hp", "Wp", "Dp"] = torch.einsum(
                "nchwd,kn -> kchwd",
                local_patches,
                (
                    one_hots_flatten
                    if key == LAB
                    else (
                        one_hots_flatten * soft_hots_flatten
                        if mode == TRAIN
                        else one_hots_flatten
                    )
                ),
            )
            # attach modified meta tensor
            patches.meta["slice"] = slice_meta

            # delete non-updated meta
            del patches.meta["crop_center"]
            # patch.meta["crop_center"] = crop_center_meta

            output_d[key] = patches

        # DEBUG
        # putting softmax saturates values
        #   -> looking like to have one high value and rest low
        #   -> hard to visualize
        # so dont do softmax for better vis

        # convert logit to probability for visualization later
        masked_logit_flatten = masked_logit.flatten(1)
        # masked_probability_flatten = nn.functional.softmax(masked_logit_flatten, dim=-1)
        masked_probability_flatten = masked_logit_flatten
        masked_probability: TensorType["B", "1", "Hz", "Wz", "Dz"] = (
            masked_probability_flatten.view(B, *CHWD)
        )

        return (output_d, masked_logit, slice_meta)


if __name__ == "__main__":

    # gumbel_top_k.tau = 100
    # torch.save(gumbel_top_k.state_dict(), "./test.ckpt")
    # model = GumbelTopK(tau=0.00001, hard=True)
    # model.load_state_dict(torch.load("./test.ckpt", weights_only=True))
    # print(model.tau_tensor)

    from utils.visualzation import VisVolLab, ShapeGenerator
    from model.inference import PatchInferer
    from matplotlib import pyplot as plt
    from utils.misc import get_patch_dim
    import time

    device = "cuda:6"
    batch_size = 1
    input_shape = [1, 128, 128, 128]
    patch_size = [64, 64, 64]
    overlap_r = 0.5
    num_paches_sampled_per_batch = 27
    patch_dim = get_patch_dim(
        image_size=input_shape[1:],
        patch_size=patch_size,
        overlap_r=overlap_r,
    )

    patch_inferer = PatchInferer()

    vis = VisVolLab()

    generator = ShapeGenerator(
        image_size=input_shape[1:],
        batch_size=batch_size,
        shape_type="circle",
    )

    vol = generator.generate().to(device)
    logit = torch.randn(batch_size, 1, *patch_dim).to(device)

    # logit[:, :, 3:, 3:, 3:] = -99
    # logit = torch.randint(0, 2, size=(batch_size, 1, *patch_dim)).to(device) * 100
    # logit = torch.ones(batch_size, 1, *patch_dim).to(device) * np.log(
    #     np.finfo(np.float32).tiny
    # )
    # logit[0, :, 0, 0, 0] = 10
    # logit[0, :, 0, 1, 0] = 10
    # logit[1, :, 0, 0, 0] = 10
    # logit[1, :, 0, 1, 0] = 10
    logit.requires_grad = True
    vol.requires_grad = True

    mask = torch.zeros(batch_size, 1, *patch_dim).to(device)
    # mask[0, :, 1:, :, :] = 1
    # mask = torch.ones(batch_size, 1, *patch_dim).to(device)

    sample_topk_patch = SampleTopKPatch(
        keys=[VOL, LAB],
        patch_size=patch_size,
        overlap_r=overlap_r,
    ).to(device)

    xs = []
    for i in range(1):
        start = time.perf_counter()
        vol.requires_grad = True
        sampled_patches_d, probability = sample_topk_patch(
            {VOL: vol, LAB: vol},
            logit,
            mask,
            num_paches_sampled_per_batch,
            hard=False,
        )
        end = time.perf_counter()
        xs.append(end - start)

    new_vol = patch_inferer(sampled_patches_d[VOL], torch.zeros_like(vol))
    new_lab = patch_inferer(sampled_patches_d[LAB], torch.zeros_like(vol))

    plt.imshow(vis.vis(vol=vol.detach().cpu()))
    plt.show()
    plt.imshow(vis.vis(vol=new_vol.detach().cpu()))
    plt.show()
    plt.imshow(vis.vis(vol=new_lab.detach().cpu()))
    plt.show()
    plt.imshow(vis.vis(vol=probability.detach().cpu()))
    plt.show()

# class SampleTopKPatch2(nn.Module):

#     def __init__(
#         self,
#         keys: list[str],
#         patch_size: list[int],
#         overlap_r: float | list[float],
#         tau: float = 0.01,
#     ):

#         super().__init__()

#         self.keys = keys
#         self.patch_size = patch_size
#         self.overlap_r = overlap_r

#         self.patchify = toBatch(Patchify)(
#             keys=self.keys,
#             patch_size=self.patch_size,
#             overlap_r=self.overlap_r,
#         )

#         self.get_top_k_sample = GumbelTopK(tau=tau)

#     @property
#     def tau(self):
#         return self.get_top_k_sample.tau

#     @tau.setter
#     def tau(self, value):
#         self.get_top_k_sample.tau = value

#     def forward(
#         self,
#         input_d: dict[str, torch.tensor],
#         logit: TensorType["B", "1", "Hz", "Wz", "Dz"],
#         mask: float | TensorType["B", "1", "Hz", "Wz", "Dz"] = torch.tensor(0),
#         num_paches_sampled_per_batch: int = 5,
#     ) -> TensorType["BK", "C", "Hp", "Wp", "Dp"]:

#         # prepare variables to be used
#         batch_size: int = input_d[self.keys[0]].shape[0]
#         input_channel_size: int = input_d[self.keys[0]].shape[1]
#         B, *CHWD = logit.shape

#         full_num_patches_per_batch: int = reduce(mul, CHWD[1:])
#         device = input_d[self.keys[0]].device

#         # get all the patches from the full-res input
#         local_patches_d: TensorType["BHzWzDz", "C", "Hp", "Wp", "Dp"] = self.patchify(
#             input_d
#         )

#         # prevent samplign from background
#         masked_logit: TensorType["B", "1", "Hz", "Wz", "Dz"] = logit + torch.log(
#             torch.max(
#                 1.0 - mask, torch.tensor([self.get_top_k_sample.epsilon], device=device)
#             )
#         )
#         # convert logit to probability for visualization later
#         masked_probability: TensorType["B", "1", "Hz", "Wz", "Dz"] = torch.exp(
#             masked_logit
#         ) / (1 + torch.exp(masked_logit))
#         # sample one-hots from logit
#         one_hots: TensorType["K", "B", "1", "Hz", "Wz", "Dz"] = self.get_top_k_sample(
#             masked_logit,
#             k=num_paches_sampled_per_batch,
#         )
#         one_hots_flatten: TensorType["K", "B", "HzWzDz"] = one_hots.flatten(2)
#         # samples top-k patches

#         output_d = {}
#         for k, local_patches in local_patches_d.items():
#             sampled_patches: list[TensorType["C", "Hp", "Wp", "Dp"]] = []
#             for b in range(batch_size):
#                 for i in range(num_paches_sampled_per_batch):

#                     patch_index = int(one_hots_flatten[i][b].argmax())
#                     # batch_splitted_local_patches: TensorType[
#                     #     "B", "HzWzDz", "C", "Hp", "Wp", "Dp"
#                     # ] = local_patches.view(
#                     #     batch_size,
#                     #     full_num_patches_per_batch,
#                     #     input_channel_size,
#                     #     *self.patch_size,
#                     # )

#                     batch_splitted_local_patches = local_patches.unsqueeze(0)

#                     patch: TensorType["C", "Hp", "Wp", "Dp"] = torch.tensordot(
#                         batch_splitted_local_patches[b],
#                         one_hots_flatten[i][b],
#                         dims=([0], [0]),
#                     )
#                     # update meta tensor after inner-product
#                     batch_patch_index = b * full_num_patches_per_batch + patch_index
#                     patch.meta["crop_center"] = local_patches[batch_patch_index].meta[
#                         "crop_center"
#                     ]
#                     patch.meta["slice"] = local_patches[batch_patch_index].meta["slice"]
#                     sampled_patches.append(patch)

#             sampled_patches: TensorType["BK", "C", "Hp", "Wp", "Dp"] = (
#                 list_data_collate(sampled_patches)
#             )

#             output_d[k] = sampled_patches

#         return (output_d, one_hots, masked_probability)


# torch.manual_seed(0)
# vol.requires_grad = True
# sampled_patches_d2, one_hots2, probability2 = sample_topk_patch2(
#     {VOL: vol, LAB: vol},
#     logit,
#     mask,
#     num_paches_sampled_per_batch,
# )

# sampled_patches2 = sampled_patches_d2[VOL]
# new_vol2 = patch_inferer(sampled_patches2, torch.zeros_like(vol))

# print((new_vol == new_vol2).all())
# print(
#     (sampled_patches_d[VOL].meta["slice"] == sampled_patches_d2[VOL].meta["slice"])
# )
# print(
#     (
#         torch.stack(sampled_patches_d[VOL].meta["crop_center"])
#         == torch.stack(sampled_patches_d2[VOL].meta["crop_center"])
#     ).all()
# )

# plt.imshow(vis.vis(vol=vol.detach().cpu()))
# plt.show()
# plt.imshow(vis.vis(vol=new_vol.detach().cpu()))
# plt.show()
# plt.imshow(vis.vis(vol=probability.detach().cpu()))
# plt.show()

# sample_topk_patch2 = SampleTopKPatch2(
#     keys=[VOL, LAB],
#     patch_size=patch_size,
#     overlap_r=overlap_r,
#     tau=0.02,
# ).to(device)
