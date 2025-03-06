from torchtyping import TensorType

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import torch
from torch import nn
from monai.transforms import Compose, SelectItemsD

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

    device = "cuda:0"
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

    start = time.perf_counter()
    vol.requires_grad = True
    sampled_patches_d, probability, slice_meta = sample_topk_patch(
        {VOL: vol, LAB: vol},
        logit,
        mask,
        num_paches_sampled_per_batch,
    )

    print(next(sampled_patches_d[VOL]).shape)
    print(next(sampled_patches_d[LAB]).shape)
