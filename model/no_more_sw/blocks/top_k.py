""""
refer to https://arxiv.org/pdf/1611.01144 for more detail.
"""

from torchtyping import TensorType

from toolz import *
from toolz.curried import *
from toolz.curried.operator import *

import numpy as np
import torch
from torch import nn


class GumbelTopK(nn.Module):

    # this is not small enough
    epsilon = np.finfo(np.float32).tiny
    epsilon_test = 1e-200
    tau: None

    def __init__(
        self,
        tau: float = 2 / 3,
    ):

        super().__init__()
        self.tau = tau
        self.register_buffer("epsilon_tensor", torch.tensor([self.epsilon]))
        self.register_buffer("epsilon_tensor_test", torch.tensor([self.epsilon_test]))

    def forward(
        self,
        logits: TensorType["B", "N"],
        k: int,
        mode="train",
    ) -> TensorType["K", "B", "N"]:

        cur_device = logits.device
        self.cur_device = cur_device

        # add gumbel noise
        flatten_logits = logits.flatten(1)
        # log_p = self.logSoftmax(flatten_logits)
        log_p = flatten_logits

        # sample soft topk
        soft_one_hots = []
        onehot_approx = torch.zeros_like(log_p, device=cur_device)
        noisy_logits = log_p + self.get_gumbel_noise(log_p.shape)

        for i in range(k):
            if i == 0:
                mask = torch.max(
                    1.0 - onehot_approx,
                    (
                        self.epsilon_tensor
                        if mode == "train"
                        else self.epsilon_tensor_test
                    ),
                )
            else:
                hard_one_hot = torch.zeros_like(onehot_approx, device=cur_device)
                _, ind = torch.topk(soft_one_hots[i - 1], 1, dim=1)
                hard_one_hard = hard_one_hot.scatter_(1, ind, 1)
                mask = torch.max(
                    1.0 - hard_one_hard,
                    (
                        self.epsilon_tensor
                        if mode == "train"
                        else self.epsilon_tensor_test
                    ),
                )

            # prev onehot has large negative value
            noisy_logits = noisy_logits + torch.log(mask)

            # during softmax, large negative value maps to a value close to 0.
            onehot_approx = torch.nn.functional.softmax(noisy_logits / self.tau, dim=1)
            soft_one_hots.append(onehot_approx)

        # DEBUG
        # sample soft topk
        # soft_one_hots = []
        # cumulative_mask = torch.zeros_like(logits, device=cur_device)
        # for i in range(k):
        #     # Apply cumulative mask
        #     masked_logits = noisy_logits + torch.log(
        #         torch.max(1.0 - cumulative_mask, self.epsilon_tensor)
        #     )

        #     # Compute softmax for current top-k
        #     onehot_approx = torch.nn.functional.softmax(
        #         masked_logits / (self.tau), dim=1
        #     )

        #     soft_one_hots.append(onehot_approx)

        #     # Update cumulative mask (hard selection)
        #     _, ind = onehot_approx.topk(1, dim=1)
        #     cumulative_mask.scatter_(1, ind, 1)
        #     # cumulative_mask += onehot_approx

        # Straight-through estimation
        st_one_hots = []
        for i in range(k):
            hard_one_hot = torch.zeros_like(soft_one_hots[i], device=cur_device)
            _, ind = torch.topk(soft_one_hots[i], 1, dim=1)
            hard_one_hot.scatter_(1, ind, 1)
            st_one_hots.append(
                (hard_one_hot - soft_one_hots[i]).detach() + soft_one_hots[i]
            )
        return (
            torch.stack(st_one_hots, dim=0),
            torch.stack(soft_one_hots, dim=0),
        )

    def get_gumbel_noise(self, shape):
        return -torch.log(
            -torch.log(torch.rand(shape, device=self.cur_device) + 1e-20) + 1e-20
        )


if __name__ == "__main__":

    from pprint import pprint
    from matplotlib import pyplot as plt

    TEST_TOP_K = True

    if TEST_TOP_K:

        sample_size = 1000
        tau = 2 / 3
        top_k = 3
        log_p = torch.log(
            torch.tensor(
                [[100, 50, 1]],
                requires_grad=True,
                dtype=torch.float32,
            )
        )
        gumbel_top_k = GumbelTopK(tau=tau)

        hard_sampless = []
        soft_sampless = []
        for _ in range(sample_size):
            hard_samples, soft_samples = gumbel_top_k(log_p, top_k)
            hard_sampless.append(hard_samples.argmax(-1))
            soft_sampless.append(soft_samples.argmax(-1))

        hard_sampless = torch.cat(hard_sampless, dim=1)
        soft_sampless = torch.cat(soft_sampless, dim=1)

        # print(all([len(set(np.array(tup))) == top_k for tup in sampless.permute(1, 0)]))

        for k, samples in enumerate(hard_sampless):
            plt.hist(
                samples,
                bins=3,
                color="blue",
                edgecolor="black",
            )
            plt.title(f"hard k:{k}")
            plt.show()

        # for k, samples in enumerate(soft_sampless):
        #     plt.hist(
        #         samples,
        #         bins=3,
        #         color="blue",
        #         edgecolor="black",
        #     )
        #     plt.title(f"soft k:{k}")
        #     plt.show()
