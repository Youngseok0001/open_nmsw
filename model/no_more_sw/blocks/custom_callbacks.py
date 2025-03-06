from lightning.pytorch.callbacks import Callback
import torch
import numpy as np


class ReducingTau(Callback):
    def __init__(
        self,
        starting_tau=None,
        final_tau=0.1,
        reduction_mutiplier=0.5,
    ):
        """Reduce tau in top_k

        Gradually reduce tau to final_tau exponentially using approx. half of total training epochs, and maintain tau to final_tau for the rest of the training epochs.

        This callback will reduce tau if pl_module.net.sample_topk_patch.get_top_k_sample.tau exists

        Args:
            final_tau (_type_, optional): _description_. Defaults to torch.tensor(0.1).
        """
        self.starting_epoch = 0
        self.total_epoch = None
        self.final_tau = final_tau
        self.starting_tau = starting_tau
        self.state = {"final_tau": final_tau}
        self.k = reduction_mutiplier

    def on_train_start(self, trainer, pl_module):
        self.starting_epoch = trainer.current_epoch
        self.total_epoch = trainer.max_epochs

        try:
            if self.starting_tau is not None:
                pl_module.net.tau = self.starting_tau
            else:
                self.starting_tau = pl_module.net.tau

            self.r = (
                -1
                / (max(1, self.total_epoch * self.k - self.starting_epoch))
                * np.log(self.final_tau / self.starting_tau)
            )
        except AttributeError:
            pass

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch

        if current_epoch == 0:
            pass
        else:
            try:
                current_tau = pl_module.net.tau

                if current_epoch > self.total_epoch * self.k:
                    updated_tau = current_tau
                else:
                    updated_tau = current_tau * np.exp(-self.r)

                pl_module.net.tau = updated_tau
            except AttributeError:
                pass

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()


if __name__ == "__main__":

    import numpy as np
    from matplotlib import pyplot as plt

    total_epoch = 1000
    starting_epoch = 0.0
    starting_tau = 2
    final_tau = 1 / 3

    k = 1

    r = (
        -1
        / (max(1.0, total_epoch * k - starting_epoch))
        * np.log(final_tau / starting_tau)
    )

    taus = []
    updated_tau = starting_tau
    for i in range(total_epoch):
        if i > total_epoch * k:
            updated_tau = updated_tau
        else:
            updated_tau = updated_tau * np.exp(-r)
        taus.append(updated_tau)

    plt.plot(taus)
