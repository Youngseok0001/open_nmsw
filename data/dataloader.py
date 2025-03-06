import torch
from monai.data import DataLoader, ThreadDataLoader
from monai.data.utils import list_data_collate
from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from utils.definitions import *


def custum_collate_fn(data_list):
    """
    To reduce memory.
    assume batch size of 1
    """
    collated_data = list_data_collate(data_list)
    return {k: (v[:1] if PATCH not in k else v) for k, v in collated_data.items()}


class NSWDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=8,
        iteration_per_epoch: int | None = None,
        collate_fn=custum_collate_fn,
        pin_memory=True,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor=None,
        persistent_workers=False,
        **kwargs
    ):

        if iteration_per_epoch is not None:
            sampler = torch.utils.data.RandomSampler(
                dataset, replacement=True, num_samples=iteration_per_epoch
            )
        else:
            sampler = None

        # Initialize the parent class with all arguments
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            **kwargs
        )


# class NSWDataLoader(DataLoader):
#     def __init__(
#         self,
#         dataset,
#         num_workers: int = 8,
#         iteration_per_epoch: int | None = None,
#         batch_size: int = 1,
#     ):

#         if iteration_per_epoch is not None:
#             sampler = torch.utils.data.RandomSampler(
#                 dataset, replacement=True, num_samples=iteration_per_epoch
#             )
#         else:
#             sampler = None
#         super().__init__(
#             dataset,
#             batch_size=1,  # only batch_size 1 supported currently
#             num_workers=num_workers,
#             collate_fn=custum_collate_fn,
#             shuffle=iteration_per_epoch is None,
#             sampler=sampler,
#             persistent_workers=False,
#             # pin_memory=True,
#         )


if __name__ == "__main__":

    from tqdm import tqdm
    from data.registry import data_registry
    from matplotlib import pyplot as plt
    from utils.visualzation import VisVolLab

    TEST_GET_GLOABL_SIZE = True

    if TEST_GET_GLOABL_SIZE:

        dataset_name = "Amos"
        rand_aug_type = "none"
        num_workers = 16
        mode = TRAIN

        dataset = data_registry[dataset_name](
            mode=mode,
            fold_n=0,
            rand_aug_type=rand_aug_type,
        )

        visualizer = VisVolLab(num_classes=dataset.num_classes)

        loader = NSWDataLoader(
            dataset,
            num_workers=num_workers,
            iteration_per_epoch=None,
        )

        # shapes = []
        for batch in tqdm(loader):

            plt.imshow(
                visualizer.vis(
                    batch[VOL][0],
                    batch[LAB][0],
                )
            )
            plt.show()
            print(batch[VOL][0].shape)

            # batch[VOL].shape
            # print(batch[VOL].shape)
            # print(batch[PATCH_VOL].shape)
            # vol = batch[VOL].squeeze()
            # shapes.append(vol.shape)

        # print(torch.tensor(shapes).max(0))
