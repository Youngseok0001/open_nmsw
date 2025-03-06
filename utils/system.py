import os
from subprocess import check_output, getoutput
from time import localtime, strftime, time

from toolz import *
from toolz.curried import *

import numpy as np
import torch


def read_branch_and_hash():
    branch_name = getoutput("git rev-parse --abbrev-ref HEAD")
    commit_hash = getoutput("git rev-parse --short HEAD")
    return (branch_name, commit_hash)


def set_gpu(gpu_id=1):

    if gpu_id == -1:

        # pick gpus that has the largest space

        function = "nvidia-smi"
        param1 = "--query-gpu=memory.free"
        param2 = "--format=csv,nounits,noheader"

        command = [function, param1, param2]
        memories = check_output(command, encoding="utf-8").split("\n")[:-1]
        memories = list(map(int)(memories))

        numbers = []
        for number, memory in take(1)(  # fix later
            sorted(enumerate(memories), key=second, reverse=True)
        ):
            # print(f"CUDA:{number} with {memory}Gb of memory is selected.\n")
            numbers.append(number)

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str)(numbers))

    else:
        # DEBUG
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"


def set_seed(seed):
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_now():
    return strftime("%Y-%m-%d_%H-%M-%S", localtime())


def timeit(f):
    def inner(*args, **kwargs):
        t1 = time()
        result = f(*args, **kwargs)
        print(f"{f.__name__} took {round(time() - t1)} seconds.")
        return result

    return inner


def load_weights(model, ckptPath, module_name_to_delete):
    print(f"loading {ckptPath} ...")
    modelDict = torch.load(ckptPath)["state_dict"]
    modelDict = keymap(lambda x: x.replace(f"{module_name_to_delete}.", ""))(modelDict)
    model.load_state_dict(modelDict)
    print("done")


if __name__ == "__main__":

    TEST_LOAD_CKPT = True

    if TEST_LOAD_CKPT:

        from model.registry import model_registry
        import copy

        ckpt_path = "/hpc/home/jeon74/no-more-sw/outputs/2024-10-10/11-43-47/ckpts/Word_LocalSeg3D.ckpt"

        input_shape = [1, 384, 384, 384]
        output_shape = [17, 384, 384, 384]
        patch_size = [128, 128, 128]
        down_size_rate = [2, 2, 2]
        overlap_r = 0.25
        patch_weight = 0.95
        add_gaussian_wt = True
        add_learnable_wt = True
        batch_size = 1
        num_train_pos_patches = 5
        num_train_neg_patches = 5
        num_infernce_patches = 5

        net = model_registry["NSWNet3D"](
            input_shape=input_shape,
            output_shape=output_shape,
            patch_size=patch_size,
            down_size_rate=down_size_rate,
            local_backbone_name="FasterUNet",
            global_backbone_name="FasterUNet",
            overlap_r=overlap_r,
            num_train_pos_patches=num_train_pos_patches,
            num_train_neg_patches=num_train_neg_patches,
            num_infernce_patches=num_infernce_patches,
            patch_weight=patch_weight,
            add_gaussian_wt=add_gaussian_wt,
            add_learnable_wt=add_learnable_wt,
        )

        net_prev = copy.deepcopy(net.local_backbone)
        load_weights(net.local_backbone, ckpt_path, "backbone")
        net_cur = copy.deepcopy(net.local_backbone)
