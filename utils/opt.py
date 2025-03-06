from toolz import *
from toolz.curried import *
from torch.optim import SGD, Adam, AdamW


def set_optimizer(model, opt_kwargs):

    theta = model.parameters()
    
    opt_name = opt_kwargs["name"]
    opt_kwargs.pop("name")
    print(f"**{opt_kwargs}**")
    return {
        "SGD": lambda: SGD(theta, **opt_kwargs),
        "Adam": lambda: Adam(theta, **opt_kwargs),
        "AdamW_1": lambda: AdamW(theta, **opt_kwargs),
        "AdamW_2": lambda: AdamW(theta, **opt_kwargs),
    }[opt_name]()
