from typing import Mapping, Iterable
from toolz import *
from toolz.curried import *
from toolz.curried.operator import *
from functools import partialmethod
from itertools import product, starmap

div = lambda x, y: x / y


@curry
def list_op(
    binary_op: Mapping, xs: int | float | list[float], ys: int | float | list[float]
):

    if isinstance(xs, Iterable) and isinstance(ys, Iterable):
        assert len(xs) == len(ys)
        return list(starmap(binary_op, zip(xs, ys)))

    if type(xs) in [float, int] and isinstance(ys, Iterable):
        return list(map(partial(binary_op, xs))(ys))

    if type(ys) in [float, int] and isinstance(xs, Iterable):
        return list(map(partial(binary_op, ys))(xs))

    if type(ys) in [float, int] and type(ys) in [float, int]:
        return binary_op(xs, ys)


def uncurry(f):
    return lambda args: f(*args)


def partialclass(cls, *args, **kwargs):
    """
    wrapper to predefine args of a class
    """

    class PartialClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return PartialClass


def no_list_in_dict(d):

    # termination state
    if all([type(val) != dict for val in d.values()]):
        if all([type(val) != list for val in d.values()]):
            return True
        else:
            return False

    else:
        return all(
            [no_list_in_dict(val) if type(val) == dict else True for val in d.values()]
        )


def flatten_dict(args):

    if type(args) == dict and no_list_in_dict(args):
        return [args]
    if type(args) != list and type(args) != dict:
        return [args]
    if type(args) == list:
        return args

    else:
        keys = args.keys()
        vals = args.values()

        expanded_vals = product(*[flatten_dict(val) for val in vals])

        return [dict(zip(keys, vals)) for vals in expanded_vals]


if __name__ == "__main__":

    TEST_LIST_OP = True

    if TEST_LIST_OP:

        xs = [1, 2, 3]
        ys = [1, 2, 3]
        op = mul
        print(list_op(op)(xs, ys))

        xs = [1, 2, 3]
        ys = 2
        op = mul
        print(list_op(op)(xs, ys))

        xs = 2
        ys = [1, 2, 3]
        op = mul
        print(list_op(op)(xs, ys))

        xs = 1
        ys = 2
        op = mul
        print(list_op(op)(xs, ys))
