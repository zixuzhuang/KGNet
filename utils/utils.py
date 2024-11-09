import os

import numpy as np
import torch as tc


def one_hot(seg, num_cls=4):
    if type(seg) is tc.Tensor:
        seg = tc.eye(num_cls)[seg]
        
    else:
        seg = np.eye(num_cls)[seg]
    return seg


def mkdirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return
