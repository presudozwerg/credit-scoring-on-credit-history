import numpy as np
from torch import tensor, stack, float32

def collate_fn_tr(input_batch):
    tensors = []
    flags = []
    for elem in input_batch:
        tensors.append(elem[0])
        flags.append(elem[1])
    tensors = stack(tensors)
    flags = np.array(flags)
    flags = tensor(flags).to(float32)
    return tensors, flags

def collate_fn_te(input_batch):
    tensors = []
    for elem in input_batch:
        tensors.append(elem)
    tensors = stack(tensors)
    return tensors