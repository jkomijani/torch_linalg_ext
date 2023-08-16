import torch
import warnings

from .eig_2x2 import eigh2, eigsu2
from .eigh_3x3 import eigvalsh3, eigh3
from .eigsu_3x3 import eigvalssu3, eigsu3

from . import generic


def eigsu(matrix, **kwargs):
    if matrix.shape[-1] == 2:
        func = eigsu2
    elif matrix.shape[-1] == 3:
        func = eigsu3
    else:
        warnings.warn("Using torch.linalg.eigh")
        func = torch.linalg.eig
    return func(matrix, **kwargs)


def eigh(matrix, **kwargs):
    if matrix.shape[-1] == 2:
        func = eigh2
    elif matrix.shape[-1] == 3:
        func = eigh3
    else:
        warnings.warn("Using torch.linalg.eigh")
        func = torch.linalg.eigh
    return func(matrix, **kwargs)
