import torch
import warnings

from . import _autograd
from . import generic

eigh2 = _autograd.Eigh2.apply
eigh3 = _autograd.Eigh3.apply
eigsu2 = _autograd.Eigsu2.apply
eigsu3 = _autograd.Eigsu3.apply
eigvalsh3 = _autograd.Eigvalsh3.apply
eigvalssu3 = _autograd.Eigvalssu3.apply

reverse_eig = _autograd.ReverseEig.apply

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
