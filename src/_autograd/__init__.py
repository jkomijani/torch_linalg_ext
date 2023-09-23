
import torch

from . import eig_autograd

eigh2x2 = eig_autograd.Eigh2x2.apply
eigh3x3 = eig_autograd.Eigh3x3.apply

eigu2x2 = eig_autograd.Eigu2x2.apply
eigu3x3 = eig_autograd.Eigu3x3.apply


def eigh(matrix, **kwargs):
    match matrix.shape[-1]:
        case 2:
            return eigh2x2(matrix, **kwargs)
        case 3:
            return eigh3x3(matrix, **kwargs)
        case _:
            return torch.linalg.eigh(matrix, **kwargs)


def eigu(matrix, **kwargs):
    match matrix.shape[-1]:
        case 2:
            return eigu2x2(matrix, **kwargs)
        case 3:
            return eigu3x3(matrix, **kwargs)
        case _:
            return torch.linalg.eig(matrix, **kwargs)
