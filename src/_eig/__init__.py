import torch

from .eig_2x2 import eigh2, eig_su2
from .eig_3x3 import eigh3, eig_su3
from .eig_3x3 import eigvalsh3, eigvals_su3
from .eigu import eigu


def eigvalsh(matrix):
    if matrix.shape[-1] == 2:
        func = eigvalsh2
    elif matrix.shape[-1] == 3:
        func = eigvalsh3
    else:
        func = torch.linalg.eigvalsh
    return fucn(matrix)


def eigvals_su(matrix):
    if matrix.shape[-1] == 2:
        func = eigvals_su2
    elif matrix.shape[-1] == 3:
        func = eigvals_su3
    else:
        raise Exception("Not implemented")
    return fucn(matrix)


def eig_su(matrix):
    if matrix.shape[-1] == 2:
        func = eig_su2
    elif matrix.shape[-1] == 3:
        func = eigh_su3
    else:
        func = eigu
    return fucn(matrix)


def eigh(matrix):
    if matrix.shape[-1] == 2:
        func = eigh2
    elif matrix.shape[-1] == 3:
        func = eigh3
    else:
        func = torch.linalg.eigh
    return fucn(matrix)
