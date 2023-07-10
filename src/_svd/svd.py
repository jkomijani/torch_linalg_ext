# Copyright (c) 2023 Javad Komijani

import torch

from .._eig import eigh


class AttributeDict:
    """For accessing a dict key like an attribute."""

    def __init__(self, **dict_):
        self.__dict__.update(**dict_)

    def __repr__(self):
        return str(self.__dict__)


def svd(matrix):
    s_sq, u = eigh(matrix @ matrix.adjoint())
    s = torch.sqrt(s_sq)
    uvh = torch.matmul(u, s.unsqueeze(-1) * u.adjoint()) @ matrix
    vh = u.adjoint() @ uvh
    return AttributeDict(U=u, S=s, Vh=vh, UVh=uvh)


def svd_su2(x):
    """Special case where x is a sum of SU(2) matrices, for which one can show
    x.adjoint() @ x is proportional to the identity matrix.
    """
    s = (torch.abs(torch.linalg.det(x))**0.5).unsqueeze(-1)
    vh = x / s.unsqueeze(-1)
    u = torch.zeros_like(vh)
    u[..., 0, 0] = 1.
    u[..., 1, 1] = 1.
    return AttributeDict(U=u, S=s, UVh=vh, Vh=vh)
