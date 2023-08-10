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
    """Return singular value decomposition of the input matrix.

    As torch.linalg.svd, we can only return (u, s, vh), but we also return

    .. math::

         (U @ V^\dagger) * phase

    where phase is constructed such that the matrix turns to SU(n). We call
    this matrix `sUVh`. We also return determinant of `(U @ V^\dagger)`.
    """
    # First obtain S^2 and U
    s_sq, u = eigh(matrix @ matrix.adjoint())

    # V can be obtained by multiplying S^{-1} U^\dagger and matrix
    # The method fails if S^{-1} diverges

    s = torch.sqrt(s_sq)

    vh = ((1 / s).unsqueeze(-1) * u.adjoint()) @ matrix

    # To do, if s==0, one can perform qr decomposition on matrix.adjoint() @ u

    uvh = u @ vh
    det_uvh = torch.det(uvh)
    uvh[..., 0] = uvh[..., 0] * det_uvh.unsqueeze(-1)  # change only 1st column
    return AttributeDict(U=u, S=s, Vh=vh, det_uvh=det_uvh, sUVh=uvh)


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
