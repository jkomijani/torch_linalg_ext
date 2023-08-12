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
    """Return singular value decomposition of the input complex matrix.

    As torch.linalg.svd, we can only return (u, s, vh), but we also return

    .. math::

         (U @ V^\dagger) * phase_factor

    where the phase factor is constructed such that the matrix turns to SU(n).
    We call this matrix `sUVh`. We also return determinant of `(U @ V^\dagger)`.
    """
    # First obtain S^2 and U
    s_sq, u = eigh(matrix @ matrix.adjoint())

    # V can be obtained by multiplying S^{-1} U^\dagger and matrix
    s = torch.sqrt(s_sq)
    vh = ((1 / s).unsqueeze(-1) * u.adjoint()) @ matrix

    # The method fails if S^{-1} diverges, which will be taken care separately.
    cond = torch.sum(s_sq <= 0, dim=-1).ravel()
    if torch.sum(cond) > 0:
        n1, n2 = matrix.shape[-2:]
        vh.view(-1, n2, n2)[cond] = singular_svd(
                matrix.view(-1, n1, n2)[cond],
                u.view(-1, n1, n1)[cond],
                )

    uvh = u @ vh
    det_uvh = torch.det(uvh)
    uvh[..., 0] = uvh[..., 0] * det_uvh.unsqueeze(-1)  # change only 1st column
    return AttributeDict(U=u, S=s, Vh=vh, det_uvh=det_uvh, sUVh=uvh)


def singular_svd(args):
    pass  # NOT READY YET
