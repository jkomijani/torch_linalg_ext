# Copyright (c) 2023 Javad Komijani

import torch

# from .._eig import eigh
from .._autograd import eigh


class AttributeDict:
    """For accessing a dict key like an attribute."""

    def __init__(self, **dict_):
        self.__dict__.update(**dict_)

    def __repr__(self):
        str_ = "svd:\n"
        for key, value in self.__dict__.items():
            str_ += f"{key}={value}\n"
        return str_


def svd(matrix, include_suvd=False):
    """Return singular value decomposition of the input complex, square matrix.

    The singular value decomposition of matrix :math:`M` is

    .. math::

         M = U S V^\dagger

    If :math:`S^{-1}` exists, then :math:`U V^\dagger` is unique, otherwise
    it is not.

    As torch.linalg.svd, we can only return (u, s, vh), but there is an option
    to return more quantities.
    To this end, set `include_suvd=True` to also have we also return

    .. math::

         (U @ V^\dagger) * phase_factor

    where the phase factor is constructed such that the matrix turns to SU(n).
    We call this matrix `sUVh`. We also return determinant of `(U @ V^\dagger)`.
    """
    # First obtain S^2 and U
    s_sq, u = eigh(matrix @ matrix.adjoint())

    # V can be obtained by multiplying S^{-1} U^\dagger and matrix
    s = torch.sqrt(s_sq)
    s[ s_sq < 0 ] = 0  # to remove possible roundoff error
    inv_s = 1 / s
    inv_s[ s == 0 ] = 0

    vh = (inv_s.unsqueeze(-1) * u.adjoint()) @ matrix

    # The method fails if S^{-1} diverges, which will be taken care separately.
    # cond = (torch.sum(s == 0, dim=-1) > 0).ravel()  # not precise (roundoff)
    cond = (torch.linalg.matrix_norm(vh) < 0.99 * vh.shape[-1]**0.5).ravel()
    if torch.sum(cond) > 0:
        n = matrix.shape[-1]
        vh.view(-1, n, n)[cond] = slow_svd(matrix.view(-1, n, n)[cond]).Vh

    if include_suvd:
        return append_suvd(AttributeDict(U=u, S=s, Vh=vh))
    else:
        return AttributeDict(U=u, S=s, Vh=vh)


def slow_svd(matrix, include_suvd=False):
    """Return singular value decomposition of the input complex, square matrix.

    The singular value decomposition of matrix :math:`M` is

    .. math::

         M = U S V^\dagger

    If :math:`S^{-1}` exists, then :math:`U V^\dagger` is unique, otherwise
    it is not.
    We explain it now. let us introduce unitary matrices :math:`D_u` and
    :math:`D_v` that satisfy :math:`[D_u, S] = [D_v, S] = 0`.
    Then, one can show that

    .. math::

         M = (U D_u) S (V D_v)^\dagger

    is another valid decomposition only if :math:`S D_u D_v^\dagger = S`.
    When :math:`S` is invertible, the condition inidcates :math:`D_u = D_v`.
    This then implies that

    .. math::

        U V^\dagger

    is unique. If some elements of :math:`S` are zero, the above constraint
    does not fully relate :math:`D_v` to :math:`D_u` and :math:`U V^\dagger` is
    not unique anymore. We use a particular presciption to handle this
    situation.
    """
    s_sq, u = eigh(matrix @ matrix.adjoint())
    _, naive_v = eigh(matrix.adjoint() @ matrix)  # v = naive_v @ D.adjoint()

    s = torch.sqrt(s_sq)
    s[ s_sq < 0 ] = 0  # to remove possible roundoff error
    inv_s = 1 / s
    inv_s[ s == 0 ] = 0

    # If all singular values are nonzero, the following expression yields `D`
    # such that `vh = D @ naive_v.adjoint()`.
    # Note that D is block diagonal, each block corresponds to a unique
    # singular value and the block is unitary itself.
    # We replace the block correspoding to vanishing sigular values to I; it is
    # numerically more precise to look at the vanishing diagonal terms in
    # s_times_d than s.

    naive_d = inv_s.unsqueeze(-1) * (u.adjoint() @ matrix @ naive_v)

    fixer = torch.zeros_like(s)
    # fixer[ s == 0] = 1  # not precise because of round off errors
    fixer[ torch.linalg.vector_norm(naive_d, dim=-1) < 0.01 ] = 1

    vh = (naive_d + torch.diag_embed(fixer)) @ naive_v.adjoint()

    if include_suvd:
        return append_suvd(AttributeDict(U=u, S=s, Vh=vh))
    else:
        return AttributeDict(U=u, S=s, Vh=vh)


def append_suvd(svd):
    """Return a new svd object that includes U V^\dagger"""
    uvh = svd.U @ svd.Vh
    rdet = torch.det(uvh)**(1 / uvh.shape[-1])  # root of determinant
    # we now make determinant of uvh unity:
    uvh = uvh / rdet.reshape(*rdet.shape, 1, 1)
    return AttributeDict(U=svd.U, S=svd.S, Vh=svd.Vh, rdet_uvh=rdet, sUVh=uvh)
