# Copyright (c) 2023 Javad Komijani

"""
In this module we implement our own custom autograd Functions by subclassing
torch.autograd.Function and implementing the forward and backward passes
which operate on Tensors.

For example, one can see "An extended collection of matrix derivative results
for forward and reverse mode algorithmic differentiation."
"""

import torch

from .eig_2x2 import eigh2, eigsu2
from .eigh_3x3 import eigvalsh3, eigh3
from .eigsu_3x3 import eigvalssu3, eigsu3


EPS_EIGH = 1e-16


class Eig(torch.autograd.Function):
    r"""
    For more details see
    "An extended collection of matrix derivative results for forward and
    reverse mode algorithmic differentiation."
    We follow the notation introduced in the above reference as
    "We consider a computation which begins with a single scalar input variable
    :math:`s_i` and eventually, through a sequence of calculations, computes a
    single scalar output :math:`s_o` . Using standard automatic differentiation
    (AD) terminology, if :math:`A` is a matrix which is an intermediate
    variable within the computation, then
    1. :math:`\dot A` denotes the derivative of A with respect to s_i
    2. :math:`\bar A` denotes the derivative of s_o w.r.t each element of A.

    https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
    """

    @staticmethod
    def forward(ctx, matrix):
        u, v = eig(matrix)
        ctx.save_for_backward(u, v)
        return u, v

    @staticmethod
    def backward(ctx, grad_u, grad_v):
        # grad_? denotes derivative of a scalar output wrt each element of '?'
        u, v = ctx.saved_tensors
        diff = calc_eig_diff(u)
        inv = diff / (diff*diff + EPS_EIGH**2)  # (pseudo) inverse of diff
        grad_u = torch.diag_embed(grad_u)  # for matrix operations
        grad_matrix = v @ (grad_u + inv * (v.adjoint() @ grad_v)) @ v.adjoint()
        return grad_matrix


class Eigh2(Eig):

    @staticmethod
    def forward(ctx, matrix):
        u, v = eigh2(matrix)
        ctx.save_for_backward(u, v)
        return u, v


class Eigh3(Eig):

    @staticmethod
    def forward(ctx, matrix):
        u, v = eigh3(matrix)
        ctx.save_for_backward(u, v)
        return u, v


class Eigsu2(Eig):

    @staticmethod
    def forward(ctx, matrix):
        u, v = eigsu2(matrix)
        ctx.save_for_backward(u, v)
        return u, v


class Eigsu3(Eig):

    @staticmethod
    def forward(ctx, matrix):
        u, v = eigsu3(matrix)
        ctx.save_for_backward(u, v)
        return u, v


class Eigvals(torch.autograd.Function):
    """A thinner version of Eig."""

    @staticmethod
    def forward(ctx, matrix):
        u, v = eig(matrix)
        ctx.save_for_backward(u, v)
        return u, v

    @staticmethod
    def backward(ctx, grad_u):
        u, v = ctx.saved_tensors
        grad_matrix = v @ (grad_u.unsqueeze(-1) * v.adjoint())
        return grad_matrix


class Eigvalsh3(Eigvals):

    @staticmethod
    def forward(ctx, matrix):
        u, v = eigh3(matrix)
        ctx.save_for_backward(u, v)
        return u, v


class Eigvalssu3(Eigvals):

    @staticmethod
    def forward(ctx, matrix):
        u, v = eigsu3(matrix)
        ctx.save_for_backward(u, v)
        return u, v


class ReverseEig(torch.autograd.Function):
    """Reverse of eigenvalue decoposition."""

    @staticmethod
    def forward(ctx, u, v):
        matrix = v @ (u.unsqueeze(-1) * v.adjoint())
        ctx.save_for_backward(u, v)  # no need to save matrix
        return matrix

    @staticmethod
    def backward(ctx, grad_matrix):
        # grad_? denotes derivative of a scalar output wrt each element of '?'
        u, v = ctx.saved_tensors
        grad_matrix = v.adjoint() @ grad_matrix @ v  # new definiton
        grad_u = torch.diagonal(grad_matrix, dim1=-1, dim2=-2)
        diff = calc_eig_diff(u)
        grad_v = v @ (diff * grad_matrix)
        return grad_u, grad_v


def calc_eig_diff(u):
    """u is the list of eigenvalues"""
    n = u.shape[-1]
    diff = u.view(-1, 1, n).repeat(1, n, 1) - u.view(-1, n, 1).repeat(1, 1, n)
    return diff.view(*u.shape[:-1], n, n)
