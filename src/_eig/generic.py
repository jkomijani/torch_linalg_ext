# Copyright (c) 2023 Javad Komijani

import torch


def sort_eig(eigvals, eigvecs):
    """sort eigenvalues in ascending order."""
    eigvals, sorted_ind = torch.sort(eigvals, dim=-1)
    n = eigvecs.shape[-1]
    sorted_ind = sorted_ind.unsqueeze(-2).repeat(*[1]*(eigvecs.ndim - 2), n, 1)
    eigvecs = eigvecs.gather(-1, sorted_ind)
    return eigvals, eigvecs


def fix_phase(eigvecs, max_row=True, firs_row=False):
    """Fix the arbitrary phase of each eigenvector.

    When `max_row` is True, for each eigenvector, the element with largest
    absolute value is changed to becomes real.

    For a bahavior similar to pytorch set `max_row = False, first_row = True`.
    """
    if max_row:
        ind_max = torch.max(torch.abs(eigvecs), dim=-2)[1]
        angle = torch.angle(eigvecs.gather(-2, ind_max.unsqueeze(-2)))
    elif first_row:
        angle = torch.angle(eigvecs[..., 0, :]).unsqueeze(-2)
    phasor = torch.exp(-1j * angle)
    n = eigvecs.shape[-1]
    return eigvecs * phasor.repeat(*[1]*(eigvecs.ndim - 2), n, 1)
