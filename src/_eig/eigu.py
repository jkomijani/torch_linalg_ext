# Copyright (c) 2023 Javad Komijani

import torch


def eigu(x):
    r"""Return eigenvalues and eigenvectors of unitary matrices.

    The implementation is with torch.linalg.eigh, which is for hermitian
    matrices. Using torch.linalg.eig instead of `eigu` seems to accumulate
    error with large number of layers.

    We use

    .. math::

        U \Omega = \Omega \Lambda
        U^\dagger \Omega = \Omega \Lambda^\dagger

    to write

    .. math::

       (U + U^\dagger) \Omega = \Omega (\Lambda + \Lambda^\dagger)
       (U - U^\dagger) \Omega = \Omega (\Lambda - \Lambda^\dagger)

    to obtain eigenvalues and eigencetors of unitary matrices.

    Warning: The algorithm used here can lead to wrong decomposition if there
    is a degeneracy in :math:`\sin(\theta_i)` while corresponding
    :math:`\cos(\theta_i) are not degenerate; e.g., this happend when
    :math:`\theta_0 = \pi - \theta_1`.
    However, it is unlikely to happend with random matrices.
    """
    eig_2sin, modal_matrix = torch.linalg.eigh(1J * (x.adjoint() - x))
    eig_2cos = torch.diagonal(
            modal_matrix.adjoint() @ (x.adjoint() + x) @ modal_matrix,
            dim1=-1, dim2=-2
            )
    eig = (eig_2cos + eig_2sin * 1J) / 2
    return eig, modal_matrix
