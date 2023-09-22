# Copyright (c) 2023 Javad Komijani

import torch

from .generic import fix_phase, eyes_like, get_default_tolerance
from .eigh_3x3 import eigh3, cross_product
from .eigh_3x3 import nullspace3_from_parallelization
from .eigh_3x3 import nullspace3_from_cross_product

from .eig_3x3 import eign3x3


# =============================================================================
def eigvalssu3x3(matrix, return_invariants=False):
    r"""
    Return eigenvalues of 3x3 special unitary matrices using closed form
    expressions.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.
        We assume `matrix` is special unitary (over last two dimensions),
        for which the eigenvalues depend only on the trace of the matrix.
        (We do not check if the matrix is truely special unitary.)

    For special unitary matrices, we can exploit the fact that the deteriminant
    is unity and the absolute value of eigenvalues are 1 to simplify the
    characteristic equation to

    .. math::

        \lambda^3 - 3 \mu \lambda^2 + 3 \mu^* \lambda - 1 = 0,

    where :math:`\mu` is one third of the trace of the matrix. A change of
    variable as :math:`\lambda = x + \mu` turns the above equation to

    .. math::

        x^3 - 3 p x = 2 q

    where 

    .. math::

        p = \mu^2 - \mu^*
        q = \mu^3 + (1 - 3 |\mu|^2) / 2

    The damped cubic equation can be solved exactly. For instance, one can
    follow Appendix A in [http://arxiv.org/abs/physics/0610206],
    "Efficient numerical diagonalization of hermitian 3x3 matrices",
    to orgainze the solutions.

    Note that when :math:`|\mu| \approx 1`, i.e., when the SU(3) matrix is
    close to its center, we have :math:`|r_2| \approx |r_3|`, but with
    different phases. As a result the cancelation between these two terms is
    vulnerable to round-off error.

    Let us formulate this problem in a slightly different way: we could write
    the special unitary matrix as

    .. math::

        U = \mu I + \theta M

    where :math:`\text{Tr}\, M = 0` and :math:`\text{Tr}\, M M^\dagger = 2`,
    indicating

    .. math::

         \theta^2 = \frac{1}{2} \text{Tr}\ (A - \mu I) (A - \mu I)^\dagger 
         \theta^2 = \frac{3}{2} (1 - |\mu|^2)

    we can then use :math:`theta` as a measure that implies how close the SU(3)
    matrix is to its center, where :math:`|\mu| = 1`.
    """
    assert matrix.shape[-2:] == (3, 3), "matrix is supposed to be 3x3"

    mu = torch.mean(matrix.diagonal(dim1=-1, dim2=-2), dim=-1)

    p = mu*mu - mu.conj()
    q = p * mu + 0.5 * (1 - mu.real*mu.real - mu.imag*mu.imag)

    r_1 = mu
    r_2 = (q + torch.sqrt((q*q - p*p*p).real))**(1/3.)
    r_3 = p / r_2

    # Note that (q*q - p*p*p) is always real and typically positive

    cond = (torch.abs(p) < 1e-16)
    r_2[cond] = (2 * q[cond])**(1/3)
    r_3[cond] = 0
    cond = cond & (torch.abs(q) < 1e-16)
    r_2[cond] = 0

    w1 = (-1 + 1j * 3**0.5) / 2  # w = exp(i * 2 pi /3)
    w2 = (-1 - 1j * 3**0.5) / 2  # w = exp(i * 4 pi /3)

    eigvals = torch.stack(
               [r_1 + r_2 + r_3,
                r_1 + r_2 * w1 + r_3 * w2,
                r_1 + r_2 * w2 + r_3 * w1
               ],
               dim=-1
               )

    _, sorted_ind = torch.sort(eigvals.imag, dim=-1)
    eigvals = eigvals.gather(-1, sorted_ind)

    if return_invariants:
        return eigvals, (mu,)
    else:
        return eigvals


# =============================================================================
def eigsu3x3(matrix, **kwargs):
    """
    Althought this function is special for SU(3) matrices, it is better to
    avoid it because if the matrix is slightly different from a SU(3) matrix,
    the eigenvalues become slightly wrong too, which then affects the precision
    of eigenvectors and the fact that they must be perpendicular for
    non-degenerate eigenvalues.
    """
    return eign3x3(matrix, func_4_eigvals = eigvalssu3x3, **kwargs)


# =============================================================================
eigvalssu3 = eigvalssu3x3


def eigsu3(matrix, method='parallelization', eps_mu=1e-6, tol=None):
    """
    Return eigenvalues and eigenvectors of 3x3 special unitary matrices.

    The eigenvalues are obtained by calling `eigvalssu3`, which uses a closed
    form expression.

    Althought this function is special for SU(3) matrices, it is better to
    avoid it because if the matrix is slightly different from a SU(3) matrix,
    the eigenvalues become slightly wrong too, which then affects the precision
    of eigenvectors and the fact that they must be perpendicular for
    non-degenerate eigenvalues.

    To calculate the eigenvectors, one can exploit vector cross products in 3
    dimensions as described in [http://arxiv.org/abs/physics/0610206]
    "Efficient numerical diagonalization of hermitian 3x3 matrices".
    (With a small modification the method works for any matrices.)
    This method is pretty fast, but the error can be relatively large when the
    cross product is close to zero.
    We developed an alternative version that is about 30% slower but in general
    more accurate. For the former one, set 'method=cross-product' and for the
    latter one use 'method=parallelization'. The default is the latter one.
    Note that both cases are about two times faster than `torch.linalg.eigh`.
    (One can work with a combination of both methods to get something in between
    both methods with respect to time and accuracy. One can also decrease
    eps_theta for the former case to obtain more accurate results, but it
    increased the time anyway.)

    Here are a few remaks about both methods:

    1. They are senstive to round-off errors when the condition number is
    large or when the eigenvalues are very close to each other. As a remedy, we
    consider following cases:

    2a. When there are (almost) two degenerate eigenvectors, one eigenvector
    can be constructed by cross product of the other two as suggested in the
    above paper. Therefore, with sorted eigenvalues, we always construct the
    middle eigenvector by the cross product of the first and last eigenvectors.

    2b. When there are (almost) three degenerate eigenvectors, the matrix is
    close to a diagonal matrix, one should obtain the eigenvectos with a
    different algorithm to get a more precise result; we use eigu_from_h, which
    can give very precise result for such matrices.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.
        We assume `matrix` is hermition (over last two dimensions).

    method : string
        can be either 'cross-product' or 'parallelization'; the latter one is
        the default case.

    eps_mu : float, optional
        specifies when to switch from the main method to the alternative one.
        (Default is 1e-6)
    """
    # For benchmarking see
    # test_and_studies/eig_3x3/eig_decomposition_3x3_benchmarking.html 

    eigvals, (mu,) = eigvalssu3(matrix, return_invariants=True)

    # We assume eigvals are sorted. There are three cases:
    #     1. all eigenvalues are different
    #     2. the first & second or the second & third eigenvalues are equal
    #     3. all three eigenvalues are (almost) degenerate
    # For case (3), matrix is almost proportional to unity matrix and theta
    # is very small. We take care of this case using `eps_theta`.
    # Putting case (3) aside we can assume that eigvals[1] is not equal to
    # both eigvals[0] and eigvals[2] at the same time. We therefore obtain the
    # eigvectors corresponding to eigvals[0] and eigvals[2] first and then
    # construct eigvals[1] as cross poduct of them.

    if method == 'parallelization':
        nullspace = nullspace3_from_parallelization
    elif method == 'cross-product':
        nullspace = nullspace3_from_cross_product

    if tol is None:
        tol = get_default_tolerance()

    eigvecs = torch.zeros_like(matrix)
    eye = eyes_like(matrix)

    for k in [0, 2]:
        indices = [k % 3, (k + 1) % 3, (k + 2) % 3]
        eigval = eigvals[..., k:k+1].unsqueeze(-1)
        eigvecs[..., k] = nullspace(
                matrix - eigval * eye,
                indices=indices,
                scale=(1 - mu).abs().ravel(),
                tol=tol
                )
    eigvecs[..., 1] = cross_product(eigvecs[..., 2], eigvecs[..., 0]).conj()

    # We now search for case (3) as explained above.
    case3 = torch.abs(mu).ravel() >= (1 - eps_mu)
   
    if torch.sum(case3) > 0:
        eigvals_view = eigvals.reshape(-1, 3)
        eigvecs_view = eigvecs.reshape(-1, 3, 3)
        u, v = eigu3_from_h(matrix.reshape(-1, 3, 3)[case3])
        eigvals_view[case3] = u
        eigvecs_view[case3] = v

    eigvecs = fix_phase(eigvecs)

    return eigvals, eigvecs


# =============================================================================
def eigu3_from_h(x):
    r"""Return eigenvalues and eigenvectors of unitary matrices via converting
    them to Hermitian matrices.

    The implementation, employing hemitian matrices, does not accumulate errors
    unlike the regular one.

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
    Moreover, if this function is called only for SU(3) matrices that are close
    to their centers, this situation will not happen because if
    :math:`\sin(\theta_i)` are degenerate, corresponding :math:`\cos(\theta_i)`
    will be degenerate too. More specifically, as long as :math:`|\mu| >
    \sqrt{5} / 3` the trouble-making situation does not happen; at the boundary
    we have :math:`(\theta_0, \theta_1, \theta_2) = (\pi/2, \pi/2, \pi)`.
    """
    eig_2sin, modal_matrix = eigh3(1J * (x.adjoint() - x))
    eig_2cos = torch.diagonal(
            modal_matrix.adjoint() @ (x.adjoint() + x) @ modal_matrix,
            dim1=-1, dim2=-2
            )
    eig = (eig_2cos + eig_2sin * 1J) / 2
    return eig, modal_matrix
