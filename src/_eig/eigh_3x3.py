# Copyright (c) 2023 Javad Komijani

import torch
import numpy as np


from .generic import fix_phase, eyes_like, get_default_tolerance
from .eigh_jacobi import jacobi_diagonalization
from .eig_3x3 import eign3x3


# =============================================================================
def eigvalsh3x3(matrix, return_invariants=False):
    r"""
    Return eigenvalues of 3x3 hermitian matrices using closed form expressions.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.
        We assume `matrix` is hermition (over last two dimensions),
        otherwise we treat it similar to pytorch by discarding the
        upper diagonal terms.

    return_invariants : boolean
        in addition to the eigenvalues, return parameters mu, theta and phi;
        see the discription of algorithm below. (Default is False.)
    
    We now give the algorithm for eigenvalue determination of 3x3 Hermitian
    matrices.

    Without loss of generality we write matrix :math:`A` as

    .. math::

        A = \mu I + \theta M

    where :math:`\text{Tr} M = 0` and :math:`\text{Tr} M^2 = 2`,
    indicating :math:`\theta^2 = \frac{1}{2} \text{Tr} (A - \mu I)^2`.
    Exploiting Cayley-Hamilton theorem for 3x3 matrices, matrix :math:`M`
    satisfies

    .. math::

        M^3 = M + \text{det}(M) I

    and therefore :math:`\frac{1}{3} \text{Tr} M^3 = \text{det} M`.
    Likewise, the eigenvalues of :math:`M` satisfy:

    .. math::

        \lambda^3 = \lambda + \text{det}(M).

    The three solutions to this equation can be expressed in terms of
    trigonometric functions with real arguments for Hermitian matrices.
    To this end we first perform a change of variable as

    .. math::

         \text{det}(M) = -\frac{2}{\sqrt{27}} \sin 3 \phi

    Then, exploiting the triple-angle trigonometric identity

    .. math::

         \sin 3\phi = 3\sin \phi - 4 \sin^3\phi

    the three eigenvalues of matrix :math:`M` read

    .. math::

        \lambda_k = \frac{2}{\sqrt{3}} \sin \left(\phi + 2\pi k /3\right).

    Note that the sum of the eigenvalues is zero.
    """

    assert matrix.shape[-2:] == (3, 3), "matrix is supposed to be 3x3"

    # We use following parametrization of the hermitian matrices
    # H = [[a,       x - I y,  s - I t]
    #      [x + I y,    b,     p - I q]
    #      [s + I t, p + I q,    c    ]]
    
    a = torch.real(matrix[..., 0, 0])
    b = torch.real(matrix[..., 1, 1])
    c = torch.real(matrix[..., 2, 2])
    x = torch.real(matrix[..., 1, 0])
    y = torch.imag(matrix[..., 1, 0])
    s = torch.real(matrix[..., 2, 0])
    t = torch.imag(matrix[..., 2, 0])
    p = torch.real(matrix[..., 2, 1])
    q = torch.imag(matrix[..., 2, 1])

    # We now define H = \mu I + \theta M, where M is a traceless
    # Hermitian matrix, in which c = - a - b, thus below we won't have c.
    # Moreover, we choose \theta such that Tr M^2 = 2;
    # then \theta^2 = 1/2 Tr(H - \mu I)^2

    mu = (a + b + c) * (1/3.)
    a = a - mu
    b = b - mu
    # c = c - mu  # no need to calculate c, which is set to -(a + b)
    
    theta = torch.sqrt(a*a + b*b + a*b + x*x + y*y + s*s + t*t + p*p + q*q)

    # minus_det is minus determinant of (H - \mu I)
    minus_det = a * (p*p + q*q) + b * (s*s + t*t) \
       + (a + b) * (a*b - x*x - y*y) - 2*(p*s*x + q*t*x - q*s*y + p*t*y)

    argument = minus_det / theta**3 * (27**0.5 / 2)
    phi = torch.asin(argument) / 3  # phi \in [-pi/6, pi/6]

    # Let us now fix phi for case theta = 0 or for numerical error the
    # absolute value of the argument inside arcsin is larger than 1.
    # case 1: theta = 0: phi is irrelevant and we set it to zero
    phi[theta == 0] = 0
    phi[argument > 1] = np.pi/6
    phi[argument < -1] = -np.pi/6

    eigvals = mu.unsqueeze(-1) \
            + (2/3**0.5) * theta.unsqueeze(-1) \
              * torch.sin(
                   torch.stack([phi - 2*np.pi/3, phi, phi + 2*np.pi/3], dim=-1)
                )  # does not have contribution from mu

    if return_invariants:
        return eigvals, (mu, theta, phi)
    else:
        return eigvals


# =============================================================================
def eigh3x3(matrix, **kwargs):
    u, v = eign3x3(matrix, func_4_eigvals = eigvalsh3x3, **kwargs)
    return u.real, v


# =============================================================================
eigvalsh3 = eigvalsh3x3


def eigh3(matrix, method='parallelization', eps_theta=1e-6, tol=None):
    """
    Return eigenvalues and eigenvectors of 3x3 hermitian matrices.

    The eigenvalues are obtained by calling `eigvalsh3`, which uses a closed
    form expression.

    To calculate the eigenvectors, one can exploit vector cross products in 3
    dimensions as described in [http://arxiv.org/abs/physics/0610206]
    "Efficient numerical diagonalization of hermitian 3x3 matrices".
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
    different algorithm to get a more precise result. As a remedy for hermitian
    matrices, when the matrix is close to a diagonal matrix, we obtain the
    eigenvectos with Jacobi method, which converges relatively fast for such
    matrices.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.
        We assume `matrix` is hermition (over last two dimensions).

    method : string
        can be either 'cross-product' or 'parallelization'; the latter one is
        the default case.

    eps_theta : float, optional
        specifies when to switch from the main method to jacobi method.
        (Default is 1e-6)
    """
    # For benchmarking see
    # test_and_studies/eig_3x3/eig_decomposition_3x3_benchmarking.html 

    eigvals, (mu, theta, phi) = eigvalsh3(matrix, return_invariants=True)

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
                scale=theta.ravel(),  # theta is the only relevant scale
                tol=tol
                )
    eigvecs[..., 1] = cross_product(eigvecs[..., 2], eigvecs[..., 0]).conj()

    # We now search for case (3) as explained above.
    case3 = (theta <= eps_theta * torch.abs(mu)).ravel()
    
    if torch.sum(case3) > 0:
        eigvecs_view = eigvecs.reshape(-1, 3, 3)
        eigvecs_view[case3] = jacobi_diagonalization(
                matrix.reshape(-1, 3, 3)[case3], tol=tol
                )[1]

    eigvecs = fix_phase(eigvecs)

    return eigvals, eigvecs


# =============================================================================
def nullspace3_from_cross_product(matrix, indices=[0, 1, 2], scale=1, tol=0):
    """For 3x3 hermitian matrices that have zero eigenvalues.

    For determination of the nullspace of a 3x3 matrix, we follow
    "Efficient numerical diagonalization of hermitian 3x3 matrices"
    [http://arxiv.org/abs/physics/0610206],
    which uses vector cross products to construct eigenvectors.

    Here are a few remaks related to the method with cross-product.

    1. The method spelled out in the mentioned paper returns the conjugate of
    left eigenvectors which are equal to right eigenvectors for hermitian
    matrices as well as unitary matrices. The eigenevectors are mainly obtianed
    by exploiting the properties of vector cross products in three dimensions.

    2. The method can be used for any 3x3 matrices as long as the cross product
    does not vanish. Otherwise, the recipe suggested in the mentioned reference
    is specific to hermitian matrices or any matrix that the conjugate of left
    eigenvectors equal the right eigenvectors (like unitary matrices).
    Otherwise, one should modify the method.
    """                                                                         
    a = matrix[..., indices[0], :]
    b = matrix[..., indices[1], :]

    nullspace = cross_product(a, b)

    nullnorm = torch.linalg.vector_norm(nullspace, dim=-1, keepdim=True)
    nullspace = nullspace / nullnorm

    cond = (nullnorm.ravel() <= scale**2 * tol)  # nullnorm = |a x b|
    if torch.sum(cond) > 0:  # if nullnorm is zero at least for one case   
        a = matrix[..., indices[0]].view(-1, 3)
        b = matrix[..., indices[1]].view(-1, 3)
        nullspace[cond] = \
            orthonormal_to_parallel_vectors(a[cond], b[cond], indices=indices)

    return nullspace.view(*matrix.shape[:-1])


# =============================================================================
def nullspace3_from_parallelization(matrix, indices=[0, 1, 2], scale=1, tol=0):
    """For 3x3 matrices that have zero eigenvalues.

    Instead of using the cross product, we first obtain the nullspace of a
    transformed matrix in which the first and second columns are perpendicular
    to the third column; hence the transformed first and second coulmns must be
    parallel as it is assumed the matrix has a nullspace.
    """
    a = matrix[..., indices[0]].view(-1, 3)
    b = matrix[..., indices[1]].view(-1, 3)
    c = matrix[..., indices[2]].view(-1, 3)

    c_sq = torch.sum((c.conj() * c).real, dim=-1)
    coef_a = torch.sum(c.conj() * a, dim=-1) / c_sq
    coef_b = torch.sum(c.conj() * b, dim=-1) / c_sq

    a = a - coef_a.unsqueeze(-1) * c
    b = b - coef_b.unsqueeze(-1) * c
    nullspace = orthonormal_to_parallel_vectors(a, b, indices=indices)
    nullspace[:, indices[2]] = -coef_a * nullspace[:, indices[0]] \
                               -coef_b * nullspace[:, indices[1]]

    nullnorm = torch.linalg.vector_norm(nullspace, dim=-1, keepdim=True)
    nullspace = nullspace / nullnorm

    cond = (c_sq.ravel() <= scale**2 * tol)  # c_sq = |c.c|
    if torch.sum(cond) > 0:
        nullspace[:, indices[0]][cond] = 0 
        nullspace[:, indices[1]][cond] = 0
        nullspace[:, indices[2]][cond] = 1

    return nullspace.view(*matrix.shape[:-1])


# =============================================================================
def cross_product(vec1, vec2):
    """Return cross product of three dimensional vectors vec1 & vec2
    (over the last axis).
    """
    vec3 = torch.zeros_like(vec1)
    vec3[..., 0] = (vec1[..., 1] * vec2[..., 2] - vec1[..., 2] * vec2[..., 1])
    vec3[..., 1] = (vec1[..., 2] * vec2[..., 0] - vec1[..., 0] * vec2[..., 2])
    vec3[..., 2] = (vec1[..., 0] * vec2[..., 1] - vec1[..., 1] * vec2[..., 0])
    return vec3


# =============================================================================
def orthonormal_to_parallel_vectors(vec1, vec2, indices=[0, 1, 2]):
    """Return an orthornormal vector to three dimensional vectors vec1 & vec2
    that are assumed to be parallel.
    """
    x = torch.sum(vec1.conj() * vec2, dim=-1)
    y = torch.sum(vec1.conj() * vec1, dim=-1)
    z = torch.sqrt(x * x + y * y)  # if z = 0, then x = y = 0, then vec1 = 0

    vec3 = torch.zeros_like(vec1)
    vec3[..., indices[0]] = -x / z
    vec3[..., indices[1]] = y / z

    cond = (z == 0).ravel()
    if torch.sum(cond) > 0:
        vec3[..., indices[0]].view(-1)[cond] = 1
        vec3[..., indices[1]].view(-1)[cond] = 0

    return vec3
