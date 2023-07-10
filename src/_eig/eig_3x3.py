# Copyright (c) 2023 Javad Komijani

import torch
import numpy as np


from .generic import fix_phase
from .eigh_jacobi import jacobi_diagonalization


# =============================================================================
def eigvalsh3(matrix, return_theta_phi=False):
    r"""
    Return eigenvalues of 3x3 hermitian matrices using closed form expressions.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.
        We assume `matrix` is hermition (over last two dimensions),
        otherwise we treat it similar to pytorch by discarding the
        upper diagonal terms.

    return_theta_phi : boolean
       in addition to the eigenvalues, return theta and phi;
       see the discription of algorithm below.
       (Default is False.)
    
    The algorithm that we use is very common and can be used for any
    :math:`3\times 3` matrix; e.g., see,
    https://hal.science/hal-01501221v1/preview/matrix_exp_and_log_formula.pdf,
    "Efficient numerical diagonalization of hermitian 3x3 matrices"
    [http://arxiv.org/abs/physics/0610206],
    and
    "Symbolic spectral decomposition of 3x3 matrices"
    [https://arxiv.org/abs/2111.02117].
    However, here we assume the matrix is hermitian.

    We now give the algorithm for eigenvalue determination of
    :math:`3\times 3` matrices.

    Without loss of generality we write matrix :math:`A` as

    .. math::

        A = \mu I + \theta M

    where :math:`\text{Tr}\, M = 0` and :math:`\text{Tr}\,M^2 = 2`,
    indicating
    :math:`\theta^2 = \frac{1}{2}\text{Tr}\,\left(A - \mu I\right)^2`.
    Exploiting Cayley-Hamilton theorem for :math:`3\times 3` matrices,
    matrix :math:`M` satisfies

    .. math::

        M^3 = M + \text{det}(M) I

    and therefore :math:`\frac{1}{3}\text{Tr}\, M^3 = \text{det} M`.
    Likewise, the eigenvalues of :math:`M` satisfy:

    .. math::

        \lambda^3 = \lambda + \text{det}(M).

    The three solutions to this equation can be expressed in terms of
    trigonometric functions. To this end we first perform a change of
    variable as :math:`\text{det}(M) = -\frac{2}{\sqrt{27}} \sin 3 \phi`.
    Then, Exploiting the triple-angle trigonometric identity
    :math:`\sin 3\phi = 3\sin \phi - 4 \sin^3\phi`, the three eigenvalues of
    matrix :math:`M` read

    .. math::

        \lambda_k = \frac{2}{\sqrt{3}} \sin \left(\phi + 2\pi k /3\right).

    Note that the sum of the eigenvalues is zero.
    """
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
    
    theta = (a**2 + b**2 + a*b + x**2 + y**2 + s**2 + t**2 + p**2 + q**2)**0.5

    # minus_det is minus determinant of (H - \mu I)
    minus_det = a * (p**2 + q**2) + b * (s**2 + t**2) \
       + (a + b) * (a*b - x**2 - y**2) - 2*(p*s*x + q*t*x - q*s*y + p*t*y)

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
                )
    
    if return_theta_phi:
        return eigvals, (theta, phi)
    else:
        return eigvals


# =============================================================================
def eigvals_su3(matrix):
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

    For special unitary matrices, we can exploit the fact that deteriminant is
    unity and the absolute value of eigenvalues are 1 to simplify the
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
    """
    assert matrix.shape[-2:] == (3, 3), "matrix is supposed to be 3x3"

    mu = matrix.diagonal(dim1=-1, dim2=-2).sum(-1) / 3

    p = mu**2 - mu.conj()
    q = mu**3 + 0.5 - 1.5 * (mu.real**2 + mu.imag**2)

    r_1 = mu
    r_2 = (q + (q**2 - p**3)**0.5)**(1/3.)
    r_3 = p / r_2
    r_3[p == 0] = 0

    w1 = (-1 + 1j * 3**0.5) / 2  # w = exp(i * 2 pi /3)
    w2 = (-1 - 1j * 3**0.5) / 2  # w = exp(i * 4 pi /3)

    eigvals = torch.stack(
               [r_1 + r_2 + r_3,
                r_1 + r_2 * w1 + r_3 * w2,
                r_1 + r_2 * w2 + r_3 * w1
               ],
               dim=-1
               )
    
    _, sorted_ind = torch.sort(eigvals.real, dim=-1)
    eigvals = eigvals.gather(-1, sorted_ind)

    return eigvals


# =============================================================================
def eigh3(matrix, eps_theta=0.01, return_theta_phi=False):
    """
    Return eigenvalues and eigenvectors of 3x3 hermitian matrices.

    The eigenvalues are obtained by calling `eigvalsh3`, which uses closed
    form expressions, and the eigenvectors constructed exploiting vector cross
    products in 3 dimensions as described in
    "Efficient numerical diagonalization of hermitian 3x3 matrices"
    [http://arxiv.org/abs/physics/0610206]. Note that this method is senstive
    to round-off errors when the condition number is large or when the
    eigenvalues are very close to each other. As a remedy, when the matrix is
    close to a diagonal matrix, we obtain the eigenvectos with Jacobi method,
    which converges relatively fast for such matrices.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.
        We assume `matrix` is hermition (over last two dimensions),
        otherwise we treat it similar to pytorch by discarding the
        upper diagonal terms.

    eps_theta: float
        specifies when to switch from the main method to jacobi method.

    return_theta_phi : boolean
       in addition to the eigenvalues, return theta and phi;
       see the discription of algorithm below.
       (Default is False.)
    """

    eigvals, (theta, phi) = eigvalsh3(matrix, return_theta_phi=True)
    
    # comment:
    # eigvals include the eigenvalues in ascending order.
    # There are three cases:
    #     1. all eigenvalues are different
    #     2. the first & second or the second & third eigenvalues are equal
    #     3. all three eigenvalues are (almost) degenerate
    # For case (3), matrix is almost proportional to unity matrix and theta
    # is very small. We take care of this case using `eps_theta`.
    # Putting case (3) asidem we can assume that eigvals[1] is not equal to
    # both eigvals[0] and eigvals[2] at the same time. We therefore obtain the
    # eigvectors corresponding to eigvals[0] and eigvals[2] first and then
    # construct eigvals[1] as cross poduct of them.

    eigvecs = torch.zeros_like(matrix)
    for k in [0, 2]:
        eigval = eigvals[..., k]
        # For efficieny, instead of subtracting `eigval * I` from the matrix,
        # we pass eigval to `extract_eigvec`, which subtracts `eigval` from
        # relevant terms.
        # Although it is not important, we change the columns used for
        # extracting the eigenvectors as `k` changes.
        indices = [k % 3, (k + 1) % 3, (k + 2) % 3]
        a = matrix[..., indices[0]]
        b = matrix[..., indices[1]]
        eigvecs[..., k] = extract_eigvec(
                        a, b, diag_shift=eigval, indices=indices, theta=theta
                        )
    eigvecs[..., 1] = cross_product(eigvecs[..., 2], eigvecs[..., 0]).conj()

    # We now search for case (3) as explained above.
    case3 = (theta <= eps_theta * torch.sum(torch.abs(eigvals), dim=-1)).ravel()
    
    if sum(case3) != 0:
        eigvecs_view = eigvecs.reshape(-1, 3, 3)
        eigvecs_view[case3] = \
                    jacobi_diagonalization(matrix.reshape(-1, 3, 3)[case3])[1]

    eigvecs = fix_phase(eigvecs)

    if return_theta_phi:
        return eigvals, eigvecs, (theta, phi)
    else:
        return eigvals, eigvecs


# =============================================================================
def eig_su3(matrix):
    """
    Return eigenvalues and eigenvectors of 3x3 special unitary matrices.

    The eigenvalues are obtained by calling `eigvals_su3`, which uses closed
    form expressions, and the eigenvectors constructed exploiting vector cross
    products in 3 dimensions as described in
    "Efficient numerical diagonalization of hermitian 3x3 matrices"
    [http://arxiv.org/abs/physics/0610206]. Note that this method is senstive
    to round-off errors when the condition number is large or when the
    eigenvalues are very close to each other.

    Parameters
    ----------
    matrix : tensor
        can be a single matrix or a batch of matrices.
        We assume `matrix` is hermition (over last two dimensions),
        otherwise we treat it similar to pytorch by discarding the
        upper diagonal terms.
    """

    eigvals = eigvals_su3(matrix)
    
    eigvecs = torch.zeros_like(matrix)
    for k in [0, 2]:
        eigval = eigvals[..., k]
        # For efficieny, instead of subtracting `eigval * I` from the matrix,
        # we pass eigval to `extract_eigvec`, which subtracts `eigval` from
        # relevant terms.
        # Although it is not important, we change the columns used for
        # extracting the eigenvectors as `k` changes.
        indices = [k % 3, (k + 1) % 3, (k + 2) % 3]
        a = matrix[..., indices[0]]
        b = matrix[..., indices[1]]
        eigvecs[..., k] = extract_eigvec(
                        a, b, diag_shift=eigval, indices=indices
                        )
    eigvecs[..., 1] = cross_product(eigvecs[..., 2], eigvecs[..., 0]).conj()

    eigvecs = fix_phase(eigvecs)

    return eigvals, eigvecs


# =============================================================================
def extract_eigvec(a, b, diag_shift=None, indices=None, theta=None, tol=1e-14):
    """For 3x3 hermitian mtrices.

    For determination of eigenvectors we follow 
    "Efficient numerical diagonalization of hermitian 3x3 matrices"
    [http://arxiv.org/abs/physics/0610206],
    which uses vector cross products to construct eigenvectors.
    
    We use theta as a reference to see if the cross product is small or not.
    """
    eigvec = cross_product(a, b, diag_shift=diag_shift, indices=indices).conj()
    norm_eigvec = torch.linalg.vector_norm(eigvec, dim=-1, keepdim=True)
    
    eigvec = eigvec / norm_eigvec

    if theta is None:
        cond = norm_eigvec.ravel() < tol
    else:
        cond = norm_eigvec.ravel() / theta.ravel()**2 < tol

    if sum(cond) != 0:  # if norm_c is zero at least for one case
        eigvec.reshape(-1, 3)[cond] = \
            find_vector_perpendicular_to_parallel_a_b(
                 a.reshape(-1, 3)[cond],
                 b.reshape(-1, 3)[cond],
                 diag_shift=diag_shift.reshape(-1)[cond],
                 indices=indices
            )

    return eigvec


def find_vector_perpendicular_to_parallel_a_b(a, b, *, diag_shift, indices):
    """Special case where a and b are parallel"""
    # note that a and b have only two dimensons

    eye = torch.eye(3).reshape(-1, 3, 3).repeat(len(diag_shift), 1, 1)
    a = a - diag_shift * eye[:, :, indices[0]]
    b = b - diag_shift * eye[:, :, indices[1]]
    
    norm_a = torch.linalg.vector_norm(a, dim=-1)
    norm_b = torch.linalg.vector_norm(b, dim=-1)

    
    mylist = [None] * 3
    angle = torch.angle(torch.sum(a.conj() * b, dim=-1))
    norm_ab = (norm_a**2 + norm_b**2)**0.5
    mylist[indices[0]] = norm_b / norm_ab
    mylist[indices[1]] = - torch.exp(-1j* angle) * norm_a / norm_ab
    mylist[indices[2]] = torch.zeros_like(norm_a)

    # what if both norm_a & norm_b are zero? we now fix it as follows:
    cond = norm_ab == 0
    if sum(cond) != 0:
        mylist[indices[0]][cond] = torch.ones_like(norm_a)
        mylist[indices[1]][cond] = torch.zeros_like(norm_a)

    return torch.stack(mylist , dim=-1) + 0J


def cross_product(vec1, vec2, diag_shift=None, indices=None):
    """Use diag_shift to shift vec1[0] & vec2[1]."""
    
    vec3 = torch.zeros_like(vec1)
    vec1 = list(torch.unbind(vec1, dim=-1))
    vec2 = list(torch.unbind(vec2, dim=-1))

    if diag_shift is not None:
        vec1[indices[0]] = vec1[indices[0]] - diag_shift
        vec2[indices[1]] = vec2[indices[1]] - diag_shift
    
    vec3[..., 0] = (vec1[1] * vec2[2] - vec1[2] * vec2[1])
    vec3[..., 1] = (vec1[2] * vec2[0] - vec1[0] * vec2[2])
    vec3[..., 2] = (vec1[0] * vec2[1] - vec1[1] * vec2[0])

    return vec3
