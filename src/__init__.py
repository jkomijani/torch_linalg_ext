
# functions from _autograd **reliably** support algorithmic differentiation
from ._autograd import eigh
from ._autograd import eigu
from ._autograd import inverse_eig
from ._autograd import svd
from ._autograd import svd_with_simplified_ad
from ._autograd import reciprocal


# functions from functions **reliably** support algorithmic differentiation
from .functions import eyes_like
from .functions import kronecker_product
