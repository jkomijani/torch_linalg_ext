
import torch

from . import eig_autograd
from . import svd_autograd

eigh = eig_autograd.Eigh.apply
eigu = eig_autograd.Eigu.apply

svd = svd_autograd.SVD.apply_wrapper
