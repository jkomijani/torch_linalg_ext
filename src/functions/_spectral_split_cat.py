# Copyright (c) 2024 Javad Komijani


import torch

ifftn, fftn = torch.fft.ifftn, torch.fft.fftn


# =============================================================================
def spectral_split(x, *, dims,
        spectral_scale=None, symmetric_norm=True, as_nested=False
        ):
    """
    Computes multi dimensional FFT of input tensor, along the specified
    dimensions, divides the spectrum in two pieces along each of those
    dimensions, performs the multi dimenstional inverse FFT of each block,
    and returns all blocks as a list. A symmetric normalization factor for both
    FFT and its inverse are assumed by default (unlike torch convention).

    Paramerers
    ----------
    x : Tensor
        the input tensor.
    dims : Tuple[int]
        dimensions to be transformed.
    spectral_scale : Tensor or number, optional
        if provided, the FFT of the input tensor gets scaled accordingly
        before getting splitted. (Default is None.)
        Note that if the input tensor ``x`` is real, ``spectral_scale`` cannot
        be arbitrary; it must be Hermitian-symmetric like the FFT of ``x``,
        otherwise ``spectral_cat`` and ``spectral_split`` are not each others
        inverse. (To this end, one can use ``conjugate_counterpart`` function.)
    symmetric_norm : Boolean, optional
        when True/False, symmetric/torch conventions for normalization are
        employed (Default is True).
    as_nested : Boolean, optional
        if True the output is organized as nested lists. (Default is False.)
    """

    if symmetric_norm:
        norm = 2 ** (-len(dims) / 2)
        scale = norm * (1 if spectral_scale is None else spectral_scale)
    else:
        scale = spectral_scale

    if scale is None:
        x_tilde = fftn(x, dim=dims)
    else:
        x_tilde = fftn(x, dim=dims) * scale

    split = Splitter.apply

    if torch.is_complex(x):
        blocks = [ifftn(blk, dim=dims) for blk in split(dims, x_tilde)]
    else:
        blocks = [ifftn(blk, dim=dims).real for blk in split(dims, x_tilde)]

    if as_nested:
        blocks = pack_as_nested(blocks)

    return blocks


def spectral_cat(blocks, *, dims,
        spectral_scale=None, symmetric_norm=True, as_nested=False
        ):
    """Reverse of spectral_split"""

    if as_nested:
        blocks = unpack(blocks, len(dims))

    x_tilde = \
            Concatenator.apply(dims, *[fftn(blk, dim=dims) for blk in blocks])

    if symmetric_norm:
        norm = 2 ** (-len(dims) / 2)
        scale = norm * (1 if spectral_scale is None else spectral_scale)
    else:
        scale = spectral_scale

    if scale is None:
        x = ifftn(x_tilde, dim=dims)
    else:
        x = ifftn(x_tilde / scale, dim=dims)

    if not torch.is_complex(blocks[0]):
        x = x.real

    return x


# =============================================================================
def splitted_fftn(x, *, dims,
        spectral_scale=None, symmetric_norm=False, as_nested=False
        ):
    """Similar to ``spectal_split``, except the returned blocks are in Fourier
    space and the default value of ``symmetric_norm`` is set to False.
    """

    if symmetric_norm:
        norm = 2 ** (-len(dims) / 2)
        scale = norm * (1 if spectral_scale is None else spectral_scale)
    else:
        scale = spectral_scale

    if scale is None:
        x_tilde = fftn(x, dim=dims)
    else:
        x_tilde = fftn(x, dim=dims) * scale

    blocks = Splitter.apply(dims, x_tilde)

    if as_nested:
        blocks = pack_as_nested(blocks)

    return blocks


def splitted_ifftn(blocks, *, dims,
        spectral_scale=None, symmetric_norm=False, as_nested=False
        ):
    """Reverse of splitted_ifftn"""

    if as_nested:
        blocks = unpack(blocks, len(dims))

    x_tilde = Concatenator.apply(dims, *blocks)

    if symmetric_norm:
        norm = 2 ** (-len(dims) / 2)
        scale = norm * (1 if spectral_scale is None else spectral_scale)
    else:
        scale = spectral_scale

    if scale is None:
        x = ifftn(x_tilde, dim=dims)
    else:
        x = ifftn(x_tilde / scale, dim=dims)


    return x


# =============================================================================
class Splitter(torch.autograd.Function):
    """Manipulate the input for RFFT partitioning"""

    @staticmethod
    def forward(ctx, dims, x_tilde):

        pre_blocks = [x_tilde]

        inds = tuple([slice(None)] * x_tilde.ndim)
        coeff = 2**(-1/2)

        def fix_cutlines(x_tilde_n, axis):
            cut = x_tilde_n.shape[axis] // 4
            inds1, inds3 = list(inds), list(inds)
            inds1[axis] = cut
            inds3[axis] = - cut
            x_tilde_n[inds1], x_tilde_n[inds3] = (
                    (x_tilde_n[inds1] + x_tilde_n[inds3]) * coeff,
                    (x_tilde_n[inds1] - x_tilde_n[inds3]) * (1j * coeff)
                    )
            return x_tilde_n

        def splitcat(x_tilde_n, axis):
            ell = x_tilde_n.shape[axis]
            assert ell > 3, f"{axis} axis is too small to be splitted"
            cut = ell // 4
            splt = torch.split(x_tilde_n, [cut, ell - 2 * cut, cut], dim=axis)
            return torch.cat([splt[0], splt[2]], dim=axis), splt[1]

        for axis in dims:
            blocks = [None] * (2 * len(pre_blocks))
            for n, x_tilde_n in enumerate(pre_blocks):
                x_tilde_n = fix_cutlines(x_tilde_n, axis)
                blocks[2 * n: 2 * n + 2] = splitcat(x_tilde_n, axis)
            pre_blocks = blocks

        if ctx is not None:
            ctx.save_for_backward(makesure_tensor(dims))

        # The output is returned as tuple for "torch.autograd.gradcheck"
        return tuple(blocks)

    @staticmethod
    def backward(ctx, *grad_blocks):
        dims = ctx.saved_tensors[0].tolist()
        grad_dims = None
        grad_x_tilde = Concatenator.forward(None, dims, *grad_blocks)
        return grad_dims, grad_x_tilde


class Concatenator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, dims, *blocks):
        # The input is taken as tuple for "torch.autograd.gradcheck"

        inds = tuple([slice(None)] * blocks[0].ndim)
        coeff = 2**(-1/2)

        def undo_cutlines(x_tilde_n, axis):
            cut = x_tilde_n.shape[axis] // 4
            inds1, inds3 = list(inds), list(inds)
            inds1[axis] = cut
            inds3[axis] = - cut
            x_tilde_n[inds1], x_tilde_n[inds3] = (
                    (x_tilde_n[inds1] - 1j * x_tilde_n[inds3]) * coeff,
                    (x_tilde_n[inds1] + 1j * x_tilde_n[inds3]) * coeff
                    )
            return x_tilde_n

        def undo_splitcat(splt_0, splt_1, axis):
            cut = splt_0.shape[axis] // 2
            splt = torch.split(splt_0, [cut, cut], dim=axis)
            return torch.cat([splt[0], splt_1, splt[1]], dim=axis)

        for axis in dims[::-1]:
            pre_blocks = [None] * (len(blocks) // 2)
            for n, _ in enumerate(pre_blocks):
                x_tilde_n = undo_splitcat(*blocks[2 * n: 2 * n + 2], axis)
                pre_blocks[n] = undo_cutlines(x_tilde_n, axis)
            blocks = pre_blocks

        assert len(blocks) == 1
        x_tilde = blocks[0]

        if ctx is not None:
            ctx.save_for_backward(makesure_tensor(dims))

        return x_tilde

    @staticmethod
    def backward(ctx, grad_x_tilde):
        dims = ctx.saved_tensors[0].tolist()
        grad_dims = None
        grad_blocks = Splitter.forward(None, dims, grad_x_tilde)
        return (grad_dims, *grad_blocks)


# ONE CAN USE FOLLOWING FOR *** rfft ***, BUT BACKWARD PROPAGATION OF
# DERIVATIVES WORKS WITH A STRANGE COMBINATION OF COEEFFICIENTS, INDICATING I
# AM NOT SURE HOW PYTORCH HANDLES THEM.
#
# def take_care_rfft_axis(x_tilde_n):
#     cut = (x_tilde_n.shape[-1] - 1) // 2
#     x_inds1 = x_tilde_n[..., cut]
#     x_inds3 = conjugate_counterpart(x_inds1).conj()
#     x_inds1, x_inds3 = (
#             (x_inds1 + x_inds3) * 0.5,
#             (x_inds1 - x_inds3) * (1j * 0.5)
#             )
#     return (
#         torch.cat([x_tilde_n[..., :cut], x_inds3.unsqueeze(-1)], dim=-1),
#         torch.cat([x_inds1.unsqueeze(-1), x_tilde_n[..., cut+1:]], dim=-1)
#         )
#
# def undo_take_care_rfft_axis(splt_0, splt_1):
#     x_inds1 = (splt_1[..., 0] - 1j * splt_0[..., -1])
#     return torch.cat(
#             [splt_0[..., :-1], x_inds1.unsqueeze(-1), splt_1[..., 1:]],
#             dim=-1
#            )


# =============================================================================
def makesure_tensor(dims):
    if torch.is_tensor(dims):
        return dims
    else:
        return torch.tensor(dims)


def conjugate_counterpart(x, dims=None):
    """Return ``x[:, -i_1, -i_2, ..., -i_n]``.

    Note that the FFT of a real signal is Hermitian-symmetric as
    ``x[:, -i_1, -i_2, ..., -i_n] = conj(x[:, i_1, i_2, ..., i_n])``.
    """
    if dims is None:
        dims = tuple(range(1, x.ndim))
    return torch.flip(torch.roll(x, [-1] * len(dims), dims=dims), dims=dims)


def pack_as_nested(blocks):
    n = len(blocks) // 2
    if n == 1:
        return [blocks[0], blocks[1]]
    else:
        return [pack_as_nested(blocks[:n]), pack_as_nested(blocks[n:])]


def unpack(nested, depth):
    if depth == 1:
        return nested
    else:
        return [*unpack(nested[0], depth - 1), *unpack(nested[1], depth - 1)]


# =============================================================================
def gradcheck(*, shape, dims, complex_input=False):
    """For sanity check of the implemented backward propagations.
    For example use:
        >>> gradcheck(shape=(2, 8, 3, 9), dims=(1, 3))
    """

    torch.set_default_dtype(torch.float64)  # double precision

    scale = torch.randn(*shape)
    if not complex_input:
        # then, scale must be Hermitian-symmetric
        scale = (scale + conjugate_counterpart(scale, dims=dims)) / 2.

    split = lambda x: spectral_split(x, dims=dims, spectral_scale=scale)
    cat = lambda *blocks: spectral_cat(blocks, dims=dims, spectral_scale=scale)

    if complex_input:
        x = torch.randn(*shape) + 1j * torch.randn(*shape)
    else:
        x = torch.randn(*shape)
    x.requires_grad = True
    blocks = split(x)

    print("spectral_cat(spectral_split(.)) == Identity:", end='\t')
    x_hat = spectral_cat(blocks, dims=dims, spectral_scale=scale)
    print(torch.mean((x_hat - x).abs()).item() < 1e-13)

    print("gradcheck(spectral_split):", end='\t')
    print(torch.autograd.gradcheck(split, x))
    print("gradcheck(spectral_cat):", end='\t')
    print(torch.autograd.gradcheck(cat, blocks))

    fftn = lambda x: splitted_fftn(x, dims=dims, spectral_scale=scale)
    ifftn = lambda *blks: splitted_ifftn(blks, dims=dims, spectral_scale=scale)

    blocks = fftn(x)

    print("splitted_ifftn(splitted_fftn(.)) == Identity:", end='\t')
    x_hat = splitted_ifftn(blocks, dims=dims, spectral_scale=scale)
    print(torch.mean((x_hat - x).abs()).item() < 1e-13)

    print("gradcheck(splitted_fftn):", end='\t')
    print(torch.autograd.gradcheck(fftn, x))
    print("gradcheck(splitted_ifftn):", end='\t')
    print(torch.autograd.gradcheck(ifftn, blocks))
