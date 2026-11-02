import hashlib

import torch

__all__ = ['gen_add_share', 'gen_mul_share', 'gen_rand_pos', 'block_shuffle_perm',
           'gen_shuffle', 'gen_reverse_shuffle', 'shuffle_tensor', 'deshuffle_tensor',
           'pad_count', 'pad_tensor', 'depad_tensor']

__ADD_SHARE_RANGE__ = 16.0
__MUL_SHARE_RANGE__ = 16.0
__POSITIVE_EPS__ = 2**-4
# relative gaussian perturbation of the resampled pad values
__PAD_NOISE_RATIO__ = 0.1


def gen_add_share(shape: tuple, dtype=None, device=None) -> torch.Tensor:
    '''
    Range: [-__ADD_SHARE_RANGE__/2, __ADD_SHARE_RANGE__/2)
    mean = 0.0; variance = __ADD_SHARE_RANGE__**2/12
    '''
    return __ADD_SHARE_RANGE__*torch.rand(shape, dtype=dtype, device=device) - __ADD_SHARE_RANGE__/2


def gen_mul_share(shape: tuple, dtype=None, device=None) -> torch.Tensor:
    '''
    Range: [__POSITIVE_EPS__, __MUL_SHARE_RANGE__)
    mean ~ __MUL_SHARE_RANGE__ /2 ; = (__MUL_SHARE_RANGE__ + __POSITIVE_EPS__)/2;
    variance ~ __MUL_SHARE_RANGE__**2/12 ; = (__MUL_SHARE_RANGE__ - __POSITIVE_EPS__)**2/12;
    '''
    return (__MUL_SHARE_RANGE__ - __POSITIVE_EPS__)*torch.rand(shape, dtype=dtype, device=device) + __POSITIVE_EPS__


def gen_rand_pos(shape: tuple, low: float, high: float, dtype=None, device=None) -> torch.Tensor:
    '''
    Range: [low, high)
    mean = (high - low) / 2
    variance = (high - low)**2/12
    '''
    return (high - low)*torch.rand(shape, dtype=dtype, device=device) + low


def block_shuffle_perm(shape: tuple, stride, device=None):
    '''
    Block-wise shuffle permutations for a feature map, used to keep pooling
    (kernel <= stride, so windows do not overlap) correct while the client
    never sees the real spatial layout.

    The feature map is split into spatial blocks of ``stride`` (i.e. one
    pooling window each, spanning all channels). The block grid is permuted as
    a whole (the same block arrangement for every channel), and inside each
    block every channel is permuted independently over its ``stride`` spatial
    elements. Because a pooling window never crosses a block boundary and
    pooling commutes with any within-window reordering, the shuffled tensor
    pools to the *permuted* pooled output.

    Args:
        shape: feature-map shape, at least 3-D; the last three dims are
            ``(C, H, W)`` and must be divisible by ``stride``. Any leading
            (batch) dims must have size 1.
        stride: pooling stride, an int or a ``(s_h, s_w)`` tuple.
        device: torch device for the generated index tensors.

    Returns:
        ``(p_full, p_block)`` gather permutations:
          * ``p_full`` has ``shape`` and permutes all ``C*H*W`` elements of the
            feature map (apply as ``x.ravel()[p_full]``);
          * ``p_block`` has the pooled shape ``shape[:-3] + (C, H//s_h, W//s_w)``
            and is the induced permutation of the pooled output, i.e.
            ``maxpool(x.ravel()[p_full]) == maxpool(x).ravel()[p_block]``.
    '''
    if isinstance(stride, int):
        stride = (stride, stride)
    s_h, s_w = stride
    shape = tuple(shape)
    assert len(shape) >= 3, "shape must be at least 3-D (C, H, W)"
    assert all(d == 1 for d in shape[:-3]), "leading (batch) dims must have size 1"
    C, H, W = shape[-3:]
    assert H % s_h == 0 and W % s_w == 0, "H and W must be divisible by the stride"
    h, w = H // s_h, W // s_w
    if device is None:
        device = 'cpu'

    # block grid permutation, shared across all channels
    block_perm = torch.randperm(h*w, device=device)
    # independent spatial permutation of the s_h*s_w elements inside each
    # (block, channel)
    intra = torch.rand(h*w, C, s_h*s_w, device=device).argsort(dim=-1)

    # derive the full gather permutation by applying the block-wise shuffle to
    # an index tensor laid out as (C, H, W)
    idx = torch.arange(C*H*W, device=device).reshape(C, H, W)
    b = idx.reshape(C, h, s_h, w, s_w).permute(1, 3, 0, 2, 4)  # (h, w, C, s_h, s_w)
    b = b.reshape(h*w, C, s_h*s_w)[block_perm]  # permute the block grid
    b = torch.gather(b, 2, intra)  # permute inside each (block, channel)
    p_full = b.reshape(h, w, C, s_h, s_w).permute(2, 0, 3, 1, 4).reshape(shape)

    # induced permutation of the pooled output: the block grid permutation
    # tiled over channels (flat index = c*(h*w) + block_perm[k])
    base = torch.arange(C, device=device).reshape(C, 1, 1)*(h*w)
    p_block = (base + block_perm.reshape(1, h, w)).reshape(shape[:-3] + (C, h, w))
    return p_full, p_block


def _numel(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def gen_shuffle(shape, dtype=None, device=None) -> torch.Tensor:
    '''Random shuffle pattern: a permutation of ``numel(shape)`` shaped as
    ``shape`` (gather order, i.e. ``x.ravel()[pattern]`` permutes ``x``).'''
    return torch.randperm(_numel(shape), dtype=dtype, device=device).reshape(shape)


def gen_reverse_shuffle(pattern: torch.Tensor, dtype=None, device=None) -> torch.Tensor:
    '''Inverse of a shuffle ``pattern`` (same shape), so that
    ``x.ravel()[gen_reverse_shuffle(p)]`` undoes ``x.ravel()[p]``.'''
    rev = torch.argsort(pattern.ravel().to(device=device))
    if dtype is not None:
        rev = rev.to(dtype)
    return rev.reshape(pattern.shape)


def shuffle_tensor(tensor: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    '''Permute ``tensor`` by the gather ``pattern`` (reshaped to pattern.shape);
    ``pattern`` must have the same number of elements as ``tensor``.'''
    return tensor.ravel()[pattern.ravel()].reshape(pattern.shape)


def deshuffle_tensor(tensor: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    '''Undo a shuffle: ``pattern`` is the *reverse* pattern produced by
    ``gen_reverse_shuffle``; the tensor is gathered by it (reshaped to
    pattern.shape).'''
    return tensor.ravel()[pattern.ravel()].reshape(pattern.shape)


def _data_seed(t: torch.Tensor) -> int:
    '''Lightweight data-dependent seed: blake2b over the raw tensor bytes.
    Binding the pad RNG to the data defeats replay attacks: equal inputs
    reproduce equal pads while different inputs produce unrelated pads, so a
    client comparing several rounds cannot spot the pad positions.'''
    b = t.detach().cpu().numpy().tobytes()
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), 'little')


def _gen_pad(real: torch.Tensor, count: int, device) -> torch.Tensor:
    '''``count`` pads resampled from ``real`` plus a small gaussian
    perturbation, so their distribution matches the real elements.'''
    cpu = real.detach().cpu()
    g = torch.Generator()
    g.manual_seed(_data_seed(cpu))
    idx = torch.randint(0, cpu.numel(), (count,), generator=g)
    noise = torch.randn(count, generator=g)
    std = cpu.std() if cpu.numel() > 1 else torch.zeros(())
    pad = cpu[idx] + __PAD_NOISE_RATIO__ * std * noise
    return pad.to(device)


def pad_count(n: int, ratio: float) -> int:
    '''Number of pads appended to an ``n``-element tensor (ratio 0 = none,
    1 = as many pads as real elements).'''
    return int(round(n * ratio))


def pad_tensor(data: torch.Tensor, oshape, dtype=None, device=None) -> torch.Tensor:
    '''Pad ``data`` up to ``oshape``, comparing every dimension: the real
    elements are kept in the leading slice of each dimension (the "front")
    and the extra trailing region is filled with random pads sampled from the
    data itself. ``oshape`` must have as many dims as ``data`` and be no
    smaller dimension-wise.'''
    oshape = tuple(oshape)
    assert data.dim() == len(oshape), 'oshape must have as many dims as data'
    dshape = tuple(data.shape)
    assert all(o >= s for o, s in zip(oshape, dshape)), 'oshape must not shrink data'
    if device is None:
        device = data.device
    lead = tuple(slice(0, s) for s in dshape)
    out = torch.empty(oshape, dtype=data.dtype, device=device)
    out[lead] = data
    extra = out.numel() - data.numel()
    if extra > 0:
        pad = _gen_pad(data.reshape(-1), extra, device)
        mask = torch.ones(oshape, dtype=torch.bool, device=device)
        mask[lead] = False
        out[mask] = pad
    if dtype is not None:
        out = out.to(dtype)
    return out


def depad_tensor(data: torch.Tensor, ishape, dtype=None, device=None) -> torch.Tensor:
    '''Undo ``pad_tensor``: take the leading slice of each dimension, i.e. the
    real elements, restoring shape ``ishape``. ``ishape`` must have as many
    dims as ``data`` and be no larger dimension-wise.'''
    ishape = tuple(ishape)
    assert data.dim() == len(ishape), 'ishape must have as many dims as data'
    assert all(o >= s for s, o in zip(ishape, data.shape)), 'ishape must not exceed data'
    res = data[tuple(slice(0, s) for s in ishape)].contiguous()
    if dtype is not None:
        res = res.to(dtype)
    return res