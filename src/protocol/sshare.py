import torch

__all__ = ['gen_add_share', 'gen_mul_share']

__ADD_SHARE_RANGE__ = 16.0
__MUL_SHARE_RANGE__ = 16.0
__POSITIVE_EPS__ = 2**-4


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
