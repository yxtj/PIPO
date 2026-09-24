try:  # Pyfhel is optional; only needed when use_he is enabled
    from Pyfhel import Pyfhel
except ImportError:  # pragma: no cover
    Pyfhel = None
import torch
import numpy as np

# linear layers

def conv(x: np.ndarray, w: torch.Tensor) -> np.ndarray:
    pass

def fc(x: np.ndarray, w: torch.Tensor) -> np.ndarray:
    pass

# pooling layers

def avgpool(x: np.ndarray, kernel: int, padding: int=0, stride: int=1) -> np.ndarray:
    pass

def maxpool(x: np.ndarray, kernel: int, padding: int=0, stride: int=1) -> np.ndarray:
    pass

# activation functions

def relu(x: np.ndarray) -> np.ndarray:
    pass

