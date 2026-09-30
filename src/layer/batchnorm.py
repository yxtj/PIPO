from .base import LayerClient, LayerServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
try:  # Pyfhel is optional; only needed when use_he is enabled
    from Pyfhel import Pyfhel
except ImportError:  # pragma: no cover
    Pyfhel = None
class BnClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)

class BnServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
        # inference-time BatchNorm is a fixed per-channel affine
        #   y = gamma * (x - running_mean) / sqrt(running_var + eps) + beta
        # batch statistics would be data-dependent and cannot be masked
        assert layer.running_mean is not None, \
            "BatchNorm requires tracked running statistics (eval-mode affine)."
        super().__init__(socket, ishape, oshape, layer, device)
        # offline must compute the linear (homogeneous) part only. Stripping
        # .bias (beta) is not enough: the -gamma*running_mean/std shift is not
        # stored in .bias and must not appear in the offline term either, so
        # scale by gamma/sqrt(running_var + eps) directly. The full effective
        # bias (beta - gamma*running_mean/std) is applied online, where the
        # client's offline+online sum cancels into gamma*x/std + beta.
        w = layer.weight if layer.weight is not None else 1.0
        # ishape is the runtime (batch-prefixed) shape, e.g. (1, C) or
        # (1, C, H, W): the channel dim is ishape dim 0 / runtime dim 1.
        # Reshape the per-channel scale to (1, C, 1, ..., 1) to broadcast
        # onto the channel dim only.
        self.w_lin = (w / torch.sqrt(layer.running_var + layer.eps)).detach()
        self.w_lin = self.w_lin.reshape((1, -1) + (1,) * (len(ishape) - 2))

    def offline(self) -> np.ndarray:
        t0 = time.time()
        rm = self.protocol.recv_offline() # recv: r'_i = r_i / m_{i-1}
        t1 = time.time()
        data = self.w_lin * rm # (gamma/std) * r'_i, no shift
        t2 = time.time()
        self.protocol.send_offline(data) # send: ((gamma/std) * r'_i) .* m_i + s_i
        t3 = time.time()
        self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline += t3 - t0
        return rm

    def online(self) -> torch.Tensor:
        t0 = time.time()
        xmr_i = self.protocol.recv_online() # recv: xmr_i = x_i - r_i / m_{i-1}
        t1 = time.time()
        data = self.layer(xmr_i) # gamma*(xmr_i - mean)/std + beta
        t2 = time.time()
        self.protocol.send_online(data) # send: (...) .* m_i - s_i
        t3 = time.time()
        self.stat.time_online_comp += t2 - t1
        self.stat.time_online += t3 - t0
        return xmr_i
