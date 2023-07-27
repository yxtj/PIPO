from .base import LayerClient, LayerServer
from .base import ProtocolServer

from socket import socket
from typing import Union
import numpy as np
import time
import torch
from Pyfhel import Pyfhel


class MaxPoolClient(LayerClient):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, he:Pyfhel,
                 layer:torch.nn.MaxPool2d, device: str) -> None:
        super().__init__(socket, ishape, oshape, he, device)
        self.layer = layer
        
    def online(self, xm) -> torch.Tensor:
        t0 = time.time()
        self.protocol.send_online(xm) # send: x_i .* m_{i-1} - r_i
        t1 = time.time()
        data = self.protocol.recv_online() # x .* mp
        t2 = time.time()
        data = self.layer(data) # max_pool(x) .* m
        t3 = time.time()
        self.stat.time_online_recv += t2 - t1
        self.stat.time_online += t3 - t0
        return data


class MaxPoolServer(LayerServer):
    def __init__(self, socket: socket, ishape: tuple, oshape: tuple, layer: torch.nn.Module, device: str) -> None:
        assert isinstance(layer, torch.nn.MaxPool2d)
        # kernel_size must be no greater than stride
        if isinstance(layer.kernel_size, int):
            assert layer.kernel_size <= layer.stride
        elif isinstance(layer.kernel_size, tuple):
            assert len(layer.kernel_size) == 2 and len(layer.stride) == 2
            assert layer.kernel_size[0] <= layer.stride[0] and layer.kernel_size[1] <= layer.stride[1]
        else:
            raise ValueError("kernel_size must be int or tuple")
        assert layer.padding == 0
        assert layer.dilation == 1
        
        if isinstance(layer.stride, int):
            stride_shape = (layer.stride, layer.stride)
        else:
            stride_shape = layer.stride
        assert ishape[-2]//stride_shape[0] == oshape[-2] and ishape[-1]//stride_shape[1] == oshape[-1]
        
        super().__init__(socket, ishape, oshape, layer, device)
        self.stride_shape = stride_shape
        # make a protocol for pooling (its output shape is the same as the input shape)
        self.protocol_pool = ProtocolServer(self.socket, self.stat, self.he)
    
    def setup(self, last_lyr: LayerServer, m: Union[torch.Tensor, float, int]=None, **kwargs) -> None:
        super().setup(last_lyr, m)
        t = time.time()
        last_pto = last_lyr.protocol if last_lyr is not None else None
        pto = self.protocol
        # set mp and sp
        block = torch.ones(self.stride_shape, device=self.device)
        mp = torch.kron(pto.m, block) # kronecker product
        sp = torch.rand_like(mp)
        self.protocol_pool.setup(self.ishape, mp.shape, s=sp, m=mp, last=last_pto)
        # print("mp", mp)
        self.stat.time_offline += time.time() - t
        
    def cut_input(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        h = self.oshape[-2] * self.stride_shape[0]
        w = self.oshape[-1] * self.stride_shape[1]
        if (h, w) == x.shape[-2:]:
            return x
        else:
            return x[..., :h, :w]
    
    def offline(self) -> np.ndarray:
        t0 = time.time()
        rm = self.protocol_pool.recv_offline() # rm = r_i / m_{i-1}
        t1 = time.time()
        data = self.cut_input(rm)
        t2 = time.time()
        self.protocol_pool.send_offline(data) # r_i / m_{i-1} .* m^p_{i} - s_i
        t3 = time.time()
        self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline += t3 - t0
        return rm
    
    def online(self) -> torch.Tensor:
        t0 = time.time()
        xrm = self.protocol_pool.recv_online() # xrm = (x_i - r_i / m_{i-1})
        t1 = time.time()
        data = self.cut_input(xrm) 
        t2 = time.time()
        self.protocol_pool.send_online(data) # (x_i - r_i / m_{i-1}) .* m^p_{i} + s_i
        t3 = time.time()
        self.stat.time_online_comp += t2 - t1
        self.stat.time_online += t3 - t0
        return xrm
    