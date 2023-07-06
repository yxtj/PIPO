from .ptobase import ProBaseServer, ProBaseClient, NumberType

import torch
import numpy as np
import comm
from typing import Union

from setting import USE_HE


class ProtocolClient(ProBaseClient):
    def setup(self, ishape: tuple, oshape: tuple, r: NumberType=None) -> None:
        super().setup(ishape, oshape, r)

    def send_online(self, data: torch.Tensor) -> None:
        data = data - self.r
        self.stat.byte_online_send += comm.send_torch(self.socket, data)

    def recv_online(self) -> torch.Tensor:
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        data  = data + self.pre
        return data


class ProtocolServer(ProBaseServer):
    def __init__(self, s, stat, he):
        super().__init__(s, stat, he)
        self.offline_buffer = None # used to generate random data for offline phase
    
    def setup(self, ishape: tuple, oshape: tuple, s: NumberType=None, m: NumberType=None,
              last: ProBaseServer=None, **kwargs) -> None:
            #   mlast: NumberType=1, Rlast: torch.Tensor=None
        super().setup(ishape, oshape, s, m, last, **kwargs)
        Rlast = last.Rlast if last is not None else None
        assert len(Rlast) == np.prod(self.oshape)
        self.Rlast = Rlast
        n = np.prod(ishape)
        self.S = torch.randperm(n).reshape(self.ishape) # shuffle matrix
        self.R = torch.argsort(self.S.ravel()).reshape(self.ishape) # unshuffle matrix
    
    def confuse_data(self, data: torch.Tensor) -> torch.Tensor:
        '''
        Shuffle input data.
        Input: a tensor of shape "self.ishape".
        Output: a tensor of shape "self.ishape".
        '''
        res = data.ravel()[self.S].reshape(self.ishape)
        return res
    
    def clearify_data(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        '''
        Pick and unshuffle.
        Input: a tensor of shape "self.oshape".
        Output: a tensor of shape "self.oshape".
        '''
        #assert len(self.Rlast) == 2*np.prod(self.oshape) == np.prod(data.shape)
        res = data.ravel()[self.Rlast].reshape(self.oshape)
        return res
    
    def recv_offline(self) -> torch.Tensor:
        if USE_HE:
            data, nbytes = comm.recv_he_matrix(self.socket, self.he)
        else:
            data, nbytes = comm.recv_torch(self.socket)
        self.stat.byte_offline_recv += nbytes
        self.offline_buffer = data
        data = self.clearify_data(data)
        return data
        
    def send_offline(self, data: np.ndarray) -> None:
        data = self.confuse_data(data)
        if USE_HE:
            self.stat.byte_offline_send += comm.send_he_matrix(self.socket, data, self.he)
        else:
            self.stat.byte_offline_send += comm.send_torch(self.socket, data)

    def recv_online(self) -> torch.Tensor:
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        data = self.clearify_data(data)
        data = data/self.mlast
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        data = data*self.m
        data = self.confuse_data(data)
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
