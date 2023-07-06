from .ptobase import ProBaseServer, ProBaseClient, NumberType
from typing import Union

import torch
import numpy as np
import comm

__ALL__ = ['ProtocolClient', 'ProtocolServer']

class ProtocolClient(ProBaseClient):
    def send_offline(self) -> None:
        self.basic_send_offline(self.r)
    
    def recv_offline(self) -> torch.Tensor:
        data = self.basic_recv_offline()
        self.pre = data
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        data = data - self.r
        self.stat.byte_online_send += comm.send_torch(self.socket, data)

    def recv_online(self) -> torch.Tensor:
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        data = data + self.pre
        return data


class ProtocolServer(ProBaseServer):
    def setup(self, ishape: tuple, oshape: tuple, s: NumberType=None, m: NumberType=None,
              last: ProBaseServer=None, **kwargs) -> None:
        assert isinstance(last, ProtocolServer)
        assert 'noise_sigma' in kwargs, 'noise_sigma is required in kwargs'
        super().setup(ishape, oshape, s, m, last, **kwargs)
        sigma = kwargs['noise_sigma']
        assert isinstance(sigma, (int, float))
        self.noise_sigma = sigma
        
    def recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        data = self.basic_recv_offline()
        if not (isinstance(self.mlast, (int, float)) and self.mlast != 1):
            data /= self.mlast
        return data
    
    def send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        data *= self.m
        data -= self.s
        self.basic_send_offline(data)
    
    def recv_online(self) -> torch.Tensor:
        '''
        Receive (data/mlast) = (x - r/mlast) from client.
        '''
        data, nbyte = comm.recv_torch(self.socket)
        self.stat.byte_online_recv += nbyte
        data = data/self.mlast
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        '''
        Send (data*m + s) to client. (data = W(x-r/mlast))
        '''
        data += torch.normal(0, self.noise_sigma, data.shape)
        data *= self.m
        data += self.s
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
