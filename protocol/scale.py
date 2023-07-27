from .ptobase import ProBaseServer, ProBaseClient, NumberType
from typing import Union

import torch
import time
import numpy as np
import comm

__ALL__ = ['ProtocolClient', 'ProtocolServer']

class ProtocolClient(ProBaseClient):
    def send_offline(self) -> None:
        t = time.time()
        self.basic_send_offline(self.r)
        self.stat.time_offline_send += time.time() - t
    
    def recv_offline(self) -> torch.Tensor:
        t0 = time.time()
        data = self.basic_recv_offline()
        t1 = time.time()
        self.pre = data
        t2 = time.time()
        self.stat.time_offline_recv += t1 - t0
        self.stat.time_offline_comp += t2 - t1
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        t0 = time.time()
        data = data - self.r
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online_send += t2 - t1

    def recv_online(self) -> torch.Tensor:
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        t1 = time.time()
        data = data + self.pre
        t2 = time.time()
        self.stat.byte_online_recv += nbyte
        self.stat.time_online_recv += t1 - t0
        self.stat.time_online_comp += t2 - t1
        return data


class ProtocolServer(ProBaseServer):
    def recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        t0 = time.time()
        data = self.basic_recv_offline()
        t1 = time.time()
        if not (isinstance(self.mlast, (int, float)) and self.mlast != 1):
            data /= self.mlast
            t2 = time.time()
            self.stat.time_offline_comp += t2 - t1
        self.stat.time_offline_recv += t1 - t0
        return data
    
    def send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        t0 = time.time()
        data *= self.m
        data -= self.s
        t1 = time.time()
        self.basic_send_offline(data)
        t2 = time.time()
        self.stat.time_offline_comp += t1 - t0
        self.stat.time_offline_send += t2 - t1
    
    def recv_online(self) -> torch.Tensor:
        '''
        Receive (data/mlast) = (x - r/mlast) from client.
        '''
        t0 = time.time()
        data, nbyte = comm.recv_torch(self.socket)
        t1 = time.time()
        data = data/self.mlast
        t2 = time.time()
        self.stat.byte_online_recv += nbyte
        self.stat.time_online_recv += t1 - t0
        self.stat.time_online_comp += t2 - t1
        return data
    
    def send_online(self, data: torch.Tensor) -> None:
        '''
        Send (data*m + s) to client. (data = W(x-r/mlast))
        '''
        t0 = time.time()
        data *= self.m
        data += self.s
        t1 = time.time()
        self.stat.byte_online_send += comm.send_torch(self.socket, data)
        t2 = time.time()
        self.stat.time_online_comp += t1 - t0
        self.stat.time_online_send += t2 - t1
