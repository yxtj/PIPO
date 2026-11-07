import socket
import torch
import numpy as np
from typing import Union

import src.comm as comm
from src.heutil import HEEngine
from src.layer_basic.stat import Stat
from .crypto import gen_add_share, gen_mul_share

NumberType = Union[int, float, torch.Tensor, np.ndarray]

# Protocol API: setup, send_online, recv_online, send_offline, recv_offline
# The send/recv methods should be state-free functions.

class ProtocolBase():
    # Offline-resource fields persisted by save_res() and restored by passing
    # them back as setup() kwargs; each protocol class declares its own list.
    # Derived fields (mlast / rplast / mp / kin) are intentionally absent:
    # setup() recomputes them from the restored values and the previous layer.
    res_fields = ()

    def __init__(self, s: socket.socket, stat: Stat, he: HEEngine, device: str):
        self.socket = s
        self.stat = stat
        self.he = he
        self.device = device
        self.ishape = None
        self.oshape = None

    def setup(self, ishape:tuple, oshape:tuple, **kwargs) -> None:
        self.ishape = ishape
        self.oshape = oshape

    def send_online(self, data: torch.Tensor) -> None:
        raise NotImplementedError

    def recv_online(self) -> torch.Tensor:
        raise NotImplementedError

    # helpers for serializing and deserializing local state

    def to_bytes(self) -> bytes:
        return b''

    def from_bytes(self, b: bytes) -> None:
        return None

    # helpers for persisting / restoring offline resources

    def save_res(self) -> dict:
        """Snapshot the offline-resource fields (tensors moved to cpu)."""
        state = {}
        for f in self.res_fields:
            v = getattr(self, f)
            state[f] = v.detach().cpu() if torch.is_tensor(v) else v
        return state

    def res_kwargs(self, state: dict) -> dict:
        """Pick the persisted fields out of a loaded resource dict as setup()
        kwargs (tensors moved to this protocol's device)."""
        kwargs = {}
        for f in self.res_fields:
            if f not in state or state[f] is None:
                continue
            v = state[f]
            kwargs[f] = v.to(self.device) if torch.is_tensor(v) else v
        return kwargs

    # helpers for generating shares

    def _gen_add_share_(self, v: NumberType, shape: tuple) -> torch.Tensor:
        if v is not None:
            assert isinstance(v, (int, float, torch.Tensor, np.ndarray))
            if isinstance(v, torch.Tensor):
                assert v.shape == shape
                res = v
            elif isinstance(v, np.ndarray):
                assert v.shape == shape
                res = torch.from_numpy(v)
            elif isinstance(v, (int, float)):
                # res = v
                res = torch.zeros(shape, device=self.device).fill_(v)
            res = res.to(self.device)
        else:
            res = gen_add_share(shape, device=self.device)
        return res

    def _gen_mul_share_(self, v: NumberType, shape: tuple) -> torch.Tensor:
        if v is not None:
            assert isinstance(v, (int, float, torch.Tensor, np.ndarray))
            if isinstance(v, torch.Tensor):
                assert v.shape == shape
                res = v
            elif isinstance(v, np.ndarray):
                assert v.shape == shape
                res = torch.from_numpy(v)
            elif isinstance(v, (int, float)):
                # res = v
                if v == 0:
                    res = torch.zeros(shape, device=self.device)
                elif v == 1:
                    res = torch.ones(shape, device=self.device)
                else:
                    res= torch.zeros(shape, device=self.device).fill_(v)
            res = res.to(self.device)
        else:
            res = gen_mul_share(shape, device=self.device)
        return res


# Client workflow: send -> recv
class ProBaseClient(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: HEEngine, device: str='cpu'):
        super().__init__(s, stat, he, device)

    def setup(self, ishape:tuple, oshape:tuple, **kwargs) -> None:
        super().setup(ishape, oshape)
        # restore a persisted offline value for `pre` (normally received
        # during the offline interaction; without a resource it stays None)
        pre = kwargs.get('pre')
        if pre is not None:
            self.pre = pre.to(self.device)

    def send_offline(self) -> None:
        raise NotImplementedError

    def recv_offline(self) -> torch.Tensor:
        raise NotImplementedError

    def basic_send_offline(self, data: torch.Tensor) -> None:
        data = self.he.encrypt(data)
        nbyte = comm.send_chunk(self.socket, self.he.serialize_array(data))
        self.stat.byte_offline_send += nbyte

    def basic_recv_offline(self) -> torch.Tensor:
        blob, nbyte = comm.recv_chunk(self.socket)
        data = self.he.decrypt(self.he.deserialize_array(blob))
        data = data.to(self.device)
        self.stat.byte_offline_recv += nbyte
        return data


# Server workflow: recv -> process -> send
class ProBaseServer(ProtocolBase):
    def __init__(self, s: socket.socket, stat: Stat, he: HEEngine, device: str='cpu'):
        super().__init__(s, stat, he, device)
        self.last = None
        self.ltype = 'linear'
        self.is_local_layer = False
        self.is_last_layer = False

    def setup(self, ishape: tuple, oshape: tuple, last: ProtocolBase=None, **kwargs) -> None:
        super().setup(ishape, oshape)
        self.last = last
        # semantic layer information; protocols decide from it whether to
        # mask/shuffle/pad the output and how to propagate the last state
        self.ltype = kwargs.get('ltype', 'linear')
        self.is_local_layer = kwargs.get('is_local_layer', False)
        self.is_last_layer = kwargs.get('is_last_layer', False)

    def gen_mpooling(self, stride_shape: tuple) -> ProtocolBase:
        raise NotImplementedError

    def send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        raise NotImplementedError

    def recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        raise NotImplementedError

    def basic_send_offline(self, data: Union[torch.Tensor, np.ndarray]) -> None:
        nbyte = comm.send_chunk(self.socket, self.he.serialize_array(data))
        self.stat.byte_offline_send += nbyte

    def basic_recv_offline(self) -> Union[torch.Tensor, np.ndarray]:
        blob, nbyte = comm.recv_chunk(self.socket)
        data = self.he.deserialize_array(blob)
        if isinstance(data, torch.Tensor):
            data = data.to(self.device)
        self.stat.byte_offline_recv += nbyte
        return data
