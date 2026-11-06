"""No-encryption HE backend: identity crypto and plain float serialization."""

import numpy as np
import torch

from .engine import _DTYPE_FLOAT32, _DTYPE_FLOAT64, HEEngine


class PlainEngine(HEEngine):
    """Complete no-encryption backend: identity crypto and plain float
    serialization, so callers need no special casing."""

    enabled = False

    def encrypt(self, plain: torch.Tensor) -> torch.Tensor:
        return plain

    def decrypt(self, cipher: torch.Tensor) -> torch.Tensor:
        return cipher

    def to_bytes_context(self) -> bytes:
        return b""

    def from_bytes_context(self, data: bytes) -> None:
        return None

    def to_bytes_public_key(self) -> bytes:
        return b""

    def from_bytes_public_key(self, data: bytes) -> None:
        return None

    def to_bytes_secret_key(self) -> bytes:
        return b""

    def from_bytes_secret_key(self, data: bytes) -> None:
        return None

    def serialize_ciphertext(self, ct) -> bytes:
        return np.asarray(ct, dtype=np.float64).tobytes()

    def deserialize_ciphertext(self, data: bytes) -> np.float64:
        return np.float64(np.frombuffer(data, dtype=np.float64)[0])

    def serialize_array(self, arr: torch.Tensor) -> bytes:
        t = arr.detach().cpu().contiguous().numpy()
        if t.dtype == np.float64:
            dtype_code = _DTYPE_FLOAT64
        elif t.dtype == np.float32:
            dtype_code = _DTYPE_FLOAT32
        else:
            raise TypeError("NoHEEngine serializes float tensors only, got {}".format(t.dtype))
        header = self._pack_header_(t.shape, dtype_code, t.itemsize)
        return header + t.tobytes()

    def deserialize_array(self, data: bytes) -> torch.Tensor:
        shape, dtype_code, elem_len, offset = self._unpack_header_(data)
        dtype = {_DTYPE_FLOAT32: np.float32, _DTYPE_FLOAT64: np.float64}[dtype_code]
        flat = np.frombuffer(data[offset:], dtype=dtype)
        return torch.from_numpy(flat.reshape(shape).copy())
