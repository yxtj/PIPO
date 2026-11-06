"""Pyfhel CKKS HE backend; the only module that imports Pyfhel (lazily)."""

import numpy as np
import torch

from .engine import HEEngine


class PyfhelEngine(HEEngine):
    """Pyfhel CKKS backend. Created without keys (server side) it stays an
    empty instance until `from_bytes_context` injects the client context."""

    enabled = True

    DEFAULT_PARAMS = {"scheme": "ckks", "n": 2 ** 13, "scale": 2 ** 30, "qi_sizes": [30] * 5}

    def __init__(self, params: dict = None, init_keys: bool = False):
        try:
            from Pyfhel import Pyfhel, PyCtxt
        except ImportError as e:
            raise ImportError(
                "Pyfhel is required for the 'pyfhel' HE backend; install it "
                "or set [common.he] backend = 'none'") from e
        self._PyCtxt = PyCtxt
        self._he = Pyfhel()
        merged = {**self.DEFAULT_PARAMS, **(params or {})}
        if init_keys:
            self._he.contextGen(**merged)
            self._he.keyGen()

    def encrypt(self, plain: torch.Tensor) -> np.ndarray:
        if isinstance(plain, torch.Tensor):
            plain = plain.detach().cpu().numpy()
        return self._he.encryptMatrix(plain)

    def decrypt(self, cipher: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self._he.decryptMatrix(cipher))

    def to_bytes_context(self) -> bytes:
        return self._he.to_bytes_context()

    def from_bytes_context(self, data: bytes) -> None:
        self._he.from_bytes_context(data)

    def to_bytes_public_key(self) -> bytes:
        return self._he.to_bytes_public_key()

    def from_bytes_public_key(self, data: bytes) -> None:
        self._he.from_bytes_public_key(data)

    def to_bytes_secret_key(self) -> bytes:
        return self._he.to_bytes_secret_key()

    def from_bytes_secret_key(self, data: bytes) -> None:
        self._he.from_bytes_secret_key(data)

    def serialize_ciphertext(self, ct) -> bytes:
        return ct.to_bytes()

    def deserialize_ciphertext(self, data: bytes):
        return self._PyCtxt(pyfhel=self._he, bytestring=data)
