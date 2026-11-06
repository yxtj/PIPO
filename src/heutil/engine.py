"""HE engine interface and backend factory.

`HEEngine` is the interface used by the layer / protocol / comm modules; the
backends live in sibling modules (`engine_none.NoHEEngine`,
`engine_pyfhel.PyfhelEngine`) and `create_he` picks one from the effective
config (or explicit arguments), so new backends (e.g. OpenFHE) plug in
without touching callers.

Type conventions:
- plaintext  : torch.Tensor (the online/offline data flow is torch-based)
- ciphertext : np.ndarray with dtype=object holding backend cipher objects
  (Pyfhel PyCtxt); NoHEEngine keeps the torch tensor as-is instead.

Wire format of `serialize_array` / `deserialize_array` (all backends):

    !I ndim | ndim * !Q shape | !B dtype_code | !I elem_len | payload

dtype_code: 0 = backend ciphertext objects, 1 = float32, 2 = float64.
payload holds prod(shape) fixed-size elements; ciphertexts of one array are
equal-sized under a fixed CKKS context, which keeps the format identical for
all backends.
"""

import struct

import numpy as np
import torch

_DTYPE_OBJECT = 0
_DTYPE_FLOAT32 = 1
_DTYPE_FLOAT64 = 2


class HEEngine:
    """Interface shared by every HE backend."""

    enabled: bool = False

    # encryption

    def encrypt(self, plain: torch.Tensor):
        """torch.Tensor -> ciphertext container (backend specific)."""
        raise NotImplementedError

    def decrypt(self, cipher) -> torch.Tensor:
        """Ciphertext container -> torch.Tensor."""
        raise NotImplementedError

    # context / key serialization

    def to_bytes_context(self) -> bytes:
        raise NotImplementedError

    def from_bytes_context(self, data: bytes) -> None:
        raise NotImplementedError

    def to_bytes_public_key(self) -> bytes:
        raise NotImplementedError

    def from_bytes_public_key(self, data: bytes) -> None:
        raise NotImplementedError

    def to_bytes_secret_key(self) -> bytes:
        raise NotImplementedError

    def from_bytes_secret_key(self, data: bytes) -> None:
        raise NotImplementedError

    # ciphertext serialization

    def serialize_ciphertext(self, ct) -> bytes:
        raise NotImplementedError

    def deserialize_ciphertext(self, data: bytes):
        raise NotImplementedError

    # array-level packing of ciphertext containers

    def serialize_array(self, arr) -> bytes:
        """Pack a ciphertext object array into one self-describing blob."""
        arr = np.asarray(arr, dtype=object)
        flat = arr.reshape(-1)
        elems = [self.serialize_ciphertext(e) for e in flat]
        elem_len = len(elems[0]) if elems else 0
        header = self._pack_header_(arr.shape, _DTYPE_OBJECT, elem_len)
        return header + b"".join(elems)

    def deserialize_array(self, data: bytes) -> np.ndarray:
        """Inverse of serialize_array; returns a ciphertext object array."""
        shape, dtype_code, elem_len, offset = self._unpack_header_(data)
        assert dtype_code == _DTYPE_OBJECT
        arr = np.empty(shape, dtype=object)
        flat = arr.reshape(-1)
        for i in range(flat.size):
            flat[i] = self.deserialize_ciphertext(data[offset:offset + elem_len])
            offset += elem_len
        return arr

    # header helpers shared by all backends

    @staticmethod
    def _pack_header_(shape: tuple, dtype_code: int, elem_len: int) -> bytes:
        return (struct.pack("!I", len(shape))
                + struct.pack("!" + "Q" * len(shape), *shape)
                + struct.pack("!BI", dtype_code, elem_len))

    @staticmethod
    def _unpack_header_(data: bytes):
        ndim, = struct.unpack("!I", data[:4])
        offset = 4
        shape = struct.unpack("!" + "Q" * ndim, data[offset:offset + 8 * ndim])
        offset += 8 * ndim
        dtype_code, elem_len = struct.unpack("!BI", data[offset:offset + 5])
        offset += 5
        return shape, dtype_code, elem_len, offset


def create_he(backend: str = None, enabled: bool = None,
              params: dict = None, init_keys: bool = False) -> HEEngine:
    """Build the HE engine from explicit arguments or the effective config.

    enabled defaults to common.use_he; backend defaults to common.he.backend.
    enabled=False always yields NoHEEngine; backend='none' does the same even
    when enabled (use_he=True + backend='none' is treated as no encryption).
    """
    from src.common.config import get_config

    cfg = get_config()
    if enabled is None:
        enabled = cfg.common.use_he
    if backend is None:
        backend = cfg.common.he.backend
    if not enabled or backend == "none":
        from .engine_none import NoHEEngine
        return NoHEEngine()
    if backend == "pyfhel":
        from .engine_pyfhel import PyfhelEngine
        if params is None:
            p = cfg.common.he.pyfhel
            params = {"n": p.n, "scale": p.scale, "qi_sizes": list(p.qi_sizes)}
        return PyfhelEngine(params=params, init_keys=init_keys)
    raise ValueError("unknown HE backend: {!r} (expected 'none' or 'pyfhel')".format(backend))
