from Pyfhel import Pyfhel
import torch
import numpy as np

def encrypt(plain: torch.Tensor, he: Pyfhel) -> np.ndarray:
    assert isinstance(plain, torch.Tensor)
    assert isinstance(he, Pyfhel)
    plain = plain.detach().numpy()
    plain = he.encryptMatrix(plain)
    return plain


def decrypt(cipher: np.ndarray, he: Pyfhel) -> torch.Tensor:
    assert isinstance(cipher, np.ndarray)
    assert isinstance(he, Pyfhel)
    cipher = he.decryptMatrix(cipher)
    cipher = torch.from_numpy(cipher)
    return cipher
