from Pyfhel import Pyfhel
import torch
import numpy as np

def encrypt(plain: torch.Tensor, he: Pyfhel) -> np.ndarray:
    assert isinstance(plain, torch.Tensor)
    assert isinstance(he, Pyfhel)
    plain = plain.detach().numpy()
    cipher=[]
    i=0
    for row in plain: 
        cipher.append([])
        for elemt in row:
            print(type(elemt))
            cipher[i].append(he.encryptFrac(np.array([elemt],np.float64)))
        i=i+1
            
    return np.array(cipher)


def decrypt(cipher: np.ndarray, he: Pyfhel) -> torch.Tensor:
    assert isinstance(cipher, np.ndarray)
    assert isinstance(he, Pyfhel)
    plain = np.zeros(cipher.shape,np.float64)
    i=0
    j=0
    for row in cipher: 
        for elemt in row:
            plain[i][j] = he.decryptFrac(elemt)[0]
            j=j+1
        i=i+1
        j=0
    plain = torch.from_numpy(plain)
    return plain

if __name__ == "__main__":
    input=torch.tensor([[1.0,2,3],[3.0,4,5]])
    HE = Pyfhel()           # Creating empty Pyfhel object
    ckks_params = {
    'scheme': 'CKKS',   # can also be 'ckks'
    'n': 2**14,         # Polynomial modulus degree. For CKKS, n/2 values can be
                        #  encoded in a single ciphertext.
                        #  Typ. 2^D for D in [10, 15]
    'scale': 2**30,     # All the encodings will use it for float->fixed point
                        #  conversion: x_fix = round(x_float * scale)
                        #  You can use this as default scale or use a different
                        #  scale on each operation (set in HE.encryptFrac)
    'qi_sizes': [60, 30, 30, 30, 60] # Number of bits of each prime in the chain.
                        # Intermediate values should be  close to log2(scale)
                        # for each operation, to have small rounding errors.
    }
    HE.contextGen(**ckks_params)
    HE.keyGen()             # Key Generation: generates a pair of public/secret keys
    output=encrypt(input,HE)
    output=decrypt(output,HE)
    print(output)