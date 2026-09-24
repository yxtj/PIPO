try:  # Pyfhel is optional; only needed when use_he is enabled
    from Pyfhel import Pyfhel
except ImportError:  # pragma: no cover
    Pyfhel = None
import torch
import numpy as np
import encoding
# linear layers
  
def elemt_mul(x,i,j,w):
    result=0
    k=j
    initial_assign=1
    for row in w:
        for elemt in row:
            oprand=elemt*x[i][k]
            if oprand:
                if initial_assign:
                    result=elemt*x[i][k]
                    initial_assign=0
                else:
                    result=result+elemt*x[i][k]
            k=k+1
        k=j
        i=i+1
    return result
def avg(x,i,j,kernel):
    result=0
    initial_assign=1
    for index_i in range(kernel):
        for index_j in range(kernel):
            if initial_assign:
                result=x[i+index_i][j+index_j]
                initial_assign=0
            else:
                result=result+x[i+index_i][j+index_j]
    return result/(kernel*kernel)

def conv(x: np.ndarray, w: torch.Tensor,padding: int=0, stride: int=1) -> np.ndarray:
    w=w.detach().numpy()
    
    if padding:
        padded_x=[]
        for i in range(x.shape[0]+padding*2):
            padded_x.append((x.shape[1]+padding*2)*[0])
        i=padding
        j=padding
        for row in x:
            for elemt in row:
                padded_x[i][j]=elemt
                j=j+1
            j=padding
            i=i+1
    else:
        padded_x=x.tolist()
    i=0
    j=0
    result=[]
    result_index=0
    while i+w.shape[0]<= len(padded_x):
        result.append([])
        while j+w.shape[1]<= len(padded_x[0]):
            result[result_index].append(elemt_mul(padded_x,i,j,w))
            j=j+stride
        result_index=result_index+1
        j=0
        i=i+stride
    return np.array(result)

def fc(x: np.ndarray, w: torch.Tensor) -> np.ndarray:
    w=w.detach().numpy()
    result=np.empty((x.shape[0],w.shape[1]),dtype=object)
    i=0
    j=0
    k=1
    while i<result.shape[0]:
        while j<result.shape[1]:
            result[i][j]=w[0][j]*x[i][0]
            while k<x.shape[1]:
                result[i][j]=result[i][j]+w[k][j]*x[i][k]
                k=k+1
            k=1
            j=j+1
        j=0
        i=i+1
    
    return result

# pooling layers

def avgpool(x: np.ndarray, kernel: int, padding: int=0, stride: int=1) -> np.ndarray:
    if padding:
        padded_x=[]
        for i in range(x.shape[0]+padding*2):
            padded_x.append((x.shape[1]+padding*2)*[0])
        i=padding
        j=padding
        for row in x:
            for elemt in row:
                padded_x[i][j]=elemt
                j=j+1
            j=padding
            i=i+1
    else:
        padded_x=x.tolist()
    i=0
    j=0
    result=[]
    result_index=0
    while i+kernel<= len(padded_x):
        result.append([])
        while j+kernel<= len(padded_x[0]):
            result[result_index].append(avg(padded_x,i,j,kernel))
            j=j+stride
        result_index=result_index+1
        j=0
        i=i+stride
    return np.array(result)



if __name__ == "__main__":
    
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
    x=torch.tensor([
                    [1.0,2,3],
                    [3.0,4,5]
                    ])
    w=torch.tensor([[1,-1,1],[1,-1,1],[1,-1,1]])
    cipher=encoding.encrypt(x,HE)
    # 0 0 0 0 0     1 -1 1
    # 0 1 2 3 0     1 -1 1
    # 0 3 4 5 0     1 -1 1
    # 0 0 0 0 0
    convolution_result=fc(cipher,w)
    output=encoding.decrypt(convolution_result,HE)
    print(output)

