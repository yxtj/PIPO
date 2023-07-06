import Crypto.PublicKey.RSA
import Crypto.Util
import numpy as np
import struct


class RSA():
    def __init__(self, nbits:int=2048, n=None, e=None) -> None:
        self.nbits = nbits
        self.nbyte = nbits//8
        self.n = None
        self.e = None
        self.d = None
        self.mlength = self.nbyte - 1 # max length of message, may be updated upon setup
    
    def setup(self, n:int=None, e:int=None):
        '''
        Setup the RSA key pair or public key.
        If n and e are given, then use them as the public key.
        Otherwise, generate a key pair (n, e) and (n, d).
        '''
        if n is not None and e is not None:
            self.n = n
            self.e = e
        else:
            key_pair = Crypto.PublicKey.RSA.generate(self.nbits, np.random.bytes)
            self.n = key_pair.n
            self.d = key_pair.d
            self.e = key_pair.e
        self._set_max_message_length_()
    
    def can_encrypt(self, data_bytes:bytes) -> bool:
        return len(data_bytes) <= self.mlength
    
    def encrypt(self, data_bytes:bytes) -> bytes:
        assert len(data_bytes) <= self.mlength
        data_n = Crypto.Util.number.bytes_to_long(data_bytes)
        c = pow(data_n, self.e, self.n)
        mc = Crypto.Util.number.long_to_bytes(c, self.nbyte)
        return mc
    
    def decrypt(self, data_bytes:bytes) -> bytes:
        assert len(data_bytes) <= self.nbyte
        data_n = Crypto.Util.number.bytes_to_long(data_bytes)
        r = pow(data_n, self.d, self.n)
        # mr = Crypto.Util.number.long_to_bytes(r, self.mlength) # this is wrong for large ciphertext
        mr = Crypto.Util.number.long_to_bytes(r, self.nbyte)
        mr = mr[-self.mlength:] # remove the leading byte
        return mr
    
    def encrypt_big(self, data_types:bytes) -> bytes:
        ld = len(data_types)
        m, r = divmod(ld, self.mlength)
        # the last 2 bytes are used to store the length of the last message
        buffer = [ self.encrypt(data_types[i*self.mlength:(i+1)*self.mlength]) for i in range(m) ]
        if r == 0:
            buffer.append(self.encrypt(struct.pack('!h', 0)))
        elif r <= self.mlength - 2:
            buffer.append(self.encrypt(data_types[-r:] + struct.pack('!h', r)))
        else:
            buffer.append(self.encrypt(data_types[-r:]))
            buffer.append(struct.pack('!h', r - self.mlength))
        return b''.join(buffer)
    
    def decrypt_big(self, data_types:bytes) -> bytes:
        ld = len(data_types)
        m, r = divmod(ld, self.nbyte)
        assert r == 0
        buffer = [ self.decrypt(data_types[i*self.nbyte:(i+1)*self.nbyte]) for i in range(m) ]
        r = struct.unpack('!h', buffer[-1][-2:])[0]
        print(m, r)
        if r == 0:
            buffer.pop()
        elif r > 0:
            print(buffer[-1][-r-4:])
            buffer[-1] = buffer[-1][-r-2:-2]
        elif r < 0:
            buffer.pop()
            buffer[-1] = buffer[-1][:r]
        return b''.join(buffer)
    
    def _set_max_message_length_(self):
        assert self.n is not None
        bytes_n = Crypto.Util.number.long_to_bytes(self.n, self.nbyte)
        for i in range(self.nbyte):
            if bytes_n[i] != 0:
                break
        self.mlength = self.nbyte - i - 1
        
        