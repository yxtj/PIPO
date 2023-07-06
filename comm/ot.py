from .rsa import RSA
from socket import socket
import struct
import Crypto.Util.strxor
from .basic import send_chunk, recv_chunk

'''
1-of-2 Oblivious Transfer Protocol.

          Sender                                Receiver
have data: m0, m1                have: b (0/1) indicating which data to require

# RSA setup phase (only once)
generate RSK key (n, e, d)      |
send public key: n, e           =>          receive: n, e

# indicator preparation phase
generate random byte: k0, k1    =>          receive: k0, k1
                                |   generate random mask: h of length m
                                |   compute: c = rsa.encrypt(h) XOR k_b
receive c                       <=          send c
h0 = rsa.decrypt(c XOR k0)      |
h1 = rsa.decrypt(c XOR k1)      |

# data transfer phase
send: m0' = m0 XOR h0           =>          receive: m0'
      m1' = m1 XOR h1           =>          receive: m1'
                                |           pick m_b'
                                |           m_b = m_b' XOR h
'''

'''
A general 1-of-2 OT protocol using an indicator function.

          Sender                                Receiver
have data: m0, m1                       have: indicator i
have data function: f(m0, m1, i)      have: selector function: b = f(i)

# RSA setup phase (only once)
generate RSK key (n, e, d)      |
send public key: n, e           =>          receive: n, e

# indicator preparation phase
generate random byte: k0, k1    =>          receive: k0, k1
                                |   generate random mask: h of length m
                                |   compute: c = rsa.encrypt(h) XOR k_b
                                |   compute: i' = i XOR h
receive c, i'                   <=          send c and i'
h0 = rsa.decrypt(c XOR k0)      |
h1 = rsa.decrypt(c XOR k1)      |

# data transfer phase
indicator: i0 = i' XOR h0      |
            i1 = i' XOR h1      |
data: r0 = f(m0, m1, i0)        |
      r1 = f(m0, m1, i1)        |
send: r0' = r0 XOR h0           =>          receive: r0'
      r1' = r1 XOR h1           =>          receive: r1'
                                |           pick r_b'
                                |           r_b = r_b' XOR h
'''

def xor_mask_char(data_bytes:bytes, mask:int) -> bytes:
    assert 0<= mask <= 255
    return Crypto.Util.strxor.strxor_c(data_bytes, mask)

def xor_mask_bytes(data_bytes:bytes, mask_bytes:bytes) -> bytes:
    return Crypto.Util.strxor.strxor(data_bytes, mask_bytes)

def mask_data(data_bytes:bytes, mask:bytes) -> bytes:
    nbyte = len(mask)
    n = len(data_bytes)
    m, r = divmod(n, nbyte)
    buffer = [xor_mask_bytes(data_bytes[i*nbyte:(i+1)*nbyte], mask) for i in range(m)]
    if r > 0:
        buffer.append(xor_mask_bytes(data_bytes[-r:], mask[:r]))
    return b''.join(buffer)


class ObliviousTransferSender():
    def __init__(self, socket:socket, nbits:int=2048) -> None:
        self.socket = socket
        self.nbits = nbits
        self.nbyte = nbits//8 + (1 if nbits % 8 > 0 else 0)
        self.rsa = RSA(nbits)
        self.mask_len = self.nbyte - 1
        
    def setup(self):
        self.rsa.setup()
        # print(f'Client: n={self.rsa.n}\nd ={self.rsa.d}')
        self.socket.sendall(self.rsa.n.to_bytes(self.nbyte, 'big'))
        self.socket.sendall(self.rsa.e.to_bytes(self.nbyte, 'big'))
    
    def run(self, x0:bytes, x1:bytes) -> tuple[int, int]:
        '''
        Oblivious transfer for sending data x0 or x1.
        Return the number of bytes sent and received.
        '''
        # indicator preparation phase
        k0, k1 = self.random_pair()
        data = struct.pack('!BB', k0, k1)
        self.socket.sendall(data)
        c = self.socket.recv(self.nbyte)
        # print("Client c:", c)
        h0 = self.decrypt_with_mask(c, k0)
        # print("Client h0:", h0)
        h1 = self.decrypt_with_mask(c, k1)
        # print("Client h1:", h1)
        # data transfer phase
        cnt0 = self.send_with_mask(x0, h0)
        cnt1 = self.send_with_mask(x1, h1)
        return 2 + cnt0 + cnt1, self.nbyte
    
    def run_customized(self, x0, x1, fun_data:callable):
        '''
        OT with customized data generation function.
        fun(x0, x1, idtf) -> data in bytes
        '''
        # send keys
        k0, k1 = self.random_pair()
        data = struct.pack('!BB', k0, k1)
        self.socket.sendall(data)
        # receive indicator
        c = self.socket.recv(self.nbyte) # encrypted mask
        id_mask, n_recv = recv_chunk(self.socket) # masked indicator
        # print("Client idmask:", id_mask.hex())
        h0 = self.decrypt_with_mask(c, k0)
        # print("Client h0:", h0.hex())
        h1 = self.decrypt_with_mask(c, k1)
        # print("Client h1:", h1.hex())
        id0 = mask_data(id_mask, h0)
        # print("Client id0:", id0.hex())
        id1 = mask_data(id_mask, h1)
        # print("Client id1:", id1.hex())
        # send data
        msg0 = fun_data(x0, x1, id0)
        msg1 = fun_data(x0, x1, id1)
        cnt0 = self.send_with_mask(msg0, h0)
        cnt1 = self.send_with_mask(msg1, h1)
        return 2 + cnt0 + cnt1, self.nbyte + n_recv
    
    # local functions
    
    def random_pair(self) -> tuple[int, int]:
        mask1 = Crypto.Random.random.randint(0, 255)
        mask2 = Crypto.Random.random.randint(0, 255)
        return mask1, mask2

    def decrypt_with_mask(self, data_bytes:bytes, mask:int) -> bytes:
        d = xor_mask_char(data_bytes, mask)
        d = self.rsa.decrypt(d)
        return d
    
    def send_with_mask(self, data_bytes:bytes, mask:bytes) -> int:
        # assert len(mask) == self.nbyte
        n = len(data_bytes)
        self.socket.sendall(struct.pack('!i', n))
        if len(mask) > self.mask_len:
            mask = mask[-self.mask_len:]
        data = mask_data(data_bytes, mask)
        self.socket.sendall(data)
        return 4 + n
    
    def receive_indicator(self) -> tuple[bytes, bytes, int]:
        c = self.socket.recv(self.nbyte)
        indicator, n = recv_chunk(self.socket)
        return c, indicator, self.nbyte + n

class ObliviousTransferReceiver():
    def __init__(self, socket:socket, nbits:int=2048) -> None:
        self.socket = socket
        self.nbits = nbits
        self.nbyte = nbits//8 + (1 if nbits % 8 > 0 else 0)
        self.rsa = RSA(nbits)
        self.mask_len = self.nbyte - 1

    def setup(self):
        data = self.socket.recv(self.nbyte)
        n = int.from_bytes(data, 'big')
        data = self.socket.recv(self.nbyte)
        e = int.from_bytes(data, 'big')
        self.rsa.setup(n, e)
        self.mask_len = self.rsa.mlength
    
    def run(self, b:int) -> tuple[bytes, int, int]:
        '''
        Oblivious transfer for receiving data with index b.
        Return the data, the number of bytes sent and received.
        '''
        assert b == 0 or b == 1
        # indicator preparation phase
        h = self.random_mask()
        k0, k1 = self.receive_pair()
        # print("Server: h =", h[:])
        c = self.rsa.encrypt(h)
        # print("Server: c =", c[:])
        if b == 0:
            c = xor_mask_char(c, k0)
        else:
            c = xor_mask_char(c, k1)
        # print("Server: c xor k_b =", c[:])
        self.socket.sendall(c)
        # data transfer phase
        m0, cnt0 = recv_chunk(self.socket)
        # print("Server: m XOR h =", m0[:32])
        m1, cnt1 = recv_chunk(self.socket)
        if b == 0:
            m = mask_data(m0, h)
        else:
            m = mask_data(m1, h)
        # print("Server: m =", m[:32])
        return m, self.nbyte, 2 + cnt0 + cnt1
    
    def run_customized(self, indicator:bytes, fun_select:callable=None) -> tuple[bytes, int, int]:
        if fun_select is None:
            b = sum(indicator[:32]) % 2
        else:
            b = fun_select(indicator)
        # preparing keys
        k0, k1 = self.receive_pair()
        h = self.random_mask()
        # print("Server h :", h.hex())
        c = self.rsa.encrypt(h)
        if b == 0:
            c = xor_mask_char(c, k0)
        else:
            c = xor_mask_char(c, k1)
        self.socket.sendall(c)
        # send indicator
        data = mask_data(indicator, h)
        # print("Server idmask:", data.hex())
        n_send = send_chunk(self.socket, data)
        # receive data
        m0, cnt0 = recv_chunk(self.socket)
        m1, cnt1 = recv_chunk(self.socket)
        if b == 0:
            m = mask_data(m0, h)
        else:
            m = mask_data(m1, h)
        return m, len(c) + n_send, 2 + cnt0 + cnt1
    
    # local functions
    
    def receive_pair(self) -> tuple[int, int]:
        data = self.socket.recv(2)
        p1, p2 = struct.unpack('!BB', data)
        return p1, p2
    
    def random_mask(self) -> bytes:
        return Crypto.Random.get_random_bytes(self.mask_len)
        # return Crypto.Random.get_random_bytes(self.nbyte)
