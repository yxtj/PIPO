import numpy as np
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP
import binascii

import Crypto.Util
import Crypto.Util.Padding
import Crypto.Random


def rsa_test():
    # RSA Key Generation
    keyPair = RSA.generate(1024)

    pubKey = keyPair.publickey()
    print(f"Public key:  (n={hex(pubKey.n)}, e={hex(pubKey.e)})")
    # pubKeyPEM = pubKey.exportKey()
    # print(pubKeyPEM.decode('ascii'))

    print(f"Private key: (n={hex(pubKey.n)}, d={hex(keyPair.d)})")
    # privKeyPEM = keyPair.exportKey()
    # print(privKeyPEM.decode('ascii'))

    # RSA Encryption
    msg = b'A message for encryption'
    encryptor = PKCS1_OAEP.new(pubKey)
    encrypted = encryptor.encrypt(msg)
    tmp = binascii.hexlify(encrypted)
    print("Encrypted:", tmp)
    print("Message length:", len(msg), "Encrypted message length:", len(encrypted), "Encrypted message length (hex):", len(tmp))

    # RSA Decryption
    decryptor = PKCS1_OAEP.new(keyPair)
    decrypted = decryptor.decrypt(encrypted)
    print('Decrypted:', decrypted)


def mask_rsa_crypto():
    key_bits = 1024
    keyPair = RSA.generate(key_bits)
    pubKey = keyPair.publickey()
    
    encryptor = PKCS1_OAEP.new(pubKey)
    decryptor = PKCS1_OAEP.new(keyPair)
    
    msg = b'A message for encryption'
    mask_byte = Crypto.Random.random.randint(0, 255) # [0, 255]
    mask_str = Crypto.Random.get_random_bytes(key_bits//8) # 128 bytes
    print("mask_byte:", mask_byte, "mask_str:", mask_str)

    c = encryptor.encrypt(msg)
    mc1 = Crypto.Util.strxor.strxor_c(c, mask_byte)
    mc2 = Crypto.Util.strxor.strxor(c, mask_str)
    print("masked by byte:", mc1)
    print("masked by str:", mc2)
    
    try:
        dc1 = decryptor.decrypt(mc1) # error
        print("decrypted by byte:", dc1)
    except:
        print("error")
    try:
        dc2 = decryptor.decrypt(mc2) # error
        print("decrypted by str:", dc2)
    except:
        print("error")
    
    print("The RSA module provided by `PyCryptodome` does not support masking. It detects whether a ciphertext is valid or not by checking the padding.")
    print("Similarly, module provided by the `cryptography` and `rsa` package do not support masking either.")
    
    
def rsa_my():
    key_bits = 1024
    keyPair = RSA.generate(key_bits)
    pubKey = keyPair.publickey()
    
    n = keyPair.n
    d = keyPair.d
    e = keyPair.e
    
    print("key pair:")
    print("log2(n)={:.2f}, log2(d)={:.2f}, log2(e)={:.2f}".format(
          np.log2(float(n)), np.log2(float(d)), np.log2(float(e))))
    print("n", keyPair.n)
    print("d", keyPair.d)
    print("e", keyPair.e)
    
    print("public key:")
    print("n", pubKey.n)
    print("e", pubKey.e)
    
    m = b'A message for encryption'
    print("message length", len(m))
    print("message:", m)
    
    # encode
    plain = Crypto.Util.number.bytes_to_long(m)
    print("plain length", np.log2(float(plain)))
    print("plain:", plain)
    
    # encrypt
    cipher = pow(plain, e, n)
    print("cipher length", np.log2(float(cipher)))
    print("cipher:", cipher)
    
    # decrypt
    dec = pow(cipher, d, n)
    print("dec length", np.log2(float(dec)))
    print("dec:", dec)
    
    # decode
    dmsg = Crypto.Util.number.long_to_bytes(dec)
    print("dmsg length", len(dmsg))
    print("dmsg:", dmsg)
    

def mask_rsa_my():
    key_bits = 1024
    keyPair = RSA.generate(key_bits)
    
    n = keyPair.n
    d = keyPair.d
    e = keyPair.e
    
    def encrypt(data_bytes, n, e):
        data_n = Crypto.Util.number.bytes_to_long(data_bytes)
        c = pow(data_n, e, n)
        mc = Crypto.Util.number.long_to_bytes(c, key_bits//8)
        return c, mc
    
    def decrypt(data_bytes, n, d):
        data_n = Crypto.Util.number.bytes_to_long(data_bytes)
        r = pow(data_n, d, n)
        mr = Crypto.Util.number.long_to_bytes(r)
        return r, mr
    
    m = b'A message for encryption'
    plain = Crypto.Util.number.bytes_to_long(m)
    cipher = pow(plain, e, n)
    dec = pow(cipher, d, n)
    
    print("plain:", np.log2(float(plain)), plain)
    print("cipher:", np.log2(float(cipher)), cipher)
    print("decrypted:", np.log2(float(dec)), dec)
    print("result:", Crypto.Util.number.long_to_bytes(dec))
    
    cipher, cipher_bytes = encrypt(m, n, e)
    
    # mask the ciphertext
    mask_byte = Crypto.Random.random.randint(0, 255) # [0, 255]
    mask_str = Crypto.Random.get_random_bytes(key_bits//8) # 128 bytes
    
    print("cipher_bytes_pad:", len(cipher_bytes), cipher_bytes)
    
    print("work with masked ciphertext")
    mc1 = Crypto.Util.strxor.strxor_c(cipher_bytes, mask_byte)
    print(mc1 == cipher_bytes)
    mc2 = Crypto.Util.strxor.strxor(cipher_bytes, mask_str)
    print(mc2 == cipher_bytes)
    
    r1, mr1 = decrypt(mc1, n, d)
    r2, mr2 = decrypt(mc2, n, d)
    
    print("decrypted by byte:", np.log2(float(r1)))
    print(r1)
    print(mr1)
    print("decrypted by str:", np.log2(float(r2)))
    print(r2)
    print(mr2)
    
    # mask back
    print("mask back")
    mc1p = Crypto.Util.strxor.strxor_c(mc1, mask_byte)
    print(mc1p == cipher_bytes)
    mc2p = Crypto.Util.strxor.strxor(mc2, mask_str)
    print(mc2p == cipher_bytes)
    # mc1p = Crypto.Util.Padding.unpad(mc1p, key_bits//8)
    # mc2p = Crypto.Util.Padding.unpad(mc2p, key_bits//8)
    r1, mr1 = decrypt(mc1p, n, d)
    r2, mr2 = decrypt(mc2p, n, d)
    print("decrypted by byte:", np.log2(float(r1)))
    print(r1)
    print(mr1)
    print("decrypted by str:", np.log2(float(r2)))
    print(r2)
    print(mr2)
    

if __name__ == "__main__":
    # rsa_crypto()
    # mask_rsa_crypto()
    # rsa_my()
    mask_rsa_my()
