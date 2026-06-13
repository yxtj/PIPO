import os

__all__ = ['USE_HE', 'PROTOCOL']

# whether to use HE in the offline phase
USE_HE = os.environ.get('PIPO_USE_HE', '0').lower() in ['1', 'true', 't', 'yes', 'y']

print("USE_HE:", USE_HE)

# which protocol to use
PROTOCOL = os.environ.get('PIPO_PROTOCOL', 'scale')
# protocols: plaintext, scale, shuffle, noise
print("PROTOCOL:", PROTOCOL)
