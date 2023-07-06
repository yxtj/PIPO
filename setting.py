import os

__all__ = ['USE_HE', 'PROTOCOL']

# whether to use HE in the offline phase
USE_HE = os.environ.get('FASENET_USE_HE', '0').lower() in ['1', 'true', 't', 'yes', 'y']

print("USE_HE:", USE_HE)

# which protocol to use
PROTOCOL = os.environ.get('FASENET_PROTOCOL', 'scale')
# protocols: scale, dual, shuffle, pad
# scale and dual are mutually exclusive
# dual is scale + oblivious transfer (OT)
# shuffle is scale + shuffle
# pad is scale + shuffle + pad
print("PROTOCOL:", PROTOCOL)
