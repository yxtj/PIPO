from src.common.config import get_config

AVAILABLE_PROTOCOL = ['plaintext', 'scale', 'shuffle', 'noise']

PROTOCOL = get_config().common.protocol
assert PROTOCOL in AVAILABLE_PROTOCOL, f"Invalid protocol: {PROTOCOL}. Available protocols: {AVAILABLE_PROTOCOL}"

if PROTOCOL == 'plaintext':
    from .plaintext import *
elif PROTOCOL == 'scale':
    from .scale import *
elif PROTOCOL == 'shuffle':
    from .shuffle import *
elif PROTOCOL == 'noise':
    from .noise import *
