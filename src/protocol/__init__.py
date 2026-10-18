from src.common.config import get_config

AVAILABLE_PROTOCOLS = ['plaintext', 'scale', 'shuffle', 'scaleshuffle', 'noise']

def validate_protocol(name: str) -> str:
    """Return ``name`` if it names a known protocol, else raise ValueError."""
    if name not in _MODULES:
        raise ValueError(
            "Invalid protocol: {!r}. Available: {}".format(name, ", ".join(AVAILABLE_PROTOCOLS))
        )
    return name

PROTOCOL = get_config().common.protocol
validate_protocol(PROTOCOL)

if PROTOCOL == 'plaintext':
    from .plaintext import *
elif PROTOCOL == 'scale':
    from .scale import *
elif PROTOCOL == 'shuffle':
    from .shuffle import *
elif PROTOCOL == 'scaleshuffle':
    from .scaleshuffle import *
elif PROTOCOL == 'noise':
    from .noise import *
