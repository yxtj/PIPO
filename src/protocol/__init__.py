import importlib

AVAILABLE_PROTOCOLS = ['plaintext', 'scale', 'shuffle', 'pad', 'padshuffle',
                       'scaleshuffle', 'padscaleshuffle', 'noise']


def validate_protocol(name: str) -> str:
    """Return ``name`` if it names a known protocol, else raise ValueError."""
    if name not in AVAILABLE_PROTOCOLS:
        raise ValueError(
            "Invalid protocol: {!r}. Available: {}".format(name, ", ".join(AVAILABLE_PROTOCOLS))
        )
    return name


def load_protocol(name: str):
    """(ProtocolServer, ProtocolClient) classes of the named protocol variant."""
    validate_protocol(name)
    mod = importlib.import_module("." + name, __name__)
    return mod.ProtocolServer, mod.ProtocolClient
