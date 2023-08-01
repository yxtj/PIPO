import setting

AVAILABLE_PROTOCOL = ['plaintext', 'scale', 'dual', 'shuffle', 'pad']

assert setting.PROTOCOL in AVAILABLE_PROTOCOL, f"Invalid protocol: {setting.PROTOCOL}. Available protocols: {AVAILABLE_PROTOCOL}"

if setting.PROTOCOL == 'plaintext':
    from .plaintext import *
elif setting.PROTOCOL == 'scale':
    from .scale import *
elif setting.PROTOCOL == 'dual':
    from .dual import *
elif setting.PROTOCOL == 'shuffle':
    from .shuffle import *
elif setting.PROTOCOL == 'pad':
    from .pad import *
