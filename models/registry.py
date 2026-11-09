"""Model registry: model name -> (inshape, builder).

Aggregates every model definition under ``models`` behind one name space.
``load(name, ...)`` builds a fresh model instance; ``torch.manual_seed(seed)``
is called first so that the client and the server can share identical random
weights for randomly-initialized  Pretrained weights can be applied
afterwards through ``weights`` (a state dict file), falling back to per-model
defaults when ``wfile`` is empty.
"""

import torch

from pathlib import Path

from . import minionn, openpose, poc, resnet, resnet_cifar, vgg


# Protocols that keep a multiplicative mask on the client side; activations
# that are not positive-homogeneous cannot be evaluated under them.
MULTIPLICATIVE_PROTOCOLS = frozenset({"scale", "scaleshuffle", "padscaleshuffle", "noise"})

# Fully-qualified model names excluded from those protocols by default.
SCALE_UNSUPPORTED = frozenset("poc-" + key for key in poc.UNSUPPORTED_SCALE)


def unsupported_models(protocol: str) -> frozenset:
    """Model names to skip by default when running ``protocol``."""
    if protocol in MULTIPLICATIVE_PROTOCOLS:
        return SCALE_UNSUPPORTED
    return frozenset()


def _uniform_init(model, low: float = -1.0, high: float = 1.0):
    for p in model.parameters():
        p.data.uniform_(low, high)
    return model


def _builders() -> dict:
    b = {}
    for key, (inshape, model) in poc.map.items():
        # poc models are stored as plain instances; re-randomize at load time
        # (use_random_weights) so the client and the server share the seed.
        b[f"poc-{key}"] = (inshape, lambda md=model: _uniform_init(md))
    b["minionn"] = (minionn.inshape, lambda: minionn.build())
    for depth in (11, 13, 16, 19):
        b[f"vgg-{depth}"] = (vgg.inshape, lambda d=depth: vgg.build_vgg(d))
    for depth in (18, 34, 50, 101, 152):
        b[f"resnet-{depth}"] = (resnet.inshape, lambda d=depth: resnet.build(d))
    for depth in (20, 32, 44, 56, 110, 152):
        b[f"resnet-cifar-{depth}"] = (
            resnet_cifar.inshape,
            lambda d=depth: resnet_cifar.build(d),
        )
    b["openpose-body"] = (
        openpose.inshape,
        lambda wfile=None: openpose.build_body_model(wfile),
    )
    b["openpose-hand"] = (
        openpose.inshape,
        lambda wfile=None: openpose.build_hand_model(wfile),
    )
    return b


def resolve_weight_file(name: str, wfile: str | None = None, wfolder: str | None = None):
    """Resolve the weights file for ``name``.

    An explicit ``wfile`` wins; otherwise ``wfolder`` is searched for
    ``<name>.pt`` and ``<name>.pth``. Returns ``None`` when none exists.
    """
    if wfile:
        return wfile
    if wfolder:
        for suffix in (".pt", ".pth"):
            path = Path(wfolder) / "{}{}".format(name, suffix)
            if path.is_file():
                return str(path)
    return None


def load(name: str, seed=None, wfile=None, wfolder="pretrained", device="cpu"):
    """Build the model registered under ``name``.

    ``seed`` seeds torch before building so random weights match between the
    client and the server. ``wfile`` (a state-dict file) is applied either by
    the builder itself (openpose) or post-hoc via ``load_state_dict``. When
    ``wfile`` is empty, ``<wfolder>/<name>.pt|.pth`` is used if present.
    """
    builders = _builders()
    if name not in builders:
        raise KeyError(f"Unknown model: {name}. Available models: {sorted(builders)}")
    if seed is not None:
        torch.manual_seed(seed)
    wfile = resolve_weight_file(name, wfile, wfolder)
    inshape, builder = builders[name]
    if name.startswith("openpose-"):
        # openpose weights are loaded into the original op_impl model during
        # construction (the checkpoint keys name those modules, not the DAG)
        model = builder(wfile)
    else:
        model = builder()
        if wfile is not None:
            model.load_state_dict(torch.load(wfile, map_location="cpu"))
    model.eval()
    model = model.to(device)
    return inshape, model
