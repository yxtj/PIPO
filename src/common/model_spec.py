"""Protocol-agnostic client-side layer specification derived from a ModelInfo.

The server owns model parsing; the client receives a ``ModelInfo`` and turns it
into :class:`ClientLayerSpec` entries here, without needing any local model
instance. ``src.system.util`` maps these specs to concrete layer classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from src.common.types import ModelInfo


#: Execution kind of each op, mirroring the old ``isinstance`` dispatch in
#: ``system.util``: local layers run on the client, linear/remote layers on the
#: server, pooling has its own sub-protocol, and shortcuts merge feature maps.
OP_KIND = {
    "conv2d": "linear",
    "linear": "linear",
    "batchnorm1d": "linear",
    "batchnorm2d": "linear",
    "avgpool2d": "linear",
    "identity": "linear",
    "relu": "local",
    "sigmoid": "local",
    "leakyrelu": "local",
    "gelu": "local",
    "flatten": "local",
    "softmax": "local",
    "maxpool2d": "pool",
    "jumpop": "shortcut",
    "addop": "shortcut",
    "concatop": "shortcut",
}

#: Ops whose client layer needs a real nn.Module rebuilt from ``attrs``.
_MODULE_OPS = frozenset({
    "sigmoid", "leakyrelu", "gelu", "maxpool2d", "avgpool2d", "flatten", "softmax",
})


@dataclass
class ClientLayerSpec:
    """One client layer, fully described by server-side model info."""

    index: int
    op: str
    kind: str
    ishape: tuple[int, ...]
    oshape: tuple[int, ...]
    sources: list[int] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)


def build_client_specs(info: ModelInfo) -> list[ClientLayerSpec]:
    """Validate a ModelInfo and expand it into per-layer client specs."""
    if len(info.shapes) != len(info.steps) + 1:
        raise ValueError(
            "invalid ModelInfo: {} shapes for {} steps (expected one more shape)".format(
                len(info.shapes), len(info.steps))
        )
    specs: list[ClientLayerSpec] = []
    for i, step in enumerate(info.steps):
        kind = OP_KIND.get(step.op)
        if kind is None:
            raise ValueError(
                "unknown layer type {!r} at layer {}; supported: {}".format(
                    step.op, i, ", ".join(sorted(OP_KIND)))
            )
        ishape = tuple(step.ishape) if step.ishape is not None else tuple(info.shapes[i])
        oshape = tuple(step.oshape) if step.oshape is not None else tuple(info.shapes[i + 1])
        if tuple(info.shapes[i]) != ishape or tuple(info.shapes[i + 1]) != oshape:
            raise ValueError(
                "inconsistent shapes at layer {} ({}): step {}->{} vs ModelInfo {}->{}".format(
                    i, step.op, ishape, oshape, tuple(info.shapes[i]), tuple(info.shapes[i + 1]))
            )
        specs.append(ClientLayerSpec(
            index=i,
            op=step.op,
            kind=kind,
            ishape=ishape,
            oshape=oshape,
            sources=list(step.sources),
            attrs=dict(step.attrs or {}),
        ))
    return specs


def build_client_module(spec: ClientLayerSpec) -> nn.Module | None:
    """Rebuild the lightweight module a client layer needs, if any.

    These modules carry no server weights; they only provide the forward
    behavior of local layers (activations, pooling on masked data, reshape).
    """
    attrs = spec.attrs
    if spec.op == "sigmoid":
        return nn.Sigmoid()
    if spec.op == "leakyrelu":
        return nn.LeakyReLU(float(attrs.get("negative_slope", 0.01)))
    if spec.op == "gelu":
        return nn.GELU(approximate=attrs.get("approximate", "none"))
    if spec.op == "maxpool2d":
        return nn.MaxPool2d(
            kernel_size=attrs.get("kernel_size", 2),
            stride=attrs.get("stride"),
            padding=attrs.get("padding", 0),
            dilation=attrs.get("dilation", 1),
            return_indices=bool(attrs.get("return_indices", False)),
            ceil_mode=bool(attrs.get("ceil_mode", False)),
        )
    if spec.op == "avgpool2d":
        return nn.AvgPool2d(
            kernel_size=attrs.get("kernel_size", 2),
            stride=attrs.get("stride"),
            padding=attrs.get("padding", 0),
            ceil_mode=bool(attrs.get("ceil_mode", False)),
            count_include_pad=bool(attrs.get("count_include_pad", True)),
            divisor_override=attrs.get("divisor_override"),
        )
    if spec.op == "flatten":
        return nn.Flatten(
            start_dim=int(attrs.get("start_dim", 1)),
            end_dim=int(attrs.get("end_dim", -1)),
        )
    if spec.op == "softmax":
        return nn.Softmax(dim=int(attrs.get("dim", 1)))
    return None