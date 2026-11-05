from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionStep:
    """One execution step of a flattened model.

    sources: global feature-map indices read by the op
    dest: global feature-map index written by the op
    op: lowercase module class name (e.g. "conv2d", "relu")
    ishape/oshape: execution-order feature-map shapes (no batch dim)
    attrs: op-specific parameters (kernel/stride/padding, concat axis, layer
        index, activation slope, pooling/flatten/softmax params, ...)

    Only fields every op needs live on the dataclass; anything used by a
    specific op family goes into ``attrs``.
    """

    sources: list[int]
    dest: int
    op: str
    ishape: Optional[tuple[int, ...]] = None
    oshape: Optional[tuple[int, ...]] = None
    attrs: dict = field(default_factory=dict)


@dataclass
class ModelInfo:
    """Structural IR of a model: steps, per-FM shapes, input shape."""

    steps: list[ExecutionStep] = field(default_factory=list)
    shapes: list[tuple[int, ...]] = field(default_factory=list)
    inshape: tuple[int, ...] = ()
