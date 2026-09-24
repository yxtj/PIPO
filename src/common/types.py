from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutionStep:
    """One execution step of a flattened model.

    sources: global feature-map indices read by the op
    dest: global feature-map index written by the op
    op: lowercase module class name (e.g. "conv2d", "relu")
    layer_idx: index into the flat leaf list, only for SERVER_OPS steps
    k/s/p: kernel/stride/padding for kernel-bearing ops
    concat_dim: concatenation axis for concat steps
    """

    sources: list[int]
    dest: int
    op: str
    layer_idx: Optional[int] = None
    k: Optional[tuple[int, int]] = None
    s: Optional[tuple[int, int]] = None
    p: Optional[tuple[int, int]] = None
    concat_dim: Optional[int] = None


@dataclass
class ModelInfo:
    """Structural IR of a model: steps, per-FM shapes, input shape."""

    steps: list[ExecutionStep] = field(default_factory=list)
    shapes: list[tuple[int, ...]] = field(default_factory=list)
    inshape: tuple[int, ...] = ()
