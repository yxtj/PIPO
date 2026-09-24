import functools
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn


class AddOp(nn.Module):
    """Element-wise addition. Forward accepts multiple tensors as *inputs."""

    def forward(self, *inputs):
        return functools.reduce(torch.add, inputs)


class MulOp(nn.Module):
    """Element-wise multiplication. Forward accepts multiple tensors as *inputs."""

    def forward(self, *inputs):
        return functools.reduce(torch.mul, inputs)


class JumpOp(nn.Module):
    """Copy the given source feature-map and output it as this layer's FM.

    Unlike `DagModel` default chaining, a `JumpOp` leaf declares an explicit
    source and passes it to its destination regardless of the previous layer's
    output: `forward` simply returns its single input.
    """

    def forward(self, x):
        return x


class ConcatOp(nn.Module):
    """Concatenation along a dimension."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


@dataclass
class FlatEntry:
    """One node of the flattened depth-first walk over a (nested) DagModel.

    Emitted by :meth:`DagModel.flat_entries` — the single numbering authority
    shared by ``expand_model`` (src/common/model_info.py, ExecutionStep IR
    construction) and ``dag_repr(global_fm=True)`` rendering.
    """

    module: nn.Module
    name: str
    sources: list[int]
    dest: int
    depth: int
    is_block: bool


class DagModel(nn.Sequential):
    """A DAG (Directed Acyclic Graph) model with explicit per-layer feature-map source indices.

    Supports arbitrary data-flow graphs, including skip connections and multi-branch concatenations.

    Each layer specifies which feature-map indices it reads from. The final
    output is the feature map produced by the last layer.

    Construction
    ------------
    Pass module instances (or ``(module, source_list)`` tuples) as positional
    args. Each arg becomes one layer in execution order.

    Source-resolution rules (applied in ``source_list``):
        * ``None``       -> resolved to the current layer index (relative 0).
        * ``i >= 0``     -> absolute feature-map index ``i``.
        * ``i < 0``      -> relative offset: ``current_layer_index + i``.

    Special cases for ``AddOp`` and ``ConcatOp``:
        * If only one source is given, the previous layer's output
          (feature-map index = this layer's index) is automatically
          added as a second source.

    Examples
    --------
    # Sequential chain: each layer reads the previous layer's output (default).
    >>> model = DagModel(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))

    # Two-branch concat: layer 3 reads FM[0] (input) and FM[2] (ReLU output).
    >>> model = DagModel(
    ...     nn.Linear(10, 5),          # 0: reads FM[0]
    ...     nn.ReLU(),                  # 1: reads FM[1]
    ...     (nn.Linear(10, 3), [0]),    # 2: reads FM[0]
    ...     nn.ReLU(),                  # 3: reads FM[2]
    ...     (ConcatOp(dim=1), [2, 4]),  # 4: concat FM[2] + FM[4]
    ... )

    # Skip-addition shorthand: AddOp with one relative source.
    # Produces same result as (AddOp(), [3, 5]) - adds FM[3] + FM[5].
    >>> model = DagModel(
    ...     nn.Conv2d(3, 8, 3, padding=1),  # 0
    ...     nn.BatchNorm2d(8),               # 1
    ...     nn.ReLU(),                       # 2 -> output FM[3]
    ...     nn.Conv2d(8, 8, 3, padding=1),   # 3
    ...     nn.BatchNorm2d(8),               # 4 -> output FM[5]
    ...     (AddOp(), [-2]),                 # 5: FM[5-2]=FM[3] + FM[5]
    ...     nn.ReLU(),                       # 6
    ... )

    Naming
    ------
    Layers may be named by passing ``(name, module)`` or ``(name, module,
    sources)`` tuples, or a single ``OrderedDict`` mapping name -> module (or
    ``(module, sources)``). Unnamed layers keep the positional index as their
    name. Names are display labels and attribute-access keys; feature-map
    sources always use indices, never names.

    Display
    -------
    ``repr(model)`` renders a PyTorch-style tree. Nested DagModels (e.g.
    ResidualBlock subclasses) are collapsed to a single line; ``flat=True``
    (``dag_repr(model, flat=True)``) expands them fully and appends a summary
    line. Every layer line carries its connection annotation
    ``FM[<sources>] -> FM[<dest>]`` in the feature-map indices local to that
    (sub)model.

    ``global_fm=True`` (implies expansion) numbers layers and feature maps
    *globally*, across all nested blocks, exactly as the flattened walk
    (:meth:`flat_entries`, which ``expand_model`` consumes) does:
    each leaf gets a global layer label ``L<i>`` (its position in the flattened
    execution order) and all ``FM[...]`` indices refer to the global feature
    maps the server sees. Nested blocks show a fixed-width ``L*`` placeholder in
    the layer-label column so the ``FM[...]`` column stays aligned.
    """

    def __init__(self, *args):
        super().__init__()
        self._sources: list[tuple[int, ...]] = []
        self._refcounts: list[int] = []
        self._build(args)

    def _build(self, args):
        layer_idx = 0
        specs = args[0].items() if len(args) == 1 and isinstance(args[0], Mapping) else ((None, a) for a in args)
        for fallback_name, arg in specs:
            name, module, src_list = _parse_layer(arg, fallback_name)
            abs_sources: tuple[int, ...]
            if src_list is None:
                abs_sources = (layer_idx,)
            else:
                abs_sources = tuple(layer_idx if s is None else layer_idx + s if s < 0 else s for s in src_list)
            # special case - AddOp/ConcatOp with one source: add previous layer as second source
            if isinstance(module, (AddOp, ConcatOp)) and len(abs_sources) == 1:
                abs_sources = abs_sources + (layer_idx,)
            # check AddOp/ConcatOp has no duplicate sources
            if isinstance(module, (AddOp, ConcatOp)) and len(abs_sources) != len(set(abs_sources)):
                raise ValueError(
                    f"Layer {layer_idx} ({type(module).__name__}): duplicate sources {abs_sources} not allowed"
                )
            # check that all sources are in range [0, layer_idx]
            for s in abs_sources:
                if not (0 <= s <= layer_idx):
                    raise ValueError(
                        f"Layer {layer_idx} ({type(module).__name__}): source index {s} out of range [0, {layer_idx}]"
                    )
            self._sources.append(abs_sources)
            self.add_module(name if name is not None else str(layer_idx), module)
            layer_idx += 1
        refcounts = [0] * (len(self._sources) + 1)
        for sources in self._sources:
            for s in set(sources):
                refcounts[s] += 1
        self._refcounts = refcounts

    def flat_entries(self) -> list[FlatEntry]:
        """Flattened depth-first walk with global feature-map numbering.

        Leaves appear in execution order; a nested DagModel child appears as a
        block entry whose ``dest`` is its output FM — the last FM written inside
        it, or its input FM when the block writes none (pass-through). Local FM
        indices are re-mapped to global ones, so a later sibling referencing a
        block reads its output FM. ``expand_model`` builds ExecutionSteps from
        this walk and ``dag_repr(global_fm=True)`` renders it.
        """
        entries: list[FlatEntry] = []
        next_fm = 1

        def walk(model, fm_lut, depth):
            nonlocal next_fm
            for (name, module), local_sources in zip(model.named_children(), model._sources):
                sources = [fm_lut[s] for s in local_sources]
                if isinstance(module, DagModel):
                    block = FlatEntry(module, name, sources, -1, depth, is_block=True)
                    entries.append(block)
                    inner_lut = {0: sources[0]}
                    walk(module, inner_lut, depth + 1)
                    block.dest = inner_lut[len(inner_lut) - 1]
                    fm_lut[len(fm_lut)] = block.dest
                else:
                    dest = next_fm
                    next_fm += 1
                    entries.append(FlatEntry(module, name, sources, dest, depth, is_block=False))
                    fm_lut[len(fm_lut)] = dest

        walk(self, {0: 0}, 1)
        return entries

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fms: list[torch.Tensor | None] = [x] + [None] * len(self._sources)
        output = x
        for i, module in enumerate(self):
            sources = self._sources[i]
            if len(sources) == 1 and sources[0] == i:
                output = module(output)
                if self._refcounts[i] == 1:
                    fms[i] = None
            elif len(sources) >= 1:
                inputs = [fms[idx] for idx in sources]
                output = module(*inputs) if len(inputs) > 1 else module(inputs[0])
            else:
                raise ValueError(f"Invalid sources {sources} for layer {i}")
            fms[i + 1] = output
        return output

    def __repr__(self) -> str:
        return dag_repr(self)

    def dag_repr(self, flat: bool = False, global_fm: bool = False) -> str:
        """Readable tree rendering; ``flat=True`` expands nested DagModels and
        ``global_fm=True`` (implies expansion) numbers layers/FMs globally."""
        return dag_repr(self, flat, global_fm)


def _parse_layer(arg, name=None):
    """Resolve a constructor arg into ``(name, module, src_list)``.

    Accepted forms: a module; ``(module, sources)``; named ``(name, module)`` /
    ``(name, module, sources)`` / ``(name, (module, sources))``. ``name`` is the
    fallback (e.g. an OrderedDict key); a name embedded in the spec wins.
    """
    if isinstance(arg, nn.Module):
        return name, arg, None
    if not isinstance(arg, tuple):
        raise TypeError(f"invalid layer spec: {arg!r}")
    if isinstance(arg[0], nn.Module):
        if len(arg) != 2:
            raise TypeError(f"expected (module, sources), got {len(arg)}-tuple: {arg!r}")
        return name, arg[0], arg[1]
    if isinstance(arg[0], str):
        if len(arg) == 2:
            rest = arg[1]
            if isinstance(rest, nn.Module):
                return arg[0], rest, None
            if isinstance(rest, tuple) and isinstance(rest[0], nn.Module):
                return arg[0], rest[0], rest[1]
        elif len(arg) == 3:
            return arg[0], arg[1], arg[2]
    raise TypeError(f"invalid layer spec: {arg!r}")


def _conn(sources, dest):
    src = ", ".join(str(s) for s in sources)
    return f"FM[{src}] -> FM[{dest}]"


def _stats(model):
    """Recursively count (leaf layers, params, non-sequential layers)."""
    layers = params = skips = 0
    for i, (_, module) in enumerate(model.named_children()):
        if isinstance(module, DagModel):
            sub_layers, sub_params, sub_skips = _stats(module)
            layers += sub_layers
            params += sub_params
            skips += sub_skips
        else:
            layers += 1
            params += sum(p.numel() for p in module.parameters())
        if model._sources[i] != (i,):
            skips += 1
    return layers, params, skips


def _render_children(model, lines, depth, flat):
    children = list(model.named_children())
    if not children:
        return
    indent = "  " * depth
    heads = []
    reprs = {}
    for i, (_, module) in enumerate(children):
        if isinstance(module, DagModel):
            head = f"{type(module).__name__}(" if flat else f"{type(module).__name__}({len(module)} layers)"
        else:
            mod_repr = repr(module)
            head = mod_repr.splitlines()[0]
            reprs[i] = mod_repr
        heads.append(head)
    col = max(len(f"({name}): {head}") for (name, _), head in zip(children, heads))
    for i, ((name, module), head) in enumerate(zip(children, heads)):
        left = f"({name}): {head}"
        conn = _conn(model._sources[i], i + 1)
        lines.append(f"{indent}{left}{' ' * (col - len(left) + 2)}{conn}")
        if isinstance(module, DagModel) and flat:
            _render_children(module, lines, depth + 1, flat)
            lines.append(f"{indent})")
        elif not isinstance(module, DagModel):
            for cont in reprs[i].splitlines()[1:]:
                lines.append(f"{indent}  {cont}")


def _render_global(model, lines, label_w):
    """Render with global (flattened) layer and FM numbering.

    All numbering (global FM sources/dests, ``L<i>`` labels) comes from
    ``model.flat_entries()`` — the same walk ``expand_model`` builds
    ExecutionSteps from — so rendering and IR can never drift. This function is
    purely presentational: it groups the entries by parent submodel to align the
    layer/FM columns, with nested blocks showing a fixed-width ``L*``
    placeholder (``label_w``) so the ``FM[...]`` column stays aligned.
    """
    entries = model.flat_entries()

    # Group entries by their parent submodel: a block owns every following
    # entry at depth + 1, until a sibling at <= its depth closes it.
    groups: dict[int, list[FlatEntry]] = {0: []}
    stack: list[FlatEntry] = []
    for e in entries:
        while stack and e.depth <= stack[-1].depth:
            stack.pop()
        groups[id(stack[-1]) if stack else 0].append(e)
        if e.is_block:
            stack.append(e)
            groups[id(e)] = []

    # Head line of each child's repr, plus the per-group column width.
    heads: dict[int, str] = {}
    conts: dict[int, list[str]] = {}
    widths: dict[int, int] = {}
    for key, group in groups.items():
        group_widths = []
        for e in group:
            if e.is_block:
                heads[id(e)] = f"{type(e.module).__name__}("
                conts[id(e)] = []
            else:
                mod_repr = repr(e.module)
                heads[id(e)] = mod_repr.splitlines()[0]
                conts[id(e)] = mod_repr.splitlines()[1:]
            group_widths.append(len(f"({e.name}): {heads[id(e)]}"))
        widths[key] = max(group_widths, default=0)

    layer_idx = 0

    def render(key, depth):
        nonlocal layer_idx
        indent = "  " * depth
        for e in groups[key]:
            left = f"({e.name}): {heads[id(e)]}"
            pad = " " * (widths[key] - len(left) + 2)
            if e.is_block:
                label = f"{'L*':<{label_w}}  "
                lines.append(f"{indent}{left}{pad}{label}{_conn(e.sources, e.dest)}")
                render(id(e), depth + 1)
                lines.append(f"{indent})")
            else:
                label = f"{f'L{layer_idx}':<{label_w}}  "
                lines.append(f"{indent}{left}{pad}{label}{_conn(e.sources, e.dest)}")
                layer_idx += 1
                for cont in conts[id(e)]:
                    lines.append(f"{indent}  {cont}")

    render(0, 1)


def dag_repr(model, flat=False, global_fm=False):
    """Render a DagModel as a PyTorch-style readable tree.

    Nested DagModels are collapsed to a single line by default; ``flat=True``
    expands them fully and appends a summary line. ``global_fm=True`` implies
    expansion and numbers layers (``L<i>``) and feature maps globally across
    all nested blocks, matching :meth:`DagModel.flat_entries` (the walk
    ``expand_model`` builds ExecutionSteps from); nested blocks carry a ``L*``
    placeholder so the ``FM[...]`` column stays aligned. Each layer line shows
    its feature-map connection ``FM[<sources>] -> FM[<dest>]`` (indices local
    to that (sub)model, unless ``global_fm``). Non-DagModel inputs fall back to
    their native repr.
    """
    if not isinstance(model, DagModel):
        return repr(model)
    children = list(model.named_children())
    stats = _stats(model) if (flat or global_fm) else None
    lines = []
    if not children:
        lines.append(f"{type(model).__name__}()")
    else:
        lines.append(f"{type(model).__name__}(")
        if global_fm:
            label_w = 1 + len(str(stats[0] - 1))
            _render_global(model, lines, label_w)
        else:
            _render_children(model, lines, 1, flat)
        lines.append(")")
    if stats:
        layers, params, skips = stats
        lines.append(
            f"Summary: {layers} layers, {params:,} params, {layers + 1} feature maps, "
            f"{skips} non-sequential layer{'s' if skips != 1 else ''}"
        )
    return "\n".join(lines)
