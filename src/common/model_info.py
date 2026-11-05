import math

import torch
import torch.nn as nn

from src.common.types import ExecutionStep, ModelInfo
from src.model.dag_model import AddOp, ConcatOp, DagModel, JumpOp

#: Ops dispatched to the server. Inference-time BatchNorm is a fixed per-channel
#: affine (y = scale*x + shift, parameters frozen at training time), so it is a
#: linear op and runs server-side like conv/linear.
SERVER_OPS = (nn.Conv2d, nn.Linear, nn.AvgPool2d, nn.Identity, nn.BatchNorm2d, nn.BatchNorm1d)

#: Modules that preserve their input shape (elementwise / normalization / identity.
#: ``nn.Softmax`` is included for the PIPO plaintext head layer.
_IDENTITY_SHAPE_OPS = (nn.ReLU, nn.Sigmoid, nn.LeakyReLU, nn.GELU, nn.BatchNorm1d, nn.BatchNorm2d, nn.Identity, nn.Softmax)


def _ext_params(module: nn.Module) -> dict:
    """Extract op parameters (k, s, p for kernel-bearing modules, dim for concat).

    BatchNorm carries no kernel but acts as a 1x1 op (each output position
    depends only on the input at the same position), so it gets the
    ``k=s=(1,1), p=(0,0)`` geometry for the incremental block paths.

    ``attrs`` holds every op-specific parameter (kernel geometry, concat axis,
    activation/pooling/flatten/softmax params); generic fields stay on
    ``ExecutionStep``.
    """
    attrs: dict = {}
    if hasattr(module, "kernel_size"):
        attrs.update({"k": list(_pair(module.kernel_size)),
                      "s": list(_pair(module.stride)),
                      "p": list(_pair(module.padding))})
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        attrs.update({"k": [1, 1], "s": [1, 1], "p": [0, 0]})
    if isinstance(module, ConcatOp):
        attrs["concat_dim"] = module.dim
    if isinstance(module, nn.LeakyReLU):
        attrs["negative_slope"] = float(module.negative_slope)
    elif isinstance(module, nn.GELU):
        attrs["approximate"] = module.approximate
    elif isinstance(module, nn.MaxPool2d):
        attrs.update({
            "kernel_size": list(_pair(module.kernel_size)),
            "stride": list(_pair(module.stride)),
            "padding": list(_pair(module.padding)),
            "dilation": list(_pair(module.dilation)),
            "return_indices": bool(module.return_indices),
            "ceil_mode": bool(module.ceil_mode),
        })
    elif isinstance(module, nn.AvgPool2d):
        attrs.update({
            "kernel_size": list(_pair(module.kernel_size)),
            "stride": list(_pair(module.stride)),
            "padding": list(_pair(module.padding)),
            "ceil_mode": bool(module.ceil_mode),
            "count_include_pad": bool(module.count_include_pad),
            "divisor_override": module.divisor_override,
        })
    elif isinstance(module, nn.Flatten):
        attrs.update({"start_dim": int(module.start_dim), "end_dim": int(module.end_dim)})
    elif isinstance(module, nn.Softmax):
        attrs["dim"] = int(module.dim)
    return {"attrs": attrs} if attrs else {}


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


def _pool_out(dim: int, k: int, s: int, p: int, d: int) -> int:
    """Output spatial side length for a kernel-bearing op (conv / pool)."""
    return (dim + 2 * p - d * (k - 1) - 1) // s + 1


def _flatten_shape(shape: tuple[int, ...], start_dim: int, end_dim: int) -> tuple[int, ...]:
    """Output shape of `nn.Flatten(start_dim, end_dim)` for a batch-less shape.

    `shape` excludes the batch dim; Flatten's `start_dim`/`end_dim` are as given
    on the full (batch-prefixed) tensor, so we shift by one and strip the batch.
    """
    ndim = len(shape) + 1
    start = start_dim if start_dim >= 0 else ndim + start_dim
    end = end_dim if end_dim >= 0 else ndim + end_dim
    full = [1, *shape]
    flat = math.prod(full[start : end + 1])
    out = full[:start] + [flat] + full[end + 1 :]
    return tuple(out[1:])


def _static_shape(module: nn.Module, input_shapes: list[tuple[int, ...]]) -> tuple[int, ...]:
    """Output feature-map shape (no batch dim) for a module given its source shapes.

    Pure static analysis: reads module parameters, never runs a forward pass.
    """
    x = input_shapes[0]
    if isinstance(module, nn.Conv2d):
        attrs = _ext_params(module)["attrs"]
        k, s, p = attrs["k"], attrs["s"], attrs["p"]
        d = _pair(module.dilation)
        return (
            module.out_channels,
            _pool_out(x[1], k[0], s[0], p[0], d[0]),
            _pool_out(x[2], k[1], s[1], p[1], d[1]),
        )
    if isinstance(module, nn.Linear):
        return (module.out_features,)
    if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
        attrs = _ext_params(module)["attrs"]
        k, s, p = attrs["k"], attrs["s"], attrs["p"]
        d = _pair(getattr(module, "dilation", 1))
        return (
            x[0],
            _pool_out(x[1], k[0], s[0], p[0], d[0]),
            _pool_out(x[2], k[1], s[1], p[1], d[1]),
        )
    if isinstance(module, nn.Flatten):
        return _flatten_shape(x, module.start_dim, module.end_dim)
    if isinstance(module, AddOp):
        return x
    if isinstance(module, JumpOp):
        # a jump simply copies its single source feature map
        return x
    if isinstance(module, ConcatOp):
        # torch dim is on the batch-prefixed tensor; FM shapes strip the batch
        dim = module.dim - 1 if module.dim > 0 else module.dim
        out = list(x)
        if not (-len(out) <= dim < len(out)):
            raise ValueError(f"concat dim {module.dim} out of range for feature-map shape {x}")
        out[dim] = sum(inp[dim] for inp in input_shapes)
        return tuple(out)
    if isinstance(module, _IDENTITY_SHAPE_OPS):
        return x
    raise NotImplementedError(f"no static shape rule for {type(module).__name__}")


def expand_model(model: DagModel) -> tuple[list[ExecutionStep], list[nn.Module]]:
    """Expand a (possibly nested) DagModel into flat ExecutionSteps with global FM sources.

    Numbering comes entirely from ``model.flat_entries()`` — DagModel's own
    flattened walk (the same one ``dag_repr(global_fm=True)`` renders); this
    function only attaches IR/protocol semantics: ``ExecutionStep`` fields plus
    op-specific parameters (``layer_idx`` for ``SERVER_OPS``, kernel geometry,
    ...) inside ``attrs``.

    Returns:
        steps: list of ExecutionStep with global FM indices
        leaves: list of leaf nn.Module in step order
    """
    steps: list[ExecutionStep] = []
    leaves: list[nn.Module] = []
    for e in model.flat_entries():
        if e.is_block:
            continue
        op = type(e.module).__name__.lower()
        params = _ext_params(e.module)
        if isinstance(e.module, SERVER_OPS):
            params.setdefault("attrs", {})["layer_idx"] = len(leaves)
        steps.append(
            ExecutionStep(
                sources=list(e.sources),
                dest=e.dest,
                op=op,
                **params,
            )
        )
        leaves.append(e.module)
    return steps, leaves


def flat_modules(model: nn.Module) -> list[nn.Module]:
    """All leaf modules in flattened execution order — the `layer_idx` index space.

    This is the lookup table for a request's ``layer_id``: a nested ``DagModel``
    child expands to its leaves (matching ``expand_model``), so
    ``flat_modules(model)[layer_idx]`` is always the module the client means.
    Indexing the top-level ``nn.Sequential`` directly would mis-resolve
    ``layer_id`` for nested models. A plain ``nn.Sequential`` is already the
    leaf list.
    """
    if isinstance(model, DagModel):
        return [e.module for e in model.flat_entries() if not e.is_block]
    return list(model)


def _static_shapes(model, inshape):
    """Compute every feature-map shape statically - no dummy forward pass.

    Walks the model graph (nested DagModels via ``expand_model``), propagating
    spatial dims through each op by reading module parameters.
    """
    if isinstance(model, DagModel):
        steps, leaves = expand_model(model)
        shapes = [tuple(inshape)]
        fms = {0: tuple(inshape)}
        for step, module in zip(steps, leaves):
            inputs = [fms[s] for s in step.sources]
            out = _static_shape(module, inputs)
            shapes.append(out)
            fms[step.dest] = out
        return shapes

    shapes = [tuple(inshape)]
    fm = tuple(inshape)
    for layer in model:
        fm = _static_shape(layer, [fm])
        shapes.append(fm)
    return shapes


def _dummy_forward_shapes(model, inshape):
    """Compute every feature-map shape by running a real (dummy) forward pass.

    Retained as the reference / cross-check path for ``compute_shapes(fast=False)``
    and for parity tests against the static analysis.
    """
    shapes = [tuple(inshape)]
    if isinstance(model, DagModel):
        steps, leaves = expand_model(model)
        fms = {0: torch.zeros(1, *inshape)}
        for step, module in zip(steps, leaves):
            inputs = [fms[s] for s in step.sources]
            with torch.no_grad():
                output = module(*inputs) if len(inputs) > 1 else module(inputs[0])
            shapes.append(tuple(output.shape[1:]))
            fms[step.dest] = output
        return shapes

    fms = [torch.zeros(1, *inshape)]
    for i, layer in enumerate(model):
        sources = [i]
        inputs = [fms[s] for s in sources]
        with torch.no_grad():
            output = layer(*inputs) if len(inputs) > 1 else layer(inputs[0])
        shapes.append(tuple(output.shape[1:]))
        fms.append(output)
    return shapes


def compute_shapes(model, inshape, fast: bool = True):
    """Compute every feature-map shape.

    ``fast=True`` (default) uses static shape analysis - reading module
    parameters, never a forward pass. ``fast=False`` falls back to the
    dummy-forward reference path (``_dummy_forward_shapes``), useful as a
    cross-check in tests.
    """
    if fast:
        return _static_shapes(model, inshape)
    return _dummy_forward_shapes(model, inshape)


def build_steps(model, all_shapes):
    if isinstance(model, DagModel):
        steps, _ = expand_model(model)
        return steps, all_shapes

    steps = []
    for i, layer in enumerate(model):
        op = type(layer).__name__.lower()
        params = _ext_params(layer)
        if isinstance(layer, SERVER_OPS):
            params.setdefault("attrs", {})["layer_idx"] = i
        step = ExecutionStep(sources=[i], dest=i + 1, op=op, **params)
        steps.append(step)
    return steps, all_shapes


def build_model_info(model, inshape):
    """Build the server-side model IR, including per-step execution shapes.

    ``ishape``/``oshape`` are stored without the batch dimension; the client
    prepends the runtime batch (1) when constructing its layer objects.
    """
    inshape = tuple(inshape)
    all_shapes = compute_shapes(model, inshape)
    steps, shapes = build_steps(model, all_shapes)
    if len(shapes) != len(steps) + 1:
        raise ValueError(
            "shape/step mismatch: {} shapes for {} steps".format(len(shapes), len(steps))
        )
    for i, step in enumerate(steps):
        step.ishape = tuple(shapes[i])
        step.oshape = tuple(shapes[i + 1])
    return ModelInfo(steps=steps, shapes=[tuple(s) for s in shapes], inshape=inshape)
