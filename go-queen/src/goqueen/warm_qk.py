"""APS warm start for the Q/K-only recipe: in-place conversion, freezing and full checkpoint.

The fitted APS object is switched back in place to the original M classes, so
it computes the original natural attention over all 361 board tokens with no
point index or one-point mask. No base is rewrapped, no tensor is created or
copied and no random number is drawn. Afterwards only the 44 query_norm /
query / key tensors are trainable.

The existing adapter save/load helpers select by requires_grad, so they would
see only those 44. The full 101-tensor list is therefore collected by object
identity, and the checkpoint is written and load-checked from that list.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
import safetensors.torch
from safetensors import safe_open

from goqueen import checkpoints
from goqueen.assisted_point import AssistedPointGatedBoardAttention, AssistedPointGoQueen
from goqueen.fusion import ExpertDecoderLayer, GatedBoardAttention, GoQueen

INJECTED_LAYERS = 11
# GatedBoardAttention's own alpha first, then its submodules in registration order,
# as named_parameters() yields them.
ADAPTER_SUFFIXES = ("alpha", "query_norm.weight", "query_norm.bias", "expert_norm.weight",
                    "expert_norm.bias", "query.weight", "key.weight", "value.weight", "output.weight")
QK_SUFFIXES = ("query_norm.weight", "query_norm.bias", "query.weight", "key.weight")
ADAPTER_TENSORS = INJECTED_LAYERS * len(ADAPTER_SUFFIXES) + 2   # 101, with rows and columns
VARIANT, TRAINING_RECIPE = "m", "aps_warm_qk"

_APS_FIELDS = ("point_mask", "point_indices")
_HOOKS = ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")


def adapter_parameters(model):
    """The complete ordered 101 adapter (name, Parameter) pairs, whatever their requires_grad flags.

    Base tensors are excluded by object identity, so a tied base weight that
    named_parameters(remove_duplicate=False) lists twice is excluded both times.
    """
    if not isinstance(model, GoQueen):
        raise TypeError("Expected a GoQueen model")
    layers, indices = model.language_model.model.layers, list(model.layer_indices)
    if (len(indices) != INJECTED_LAYERS or len(set(indices)) != INJECTED_LAYERS
            or not all(0 <= i < len(layers) and type(layers[i]) is ExpertDecoderLayer for i in indices)
            or sum(isinstance(m, ExpertDecoderLayer) for m in model.modules()) != INJECTED_LAYERS
            or sum(isinstance(m, GatedBoardAttention) for m in model.modules()) != INJECTED_LAYERS):
        raise ValueError("Expected exactly 11 injected adapter layers, none nested or extra")
    base = {id(p) for p in model.base_parameters}
    named = tuple((name, p) for name, p in model.named_parameters(remove_duplicate=False)
                  if id(p) not in base)
    expected = [f"language_model.model.layers.{i}.adapter.{suffix}"
                for i in sorted(indices) for suffix in ADAPTER_SUFFIXES] + ["rows.weight", "columns.weight"]
    if [name for name, _ in named] != expected:
        raise ValueError("Non-base parameters are not exactly the 101-tensor adapter layout")
    if len({id(p) for _, p in named}) != ADAPTER_TENSORS:
        raise ValueError("Adapter tensors must be distinct objects")
    return named


def convert_to_natural(model):
    """Switch a clean fitted APS object in place to original M and return the same object.

    Every precondition is checked before anything changes. The caller must
    leave any binding and close its routing monitor first: live state is
    rejected, never discarded. Only the plain APS attributes that a finished
    binding leaves behind as None are removed.
    """
    if type(model) is not AssistedPointGoQueen:
        raise TypeError("Conversion requires an exact AssistedPointGoQueen")
    before = adapter_parameters(model)
    layers = model.injected_layers
    if any(type(layer.adapter) is not AssistedPointGatedBoardAttention for layer in layers):
        raise TypeError("Conversion requires 11 exact APS adapters")
    if model._bound or any(layer.expert_features is not None for layer in layers):
        raise RuntimeError("Expert features are still bound")
    if any(getattr(layer.adapter, field) is not None for layer in layers for field in _APS_FIELDS):
        raise RuntimeError("An APS point mask or point indices are still stored")
    if any(getattr(layer.adapter, hooks) for layer in layers for hooks in _HOOKS):
        raise RuntimeError("Adapter hooks remain; close the routing monitor first")
    flags = [p.requires_grad for _, p in before]
    for layer in layers:
        for field in _APS_FIELDS:
            vars(layer.adapter).pop(field, None)
        layer.adapter.__class__ = GatedBoardAttention
    model.__class__ = GoQueen
    after = adapter_parameters(model)
    if ([(n, id(p)) for n, p in after] != [(n, id(p)) for n, p in before]
            or [p.requires_grad for _, p in after] != flags):
        raise RuntimeError("Conversion changed the adapter parameter list")
    return model


def freeze_except_qk(model, named):
    """Make exactly the 44 query_norm/query/key tensors trainable; freeze the other 57 and the base.

    Returns the ordered 44 (name, Parameter) pairs for building the optimizer.
    """
    current = _natural_adapters(model, named)
    selected = tuple((name, p) for name, p in current
                     if name.rpartition(".adapter.")[2] in QK_SUFFIXES)
    chosen = {id(p) for _, p in selected}
    for p in model.base_parameters:
        p.requires_grad_(False)
    for _, p in current:
        p.requires_grad_(id(p) in chosen)
    return selected


def save_full_checkpoint(model, named, path, metadata):
    """Write all 101 adapter tensors to a new file, read them back, then load-check with the existing loader.

    The loader selects by requires_grad, so all 101 adapter flags are enabled
    only around that one call and restored in finally; the base stays frozen.
    A failed write or check leaves the file in place for inspection; nothing
    is deleted or retried. The check uses this model object in this process.
    """
    current = _natural_adapters(model, named)
    if any(p.requires_grad for p in model.base_parameters):
        raise ValueError("Base parameters must be frozen")
    if not isinstance(metadata, dict) or not all(type(k) is str and type(v) is str
                                                 for k, v in metadata.items()):
        raise ValueError("Checkpoint metadata must map strings to strings")
    if metadata.get("variant") != VARIANT or metadata.get("training_recipe") != TRAINING_RECIPE:
        raise ValueError("Checkpoint metadata requires variant=m and training_recipe=aps_warm_qk")
    if "routing_contract" in metadata:
        raise ValueError("A natural M checkpoint cannot declare an APS routing contract")
    metadata, path = dict(metadata), Path(path)
    if path.is_symlink() or path.exists():
        raise FileExistsError("Checkpoint destination already exists; it is never overwritten")
    # Independent copies: the later comparison must not compare a parameter with itself.
    state = {name: p.detach().cpu().clone().contiguous() for name, p in current}
    for name, tensor in state.items():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Non-finite adapter tensor: " + name)
    data = safetensors.torch.save(state, metadata=metadata)
    with path.open("xb") as stream:
        stream.write(data)
    del data
    digest = _file_sha256(path)

    with safe_open(str(path), framework="pt", device="cpu") as saved:
        if (saved.metadata() or {}) != metadata or set(saved.keys()) != set(state):
            raise RuntimeError("Saved names or metadata differ from the full adapter state")
        for name, tensor in state.items():
            if not _same_bytes(saved.get_tensor(name), tensor):
                raise RuntimeError("Saved tensor differs from the full adapter state: " + name)

    flags = [(p, p.requires_grad) for _, p in current]
    try:
        for _, p in current:
            p.requires_grad_(True)
        loaded = checkpoints.load_adapter_checkpoint(model, path)
    finally:
        for p, flag in flags:
            p.requires_grad_(flag)
    if loaded != metadata:
        raise RuntimeError("The existing loader returned different metadata")
    for name, p in current:
        if not _same_bytes(p, state[name]):
            raise RuntimeError("Same-model load did not keep the full adapter state: " + name)
    if any(p.requires_grad for p in model.base_parameters):
        raise RuntimeError("Base parameters became trainable")
    return {"sha256": digest, "count": len(state), "values": sum(t.numel() for t in state.values()),
            "file_bytes": path.stat().st_size, "metadata": metadata,
            "readback_verified": True, "loader_verified": True,
            "loader_scope": "goqueen.checkpoints.load_adapter_checkpoint on this same model in this "
                            "process; no separate-process reload or restored forward"}


def _natural_adapters(model, named):
    """This converted model's full adapter list, after checking `named` is exactly that list."""
    if type(model) is not GoQueen:
        raise TypeError("Expected the converted natural GoQueen")
    current = adapter_parameters(model)
    if any(type(layer.adapter) is not GatedBoardAttention for layer in model.injected_layers):
        raise TypeError("Expected original M adapters")
    try:
        given = [(name, id(p)) for name, p in named]
    except (TypeError, ValueError):
        raise ValueError("named must be (name, parameter) pairs") from None
    if given != [(name, id(p)) for name, p in current]:
        raise ValueError("named is not this model's complete ordered 101-tensor adapter list")
    return current


def _same_bytes(a, b):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    return (a.dtype == b.dtype and a.shape == b.shape
            and torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8)))


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
