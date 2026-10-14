"""Board fusion for an already-loaded Qwen3_5ForCausalLM text backbone.

This module does not load Qwen, choose quantization/distribution, create an
optimizer, or enable checkpointing.  Its new narrow adapter separates the
attention width (default 1024) from the language hidden width (5120 for 27B).
It does not change the historical ``fusion.GatedBoardAttention`` implementation.
The reference decoder API is Transformers v5.19.0 ``modeling_qwen3_5.py``:
``model.layers``, ``block_type``, and a Tensor decoder return value.

Training must keep the binding alive THROUGH backward, one microbatch at a time::

    with model.bind_expert(features):
        output = model(input_ids=ids, attention_mask=mask, logits_to_keep=1)
        loss = loss_function(output.logits)
        loss.backward()

Do not return a loss from this context and call backward later.  The wrappers
recompute positional embeddings on every call, including checkpoint replay;
they never cache a positional tensor's autograd graph across calls.  Use
non-reentrant activation checkpointing for frozen-base training and verify the
actual cloud execution path before adopting a distributed training setup.
Concurrent forwards and overlapping microbatch graphs are unsupported.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F


INJECTION_LAYERS = (0, 6, 13, 19, 25, 32, 38, 44, 50, 57, 63)
_BLOCK_TYPES = {"linear_attention", "full_attention"}


class NarrowBoardCrossAttention(nn.Module):
    """Board attention with an explicit internal width and residual dtype.

    Attention, projections, normalization and gate arithmetic use the adapter
    parameter dtype, even inside an enclosing autocast context.  The completed
    delta is cast to the original hidden dtype before residual addition.  This
    is a new architecture/precision contract, not a historical checkpoint alias.
    """

    def __init__(self, language_width: int, expert_width: int = 768,
                 heads: int = 8, attention_width: int = 1024):
        super().__init__()
        for name, value in (("language_width", language_width), ("expert_width", expert_width),
                            ("heads", heads), ("attention_width", attention_width)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if attention_width % heads:
            raise ValueError("Attention width must be divisible by its head count")
        self.heads = heads
        self.attention_width = attention_width
        self.head_width = attention_width // heads
        self.query_norm = nn.LayerNorm(language_width)
        self.expert_norm = nn.LayerNorm(expert_width)
        self.query = nn.Linear(language_width, attention_width, bias=False)
        self.key = nn.Linear(expert_width, attention_width, bias=False)
        self.value = nn.Linear(expert_width, attention_width, bias=False)
        self.output = nn.Linear(attention_width, language_width, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        if width != self.query.in_features or expert.shape != (batch, 361, self.key.in_features):
            raise ValueError("Language or board feature shape does not match the adapter")
        if hidden.device != self.query.weight.device or expert.device != hidden.device:
            raise ValueError("Hidden, features and adapter must share a device")
        dtype = self.query.weight.dtype
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            query = self.query(self.query_norm(hidden.to(dtype=dtype)))
            normalized_expert = self.expert_norm(expert.to(dtype=dtype))
            key, value = self.key(normalized_expert), self.value(normalized_expert)
            query = query.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
            key = key.reshape(batch, 361, self.heads, self.head_width).transpose(1, 2)
            value = value.reshape(batch, 361, self.heads, self.head_width).transpose(1, 2)
            attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            attended = attended.transpose(1, 2).reshape(batch, length, self.attention_width)
            delta = torch.tanh(self.alpha) * self.output(attended)
        return hidden + delta.to(dtype=hidden.dtype)


class Qwen38ExpertDecoderLayer(nn.Module):
    """Preserve the hybrid decoder interface and differentiable device transfer."""

    def __init__(
        self,
        decoder: nn.Module,
        adapter: NarrowBoardCrossAttention,
        position_provider: Callable[[], torch.Tensor],
    ):
        super().__init__()
        if getattr(decoder, "block_type", None) not in _BLOCK_TYPES:
            raise ValueError("Expected a Qwen3_5 hybrid decoder with block_type")
        self.decoder = decoder
        self.adapter = adapter
        self.block_type = decoder.block_type
        # A callable, not another registered copy of the shared embedding module.
        self._position_provider = position_provider
        self.expert_features = None

    def forward(self, hidden_states, *args, **kwargs):
        if self.expert_features is None:
            raise RuntimeError("Decoder called outside its expert binding")
        parameter = self.adapter.query.weight
        # Loading the base with an explicit device map attaches dispatch hooks to
        # the ORIGINAL decoder.  This new pre-decoder adapter runs before those
        # hooks, so it must move the boundary activation itself.  Tensor.to keeps
        # autograd connected across devices; never detach this activation.
        hidden_states = hidden_states.to(device=parameter.device)
        # Copies/casts retain positional gradients; expert input was detached at bind.
        # Compute the shared positions here so checkpoint replay builds a fresh graph.
        expert = self.expert_features.to(device=parameter.device, dtype=parameter.dtype)
        position = self._position_provider().to(device=parameter.device, dtype=parameter.dtype)
        hidden_states = self.adapter(hidden_states, expert + position.unsqueeze(0))
        # Preserve positional/cache/mask kwargs and the original Tensor return API.
        return self.decoder(hidden_states, *args, **kwargs)


class Qwen38GoQueen(nn.Module):
    """Attach eleven narrow board adapters to a 64-layer text-only backbone.

    Pass the already-loaded official text-only model, not the VL wrapper.
    ``adapter_devices`` optionally maps every injected layer index to its device;
    otherwise each adapter follows that decoder's first parameter device.
    ``adapter_dtype`` and ``position_device`` are placement choices supplied by
    the caller; this class does not move/quantize the frozen base.  An omitted
    adapter dtype retains PyTorch's current default dtype, as the old adapter does.

    All adapter parameters initially remain trainable.  Call
    ``set_trainable_adapter_names`` with an explicit subset if the adopted recipe
    requires one.  Selecting only Q/K with every gate zero is not warm starting.
    Parameter sharding/dispatch hooks and optimizer construction belong to the
    caller; this wrapper alone is not a tested FSDP/ZeRO integration.
    """

    def __init__(
        self,
        language_model: nn.Module,
        *,
        expert_width: int = 768,
        heads: int = 8,
        attention_width: int = 1024,
        adapter_dtype: torch.dtype | None = None,
        adapter_devices: Mapping[int, object] | None = None,
        position_device: object | None = None,
    ):
        super().__init__()
        config = language_model.config
        if getattr(config, "model_type", None) != "qwen3_5_text":
            raise ValueError("Load Qwen3_5ForCausalLM with its text-only configuration first")
        layers = language_model.model.layers
        if config.num_hidden_layers != 64 or len(layers) != 64:
            raise ValueError("This integration requires the requested 64-layer backbone")
        if len(config.layer_types) != 64:
            raise ValueError("Expected one hybrid block type per decoder")
        if isinstance(expert_width, bool) or not isinstance(expert_width, int) or expert_width < 1:
            raise ValueError("expert_width must be a positive integer")
        if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
            raise ValueError("heads must be a positive integer")
        if (isinstance(attention_width, bool) or not isinstance(attention_width, int)
                or attention_width < 1 or attention_width % heads):
            raise ValueError("attention_width must be positive and divisible by heads")
        if adapter_devices is not None and set(adapter_devices) != set(INJECTION_LAYERS):
            raise ValueError("adapter_devices must name exactly the eleven injection layers")
        if adapter_dtype is not None and not adapter_dtype.is_floating_point:
            raise ValueError("Adapters require a floating point dtype")
        devices = {}
        for index, decoder in enumerate(layers):
            if isinstance(decoder, Qwen38ExpertDecoderLayer):
                raise ValueError("Board adapters have already been installed")
            if getattr(decoder, "block_type", None) != config.layer_types[index]:
                raise ValueError(f"Decoder {index} does not match its configured block type")
            if config.layer_types[index] not in _BLOCK_TYPES:
                raise ValueError(f"Unsupported block type at decoder {index}")
            if index in INJECTION_LAYERS:
                target = (adapter_devices[index] if adapter_devices is not None
                          else next(decoder.parameters()).device)
                devices[index] = torch.device(target)
                if devices[index].type == "meta":
                    raise ValueError("Materialize the base or supply actual adapter devices first")

        self.language_model = language_model
        self.base_parameters = tuple(language_model.parameters())
        self.expert_width = expert_width
        self.attention_width = attention_width
        self.layer_indices = list(INJECTION_LAYERS)
        self._bound = False
        self._bound_batch_size = None
        self.freeze_base()
        placement = torch.device(position_device) if position_device is not None else devices[0]
        if placement.type == "meta":
            raise ValueError("Position embeddings need a materialized device")
        self.rows = nn.Embedding(19, expert_width, device=placement, dtype=adapter_dtype)
        self.columns = nn.Embedding(19, expert_width, device=placement, dtype=adapter_dtype)
        nn.init.normal_(self.rows.weight, std=0.02)
        nn.init.normal_(self.columns.weight, std=0.02)
        self.register_buffer("row_indices", torch.arange(19, device=placement).repeat_interleave(19),
                             persistent=False)
        self.register_buffer("column_indices", torch.arange(19, device=placement).repeat(19),
                             persistent=False)
        for index in self.layer_indices:
            adapter = NarrowBoardCrossAttention(config.hidden_size, expert_width, heads, attention_width)
            adapter.to(device=devices[index], dtype=adapter_dtype)
            layers[index] = Qwen38ExpertDecoderLayer(layers[index], adapter, self.position_embeddings)
        # Capture the full adapter name set independently of future requires_grad flags.
        base_ids = {id(parameter) for parameter in self.base_parameters}
        self._adapter_names = tuple(name for name, parameter in self.named_parameters()
                                    if id(parameter) not in base_ids)

    @property
    def config(self):
        return self.language_model.config

    @property
    def injected_layers(self):
        return [self.language_model.model.layers[index] for index in self.layer_indices]

    def position_embeddings(self):
        return self.rows(self.row_indices) + self.columns(self.column_indices)

    def freeze_base(self):
        for parameter in self.base_parameters:
            parameter.requires_grad_(False)
            parameter.grad = None

    def adapter_named_parameters(self):
        """All adapters, including currently frozen ones; call before FSDP wrapping."""
        parameters = dict(self.named_parameters())
        return {name: parameters[name] for name in self._adapter_names}

    def set_trainable_adapter_names(self, names: Iterable[str] | None):
        """Freeze the base and select an explicit adapter subset; None selects all."""
        if self._bound:
            raise RuntimeError("Cannot change trainable parameters during a bound microbatch")
        parameters = self.adapter_named_parameters()
        supplied = list(parameters) if names is None else list(names)
        if len(set(supplied)) != len(supplied) or not set(supplied).issubset(parameters):
            raise ValueError("Trainable names must be distinct names from the full adapter set")
        self.freeze_base()
        selected = set(supplied)
        for name, parameter in parameters.items():
            parameter.requires_grad_(name in selected)
            parameter.grad = None
        return {name: parameter for name, parameter in parameters.items() if name in selected}

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @contextmanager
    def bind_expert(self, features):
        """Hold immutable detached features until all backward work has returned.

        Use one context per microbatch, including evaluation/generation.  Calling
        backward after leaving this context violates the checkpointing contract.
        Do not mutate the source features while this context is active.
        """
        if self._bound:
            raise RuntimeError("Nested or concurrent expert binding is unsupported")
        if features.ndim != 4 or features.shape[1:] != (11, 361, self.expert_width):
            raise ValueError(f"Expected [B,11,361,{self.expert_width}] expert features")
        if features.shape[0] < 1 or not features.is_floating_point():
            raise ValueError("Expert features must have a nonempty floating point batch")
        self._bound = True
        self._bound_batch_size = features.shape[0]
        try:
            detached = features.detach()
            for slot, layer in enumerate(self.injected_layers):
                layer.expert_features = detached[:, slot]
            yield self
        finally:
            for layer in self.injected_layers:
                layer.expert_features = None
            self._bound_batch_size = None
            self._bound = False

    def forward(self, **inputs):
        if not self._bound:
            raise RuntimeError("Use bind_expert around forward AND backward")
        for key in ("input_ids", "inputs_embeds"):
            value = inputs.get(key)
            if value is not None and value.shape[0] != self._bound_batch_size:
                raise ValueError("Language and expert batch sizes differ")
        inputs.setdefault("use_cache", False)
        if torch.is_grad_enabled() and (inputs["use_cache"] or inputs.get("past_key_values") is not None):
            raise ValueError("Use an uncached full prefix for gradient-enabled training")
        return self.language_model(**inputs)

    def generate(self, *, expert_features, **inputs):
        if inputs.get("num_beams", 1) != 1 or inputs.get("num_return_sequences", 1) != 1:
            raise ValueError("Only single-sequence generation is supported")
        # Also override potentially different defaults in GenerationConfig.
        inputs.setdefault("num_beams", 1)
        inputs.setdefault("num_return_sequences", 1)
        with torch.no_grad(), self.bind_expert(expert_features):
            return self.language_model.generate(**inputs)

    def architecture(self):
        return {
            "language_model_type": self.config.model_type,
            "language_model_class": type(self.language_model).__name__,
            "decoder_layers": self.config.num_hidden_layers,
            "language_width": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "expert_outer_blocks": 11,
            "expert_width": self.expert_width,
            "board_tokens": 361,
            "injection_decoder_layers_zero_based": list(self.layer_indices),
            "injection_block_types": [layer.block_type for layer in self.injected_layers],
            "injection_location": "before the original decoder layer",
            "adapter_implementation": "goqueen.qwen38_fusion.NarrowBoardCrossAttention",
            "adapter_attention_width": self.attention_width,
            "adapter_attention_heads": self.injected_layers[0].adapter.heads,
            "adapter_arithmetic": "parameter dtype, autocast disabled; delta cast before residual addition",
            "gate": "hidden + tanh(alpha) * cross_attention(hidden, expert)",
            "initial_gate": 0.0,
            "position_embedding": "shared trainable row + column before expert LayerNorm",
            "all_adapter_names": list(self._adapter_names),
            "trainable_adapter_names": [name for name, parameter in self.adapter_named_parameters().items()
                                        if parameter.requires_grad],
            "binding_lifetime": "one microbatch forward through completed backward",
            "distributed_runtime_verified": False,
        }
