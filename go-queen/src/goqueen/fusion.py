"""Frozen Qwen with zero-initialized gates and trainable board cross-attention."""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class GatedBoardAttention(nn.Module):
    def __init__(self, language_width: int, expert_width: int = 768, heads: int = 8):
        super().__init__()
        if language_width % heads:
            raise ValueError("Adapter width must be divisible by its head count")
        self.heads = heads
        self.head_width = language_width // heads
        self.query_norm = nn.LayerNorm(language_width)
        self.expert_norm = nn.LayerNorm(expert_width)
        self.query = nn.Linear(language_width, language_width, bias=False)
        self.key = nn.Linear(expert_width, language_width, bias=False)
        self.value = nn.Linear(expert_width, language_width, bias=False)
        # Ordinary nonzero initialization lets alpha receive gradients at alpha=0.
        self.output = nn.Linear(language_width, language_width, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        if expert.shape != (batch, 361, self.key.in_features):
            raise ValueError(f"Bad expert shape: {tuple(expert.shape)}")
        query = self.query(self.query_norm(hidden))
        expert = self.expert_norm(expert)
        key, value = self.key(expert), self.value(expert)
        query = query.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = key.reshape(batch, 361, self.heads, self.head_width).transpose(1, 2)
        value = value.reshape(batch, 361, self.heads, self.head_width).transpose(1, 2)
        # The entire board is an observation, so these 361 keys need no causal mask.
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return hidden + torch.tanh(self.alpha) * self.output(attended)


class ExpertDecoderLayer(nn.Module):
    """Inject before the pinned HF decoder while preserving its return/cache API."""
    def __init__(self, decoder: nn.Module, adapter: GatedBoardAttention):
        super().__init__()
        self.decoder = decoder
        self.adapter = adapter
        self.attention_type = decoder.attention_type
        self.expert_features = None

    def forward(self, hidden_states, *args, **kwargs):
        if self.expert_features is not None:
            hidden_states = self.adapter(hidden_states, self.expert_features)
        return self.decoder(hidden_states, *args, **kwargs)


class GoQueen(nn.Module):
    def __init__(self, language_model, layer_indices=None, expert_width=768, heads=8):
        super().__init__()
        if language_model.config.model_type != "qwen3":
            raise ValueError("This milestone implements the pinned Qwen3 architecture")
        self.language_model = language_model
        self.base_parameters = tuple(language_model.parameters())
        for parameter in self.base_parameters:
            parameter.requires_grad_(False)
        layers = language_model.model.layers
        if layer_indices is None:
            layer_indices = [round(i * (len(layers) - 1) / 10) for i in range(11)]
        if len(layer_indices) != 11 or len(set(layer_indices)) != 11:
            raise ValueError("Exactly 11 distinct decoder injection layers are required")
        if any(index < 0 or index >= len(layers) for index in layer_indices):
            raise ValueError("Decoder injection layer is out of range")
        self.layer_indices = list(layer_indices)
        self.expert_width = expert_width
        self.rows = nn.Embedding(19, expert_width)
        self.columns = nn.Embedding(19, expert_width)
        nn.init.normal_(self.rows.weight, std=0.02)
        nn.init.normal_(self.columns.weight, std=0.02)
        self.register_buffer("row_indices", torch.arange(19).repeat_interleave(19), persistent=False)
        self.register_buffer("column_indices", torch.arange(19).repeat(19), persistent=False)
        for index in self.layer_indices:
            if isinstance(layers[index], ExpertDecoderLayer):
                raise ValueError("Adapters have already been installed")
            layers[index] = ExpertDecoderLayer(layers[index], GatedBoardAttention(
                language_model.config.hidden_size, expert_width, heads))
        self._bound = False

    @property
    def injected_layers(self):
        return [self.language_model.model.layers[index] for index in self.layer_indices]

    @contextmanager
    def bind_expert(self, features):
        # A scoped binding keeps HF generation/cache support without a source fork.
        # This minimal single-process wrapper is not safe for concurrent requests.
        if self._bound:
            raise RuntimeError("Nested or concurrent feature binding is unsupported")
        self._bound = True
        try:
            if features is not None:
                if features.ndim != 4 or features.shape[1:] != (11, 361, self.expert_width):
                    raise ValueError(f"Expected [B,11,361,{self.expert_width}], got {tuple(features.shape)}")
                features = features.detach()
                position = self.rows(self.row_indices) + self.columns(self.column_indices)
                for i, layer in enumerate(self.injected_layers):
                    layer.expert_features = features[:, i] + position.unsqueeze(0)
            yield
        finally:
            for layer in self.injected_layers:
                layer.expert_features = None
            self._bound = False

    def forward(self, *, expert_features=None, **inputs):
        inputs.setdefault("use_cache", False)
        with self.bind_expert(expert_features):
            return self.language_model(**inputs)

    def generate(self, *, expert_features, **inputs):
        if inputs.get("num_beams", 1) != 1 or inputs.get("num_return_sequences", 1) != 1:
            raise ValueError("This milestone supports single-sequence generation only")
        with self.bind_expert(expert_features):
            return self.language_model.generate(**inputs)

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def adapter_state_dict(self):
        return {name: parameter.detach().cpu().contiguous()
                for name, parameter in self.named_parameters() if parameter.requires_grad}

    def load_adapter_state_dict(self, state):
        parameters = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if set(state) != set(parameters):
            raise ValueError("Adapter checkpoint parameter names do not match this architecture")
        with torch.no_grad():
            for name, parameter in parameters.items():
                if state[name].shape != parameter.shape:
                    raise ValueError(f"Adapter checkpoint shape mismatch: {name}")
                parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))

    def gate_values(self):
        return [float(layer.adapter.alpha.detach()) for layer in self.injected_layers]

    def architecture(self):
        return {
            "language_model_type": self.language_model.config.model_type,
            "decoder_layers": self.language_model.config.num_hidden_layers,
            "language_width": self.language_model.config.hidden_size,
            "expert_outer_blocks": 11, "expert_width": self.expert_width,
            "board_tokens": 361, "injection_decoder_layers_zero_based": self.layer_indices,
            "injection_location": "before the original decoder layer",
            "mapping_rule": "round(i*(decoder_layers-1)/10), i=0..10",
            "adapter_attention_heads": self.injected_layers[0].adapter.heads,
            "gate": "hidden + tanh(alpha) * cross_attention(hidden, expert)",
            "gate_initialization": 0.0,
            "position_embedding": "shared trainable row + column embeddings, added to expert tokens before expert LayerNorm",
            "base_language_parameters": sum(p.numel() for p in self.base_parameters),
            "trainable_parameters": sum(p.numel() for p in self.trainable_parameters()),
            "vocabulary": "unchanged; GTP coordinates use existing text tokens",
        }
