"""Opt-in M-APS variant: each adapter may attend only to an externally supplied board point.

The original M formulas are kept exactly:

    query   W_Q(LN_query(hidden))
    key     W_K(LN_expert(f_l + p))
    value   W_V(LN_expert(f_l + p))
    output  hidden + tanh(alpha) * W_O(attended)

p = rows[row] + columns[column] is the live shared position code; the expert
features stay detached. The only change is a boolean attention mask
[B, 1, 1, 361] with exactly one True (= allowed) per sample, at that sample's
point index, shared by all 11 adapters and broadcast over every head and query
position. The caller supplies the indices per actual forward/generation call;
this module never reads labels or question records.

The original GoQueen constructor runs unchanged, so parameter names, order,
initial values and RNG consumption match M. Each adapter's class is then
switched in place, which creates no tensor and draws no random number.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
from torch.nn import functional as F

from goqueen.fusion import GatedBoardAttention, GoQueen

VARIANT = "m_assisted_point"
ROUTING_CONTRACT = "single_point_bool_mask_v1"


class AssistedPointGatedBoardAttention(GatedBoardAttention):
    # Set only by AssistedPointGoQueen.bind_expert: plain attributes, never parameters or buffers.
    point_mask = None
    point_indices = None

    def forward(self, hidden: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        if expert.shape != (batch, 361, self.key.in_features):
            raise ValueError(f"Bad expert shape: {tuple(expert.shape)}")
        mask = self.point_mask
        if mask is None:
            raise RuntimeError("APS adapters need a point mask bound by AssistedPointGoQueen")
        if mask.dtype != torch.bool or mask.shape != (batch, 1, 1, 361):
            raise RuntimeError(f"Bound APS point mask {tuple(mask.shape)} does not match batch {batch}")
        query = self.query(self.query_norm(hidden))
        expert = self.expert_norm(expert)
        key, value = self.key(expert), self.value(expert)
        query = query.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = key.reshape(batch, 361, self.heads, self.head_width).transpose(1, 2)
        value = value.reshape(batch, 361, self.heads, self.head_width).transpose(1, 2)
        # Looked up on F at call time, so an observing wrapper sees the real call.
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        branch = self.output(attended)
        # Checked even at alpha=0: tanh(0) * NaN is NaN. A non-finite unselected
        # token also reaches the branch through 0 * NaN or a NaN score.
        if not bool(torch.isfinite(branch).all()):
            raise FloatingPointError("Non-finite APS attention branch")
        return hidden + torch.tanh(self.alpha) * branch


def _point_mask(point_indices, batch, device):
    """Validate per-sample indices; return a [B,1,1,361] bool mask and a private index copy."""
    if point_indices is None:
        raise ValueError("APS expert features require point_indices for every sample")
    if not isinstance(point_indices, torch.Tensor):
        raise TypeError("point_indices must be a torch.Tensor")
    if point_indices.dtype != torch.long:
        raise TypeError(f"point_indices must have dtype torch.long, got {point_indices.dtype}")
    if point_indices.ndim != 1 or point_indices.shape[0] != batch:
        raise ValueError(f"point_indices must have shape [{batch}], got {tuple(point_indices.shape)}")
    if batch == 0:
        raise ValueError("APS requires a nonempty batch")
    points = point_indices.detach().to(device=device).clone()
    if bool((points < 0).any()) or bool((points > 360).any()):
        raise ValueError("point_indices must lie in [0, 360]")
    mask = torch.zeros((batch, 361), dtype=torch.bool, device=device)
    mask[torch.arange(batch, device=device), points] = True
    return mask.view(batch, 1, 1, 361), points


class AssistedPointGoQueen(GoQueen):
    def __init__(self, language_model, layer_indices=None, expert_width=768, heads=8):
        super().__init__(language_model, layer_indices, expert_width, heads)
        for layer in self.injected_layers:
            layer.adapter.__class__ = AssistedPointGatedBoardAttention

    @contextmanager
    def bind_expert(self, features, point_indices=None):
        # GoQueen.bind_expert, plus one shared runtime mask for this actual batch.
        # This minimal single-process wrapper is not safe for concurrent requests.
        if self._bound:
            raise RuntimeError("Nested or concurrent feature binding is unsupported")
        self._bound = True
        try:
            if features is None:
                if point_indices is not None:
                    raise ValueError("APS point_indices require expert_features")
            else:
                if features.ndim != 4 or features.shape[1:] != (11, 361, self.expert_width):
                    raise ValueError(f"Expected [B,11,361,{self.expert_width}], got {tuple(features.shape)}")
                mask, points = _point_mask(point_indices, features.shape[0], features.device)
                features = features.detach()
                position = self.rows(self.row_indices) + self.columns(self.column_indices)
                for i, layer in enumerate(self.injected_layers):
                    layer.expert_features = features[:, i] + position.unsqueeze(0)
                    layer.adapter.point_mask = mask
                    layer.adapter.point_indices = points
            yield
        finally:
            for layer in self.injected_layers:
                layer.expert_features = None
                layer.adapter.point_mask = None
                layer.adapter.point_indices = None
            self._bound = False

    def forward(self, *, expert_features=None, point_indices=None, **inputs):
        inputs.setdefault("use_cache", False)
        with self.bind_expert(expert_features, point_indices):
            return self.language_model(**inputs)

    def generate(self, *, expert_features, point_indices, **inputs):
        if inputs.get("num_beams", 1) != 1 or inputs.get("num_return_sequences", 1) != 1:
            raise ValueError("This milestone supports single-sequence generation only")
        with self.bind_expert(expert_features, point_indices):
            return self.language_model.generate(**inputs)

    def architecture(self):
        description = super().architecture()
        description.update({
            "variant": VARIANT,
            "routing_contract": ROUTING_CONTRACT,
            "point_selection": "external and label-free: point_indices torch.long [B] in 0..360, "
                               "supplied per actual forward/generation call in expert_features batch order",
            "attention_mask": "torch.bool [B,1,1,361], True = allowed, exactly one True per sample at its point; "
                              "broadcast over all heads and query positions; one shared mask for all 11 adapters",
            "attention": "torch scaled_dot_product_attention over the full 361-token board, dropout_p=0.0",
            "query": "W_Q(LN_query(hidden))",
            "key": "W_K(LN_expert(f_l + p))",
            "value": "W_V(LN_expert(f_l + p))",
            "output": "hidden + tanh(alpha) * W_O(attended)",
            "position_code": "p = rows[row] + columns[column]; live shared code; expert features detached",
            "expected_zero_gradient_tensors": "query_norm.weight, query_norm.bias, query.weight, key.weight in "
                                              "each adapter (single allowed token, math SDPA); not a verified result",
            "runtime_binding": "scoped; expert features, mask and point indices cleared in finally on success and "
                               "error; nested binding rejected; single-process only",
            "nonfinite_attention_branch": "raises FloatingPointError, including at alpha=0",
        })
        return description
