"""Opt-in M-Key variant: a fixed-scale copy of the shared position code in every adapter key.

    key input    LN_l(f_l + p) + c * p
    value input  LN_l(f_l + p)            (the original M formula)

p = rows[row] + columns[column] is the original shared trainable code. It is
computed once per binding and handed live (not detached) to both branches; the
expert features stay detached. The original GoQueen constructor runs unchanged,
so parameter names, order, initial values and RNG consumption match M. Each
adapter's class is then switched in place, which creates no tensor and draws no
random number.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch.nn import functional as F

from goqueen.fusion import GatedBoardAttention, GoQueen

# Under independent N(0, 0.02^2) row and column codes, E[(c * p_d)^2] = 1.
POSITION_SCALE = 1.0 / (0.02 * math.sqrt(2.0))


class MKeyGatedBoardAttention(GatedBoardAttention):
    # Set only by MKeyGoQueen.bind_expert: a plain attribute, never a parameter or buffer.
    position = None

    def forward(self, hidden: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        if expert.shape != (batch, 361, self.key.in_features):
            raise ValueError(f"Bad expert shape: {tuple(expert.shape)}")
        if self.position is None:
            raise RuntimeError("M-Key adapters need the live position code bound by MKeyGoQueen")
        query = self.query(self.query_norm(hidden))
        expert = self.expert_norm(expert)
        key = self.key(expert + POSITION_SCALE * self.position)
        value = self.value(expert)
        query = query.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        key = key.reshape(batch, 361, self.heads, self.head_width).transpose(1, 2)
        value = value.reshape(batch, 361, self.heads, self.head_width).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, length, width)
        branch = self.output(attended)
        # Checked even at alpha=0: tanh(0) * NaN is NaN, so a zero gate does not neutralise a broken branch.
        if not bool(torch.isfinite(branch).all()):
            raise FloatingPointError("Non-finite M-Key attention branch")
        return hidden + torch.tanh(self.alpha) * branch


class MKeyGoQueen(GoQueen):
    def __init__(self, language_model, layer_indices=None, expert_width=768, heads=8):
        super().__init__(language_model, layer_indices, expert_width, heads)
        for layer in self.injected_layers:
            layer.adapter.__class__ = MKeyGatedBoardAttention

    @contextmanager
    def bind_expert(self, features):
        # GoQueen.bind_expert, plus one shared live p handed to every adapter key.
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
                    layer.adapter.position = position
            yield
        finally:
            for layer in self.injected_layers:
                layer.expert_features = None
                layer.adapter.position = None
            self._bound = False

    def architecture(self):
        description = super().architecture()
        description.update({
            "variant": "m_key",
            "expert_key_input": "LN_l(f_l + p) + c * p",
            "expert_value_input": "LN_l(f_l + p)",
            "position_code": "p = rows[row] + columns[column]; the same live shared code, not detached, computed once per binding",
            "position_scale": POSITION_SCALE,
            "position_scale_definition": "c = 1 / (0.02 * sqrt(2)), fixed and untrained",
            "nonfinite_attention_branch": "raises FloatingPointError, including at alpha=0",
        })
        return description
