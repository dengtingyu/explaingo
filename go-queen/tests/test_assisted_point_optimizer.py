"""Tiny-model APS AdamW checks; no pretrained model or factual fitting run."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_assisted_point import (INACTIVE, assert_unbound, board_features,
                                 fresh, points, raw_bytes, token_ids)


class AssistedOptimizerTests(unittest.TestCase):
    def test_fresh_optimizer_has_all_101_adapters_and_no_base(self):
        model = fresh()
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([p for _, p in named], lr=.001,
                                     betas=(.9, .999), eps=1e-8, weight_decay=0)
        members = [p for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(named), 101)
        self.assertEqual([id(p) for p in members], [id(p) for _, p in named])
        self.assertEqual(len(set(map(id, members))), 101)
        self.assertFalse(set(map(id, members)) & set(map(id, model.base_parameters)))
        self.assertEqual(len(optimizer.state), 0)

    def test_three_tiny_updates_keep_44_inactive_tensors_and_base_bytes(self):
        model = fresh()
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        inactive = [(n, p) for n, p in named
                    if any(n.endswith(".adapter." + s) for s in INACTIVE)]
        self.assertEqual(len(inactive), 44)
        old = {n: raw_bytes(p).clone() for n, p in inactive}
        base = [raw_bytes(p).clone() for p in model.base_parameters]
        initial_gates = model.gate_values()
        features = board_features(888, batch=4).requires_grad_()
        ids = token_ids(777, batch=4, length=5)
        selected = points(0, 19, 288, 360)
        optimizer = torch.optim.AdamW([p for _, p in named], lr=.001,
                                     betas=(.9, .999), eps=1e-8, weight_decay=0)
        with sdpa_kernel(SDPBackend.MATH):
            for _ in range(3):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=ids, expert_features=features,
                               point_indices=selected).logits[:, -1]
                F.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
                for n, p in inactive:
                    self.assertIsNotNone(p.grad, n)
                    self.assertTrue(torch.equal(p.grad, torch.zeros_like(p.grad)), n)
                self.assertIsNone(features.grad)
                self.assertTrue(all(p.grad is None for p in model.base_parameters))
                assert_unbound(self, model)
                optimizer.step()
                for n, p in inactive:
                    self.assertTrue(torch.equal(raw_bytes(p), old[n]), n)
        self.assertNotEqual(model.gate_values(), initial_gates)
        for p, saved in zip(model.base_parameters, base):
            self.assertTrue(torch.equal(raw_bytes(p), saved))
        # A zero gradient may create Adam moment buffers and advance its counter.
        # Invariance concerns parameter bytes, not absence of optimizer state.
        self.assertTrue(all(float(optimizer.state[p]["step"]) == 3 for _, p in inactive))


if __name__ == "__main__":
    unittest.main()
