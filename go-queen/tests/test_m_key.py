"""Synthetic checks for the opt-in M-Key variant.

Uses torch and a tiny fake Qwen3-shaped base only: no transformers, downloads,
network, GPU or cached data. Run from the project root:

    PYTHONPATH=src python -m unittest discover -s tests -p 'test_m_key.py' -v
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
CANDIDATE = (HERE / "m_key.py").is_file()
# Candidate layout: m-key-candidate/{m_key.py,test_m_key.py} beside project/src.
# Integrated layout: project/tests/test_m_key.py beside project/src/goqueen/m_key.py.
for _path in ((HERE.parent / "project" / "src", HERE) if CANDIDATE else (HERE.parent / "src",)):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from goqueen import fusion  # noqa: E402

# Captured before the variant is imported, so a patch of the original classes would show.
ORIGINAL_ATTRIBUTES = (fusion.GatedBoardAttention.forward, fusion.ExpertDecoderLayer.forward,
                       fusion.GoQueen.__init__, fusion.GoQueen.bind_expert, fusion.GoQueen.architecture)

if CANDIDATE:
    import m_key  # noqa: E402
else:
    from goqueen import m_key  # noqa: E402

WIDTH, EXPERT, HEADS, LAYERS, VOCAB = 16, 8, 4, 11, 23
SEED = 1234
ROW, COLUMN = torch.arange(361) // 19, torch.arange(361) % 19
ADAPTER_SUFFIXES = ["alpha", "query_norm.weight", "query_norm.bias", "expert_norm.weight",
                    "expert_norm.bias", "query.weight", "key.weight", "value.weight", "output.weight"]


class FakeDecoderLayer(nn.Module):
    attention_type = "full_attention"

    def __init__(self):
        super().__init__()
        self.mix = nn.Linear(WIDTH, WIDTH)
        self.last_kwargs = None

    def forward(self, hidden_states, *args, **kwargs):
        self.last_kwargs = dict(kwargs)
        return hidden_states + torch.tanh(self.mix(hidden_states))


class FakeQwen3(nn.Module):
    """Qwen3-shaped causal LM: config, model.layers, logits and a greedy generate."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=WIDTH, num_hidden_layers=LAYERS)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, WIDTH)
        self.model.layers = nn.ModuleList(FakeDecoderLayer() for _ in range(LAYERS))
        self.lm_head = nn.Linear(WIDTH, VOCAB, bias=False)
        self.calls = []
        self.positions_during_generate = None

    def forward(self, input_ids, use_cache=None, **kwargs):
        self.calls.append({"use_cache": use_cache, **kwargs})
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, use_cache=use_cache)
        return SimpleNamespace(logits=self.lm_head(hidden))

    def generate(self, input_ids, max_new_tokens=2, **kwargs):
        self.positions_during_generate = [layer.adapter.position is not None for layer in self.model.layers]
        for _ in range(max_new_tokens):
            logits = self(input_ids, use_cache=True).logits
            input_ids = torch.cat([input_ids, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        return input_ids


def build(cls, seed=SEED):
    """Seeded fresh fake base, then the wrapper; returns (model, RNG before wrapping, RNG after)."""
    torch.manual_seed(seed)
    base = FakeQwen3()
    before = torch.get_rng_state()
    model = cls(base, expert_width=EXPERT, heads=HEADS)
    return model, before, torch.get_rng_state()


def fresh(cls):
    return build(cls)[0]


def board_features(seed, batch=2):
    return torch.randn(batch, 11, 361, EXPERT, generator=torch.Generator().manual_seed(seed))


def token_ids(seed, batch=2, length=5):
    return torch.randint(0, VOCAB, (batch, length), generator=torch.Generator().manual_seed(seed))


def set_gates(model, value):
    with torch.no_grad():
        for layer in model.injected_layers:
            layer.adapter.alpha.fill_(value)


def perturb_norms(model, seed):
    """Non-identity LayerNorm affines, identical across models given the same seed."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.injected_layers:
            for norm in (layer.adapter.query_norm, layer.adapter.expert_norm):
                norm.weight.copy_(1 + 0.3 * torch.randn(norm.weight.shape, generator=generator))
                norm.bias.copy_(0.3 * torch.randn(norm.bias.shape, generator=generator))


def raw_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def reference_adapter(adapter, hidden, features, position, key_scale, detach_key_copy=False):
    """Direct per-head softmax attention with x_K = LN(f+p) + key_scale*p and x_V = LN(f+p)."""
    normed = F.layer_norm(features + position, (EXPERT,), adapter.expert_norm.weight,
                          adapter.expert_norm.bias, adapter.expert_norm.eps)
    key_copy = position.detach() if detach_key_copy else position
    key = (normed + key_scale * key_copy) @ adapter.key.weight.T
    value = normed @ adapter.value.weight.T
    query = F.layer_norm(hidden, (WIDTH,), adapter.query_norm.weight, adapter.query_norm.bias,
                         adapter.query_norm.eps) @ adapter.query.weight.T
    batch, length, _ = hidden.shape
    width = WIDTH // HEADS
    query = query.reshape(batch, length, HEADS, width).transpose(1, 2)
    key = key.reshape(batch, 361, HEADS, width).transpose(1, 2)
    value = value.reshape(batch, 361, HEADS, width).transpose(1, 2)
    weights = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(width), dim=-1)
    attended = (weights @ value).transpose(1, 2).reshape(batch, length, WIDTH)
    return hidden + torch.tanh(adapter.alpha) * (attended @ adapter.output.weight.T)


def assert_unbound(test, model):
    test.assertFalse(model._bound)
    for layer in model.injected_layers:
        test.assertIsNone(layer.expert_features)
        test.assertIsNone(layer.adapter.position)


class ConstructionTests(unittest.TestCase):
    def test_public_constant_and_api(self):
        self.assertEqual(m_key.POSITION_SCALE, 1 / (0.02 * math.sqrt(2)))
        self.assertAlmostEqual(2 * (0.02 * m_key.POSITION_SCALE) ** 2, 1.0, places=12)
        self.assertTrue(issubclass(m_key.MKeyGoQueen, fusion.GoQueen))
        for name in ("__init__", "bind_expert", "forward", "generate", "trainable_parameters",
                     "adapter_state_dict", "load_adapter_state_dict", "gate_values", "architecture"):
            with self.subTest(name=name):
                self.assertEqual(inspect.signature(getattr(m_key.MKeyGoQueen, name)),
                                 inspect.signature(getattr(fusion.GoQueen, name)))

    def test_construction_matches_original_bytes_order_and_rng(self):
        original, original_before, original_after = build(fusion.GoQueen)
        variant, variant_before, variant_after = build(m_key.MKeyGoQueen)
        self.assertTrue(torch.equal(original_before, variant_before))
        # Construction really draws, so equal post-construction states are informative.
        self.assertFalse(torch.equal(original_before, original_after))
        self.assertTrue(torch.equal(original_after, variant_after))
        original_named, variant_named = list(original.named_parameters()), list(variant.named_parameters())
        self.assertEqual([name for name, _ in original_named], [name for name, _ in variant_named])
        for (name, a), (_, b) in zip(original_named, variant_named):
            with self.subTest(name=name):
                self.assertEqual((a.shape, a.dtype, a.requires_grad), (b.shape, b.dtype, b.requires_grad))
                self.assertTrue(torch.equal(raw_bytes(a), raw_bytes(b)))
        self.assertEqual(list(original.state_dict()), list(variant.state_dict()))
        self.assertEqual([name for name, _ in original.named_buffers()],
                         [name for name, _ in variant.named_buffers()])
        trainable = [name for name, p in variant.named_parameters() if p.requires_grad]
        self.assertEqual(trainable, [f"language_model.model.layers.{i}.adapter.{suffix}"
                                     for i in range(LAYERS) for suffix in ADAPTER_SUFFIXES]
                         + ["rows.weight", "columns.weight"])
        for a, b in zip(original.injected_layers, variant.injected_layers):
            self.assertIs(type(a.adapter), fusion.GatedBoardAttention)
            self.assertIs(type(b.adapter), m_key.MKeyGatedBoardAttention)
            self.assertIsInstance(b.adapter, fusion.GatedBoardAttention)
            self.assertEqual(set(vars(a.adapter)), set(vars(b.adapter)))

    def test_original_classes_are_untouched(self):
        self.assertEqual(ORIGINAL_ATTRIBUTES, (
            fusion.GatedBoardAttention.forward, fusion.ExpertDecoderLayer.forward,
            fusion.GoQueen.__init__, fusion.GoQueen.bind_expert, fusion.GoQueen.architecture))
        fresh(m_key.MKeyGoQueen)
        original = fresh(fusion.GoQueen)
        self.assertTrue(all(type(layer.adapter) is fusion.GatedBoardAttention
                            for layer in original.injected_layers))
        self.assertNotIn("variant", original.architecture())


class FormulaTests(unittest.TestCase):
    def setUp(self):
        self.hidden = torch.randn(2, 5, WIDTH, generator=torch.Generator().manual_seed(12))
        self.features = board_features(11)

    def test_adapter_matches_key_only_formula_and_differs_from_m(self):
        model = fresh(m_key.MKeyGoQueen)
        perturb_norms(model, 7)
        set_gates(model, 0.6)
        layer = model.injected_layers[3]
        with torch.no_grad():
            expected_position = model.rows.weight[ROW] + model.columns.weight[COLUMN]
            with model.bind_expert(self.features):
                torch.testing.assert_close(layer.adapter.position, expected_position)
                actual = layer.adapter(self.hidden, layer.expert_features)
                # The original method on the same parameters is the M formula.
                as_m = fusion.GatedBoardAttention.forward(layer.adapter, self.hidden, layer.expert_features)
            expected = reference_adapter(layer.adapter, self.hidden, self.features[:, 3],
                                         expected_position, m_key.POSITION_SCALE)
            expected_m = reference_adapter(layer.adapter, self.hidden, self.features[:, 3],
                                           expected_position, 0.0)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(as_m, expected_m, rtol=1e-4, atol=1e-5)
        self.assertGreater(float((actual - as_m).abs().max()), 1e-3)

    def test_value_formula_is_m_when_keys_cannot_matter(self):
        # A zero query ties every score, so attention averages the values and the key formula drops out.
        original, variant = fresh(fusion.GoQueen), fresh(m_key.MKeyGoQueen)
        for model in (original, variant):
            perturb_norms(model, 7)
            set_gates(model, 0.6)
            with torch.no_grad():
                for layer in model.injected_layers:
                    layer.adapter.query.weight.zero_()
        ids = token_ids(13)
        with torch.no_grad():
            torch.testing.assert_close(variant(input_ids=ids, expert_features=self.features).logits,
                                       original(input_ids=ids, expert_features=self.features).logits)

    def test_zero_gate_is_exact_identity_and_open_gate_differs_from_m(self):
        original, variant = fresh(fusion.GoQueen), fresh(m_key.MKeyGoQueen)
        for model in (original, variant):
            perturb_norms(model, 7)
        ids = token_ids(13)
        with torch.no_grad():
            plain = variant(input_ids=ids).logits
            self.assertTrue(torch.equal(variant(input_ids=ids, expert_features=self.features).logits, plain))
            self.assertTrue(torch.equal(original(input_ids=ids, expert_features=self.features).logits, plain))
            for model in (original, variant):
                set_gates(model, 0.6)
            difference = (variant(input_ids=ids, expert_features=self.features).logits
                          - original(input_ids=ids, expert_features=self.features).logits).abs().max()
        self.assertGreater(float(difference), 1e-4)

    def test_position_gradient_includes_the_key_copy(self):
        model = fresh(m_key.MKeyGoQueen)
        perturb_norms(model, 7)
        set_gates(model, 0.6)
        weights = torch.randn(2, 5, WIDTH, generator=torch.Generator().manual_seed(14))
        layer = model.injected_layers[3]
        with model.bind_expert(self.features):
            loss = (layer.adapter(self.hidden, layer.expert_features) * weights).sum()
            (actual,) = torch.autograd.grad(loss, model.rows.weight)

        def reference_gradient(detach_key_copy):
            rows = model.rows.weight.detach().clone().requires_grad_(True)
            position = rows[ROW] + model.columns.weight.detach()[COLUMN]
            output = reference_adapter(layer.adapter, self.hidden, self.features[:, 3], position,
                                       m_key.POSITION_SCALE, detach_key_copy)
            return torch.autograd.grad((output * weights).sum(), rows)[0]

        full, without_key_copy = reference_gradient(False), reference_gradient(True)
        torch.testing.assert_close(actual, full, rtol=1e-3, atol=1e-5)
        # The c*p path carries real gradient, so a detached key copy would be caught above.
        self.assertGreater(float((full - without_key_copy).abs().max()), 1e-4)

    def test_one_live_position_is_shared_and_features_stay_detached(self):
        model = fresh(m_key.MKeyGoQueen)
        set_gates(model, 0.6)
        features = self.features.clone().requires_grad_(True)
        with model.bind_expert(features):
            positions = [layer.adapter.position for layer in model.injected_layers]
            self.assertTrue(all(position is positions[0] for position in positions))
            self.assertTrue(positions[0].requires_grad)
            self.assertIsNotNone(positions[0].grad_fn)
            for i, layer in enumerate(model.injected_layers):
                torch.testing.assert_close(layer.expert_features.detach(),
                                           (features[:, i] + positions[0]).detach())
        model(input_ids=token_ids(13), expert_features=features).logits.square().mean().backward()
        self.assertIsNone(features.grad)
        self.assertTrue(all(p.grad is None for p in model.base_parameters))
        for table in (model.rows, model.columns):
            self.assertTrue(bool(torch.isfinite(table.weight.grad).all()))
            self.assertGreater(float(table.weight.grad.abs().max()), 0.0)


class SafetyAndApiTests(unittest.TestCase):
    def test_nonfinite_branch_raises_even_at_zero_gate(self):
        original, variant = fresh(fusion.GoQueen), fresh(m_key.MKeyGoQueen)
        ids, bad = token_ids(13), board_features(11)
        bad[0, 4, 100, 2] = float("nan")
        with torch.no_grad():
            # The original wrapper lets tanh(0) * NaN reach the logits; M-Key stops at the branch.
            self.assertTrue(bool(torch.isnan(original(input_ids=ids, expert_features=bad).logits).any()))
            with self.assertRaises(FloatingPointError):
                variant(input_ids=ids, expert_features=bad)
            assert_unbound(self, variant)
            variant.injected_layers[0].adapter.output.weight[0, 0] = float("inf")
            with self.assertRaises(FloatingPointError):
                variant(input_ids=ids, expert_features=board_features(11))
        assert_unbound(self, variant)

    def test_binding_rules_and_cleanup(self):
        model = fresh(m_key.MKeyGoQueen)
        features = board_features(11)
        with self.assertRaises(RuntimeError):
            model.injected_layers[0].adapter(torch.zeros(2, 5, WIDTH), features[:, 0])
        with model.bind_expert(features):
            with self.assertRaises(RuntimeError):
                with model.bind_expert(features):
                    pass
            # As in GoQueen, a rejected inner binding leaves the outer one intact.
            self.assertTrue(model._bound)
            self.assertTrue(all(layer.adapter.position is not None for layer in model.injected_layers))
        assert_unbound(self, model)
        with self.assertRaises(KeyError):
            with model.bind_expert(features):
                raise KeyError("synthetic failure inside the binding")
        assert_unbound(self, model)
        with self.assertRaises(ValueError):
            with model.bind_expert(features[:, :10]):
                pass
        assert_unbound(self, model)
        with model.bind_expert(None):
            self.assertTrue(model._bound)
            self.assertTrue(all(layer.expert_features is None and layer.adapter.position is None
                                for layer in model.injected_layers))
        assert_unbound(self, model)

    def test_features_none_and_cache_default(self):
        original, variant = fresh(fusion.GoQueen), fresh(m_key.MKeyGoQueen)
        for model in (original, variant):
            set_gates(model, 0.6)
        ids = token_ids(13)
        with torch.no_grad():
            self.assertTrue(torch.equal(variant(input_ids=ids).logits, original(input_ids=ids).logits))
            self.assertIs(variant.language_model.calls[-1]["use_cache"], False)
            self.assertIs(variant.injected_layers[0].decoder.last_kwargs["use_cache"], False)
            variant(input_ids=ids, use_cache=True, expert_features=board_features(11))
            self.assertIs(variant.language_model.calls[-1]["use_cache"], True)
        assert_unbound(self, variant)

    def test_generate_restriction_and_binding(self):
        model = fresh(m_key.MKeyGoQueen)
        features, ids = board_features(11, batch=1), token_ids(13, batch=1)
        for bad in ({"num_beams": 2}, {"num_return_sequences": 2}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                model.generate(expert_features=features, input_ids=ids, **bad)
            assert_unbound(self, model)
        with torch.no_grad():
            generated = model.generate(expert_features=features, input_ids=ids, max_new_tokens=2)
        self.assertEqual(tuple(generated.shape), (1, ids.shape[1] + 2))
        self.assertEqual(model.language_model.positions_during_generate, [True] * LAYERS)
        self.assertIs(model.language_model.calls[-1]["use_cache"], True)
        assert_unbound(self, model)

    def test_adapter_checkpoint_round_trip_and_checks(self):
        source = fresh(m_key.MKeyGoQueen)
        perturb_norms(source, 7)
        set_gates(source, 0.6)
        state = {name: tensor.clone() for name, tensor in source.adapter_state_dict().items()}
        self.assertEqual(list(state), list(fresh(fusion.GoQueen).adapter_state_dict()))
        target = fresh(m_key.MKeyGoQueen)
        target.load_adapter_state_dict(state)
        for name, tensor in target.adapter_state_dict().items():
            with self.subTest(name=name):
                self.assertTrue(torch.equal(raw_bytes(tensor), raw_bytes(state[name])))
        features, ids = board_features(11), token_ids(13)
        with torch.no_grad():
            self.assertTrue(torch.equal(target(input_ids=ids, expert_features=features).logits,
                                        source(input_ids=ids, expert_features=features).logits))
        missing = {name: tensor for name, tensor in state.items() if name != "rows.weight"}
        wrong_shape = dict(state)
        wrong_shape["rows.weight"] = torch.zeros(18, EXPERT)
        for bad in (missing, wrong_shape):
            with self.assertRaises(ValueError):
                target.load_adapter_state_dict(bad)

    def test_architecture_metadata(self):
        before, after = fresh(fusion.GoQueen).architecture(), fresh(m_key.MKeyGoQueen).architecture()
        self.assertEqual({key: after[key] for key in before}, before)
        self.assertNotIn("variant", before)
        self.assertEqual(after["variant"], "m_key")
        self.assertEqual(after["expert_key_input"], "LN_l(f_l + p) + c * p")
        self.assertEqual(after["expert_value_input"], "LN_l(f_l + p)")
        self.assertEqual(after["position_scale"], m_key.POSITION_SCALE)
        self.assertEqual(json.loads(json.dumps(after)), after)


if __name__ == "__main__":
    unittest.main()
