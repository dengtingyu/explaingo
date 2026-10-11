"""Source-only checks for the opt-in M-APS variant.

Written without being executed by its author; results count only once run.
Torch and a tiny fake Qwen3-shaped base only: no transformers, downloads,
network, GPU, cached features or optimizer step. Exactness checks force math
SDPA, as the planned runner and scorer do. From the directory holding
aps-candidate/ and project/:

    python -m unittest discover -s aps-candidate -p 'test_assisted_point.py' -v
"""
from __future__ import annotations

from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
CANDIDATE = (HERE / "assisted_point.py").is_file()
# Candidate layout: aps-candidate/{assisted_point.py,test_assisted_point.py} beside project/src.
# Integrated layout: project/tests/test_assisted_point.py beside project/src/goqueen/assisted_point.py.
for _path in ((HERE.parent / "project" / "src", HERE) if CANDIDATE else (HERE.parent / "src",)):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from goqueen import fusion  # noqa: E402

# Captured before the variant is imported, so a patch of the original classes would show.
ORIGINAL_ATTRIBUTES = (fusion.GatedBoardAttention.forward, fusion.ExpertDecoderLayer.forward,
                       fusion.GoQueen.__init__, fusion.GoQueen.bind_expert, fusion.GoQueen.forward,
                       fusion.GoQueen.generate, fusion.GoQueen.architecture)

if CANDIDATE:
    import assisted_point as aps  # noqa: E402
else:
    from goqueen import assisted_point as aps  # noqa: E402

WIDTH, EXPERT, HEADS, LAYERS, VOCAB = 8, 4, 2, 11, 13
SEED = 2024
ROW, COLUMN = torch.arange(361) // 19, torch.arange(361) % 19
SUFFIXES = ["alpha", "query_norm.weight", "query_norm.bias", "expert_norm.weight", "expert_norm.bias",
            "query.weight", "key.weight", "value.weight", "output.weight"]
INACTIVE = ("query_norm.weight", "query_norm.bias", "query.weight", "key.weight")


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
    """Qwen3-shaped causal LM; its greedy generate feeds one new token per cached step."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=WIDTH, num_hidden_layers=LAYERS)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(VOCAB, WIDTH)
        self.model.layers = nn.ModuleList(FakeDecoderLayer() for _ in range(LAYERS))
        self.lm_head = nn.Linear(WIDTH, VOCAB, bias=False)
        self.calls = []
        self.generation_steps = []

    def forward(self, input_ids, use_cache=None, **kwargs):
        self.calls.append({"use_cache": use_cache, "length": input_ids.shape[1], **kwargs})
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden, use_cache=use_cache)
        return SimpleNamespace(logits=self.lm_head(hidden))

    def generate(self, input_ids, max_new_tokens=3, use_cache=True, **kwargs):
        sequence, step_input = input_ids, input_ids
        for _ in range(max_new_tokens):
            self.generation_steps.append([layer.adapter.point_mask for layer in self.model.layers])
            logits = self(input_ids=step_input, use_cache=use_cache).logits
            token = logits[:, -1].argmax(-1, keepdim=True)
            sequence = torch.cat([sequence, token], dim=1)
            step_input = token if use_cache else sequence
        return sequence


def build(cls, seed=SEED):
    """Seeded fresh fake base, then the wrapper; returns (model, RNG before wrapping, RNG after)."""
    torch.manual_seed(seed)
    base = FakeQwen3()
    before = torch.get_rng_state()
    model = cls(base, expert_width=EXPERT, heads=HEADS)
    return model, before, torch.get_rng_state()


def fresh(cls=None):
    return build(cls or aps.AssistedPointGoQueen)[0]


def generator(seed):
    return torch.Generator().manual_seed(seed)


def board_features(seed, batch=2):
    return torch.randn(batch, 11, 361, EXPERT, generator=generator(seed))


def token_ids(seed, batch=2, length=5):
    return torch.randint(0, VOCAB, (batch, length), generator=generator(seed))


def points(*values):
    return torch.tensor(values, dtype=torch.long)


def set_known_weights(model, seed):
    """Known non-identity LN affines, projections and position tables, identical for a given seed."""
    g = generator(seed)
    with torch.no_grad():
        model.rows.weight.copy_(0.5 * torch.randn(model.rows.weight.shape, generator=g))
        model.columns.weight.copy_(0.5 * torch.randn(model.columns.weight.shape, generator=g))
        for layer in model.injected_layers:
            adapter = layer.adapter
            for norm in (adapter.query_norm, adapter.expert_norm):
                norm.weight.copy_(1 + 0.3 * torch.randn(norm.weight.shape, generator=g))
                norm.bias.copy_(0.3 * torch.randn(norm.bias.shape, generator=g))
            for linear in (adapter.query, adapter.key, adapter.value, adapter.output):
                linear.weight.copy_(torch.randn(linear.weight.shape, generator=g) / 2)


def set_gates(model, value):
    with torch.no_grad():
        for layer in model.injected_layers:
            layer.adapter.alpha.fill_(value)


def raw_bytes(tensor):
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)


def layer_norm(x, norm):
    mean = x.mean(-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(-1, keepdim=True)
    return (x - mean) / torch.sqrt(variance + norm.eps) * norm.weight + norm.bias


def expected_branch(model, layer_number, features, sample, point):
    """Original M value branch W_O W_V LN_expert(f + p) at one point, without the adapter or SDPA."""
    adapter = model.injected_layers[layer_number].adapter
    x = features[sample, layer_number, point] + model.rows.weight[point // 19] + model.columns.weight[point % 19]
    return adapter.output.weight @ (adapter.value.weight @ layer_norm(x, adapter.expert_norm))


@contextmanager
def observe_sdpa(records):
    """Record the real SDPA calls made through torch.nn.functional, without changing them."""
    real = F.scaled_dot_product_attention

    def observed(query, key, value, *args, **kwargs):
        attended = real(query, key, value, *args, **kwargs)
        records.append({"query": query, "key": key, "value": value, "args": args,
                        "kwargs": kwargs, "attended": attended})
        return attended

    with patch.object(F, "scaled_dot_product_attention", observed):
        yield


def assert_unbound(test, model):
    test.assertFalse(model._bound)
    for layer in model.injected_layers:
        test.assertIsNone(layer.expert_features)
        test.assertIsNone(layer.adapter.point_mask)
        test.assertIsNone(layer.adapter.point_indices)


class ConstructionTests(unittest.TestCase):
    def test_construction_matches_original_names_values_order_and_rng(self):
        original, original_before, original_after = build(fusion.GoQueen)
        variant, variant_before, variant_after = build(aps.AssistedPointGoQueen)
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
                                     for i in range(LAYERS) for suffix in SUFFIXES]
                         + ["rows.weight", "columns.weight"])
        self.assertEqual(len(trainable), 101)
        for a, b in zip(original.injected_layers, variant.injected_layers):
            self.assertIs(type(a.adapter), fusion.GatedBoardAttention)
            self.assertIs(type(b.adapter), aps.AssistedPointGatedBoardAttention)
            self.assertIsInstance(b.adapter, fusion.GatedBoardAttention)
            self.assertEqual(set(vars(a.adapter)), set(vars(b.adapter)))

    def test_binding_registers_no_tensor(self):
        model = fresh()

        def registered():
            return (list(model.state_dict()), [name for name, _ in model.named_buffers()],
                    [name for name, _ in model.named_parameters()])

        before = registered()
        with model.bind_expert(board_features(1), points(3, 200)):
            self.assertEqual(registered(), before)
        self.assertEqual(registered(), before)

    def test_zero_gate_output_equals_base_and_original_M(self):
        original, variant = fresh(fusion.GoQueen), fresh()
        features, ids = board_features(1), token_ids(2)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            plain = variant(input_ids=ids).logits
            routed = variant(input_ids=ids, expert_features=features, point_indices=points(5, 300)).logits
            unrouted = original(input_ids=ids, expert_features=features).logits
        self.assertTrue(torch.equal(routed, plain))
        self.assertTrue(torch.equal(unrouted, plain))

    def test_public_api_and_original_classes_untouched(self):
        self.assertEqual(ORIGINAL_ATTRIBUTES, (
            fusion.GatedBoardAttention.forward, fusion.ExpertDecoderLayer.forward,
            fusion.GoQueen.__init__, fusion.GoQueen.bind_expert, fusion.GoQueen.forward,
            fusion.GoQueen.generate, fusion.GoQueen.architecture))
        cls = aps.AssistedPointGoQueen
        self.assertEqual(inspect.signature(cls.__init__), inspect.signature(fusion.GoQueen.__init__))
        forward = inspect.signature(cls.forward).parameters
        self.assertEqual(forward["expert_features"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(forward["point_indices"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(forward["point_indices"].default)
        generate = inspect.signature(cls.generate).parameters
        self.assertIs(generate["expert_features"].default, inspect.Parameter.empty)
        self.assertIs(generate["point_indices"].default, inspect.Parameter.empty)
        bind = inspect.signature(cls.bind_expert).parameters
        self.assertEqual(list(bind), ["self", "features", "point_indices"])
        self.assertIsNone(bind["point_indices"].default)
        # No M-Key position copy or M-Key object enters this module.
        self.assertFalse(hasattr(aps, "POSITION_SCALE"))
        self.assertFalse(any(getattr(value, "__module__", "") == "goqueen.m_key" for value in vars(aps).values()))
        fresh()
        original = fresh(fusion.GoQueen)
        self.assertTrue(all(type(layer.adapter) is fusion.GatedBoardAttention for layer in original.injected_layers))

    def test_architecture_describes_the_variant(self):
        before, after = fresh(fusion.GoQueen).architecture(), fresh().architecture()
        self.assertEqual({key: after[key] for key in before}, before)
        self.assertNotIn("variant", before)
        self.assertEqual(after["variant"], aps.VARIANT)
        self.assertEqual(after["variant"], "m_assisted_point")
        self.assertEqual(after["routing_contract"], "single_point_bool_mask_v1")
        self.assertEqual((after["query"], after["key"], after["value"], after["output"]),
                         ("W_Q(LN_query(hidden))", "W_K(LN_expert(f_l + p))", "W_V(LN_expert(f_l + p))",
                          "hidden + tanh(alpha) * W_O(attended)"))
        self.assertNotIn("position_scale", after)
        self.assertEqual(json.loads(json.dumps(after)), after)


class MaskTests(unittest.TestCase):
    def test_mask_dtype_shape_single_true_shared_and_private_points(self):
        model = fresh()
        chosen = points(0, 300, 360)
        features = board_features(1, batch=3)
        with model.bind_expert(features, chosen):
            masks = [layer.adapter.point_mask for layer in model.injected_layers]
            mask = masks[0]
            self.assertTrue(all(item is mask for item in masks))
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(tuple(mask.shape), (3, 1, 1, 361))
            self.assertEqual(mask.device, features.device)
            self.assertEqual(mask.sum(dim=-1).reshape(-1).tolist(), [1, 1, 1])
            self.assertEqual(mask.reshape(3, 361).nonzero().tolist(), [[0, 0], [1, 300], [2, 360]])
            for layer in model.injected_layers:
                self.assertTrue(torch.equal(layer.adapter.point_indices, chosen))
                self.assertIsNot(layer.adapter.point_indices, chosen)
        assert_unbound(self, model)

    def test_every_adapter_call_passes_the_same_bool_mask_to_sdpa(self):
        model = fresh()
        records = []
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), observe_sdpa(records):
            with model.bind_expert(board_features(1), points(17, 300)):
                bound = model.injected_layers[0].adapter.point_mask
                model.language_model(input_ids=token_ids(2), use_cache=False)
        self.assertEqual(len(records), LAYERS)
        for record in records:
            self.assertEqual(record["args"], ())
            self.assertIs(record["kwargs"]["attn_mask"], bound)
            self.assertEqual(record["kwargs"]["attn_mask"].dtype, torch.bool)
            self.assertEqual(record["kwargs"]["dropout_p"], 0.0)
            self.assertNotIn("is_causal", record["kwargs"])
            self.assertEqual(tuple(record["key"].shape), (2, HEADS, 361, WIDTH // HEADS))
        assert_unbound(self, model)

    def test_sdpa_receives_original_qkv_and_returns_exactly_the_selected_value(self):
        model = fresh()
        set_known_weights(model, 3)
        features, chosen = board_features(1), points(17, 300)
        hidden = torch.randn(2, 5, WIDTH, generator=generator(4))
        layer = model.injected_layers[6]
        adapter = layer.adapter
        records = []
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), observe_sdpa(records):
            with model.bind_expert(features, chosen):
                adapter(hidden, layer.expert_features)
            position = model.rows.weight[ROW] + model.columns.weight[COLUMN]
            normed = layer_norm(features[:, 6] + position, adapter.expert_norm)
            heads = lambda x, n: x.reshape(2, n, HEADS, WIDTH // HEADS).transpose(1, 2)  # noqa: E731
            expected_query = heads(layer_norm(hidden, adapter.query_norm) @ adapter.query.weight.T, 5)
            expected_key = heads(normed @ adapter.key.weight.T, 361)
            expected_value = heads(normed @ adapter.value.weight.T, 361)
        self.assertEqual(len(records), 1)
        record = records[0]
        torch.testing.assert_close(record["query"], expected_query, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(record["key"], expected_key, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(record["value"], expected_value, rtol=1e-5, atol=1e-5)
        # Theoretical exactness under math SDPA with one allowed finite score:
        # weights are exactly [1, 0, ...], so every head and query position gets V[point].
        for sample, point in enumerate(chosen.tolist()):
            attended = record["attended"][sample]
            selected = record["value"][sample, :, point, :].unsqueeze(1).expand_as(attended)
            self.assertTrue(torch.equal(attended, selected), f"sample {sample}")


class SelectedValueTests(unittest.TestCase):
    def setUp(self):
        self.model = fresh()
        set_known_weights(self.model, 3)
        set_gates(self.model, 0.7)
        self.features = board_features(1)
        self.hidden = torch.randn(2, 5, WIDTH, generator=generator(4))

    def adapter_output(self, layer_number, chosen, features=None):
        layer = self.model.injected_layers[layer_number]
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            with self.model.bind_expert(self.features if features is None else features, chosen):
                return layer.adapter(self.hidden, layer.expert_features)

    def expected_output(self, layer_number, chosen, features=None):
        features = self.features if features is None else features
        adapter = self.model.injected_layers[layer_number].adapter
        with torch.no_grad():
            branches = torch.stack([expected_branch(self.model, layer_number, features, sample, point)
                                    for sample, point in enumerate(chosen.tolist())])
            return self.hidden + torch.tanh(adapter.alpha) * branches[:, None, :]

    def test_output_is_the_selected_ln_value_branch_for_each_sample_and_query_position(self):
        chosen = points(17, 300)
        for layer_number in (0, 6, 10):
            with self.subTest(layer=layer_number):
                actual = self.adapter_output(layer_number, chosen)
                expected = self.expected_output(layer_number, chosen)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
                delta = actual - self.hidden
                self.assertTrue(torch.allclose(delta, delta[:, :1].expand_as(delta), rtol=0, atol=1e-6))
                # Distinct samples really select distinct values.
                self.assertGreater(float((expected[0, 0] - self.hidden[0, 0]
                                          - expected[1, 0] + self.hidden[1, 0]).abs().max()), 1e-3)

    def test_a_different_allowed_point_yields_its_own_value(self):
        first, second = points(17, 300), points(18, 42)
        a, b = self.adapter_output(6, first), self.adapter_output(6, second)
        torch.testing.assert_close(b, self.expected_output(6, second), rtol=1e-5, atol=1e-5)
        self.assertGreater(float((a - b).abs().max()), 1e-3)

    def test_unselected_tokens_and_query_key_weights_cannot_contribute(self):
        chosen = points(17, 300)
        baseline = self.adapter_output(6, chosen)
        samples = torch.arange(2)
        changed = self.features.clone()
        kept = changed[samples, :, chosen].clone()
        changed.add_(5 * torch.randn(changed.shape, generator=generator(9)))
        changed[samples, :, chosen] = kept
        self.assertTrue(torch.equal(self.adapter_output(6, chosen, changed), baseline))
        adapter = self.model.injected_layers[6].adapter
        with torch.no_grad():
            for tensor in (adapter.query.weight, adapter.key.weight,
                           adapter.query_norm.weight, adapter.query_norm.bias):
                tensor.mul_(-3).add_(0.25)
        self.assertTrue(torch.equal(self.adapter_output(6, chosen), baseline))
        # Control: the selected token itself does matter. Change one coordinate only,
        # because LayerNorm would remove a shift applied equally to every coordinate.
        moved = self.features.clone()
        moved[0, 6, 17, 0] += 1.0
        self.assertFalse(torch.equal(self.adapter_output(6, chosen, moved), baseline))


class GradientTests(unittest.TestCase):
    def run_backward(self, gate):
        model = fresh()
        set_known_weights(model, 3)
        set_gates(model, gate)
        features = board_features(1).requires_grad_(True)
        weights = torch.randn(2, 5, VOCAB, generator=generator(5))
        with sdpa_kernel(SDPBackend.MATH):
            logits = model(input_ids=token_ids(2), expert_features=features,
                           point_indices=points(17, 300)).logits
            (logits * weights).sum().backward()
        assert_unbound(self, model)
        return model, features

    def test_nonzero_gate_paths_live_and_qk_derivatives_exactly_zero(self):
        model, features = self.run_backward(0.6)
        self.assertIsNone(features.grad)
        self.assertTrue(all(p.grad is None for p in model.base_parameters))
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        for name, p in trainable:
            with self.subTest(name=name):
                self.assertIsNotNone(p.grad)
                self.assertTrue(bool(torch.isfinite(p.grad).all()))
        inactive = [(name, p) for name, p in trainable if name.endswith(INACTIVE)]
        self.assertEqual(len(inactive), 44)
        for name, p in inactive:
            with self.subTest(inactive=name):
                # Theoretical exact zero: softmax backward with weights [1, 0, ...]. Not a tolerance check.
                self.assertTrue(bool(p.grad.eq(0).all()))
        for group in (".alpha", ".value.weight", ".output.weight", ".expert_norm.weight", ".expert_norm.bias"):
            with self.subTest(active=group):
                self.assertTrue(any(bool(p.grad.ne(0).any()) for name, p in trainable if name.endswith(group)))
        for table in (model.rows, model.columns):
            self.assertTrue(bool(table.weight.grad.ne(0).any()))

    def test_zero_gate_only_gates_receive_gradient(self):
        model, features = self.run_backward(0.0)
        self.assertIsNone(features.grad)
        for name, p in model.named_parameters():
            if p.requires_grad:
                with self.subTest(name=name):
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(bool(torch.isfinite(p.grad).all()))
                    if not name.endswith(".alpha"):
                        self.assertTrue(bool(p.grad.eq(0).all()))
        self.assertTrue(any(bool(p.grad.ne(0).any()) for name, p in model.named_parameters()
                            if name.endswith(".alpha")))

    def test_bound_expert_input_keeps_live_position_and_detached_features(self):
        model = fresh()
        features = board_features(1).requires_grad_(True)
        with model.bind_expert(features, points(17, 300)):
            position = model.rows.weight[ROW] + model.columns.weight[COLUMN]
            for i, layer in enumerate(model.injected_layers):
                self.assertTrue(layer.expert_features.requires_grad)
                self.assertIsNotNone(layer.expert_features.grad_fn)
                torch.testing.assert_close(layer.expert_features.detach(), (features[:, i] + position).detach())
        assert_unbound(self, model)


class BindingTests(unittest.TestCase):
    def assert_reusable(self, model):
        assert_unbound(self, model)
        with model.bind_expert(board_features(1), points(3, 200)):
            mask = model.injected_layers[0].adapter.point_mask
            self.assertEqual(mask.reshape(2, 361).nonzero().tolist(), [[0, 3], [1, 200]])
        assert_unbound(self, model)

    def test_malformed_points_rejected_and_cleaned(self):
        model = fresh()
        features = board_features(1)
        cases = [
            (None, ValueError),
            ([3, 200], TypeError),
            (torch.tensor([3, 200], dtype=torch.int32), TypeError),
            (torch.tensor([True, False]), TypeError),
            (torch.tensor([3.0, 200.0]), TypeError),
            (torch.tensor(3), ValueError),
            (torch.tensor([[3], [200]]), ValueError),
            (points(3), ValueError),
            (points(3, 200, 5), ValueError),
            (points(-1, 200), ValueError),
            (points(3, 361), ValueError),
        ]
        for value, error in cases:
            with self.subTest(points=value), self.assertRaises(error):
                with model.bind_expert(features, value):
                    self.fail("A malformed binding must not open")
            assert_unbound(self, model)
        with self.assertRaises(ValueError):
            with model.bind_expert(torch.randn(0, 11, 361, EXPERT), torch.zeros(0, dtype=torch.long)):
                self.fail("An empty batch must not open")
        with self.assertRaises(ValueError):
            with model.bind_expert(board_features(1)[:, :10], points(3, 200)):
                self.fail("Malformed features must not open")
        self.assert_reusable(model)

    def test_none_bypass_only_without_points(self):
        original, model = fresh(fusion.GoQueen), fresh()
        set_gates(original, 0.6)
        set_gates(model, 0.6)
        ids = token_ids(2)
        with model.bind_expert(None):
            self.assertTrue(model._bound)
            self.assertTrue(all(layer.expert_features is None and layer.adapter.point_mask is None
                                for layer in model.injected_layers))
        assert_unbound(self, model)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(input_ids=ids).logits, original(input_ids=ids).logits))
        with self.assertRaises(ValueError):
            with model.bind_expert(None, points(3, 200)):
                self.fail("Points without features must not open")
        with self.assertRaises(ValueError):
            model(input_ids=ids, point_indices=points(3, 200))
        self.assert_reusable(model)

    def test_nested_binding_rejected_without_erasing_outer(self):
        model = fresh()
        with model.bind_expert(board_features(1), points(3, 200)):
            outer = model.injected_layers[0].adapter.point_mask
            with self.assertRaises(RuntimeError):
                with model.bind_expert(board_features(2), points(5, 6)):
                    pass
            self.assertTrue(model._bound)
            self.assertTrue(all(layer.adapter.point_mask is outer and layer.expert_features is not None
                                for layer in model.injected_layers))
        self.assert_reusable(model)

    def test_exception_inside_binding_and_failing_forward_clean_up(self):
        model = fresh()
        with self.assertRaises(KeyError):
            with model.bind_expert(board_features(1), points(3, 200)):
                raise KeyError("synthetic failure inside the binding")
        self.assert_reusable(model)
        with self.assertRaises(ValueError):
            # Input batch 3 against a bound batch of 2 fails inside the first adapter.
            model(input_ids=token_ids(2, batch=3), expert_features=board_features(1),
                  point_indices=points(3, 200))
        self.assert_reusable(model)

    def test_adapter_rejects_missing_or_mismatched_mask(self):
        model = fresh()
        layer = model.injected_layers[0]
        hidden = torch.zeros(1, 5, WIDTH)
        with self.assertRaises(RuntimeError):
            layer.adapter(hidden, board_features(1, batch=1)[:, 0])
        with model.bind_expert(board_features(1), points(3, 200)):
            with self.assertRaises(RuntimeError):
                layer.adapter(hidden, layer.expert_features[:1])
        self.assert_reusable(model)


class ForwardAndGenerationTests(unittest.TestCase):
    def test_forward_cache_default_and_argument_passthrough(self):
        model = fresh()
        ids = token_ids(2)
        with torch.no_grad():
            model(input_ids=ids, expert_features=board_features(1), point_indices=points(3, 200),
                  logits_to_keep=1)
            call = model.language_model.calls[-1]
            self.assertIs(call["use_cache"], False)
            self.assertEqual(call["logits_to_keep"], 1)
            self.assertNotIn("point_indices", call)
            self.assertNotIn("expert_features", call)
            self.assertIs(model.injected_layers[0].decoder.last_kwargs["use_cache"], False)
            model(input_ids=ids, use_cache=True, expert_features=board_features(1), point_indices=points(3, 200))
            self.assertIs(model.language_model.calls[-1]["use_cache"], True)
        assert_unbound(self, model)

    def test_generation_restrictions_and_mask_persisting_across_cached_steps(self):
        model = fresh()
        set_known_weights(model, 3)
        set_gates(model, 0.6)
        features, ids, chosen = board_features(1, batch=1), token_ids(2, batch=1), points(123)
        for bad in ({"num_beams": 2}, {"num_return_sequences": 2}):
            with self.subTest(**bad), self.assertRaises(ValueError):
                model.generate(expert_features=features, point_indices=chosen, input_ids=ids, **bad)
            assert_unbound(self, model)
        with self.assertRaises(TypeError):
            model.generate(expert_features=features, input_ids=ids)
        assert_unbound(self, model)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            generated = model.generate(expert_features=features, point_indices=chosen, input_ids=ids,
                                       max_new_tokens=3, use_cache=True)
        self.assertEqual(tuple(generated.shape), (1, ids.shape[1] + 3))
        steps = model.language_model.generation_steps
        self.assertEqual(len(steps), 3)
        first = steps[0][0]
        self.assertIsNotNone(first)
        self.assertTrue(all(mask is first for step in steps for mask in step))
        self.assertEqual(first.reshape(-1).nonzero().reshape(-1).tolist(), [123])
        self.assertEqual([call["length"] for call in model.language_model.calls[-3:]], [ids.shape[1], 1, 1])
        assert_unbound(self, model)


class NonFiniteTests(unittest.TestCase):
    def assert_rejected_at_zero_gate(self, model, features, chosen):
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), self.assertRaises(FloatingPointError):
            model(input_ids=token_ids(2), expert_features=features, point_indices=chosen)
        assert_unbound(self, model)

    def test_nonfinite_branch_rejected_even_at_zero_gate(self):
        model = fresh()
        self.assertEqual(model.gate_values(), [0.0] * LAYERS)
        chosen = points(17, 300)
        selected = board_features(1)
        selected[0, 4, 17, 1] = float("nan")
        self.assert_rejected_at_zero_gate(model, selected, chosen)
        # The SDPA mask does not shield a non-finite unselected token: its NaN key
        # score and its 0 * NaN value product reach the branch, which is rejected.
        unselected = board_features(1)
        unselected[1, 4, 5, 2] = float("nan")
        self.assert_rejected_at_zero_gate(model, unselected, chosen)
        with torch.no_grad():
            model.injected_layers[0].adapter.output.weight[0, 0] = float("inf")
        self.assert_rejected_at_zero_gate(model, board_features(1), chosen)


if __name__ == "__main__":
    unittest.main()
