"""Tiny-model checks for the APS-to-natural warm start; no pretrained model, data or fit.

Written without being executed by its author; results count only once run.
Torch, safetensors and the tiny fake Qwen3-shaped base from test_assisted_point
only; here its embedding and output weights are tied, as in Qwen3-0.6B.
Exactness checks force math SDPA. Candidate layout, from the directory holding
candidate/ and project/:

    python -m unittest discover -s candidate -p 'test_warm_qk.py' -v

Integrated layout, from project/:

    python -m unittest discover -s tests -p 'test_warm_qk.py' -v
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
CANDIDATE = (HERE / "warm_qk.py").is_file()
# Candidate layout: candidate/{warm_qk.py,test_warm_qk.py} beside project/.
# Integrated layout: project/tests/test_warm_qk.py beside project/src/goqueen/warm_qk.py.
PROJECT = HERE.parent / "project" if CANDIDATE else HERE.parent
for _path in (PROJECT / "src", PROJECT / "tests", HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from goqueen import checkpoints, fusion  # noqa: E402
from goqueen.assisted_point import AssistedPointGatedBoardAttention, AssistedPointGoQueen  # noqa: E402
from goqueen.m_key import MKeyGoQueen  # noqa: E402
from test_assisted_point import (COLUMN, EXPERT, HEADS, INACTIVE, LAYERS, ROW, SEED,  # noqa: E402
                                 SUFFIXES, WIDTH, FakeQwen3, board_features, generator, layer_norm,
                                 points, raw_bytes, set_gates, set_known_weights, token_ids)

if CANDIDATE:
    import warm_qk  # noqa: E402
else:
    from goqueen import warm_qk  # noqa: E402

# Independent expectations: written out here, not read from the module under test.
ADAPTER_NAMES = [f"language_model.model.layers.{i}.adapter.{suffix}"
                 for i in range(LAYERS) for suffix in SUFFIXES] + ["rows.weight", "columns.weight"]
QK_NAMES = [f"language_model.model.layers.{i}.adapter.{suffix}" for i in range(LAYERS) for suffix in INACTIVE]
QK_VALUES = LAYERS * (2 * WIDTH + WIDTH * WIDTH + WIDTH * EXPERT)
METADATA = {"variant": "m", "training_recipe": "aps_warm_qk", "updates": "0", "source_aps_sha256": "0" * 64}


def tied(cls=AssistedPointGoQueen, seed=SEED):
    """Seeded fake base with tied embedding/output weights, then the wrapper."""
    torch.manual_seed(seed)
    base = FakeQwen3()
    base.lm_head.weight = base.model.embed_tokens.weight
    return cls(base, expert_width=EXPERT, heads=HEADS)


def converted(seed=SEED):
    """A fitted-looking APS model with nonzero gates, its 101 list, then in-place conversion."""
    model = tied(seed=seed)
    set_known_weights(model, 3)
    set_gates(model, 0.6)
    named = warm_qk.adapter_parameters(model)
    warm_qk.convert_to_natural(model)
    return model, named


def last_layer(model):
    """Dirty state is planted in the last layer, so every layer must be checked before mutation."""
    return model.injected_layers[10]


def flags(model):
    return [(name, p.requires_grad) for name, p in model.named_parameters(remove_duplicate=False)]


def snapshot(named):
    return {name: (tuple(p.shape), raw_bytes(p).clone()) for name, p in named}


def same(tensor, entry):
    return tuple(tensor.shape) == entry[0] and torch.equal(raw_bytes(tensor), entry[1])


def natural_reference(model, number, features, hidden):
    """Original M adapter written out: softmax over all 361 tokens without a mask, then the gate."""
    adapter = model.injected_layers[number].adapter
    tokens = layer_norm(features[:, number] + model.rows.weight[ROW] + model.columns.weight[COLUMN],
                        adapter.expert_norm)

    def heads(x):
        return x.reshape(x.shape[0], x.shape[1], HEADS, WIDTH // HEADS).transpose(1, 2)

    query = heads(layer_norm(hidden, adapter.query_norm) @ adapter.query.weight.T)
    key, value = heads(tokens @ adapter.key.weight.T), heads(tokens @ adapter.value.weight.T)
    weights = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(WIDTH // HEADS), dim=-1)
    attended = (weights @ value).transpose(1, 2).reshape(hidden.shape)
    return hidden + torch.tanh(adapter.alpha) * (attended @ adapter.output.weight.T)


class AdapterListTests(unittest.TestCase):
    def test_complete_ordered_list_independent_of_flags_and_excluding_tied_base(self):
        model = tied()
        # Premise: the output weight is a second name for a base object.
        self.assertIs(model.language_model.lm_head.weight, model.language_model.model.embed_tokens.weight)
        named = warm_qk.adapter_parameters(model)
        self.assertIsInstance(named, tuple)
        self.assertEqual([name for name, _ in named], ADAPTER_NAMES)
        for name, p in named:
            self.assertIs(p, model.get_parameter(name))
        self.assertEqual(len({id(p) for _, p in named}), 101)
        self.assertFalse({id(p) for p in model.base_parameters} & {id(p) for _, p in named})
        model.requires_grad_(False)
        self.assertEqual(model.adapter_state_dict(), {})  # the requires_grad view now sees nothing
        self.assertEqual([(n, id(p)) for n, p in warm_qk.adapter_parameters(model)],
                         [(n, id(p)) for n, p in named])

    def test_malformed_layouts_rejected(self):
        def extra_adapter_tensor(model):
            model.injected_layers[4].adapter.extra = nn.Parameter(torch.zeros(1))

        def tensor_shared_by_two_adapters(model):
            model.injected_layers[1].adapter.alpha = model.injected_layers[0].adapter.alpha

        def base_tensor_not_listed_as_base(model):
            model.base_parameters = model.base_parameters[1:]

        def adapter_tensor_listed_as_base(model):
            model.base_parameters = model.base_parameters + (model.rows.weight,)

        def ten_injected_layers(model):
            model.layer_indices = model.layer_indices[:10]

        def nested_wrapper(model):
            layer = model.injected_layers[2]
            layer.decoder = fusion.ExpertDecoderLayer(layer.decoder, fusion.GatedBoardAttention(WIDTH, EXPERT, HEADS))

        for corrupt in (extra_adapter_tensor, tensor_shared_by_two_adapters, base_tensor_not_listed_as_base,
                        adapter_tensor_listed_as_base, ten_injected_layers, nested_wrapper):
            with self.subTest(corrupt.__name__):
                model = tied()
                corrupt(model)
                with self.assertRaises(ValueError):
                    warm_qk.adapter_parameters(model)
        with self.assertRaises(TypeError):
            warm_qk.adapter_parameters(FakeQwen3())


class ConversionTests(unittest.TestCase):
    def assert_rejected_unchanged(self, model, error):
        types = [type(model)] + [type(layer.adapter) for layer in model.injected_layers]
        with self.assertRaises(error):
            warm_qk.convert_to_natural(model)
        self.assertEqual([type(model)] + [type(layer.adapter) for layer in model.injected_layers], types)

    def test_in_place_switch_preserves_objects_bytes_flags_rng_and_clears_aps_fields(self):
        model = tied()
        set_known_weights(model, 3)
        set_gates(model, 0.4)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            model(input_ids=token_ids(2), expert_features=board_features(1), point_indices=points(17, 300))
        # Premise: a finished APS binding leaves its plain fields behind as None instance attributes.
        self.assertIn("point_mask", vars(model.injected_layers[0].adapter))
        model.injected_layers[3].adapter.value.weight.requires_grad_(False)
        model.rows.weight.requires_grad_(False)
        base = model.language_model
        wrappers = [(layer, layer.decoder, layer.adapter) for layer in model.injected_layers]
        listing = [(n, id(p), tuple(p.shape), p.dtype, p.requires_grad)
                   for n, p in model.named_parameters(remove_duplicate=False)]
        before = snapshot(model.named_parameters())
        state_names, buffers = list(model.state_dict()), [n for n, _ in model.named_buffers()]
        forwards = len(base.calls)
        rng = torch.get_rng_state()

        self.assertIs(warm_qk.convert_to_natural(model), model)

        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(len(base.calls), forwards)
        self.assertIs(type(model), fusion.GoQueen)
        self.assertIs(model.language_model, base)
        reference = tied(fusion.GoQueen).injected_layers[0].adapter
        for layer, (wrapper, decoder, adapter) in zip(model.injected_layers, wrappers):
            self.assertIs(layer, wrapper)
            self.assertIs(layer.decoder, decoder)
            self.assertIs(layer.adapter, adapter)
            self.assertIs(type(adapter), fusion.GatedBoardAttention)
            self.assertEqual(set(vars(adapter)), set(vars(reference)))
            self.assertFalse(hasattr(adapter, "point_mask") or hasattr(adapter, "point_indices"))
        self.assertEqual([(n, id(p), tuple(p.shape), p.dtype, p.requires_grad)
                          for n, p in model.named_parameters(remove_duplicate=False)], listing)
        for name, p in model.named_parameters():
            self.assertTrue(same(p, before[name]), name)
        self.assertEqual(list(model.state_dict()), state_names)
        self.assertEqual([n for n, _ in model.named_buffers()], buffers)

    def test_live_binding_stored_fields_or_hooks_rejected_before_any_change(self):
        model = tied()
        with model.bind_expert(board_features(1), points(3, 200)):
            bound = model.injected_layers[0].adapter.point_mask
            self.assert_rejected_unchanged(model, RuntimeError)
            # The live binding is kept, not discarded.
            self.assertTrue(model._bound)
            self.assertTrue(all(layer.adapter.point_mask is bound and layer.expert_features is not None
                                for layer in model.injected_layers))
        warm_qk.convert_to_natural(model)

        for name, value in (("expert_features", torch.zeros(2, 361, EXPERT)),
                            ("point_mask", torch.ones(2, 1, 1, 361, dtype=torch.bool)),
                            ("point_indices", points(3, 200))):
            with self.subTest(stored=name):
                model = tied()
                owner = last_layer(model) if name == "expert_features" else last_layer(model).adapter
                setattr(owner, name, value)
                self.assert_rejected_unchanged(model, RuntimeError)
                self.assertIs(getattr(owner, name), value)

        for register in ("register_forward_pre_hook", "register_forward_hook"):
            with self.subTest(hook=register):
                model = tied()
                handle = getattr(last_layer(model).adapter, register)(lambda *args: None)
                self.assert_rejected_unchanged(model, RuntimeError)
                handle.remove()
                warm_qk.convert_to_natural(model)

    def test_only_an_exact_unconverted_aps_model_is_accepted(self):
        class SubclassedAPS(AssistedPointGoQueen):
            pass

        class SubclassedAdapter(AssistedPointGatedBoardAttention):
            pass

        odd_adapter = tied()
        last_layer(odd_adapter).adapter.__class__ = SubclassedAdapter
        already = tied()
        warm_qk.convert_to_natural(already)
        for model in (tied(fusion.GoQueen), tied(MKeyGoQueen), tied(SubclassedAPS), odd_adapter, already):
            with self.subTest(model=type(model).__name__):
                self.assert_rejected_unchanged(model, TypeError)


class NaturalFormulaTests(unittest.TestCase):
    def test_converted_adapter_is_the_written_out_natural_formula_not_the_one_point_path(self):
        model = tied()
        set_known_weights(model, 3)
        set_gates(model, 0.7)
        features, chosen = board_features(1), points(17, 300)
        hidden = torch.randn(2, 5, WIDTH, generator=generator(4))
        layer = model.injected_layers[6]
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            with model.bind_expert(features, chosen):
                assisted = layer.adapter(hidden, layer.expert_features)
            warm_qk.convert_to_natural(model)
            with model.bind_expert(features):
                natural = layer.adapter(hidden, layer.expert_features)
            expected = natural_reference(model, 6, features, hidden)
        torch.testing.assert_close(natural, expected, rtol=1e-5, atol=1e-5)
        # In this constructed example the one-point path adds one selected value at every query
        # position; natural attention mixes all 361 values with query-dependent weights.
        self.assertGreater(float((natural - assisted).abs().max()), 1e-3)
        step = assisted - hidden
        self.assertTrue(torch.allclose(step, step[:, :1].expand_as(step), rtol=0, atol=1e-6))
        step = natural - hidden
        self.assertFalse(torch.allclose(step, step[:, :1].expand_as(step), rtol=0, atol=1e-4))
        with self.assertRaises(TypeError):
            model.bind_expert(features, chosen)  # no point-index argument exists after conversion

    def test_converted_model_equals_original_M_with_identical_weights(self):
        original, model = tied(fusion.GoQueen), tied()
        for item in (original, model):
            set_known_weights(item, 3)
            set_gates(item, 0.7)
        warm_qk.convert_to_natural(model)
        ids, features = token_ids(2), board_features(1)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            natural = model(input_ids=ids, expert_features=features).logits
            reference = original(input_ids=ids, expert_features=features).logits
            plain = model(input_ids=ids).logits
        self.assertTrue(torch.equal(natural, reference))
        # The board path is live, so the equality above is informative.
        self.assertGreater(float((natural - plain).abs().max()), 1e-4)


class FreezeTests(unittest.TestCase):
    def test_exactly_44_in_optimizer_and_one_update_keeps_57_and_base_bytes(self):
        model, named = converted()
        self.assertTrue(all(p.requires_grad for _, p in named))  # all 101 trainable before freezing
        stray = model.base_parameters[0]
        stray.requires_grad_(True)
        trainable = warm_qk.freeze_except_qk(model, named)
        self.assertIsInstance(trainable, tuple)
        self.assertEqual([name for name, _ in trainable], QK_NAMES)
        for name, p in trainable:
            self.assertIs(p, model.get_parameter(name))
        chosen = {id(p) for _, p in trainable}
        self.assertEqual({id(p) for p in model.parameters() if p.requires_grad}, chosen)
        self.assertFalse(stray.requires_grad)
        self.assertEqual(sum(p.numel() for _, p in trainable), QK_VALUES)
        frozen = [(n, p) for n, p in named if id(p) not in chosen]
        self.assertEqual(len(frozen), 57)

        optimizer = torch.optim.AdamW([p for _, p in trainable], lr=.001,
                                      betas=(.9, .999), eps=1e-8, weight_decay=0)
        self.assertEqual([id(p) for group in optimizer.param_groups for p in group["params"]],
                         [id(model.get_parameter(name)) for name in QK_NAMES])
        old_frozen, old_trainable = snapshot(frozen), snapshot(trainable)
        old_base = [raw_bytes(p).clone() for p in model.base_parameters]
        with sdpa_kernel(SDPBackend.MATH):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids=token_ids(777, batch=4),
                           expert_features=board_features(888, batch=4)).logits[:, -1]
            F.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
        for name, p in trainable:
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(bool(torch.isfinite(p.grad).all()), name)
        # Under natural attention every Q/K kind is reached; under APS these were exactly zero.
        for suffix in INACTIVE:
            self.assertTrue(any(bool(p.grad.ne(0).any()) for n, p in trainable
                                if n.endswith(".adapter." + suffix)), suffix)
        self.assertTrue(all(p.grad is None for _, p in frozen))
        self.assertTrue(all(p.grad is None for p in model.base_parameters))
        optimizer.step()
        for name, p in frozen:
            self.assertTrue(same(p, old_frozen[name]), name)
        for p, saved in zip(model.base_parameters, old_base):
            self.assertTrue(torch.equal(raw_bytes(p), saved))
        # Descriptive only for the real recipe; in this constructed example something must move.
        self.assertTrue(any(not same(p, old_trainable[name]) for name, p in trainable))

    def test_foreign_filtered_reordered_lists_and_unconverted_model_rejected(self):
        model, named = converted()
        first = warm_qk.freeze_except_qk(model, named)
        before = flags(model)
        _, foreign = converted(seed=SEED + 1)
        cases = {
            "foreign_objects": foreign,
            "trainable_44_only": first,
            "reordered": named[1:] + named[:1],
            "base_object_substituted": ((named[0][0], model.base_parameters[0]),) + named[1:],
            "not_pairs": [p for _, p in named],
            "names_only": dict(named),
        }
        for label, candidate in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    warm_qk.freeze_except_qk(model, candidate)
                self.assertEqual(flags(model), before)
        aps = tied()
        aps_named = warm_qk.adapter_parameters(aps)
        aps_flags = flags(aps)
        with self.assertRaises(TypeError):
            warm_qk.freeze_except_qk(aps, aps_named)
        self.assertEqual(flags(aps), aps_flags)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "adapter.safetensors"

    def prepared(self):
        model, named = converted()
        warm_qk.freeze_except_qk(model, named)
        return model, named

    def assert_kept(self, path, metadata, expected):
        """The file on disk holds exactly the full 101 state in `expected`, with this metadata."""
        with safe_open(str(path), framework="pt", device="cpu") as saved:
            self.assertEqual(saved.metadata(), metadata)
            self.assertEqual(set(saved.keys()), set(ADAPTER_NAMES))
            for name in ADAPTER_NAMES:
                tensor = saved.get_tensor(name)
                self.assertEqual(tensor.dtype, torch.float32, name)
                self.assertTrue(same(tensor, expected[name]), name)

    def assert_state(self, named, expected):
        for name, p in named:
            self.assertTrue(same(p, expected[name]), name)

    def test_full_101_file_via_existing_loader_with_all_flags_then_44_restored(self):
        model, named = self.prepared()
        before_flags, before = flags(model), snapshot(named)
        forwards = len(model.language_model.calls)
        real_loader, calls = checkpoints.load_adapter_checkpoint, []

        def observed(target, path):
            calls.append((target, Path(path), [p.requires_grad for _, p in named],
                          [p.requires_grad for p in target.base_parameters]))
            return real_loader(target, path)

        with patch.object(checkpoints, "load_adapter_checkpoint", side_effect=observed), \
                patch.object(model, "load_adapter_state_dict", wraps=model.load_adapter_state_dict) as copy:
            result = warm_qk.save_full_checkpoint(model, named, self.path, METADATA)

        self.assertEqual(len(calls), 1)
        target, path, adapter_flags, base_flags = calls[0]
        self.assertIs(target, model)
        self.assertEqual(path, self.path)
        self.assertEqual(adapter_flags, [True] * 101)
        self.assertFalse(any(base_flags))
        copy.assert_called_once()
        self.assertEqual(sorted(copy.call_args.args[0]), sorted(ADAPTER_NAMES))
        self.assertEqual(flags(model), before_flags)
        self.assertEqual([n for n, p in model.named_parameters() if p.requires_grad], QK_NAMES)
        self.assertEqual(len(model.language_model.calls), forwards)
        self.assertEqual(result["sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(result["count"], 101)
        self.assertIs(result["loader_verified"], True)
        self.assertEqual(result["metadata"], METADATA)
        self.assertNotIn(str(self.path.parent), json.dumps(result))  # report-ready, no local path
        self.assert_kept(self.path, METADATA, before)
        self.assert_state(named, before)

    def test_file_is_a_standard_M_checkpoint_for_the_existing_loader(self):
        model, named = self.prepared()
        warm_qk.save_full_checkpoint(model, named, self.path, METADATA)
        before = flags(model)
        # Alone, the existing loader sees only the 44 trainable tensors and rejects the full file.
        with self.assertRaises(ValueError):
            checkpoints.load_adapter_checkpoint(model, self.path)
        self.assertEqual(flags(model), before)
        other = tied(fusion.GoQueen, seed=SEED + 1)  # all 101 trainable by construction
        other_named = warm_qk.adapter_parameters(other)
        self.assertTrue(any(not torch.equal(raw_bytes(p), raw_bytes(q))
                            for (_, p), (_, q) in zip(named, other_named)))
        self.assertEqual(checkpoints.load_adapter_checkpoint(other, self.path), METADATA)
        self.assert_state(other_named, snapshot(named))
        with self.assertRaises(ValueError):
            checkpoints.load_adapter_checkpoint(tied(), self.path)  # APS: variant mismatch

    def test_existing_destination_is_never_overwritten(self):
        model, named = self.prepared()
        before = flags(model)
        self.path.write_bytes(b"keep")
        with patch.object(checkpoints, "load_adapter_checkpoint") as loader, \
                self.assertRaises(FileExistsError):
            warm_qk.save_full_checkpoint(model, named, self.path, METADATA)
        loader.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b"keep")
        self.assertEqual(flags(model), before)

    def test_loader_failures_restore_all_flags_and_keep_the_artifact(self):
        model, named = self.prepared()
        before_flags, before = flags(model), snapshot(named)
        bad = {**METADATA, "position_scale": "35.35"}
        with self.assertRaises(ValueError):  # rejected by the real existing loader after the write
            warm_qk.save_full_checkpoint(model, named, self.path, bad)
        self.assertEqual(flags(model), before_flags)
        self.assert_kept(self.path, bad, before)

        second = self.path.with_name("second.safetensors")
        with patch.object(checkpoints, "load_adapter_checkpoint", side_effect=TimeoutError("deadline")), \
                self.assertRaises(TimeoutError):
            warm_qk.save_full_checkpoint(model, named, second, METADATA)
        self.assertEqual(flags(model), before_flags)
        self.assert_kept(second, METADATA, before)
        self.assert_state(named, before)

    def test_post_load_change_detected_against_an_independent_snapshot(self):
        model, named = self.prepared()
        before_flags, before = flags(model), snapshot(named)
        real_copy = model.load_adapter_state_dict

        def copy_then_change(state):
            real_copy(state)
            with torch.no_grad():
                model.rows.weight[0, 0] += 1.0

        with patch.object(model, "load_adapter_state_dict", side_effect=copy_then_change), \
                self.assertRaises(RuntimeError):
            warm_qk.save_full_checkpoint(model, named, self.path, METADATA)
        self.assertEqual(flags(model), before_flags)
        self.assert_kept(self.path, METADATA, before)

    def test_invalid_metadata_nonfinite_or_wrong_inputs_write_nothing(self):
        model, named = self.prepared()
        before = flags(model)
        for metadata in (None, {}, {"variant": "m"}, {"training_recipe": "aps_warm_qk"},
                         {**METADATA, "variant": "m_assisted_point"}, {**METADATA, "training_recipe": "aps"},
                         {**METADATA, "routing_contract": "single_point_bool_mask_v1"},
                         {**METADATA, "updates": 0}):
            with self.subTest(metadata=metadata):
                with patch.object(checkpoints, "load_adapter_checkpoint") as loader, \
                        self.assertRaises(ValueError):
                    warm_qk.save_full_checkpoint(model, named, self.path, metadata)
                loader.assert_not_called()
                self.assertFalse(self.path.exists())
                self.assertEqual(flags(model), before)

        _, foreign = converted(seed=SEED + 1)
        with self.assertRaises(ValueError):
            warm_qk.save_full_checkpoint(model, foreign, self.path, METADATA)
        model.base_parameters[1].requires_grad_(True)
        with self.assertRaises(ValueError):
            warm_qk.save_full_checkpoint(model, named, self.path, METADATA)
        model.base_parameters[1].requires_grad_(False)
        aps = tied()
        aps.requires_grad_(False)
        with self.assertRaises(TypeError):
            warm_qk.save_full_checkpoint(aps, warm_qk.adapter_parameters(aps), self.path, METADATA)
        self.assertFalse(self.path.exists())

        # A frozen tensor is included: a check over the 44 trainable tensors alone would miss it.
        for name, value in (("language_model.model.layers.5.adapter.value.weight", float("nan")),
                            ("language_model.model.layers.2.adapter.key.weight", float("nan")),
                            ("columns.weight", float("inf"))):
            with self.subTest(nonfinite=name):
                model, named = self.prepared()
                with torch.no_grad():
                    model.get_parameter(name)[0, 0] = value
                with self.assertRaises(ValueError):
                    warm_qk.save_full_checkpoint(model, named, self.path, METADATA)
                self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
