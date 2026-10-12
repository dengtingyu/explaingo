"""Actual-batch APS facade routing and safe disk variants, without pretrained models."""
from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goqueen import assisted_point_eval as routed
from goqueen.assisted_point import AssistedPointGoQueen
from goqueen.checkpoints import APS_ROUTING_CONTRACT, load_adapter_checkpoint
from goqueen.fusion import GoQueen
from goqueen.grounding_data import occupancy_prompt, point_to_gtp
from goqueen.m_key import MKeyGoQueen, POSITION_SCALE


def question(index, label=0):
    point = point_to_gtp(index)
    return {"point_index": index, "point": point, "to_play": "B",
            "prompt": occupancy_prompt("B", point), "label": label,
            "position_id": "test", "question_id": str(index)}


class CallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, **inputs):
        self.calls.append(inputs)
        return SimpleNamespace(logits=torch.zeros(len(inputs["point_indices"]), 1, 3))


class RoutingTests(unittest.TestCase):
    def test_route_reads_no_answer_fields(self):
        a = question(0, 0)
        b = question(0, 2)
        del b["label"]
        self.assertTrue(torch.equal(routed.point_indices_for([a], "cpu"),
                                    routed.point_indices_for([b], "cpu")))

    def test_coordinate_validation_precedes_forward(self):
        q = question(300)
        for changes in ({"point_index": True}, {"point_index": -1},
                        {"point_index": 361}, {"point": "A1"},
                        {"prompt": "read whatever"}, {"to_play": "X"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                routed.point_indices_for([{**q, **changes}], "cpu")

    def test_each_internal_scorer_forward_gets_its_actual_chunk(self):
        model = CallModel()
        observed = []

        def scorer(facade, tokenizer, selected, features, batch_size, deadline):
            self.assertEqual(batch_size, len(selected))
            facade(input_ids=torch.ones(len(selected), 2, dtype=torch.long))
            observed.append(model.calls[-1]["point_indices"].tolist())
            return [{"question_id": q["question_id"]} for q in selected]

        qs = [question(i) for i in (0, 300, 360, 15, 19)]
        with patch.object(routed.original, "score_questions", scorer):
            result = routed.score_questions(model, None, qs, {}, batch_size=2)
        self.assertEqual(observed, [[0, 300], [360, 15], [19]])
        self.assertEqual(len(result), 5)

    def test_partial_original_batch_stops_outer_scoring(self):
        model = CallModel()
        with patch.object(routed.original, "score_questions", return_value=[]) as scorer:
            self.assertEqual(routed.score_questions(model, None, [question(0), question(1)], {}, 1), [])
            self.assertEqual(scorer.call_count, 1)

    def test_expired_deadline_never_routes_or_calls_model(self):
        model = CallModel()
        self.assertEqual(routed.score_questions(model, None, [question(0)], {}, deadline=0), [])
        self.assertEqual(model.calls, [])

    def test_cached_prefix_mismatch_is_rejected(self):
        model = CallModel()
        with patch.object(routed.original, "encode_prefix", return_value=(torch.tensor([1, 2]), (0, 1, 2))), \
                patch.object(routed.original, "_forward_logits") as forwarded:
            with self.assertRaises(ValueError):
                routed._forward_logits(model, None, [torch.tensor([1, 3])], [question(0)], {})
            forwarded.assert_not_called()

    def test_forward_uses_same_features_and_correct_point(self):
        model = CallModel()
        selected = [question(300), question(0)]
        prefixes = [torch.tensor([1, 2]), torch.tensor([1, 2])]
        features = {"test": "feature-payload"}

        def forward(facade, tokenizer, actual_prefixes, actual_selected, actual_features):
            self.assertIs(actual_features, features)
            self.assertIs(actual_selected, selected)
            self.assertIs(actual_prefixes, prefixes)
            facade(input_ids=torch.ones(2, 2, dtype=torch.long))
            return model.calls[-1]["point_indices"]

        with patch.object(routed.original, "encode_prefix", return_value=(prefixes[0], (0, 1, 2))), \
                patch.object(routed.original, "_forward_logits", forward):
            result = routed._forward_logits(model, None, prefixes, selected, features)
        self.assertEqual(result.tolist(), [300, 0])


class TinyDecoder(nn.Module):
    attention_type = "full_attention"


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=4, num_hidden_layers=11)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(TinyDecoder() for _ in range(11))


def fresh(kind):
    return kind(TinyQwen(), expert_width=2, heads=1)


def metadata():
    return {"variant": "m_assisted_point", "routing_contract": APS_ROUTING_CONTRACT,
            "updates": "0", "engineering_only": "true",
            "source_manifest_sha256": "a" * 64, "questions_sha256": "b" * 64,
            "source_features_sha256": "c" * 64}


class APSCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "adapter.safetensors"

    def write(self, model, meta, state=None):
        save_file(model.adapter_state_dict() if state is None else state, str(self.path), metadata=meta)

    def rejected_without_copy(self, model):
        before = {name: tensor.clone() for name, tensor in model.adapter_state_dict().items()}
        with patch.object(model, "load_adapter_state_dict", wraps=model.load_adapter_state_dict) as copied:
            with self.assertRaises(ValueError):
                load_adapter_checkpoint(model, self.path)
            copied.assert_not_called()
        for name, tensor in model.adapter_state_dict().items():
            self.assertTrue(torch.equal(tensor, before[name]), name)

    def test_APS_roundtrip_preflights_and_restores(self):
        model = fresh(AssistedPointGoQueen)
        state = {name: torch.full_like(tensor, .25) for name, tensor in model.adapter_state_dict().items()}
        self.write(model, metadata(), state)
        self.assertEqual(load_adapter_checkpoint(model, self.path), metadata())
        for name, tensor in model.adapter_state_dict().items():
            self.assertTrue(torch.equal(tensor, state[name]), name)

    def test_cross_variants_rejected_both_directions(self):
        kinds = [GoQueen, MKeyGoQueen, AssistedPointGoQueen]
        metas = [{"variant": "m"}, {"variant": "m_key", "position_scale": repr(POSITION_SCALE)}, metadata()]
        for source, meta in zip(kinds, metas):
            self.write(fresh(source), meta)
            for target in kinds:
                if target is not source:
                    with self.subTest(source=source.__name__, target=target.__name__):
                        self.rejected_without_copy(fresh(target))

    def test_APS_required_metadata_is_not_optional(self):
        model = fresh(AssistedPointGoQueen)
        for name in metadata():
            meta = metadata()
            del meta[name]
            with self.subTest(missing=name):
                self.write(model, meta)
                self.rejected_without_copy(model)

    def test_APS_invalid_contract_update_count_or_digest_rejected(self):
        model = fresh(AssistedPointGoQueen)
        for extra in ({"routing_contract": "all_points"}, {"updates": "1"},
                      {"updates": "-1"}, {"updates": "00"}, {"engineering_only": "yes"},
                      {"questions_sha256": "x" * 64}, {"source_manifest_sha256": "a" * 63},
                      {"position_scale": repr(POSITION_SCALE)}):
            with self.subTest(extra=extra):
                self.write(model, {**metadata(), **extra})
                self.rejected_without_copy(model)

    def test_nonfinite_last_tensor_is_rejected_before_any_copy(self):
        model = fresh(AssistedPointGoQueen)
        state = {name: tensor.clone() for name, tensor in model.adapter_state_dict().items()}
        state[next(reversed(state))].fill_(float("nan"))
        self.write(model, metadata(), state)
        self.rejected_without_copy(model)

    def test_routing_metadata_cannot_be_disguised_as_M(self):
        model = fresh(GoQueen)
        self.write(model, {"variant": "m", "routing_contract": APS_ROUTING_CONTRACT})
        self.rejected_without_copy(model)


if __name__ == "__main__":
    unittest.main()
