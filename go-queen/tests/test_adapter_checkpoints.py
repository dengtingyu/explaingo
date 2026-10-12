"""Tiny safetensors variant guards; no pretrained model, forward or optimizer.

    python -m unittest discover -s tests -p 'test_adapter_checkpoints.py' -v
"""
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

from goqueen.checkpoints import load_adapter_checkpoint
from goqueen.fusion import GoQueen
from goqueen.m_key import MKeyGoQueen, POSITION_SCALE


class TinyDecoder(nn.Module):
    attention_type = "full_attention"


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=4, num_hidden_layers=11)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(TinyDecoder() for _ in range(11))


def fresh(model_type=GoQueen):
    return model_type(TinyQwen(), expert_width=2, heads=1)


def distinguishable_state(model):
    return {name: torch.full_like(tensor, (index + 1) / 8)
            for index, (name, tensor) in enumerate(model.adapter_state_dict().items())}


class AdapterCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "adapter.safetensors"

    def write(self, model, metadata=None, state=None):
        state = distinguishable_state(model) if state is None else state
        save_file(state, str(self.path), metadata=metadata)
        return state

    def assert_restored(self, model, state):
        self.assertEqual(set(model.adapter_state_dict()), set(state))
        for name, tensor in model.adapter_state_dict().items():
            self.assertTrue(torch.equal(tensor, state[name]), name)

    def assert_rejected_without_copy(self, model):
        before = {name: tensor.clone() for name, tensor in model.adapter_state_dict().items()}
        with patch.object(model, "load_adapter_state_dict", wraps=model.load_adapter_state_dict) as copy:
            with self.assertRaises(ValueError):
                load_adapter_checkpoint(model, self.path)
            copy.assert_not_called()
        self.assert_restored(model, before)

    def test_unmarked_historical_M_restores_exactly(self):
        model = fresh()
        state = self.write(model)
        self.assertEqual(load_adapter_checkpoint(model, self.path), {})
        self.assert_restored(model, state)

    def test_historical_stage_metadata_remains_original_M(self):
        model = fresh()
        for metadata in ({"stage": "a0_value_token_microfit", "updates": "180"},
                         {"stage": "a0_prompt_control", "updates": "180"},
                         {"stage": "phase0_text_fit_check", "updates": "60"},
                         {"scope": "GoQueen adapters only", "epoch": "3"}):
            with self.subTest(metadata=metadata):
                state = self.write(model, metadata)
                self.assertEqual(load_adapter_checkpoint(model, self.path), metadata)
                self.assert_restored(model, state)

    def test_explicit_M_restores_exactly(self):
        model = fresh()
        state = self.write(model, {"variant": "m"})
        load_adapter_checkpoint(model, self.path)
        self.assert_restored(model, state)

    def test_explicit_M_key_restores_exactly(self):
        model = fresh(MKeyGoQueen)
        metadata = {"variant": "m_key", "position_scale": repr(POSITION_SCALE)}
        state = self.write(model, metadata)
        self.assertEqual(load_adapter_checkpoint(model, self.path), metadata)
        self.assert_restored(model, state)

    def test_M_key_checkpoint_cannot_enter_M(self):
        model = fresh()
        self.write(model, {"variant": "m_key", "position_scale": repr(POSITION_SCALE)})
        self.assert_rejected_without_copy(model)

    def test_M_checkpoint_cannot_enter_M_key(self):
        model = fresh(MKeyGoQueen)
        for metadata in (None, {"variant": "m"}):
            with self.subTest(metadata=metadata):
                self.write(model, metadata)
                self.assert_rejected_without_copy(model)

    def test_unknown_or_blank_variant_rejected(self):
        model = fresh()
        for variant in ("", " ", "M", "other", "point_table"):
            with self.subTest(variant=variant):
                self.write(model, {"variant": variant})
                self.assert_rejected_without_copy(model)

    def test_unmarked_M_key_semantics_rejected(self):
        model = fresh()
        for metadata in ({"position_scale": repr(POSITION_SCALE)}, {"stage": "a0_m_key"}):
            with self.subTest(metadata=metadata):
                self.write(model, metadata)
                self.assert_rejected_without_copy(model)

    def test_explicit_M_cannot_declare_M_key_semantics(self):
        model = fresh()
        for extra in ({"position_scale": repr(POSITION_SCALE)}, {"stage": "a0_m_key"}):
            with self.subTest(extra=extra):
                self.write(model, {"variant": "m", **extra})
                self.assert_rejected_without_copy(model)

    def test_missing_nonfinite_or_wrong_scale_rejected(self):
        model = fresh(MKeyGoQueen)
        for scale in (None, "", "nonsense", "nan", "inf", "-inf", "0", repr(POSITION_SCALE + 1)):
            with self.subTest(scale=scale):
                metadata = {"variant": "m_key"}
                if scale is not None:
                    metadata["position_scale"] = scale
                self.write(model, metadata)
                self.assert_rejected_without_copy(model)

    def test_unknown_model_subclass_rejected_before_open(self):
        class UnknownGoQueen(GoQueen):
            pass
        model = fresh(UnknownGoQueen)
        with patch("goqueen.checkpoints.safe_open") as opened:
            self.assert_rejected_without_copy(model)
            opened.assert_not_called()

    def test_extra_point_table_parameter_rejected_without_copy(self):
        model = fresh()
        state = distinguishable_state(model)
        state["point_embedding.table"] = torch.zeros(361, 4)
        self.write(model, state=state)
        self.assert_rejected_without_copy(model)

    def test_missing_adapter_parameter_rejected_without_copy(self):
        model = fresh()
        state = distinguishable_state(model)
        del state["rows.weight"]
        self.write(model, state=state)
        self.assert_rejected_without_copy(model)

    def test_late_shape_mismatch_does_not_partially_copy(self):
        model = fresh()
        state = distinguishable_state(model)
        last = next(reversed(state))
        state[last] = torch.zeros(1)
        self.write(model, state=state)
        self.assert_rejected_without_copy(model)

    def test_metadata_and_tensors_use_one_open_context(self):
        model = fresh(MKeyGoQueen)
        state = self.write(model, {"variant": "m_key", "position_scale": repr(POSITION_SCALE)})
        from goqueen.checkpoints import safe_open
        with patch("goqueen.checkpoints.safe_open", wraps=safe_open) as opened:
            load_adapter_checkpoint(model, self.path)
            opened.assert_called_once_with(str(self.path), framework="pt", device="cpu")
        self.assert_restored(model, state)


if __name__ == "__main__":
    unittest.main()
