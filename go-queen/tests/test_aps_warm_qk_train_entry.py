"""Stdlib entry checks using saved JSON and tiny call doubles only.

No test imports a model dependency, decodes a tensor, starts a child, or makes
a model forward. Valid model execution is never entered. The helper's own
existing unit suite is not repeated here.
"""
from __future__ import annotations

import builtins
import copy
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/eval_aps_warm_qk_train.py"
HEAVY = {"torch", "numpy", "transformers", "safetensors"}


def load_entry():
    spec = importlib.util.spec_from_file_location("warm_train_eval_entry_check", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = load_entry()


def forbid_heavy_import(original):
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in HEAVY:
            raise AssertionError("Entry requested a heavy dependency: " + name)
        return original(name, *args, **kwargs)
    return guarded


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(entry.MANIFEST.read_text())
        cls.reference = json.loads(entry.REFERENCE.read_text())

    def test_actual_returned_exposures_give_fixed_unique_list_and_six_references(self):
        manifest, saved, restored, questions = entry.prepare_selection(
            self.data, self.reference, entry.PINS)
        observed = [qid for step in self.reference["step_attempts"] for qid in step["question_ids"]]
        expected = sorted(set(observed), key=lambda q: (hashlib.sha256(q.encode()).hexdigest(), q))
        self.assertEqual([x["question"]["question_id"] for x in manifest["entries"]], expected)
        self.assertEqual((manifest["unique_questions"], manifest["slots"]), (2702, 4000))
        self.assertEqual(sum(x["exposure_count"] for x in manifest["entries"]), len(observed))
        self.assertEqual(manifest["source"], entry.PINS)
        self.assertIsNot(manifest["source"], entry.PINS)
        self.assertEqual(manifest["separation"]["comparison_prompts_in_list"], 54)
        self.assertEqual(len({q["prompt"] for q in self.reference["validation_questions"]}), 52)
        final = next(e for e in self.reference["evaluations"] if e["name"] == "final")
        self.assertEqual(restored, final["rows"][:6])
        self.assertEqual([q["question_id"] for q in questions], [r["question_id"] for r in restored])
        self.assertEqual({q["to_play"] for q in questions}, {"B", "W"})
        self.assertEqual(len({q["position_id"] for q in questions}), 2)
        self.assertEqual([q["label"] for q in questions], [1, 2, 0, 1, 2, 0])
        for name in ("forced_choice_correct", "confusion_matrix", "macro_recall",
                     "three_label_ce", "full_vocabulary_ce"):
            self.assertEqual(saved["summary"][name], final[name])
        self.assertEqual(entry.MAX_FORWARDS, len(expected) + len(restored))

    def test_complete_status_cannot_substitute_for_run_completed(self):
        for status, completed in (("complete", False), ("stopped", True), ("complete", 1)):
            reference = dict(self.reference, status=status, run_completed=completed)
            with self.subTest(status=status, completed=completed), self.assertRaises(ValueError):
                entry.prepare_selection(self.data, reference, entry.PINS)

    def test_actual_steps_and_both_schedules_must_agree(self):
        for change in ("unreturned", "changed_slot", "changed_report_schedule", "changed_manifest_schedule"):
            reference, data = copy.deepcopy(self.reference), copy.deepcopy(self.data)
            if change == "unreturned":
                reference["step_attempts"][-1]["optimizer_step_returned"] = False
            elif change == "changed_slot":
                reference["step_attempts"][0]["question_ids"][0] = "not-an-observed-id"
            elif change == "changed_report_schedule":
                reference["train_order"][0] = "not-the-original-order"
            else:
                data["train_order"][0] = "not-the-original-order"
            with self.subTest(change=change), self.assertRaises(ValueError):
                entry.prepare_selection(data, reference, entry.PINS)

    def test_missing_duplicate_partial_or_corrupt_final_reference_is_rejected(self):
        for change in ("missing", "duplicate", "wrong_updates", "partial", "missing_row", "bad_summary"):
            reference = copy.deepcopy(self.reference)
            final = next(e for e in reference["evaluations"] if e["name"] == "final")
            if change == "missing":
                reference["evaluations"].remove(final)
            elif change == "duplicate":
                reference["evaluations"].append(copy.deepcopy(final))
            elif change == "wrong_updates":
                final["updates_completed"] = 999
            elif change == "partial":
                final["coverage_complete"] = False
            elif change == "missing_row":
                final["rows"].pop()
            else:
                final["forced_choice_correct"] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                entry.prepare_selection(self.data, reference, entry.PINS)


class RestorationAndCountTests(unittest.TestCase):
    def test_restoration_requires_complete_fixed_tolerance_and_finite_difference(self):
        good = {"passed": True, "coverage_complete": True, "predictions_match": True,
                "current_questions": 6, "saved_questions": 6,
                "max_abs_logp_difference": 0., "tolerance": 1e-5}
        self.assertTrue(entry.restore_allows_training(good))
        self.assertTrue(entry.restore_allows_training(dict(good, max_abs_logp_difference=1e-5)))
        bad_fields = [(key, False) for key in ("passed", "coverage_complete", "predictions_match")]
        bad_fields += [("tolerance", 1e-4), ("max_abs_logp_difference", 1.0001e-5)]
        bad_fields += [(field, value) for field in ("current_questions", "saved_questions")
                       for value in (5, 54, 6.0, True, None)]
        bad_fields += [("max_abs_logp_difference", value) for value in (-1., math.nan, math.inf, True, None)]
        for key, value in bad_fields:
            with self.subTest(key=key, value=value):
                self.assertFalse(entry.restore_allows_training(dict(good, **{key: value})))
        for value in (None, [], {}, "passed"):
            self.assertFalse(entry.restore_allows_training(value))

    def test_counter_passes_arguments_result_identity_and_attributes_unchanged(self):
        result, inputs = object(), object()
        model = Mock(return_value=result)
        model.some_existing_attribute = object()
        report = {"forward_attempts": 0, "forward_returns": 0}
        counter = entry.ForwardCounter(model, report)
        self.assertIs(counter(inputs, use_cache=False), result)
        model.assert_called_once_with(inputs, use_cache=False)
        self.assertIs(counter.some_existing_attribute, model.some_existing_attribute)
        self.assertEqual(report, {"forward_attempts": 1, "forward_returns": 1})

    def test_counter_retains_failed_attempt_without_claiming_a_return(self):
        model = Mock(side_effect=RuntimeError("synthetic call failure"))
        report = {"forward_attempts": 6, "forward_returns": 6}
        with self.assertRaisesRegex(RuntimeError, "synthetic call failure"):
            entry.ForwardCounter(model, report)("synthetic input")
        self.assertEqual(report, {"forward_attempts": 7, "forward_returns": 6})
        model.assert_called_once()

    def test_counter_refuses_2709th_call_before_invoking_wrapped_object(self):
        model = Mock(return_value="last allowed")
        report = {"forward_attempts": 2707, "forward_returns": 2707}
        counter = entry.ForwardCounter(model, report)
        self.assertEqual(counter(), "last allowed")
        with self.assertRaisesRegex(ValueError, "Forward ceiling"):
            counter()
        self.assertEqual(report, {"forward_attempts": 2708, "forward_returns": 2708})
        model.assert_called_once()


class ImportAndEntryTests(unittest.TestCase):
    def test_import_has_no_heavy_dependency_or_process_side_effect(self):
        with patch("builtins.__import__", side_effect=forbid_heavy_import(builtins.__import__)), \
                patch("subprocess.Popen") as child, patch.object(Path, "write_text") as write, \
                patch.object(Path, "write_bytes") as write_bytes:
            module = load_entry()
        self.assertEqual(module.STAGE, "aps_warm_qk_train_eval")
        self.assertEqual((module.TOTAL_SECONDS, module.RESERVE_SECONDS, module.MAX_FORWARDS), (1500, 60, 2708))
        self.assertEqual(set(module.PINS), {
            "outputs/aps-warm-qk/process-completion.json", "outputs/grounding-data/manifest.json",
            "outputs/aps-warm-qk/report.json", "outputs/grounding-assisted-point/report.json",
            "outputs/aps-warm-qk/adapter.safetensors"})
        for digest in module.PINS.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        child.assert_not_called()
        write.assert_not_called()
        write_bytes.assert_not_called()

    def test_invalid_budget_or_origin_rejects_before_read_or_write(self):
        cases = [(seconds, "1") for seconds in (0, 1499, 1501)]
        cases += [(1500, origin) for origin in ("0", "-1", "nan", "inf", "101")]
        for seconds, origin in cases:
            with self.subTest(seconds=seconds, origin=origin), \
                    patch.object(entry, "PROGRAM_STARTED", 100.), \
                    patch.object(Path, "read_text") as read, patch.object(Path, "open") as opened, \
                    self.assertRaises(ValueError):
                entry.main(["--seconds", str(seconds), "--origin-start", origin])
            read.assert_not_called()
            opened.assert_not_called()

    def test_foreign_parent_or_reserved_result_preserved_before_heavy_import(self):
        for retained in (None, "report.json", "report.tmp", "run-start.json"):
            with self.subTest(retained=retained), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                parent = {"launcher_pid": -1 if retained is None else entry.os.getppid(),
                          "origin_monotonic": 1., "seconds": 1500,
                          "runner_sha256": entry.file_sha256(entry.__file__)}
                (out / "process-start.json").write_text(json.dumps(parent))
                if retained:
                    (out / retained).write_text("historical bytes")
                with patch.object(entry, "OUTPUT", out), \
                        patch("builtins.__import__", side_effect=forbid_heavy_import(builtins.__import__)), \
                        self.assertRaises(ValueError):
                    entry.main(["--origin-start", "1"])
                self.assertFalse((out / "runner-at-run.py").exists())
                if retained:
                    self.assertEqual((out / retained).read_text(), "historical bytes")
                else:
                    self.assertFalse((out / "run-start.json").exists())

    def test_wrong_initial_pin_stops_before_json_selection_or_heavy_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            original_hash = entry.file_sha256
            parent = {"launcher_pid": entry.os.getppid(), "origin_monotonic": 100.,
                      "seconds": 1500, "runner_sha256": original_hash(entry.__file__)}
            (out / "process-start.json").write_text(json.dumps(parent))

            def changed_pin(path):
                path = Path(path)
                if path == entry.MANIFEST:
                    return "0" * 64
                if path.is_relative_to(entry.ROOT):
                    relative = path.relative_to(entry.ROOT).as_posix()
                    if relative in entry.PINS:
                        return entry.PINS[relative]
                return original_hash(path)

            with patch.object(entry, "OUTPUT", out), patch.object(entry, "PROGRAM_STARTED", 101.), \
                    patch.object(entry.time, "perf_counter", return_value=102.), \
                    patch.object(entry.signal, "signal"), patch.object(entry.signal, "setitimer"), \
                    patch.object(entry, "file_sha256", side_effect=changed_pin), \
                    patch.object(entry, "prepare_selection") as select, \
                    patch.object(entry, "load_frozen_final_model") as load, \
                    patch("builtins.__import__", side_effect=forbid_heavy_import(builtins.__import__)), \
                    patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(entry.main(["--origin-start", "100"]), 1)
            select.assert_not_called()
            load.assert_not_called()
            report = json.loads((out / "report.json").read_text())
            self.assertEqual((report["status"], report["failed_phase"]), ("error", "source_validation"))
            self.assertIn("Pinned source input changed", report["error"])
            self.assertFalse(report["valid_complete"])
            self.assertEqual((report["forward_attempts"], report["forward_returns"]), (0, 0))
            self.assertFalse((out / "eval-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
