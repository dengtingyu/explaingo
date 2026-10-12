"""Standard-library guards only; no model, feature decoding or evaluation runs."""
from __future__ import annotations

import builtins
import copy
from contextlib import ExitStack
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_assisted_point_transfer as runner


def selection_fixture():
    train = [{"question_id": f"train-{i}", "position_id": f"tp-{i // 3}",
              "game_id": "train-game", "prompt": f"train prompt {i}",
              "point_index": i, "label": i % 3, "to_play": "B", "split": "train"}
             for i in range(21)]
    val = [{"question_id": f"val-{i}", "position_id": f"vp-{i // 3}",
            "game_id": f"val-game-{i // 18}", "prompt": f"val prompt {i}",
            "point_index": 0 if i == 0 else 100 + i, "label": i % 3,
            "to_play": "B" if (i // 3) % 2 == 0 else "W", "split": "val"}
           for i in range(54)]
    fit = {"fixed_questions": train,
           "source": {"fixed_questions_sha256": runner.content_sha256(train)},
           "evaluations": [{"name": "final", "updates_completed": 180,
                            "coverage_complete": True,
                            "rows": [{"question_id": q["question_id"]} for q in train]}]}
    return {"original_occupancy_questions": val}, fit


def restoration_fixture():
    return {"passed": True, "coverage_complete": True, "predictions_match": True,
            "max_abs_logp_difference": 0.0, "tolerance": runner.TOLERANCE}


class ImportBoundaryTests(unittest.TestCase):
    def test_runner_module_import_needs_only_standard_library(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "transformers", "safetensors", "goqueen"}:
                self.fail("Pure preflight imported a tensor/model dependency: " + name)
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location("aps_transfer_import_guard", runner.__file__)
        module = importlib.util.module_from_spec(spec)
        with mock.patch("builtins.__import__", side_effect=guarded):
            spec.loader.exec_module(module)
        self.assertEqual(module.TOLERANCE, 1e-5)


class SelectionTests(unittest.TestCase):
    def test_preserves_exact_order_and_filters_only_existing_val_split(self):
        data, fit = selection_fixture()
        val = data["original_occupancy_questions"]
        val.reverse()
        data["original_occupancy_questions"] = [copy.deepcopy(fit["fixed_questions"][0]), *val]
        train, selected, saved, separation = runner.select_questions(data, fit)
        self.assertIs(train, fit["fixed_questions"])
        self.assertEqual([q["question_id"] for q in selected], [q["question_id"] for q in val])
        self.assertTrue(all(a is b for a, b in zip(selected, val)))
        self.assertIs(saved, fit["evaluations"][0]["rows"])
        self.assertEqual(separation, {"game_id_overlap": 0, "position_id_overlap": 0,
                                     "question_id_overlap": 0, "prompt_overlap": 0,
                                     "validation_cells_with_training_point": 1})

    def test_rejects_missing_or_extra_training_and_validation_cells(self):
        for group in ("train", "val"):
            for delta in (-1, 1):
                with self.subTest(group=group, delta=delta):
                    data, fit = selection_fixture()
                    rows = fit["fixed_questions"] if group == "train" else data["original_occupancy_questions"]
                    rows.pop() if delta < 0 else rows.append(copy.deepcopy(rows[-1]))
                    with self.assertRaisesRegex(ValueError, "Expected 21 restoration and 54"):
                        runner.select_questions(data, fit)

    def test_rejects_duplicate_question_identity_in_either_list(self):
        for group in ("train", "val"):
            data, fit = selection_fixture()
            rows = fit["fixed_questions"] if group == "train" else data["original_occupancy_questions"]
            rows[-1]["question_id"] = rows[0]["question_id"]
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "Duplicate"):
                runner.select_questions(data, fit)

    def test_rejects_changed_class_side_game_and_position_coverage(self):
        for field, value, message in (("label", 1, "class"), ("to_play", "W", "side"),
                                       ("game_id", "val-game-1", "game"),
                                       ("position_id", "unexpected-position", "position")):
            data, fit = selection_fixture()
            data["original_occupancy_questions"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                runner.select_questions(data, fit)

    def test_rejects_each_train_validation_overlap_without_changing_counts(self):
        for field in ("game_id", "position_id", "question_id", "prompt"):
            data, fit = selection_fixture()
            val = data["original_occupancy_questions"]
            previous = val[0][field]
            replacement = fit["fixed_questions"][0][field]
            for q in val:
                if q[field] == previous:
                    q[field] = replacement
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "overlap: " + field):
                runner.select_questions(data, fit)

    def test_coordinate_overlap_counts_validation_cells_not_distinct_coordinates(self):
        for number in (0, 2):
            data, fit = selection_fixture()
            val = data["original_occupancy_questions"]
            if number == 0:
                val[0]["point_index"] = 200
            else:
                val[1]["point_index"] = val[0]["point_index"]
            with self.subTest(number=number), self.assertRaisesRegex(ValueError, "Coordinate overlap"):
                runner.select_questions(data, fit)

    def test_retained_training_content_hash_prevents_silent_label_change(self):
        data, fit = selection_fixture()
        fit["fixed_questions"][0]["label"] = 2
        with self.assertRaisesRegex(ValueError, "Training list changed"):
            runner.select_questions(data, fit)

    def test_only_complete_saved_180_endpoint_is_a_restoration_reference(self):
        for mutation in ("missing", "duplicate", "intermediate", "partial", "few_rows"):
            data, fit = selection_fixture()
            final = fit["evaluations"][0]
            if mutation == "missing":
                fit["evaluations"] = []
            elif mutation == "duplicate":
                fit["evaluations"].append(copy.deepcopy(final))
            elif mutation == "intermediate":
                final["updates_completed"] = 120
            elif mutation == "partial":
                final["coverage_complete"] = False
            else:
                final["rows"].pop()
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "final reference incomplete"):
                runner.select_questions(data, fit)


class RestorationGateTests(unittest.TestCase):
    def test_zero_and_exact_declared_boundary_allow_validation(self):
        for difference in (0.0, runner.TOLERANCE):
            value = restoration_fixture()
            value["max_abs_logp_difference"] = difference
            self.assertTrue(runner.restore_allows_validation(value))

    def test_any_failed_coverage_prediction_or_pass_flag_stops_validation(self):
        for key in ("passed", "coverage_complete", "predictions_match"):
            for value in (False, None, 1):
                comparison = restoration_fixture()
                comparison[key] = value
                with self.subTest(key=key, value=value):
                    self.assertFalse(runner.restore_allows_validation(comparison))

    def test_nonfinite_negative_bool_or_over_tolerance_difference_stops(self):
        for value in (math.nan, math.inf, -math.inf, -1e-12, 1.000001e-5, True, None, "0"):
            comparison = restoration_fixture()
            comparison["max_abs_logp_difference"] = value
            with self.subTest(value=value):
                self.assertFalse(runner.restore_allows_validation(comparison))

    def test_cannot_relax_or_omit_the_declared_tolerance(self):
        for value in (None, 1e-4, 0, math.nan):
            comparison = restoration_fixture()
            comparison["tolerance"] = value
            with self.subTest(value=value):
                self.assertFalse(runner.restore_allows_validation(comparison))

    def test_malformed_restoration_record_stops_instead_of_becoming_success(self):
        for comparison in (None, [], "passed", {}, {"passed": True}):
            with self.subTest(comparison=comparison):
                self.assertFalse(runner.restore_allows_validation(comparison))


class ParentAndExclusiveOutputTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="aps-transfer-preflight-")
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name).resolve()
        self.parent = {"launcher_pid": 777, "origin_monotonic": 100.0, "seconds": 300}
        (self.output / "process-start.json").write_text(json.dumps(self.parent))

    def invoke(self, argv=None, **changes):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner, "OUTPUT", self.output))
            stack.enter_context(mock.patch.object(runner, "PROGRAM_STARTED", 100.0))
            stack.enter_context(mock.patch.object(runner.os, "getppid", return_value=777))
            signal_mock = stack.enter_context(mock.patch.object(runner.signal, "signal"))
            stack.enter_context(mock.patch.object(runner.signal, "setitimer"))
            for name, value in changes.items():
                stack.enter_context(mock.patch.object(runner, name, value))
            result = runner.main(argv or ["--seconds", "300", "--origin-start", "100"])
            return result, signal_mock

    def test_invalid_budget_or_origin_is_rejected_before_writing(self):
        for seconds, origin in ((15, "100"), (301, "100"), (300, "0"),
                                (300, "101"), (300, "nan"), (300, "inf")):
            with self.subTest(seconds=seconds, origin=origin), self.assertRaisesRegex(ValueError, "Invalid total budget"):
                self.invoke(["--seconds", str(seconds), "--origin-start", origin])
            self.assertEqual(sorted(p.name for p in self.output.iterdir()), ["process-start.json"])

    def test_parent_pid_origin_and_budget_must_all_match(self):
        for key, value in (("launcher_pid", 888), ("origin_monotonic", 99.0), ("seconds", 299)):
            record = dict(self.parent, **{key: value})
            (self.output / "process-start.json").write_text(json.dumps(record))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "single-run launcher"):
                self.invoke()
            self.assertEqual(sorted(p.name for p in self.output.iterdir()), ["process-start.json"])

    def test_each_retained_child_artifact_prevents_repeat_and_preserves_bytes(self):
        for name in ("report.json", "report.tmp", "run-start.json"):
            path = self.output / name
            path.write_bytes(b"retained")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Existing run retained"):
                self.invoke()
            self.assertEqual(path.read_bytes(), b"retained")
            path.unlink()

    def test_changed_pinned_input_stops_before_any_heavy_import(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"numpy", "torch", "transformers", "safetensors", "goqueen"}:
                self.fail("Changed input reached a heavy import")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded), \
                mock.patch.object(runner.time, "perf_counter", return_value=101.0), \
                mock.patch("sys.stdout", new_callable=io.StringIO):
            result, _ = self.invoke(file_sha256=lambda _p: "0" * 64,
                                    relative=lambda p: Path(p).name)
        self.assertEqual(result, 1)
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error"]["phase"], "source_validation")
        self.assertIn("Pinned APS inputs changed", report["error"]["message"])
        self.assertFalse(report["validation_started"])
        self.assertEqual(report["coverage"], {"restoration": 0, "validation": 0,
                                             "requested_restoration": 21, "requested_validation": 54})


if __name__ == "__main__":
    unittest.main()
