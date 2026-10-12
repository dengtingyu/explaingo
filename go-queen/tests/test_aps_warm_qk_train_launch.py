"""Stdlib-only checks: synthetic reports, fake child and clock.

No test calls the real runner. The real runner and model are never started by these tests.
"""
from __future__ import annotations

import builtins
import copy
from contextlib import ExitStack
import importlib.util
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

DRAFT = Path(__file__).resolve().parents[1] / "scripts/run_aps_warm_qk_train.py"


def load_draft():
    spec = importlib.util.spec_from_file_location("warm_train_eval_launcher_draft", DRAFT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = load_draft()


def completed_report():
    train = [{"question_id": f"train-{i}"} for i in range(2702)]
    restore = [{"question_id": f"restore-{i}"} for i in range(6)]
    rows = lambda questions: [{**q, "label_log_probs": [-.2, -2.2, -3.2],
                               "off_label_mass": .03} for q in questions]
    return {
        "stage": "aps_warm_qk_train_eval", "status": "complete", "valid_complete": True,
        "budget_exceeded": False, "training_questions": train, "training_rows": rows(train),
        "restoration_questions": restore, "restoration_rows": rows(restore),
        "forward_attempts": 2708, "forward_returns": 2708,
        "base_sha256_before": "a" * 64, "base_sha256_after": "a" * 64,
        "adapter_sha256_before": "b" * 64, "adapter_sha256_after": "b" * 64,
        "source_hashes_unchanged": True, "features_unchanged": True,
        "manifest_rebuilt_equal": True, "runtime_binding_cleared": True,
        "all_parameters_frozen_and_gradients_absent": True,
        "optimizer_constructed": False, "optimizer_updates": 0, "backward_calls": 0,
        "restoration": {"passed": True, "coverage_complete": True, "predictions_match": True,
                        "current_questions": 6, "saved_questions": 6,
                        "max_abs_logp_difference": 0.0, "tolerance": 1e-5},
        "summary": {"summary": {"coverage_complete": True, "completed_questions": 2702,
                                 "requested_questions": 2702, "forced_choice_correct": 0}},
    }


class EndpointTests(unittest.TestCase):
    def accepts(self, report=None, exit_code=0, elapsed=600., cap=1500):
        return launch.effective_complete(completed_report() if report is None else report,
                                         exit_code, elapsed, cap)

    def test_complete_execution_needs_no_accuracy_threshold(self):
        self.assertTrue(self.accepts())  # Synthetic accuracy is deliberately zero.
        self.assertTrue(self.accepts(elapsed=1500))

    def test_nonzero_signal_or_overrun_is_not_repaired_by_complete_rows(self):
        for code in (1, -9, -15, None, False):
            with self.subTest(code=code):
                self.assertFalse(self.accepts(exit_code=code))
        for elapsed in (-1, 1500.0001, math.nan, math.inf, True, "600"):
            with self.subTest(elapsed=elapsed):
                self.assertFalse(self.accepts(elapsed=elapsed))
        for cap in (1499, 1501, 1500., False, None):
            with self.subTest(cap=cap):
                self.assertFalse(self.accepts(cap=cap))

    def test_missing_duplicate_reordered_or_foreign_rows_cannot_be_full_coverage(self):
        for group in ("training", "restoration"):
            for mutation in ("missing", "extra", "duplicate", "reorder", "foreign"):
                report = completed_report()
                rows = report[group + "_rows"]
                if mutation == "missing":
                    rows.pop()
                elif mutation == "extra":
                    rows.append(copy.deepcopy(rows[0]))
                elif mutation == "duplicate":
                    rows[-1]["question_id"] = rows[0]["question_id"]
                elif mutation == "reorder":
                    rows.reverse()
                else:
                    rows[-1]["question_id"] = "not-selected"
                with self.subTest(group=group, mutation=mutation):
                    self.assertFalse(self.accepts(report))
        report = completed_report()
        report["restoration_questions"][0]["question_id"] = "train-0"
        report["restoration_rows"][0]["question_id"] = "train-0"
        self.assertFalse(self.accepts(report))

    def test_nonfinite_malformed_scores_and_probabilities_fail_closed(self):
        for group in ("training", "restoration"):
            for scores in ([], [-1, -2], [-1, -2, -3, -4], [math.nan, -1, -2],
                           [math.inf, -1, -2], [True, -1, -2], [.1, -1, -2]):
                report = completed_report()
                report[group + "_rows"][0]["label_log_probs"] = scores
                with self.subTest(group=group, scores=scores):
                    self.assertFalse(self.accepts(report))
            for mass in (math.nan, -0.1, 1.1, False):
                report = completed_report()
                report[group + "_rows"][0]["off_label_mass"] = mass
                self.assertFalse(self.accepts(report))

    def test_no_update_no_extra_forward_and_all_integrity_are_required(self):
        for key, value in (("stage", "aps_original54_transfer"), ("status", "stopped"),
                           ("valid_complete", False), ("budget_exceeded", True),
                           ("optimizer_constructed", True), ("optimizer_updates", 1),
                           ("backward_calls", 1), ("optimizer_updates", False),
                           ("forward_attempts", 2709), ("forward_returns", 2707),
                           ("forward_returns", 2708.0)):
            report = completed_report()
            report[key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(self.accepts(report))
        for key in ("source_hashes_unchanged", "features_unchanged", "manifest_rebuilt_equal",
                    "runtime_binding_cleared", "all_parameters_frozen_and_gradients_absent"):
            report = completed_report()
            report[key] = False
            self.assertFalse(self.accepts(report))

    def test_restoration_is_exact_six_and_fixed_tolerance(self):
        for key, value in (("passed", False), ("coverage_complete", False),
                           ("predictions_match", False), ("current_questions", 5),
                           ("saved_questions", 54), ("max_abs_logp_difference", 1.1e-5),
                           ("max_abs_logp_difference", math.nan), ("tolerance", 1e-4)):
            report = completed_report()
            report["restoration"][key] = value
            with self.subTest(key=key, value=value):
                self.assertFalse(self.accepts(report))

    def test_measured_digests_and_nested_summary_are_required(self):
        for prefix in ("base", "adapter"):
            report = completed_report()
            report[prefix + "_sha256_after"] = "c" * 64
            self.assertFalse(self.accepts(report))
            for value in (None, "", "z" * 64):
                report = completed_report()
                report[prefix + "_sha256_before"] = report[prefix + "_sha256_after"] = value
                self.assertFalse(self.accepts(report))
        for value in (None, [], {}, {"coverage_complete": True}, {"summary": None}):
            report = completed_report()
            report["summary"] = value
            self.assertFalse(self.accepts(report))
        report = completed_report()
        report["summary"]["summary"]["completed_questions"] = 2701
        self.assertFalse(self.accepts(report))
        for report in ([], {}, "complete"):
            self.assertFalse(launch.effective_complete(report, 0, 600, 1500))


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="warm-train-eval-launch-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        (self.root / "scripts").mkdir()
        (self.root / "outputs").mkdir()
        self.runner = self.root / "scripts/eval_aps_warm_qk_train.py"
        self.launcher = self.root / "scripts/run_aps_warm_qk_train.py"
        self.runner.write_text("# Synthetic source, never executed.\n")
        self.launcher.write_text("# Synthetic launcher bytes.\n")
        self.output = self.root / "outputs/aps-warm-qk-train-eval"
        self.child = mock.Mock(pid=1234)
        self.child.wait.return_value = 0

    def run_fake(self, *, report=None, raw=None, missing=False, mutate=None,
                 spawn_error=None, clock=None, completion_write_error=False):
        report = completed_report() if report is None else report
        report.setdefault("source_script_sha256", launch.file_sha256(self.runner))

        def spawn(command, **kwargs):
            self.assertEqual(command, [sys.executable, "-B", str(self.runner), "--seconds", "1500",
                                       "--origin-start", "100.0"])
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertEqual(Path(kwargs["stdout"].name), self.output / "process-stdout.log")
            self.assertEqual(Path(kwargs["stderr"].name), self.output / "process-stderr.log")
            start = json.loads((self.output / "process-start.json").read_text())
            self.assertEqual((start["origin_monotonic"], start["seconds"], start["reserve_seconds"]),
                             (100.0, 1500, 60))
            self.assertEqual(start["runner_sha256"], launch.file_sha256(self.runner))
            self.assertFalse(start["automatic_retry"])
            self.assertEqual((self.output / "launcher-at-run.py").read_bytes(), self.launcher.read_bytes())
            if spawn_error:
                raise spawn_error
            if not missing:
                (self.output / "report.json").write_text(raw if raw is not None else json.dumps(report))
            if mutate:
                mutate()
            return self.child

        original_open = Path.open

        def open_path(path, *args, **kwargs):
            if completion_write_error and path.name == "process-completion.json" and args == ("x",):
                raise OSError("synthetic completion write failure")
            return original_open(path, *args, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(launch, "ROOT", self.root))
            stack.enter_context(mock.patch.object(launch, "OUTPUT", self.output))
            stack.enter_context(mock.patch.object(launch, "__file__", str(self.launcher)))
            stack.enter_context(mock.patch.object(launch, "PROGRAM_STARTED", 100.0))
            stack.enter_context(mock.patch.object(launch.time, "perf_counter", side_effect=clock)
                                if clock is not None else mock.patch.object(launch.time, "perf_counter", return_value=700.0))
            popen = stack.enter_context(mock.patch.object(launch.subprocess, "Popen", side_effect=spawn))
            stack.enter_context(mock.patch.object(Path, "open", new=open_path))
            stdout = stack.enter_context(mock.patch("sys.stdout", new_callable=io.StringIO))
            if completion_write_error:
                with self.assertRaises(OSError):
                    launch.main([])
                return None, None, popen, [json.loads(line) for line in stdout.getvalue().splitlines()]
            code = launch.main([])
        result = json.loads((self.output / "process-completion.json").read_text())
        return code, result, popen, [json.loads(line) for line in stdout.getvalue().splitlines()]

    def test_one_mock_child_preserves_shared_origin_and_terminates_no_extra_work(self):
        code, result, popen, printed = self.run_fake()
        self.assertEqual(code, 0)
        self.assertTrue(result["valid_complete"])
        self.assertEqual(result["seconds_to_child_exit"], 600.)
        self.assertTrue(result["source_unchanged"])
        self.assertTrue(result["report_source_matches"])
        popen.assert_called_once()
        self.child.wait.assert_called_once_with(timeout=930.)
        self.child.terminate.assert_not_called()
        self.child.kill.assert_not_called()
        self.assertEqual(printed[0]["child_exit_code"], 0)
        self.assertEqual(printed[-1]["seconds_through_completion_record"], 600.)

    def test_existing_partial_directory_is_preserved_before_hash_or_spawn(self):
        self.output.mkdir()
        retained = self.output / "run-start.json"
        retained.write_bytes(b"historical partial attempt")
        with mock.patch.object(launch, "OUTPUT", self.output), \
                mock.patch.object(launch, "file_sha256") as hashing, \
                mock.patch.object(launch.subprocess, "Popen") as popen:
            with self.assertRaises(FileExistsError):
                launch.main([])
        hashing.assert_not_called()
        popen.assert_not_called()
        self.assertEqual(retained.read_bytes(), b"historical partial attempt")

    def test_different_budget_or_output_is_rejected_without_creating_attempt(self):
        with mock.patch.object(launch, "OUTPUT", self.output), \
                mock.patch.object(launch.subprocess, "Popen") as popen:
            for cap in (0, 1499, 1501):
                with self.subTest(cap=cap), self.assertRaises(ValueError):
                    launch.main(["--seconds", str(cap)])
            with mock.patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
                launch.main(["--output", "elsewhere"])
        popen.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_partial_zero_exit_is_preserved_but_not_complete_and_never_retried(self):
        report = completed_report()
        report.update(status="stopped", valid_complete=False, training_rows=[], forward_returns=6)
        code, result, popen, _ = self.run_fake(report=report)
        self.assertEqual(code, 0)
        self.assertFalse(result["valid_complete"])
        self.assertFalse(result["automatic_retry"])
        popen.assert_called_once()

    def test_truncated_report_retains_exit_zero_with_parse_failure(self):
        code, result, popen, printed = self.run_fake(raw='{"status":"complete"')
        self.assertEqual(code, 1)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["report_parse_error"], "JSONDecodeError")
        self.assertFalse(result["valid_complete"])
        self.assertEqual(printed[0]["child_exit_code"], 0)
        popen.assert_called_once()

    def test_missing_report_records_hash_failure_without_losing_child_outcome(self):
        code, result, _, printed = self.run_fake(missing=True)
        self.assertEqual(code, 1)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["hash_errors"]["report"], "FileNotFoundError")
        self.assertFalse(result["valid_complete"])
        self.assertEqual(printed[0]["child_exit_code"], 0)

    def test_spawn_error_is_terminal_without_second_attempt(self):
        code, result, popen, _ = self.run_fake(spawn_error=OSError("synthetic spawn failure"))
        self.assertEqual(code, 1)
        self.assertIsNone(result["exit_code"])
        self.assertEqual(result["launch_error"], "OSError")
        self.child.wait.assert_not_called()
        popen.assert_called_once()

    def test_signal_and_exhausted_native_grace_do_not_extend_valid_budget(self):
        self.child.wait.side_effect = [subprocess.TimeoutExpired("fake", 1530),
                                       subprocess.TimeoutExpired("fake", 5), -9]
        code, result, popen, _ = self.run_fake()
        self.assertEqual(code, 1)
        self.assertEqual(result["terminated_by_signal"], 9)
        self.assertTrue(result["terminated_after_grace"])
        self.assertFalse(result["valid_complete"])
        self.child.terminate.assert_called_once_with()
        self.child.kill.assert_called_once_with()
        self.assertEqual(self.child.wait.call_args_list, [mock.call(timeout=930.), mock.call(timeout=5), mock.call()])
        popen.assert_called_once()

    def test_source_deletion_after_exit_still_records_terminal_evidence(self):
        code, result, _, printed = self.run_fake(mutate=self.runner.unlink)
        self.assertEqual(code, 1)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["hash_errors"]["runner"], "FileNotFoundError")
        self.assertFalse(result["source_unchanged"])
        self.assertFalse(result["valid_complete"])
        self.assertEqual(printed[0]["child_exit_code"], 0)

    def test_report_from_different_runner_cannot_be_promoted(self):
        report = completed_report()
        report["source_script_sha256"] = "c" * 64
        code, result, _, _ = self.run_fake(report=report)
        self.assertEqual(code, 0)
        self.assertFalse(result["report_source_matches"])
        self.assertFalse(result["valid_complete"])

    def test_changed_launcher_source_invalidates_an_exit_zero_report(self):
        code, result, _, _ = self.run_fake(mutate=lambda: self.launcher.write_text("# changed\n"))
        self.assertEqual(code, 0)
        self.assertFalse(result["source_unchanged"])
        self.assertFalse(result["valid_complete"])

    def test_post_exit_parsing_and_hash_time_cannot_use_the_native_grace(self):
        code, result, _, _ = self.run_fake(clock=[101., 1599., 1600.01, 1600.02, 1600.03])
        self.assertEqual(code, 0)
        self.assertEqual(result["seconds_to_child_exit"], 1499.)
        self.assertGreater(result["seconds_before_completion_record"], 1500)
        self.assertTrue(result["budget_exceeded"])
        self.assertFalse(result["valid_complete"])

    def test_completion_write_crossing_cap_durably_revokes_validity(self):
        code, result, _, printed = self.run_fake(clock=[101., 1599., 1599.5, 1600.01, 1600.02])
        self.assertEqual(code, 0)
        self.assertEqual(result["seconds_to_child_exit"], 1499.)
        self.assertEqual(result["seconds_before_completion_record"], 1499.5)
        self.assertGreater(result["seconds_after_initial_completion_record"], 1500)
        self.assertTrue(result["budget_exceeded"])
        self.assertFalse(result["valid_complete"])
        self.assertFalse(printed[-1]["valid_complete"])

    def test_child_terminal_evidence_survives_completion_file_failure(self):
        _, _, popen, printed = self.run_fake(completion_write_error=True)
        popen.assert_called_once()
        self.assertEqual(len(printed), 1)
        self.assertEqual(printed[0]["child_exit_code"], 0)
        self.assertEqual(printed[0]["seconds_to_child_exit"], 600.)


class ImportBoundaryTests(unittest.TestCase):
    def test_module_import_never_requests_heavy_dependencies_or_starts_a_child(self):
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "numpy", "transformers", "safetensors", "goqueen"}:
                raise AssertionError("Launcher attempted a heavy import: " + name)
            return original(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded), \
                mock.patch("subprocess.Popen") as popen:
            module = load_draft()
        self.assertEqual(module.STAGE, "aps_warm_qk_train_eval")
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
