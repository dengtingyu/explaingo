"""Process acceptance and rejection checks without launching a training child."""
from __future__ import annotations

import copy
import io
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_grounding_assisted_point as launch


def completed_report(category="beyond_additive"):
    questions = [{"question_id": f"q{i}"} for i in range(21)]
    return {
        "stage": "a0_assisted_point", "status": "complete", "run_completed": True, "updates_completed": 180,
        "validity": {"endpoint": True}, "fixed_questions": questions,
        "evaluations": [{"name": "initial", "updates_completed": 0},
                        {"name": "final", "updates_completed": 180,
                         "coverage_complete": True, "completed_questions": 21,
                         "rows": copy.deepcopy(questions)}],
        "frozen_base_final_hash_status": "measured",
        "frozen_base_sha256_before": "a" * 64,
        "frozen_base_sha256_after": "a" * 64,
        "frozen_base_gradients_absent": True,
        "final_assisted_point_endpoint": {"category": category},
    }


class EndpointTests(unittest.TestCase):
    def endpoint(self, report, **kwargs):
        values = {"exit_code": 0, "wall_seconds": 218.0, "cap_seconds": 300.0}
        values.update(kwargs)
        return launch.effective_endpoint(report, **values)

    def assert_invalid(self, report, **kwargs):
        result = self.endpoint(report, **kwargs)
        self.assertIs(result["valid"], False)
        self.assertIsNone(result["category"])

    def test_complete_categories_include_finite_failure(self):
        for category in ("full_fit", "beyond_additive", "not_beyond_additive"):
            with self.subTest(category=category):
                result = self.endpoint(completed_report(category))
                self.assertIs(result["valid"], True)
                self.assertEqual(result["category"], category)

    def test_os_exit_and_signal_override_success_claim(self):
        for code in (1, -15, -9, None, False):
            with self.subTest(code=code):
                self.assert_invalid(completed_report("full_fit"), exit_code=code)

    def test_external_total_cap_and_nonfinite_times(self):
        for wall in (-0.1, 300.000001, math.inf, math.nan, False):
            with self.subTest(wall=wall):
                self.assert_invalid(completed_report(), wall_seconds=wall)
        for cap in (0, -1, 301, math.inf, math.nan, True):
            with self.subTest(cap=cap):
                self.assert_invalid(completed_report(), cap_seconds=cap)
        self.assertTrue(self.endpoint(completed_report(), wall_seconds=300)["valid"])

    def test_partial_run_and_reproduction_failure_override_category(self):
        for field, value in (("status", "incomplete_budget"), ("run_completed", False),
                             ("updates_completed", 179), ("validity", {"endpoint": False}),
                             ("validity", None)):
            with self.subTest(field=field, value=value):
                report = completed_report()
                report[field] = value
                self.assert_invalid(report)

    def test_non_aps_stage_cannot_reuse_a_successful_report(self):
        for stage in (None, "a0_m_key", "a0_value_token", "m_assisted_point"):
            with self.subTest(stage=stage):
                report = completed_report("full_fit")
                report["stage"] = stage
                self.assert_invalid(report)

    def test_final_complete_marker_does_not_replace_actual_21_rows(self):
        for count in (0, 20, 22):
            with self.subTest(count=count):
                report = completed_report()
                report["evaluations"][1]["rows"] = [{"question_id": f"q{i}"} for i in range(count)]
                self.assert_invalid(report)
        report = completed_report()
        report["evaluations"][1]["rows"].append(None)
        self.assert_invalid(report)

    def test_final_ids_unique_and_match_fixed_factual_selection(self):
        for mutation in ("duplicate", "foreign", "missing_id", "non_string"):
            with self.subTest(mutation=mutation):
                report = completed_report()
                row = report["evaluations"][1]["rows"][20]
                if mutation == "duplicate":
                    row["question_id"] = "q0"
                elif mutation == "foreign":
                    row["question_id"] = "other_question"
                elif mutation == "missing_id":
                    row.clear()
                else:
                    row["question_id"] = 20
                self.assert_invalid(report)
        report = completed_report()
        report["fixed_questions"][20]["question_id"] = "q0"
        self.assert_invalid(report)

    def test_exactly_one_final_evaluation_at_fixed_endpoint(self):
        for mutation in ("absent", "duplicate", "false_coverage", "wrong_count", "intermediate"):
            with self.subTest(mutation=mutation):
                report = completed_report()
                final = report["evaluations"][1]
                if mutation == "absent":
                    report["evaluations"].pop()
                elif mutation == "duplicate":
                    report["evaluations"].append(copy.deepcopy(final))
                elif mutation == "false_coverage":
                    final["coverage_complete"] = False
                elif mutation == "wrong_count":
                    final["completed_questions"] = 20
                else:
                    final["updates_completed"] = 120
                self.assert_invalid(report)

    def test_base_equality_requires_measured_valid_digest_and_no_gradient(self):
        for field, value in (("frozen_base_final_hash_status", "not_measured"),
                             ("frozen_base_sha256_after", "b" * 64),
                             ("frozen_base_gradients_absent", False)):
            with self.subTest(field=field):
                report = completed_report()
                report[field] = value
                self.assert_invalid(report)
        for digest in (None, "", "a" * 63, "z" * 64):
            with self.subTest(digest=digest):
                report = completed_report()
                report["frozen_base_sha256_before"] = report["frozen_base_sha256_after"] = digest
                self.assert_invalid(report)

    def test_unknown_or_missing_category_is_unaccepted(self):
        for category in (None, "partial", "passed", "accuracy_crossing"):
            with self.subTest(category=category):
                report = completed_report(category)
                result = self.endpoint(report)
                self.assertIs(result["valid"], False)
                self.assertIsNone(result["category"])
                self.assertEqual(result["child_claimed_category"], category)

    def test_malformed_or_missing_report_never_gains_acceptance(self):
        for report in (None, [], "complete", {}, {"final_assisted_point_endpoint": []},
                       {"evaluations": "not_a_list"}):
            with self.subTest(report=report):
                self.assert_invalid(report)


class PreLaunchRejectionTests(unittest.TestCase):
    def reject_without_side_effects(self, args, *, existing=()):
        existing = {Path(p).resolve() for p in existing}
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", ["run_grounding_assisted_point.py", *args]))
            child = stack.enter_context(mock.patch.object(launch.subprocess, "Popen",
                side_effect=AssertionError("A rejected launch must not create a child")))
            hash_source = stack.enter_context(mock.patch.object(launch, "file_sha256",
                side_effect=AssertionError("Rejection precedes reading the runner or heavy assets")))
            writes = [stack.enter_context(mock.patch.object(Path, method,
                       side_effect=AssertionError("A rejected launch must not write an artifact")))
                      for method in ("mkdir", "open", "write_text", "write_bytes")]
            stack.enter_context(mock.patch.object(Path, "exists", lambda p: p.resolve() in existing))
            with self.assertRaises(ValueError):
                launch.main()
            child.assert_not_called()
            hash_source.assert_not_called()
            for write in writes:
                write.assert_not_called()

    def test_bad_budget_rejected_before_process_and_artifacts(self):
        for seconds in ("0", "-1", "301"):
            with self.subTest(seconds=seconds):
                self.reject_without_side_effects(["--seconds", seconds])

    def test_original_m_and_p_directories_are_protected(self):
        for directory in ("grounding-microfit", "grounding-prompt-control", "grounding-m-key",
                          "assisted-point-engineering", "assisted-point-unit-check"):
            with self.subTest(directory=directory):
                self.reject_without_side_effects(["--output", str(ROOT / "outputs" / directory / "new-report.json")])

    def test_path_outside_project_outputs_is_rejected(self):
        for output in (ROOT / "report.json", ROOT / "src/report.json", Path("/tmp/aps-report.json")):
            with self.subTest(output=output):
                self.reject_without_side_effects(["--output", str(output)])

    def test_reserved_names_cannot_alias_process_or_child_artifacts(self):
        for name in ("process-start.json", "process-completion.json", "run-start.json",
                     "PROCESS-START.json", "adapter.safetensors", "runner-at-run.py"):
            with self.subTest(name=name):
                self.reject_without_side_effects(["--output", str(ROOT / "outputs/aps-unused-tests" / name)])

    def test_existing_any_retained_artifact_blocks_resume(self):
        directory = ROOT / "outputs/aps-unused-tests"
        for name in ("report.json", "adapter.safetensors", "adapter.partial.safetensors",
                     "report.tmp", "runner-at-run.py", "run-start.json",
                     "process-start.json", "process-completion.json", "process-stdout.log", "process-stderr.log",
                     "launcher-at-run.py"):
            with self.subTest(name=name):
                self.reject_without_side_effects(["--output", str(directory / "report.json")],
                                                 existing=[directory / name])


class ProcessLifecycleTests(unittest.TestCase):
    """Exercise real file records and orchestration with a mocked child process.

    No subprocess, model, tensor library, optimizer or training is launched.
    """

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="goqueen-aps-launch-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        scripts = self.root / "scripts"
        scripts.mkdir()
        self.runner = scripts / "train_grounding_assisted_point.py"
        self.runner.write_text("# Synthetic runner; it is never executed.\n")
        self.launcher = scripts / "run_grounding_assisted_point.py"
        self.launcher.write_text("# Synthetic launcher source pin.\n")
        self.output = self.root / "outputs/grounding-assisted-point/report.json"
        self.child = mock.Mock(pid=4321)
        self.child.wait.return_value = 0

    def run_mocked(self, *, report=None, wall=218.0, seconds=300,
                   missing_report=False, raw_report=None, mutate=None, spawn_error=None):
        if report is None:
            report = completed_report()

        def spawn(command, **kwargs):
            self.assertEqual(command,
                [sys.executable, "-B", str(self.runner), "--seconds", str(seconds),
                 "--output", str(self.output)])
            self.assertEqual(kwargs["cwd"], self.root)
            self.assertEqual(Path(kwargs["stdout"].name), self.output.parent / "process-stdout.log")
            self.assertEqual(Path(kwargs["stderr"].name), self.output.parent / "process-stderr.log")
            if spawn_error is not None:
                raise spawn_error
            if not missing_report:
                self.output.write_text(raw_report if raw_report is not None else json.dumps(report))
            if mutate is not None:
                mutate()
            return self.child

        clock = iter([100.1, 100 + wall] if spawn_error is None else [100 + wall])
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(launch, "ROOT", self.root))
            stack.enter_context(mock.patch.object(launch, "__file__", str(self.launcher)))
            stack.enter_context(mock.patch.object(launch, "PROGRAM_STARTED", 100.0))
            stack.enter_context(mock.patch.object(launch.time, "perf_counter", lambda: next(clock)))
            stack.enter_context(mock.patch.object(launch, "relative",
                lambda path: Path(path).resolve().relative_to(self.root).as_posix()))
            stack.enter_context(mock.patch.object(sys, "argv", [str(self.launcher),
                "--seconds", str(seconds), "--output", str(self.output)]))
            popen = stack.enter_context(mock.patch.object(launch.subprocess, "Popen", side_effect=spawn))
            console = io.StringIO()
            stack.enter_context(mock.patch.object(sys, "stdout", console))
            code = launch.main()
            popen.assert_called_once()
        result = json.loads((self.output.parent / "process-completion.json").read_text())
        self.assertEqual(json.loads(console.getvalue()), result)
        return code, result

    def test_completed_child_retains_pinned_sources_and_os_records(self):
        code, result = self.run_mocked()
        self.assertEqual(code, 0)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["child_pid"], 4321)
        self.assertEqual(result["external_wall_seconds"], 218.0)
        self.assertIsNone(result["terminated_by_signal"])
        self.assertFalse(result["automatic_retry"])
        self.assertTrue(result["effective_endpoint"]["valid"])
        start = json.loads((self.output.parent / "process-start.json").read_text())
        self.assertEqual(start["runner"], "scripts/train_grounding_assisted_point.py")
        self.assertEqual(start["runner_sha256"], launch.file_sha256(self.runner))
        self.assertEqual(start["launcher_sha256"], result["launcher_sha256"])
        self.assertEqual(result["launcher_at_run_copy_sha256"], result["launcher_sha256"])
        self.assertEqual((self.output.parent / "launcher-at-run.py").read_bytes(),
                         self.launcher.read_bytes())
        self.child.wait.assert_called_once_with(timeout=329.9)

    def test_overrun_keeps_successful_os_exit_but_invalidates_endpoint(self):
        code, result = self.run_mocked(report=completed_report("full_fit"), wall=300.01)
        self.assertEqual(code, 0)
        self.assertTrue(result["budget_exceeded"])
        self.assertFalse(result["effective_endpoint"]["valid"])
        self.assertIsNone(result["effective_endpoint"]["category"])
        self.assertEqual(result["effective_endpoint"]["child_claimed_category"], "full_fit")

    def test_missing_report_is_process_failure_and_keeps_exit_record(self):
        code, result = self.run_mocked(missing_report=True)
        self.assertEqual(code, 1)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["report_parse_error"], "FileNotFoundError")
        self.assertIsNone(result["report_sha256"])
        self.assertFalse(result["effective_endpoint"]["valid"])

    def test_array_report_cannot_claim_process_acceptance(self):
        code, result = self.run_mocked(raw_report="[]")
        self.assertEqual(code, 1)
        self.assertEqual(result["report_parse_error"], "ValueError")
        self.assertFalse(result["effective_endpoint"]["valid"])

    def test_runner_mutation_invalidates_a_complete_child_claim(self):
        code, result = self.run_mocked(mutate=lambda: self.runner.write_text("# Changed.\n"))
        self.assertEqual(code, 0)
        self.assertFalse(result["runner_source_unchanged"])
        self.assertFalse(result["effective_endpoint"]["valid"])
        self.assertIsNone(result["effective_endpoint"]["category"])

    def test_launcher_mutation_is_distinct_from_retained_source_copy(self):
        code, result = self.run_mocked(mutate=lambda: self.launcher.write_text("# Changed.\n"))
        self.assertEqual(code, 0)
        self.assertFalse(result["launcher_source_unchanged"])
        self.assertEqual(result["launcher_at_run_copy_sha256"], result["launcher_sha256"])
        self.assertFalse(result["effective_endpoint"]["valid"])

    def test_native_grace_terminates_once_without_retry(self):
        timeout = launch.subprocess.TimeoutExpired("synthetic", 330)
        self.child.wait.side_effect = [timeout, -15]
        code, result = self.run_mocked(wall=330.1)
        self.assertEqual(code, 1)
        self.assertEqual(result["exit_code"], -15)
        self.assertEqual(result["terminated_by_signal"], 15)
        self.assertTrue(result["terminated_after_native_return_grace"])
        self.child.terminate.assert_called_once_with()
        self.child.kill.assert_not_called()
        self.assertEqual(self.child.wait.call_args_list,
                         [mock.call(timeout=329.9), mock.call(timeout=5)])

    def test_child_unresponsive_to_terminate_is_killed_then_reaped(self):
        timeout = launch.subprocess.TimeoutExpired("synthetic", 330)
        self.child.wait.side_effect = [timeout, timeout, -9]
        code, result = self.run_mocked(wall=335.1)
        self.assertEqual(code, 1)
        self.assertEqual(result["terminated_by_signal"], 9)
        self.child.terminate.assert_called_once_with()
        self.child.kill.assert_called_once_with()
        self.assertEqual(self.child.wait.call_args_list,
                         [mock.call(timeout=329.9), mock.call(timeout=5), mock.call()])
        self.assertFalse(result["effective_endpoint"]["valid"])

    def test_spawn_failure_has_terminal_record_and_no_fake_os_exit(self):
        code, result = self.run_mocked(spawn_error=FileNotFoundError("synthetic"))
        self.assertEqual(code, 1)
        self.assertEqual(result["launch_error"], "FileNotFoundError")
        self.assertIsNone(result["exit_code"])
        self.assertIsNone(result["child_pid"])
        self.assertIsNone(result["terminated_by_signal"])
        self.child.wait.assert_not_called()
        self.assertFalse(result["effective_endpoint"]["valid"])


if __name__ == "__main__":
    unittest.main()
