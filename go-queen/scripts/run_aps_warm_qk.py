#!/usr/bin/env python3
"""Launch one APS-warm Q/K run, preserving its external outcome without retries.

The output directory is exclusive. The total budget is soft: a native child
call gets a 30-second return grace before termination. Actual overruns cannot
receive a valid complete endpoint, even when the child exits successfully.
"""
from __future__ import annotations

import time

PROGRAM_STARTED = time.perf_counter()

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_grounding_microfit import file_sha256, require

STAGE = "aps_warm_qk"


def effective_complete(report, *, exit_code, wall_seconds, cap_seconds, source_sha256):
    """Combine the child's contract with actual process, time and source evidence."""
    if (not isinstance(report, dict) or type(exit_code) is not int or exit_code != 0
            or type(cap_seconds) is not int or not 1 <= cap_seconds <= 1800
            or type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds)
            or not 0 <= wall_seconds <= cap_seconds
            or not isinstance(source_sha256, str) or len(source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in source_sha256)):
        return False
    total, maximum = report.get("total_seconds"), report.get("max_seconds")
    conditions, questions, evaluations = (report.get(k) for k in
                                          ("endpoint_conditions", "validation_questions", "evaluations"))
    if (report.get("stage") != STAGE or report.get("status") != "complete"
            or report.get("run_completed") is not True
            or type(report.get("updates_completed")) is not int or report["updates_completed"] != 1000
            or type(report.get("sample_slots")) is not int or report["sample_slots"] != 4000
            or report.get("source_script_sha256") != source_sha256
            or type(total) not in (int, float) or not math.isfinite(total)
            or not 0 <= total <= wall_seconds
            or type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum != cap_seconds
            or not isinstance(conditions, dict) or not conditions
            or not all(value is True for value in conditions.values())
            or not isinstance(questions, list) or len(questions) != 54
            or not isinstance(evaluations, list) or len(evaluations) != 2):
        return False
    try:
        ids = [q["question_id"] for q in questions]
        if (not all(isinstance(qid, str) and qid for qid in ids) or len(set(ids)) != 54
                or [e["name"] for e in evaluations] != ["initial", "final"]):
            return False
        for evaluation in evaluations:
            rows = evaluation.get("rows")
            if (evaluation.get("coverage_complete") is not True
                    or not isinstance(rows, list) or len(rows) != 54
                    or [row["question_id"] for row in rows] != ids):
                return False
    except (KeyError, TypeError, AttributeError):
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/aps-warm-qk/report.json")
    args = parser.parse_args(argv)
    require(1 <= args.seconds <= 1800, "Total budget must be 1..1800 seconds")
    output = args.output.resolve()
    require(output.parent.parent == (ROOT / "outputs").resolve(),
            "Use a new direct child directory of outputs; retained directories are not reused")
    reserved = [output, output.with_suffix(".tmp")]
    reserved += [output.parent / name for name in (
        "process-start.json", "process-completion.json", "process-stdout.log", "process-stderr.log",
        "launcher-at-run.py", "run-start.json", "runner-at-run.py", "adapter.safetensors",
        "adapter.partial.safetensors")]
    require(output.suffix == ".json"
            and len({p.name.casefold() for p in reserved}) == len(reserved),
            "Output name must not collide with retained process or child artifacts")
    # Atomic directory creation protects any historical content, not only known filenames.
    output.parent.mkdir(parents=False, exist_ok=False)
    runner, launcher = ROOT / "scripts/train_aps_warm_qk.py", Path(__file__).resolve()
    source_sha, launcher_sha = file_sha256(runner), file_sha256(launcher)
    start = {"stage": STAGE, "launcher_pid": os.getpid(), "outer_start_monotonic": PROGRAM_STARTED,
             "total_soft_cap_seconds": args.seconds, "native_return_grace_seconds": 30,
             "runner": runner.relative_to(ROOT).as_posix(), "runner_sha256": source_sha,
             "launcher": launcher.relative_to(ROOT).as_posix(), "launcher_sha256": launcher_sha,
             "automatic_retry": False}
    with (output.parent / "process-start.json").open("x") as stream:
        json.dump(start, stream, indent=2, allow_nan=False)
        stream.write("\n")
    launcher_copy = output.parent / "launcher-at-run.py"
    with launcher_copy.open("xb") as stream:
        stream.write(launcher.read_bytes())
    require(file_sha256(launcher_copy) == launcher_sha, "Launcher copy differs from pinned source")
    code, child_pid, launch_error, killed = None, None, None, False
    with (output.parent / "process-stdout.log").open("x") as out, \
            (output.parent / "process-stderr.log").open("x") as err:
        try:
            child = subprocess.Popen([sys.executable, "-B", str(runner), "--seconds", str(args.seconds),
                                      "--output", str(output), "--outer-start", repr(PROGRAM_STARTED)],
                                     cwd=ROOT, stdout=out, stderr=err)
            child_pid = child.pid
            try:
                code = child.wait(timeout=max(.001, args.seconds + 30
                                               - (time.perf_counter() - PROGRAM_STARTED)))
            except subprocess.TimeoutExpired:
                killed = True
                child.terminate()
                try:
                    code = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
        except OSError as exc:
            launch_error = type(exc).__name__
    child_wall = time.perf_counter() - PROGRAM_STARTED
    print(json.dumps({"child_exit_code": code, "child_pid": child_pid,
                      "seconds_to_child_exit": child_wall}), flush=True)
    report, parse_error = {}, None
    try:
        report = json.loads(output.read_text())
        require(isinstance(report, dict), "Report must be an object")
    except (OSError, ValueError) as exc:
        parse_error = type(exc).__name__
    hashes, hash_errors = {}, {}
    for name, path in (("runner", runner), ("launcher", launcher),
                       ("launcher_copy", launcher_copy), ("report", output)):
        try:
            hashes[name] = file_sha256(path)
        except OSError as exc:
            hashes[name] = None
            hash_errors[name] = type(exc).__name__
    source_same = (hashes["runner"] == source_sha
                   and hashes["launcher"] == hashes["launcher_copy"] == launcher_sha)
    wall = time.perf_counter() - PROGRAM_STARTED
    result = {"stage": STAGE, "exit_code": code, "child_pid": child_pid,
              "terminated_by_signal": -code if type(code) is int and code < 0 else None,
              "launch_error": launch_error, "report_parse_error": parse_error,
              "seconds_to_child_exit": child_wall, "seconds_before_completion_record": wall,
              "timing_scope": "Origin precedes launcher imports; second duration includes child exit, log closure, report parsing and source/report hashing, but excludes this completion write. Printed terminal wall includes that write.",
              "total_soft_cap_seconds": args.seconds, "budget_exceeded": wall > args.seconds,
              "terminated_after_native_return_grace": killed, "automatic_retry": False,
              "runner_sha256": source_sha, "launcher_sha256": launcher_sha,
              "launcher_at_run_copy_sha256": hashes["launcher_copy"],
              "source_unchanged": source_same, "report_sha256": hashes["report"],
              "hash_errors": hash_errors,
              "valid_complete": (not parse_error and not launch_error and not killed and not hash_errors
                                 and source_same and effective_complete(report, exit_code=code,
                                     wall_seconds=wall, cap_seconds=args.seconds, source_sha256=source_sha))}
    completion = output.parent / "process-completion.json"
    with completion.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    after_write = time.perf_counter() - PROGRAM_STARTED
    if after_write > args.seconds:
        result.update(valid_complete=False, budget_exceeded=True,
                      seconds_after_initial_completion_record=after_write)
        completion.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({**result, "seconds_through_completion_record": time.perf_counter() - PROGRAM_STARTED},
                     allow_nan=False), flush=True)
    # Process success and protocol completion remain separate (a clean stop can exit 0).
    return 0 if code == 0 and not parse_error and not launch_error else 1


if __name__ == "__main__":
    raise SystemExit(main())
