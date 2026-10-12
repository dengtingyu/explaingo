#!/usr/bin/env python3
"""Launch one explicit M-APS fit and retain its external process outcome.

This entry point never resumes or retries a run. Its soft total budget includes
imports and setup; a delayed native child operation gets a 30-second return
grace, followed by termination. A report category is accepted only when both
the child validity checks and the measured external outcome pass.
"""
from __future__ import annotations

import time

PROGRAM_STARTED = time.perf_counter()

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_grounding_microfit import file_sha256, relative, require


def effective_endpoint(report, *, exit_code, wall_seconds, cap_seconds):
    """Require external evidence in addition to the child's measured contract."""
    if not isinstance(report, dict):
        report = {}
    endpoint = report.get("final_assisted_point_endpoint")
    claimed = endpoint.get("category") if isinstance(endpoint, dict) else None
    evaluations = report.get("evaluations")
    finals = ([e for e in evaluations if isinstance(e, dict) and e.get("name") == "final"]
              if isinstance(evaluations, list) else [])
    final = finals[0] if len(finals) == 1 else {}
    rows, questions = final.get("rows"), report.get("fixed_questions")
    final_ids = ([r.get("question_id") for r in rows if isinstance(r, dict)]
                 if isinstance(rows, list) else [])
    planned_ids = ([q.get("question_id") for q in questions if isinstance(q, dict)]
                   if isinstance(questions, list) else [])
    complete_final = (final.get("coverage_complete") is True
                      and final.get("completed_questions") == 21
                      and final.get("updates_completed") == 180
                      and isinstance(rows, list) and len(rows) == 21
                      and isinstance(questions, list) and len(questions) == 21
                      and len(final_ids) == len(planned_ids) == 21
                      and all(isinstance(qid, str) for qid in final_ids + planned_ids)
                      and len(set(final_ids)) == len(set(planned_ids)) == 21
                      and set(final_ids) == set(planned_ids))
    before = report.get("frozen_base_sha256_before")
    measured_hash = (isinstance(before, str) and len(before) == 64
                     and all(c in "0123456789abcdef" for c in before)
                     and report.get("frozen_base_final_hash_status") == "measured"
                     and before == report.get("frozen_base_sha256_after"))
    validity = report.get("validity")
    valid = (type(exit_code) is int and exit_code == 0
             and type(wall_seconds) in (int, float) and math.isfinite(wall_seconds)
             and type(cap_seconds) in (int, float) and math.isfinite(cap_seconds)
             and 0 < cap_seconds <= 300 and 0 <= wall_seconds <= cap_seconds
             and report.get("stage") == "a0_assisted_point"
             and report.get("status") == "complete" and report.get("run_completed") is True
             and isinstance(validity, dict) and validity.get("endpoint") is True
             and report.get("updates_completed") == 180
             and complete_final and measured_hash
             and report.get("frozen_base_gradients_absent") is True
             and claimed in ("full_fit", "beyond_additive", "not_beyond_additive"))
    return {"valid": valid, "category": claimed if valid else None,
            "child_claimed_category": claimed,
            "scope": "External process completion and total-cap gate, combined with the child's measured APS validity; no independent model reproduction."}


def main():
    started = PROGRAM_STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/grounding-assisted-point/report.json")
    args = parser.parse_args()
    require(1 <= args.seconds <= 300, "Total budget must be 1..300 seconds")
    output = args.output.resolve()
    relative(output)
    protected = ("grounding-microfit", "grounding-prompt-control", "grounding-m-key",
                 "assisted-point-engineering", "assisted-point-unit-check")
    require(output.is_relative_to(ROOT / "outputs")
            and not any(output.is_relative_to(ROOT / "outputs" / name) for name in protected),
            "Use a new fit output directory outside retained M/P/M-Key and APS engineering checks")
    start = output.parent / "process-start.json"
    completion = output.parent / "process-completion.json"
    stdout = output.parent / "process-stdout.log"
    stderr = output.parent / "process-stderr.log"
    runner = ROOT / "scripts/train_grounding_assisted_point.py"
    launcher = Path(__file__).resolve()
    launcher_copy = output.parent / "launcher-at-run.py"
    reserved = [output, start, completion, stdout, stderr, launcher_copy,
                output.parent / "run-start.json", output.parent / "runner-at-run.py",
                output.parent / "adapter.safetensors",
                output.parent / "adapter.partial.safetensors", output.with_suffix(".tmp")]
    require(output.suffix == ".json"
            and len({p.as_posix().casefold() for p in reserved}) == len(reserved),
            "Output names must not collide with retained process or child artifacts")
    require(not any(p.exists() for p in reserved), "Preserve existing outputs; no resume or retry")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_sha = file_sha256(runner)
    launcher_sha = file_sha256(launcher)
    with start.open("x", encoding="utf-8") as f:
        json.dump({"runner": relative(runner), "runner_sha256": source_sha,
                   "launcher": relative(launcher), "launcher_sha256": launcher_sha,
                   "launcher_at_run_copy": relative(launcher_copy),
                   "total_soft_cap_seconds": args.seconds, "native_return_grace_seconds": 30,
                   "automatic_retry": False}, f, indent=2, allow_nan=False)
    with launcher_copy.open("xb") as f:
        f.write(launcher.read_bytes())
    require(file_sha256(launcher_copy) == launcher_sha,
            "Launcher at-run copy differs from the pinned source")
    killed_after_grace = False
    launch_error, code, child_pid = None, None, None
    with stdout.open("x") as out, stderr.open("x") as err:
        try:
            child = subprocess.Popen([sys.executable, "-B", str(runner),
                                      "--seconds", str(args.seconds), "--output", str(output)],
                                     cwd=ROOT, stdout=out, stderr=err)
        except OSError as exc:
            launch_error = type(exc).__name__
        else:
            child_pid = child.pid
            try:
                code = child.wait(timeout=max(.001, args.seconds + 30
                                               - (time.perf_counter() - started)))
            except subprocess.TimeoutExpired:
                killed_after_grace = True
                child.terminate()
                try:
                    code = child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
    wall = time.perf_counter() - started
    report, report_error = {}, None
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
        require(isinstance(report, dict), "Report must be an object")
    except (OSError, ValueError) as exc:
        report_error = type(exc).__name__
    result = {"exit_code": code,
              "terminated_by_signal": -code if type(code) is int and code < 0 else None,
              "child_pid": child_pid, "launch_error": launch_error,
              "external_wall_seconds": wall, "total_soft_cap_seconds": args.seconds,
              "budget_exceeded": wall > args.seconds,
              "terminated_after_native_return_grace": killed_after_grace,
              "report_parse_error": report_error, "runner_sha256": source_sha,
              "runner_source_unchanged": file_sha256(runner) == source_sha,
              "launcher_sha256": launcher_sha,
              "launcher_source_unchanged": file_sha256(launcher) == launcher_sha,
              "launcher_at_run_copy_sha256": file_sha256(launcher_copy),
              "report_sha256": file_sha256(output) if output.exists() else None,
              "automatic_retry": False,
              "effective_endpoint": effective_endpoint(report, exit_code=code,
                  wall_seconds=wall, cap_seconds=args.seconds)}
    if (launch_error or report_error or not result["runner_source_unchanged"]
            or not result["launcher_source_unchanged"]
            or result["launcher_at_run_copy_sha256"] != launcher_sha):
        result["effective_endpoint"]["valid"] = False
        result["effective_endpoint"]["category"] = None
    with completion.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if code == 0 and report_error is None and launch_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
