#!/usr/bin/env python3
"""Audit saved warm-QK training-item scores after the external child has ended.

Standard-library JSON arithmetic and file-byte hashing only. No model, tokenizer,
tensor payload or NPZ array is loaded; no inference or training is repeated.
Run once, only after the corresponding external process-completion record exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/aps-warm-qk-train-eval"
STAGE, CAP, TOLERANCE = "aps_warm_qk_train_eval", 1500, 1e-5
LABELS = ("empty", "black", "white")
IDENTITY = ("question_id", "position_id", "game_id", "split", "ply", "to_play",
            "point_index", "point", "label", "source_task_ids")
EXPECTED_INPUTS = {
    "outputs/aps-warm-qk/process-completion.json": "2eeb001e3764e6ed83a7d3bc24c2fec22c604687ef59280abb7019ae37113861",
    "outputs/grounding-data/manifest.json": "2c3deae2c0bde1cdd52b105889f0558e295662eb0dc222ffdbf2acbe07bd6279",
    "outputs/aps-warm-qk/report.json": "68456efe52869550ffcf6542c3150c9c739365b43b14ad5ab6bd801fab1b46d2",
    "outputs/grounding-assisted-point/report.json": "2eb4bab5e4e86ff33e586bae94ee010a4cb005a84c0d76ada40903e94d9e75a4",
    "outputs/aps-warm-qk/adapter.safetensors": "0db384223a90fd05f216cf497355d5036ddd667426b9376f2087614b46760e30",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def real(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def same(actual, expected):
    """Exact counts/identities; 1e-12 arithmetic tolerance, unrelated to model restoration."""
    if type(expected) is float:
        return real(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            same(actual[k], value) for k, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            same(a, b) for a, b in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def arithmetic(rows, weights=None):
    """Independent raw-score arithmetic; vocabulary correctness uses the saved argmax token."""
    matrix = [[0] * 3 for _ in range(3)]
    ces, nlls, class_ce, class_nll = [], [], [[], [], []], [[], [], []]
    total = full_correct = off_label = ties = 0
    for row in rows:
        weight = 1 if weights is None else weights[row["question_id"]]
        gold, values = row["label"], row["label_log_probs"]
        predicted = max(range(3), key=lambda y: values[y])
        token = row["unconstrained_token_id"]
        tokens = row["label_token_ids"]
        mapped = tokens.index(token) if token in tokens else None
        top = max(values)
        ce = math.log(math.fsum(math.exp(v - top) for v in values)) + top - values[gold]
        nll = -values[gold]
        matrix[gold][predicted] += weight
        total += weight
        full_correct += weight * (mapped == gold)
        off_label += weight * (mapped is None)
        ties += weight * (values.count(top) > 1)
        ces.append(weight * ce)
        nlls.append(weight * nll)
        class_ce[gold].append(weight * ce)
        class_nll[gold].append(weight * nll)
    counts = [sum(row) for row in matrix]
    correct = sum(matrix[y][y] for y in range(3))
    recalls = [matrix[y][y] / counts[y] if counts[y] else None for y in range(3)]
    return {
        "completed": total, "confusion_matrix": matrix, "correct": correct,
        "accuracy": correct / total if total else None,
        "macro_recall": math.fsum(recalls) / 3 if all(r is not None for r in recalls) else None,
        "three_label_ce": math.fsum(ces) / total if total else None,
        "full_vocabulary_ce": math.fsum(nlls) / total if total else None,
        "full_vocabulary_correct": full_correct, "off_label_choices": off_label, "exact_top_ties": ties,
        "by_class": {name: {"completed": counts[y], "correct": matrix[y][y], "recall": recalls[y],
            "three_label_ce": math.fsum(class_ce[y]) / counts[y] if counts[y] else None,
            "full_vocabulary_ce": math.fsum(class_nll[y]) / counts[y] if counts[y] else None}
            for y, name in enumerate(LABELS)},
    }


def inspect(report, completion, start, check, audit):
    # These two modules were read as pure Python. Do not import an evaluation runner.
    sys.path.insert(0, str(ROOT / "src"))
    from goqueen.train_eval_selection import (build_eval_manifest, validate_eval_manifest,
        questions_from_manifest, restoration_subset, describe, exposure_weighted_summary)
    from goqueen.transfer_metrics import compare_restoration

    source_data = read_json(ROOT / "outputs/grounding-data/manifest.json")
    reference = read_json(ROOT / "outputs/aps-warm-qk/report.json")
    reference_completion = read_json(ROOT / "outputs/aps-warm-qk/process-completion.json")
    pins = report.get("input_sha256", {})
    for rel, expected in EXPECTED_INPUTS.items():
        check("fixed_input:" + rel, sha(ROOT / rel) == expected)
        if pins:
            check("recorded_input:" + rel, pins.get(rel) == expected)
    check("source_run_completed", reference["status"] == "complete" and reference["run_completed"] is True)
    check("source_run_external_outcome", reference_completion["exit_code"] == 0
          and reference_completion["valid_complete"] is True
          and reference_completion["report_sha256"] == EXPECTED_INPUTS["outputs/aps-warm-qk/report.json"])

    # Reconstruct only the required observed schedule, not the old training audit.
    steps = reference["step_attempts"]
    check("source_1000_returned_steps", len(steps) == 1000 and all(
        type(s["step"]) is int and s["step"] == i and s["optimizer_step_started"] is True
        and s["optimizer_step_returned"] is True and len(s["question_ids"]) == 4
        for i, s in enumerate(steps, 1)))
    slots = [qid for step in steps for qid in step["question_ids"]]
    exposure = Counter(slots)
    check("source_actual4000_order", len(slots) == 4000 and slots == reference["train_order"] == source_data["train_order"])
    check("source_unique2702_exposure", len(exposure) == 2702 and Counter(exposure.values()) == {1: 1404, 2: 1298})
    manifest = build_eval_manifest(slots, source_data["train_order"], source_data["training_pool_questions"],
        reference["validation_questions"], order_sha256=reference["train_order_sha256"], source=pins)
    validate_eval_manifest(manifest)
    questions = questions_from_manifest(manifest)
    pool = {q["question_id"]: q for q in source_data["training_pool_questions"]}
    first, last = {}, {}
    for slot, qid in enumerate(slots):
        first.setdefault(qid, slot // 4 + 1)
        last[qid] = slot // 4 + 1
    check("manifest_entries_from_actual_slots", all(
        e["question"] == pool[e["question"]["question_id"]]
        and e["exposure_count"] == exposure[e["question"]["question_id"]]
        and e["first_step"] == first[e["question"]["question_id"]]
        and e["last_step"] == last[e["question"]["question_id"]]
        for e in manifest["entries"]))
    check("sha_order_independent", [q["question_id"] for q in questions] == sorted(exposure,
          key=lambda qid: (hashlib.sha256(qid.encode()).hexdigest(), qid)))
    check("selected_class_counts", Counter(q["label"] for q in questions) == {0: 1334, 1: 690, 2: 678})
    check("selected_board_coverage", (len({q["game_id"] for q in questions}),
          len({q["position_id"] for q in questions}), len({q["point_index"] for q in questions})) == (5, 30, 360))
    final = [e for e in reference["evaluations"] if e["name"] == "final"]
    check("source_final54", len(final) == 1 and final[0]["updates_completed"] == 1000
          and final[0]["coverage_complete"] is True and len(final[0]["rows"]) == 54)
    saved54 = final[0]["rows"]
    restore_saved = restoration_subset(saved54)
    val_lookup = {q["question_id"]: q for q in reference["validation_questions"]}
    restore_questions = [val_lookup[row["question_id"]] for row in restore_saved]
    check("restore_six_both_sides", len(restore_questions) == 6
          and {q["to_play"] for q in restore_questions} == {"B", "W"})

    selected_saved = report.get("training_questions") is not None
    if selected_saved:
        check("reported_training_selection", report["training_questions"] == questions)
        check("reported_restoration_selection", report["restoration_questions"] == restore_questions)
        check("eval_manifest_rebuild", read_json(OUTPUT / "eval-manifest.json") == manifest)
        check("eval_manifest_content_pin", report["eval_manifest_content_sha256"] == manifest["content_sha256"])
        check("eval_manifest_file_pin", sha(OUTPUT / "eval-manifest.json") == report["eval_manifest_file_sha256"])
        check("saved54_summary", same(report["saved54"], describe(saved54, reference["validation_questions"])))
        train_prompts = {q["prompt"] for q in questions}
        comparison = reference["validation_questions"]
        overlap = {"comparison_rows": sum(q["prompt"] in train_prompts for q in comparison),
                   "distinct_strings": len({q["prompt"] for q in comparison}),
                   "distinct_prompt_intersection": len(train_prompts & {q["prompt"] for q in comparison})}
        check("prompt_overlap_recomputed", same(report["prompt_overlap"], overlap)
              and overlap == {"comparison_rows": 54, "distinct_strings": 52, "distinct_prompt_intersection": 52})
    else:
        check("early_failure_not_complete", report.get("valid_complete") is False)

    def rows_for(group, expected):
        rows = report.get(group + "_rows", [])
        check(group + "_ordered_prefix", len(rows) <= len(expected)
              and [r["question_id"] for r in rows] == [q["question_id"] for q in expected[:len(rows)]])
        for index, (row, question) in enumerate(zip(rows, expected)):
            values, tokens = row["label_log_probs"], row["label_token_ids"]
            check(f"{group}_row_{index}_metadata", all(row.get(k) == question.get(k) for k in IDENTITY)
                  and type(row["prefix_tokens"]) is int and row["prefix_tokens"] > 0)
            good = (isinstance(values, list) and len(values) == 3
                    and all(real(v) and v <= 0 for v in values)
                    and real(row["off_label_mass"]) and 0 <= row["off_label_mass"] <= 1
                    and tokens == [3194, 11453, 5782] and type(row["unconstrained_token_id"]) is int)
            check(f"{group}_row_{index}_finite_scores", good)
            if not good:
                raise ValueError("Malformed score; arithmetic cannot continue")
            forced = max(range(3), key=lambda y: values[y])
            mapped = tokens.index(row["unconstrained_token_id"]) if row["unconstrained_token_id"] in tokens else None
            check(f"{group}_row_{index}_choice_arithmetic",
                  type(row["forced_choice_label"]) is int and row["forced_choice_label"] == forced
                  and row["unconstrained_label"] == mapped
                  and row["forced_choice_correct"] is (forced == row["label"])
                  and row["unconstrained_label_correct"] is (mapped == row["label"])
                  and same(row["gold_value_nll"], -float(values[row["label"]])))
        jsonl = OUTPUT / ("training-rows.jsonl" if group == "training" else "restoration-rows.jsonl")
        check(group + "_jsonl_report_identity", ([json.loads(line) for line in jsonl.read_text().splitlines()]
              if jsonl.exists() else []) == rows)
        return rows

    rows, restored = rows_for("training", questions), rows_for("restoration", restore_questions)
    # Count completed composition from actual rows, independently of the runner's
    # selected[:completed] construction; rows_for already checked that prefix.
    composition = {field: {"requested": dict(Counter(str(q[field]) for q in questions)),
                           "completed": dict(Counter(str(row[field]) for row in rows))}
                   for field in ("label", "game_id", "position_id")}
    composition["exposure_count"] = {
        "requested": dict(Counter(str(exposure[q["question_id"]]) for q in questions)),
        "completed": dict(Counter(str(exposure[row["question_id"]]) for row in rows))}
    if report.get("prefix_composition") is not None:
        check("prefix_composition_recomputed", same(report["prefix_composition"], composition))
        check("prefix_composition_denominators", all(sum(part["requested"].values()) == 2702
              and sum(part["completed"].values()) == len(rows) for part in composition.values()))
    else:
        check("missing_composition_not_complete", report.get("valid_complete") is False)
    restoration = compare_restoration(restored, restore_saved, TOLERANCE)
    if report.get("restoration") is not None:
        check("restoration_comparison_recomputed", same(report["restoration"], restoration))
    check("no_scoring_after_failed_restoration", not rows or restoration["passed"] is True)
    if report.get("summary") is not None:
        check("whole_summary_helper_recomputed", same(report["summary"], describe(rows, questions)))
    if report.get("exposure_weighted") is not None:
        check("weighted_helper_recomputed", same(report["exposure_weighted"], exposure_weighted_summary(rows, manifest)))

    independent, weighted = arithmetic(rows), arithmetic(rows, exposure)
    if report.get("summary") is not None:
        summary = report["summary"]["summary"]
        mapping = {"completed_questions": "completed", "forced_choice_correct": "correct",
                   "forced_choice_accuracy": "accuracy", "confusion_matrix": "confusion_matrix",
                   "macro_recall": "macro_recall", "three_label_ce": "three_label_ce",
                   "full_vocabulary_ce": "full_vocabulary_ce", "exact_top_tie_count": "exact_top_ties"}
        for saved_key, independent_key in mapping.items():
            check("independent_unique:" + saved_key, same(summary[saved_key], independent[independent_key]))
        check("independent_full_vocab_correct", summary["full_vocabulary_argmax"]["correct"] == independent["full_vocabulary_correct"])
        check("independent_full_vocab_offlabel", summary["full_vocabulary_argmax"]["off_label_choices"] == independent["off_label_choices"])
        for label in LABELS:
            part, own = report["summary"]["per_class"][label], independent["by_class"][label]
            check("independent_class:" + label, same(summary["by_class"][label]["recall"], own["recall"])
                  and same(part["three_label_ce"], own["three_label_ce"])
                  and same(part["full_vocabulary_ce"], own["full_vocabulary_ce"]))
    if report.get("exposure_weighted") is not None:
        summary = report["exposure_weighted"]
        for saved_key, own_key in (("completed_slots", "completed"), ("forced_choice_accuracy", "accuracy"),
                ("confusion_matrix", "confusion_matrix"), ("macro_recall", "macro_recall"),
                ("three_label_ce", "three_label_ce"), ("full_vocabulary_ce", "full_vocabulary_ce")):
            check("independent_weighted:" + saved_key, same(summary[saved_key], weighted[own_key]))
        check("independent_weighted_requested", summary["requested_slots"] == 4000)
        for label in LABELS:
            part, own = summary["by_class"][label], weighted["by_class"][label]
            check("independent_weighted_class:" + label, part["completed_slots"] == own["completed"]
                  and part["correct_slots"] == own["correct"] and same(part["recall"], own["recall"]))

    # Hash current source bytes and the retained runtime copies; never decode weights/features.
    all_pins = {**pins, **report.get("feature_sha256", {}), **report.get("implementation_sha256", {}),
                **report.get("local_model_config_sha256", {})}
    for rel, expected in all_pins.items():
        path = ROOT / rel
        if Path(rel).is_absolute() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Source record is not a project-relative path")
        check("file_bytes:" + rel, digest(expected) and sha(path) == expected)
    for rel, expected in report.get("implementation_sha256", {}).items():
        check("runtime_copy:" + rel, sha(OUTPUT / "source-at-run" / rel) == expected)
    for key, name in (("runner_sha256", "runner-at-run.py"), ("launcher_sha256", "launcher-at-run.py")):
        check("process_source:" + name, start[key] == completion[key] == sha(OUTPUT / name))
        current = ROOT / start["runner" if key == "runner_sha256" else "launcher"]
        check("current_process_source:" + name, sha(current) == start[key])
    check("report_runner_pin", report["source_script_sha256"] == start["runner_sha256"])
    check("process_report_pin", sha(OUTPUT / "report.json") == completion["report_sha256"])

    # These are saved runtime claims, not independently observed model state.
    for prefix in ("base", "adapter"):
        before, after = report.get(prefix + "_sha256_before"), report.get(prefix + "_sha256_after")
        if before is not None:
            check(prefix + "_saved_before_digest", digest(before))
        if after is not None:
            check(prefix + "_saved_after_digest", digest(after))
            check(prefix + "_saved_equality_flag", report[prefix + ("_unchanged" if prefix == "base" else "s_unchanged")] == (before == after))
    if report.get("base_sha256_before") is not None:
        check("saved_base_matches_reference", report["base_sha256_before"] == reference["base_sha256_after"])
        check("saved_checkpoint_metadata", report["checkpoint_metadata"] == reference["checkpoint_metadata"])
        fit = read_json(ROOT / "outputs/grounding-assisted-point/report.json")
        config_path = (Path(fit["local_model"]["directory"]) / "config.json").as_posix()
        check("saved_base_config_pin", report["local_model_config_sha256"]
              == {config_path: fit["local_model"]["config_sha256"]})
    if "feature_memory_sha256_after" in report:
        before, after = report["feature_memory_sha256_before"], report["feature_memory_sha256_after"]
        check("saved_feature_digest_values", all(digest(v) for v in list(before.values()) + list(after.values())))
        check("saved_feature_equality_flag", report["features_unchanged"] == (before == after))

    attempts, returns = report["forward_attempts"], report["forward_returns"]
    check("actual_forward_bounds", type(attempts) is type(returns) is int and 0 <= returns <= attempts <= 2708
          and len(rows) + len(restored) <= returns)
    check("zero_update_record", report["optimizer_constructed"] is False
          and type(report["optimizer_updates"]) is int and report["optimizer_updates"] == 0
          and type(report["backward_calls"]) is int and report["backward_calls"] == 0)
    check("fixed_runtime_record", report["runtime"] == {"device": "cpu", "dtype": "float32", "threads": 4,
          "batch_size": 1, "sdpa": "math", "soft_total_seconds": CAP, "closing_reserve_seconds": 60})
    check("process_start_contract", start["stage"] == STAGE and start["seconds"] == CAP
          and start["reserve_seconds"] == 60 and start["native_return_grace_seconds"] == 30
          and start["automatic_retry"] is False and real(start["origin_monotonic"]))
    check("terminal_status", report["stage"] == completion["stage"] == STAGE
          and report["status"] in ("complete", "stopped", "error"))
    check("external_fixed_budget", completion["soft_total_seconds"] == CAP
          and completion["reserve_seconds"] == 60 and completion["automatic_retry"] is False)
    child_times = [report["seconds_before_report_write"]]
    if "seconds_after_initial_report_write" in report:
        child_times.append(report["seconds_after_initial_report_write"])
    check("child_saved_times", all(real(t) and t >= 0 for t in child_times))
    check("child_overrun_record", report["budget_exceeded"] == (max(child_times) > CAP))
    external_times = [completion["seconds_to_child_exit"], completion["seconds_before_completion_record"]]
    if "seconds_after_initial_completion_record" in completion:
        external_times.append(completion["seconds_after_initial_completion_record"])
    check("external_saved_times", all(real(t) and t >= 0 for t in external_times)
          and external_times == sorted(external_times))
    check("external_overrun_record", completion["budget_exceeded"] == (max(external_times) > CAP))
    if report["valid_complete"]:
        check("complete_claim_exact_coverage", len(rows) == 2702 and len(restored) == 6 and attempts == returns == 2708)
        check("complete_claim_restoration", restoration["passed"] is True)
        check("complete_claim_saved_integrity", all(report.get(k) is True for k in (
            "base_unchanged", "adapters_unchanged", "features_unchanged", "source_hashes_unchanged",
            "manifest_rebuilt_equal", "all_parameters_frozen_and_gradients_absent", "runtime_binding_cleared")))
        check("complete_claim_no_error", report["status"] == "complete" and report.get("error") is None
              and report.get("closing_error") is None and report["budget_exceeded"] is False)
    external_eligible = (report["valid_complete"] is True and completion["exit_code"] == 0
        and completion["terminated_by_signal"] is None and completion["launch_error"] is None
        and completion["report_parse_error"] is None and not completion["hash_errors"]
        and completion["terminated_after_grace"] is False and completion["source_unchanged"] is True
        and completion["report_source_matches"] is True and max(external_times) <= CAP)
    check("external_validity_agrees", completion["valid_complete"] is external_eligible)
    audit.update(run_status=report["status"], external_exit_code=completion["exit_code"],
        reported_valid_complete=report["valid_complete"], external_valid_complete=completion["valid_complete"],
        coverage={"training": len(rows), "requested_training": 2702, "restoration": len(restored),
                  "requested_restoration": 6, "forward_attempts": attempts, "forward_returns": returns},
        independent_unique=independent, independent_exposure_weighted=weighted,
        independent_prefix_composition=composition,
        restoration_recomputed=restoration, input_files_hashed=len(all_pins),
        saved_parameter_evidence="Compared saved before/after digests and flags only; no base or adapter tensor was independently reloaded.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-process-completion", action="store_true", required=True)
    parser.parse_args(argv)
    # An actual child terminal record must already exist, not just a report file.
    completion = read_json(OUTPUT / "process-completion.json")
    if (completion.get("stage") != STAGE or type(completion.get("exit_code")) is not int
            or type(completion.get("child_pid")) is not int):
        raise ValueError("A finalized external child outcome is required before inspection")
    target, source_copy = OUTPUT / "saved-audit.json", OUTPUT / "inspect-at-run.py"
    if target.exists() or source_copy.exists():
        raise FileExistsError("Existing inspection artifacts are retained; no audit retry or overwrite")
    with source_copy.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    started = time.perf_counter()
    audit = {"status": "running", "stage": STAGE, "script_sha256": sha(__file__),
        "run": "outputs/aps-warm-qk-train-eval", "model_loaded": False, "tokenizer_loaded": False,
        "tensor_payload_decoded": False, "tests_executed": False, "checks_completed": 0, "failures": [],
        "scope": "New saved score arithmetic, selection, records and file-byte hashes only; the historical training audit is not repeated.",
        "limitations": ["Runtime frozen/gradient/parameter-memory claims are checked as saved records, not independently remeasured.",
            "Full-vocabulary correctness uses the saved argmax token; three retained log probabilities cannot reconstruct all logits.",
            "Reported timings come from saved process records; no new inference timing or experiment is performed.",
            "A consistent partial prefix does not constitute a complete-list estimate or a new sample."]}

    def check(name, condition):
        audit["checks_completed"] += 1
        if not condition:
            audit["failures"].append(name)

    with target.open("x") as stream:
        try:
            report, start = read_json(OUTPUT / "report.json"), read_json(OUTPUT / "process-start.json")
            audit["report_sha256"] = sha(OUTPUT / "report.json")
            audit["process_completion_sha256"] = sha(OUTPUT / "process-completion.json")
            inspect(report, completion, start, check, audit)
        except Exception as exc:
            audit["failures"].append(type(exc).__name__ + ": " + str(exc).replace(str(ROOT), "."))
        audit["status"] = "consistent_saved_records" if not audit["failures"] else "inconsistent_saved_records"
        audit["complete_endpoint"] = (not audit["failures"] and audit.get("reported_valid_complete") is True
                                      and audit.get("external_valid_complete") is True)
        audit["seconds"] = time.perf_counter() - started
        stream.write(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": audit["status"], "checks_completed": audit["checks_completed"],
                      "complete_endpoint": audit["complete_endpoint"], "seconds": audit["seconds"]}), flush=True)
    return 0 if not audit["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
