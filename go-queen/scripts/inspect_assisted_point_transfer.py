#!/usr/bin/env python3
"""Independently audit one saved original54 APS evaluation, without a model.

Uses only the standard library. Tensor files are hashed as opaque bytes, never
decoded. Existing audit artifacts are retained; this script does not retry.
"""
from __future__ import annotations

import time
STARTED = time.perf_counter()

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/assisted-point-transfer"
PINS = {
    "outputs/grounding-data/manifest.json": "2c3deae2c0bde1cdd52b105889f0558e295662eb0dc222ffdbf2acbe07bd6279",
    "outputs/grounding-assisted-point/report.json": "2eb4bab5e4e86ff33e586bae94ee010a4cb005a84c0d76ada40903e94d9e75a4",
    "outputs/grounding-assisted-point/adapter.safetensors": "1532a7420e946c53983a6a6b3ce95fe9bf7dd7dd35e5ef1d1e4a02bc97bbad79",
    "outputs/grounding-baseline/original-val-calibrated-report.json": "ace7d6db92f25cefaf5d8b63b1507827b7e3701002d088d013ec1e4e2e398c72",
}
METADATA = ("question_id", "position_id", "game_id", "split", "ply", "to_play",
            "point_index", "point", "label", "source_task_ids")
LABEL_TOKENS = [3194, 11453, 5782]
TOLERANCE = 1e-5


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def numeric_equal(actual, expected):
    if expected is None:
        return actual is None
    return finite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10)


def independently_summarize(rows, questions, check):
    """Recompute from raw scores; never import the production metric helper."""
    check(len(rows) <= len(questions), "score rows do not exceed selection")
    check([r["question_id"] for r in rows] == [q["question_id"] for q in questions[:len(rows)]],
          "score rows are the exact ordered selection prefix")
    check(len({r["question_id"] for r in rows}) == len(rows), "score IDs are unique")
    confusion = [[0] * 3 for _ in range(3)]
    ce3, ce_full, ties, unique_correct, full_correct, off_choices = [], [], 0, 0, 0, 0
    for row, question in zip(rows, questions):
        gold, values = row["label"], row["label_log_probs"]
        token = row["unconstrained_token_id"]
        check(all(key in row and row[key] == question[key] for key in METADATA)
              and type(gold) is int and gold in (0, 1, 2)
              and isinstance(values, list) and len(values) == 3 and all(finite(x) for x in values)
              and row["label_token_ids"] == LABEL_TOKENS and type(row["prefix_tokens"]) is int
              and row["prefix_tokens"] > 0 and type(token) is int and token >= 0
              and finite(row["off_label_mass"]) and 0 <= row["off_label_mass"] <= 1,
              "row identity, factual metadata, encoding and finite scores")
        maximum = max(values)
        predicted = values.index(maximum)
        full_label = LABEL_TOKENS.index(token) if token in LABEL_TOKENS else None
        check(type(row["forced_choice_label"]) is int and row["forced_choice_label"] == predicted
              and row["unconstrained_label"] == full_label
              and row["forced_choice_correct"] is (predicted == gold)
              and row["unconstrained_label_correct"] is (full_label == gold)
              and numeric_equal(row["gold_value_nll"], -values[gold]), "first argmax, token mapping and redundant scores")
        tied = values.count(maximum) > 1
        confusion[gold][predicted] += 1
        ties += tied
        unique_correct += predicted == gold and not tied
        full_correct += full_label == gold
        off_choices += full_label is None
        ce3.append(maximum + math.log(sum(math.exp(x - maximum) for x in values)) - values[gold])
        ce_full.append(-values[gold])
    n, requested = len(rows), len(questions)
    correct = sum(confusion[i][i] for i in range(3))
    class_counts = [sum(row) for row in confusion]
    recalls = [confusion[i][i] / class_counts[i] if class_counts[i] else None for i in range(3)]
    def groups(key):
        answer = {}
        for value in sorted({q[key] for q in questions}):
            selected = [r for r in rows if r[key] == value]
            hits = sum(r["label_log_probs"].index(max(r["label_log_probs"])) == r["label"] for r in selected)
            answer[value] = {"correct": hits, "completed": len(selected),
                             "requested": sum(q[key] == value for q in questions),
                             "accuracy": hits / len(selected) if selected else None}
        return answer
    occupied_n = sum(class_counts[1:])
    occupied_correct = sum(confusion[g][p] for g in range(3) for p in range(3) if (g > 0) == (p > 0))
    color_correct = confusion[1][1] + confusion[2][2]
    return {"coverage_complete": n == requested, "completed_questions": n, "requested_questions": requested,
            "forced_choice_correct": correct, "forced_choice_accuracy": correct / n if n else None,
            "macro_recall": sum(recalls) / 3 if all(x is not None for x in recalls) else None,
            "macro_recall_convention": "null if any factual class is uncovered",
            "confusion_matrix": confusion, "exact_top_tie_count": ties, "unique_max_correct": unique_correct,
            "three_label_ce": math.fsum(ce3) / n if n else None,
            "full_vocabulary_ce": math.fsum(ce_full) / n if n else None,
            "by_game": groups("game_id"), "by_side": groups("to_play"),
            "occupied_vs_empty": {"correct": occupied_correct, "completed": n, "requested": requested,
                                   "accuracy": occupied_correct / n if n else None},
            "true_occupied_color": {"correct": color_correct, "completed": occupied_n,
                                    "requested": sum(q["label"] > 0 for q in questions),
                                    "accuracy": color_correct / occupied_n if occupied_n else None},
            "full_vocabulary_correct": full_correct, "off_label_argmax_count": off_choices}


def compare_summary(actual, expected, check, context):
    """Compare the integrated helper's concrete schema with independent counts."""
    counts = ("coverage_complete", "completed_questions", "requested_questions", "forced_choice_correct",
              "confusion_matrix", "exact_top_tie_count", "unique_max_correct")
    rates = ("forced_choice_accuracy", "macro_recall", "three_label_ce", "full_vocabulary_ce")
    check(all(actual[key] == expected[key] for key in counts)
          and all(numeric_equal(actual[key], expected[key]) for key in rates), context + ": aggregate arithmetic")
    for key in ("by_game", "by_side"):
        check(actual[key] == expected[key], context + ": " + key + " counts, coverage and rates")
    for saved_key, computed_key in (("occupancy", "occupied_vs_empty"),
                                    ("color_on_true_occupied", "true_occupied_color")):
        saved, computed = actual[saved_key], expected[computed_key]
        check(all(saved[key] == computed[key] for key in ("correct", "completed", "requested"))
              and numeric_equal(saved["rate"], computed["accuracy"]), context + ": " + saved_key)
    vocab, n = actual["full_vocabulary_argmax"], expected["completed_questions"]
    check(vocab["correct"] == expected["full_vocabulary_correct"]
          and vocab["off_label_choices"] == expected["off_label_argmax_count"] and vocab["completed"] == n
          and numeric_equal(vocab["rate"], expected["full_vocabulary_correct"] / n if n else None),
          context + ": stored full-vocabulary choices and rate")


def main():
    audit_path, copy_path = OUTPUT / "saved-audit.json", OUTPUT / "auditor-at-run.py"
    if audit_path.exists() or copy_path.exists():
        raise FileExistsError("Retain the existing saved audit; no overwrite or retry")
    audit = {"status": "running", "checks": [], "model_loaded": False, "tensor_payload_decoded": False,
             "limitations": [
                 "No forward, backward, optimizer, base-model reload or tensor decoding occurred in this audit.",
                 "Runtime base/adapter equality, frozen state, routing and zero-optimizer declarations are checked as saved records, not independently rerun measurements.",
                 "The stored full-vocabulary argmax cannot be reconstructed from three label scores; token mapping and saved reference equality are checked.",
                 "Feature/checkpoint files are verified only by byte hashes; labels are checked against the pinned manifest, not independently replayed SGF.",
                 "Only three validation games; no independent-sample significance or causal/generalization claim."]}
    with audit_path.open("x") as stream:
        json.dump(audit, stream, indent=2)
    def check(condition, description):
        audit["checks"].append({"check": description, "passed": bool(condition)})
        if not condition:
            raise ValueError(description)
    try:
        with copy_path.open("xb") as stream:
            stream.write(Path(__file__).read_bytes())
        audit["auditor_sha256"] = file_hash(copy_path)
        for name, digest in PINS.items():
            check(file_hash(ROOT / name) == digest, "pinned input bytes: " + name)
        manifest = read_json(ROOT / "outputs/grounding-data/manifest.json")
        fit = read_json(ROOT / "outputs/grounding-assisted-point/report.json")
        baseline = read_json(ROOT / "outputs/grounding-baseline/original-val-calibrated-report.json")
        report = read_json(OUTPUT / "report.json")
        external = read_json(OUTPUT / "process-completion.json")
        process_start = read_json(OUTPUT / "process-start.json")
        run_start = read_json(OUTPUT / "run-start.json")
        report_sha = file_hash(OUTPUT / "report.json")
        audit["inputs"] = {"report_sha256": report_sha,
                           "completion_sha256": file_hash(OUTPUT / "process-completion.json"),
                           "process_start_sha256": file_hash(OUTPUT / "process-start.json"),
                           "pinned_inputs": dict(PINS)}
        check(external["report_sha256"] == report_sha, "external completion pins exact report bytes")
        if "input_sha256" in report:
            check(report["input_sha256"] == PINS, "report input identities")
        train = fit["fixed_questions"]
        questions = [q for q in manifest["original_occupancy_questions"] if q["split"] == "val"]
        check(len(train) == 21 and len(questions) == 54, "fixed 21 restoration and original54 sizes")
        if "restoration_questions" in report:
            check(report["restoration_questions"] == train and report["validation_questions"] == questions,
                  "exact pinned selections and order")
            check(report["validation_questions_sha256"] == content_hash(questions), "selection content digest")
        check(content_hash(train) == fit["source"]["fixed_questions_sha256"], "saved training selection digest")
        check(Counter(q["label"] for q in questions) == {0: 18, 1: 18, 2: 18}, "balanced 18/18/18 facts")
        for key in ("game_id", "position_id", "question_id", "prompt"):
            check(not ({q[key] for q in train} & {q[key] for q in questions}), "selection separation: " + key)
        final = [e for e in fit["evaluations"] if e["name"] == "final"]
        check(len(final) == 1 and final[0]["updates_completed"] == 180 and final[0]["coverage_complete"],
              "unique pinned final 180-update reference")
        saved = final[0]["rows"]
        audit["restoration_metrics"] = independently_summarize(report["restoration_rows"], train, check)
        audit["validation_metrics"] = independently_summarize(report["validation_rows"], questions, check)
        restored = report["restoration_rows"]
        check(len(saved) == 21 and len(restored) <= 21, "restoration coverage against full21 reference")
        differences = []
        for new, old in zip(restored, saved):
            check(all(new[key] == old[key] for key in METADATA + ("label_token_ids", "prefix_tokens")),
                  "restoration metadata and input encoding")
            check(len(old["label_log_probs"]) == 3 and all(finite(x) for x in old["label_log_probs"]),
                  "finite old restoration scores")
            delta = max(abs(a - b) for a, b in zip(new["label_log_probs"], old["label_log_probs"]))
            choices = (new["forced_choice_label"] == old["forced_choice_label"]
                       and new["unconstrained_token_id"] == old["unconstrained_token_id"])
            differences.append({"question_id": new["question_id"], "max_abs_logp_difference": delta,
                                "predictions_match": choices})
        restoration_passed = len(restored) == 21 and all(x["max_abs_logp_difference"] <= TOLERANCE
                                                        and x["predictions_match"] for x in differences)
        audit["restoration_comparison"] = {"passed": restoration_passed, "coverage_complete": len(restored) == 21,
                                           "max_abs_logp_difference": max((x["max_abs_logp_difference"] for x in differences), default=None),
                                           "predictions_match": all(x["predictions_match"] for x in differences),
                                           "tolerance": TOLERANCE, "differences": differences}
        restoration = report.get("restoration")
        if restoration is not None:
            expected = audit["restoration_comparison"]
            check(all(restoration[key] == expected[key] for key in ("passed", "coverage_complete", "predictions_match", "tolerance"))
                  and numeric_equal(restoration["max_abs_logp_difference"], expected["max_abs_logp_difference"]),
                  "reported restoration agrees with independent arithmetic")
        check(not report["validation_rows"] or restoration_passed, "validation only after passed restoration")
        metrics, summary = audit["validation_metrics"], report.get("summary")
        if summary is not None:
            compare_summary(summary, metrics, check, "production validation summary")
        old_by_id = {r["question_id"]: r for r in baseline["rows"]}
        check(len(old_by_id) == len(baseline["rows"]) == 54, "54 unique old natural reference rows")
        old_rows = [old_by_id[q["question_id"]] for q in questions]
        audit["historical_natural_metrics"] = independently_summarize(old_rows, questions, check)
        check(audit["historical_natural_metrics"]["forced_choice_correct"] == 20, "old natural20 reference")
        if "references" in report:
            compare_summary(report["references"]["old_natural"], audit["historical_natural_metrics"],
                            check, "production historical natural summary")
        for new, old in zip(report["validation_rows"], old_rows):
            check(new["prefix_tokens"] == old["prefix_tokens"]
                  and new["label_token_ids"] == old["label_token_ids"], "canonical input encoding matches old reference")
        for name, count in (("train_majority", 18), ("coordinate_side_empirical_prior", 19)):
            prior = baseline["train_only_baselines"][name]["summary"]
            if "references" in report:
                check(report["references"]["categorical_priors"][name] == prior, "saved categorical reference: " + name)
            check(prior["all"]["forced_choice_correct"] == count, "categorical reference correct count")
        audit["count_differences_from_historical_references"] = ({
            str(reference): metrics["forced_choice_correct"] - reference for reference in (18, 19, 20)}
            if metrics["coverage_complete"] else None)
        positions = {p["position_id"]: p for p in manifest["positions"]}
        expected_positions = {q["position_id"] for q in train + questions}
        feature_sources = report.get("feature_sources", {})
        check(not (restored or report["validation_rows"]) or set(feature_sources) == expected_positions,
              "scores have all feature provenance records")
        for pid, source in sorted(feature_sources.items()):
            position = positions[pid]
            check(source == {"file": position["source_feature_file"], "sha256": position["feature_sha256"]},
                  "pinned feature metadata: " + pid)
            check(file_hash(ROOT / source["file"]) == source["sha256"], "feature bytes: " + pid)
        for question in train + questions:
            position = positions[question["position_id"]]
            check(question["label"] == position["absolute_board_labels"][question["point_index"]],
                  "factual label matches pinned board")
        implementation = report.get("implementation_sha256", {})
        required = {"scripts/eval_assisted_point_transfer.py", "src/goqueen/assisted_point.py",
                    "src/goqueen/assisted_point_eval.py", "src/goqueen/grounding_eval.py",
                    "src/goqueen/checkpoints.py", "src/goqueen/transfer_metrics.py"}
        check(not (restored or report["validation_rows"]) or required <= set(implementation),
              "scores have scoring/routing/loading/metric source records")
        for name, digest in implementation.items():
            check(file_hash(ROOT / name) == digest, "current implementation bytes: " + name)
            check(file_hash(OUTPUT / "source-at-run" / name) == digest, "saved implementation bytes: " + name)
        for kind, script in (("runner", "eval_assisted_point_transfer.py"), ("launcher", "run_assisted_point_transfer.py")):
            digest = file_hash(OUTPUT / (kind + "-at-run.py"))
            check(digest == process_start[kind + "_sha256"] == external[kind + "_sha256"]
                  == file_hash(ROOT / "scripts" / script), "launcher/runner source chain: " + kind)
        check(run_start["runner_sha256"] == process_start["runner_sha256"], "child source matches launch")
        if "checkpoint_metadata" in report:
            check(report["checkpoint_metadata"] == fit["checkpoint_metadata"]
                  and report["checkpoint_metadata"]["engineering_only"] == "false"
                  and report["checkpoint_metadata"]["updates"] == "180", "fitted checkpoint metadata record")
        recorded_equalities = {}
        for key in ("base", "adapter"):
            before, after = report.get(key + "_sha256_before"), report.get(key + "_sha256_after")
            recorded_equalities[key] = (isinstance(before, str) and len(before) == 64
                                         and all(c in "0123456789abcdef" for c in before) and before == after)
        if "base_sha256_before" in report:
            check(report["base_sha256_before"] == fit["frozen_base_sha256_after"], "recorded base agrees with fit")
        runtime = report["runtime"]
        check(runtime["device"] == "cpu" and runtime["threads"] == 4 and runtime["dtype"] == "float32"
              and runtime["sdpa"] == "math" and runtime["batch_size"] == 1, "recorded frozen evaluation configuration")
        routing = report.get("routing_observation")
        if routing is not None:
            forwards = routing["forward_calls"]["evaluation"]
            covered = len(restored) + len(report["validation_rows"])
            check(type(forwards) is int and covered <= forwards <= min(75, covered + 1)
                  and routing["adapter_calls"] == 11 * forwards and routing["routed_samples"] == forwards
                  and routing["forward_calls"]["training"] == 0, "recorded routing counts match returned score coverage")
        check(report["optimizer_constructed"] is False and report["optimizer_updates"] == 0
              and report["backward_calls"] == 0, "recorded zero optimizer/backward work")
        if "coverage" in report:
            check(report["coverage"] == {"restoration": len(restored), "validation": len(report["validation_rows"]),
                                        "requested_restoration": 21, "requested_validation": 54}, "honest actual coverage")
        check(report["stage"] == external["stage"] == process_start["stage"] == run_start["stage"]
              == "aps_original54_transfer", "stage record agreement")
        cap = external["soft_total_seconds"]
        check(finite(cap) and 15 < cap <= 300 and cap == runtime["soft_total_seconds"] == process_start["seconds"],
              "matching bounded total-time contract")
        check(type(external["exit_code"]) is int, "saved real OS exit code")
        check(external["terminated_by_signal"] == (-external["exit_code"] if external["exit_code"] < 0 else None),
              "signal record matches OS exit")
        for key in ("seconds_to_child_exit", "seconds_before_completion_record"):
            check(finite(external[key]) and 0 <= external[key], "external time: " + key)
        check(external["seconds_to_child_exit"] <= external["seconds_before_completion_record"], "external chronology")
        requirements = {
            "restored21": restoration_passed, "all54": metrics["coverage_complete"],
            "summary_present": summary is not None, "runtime_hash_equality": all(recorded_equalities.values()),
            "runtime_assertions": all(report.get(key) is True for key in
                                       ("source_hashes_unchanged", "runtime_binding_cleared", "all_parameters_frozen_and_gradients_absent")),
            "routing75x11": routing == {"adapter_calls": 825, "routed_samples": 75,
                                        "forward_calls": {"training": 0, "evaluation": 75}},
            "child_complete": report["status"] == "complete" and report["valid_complete"] is True
                              and report["budget_exceeded"] is False,
            "external_success": external["exit_code"] == 0 and not external["terminated_after_grace"]
                                and external["launch_error"] is None and external["report_parse_error"] is None,
            "within_total_budget": external["seconds_before_completion_record"] <= cap
                                   and external["budget_exceeded"] is False,
            "source_chain": external["source_unchanged"] is True,
        }
        audit["complete_endpoint_requirements"] = requirements
        audit["evaluation_valid_complete"] = all(requirements.values())
        check(external["valid_complete"] is audit["evaluation_valid_complete"], "external completeness agrees with independent evidence checks")
        check(report["automatic_retry"] is False and external["automatic_retry"] is False
              and process_start["automatic_retry"] is False and run_start["automatic_retry"] is False,
              "no automatic retry declarations")
        check(file_hash(OUTPUT / "report.json") == report_sha, "report unchanged throughout audit")
        audit["status"] = "passed"
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = {"type": type(exc).__name__, "message": str(exc).replace(str(ROOT), ".")}
    finally:
        audit["checks_completed"] = len(audit["checks"])
        audit["seconds"] = time.perf_counter() - STARTED
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": audit["status"], "checks_completed": audit["checks_completed"],
                      "seconds": audit["seconds"]}), flush=True)
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
