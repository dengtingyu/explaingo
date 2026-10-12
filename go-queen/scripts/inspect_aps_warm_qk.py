#!/usr/bin/env python3
"""Independently audit saved APS-warm-Q/K records after external completion.

Standard library only: no model imports, forwards, training or tensor decoding.
The explicit --after-process-completion flag is a caller acknowledgement, not
evidence; the matching process-completion.json is also required. Every audit
uses exclusive output files and preserves the exact audit source.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
import time

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("empty", "black", "white")
LAYERS = (0, 3, 5, 8, 11, 14, 16, 19, 22, 24, 27)
SUFFIXES = ("query_norm.weight", "query_norm.bias", "query.weight", "key.weight")
EXPECTED_QK = [f"language_model.model.layers.{i}.adapter.{s}" for i in LAYERS for s in SUFFIXES]
PINNED = {
    "outputs/grounding-data/manifest.json": "2c3deae2c0bde1cdd52b105889f0558e295662eb0dc222ffdbf2acbe07bd6279",
    "outputs/grounding-assisted-point/report.json": "2eb4bab5e4e86ff33e586bae94ee010a4cb005a84c0d76ada40903e94d9e75a4",
    "outputs/grounding-assisted-point/adapter.safetensors": "1532a7420e946c53983a6a6b3ce95fe9bf7dd7dd35e5ef1d1e4a02bc97bbad79",
    "scripts/train_aps_warm_qk.py": "72ba36f568daa4cc2527dfc8195c5211a23830ef0393dacb9d6ee4684ca7ab20",
    "scripts/run_aps_warm_qk.py": "f0d9520f1e50c7c2851aa50ff784a214506141d476fa055890ac96dc000c6ea7",
}
METADATA = ("question_id", "position_id", "game_id", "split", "ply", "to_play", "point_index", "point", "label", "source_task_ids")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(), parse_constant=lambda s: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + s)))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def relpath(value):
    path = ROOT / value
    if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("Expected project-relative artifact path")
    return path


class Audit:
    def __init__(self):
        self.count = 0
        self.failures = []

    def check(self, name, passed, detail=None):
        self.count += 1
        if not passed:
            self.failures.append({"check": name, "detail": detail})
        return bool(passed)

    def equal(self, name, actual, expected):
        if isinstance(expected, dict):
            if not self.check(name + ":mapping", isinstance(actual, dict)):
                return
            for key, value in expected.items():
                self.equal(name + "." + key, actual.get(key), value)
        elif isinstance(expected, list):
            if not self.check(name + ":list-length", isinstance(actual, list) and len(actual) == len(expected)):
                return
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.equal(f"{name}[{i}]", a, b)
        elif type(expected) is float:
            self.check(name, finite(actual) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12),
                       {"saved": actual, "recomputed": expected})
        else:
            self.check(name, type(actual) is type(expected) and actual == expected,
                       {"saved": actual, "recomputed": expected})


def metrics(rows, questions, audit, prefix):
    """Arithmetic from raw label log probabilities; never call the original metrics."""
    if not audit.check(prefix + ":ordered-prefix", len(rows) <= len(questions)
                       and [r.get("question_id") for r in rows] == [q["question_id"] for q in questions[:len(rows)]]):
        return None
    confusion, nll3, nllfull = [[0] * 3 for _ in range(3)], [], []
    full_correct = off_choices = ties = unique_correct = 0
    for i, (row, question) in enumerate(zip(rows, questions)):
        name = f"{prefix}.row{i}"
        for key in METADATA:
            if key in question:
                audit.equal(name + "." + key, row.get(key), question[key])
        scores, tokens, gold = row.get("label_log_probs"), row.get("label_token_ids"), question["label"]
        if not audit.check(name + ":valid-label-scores", isinstance(scores, list) and len(scores) == 3
                           and all(finite(x) and x <= 1e-6 for x in scores)):
            return None
        if not audit.check(name + ":tokens", isinstance(tokens, list) and len(tokens) == 3
                           and all(type(t) is int and t >= 0 for t in tokens) and len(set(tokens)) == 3):
            return None
        maximum = max(scores)
        pred = scores.index(maximum)
        tied = scores.count(maximum) > 1
        total_logp = maximum + math.log(math.fsum(math.exp(x - maximum) for x in scores))
        nll3.append(total_logp - scores[gold])
        nllfull.append(-scores[gold])
        confusion[gold][pred] += 1
        ties += tied
        unique_correct += pred == gold and not tied
        token = row.get("unconstrained_token_id")
        audit.check(name + ":unconstrained-token", type(token) is int and token >= 0)
        mapped = tokens.index(token) if token in tokens else None
        full_correct += mapped == gold
        off_choices += mapped is None
        audit.equal(name + ".forced_choice_label", row.get("forced_choice_label"), pred)
        audit.equal(name + ".forced_choice_correct", row.get("forced_choice_correct"), pred == gold)
        audit.equal(name + ".unconstrained_label", row.get("unconstrained_label"), mapped)
        audit.equal(name + ".unconstrained_label_correct", row.get("unconstrained_label_correct"), mapped == gold)
        audit.equal(name + ".gold_value_nll", row.get("gold_value_nll"), -scores[gold])
        off_mass = max(0.0, min(1.0, 1.0 - math.fsum(math.exp(x) for x in scores)))
        audit.check(name + ":offmass-rounding", finite(row.get("off_label_mass"))
                    and abs(row["off_label_mass"] - off_mass) <= 2e-6)
    count = len(rows)
    correct = sum(confusion[c][c] for c in range(3))
    by_class = {}
    for c, label in enumerate(LABELS):
        completed = sum(confusion[c])
        by_class[label] = {"requested": sum(q["label"] == c for q in questions), "completed": completed,
                           "correct": confusion[c][c], "recall": confusion[c][c] / completed if completed else None}
    grouped = {}
    for field, output in (("game_id", "by_game"), ("to_play", "by_side")):
        grouped[output] = {}
        for key in sorted({q[field] for q in questions}):
            selected = [r for r in rows if r[field] == key]
            good = sum(r["label_log_probs"].index(max(r["label_log_probs"])) == r["label"] for r in selected)
            grouped[output][key] = {"requested": sum(q[field] == key for q in questions),
                "completed": len(selected), "correct": good, "accuracy": good / len(selected) if selected else None}
    occupied_n = sum(sum(confusion[c]) for c in (1, 2))
    occupancy_correct = confusion[0][0] + sum(confusion[g][p] for g in (1, 2) for p in (1, 2))
    return {"coverage_complete": count == len(questions), "completed_questions": count,
        "requested_questions": len(questions), "forced_choice_correct": correct,
        "forced_choice_accuracy": correct / count if count else None,
        "macro_recall": sum(v["recall"] for v in by_class.values()) / 3 if all(v["recall"] is not None for v in by_class.values()) else None,
        "by_class": by_class, "confusion_matrix": confusion, "exact_top_tie_count": ties,
        "unique_max_correct": unique_correct, "three_label_ce": math.fsum(nll3) / count if count else None,
        "full_vocabulary_ce": math.fsum(nllfull) / count if count else None,
        "occupancy": {"correct": occupancy_correct, "completed": count, "requested": len(questions),
                      "rate": occupancy_correct / count if count else None},
        "color_on_true_occupied": {"correct": confusion[1][1] + confusion[2][2], "completed": occupied_n,
            "requested": sum(q["label"] != 0 for q in questions),
            "rate": (confusion[1][1] + confusion[2][2]) / occupied_n if occupied_n else None},
        "full_vocabulary_argmax": {"correct": full_correct, "off_label_choices": off_choices,
                                   "completed": count, "rate": full_correct / count if count else None}, **grouped}


def audit_run(run, audit):
    report, start, completion = [read_json(run / n) for n in ("report.json", "process-start.json", "process-completion.json")]
    if report.get("status") == "running" or type(completion.get("exit_code")) is not int:
        raise ValueError("No reliable terminal saved outcome; do not audit an active run")
    inputs = {n: digest(run / n) for n in ("report.json", "process-start.json", "process-completion.json", "run-start.json")}
    audit.equal("stages", [report.get("stage"), start.get("stage"), completion.get("stage")], ["aps_warm_qk"] * 3)
    audit.equal("completion.report-pin", completion.get("report_sha256"), inputs["report.json"])
    for name, expected in PINNED.items():
        audit.equal("pinned." + name, digest(ROOT / name), expected)
    manifest = read_json(ROOT / "outputs/grounding-data/manifest.json")
    fit = read_json(ROOT / "outputs/grounding-assisted-point/report.json")
    validation = [q for q in manifest["original_occupancy_questions"] if q["split"] == "val"]
    restoration = fit["fixed_questions"]
    if "validation_questions" in report:
        audit.equal("validation-list", report["validation_questions"], validation)
    if "restoration_questions" in report:
        audit.equal("restoration-list", report["restoration_questions"], restoration)
    order = manifest["train_order"]
    if "train_order" in report:
        audit.equal("planned-order", report["train_order"], order)
        audit.equal("order-hash", report.get("train_order_sha256"), content_digest(order))
    pool = {q["question_id"]: q for q in manifest["training_pool_questions"]}
    audit.check("fixed-schedule-counts", len(order) == 4000 and len(set(order)) == 2702
                and Counter(pool[q]["label"] for q in order) == {0: 1334, 1: 1333, 2: 1333})
    restore_rows = report.get("restoration_rows", [])
    restored_metrics = metrics(restore_rows, restoration, audit, "restore")
    saved_rows = next(e["rows"] for e in fit["evaluations"] if e["name"] == "final")
    max_delta, choices_match, restore_metadata = 0.0, True, True
    for i, (row, old) in enumerate(zip(restore_rows, saved_rows)):
        for key in METADATA + ("prefix_tokens", "label_token_ids"):
            if row.get(key) != old.get(key):
                restore_metadata = False
            audit.equal(f"restore-reference.{i}.{key}", row.get(key), old.get(key))
        if len(row.get("label_log_probs", [])) == len(old["label_log_probs"]) == 3:
            max_delta = max(max_delta, *(abs(a - b) for a, b in zip(row["label_log_probs"], old["label_log_probs"])))
        for key in ("forced_choice_label", "unconstrained_token_id"):
            choices_match = choices_match and row.get(key) == old[key]
    restored = (len(restore_rows) == len(saved_rows) == 21 and restore_metadata
                and choices_match and max_delta <= 1e-5 and restored_metrics is not None)
    if report.get("restoration") is not None:
        audit.equal("restore-passed", report["restoration"].get("passed"), restored)
        audit.equal("restore-maxdelta", report["restoration"].get("max_abs_logp_difference"), max_delta)
        audit.equal("restore-choice", report["restoration"].get("predictions_match"), choices_match)
    evaluations = {}
    for evaluation in report.get("evaluations", []):
        name = evaluation.get("name")
        audit.check("unique-known-evaluation", name in ("initial", "final") and name not in evaluations)
        computed = metrics(evaluation.get("rows", []), validation, audit, str(name))
        evaluations[name] = computed
        if computed is not None:
            # In-flight rows can remain without final summary after an interrupted evaluation.
            if "completed_questions" in evaluation:
                audit.equal("summary." + str(name), evaluation, computed)
            else:
                audit.check("unfinished-evaluation-not-complete." + str(name), evaluation.get("coverage_complete") is False)
        audit.equal("evaluation-update." + str(name), evaluation.get("updates_completed"),
                    0 if name == "initial" else report.get("updates_completed"))
    names = report.get("trainable_names", [])
    if names:
        audit.equal("44-names", names, EXPECTED_QK)
    attempts = report.get("step_attempts", [])
    returned = 0
    for i, attempt in enumerate(attempts):
        audit.equal(f"step.{i}.ordinal", attempt.get("step"), i + 1)
        audit.equal(f"step.{i}.schedule", attempt.get("question_ids"), order[4 * i:4 * i + 4])
        done = attempt.get("optimizer_step_returned") is True
        if done:
            returned += 1
            audit.check(f"step.{i}.started", attempt.get("optimizer_step_started") is True)
            norms = attempt.get("gradient_norms", {})
            audit.check(f"step.{i}.44-finite-gradients", set(norms) == set(EXPECTED_QK)
                        and all(finite(v) and v >= 0 for v in norms.values()))
            audit.check(f"step.{i}.finite-loss", finite(attempt.get("full_vocabulary_ce"))
                        and attempt["full_vocabulary_ce"] >= 0)
            if "seconds" in attempt:
                audit.check(f"step.{i}.finite-time", finite(attempt["seconds"]) and attempt["seconds"] >= 0)
            else:
                # An exception after optimizer.step returns can precede the timing assignment.
                audit.check(f"step.{i}.unfinished-post-step", i == len(attempts) - 1 and report.get("status") != "complete")
        else:
            audit.check(f"step.{i}.incomplete-only-last", i == len(attempts) - 1)
    audit.check("at-most-1000-attempts", len(attempts) <= 1000)
    audit.equal("returned-updates", report.get("updates_completed"), returned)
    audit.equal("returned-slots", report.get("sample_slots"), returned * 4)
    if report.get("optimizer_constructed"):
        audit.equal("optimizer-settings", report.get("optimizer_contract"),
                    {"lr": .001, "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0,
                     "amsgrad": False, "maximize": False, "foreach": None,
                     "capturable": False, "differentiable": False, "fused": None})
    sources_available = bool(report.get("implementation_sha256"))
    for path, pinned in report.get("implementation_sha256", {}).items():
        copied = run / "source-at-run" / path
        same = copied.is_file() and digest(copied) == pinned and digest(relpath(path)) == pinned
        audit.check("source-copy." + path, same)
        sources_available = sources_available and same
    for path, pinned in {**report.get("input_sha256", {}), **report.get("feature_sha256", {})}.items():
        audit.equal("input-bytes." + path, digest(relpath(path)), pinned)
    for filename, source in (("runner-at-run.py", "scripts/train_aps_warm_qk.py"), ("launcher-at-run.py", "scripts/run_aps_warm_qk.py")):
        audit.equal(filename, digest(run / filename), PINNED[source])
    for key, source in (("runner_sha256", "scripts/train_aps_warm_qk.py"), ("launcher_sha256", "scripts/run_aps_warm_qk.py")):
        audit.equal("start." + key, start.get(key), PINNED[source])
        audit.equal("completion." + key, completion.get(key), PINNED[source])
    audit.equal("completion.launcher-copy-pin", completion.get("launcher_at_run_copy_sha256"), PINNED["scripts/run_aps_warm_qk.py"])
    audit.equal("source-script", report.get("source_script_sha256"), PINNED["scripts/train_aps_warm_qk.py"])
    checkpoint = {"present": (run / "adapter.safetensors").is_file(), "tensor_payload_decoded": False}
    checkpoint_ok = False
    if checkpoint["present"]:
        path = run / "adapter.safetensors"
        with path.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < size < min(path.stat().st_size, 10_000_000):
                raise ValueError("Invalid safetensors header length")
            header = json.loads(stream.read(size))
        metadata = header.pop("__metadata__", {})
        count, values = len(header), sum(math.prod(t["shape"]) for t in header.values())
        checkpoint.update(sha256=digest(path), declared_tensor_count=count, declared_values=values,
                          declared_metadata=metadata, file_bytes=path.stat().st_size)
        expected_metadata = {"variant": "m", "training_recipe": "aps_warm_qk", "engineering_only": "false",
            "updates": str(returned), "source_aps_sha256": PINNED["outputs/grounding-assisted-point/adapter.safetensors"],
            "trainable_names": json.dumps(EXPECTED_QK, separators=(",", ":")),
            "train_order_sha256": content_digest(order), "manifest_sha256": PINNED["outputs/grounding-data/manifest.json"],
            "runner_sha256": PINNED["scripts/train_aps_warm_qk.py"]}
        audit.equal("checkpoint-header-metadata", metadata, expected_metadata)
        audit.equal("checkpoint-report-metadata", report.get("checkpoint_metadata"), metadata)
        audit.equal("checkpoint-summary", report.get("checkpoint"), {"sha256": checkpoint["sha256"],
            "count": count, "values": values, "file_bytes": checkpoint["file_bytes"], "metadata": metadata})
        checkpoint_ok = count == 101 and values == 40438795 and metadata == expected_metadata and all(t.get("dtype") == "F32" for t in header.values())
        audit.check("checkpoint-declared-layout", checkpoint_ok)
    elif report.get("checkpoint"):
        audit.check("reported-checkpoint-exists", False)
    frozen = {name: {"before": report.get(before), "after": report.get(after),
                    "runtime_record_matches": isinstance(report.get(before), str) and report.get(before) == report.get(after)}
              for name, before, after in (("base", "base_sha256_before", "base_sha256_after"),
                  ("other_57_adapters", "frozen_adapter_sha256_before", "frozen_adapter_sha256_after"))}
    conditions = report.get("endpoint_conditions", {})
    for key, value in (("base_frozen", frozen["base"]["runtime_record_matches"]),
                       ("other_adapters_frozen", frozen["other_57_adapters"]["runtime_record_matches"]),
                       ("updates", returned == 1000), ("slots", returned * 4 == 4000)):
        if key in conditions:
            audit.equal("condition." + key, conditions[key], value)
    if report.get("trainable_sha256_after"):
        audit.check("before-after-44-names", set(report["trainable_sha256_before"]) == set(report["trainable_sha256_after"]) == set(EXPECTED_QK))
        changes = {n: report["trainable_sha256_before"][n] != report["trainable_sha256_after"][n] for n in EXPECTED_QK}
        audit.equal("declared-qk-changes", report.get("trainable_bytes_changed"), changes)
    else:
        changes = None
    cap, wall = completion.get("total_soft_cap_seconds"), completion.get("seconds_before_completion_record")
    budget_ok = (type(cap) is int and cap == 1800 and finite(wall) and 0 <= wall <= cap
                 and finite(report.get("total_seconds")) and 0 <= report["total_seconds"] <= wall
                 and not completion.get("budget_exceeded"))
    if "seconds_after_initial_completion_record" in completion:
        budget_ok = budget_ok and completion["seconds_after_initial_completion_record"] <= cap
    complete_coverage = (set(evaluations) == {"initial", "final"}
                         and all(e and e["coverage_complete"] for e in evaluations.values()))
    endpoint = (report.get("status") == "complete" and report.get("run_completed") is True
                and returned == 1000 and restored and complete_coverage and checkpoint_ok and sources_available
                and names == EXPECTED_QK and report.get("train_order") == order
                and bool(conditions) and all(v is True for v in conditions.values())
                and all(v["runtime_record_matches"] for v in frozen.values())
                and completion.get("exit_code") == 0 and budget_ok
                and completion.get("source_unchanged") is True
                and not any(completion.get(k) for k in ("launch_error", "report_parse_error", "hash_errors", "terminated_after_native_return_grace")))
    audit.equal("external-valid-complete", completion.get("valid_complete"), endpoint)
    for name, pinned in inputs.items():
        audit.equal("input-still-stable." + name, digest(run / name), pinned)
    difference = None
    if complete_coverage:
        initial, final = evaluations["initial"], evaluations["final"]
        difference = {"forced_correct_final_minus_initial": final["forced_choice_correct"] - initial["forced_choice_correct"],
            "full_vocabulary_correct_final_minus_initial": final["full_vocabulary_argmax"]["correct"] - initial["full_vocabulary_argmax"]["correct"],
            "three_label_ce_final_minus_initial": final["three_label_ce"] - initial["three_label_ce"],
            "full_vocabulary_ce_final_minus_initial": final["full_vocabulary_ce"] - initial["full_vocabulary_ce"]}
    return {"inputs": inputs, "run_status": report.get("status"), "returned_updates": returned,
        "sample_slots": returned * 4, "restoration": {"passed": restored, "max_abs_logp_difference": max_delta,
        "predictions_match": choices_match, "metrics": restored_metrics}, "evaluations": evaluations,
        "descriptive_final_minus_initial": difference,
        "checkpoint": checkpoint, "runtime_frozen_hash_records": frozen, "qk_declared_bytes_changed": changes,
        "external_process": completion, "complete_endpoint_before_audit_failures": endpoint,
        "complete_endpoint": endpoint and not audit.failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-process-completion", action="store_true", required=True)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/aps-warm-qk")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/aps-warm-qk/saved-audit.json")
    args = parser.parse_args(argv)
    run, output = args.run.resolve(), args.output.resolve()
    if run.parent != ROOT / "outputs" or output.parent != run or output.name != "saved-audit.json":
        parser.error("Audit output must be saved-audit.json inside the selected direct outputs run directory")
    if not (run / "process-completion.json").is_file():
        parser.error("The real process completion record is required; never inspect a running attempt")
    completion = read_json(run / "process-completion.json")
    if type(completion.get("exit_code")) is not int:
        parser.error("Actual child exit code is missing")
    began = time.perf_counter()
    source_copy = run / "inspect-at-run.py"
    if output.exists() or source_copy.exists():
        parser.error("Preserve the existing audit report and source; no overwrite or automatic retry")
    source = Path(__file__).read_bytes()
    with source_copy.open("xb") as stream:
        stream.write(source)
    audit = Audit()
    result = {"status": "pending", "script_sha256": hashlib.sha256(source).hexdigest(),
        "run": run.relative_to(ROOT).as_posix(), "model_loaded": False, "tensor_payload_decoded": False,
        "limitations": ["Saved arithmetic and byte hashes only, not a repeat of runtime or training.",
            "Base and frozen-57 equality compare runtime-recorded digests; no independent parameter remeasurement.",
            "Checkpoint header declarations and file bytes are checked, but tensor payloads, finiteness and loader execution are not independently decoded or repeated.",
            "Full-vocabulary argmax cannot be derived from only three label scores; its saved token-to-label mapping is checked.",
            "Full-vocabulary CE uses saved gold log probability; three-class CE independently renormalizes the three scores.",
            "A consistent partial or error record is not a valid fixed endpoint or an effective negative scientific result.",
            "54 previously inspected questions are correlated within three games; these counts do not establish generalization or natural point-addressing mechanism."]}
    try:
        result.update(audit_run(run, audit))
        result["status"] = "consistent_saved_records" if not audit.failures else "inconsistent_saved_records"
    except Exception as exc:
        result.update(status="audit_error", error={"type": type(exc).__name__, "message": str(exc).replace(str(ROOT), ".")}, complete_endpoint=False)
    result.update(checks_completed=audit.count, failed_checks=audit.failures, seconds=time.perf_counter() - began)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result.get(k) for k in ("status", "checks_completed", "complete_endpoint", "seconds")}, allow_nan=False))
    return 0 if result["status"] == "consistent_saved_records" else 1


if __name__ == "__main__":
    raise SystemExit(main())
