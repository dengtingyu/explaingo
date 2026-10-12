#!/usr/bin/env python3
"""Describe saved warm-Q/K scores by ply and side; no model or project imports."""
import collections
import hashlib
import json
import math
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/aps-warm-qk-training-strata"
INPUTS = {
    "training_rows": "outputs/aps-warm-qk-train-eval/training-rows.jsonl",
    "manifest": "outputs/aps-warm-qk-train-eval/eval-manifest.json",
    "warm_qk_report": "outputs/aps-warm-qk/report.json",
}
LABELS = ["empty", "black", "white"]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def describe(rows, weights):
    """All counts use the supplied multiplicity, including conditional counts."""
    matrix = [[0] * 3 for _ in range(3)]
    relative = [[0] * 2 for _ in range(2)]
    ce, full_ce = [], []
    ties = 0
    for row in rows:
        weight = weights[row["question_id"]]
        gold = row["label"]
        scores = row["label_log_probs"]
        top = max(scores)
        predicted = scores.index(top)
        matrix[gold][predicted] += weight
        ce.append(weight * (math.log(math.fsum(math.exp(x - top) for x in scores)) + top - scores[gold]))
        full_ce.append(-weight * scores[gold])
        ties += weight * (scores.count(top) > 1)
        if gold != 0 and predicted != 0:
            own = 1 if row["to_play"] == "B" else 2
            relative[int(gold != own)][int(predicted != own)] += weight
    true_counts = [sum(line) for line in matrix]
    predicted_counts = [sum(matrix[g][p] for g in range(3)) for p in range(3)]
    total = sum(true_counts)
    occupied = true_counts[1] + true_counts[2]
    occupied_called_empty = matrix[1][0] + matrix[2][0]
    occupied_called_occupied = occupied - occupied_called_empty
    empty_called_occupied = matrix[0][1] + matrix[0][2]
    color_correct = matrix[1][1] + matrix[2][2]
    occupancy_matrix = [[matrix[0][0], empty_called_occupied],
                        [occupied_called_empty, occupied_called_occupied]]
    return {
        "unique_rows": len(rows), "count": total,
        "true_label_counts": true_counts, "predicted_label_counts": predicted_counts,
        "confusion_matrix": matrix,
        "correct": sum(matrix[i][i] for i in range(3)),
        "accuracy": ratio(sum(matrix[i][i] for i in range(3)), total),
        "three_label_ce": ratio(math.fsum(ce), total),
        "full_vocabulary_ce": ratio(math.fsum(full_ce), total),
        "exact_top_tie_count": ties,
        "occupancy": {
            "confusion_matrix": occupancy_matrix,
            "correct": matrix[0][0] + occupied_called_occupied,
            "accuracy": ratio(matrix[0][0] + occupied_called_occupied, total),
            "predicted_occupied_count": sum(predicted_counts[1:]),
            "predicted_occupied_rate": ratio(sum(predicted_counts[1:]), total),
            "called_occupied_given_true_occupied": ratio(occupied_called_occupied, occupied),
            "called_occupied_given_true_empty": ratio(empty_called_occupied, true_counts[0]),
            "in_sample_majority_correct": max(occupied, true_counts[0]),
        },
        "color_given_true_and_predicted_occupied": {
            "absolute_confusion_matrix": [matrix[1][1:], matrix[2][1:]],
            "correct": color_correct, "denominator": occupied_called_occupied,
            "accuracy": ratio(color_correct, occupied_called_occupied),
            "predict_black_given_black": ratio(matrix[1][1], matrix[1][1] + matrix[1][2]),
            "predict_black_given_white": ratio(matrix[2][1], matrix[2][1] + matrix[2][2]),
            "own_opponent_confusion_matrix": relative,
            "predicted_own_count": relative[0][0] + relative[1][0],
            "predicted_own_rate": ratio(relative[0][0] + relative[1][0], occupied_called_occupied),
            "predict_own_given_own": ratio(relative[0][0], sum(relative[0])),
            "predict_own_given_opponent": ratio(relative[1][0], sum(relative[1])),
        },
        "color_on_all_true_occupied": {
            "correct": color_correct, "denominator": occupied,
            "accuracy": ratio(color_correct, occupied),
        },
    }


def grouped(rows, weights, field):
    return {str(value): describe([row for row in rows if row[field] == value], weights)
            for value in sorted({row[field] for row in rows})}


def empirical_priors(rows, weights):
    """MLE lookup references fitted and evaluated on the same saved labels."""
    global_counts = [sum(weights[r["question_id"]] for r in rows if r["label"] == k) for k in range(3)]
    total = sum(global_counts)

    def fitted(counts):
        count = sum(counts)
        entropy = -math.fsum(n * math.log(n / count) for n in counts if n) / count
        return {"count": count, "class_counts": counts,
                "class_probabilities": [n / count for n in counts],
                "three_label_ce": entropy,
                "in_sample_majority_correct": max(counts)}

    by_ply = {str(ply): fitted([sum(weights[r["question_id"]] for r in rows
                                  if r["ply"] == ply and r["label"] == k) for k in range(3)])
              for ply in sorted({r["ply"] for r in rows})}
    global_fit = fitted(global_counts)
    ply_ce = math.fsum(g["count"] * g["three_label_ce"] for g in by_ply.values()) / total
    model_ce = describe(rows, weights)["three_label_ce"]
    return {"global": global_fit, "by_ply": by_ply,
            "ply_lookup_three_label_ce": ply_ce,
            "ply_lookup_majority_correct": sum(g["in_sample_majority_correct"] for g in by_ply.values()),
            "model_three_label_ce": model_ce,
            "model_minus_global_prior_ce": model_ce - global_fit["three_label_ce"],
            "model_minus_ply_prior_ce": model_ce - ply_ce,
            "uniform_three_label_ce": math.log(3)}


def prediction_structure(rows):
    """Check position balance separately from whether actual predictions agree."""
    positions = []
    for position in sorted({r["position_id"] for r in rows}):
        members = [r for r in rows if r["position_id"] == position]
        golds = collections.Counter(r["label"] for r in members)
        predictions = collections.Counter(r["forced_choice_label"] for r in members)
        positions.append({"position_id": position, "plies": sorted({r["ply"] for r in members}),
                          "sides": sorted({r["to_play"] for r in members}),
                          "questions": len(members), "true_label_counts": [golds[k] for k in range(3)],
                          "predicted_label_counts": [predictions[k] for k in range(3)],
                          "constant_prediction": len(predictions) == 1,
                          "correct": sum(r["label"] == r["forced_choice_label"] for r in members)})
    per_ply = {}
    for ply in sorted({r["ply"] for r in rows}):
        subset = [r for r in rows if r["ply"] == ply]
        groups = [p for p in positions if p["plies"] == [ply]]
        predictions = collections.Counter(r["forced_choice_label"] for r in subset)
        per_ply[str(ply)] = {
            "questions": len(subset), "positions": len(groups),
            "sides": sorted({r["to_play"] for r in subset}),
            "predicted_label_counts": [predictions[k] for k in range(3)],
            "distinct_prediction_labels": len(predictions),
            "constant_prediction_positions": sum(p["constant_prediction"] for p in groups),
            "mixed_prediction_positions": sum(not p["constant_prediction"] for p in groups),
            "correct": sum(r["label"] == r["forced_choice_label"] for r in subset),
        }
    return {"positions": positions, "position_count": len(positions),
            "every_position_exactly_one_each_label": all(p["true_label_counts"] == [1, 1, 1] for p in positions),
            "every_position_one_side_and_ply": all(len(p["sides"]) == len(p["plies"]) == 1 for p in positions),
            "constant_prediction_positions": sum(p["constant_prediction"] for p in positions),
            "mixed_prediction_positions": sum(not p["constant_prediction"] for p in positions),
            "by_ply": per_ply}


def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (OUTPUT / "analysis-at-run.py").write_bytes(source)
    contents = {name: (ROOT / rel).read_bytes() for name, rel in INPUTS.items()}
    rows = [json.loads(line) for line in contents["training_rows"].decode().splitlines()]
    manifest = json.loads(contents["manifest"])
    old = json.loads(contents["warm_qk_report"])
    final54 = [e for e in old["evaluations"] if e["name"] == "final"]
    if len(final54) != 1 or not final54[0]["coverage_complete"] or not old["run_completed"]:
        raise ValueError("Expected one completed saved final54 evaluation")
    reference = final54[0]["rows"]
    entries = manifest["entries"]
    weights = {e["question"]["question_id"]: e["exposure_count"] for e in entries}
    actual_slots = [qid for step in old["step_attempts"] if step["optimizer_step_returned"] for qid in step["question_ids"]]
    if len(rows) != 2702 or len(reference) != 54 or len(actual_slots) != 4000:
        raise ValueError("Unexpected fixed saved-row or exposure counts")
    if weights != dict(collections.Counter(actual_slots)):
        raise ValueError("Manifest exposure weights differ from saved returned training slots")
    if [r["question_id"] for r in rows] != [e["question"]["question_id"] for e in entries] or len(weights) != len(rows):
        raise ValueError("Training rows do not match unique ordered manifest")
    for row, entry in zip(rows, entries):
        if any(row[k] != entry["question"][k] for k in ["label", "ply", "to_play", "position_id", "game_id"]):
            raise ValueError("Saved row facts differ from manifest")
    for row in rows + reference:
        scores = row["label_log_probs"]
        if row["label"] not in range(3) or row["to_play"] not in ("B", "W") or len(scores) != 3 or not all(math.isfinite(x) for x in scores):
            raise ValueError("Invalid saved scoring row")
        if row["forced_choice_label"] != scores.index(max(scores)):
            raise ValueError("Stored forced choice differs from first maximum")
    training = {}
    for name, multiplicity in [("unique", {r["question_id"]: 1 for r in rows}), ("exposure_weighted", weights)]:
        training[name] = {"overall": describe(rows, multiplicity),
                          "by_ply": grouped(rows, multiplicity, "ply"),
                          "by_side": grouped(rows, multiplicity, "to_play"),
                          "empirical_priors": empirical_priors(rows, multiplicity)}
    ref_weights = {r["question_id"]: 1 for r in reference}
    report = {
        "stage": "aps_warm_qk_training_strata", "status": "complete",
        "scope": "Descriptive arithmetic on saved fixed-final score rows and original exposure multiplicities",
        "source_script_sha256": hashlib.sha256(source).hexdigest(),
        "inputs": {INPUTS[name]: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
        "model_loaded": False, "tensor_payload_decoded": False, "model_forward_calls": 0,
        "training": training, "training_prediction_structure": prediction_structure(rows),
        "saved54": {"overall": describe(reference, ref_weights),
                    "by_ply": grouped(reference, ref_weights, "ply"),
                    "by_side": grouped(reference, ref_weights, "to_play"),
                    "prediction_structure": prediction_structure(reference)},
        "definitions": {
            "label_order": LABELS,
            "confusion_axes": "Rows=true label, columns=predicted label",
            "occupancy_axes": "Rows=true empty/occupied, columns=predicted empty/occupied",
            "conditional_color": "Only truly occupied rows also predicted occupied; excludes occupied points predicted empty",
            "relative_color_axes": "Rows=true own/opponent stone, columns=predicted own/opponent; own means side to play",
            "exposure_weighting": "Each fixed scored row weighted by its exact count among 4000 returned training slots; no additional observations",
            "empirical_prior": "Descriptive oracle lookup: class probabilities fitted and scored using the same labels and same weights, globally or separately at each ply; no held-out fitting",
            "prior_comparison": "References are three-class cross-entropies; comparisons use model three-label CE, not full-vocabulary CE",
            "majority_reference": "Same-data oracle count choosing the majority occupancy or class within that reported stratum",
        },
        "limitations": [
            "No new model scoring, parameter update, attention capture, tensor decoding or execution of a prior audit.",
            "Stratification is descriptive. Ply, side, class, coordinate and board composition can be confounded; residual differences do not identify point reading or its absence.",
            "Conditional colour near one half is a finite rate, not proof of no signal, random guessing or a causal failure mechanism.",
            "Same-data empirical priors use outcome labels and need not be realizable by the model; their CE is not a held-out baseline or a bound on architecture performance.",
            "If each position has one of each class, any position-constant predictor gets one correct per position. A total of 18/54 does not imply actual predictions are position-constant.",
            "Within-ply prediction heterogeneity rules out an exactly ply-only deterministic classification description on these rows; it does not establish causal board or coordinate use.",
            "No before-training scores exist for this training list. Training-versus54 and exposure strata are not causal comparisons or independent-sample significance tests.",
            "The 54 rows were saved during the prior run; they were not rescored here. Runtime model integrity is outside this saved-row analysis.",
        ],
        "seconds_before_report_write": time.perf_counter() - started,
    }
    with (OUTPUT / "report.json").open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "unique_rows": len(rows), "exposure_slots": sum(weights.values()),
                      "saved54_rows": len(reference), "seconds": time.perf_counter() - started}))


if __name__ == "__main__":
    main()
