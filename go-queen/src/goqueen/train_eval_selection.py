"""Fixed evaluation list and multiplicity glue for the fixed-weight training-sample evaluation.

Standard library plus goqueen.transfer_metrics (itself pure Python). Nothing here
loads a model, reads a tensor or touches a file.

* observed_slots: the question IDs the completed warm-Q/K run actually used, taken
  from its saved step_attempts in step order (observed exposure, not a schedule).
* build_eval_manifest: checks those slots against the pinned train_order, the
  training pool and the saved comparison list. Keeps every unique ID once with its
  exposure count and first/last step, ordered by sha256(question_id): label-, score-
  and recency-independent, so a deadline prefix is not a time-ordered slice.
* validate_eval_manifest: recomputes every invariant and the content digest.
* restoration_subset: the first saved positions' rows, for the engineering
  restoration check only; never part of any metric.
* describe / exposure_weighted_summary: call transfer_metrics.summarize_transfer
  on the whole list and on class or exposure-count partitions, then combine
  linearly. Accuracy and CE are never re-implemented.

Scores are descriptive only: no bands, significance or causal labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from goqueen.transfer_metrics import summarize_transfer

RULE = ("all unique question IDs in the completed run's saved step_attempts, each scored once "
        "at the fixed final weights, ordered by sha256(question_id) then question_id")
LABEL_NAMES = ("empty", "black", "white")
WEIGHTING = ("exposure-weighted: each unique question's row counts once per training slot that used it; "
             "equivalent to scoring every slot, because the weights are fixed and each ID is scored once")


def observed_slots(step_attempts, *, steps=1000, batch_size=4):
    """Flatten the saved step_attempts into the ordered list of training slots actually used."""
    if not isinstance(step_attempts, list) or len(step_attempts) != steps:
        raise ValueError(f"expected exactly {steps} saved step attempts")
    slots = []
    for number, attempt in enumerate(step_attempts, start=1):
        if not isinstance(attempt, dict) or type(attempt.get("step")) is not int or attempt["step"] != number:
            raise ValueError(f"step attempt {number}: missing or out of order")
        if attempt.get("optimizer_step_started") is not True or attempt.get("optimizer_step_returned") is not True:
            raise ValueError(f"step {number}: the optimizer step did not both start and return")
        ids = attempt.get("question_ids")
        if (not isinstance(ids, list) or len(ids) != batch_size
                or not all(isinstance(qid, str) and qid for qid in ids)):
            raise ValueError(f"step {number}: expected {batch_size} question IDs")
        slots.extend(ids)
    return slots


def build_eval_manifest(slots, intended_order, pool_questions, comparison_questions, *,
                        order_sha256, batch_size=4, source=None):
    """The fixed evaluation list: every observed unique ID once, with its exposure record."""
    slots, intended = list(slots), list(intended_order)
    if not slots or len(slots) % batch_size:
        raise ValueError(f"slots must be a non-empty multiple of the batch size {batch_size}")
    if _content_sha256(intended) != order_sha256:
        raise ValueError("train_order does not match its pinned content digest")
    if slots != intended:
        raise ValueError("observed training slots differ from the pinned train_order")
    pool = {}
    for index, question in enumerate(pool_questions):
        if not isinstance(question, dict) or not isinstance(question.get("question_id"), str):
            raise ValueError(f"pool question {index}: missing question_id")
        if question["question_id"] in pool:
            raise ValueError(f"duplicate pool question_id {question['question_id']!r}")
        pool[question["question_id"]] = question
    exposures = {}
    for index, qid in enumerate(slots):
        if qid not in pool:
            raise ValueError(f"slot {index}: {qid!r} is not in the training pool")
        if pool[qid].get("split") != "train" or pool[qid].get("label") not in (0, 1, 2):
            raise ValueError(f"slot {index}: {qid!r} is not a labelled training question")
        step = index // batch_size + 1
        record = exposures.setdefault(qid, {"exposure_count": 0, "first_step": step, "last_step": step})
        record["exposure_count"] += 1
        record["last_step"] = step
    order = sorted(exposures, key=_order_key)
    entries = [{"question": copy.deepcopy(pool[qid]), **exposures[qid]} for qid in order]
    manifest = {"rule": RULE, "batch_size": batch_size, "train_order_sha256": order_sha256,
                "source": copy.deepcopy(source or {}), "slots": len(slots),
                "unique_questions": len(entries), "counts": _counts(entries),
                "separation": _separation(entries, comparison_questions), "entries": entries}
    manifest["content_sha256"] = _content_sha256(manifest)
    return manifest


def validate_eval_manifest(manifest):
    """Recompute every invariant; raise ValueError on any inconsistency."""
    if not isinstance(manifest, dict) or manifest.get("rule") != RULE:
        raise ValueError("not an evaluation manifest built by this rule")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest has no entries")
    ids = [entry["question"]["question_id"] for entry in entries]
    if ids != sorted(ids, key=_order_key) or len(set(ids)) != len(ids):
        raise ValueError("entries are not unique and in sha256 order")
    for entry in entries:
        count, first, last = entry.get("exposure_count"), entry.get("first_step"), entry.get("last_step")
        if not all(type(v) is int for v in (count, first, last)) or count < 1 or not 1 <= first <= last:
            raise ValueError(f"{entry['question']['question_id']}: invalid exposure record")
        if entry["question"].get("split") != "train":
            raise ValueError(f"{entry['question']['question_id']}: not a training question")
    if sum(entry["exposure_count"] for entry in entries) != manifest.get("slots"):
        raise ValueError("exposure counts do not sum to the slot count")
    if manifest.get("unique_questions") != len(entries) or manifest.get("counts") != _counts(entries):
        raise ValueError("manifest counts disagree with its entries")
    separation = manifest.get("separation") or {}
    if any(separation.get(key) != 0 for key in ("game_id_overlap", "position_id_overlap", "question_id_overlap")):
        raise ValueError("manifest does not record complete separation from the comparison list")
    if manifest.get("content_sha256") != _content_sha256(manifest):
        raise ValueError("manifest content digest does not match")
    return True


def questions_from_manifest(manifest):
    """The expected question list, in scoring order, for the scorer and summarize_transfer."""
    return [copy.deepcopy(entry["question"]) for entry in manifest["entries"]]


def restoration_subset(saved_rows, positions=2):
    """Saved rows of the first `positions` position IDs in saved order; one per label, both sides covered."""
    if not isinstance(saved_rows, list) or not saved_rows:
        raise ValueError("saved rows must be a non-empty list")
    chosen = list(dict.fromkeys(row["position_id"] for row in saved_rows))[:positions]
    if len(chosen) != positions:
        raise ValueError(f"saved rows contain fewer than {positions} positions")
    subset = [copy.deepcopy(row) for row in saved_rows if row["position_id"] in chosen]
    if len(subset) != 3 * positions or any(
            sorted(row["label"] for row in subset if row["position_id"] == pid) != [0, 1, 2] for pid in chosen):
        raise ValueError("each restoration position must contribute exactly one row per label")
    if {row["to_play"] for row in subset} != {"B", "W"}:
        raise ValueError("restoration rows must cover both sides to play")
    return subset


def describe(rows, questions):
    """summarize_transfer on the whole list plus per-class CE from class subsets."""
    summary = summarize_transfer(rows, questions)  # also validates rows as an ordered prefix
    per_class = {}
    for label, name in enumerate(LABEL_NAMES):
        subset = [q for q in questions if q["label"] == label]
        if not subset:
            per_class[name] = None
            continue
        part = summarize_transfer([r for r in rows if r["label"] == label], subset)
        per_class[name] = {key: part[key] for key in (
            "requested_questions", "completed_questions", "forced_choice_correct",
            "forced_choice_accuracy", "three_label_ce", "full_vocabulary_ce")}
        per_class[name]["predicted_counts"] = part["confusion_matrix"][label]
    return {"summary": summary, "per_class": per_class}


def exposure_weighted_summary(rows, manifest):
    """Summarize each exposure-count stratum, then weight every row by its training-slot count."""
    questions = questions_from_manifest(manifest)
    summarize_transfer(rows, questions)  # validates rows as an ordered prefix of the list
    counts = {entry["question"]["question_id"]: entry["exposure_count"] for entry in manifest["entries"]}
    parts = {}
    for k in sorted(set(counts.values())):
        parts[k] = summarize_transfer([r for r in rows if counts[r["question_id"]] == k],
                                      [q for q in questions if counts[q["question_id"]] == k])
    matrix = [[sum(k * part["confusion_matrix"][y][p] for k, part in parts.items()) for p in range(3)]
              for y in range(3)]
    completed = sum(k * part["completed_questions"] for k, part in parts.items())
    requested = sum(k * part["requested_questions"] for k, part in parts.items())

    def weighted_mean(field):
        pairs = [(k * part["completed_questions"], part[field]) for k, part in parts.items()
                 if part["completed_questions"]]
        total = sum(weight for weight, _ in pairs)
        return math.fsum(weight * value for weight, value in pairs) / total if total else None

    by_class = {}
    for y, name in enumerate(LABEL_NAMES):
        done = sum(matrix[y])
        by_class[name] = {"completed_slots": done, "correct_slots": matrix[y][y],
                          "recall": matrix[y][y] / done if done else None}
    recalls = [entry["recall"] for entry in by_class.values()]
    correct = sum(matrix[y][y] for y in range(3))
    return {
        "weighting": WEIGHTING,
        "requested_slots": requested,
        "completed_slots": completed,
        "coverage_complete": completed == requested,
        "confusion_matrix": matrix,
        "forced_choice_accuracy": correct / completed if completed else None,
        "by_class": by_class,
        "macro_recall": math.fsum(recalls) / 3 if all(r is not None for r in recalls) else None,
        "three_label_ce": weighted_mean("three_label_ce"),
        "full_vocabulary_ce": weighted_mean("full_vocabulary_ce"),
        "strata": {str(k): {"exposure_count": k, "requested_questions": part["requested_questions"],
                            "completed_questions": part["completed_questions"],
                            "forced_choice_accuracy": part["forced_choice_accuracy"],
                            "three_label_ce": part["three_label_ce"]}
                   for k, part in parts.items()},
    }


def _order_key(question_id):
    return hashlib.sha256(question_id.encode("utf-8")).hexdigest(), question_id


def _counts(entries):
    by_class = {name: {"questions": 0, "slots": 0} for name in LABEL_NAMES}
    by_exposure = {}
    for entry in entries:
        question, count = entry["question"], entry["exposure_count"]
        name = LABEL_NAMES[question["label"]]
        by_class[name]["questions"] += 1
        by_class[name]["slots"] += count
        key = f"{name}:{count}"
        by_exposure[key] = by_exposure.get(key, 0) + 1
    return {"by_class": by_class, "by_class_and_exposure": dict(sorted(by_exposure.items())),
            "games": len({e["question"]["game_id"] for e in entries}),
            "positions": len({e["question"]["position_id"] for e in entries}),
            "point_indices": len({e["question"]["point_index"] for e in entries})}


def _separation(entries, comparison_questions):
    if not isinstance(comparison_questions, list) or not comparison_questions:
        raise ValueError("comparison questions must be a non-empty list")
    selected = [entry["question"] for entry in entries]
    overlap = {f"{key}_overlap": len({q[key] for q in selected} & {q[key] for q in comparison_questions})
               for key in ("game_id", "position_id", "question_id")}
    if any(overlap.values()):
        raise ValueError(f"evaluation list overlaps the comparison list: {overlap}")
    # Shared prompt text is expected (prompts depend only on side and point); it is recorded, not rejected.
    prompts = {q.get("prompt") for q in selected} - {None}
    overlap["comparison_prompts_in_list"] = sum(q.get("prompt") in prompts for q in comparison_questions)
    return overlap


def _content_sha256(value):
    # Same canonical JSON convention as the project's content_sha256.
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if key != "content_sha256"}
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
