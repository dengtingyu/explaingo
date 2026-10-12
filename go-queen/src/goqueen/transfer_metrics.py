"""Pure count, CE and alignment helpers for the frozen APS original54 evaluation.

Standard library only. Inputs are JSON-compatible score rows as produced by
goqueen.grounding_eval.score_questions (label order empty=0, black=1, white=2)
and the expected question records. Nothing is mutated, no file is touched and
no model runs; results are JSON-compatible and never contain NaN or infinity.

Malformed input raises ValueError naming the item and field: missing or
ill-typed identity/score fields, rows contradicting their own scores, or
ordering/identity/metadata violations. compare_restoration additionally
reports (rather than raises) content mismatches, so a failed restoration stays
visible as data; it never passes on malformed, partial or non-finite input.
"""
from __future__ import annotations

import copy
import math

LABEL_NAMES = ("empty", "black", "white")
SIDES = ("B", "W")
ESSENTIAL_FIELDS = ("question_id", "position_id", "game_id", "to_play", "point_index", "point", "label")
OPTIONAL_COMMON_FIELDS = ("split", "ply", "source_task_ids")
SCORE_FIELDS = ("label_log_probs", "forced_choice_label", "unconstrained_token_id",
                "unconstrained_label", "label_token_ids")


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _first_max(values):
    # max() keeps the first of equal maxima, matching the scorer's fixed label order.
    return max(range(3), key=values.__getitem__)


def _check_identity(item, where):
    if not isinstance(item, dict):
        raise ValueError(f"{where}: expected a dict")
    missing = [field for field in ESSENTIAL_FIELDS if field not in item]
    if missing:
        raise ValueError(f"{where}: missing essential fields {missing}")
    for field in ("question_id", "position_id", "game_id", "point"):
        if not isinstance(item[field], str) or not item[field]:
            raise ValueError(f"{where}: {field} must be a non-empty string")
    if item["to_play"] not in SIDES:
        raise ValueError(f"{where}: to_play must be B or W")
    if not _is_int(item["point_index"]) or not 0 <= item["point_index"] <= 360:
        raise ValueError(f"{where}: point_index must be an integer in [0, 360]")
    if not _is_int(item["label"]) or not 0 <= item["label"] <= 2:
        raise ValueError(f"{where}: label must be 0 (empty), 1 (black) or 2 (white)")


def _metadata_mismatches(first, second):
    """Common fields that differ; essential fields are already present in both."""
    return [field for field in ESSENTIAL_FIELDS + OPTIONAL_COMMON_FIELDS
            if field in first and field in second and first[field] != second[field]]


def _check_scores(row, where, require_finite=True):
    """Validate one score row against its own stored choices; return (scores, all_finite)."""
    missing = [field for field in SCORE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"{where}: missing score fields {missing}")
    raw = row["label_log_probs"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 3 or not all(_is_real(v) for v in raw):
        raise ValueError(f"{where}: label_log_probs must be three real numbers")
    values = [float(v) for v in raw]
    finite = all(math.isfinite(v) for v in values)
    if require_finite and not finite:
        raise ValueError(f"{where}: non-finite label_log_probs")
    if any(math.isfinite(v) and v > 0.0 for v in values):
        raise ValueError(f"{where}: a full-vocabulary log probability exceeds zero")
    tokens = row["label_token_ids"]
    if (not isinstance(tokens, (list, tuple)) or len(tokens) != 3
            or not all(_is_int(t) for t in tokens) or len(set(tokens)) != 3):
        raise ValueError(f"{where}: label_token_ids must be three distinct integers")
    forced = row["forced_choice_label"]
    if not _is_int(forced) or not 0 <= forced <= 2:
        raise ValueError(f"{where}: forced_choice_label must be 0, 1 or 2")
    if finite and forced != _first_max(values):
        raise ValueError(f"{where}: forced_choice_label is not the first maximum of label_log_probs")
    token, stored = row["unconstrained_token_id"], row["unconstrained_label"]
    if not _is_int(token):
        raise ValueError(f"{where}: unconstrained_token_id must be an integer")
    if stored is not None and (not _is_int(stored) or not 0 <= stored <= 2):
        raise ValueError(f"{where}: unconstrained_label must be null, 0, 1 or 2")
    mapped = list(tokens).index(token) if token in tokens else None
    if stored != mapped:
        raise ValueError(f"{where}: unconstrained_label disagrees with label_token_ids")
    if stored is not None and finite and values[stored] != max(values):
        raise ValueError(f"{where}: stored full-vocabulary argmax is not a maximum of its label scores")
    return values, finite


def _expected_list(questions):
    if not isinstance(questions, (list, tuple)) or not questions:
        raise ValueError("expected questions must be a non-empty list")
    seen = set()
    for index, question in enumerate(questions):
        _check_identity(question, f"question {index}")
        if question["question_id"] in seen:
            raise ValueError(f"question {index}: duplicate question_id {question['question_id']!r}")
        seen.add(question["question_id"])
    return list(questions)


def _three_label_ce(values, gold):
    top = max(values)
    return top + math.log(math.fsum(math.exp(v - top) for v in values)) - values[gold]


def _groups(key, names, expected, checked):
    groups = {}
    for name in names:
        rows = [row for row, _ in checked if row[key] == name]
        correct = sum(row["forced_choice_label"] == row["label"] for row in rows)
        groups[name] = {"requested": sum(q[key] == name for q in expected), "completed": len(rows),
                        "correct": correct, "accuracy": _rate(correct, len(rows))}
    return groups


def summarize_transfer(rows, expected_questions):
    """Counts and CE for an ordered completed prefix (possibly empty or complete) of the expected list."""
    expected = _expected_list(expected_questions)
    if not isinstance(rows, (list, tuple)):
        raise ValueError("rows must be a list")
    if len(rows) > len(expected):
        raise ValueError(f"{len(rows)} rows exceed the {len(expected)} expected questions")
    checked, seen = [], set()
    for index, row in enumerate(rows):
        where = f"row {index}"
        _check_identity(row, where)
        question_id = row["question_id"]
        if question_id in seen:
            raise ValueError(f"{where}: duplicate question_id {question_id!r}")
        seen.add(question_id)
        if question_id != expected[index]["question_id"]:
            raise ValueError(f"{where}: {question_id!r} is not expected question {index} "
                             f"({expected[index]['question_id']!r}); rows must be an ordered prefix")
        mismatched = _metadata_mismatches(row, expected[index])
        if mismatched:
            raise ValueError(f"{where}: metadata differs from the expected question: {mismatched}")
        checked.append((row, _check_scores(row, where)[0]))

    matrix = [[0, 0, 0] for _ in range(3)]
    ties = unique_max_correct = vocabulary_correct = off_label = 0
    three_label, full_vocabulary = [], []
    for row, values in checked:
        gold = row["label"]
        matrix[gold][row["forced_choice_label"]] += 1
        ties += values.count(max(values)) > 1
        unique_max_correct += all(values[gold] > v for y, v in enumerate(values) if y != gold)
        off_label += row["unconstrained_label"] is None
        vocabulary_correct += row["unconstrained_label"] == gold
        three_label.append(_three_label_ce(values, gold))
        full_vocabulary.append(-values[gold])

    completed, requested = len(checked), len(expected)
    correct = sum(matrix[y][y] for y in range(3))
    classes = {}
    for y, name in enumerate(LABEL_NAMES):
        done = sum(matrix[y])
        classes[name] = {"requested": sum(q["label"] == y for q in expected), "completed": done,
                         "correct": matrix[y][y], "recall": _rate(matrix[y][y], done)}
    recalls = [entry["recall"] for entry in classes.values()]
    occupancy_correct = matrix[0][0] + sum(matrix[y][p] for y in (1, 2) for p in (1, 2))
    color_correct = matrix[1][1] + matrix[2][2]
    occupied_completed = sum(matrix[1]) + sum(matrix[2])
    coverage_complete = completed == requested
    return {
        "coverage_complete": coverage_complete,
        "coverage_scope": "complete" if coverage_complete else "partial_prefix" if completed else "empty",
        "completed_questions": completed,
        "requested_questions": requested,
        "label_order": list(LABEL_NAMES),
        "forced_choice_correct": correct,
        "forced_choice_accuracy": _rate(correct, completed),
        "denominator_note": "Rates use completed rows only; requested counts are the fixed list. A partial prefix is not the full-list result.",
        "macro_recall": math.fsum(recalls) / 3 if all(r is not None for r in recalls) else None,
        "macro_recall_convention": "Mean of the three class recalls over completed rows; null if any class has no completed row (never counted as zero).",
        "by_class": classes,
        "confusion_matrix": matrix,
        "confusion_matrix_axes": "rows=true label, columns=forced choice, order empty/black/white",
        "exact_top_tie_count": ties,
        "unique_max_correct": unique_max_correct,
        "tie_rule": "Forced choice keeps the scorer's first maximum in label order; exact ties use no tolerance and stay in every denominator.",
        "three_label_ce": _mean(three_label),
        "full_vocabulary_ce": _mean(full_vocabulary),
        "occupancy": {"correct": occupancy_correct, "completed": completed, "requested": requested,
                      "rate": _rate(occupancy_correct, completed),
                      "definition": "Occupied (black or white) versus empty, collapsed from the confusion matrix."},
        "color_on_true_occupied": {"correct": color_correct, "completed": occupied_completed,
                                   "requested": sum(q["label"] != 0 for q in expected),
                                   "rate": _rate(color_correct, occupied_completed),
                                   "definition": "Exact colour on truly occupied items; an empty prediction counts as wrong."},
        "full_vocabulary_argmax": {"correct": vocabulary_correct, "off_label_choices": off_label,
                                   "completed": completed, "rate": _rate(vocabulary_correct, completed),
                                   "caveat": "Uses the stored full-vocabulary argmax token and its label mapping; three label scores cannot reconstruct the vocabulary."},
        "by_game": _groups("game_id", dict.fromkeys(q["game_id"] for q in expected), expected, checked),
        "by_side": _groups("to_play", [s for s in SIDES if any(q["to_play"] == s for q in expected)],
                           expected, checked),
    }


def align_saved_reference(questions, saved_rows):
    """Return deep copies of saved rows in question order; one-to-one IDs and consistent metadata."""
    expected = _expected_list(questions)
    if not isinstance(saved_rows, (list, tuple)):
        raise ValueError("saved rows must be a list")
    by_id = {}
    for index, row in enumerate(saved_rows):
        _check_identity(row, f"saved row {index}")
        if row["question_id"] in by_id:
            raise ValueError(f"saved row {index}: duplicate question_id {row['question_id']!r}")
        by_id[row["question_id"]] = row
    expected_ids = [q["question_id"] for q in expected]
    missing = [qid for qid in expected_ids if qid not in by_id]
    extra = sorted(set(by_id) - set(expected_ids))
    if missing or extra:
        raise ValueError(f"saved rows are not one-to-one with the questions; missing={missing}, extra={extra}")
    aligned = []
    for question in expected:
        row = by_id[question["question_id"]]
        mismatched = _metadata_mismatches(row, question)
        if mismatched:
            raise ValueError(f"saved row {question['question_id']!r}: metadata differs: {mismatched}")
        aligned.append(copy.deepcopy(row))
    return aligned


def compare_restoration(current_rows, saved_rows, tolerance=1e-5):
    """Compare re-scored rows with the full saved endpoint; pass only on complete, exact-choice agreement."""
    if not _is_real(tolerance) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative number")
    checked = {}
    for name, rows in (("current", current_rows), ("saved", saved_rows)):
        if not isinstance(rows, (list, tuple)):
            raise ValueError(f"{name} rows must be a list")
        entries = []
        for index, row in enumerate(rows):
            where = f"{name} row {index}"
            _check_identity(row, where)
            values, finite = _check_scores(row, where, require_finite=False)
            if not _is_int(row.get("prefix_tokens")) or row["prefix_tokens"] < 1:
                raise ValueError(f"{where}: prefix_tokens must be a positive integer")
            entries.append((row, values, finite))
        checked[name] = entries
    if not checked["saved"]:
        raise ValueError("saved rows must contain the full saved endpoint")

    current_ids = [row["question_id"] for row, _, _ in checked["current"]]
    saved_ids = [row["question_id"] for row, _, _ in checked["saved"]]
    issues = [f"duplicate question IDs in {name} rows"
              for name, ids in (("current", current_ids), ("saved", saved_ids)) if len(set(ids)) != len(ids)]
    if len(current_ids) > len(saved_ids):
        issues.append(f"{len(current_ids) - len(saved_ids)} current rows exceed the saved endpoint")
    elif len(current_ids) < len(saved_ids):
        issues.append(f"only {len(current_ids)} of {len(saved_ids)} saved questions were re-scored")
    coverage_complete = current_ids == saved_ids

    differences, maxima, predictions_match, nonfinite = [], [], True, False
    for index, ((current, cur_values, cur_finite), (saved, saved_values, saved_finite)) in enumerate(
            zip(checked["current"], checked["saved"])):
        entry = {"index": index, "question_id": saved["question_id"], "issues": []}
        differences.append(entry)
        if current["question_id"] != saved["question_id"]:
            entry["current_question_id"] = current["question_id"]
            entry["issues"].append("question_id differs from the saved order")
            predictions_match = False
            continue
        mismatched = _metadata_mismatches(current, saved)
        if mismatched:
            entry["issues"].append(f"metadata differs: {mismatched}")
        if list(current["label_token_ids"]) != list(saved["label_token_ids"]):
            entry["issues"].append("label_token_ids differ")
        if current["prefix_tokens"] != saved["prefix_tokens"]:
            entry["issues"].append("prefix_tokens differ")
        if cur_finite and saved_finite:
            # All scores are finite and <= 0, so these differences cannot overflow.
            entry["max_abs_logp_difference"] = max(abs(a - b) for a, b in zip(cur_values, saved_values))
            maxima.append(entry["max_abs_logp_difference"])
            if entry["max_abs_logp_difference"] > tolerance:
                entry["issues"].append("label log probability difference exceeds tolerance")
        else:
            nonfinite = True
            entry["max_abs_logp_difference"] = None
            entry["issues"].append("non-finite label log probabilities")
        entry.update(current_forced_choice=current["forced_choice_label"],
                     saved_forced_choice=saved["forced_choice_label"],
                     current_unconstrained_token_id=current["unconstrained_token_id"],
                     saved_unconstrained_token_id=saved["unconstrained_token_id"])
        if current["forced_choice_label"] != saved["forced_choice_label"]:
            entry["issues"].append("forced choice differs")
            predictions_match = False
        if (current["unconstrained_token_id"] != saved["unconstrained_token_id"]
                or current["unconstrained_label"] != saved["unconstrained_label"]):
            entry["issues"].append("full-vocabulary argmax differs")
            predictions_match = False

    max_difference = max(maxima) if maxima else None
    passed = (coverage_complete and not issues and not nonfinite and predictions_match
              and max_difference is not None and max_difference <= tolerance
              and not any(entry["issues"] for entry in differences))
    return {"passed": passed, "coverage_complete": coverage_complete,
            "current_questions": len(current_ids), "saved_questions": len(saved_ids),
            "max_abs_logp_difference": max_difference, "predictions_match": predictions_match,
            "tolerance": tolerance, "issues": issues, "differences": differences,
            "rule": "Pass only if every saved question is re-scored in order with equal metadata, label tokens and prefix length, finite scores within tolerance and identical forced and full-vocabulary choices. A failure stops interpretation; it does not justify a different tolerance."}
