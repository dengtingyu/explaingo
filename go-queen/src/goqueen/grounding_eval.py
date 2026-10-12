"""Read-only, causal occupancy-value scoring and descriptive board contrasts.

Scores are full-vocabulary log probabilities, never a renormalized three-way
softmax. Labels enter result calculations only, not the forward input.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import itertools
import json
import math
import time
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .grounding_data import LABEL_NAMES
from .smoke_data import encode_answer

TIE_TOLERANCE = 1e-6


class MeasurementDeadline(TimeoutError):
    """The CLI time budget expired; already completed rows remain usable."""


def encode_prefix(tokenizer, prompt: str) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Return the label-independent prefix and empty/black/white token IDs.

    Assert that all three independently encoded target strings share precisely
    the same prefix and suffix, and each value occupies exactly one BPE token.
    The returned prefix stops before that token; no gold value is supplied.
    """
    prefixes, suffixes, label_ids = [], [], []
    for name in LABEL_NAMES:
        target = json.dumps({"occupancy": name}, separators=(",", ":"))
        encoded, metadata = encode_answer(tokenizer, prompt, target)
        text = target + "<|im_end|>"
        value_start = target.index(":") + 2
        value_end = value_start + len(name)
        tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if tokens["input_ids"] != metadata["target_token_ids"]:
            raise ValueError("Answer tokenizer disagrees with established encoding")
        indices = [i for i, (a, b) in enumerate(tokens["offset_mapping"])
                   if a < value_end and b > value_start]
        if len(indices) != 1:
            raise ValueError("Each occupancy value must be a single BPE token")
        value_index = indices[0]
        # Exact coverage rules out a value token swallowing label-dependent
        # punctuation or only part of the value.
        if tuple(tokens["offset_mapping"][value_index]) != (value_start, value_end):
            raise ValueError("Occupancy value token must cover exactly the value")
        stop = metadata["answer_start"] + value_index
        prefixes.append(encoded["input_ids"][0, :stop])
        suffixes.append(metadata["target_token_ids"][value_index + 1:])
        label_ids.append(metadata["target_token_ids"][value_index])
    if not all(torch.equal(prefixes[0], x) for x in prefixes[1:]):
        raise ValueError("Occupancy labels do not share an identical token prefix")
    if not all(suffixes[0] == x for x in suffixes[1:]) or len(set(label_ids)) != 3:
        raise ValueError("Occupancy target suffixes or label token IDs disagree")
    return prefixes[0], tuple(label_ids)


def _metadata(question: dict) -> dict:
    # Retain identity and audit metadata, not the answer text in model inputs.
    names = ("question_id", "position_id", "game_id", "split", "ply", "to_play",
             "point_index", "point", "label", "source_task_ids")
    return {name: question[name] for name in names if name in question}


def _left_pad(prefixes: list[torch.Tensor], pad_id: int, device) -> dict:
    length = max(len(ids) for ids in prefixes)
    ids = torch.full((len(prefixes), length), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for row, prefix in enumerate(prefixes):
        ids[row, -len(prefix):] = prefix.to(device=device)
        mask[row, -len(prefix):] = 1
    positions = mask.cumsum(dim=-1) - 1
    positions.masked_fill_(mask.eq(0), 0)
    if not bool(mask[:, -1].eq(1).all()):
        raise ValueError("Every last column must be the actual prefix endpoint")
    return {"input_ids": ids, "attention_mask": mask, "position_ids": positions}


def _forward_logits(model, tokenizer, prefixes, selected, features_by_position):
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no padding or end-of-sequence token")
    inputs = _left_pad(prefixes, pad_id, device)
    features = []
    for question in selected:
        payload = features_by_position[question["position_id"]]
        tensor = payload["features"] if isinstance(payload, Mapping) else payload
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (11, 361, 768):
            raise ValueError("Expected a saved [11,361,768] feature tensor")
        features.append(tensor)
    inputs["expert_features"] = torch.stack(features).to(device=device, dtype=torch.float32)
    # Supported by the pinned local Qwen3 implementation. The full prefix is
    # processed, but only its last position is projected to the vocabulary.
    output = model(**inputs, logits_to_keep=1, use_cache=False)
    if output.logits.ndim != 3 or output.logits.shape[:2] != (len(selected), 1):
        raise ValueError("Expected exactly one output position for each prefix")
    return output.logits[:, -1].float()


def score_questions(model, tokenizer, questions: list[dict], features_by_position,
                    batch_size: int = 4, deadline: float | None = None) -> list[dict]:
    """Score completed batches until an absolute perf_counter deadline.

    One causal forward per question scores every candidate. The caller can set
    a process alarm for interruption; an in-flight batch is never partially
    reported. No generation or parameter update occurs. Training mode is restored.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be positive")
    rows, was_training = [], model.training
    model.eval()
    try:
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            for start in range(0, len(questions), batch_size):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                selected = questions[start:start + batch_size]
                try:
                    encoded = [encode_prefix(tokenizer, q["prompt"]) for q in selected]
                    label_ids = encoded[0][1]
                    if any(x[1] != label_ids for x in encoded):
                        raise ValueError("Label token IDs changed across prompts")
                    logits = _forward_logits(model, tokenizer, [x[0] for x in encoded],
                                             selected, features_by_position)
                    log_probs = logits.log_softmax(dim=-1)
                    candidates = log_probs[:, list(label_ids)]
                    off_mass = (1.0 - candidates.exp().sum(dim=-1)).clamp(0, 1)
                    unconstrained = logits.argmax(dim=-1)
                    inverse = {token: label for label, token in enumerate(label_ids)}
                    completed = []
                    for row, question in enumerate(selected):
                        gold = question["label"]
                        if type(gold) is not int or not 0 <= gold < 3:
                            raise ValueError("Question label is not an absolute occupancy class")
                        values = candidates[row].tolist()
                        token = int(unconstrained[row])
                        forced = int(candidates[row].argmax())
                        predicted = inverse.get(token)
                        completed.append({**_metadata(question),
                                          "prefix_tokens": len(encoded[row][0]),
                                          "label_token_ids": list(label_ids),
                                          "label_log_probs": values,
                                          "off_label_mass": float(off_mass[row]),
                                          "forced_choice_label": forced,
                                          "unconstrained_token_id": token,
                                          "unconstrained_label": predicted,
                                          "forced_choice_correct": forced == gold,
                                          "unconstrained_label_correct": predicted == gold,
                                          "gold_value_nll": -values[gold]})
                    rows.extend(completed)
                    del logits, log_probs, candidates, off_mass, unconstrained
                except MeasurementDeadline:
                    break
    finally:
        model.train(was_training)
    return rows


def verify_batch_equivalence(model, tokenizer, questions: list[dict], features_by_position,
                             deadline: float | None = None,
                             tolerance: float = 5e-5) -> dict:
    """Select two different prefix lengths and compare batch versus singles."""
    chosen, lengths = [], set()
    for question in questions:
        length = len(encode_prefix(tokenizer, question["prompt"])[0])
        if length not in lengths:
            chosen.append(question)
            lengths.add(length)
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise ValueError("Equivalence check needs different prefix lengths")
    batched = score_questions(model, tokenizer, chosen, features_by_position, 2, deadline)
    singles = score_questions(model, tokenizer, chosen, features_by_position, 1, deadline)
    if len(batched) != 2 or len(singles) != 2:
        raise MeasurementDeadline("Batch-equivalence check did not finish")
    delta = max(abs(a - b) for x, y in zip(batched, singles)
                for a, b in zip(x["label_log_probs"], y["label_log_probs"]))
    mass_delta = max(abs(x["off_label_mass"] - y["off_label_mass"])
                     for x, y in zip(batched, singles))
    predictions_match = all(x["forced_choice_label"] == y["forced_choice_label"]
                            and x["unconstrained_token_id"] == y["unconstrained_token_id"]
                            for x, y in zip(batched, singles))
    return {"question_ids": [q["question_id"] for q in chosen],
            "prefix_lengths": [len(encode_prefix(tokenizer, q["prompt"])[0]) for q in chosen],
            "max_label_log_prob_abs_difference": delta,
            "max_off_label_mass_abs_difference": mass_delta,
            "prediction_argmaxes_match": predictions_match,
            "tolerance": tolerance, "within_tolerance": delta <= tolerance and mass_delta <= tolerance and predictions_match,
            "numeric_policy": "FP32 batch-shape rounding tolerance, with identical full and forced argmax required; original saved mean CE has a separate 1e-5 guard.",
            "left_padding": True, "explicit_cumulative_position_ids": True,
            "forward_question_count": 4}


def summarize_rows(rows: list[dict]) -> dict:
    """Token-level diagnostic summary; never a generated JSON accuracy."""
    def aggregate(selected):
        n = len(selected)
        with_mass = [r for r in selected if "off_label_mass" in r]
        with_unconstrained = [r for r in selected if "unconstrained_label" in r]
        forced = sum(r["forced_choice_label"] == r["label"] for r in selected)
        unconstrained = sum(r["unconstrained_label"] == r["label"] for r in with_unconstrained)
        return {"questions": n, "forced_choice_correct": forced,
                "forced_choice_accuracy": forced / n if n else None,
                "unconstrained_label_correct": unconstrained if with_unconstrained else None,
                "unconstrained_label_accuracy": unconstrained / len(with_unconstrained) if with_unconstrained else None,
                "off_label_argmax_count": sum(r["unconstrained_label"] is None for r in with_unconstrained) if with_unconstrained else None,
                "mean_off_label_mass": sum(r["off_label_mass"] for r in with_mass) / len(with_mass) if with_mass else None,
                "max_off_label_mass": max((r["off_label_mass"] for r in with_mass), default=None),
                "value_token_ce": -sum(r["label_log_probs"][r["label"]] for r in selected) / n if n else None,
                "label_counts": {name: sum(r["label"] == i for r in selected) for i, name in enumerate(LABEL_NAMES)},
                "forced_choice_prediction_counts": {name: sum(r["forced_choice_label"] == i for r in selected) for i, name in enumerate(LABEL_NAMES)}}
    return {"scope": "Causal next-value-token scoring after a supplied JSON prefix; no generation. Forced choice is restricted to three candidate tokens; unconstrained label accuracy uses the full-vocabulary argmax.",
            "label_order": list(LABEL_NAMES), "all": aggregate(rows),
            "by_label": {name: aggregate([r for r in rows if r["label"] == i]) for i, name in enumerate(LABEL_NAMES)},
            "by_game": {game: aggregate([r for r in rows if r.get("game_id") == game])
                        for game in sorted({r["game_id"] for r in rows if "game_id" in r})}}


def _contrast_summary(values: list[float], tolerance: float) -> dict:
    n = len(values)
    positive = sum(v > tolerance for v in values)
    negative = sum(v < -tolerance for v in values)
    ties = n - positive - negative
    return {"count": n, "mean_dd": sum(values) / n if n else None,
            "positive": positive, "negative": negative, "ties": ties,
            "positive_rate_ties_half": (positive + .5 * ties) / n if n else None}


def counterfactual_metrics(data: dict, rows: list[dict], tie_tolerance: float = TIE_TOLERANCE) -> dict:
    """Descriptive same-ply board DD and reciprocal-coordinate interactions.

    Each changed-label pair contributes one DD. All available Cartesian
    reciprocal coordinate pairs are used without choosing favorable examples.
    These observations share three games and do not yield independent trials.
    """
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("Tie tolerance must be finite and nonnegative")
    lookup = {}
    for row in rows:
        key = (row["position_id"], row["point_index"])
        if key in lookup:
            raise ValueError("Duplicate board/coordinate scores")
        if len(row["label_log_probs"]) != 3 or not all(math.isfinite(v) for v in row["label_log_probs"]):
            raise ValueError("Expected three finite log probabilities")
        lookup[key] = row
    groups, records, missing, unchanged = defaultdict(list), [], 0, 0
    for pair in data["validation_pairs"]:
        if not pair["label_changed"]:
            unchanged += 1
            continue
        a = lookup.get((pair["position_a"], pair["point_index"]))
        b = lookup.get((pair["position_b"], pair["point_index"]))
        if a is None or b is None:
            missing += 1
            continue
        la, lb = pair["label_a"], pair["label_b"]
        if a["label"] != la or b["label"] != lb:
            raise ValueError("Score labels disagree with fixed pair labels")
        pa, pb = a["label_log_probs"], b["label_log_probs"]
        dd = (pa[la] - pa[lb]) - (pb[la] - pb[lb])
        stratum = "stone_empty" if 0 in (la, lb) else "black_white"
        record = {"pair_id": pair["pair_id"], "pair_question_id": pair["pair_question_id"],
                  "ply": pair["ply"], "point_index": pair["point_index"],
                  "label_a": la, "label_b": lb, "stratum": stratum, "dd": dd}
        records.append(record)
        groups[pair["pair_id"]].append(record)
    reciprocal, reciprocal_groups = [], []
    for group_id, selected in sorted(groups.items()):
        for lo, hi in itertools.combinations(range(3), 2):
            forward = [r for r in selected if (r["label_a"], r["label_b"]) == (lo, hi)]
            reverse = [r for r in selected if (r["label_a"], r["label_b"]) == (hi, lo)]
            # reverse.dd is oriented towards its reversed correct labels, so
            # summing is C(q)-C(r) for a fixed lo-versus-hi contrast C.
            values = [q["dd"] + r["dd"] for q in forward for r in reverse]
            if values:
                stratum = "stone_empty" if lo == 0 else "black_white"
                reciprocal.extend((stratum, v) for v in values)
                reciprocal_groups.append({"pair_id": group_id, "labels": [LABEL_NAMES[lo], LABEL_NAMES[hi]],
                                          "forward_coordinates": len(forward), "reverse_coordinates": len(reverse),
                                          "stratum": stratum, **_contrast_summary(values, tie_tolerance)})
    return {"scope": "Descriptive diagnostic on previously inspected validation games. Counts reuse the same three games and board-pair groups; no confidence interval, threshold, causal proof, or formal M1 verdict is asserted.",
            "label_order": list(LABEL_NAMES), "tie_tolerance": tie_tolerance,
            "tie_rule": "DD within [-tolerance,+tolerance] receives one half in the positive rate",
            "dd_formula": "[logP_A(label_A|q)-logP_A(label_B|q)]-[logP_B(label_A|q)-logP_B(label_B|q)]",
            "pair_selection": "Fixed same-ply, same-coordinate, label-changing pairs from the manifest; both questions must have completed scores",
            "requested_changed_pairs": sum(p["label_changed"] for p in data["validation_pairs"]),
            "missing_changed_pairs": missing, "unchanged_pairs_excluded": unchanged,
            "all": _contrast_summary([r["dd"] for r in records], tie_tolerance),
            "by_stratum": {s: _contrast_summary([r["dd"] for r in records if r["stratum"] == s], tie_tolerance)
                           for s in ("stone_empty", "black_white")},
            "by_board_pair_group": [{"pair_id": group, "ply": selected[0]["ply"],
                                     **_contrast_summary([r["dd"] for r in selected], tie_tolerance)}
                                    for group, selected in sorted(groups.items())],
            "pair_rows": records,
            "reciprocal_coordinate_interaction": {
                "formula": "C(q)-C(r), C(x)=[logP_A(lo|x)-logP_A(hi|x)]-[logP_B(lo|x)-logP_B(hi|x)]; q labels=(lo,hi), r labels=(hi,lo)",
                "scope": "Subtracts a coordinate-invariant board-global label bias. Uses all fixed Cartesian reciprocal coordinate pairs within each same-ply board pair and unordered label pair. This restricted reversing-label selection differs from the overall DD sample; it does not remove arbitrary coordinate-dependent confounding or establish causality.",
                "all": _contrast_summary([v for _, v in reciprocal], tie_tolerance),
                "by_stratum": {s: _contrast_summary([v for k, v in reciprocal if k == s], tie_tolerance)
                               for s in ("stone_empty", "black_white")},
                "by_group_and_label_pair": reciprocal_groups}}


def empirical_prior_baselines(data: dict, questions: list[dict], smoothing: float = 1.0) -> dict:
    """Training-pool-only global majority and coordinate-by-side priors."""
    if not math.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Smoothing must be finite and positive")
    training = data["training_pool_questions"]
    if not training or any(q["split"] != "train" for q in training):
        raise ValueError("Prior must be fitted only to the declared training pool")
    counts = Counter(q["label"] for q in training)
    cells = defaultdict(Counter)
    for q in training:
        cells[(q["point_index"], q["to_play"])][q["label"]] += 1
    outputs = {}
    for name in ("train_majority", "coordinate_side_empirical_prior"):
        rows = []
        for q in questions:
            observed = counts if name == "train_majority" else cells.get((q["point_index"], q["to_play"]), Counter())
            denominator = sum(observed.values()) + 3 * smoothing
            probabilities = [(observed[label] + smoothing) / denominator for label in range(3)]
            rows.append({**_metadata(q), "label_log_probs": [math.log(p) for p in probabilities],
                         "forced_choice_label": max(range(3), key=lambda label: probabilities[label])})
        outputs[name] = {"summary": summarize_rows(rows),
                         "counterfactual": counterfactual_metrics(data, rows)}
    return {"scope": "Categorical label priors, not language-model vocabulary probabilities; no unconstrained-token or off-label metric applies. Both fit only the declared training pool, with no validation labels used for fitting. Ties choose empty, then black, then white. DD is zero because these priors do not depend on board identity.",
            "training_questions": len(training), "smoothing_per_class": smoothing,
            "training_label_counts": {name: counts[i] for i, name in enumerate(LABEL_NAMES)},
            "majority_label": LABEL_NAMES[max(range(3), key=lambda label: counts[label])],
            "coordinate_side_cells_observed": len(cells),
            "evaluation_questions": len(questions), **outputs}
