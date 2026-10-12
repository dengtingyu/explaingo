"""Pure numerical diagnostics for small board-by-query occupancy score grids.

Standard library only. The functions consume already-computed label
log-probabilities for every cell of a complete board-by-query grid. They do not
train, tokenize, select data, apply thresholds or make pass/fail decisions.

Conventions
-----------
* Labels are integers 0, 1, 2 in ``LABEL_ORDER`` (empty, black, white).
* ``scores[b][q]`` holds the three label log-probabilities of board ``b`` and
  query ``q``. Full-vocabulary log-probabilities (total label mass <= 1) and
  three-label-normalized log-probabilities (mass == 1) are both accepted.
* ``lambda`` is the label-centred log-probability: each cell minus its own
  three-label mean. It removes any per-cell constant, so every interaction
  statistic below is independent of the cell's normalization; cross-entropy
  and off-label mass are not.
* RMS values average over all boards x queries x 3 labels.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import itertools
import math
from typing import Any

LABEL_ORDER = ("empty", "black", "white")
LN2 = math.log(2.0)
LN3 = math.log(3.0)


class GridError(ValueError):
    """The grid is malformed, partial or non-finite; nothing is inferred or filled in."""


def _sequence(value: Any, name: str) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise GridError(f"{name} must be a sequence")
    return list(value)


def _real(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GridError(f"{where}: expected a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise GridError(f"{where}: non-finite value")
    return number


def _identifiers(value: Any, count: int, name: str) -> list[str]:
    ids = _sequence(value, name)
    if len(ids) != count:
        raise GridError(f"{name} has {len(ids)} entries for {count} grid lines")
    if any(not isinstance(item, str) or not item for item in ids):
        raise GridError(f"{name} must contain non-empty strings")
    if len(set(ids)) != len(ids):
        raise GridError(f"{name} must be unique")
    return ids


def _validated(scores, labels, board_ids, query_ids, mass_tolerance):
    tolerance = _real(mass_tolerance, "mass_tolerance")
    if tolerance < 0:
        raise GridError("mass_tolerance must be non-negative")
    score_rows, label_rows = _sequence(scores, "scores"), _sequence(labels, "labels")
    if len(score_rows) < 2:
        raise GridError("At least two boards are required")
    if len(label_rows) != len(score_rows):
        raise GridError("labels and scores have different board counts")
    width, raw, lab = None, [], []
    largest_log_prob = math.log1p(tolerance)
    for b, (score_row, label_row) in enumerate(zip(score_rows, label_rows)):
        score_row = _sequence(score_row, f"scores[{b}]")
        label_row = _sequence(label_row, f"labels[{b}]")
        width = len(score_row) if width is None else width
        if len(score_row) != width or len(label_row) != width:
            raise GridError("The grid is ragged; every board needs every query")
        values_row, labels_row = [], []
        for q, (cell, label) in enumerate(zip(score_row, label_row)):
            where = f"cell[{b}][{q}]"
            if isinstance(cell, (str, bytes)) or not isinstance(cell, Sequence) or len(cell) != 3:
                raise GridError(f"{where}: expected three label log-probabilities")
            values = tuple(_real(v, where) for v in cell)
            if max(values) > largest_log_prob:
                raise GridError(f"{where}: a label probability exceeds 1; supply log-probabilities")
            mass = math.fsum(math.exp(v) for v in values)
            if mass > 1.0 + tolerance:
                raise GridError(f"{where}: label probability mass {mass!r} exceeds 1 + tolerance")
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < 3:
                raise GridError(f"{where}: label must be an int 0, 1 or 2")
            values_row.append(values)
            labels_row.append(label)
        raw.append(values_row)
        lab.append(labels_row)
    if width < 2:
        raise GridError("At least two queries are required")
    return (len(raw), width, raw, lab, _identifiers(board_ids, len(raw), "board_ids"),
            _identifiers(query_ids, width, "query_ids"))


def _double_centre(table, boards, queries):
    """Two-way decomposition per label: table = mean + board + query + interaction."""
    mean = [math.fsum(table[b][q][y] for b in range(boards) for q in range(queries)) / (boards * queries)
            for y in range(3)]
    board = [[math.fsum(table[b][q][y] for q in range(queries)) / queries - mean[y] for y in range(3)]
             for b in range(boards)]
    query = [[math.fsum(table[b][q][y] for b in range(boards)) / boards - mean[y] for y in range(3)]
             for q in range(queries)]
    interaction = [[[table[b][q][y] - mean[y] - board[b][y] - query[q][y] for y in range(3)]
                    for q in range(queries)] for b in range(boards)]
    return board, query, interaction


def _rms(values: list[float]) -> float:
    return math.sqrt(math.fsum(v * v for v in values) / len(values))


def _logsumexp(values) -> float:
    top = max(values)
    return top + math.log(math.fsum(math.exp(v - top) for v in values))


def _additive_reference(lab, boards, queries) -> dict[str, Any]:
    cells = boards * queries
    columns = list(zip(*lab))
    if (boards == 3 and queries == 3 and all(sorted(row) == [0, 1, 2] for row in lab)
            and all(sorted(column) == [0, 1, 2] for column in columns)):
        return {"structure": "latin_3x3", "cells": cells, "accuracy_ceiling_cells": 7,
                "three_label_ce_infimum": LN3, "infimum_attained": True,
                "basis": "Every label occurs once per board and per query. Additive scores give "
                         "contradictory pairwise sums (>=2 wrong cells; 7 is reachable), and the "
                         "summed true-label score equals the summed cell mean, so mean CE >= ln 3 "
                         "with equality only for uniform cells."}
    if boards == 2:
        transitions = [(lab[0][q], lab[1][q]) for q in range(queries)]
        counts = Counter(transitions)
        if (all(a != b for a, b in transitions)
                and all(counts[(lo, hi)] == counts[(hi, lo)] for lo, hi in itertools.combinations(range(3), 2))):
            pairs = queries // 2
            return {"structure": "two_board_reciprocal_matched", "cells": cells,
                    "accuracy_ceiling_cells": cells - pairs, "three_label_ce_infimum": LN2,
                    "infimum_attained": False,
                    "basis": "Every query changes label and the queries pair into reversed transitions. "
                             "Each pair forces one wrong cell (one is reachable per pair). Restricting each "
                             "cell to its pair's two labels and applying convexity gives CE >= ln 2 per "
                             "cell, approached only as the third label's probability tends to zero."}
    return {"structure": "other", "cells": cells, "accuracy_ceiling_cells": None,
            "three_label_ce_infimum": None, "infimum_attained": None,
            "basis": "No additive-model ceiling or CE infimum is derived for this label structure."}


def analyze_grid(scores, labels, board_ids, query_ids, *, mass_tolerance: float = 1e-6) -> dict[str, Any]:
    """Describe one complete grid; inputs are copied and never modified.

    * ``G``: RMS of the board main effect of lambda.
    * ``T``: RMS of the query main effect.
    * ``D``: RMS of lambda minus its across-board mean at each query and label (total
      board dependence). For a complete grid ``D**2 == G**2 + I_RMS**2``.
    * ``I_RMS``: RMS of the double-centred interaction; zero for any additive model
      ``f(query, label) + g(board, label) + h(board, query)``.
    * ``S``: signed mean interaction margin toward the true label in nats; zero for
      additive models, negative for wrong-direction interaction.
    * ``S_cosine``: cosine between the interaction and the double-centred true-label
      pattern; ``None`` when either norm is zero, meaningless at rounding-level I_RMS.
    * ``reciprocal``: per board pair, changed-label DD and every reversed-label query
      combination ``R = C(forward) - C(reverse)``,
      ``C(x) = [l_A(lo|x) - l_A(hi|x)] - [l_B(lo|x) - l_B(hi|x)]``.
    """
    boards, queries, raw, lab, board_names, query_names = _validated(
        scores, labels, board_ids, query_ids, mass_tolerance)
    lam = [[[v - math.fsum(cell) / 3 for v in cell] for cell in row] for row in raw]
    board_effect, query_effect, interaction = _double_centre(lam, boards, queries)
    across_board = [[math.fsum(lam[b][q][y] for b in range(boards)) / boards for y in range(3)]
                    for q in range(queries)]
    flat = lambda table: [table[b][q][y] for b in range(boards) for q in range(queries) for y in range(3)]
    g_value = _rms([board_effect[b][y] for b in range(boards) for y in range(3)])
    t_value = _rms([query_effect[q][y] for q in range(queries) for y in range(3)])
    d_value = _rms([lam[b][q][y] - across_board[q][y]
                    for b in range(boards) for q in range(queries) for y in range(3)])
    i_values = flat(interaction)
    i_rms = _rms(i_values)

    target = [[[(1.0 if y == lab[b][q] else 0.0) - 1.0 / 3 for y in range(3)]
               for q in range(queries)] for b in range(boards)]
    _, _, target_interaction = _double_centre(target, boards, queries)
    t_flat = flat(target_interaction)
    dot = math.fsum(a * b for a, b in zip(i_values, t_flat))
    i_norm = math.sqrt(math.fsum(v * v for v in i_values))
    t_norm = math.sqrt(math.fsum(v * v for v in t_flat))
    s_terms = [interaction[b][q][lab[b][q]]
               - 0.5 * math.fsum(interaction[b][q][y] for y in range(3) if y != lab[b][q])
               for b in range(boards) for q in range(queries)]

    cells = []
    for b in range(boards):
        for q in range(queries):
            values, label = raw[b][q], lab[b][q]
            normalizer = _logsumexp(values)
            top = max(values)
            forced = values.index(top)
            margin = values[label] - max(values[y] for y in range(3) if y != label)
            cells.append({
                "board_id": board_names[b], "query_id": query_names[q],
                "label": label, "label_name": LABEL_ORDER[label],
                "log_probs": list(values),
                "three_label_log_probs": [v - normalizer for v in values],
                "lambda": list(lam[b][q]), "interaction": list(interaction[b][q]),
                "full_vocabulary_ce": -values[label],
                "three_label_ce": normalizer - values[label],
                "margin": margin, "forced_choice_label": forced,
                "forced_choice_tie": values.count(top) > 1,
                "forced_choice_correct": forced == label,
                "off_label_mass": 1.0 - math.fsum(math.exp(v) for v in values)})

    n = len(cells)
    margins = [cell["margin"] for cell in cells]
    correct_margins = [cell["margin"] for cell in cells if cell["forced_choice_correct"]]
    off_mass = [cell["off_label_mass"] for cell in cells]
    per_label = {}
    for label, name in enumerate(LABEL_ORDER):
        selected = [cell for cell in cells if cell["label"] == label]
        per_label[name] = {
            "cells": len(selected),
            "forced_choice_correct": sum(cell["forced_choice_correct"] for cell in selected),
            "mean_three_label_ce": (math.fsum(cell["three_label_ce"] for cell in selected) / len(selected)
                                    if selected else None)}
    summary = {
        "cells": n,
        "forced_choice_correct": sum(cell["forced_choice_correct"] for cell in cells),
        "strict_margin_correct": sum(m > 0 for m in margins),
        "forced_choice_ties": sum(cell["forced_choice_tie"] for cell in cells),
        "mean_full_vocabulary_ce": math.fsum(cell["full_vocabulary_ce"] for cell in cells) / n,
        "mean_three_label_ce": math.fsum(cell["three_label_ce"] for cell in cells) / n,
        "margins_sorted": sorted(margins),
        "min_margin": min(margins),
        "min_margin_among_forced_correct": min(correct_margins) if correct_margins else None,
        "mean_off_label_mass": math.fsum(off_mass) / n,
        "min_off_label_mass": min(off_mass), "max_off_label_mass": max(off_mass),
        "per_label": per_label}

    reciprocal = []
    for i, j in itertools.combinations(range(boards), 2):
        contrast = lambda q, lo, hi: ((raw[i][q][lo] - raw[i][q][hi]) - (raw[j][q][lo] - raw[j][q][hi]))
        changed = [{"query_id": query_names[q], "labels": [lab[i][q], lab[j][q]],
                    "dd": contrast(q, lab[i][q], lab[j][q])}
                   for q in range(queries) if lab[i][q] != lab[j][q]]
        combinations = []
        for lo, hi in itertools.combinations(range(3), 2):
            forward = [q for q in range(queries) if (lab[i][q], lab[j][q]) == (lo, hi)]
            reverse = [q for q in range(queries) if (lab[i][q], lab[j][q]) == (hi, lo)]
            for qf, qr in itertools.product(forward, reverse):
                c_forward, c_reverse = contrast(qf, lo, hi), contrast(qr, lo, hi)
                combinations.append({"labels": [LABEL_ORDER[lo], LABEL_ORDER[hi]],
                                     "forward_query_id": query_names[qf], "reverse_query_id": query_names[qr],
                                     "c_forward": c_forward, "c_reverse": c_reverse,
                                     "R": c_forward - c_reverse})
        values = [item["R"] for item in combinations]
        reciprocal.append({
            "board_ids": [board_names[i], board_names[j]], "changed_label_dd": changed,
            "reversed_label_combinations": combinations,
            "R_count": len(values),
            "R_mean": math.fsum(values) / len(values) if values else None,
            "R_min": min(values) if values else None, "R_max": max(values) if values else None,
            "R_positive": sum(v > 0 for v in values), "R_negative": sum(v < 0 for v in values),
            "R_zero": sum(v == 0 for v in values)})

    reference = _additive_reference(lab, boards, queries)
    ceiling, infimum = reference["accuracy_ceiling_cells"], reference["three_label_ce_infimum"]
    reference.update({
        "observed_forced_choice_correct_minus_ceiling":
            summary["forced_choice_correct"] - ceiling if ceiling is not None else None,
        "observed_mean_three_label_ce_minus_infimum":
            summary["mean_three_label_ce"] - infimum if infimum is not None else None,
        "observed_mean_full_vocabulary_ce_minus_infimum":
            summary["mean_full_vocabulary_ce"] - infimum if infimum is not None else None})

    return {
        "shape": {"boards": boards, "queries": queries, "cells": n},
        "board_ids": board_names, "query_ids": query_names, "label_order": list(LABEL_ORDER),
        "normalization": {
            "lambda": "cell log-probabilities minus their three-label mean",
            "rms": "mean of squares over boards x queries x 3 labels, then square root",
            "full_vocabulary_ce": "-log_prob[true label] as supplied",
            "three_label_ce": "logsumexp(three log_probs) - log_prob[true label]",
            "off_label_mass": "1 - sum(exp(log_probs)), unclamped",
            "forced_choice": "first index of the maximum supplied log-probability; exact ties flagged",
            "mass_tolerance": float(mass_tolerance)},
        "cells": cells, "summary": summary,
        "decomposition": {
            "G": g_value, "T": t_value, "D": d_value, "I_RMS": i_rms,
            "max_abs_interaction": max(abs(v) for v in i_values),
            "D_squared_minus_G_squared_plus_I_squared": d_value ** 2 - (g_value ** 2 + i_rms ** 2),
            "S": math.fsum(s_terms) / len(s_terms),
            "S_cosine": dot / (i_norm * t_norm) if i_norm > 0 and t_norm > 0 else None,
            "true_label_pattern_interaction_rms": _rms(t_flat)},
        "reciprocal": reciprocal, "additive_reference": reference}


def combine_disjoint(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pool counts and additive references of grids sharing no board or query.

    An additive model then splits into independent per-grid parts, so ceilings
    add and the CE infimum is the cell-weighted mean. Grids sharing a board
    identity or an exact query prompt are rejected rather than pooled.
    """
    results = _sequence(results, "results")
    if not results:
        raise GridError("Nothing to combine")
    boards, queries = [], []
    for result in results:
        boards.extend(result["board_ids"])
        queries.extend(result["query_ids"])
    if len(set(boards)) != len(boards) or len(set(queries)) != len(queries):
        raise GridError("Grids share a board or query identity; additive parts are not separable")
    cells = sum(result["shape"]["cells"] for result in results)
    correct = sum(result["summary"]["forced_choice_correct"] for result in results)
    references = [result["additive_reference"] for result in results]
    known = all(reference["accuracy_ceiling_cells"] is not None for reference in references)
    weighted = lambda key: math.fsum(result["summary"][key] * result["shape"]["cells"] for result in results) / cells
    infimum = (math.fsum(reference["three_label_ce_infimum"] * reference["cells"] for reference in references) / cells
               if known else None)
    ceiling = sum(reference["accuracy_ceiling_cells"] for reference in references) if known else None
    return {"grids": len(results), "cells": cells, "forced_choice_correct": correct,
            "mean_three_label_ce": weighted("mean_three_label_ce"),
            "mean_full_vocabulary_ce": weighted("mean_full_vocabulary_ce"),
            "accuracy_ceiling_cells": ceiling, "three_label_ce_infimum": infimum,
            "forced_choice_correct_minus_ceiling": correct - ceiling if known else None,
            "mean_three_label_ce_minus_infimum": weighted("mean_three_label_ce") - infimum if known else None}
