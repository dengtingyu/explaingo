"""Coordinate-only APS routing at each actual scoring/training batch boundary.

Reuse the established prefix, feature and score handling without patching its
module globals. This wrapper does not supply occupancy labels to the model.
"""
from __future__ import annotations

import time

import torch

from . import grounding_eval as original
from .grounding_data import occupancy_prompt, point_to_gtp


def point_indices_for(selected, device):
    """Validate canonical coordinate inputs without inspecting answer fields."""
    points = []
    for question in selected:
        index = question.get("point_index")
        if type(index) is not int or not 0 <= index < 361:
            raise ValueError("APS point_index must be an integer in [0,360]")
        point = point_to_gtp(index)
        if question.get("point") != point:
            raise ValueError("APS coordinate and point_index disagree")
        player = question.get("to_play")
        if player not in ("B", "W"):
            raise ValueError("APS player must be B or W")
        if question.get("prompt") != occupancy_prompt(player, point):
            raise ValueError("APS prompt is not the canonical coordinate input")
        points.append(index)
    if not points:
        raise ValueError("APS requires a nonempty actual batch")
    return torch.tensor(points, dtype=torch.long, device=device)


class _RoutedBatch:
    """Forward a fixed actual batch to APS; delegate mode/device APIs to it."""
    def __init__(self, model, selected):
        self.model = model
        self.points = point_indices_for(selected, next(model.parameters()).device)

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, **inputs):
        if "point_indices" in inputs:
            raise ValueError("APS routing may only be supplied by this batch wrapper")
        if inputs["input_ids"].shape[0] != len(self.points):
            raise ValueError("APS actual forward batch does not match routed points")
        return self.model(point_indices=self.points, **inputs)


def _forward_logits(model, tokenizer, prefixes, selected, features_by_position):
    if len(prefixes) != len(selected):
        raise ValueError("APS prefixes and selected questions have different lengths")
    # Caller-provided cached prefixes must also correspond to the routed text.
    for prefix, question in zip(prefixes, selected):
        expected, _ = original.encode_prefix(tokenizer, question["prompt"])
        if not torch.equal(prefix.detach().cpu(), expected):
            raise ValueError("APS prefix does not match the routed canonical prompt")
    routed = _RoutedBatch(model, selected)
    return original._forward_logits(routed, tokenizer, prefixes, selected, features_by_position)


def score_questions(model, tokenizer, questions, features_by_position,
                    batch_size=4, deadline=None):
    """Bind exactly one selected chunk for each original scorer invocation.

    The original scorer's internal forward receives the routed facade. Its
    batch size equals this chunk's length, so it cannot reuse a larger point
    vector for a smaller inner batch. APS clears all temporary binding state
    in its own forward, including on exceptions.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Batch size must be positive")
    rows = []
    for start in range(0, len(questions), batch_size):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        selected = questions[start:start + batch_size]
        routed = _RoutedBatch(model, selected)
        completed = original.score_questions(
            routed, tokenizer, selected, features_by_position,
            batch_size=len(selected), deadline=deadline)
        rows.extend(completed)
        if len(completed) != len(selected):
            break
    return rows
