"""Deterministic occupancy diagnostic questions from saved board input planes.

This module does not load models, extract features or select training settings
from validation performance. Labels are absolute colours in top-row-first order.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

SIZE = 19
COLUMNS = "ABCDEFGHJKLMNOPQRST"
LABEL_NAMES = ("empty", "black", "white")
SEED = 42
MAX_UPDATES = 1000
BATCH_SIZE = 4
HELD_BACK_NUMERATOR = 1
HELD_BACK_DENOMINATOR = 5


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_sha256(value: Any) -> str:
    """Hash JSON content independently of pretty-printing and output location."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def point_to_gtp(index: int) -> str:
    if type(index) is not int or not 0 <= index < SIZE * SIZE:
        raise ValueError("Point index must be an integer in [0, 360]")
    return f"{COLUMNS[index % SIZE]}{SIZE - index // SIZE}"


def gtp_to_index(point: str) -> int:
    if not isinstance(point, str) or len(point) not in (2, 3):
        raise ValueError("Invalid 19x19 coordinate")
    try:
        column = COLUMNS.index(point[0])
        row = int(point[1:])
    except ValueError as error:
        raise ValueError("Invalid 19x19 coordinate") from error
    if not 1 <= row <= SIZE or point != f"{COLUMNS[column]}{row}":
        raise ValueError("Coordinate must be canonical GTP")
    return (SIZE - row) * SIZE + column


def occupancy_prompt(to_play: str, point: str) -> str:
    """The exact point_occupancy template used by rule_qa.question_examples."""
    if to_play not in ("B", "W"):
        raise ValueError("Unknown side to play")
    gtp_to_index(point)
    player = "黑方" if to_play == "B" else "白方"
    return (f"19路围棋，中国面积计分，简单劫，禁止自杀，贴7.5目，当前行棋方：{player}。"
            f"根据棋盘特征，{point}交叉点的占用状态是什么？只输出JSON，字段occupancy取black、white或empty。")


def _rng(namespace: str) -> random.Random:
    # Separate deterministic streams keep choices independent of list lengths.
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{namespace}".encode()).digest(), "big")
    return random.Random(seed)


def _project_relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _absolute_labels(feature_path: Path, to_play: str) -> list[int]:
    with np.load(feature_path, allow_pickle=False) as saved:
        spatial = saved["input_spatial"]
        if spatial.shape != (1, 22, SIZE, SIZE):
            raise ValueError("Unexpected saved input plane shape")
        own, other = spatial[0, 1].reshape(-1), spatial[0, 2].reshape(-1)
        if not np.isfinite(own).all() or not np.isfinite(other).all():
            raise ValueError("Non-finite occupancy input planes")
        if not np.isin(own, [0, 1]).all() or not np.isin(other, [0, 1]).all():
            raise ValueError("Occupancy planes must be binary")
        if np.any((own != 0) & (other != 0)):
            raise ValueError("Both players occupy the same point")
        own_colour = 1 if to_play == "B" else 2 if to_play == "W" else None
        if own_colour is None:
            raise ValueError("Unknown side to play")
        return np.where(own != 0, own_colour,
                        np.where(other != 0, 3 - own_colour, 0)).astype(int).tolist()


def _question(position: dict, index: int, source_tasks: list[dict],
              question_id: str | None = None) -> dict:
    point = point_to_gtp(index)
    label = position["absolute_board_labels"][index]
    return {"question_id": question_id or f"{position['position_id']}-occ-p{index:03d}",
            "position_id": position["position_id"], "game_id": position["game_id"],
            "split": position["split"], "ply": position["ply"],
            "to_play": position["to_play"], "point_index": index, "point": point,
            "label": label, "answer": {"occupancy": LABEL_NAMES[label]},
            "prompt": occupancy_prompt(position["to_play"], point),
            "source_task_ids": [task["task_id"] for task in source_tasks],
            "source_task_content_sha256": {task["task_id"]: content_sha256(task)
                                           for task in source_tasks}}


def _label_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["label"] for row in rows)
    return {name: counts[label] for label, name in enumerate(LABEL_NAMES)}


def build_manifest(source_path: Path, root: Path) -> dict:
    source = json.loads(source_path.read_text())
    if source.get("format_version") != 2:
        raise ValueError("Expected the established position/task manifest v2")
    # This bounded diagnostic intentionally retains the original split and plies.
    positions = copy.deepcopy(source["positions"])
    ids = [position["position_id"] for position in positions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source position IDs")
    game_sets = {split: {p["game_id"] for p in positions if p["split"] == split}
                 for split in ("train", "val")}
    if len(game_sets["train"]) != 5 or len(game_sets["val"]) != 3:
        raise ValueError("Diagnostic requires the existing five/three game split")
    if game_sets["train"] & game_sets["val"]:
        raise ValueError("Game split overlap")
    if any(p["split"] not in game_sets for p in positions):
        raise ValueError("Unknown source split")
    for position in positions:
        feature_path = source_path.parent / position["feature_file"]
        feature_sha = file_sha256(feature_path)
        if feature_sha != position["feature_sha256"]:
            raise ValueError("Saved feature file changed")
        position["source_feature_file"] = _project_relative(feature_path, root)
        position["absolute_board_labels"] = _absolute_labels(feature_path, position["to_play"])
        position["absolute_board_labels_sha256"] = content_sha256(position["absolute_board_labels"])
    position_by_id = {position["position_id"]: position for position in positions}
    occupancy_tasks = [task for task in source["tasks"]
                       if task["task_type"] == "point_occupancy"]
    if len(occupancy_tasks) != 144:
        raise ValueError("Expected all 144 established occupancy tasks")
    tasks_at_point = defaultdict(list)
    for task in occupancy_tasks:
        tasks_at_point[(task["position_id"], gtp_to_index(task["question"]["point"]))].append(task)
    original = []
    for task in occupancy_tasks:
        position = position_by_id[task["position_id"]]
        index = gtp_to_index(task["question"]["point"])
        row = _question(position, index, [task], question_id=task["task_id"])
        if row["answer"] != task["answer"] or row["prompt"] != task["prompt"]:
            raise ValueError("Established occupancy label/prompt disagrees with saved board")
        original.append(row)
    if any(gtp_to_index(point_to_gtp(index)) != index for index in range(SIZE * SIZE)):
        raise ValueError("GTP index roundtrip failed")

    training, heldback, holdback_counts = [], [], []
    for position in positions:
        if position["split"] != "train":
            continue
        selected = set()
        for label, name in enumerate(LABEL_NAMES):
            candidates = [i for i, value in enumerate(position["absolute_board_labels"])
                          if value == label]
            _rng(f"holdback:{position['position_id']}:{label}").shuffle(candidates)
            n = len(candidates) * HELD_BACK_NUMERATOR // HELD_BACK_DENOMINATOR
            selected.update(candidates[:n])
            holdback_counts.append({"position_id": position["position_id"], "label": name,
                                    "available": len(candidates), "heldback": n,
                                    "training_pool": len(candidates) - n,
                                    "heldback_fraction": n / len(candidates) if candidates else None})
        for index in range(SIZE * SIZE):
            row = _question(position, index, tasks_at_point[(position["position_id"], index)])
            (heldback if index in selected else training).append(row)

    validation, pairs, coordinate_groups = [], [], []
    by_ply = defaultdict(list)
    for position in positions:
        if position["split"] == "val":
            by_ply[position["ply"]].append(position)
    for ply, group in sorted(by_ply.items()):
        group = sorted(group, key=lambda p: p["game_id"])
        if len(group) != 3 or {p["game_id"] for p in group} != game_sets["val"]:
            raise ValueError("Each validation ply must contain all three held-out games")
        if len({p["to_play"] for p in group}) != 1:
            raise ValueError("Paired validation positions must share side to play")
        foreground = [i for i in range(SIZE * SIZE)
                      if any(p["absolute_board_labels"][i] != 0 for p in group)]
        background = [i for i in range(SIZE * SIZE)
                      if all(p["absolute_board_labels"][i] == 0 for p in group)]
        _rng(f"validation-background:{ply}").shuffle(background)
        selected_background = sorted(background[:min(len(foreground), len(background))])
        selected = sorted(foreground + selected_background)
        coordinate_groups.append({"ply": ply, "position_ids": [p["position_id"] for p in group],
                                  "foreground_point_indices": foreground,
                                  "background_point_indices": selected_background,
                                  "background_available": len(background),
                                  "background_requested": len(foreground),
                                  "background_selected": len(selected_background),
                                  "point_indices": selected})
        for position in group:
            for index in selected:
                validation.append(_question(position, index,
                                            tasks_at_point[(position["position_id"], index)]))
        for a, b in itertools.combinations(group, 2):
            pair_id = f"ply-{ply:03d}:{a['game_id']}:{b['game_id']}"
            for index in selected:
                label_a, label_b = a["absolute_board_labels"][index], b["absolute_board_labels"][index]
                pairs.append({"pair_id": pair_id, "pair_question_id": f"{pair_id}:p{index:03d}",
                              "ply": ply, "position_a": a["position_id"], "position_b": b["position_id"],
                              "point_index": index, "point": point_to_gtp(index),
                              "question_a": f"{a['position_id']}-occ-p{index:03d}",
                              "question_b": f"{b['position_id']}-occ-p{index:03d}",
                              "label_a": label_a, "label_b": label_b,
                              "label_changed": label_a != label_b,
                              "coordinate_group": "foreground" if index in foreground else "background"})

    # One duplicate class per batch rotates 0,1,2. Each batch is then shuffled.
    # Cycling separately shuffled per-class pools avoids validation-driven choices.
    label_pools = {label: [row["question_id"] for row in training if row["label"] == label]
                   for label in range(len(LABEL_NAMES))}
    if any(not pool for pool in label_pools.values()):
        raise ValueError("All three training-pool classes must be present")
    order_rng = _rng("train-order")
    cursors = {label: 0 for label in label_pools}
    for pool in label_pools.values():
        order_rng.shuffle(pool)
    train_order = []
    for update in range(MAX_UPDATES):
        batch_labels = [0, 1, 2, update % 3]
        order_rng.shuffle(batch_labels)
        for label in batch_labels:
            if cursors[label] == len(label_pools[label]):
                order_rng.shuffle(label_pools[label])
                cursors[label] = 0
            train_order.append(label_pools[label][cursors[label]])
            cursors[label] += 1

    training_ids = {row["question_id"] for row in training}
    heldback_ids = {row["question_id"] for row in heldback}
    if training_ids & heldback_ids or len(training) + len(heldback) != 30 * SIZE * SIZE:
        raise ValueError("Training point partition is invalid")
    if not set(train_order) <= training_ids:
        raise ValueError("Training order contains nontraining questions")
    validation_ids = {row["question_id"] for row in validation}
    if len(validation_ids) != len(validation) or len({p["pair_question_id"] for p in pairs}) != len(pairs):
        raise ValueError("Diagnostic row IDs are not unique")
    if any(p["question_a"] not in validation_ids or p["question_b"] not in validation_ids for p in pairs):
        raise ValueError("A validation pair references an absent question")
    train_rows = {row["question_id"]: row for row in training}
    for start in range(0, len(train_order), BATCH_SIZE):
        batch_counts = sorted(Counter(train_rows[q]["label"]
                                      for q in train_order[start:start + BATCH_SIZE]).values())
        if batch_counts != [1, 1, 2]:
            raise ValueError("Training-order batch is not class balanced")

    config = {"seed": SEED, "board_size": SIZE, "label_names": list(LABEL_NAMES),
              "token_order": "index=y*19+x; x=0 left, y=0 top; GTP skips I",
              "max_updates": MAX_UPDATES, "batch_size": BATCH_SIZE,
              "heldback_fraction": "1/5", "heldback_rounding": "floor(class_count/5)",
              "heldback_classes": "Every class with members, including the empty label; per board",
              "random_streams": "Python Random seeded by SHA256 of '42:<namespace>'; independent namespaces",
              "validation_selection": "All coordinates occupied on at least one board in each same-ply three-game group, plus an equal-size seeded sample empty on all three, capped to available background; ask all three boards at every selected coordinate",
              "validation_pairs": "All three unordered board pairs at each selected coordinate; stable board-pair group IDs",
              "train_order": "1000 batches of four: labels 0,1,2 and update%3, shuffled within each batch; per-class training-only pools shuffled and cycled",
              "prompt_template": occupancy_prompt("B", "A19").replace("黑方", "{player_name}").replace("A19", "{point}")}
    changed = sum(pair["label_changed"] for pair in pairs)
    pair_stats = []
    for pair_id in dict.fromkeys(pair["pair_id"] for pair in pairs):
        rows = [pair for pair in pairs if pair["pair_id"] == pair_id]
        transition_counts = Counter(f"{LABEL_NAMES[p['label_a']]}->{LABEL_NAMES[p['label_b']]}" for p in rows)
        pair_stats.append({"pair_id": pair_id, "ply": rows[0]["ply"], "coordinates": len(rows),
                           "changed_labels": sum(row["label_changed"] for row in rows),
                           "unchanged_labels": sum(not row["label_changed"] for row in rows),
                           "label_transitions": dict(sorted(transition_counts.items()))})
    lists = {"original_occupancy_questions": original, "training_pool_questions": training,
             "heldback_train_questions": heldback, "validation_questions": validation,
             "validation_pairs": pairs, "train_order": train_order,
             "validation_coordinate_groups": coordinate_groups}
    result = {"format_version": 1, "stage": "occupancy_coordinate_grounding_diagnostic",
              "scope": "Diagnostic only. Existing validation games have already been inspected; this is not a blind test or a formal M1 acceptance set.",
              "source": {"manifest": _project_relative(source_path, root),
                         "manifest_sha256": file_sha256(source_path),
                         "task_type": "point_occupancy",
                         "source_occupancy_tasks_sha256": content_sha256(occupancy_tasks),
                         "source_task_content_sha256": {task["task_id"]: content_sha256(task)
                                                        for task in occupancy_tasks}},
              "config": config, "games": copy.deepcopy(source["games"]),
              "positions": positions, **lists,
              "heldback_rounding_counts": holdback_counts,
              "validation_pair_stats": pair_stats,
              "counts": {"games": {split: len(games) for split, games in game_sets.items()},
                         "positions": {split: sum(p["split"] == split for p in positions)
                                       for split in game_sets},
                         "questions": {name: len(rows) for name, rows in lists.items()},
                         "label_counts": {name: _label_counts(rows) for name, rows in lists.items()
                                          if name.endswith("questions")},
                         "train_order_label_counts": _label_counts([train_rows[q] for q in train_order]),
                         "validation_board_pair_groups": len(pair_stats),
                         "validation_pair_label_changed": changed,
                         "validation_pair_label_unchanged": len(pairs) - changed},
              "hashes": {"config": content_sha256(config), "games": content_sha256(source["games"]),
                         "positions": content_sha256(positions),
                         "question_lists": {name: content_sha256(rows) for name, rows in lists.items()},
                         "heldback_rounding_counts": content_sha256(holdback_counts),
                         "validation_pair_stats": content_sha256(pair_stats)},
              "checks": {"old_occupancy_labels_and_prompts_checked": len(original),
                         "coordinate_roundtrips_checked": SIZE * SIZE,
                         "feature_files_hash_checked": len(positions),
                         "train_val_game_overlap": 0,
                         "training_pool_heldback_overlap": 0,
                         "train_order_samples_only_training_pool": True,
                         "every_train_batch_label_counts": [1, 1, 2],
                         "board_pair_questions_exist": True},
              "limitations": ["The eight-game source and early/midgame plies 8-95 are a small diagnostic sample, not a representative Go benchmark.",
                              "The validation boards and original occupancy tasks have been inspected in earlier stages; labels are available for auditing and must not enter model inputs.",
                              "Heldback training coordinates are excluded only from this new diagnostic training pool; earlier models may have seen original questions at some heldback coordinates.",
                              "Holding back coordinates on a training board does not withhold the board or game itself.",
                              "Background sampling is capped when fewer all-three-empty coordinates are available than union-occupied coordinates.",
                              "Absolute labels are reconstructed from saved current-player/opponent input planes and the source side to play; this stage does not perform a new independent SGF replay.",
                              "Question counts and sampled-coordinate distributions differ from the established 144-question diagnostic; compare model arms using the same list and train order."]}
    return result
