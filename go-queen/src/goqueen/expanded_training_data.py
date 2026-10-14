"""Validate an immutable, game-separated occupancy corpus without model imports."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


LABELS = ("empty", "black", "white")


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inside(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Expected a contained project-relative path")
    result = (Path(root) / relative).resolve()
    if not result.is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes project")
    return result


def load_expanded_corpus(project_root, source_manifest, feature_manifest, *, minimum_train=100_000):
    """Join saved questions to verified expert positions and enforce split identity.

    Feature payload hashes are checked by BoundedFeatureCache on use. This loader
    checks the source contract and metadata; it never decodes a tensor or model.
    """
    root = Path(project_root).resolve()
    source_path, feature_path = Path(source_manifest).resolve(), Path(feature_manifest).resolve()
    if not source_path.is_relative_to(root) or not feature_path.is_relative_to(root):
        raise ValueError("Manifests must reside inside the project")
    sources = json.loads(source_path.read_text())
    features = json.loads(feature_path.read_text())
    if sources.get("status") != "ready" or sources.get("stage") != "katago_public_occupancy_preparation":
        raise ValueError("Source corpus is not ready")
    if features.get("complete") is not True or features.get("status") != "complete":
        raise ValueError("Expert feature extraction is incomplete")
    if features["contract"]["source_manifest_sha256"] != file_sha256(source_path):
        raise ValueError("Features were built from a different source manifest")
    questions_path = inside(root, sources["questions_file"])
    if file_sha256(questions_path) != sources["questions_sha256"]:
        raise ValueError("Questions file changed")
    games, positions, feature_positions = {}, {}, {}
    for group, records, field in ((games, sources["games"], "game_id"),
                                  (positions, sources["positions"], "position_id"),
                                  (feature_positions, features["positions"], "position_id")):
        for record in records:
            key = record[field]
            if key in group:
                raise ValueError("Repeated source identity")
            group[key] = record
    if set(positions) != set(feature_positions):
        raise ValueError("Feature and source position sets differ")
    for key, position in positions.items():
        feature = feature_positions[key]
        if games[position["game_id"]]["split"] != position["split"]:
            raise ValueError("A game spans multiple data splits")
        for field in ("game_id", "split", "board_sha256", "to_play", "ply"):
            if position[field] != feature[field]:
                raise ValueError("Feature position metadata mismatch: " + field)
        if position["board_labels361"] != feature["absolute_board_labels"]:
            raise ValueError("Expert and question board labels differ")
    splits = {"train": [], "val": []}
    ids, contents = set(), set()
    with questions_path.open() as stream:
        for line in stream:
            q = json.loads(line)
            if q["question_id"] in ids:
                raise ValueError("Repeated question ID")
            ids.add(q["question_id"])
            p = positions[q["position_id"]]
            if q["split"] not in splits:
                raise ValueError("Unexpected question split")
            for field in ("game_id", "split", "board_sha256", "to_play"):
                if q[field] != p[field]:
                    raise ValueError("Question source mismatch: " + field)
            index = q["point_index"]
            if type(index) is not int or not 0 <= index < 361:
                raise ValueError("Invalid board coordinate")
            label = p["board_labels361"][index]
            if q["label"] != label or q["label_name"] != LABELS[label]:
                raise ValueError("Question label differs from the board")
            identity = (q["board_sha256"], q["to_play"], index)
            if identity in contents:
                raise ValueError("Duplicate actual question across corpus splits")
            contents.add(identity)
            expected_coordinate = "ABCDEFGHJKLMNOPQRST"[index % 19] + str(19-index//19)
            if q["coordinate"] != expected_coordinate or f"判断{expected_coordinate}点" not in q["prompt"]:
                raise ValueError("Prompt does not address the saved point")
            splits[q["split"]].append(q)
    if len(splits["train"]) < minimum_train or not splits["val"]:
        raise ValueError("Training minimum or independent validation not met")
    counts = {s: len(rows) for s, rows in splits.items()}
    if counts != sources["counts"]["questions"]:
        raise ValueError("Manifest question counts differ from the actual list")
    report = {"source_sha256": file_sha256(source_path), "feature_manifest_sha256": file_sha256(feature_path),
              "questions_sha256": file_sha256(questions_path), "counts": counts,
              "labels": {s: dict(Counter(q["label_name"] for q in rows)) for s, rows in splits.items()},
              "games": {s: len({q["game_id"] for q in rows}) for s, rows in splits.items()},
              "minimum_unique_train": minimum_train, "validation_in_training_count": False,
              "unique_input_identity_count": len(contents), "tensor_payload_decoded": False}
    return splits, features, report
