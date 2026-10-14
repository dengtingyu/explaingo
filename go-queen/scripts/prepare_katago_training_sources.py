#!/usr/bin/env python3
"""Select pinned public SGFs and unique occupancy questions; never load a model.

Only original simple-ko, area-scoring, no-tax, no-suicide, no-button rules are
accepted. SGF metadata and replay are separate from expert feature extraction.
Imports are inert; sgfmill and the vendor Board are imported only when needed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_BASE = "https://katagoarchive.org/kata1/traininggames/"
INDEX_URL = ARCHIVE_BASE + "index.html"
FIRST_SHA = "395371e7a2a7ceef129ce2fcba47c74d411c6674b97bb8f32bc8c7f21bb6708e"
DEFAULT_PLIES = (20, 39, 60, 79, 100, 119)
LABEL_NAMES = ("empty", "black", "white")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def parse_rules_exact(raw: str, komi: float) -> dict:
    """Reject unsupported fields instead of silently dropping SGF rule suffixes."""
    if type(komi) not in (int, float) or not math.isfinite(komi):
        raise ValueError("missing_or_nonfinite_komi")
    if not isinstance(raw, str) or raw not in ("koSIMPLEscoreAREAtaxNONEsui0", "Chinese", "chinese"):
        raise ValueError("unsupported_original_rules")
    return {"koRule": "KO_SIMPLE", "scoringRule": "SCORING_AREA", "taxRule": "TAX_NONE",
            "multiStoneSuicideLegal": False, "hasButton": False, "encorePhase": 0,
            "passWouldEndPhase": False, "whiteKomi": float(komi), "asymPowersOfTwo": 0.0}


def parse_game_bytes(raw: bytes, min_moves: int = 120, *, require_supported_rules: bool = True) -> dict:
    """Return normalized B/W, top-left row-major moves and exact original rules.

    The rule-agnostic option is only for duplicate exclusion of old project games;
    those records are never candidates for the new dataset.
    """
    from sgfmill import sgf
    game = sgf.Sgf_game.from_bytes(raw)
    root = game.get_root()
    if game.get_size() != 19:
        raise ValueError("not_19x19")
    if any(root.get_setup_stones()) or (root.has_property("HA") and root.get("HA") not in (None, 0)):
        raise ValueError("setup_or_handicap")
    if root.has_property("PL") and root.get("PL") != "b":
        raise ValueError("nondefault_initial_player")
    original_rules = root.get("RU") if root.has_property("RU") else None
    komi = root.get("KM") if root.has_property("KM") else None
    rules = parse_rules_exact(original_rules, komi) if require_supported_rules else None
    moves = []
    for offset, node in enumerate(game.get_main_sequence()):
        if offset and any(node.get_setup_stones()):
            raise ValueError("setup_after_root")
        color, point = node.get_move()
        if color is None:
            continue
        player = color.upper()
        if player != ("B" if len(moves) % 2 == 0 else "W"):
            raise ValueError("nonalternating_moves")
        index = None if point is None else (18 - point[0]) * 19 + point[1]
        moves.append((player, index))
    if len(moves) < min_moves:
        raise ValueError("too_few_moves")
    return {"size": 19, "original_rules": original_rules, "komi": komi,
            "rules": rules, "moves": moves, "move_count": len(moves)}


def transformed_point(point: int | None, symmetry: int) -> int | None:
    if point is None:
        return None
    if type(point) is not int or not 0 <= point < 361 or not 0 <= symmetry < 8:
        raise ValueError("invalid_point_or_symmetry")
    y, x = divmod(point, 19)
    if symmetry >= 4:
        x = 18 - x
    for _ in range(symmetry % 4):
        x, y = 18 - y, x
    return y * 19 + x


def canonical_moves(moves, limit: int | None = None) -> str:
    selected = moves if limit is None else moves[:limit]
    return min(digest(canonical_bytes([[p, transformed_point(i, s)] for p, i in selected])) for s in range(8))


def load_board_class():
    vendor = str(ROOT / "vendor/KataGo/python")
    if vendor not in sys.path:
        sys.path.insert(0, vendor)
    from katago.game.board import Board
    return Board


def replay_game(parsed: dict, selected_plies) -> list[dict]:
    """Replay every recorded move under the supported rules, retaining <=6 states."""
    plies = tuple(selected_plies)
    if (not plies or len(plies) > 6 or len(set(plies)) != len(plies)
            or any(type(p) is not int or not 0 < p <= len(parsed["moves"]) for p in plies)):
        raise ValueError("invalid_selected_plies")
    expected = parse_rules_exact(parsed["original_rules"], parsed["komi"])
    if parsed["rules"] != expected:
        raise ValueError("parsed_rules_changed")
    Board = load_board_class()
    board = Board(19)
    positions = []
    consecutive_passes = 0
    for ply, (player, point) in enumerate(parsed["moves"], 1):
        if consecutive_passes >= 2:
            raise ValueError("moves_after_two_passes")
        color = Board.BLACK if player == "B" else Board.WHITE
        if color != board.pla:
            raise ValueError("replay_wrong_side")
        loc = Board.PASS_LOC if point is None else board.loc(point % 19, point // 19)
        if point is not None and board.would_be_suicide(color, loc):
            raise ValueError("forbidden_suicide")
        board.play(color, loc)
        consecutive_passes = consecutive_passes + 1 if point is None else 0
        if ply in plies:
            if consecutive_passes >= 2:
                raise ValueError("selected_position_after_two_passes")
            labels = [int(board.board[board.loc(x, y)]) for y in range(19) for x in range(19)]
            if not set(labels).issubset({0, 1, 2}):
                raise ValueError("invalid_board_label")
            positions.append({"ply": ply, "to_play": "B" if board.pla == Board.BLACK else "W",
                              "board_labels361": labels, "board_sha256": digest(bytes(labels)),
                              "pass_would_end_phase": consecutive_passes == 1})
    return sorted(positions, key=lambda p: p["ply"])


def game_split(game_key: str, seed: int = 42) -> str:
    """A stable whole-game split, unchanged when more archives are appended."""
    return "val" if int(digest(f"{seed}:whole-game:{game_key}".encode())[:16], 16) % 10 == 0 else "train"


def balanced_points(labels, seed: int, identity: str) -> list[int]:
    if len(labels) != 361 or not set(labels).issubset({0, 1, 2}):
        raise ValueError("invalid_board_labels")
    groups = [[i for i, label in enumerate(labels) if label == cls] for cls in range(3)]
    n = min(map(len, groups))
    for cls, group in enumerate(groups):
        random.Random(int(digest(f"{seed}:{identity}:{cls}".encode()), 16)).shuffle(group)
    return [groups[cls][j] for j in range(n) for cls in range(3)]


def coordinate(index: int) -> str:
    y, x = divmod(index, 19)
    return "ABCDEFGHJKLMNOPQRST"[x] + str(19 - y)


def question_prompt(to_play: str, point: int, komi: float) -> str:
    # Describe only supported actual rules; do not relabel the record as Chinese.
    side = "黑方" if to_play == "B" else "白方"
    return (f"19路围棋，面积计分，简单劫，无税，禁止自杀，无按钮，贴{komi:g}目，当前行棋方：{side}。"
            f"请判断{coordinate(point)}点的占用状态。只输出JSON，格式为{{\"occupancy\":\"black|white|empty\"}}。")


def old_duplicate_sets() -> tuple[set, dict, list]:
    full, prefixes, evidence = set(), {}, []
    for name in ("data/pilot-sources.json", "data/expanded-sources.json"):
        manifest = ROOT / name
        content = json.loads(manifest.read_text())
        for row in content["games"]:
            path = (ROOT / row["file"]).resolve()
            if not path.is_relative_to(ROOT) or path.is_symlink():
                raise ValueError("invalid_old_game_path")
            raw = path.read_bytes()
            if digest(raw) != row["sgf_sha256"]:
                raise ValueError("old_game_source_changed")
            parsed = parse_game_bytes(raw, min_moves=0, require_supported_rules=False)
            full.add(canonical_moves(parsed["moves"]))
            limit = min(120, len(parsed["moves"]))
            prefixes.setdefault(limit, set()).add(canonical_moves(parsed["moves"], limit))
            evidence.append({"file": row["file"], "sgf_sha256": digest(raw), "prefix_plies": limit})
    return full, prefixes, evidence


def download_archive(url: str, target: Path, index_text: str, bytes_left: int) -> None:
    if url != ARCHIVE_BASE + target.name or target.name not in index_text:
        raise ValueError("archive_not_in_official_index")
    if target.exists():
        return
    temporary = target.with_suffix(target.suffix + ".part")
    if temporary.exists():
        raise FileExistsError("Preserve previous partial download")
    try:
        # Standard verified-HTTPS curl works with this public archive's server;
        # Python urllib is rejected by the server. No credentials or TLS bypass.
        subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                        "--proto", "=https", "--proto-redir", "=https", "--max-time", "180",
                        "--max-filesize", str(bytes_left), "--output", str(temporary), url], check=True)
        if temporary.stat().st_size > bytes_left:
            raise ValueError("download_exceeds_total_byte_cap")
        temporary.replace(target)
    except Exception:
        # A nonempty partial is evidence and is never silently reused.
        raise


def archive_payloads(path: Path) -> list[tuple[str, bytes]]:
    """Read compressed members once in archive order, then sort in memory.

    Sorting TarInfo objects before extractfile causes quadratic decompression
    on bzip2 archives. The bounded payload copy avoids that repeated seeking.
    """
    payloads, total = [], 0
    with tarfile.open(path, "r:bz2") as handle:
        for member in handle:
            if not member.isfile() or not member.name.endswith(".sgf"):
                continue
            if member.size > 1_000_000 or PurePosixPath(member.name).is_absolute() or ".." in PurePosixPath(member.name).parts:
                raise ValueError("unsafe_or_oversized_archive_member")
            raw = handle.extractfile(member).read()
            total += len(raw)
            if total > 256_000_000:
                raise ValueError("archive_expanded_payload_limit")
            payloads.append((member.name, raw))
    return sorted(payloads)


def select_dataset(archive_records: list[dict], plies=DEFAULT_PLIES, seed=42, *, state=None) -> dict:
    # The optional state makes a bounded date-by-date preparation incremental.
    # Previously inspected archives are not reparsed to decide when to stop.
    if state is None:
        state = {}
    if not state:
        full_seen, old_prefixes, old_evidence = old_duplicate_sets()
        state.update(full_seen=full_seen, old_prefixes=old_prefixes, old_evidence=old_evidence,
                     prefix_seen=set(old_prefixes.get(120, set())), boards_seen=set(), questions_seen=set(),
                     rejection_counts=Counter(), rules_inventory=Counter(), games=[], positions=[],
                     questions=[], source_rejections=[], archives_seen=set())
    full_seen, old_prefixes, old_evidence = (state[k] for k in ("full_seen", "old_prefixes", "old_evidence"))
    prefix_seen, boards_seen, questions_seen = (state[k] for k in ("prefix_seen", "boards_seen", "questions_seen"))
    rejection_counts, rules_inventory = state["rejection_counts"], state["rules_inventory"]
    games, positions, questions, source_rejections = (state[k] for k in ("games", "positions", "questions", "source_rejections"))
    for archive in archive_records:
        if archive["sha256"] in state["archives_seen"]:
            raise ValueError("archive_already_inspected")
        state["archives_seen"].add(archive["sha256"])
        path = ROOT / archive["file"]
        if digest(path.read_bytes()) != archive["sha256"]:
            raise ValueError("archive_digest_changed")
        for member_name, raw in archive_payloads(path):
            try:
                parsed = parse_game_bytes(raw, max(120, max(plies)))
                rules_inventory[parsed["original_rules"]] += 1
                full = canonical_moves(parsed["moves"])
                prefix = canonical_moves(parsed["moves"], 120)
                if full in full_seen:
                    raise ValueError("duplicate_full_game_d4_or_old_game")
                if prefix in prefix_seen or any(canonical_moves(parsed["moves"], length) in keys for length, keys in old_prefixes.items()):
                    raise ValueError("duplicate_long_prefix_d4_or_old_game")
                board_rows = replay_game(parsed, plies)
                keys = [(r["board_sha256"], r["to_play"]) for r in board_rows]
                if len(set(keys)) != len(keys) or any(k in boards_seen for k in keys):
                    raise ValueError("duplicate_selected_board_state_side")
                if any(not balanced_points(r["board_labels361"], seed, full+str(r["ply"])) for r in board_rows):
                    raise ValueError("selected_position_missing_class")
            except Exception as exc:
                reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                # Retain bounded machine-readable reasons, never arbitrary SGF comments.
                if not re.fullmatch(r"[A-Za-z0-9_]{1,100}", reason):
                    reason = type(exc).__name__
                rejection_counts[reason] += 1
                source_rejections.append({"archive": archive["file"], "member": member_name,
                                          "sgf_sha256": digest(raw), "reason": reason})
                continue
            full_seen.add(full); prefix_seen.add(prefix); boards_seen.update(keys)
            game_id = "katago-public-" + full[:20]
            split = game_split(full, seed)
            game = {"game_id": game_id, "split": split,
                    "file": "", "sgf_sha256": digest(raw), "archive": archive["file"],
                    "archive_sha256": archive["sha256"], "archive_member": member_name,
                    "selected_plies": list(plies), "original_rules": parsed["original_rules"],
                    "komi": parsed["komi"], "rules": parsed["rules"],
                    "move_count": parsed["move_count"], "full_replay_legal": True,
                    "d4_game_sha256": full, "d4_prefix120_sha256": prefix}
            games.append(game)
            for row in board_rows:
                position_id = f"{game_id}-ply-{row['ply']:03d}"
                position = {**row, "position_id": position_id, "game_id": game_id, "split": split,
                            "komi": parsed["komi"], "original_rules": parsed["original_rules"], "rules": parsed["rules"]}
                positions.append(position)
                for point in balanced_points(row["board_labels361"], seed, full+str(row["ply"])):
                    key = (row["board_sha256"], row["to_play"], point)
                    if key in questions_seen:
                        raise ValueError("question_identity_repeated")
                    questions_seen.add(key)
                    label = row["board_labels361"][point]
                    questions.append({"question_id": digest(canonical_bytes(key)),
                                      "position_id": position_id, "game_id": game_id, "split": split,
                                      "board_sha256": row["board_sha256"], "to_play": row["to_play"],
                                      "point_index": point, "coordinate": coordinate(point),
                                      "label": label, "label_name": LABEL_NAMES[label],
                                      "prompt": question_prompt(row["to_play"], point, parsed["komi"]),
                                      "answer": '{"occupancy":"'+LABEL_NAMES[label]+'"}'})
    counts = {"games": dict(Counter(g["split"] for g in games)),
              "positions": dict(Counter(p["split"] for p in positions)),
              "questions": dict(Counter(q["split"] for q in questions)),
              "labels": {s: dict(Counter(q["label_name"] for q in questions if q["split"] == s)) for s in ("train", "val")}}
    return {"games": games, "positions": positions, "questions": questions, "counts": counts,
            "rejection_counts": dict(rejection_counts), "rejections": source_rejections,
            "accepted_rules_inventory": dict(rules_inventory), "old_game_exclusion_sources": old_evidence}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/katago-public-20261013")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/katago-training-sources-20261013")
    parser.add_argument("--first-date", default="2020-12-08")
    parser.add_argument("--max-archives", type=int, default=20)
    parser.add_argument("--max-total-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--min-train-questions", type=int, default=100_000)
    parser.add_argument("--min-val-questions", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--index-record", type=Path,
                        help="Optional preserved root-observed official index link record; not claimed raw HTML")
    args = parser.parse_args(argv)
    if not (1 <= args.max_archives <= 20 and 0 < args.max_total_bytes <= 1_000_000_000
            and args.min_train_questions >= 100_000 and args.min_val_questions > 0):
        raise ValueError("invalid_fixed_limits")
    relative(args.directory); relative(args.output)
    if args.output.exists():
        raise FileExistsError("Choose a fresh output; existing preparation is preserved")
    args.directory.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    index_bytes, index_evidence_kind, index_evidence_sha = None, None, None
    if args.allow_download:
        if args.index_record:
            index_record_bytes = args.index_record.read_bytes()
            index_record = json.loads(index_record_bytes)
            if (index_record.get("index_url") != INDEX_URL
                    or index_record.get("kind") != "root_observed_official_index_transcription"
                    or not isinstance(index_record.get("entries"), list)):
                raise ValueError("invalid_official_index_record")
            for entry in index_record["entries"]:
                if entry["url"] != ARCHIVE_BASE + entry["name"] or not re.fullmatch(r"2020-12-\d{2}sgfs\.tar\.bz2", entry["name"]):
                    raise ValueError("invalid_index_record_link")
            index_bytes = "\n".join(e["name"] for e in index_record["entries"]).encode()
            index_evidence_kind = index_record["kind"]
            index_evidence_sha = digest(index_record_bytes)
        else:
            with urllib.request.urlopen(INDEX_URL, timeout=60) as response:
                index_bytes = response.read(10_000_001)
            index_evidence_kind = "downloaded_official_index_html"
            index_evidence_sha = digest(index_bytes)
        if len(index_bytes) > 10_000_000:
            raise ValueError("unexpected_index_size")
    records, used_bytes, dataset, selection_state = [], 0, None, {}
    first = date.fromisoformat(args.first_date)
    for offset in range(args.max_archives):
        name = str(first + timedelta(days=offset)) + "sgfs.tar.bz2"
        path = args.directory / name
        if not path.exists():
            if not args.allow_download:
                break
            download_archive(ARCHIVE_BASE+name, path, index_bytes.decode(), args.max_total_bytes-used_bytes)
        raw = path.read_bytes(); used_bytes += len(raw)
        if used_bytes > args.max_total_bytes:
            raise ValueError("existing_archives_exceed_byte_limit")
        hashed = digest(raw)
        if name == "2020-12-08sgfs.tar.bz2" and hashed != FIRST_SHA:
            raise ValueError("first_archive_changed")
        records.append({"file": relative(path), "url": ARCHIVE_BASE+name,
                        "sha256": hashed, "bytes": len(raw)})
        dataset = select_dataset([records[-1]], seed=args.seed, state=selection_state)
        print(json.dumps({"archives":len(records), "bytes":used_bytes, "counts":dataset["counts"]},ensure_ascii=False),flush=True)
        if (dataset["counts"]["questions"].get("train",0) >= args.min_train_questions
                and dataset["counts"]["questions"].get("val",0) >= args.min_val_questions):
            break
    if dataset is None:
        raise ValueError("no_available_archives")
    args.output.mkdir()
    selected_dir = args.output / "sgf"; selected_dir.mkdir()
    lookup = {(g["archive"], g["archive_member"]): g for g in dataset["games"]}
    for archive in records:
        with tarfile.open(ROOT/archive["file"],"r:bz2") as handle:
            for member in handle:
                game = lookup.get((archive["file"],member.name))
                if game is None:
                    continue
                raw = handle.extractfile(member).read()
                if digest(raw) != game["sgf_sha256"]:
                    raise ValueError("selected_sgf_changed")
                path = selected_dir / (game["game_id"]+".sgf")
                path.write_bytes(raw); game["file"] = relative(path)
    questions_path = args.output / "questions.jsonl"
    with questions_path.open("x") as stream:
        for row in dataset.pop("questions"):
            stream.write(json.dumps(row,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
    enough = (dataset["counts"]["questions"].get("train",0) >= args.min_train_questions
              and dataset["counts"]["questions"].get("val",0) >= args.min_val_questions)
    manifest = {"format_version":1,"stage":"katago_public_occupancy_preparation","status":"ready" if enough else "insufficient_data",
                "model_execution":False,"seed":args.seed,"archives":records,
                "index_url":INDEX_URL,"index_evidence_kind":index_evidence_kind,"index_evidence_sha256":index_evidence_sha,
                "source_readme_url":"https://katagoarchive.org/kata1/README.txt",
                "source_disclaimer_url":"https://katagoarchive.org/kata1/DISCLAIMER.txt",
                "selected_plies":list(DEFAULT_PLIES),"requirements":{"min_unique_training_questions":args.min_train_questions,
                "min_separate_validation_questions":args.min_val_questions,"max_archives":args.max_archives,"max_archive_bytes":args.max_total_bytes},
                "questions_file":relative(questions_path),"questions_sha256":digest(questions_path.read_bytes()),
                "selection":"date-ordered bounded archives; full/D4 and 120-ply prefix exclusion; seed42 hash whole-game split; per-position balanced distinct points",
                "question_identity":"(absolute board SHA256, side to play, point index), globally unique across splits",
                "split_note":"Approximately 10% whole-game hash split; self-play source, not human-game or blind-evaluation claim",
                "limits":["Only original simple-ko, area, no-tax, no-suicide, no-button rules; original finite komi retained",
                          "At most six recorded positions per game; no expert features or Qwen executed",
                          "No claim of endgame, broad-rules, independent-question or human-play representativeness"],
                "producer_sha256":digest(Path(__file__).read_bytes()),"elapsed_seconds":time.monotonic()-start,**dataset}
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    (args.output/"script-at-run.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status":manifest["status"],"manifest":relative(args.output/"manifest.json"),"counts":manifest["counts"],"seconds":manifest["elapsed_seconds"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
