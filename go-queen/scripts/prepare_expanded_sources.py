#!/usr/bin/env python3
"""Download and select small, pinned public Pachi SGFs without model inference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess

from sgfmill import sgf

from build_batch_data import (
    ROOT, Board, canonical_game_sha256, file_sha256,
    load_sgf_moves_exn, prepare_games, project_file,
)

REPOSITORY = "https://github.com/pasky/pachi"
REVISION = "617986f4570d56174cf3759392e02d7af9f58fa6"
TREE_URL = f"https://api.github.com/repos/pasky/pachi/git/trees/{REVISION}?recursive=1"
RAW_BASE = f"https://raw.githubusercontent.com/pasky/pachi/{REVISION}/"


def public_bytes(url: str) -> bytes:
    return subprocess.check_output([
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--retry", "2", "--max-time", "60", url,
    ])


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def d4_hashes(setup: list, moves: list, limit: int | None = None) -> list[str]:
    """Hash all eight square-board transforms; retain color and move ordering."""
    board = Board(19)
    hashes = []
    for reflected in (False, True):
        for turns in range(4):
            def transformed(pla: int, loc: int) -> tuple[int, int]:
                if loc == Board.PASS_LOC:
                    return pla, loc
                x, y = board.loc_x(loc), board.loc_y(loc)
                if reflected:
                    x = 18 - x
                for _ in range(turns):
                    x, y = 18 - y, x
                return pla, board.loc(x, y)

            rotated_setup = [transformed(pla, loc) for pla, loc in setup]
            rotated_moves = [transformed(pla, loc) for pla, loc in moves[:limit]]
            hashes.append(canonical_game_sha256(19, rotated_setup, rotated_moves))
    return hashes


def download(entry: dict, directory: Path) -> dict:
    source_path = entry["path"]
    path = directory / "upstream" / source_path
    raw_url = RAW_BASE + source_path
    if path.exists() and git_blob_sha1(path.read_bytes()) == entry["sha"]:
        data = path.read_bytes()
    else:
        data = public_bytes(raw_url)
        if len(data) != entry["size"] or git_blob_sha1(data) != entry["sha"]:
            raise ValueError(f"Pinned Git blob mismatch: {source_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {
        "source_path": source_path,
        "file": project_file(path),
        "raw_url": raw_url,
        "sgf_sha256": hashlib.sha256(data).hexdigest(),
        "git_blob_sha1": entry["sha"],
        "bytes": len(data),
    }


def game_id(source_path: str) -> str:
    stem = re.sub(r"[^a-z0-9]+", "-", Path(source_path).stem.lower()).strip("-")
    suffix = hashlib.sha256(source_path.encode()).hexdigest()[:8]
    return f"pachi-expanded-{stem}-{suffix}"


def selected_public_metadata(root) -> dict:
    """Read game headers only; never copy comments, chat, venue or contact fields."""
    players = {color: root.get(prop) if root.has_property(prop) else None
               for color, prop in (("B", "PB"), ("W", "PW"))}
    counterpart = [value for value in players.values()
                   if value and not str(value).casefold().startswith("pachi")]
    opponent = str(counterpart[0]).casefold() if len(counterpart) == 1 else (
        "pair:" + "|".join(sorted(str(value).casefold() for value in players.values() if value))
    )
    # Infer only a declared engine flag; never save any SGF comment text.
    comment = root.get("C") if root.has_property("C") else ""
    declared_engine = (len(counterpart) == 1 and
                       re.search(r"GTP Engine for " + re.escape(str(counterpart[0])) + r"\s*\(",
                                 comment, flags=re.IGNORECASE) is not None)
    return {
        "players": players,
        "opponent_group": opponent,
        "opponent_kind": "declared_gtp_engine" if declared_engine else "unverified_account_type",
        "date": root.get("DT") if root.has_property("DT") else None,
    }


def replay_candidate(record: dict, pilot_anchor: dict) -> dict:
    # prepare_games loads no checkpoint; the distinct, known-good anchor satisfies
    # its >=2-game split contract while checking every candidate's actual moves.
    prepared = prepare_games({"games": [record, pilot_anchor]}, seed=42, validation_games=1)
    item = next(game for game in prepared if game["metadata"]["game_id"] == record["game_id"])
    return item["metadata"]


def proposed_group_split(selected: list[dict], requested: int, seed: int) -> dict:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in selected:
        grouped[record["opponent_group"]].append(record["game_id"])
    groups = sorted(grouped)
    random.Random(seed).shuffle(groups)
    # Subset sum chooses entire opponent groups and reserves >=1 training game.
    choices: dict[int, list[str]] = {0: []}
    for group in groups:
        for count, used in list(choices.items()):
            count += len(grouped[group])
            if count < len(selected) and count not in choices:
                choices[count] = used + [group]
    nonempty = [count for count in choices if count > 0]
    if not nonempty:
        return {"status": "insufficient_opponent_groups", "requested_holdout_games": requested}
    actual = min(nonempty, key=lambda count: (abs(count - requested), count > requested, count))
    validation_groups = set(choices[actual])
    validation_ids = sorted(record["game_id"] for record in selected
                            if record["opponent_group"] in validation_groups)
    train_ids = sorted(record["game_id"] for record in selected
                      if record["opponent_group"] not in validation_groups)
    return {
        "status": "assigned_in_source_manifest_requires_builder_support",
        "method": f"Keep every non-Pachi public opponent account group wholly in one split; choose complete groups closest to {requested} holdout games with seed {seed}",
        "seed": seed,
        "requested_holdout_games": requested,
        "actual_holdout_games": actual,
        "train_game_ids": train_ids,
        "validation_game_ids": validation_ids,
        "validation_opponent_groups": sorted(validation_groups),
        "train_games": len(train_ids),
        "validation_games": len(validation_ids),
        "opponent_groups_disjoint": True,
        "limitation": "Public account identity does not establish human identity or account aliases; this is opponent-account grouping",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-games", type=int, default=24)
    parser.add_argument("--minimum-games", type=int, default=20)
    parser.add_argument("--minimum-moves", type=int, default=74)
    parser.add_argument("--validation-games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/expanded-sgf")
    parser.add_argument("--output", type=Path, default=ROOT / "data/expanded-sources.json")
    args = parser.parse_args()
    if not 2 <= args.minimum_games <= args.target_games or args.minimum_moves < 1:
        raise ValueError("Require target >= minimum >=2, and a positive minimum move count")
    if not 1 <= args.workers <= 8:
        raise ValueError("Choose 1-8 small-file download workers")
    pilot = json.loads((ROOT / "data/pilot-sources.json").read_text())
    pilot_shas = {game["sgf_sha256"] for game in pilot["games"]}
    pilot_groups = set()
    pilot_prefixes: dict[int, set[str]] = defaultdict(set)
    for game in pilot["games"]:
        _, setup, moves, _ = load_sgf_moves_exn(str(ROOT / game["file"]))
        pilot_groups.add(min(d4_hashes(setup, moves)))
        prefix_length = min(160, len(moves))
        pilot_prefixes[prefix_length].add(min(d4_hashes(setup, moves, prefix_length)))
    tree = json.loads(public_bytes(TREE_URL))
    if tree.get("sha") != REVISION or tree.get("truncated"):
        raise ValueError("The pinned public Git tree is incomplete or mismatched")
    entries = sorted((entry for entry in tree["tree"]
                      if entry["type"] == "blob" and entry["path"].endswith(".sgf")),
                     key=lambda entry: entry["path"])
    if any(entry["size"] > 1_000_000 for entry in entries):
        raise ValueError("Unexpectedly large SGF; this script downloads only small public files")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        downloaded = list(pool.map(lambda entry: download(entry, args.directory), entries))
    eligible = []
    rejected = []

    def reject(record: dict, reason: str, **metadata) -> None:
        rejected.append({**record, "reason": reason, **metadata})

    for base in downloaded:
        if base["sgf_sha256"] in pilot_shas:
            reject(base, "pilot_original_file_excluded")
            continue
        if not base["source_path"].startswith("t-regress/old/games/"):
            reject(base, "not_historical_full_game_directory")
            continue
        path = ROOT / base["file"]
        try:
            root = sgf.Sgf_game.from_bytes(path.read_bytes()).get_root()
        except Exception as exc:
            reject(base, "not_parseable_sgf", error_class=type(exc).__name__)
            continue
        try:
            original_rules = root.get("RU") if root.has_property("RU") else None
            if original_rules is None or str(original_rules).casefold() != "chinese":
                reject(base, "missing_or_non_chinese_original_rules")
                continue
            if not root.has_property("SZ") or root.get("SZ") != 19:
                reject(base, "missing_or_non_19x19_board_size")
                continue
            if not root.has_property("KM") or root.get("KM") != 7.5:
                reject(base, "missing_or_non_7_5_komi")
                continue
            metadata, setup, moves, _ = load_sgf_moves_exn(str(path))
            if setup or metadata.handicap not in (None, 0):
                reject(base, "setup_or_handicap_game", setup_stones=len(setup))
                continue
            count = len(moves)
            if count < args.minimum_moves:
                reject(base, "fewer_than_minimum_moves", move_count=count)
                continue
            hashes = d4_hashes(setup, moves)
            d4_group = min(hashes)
            if d4_group in pilot_groups:
                reject(base, "pilot_game_d4_variant_excluded", move_count=count)
                continue
            if any(min(d4_hashes(setup, moves, length)) in values
                   for length, values in pilot_prefixes.items()):
                reject(base, "pilot_first_moves_d4_variant_excluded", move_count=count)
                continue
            record = {
                **base, "game_id": game_id(base["source_path"]), "move_count": count,
                "original_rules": original_rules, "original_komi": metadata.komi,
                "rules_normalized": False, "canonical_game_sha256": canonical_game_sha256(19, setup, moves),
                "d4_game_group_sha256": d4_group, "d4_variant_sha256": hashes,
                "d4_game_sha256": d4_group,
                "d4_prefix160_sha256": min(d4_hashes(setup, moves, 160)),
                "d4_prefix_length": min(160, len(moves)),
                **selected_public_metadata(root),
            }
            checked = replay_candidate(record, pilot["games"][0])
            record["full_replay_legal"] = checked["full_replay_legal"]
            eligible.append(record)
        except Exception as exc:
            reject(base, "invalid_sgf_or_illegal_full_replay", error_class=type(exc).__name__)
    # Prefer the original historical full-game record to derivative regression files.
    eligible.sort(key=lambda record: (not record["source_path"].startswith("t-regress/old/games/"),
                                      record["date"] or "", record["source_path"]))
    unique = []
    groups = {}
    prefixes = {}
    for record in eligible:
        group = record["d4_game_group_sha256"]
        prefix = record["d4_prefix160_sha256"]
        if group in groups:
            reject(record, "duplicate_full_game_under_d4", duplicate_of=groups[group])
        elif prefix in prefixes:
            reject(record, "same_first_160_moves_under_d4", duplicate_of=prefixes[prefix])
        else:
            groups[group] = record["game_id"]
            prefixes[prefix] = record["game_id"]
            unique.append(record)
    by_opponent = defaultdict(list)
    for record in unique:
        by_opponent[record["opponent_group"]].append(record)
    opponent_order = sorted(by_opponent)
    random.Random(args.seed).shuffle(opponent_order)
    selected = []
    rank = 0
    while len(selected) < args.target_games:
        current = [by_opponent[opponent][rank] for opponent in opponent_order
                   if len(by_opponent[opponent]) > rank]
        if not current:
            break
        selected.extend(current[:args.target_games - len(selected)])
        rank += 1
    selected.sort(key=lambda record: (record["date"] or "", record["source_path"]))
    selected_ids = {record["game_id"] for record in selected}
    for record in unique:
        if record["game_id"] not in selected_ids:
            reject(record, "eligible_but_target_cap_reached")
    status = "selected_target_reached" if len(selected) >= args.minimum_games else "source_exhausted_below_minimum"
    split_proposal = proposed_group_split(selected, args.validation_games, args.seed)
    validation_ids = set(split_proposal.get("validation_game_ids", []))
    for record in selected:
        record["split"] = "val" if record["game_id"] in validation_ids else "train"
    manifest = {
        "format_version": 1,
        "source_type": "Public historical full KGS game records from the official Pachi author repository; opponents can be human or bot accounts",
        "repository": REPOSITORY,
        "revision": REVISION,
        "tree_url": TREE_URL,
        "selection_note": "No source or original rules were substituted. Raw public SGFs are retained locally; selection metadata contains no chat, comments, venue or contact fields.",
        "games": selected,
        "selection_report": {
            "status": status,
            "target_games": args.target_games,
            "minimum_games": args.minimum_games,
            "minimum_moves": args.minimum_moves,
            "inventory_sgf_files": len(entries),
            "downloaded_bytes": sum(entry["size"] for entry in entries),
            "eligible_before_deduplication": len(eligible),
            "unique_eligible_games": len(unique),
            "selected_games": len(selected),
            "shortfall_to_minimum": max(0, args.minimum_games - len(selected)),
            "selected_total_moves": sum(record["move_count"] for record in selected),
            "unique_opponent_groups": len({record["opponent_group"] for record in selected}),
            "pilot_games_excluded": [game["game_id"] for game in pilot["games"]],
            "rejection_counts": dict(sorted(Counter(record["reason"] for record in rejected).items())),
            "rejected": sorted(rejected, key=lambda record: record["source_path"]),
            "deduplication": "Canonical full-game SHA256; minimum of all eight D4 full-game hashes; additionally reject a shared up-to-160-move D4 prefix as a possible derivative/truncated record",
            "checks": {"all_selected_games_replayed_legally": True,
                       "selected_full_game_d4_groups_unique": True,
                       "selected_up_to_160_move_d4_prefix_groups_unique": True,
                       "selected_rules_normalized": False,
                       "no_expert_checkpoint_loaded": True},
        },
        "proposed_split": split_proposal,
        "split_method": "explicit_game_split_by_non_pachi_opponent_account_group",
        "builder_compatibility": "games entries retain game_id/file/sgf_sha256 and add explicit split=train|val. A builder supporting source-provided splits should honor these fields; the original random per-game splitter does not.",
        "limitations": ["This small regression-source collection is not a representative professional-game corpus",
                        "Opponent account names do not determine human/bot identity or account aliases",
                        "Conservative 160-move prefix deduplication can merge genuinely different games sharing a long opening",
                        "No features, teacher labels, language-model training or cloud jobs were produced"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    summary = {key: value for key, value in manifest["selection_report"].items()
               if key not in ("rejected", "checks")}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(manifest["proposed_split"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
