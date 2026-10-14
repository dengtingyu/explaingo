#!/usr/bin/env python3
"""Build a small, game-split 19x19 dataset with frozen KataGo NN targets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from sgfmill import sgf

# Reuse the verified extraction math and official KataGo imports unchanged.
from extract_features import (
    ROOT, NETWORK, Board, Features, GameState,
    board_tokens, file_sha256, gtp_coord, infer, load_model, load_sgf_moves_exn,
)

EXPERT_COMMIT = "d91ea855110dae533f0aada947b2b7d78cc8a4e1"
TARGET_SOURCE = "katago_nn_legal_policy_argmax"


def project_file(path: Path) -> str:
    """Record project paths without embedding the operator's home directory."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("Input files must be inside the project") from exc


def canonical_game_sha256(size: int, setup: list, moves: list) -> str:
    board = Board(size)

    def point(pla: int, loc: int) -> list:
        coord = None if loc == Board.PASS_LOC else [board.loc_x(loc), board.loc_y(loc)]
        return ["B" if pla == Board.BLACK else "W", coord]

    normalized = {
        "size": size,
        "setup": sorted((point(pla, loc) for pla, loc in setup), key=str),
        "moves": [point(pla, loc) for pla, loc in moves],
    }
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def arrays_sha256(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        contiguous = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(contiguous.dtype).encode())
        digest.update(json.dumps(list(contiguous.shape)).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def prepare_games(sources: dict, seed: int, validation_games: int) -> list[dict]:
    """Verify pinned SGFs and all moves before loading the large expert."""
    source_games = sources["games"]
    if len(source_games) < 2 or not 0 < validation_games < len(source_games):
        raise ValueError("Need at least one training game and one held-out game")
    ids = [item["game_id"] for item in source_games]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate game_id in source manifest")
    shuffled = sorted(ids)
    random.Random(seed).shuffle(shuffled)
    held_out = set(shuffled[-validation_games:])
    canonical_seen = set()
    prepared = []
    for source in sorted(source_games, key=lambda item: item["game_id"]):
        path = ROOT / source["file"]
        project_file(path)
        if file_sha256(path) != source["sgf_sha256"]:
            raise ValueError(f"Source SHA256 mismatch: {source['game_id']}")
        metadata, setup, moves, parsed_rules = load_sgf_moves_exn(str(path))
        root = sgf.Sgf_game.from_bytes(path.read_bytes()).get_root()
        original_rules = root.get("RU") if root.has_property("RU") else None
        if metadata.size != 19 or setup or metadata.handicap not in (None, 0):
            raise ValueError(f"Need a 19x19 game without setup/handicap: {source['game_id']}")
        if original_rules is None or original_rules.lower() != "chinese" or not parsed_rules:
            raise ValueError("This pilot requires explicit Chinese rules; it does not normalize original rules")
        rules = dict(GameState.RULES_CHINESE)
        rules.update(parsed_rules)
        if metadata.komi != 7.5:
            raise ValueError(f"This pilot requires explicit komi 7.5: {source['game_id']}")
        rules["whiteKomi"] = metadata.komi
        if (rules["koRule"] != "KO_SIMPLE" or rules["scoringRule"] != "SCORING_AREA"
                or rules["multiStoneSuicideLegal"] or rules["whiteKomi"] != 7.5):
            raise ValueError(f"Unsupported original rules/komi: {source['game_id']}")
        state = GameState(19, rules)
        for ply, (pla, loc) in enumerate(moves):
            if pla != state.board.pla:
                raise ValueError(f"Wrong side to play at ply {ply}: {source['game_id']}")
            # The vendor board permits multi-stone suicide; enforce the chosen rule here.
            if loc != Board.PASS_LOC and state.board.would_be_suicide(pla, loc):
                raise ValueError(f"Suicide is forbidden at ply {ply}: {source['game_id']}")
            state.play(pla, loc)
        canonical_sha = canonical_game_sha256(19, setup, moves)
        if canonical_sha in canonical_seen:
            raise ValueError("The same normalized game appears under multiple names")
        canonical_seen.add(canonical_sha)
        prepared.append({
            "metadata": {
                "game_id": source["game_id"],
                "canonical_game_sha256": canonical_sha,
                "sgf_sha256": source["sgf_sha256"],
                "file": project_file(path),
                "split": "val" if source["game_id"] in held_out else "train",
                "rules": rules,
                "original_rules": original_rules,
                "rules_normalized": False,
                "move_count": len(moves),
                "full_replay_legal": True,
            },
            "moves": moves,
        })
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "data/pilot-sources.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / f"{NETWORK}.ckpt")
    parser.add_argument("--plies", type=int, nargs="+", default=[8, 25, 48, 73])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-games", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/batch-data")
    args = parser.parse_args()
    if len(set(args.plies)) != len(args.plies) or any(ply <= 0 for ply in args.plies):
        raise ValueError("Choose distinct non-empty positions")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    source_json = json.loads(args.sources.read_text())
    if source_json.get("format_version") != 1:
        raise ValueError("Unsupported source manifest format")
    games = prepare_games(source_json, args.seed, args.validation_games)
    for game in games:
        if max(args.plies) >= len(game["moves"]):
            raise ValueError("Every selected position must precede another recorded move")
    source_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "vendor/KataGo"), "rev-parse", "HEAD"], text=True,
    ).strip()
    if source_commit != EXPERT_COMMIT:
        raise ValueError("KataGo source differs from the previously verified pinned commit")
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Dataset already exists; select a fresh --output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output / "features"
    feature_dir.mkdir(exist_ok=True)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    regular, model, _ = load_model(str(args.checkpoint), use_swa=True, device="cpu", pos_len=19)
    if model is None:
        raise ValueError("Checkpoint has no SWA model")
    del regular
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if len(model.blocks) != 11 or model.config["trunk_num_channels"] != 768:
        raise ValueError("Unexpected expert feature layout")
    features = Features(model.config, 19)
    names = [f"trunk_after_block{i}" for i in range(11)]
    samples = []
    positions: dict[str, str] = {}
    for game in games:
        info = game["metadata"]
        for ply in args.plies:
            state = GameState(19, info["rules"])
            for pla, loc in game["moves"][:ply]:
                state.play(pla, loc)
            spatial, global_input = state.get_input_features(features)
            legal = np.zeros(362, dtype=bool)
            for index in range(361):
                loc = features.tensor_pos_to_loc(index, state.board)
                legal[index] = (state.board.would_be_legal(state.board.pla, loc)
                                and not state.board.would_be_suicide(state.board.pla, loc))
            legal[361] = True
            position_sha = arrays_sha256(
                input_spatial=spatial, input_global=global_input, legal_mask=legal,
            )
            if position_sha in positions:
                if positions[position_sha] != info["split"]:
                    raise ValueError("Identical model input appears across train and validation")
                raise ValueError("Repeated position in this small pilot dataset")
            positions[position_sha] = info["split"]
            inference_start = time.perf_counter()
            outputs = infer(model, spatial, global_input, names)
            inference_seconds = time.perf_counter() - inference_start
            if any(name not in outputs for name in names):
                raise ValueError("Official ExtraOutputs omitted a requested feature")
            policy = np.where(legal, outputs["policy0"], 0).astype(np.float32)
            policy_mass = float(policy.sum())
            if not np.isfinite(policy_mass) or policy_mass <= 0:
                raise ValueError("Invalid legal-policy probability mass")
            policy /= policy_mass
            target_index = int(policy.argmax())
            if not legal[target_index]:
                raise ValueError("Teacher argmax is illegal")
            target_move = gtp_coord(state.board, features.tensor_pos_to_loc(target_index, state.board))
            arrays = {name: board_tokens(outputs[name], 768) for name in names}
            arrays.update(input_spatial=spatial, input_global=global_input,
                          legal_mask=legal, policy_legal=policy)
            if any(not np.isfinite(value).all() for value in arrays.values()):
                raise ValueError("Non-finite expert feature or probability")
            sample_id = f"{info['game_id']}-ply-{ply:03d}"
            relative_file = f"features/{sample_id}.npz"
            target_path = args.output / relative_file
            temporary_path = target_path.with_suffix(".npz.tmp")
            with temporary_path.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            temporary_path.replace(target_path)
            sample = {
                "sample_id": sample_id,
                "game_id": info["game_id"],
                "split": info["split"],
                "ply": ply,
                "to_play": "B" if state.board.pla == Board.BLACK else "W",
                "komi": info["rules"]["whiteKomi"],
                "target_move": target_move,
                "target_source": TARGET_SOURCE,
                "feature_file": relative_file,
                "feature_sha256": file_sha256(target_path),
                "position_sha256": position_sha,
                "moves": [["B" if pla == Board.BLACK else "W", gtp_coord(state.board, loc)]
                          for pla, loc in game["moves"][:ply]],
                "inference_seconds": inference_seconds,
            }
            samples.append(sample)
            print(f"{sample_id}: {info['split']}, {sample['to_play']} to play, "
                  f"NN target {target_move}, {sample['inference_seconds']:.3f}s", flush=True)
    manifest = {
        "format_version": 1,
        "seed": args.seed,
        "source": {"manifest": project_file(args.sources),
                   "manifest_sha256": file_sha256(args.sources), **source_json},
        "expert": {"network": NETWORK, "source_commit": source_commit,
                   "checkpoint_sha256": file_sha256(args.checkpoint), "weights": "swa",
                   "outer_blocks": 11, "width": 768},
        "split_method": "Sort game_id, shuffle with Python random.Random(seed), reserve the final N games for validation",
        "validation_games": args.validation_games,
        "selected_plies": args.plies,
        "rules_normalized": False,
        "token_order": "index=y*19+x; x=0 left, y=0 top; pass policy index=361; GTP skips I",
        "feature_shape": [11, 361, 768],
        "position_hash_definition": "SHA256 of sorted input_spatial, input_global, legal_mask names, dtypes, shapes and contiguous bytes",
        "games": [game["metadata"] for game in games],
        "samples": samples,
        "counts": {split: sum(sample["split"] == split for sample in samples)
                   for split in ("train", "val")},
        "checks": {"all_games_replayed_legally": True, "canonical_games_unique": True,
                   "position_hashes_unique": True, "cross_split_duplicate_positions": 0,
                   "all_targets_legal": True},
        "runtime": {"device": "cpu", "dtype": "float32", "threads": args.threads,
                    "elapsed_seconds": time.perf_counter() - started},
        "limitations": [f"{len(samples)} positions from {len(games)} historical human-versus-Pachi games are a pipeline pilot, not a statistically representative benchmark",
                        "Targets are one frozen neural-network policy argmax, not MCTS labels or the human next move",
                        "No positional augmentation, self-play or language-model training occurs in this script"],
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    print(f"Saved {len(samples)} samples: {manifest['counts']}", flush=True)


if __name__ == "__main__":
    main()
