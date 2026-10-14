#!/usr/bin/env python3
"""Extract fixed SGF positions with the pinned frozen KataGo on CUDA.

No download, training or cloud provisioning occurs here. Sources must supply
whole-game splits and selected_plies per game (or an explicit shared --plies).
Resume reuses only manifest-recorded, hash-matching caches; damaged or orphaned
files are preserved and rejected. Output is an independent position manifest,
not the historical fixed-five/three-game grounding manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
NETWORK = "kata1-tf3-b11c768-s11003M-d5973M-7gres"
EXPERT_COMMIT = "d91ea855110dae533f0aada947b2b7d78cc8a4e1"
BLOCK_NAMES = tuple(f"trunk_after_block{i}" for i in range(11))


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def project_path(value, root=ROOT):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Inputs and output must stay inside the project")
    return path


def selection_plan(sources, shared_plies=None):
    games = sources.get("games")
    if not isinstance(games, list) or not games:
        raise ValueError("Source manifest needs at least one game")
    seen, plan = set(), []
    for game in games:
        identity = game.get("game_id")
        if (not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity)
                or identity in seen):
            raise ValueError("Game IDs must be unique filename-safe strings")
        seen.add(identity)
        if game.get("split") not in ("train", "val", "test"):
            raise ValueError("Every source game needs an explicit whole-game split")
        if not re.fullmatch(r"[0-9a-f]{64}", game.get("sgf_sha256", "")):
            raise ValueError("Every game needs its pinned SGF SHA256")
        source_file = game.get("file")
        if (not isinstance(source_file, str) or not source_file
                or Path(source_file).is_absolute() or ".." in Path(source_file).parts):
            raise ValueError("SGF paths must be explicit project-relative paths")
        plies = game.get("selected_plies", sources.get("selected_plies", shared_plies))
        if shared_plies is not None and plies != shared_plies:
            raise ValueError("CLI plies cannot override a different source selection")
        if (not isinstance(plies, list) or not plies or any(type(p) is not int or p < 0 for p in plies)
                or len(set(plies)) != len(plies)):
            raise ValueError("Provide distinct nonnegative selected_plies; no implicit selection")
        for ply in sorted(plies):
            plan.append({"position_id": f"{identity}-ply-{ply:03d}", "game_id": identity,
                         "split": game["split"], "ply": ply,
                         "source_sgf": game["file"], "sgf_sha256": game["sgf_sha256"]})
    return plan


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_recorded_positions(manifest, contract, output):
    """Validate resume metadata and file bytes without NumPy/PyTorch imports."""
    if manifest.get("contract") != contract or manifest.get("contract_sha256") != content_sha256(contract):
        raise ValueError("Resume contract/source/implementation changed")
    planned = {row["position_id"]: row for row in contract["selection"]}
    records, seen_inputs = {}, {}
    for record in manifest.get("positions", []):
        identity = record.get("position_id")
        if identity not in planned or identity in records:
            raise ValueError("Resume has unknown or duplicate positions")
        if any(record.get(key) != value for key, value in planned[identity].items()):
            raise ValueError("Resume position metadata changed")
        expected_path = f"features/{identity}.npz"
        if record.get("feature_file") != expected_path:
            raise ValueError("Resume cache path changed")
        path = (output / expected_path).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file():
            raise ValueError("Resume cache is absent or outside output")
        if sha256_file(path) != record.get("feature_sha256"):
            raise ValueError("Resume cache SHA256 mismatch; retained without overwrite")
        digest = record.get("position_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Resume needs actual position-input digest")
        if digest in seen_inputs:
            raise ValueError("Duplicate model input in resume manifest")
        seen_inputs[digest] = identity
        records[identity] = record
    expected_files = {record["feature_file"] for record in records.values()}
    actual_files = {path.relative_to(output).as_posix()
                    for path in (output / "features").rglob("*") if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("Unrecorded or missing feature files retained; refusing resume")
    return records, seen_inputs


def _prepare_games(sources, plan):
    """Verify modern source rules and replay, before any checkpoint load."""
    from extract_features import Board
    from build_batch_data import canonical_game_sha256
    from prepare_expanded_sources import d4_hashes
    from prepare_katago_training_sources import parse_game_bytes, replay_game

    result, full_groups, prefix_groups = {}, {}, {}
    source_positions = {}
    for item in sources.get("positions", []):
        identity = item["position_id"]
        if identity in source_positions:
            raise ValueError("Repeated position metadata in source manifest")
        source_positions[identity] = item
    plies = {}
    for row in plan:
        plies.setdefault(row["game_id"], []).append(row["ply"])
    for game in sources["games"]:
        identity, path = game["game_id"], project_path(game["file"])
        if sha256_file(path) != game["sgf_sha256"]:
            raise ValueError("Source SGF SHA256 changed: " + identity)
        # The vendor SGF loader predates modern `...taxNONEsui0` rule strings.
        # Reuse the source selector's exact parser and full-main-line replay;
        # never map unknown or superko rules onto the older Chinese default.
        parsed = parse_game_bytes(path.read_bytes(), min_moves=0)
        replayed = {item["ply"]: item for item in replay_game(parsed, plies[identity])}
        for ply, state_record in replayed.items():
            saved = source_positions.get(f"{identity}-ply-{ply:03d}")
            if saved is not None and any(saved.get(key) != state_record[key]
                                         for key in ("to_play", "board_sha256", "board_labels361")):
                raise ValueError("Source saved board differs from SGF replay: " + identity)
        for key in ("rules", "original_rules", "komi"):
            if key in game and game[key] != parsed[key]:
                raise ValueError("Source rule metadata disagrees with SGF: " + identity)
        rules = parsed["rules"]
        board = Board(19)
        moves = [(Board.BLACK if color == "B" else Board.WHITE,
                  Board.PASS_LOC if point is None else board.loc(point % 19, point // 19))
                 for color, point in parsed["moves"]]
        full, prefix = min(d4_hashes([], moves)), min(d4_hashes([], moves, 160))
        if full in full_groups or prefix in prefix_groups:
            raise ValueError("Duplicate/symmetry-related game or long opening prefix")
        full_groups[full], prefix_groups[prefix] = identity, identity
        result[identity] = {"moves": moves, "rules": rules, "positions": replayed, "record": {
            "game_id": identity, "split": game["split"], "file": game["file"],
            "sgf_sha256": game["sgf_sha256"], "move_count": len(moves), "rules": rules,
            "original_rules": parsed["original_rules"], "komi": parsed["komi"],
            "rules_normalized": False,
            "canonical_game_sha256": canonical_game_sha256(19, [], moves),
            "d4_game_sha256": full, "d4_prefix160_sha256": prefix,
            "full_replay_legal": True}}
    return result


def _infer_cuda(model, prepared):
    import numpy as np
    import torch
    from extract_features import ExtraOutputs, board_tokens
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        spatial = torch.from_numpy(np.concatenate([row["spatial"] for row in prepared])).to("cuda")
        global_input = torch.from_numpy(np.concatenate([row["global_input"] for row in prepared])).to("cuda")
        extra = ExtraOutputs(list(BLOCK_NAMES))
        heads = model.postprocess_output(model(spatial, global_input, extra_outputs=extra))[0]
        policy = torch.softmax(heads[0][:, 0], dim=-1).float().cpu().numpy()
        activations = {name: extra.returned[name].detach().float().cpu().numpy() for name in BLOCK_NAMES}
    if policy.shape != (len(prepared), 362) or not bool(np.isfinite(policy).all()):
        raise ValueError("Invalid CUDA policy batch")
    if any(len(value) != len(prepared) for value in activations.values()):
        raise ValueError("Expert activation batch size mismatch")
    return [{"blocks": {name: board_tokens(activations[name][index], 768) for name in BLOCK_NAMES},
             "policy": policy[index]} for index in range(len(prepared))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / f"{NETWORK}.ckpt")
    parser.add_argument("--plies", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.seconds < 1 or args.threads < 1:
        parser.error("batch-size, threads and seconds must be positive")
    started = time.perf_counter()
    sources_path, checkpoint, output = map(project_path, (args.sources, args.checkpoint, args.output))
    sources = json.loads(sources_path.read_text())
    if (sources.get("stage") == "katago_public_occupancy_preparation"
            and sources.get("status") != "ready"):
        raise ValueError("Source preparation has not met its recorded data requirements")
    plan = selection_plan(sources, args.plies)
    expert_sources = json.loads((ROOT / "sources.json").read_text())
    checkpoint_pin = next(item for item in expert_sources["files"] if item["file"] == checkpoint.name)
    if (expert_sources["network"] != NETWORK or expert_sources["katago_commit"] != EXPERT_COMMIT
            or checkpoint.stat().st_size != checkpoint_pin["bytes"]
            or sha256_file(checkpoint) != checkpoint_pin["sha256"]):
        raise ValueError("Pinned expert checkpoint changed")
    commit = subprocess.check_output(["git", "-C", str(ROOT / "vendor/KataGo"), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPERT_COMMIT:
        raise ValueError("KataGo checkout is not the pinned expert version")
    implementation = {str(path.relative_to(ROOT)): sha256_file(path) for path in
                      (Path(__file__).resolve(), ROOT / "scripts/extract_features.py",
                       ROOT / "scripts/build_batch_data.py", ROOT / "scripts/prepare_expanded_sources.py",
                       ROOT / "scripts/prepare_katago_training_sources.py")}
    # Pin the actual Python implementation, not just a possibly modified Git HEAD.
    implementation.update({str(path.relative_to(ROOT)): sha256_file(path)
                           for path in sorted((ROOT / "vendor/KataGo/python/katago").rglob("*.py"))})
    contract = {"source_manifest": str(sources_path.relative_to(ROOT)),
                "source_manifest_sha256": sha256_file(sources_path),
                "expert_sources_sha256": sha256_file(ROOT / "sources.json"),
                "network": NETWORK, "expert_commit": commit,
                "checkpoint_sha256": checkpoint_pin["sha256"], "weights": "swa",
                "device": "cuda", "dtype": "float32", "batch_size": args.batch_size,
                "threads": args.threads, "tf32": False, "sdpa_backend": "math",
                "feature_shape": [11, 361, 768], "selection": plan,
                "position_rule_context": "For supported area rules, passWouldEndPhase equals previous move is pass; terminal two-pass positions rejected",
                "token_order": "index=y*19+x; x=0 left, y=0 top; pass policy index=361; GTP skips I",
                "implementation_sha256": implementation}
    manifest_path = output / "manifest.json"
    if output.exists():
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError("Use fresh output or explicit --resume with a recorded manifest")
        manifest = json.loads(manifest_path.read_text())
        records, seen_inputs = verify_recorded_positions(manifest, contract, output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "features").mkdir()
        manifest = {"format_version": 1, "stage": "cuda_frozen_expert_features", "status": "prepared",
                    "complete": False, "contract": contract, "contract_sha256": content_sha256(contract),
                    "games": [], "positions": [], "runs": []}
        records, seen_inputs = {}, {}
        atomic_json(manifest_path, manifest)
    run = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "soft_seconds": args.seconds, "reused_positions": len(records),
           "forward_batches_attempted": 0, "forward_batches_returned": 0, "new_positions": 0,
           "status": "running", "optimizer_updates": 0}
    manifest["runs"].append(run)
    manifest["complete"], manifest["status"] = False, "running"

    def record():
        manifest["positions"] = [records[row["position_id"]] for row in plan if row["position_id"] in records]
        manifest["counts"] = {"requested_positions": len(plan), "saved_positions": len(records),
                              "saved_by_split": dict(Counter(r["split"] for r in records.values()))}
        run["seconds_before_manifest_write"] = time.perf_counter() - started
        atomic_json(manifest_path, manifest)

    record()
    try:
        prepared_games = _prepare_games(sources, plan)
        manifest["games"] = [prepared_games[g["game_id"]]["record"] for g in sources["games"]]
        missing = [row for row in plan if row["position_id"] not in records]
        if missing:
            import numpy as np
            import torch
            from extract_features import Board, Features, GameState, load_model, gtp_coord
            from build_batch_data import arrays_sha256
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required; no automatic CPU fallback")
            torch.set_num_threads(args.threads)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            run.update(torch_version=torch.__version__, cuda_version=torch.version.cuda,
                       gpu=torch.cuda.get_device_name(0), tf32=False)
            if time.perf_counter() - started >= args.seconds:
                raise TimeoutError("Soft budget exhausted before expert load")
            regular, model, _ = load_model(str(checkpoint), use_swa=True, device="cuda", pos_len=19)
            del regular
            if model is None or len(model.blocks) != 11 or model.config["trunk_num_channels"] != 768:
                raise ValueError("Unexpected frozen expert layout")
            model.eval().requires_grad_(False)
            if any(p.device.type != "cuda" or p.dtype != torch.float32 for p in model.parameters()):
                raise ValueError("Require CUDA FP32 expert parameters")
            feature_encoder = Features(model.config, 19)
            pending = []

            def flush():
                if not pending:
                    return
                if time.perf_counter() - started >= args.seconds:
                    raise TimeoutError("Soft budget exhausted before next expert batch")
                run["forward_batches_attempted"] += 1
                record()
                predictions = _infer_cuda(model, pending)
                run["forward_batches_returned"] += 1
                for item, predicted in zip(pending, predictions):
                    row, legal = item["row"], item["legal"]
                    relative = f"features/{row['position_id']}.npz"
                    target = output / relative
                    temporary = target.with_suffix(".npz.part")
                    if target.exists() or temporary.exists():
                        raise FileExistsError("Unrecorded cache retained; refusing to overwrite " + relative)
                    policy = np.where(legal, predicted["policy"], 0).astype(np.float32)
                    mass = float(policy.sum())
                    if not np.isfinite(mass) or mass <= 0:
                        raise ValueError("Invalid legal policy mass")
                    policy /= mass
                    arrays = {**predicted["blocks"], "input_spatial": item["spatial"],
                              "input_global": item["global_input"], "legal_mask": legal, "policy_legal": policy}
                    if any(not bool(np.isfinite(v).all()) for v in arrays.values()):
                        raise ValueError("Non-finite saved expert arrays")
                    with temporary.open("xb") as stream:
                        np.savez_compressed(stream, **arrays)
                    temporary.replace(target)
                    target_index = int(policy.argmax())
                    result = {**row, **item["metadata"], "feature_file": relative,
                              "feature_sha256": sha256_file(target), "feature_bytes": target.stat().st_size,
                              "target_index": target_index,
                              "target_move": ("pass" if target_index == 361 else
                                              "ABCDEFGHJKLMNOPQRST"[target_index % 19]
                                              + str(19 - target_index // 19)),
                              "target_source": "katago_nn_legal_policy_argmax"}
                    records[row["position_id"]] = result
                    seen_inputs[result["position_sha256"]] = row["position_id"]
                    run["new_positions"] += 1
                    record()
                pending.clear()

            missing_ids = {row["position_id"] for row in missing}
            for game in sources["games"]:
                loaded = prepared_games[game["game_id"]]
                state, played = GameState(19, loaded["rules"]), 0
                for row in (row for row in plan if row["game_id"] == game["game_id"]):
                    for pla, loc in loaded["moves"][played:row["ply"]]:
                        state.play(pla, loc)
                    played = row["ply"]
                    if row["position_id"] not in missing_ids:
                        continue
                    if time.perf_counter() - started >= args.seconds:
                        raise TimeoutError("Soft budget exhausted while preparing positions")
                    if (len(state.moves) >= 2 and state.moves[-1][1] == Board.PASS_LOC
                            and state.moves[-2][1] == Board.PASS_LOC):
                        raise ValueError("Selected feature position is after terminal two passes")
                    # The supported area-rule context changes with move history;
                    # the SGF-level default False is not correct after one pass.
                    state.rules["passWouldEndPhase"] = bool(state.moves and state.moves[-1][1] == Board.PASS_LOC)
                    spatial, global_input = state.get_input_features(feature_encoder)
                    legal = np.zeros(362, dtype=np.bool_)
                    absolute = []
                    for index in range(361):
                        loc = feature_encoder.tensor_pos_to_loc(index, state.board)
                        stone = state.board.board[loc]
                        absolute.append(1 if stone == Board.BLACK else 2 if stone == Board.WHITE else 0)
                        legal[index] = (state.board.would_be_legal(state.board.pla, loc)
                                        and not state.board.would_be_suicide(state.board.pla, loc))
                    legal[361] = True
                    board_sha = hashlib.sha256(bytes(absolute)).hexdigest()
                    side = "B" if state.board.pla == Board.BLACK else "W"
                    replayed = loaded["positions"][row["ply"]]
                    if (board_sha != replayed["board_sha256"] or side != replayed["to_play"]
                            or absolute != replayed["board_labels361"]):
                        raise ValueError("Feature input board differs from verified SGF position")
                    digest = arrays_sha256(input_spatial=spatial, input_global=global_input, legal_mask=legal)
                    if digest in seen_inputs:
                        raise ValueError("Repeated model input across selected positions")
                    # Include pending inputs in duplicate checks before their batch is saved.
                    seen_inputs[digest] = row["position_id"]
                    pending.append({"row": row, "spatial": spatial, "global_input": global_input, "legal": legal,
                                    "metadata": {"position_sha256": digest, "to_play": side,
                                                 "board_sha256": board_sha,
                                                 "pass_would_end_phase": state.rules["passWouldEndPhase"],
                                                 "komi": loaded["rules"]["whiteKomi"],
                                                 "original_rules": loaded["record"]["original_rules"],
                                                 "absolute_board_labels": absolute,
                                                 "moves": [["B" if pla == Board.BLACK else "W", gtp_coord(state.board, loc)]
                                                           for pla, loc in loaded["moves"][:row["ply"]]]}})
                    if len(pending) == args.batch_size:
                        flush()
            flush()
        if (sha256_file(sources_path) != contract["source_manifest_sha256"]
                or sha256_file(checkpoint) != contract["checkpoint_sha256"]
                or sha256_file(ROOT / "sources.json") != contract["expert_sources_sha256"]
                or any(sha256_file(project_path(g["file"])) != g["sgf_sha256"] for g in sources["games"])
                or any(sha256_file(ROOT / path) != digest for path, digest in implementation.items())):
            raise ValueError("Input or implementation changed during extraction")
        manifest["complete"] = len(records) == len(plan) and time.perf_counter() - started <= args.seconds
        manifest["status"] = run["status"] = "complete" if manifest["complete"] else "stopped"
    except TimeoutError as exc:
        manifest["complete"] = False
        manifest["status"] = run["status"] = "stopped"
        run["stop_reason"] = str(exc)
    except Exception as exc:
        manifest["complete"] = False
        manifest["status"] = run["status"] = "error"
        run["error"] = {"type": type(exc).__name__, "message": str(exc).replace(str(ROOT), ".")}
    finally:
        record()
        elapsed = time.perf_counter() - started
        if elapsed > args.seconds:
            manifest["complete"] = False
            if manifest["status"] == "complete":
                manifest["status"] = run["status"] = "stopped"
            run["budget_exceeded"] = True
            record()
    print(json.dumps({"status": manifest["status"], "complete": manifest["complete"],
                      "saved_positions": len(records), "requested_positions": len(plan),
                      "seconds": time.perf_counter() - started}, ensure_ascii=False), flush=True)
    return 1 if manifest["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
