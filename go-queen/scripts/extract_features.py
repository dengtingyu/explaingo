#!/usr/bin/env python3
"""Read official KataGo weights and save 19x19 hidden features; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# The official default imports a CUDA/Triton-only acceleration kernel at construction.
# CPU uses the same unfused PyTorch math; keep the vendor source unchanged.
os.environ["KATAGO_FUSED_SWIGLU_KERNEL"] = "0"
os.environ["KATAGO_FLEX_ATTENTION"] = "0"
os.environ["KATAGO_FUSED_ROPE_BACKWARD"] = "0"
sys.path.insert(0, str(ROOT / "vendor/KataGo/python"))
from katago.game.board import Board
from katago.game.data import load_sgf_moves_exn
from katago.game.features import Features
from katago.game.gamestate import GameState
from katago.train.load_model import load_model
from katago.train.model_pytorch import ExtraOutputs

NETWORK = "kata1-tf3-b11c768-s11003M-d5973M-7gres"


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def gtp_coord(board: Board, loc: int) -> str:
    if loc == Board.PASS_LOC:
        return "pass"
    return "ABCDEFGHJKLMNOPQRST"[board.loc_x(loc)] + str(19 - board.loc_y(loc))


def board_tokens(array: np.ndarray, channels: int) -> np.ndarray:
    """Canonical order is SGF top-to-bottom, then left-to-right (y*19+x)."""
    if array.shape == (channels, 19, 19):
        result = array.transpose(1, 2, 0).reshape(361, channels)
    elif array.shape == (361, channels):
        result = array
    else:
        raise ValueError(f"Unexpected feature shape {array.shape}; refuse to guess token order")
    if not np.isfinite(result).all():
        raise ValueError("Hidden features contain NaN or infinity")
    return np.ascontiguousarray(result, dtype=np.float32)


def infer(model, spatial: np.ndarray, global_input: np.ndarray, names: list[str]) -> dict:
    # GameState.get_model_outputs at this commit assumes old Q-value heads that
    # version 17 no longer has. Read only the needed, stable heads directly.
    with torch.inference_mode():
        extra = ExtraOutputs(names)
        heads = model.postprocess_output(model(torch.from_numpy(spatial), torch.from_numpy(global_input),
                                               extra_outputs=extra))[0]
        (policy_logits, value_logits, _td_value, _td_score, ownership_pretanh,
         _scoring, _futurepos, _seki, _scoremean, _scorestdev, lead,
         _vtime, _value_error, _score_error, _scorebelief) = (head[0] for head in heads)
        return {
            "policy0": torch.softmax(policy_logits[0], dim=0).numpy(),
            "value": torch.softmax(value_logits, dim=0).numpy(),
            "ownership": torch.tanh(ownership_pretanh).reshape(361).numpy(),
            "lead": lead.item(),
            **{name: activation[0].numpy() for name, activation in extra.returned.items()},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models" / f"{NETWORK}.ckpt")
    parser.add_argument("--sgf", type=Path, default=ROOT / "examples/opening.sgf")
    parser.add_argument("--plies", type=int, nargs="+", default=[0, 1, 8])
    parser.add_argument("--weights", choices=["swa", "regular"], default="swa")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    regular, swa, _ = load_model(str(args.checkpoint), use_swa=args.weights == "swa", device="cpu", pos_len=19)
    model = swa if args.weights == "swa" else regular
    assert model is not None
    if args.weights == "swa":
        del regular
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_seconds = time.perf_counter() - start
    channels = model.config["trunk_num_channels"]
    names = [f"trunk_after_block{i}" for i in range(len(model.blocks))] + ["trunkfinal"]
    print(f"Loaded {args.weights} weights: {len(model.blocks)} outer blocks, {channels} channels", flush=True)

    metadata, setup, moves, sgf_rules = load_sgf_moves_exn(str(args.sgf))
    if metadata.size != 19 or setup:
        raise ValueError("This first milestone supports ordinary 19x19 games without setup stones")
    rules = dict(GameState.RULES_CHINESE)
    if sgf_rules:
        rules.update(sgf_rules)
    if rules["koRule"] != "KO_SIMPLE" or rules["scoringRule"] != "SCORING_AREA":
        raise ValueError("First milestone is limited to simple-ko area rules")
    if metadata.komi is not None:
        rules["whiteKomi"] = metadata.komi
    features = Features(model.config, 19)
    cases = []

    for ply in args.plies:
        if not 0 <= ply <= len(moves):
            raise ValueError(f"Invalid ply {ply}")
        state = GameState(19, rules)
        for player, loc in moves[:ply]:
            state.play(player, loc)
        spatial, global_input = state.get_input_features(features)
        # Check every stone against its known board coordinate in the actual model input.
        for y in range(19):
            for x in range(19):
                player = state.board.board[state.board.loc(x, y)]
                assert spatial[0, 1, y, x] == (player == state.board.pla)
                assert spatial[0, 2, y, x] == (player == Board.get_opp(state.board.pla))
        start = time.perf_counter()
        outputs = infer(model, spatial, global_input, names)
        elapsed = time.perf_counter() - start
        if any(name not in outputs for name in names):
            raise ValueError("Official ExtraOutputs did not return all requested features")

        legal = np.zeros(362, dtype=bool)
        for i in range(361):
            loc = features.tensor_pos_to_loc(i, state.board)
            legal[i] = state.board.would_be_legal(state.board.pla, loc)
        legal[361] = True
        policy = np.where(legal, outputs["policy0"], 0)
        policy /= policy.sum()
        arrays = {name: board_tokens(outputs[name], channels) for name in names}
        arrays.update(input_spatial=spatial, input_global=global_input,
                      policy_raw=outputs["policy0"], policy_legal=policy,
                      legal_mask=legal, value=outputs["value"],
                      ownership=outputs["ownership"], lead=np.array(outputs["lead"]))
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite output: {key}")

        # Requesting hidden features must leave the model's predictions unchanged.
        repeat_error = None
        if ply == args.plies[0]:
            repeated = infer(model, spatial, global_input, [])
            repeat_error = max(float(np.max(np.abs(outputs[key] - repeated[key])))
                               for key in ("policy0", "value", "ownership"))
            if repeat_error > 1e-6:
                raise ValueError(f"Feature extraction changed predictions: {repeat_error}")

        filename = f"ply-{ply:03d}.npz"
        np.savez_compressed(args.output / filename, **arrays)
        to_play = "B" if state.board.pla == Board.BLACK else "W"
        top = np.argsort(policy)[-5:][::-1]
        case = {
            "ply": ply, "to_play": to_play, "file": filename,
            "moves": [["B" if pla == Board.BLACK else "W", gtp_coord(state.board, loc)]
                      for pla, loc in moves[:ply]],
            "inference_seconds": elapsed,
            "raw_feature_shapes": {name: list(outputs[name].shape) for name in names},
            "saved_feature_shape": [361, channels],
            "top_moves": [{"move": gtp_coord(state.board, features.tensor_pos_to_loc(int(i), state.board)),
                           "probability": float(policy[i])} for i in top],
            "value_current_player": outputs["value"].tolist(),
            "lead_current_player": outputs["lead"],
            "feature_extraction_prediction_max_error": repeat_error,
        }
        cases.append(case)
        print(f"ply={ply}, to_play={to_play}, {elapsed:.3f}s, features={len(names)} x (361,{channels})", flush=True)

    report = {
        "milestone": "19x19 frozen expert feature extraction",
        "network": NETWORK, "weights": args.weights,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "sgf_sha256": file_sha256(args.sgf), "sgf": str(args.sgf.resolve()),
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT / "vendor/KataGo"), "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "device": "cpu", "dtype": "float32", "threads": args.threads,
        "cpu_compatibility_flags": {name: os.environ[name] for name in
                                    ("KATAGO_FUSED_SWIGLU_KERNEL", "KATAGO_FLEX_ATTENTION", "KATAGO_FUSED_ROPE_BACKWARD")},
        "parameters": sum(p.numel() for p in model.parameters()),
        "load_seconds": load_seconds, "rules": rules,
        "outer_blocks": len(model.blocks), "hidden_channels": channels,
        "token_order": "index=y*19+x; x=0 left, y=0 top; pass policy index=361; GTP skips I",
        "config": model.config, "cases": cases,
        "limitations": ["No training or language model yet", "Only these ordinary simple-ko area-rule positions were verified", "Native comparison is recorded separately"],
    }
    (args.output / "extraction-report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
