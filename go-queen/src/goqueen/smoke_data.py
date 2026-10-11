"""Three tiny, deliberately limited examples for checking the fusion wiring.

The target is the frozen expert neural network's legal policy argmax. It is a
toy label, not a searched move or a human annotation. Each feature file describes
the position *before* that target move; the answer is never played before loading
features. These examples cannot establish playing strength or generalization.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch


BOARD_SIZE = 19
BLOCKS = 11
CHANNELS = 768
GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"


def _policy_index(move: str) -> int:
    """Map GTP coordinates to KataGo's top-to-bottom, row-major policy index."""
    if move.lower() == "pass":
        return BOARD_SIZE * BOARD_SIZE
    match = re.fullmatch(r"([ABCDEFGHJKLMNOPQRST])(\d{1,2})", move.upper())
    if match is None:
        raise ValueError(f"Invalid 19x19 GTP move: {move!r}")
    row = int(match.group(2))
    if not 1 <= row <= BOARD_SIZE:
        raise ValueError(f"GTP row must be 1..19: {move!r}")
    x = GTP_COLUMNS.index(match.group(1))
    y = BOARD_SIZE - row
    return y * BOARD_SIZE + x


def load_case(output_dir: Path, ply: int) -> dict[str, Any]:
    """Load one previously extracted position and validate its toy move label.

    Returns float32 ``features`` with shape ``[1, 11, 361, 768]`` plus prompt,
    JSON target, position metadata, and the SHA-256 of the source feature file.
    """
    output_dir = Path(output_dir)
    report = json.loads((output_dir / "extraction-report.json").read_text())
    matches = [case for case in report["cases"] if case["ply"] == ply]
    if len(matches) != 1:
        raise ValueError(f"Expected one extraction case for ply {ply}, got {len(matches)}")
    case = matches[0]
    assert report["outer_blocks"] == BLOCKS
    assert report["hidden_channels"] == CHANNELS
    assert len(case["moves"]) == ply, "Features must describe the requested pre-move position"
    to_play = case["to_play"]
    if to_play not in ("B", "W"):
        raise ValueError(f"Unexpected player: {to_play!r}")
    feature_file = (output_dir / case["file"]).resolve()
    target_move = case["top_moves"][0]["move"]
    target_index = _policy_index(target_move)

    with np.load(feature_file, allow_pickle=False) as saved:
        layers = []
        for layer_index in range(BLOCKS):
            layer = saved[f"trunk_after_block{layer_index}"]
            assert layer.shape == (BOARD_SIZE * BOARD_SIZE, CHANNELS), layer.shape
            assert layer.dtype == np.float32, layer.dtype
            assert np.isfinite(layer).all(), "Non-finite expert features"
            layers.append(layer)
        stacked = np.stack(layers, axis=0)
        legal_mask = saved["legal_mask"]
        policy_legal = saved["policy_legal"]
        assert legal_mask.shape == policy_legal.shape == (362,)
        assert legal_mask.dtype == np.bool_
        assert np.isfinite(policy_legal).all()
        assert legal_mask[target_index], "Report target must be a legal move"
        assert target_index == int(policy_legal.argmax()), "Report and saved policy disagree"
        assert np.all(policy_legal[~legal_mask] == 0), "Illegal moves must have zero probability"
        assert np.isclose(policy_legal.sum(), 1.0, atol=1e-5)

    features = torch.from_numpy(stacked).unsqueeze(0)
    assert features.dtype == torch.float32
    assert features.shape == (1, BLOCKS, BOARD_SIZE * BOARD_SIZE, CHANNELS)
    player_name = "黑方" if to_play == "B" else "白方"
    prompt = (
        "请根据当前棋盘特征，在19路围棋（中国规则，贴7.5目）中"
        f"为{player_name}选择下一手。只输出JSON，例如{{\"move\":\"D4\"}}。"
    )
    target = json.dumps({"move": target_move}, ensure_ascii=False, separators=(",", ":"))
    return {
        "features": features,
        "prompt": prompt,
        "target": target,
        "ply": ply,
        "to_play": to_play,
        "feature_file": str(feature_file),
        "target_move": target_move,
        "feature_sha256": hashlib.sha256(feature_file.read_bytes()).hexdigest(),
    }


def encode_answer(
    tokenizer: Any, prompt: str, target: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return one unpadded training sequence and its answer-boundary metadata.

    The official Qwen chat template builds the generation prefix with thinking
    disabled. Append the independently tokenized JSON answer and end-of-message
    token, so the empty thinking prefix cannot differ between two template calls.
    Labels use unshifted token IDs; the causal language model performs its usual
    next-token shift exactly once when it computes loss.
    """
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt_ids, list) or not all(isinstance(token, int) for token in prompt_ids):
        raise TypeError("Expected chat template to return a flat list of token IDs")
    answer_text = target + "<|im_end|>"
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    assert prompt_ids and answer_ids
    assert answer_ids[-1] == tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert tokenizer.decode(answer_ids, skip_special_tokens=False) == answer_text

    answer_start = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :answer_start] = -100
    supervised = labels.ne(-100)
    expected_supervised = torch.arange(input_ids.shape[1]).unsqueeze(0).ge(answer_start)
    assert torch.equal(supervised, expected_supervised), "Only answer tokens may be supervised"
    assert not torch.any(supervised & attention_mask.eq(0)), "Padding must never receive labels"
    assert torch.equal(labels[supervised], input_ids[supervised]), "Do not pre-shift labels"
    assert tokenizer.decode(labels[supervised].tolist(), skip_special_tokens=False) == answer_text
    assert int(supervised.sum()) == len(answer_ids)

    inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    metadata = {
        "prompt_tokens": answer_start,
        "answer_tokens": len(answer_ids),
        "answer_start": answer_start,
        "target": target,
        "target_token_ids": answer_ids,
    }
    return inputs, metadata
