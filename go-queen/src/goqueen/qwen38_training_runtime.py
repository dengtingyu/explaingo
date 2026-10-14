"""Cloud-only Qwen tokenization, forward scoring and adapter checkpoint payloads.

No weight load occurs on import. The caller owns the fixed recipe, device map,
optimizer, epoch order, exclusive output root and process lifecycle.
"""
from __future__ import annotations

import json
from pathlib import Path

LABELS = ('empty', 'black', 'white')
JSON_PREFIX = '{"occupancy":"'


def encode_questions(tokenizer, questions):
    """Encode full nonthinking prefixes once; verify actual one-token answers."""
    result = {}
    label_ids = {}
    prompt_cache = {}
    for q in questions:
        if q['question_id'] in result:
            raise ValueError('Repeated question ID')
        if q['prompt'] in prompt_cache:
            result[q['question_id']] = prompt_cache[q['prompt']]
            continue
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': q['prompt']}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) + JSON_PREFIX
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        for label in LABELS:
            full = tokenizer.encode(prompt + label, add_special_tokens=False)
            if full[:-1] != ids or len(full) != len(ids) + 1:
                raise ValueError('Occupancy must be exactly one token after this prefix')
            if label in label_ids and label_ids[label] != full[-1]:
                raise ValueError('Label token differs across prompts')
            label_ids[label] = full[-1]
        if q['question_id'] in result or not ids:
            raise ValueError('Repeated question ID or empty prefix')
        result[q['question_id']] = ids
        prompt_cache[q['prompt']] = ids
    if len(set(label_ids.values())) != 3:
        raise ValueError('Three distinct occupancy token IDs required')
    return result, label_ids


def make_inputs(torch, tokenizer, rows, encoded, input_device):
    """Right padding leaves every causal real prefix before its padding.

    Compute logits only at the union of last-real-token positions and select the
    appropriate position per row. No assumption that padded last token is real.
    """
    sequences = [encoded[q['question_id']] for q in rows]
    width = max(map(len, sequences))
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if type(pad) is not int:
        raise ValueError('A concrete padding token is required')
    ids = torch.full((len(rows), width), pad, dtype=torch.long, device=input_device)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=input_device)
        mask[i, :len(seq)] = 1
    last = [len(seq)-1 for seq in sequences]
    unique_last = sorted(set(last))
    row_positions = [unique_last.index(i) for i in last]
    # Python integer list avoids a top-level dispatch hook moving indices to GPU0.
    # The pinned Qwen forward uses non-int logits_to_keep directly for indexing.
    inputs = {'input_ids': ids, 'attention_mask': mask, 'use_cache': False,
              'logits_to_keep': unique_last}
    return inputs, row_positions


def forward_logits(torch, model, tokenizer, rows, encoded, cache, input_device):
    """Call only inside the caller's binding context (including later backward)."""
    inputs, positions = make_inputs(torch, tokenizer, rows, encoded, input_device)
    logits = model(**inputs).logits
    logits = logits[torch.arange(len(rows), device=logits.device),
                    torch.tensor(positions, device=logits.device)].float()
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError('Nonfinite full-vocabulary logits')
    return logits


def feature_batch(torch, rows, cache):
    # Keep canonical cache on CPU. Qwen wrappers copy each block to its own GPU.
    return torch.stack([cache[q['position_id']] for q in rows]).detach()


def score_batch(torch, logits, rows, label_ids):
    full_logp = logits.log_softmax(-1)
    ids = torch.tensor([label_ids[x] for x in LABELS], device=logits.device)
    three_raw = full_logp.index_select(-1, ids)
    three_logp = three_raw.log_softmax(-1)
    forced = three_logp.argmax(-1)
    unrestricted = logits.argmax(-1)
    gold = torch.tensor([q['label'] for q in rows], device=logits.device)
    indices = torch.arange(len(rows), device=logits.device)
    forced_list, unrestricted_list = forced.tolist(), unrestricted.tolist()
    raw_list = three_raw.tolist()
    three_ce = (-three_logp[indices, gold]).tolist()
    full_ce = (-three_raw[indices, gold]).tolist()
    return [dict(question_id=q['question_id'], position_id=q['position_id'], game_id=q['game_id'],
                 ply=q.get('ply'), to_play=q['to_play'], label=q['label_name'],
                 prediction=LABELS[forced_list[i]], correct=forced_list[i] == q['label'],
                 full_token_id=unrestricted_list[i],
                 full_correct=unrestricted_list[i] == label_ids[q['label_name']],
                 raw_class_logp=raw_list[i], three_class_ce=three_ce[i], full_vocab_ce=full_ce[i])
            for i, q in enumerate(rows)]


def aggregate(rows):
    if not rows:
        raise ValueError('Cannot summarize empty evaluation')
    matrix = [[0] * 3 for _ in LABELS]
    for r in rows:
        matrix[LABELS.index(r['label'])][LABELS.index(r['prediction'])] += 1
    recalls = {label: matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else None
               for i, label in enumerate(LABELS)}
    present = [v for v in recalls.values() if v is not None]
    n = len(rows)
    return {'n': n, 'correct': sum(r['correct'] for r in rows),
            'accuracy': sum(r['correct'] for r in rows) / n,
            'full_vocab_correct': sum(r['full_correct'] for r in rows),
            'macro_recall': sum(present) / len(present), 'per_class_recall': recalls,
            'confusion_labels': list(LABELS), 'confusion_matrix': matrix,
            'three_class_ce': sum(r['three_class_ce'] for r in rows) / n,
            'full_vocab_ce': sum(r['full_vocab_ce'] for r in rows) / n}


def save_payload(torch, save_file, model, optimizer, directory, *, global_step, rng_state):
    """Save only adapter tensors and same-run trusted optimizer state, never base."""
    directory = Path(directory)
    adapters = model.adapter_named_parameters()
    if len(adapters) != 101 or any(not bool(torch.isfinite(p).all()) for p in adapters.values()):
        raise ValueError('Expected all101 finite adapter tensors')
    save_file({name: p.detach().cpu().contiguous() for name, p in adapters.items()},
              str(directory / 'adapter.safetensors'),
              metadata={'model': 'Qwen/Qwen3.8-27B', 'global_step': str(global_step),
                        'adapter_attention_width': str(model.architecture()['adapter_attention_width'])})
    # This is authored in this run and loaded only after whole-checkpoint hashes
    # match; never load optimizer files supplied by an unrelated source.
    torch.save({'optimizer': optimizer.state_dict(), 'rng': rng_state}, directory / 'training-state.pt')
    (directory / 'architecture.json').write_text(json.dumps(model.architecture(), indent=2)+'\n')
