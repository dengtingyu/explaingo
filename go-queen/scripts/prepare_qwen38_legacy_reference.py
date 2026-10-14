#!/usr/bin/env python3
"""Copy the fixed historical54 reference; standard-library metadata only.

No model, tensor payload decode, new point selection or label change. The
prompt adopts the current source recipe's wording and preserves actual rules.
"""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original = ROOT / 'outputs/grounding-data/manifest.json'
    original_m1 = ROOT / 'outputs/m1-data/manifest.json'
    data, m1 = json.loads(original.read_text()), json.loads(original_m1.read_text())
    assert sha(original_m1) == data['source']['manifest_sha256']
    assert m1['expert']['checkpoint_sha256'] == '59ccd9a0f2f472525b9a2432a16a6a92e5576c467262c9383d152e97337b6282'
    questions = [q for q in data['original_occupancy_questions'] if q['split'] == 'val']
    assert len(questions) == 54 and len({q['question_id'] for q in questions}) == 54
    assert Counter(q['label'] for q in questions) == {0: 18, 1: 18, 2: 18}
    positions = {p['position_id']: p for p in data['positions']}
    games = {g['game_id']: g for g in m1['games']}
    output = ROOT / 'outputs/qwen38-legacy-reference'
    output.mkdir(exist_ok=False)
    (output / 'features').mkdir()
    spec = importlib.util.spec_from_file_location('source_prompt', ROOT / 'scripts/prepare_katago_training_sources.py')
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)
    labels = ('empty', 'black', 'white')
    feature_records, rows = {}, []
    for q in questions:
        p = positions[q['position_id']]
        rule = games[p['game_id']]['rules']
        assert rule['koRule'] == 'KO_SIMPLE' and rule['scoringRule'] == 'SCORING_AREA'
        assert rule['taxRule'] == 'TAX_NONE' and not rule['multiStoneSuicideLegal'] and not rule['hasButton']
        assert p['komi'] == rule['whiteKomi']
        assert q['label'] == p['absolute_board_labels'][q['point_index']]
        assert all(q[k] == p[k] for k in ('game_id', 'position_id', 'to_play', 'ply'))
        if p['position_id'] not in feature_records:
            path = (ROOT / p['source_feature_file']).resolve()
            assert path.is_relative_to(ROOT) and sha(path) == p['feature_sha256']
            target = output / 'features' / path.name
            shutil.copyfile(path, target)
            assert sha(target) == p['feature_sha256']
            feature_records[p['position_id']] = dict(position_id=p['position_id'], game_id=p['game_id'],
                ply=p['ply'], to_play=p['to_play'], feature_file='features/' + target.name,
                feature_sha256=p['feature_sha256'], original_feature=p['source_feature_file'])
        row = {k: q[k] for k in ('question_id', 'position_id', 'game_id', 'ply', 'to_play', 'point_index', 'label')}
        row.update(label_name=labels[q['label']], split='legacy', coordinate=source.coordinate(q['point_index']),
                   prompt=source.question_prompt(q['to_play'], q['point_index'], p['komi']),
                   original_prompt=q['prompt'])
        rows.append(row)
    assert len(feature_records) == 18 and len({r['game_id'] for r in rows}) == 3
    feature_manifest = {'positions': list(feature_records.values()), 'expert': m1['expert'],
                        'scope': 'Existing historical features copied unchanged; no new extraction or payload decode'}
    (output / 'feature-manifest.json').write_text(json.dumps(feature_manifest, ensure_ascii=False, indent=2) + '\n')
    manifest = {'stage': 'qwen38_historical54_secondary_reference', 'questions': rows,
                'feature_manifest': 'outputs/qwen38-legacy-reference/feature-manifest.json',
                'sources': {'original_manifest_sha256': sha(original), 'm1_manifest_sha256': sha(original_m1),
                            'producer_sha256': sha(__file__), 'prompt_source_sha256': sha(ROOT / 'scripts/prepare_katago_training_sources.py')},
                'limitations': ['Previously viewed54 questions from3 games/18 positions; nonblind and correlated.',
                    'New prompt wording and new numerical/model recipe; historical old scores are not matched causal arms.',
                    'Secondary only; excluded from new training and primary14499-question validation.'],
                'model_loaded': False, 'tensor_payload_decoded': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    shutil.copyfile(__file__, output / 'producer-at-run.py')
    print(json.dumps({'questions': len(rows), 'positions': len(feature_records), 'games': 3,
                      'manifest_sha256': sha(output / 'manifest.json'), 'model_loaded': False}))


if __name__ == '__main__':
    main()
