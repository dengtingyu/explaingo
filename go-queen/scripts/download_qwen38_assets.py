#!/usr/bin/env python3
"""Download pinned public model assets on a cloud Linux host; never load a model.

This distribution helper is new and has not been executed in the recorded run.
The recorded trainer and its strict identity checks remain unchanged.
"""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'reproduction/qwen38-27b-20261014/model-assets.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    if sys.platform != 'linux' or not ROOT.is_relative_to(Path('/workspace')):
        raise RuntimeError('Run only in a cloud Linux project under /workspace')
    cfg = json.loads(CONFIG.read_text())
    out = ROOT / 'models'
    model_dir = out / 'Qwen3.8-27B'
    receipt = out / 'qwen-assets.local.json'
    hashes = out / 'qwen-asset-hashes.local.json'
    if model_dir.exists() or receipt.exists() or hashes.exists():
        raise FileExistsError('Existing model assets are retained; no automatic overwrite or retry')
    out.mkdir(exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=cfg['qwen']['model_id'], revision=cfg['qwen']['revision'],
                      local_dir=model_dir, allow_patterns=list(cfg['qwen']['files']), max_workers=4)
    for name, info in cfg['qwen']['files'].items():
        if Path(name).name != name:
            raise ValueError('Non-flat public model inventory')
        path = model_dir / name
        if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            raise ValueError('Downloaded Qwen asset differs: ' + name)
    record = {'status': 'completed', 'qwen_model_id': cfg['qwen']['model_id'],
              'qwen_revision': cfg['qwen']['revision'], 'qwen_download_complete': True,
              'qwen_local_remote_path': str(model_dir),
              'qwen_file_sizes': {n: x['bytes'] for n, x in cfg['qwen']['files'].items()}}
    write_new(receipt, record)
    write_new(hashes, {'model_id': record['qwen_model_id'], 'revision': record['qwen_revision'],
                      'asset_record_sha256': sha(receipt),
                      'files': {n: x['sha256'] for n, x in cfg['qwen']['files'].items()},
                      'file_sizes': record['qwen_file_sizes']})
    expert = cfg['katago']
    archive_path = out / expert['archive']['file']
    checkpoint = out / expert['checkpoint']['file']
    if archive_path.exists() or checkpoint.exists():
        raise FileExistsError('Existing KataGo assets are retained')
    with urllib.request.urlopen(expert['archive']['url'], timeout=600) as response, archive_path.open('xb') as target:
        for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
            target.write(chunk)
    if archive_path.stat().st_size != expert['archive']['bytes'] or sha(archive_path) != expert['archive']['sha256']:
        raise ValueError('KataGo archive differs')
    with zipfile.ZipFile(archive_path) as archive:
        candidates = [m for m in archive.infolist() if Path(m.filename).name == checkpoint.name]
        if len(candidates) != 1 or candidates[0].is_dir():
            raise ValueError('Expected exactly one named KataGo checkpoint in archive')
        with archive.open(candidates[0]) as source, checkpoint.open('xb') as target:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                target.write(chunk)
    if checkpoint.stat().st_size != expert['checkpoint']['bytes'] or sha(checkpoint) != expert['checkpoint']['sha256']:
        raise ValueError('Extracted KataGo checkpoint differs')
    print('Pinned public assets verified; no model was loaded and no training was started.')


if __name__ == '__main__':
    main()
