"""Fetch the pinned public base and verify the supplied training asset checksums.

Post-study delivery helper, not the historical download command. Importing this
module does not download anything. Use --verify-only for a fully offline check.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

R = Path(__file__).resolve().parent
MODEL_ID = 'Qwen/Qwen3.5-0.8B'
MODEL_REVISION = '2fc06364715b967f1860aea9cf38778875588b17'


def load_checksums(root):
    entries = json.loads((root / 'base_model_checksums.json').read_text(encoding='utf-8'))
    if not isinstance(entries, dict) or not entries:
        raise ValueError('Base model checksum manifest must be a nonempty mapping.')
    for name, wanted in entries.items():
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or '..' in path.parts
                or '\\' in name or str(path) != name
                or not isinstance(wanted, str) or not re.fullmatch('[0-9a-f]{64}', wanted)):
            raise ValueError('Invalid base model checksum entry: ' + repr(name))
    if 'config.json' not in entries or not any(name.endswith('.safetensors') for name in entries):
        raise ValueError('Checksum manifest must include the base config and model weights.')
    return entries


def verify(root, entries):
    for name, wanted in entries.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError('Missing pinned base model asset: ' + str(path))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != wanted:
            raise ValueError('Base model checksum mismatch: ' + name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=R / 'base_model', help='Pinned base model destination directory')
    parser.add_argument('--verify-only', action='store_true', help='Check local assets without network access')
    args = parser.parse_args(argv)
    entries = load_checksums(R)
    if not args.verify_only:
        os.environ.setdefault('HF_HOME', str(R / 'hf_cache'))
        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
        from huggingface_hub import snapshot_download
        snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, local_dir=args.output,
            allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE', 'README.md', *entries],
            max_workers=3, token=False,
        )
    verify(args.output, entries)
    print(f'Pinned public base model verified against {len(entries)} supplied SHA-256 checksums.')


if __name__ == '__main__':
    main()
