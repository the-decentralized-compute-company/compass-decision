"""Prepare a new training workspace from separately restored, checksum-verified data.

This post-study helper does not download data or start training.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(evidence, output):
    evidence = Path(evidence).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError('Output already exists; choose a new directory. Nothing overwritten.')
    if output.is_relative_to(ROOT) or output.is_relative_to(evidence):
        raise ValueError('Choose an output outside the repository and evidence archive.')
    entries = json.loads((ROOT / 'training_input_checksums.json').read_text())
    for name, wanted in entries.items():
        name_path = PurePosixPath(name)
        if name_path.is_absolute() or '..' in name_path.parts:
            raise ValueError('Unsafe input-manifest path')
        source = evidence / name
        if not source.is_file() or digest(source) != wanted:
            raise ValueError('Missing or mismatched frozen input: ' + name)
    target = output / 'tdcc_compass_v2'
    # Complete preflight before making any new directories.
    code = list((ROOT / 'research/tdcc_compass_v2').glob('*.py'))
    if not code or not (ROOT / 'upstream/joint_schema_model.py').is_file():
        raise ValueError('Incomplete training source tree')
    target.mkdir(parents=True)
    for source in code:
        shutil.copy2(source, target / source.name)
    for name in ('predict_cpu.py', 'install_cpu.sh', 'requirements-cpu.lock.txt',
                 'requirements.txt', 'download_base_model.py', 'base_model_checksums.json'):
        shutil.copy2(ROOT / name, target / name)
    shutil.copytree(ROOT / 'upstream', target / 'upstream',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'docs/TRAINING.md', target / 'README_TRAINING.md')
    for name in entries:
        # Original research tree uses sibling v2 / prior-pilot directories.
        relative = PurePosixPath(name).relative_to('research')
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(evidence / name, destination)
    for name in ('artifacts', 'feature_cache', 'logs'):
        (target / name).mkdir()
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = prepare(args.evidence_dir, args.output)
    print('Prepared:', target)
    print('No training started. Read README_TRAINING.md before continuing.')


if __name__ == '__main__':
    main()
