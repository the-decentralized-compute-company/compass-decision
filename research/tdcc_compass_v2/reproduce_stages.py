"""Portable post-study helper: rerun the frozen recipe on supplied train/dev/cal splits.

This helper was not the historical launcher. Exact executed sources and logs are
included separately. It never reads final evaluation inputs or labels. A paused
run exits with status 75; remove STOP_REQUESTED only when intentionally resuming.
Keep a started workspace at the same path: historical run configs store the
absolute source-checkpoint path. Do not use this on the delivered frozen model.
"""
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

R = Path(__file__).resolve().parent


def check_stop():
    if (R / 'STOP_REQUESTED').exists():
        print('STOP_REQUESTED is present. Remove it only when intentionally resuming.', file=sys.stderr)
        raise SystemExit(75)


def run(args):
    check_stop()
    env = os.environ.copy()
    env.setdefault('OMP_NUM_THREADS', '6')
    env.setdefault('MKL_NUM_THREADS', '6')
    completed = subprocess.run([sys.executable, *map(str, args)], cwd=R, env=env, check=False)
    if completed.returncode:
        if completed.returncode == 75:
            print('Run paused. Its resumable checkpoint is preserved; no later stage was started.', file=sys.stderr)
        raise SystemExit(completed.returncode if completed.returncode > 0 else 128 - completed.returncode)
    check_stop()


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def expected_config(name, kind, hard, epochs, checkpoint):
    return dict(
        code_sha256={f: sha(R / f) for f in ('train_v2.py', 'engine.py', 'upstream/joint_schema_model.py')},
        stage=kind, hard_weight=hard, epochs=epochs, accumulation=4,
        data_sha256={part: sha(R / f'data/{part}.jsonl') for part in ('train', 'dev')},
        name=name, from_checkpoint=str(checkpoint) if checkpoint is not None else None,
    )


def stage(name, kind, hard, epochs, checkpoint=None):
    check_stop()
    out = R / 'artifacts' / name
    result = out / 'result.json'
    config = expected_config(name, kind, hard, epochs, checkpoint)
    config_path = out / 'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Configuration or source bytes changed; refusing to reuse stage: ' + name)
    if not result.exists():
        args = ['train_v2.py', '--name', name, '--stage', kind, '--hard-weight', hard, '--epochs', epochs,
                '--accumulation', 4, '--resume']
        if checkpoint is not None:
            args += ['--from-checkpoint', checkpoint]
        run(args)
    if not result.exists():
        raise RuntimeError('Stage exited without completing: ' + name)
    if not config_path.exists() or json.loads(config_path.read_text(encoding='utf-8')) != config:
        raise ValueError('Completed stage has missing or incompatible run configuration: ' + name)
    value = json.loads(result.read_text(encoding='utf-8'))
    if value.get('status') != 'stage_complete' or value.get('name') != name or value.get('stage') != kind:
        raise ValueError('Stage result identity or completion status is invalid: ' + name)
    for field in ('best_selection_score', 'best_hard_nll'):
        number = value.get(field)
        if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number):
            raise ValueError('Nonfinite or missing stage selection metric: ' + name + '/' + field)
    for file in ('joint_head.safetensors', 'joint_head_config.json', 'compass_cpu_config.json',
                 'adapter/adapter_model.safetensors', 'adapter/adapter_config.json'):
        if not (out / 'best' / file).is_file():
            raise FileNotFoundError('Completed stage is missing its best checkpoint: ' + name + '/' + file)
    return value


def main():
    check_stop()
    if (R / 'artifacts/selected').exists() or (R / 'artifacts/FROZEN_MANIFEST.json').exists():
        raise SystemExit('A frozen or partial selection already exists. Refusing to rerun or refit it; use a fresh workspace.')
    run(['cache_features.py', 'data/train.jsonl', 'data/dev.jsonl'])
    candidates = []
    for name, hard in (('head_soft', 0.0), ('head_hybrid', 0.35)):
        result = stage(name, 'head', hard, 3)
        candidates.append(dict(name=name, hard=hard, score=result['best_selection_score'], nll=result['best_hard_nll']))
    # Exactly the finalizer's rule: score, then lower hard NLL; a complete tie
    # retains the first declared candidate. Never use a stage name as a tiebreaker.
    chosen = max(candidates, key=lambda candidate: (candidate['score'], -candidate['nll']))
    lora = 'lora_' + chosen['name'].removeprefix('head_')
    stage(lora, 'lora', chosen['hard'], 2, R / 'artifacts' / chosen['name'] / 'best')
    # The unchanged finalizer compares all three completed development results,
    # commits selection, then calibrates. It does not check STOP during calibration.
    run(['finalize_v2.py', '--candidates', 'head_soft', 'head_hybrid', lora])


if __name__ == '__main__':
    main()
