"""Clean-process inference parity against frozen calibration outputs, no test reads."""
import hashlib, json
from pathlib import Path
from predict_cpu import CompassCPU

R = Path(__file__).parent

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def main():
    rows = [json.loads(x) for x in (R/'data/calibration.jsonl').read_text().splitlines()]
    expected = {r['id']: r for r in json.loads((R/'artifacts/calibration_predictions.json').read_text())}
    selected = []
    for group, language in [('typed_decisions', 'en'), ('massive', 'tr'), ('snli', 'en')]:
        selected.append(next(r for r in rows if r['metadata']['dataset_group'] == group and r['metadata']['language'] == language))
    model = CompassCPU(R/'artifacts/selected')
    counts = []
    for row in selected:
        result = model.predict(row)
        n = 0
        for qid, saved in expected[row['id']]['answers'].items():
            actual = result['answers'][qid]
            assert list(actual['logits']) == saved['option_ids']
            assert list(actual['logits'].values()) == saved['logits'], (row['id'], qid, 'logit mismatch')
            assert actual['probabilities'] == saved['probabilities'], (row['id'], qid, 'probability mismatch')
            n += len(saved['option_ids'])
        counts.append({'id': row['id'], 'input_tokens': result['input_tokens'], 'exact_logits_and_probabilities': n})
    report = {'passed': True, 'scope': 'Three calibration records only; clean-process runtime parity, not a quality estimate.',
              'records': counts, 'config_sha256': sha(R/'artifacts/selected/compass_cpu_config.json'),
              'predictor_sha256': sha(R/'predict_cpu.py'), 'script_sha256': sha(__file__)}
    (R/'artifacts/reload_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
