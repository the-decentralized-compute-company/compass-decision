"""Lightweight public-source tests; no model download, imports or training needed."""
import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_python_sources_parse(self):
        for path in ROOT.rglob('*.py'):
            if not any(p in {'.venv', 'base_model', 'model'} for p in path.parts):
                ast.parse(path.read_text(encoding='utf8'), filename=str(path))

    def test_lightweight_runtime_import(self):
        command = [sys.executable, '-B', '-c',
                   'import predict_cpu; assert hasattr(predict_cpu, "CompassCPU")']
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True)

    def test_command_help(self):
        for script in ('predict_cpu.py', 'download_base_model.py', 'prepare_training.py', 'verify_files.py'):
            completed = subprocess.run([sys.executable, '-B', script, '--help'], cwd=ROOT,
                                       check=True, capture_output=True, text=True)
            self.assertIn('usage:', completed.stdout)

    def test_examples_cover_three_types(self):
        for language in ('en', 'tr'):
            rows = [json.loads(line) for line in
                    (ROOT / f'examples/example_{language}.jsonl').read_text().splitlines()]
            self.assertTrue(rows)
            self.assertEqual({q['type'] for q in rows[0]['questions'].values()},
                             {'choice', 'noul', 'score'})

    def test_base_pin_matches_provenance(self):
        provenance = json.loads((ROOT / 'PROVENANCE.json').read_text())
        spec = importlib.util.spec_from_file_location('download_base', ROOT / 'download_base_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.MODEL_REVISION, provenance['base_revision'])
        self.assertEqual(module.MODEL_ID, provenance['base_model'])
        self.assertEqual(len(module.load_checksums(ROOT)), 3)

    def test_model_limitations_visible(self):
        card = (ROOT / 'docs/MODEL_CARD.md').read_text()
        for value in ('66.58%', '52.50%', '67.45%', '25.00%', 'FAIL', 'teacher', 'not supported'):
            self.assertIn(value, card)

    def test_training_helper_refuses_missing_inputs(self):
        spec = importlib.util.spec_from_file_location('prepare_training', ROOT / 'prepare_training.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(ValueError):
                module.prepare(root / 'missing', root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_verifier_rejects_changed_bytes(self):
        spec = importlib.util.spec_from_file_location('verify_files', ROOT / 'verify_files.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'sample.txt').write_text('correct')
            digest = hashlib.sha256(b'correct').hexdigest()
            (root / 'CHECKSUMS.sha256').write_text(digest + '  sample.txt\n')
            self.assertEqual(module.verify(root), 1)
            (root / 'sample.txt').write_text('changed')
            with self.assertRaises(ValueError):
                module.verify(root)


if __name__ == '__main__':
    unittest.main()
