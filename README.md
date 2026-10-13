# Compass CPU v2

The Decentralized Compute Company

Custom CPU runtime, training/selection sources and evaluation scorer for an experimental Qwen3.5-0.8B + LoRA + Clef typed-decision model.

**English benchmark: 66.58% versus v1's 52.50%; lexical baseline: 67.45%. Broader bilingual gate: FAIL.** NLI collapses to neutral and Turkish unseen-domain intent regresses. This is a research experiment, with no claim of state-of-the-art, production readiness, independent teacher truth or broad bilingual reliability. Read [the model card](docs/MODEL_CARD.md) and [evaluation report](evaluation/EVALUATION_REPORT.md).

## What belongs here

This folder is the GitHub repository root. It contains code, dependency pins, examples, tests, documentation, aggregate results and license notices. Weights, source-corpus records, raw predictions, logs, environments and downloaded bases are excluded. The companion `huggingface/` upload folder contains the complete selected head/adapter/tokenizer and a standalone copy of the custom runtime. Do not push the outer release folder or its ZIP into this Git repository.

## Quick start: companion model folder

Run these commands from this `github/` folder after extracting the combined release:

```bash
python3 verify_files.py
python3 -m unittest discover -s tests -v
bash install_cpu.sh
.venv/bin/python download_base_model.py
.venv/bin/python predict_cpu.py --checkpoint ../huggingface --input examples/example_en.jsonl --output predictions.jsonl
```

For a standalone clone after the model is published, download the complete companion model into the ignored `model/` directory first:

```bash
.venv/bin/hf download YOUR_HF_NAMESPACE/compass-v2 --local-dir model
.venv/bin/python predict_cpu.py --checkpoint model --input examples/example_en.jsonl --output predictions.jsonl
```

The placeholder is not a claim that a repository already exists. Authentication is only needed if the selected model repository is private. Keep tokens out of source files and shell-history commands.
## CPU runtime

Recorded environment: Python 3.12.14, torch 2.8.0+cpu, Transformers 5.2.0, PEFT 0.19.1. Linux CPU execution is tested; other platforms and devices are not validated. Use a CPU supporting BF16 operations, several GB of RAM, and space for dependencies plus the approximately 1.7 GB base model. Speed depends on CPU kernels and hardware.

```bash
bash install_cpu.sh
.venv/bin/python download_base_model.py
```

The downloader fetches only the pinned official `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17` and checks the recorded SHA-256 of the model shard, config and index. No token is required for that public base. Do not substitute another base or revision. The tokenizer shipped with the selected checkpoint is used for inference.

The runtime loads `Qwen3_5ForConditionalGeneration`, attaches the LoRA adapter specifically to `base.model.language_model`, and loads the full FP32 `JointSchemaHead`. CPU backbone/autocast is BF16. This is a custom typed-classification runtime. `AutoModel.from_pretrained`, `AutoPeftModel.from_pretrained`, generic `pipeline`, standard chat generation, GGUF and Ollama loading are not supported entry points. It does not use `trust_remote_code` or silently execute code downloaded from a model repository. Review and run the provided Python runtime explicitly.

Input: JSONL records with `state` and a nonempty `questions` object. Question types are `choice` (option dictionary), Boolean `noul`, and ordinal `score` (ordered criteria list, indexed from zero). Output includes raw logits, uncalibrated and temperature-scaled probabilities, chosen answers, and input token count. The temperature is 0.7071067811865475. Inputs over 2,048 tokens are rejected; they are never silently truncated. Probabilities are not correctness or safety guarantees. Examples are format demonstrations, not new benchmarks.

## Source layout

- `predict_cpu.py`: explicit custom model loader and JSONL CLI
- `upstream/`: unchanged Clef head plus original license
- `research/tdcc_compass_v2/`: exact training engine, bounded/resumable training, development selection and calibration sources
- `research/tdcc_compass_bilingual/`: source-download, schema and conversion/audit code
- `evaluation/evaluate_frozen.py`: scorer accepting explicit label/prediction paths
- `evaluation/`: aggregate historical results and registered evaluation protocol
- `prepare_training.py`: prepares a new training workspace using separately restored frozen evidence data
- `tests/`: lightweight dependency-free packaging/runtime-import checks

See [training documentation](docs/TRAINING.md). Code-only publication does not include the corpora required for exact retraining. Do not run training into delivered weights. No retraining or public upload is part of packaging.

## Verification

`python3 verify_files.py` checks every shipped file in this folder against `CHECKSUMS.sha256`. `python3 -m unittest discover -s tests -v` runs packaging tests without downloading anything. The evaluator's focused tests can additionally run with NumPy: `python3 -m unittest discover -s evaluation -p test_evaluator.py -v`. Historical data-dependent training tests run only in the separately prepared workspace. See `PACKAGING_VALIDATION.json` for what was actually checked in the release.

## License scope

Original glue and documentation are Apache-2.0; preserve `NOTICE` and all third-party notices. This grant does not establish a trained-weight or source-dataset license. Read [license status](LICENSE_STATUS.md) before public model distribution. No personal author name or credential is embedded in this repository. Official source attributions remain intact.
