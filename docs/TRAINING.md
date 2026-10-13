# Training and reproducibility boundaries

Inference does not require retraining. The published checkpoint is the development-selected head/adapter, with an already fitted calibration temperature.

The original full audit archive is `tdcc_compass_cpu_trained_v2.zip` (SHA-256 `6c3333f4c745e863dc7647b53ee5d83d7f8072d22ba828f51ab46d29dd513c47`). It preserves the exact converted train/development/calibration splits and historical evidence. This clean upload package deliberately excludes all corpus records, raw predictions, execution logs and optimizer/cache state. Therefore it is not a standalone full-data reproduction archive. Source preparation code is provided for inspection and adaptation; regenerating a different split is not exact reproduction.

If you possess and have separately extracted that verified audit archive:

```bash
python3 prepare_training.py --evidence-dir /path/to/tdcc_compass_cpu_trained_v2 --output ../compass_reproduction
cd ../compass_reproduction/tdcc_compass_v2
bash install_cpu.sh
.venv/bin/python download_base_model.py
.venv/bin/python test_engine.py
.venv/bin/python test_resume.py
.venv/bin/python reproduce_stages.py
```

The preparation command refuses to overwrite an existing destination. It checks the required frozen input files against a recorded input manifest. It does not start training. The stage runner performs three-epoch soft and hybrid head comparisons, selects by the fixed development rule, performs up to two LoRA epochs, freezes selection and calibrates. It does not read final-test labels.

Training is a long CPU job. The historical head comparisons took roughly 29 minutes after feature caching; LoRA plus scheduled development checks and one interruption took about 15 hours 39 minutes. Feature extraction and final evaluation are additional. These times are not hardware-independent promises.

For a safe optimizer-boundary pause, create `STOP_REQUESTED` in the reproduction experiment directory. Remove it only when intentionally resuming, then rerun `reproduce_stages.py`. Keep the workspace at the same path and preserve source/data/configuration while resuming. The already-started final calibration phase does not support interruption through that stop file. Never reuse a frozen release directory as a training destination.

For new datasets, provide appropriately licensed JSONL records with state, questions, gold targets and metadata in the expected schema, and establish new isolated development/calibration/test splits. Review `engine.py` and source-preparation code first. Do not adapt on the exposed final evaluation and call the rerun fresh.

`evaluation/evaluate_frozen.py --help` describes generic scoring on explicit supplied label/prediction files. The aggregate JSON reports in this repository do not contain sufficient per-record information to recreate every bootstrap by themselves. Exact historical statistical reproduction requires the full audit archive's frozen predictions and labels; its portable `evaluation/rescore_delivery.py` is the appropriate wrapper.
