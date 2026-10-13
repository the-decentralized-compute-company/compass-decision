# Compass CPU v2

The Decentralized Compute Company

**Experimental research checkpoint. English benchmark gate: PASS. Broader bilingual gate: FAIL.**

This repository contains the actual selected LoRA adapter, the trained Clef joint-schema head, tokenizer, configuration and custom CPU inference runtime. The pinned Qwen base weights are downloaded separately. The adapter alone is not the complete model.

## Results and limitations

On the same 368 previously unused English test records (1,840 typed decisions), v2 obtained **66.58%** decision accuracy versus **52.50%** for v1: **+14.08 percentage points**, paired 95% interval **+10.60 to +17.45**. V2's own accuracy interval is 64.13–68.97%. Original records are synthetic and their labels are model-teacher labels; this measures agreement with those labels, not independently verified real-world correctness. Only 17.93% of records had all five decisions correct.

A lighter state-aware lexical diagnostic reached **67.45%**. V2 minus lexical is −0.87 points (paired 95% interval −3.32 to +1.58). Neural superiority and state-of-the-art performance are not demonstrated.

| Evaluation slice | V2 English | V2 Turkish |
|---|---:|---:|
| Original Typed Decisions | 66.58% | No coverage |
| MASSIVE seen-domain intent | 77.50% | 61.67% |
| MASSIVE unseen-domain intent | 33.33% | 41.67% |
| SNLI / SNLI-TR | 25.00% | 25.00% |

The broader bilingual gate failed. NLI fell from v1's 41% to 25% in both languages, below the 34% training-prior baseline; v2 predicts neutral for all 200 NLI examples. Turkish unseen-domain intent regressed from 63.33% to 41.67%. Turkish evaluation covers choice classification only, with zero independent Turkish Boolean (`noul`) or ordinal (`score`) validation. Turkish SNLI is machine-translated source data, not newly human-validated by this project.

Original English type accuracy: choice 64.49%, Boolean 73.55%, ordinal 62.91%. Some fields collapse to a constant answer: invoice duplicate=false and security disposition=investigate on their 92-record slices. English calibrated hard NLL improves over v1 (0.96301 to 0.76127), but 10-bin ECE worsens (4.60% to 7.82%). Aggregate accuracy must not imply consistent field discrimination or uniformly improved probability reliability.

No production-readiness, autonomous consequential decision, general Turkish, broad bilingual reliability, adversarial robustness, or universal unseen-schema claim is supported. Do not use this model as an autonomous authority for medical, legal, financial, employment, security or other high-impact decisions. Evaluate any proposed application separately with meaningful human oversight. Base-model pretraining contamination cannot be certified absent.

## Architecture and training

- Base: Qwen/Qwen3.5-0.8B, exact revision `2fc06364715b967f1860aea9cf38778875588b17`; 852,985,920 parameters.
- Head: unmodified Apache-licensed Cloudflare Clef reference from Unsloth source commit `0b1291bfd7c147e2edf82c085b8412c2630be4fe`; 27,324,932 parameters. Hidden size 1024, width 512, two routing layers, four layers, eight heads, feedforward 2048.
- Adapter: 5,411,328 parameters, LoRA rank 8, alpha 16, dropout 0, all linear text modules.
- Training: 1,859 records / 5,695 decisions: 959 original English Typed Decisions records, 600 MASSIVE intent records, 300 SNLI/SNLI-TR records. Development and calibration each have 360 records / 840 decisions. No private corpus, GPU, paid teacher API, or separately authored DecisionMix corpus was used.
- Selection: compare soft-target head objective with 65% soft / 35% hard objective for three epochs; select soft on the development rule, then train LoRA/head up to two epochs. Selected checkpoint is epoch 2, record position 932, not terminal weights. Selected lineage received 2,093 accepted optimizer updates and 8,368 record exposures.
- One global temperature was fitted on separate calibration hard-label NLL, using a fixed 161-point grid from 0.25 to 4. Calibration hard NLL changed from 0.83524 to 0.81932, with no accuracy change. Soft-teacher cross-entropy worsened. Training token cap was 1,536; inference cap is 2,048.
- All final results were scored after checkpoint/calibration freeze. Translation pairs and relevant source families were grouped across partitions. One source-factor-overlap training record was removed before optimization. This separate held-out evaluation pass is part of the same project, not an external institutional audit.

The complete historical evidence archive was checksum-verified before this repackaging. `evaluation/` contains aggregate reports, gate verdict and protocol; training/test corpus records, raw predictions, logs and optimizer/cache files are intentionally absent here. Repackaging did not retrain, select or tune the model. `PROVENANCE.json` records unchanged weight hashes and configuration-only/path changes. `PACKAGING_VALIDATION.json` describes checks on this actual repository layout.

The original 928 final evaluation records are now exposed. A later revision needs a newly isolated test for a fresh confirmatory claim; rerunning the existing test is not a new untouched evaluation.
## CPU runtime

Recorded environment: Python 3.12.14, torch 2.8.0+cpu, Transformers 5.2.0, PEFT 0.19.1. Linux CPU execution is tested; other platforms and devices are not validated. Use a CPU supporting BF16 operations, several GB of RAM, and space for dependencies plus the approximately 1.7 GB base model. Speed depends on CPU kernels and hardware.

```bash
bash install_cpu.sh
.venv/bin/python download_base_model.py
```

The downloader fetches only the pinned official `Qwen/Qwen3.5-0.8B` revision `2fc06364715b967f1860aea9cf38778875588b17` and checks the recorded SHA-256 of the model shard, config and index. No token is required for that public base. Do not substitute another base or revision. The tokenizer shipped with the selected checkpoint is used for inference.

The runtime loads `Qwen3_5ForConditionalGeneration`, attaches the LoRA adapter specifically to `base.model.language_model`, and loads the full FP32 `JointSchemaHead`. CPU backbone/autocast is BF16. This is a custom typed-classification runtime. `AutoModel.from_pretrained`, `AutoPeftModel.from_pretrained`, generic `pipeline`, standard chat generation, GGUF and Ollama loading are not supported entry points. It does not use `trust_remote_code` or silently execute code downloaded from a model repository. Review and run the provided Python runtime explicitly.

Input: JSONL records with `state` and a nonempty `questions` object. Question types are `choice` (option dictionary), Boolean `noul`, and ordinal `score` (ordered criteria list, indexed from zero). Output includes raw logits, uncalibrated and temperature-scaled probabilities, chosen answers, and input token count. The temperature is 0.7071067811865475. Inputs over 2,048 tokens are rejected; they are never silently truncated. Probabilities are not correctness or safety guarantees. Examples are format demonstrations, not new benchmarks.

See the repository README for model download and inference commands.
