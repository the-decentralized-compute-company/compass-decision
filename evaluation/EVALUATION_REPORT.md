> Historical study report. Evidence filenames mentioned below refer to the original complete audit archive, which is not duplicated in this source/model upload package. This is a separate evaluation pass within the same project, not an external institutional audit.

# Compass-0.8B v2: separately executed held-out evaluation

Completed 2026-10-13T04:31:39.252378+00:00. Weights and calibration were frozen before these results.

## Verdict

- Primary English teacher-benchmark performance gate: **PASS**. V2: **66.58%**, v1 on the same fresh records: **52.50%**. Paired gain +14.08 pp, 95% CI [+10.60, +17.45] pp.
- State-aware lexical comparator: **67.45%**. V2 minus lexical: -0.87 pp, paired 95% CI [-3.32, +1.58] pp. This experiment does not demonstrate an accuracy advantage over the simpler comparator.
- Secondary bilingual intent/NLI gate: **FAIL**. NLI and Turkish unseen-domain intent regressions fail the preset safeguards. NLI is 25% in both languages, down from v1’s 41%; Turkish unseen-domain intent is 41.67%, down from 63.33%. V2 predicts `neutral` for every NLI example. Report seen-domain intent gains separately from these failures.
- Integrity checks pass: frozen dataset/checkpoint/base/code hashes, independent selection and calibration reconstruction, full-precision prediction validation, and clean-process reload.
- Interpretation: a reproducible experimental improvement over v1 on a limited English benchmark, with large seen-domain bilingual intent gains and substantial regressions on NLI and Turkish unseen-domain intent. General bilingual typed decisions, correctness probabilities, and consequential autonomous use are unvalidated.

## 1. Fresh English workflow benchmark

All 368 previously unused official test records, 1,840 decisions, 92 records per workflow. The 32 exposed v1 pilot records are excluded. Targets are synthetic teacher labels, so accuracy means teacher agreement. Never compare v2 to the old 45.625% pilot number as if the sample were matched.

| Model | Accuracy | Record exact match | Hard NLL | Soft CE | Brier sum | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|---:|
| Compass v2 | 66.58% | 17.93% | 0.76127 | 0.96877 | 0.11653 | 0.07822 |
| Compass v1 | 52.50% | 7.61% | 0.96301 | 1.04441 | 0.16127 | 0.04596 |
| V1 train soft-schema prior | 50.16% | 10.05% | 1.03396 | 1.09569 | 0.18393 | 0.06552 |
| V2 train soft-schema prior | 50.49% | 10.05% | 1.03165 | 1.09320 | 0.18258 | 0.05448 |
| V2 train hard-majority prior | 48.91% | 7.34% | 1.01455 | 1.13547 | 0.19817 | 0.03376 |
| Uniform | 27.77% | 0.00% | 1.21182 | 1.21182 | 0.24028 | 0.04739 |
| TF-IDF + logistic diagnostic | 67.45% | 20.92% | 0.75741 | 1.20899 | 0.18128 | 0.06128 |

V2 accuracy 95% CI: [64.13, 68.97]%. The 65% criterion was a point-estimate threshold; this interval does not establish a population accuracy floor of 65%.

| Paired comparison: v2 minus baseline | Point gain | Paired 95% CI | Familywise 95% lower bound |
|---|---:|---:|---:|
| Compass v1 | +14.08 pp | [+10.60, +17.45] pp | +10.05 pp |
| V1 train soft-schema prior | +16.41 pp | [+13.37, +19.46] pp | +12.88 pp |
| V2 train soft-schema prior | +16.09 pp | [+13.15, +18.97] pp | +12.72 pp |
| V2 train hard-majority prior | +17.66 pp | [+14.51, +20.76] pp | +14.08 pp |
| TF-IDF + logistic diagnostic | -0.87 pp | [-3.32, +1.58] pp | Diagnostic only |

The registered gate requires accuracy ≥65%, gain ≥5 points over each of four registered baselines, a familywise paired lower bound above zero, no workflow loss over 5 points against its strongest registered baseline, and integrity checks. Lexical was frozen as a disclosed additional diagnostic, not retroactively inserted into or removed from the gate.

| Workflow (460 fields each) | V2 | V1 | Best registered prior | Lexical |
|---|---:|---:|---:|---:|
| agent_trace_observability | 58.91% | 40.87% | 36.09% | 65.00% |
| customer_service | 75.43% | 55.00% | 49.13% | 71.74% |
| invoice_processing | 64.57% | 54.57% | 59.13% | 65.65% |
| security_incidents | 67.39% | 59.57% | 59.78% | 67.39% |

## 2. Bilingual classification evidence

560 decisions from 280 distinct source families, with paired English/Turkish examples. MASSIVE uses the same full-60-intent schema throughout; the three unseen domains (general, news, datetime) have no labeled training/development/calibration examples, but their option descriptions were visible. SNLI-TR is source-corpus machine translation and was not independently human-validated here. No final-test quality or length filter was applied.

| Language/task | Decisions | V2 | V1 | Strongest train prior | Lexical |
|---|---:|---:|---:|---:|---:|
| en/massive_seen_domain | 120 | 77.50% | 33.33% | 6.67% | 47.50% |
| en/massive_unseen_domain | 60 | 33.33% | 36.67% | 0.00% | 0.00% |
| en/snli | 100 | 25.00% | 41.00% | 34.00% | 47.00% |
| tr/massive_seen_domain | 120 | 61.67% | 25.00% | 6.67% | 45.83% |
| tr/massive_unseen_domain | 60 | 41.67% | 63.33% | 0.00% | 0.00% |
| tr/snli | 100 | 25.00% | 41.00% | 34.00% | 46.00% |

Language aggregates are descriptive across these particular tasks:
- en: 49.29%, 95% CI [44.64, 53.93]%; secondary gate FAIL. Conditions: at_least_100_source_groups=True, lift_at_least_5pp_each_baseline=True, paired_ci95_lower_above_zero_each_baseline=True, no_task_loss_over_5pp=False
- tr: 44.29%, 95% CI [39.29, 49.29]%; secondary gate FAIL. Conditions: at_least_100_source_groups=True, lift_at_least_5pp_each_baseline=True, paired_ci95_lower_above_zero_each_baseline=False, no_task_loss_over_5pp=False

NLI must not be described as a learned strength. V2 chooses `neutral` in all 100 English and all 100 Turkish cases; only 25 source labels are neutral. The train-prior tie rule chooses contradiction and gets 34%. The lexical diagnostic achieves 46.5% pooled EN/TR NLI. The aggregate bilingual improvement cannot erase this task-specific regression.

Turkish unseen-domain intent regresses from 63.33% to 41.67%; English unseen-domain intent changes from 36.67% to 33.33%. Do not describe these held-out-domain results as gains over v1. The Turkish language aggregate also fails its paired confidence requirement against v1.

Turkish training and testing contain choice classification only: 450 training choices and 280 test choices. Turkish `noul` and `score` support is zero. No broad Turkish typed-decision claim is warranted.

## 3. Probability scaling and validation

Temperature 0.7071067811865475 was selected on the fixed separate 360-record/840-field calibration set, using the preregistered 161-point hard-NLL grid. Calibration hard NLL: 0.835237 → 0.819317. This is a fitted score transformation, not a correctness guarantee.

| Public v2 metric | Raw logits/temperature 1 | Saved calibrated model |
|---|---:|---:|
| accuracy | 0.665761 | 0.665761 |
| hard_nll | 0.812691 | 0.761268 |
| soft_cross_entropy | 0.959418 | 0.968770 |
| kl_from_target | 0.195133 | 0.204485 |
| brier_sum | 0.112390 | 0.116528 |
| ece_10_equal_width | 0.133120 | 0.078219 |

Clean-process delivery predictor reload matched 388 probabilities across 14 preregistered final records, max difference 0, tolerance 2e-06. Boolean/integer/string outputs, all three types, malformed inputs, and overlength rejection passed. Every final output is present; none were truncated, dropped, or selected by outcome.

## 4. Failure patterns and scope

- en/invoice_processing/duplicate: always predicts `false` across 92 cases; accuracy 89.13%. Probability variation does not establish top-label discrimination.
- en/security_incidents/disposition: always predicts `investigate` across 92 cases; accuracy 75.00%. Probability variation does not establish top-label discrimination.
- en/caption_nli/relation: always predicts `neutral` across 100 cases; accuracy 25.00%. Probability variation does not establish top-label discrimination.
- tr/caption_nli/relation: always predicts `neutral` across 100 cases; accuracy 25.00%. Probability variation does not establish top-label discrimination.

The data audit found one original public training/test generator-factor duplicate despite different rendered states. That training record was removed before optimization; the 368-record primary test stayed unchanged. Removing the corresponding test record only as a disclosed sensitivity check yields:
- Compass v2: 66.59%
- Compass v1: 52.59%
- TF-IDF + logistic diagnostic: 67.47%

Remaining limits: related synthetic templates can span public splits; teacher disagreement is not human ground truth; source-group bootstrap uncertainty may be optimistic under residual correlations; base-model pretraining contamination cannot be certified absent; only a small fixed set of task schemas was evaluated. Unseen-domain intent examples do not establish universal unseen-schema reliability.

## 5. Reproduction and audit trail

- Selected lineage: three-epoch soft-target frozen head, followed by the LoRA checkpoint at epoch 2, record position 932. The later terminal checkpoint was correctly rejected under the registered accuracy-first rule, despite lower NLL.
- Completed experiment updates: head soft 1,395; head hybrid 1,395; LoRA 930. These include discarded candidates. Dataset: 1,859 unique training records, 360 development, 360 calibration.
- An execution transport interruption was resumed from a saved optimizer/RNG checkpoint. Replayed loss matched exactly; attempted duplicate progress rows are not double-counted. The independent stop/resume control-flow tests also pass.
- Metrics use normalized full-precision probabilities; rounded source target probabilities are renormalized. Brier is summed over classes, then averaged over decisions; ECE uses 10 equal-width top-confidence bins. NLL/KL calculation clips prediction probabilities at 1e-12.
- Paired uncertainty uses 20,000 deterministic within-domain source-group bootstrap resamples, seed 48021. A workflow record’s five fields stay together; translation pairs stay together. The four primary one-sided lower bounds use Bonferroni familywise 95% coverage. Slice comparisons are descriptive unless explicitly stated.
- Full results: `public_results.json`, `bilingual_results.json`. Preregistration: `PROTOCOL.md`, `AMENDMENTS.md`. Integrity: `final_freeze_audit.json`, `full_precision_inference_audit.json`, `final_reload_audit.json`, `final_partition_audit.json`, `development_selection_audit.json`.
- Portable statistical scorer: `evaluate_frozen.py`; its CLI accepts explicit label and prediction paths. Workspace reconstruction/audit helpers preserve the original run paths and are not a claim of one-command portability on another computer.
- A mechanical post-inference scoring-wrapper fix coerced file paths to Path objects after its first orchestration call stopped before producing results. Prediction files, statistical scorer, bootstrap settings, gates, data and models were unchanged; no inference was repeated. See `scoring_wrapper_fix.json`.
- Publication was not performed. Preserve dataset/model attribution and license notices when preparing any distribution.

## 6. Observed CPU inference cost

On all 928 records, v2 took 41.09 minutes and v1 41.67 minutes, excluding model loading. Both used six CPU threads and one record per forward. The lexical pipeline took 2.49 seconds including loading with one thread and batched sparse features. These are observed implementation costs; batching/loading conventions differ, so no universal speed ratio is claimed. See `inference_runtime.json`.

All 928 final records are now exposed evaluation data. A later model revision must use a newly isolated test for a fresh confirmatory improvement claim; these records must not become a tuning set and then be relabeled as fresh.
