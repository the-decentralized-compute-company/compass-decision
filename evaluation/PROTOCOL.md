# Compass-0.8B v2 independent evaluation protocol

Preregistered 2026-10-12 before v2 training outcomes or any fresh-test model scores. This is an experimental quality gate, not a guarantee of usefulness, correctness, or publication readiness.

## Ownership and isolation

- Evaluator owns `historical-study/tdcc_compass_v2_eval`. Training workers must not inspect `*_labels.jsonl` or final scores before committing one selected checkpoint and inference/calibration configuration.
- Primary sample: every original Typed Decisions test record except the 32 records already scored in the v1 pilot. Exactly 368 records, 1,840 decisions, 92 records per each of four English workflows. IDs and SHA256 fingerprints are in `public_manifest.json`. No sampling, replacing hard examples, or result-dependent exclusion.
- The old 32 test records and all old bilingual smoke fixtures are permanently exposed development/diagnostic data. Never describe their reuse as fresh evaluation.
- Original train-origin records alone may enter train/dev/calibration. Translation, paraphrase, original image/caption families, and duplicate source examples must remain in a single partition. No label-dependent distractor construction.
- Any broken/overlength/inference-failed test record remains in the denominator as incorrect. Report it; do not silently truncate or remove it. If a genuine implementation defect requires repair, disclose the change and treat the repaired re-run as a repeated, no-longer-pristine test.
- Freeze checkpoint hashes, code hashes, training IDs/data hashes, exact tokenizer/base revision, hyperparameters, choice of calibration objective/temperature, max input length, and prior construction before final inference. Training chooses its checkpoint solely on predesignated development data. Calibration is fit on the separate calibration set. Bound and log attempted development configurations.

## Confirmatory comparisons

On the same 368 examples, compare the single selected v2 checkpoint with:

1. Read-only v1 selected checkpoint, with its saved temperature.
2. The v1 train-only soft-label schema prior.
3. The v2 train-only soft-label schema prior.
4. The v2 train-only hard-label majority schema prior.

A schema key is the exact question ID and question JSON, including instructions and option meanings; no grouping or feature uses test labels. Option order follows the actual encoder: `true,false` for noul; lexicographic option ID for choice; numerical order for score. Priors normalize teacher probabilities before accumulation. Hard priors use Laplace smoothing (one pseudocount per option); argmax is unchanged except deterministic ties. Unseen schemas fall back to uniform and their coverage is reported. Also report uniform separately, never as the sole baseline. If a content-aware non-neural baseline is developed, freeze it on train/dev and report it as a diagnostic rather than quietly changing the registered four.

## Metrics and statistical analysis

Primary metric: exact agreement with the provided teacher's hard label, micro-averaged over the fixed 1,840 decisions. The benchmark is synthetic and teacher-labeled; accuracy here is teacher agreement, not verified real-world correctness.

Report all of the following for every model/baseline: decision accuracy, record all-fields accuracy, hard-label negative log likelihood, normalized-soft-target cross entropy, KL(target || prediction), sum-over-classes Brier score, and 10 equal-width top-confidence ECE. Report unscaled and calibration-scaled probability metrics for v2; positive temperature must not change argmax. ECE is descriptive and not a calibrated correctness guarantee.

Report support and metrics by source/task, language, workflow/domain, question type and question ID. Report label and prediction distributions, per-schema prior coverage, malformed output counts, length/rejection counts, and nonfinite values. Preserve full precision in saved logits/probabilities; format rounding is presentation-only. All tie-breaking uses encoder option order.

Uncertainty: 20,000 deterministic paired, workflow-stratified bootstrap resamples of source-record groups, seed 48021. A record's five decisions remain together; translated pairs remain together for bilingual analyses. Report percentile 95% confidence intervals and paired lifts against each registered baseline. Use a one-sided Bonferroni familywise 95% lower bound across the four primary paired comparisons (1.25th percentile per comparison). These intervals quantify finite-sample uncertainty under source-group sampling; related synthetic templates and teacher noise may make them optimistic. Do not treat the 1,840 fields as independent Bernoulli trials. No significance fishing across slices. Small slices are descriptive.

## Preset improved-model gate

All conditions are required for the specific claim **improved on the held-out Typed Decisions subset**:

- At least 65% primary exact-label accuracy, a preset practical target for this experiment, not a universal quality standard.
- At least +5 percentage points above each of the four registered baselines.
- Familywise lower confidence bound of paired accuracy improvement strictly above zero for each baseline.
- In each of the four workflows, v2 accuracy no more than 5 points below the strongest registered baseline on that workflow. Per-domain comparisons are safeguards, not individually powered superiority claims.
- No test contamination, nonfinite predictions, malformed distributions, silent truncation, missing outputs, irreproducible checkpoint loading, or unexplained artifact mismatches.

If the gate fails, report failure and actual scores. Do not reduce targets, hide baselines, change the sample, combine smoke data into the primary score, or reuse this test to select another checkpoint. An honest experimental release can still document a failed result; it must not be advertised as a proven improvement or production-ready model.

## Bilingual / unseen-domain evidence

Separate secondary evaluation, never pooled into the primary score:

- Proposed MASSIVE 1.1 official-test: 120 original IDs paired EN/TR from trained domains, plus 60 paired IDs from three untouched domains selected by lexicographic SHA256(domain name) rank, before observing outcomes. Untouched domains are excluded from train/dev/calibration. Fixed domain-specific candidate schemas, no gold-dependent options.
- Proposed SNLI/SNLI-TR 1.1 official-test: 100 original source groups paired EN/TR, family-disjoint from train/dev/calibration. Label-preserving translation is a dataset assertion, not independent Turkish human verification.
- Data worker must confirm available counts/source versions and freeze deterministic hash-selected IDs before training; any amendment is logged before outcomes. Default each-language target is 280 records. Source family grouping can require changing the train set rather than trimming inconvenient test records.
- A bounded bilingual-improvement claim requires at least 100 independent held-out source groups per language, +5 points versus v1 and all matching training priors in each language, paired 95% lower bound above zero in each language, and no task loses more than 5 points against its strongest baseline. These are secondary tests, not a substitute for the primary gate. Report the unseen-domain slice separately and do not imply unseen-domain superiority unless it independently beats its baselines with uncertainty.
- MASSIVE measures intent selection with a supplied domain; SNLI measures NLI. Neither establishes general Turkish workflow decisions, fluent generation, safety, real-world correctness, or unseen-schema universality. Any authored synthetic Turkish fixtures remain smoke tests.

## Integrity and delivery checks

- Audit exact states, normalized text, source IDs, translation groups, schemas, factor/template families and input-length distributions across partitions. Report public train/test related-template exposure as a limitation, even when record IDs are disjoint.
- Confirm gold, IDs and metadata are not rendered into model inputs. Read inference and encoding code, not only logs.
- Repeat a preselected output sample in a clean process after saving; require typed output equality and full-precision probability agreement within explicit BF16 tolerance. Check the tokenizer/checkpoint/base hashes. Verify all declared train steps/selected lineage.
- Keep v1 weights read-only. Do not run heavy model inference concurrently with training without coordinating resources.
- No publication, paid API or uploading data/model artifacts is authorized here. Deliver locally with a clear model card, data licenses/provenance, reproducible commands, results JSON and disclosure of all failed gates.
