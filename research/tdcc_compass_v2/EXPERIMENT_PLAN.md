# Compass CPU v2: development-only selection plan

Written before any v2 fresh-test inference. No external teacher calls, paid compute, or publication.

- Pinned Qwen/Qwen3.5-0.8B backbone, exact unchanged Cloudflare Clef reference schema head, rank8/alpha16 all-linear text LoRA. No model substitute.
- Original public training partition: 959 train,120dev,120calibration, all from original1200train. Four workflows balanced. Dev/cal exclude all v1 training IDs.
- Add independently sourced and documented public TR/EN training/development/calibration partitions; preserve official splits and translation groups. Freeze input checksums before optimization.
- Independent evaluator seals all368unused original-test records/1840decisions (400minus32exposed). The32exposed v1 test records are development-contaminated and are excluded from v2 optimization and fresh evaluation. Fresh bilingual/domain-generalization evaluation is separately sealed.
- Stage A: cache only zero-adapter frozen features for train+dev.
- Stage B: exactly two controlled head objectives from identical random initialization and data order: soft-target crossentropy vs65%soft+35%hard crossentropy. Three epochs each, effective batch4, AdamW2e-4,5%warmup then cosine to10%LR, gradient norm1. No test selection.
- Select head/checkpoint by development composite accuracy:50% macroaccuracy across four original workflows,50% macroaccuracy across added dataset-language groups. Overall hard NLL breaks exact ties. Record unweighted per-domain/language/type results too.
- Stage C: selected head plus LoRA and joint head training, at most two epochs, effective batch4; adapter LR2e-5, head1e-4. Check development at half and whole epochs, early stop after two consecutive nonimproving checks. Compare against selected frozen-head result; do not force adoption of LoRA if worse.
- Checkpoints every100examples and every development check include optimizer,RNG,position,and input hashes; atomic replacement preserves recoverability. Completed runs cannot be overwritten.
- After weights and selection are frozen, choose one scalar temperature on separate calibration hard NLL. Report raw and calibrated hard NLL/ECE and teacher-distribution loss. Temperature does not affect accuracy.
- Final evaluation is owned by independent reviewer, with v1 and predeclared schema/majority baselines on identical fresh examples. No tuning on final outcomes. If it fails preregistered gates, report that honestly; any later changes require a separately described next study and unexposed evaluation.

This study can establish improvements on these benchmarks; it cannot certify general reliability or consequential decision autonomy. Original typed labels are teacher-generated; added translated data must carry language-quality and licensing caveats.

## Pre-optimization amendment

Independent audit found one exact source-factor overlap between original-train record tr_security_incidents_000063 and a sealed original-test record. Exclude that train row before any parameter update; train count959. No test labels were inspected and the fixed test stays unchanged. Its frozen feature may already have been computed, which is not model fitting and is never used for optimization.

## Pre-optimization bilingual schema amendment

Use one fixed full60intent MASSIVE schema rather than gold-derived domain narrowing. This avoids trivial singleton-domain predictions and tests genuine label selection. Extra train900 (MASSIVE600,SNLI300),dev78,calibration78; paired IDs grouped. Maximum added training input1449tokens, under the1536training cap. Set the global inference cap2048 identically for v1 and v2 final evaluations; no truncation and no weight changes to v1. Combined train1859records,dev198,calibration198.

## Pre-optimization development-size amendment

Expanded each extra dev/calibration partition to240records:60MASSIVE sourcepairs and60SNLI source/imagepairs, deterministic supersets of earlier78record partitions. Training bytes unchanged. Combined train1859,dev360,calibration360. This avoids checkpoint selection driven by only9NLI validation pairs. No model outcome or optimization was observed before expansion. Calibration max1454tokens; training max1449tokens.
