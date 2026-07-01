---
id: sequential-prescriptions-low-roi
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [error-analysis, features, sequential, calibration, macro-f1, gbdt]
related: [[session-grouping-requires-groupkfold]], [[data-is-jsonl-not-csv]]
summary: A hypothesis proposing 5 sequential-heavy "prescriptions" (conditional crosses, contextual tokens, pairwise heads, sequential-prior multiplier, two-stage sub-classifier) projecting Macro-F1 0.63→0.75-0.78 was empirically refuted — the GBDT already uses last_action/turn, so extra sequential handling adds ~0.
---

# Sequential re-processing prescriptions are low ROI (measured)

## Symptom
A hypothesis (data_docs/hypothesis_targeted_prescriptions.md) claimed the read/search 4-way and execution 3-way confusion could be cracked with 5 sequential-signal prescriptions, projecting Macro-F1 0.63 → 0.75–0.78 (additive +5~8, +2~4, +3~5, +1~2, +3~5 %p).

## Root cause
- The base GBDT already consumes last_action / second_last_action / turn_index / rule flags as features, so any post-hoc device built on the SAME signal is redundant.
- The projected gains were treated as additive though all five exploit one signal (last_action + turn).
- Baseline was mis-stated as 0.63; real was 0.66 (exp_001) / 0.68 (exp_010, which already implements the "conditional crosses" via action n-grams / prompt_intent / session_phase).

## Fix (measured on exp_001 OOF, honest GroupKFold)
- **Sequential-prior multiplier (처방4)**: `p_model^(1-α)·p_prior^α`, leave-one-out prior P(y|last,turn_bucket,failed) → **+0.0013** (α=0.2). Claim was +1~2%p.
- **Conditional crosses (처방1)**: exp_010 (engineered action n-grams etc.) → **+0.02** (0.66→0.68), not +5~8%p.
- **Pure-sequential ceiling**: predicting the read/search cluster from (last_action, turn_bucket, failed) alone tops out at **0.403 accuracy**, but the current text+sequential model already gets **0.505** — i.e. more sequential handling cannot beat what the model already fuses. Two-stage with "drop text, tabular only" (처방5) would REDUCE accuracy (0.51→0.40).
- mDeBERTa-based prescriptions (처방2/3) build on a base that already underperforms GBDT here (fine-tuned DeBERTa ~0.66, DeBERTa+LGBM stack 0.656 < GBDT 0.68) → low priority.

## Generalization
Before investing in a lever, measure its MARGINAL signal against what the model already encodes. Post-hoc priors / 2-stage / auxiliary heads over a signal already in the feature set add ~0. The genuine headroom for the read/search cluster is (a) NEW signal the model lacks — fine-grained text/args (glob metachars `*`/`?`/`**` → glob_pattern, explicit path+extension → read_file, last-action args) — and (b) per-class calibration for Macro-F1 (top-2 acc 0.84 ≫ top-1 0.67, so argmax is the bottleneck), and (c) model diversity for ensembling. Do not stack redundant prescriptions or trust additive %p projections.
