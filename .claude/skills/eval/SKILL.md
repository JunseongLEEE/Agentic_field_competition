---
description: "Evaluate experiment results — compares against baseline, checks for leakage/overfitting, decides if experiment is a valid submission candidate. Pass experiment name as argument."
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
---

# /eval — Experiment Evaluator

Evaluate an experiment's results for quality, stability, and leakage.

## Arguments
- `$ARGUMENTS` — experiment name (e.g., "exp_001_baseline")

## Step 1: Load Results

Read these files from the experiment directory:
- `train_log.json` — CV scores, feature importance
- `config.yaml` — model config, seed
- `oof_preds.npy` — out-of-fold predictions
- `test_preds.npy` — test predictions

Also load:
- `EXPERIMENT_LOG.csv` — for baseline comparison
- Previous experiments' `train_log.json` — for relative ranking

## Step 2: Run Checks

### A. Score Comparison
```python
import json, numpy as np

# Current experiment
with open(f'experiments/{exp}/train_log.json') as f:
    current = json.load(f)

# Find baseline (first COMPLETED experiment)
# Compare: improvement = current['cv_mean'] - baseline['cv_mean']
```

### B. Stability Analysis
- CV std: < 0.005 = A, < 0.01 = B, < 0.02 = C, >= 0.02 = D
- Max fold deviation from mean
- If any single fold > 3σ from mean → flag as suspicious

### C. Leakage Detection
```python
oof = np.load('oof_preds.npy')
test = np.load('test_preds.npy')

# 1. Distribution shift
print(f"OOF: mean={oof.mean():.4f}, std={oof.std():.4f}")
print(f"Test: mean={test.mean():.4f}, std={test.std():.4f}")
# If |oof_mean - test_mean| / oof_std > 0.5 → flag

# 2. Suspicious score jump
# If improvement > +0.05 over baseline without clear reason → flag

# 3. Check for NaN/Inf
assert not np.any(np.isnan(oof)), "NaN in OOF!"
assert not np.any(np.isnan(test)), "NaN in test!"
```

### D. Overfitting Signals
- If LB score available: check CV-LB gap trend
- If model complexity increased but CV gain is marginal → flag
- If all improvement comes from 1 fold → flag

## Step 3: Recommendation

| Condition | Decision |
|-----------|----------|
| Leakage confirmed | REJECT |
| CV worse than baseline | REJECT |
| CV-LB gap > 2x average | REVIEW |
| CV std grade D | REVIEW |
| Score up + all checks pass | CANDIDATE |

## Step 4: Save & Report

Save `evaluation.json` in experiment directory.

Update experiment status in EXPERIMENT_LOG.csv to EVALUATED or CANDIDATE.

```
========================================
EVALUATION: exp_NNN_name
========================================
CV Score:    0.XXXX ± 0.XXXX (baseline: 0.XXXX)
Improvement: +0.XXXX
Stability:   A
Leakage:     CLEAN
Distribution: OK (shift=0.XX)

→ CANDIDATE ✓
다음 단계: /pack exp_NNN_name
========================================
```

Or if rejected:
```
→ REJECT ✗
사유: [reason]
제안: [what to try instead]
========================================
```
