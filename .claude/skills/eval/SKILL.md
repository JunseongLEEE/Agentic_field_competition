---
description: "Evaluate an experiment against baseline + previous best. Detects leakage, class collapse, overfitting. Uses CV→LB correlation to predict LB score and decide CANDIDATE / REVIEW / REJECT."
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

Decide whether an experiment should become a DACON submission candidate, **without burning a DACON submission**.
You verify on local CV + the CV→LB correlation model.

## Arguments
- `$ARGUMENTS` — experiment id (e.g., `exp_001_baseline_lgbm`).

## STEP 0 — Load Everything

```bash
EXP="$ARGUMENTS"
EXP_DIR="experiments/${EXP}"

# Required artifacts
test -f "${EXP_DIR}/train_log.json" || { echo "missing train_log.json"; exit 2; }
test -f "${EXP_DIR}/oof_preds.npy" || { echo "missing oof_preds.npy"; exit 2; }
test -f "${EXP_DIR}/test_preds.npy" || { echo "missing test_preds.npy"; exit 2; }
test -f "${EXP_DIR}/config.yaml" || { echo "missing config.yaml"; exit 2; }

# Correlation model
python scripts/cv_lb_correlation.py --json > /tmp/cvlb.json
```

Also read:
- `EXPERIMENT_LOG.csv` — baseline + current best CV
- `logs/insights.jsonl` — historical class-collapse + CV-LB patterns
- The plan entry that birthed this experiment — its `verification_protocol`

## STEP 1 — Score Comparison

```python
import json, numpy as np, pandas as pd

log = json.load(open(f'experiments/{EXP}/train_log.json'))
cv = log['cv_mean']; cv_std = log['cv_std']

prev = pd.read_csv('EXPERIMENT_LOG.csv')
baseline_cv = float(prev.iloc[0]['cv_macro_f1']) if len(prev) else 0.0
current_best_cv = float(prev['cv_macro_f1'].max()) if len(prev) else 0.0

improvement_vs_baseline = cv - baseline_cv
improvement_vs_best     = cv - current_best_cv
```

## STEP 2 — Stability Grading

```
cv_std < 0.005 → A
cv_std < 0.010 → B
cv_std < 0.020 → C
cv_std ≥ 0.020 → D
```

Also compute max fold deviation. If any fold is > 3σ from the mean → flag as suspicious.

## STEP 3 — Per-Class Inspection (Macro-F1 is dragged by the worst class)

```python
per_class = log['per_class_f1']                 # dict {class_id: f1}
worst = log['worst_class']
collapsed = log.get('collapsed_classes', [])    # f1 < 0.05

# Macro-F1 sanity
recomputed_macro = sum(per_class.values()) / len(per_class)
assert abs(recomputed_macro - cv) < 0.01, "Macro-F1 mismatch between per_class and cv_mean"
```

Flags:
- `len(collapsed) > 0` → REVIEW with reason "classes {ids} collapsed; LB will suffer"
- Improvement from majority classes only, with no minority gain → REVIEW

## STEP 4 — Leakage Probes

Compute and check each:

1. **Score jump too large**:
   `improvement_vs_baseline > 0.05` AND no model/feature change justifies it → flag.

2. **Fold-specific spike**:
   any fold > `cv_mean + 3 * cv_std` → flag (possible group leak across folds).

3. **OOF vs Test distribution shift**:
   ```python
   oof  = np.load(f'experiments/{EXP}/oof_preds.npy')
   test = np.load(f'experiments/{EXP}/test_preds.npy')
   oof_class_freq  = np.bincount(oof.argmax(1),  minlength=14) / len(oof)
   test_class_freq = np.bincount(test.argmax(1), minlength=14) / len(test)
   l1 = float(np.abs(oof_class_freq - test_class_freq).sum())
   # l1 > 0.30 → REVIEW
   ```

4. **Session-level leak (HARD)**:
   The 14-class data has 9,429 sessions over 70,000 rows (99.69% multi-step), so
   CV MUST be `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
   grouped by session id (`id.rsplit("-step",1)[0]`). If `config.yaml` /
   `train.py` used plain `KFold` or `StratifiedKFold` (no session grouping), or
   the zero-overlap assert is missing → **REJECT** (CV is leaked/inflated; do not
   trust the score). If grouping is present but unverifiable → REVIEW.

5. **Generation-method leakage**:
   Grep `data_docs/generation_methodology.md` for "label" or "answer" appearing inside `history`. If found, flag.

6. **Label format**:
   The submission `action` column must be the 14 exact snake_case STRING class
   names (read_file, grep_search, list_directory, glob_pattern, edit_file,
   write_file, apply_patch, run_bash, run_tests, lint_or_typecheck, ask_user,
   plan_task, web_search, respond_only) — NOT integers 0-13. Any value outside
   this set → REJECT.

## STEP 5 — Submission Readiness

Re-check (lightweight, runs again to be safe):
```bash
python scripts/validate_submission.py --script experiments/${EXP}/script.py
du -sm experiments/${EXP}/model/   # MB
```

Conditions:
- offline check FAIL → REJECT
- `model_size_mb > 800` → REVIEW
- `estimated_full_test_minutes > 10` → REJECT (hard server cap)
- `estimated_full_test_minutes > 8` → REVIEW

## STEP 6 — Predict LB Score

```bash
python scripts/cv_lb_correlation.py --predict <cv> --json > /tmp/pred.json
```

Read:
- `predicted_lb`
- `pi_low`, `pi_high` (95% prediction interval)
- `trust_level`

Decide:
- If `trust_level == "high"`: gate on `predicted_lb - (predicted_lb - pi_low) > current_best_lb`.
- If `trust_level == "medium"`: gate but widen uncertainty ×1.5.
- If `trust_level == "low"`: do not gate; rely on CV improvement ≥ 0.005.

## STEP 7 — Decision

Decision table (apply top-down, first match wins):

| condition | recommendation |
|---|---|
| any confirmed leakage flag | REJECT |
| CV used plain KFold/StratifiedKFold (not StratifiedGroupKFold by session) | REJECT |
| submission `action` values not in the 14 string classes (e.g. ints) | REJECT |
| offline check FAIL | REJECT |
| estimated full test inference > 10 min | REJECT |
| CV worse than baseline | REJECT |
| any collapsed class (F1 < 0.05) | REVIEW |
| CV std grade D | REVIEW |
| OOF/test L1 distribution shift > 0.30 | REVIEW |
| model size > 800 MB | REVIEW |
| estimated inference 8–10 min | REVIEW |
| improvement_vs_best > 0 AND all checks pass | CANDIDATE |
| improvement_vs_best ≤ 0 AND adds diversity to ensemble | CANDIDATE_DIVERSITY |
| else | COMPLETED |

`CANDIDATE_DIVERSITY` is selectable by `/rank` only if its OOF predictions correlate < 0.95 with all current candidates.

## STEP 8 — Write evaluation.json

```json
{
  "experiment_id": "exp_NNN_name",
  "evaluation_date": "<iso>",
  "metric": "macro_f1",
  "scores": {
    "cv_macro_f1": 0.XXXX,
    "cv_std": 0.XXXX,
    "baseline_macro_f1": 0.XXXX,
    "current_best_macro_f1": 0.XXXX,
    "improvement_vs_baseline": +0.XXXX,
    "improvement_vs_best": +0.XXXX
  },
  "per_class_summary": {
    "worst_class": {"id": <int>, "f1": 0.XX},
    "collapsed_classes": [<ids>]
  },
  "stability_grade": "A|B|C|D",
  "leakage_flags": [],
  "distribution_l1": 0.XX,
  "submission_readiness": {
    "offline_check": "PASS|FAIL",
    "model_size_mb": <float>,
    "estimated_full_test_minutes": <float>
  },
  "lb_prediction": {
    "predicted_lb": 0.XXXX,
    "pi_low": 0.XXXX,
    "pi_high": 0.XXXX,
    "trust_level": "low|medium|high",
    "worth_submitting": true|false
  },
  "recommendation": "CANDIDATE | CANDIDATE_DIVERSITY | REVIEW | REJECT | COMPLETED",
  "reason": "<one sentence>",
  "actionable_next_step": "<what /plan should propose next based on this>"
}
```

Update `EXPERIMENT_LOG.csv` status column for this experiment.

## STEP 9 — Report

```
═════════════════════════════════════════════
EVALUATION: exp_NNN_name
═════════════════════════════════════════════
CV Macro-F1     : 0.XXXX ± 0.XXXX  (baseline 0.XXXX, best 0.XXXX)
ΔBest           : +0.XXXX
Worst class     : id=<N> f1=<0.XX>  collapsed=[<ids or none>]
Stability       : <A|B|C|D>
Distribution L1 : <0.XX>
Leakage flags   : <[] or list>
Offline         : PASS / FAIL
Model size      : <X> MB
Inference est.  : <X.X> min

LB Prediction   : <0.XXXX>  PI=[<0.XXXX>, <0.XXXX>]  trust=<level>
Worth submit?   : YES / NO  (vs current best LB <0.XXXX>)

→ <CANDIDATE | REVIEW | REJECT | COMPLETED>
Reason: <one sentence>
Next  : /pack <exp> (if CANDIDATE)  |  /plan (if REJECT/REVIEW)
═════════════════════════════════════════════
```

## Hard Rules

- NEVER mark CANDIDATE if any leakage flag is confirmed.
- NEVER mark CANDIDATE if offline check fails.
- NEVER mark CANDIDATE if estimated inference > 10 min.
- ALWAYS compute the LB prediction even when `trust_level == "low"` (we need to grow the dataset).
- ALWAYS write `actionable_next_step` so `/plan` has fuel for the next cycle.
