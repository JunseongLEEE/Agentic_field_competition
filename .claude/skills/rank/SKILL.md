---
description: "Rank submission candidates and recommend top picks for today. Shows composite score considering CV, stability, and diversity."
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
---

# /rank — Submission Candidate Selector

Rank all CANDIDATE experiments and recommend which to submit today.

## Process

### 1. Gather Candidates

Find all experiments with `evaluation.json` where recommendation = "CANDIDATE".

```bash
for dir in experiments/exp_*; do
  if [ -f "$dir/evaluation.json" ]; then
    echo "=== $dir ==="
    python3 -c "import json; d=json.load(open('$dir/evaluation.json')); print(d.get('recommendation','?'), d.get('cv_score','?'))" 2>/dev/null
  fi
done
```

### 2. Rank by Composite Score

```
composite = 0.5 × normalized_cv + 0.2 × stability + 0.3 × diversity
```

- **normalized_cv**: (score - min) / (max - min) among candidates
- **stability**: 1 - (cv_std / max_cv_std)
- **diversity**: 1/count(same_model_type) — rewards unique approaches

### 3. Apply Diversity Constraints

- Max 3 from same model family
- Include at least 1 ensemble if available
- Include at least 1 "safe" stable pick

### 4. Update SUBMISSION_CANDIDATES.md

Write the ranked table with:
- Rank, Experiment ID, CV Score, CV Std, Model Type, Composite Score
- Priority: SUBMIT_FIRST (top 2) | SUBMIT_IF_SLOTS (3-5) | HOLD (6+)

### 5. Report

```
========================================
TODAY'S CANDIDATES (YYYY-MM-DD)
========================================
1. exp_005_ensemble_v2    CV: 0.9234  [SUBMIT_FIRST]
2. exp_003_catboost_tuned CV: 0.9201  [SUBMIT_FIRST]
3. exp_004_xgb_features   CV: 0.9189  [SUBMIT_IF_SLOTS]
...

남은 제출 기회: X/10
⚠️  수동 제출 필요
========================================
```
