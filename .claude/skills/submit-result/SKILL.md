---
description: "Record a leaderboard score after manual submission. Extracts insights from CV-LB gap and updates experiment memory. Usage: /submit-result exp_001 0.8234"
user-invocable: true
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
---

# /submit-result — Record LB Score & Extract Insights

When the user manually submits and gets a leaderboard score, use this to record it and analyze the CV-LB relationship.

## Arguments
- `$ARGUMENTS` — format: `<experiment_name> <lb_score>` (e.g., "exp_001_baseline 0.8234")

## Step 1: Parse & Validate

Extract experiment name and LB score from arguments.
Find the experiment directory and load its results.

## Step 2: Load Experiment Data

```python
import json
from pathlib import Path

exp_dir = Path(f'experiments/{exp_name}')
train_log = json.load(open(exp_dir / 'train_log.json'))
cv_mean = train_log['cv_mean']
cv_std = train_log['cv_std']

gap = abs(cv_mean - lb_score)
gap_direction = "CV > LB" if cv_mean > lb_score else "LB > CV"
```

## Step 3: Analyze CV-LB Relationship

Load ALL previous submissions to find patterns:

```python
# Load historical gaps
import csv
with open('EXPERIMENT_LOG.csv') as f:
    history = [r for r in csv.DictReader(f) if r.get('lb_score')]

avg_gap = mean of all historical gaps
gap_trend = is the gap growing or shrinking?
```

**Insight extraction rules:**

| Pattern | Insight |
|---------|---------|
| gap < 0.005 | "CV setup is reliable, trust it" |
| gap > 0.02 | "CV-LB divergence: possible overfitting or distribution shift" |
| CV > LB consistently | "Model overfits to train distribution" |
| LB > CV consistently | "CV is pessimistic, can be more aggressive" |
| gap increasing over experiments | "Later experiments overfit more — simplify" |
| gap decreasing | "CV alignment improving — current direction is good" |
| Single model type has smaller gap | "Model X generalizes better" |

## Step 4: Update Files

### 4a. Update experiment SUMMARY.md
Edit the Results section:
```markdown
| LB Score | 0.XXXX |
| CV-LB Gap | 0.XXXX |
| Status | SUBMITTED |
```

Add to Insight section:
```markdown
- CV-LB gap was X.XXXX (direction). [generated insight]
```

### 4b. Append to `logs/insights.jsonl`
```python
insight_record = {
    "date": "2026-05-17",
    "experiment": exp_name,
    "cv_score": cv_mean,
    "cv_std": cv_std,
    "lb_score": lb_score,
    "gap": gap,
    "gap_direction": gap_direction,
    "avg_historical_gap": avg_gap,
    "gap_trend": "growing|shrinking|stable",
    "model_type": "...",
    "feature_set": "...",
    "insight": "Generated insight about what this tells us",
    "actionable": "Specific suggestion for next experiment"
}
```

### 4c. Update EXPERIMENT_LOG.csv
Set `lb_score` and `cv_lb_gap` for this experiment row.

### 4d. Update LEADERBOARD_LOG.md
Append a new row to the table.

### 4e. Rebuild digest
```bash
python scripts/build_digest.py
```

### 4f. Update orchestrator_state.json
If this is the best LB score, update. Add insight to strategy considerations.

## Step 5: Report

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBMISSION RECORDED: exp_NNN_name
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CV Score:      0.XXXX ± 0.XXXX
LB Score:      0.XXXX
CV-LB Gap:     0.XXXX ({gap_direction})
Historical Avg Gap: 0.XXXX
Gap Trend:     {growing/shrinking/stable}

INSIGHT: {what we learned from this submission}
ACTION: {what to do next based on this}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Step 6: Pattern Detection (after 3+ submissions)

When enough data exists, detect deeper patterns:

```python
# Group by model type
model_gaps = group submissions by model_type, compute avg gap per type

# Group by feature set
feature_gaps = group by feature approach

# Temporal trend
chronological_gaps = gaps in order of submission time
```

Print pattern summary:
```
MODEL GENERALIZATION RANKING:
1. CatBoost — avg gap: 0.003 (best generalizer)
2. LightGBM — avg gap: 0.008
3. XGBoost — avg gap: 0.015 (tends to overfit)

FEATURE INSIGHT:
- Text length features: reduce gap by ~0.005
- Raw text embeddings: increase gap by ~0.01

RECOMMENDATION FOR NEXT EXPERIMENT:
Based on {N} submissions, prioritize {model_type} with {feature_approach}
```
