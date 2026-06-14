---
description: "Run an experiment and capture results. Pass experiment path or name as argument (e.g., /run exp_001_baseline)."
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
---

# /run — Experiment Runner

Execute an experiment and capture all outputs.

## Arguments
- `$ARGUMENTS` — experiment name or path (e.g., "exp_001_baseline" or "experiments/exp_001_baseline")

## Execution Steps

### 1. Locate and Validate
```bash
# Find the experiment directory
ls experiments/*$0* 2>/dev/null || ls $ARGUMENTS 2>/dev/null
```

Check these exist before running:
- `config.yaml`
- `train.py`

### 2. Run Training

```bash
cd experiments/EXP_NAME && python train.py 2>&1 | tee run_output.txt
```

Monitor for:
- OOM errors → report immediately
- NaN/Inf in outputs → CRITICAL, stop
- Warnings about data leakage
- Runtime (flag if > 30min for non-NN models)

### 3. Verify Outputs

After training completes, check:
```bash
ls -la oof_preds.npy test_preds.npy train_log.json models/
python -c "import numpy as np; oof=np.load('oof_preds.npy'); test=np.load('test_preds.npy'); print(f'OOF: {oof.shape}, Test: {test.shape}, OOF range: [{oof.min():.4f}, {oof.max():.4f}]')"
```

### 4. Update EXPERIMENT_LOG.csv

After successful run, append a row:
```python
import csv, json
from datetime import datetime

with open('train_log.json') as f:
    results = json.load(f)

# Append to EXPERIMENT_LOG.csv
row = {
    'experiment_id': results['experiment_id'],
    'name': EXP_NAME,
    'hypothesis': '...',  # from config.yaml
    'status': 'COMPLETED',
    'cv_score': results['cv_mean'],
    'cv_std': results['cv_std'],
    'lb_score': '',
    'cv_lb_gap': '',
    'seed': 42,
    'git_commit': GIT_HASH,
    'created_at': datetime.now().isoformat(),
    'completed_at': datetime.now().isoformat(),
    'notes': ''
}
```

### 5. Update SUMMARY.md Results

After successful run, update the experiment's `SUMMARY.md` Results section:

```markdown
## Results
| Metric | Score |
|--------|-------|
| CV Mean | {actual cv_mean} |
| CV Std | {actual cv_std} |
| CV Fold Scores | {actual fold scores} |
| LB Score | 미제출 |
| CV-LB Gap | N/A |
| Status | COMPLETED |
```

### 6. Rebuild Digest

```bash
python scripts/build_digest.py
```

This updates `logs/experiment_digest.md` so all agents have the latest snapshot.

### 7. Report Summary

```
========================================
EXPERIMENT COMPLETE: exp_NNN_name
========================================
CV Score: 0.XXXX ± 0.XXXX
Fold scores: [...]
Runtime: X분 XX초
Outputs: oof_preds.npy ✓ | test_preds.npy ✓ | models/ ✓
SUMMARY.md: updated ✓
Digest: rebuilt ✓

다음 단계: /eval exp_NNN_name
========================================
```

## Error Handling
- If OOM: suggest `--batch_size` reduction or feature count reduction
- If NaN: check for division by zero, log transform on negatives, missing value handling
- If slow: profile and suggest optimization
- If crash: save partial logs, report stack trace, update SUMMARY.md status to FAILED
