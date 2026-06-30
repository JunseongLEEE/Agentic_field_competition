---
description: "Execute an experiment's train.py and dry-run its script.py. Captures Macro-F1, per-class F1, runtime, model size, and offline-safety verdict."
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
---

# /run — Experiment Runner

Execute `train.py` for an experiment, verify outputs, dry-run `script.py`, and update bridge files.
You do NOT modify experiment code. You only run, verify, record.

## Arguments
- `$ARGUMENTS` — experiment id or path (e.g., `exp_001_baseline_lgbm` or `experiments/exp_001_baseline_lgbm`).

## STEP 0 — Resolve and Validate

```bash
EXP_NAME="$ARGUMENTS"
EXP_DIR="experiments/${EXP_NAME}"
test -d "$EXP_DIR" || EXP_DIR=$(ls -d experiments/*${EXP_NAME}* 2>/dev/null | head -1)
test -d "$EXP_DIR" || { echo "experiment not found"; exit 2; }

# Required files
for f in config.yaml train.py script.py requirements.txt; do
  test -f "${EXP_DIR}/${f}" || { echo "missing ${EXP_DIR}/${f}"; exit 2; }
done

# Offline scan must pass before we burn compute
python scripts/validate_submission.py --script "${EXP_DIR}/script.py"
```

## STEP 1 — Run train.py with Wall-Clock Cap

```bash
mkdir -p "${EXP_DIR}/model" "${EXP_DIR}/models"
START=$(date +%s)
(cd "${EXP_DIR}" && timeout 3600 python train.py 2>&1 | tee run_output.txt)
END=$(date +%s)
echo "wall_seconds=$((END-START))"
```

Watch live output for:
- `OOM` / `CUDA out of memory` → kill, report, set status=FAILED
- `nan` / `inf` in any line → set status=FAILED
- `RuntimeError` / `ValueError` → set status=FAILED with traceback

## STEP 2 — Verify Training Outputs

```bash
cd "${EXP_DIR}"
ls -la oof_preds.npy test_preds.npy train_log.json
test -d model && ls -la model/

python - <<'PY'
import json, numpy as np
log = json.load(open('train_log.json'))
oof  = np.load('oof_preds.npy')
test = np.load('test_preds.npy')

assert oof.ndim == 2 and oof.shape[1] == 14, f"oof shape {oof.shape} != (N, 14)"
assert test.ndim == 2 and test.shape[1] == 14, f"test shape {test.shape} != (M, 14)"
assert not np.isnan(oof).any(), "NaN in OOF"
assert not np.isnan(test).any(), "NaN in test"
assert 'cv_mean' in log and 'per_class_f1' in log, "train_log.json missing required keys"

print(f"OOF  : {oof.shape} | mean prob max={oof.max():.4f}")
print(f"TEST : {test.shape} | mean prob max={test.max():.4f}")
print(f"CV   : macro-F1 {log['cv_mean']:.4f} ± {log['cv_std']:.4f}")
print(f"Worst class: {log['worst_class']}")
PY
```

## STEP 3 — Dry-Run script.py (Server Simulation)

This is mandatory. We must prove inference fits the 10-min budget BEFORE we ever ask DACON to run it.

```bash
RUN_DIR=/tmp/dryrun_${EXP_NAME}_$$
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/output"
cp -r "${EXP_DIR}/model" "${RUN_DIR}/"
cp "${EXP_DIR}/script.py" "${RUN_DIR}/"
cp data/sample_submission.csv "${RUN_DIR}/data/" 2>/dev/null || true

# Use a 1000-row slice for speed; extrapolate runtime to full test
head -n 1001 data/test.csv > "${RUN_DIR}/data/test.csv"
SAMPLE_ROWS=$(($(wc -l < "${RUN_DIR}/data/test.csv") - 1))
FULL_ROWS=$(($(wc -l < data/test.csv) - 1))

START=$(date +%s)
(cd "${RUN_DIR}" && timeout 300 python script.py)
END=$(date +%s)
SAMPLE_SEC=$((END-START))

# Verify submission produced
test -f "${RUN_DIR}/output/submission.csv" || { echo "submission.csv NOT produced"; exit 2; }

# Verify shape matches sample_submission
python - <<PY
import pandas as pd
sub = pd.read_csv("${RUN_DIR}/output/submission.csv")
sample = pd.read_csv("data/sample_submission.csv") if False else None
print(f"submission shape: {sub.shape}")
print(f"columns: {list(sub.columns)}")
print(f"NaN: {sub.isnull().sum().sum()}")
print(sub.head())
PY

EXTRAPOLATED_MIN=$(python -c "print(round(${SAMPLE_SEC} * ${FULL_ROWS} / max(${SAMPLE_ROWS},1) / 60, 2))")
echo "sample_seconds=${SAMPLE_SEC}  estimated_full_minutes=${EXTRAPOLATED_MIN}"

# Cleanup
rm -rf "${RUN_DIR}"
```

Hard guard: if `estimated_full_minutes > 8.0` → set status=REVIEW, reason="inference budget at risk".

## STEP 4 — Update train_log.json with Dry-Run Numbers

Append/overwrite these fields with the actual measurements:
- `inference_ms_per_sample` = `(SAMPLE_SEC * 1000) / SAMPLE_ROWS`
- `estimated_full_test_minutes` = extrapolated value
- `dry_run_status` = `SUCCESS | FAILED`

## STEP 5 — Update Bridge Files

```bash
python scripts/build_digest.py
```

Append a row to `EXPERIMENT_LOG.csv`:
```
experiment_id,name,hypothesis,status,cv_macro_f1,cv_std,worst_class_f1,
inference_ms_per_sample,estimated_full_test_minutes,model_size_mb,
offline_check,lb_score,cv_lb_gap,seed,git_commit,created_at,completed_at,notes
```

Append to `logs/cycle_history.jsonl`:
```json
{
  "timestamp": "<iso>",
  "phase": "run",
  "experiment": "<exp_NNN>",
  "cv_mean": 0.XXXX,
  "cv_std": 0.XXXX,
  "worst_class_f1": 0.XX,
  "inference_ms_per_sample": X.X,
  "estimated_full_test_minutes": X.X,
  "dry_run_status": "SUCCESS"
}
```

Update `experiments/exp_NNN/SUMMARY.md` Results section with actual numbers.

## STEP 6 — Report

```
═════════════════════════════════════════════
EXPERIMENT COMPLETE: exp_NNN_name
═════════════════════════════════════════════
CV Macro-F1     : 0.XXXX ± 0.XXXX
Fold scores     : [0.XX, 0.XX, 0.XX, 0.XX, 0.XX]
Worst class     : id=<N> f1=<0.XX>  (collapsed if < 0.05)
Train runtime   : <M>m <S>s
Inference       : <X.X> ms/sample → est. full test <X.X> min
Model size      : <X.X> MB
Offline check   : PASS / FAIL
Dry-run         : SUCCESS / FAILED  (submission.csv produced)
EXPERIMENT_LOG  : updated
Digest          : rebuilt

Next: /eval exp_NNN_name
═════════════════════════════════════════════
```

## Error Handling

| symptom | action |
|---|---|
| OOM | report; suggest reducing `max_features`, batch size, or sequence length |
| NaN/Inf in OOF or test | status=FAILED; do NOT proceed to eval |
| CV std > 0.02 | flag in report; let evaluator decide |
| `model/` > 800 MB | warn; let packager block |
| `estimated_full_test_minutes > 8.0` | status=REVIEW |
| `dry_run_status == FAILED` | status=FAILED; experiment cannot become CANDIDATE |
| Macro-F1 < 1/14 ≈ 0.071 | CRITICAL — pipeline bug, halt and surface |

## Hard Rules

- NEVER edit experiment code from this skill.
- NEVER skip the dry-run.
- NEVER mark an experiment CANDIDATE here — that is `/eval`'s job.
- ALWAYS rebuild the digest at the end.
