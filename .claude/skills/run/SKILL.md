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

**Rule A (foreground/wait):** run `train.py` in the FOREGROUND and WAIT for it to
exit. NEVER background-and-exit — that orphans the training and skips verify +
dry-run. You are done with this step only when `train_log.json` + `oof_preds.npy`
+ `test_preds.npy` exist. **Rule B (thread cap):** on a 128-core box, cap CPU
threads ≤ 16 (`OMP_NUM_THREADS`/model `n_jobs`) and run at most 2 heavy trainings
in parallel — full-core fan-out causes oversubscription thrash (jobs hang for
tens of minutes). Prefer GPU (local RTX 3090) when the model supports it.

Run unbuffered (`python -u`) and `tee` to the standard live-log path
`experiments/<exp>/train.log` so the user can `tail -f` progress in real time.
train.py also installs its own line-buffered Tee to that same file (covers any
launch mode), so `-u | tee -a` here is belt-and-suspenders.

```bash
mkdir -p "${EXP_DIR}/model" "${EXP_DIR}/models"
export OMP_NUM_THREADS=16               # Rule B: cap threads on the 128-core box
START=$(date +%s)
# FOREGROUND (no &): unbuffered + tee to the live log; wait for completion
(cd "${EXP_DIR}" && timeout 3600 python -u train.py 2>&1 | tee -a train.log | tee run_output.txt)
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

Note: test input is `data/test.jsonl` (JSONL, NOT csv). The repo `test.jsonl` is
only a 5-row SAMPLE; the real server test is **30,000 hidden rows**. So run
script.py once on the real `data/test.jsonl` to prove correctness, and separately
time it on a ~1000-row JSONL sample built from `train.jsonl` (labels dropped) to
extrapolate inference time to 30,000 rows. `features.py` must be copied alongside
`script.py` (script.py imports it).

```bash
RUN_DIR=/tmp/dryrun_${EXP_NAME}_$$
mkdir -p "${RUN_DIR}/data" "${RUN_DIR}/output"
cp -r "${EXP_DIR}/model" "${RUN_DIR}/"
cp "${EXP_DIR}/script.py" "${RUN_DIR}/"
cp "${EXP_DIR}/features.py" "${RUN_DIR}/" 2>/dev/null || true   # script.py imports it
cp data/sample_submission.csv "${RUN_DIR}/data/" 2>/dev/null || true

# (a) correctness run on the real 5-row test.jsonl
cp data/test.jsonl "${RUN_DIR}/data/test.jsonl"
(cd "${RUN_DIR}" && timeout 300 python script.py)
test -f "${RUN_DIR}/output/submission.csv" || { echo "submission.csv NOT produced"; exit 2; }

# (b) timing sample: ~1000 JSONL rows from train.jsonl (test.jsonl too small to time)
head -n 1000 data/train.jsonl > "${RUN_DIR}/data/test.jsonl"
SAMPLE_ROWS=$(wc -l < "${RUN_DIR}/data/test.jsonl")
FULL_ROWS=30000                                   # real server test size (hidden)
START=$(date +%s)
(cd "${RUN_DIR}" && timeout 300 python script.py)
END=$(date +%s)
SAMPLE_SEC=$((END-START))

# Verify label column: values must be the 14 STRING class names, not ints
python - <<PY
import pandas as pd
CLASSES={"read_file","grep_search","list_directory","glob_pattern","edit_file",
"write_file","apply_patch","run_bash","run_tests","lint_or_typecheck","ask_user",
"plan_task","web_search","respond_only"}
sub = pd.read_csv("${RUN_DIR}/output/submission.csv")
print(f"submission shape: {sub.shape}  columns: {list(sub.columns)}  NaN: {sub.isnull().sum().sum()}")
bad = set(sub['action'].unique()) - CLASSES
assert not bad, f"BAD action labels (must be 14 strings, not ints): {bad}"
print("label check PASS — all action values in the 14 string classes")
print(sub.head())
PY

EXTRAPOLATED_MIN=$(python -c "print(round(${SAMPLE_SEC} * ${FULL_ROWS} / max(${SAMPLE_ROWS},1) / 60, 2))")
echo "sample_rows=${SAMPLE_ROWS} sample_seconds=${SAMPLE_SEC}  estimated_30000_row_minutes=${EXTRAPOLATED_MIN}"

# Cleanup
rm -rf "${RUN_DIR}"
```

Hard guard: if `estimated_30000_row_minutes > 8.0` → set status=REVIEW, reason="inference budget at risk" (server cap is 10 min for 30,000 rows).

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
- NEVER background-and-exit `train.py` (Rule A) — run it in the FOREGROUND and WAIT until `train_log.json` + `oof_preds.npy` + `test_preds.npy` exist.
- NEVER run heavy trainings with `n_jobs=-1`/full-core fan-out (Rule B) — cap threads ≤ 16 (`OMP_NUM_THREADS`) and ≤ 2 in parallel.
- ALWAYS run unbuffered + tee to `experiments/<exp>/train.log` so progress is watchable live (`tail -f`).
- NEVER skip the dry-run.
- NEVER mark an experiment CANDIDATE here — that is `/eval`'s job.
- ALWAYS rebuild the digest at the end.
