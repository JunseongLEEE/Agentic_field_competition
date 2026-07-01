---
description: "Build the DACON submit.zip (model/ + script.py + requirements.txt). Enforces 1GB cap, offline safety, and a working dry-run before packaging."
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
---

# /pack — DACON Code Submission Packager

Package an evaluated CANDIDATE experiment into a server-ready zip.
You refuse to pack if anything would auto-disqualify on the DACON server.

## Arguments
- `$ARGUMENTS` — experiment id (e.g., `exp_001_baseline_lgbm`).

## Required zip layout (no extra files at the top level)
```
submit.zip
├── model/                # weights, encoders, anything script.py loads
├── script.py             # inference only
└── requirements.txt      # extras only (keep minimal)
```

## Hard server limits (zero tolerance)
- zip ≤ **1 GB**
- install (`pip install -r requirements.txt`) ≤ **10 min**
- inference (`python script.py`) ≤ **10 min** on T4 (3 vCPU, 12GB RAM)
- offline after install

## STEP 0 — Preconditions

```bash
EXP="$ARGUMENTS"
EXP_DIR="experiments/${EXP}"

# Evaluator must have approved
python - <<PY
import json
ev = json.load(open(f"${EXP_DIR}/evaluation.json"))
assert ev["recommendation"] in ("CANDIDATE", "CANDIDATE_DIVERSITY"), f"not a candidate: {ev['recommendation']}"
assert ev["submission_readiness"]["offline_check"] == "PASS"
print("eval gate: OK")
PY
```

If preconditions fail → STOP, print why, do nothing.

## STEP 1 — Static Offline Scan

```bash
python scripts/validate_submission.py --script "${EXP_DIR}/script.py"

# Belt-and-suspenders manual scan
Grep -n "from_pretrained\b" "${EXP_DIR}/script.py" | grep -v "model/" || true
Grep -nE "requests\.|urllib|wget|curl|\.download\(" "${EXP_DIR}/script.py" || true
Grep -nE "api_key|API_KEY|openai\.|anthropic\." "${EXP_DIR}/script.py" || true
```

Any match outside `model/` paths → BLOCK packaging.

## STEP 2 — Structural Checks

```bash
# script.py must have main guard, read data/, write output/submission.csv
Grep -n "if __name__ == '__main__'" "${EXP_DIR}/script.py"
Grep -n "data/" "${EXP_DIR}/script.py"
Grep -n "output/submission.csv" "${EXP_DIR}/script.py"
```

All three must be present. Otherwise BLOCK.

## STEP 3 — Size Check

```bash
MODEL_MB=$(du -sm "${EXP_DIR}/model" | awk '{print $1}')
echo "model dir size: ${MODEL_MB} MB"
test "$MODEL_MB" -lt 900 || { echo "model dir >= 900MB; will not fit under 1GB zip"; exit 2; }
test "$MODEL_MB" -lt 800 && echo "OK" || echo "WARN: model size approaching cap"
```

## STEP 4 — Real Dry-Run on Server-Like Layout

```bash
PACK_STAGE=/tmp/pack_${EXP}_$$
mkdir -p "${PACK_STAGE}/data" "${PACK_STAGE}/output"
cp -r "${EXP_DIR}/model" "${PACK_STAGE}/"
cp "${EXP_DIR}/script.py" "${PACK_STAGE}/"
cp "${EXP_DIR}/requirements.txt" "${PACK_STAGE}/"
# helper modules script.py imports (features.py / model.py) MUST travel with it
for m in features.py model.py; do cp "${EXP_DIR}/$m" "${PACK_STAGE}/" 2>/dev/null || true; done
# server mounts JSONL test data (NOT csv); sample_submission stays csv
cp data/test.jsonl data/sample_submission.csv "${PACK_STAGE}/data/" 2>/dev/null || true

START=$(date +%s)
(cd "${PACK_STAGE}" && timeout 600 python script.py)   # 10-min wall cap, matches server
DRY_SEC=$(( $(date +%s) - START ))

# Verify submission produced + valid shape (labels are 14 STRING class names)
python - <<PY
import pandas as pd
CLASSES={"read_file","grep_search","list_directory","glob_pattern","edit_file","write_file",
         "apply_patch","run_bash","run_tests","lint_or_typecheck","ask_user","plan_task",
         "web_search","respond_only"}
sub = pd.read_csv("${PACK_STAGE}/output/submission.csv")
sample = pd.read_csv("data/sample_submission.csv")
assert list(sub.columns) == list(sample.columns) == ["id","action"], f"cols mismatch {list(sub.columns)}"
assert sub.shape[0] == sample.shape[0], f"row mismatch {sub.shape} vs {sample.shape}"
assert list(sub["id"]) == list(sample["id"]), "id order/values differ from sample_submission"
assert sub["action"].isnull().sum() == 0, "NaN in submission"
bad = set(sub["action"].unique()) - CLASSES
assert not bad, f"invalid class labels: {bad}"
print("dry-run submission.csv OK")
PY

echo "dry_run_seconds=${DRY_SEC}"
```

`dry_run_seconds > 540` (9 min) → BLOCK. Two-minute safety margin under the 10-min cap.

## STEP 5 — Build zip (ALWAYS into submissions/)

Use the packaging script — it uses Python `zipfile` (the `zip`/`unzip` CLIs are NOT
installed here), bundles `model/ + script.py + requirements.txt + features.py/model.py`,
enforces the 1 GB cap, and writes both `submissions/${EXP}.zip` and
`submissions/${EXP}_meta.json`.

```bash
python scripts/package_submit.py --exp "${EXP_DIR}"

ZIP_PATH="submissions/${EXP}.zip"
ZIP_BYTES=$(stat -c%s "${ZIP_PATH}" 2>/dev/null || stat -f%z "${ZIP_PATH}")
ZIP_MB=$(( ZIP_BYTES / 1024 / 1024 ))
echo "zip size: ${ZIP_MB} MB   ->  ${ZIP_PATH}"
# Inspect contents (python, since the unzip CLI is absent)
python -m zipfile -l "${ZIP_PATH}"
```

## STEP 6 — Metadata (augment the script-written meta with eval/LB fields)

`package_submit.py` already wrote `submissions/${EXP}_meta.json` (sha256, zip_size_mb,
model_size_mb, cv_score, git_commit, created_at). Merge in eval + LB-prediction fields
when an `evaluation.json` exists, plus the dry-run timing.

```bash
python - <<PY
import json, os
mp = f"submissions/${EXP}_meta.json"
meta = json.load(open(mp))
evp = f"experiments/${EXP}/evaluation.json"
if os.path.exists(evp):
    ev = json.load(open(evp))
    meta["cv_macro_f1"] = ev.get("scores", {}).get("cv_macro_f1", meta.get("cv_score"))
    lb = ev.get("lb_prediction", {})
    meta["predicted_lb"] = lb.get("predicted_lb")
    meta["lb_prediction_interval"] = [lb.get("pi_low"), lb.get("pi_high")]
    meta["trust_level"] = lb.get("trust_level")
    meta["offline_check"] = ev.get("submission_readiness", {}).get("offline_check", "PASS")
meta["dry_run_seconds"] = int(${DRY_SEC})
json.dump(meta, open(mp, "w"), indent=2)
print(json.dumps(meta, indent=2))
PY

rm -rf "${PACK_STAGE}"
```

## STEP 7 — Report

```
═════════════════════════════════════════════
PACKAGED: exp_NNN_name
═════════════════════════════════════════════
zip          : submissions/exp_NNN_name.zip   (<MB> MB / 1024 MB cap)
sha256       : <hash>
model dir    : <MB> MB
dry-run      : <S>s  (cap 540s)
offline scan : PASS

CV Macro-F1    : <0.XXXX>
Predicted LB   : <0.XXXX>  PI=[<lo>, <hi>]  trust=<level>
vs current best LB <0.XXXX>: Δ=<+/-0.XXXX>

⚠️  Manual upload required — DACON web only. NEVER auto-submit.
Next: /rank to confirm this is today's best use of a submission slot.
═════════════════════════════════════════════
```

## Hard Rules

- DO NOT pack if evaluator did not return CANDIDATE / CANDIDATE_DIVERSITY.
- DO NOT pack if offline scan finds a network call.
- DO NOT pack if dry-run produces no `output/submission.csv` or it has wrong shape.
- DO NOT pack if zip > 1 GB.
- DO NOT pack if `requirements.txt` pins `numpy`/`scipy`/`pandas`/`joblib` — that causes the server ABI crash (`numpy.dtype size changed`). Pin only `scikit-learn` (matching the pickle) + the model lib. Validate requirements in a CLEAN venv first. See [[requirements-never-pin-numpy-scipy]].
- DO NOT upload anywhere. Manual submission only.
