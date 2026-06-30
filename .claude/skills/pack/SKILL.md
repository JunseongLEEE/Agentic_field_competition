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
cp data/test.csv data/sample_submission.csv "${PACK_STAGE}/data/" 2>/dev/null || true

START=$(date +%s)
(cd "${PACK_STAGE}" && timeout 600 python script.py)   # 10-min wall cap, matches server
DRY_SEC=$(( $(date +%s) - START ))

# Verify submission produced + valid shape
python - <<PY
import pandas as pd
sub = pd.read_csv("${PACK_STAGE}/output/submission.csv")
sample = pd.read_csv("data/sample_submission.csv")
assert sub.shape[0] == sample.shape[0], f"row mismatch {sub.shape} vs {sample.shape}"
assert list(sub.columns) == list(sample.columns), f"cols mismatch {list(sub.columns)} vs {list(sample.columns)}"
assert sub.isnull().sum().sum() == 0, "NaN in submission"
# Label column must be valid 14-class ids (assumes integer labels)
label_col = sub.columns[1]
assert sub[label_col].between(0, 13).all(), "predicted labels out of [0, 13]"
print("dry-run submission.csv OK")
PY

echo "dry_run_seconds=${DRY_SEC}"
```

`dry_run_seconds > 540` (9 min) → BLOCK. Two-minute safety margin under the 10-min cap.

## STEP 5 — Build zip

```bash
mkdir -p submissions
ZIP_PATH="submissions/${EXP}.zip"
(cd "${PACK_STAGE}" && zip -r "${ZIP_PATH}" model script.py requirements.txt -x "*.DS_Store")
# Move into project (was created with relative path inside PACK_STAGE)
mv "${PACK_STAGE}/${ZIP_PATH}" "${ZIP_PATH}" 2>/dev/null || true

ZIP_BYTES=$(stat -f%z "${ZIP_PATH}" 2>/dev/null || stat -c%s "${ZIP_PATH}")
ZIP_MB=$(( ZIP_BYTES / 1024 / 1024 ))
echo "zip size: ${ZIP_MB} MB"
test "${ZIP_BYTES}" -lt 1073741824 || { echo "zip exceeds 1 GB"; rm -f "${ZIP_PATH}"; exit 2; }

# Inspect contents
unzip -l "${ZIP_PATH}"
```

## STEP 6 — Metadata

```bash
SHA=$(shasum -a 256 "${ZIP_PATH}" | awk '{print $1}')
python - <<PY
import json, hashlib
from datetime import datetime
ev = json.load(open(f"experiments/${EXP}/evaluation.json"))
meta = {
  "experiment_id": "${EXP}",
  "cv_macro_f1": ev["scores"]["cv_macro_f1"],
  "predicted_lb": ev["lb_prediction"]["predicted_lb"],
  "lb_prediction_interval": [ev["lb_prediction"]["pi_low"], ev["lb_prediction"]["pi_high"]],
  "trust_level": ev["lb_prediction"]["trust_level"],
  "model_size_mb": ev["submission_readiness"]["model_size_mb"],
  "estimated_inference_min": ev["submission_readiness"]["estimated_full_test_minutes"],
  "dry_run_seconds": int(${DRY_SEC}),
  "zip_size_mb": int(${ZIP_MB}),
  "sha256": "${SHA}",
  "offline_check": "PASS",
  "packed_at": datetime.now().astimezone().isoformat(),
}
open(f"submissions/${EXP}.meta.json", "w").write(json.dumps(meta, indent=2))
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
- DO NOT upload anywhere. Manual submission only.
