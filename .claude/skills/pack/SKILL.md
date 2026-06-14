---
description: "Package experiment as DACON code submission zip (model/ + script.py + requirements.txt). Pass experiment name as argument."
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

Create a submission zip following DACON code submission format.

## Arguments
- `$ARGUMENTS` — experiment name (e.g., "exp_001_baseline")

## Required Zip Structure
```
submit.zip
├── model/              # Trained model weights
│   └── (model files)
├── script.py           # Inference-only code
└── requirements.txt    # Extra packages
```

**NO other files allowed at top level.**

## Step 1: Locate Experiment

```bash
EXP_DIR=experiments/$0  # or search for match
ls $EXP_DIR/script.py $EXP_DIR/model/ $EXP_DIR/requirements.txt
```

## Step 2: Pre-packaging Validation

### 2a. Check required files exist
- [ ] `script.py` exists
- [ ] `model/` directory exists and is not empty
- [ ] `requirements.txt` exists

### 2b. Offline compatibility scan
```bash
# MUST NOT contain these patterns in script.py:
grep -n "from_pretrained\b" $EXP_DIR/script.py | grep -v "model/"  # HF hub downloads
grep -n "requests\.\|urllib\.\|wget\|curl\|download" $EXP_DIR/script.py
grep -n "api_key\|API_KEY\|openai\.\|anthropic\." $EXP_DIR/script.py
grep -n "\.download(" $EXP_DIR/script.py
```

If any match → **BLOCK packaging** and report the violation.

### 2c. Verify script.py structure
- [ ] Reads from `data/` directory
- [ ] Writes to `output/submission.csv`
- [ ] Has `if __name__ == '__main__'` block
- [ ] Loads model from `model/` directory (local path)

### 2d. Check model size
```bash
du -sh $EXP_DIR/model/
# Warn if > 500MB (server limits vary by competition)
```

## Step 3: Local Dry Run

Test that script.py actually works locally:
```bash
cd $EXP_DIR

# Create mock data/ directory if not exists (use actual test data)
mkdir -p data
cp ../../data/test.csv data/ 2>/dev/null || echo "No test.csv — dry run skipped"

# Run inference
python script.py

# Verify output
ls output/submission.csv
python -c "
import pandas as pd
sub = pd.read_csv('output/submission.csv')
print(f'Shape: {sub.shape}')
print(f'Columns: {list(sub.columns)}')
print(sub.head())
print(f'NaN: {sub.isnull().sum().sum()}')
"

# Cleanup
rm -rf data/ output/
```

## Step 4: Package

```bash
cd $EXP_DIR
mkdir -p ../../submissions

# Create zip with ONLY the required files
zip -r ../../submissions/${EXP_NAME}.zip \
    model/ \
    script.py \
    requirements.txt

# Verify zip contents
unzip -l ../../submissions/${EXP_NAME}.zip
```

## Step 5: Validate Zip

```bash
python ../../scripts/validate_submission.py --zip ../../submissions/${EXP_NAME}.zip
```

## Step 6: Metadata

```python
import hashlib, json
from datetime import datetime

zip_path = f'../../submissions/{EXP_NAME}.zip'
sha256 = hashlib.sha256(open(zip_path, 'rb').read()).hexdigest()

meta = {
    "experiment_id": EXP_NAME,
    "cv_score": cv_mean,  # from train_log.json
    "model_size_mb": model_size,
    "inference_ms_per_sample": speed,
    "sha256": sha256,
    "created_at": datetime.now().isoformat(),
    "offline_verified": True,
    "dry_run_passed": True
}
```

## Step 7: Report

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBMISSION PACKAGED: exp_NNN_name
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Zip:            submissions/exp_NNN_name.zip
Size:           XX MB
Model size:     XX MB
Offline check:  PASS ✓
Dry run:        PASS ✓
CV Score:       0.XXXX

Contents:
  model/          XX files, XX MB
  script.py       ✓
  requirements.txt ✓

⚠️  수동 제출 필요 — DACON 사이트에서 직접 업로드
다음 단계: /rank (후보 순위 확인)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
