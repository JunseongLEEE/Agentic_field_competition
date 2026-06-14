---
description: "Implement an experiment — creates isolated experiment directory with training code AND DACON inference code. Pass experiment name or plan reference as argument."
user-invocable: true
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
  - Agent
---

# /dev — Model Developer (DACON Code Submission)

You implement experiments for an AI competition. Each experiment produces TWO codebases:
1. **train.py** — local training with CV (never submitted)
2. **script.py** — server-side inference only (this gets submitted)

## Arguments
- `$ARGUMENTS` — experiment name or description (e.g., "baseline_lgbm", "transformer_small")

## DACON Submission Format (MUST follow)

The final submission zip structure:
```
submit.zip
├── model/              # Trained model weights (saved by train.py)
│   └── model.pt        # Or .pkl, .bin, .onnx, etc.
├── script.py           # Inference ONLY (reads data/ → writes output/submission.csv)
└── requirements.txt    # Extra packages beyond server defaults
```

**Critical constraints:**
- script.py runs in **OFFLINE** environment — NO internet access
- NO `from_pretrained("model-name")` from HuggingFace Hub
- NO API calls (OpenAI, etc.)
- ALL model files must be in `model/` directory
- Output MUST go to `output/submission.csv`
- Data is at `data/test.csv` (server provides this)

## Experiment Directory Structure

```
experiments/exp_NNN_name/
├── config.yaml          # All parameters
├── train.py             # LOCAL training + CV (produces model weights)
├── script.py            # INFERENCE ONLY (copy of what gets submitted)
├── requirements.txt     # Extra packages for submission
├── model/               # Saved model weights after training
├── models/              # Per-fold models (for CV, not submitted)
├── SUMMARY.md           # Experiment memory (from template)
└── README.md            # Hypothesis, approach
```

## Step 1: Read Context

Before implementing, read:
```
logs/experiment_digest.md    — what's been tried, what worked
logs/insights.jsonl          — CV-LB patterns from past submissions
Competition_desription.md    — competition theme
```

## Step 2: Create Experiment Directory

```bash
python scripts/create_experiment.py --name "$0" --hypothesis "$ARGUMENTS"
```

Or create manually if script doesn't fit.

## Step 3: Implement train.py

The training script MUST:

1. **Load config.yaml** for all parameters
2. **Set seeds** everywhere (numpy, random, torch, etc.)
3. **Implement CV loop** (5-fold stratified by default)
4. **Save outputs**:
   - `oof_preds.npy` — out-of-fold predictions
   - `test_preds.npy` — test predictions (mean of folds)
   - `models/` — per-fold models (for CV analysis)
   - `model/` — **final model for submission** (trained on all data or best fold)
   - `train_log.json` — structured results
5. **Measure inference speed**: time per sample on test data
6. **Report model size**: total size of model/ directory in MB
7. **Be runnable** with: `cd experiments/exp_NNN && python train.py`

### train_log.json Format (MUST follow)

```json
{
  "experiment_id": "exp_NNN_name",
  "cv_scores": [0.XX, 0.XX, 0.XX, 0.XX, 0.XX],
  "cv_mean": 0.XXXX,
  "cv_std": 0.XXXX,
  "metric_name": "auc or f1 or rmse etc",
  "runtime_seconds": 123.4,
  "inference_ms_per_sample": 2.5,
  "model_size_mb": 150.0,
  "n_features": 50,
  "feature_importance": {"feat1": 0.1, "feat2": 0.05},
  "offline_compatible": true
}
```

## Step 4: Implement script.py (INFERENCE ONLY)

This is what runs on DACON server. Template:

```python
import os
import pandas as pd
# Other OFFLINE-compatible imports only

def load_model():
    """Load trained model from model/ directory."""
    model_path = os.path.join('model', 'your_model_file')
    # Load model — LOCAL FILES ONLY, no internet
    return model

def load_data():
    """Load test data from data/ directory (provided by server)."""
    data_path = os.path.join('data', 'test.csv')
    data = pd.read_csv(data_path)
    return data

def preprocess(data):
    """Apply same preprocessing as training."""
    # MUST match train.py preprocessing exactly
    return processed_data

def predict(model, data):
    """Run inference."""
    predictions = model.predict(data)
    return predictions

def save_results(predictions):
    """Save to output/submission.csv."""
    os.makedirs('output', exist_ok=True)
    # Format must match sample_submission.csv
    submission = pd.DataFrame({
        'ID': ...,          # Match sample_submission format
        'target': predictions
    })
    submission.to_csv(os.path.join('output', 'submission.csv'), index=False)

if __name__ == '__main__':
    model = load_model()
    data = load_data()
    processed = preprocess(data)
    predictions = predict(model, processed)
    save_results(predictions)
    print("추론 완료!")
```

### script.py Rules
- **NO training code** — inference only
- **NO internet calls** — everything local
- **NO `from_pretrained()` with model names** — use local paths only
- Reads from `data/` (server provides)
- Writes to `output/submission.csv`
- Must handle edge cases (missing values, unexpected data)
- Must be fast — inference speed is evaluated

## Step 5: Create requirements.txt

Only include packages NOT already on DACON server:
```txt
# Only extra packages needed for inference
# Do NOT include: pandas, numpy, scikit-learn, torch (usually pre-installed)
lightgbm==4.3.0
```

## Step 6: Create SUMMARY.md

Copy from `experiments/TEMPLATE_SUMMARY.md` and fill in Setup + Inference Constraints sections.

## Step 7: Verify Offline Compatibility

Scan script.py for violations:
```bash
# These patterns should NOT appear in script.py:
grep -n "from_pretrained\|download\|api_key\|requests.get\|urllib\|wget\|curl" script.py
```

## After Implementation

```
Experiment: exp_NNN_name
Files created:
  ✓ config.yaml
  ✓ train.py       (training + CV)
  ✓ script.py      (inference only — DACON submission)
  ✓ requirements.txt
  ✓ SUMMARY.md
  ✓ model/         (empty — populated after training)

Offline check: PASS/FAIL
Ready to run: /run exp_NNN_name
```

Do NOT run the experiment — that's `/run`'s job.
