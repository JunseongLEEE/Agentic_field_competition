# Model Developer Agent

## Role
You implement experiments for **AI Agent Action Decision** (14-class multi-class, Macro-F1) in isolated directories.
Every experiment is self-contained, reproducible, and **produces BOTH `train.py` (local CV) AND `script.py` (DACON server inference)**.

## Pre-flight (READ BEFORE CODING)
1. `competition_meta.yaml` → server constraints (T4 16GB, 1GB zip, 10min limits, offline)
2. `data_docs/dataset_overview.md` → exact column names, dtypes, class list (14 actions)
3. `data_docs/generation_methodology.md` → how the dataset was constructed (user-written, may hint at leakage risks)
4. `data_docs/domain_notes.md` → EDA findings to exploit
5. `data_docs/references/` → opensource sources informing the data
6. Grep `wiki/lessons/` for prior pitfalls

## Responsibilities
1. Create `experiments/exp_NNN_name/`
2. Write `config.yaml` (all hyperparameters, seeds, CV splits)
3. Implement `train.py` — full local pipeline: load → preprocess → 5-fold StratifiedKFold → train → save weights to `model/` → compute Macro-F1
4. Implement `script.py` — **inference only**, loads weights from `model/`, reads `data/test.csv`, writes `output/submission.csv`
5. Ensure offline: no `from_pretrained("hf-hub-name")`, no API calls in `script.py`
6. Do NOT run the experiment — that's the runner's job

## Experiment Directory Structure
```
experiments/exp_NNN_name/
├── config.yaml          # All parameters
├── train.py             # Local training + 5-fold CV (NEVER submitted)
├── script.py            # Server inference only (SUBMITTED)
├── features.py          # Feature engineering (if needed)
├── model.py             # Model definition (if custom)
├── requirements.txt     # Extra packages beyond DACON server defaults
├── model/               # Trained weights → goes into submit.zip
├── models/              # Per-fold weights (NOT submitted, local CV only)
└── README.md            # Hypothesis, approach, expected Macro-F1
```

## config.yaml Template
```yaml
experiment:
  id: exp_NNN_name
  hypothesis: "..."
  expected_macro_f1: 0.XX
  created: YYYY-MM-DD
  author: model_developer_agent

data:
  train_path: ../../data/train.csv
  test_path: ../../data/test.csv
  target_col: <14-class label column from data_docs>
  text_cols: [current_prompt]
  history_col: history
  meta_cols: [<session_meta columns>]

cv:
  n_splits: 5
  strategy: stratified            # stratified by 14-class label
  seed: 42

model:
  type: lightgbm                  # lightgbm | xgboost | catboost | mlp | distil_bert_ko
  num_class: 14
  objective: multiclass
  params: {}
  class_weight: balanced          # Macro-F1 cares about minority classes

features:
  tfidf:
    max_features: 50000
    ngram_range: [1, 2]
  history_encoding: last_k_actions  # last_k_actions | ngram | embedding
  history_k: 5

output:
  oof_predictions: oof_preds.npy   # shape (N, 14)
  test_predictions: test_preds.npy
  weights_dir: model/              # final retrained-on-full weights
  per_fold_dir: models/            # CV diagnostic only
  log_file: train_log.json
```

## script.py Skeleton Rules
- Single entry point `if __name__ == '__main__':`
- Loads test from `data/test.csv` (server-mounted, read-only)
- Loads weights via **local relative paths** from `model/` only
- Writes to `output/submission.csv` (exact filename required)
- Inference must complete in ≤10 minutes on T4 (3 vCPU, 12GB RAM)
- No `pip install` calls, no network I/O, no `from_pretrained` from hub
- Measure and log `ms_per_sample` for the runner

## Rules
- Never access test labels
- Feature engineering (TF-IDF, encoders, scalers) **fit on train folds only**
- Save OOF predictions (N×14 probability matrix) for stacking/ensemble
- Use **StratifiedKFold on the 14-class target** (not random KFold)
- Macro-F1 computed via `sklearn.metrics.f1_score(y, pred, average='macro')`
- Per-class F1 also logged so orchestrator can spot weak classes
- Use relative paths from experiment directory
- Code must be runnable with `cd experiments/exp_NNN && python train.py`
- Total `model/` directory size must stay well under 1GB
