---
description: "Structured EDA for the 14-class AI Agent Action Decision dataset. Reads data_docs first, then probes schema, class balance, history, session_meta, and class-level signal. Appends findings to data_docs/domain_notes.md."
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Glob
---

# /eda — Exploratory Data Analysis

Goal: in one pass, surface the facts that change planning decisions.
Output is appended (never overwritten) to `data_docs/domain_notes.md` and a one-shot snapshot to `logs/eda_report.md`.

## Arguments
- `$ARGUMENTS` — optional focus area (`target_distribution`, `history`, `session_meta`, `text_length`, `leakage_probe`).

## STEP 0 — Read data_docs/ FIRST

```bash
ls data_docs/ 2>/dev/null
cat data_docs/dataset_overview.md       # exact column names, dtypes
cat data_docs/generation_methodology.md # how the data was constructed
cat data_docs/domain_notes.md           # prior EDA findings (do not duplicate)
ls data_docs/references/
```

You must know the column names and the dataset construction process before profiling the data.

## STEP 1 — Schema and Shape

Data is **JSONL, not CSV**. `data/train.jsonl` (70,000 rows; keys `id`,
`session_meta`, `history`, `current_prompt`) is joined by `id` to
`data/train_labels.csv` (`id,action`). `data/test.jsonl` in the repo is a 5-row
SAMPLE (real server test = 30,000 hidden). `data/sample_submission.csv` stays CSV
(cols `id,action`). Use the loaders in
`experiments/exp_001_tfidf_lightgbm/features.py`.

```python
import pandas as pd, numpy as np, json
from sys import path; path.insert(0, 'experiments/exp_001_tfidf_lightgbm')
from features import load_jsonl

train_raw = load_jsonl('data/train.jsonl')                 # list[dict]
test_raw  = load_jsonl('data/test.jsonl')                  # 5-row sample
train = pd.json_normalize(train_raw)                       # flattens session_meta.*
test  = pd.json_normalize(test_raw)
labels = pd.read_csv('data/train_labels.csv')              # id,action (action = 14 strings)
train = train.merge(labels, on='id', how='left')          # attach target `action`
sample = pd.read_csv('data/sample_submission.csv')

print("SHAPES")
print(f"  train: {train.shape}")
print(f"  test : {test.shape}")
print(f"  sample_submission: {sample.shape}")

print("\nCOLUMNS")
print(f"  train: {list(train.columns)}")
print(f"  test : {list(test.columns)}")
print(f"  train-only (likely target + leakage candidates): {sorted(set(train.columns) - set(test.columns))}")

print("\nDTYPES")
print(train.dtypes)

print("\nMISSING (train)")
print(train.isnull().sum().sort_values(ascending=False).head(10))
print("\nMISSING (test)")
print(test.isnull().sum().sort_values(ascending=False).head(10))
```

## STEP 2 — 14-Class Target Distribution (Macro-F1 critical)

```python
target = 'action'   # 14 snake_case STRING classes (from train_labels.csv), NOT ints 0-13
vc = train[target].value_counts(normalize=True).sort_index()
print("CLASS DISTRIBUTION")
print(vc)
print(f"\nimbalance ratio (max/min): {vc.max() / max(vc.min(), 1e-9):.2f}")
print(f"smallest classes (top-3 minority):")
print(vc.sort_values().head(3))
```

Flag in report if:
- any class < 1% of train → Macro-F1 fragile; focal loss / class_weight critical
- imbalance ratio > 50 → consider SMOTE / weighted sampling experiment

## STEP 3 — Input Field Analysis

For each of `current_prompt`, `history`, `session_meta`:

```python
for col in [<from data_docs>]:
    print(f"\n=== {col} ===")
    print(f"  dtype          : {train[col].dtype}")
    if train[col].dtype == 'object':
        lens = train[col].fillna('').str.len()
        print(f"  length p50/p95/max: {lens.quantile(.5):.0f} / {lens.quantile(.95):.0f} / {lens.max()}")
        print(f"  empty/null     : {(train[col].fillna('').str.len() == 0).sum()}")
    else:
        print(train[col].describe())
```

If `history` is JSON-encoded, parse and report:
- avg number of actions in history
- distribution of "last action" type vs target (this is a candidate strong predictor)

## STEP 4 — Train/Test Distribution Shift

```python
for col in test.columns:
    if col in train.columns and train[col].dtype == 'object':
        tr = set(train[col].dropna().unique())
        te = set(test[col].dropna().unique())
        unseen = te - tr
        if unseen:
            print(f"WARN unseen-in-train: {col} has {len(unseen)} novel values in test")
```

For numeric/meta columns, run a 2-sample KS test or compare quantiles; flag if p < 0.01.

## STEP 5 — Leakage Probes (specific to AI Agent Action Decision)

1. Does `history` ever literally contain the next action label? (If so: the generation method leaks.)
2. **Session grouping (KNOWN — verify, don't re-derive wrong):** the session id is
   `id.rsplit("-step",1)[0]`. 9,429 sessions span 70,000 rows (99.69% multi-step),
   so same-session steps share workspace/meta + overlapping history prefixes.
   **`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by
   session is MANDATORY**; plain KFold/StratifiedKFold leaks session context and
   inflates CV. Report rows-per-session distribution to keep this front-of-mind.
   ```python
   sid = train['id'].str.rsplit('-step', n=1).str[0]
   print("sessions:", sid.nunique(), " rows:", len(train),
         " multi-step %:", round(100*(sid.value_counts()>1).reindex(sid).mean(),2))
   ```
3. Is `session_meta.budget_tokens_remaining` strongly correlated with target? Could be informative or leakage depending on data construction (EDA found it near-noise).

## STEP 6 — Inference-Budget Forecast

Estimate, in advance, what model sizes will fit:

```python
# rough token count per test row, assuming TF-IDF / word-level
text_tokens = test['current_prompt'].fillna('').str.split().str.len()
hist_tokens = test['history'].fillna('').str.split().str.len()
print(f"per-row tokens p50/p95: text {text_tokens.quantile(.5):.0f}/{text_tokens.quantile(.95):.0f}, history {hist_tokens.quantile(.5):.0f}/{hist_tokens.quantile(.95):.0f}")
print(f"test size: {len(test)}  → est. tokens to embed: {(text_tokens + hist_tokens).sum():.0f}")
```

Use this to filter out architectures that cannot fit the 10-minute inference budget on T4.

## STEP 7 — Write Findings

Append to `data_docs/domain_notes.md`:
```markdown
## EDA <YYYY-MM-DD>
- Train/test shape: ...
- Class imbalance: ratio max/min = ..., minority classes: [...]
- Key fields: ...
- Distribution shift flags: ...
- Leakage probes: ...
- Inference budget hint: ...

Implications for planning:
- ...
```

And to `logs/eda_report.md` (overwrite — one-shot snapshot):
```markdown
# EDA Snapshot — <date>
[full numbers, plots references, etc.]
```

## STEP 8 — Recommended Next Experiments

Print 2–3 concrete experiment ideas that `/plan` can pick up:

```
Recommended next experiments:
1. baseline: TF-IDF(current_prompt, 1-2gram) + LightGBM, class_weight=balanced
2. add last-3 actions from history as categorical features
3. probe: GroupKFold by session_id if session_id present
```

End with: `EDA done. Run /plan to convert findings into experiments.`

## Hard Rules

- ALWAYS append to `domain_notes.md`, never overwrite.
- ALWAYS read `data_docs/` before profiling.
- NEVER skip the 14-class distribution check — Macro-F1 evaluation hinges on minority classes.
- NEVER design a model from EDA alone; this skill produces facts, `/plan` produces experiments.
