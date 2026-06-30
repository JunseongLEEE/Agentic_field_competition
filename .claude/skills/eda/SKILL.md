---
description: "Quick EDA on competition data — analyzes shape, distributions, missing values, class balance, and key patterns. Run at competition start or when new data arrives."
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Glob
---

# /eda — Exploratory Data Analysis

Fast, structured EDA for competition data. Designed to extract actionable insights quickly.

## Arguments
- `$ARGUMENTS` — optional: specific focus area (e.g., "target distribution", "text lengths", "missing values")

## Step 0: Read Data Documentation

EDA를 시작하기 전 `data_docs/`를 모두 읽는다. 데이터셋이 어떻게 만들어졌는지를 알면 EDA에서 무엇을 봐야 할지가 명확해진다.

```bash
ls data_docs/ 2>/dev/null && cat data_docs/*.md
```

특히 확인할 것:
- `generation_methodology.md` — 데이터가 어떻게 만들어졌나
- `references/` — 어떤 오픈소스를 참고했나
- `domain_notes.md` — 이전 EDA에서 발견된 패턴

EDA 결과는 `domain_notes.md`에 누적 기록 (덮어쓰기 금지, append).

## Step 1: Data Overview

```python
import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

print("=== SHAPES ===")
print(f"Train: {train.shape}, Test: {test.shape}")

print("\n=== COLUMNS ===")
print(f"Train: {list(train.columns)}")
print(f"Test: {list(test.columns)}")
print(f"Train-only: {set(train.columns) - set(test.columns)}")  # likely includes target

print("\n=== DTYPES ===")
print(train.dtypes)

print("\n=== MISSING VALUES ===")
print(train.isnull().sum())
print(test.isnull().sum())

print("\n=== TARGET DISTRIBUTION ===")
target_col = (set(train.columns) - set(test.columns)).pop()  # guess target
print(train[target_col].value_counts(normalize=True))
```

## Step 2: Feature Analysis

```python
print("\n=== NUMERIC FEATURES ===")
print(train.describe())

print("\n=== CATEGORICAL FEATURES ===")
for col in train.select_dtypes(include='object').columns:
    print(f"\n{col}: {train[col].nunique()} unique values")
    print(train[col].value_counts().head(5))

print("\n=== TEXT FEATURES ===")
for col in train.select_dtypes(include='object').columns:
    lengths = train[col].str.len()
    print(f"\n{col} length: mean={lengths.mean():.0f}, median={lengths.median():.0f}, max={lengths.max()}")
```

## Step 3: Train-Test Consistency

```python
# Check for distribution shift between train and test
for col in test.columns:
    if col in train.columns:
        if train[col].dtype == 'object':
            train_vals = set(train[col].dropna().unique())
            test_vals = set(test[col].dropna().unique())
            unseen = test_vals - train_vals
            if unseen:
                print(f"⚠️  {col}: {len(unseen)} unseen values in test!")
```

## Step 4: Competition-Specific Analysis

Based on competition theme (AI Agent Action Decision):
- Check for action/label columns and their distribution
- Look for temporal patterns (timestamps, sequences)
- Check if data has group structure (sessions, users, agents)
- Measure text/feature complexity for model size estimation
- Estimate inference time budget

## Step 5: Report

Write findings to `logs/eda_report.md`:

```markdown
# EDA Report — YYYY-MM-DD

## Data Summary
- Train: NNN rows × MMM cols
- Test: NNN rows × MMM cols
- Target: [col_name], [distribution]

## Key Findings
1. [most impactful finding]
2. [second finding]
3. [third finding]

## Recommended Approach
- Model type: [suggestion]
- Key features: [list]
- CV strategy: [stratified/group/time]
- Watch out for: [pitfalls]

## Action Items
- [ ] [first thing to try]
- [ ] [second thing to try]
```

Then print: "EDA 완료. `/plan`으로 실험 계획을 세우세요."
