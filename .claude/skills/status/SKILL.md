---
description: "Show competition dashboard — all experiments, scores, submission history, and current strategy at a glance."
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Glob
---

# /status — Competition Dashboard

Show a complete overview of competition progress.

## Display

### 1. Experiment Summary

!`cat EXPERIMENT_LOG.csv 2>/dev/null || echo "No experiments yet"`

Format as a clean table:
```
EXPERIMENTS                          CV Score   Std      Status
─────────────────────────────────────────────────────────────────
exp_001_baseline_lgbm                0.8234     0.0045   COMPLETED
exp_002_feature_eng_v1               0.8301     0.0038   CANDIDATE
exp_003_catboost_tuned               0.8289     0.0051   EVALUATED
```

### 2. Best Scores

```
LEADERBOARD
─────────────────────────────
Best CV:     0.XXXX (exp_NNN)
Best LB:     0.XXXX (exp_NNN)
CV-LB Gap:   0.XXXX (avg)
```

### 3. Submission Status

!`cat LEADERBOARD_LOG.md 2>/dev/null | tail -10`

```
TODAY'S SUBMISSIONS: X/10 used
TOTAL SUBMISSIONS:   XX
```

### 4. Strategy Phase

!`cat EXPERIMENT_GOAL.md 2>/dev/null | head -30`

### 5. Quick Recommendations

Based on current state, suggest:
- What to try next (1-2 sentences)
- Any warnings (CV-LB divergence, time running out, etc.)
- Remaining competition days estimate

End with: "다음 행동: `/plan` (계획) | `/dev NAME` (구현) | `/eda` (데이터 분석)"
