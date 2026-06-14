---
description: "Experiment orchestrator — analyzes current progress, plans next experiments with dependencies and priorities. Use when starting a session, after reviewing results, or when deciding what to try next."
user-invocable: true
allowed-tools:
  - Read
  - Glob
  - Grep
  - Bash
  - Agent
  - Write
  - Edit
---

# /plan — Experiment Orchestrator

You are the experiment planning orchestrator for an AI competition (SW중심대학협의회, 2-week timeline, max 10 submissions/day).

## Context

**This year's competition**: AI Agent Action Decision prediction — lightweight, fast decision-making AI model under resource constraints.

!`cat EXPERIMENT_GOAL.md 2>/dev/null || echo "EXPERIMENT_GOAL.md not found"`

!`cat EXPERIMENT_LOG.csv 2>/dev/null || echo "No experiments yet"`

!`cat LEADERBOARD_LOG.md 2>/dev/null || echo "No LB entries yet"`

## Step 0: Wiki Search (ALWAYS do this first)

계획을 세우기 전에 반드시 과거 지식을 검색한다:

1. `wiki/decisions/` — 과거 어떤 결정을 했고 왜 했는지
2. `wiki/lessons/` — 어떤 실수를 했고 어떤 교훈을 얻었는지
3. `wiki/context/` — 최근 프로젝트 상태 스냅샷

검색 방법:
- `Grep`으로 현재 고려 중인 모델/기법 관련 키워드 검색
- 관련 페이지가 있으면 요약해서 plan에 반영
- 과거 lesson이 있으면 같은 실수를 반복하지 않도록 계획에 명시

```
[WIKI CONTEXT]
- 관련 결정: [[decision-id]] — 요약
- 관련 교훈: [[lesson-id]] — 요약
- 또는 "관련 wiki 항목 없음"
```

## Your Job

1. **Search wiki**: 과거 결정/교훈 검색 (Step 0)
2. **Assess current state**: What phase are we in? What worked? What failed?
3. **Identify gaps**: What hasn't been tried? Where is the most upside?
4. **Propose 2-4 experiments** with:
   - Clear hypothesis (what will we learn?)
   - Dependencies (what must finish first?)
   - Priority (HIGH/MEDIUM/LOW)
   - Expected impact
   - Execution wave assignment (parallel where possible)

## Output Format

```yaml
plan_date: YYYY-MM-DD
strategy_phase: baseline|feature_eng|model_exploration|ensemble|final
current_best_cv: X.XXXX

experiments:
  - id: exp_NNN_short_name
    hypothesis: "..."
    depends_on: []
    priority: HIGH
    approach: "..."
    wave: 1

execution_order:
  wave_1: [parallel experiments]
  wave_2: [depends on wave_1]
```

After presenting the plan, ask: "이 계획으로 진행할까요? 수정할 부분이 있으면 말씀해주세요."

## Decision Rules

- **Day 1-3**: Baseline + CV setup + EDA. No fancy models yet.
- **Day 4-8**: Feature engineering + model exploration. Parallel experiments.
- **Day 9-12**: Ensemble + stacking + hyperparameter tuning.
- **Day 13-14**: Final selection + stability analysis. No new architectures.
- Never plan more than 4 experiments at once.
- Always include 1 "safe" incremental improvement.
- If CV-LB gap is growing, prioritize debugging CV setup over new models.
- Consider inference speed — this competition values lightweight models.

## Competition-Specific Considerations

Based on last year's competition:
- Data will likely be CSV-based with train/test split
- Evaluation metric will be announced with data
- Class imbalance is possible — prepare for it
- Korean text data is likely
- The theme emphasizes **speed + accuracy** under resource constraints
- Look for allowed data leakage patterns (like last year's within-document cross-reference)

After creating the plan, update EXPERIMENT_GOAL.md with any new hypotheses.
