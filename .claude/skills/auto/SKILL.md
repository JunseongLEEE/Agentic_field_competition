---
description: "Full autonomous experiment cycle — plans, implements, runs, evaluates in one shot. Loops automatically until improvement stalls. Use this to let the system work while you're away."
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

# /auto — Autonomous Experiment Pipeline

You are a fully autonomous experiment orchestrator. You plan, implement, run, and evaluate experiments WITHOUT human intervention.

**CRITICAL**: You may be running in a fresh session with NO prior context (e.g., via /loop). 
ALL state comes from bridge files. NEVER assume you know what happened before — READ first.

## Arguments
- `$ARGUMENTS` — optional: number of cycles (default: 5), or "until_stall"

## Guardrails (NEVER violate these)

1. **NEVER auto-submit** to competition website
2. **STOP after 5 consecutive cycles with no CV improvement**
3. **STOP if any experiment produces NaN/Inf predictions**
4. **STOP if runtime exceeds 60 minutes per single experiment**
5. **Max 20 experiments total per /auto invocation**
6. **Always log everything** — even failures

---

## STEP 0: Context Recovery + Wiki Search (ALWAYS run first)

Every time /auto starts, reconstruct full situational awareness from files AND wiki:

```python
import json

# 1. Load orchestrator brain
state = json.load(open('logs/orchestrator_state.json'))

# 2. Load recent cycle history (only last N)
recent_n = state.get('recent_context_window', 5)
with open('logs/cycle_history.jsonl') as f:
    all_cycles = [json.loads(line) for line in f if line.strip()]
recent_cycles = all_cycles[-recent_n:]  # ONLY look at recent N

# 3. Load experiment log
import csv
with open('EXPERIMENT_LOG.csv') as f:
    experiments = list(csv.DictReader(f))

# 4. Check queue for stuck experiments
queue = json.load(open('logs/experiment_queue.json'))

# 5. Load experiment digest (ALL experiments at a glance)
digest = open('logs/experiment_digest.md').read() if Path('logs/experiment_digest.md').exists() else ""

# 6. Load recent submission insights (CV-LB feedback)
insights = []
if Path('logs/insights.jsonl').exists():
    with open('logs/insights.jsonl') as f:
        insights = [json.loads(l) for l in f if l.strip()][-5:]  # last 5 only
```

**Wiki Search** (Compound Engineering):
- `wiki/lessons/` — 과거 실수를 반복하지 않기 위해 최근 교훈 검색
- `wiki/decisions/` — 이전 결정 맥락 파악
- Grep으로 현재 phase 관련 키워드 검색
- 관련 항목이 있으면 plan에 반영

Print a situational summary:
```
[RECOVERY] Phase: {phase} | Best CV: {best_cv} ({best_exp}) | Stall: {stall}/5 | Total cycles: {n}
[RECENT] Last {N} experiments: {names and scores}
[INSIGHTS] {latest insight from submissions, or "No LB data yet"}
[NEXT] {state['next_action']} — {state['last_reasoning']}
```

**Use the experiment digest** to understand what's been tried and what worked.
**Use insights** to factor in CV-LB patterns (e.g., "CatBoost generalizes better", "text features increase gap").

Then proceed to the appropriate phase based on `state['next_action']`.

---

## Bridge Files (Agent Communication Protocol)

All agents communicate through these shared files. READ them before acting, WRITE after completing.

### `logs/orchestrator_state.json` — Orchestrator's brain
```json
{
  "last_updated": "2026-05-17T12:00:00",
  "current_phase": "baseline",
  "best_cv": 0.0,
  "best_experiment": null,
  "stall_counter": 0,
  "total_cycles": 0,
  "strategy_history": [],
  "next_action": "plan",
  "active_experiments": [],
  "blocked_approaches": []
}
```

### `logs/experiment_queue.json` — Pending experiments
```json
{
  "queue": [
    {
      "id": "exp_001_baseline_lgbm",
      "status": "PLANNED|IMPLEMENTING|RUNNING|EVALUATING|DONE|FAILED",
      "assigned_at": "...",
      "completed_at": null,
      "error": null
    }
  ]
}
```

### `logs/agent_messages.jsonl` — Inter-agent message log
Each line is a message:
```json
{"timestamp": "...", "from": "orchestrator", "to": "all", "type": "plan", "content": "Starting cycle 3, trying CatBoost with text features"}
{"timestamp": "...", "from": "runner", "to": "orchestrator", "type": "result", "content": "exp_003 completed: CV=0.8234"}
{"timestamp": "...", "from": "evaluator", "to": "orchestrator", "type": "alert", "content": "exp_003 has high fold variance, marking REVIEW"}
```

---

## Autonomous Loop

### CYCLE START: Read State

```python
import json
state = json.load(open('logs/orchestrator_state.json'))
cycle = state['total_cycles'] + 1
```

### Phase 1: PLAN (Orchestrator decides)

Based on state, decide what to try:

**Strategy progression:**
```
Phase "baseline" (cycles 1-3):
  → Try LightGBM, XGBoost, CatBoost with minimal features
  → Goal: establish CV baseline and verify pipeline works
  
Phase "feature_eng" (cycles 4-7):
  → Feature engineering on best baseline model
  → Text features, aggregations, interactions
  
Phase "model_tuning" (cycles 8-12):
  → Hyperparameter optimization on best model+features
  → Try different architectures (small NN, etc.)
  
Phase "ensemble" (cycles 13-16):
  → Blend/stack best diverse models
  → Optimize weights
  
Phase "final" (cycles 17-20):
  → Stability analysis across seeds
  → Final candidate selection
```

**Adaptation rules:**
- If stall_counter >= 3 → change strategy (skip to next phase or try blocked_approaches)
- If an approach fails → add to blocked_approaches
- If improvement found → reset stall_counter, note what worked

### Phase 2: IMPLEMENT (Spawn Dev Agent)

Write experiment plan to queue, then spawn:

```
Agent(
  description="Implement exp_NNN",
  prompt="You are a model developer. Implement the following experiment:
    
    Experiment: exp_NNN_name
    Hypothesis: [...]
    Model: [...]
    Features: [...]
    
    Create in experiments/exp_NNN_name/:
    - config.yaml (all params, seed=42)
    - train.py (5-fold CV, outputs oof_preds.npy, test_preds.npy, train_log.json)
    
    Data paths: ../../data/train.csv, ../../data/test.csv
    
    IMPORTANT: 
    - train_log.json MUST have: cv_scores (list), cv_mean (float), cv_std (float)
    - Measure inference time per sample
    - Handle missing values gracefully
    
    When done, update logs/experiment_queue.json status to IMPLEMENTING→DONE
    Write a message to logs/agent_messages.jsonl confirming completion.",
  run_in_background=false
)
```

### Phase 3: RUN (Execute)

```bash
cd experiments/exp_NNN_name && timeout 3600 python train.py 2>&1 | tee run_output.txt
```

Update queue status: RUNNING → check results.

### Phase 4: EVALUATE (Inline or Spawn Agent)

```python
import json, numpy as np

# Load results
with open(f'experiments/{exp}/train_log.json') as f:
    results = json.load(f)

cv_mean = results['cv_mean']
cv_std = results['cv_std']

# Compare against best
state = json.load(open('logs/orchestrator_state.json'))
improvement = cv_mean - state['best_cv'] if state['best_cv'] > 0 else cv_mean

# Leakage check
oof = np.load(f'experiments/{exp}/oof_preds.npy')
test = np.load(f'experiments/{exp}/test_preds.npy')
distribution_shift = abs(oof.mean() - test.mean()) / max(oof.std(), 1e-8)

# Decision
if np.any(np.isnan(oof)) or np.any(np.isnan(test)):
    status = "FAILED"
    reason = "NaN detected"
elif distribution_shift > 1.0:
    status = "REVIEW"
    reason = f"High distribution shift: {distribution_shift:.2f}"
elif improvement > 0:
    status = "CANDIDATE"
    reason = f"Improved by {improvement:+.6f}"
else:
    status = "COMPLETED"
    reason = f"No improvement ({improvement:+.6f})"

# DACON code submission checks
inference_speed = results.get('inference_ms_per_sample', None)
model_size = results.get('model_size_mb', None)
offline_ok = results.get('offline_compatible', None)

if offline_ok is False:
    status = "FAILED"
    reason += " | script.py has online dependencies"

# Check script.py exists for CANDIDATE
script_path = Path(f'experiments/{exp}/script.py')
if status == "CANDIDATE" and not script_path.exists():
    status = "REVIEW"
    reason += " | script.py missing — cannot package for DACON"
```

### Phase 5: UPDATE STATE

```python
import json
from datetime import datetime

state = json.load(open('logs/orchestrator_state.json'))

state['total_cycles'] = cycle
state['last_updated'] = datetime.now().isoformat()

if status == "CANDIDATE":
    state['best_cv'] = cv_mean
    state['best_experiment'] = exp_name
    state['stall_counter'] = 0
else:
    state['stall_counter'] += 1

# Phase transition logic
if state['stall_counter'] >= 3:
    phases = ["baseline", "feature_eng", "model_tuning", "ensemble", "final"]
    current_idx = phases.index(state['current_phase'])
    if current_idx < len(phases) - 1:
        state['current_phase'] = phases[current_idx + 1]
        state['stall_counter'] = 0
        # Log strategy change
        state['strategy_history'].append({
            "cycle": cycle,
            "change": f"Advanced to {state['current_phase']}",
            "reason": "3 cycles without improvement"
        })

json.dump(state, open('logs/orchestrator_state.json', 'w'), indent=2)
```

**Write cycle summary to `logs/cycle_history.jsonl`** (append one line):
```python
cycle_record = {
    "cycle": cycle,
    "timestamp": datetime.now().isoformat(),
    "experiment": exp_name,
    "cv_mean": cv_mean,
    "cv_std": cv_std,
    "status": status,
    "phase": state['current_phase'],
    "best_cv": state['best_cv'],
    "best_experiment": state['best_experiment'],
    "stall_counter": state['stall_counter'],
    "reasoning": "WHY you chose this experiment and what you learned",
    "next_plan": "What the NEXT cycle should try based on this result"
}
with open('logs/cycle_history.jsonl', 'a') as f:
    f.write(json.dumps(cycle_record, ensure_ascii=False) + '\n')
```

**Update `orchestrator_state.json` with reasoning for next session**:
```python
state['last_reasoning'] = cycle_record['next_plan']
state['next_action'] = 'plan'  # or 'retry' if fixable error
json.dump(state, open('logs/orchestrator_state.json', 'w'), indent=2, ensure_ascii=False)
```

Update EXPERIMENT_LOG.csv with new row.

### Phase 6: STOP CHECK

```python
if state['stall_counter'] >= 5:
    STOP("5 consecutive cycles without improvement")
if cycle >= max_cycles:
    STOP("Max cycles reached")
if status == "FAILED" and "NaN" in reason:
    STOP("Critical error: NaN in predictions")
```

### Phase 7: COMPOUND (Knowledge Capture)

매 사이클 종료 시 이번 실험에서 얻은 지식을 wiki에 축적:

1. **Lesson이 있다면** (`wiki/lessons/`에 작성):
   - 예상과 다른 결과, 디버깅 과정, 실수 등
   
2. **Decision이 있다면** (`wiki/decisions/`에 작성):
   - 모델/피처/하이퍼파라미터 선택과 그 이유

3. **Context 업데이트** (`wiki/context/`):
   - 매 5사이클마다 전체 상태 스냅샷 저장

4. `wiki/_meta/index.md` 갱신

**경량 compound**: 모든 사이클마다 full compound할 필요 없음.
- 개선이 있었거나 실패했을 때만 lesson/decision 작성
- 매 5사이클마다 context 스냅샷

### Phase 8: REPORT

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CYCLE {cycle}/{max_cycles} COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Experiment: exp_NNN_name
CV Score:   0.XXXX ± 0.XXXX
Best Ever:  0.XXXX (exp_NNN)
Inference:  XX ms/sample | Model: XX MB
Submission: script.py ✓/✗ | offline ✓/✗
Status:     {status}
Phase:      {current_phase}
Stall:      {stall_counter}/5

Next: {what_next}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Then LOOP to Phase 1 for next cycle.

---

## End Summary

When stopping:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AUTONOMOUS RUN COMPLETE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total cycles:    N
Successful:      M
Failed:          K
Best CV:         0.XXXX (exp_NNN_name)
Phase reached:   {phase}
Stop reason:     {reason}

Top candidates:
1. exp_NNN — CV 0.XXXX ± 0.XXXX [CANDIDATE]
2. exp_NNN — CV 0.XXXX ± 0.XXXX [CANDIDATE]

Strategy log: logs/orchestrator_state.json
Full log: EXPERIMENT_LOG.csv

Wiki: wiki/에 {N}개 lesson, {M}개 decision 축적됨

다음: /rank 실행 후 수동 제출
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
