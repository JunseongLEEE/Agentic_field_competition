---
description: "Competition dashboard. One screen: time state, daily quota, best CV, best LB, CV→LB correlation health, top experiments, recommended next action."
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Glob
---

# /status — Competition Dashboard

A single screen that tells you: how much time left, how many submissions used today,
what's the current best, how trustworthy CV is for predicting LB, and what to do next.

## Display

### 0. Time State + Quota

!`python scripts/check_time_state.py 2>/dev/null || echo "competition_meta.yaml missing"`

### 1. CV→LB Correlation Health

!`python scripts/cv_lb_correlation.py 2>/dev/null || echo "correlation script missing"`

### 2. Orchestrator State

!`python -c "import json; s=json.load(open('logs/orchestrator_state.json')); print('phase=',s.get('current_phase'),' best_cv=',s.get('best_cv'),' best_lb=',s.get('best_lb'),' stall=',s.get('stall_counter'),'/5  cycles=',s.get('total_cycles'))" 2>/dev/null || echo "orchestrator_state.json missing"`

### 3. Experiment Digest (top of file)

!`head -40 logs/experiment_digest.md 2>/dev/null || echo "experiment_digest.md missing — run scripts/build_digest.py"`

### 4. Recent Submissions

!`python -c "
import yaml, datetime
m = yaml.safe_load(open('competition_meta.yaml'))
log = m.get('submissions_log') or []
print(f'total submissions: {len(log)}')
for e in log[-5:]:
    print(f\"  {e.get('submitted_at','?')[:10]}  {e.get('experiment_id','?'):<28}  cv={e.get('cv_score','?')}  lb={e.get('lb_score','?')}  status={e.get('status','?')}\")
" 2>/dev/null || echo "no submissions logged"`

### 5. Latest Insights (last 3)

!`tail -n 3 logs/insights.jsonl 2>/dev/null | python -c "
import json, sys
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    print(f\"  {r.get('experiment','?'):<28}  gap={r.get('gap','?'):+.4f}  → {r.get('insight','')}\")
" 2>/dev/null || echo "  no insights yet"`

### 6. Recommended Next Action

Compute from the above:
- If `days_to_preliminary <= 1` and there is a CANDIDATE → `/rank` then manual submit.
- Else if `stall_counter >= 5` → `/plan` in diagnostic mode (CV audit, leakage probes).
- Else if `submissions_remaining_today > 0` AND `evaluation.json.recommendation == CANDIDATE` exists → `/rank`.
- Else if any experiment is missing `evaluation.json` → `/eval <that exp>`.
- Else if there is a packaged but unranked CANDIDATE → `/rank`.
- Else → `/plan`.

Print one line:
```
NEXT: /<skill> [args]   ← <one sentence why>
```

### 7. Warnings

Flag in red (text only — no color codes):
- `submissions_today >= 18` → "quota almost exhausted" (team quota is 20/day)
- `days_to_preliminary <= 2` → "deadline imminent — freeze new architectures"
- `cv_lb_trust == 'low' and submissions_count >= 3` → "submissions aren't improving the correlation model; collect more diverse picks"
- `best_lb is null and submissions_today >= 1` → "submitted but no LB recorded — run /submit-result"

## Output Skeleton

```
═════════════════════════════════════════════
COMPETITION DASHBOARD — <YYYY-MM-DD HH:MM KST>
═════════════════════════════════════════════
Deadline       : preliminary 2026-07-15  → D-<N>
Daily quota    : <used>/20  remaining=<R>

Best CV        : 0.XXXX (exp_NNN)
Best LB        : 0.XXXX (exp_NNN)
CV→LB model    : n=<k>  r=<r>  σ=<sigma>  trust=<level>

Recent experiments (top of digest):
  ...

Recent submissions:
  ...

Recent insights:
  ...

NEXT: /<skill> [args]   ← <reason>

Warnings:
  - ...
═════════════════════════════════════════════
```
