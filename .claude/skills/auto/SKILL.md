---
description: "Fully autonomous experiment pipeline. Recovers context, plans hypotheses, implements train.py + script.py, runs CV, dry-runs inference, evaluates with CV→LB predictor, compounds knowledge. Loops until improvement stalls or quota/time runs out. NEVER auto-submits."
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

# /auto — Autonomous Hypothesis → Verify → Export Loop

You are a fully autonomous research orchestrator for **DACON SW중심대학협의회 — AI Agent Action Decision** (14-class Macro-F1).

You may be running in a fresh session with no prior memory. **All state must be recovered from bridge files + wiki + competition_meta.yaml.** Never assume context you have not just read.

## Arguments
- `$ARGUMENTS` — number of cycles (default `5`) or `until_stall`.

## Mission (every cycle)

```
HYPOTHESIZE  →  IMPLEMENT  →  VERIFY  →  EXPORT
   (plan)        (dev)        (run+eval)  (pack + insights)
                                          ⇣
                                       optional /rank suggestion
                                          ⇣
                                  HUMAN submits manually
                                          ⇣
                                  /submit-result refits CV→LB
                                          ⇣
                                  better plan next cycle
```

The system replaces wasted DACON submissions with a CV→LB predictor that improves every time a real LB score comes back.

## Hard Guardrails (NEVER violate)

Execution control:
1. NEVER auto-submit to DACON.
2. STOP after 5 consecutive cycles without CV Macro-F1 improvement.
3. STOP if any experiment produces NaN/Inf predictions.
4. STOP if any single experiment's local `train.py` exceeds 60 minutes wall-clock.
5. Cap at 20 experiments per `/auto` invocation.
6. Log every cycle to `logs/cycle_history.jsonl`, even failures.

Time pressure:
7. On D-day or D-1: NO new architectures. Only seed-ensembles of already-validated candidates.
8. If today's submission quota is exhausted: keep producing CANDIDATEs but mark them HOLD until reset.

DACON server constraints (auto-disqualify if violated):
9. zip ≤ 1 GB
10. inference ≤ 10 min (verified via local dry-run)
11. install ≤ 10 min (keep `requirements.txt` lean)
12. model fits T4 16GB VRAM (fp16 or quantized if needed)
13. full test inference fits in 12GB RAM (batch conservatively)

## STEP 0 — Context Recovery (ALWAYS first)

```bash
python scripts/check_time_state.py --json     > /tmp/time.json
python scripts/cv_lb_correlation.py --json    > /tmp/cvlb.json
```

```python
import json, pathlib

time_state = json.load(open('/tmp/time.json'))
cvlb       = json.load(open('/tmp/cvlb.json'))
state      = json.load(open('logs/orchestrator_state.json'))

# Recent context window only — never load full history
N = state.get('recent_context_window', 5)
cycle_history = [json.loads(l) for l in open('logs/cycle_history.jsonl') if l.strip()][-N:]
insights      = [json.loads(l) for l in open('logs/insights.jsonl')      if l.strip()][-N:]

digest = pathlib.Path('logs/experiment_digest.md').read_text() if pathlib.Path('logs/experiment_digest.md').exists() else ""
```

Read `data_docs/` (all `.md` files). Read `wiki/lessons/` and `wiki/decisions/` for any pages related to today's planned phase.

Print a situational summary:
```
[TIME] now=<iso KST>  | D-<N> to preliminary  | quota=<used>/20  remaining=<R>
[STATE] phase=<X>  best_cv=<0.XXXX>(<exp>)  best_lb=<0.XXXX>(<exp>)  stall=<S>/5  cycles=<C>
[CVLB] n_pairs=<k>  pearson_r=<r>  σ=<sigma>  trust=<level>
[INSIGHTS] <last insight one-liner or "none yet">
[DATA] <data_docs one-liner or "no data_docs">
[NEXT] <state.next_action>  →  <state.last_reasoning>
```

## STEP 1 — HYPOTHESIZE (Phase: PLAN)

### 1a — Model Family Roster (MANDATORY auto-exploration)

`/auto` is responsible for SWEEPING the model family space without the user having to name models. Maintain `state['model_family_roster']` — the complete tournament bracket. If missing, seed it:

```python
DEFAULT_ROSTER = [
  # tier: baseline (cheap, used to validate pipeline + leakage)
  {"family":"tfidf_lightgbm",      "tier":"baseline",   "cost_min":5,  "priority":1.0},
  {"family":"tfidf_logreg",        "tier":"baseline",   "cost_min":3,  "priority":0.9},
  {"family":"tfidf_catboost",      "tier":"baseline",   "cost_min":8,  "priority":0.8},
  # tier: embedding (text semantics with cheap head)
  {"family":"st_minilm_lgb",       "tier":"embedding",  "cost_min":15, "priority":0.85},
  {"family":"st_mpnet_lgb",        "tier":"embedding",  "cost_min":20, "priority":0.8},
  # tier: encoder fine-tune (likely SOTA for this task)
  {"family":"codebert_base",       "tier":"transformer","cost_min":40, "priority":1.0},
  {"family":"graphcodebert_base",  "tier":"transformer","cost_min":45, "priority":0.95},
  {"family":"deberta_v3_small",    "tier":"transformer","cost_min":35, "priority":0.95},
  {"family":"deberta_v3_base",     "tier":"transformer","cost_min":50, "priority":0.9},
  {"family":"electra_small",       "tier":"transformer","cost_min":25, "priority":0.7},
  {"family":"xlm_roberta_base",    "tier":"transformer","cost_min":45, "priority":0.7,
   "condition":"korean_text_detected"},
  # tier: decoder/LLM (use only if encoder tier saturates)
  {"family":"qwen25_05b_cls_head", "tier":"llm",        "cost_min":60, "priority":0.6},
  {"family":"phi3_mini_lora",      "tier":"llm",        "cost_min":75, "priority":0.5},
  # tier: ensemble (last)
  {"family":"top3_blend",          "tier":"ensemble",   "cost_min":10, "priority":1.0,
   "condition":"≥3 candidates exist"},
]
```

Persist per-family stats in `state['family_stats']`:
```json
{
  "tfidf_lightgbm": {"tried":2, "best_cv":0.612, "best_exp":"exp_001", "avg_lb_gap":0.008, "status":"explored"},
  "codebert_base":  {"tried":1, "best_cv":0.683, "best_exp":"exp_007", "avg_lb_gap":null,  "status":"leader"},
  ...
}
```

### 1b — Family Selection Strategy (the actual auto-sweep)

Pick the next experiment using this decision tree, in order:

```
1. SMOKE TEST PHASE  (state['family_stats'] empty)
   → pick tfidf_lightgbm                   # validate pipeline first

2. BASELINE SWEEP    (any tier=baseline with tried==0)
   → pick highest-priority unexplored baseline

3. TIER PROGRESSION  (current tier saturated, i.e. all tried ≥1 and stall ≥ 2)
   → advance to next tier (baseline → embedding → transformer → llm → ensemble)

4. WITHIN-TIER SWEEP (current tier has untried families AND
                      best_cv_current_tier improved vs prev tier by ≥ 0.01)
   → pick highest-priority untried family in current tier

5. EXPLOIT LEADER    (leader family has improvement slope > 0 over last 2 attempts)
   → iterate the leader family (new feature_set, new hyperparams)

6. ENSEMBLE          (≥3 CANDIDATEs exist AND days_to_deadline ≤ 5)
   → trigger top3_blend

7. FALLBACK          (stall_counter ≥ 4)
   → switch to diagnostic phase (CV audit, leakage probes) NOT new family
```

Cost-budget filter — skip families whose `cost_min > available_budget_min`:
```python
available = (time_state['days_to_preliminary'] - 1) * 8 * 60   # 8h/day, reserve 1 day for ensemble
```

### 1c — Materialize the experiment

For the chosen `family`, generate the full experiment spec:
- experiment_id: `exp_<NNN>_<family>` (auto-increment NNN)
- hypothesis: `"<family> outperforms current best (<best_family>, CV=<best_cv>) by ≥ <expected_delta>"`
- expected_delta: from `family_stats` (if explored) or tier prior (baseline=0.00, embedding=+0.03, transformer=+0.05, llm=+0.02)
- verification: 5-fold CV Macro-F1, per-class F1, OOF correlation vs candidates
- approach: `model_family=<family>`, default feature_set per family, config from the templates in `/dev`

Skip if `(family, feature_set)` already in `state.blocked_approaches`.

Write the plan to `logs/agent_messages.jsonl`:
```json
{"timestamp":"<iso>","from":"orchestrator","type":"plan",
 "content":"cycle <C+1>: trying <family> (tier=<X>) — reason: <selection_rule>",
 "roster_progress": "<tried_count>/<total_families>"}
```

## STEP 2 — IMPLEMENT (Phase: DEV)

Spawn the dev role via Task tool (Agent):
```
Agent(
  description="Implement exp_NNN",
  subagent_type="general-purpose",
  prompt="""You are the model_developer agent. Follow .claude/skills/dev/SKILL.md.

  Experiment: exp_NNN_name
  Hypothesis: <from plan>
  Verification protocol: <from plan>
  Approach:
    model_family: <...>
    feature_set: <...>
    cv: stratified_group_5fold          # StratifiedGroupKFold, group = session id (id.rsplit("-step",1)[0])
  Data: JSONL (data/train.jsonl + data/train_labels.csv join on id; data/test.jsonl).
        data_docs/dataset_overview.md describes columns and the 14-class STRING target `action`.

  REQUIREMENTS:
  - Produce experiments/exp_NNN_name/{config.yaml,train.py,script.py,features.py,requirements.txt,SUMMARY.md}
  - Load data via JSONL loaders (features.py load_jsonl/build_records); labels are 14 snake_case STRINGS, not ints
  - CV = StratifiedGroupKFold(5, shuffle=True, random_state=42) grouped by session, with a zero group-overlap assert
  - Thread cap (Rule B): set n_jobs/num_threads/thread_count ≤ 16 (128-core box → oversubscription thrash otherwise)
  - train.py writes train_log.json with cv_mean/cv_std/per_class_f1/collapsed_classes/inference_ms_per_sample/model_size_mb/offline_compatible
  - script.py is OFFLINE-only, reads data/test.jsonl, writes output/submission.csv with STRING `action` labels
  - Run scripts/validate_submission.py before returning
  - Do NOT execute train.py — runner will

  When done, append to logs/agent_messages.jsonl with type=implemented.
  """
)
```

## STEP 3 — VERIFY (Phase: RUN + EVAL)

### 3a — RUN

**Rule A (foreground/wait):** the orchestrator runs `train.py` in the FOREGROUND
and BLOCKS until it exits. NEVER background the training and move on — a
background-and-exit leaves orphaned trainings and skips downstream steps
(eval, pack). The cycle is only "run" once `train_log.json` + `oof_preds.npy` +
`test_preds.npy` actually exist on disk. **Rule B (thread cap):** cap CPU threads
≤ 16 (128-core box → oversubscription thrash) and run at most 2 heavy trainings
in parallel, never a full-core fan-out.

```bash
# FOREGROUND — wait for completion (do NOT append & / do NOT run_in_background)
cd experiments/exp_NNN_name && timeout 3600 python train.py 2>&1 | tee run_output.txt
```

Validate outputs:
```python
import json, numpy as np
log = json.load(open('train_log.json'))
oof  = np.load('oof_preds.npy')
test = np.load('test_preds.npy')
assert oof.shape[1] == 14 and test.shape[1] == 14
assert not np.isnan(oof).any() and not np.isnan(test).any()
```

Dry-run `script.py` on a 1000-row test sample to estimate full inference time. If extrapolated > 8 min → mark REVIEW.

### 3b — EVAL

Compute against current best:
```python
improvement_vs_best = log['cv_mean'] - state['best_cv']
```

Leakage probes (see `.claude/skills/eval/SKILL.md` STEP 4 for details).
Per-class collapse check: `collapsed_classes` non-empty → REVIEW.
Distribution shift: L1 between OOF and test class freq > 0.30 → REVIEW.
Offline check + size + inference budget — REJECT if hard limits exceeded.

LB prediction:
```bash
python scripts/cv_lb_correlation.py --predict <cv_mean> --json
```

Decision:
```
recommendation =
  REJECT if leakage OR offline_fail OR est_inference > 10min
  REVIEW if collapsed_classes OR cv_std grade D OR L1 shift > 0.30 OR est_inference 8-10min
  CANDIDATE if improvement_vs_best > 0 AND all checks pass
  CANDIDATE_DIVERSITY if no CV gain but OOF correlation with all candidates < 0.95
  COMPLETED otherwise
```

Write `experiments/exp_NNN/evaluation.json`.

## STEP 4 — EXPORT (Phase: PACK if CANDIDATE)

If `recommendation` is CANDIDATE or CANDIDATE_DIVERSITY:
```bash
# Inline the pack logic or call the skill; ends with submissions/exp_NNN.zip + meta.json
```

If REJECT / REVIEW: do NOT pack. Append the failure mode to `state.blocked_approaches` for future cycles.

## STEP 5 — UPDATE STATE

```python
from datetime import datetime
state['total_cycles'] += 1
state['last_updated'] = datetime.now().astimezone().isoformat()

# --- Model family bookkeeping (the tournament bracket) ---
fam = chosen_family
fs = state.setdefault('family_stats', {}).setdefault(fam, {
    "tried": 0, "best_cv": None, "best_exp": None, "avg_lb_gap": None, "status": "exploring"
})
fs["tried"] += 1
if recommendation != 'REJECT' and (fs["best_cv"] is None or log['cv_mean'] > fs["best_cv"]):
    fs["best_cv"] = log['cv_mean']
    fs["best_exp"] = exp_name
# Mark status
if fs["tried"] >= 2 and (fs["best_cv"] or 0) < (state.get('best_cv') or 0) - 0.02:
    fs["status"] = "dropped"          # 2 attempts and still 0.02 below leader → stop trying
elif fs["best_cv"] == max((s.get("best_cv") or 0) for s in state['family_stats'].values()):
    fs["status"] = "leader"
else:
    fs["status"] = "explored"

# Crown overall best
if recommendation in ('CANDIDATE', 'CANDIDATE_DIVERSITY') and improvement_vs_best > 0:
    state['best_cv'] = log['cv_mean']
    state['best_experiment'] = exp_name
    state['best_family'] = fam
    state['stall_counter'] = 0
else:
    state['stall_counter'] += 1

# Tier auto-advance (replaces old phase advance) — only after current tier saturated
tiers = ['baseline','embedding','transformer','llm','ensemble']
current_tier = state.get('current_tier', 'baseline')
tier_families = [r['family'] for r in DEFAULT_ROSTER if r['tier'] == current_tier]
tier_done = all(state['family_stats'].get(f, {}).get('tried', 0) >= 1 for f in tier_families)
if tier_done and state['stall_counter'] >= 2 and current_tier != 'ensemble':
    nxt = tiers[tiers.index(current_tier) + 1]
    state['current_tier'] = nxt
    state['current_phase'] = {'baseline':'baseline','embedding':'feature_eng',
                              'transformer':'model_exploration','llm':'model_exploration',
                              'ensemble':'ensemble'}[nxt]
    state['strategy_history'].append({
        'cycle': state['total_cycles'],
        'change': f"tier advanced to {nxt}",
        'reason': f"{current_tier} tier swept, stall={state['stall_counter']}"
    })
    state['stall_counter'] = 0

if recommendation == 'REJECT':
    state.setdefault('blocked_approaches', []).append({
        'experiment': exp_name,
        'reason': reason,
        'cycle': state['total_cycles']
    })

json.dump(state, open('logs/orchestrator_state.json', 'w'), indent=2, ensure_ascii=False)
```

Append to `logs/cycle_history.jsonl`:
```json
{
  "cycle": <int>,
  "timestamp": "<iso>",
  "experiment": "exp_NNN",
  "phase": "<phase>",
  "cv_mean": 0.XXXX,
  "cv_std": 0.XXXX,
  "worst_class_f1": 0.XX,
  "predicted_lb": 0.XXXX,
  "lb_prediction_interval": [0.XXXX, 0.XXXX],
  "trust_level": "<level>",
  "recommendation": "<...>",
  "best_cv_after": 0.XXXX,
  "stall_counter_after": <int>,
  "reasoning": "<why this experiment, what it taught us>",
  "next_plan": "<what next cycle should try>"
}
```

`state['last_reasoning'] = next_plan`. `state['next_action'] = 'plan'` (or `'retry'` if fixable error).

## STEP 6 — STOP CONDITIONS

```python
if state['stall_counter'] >= 5: STOP("5 cycles without improvement")
if state['total_cycles'] >= max_cycles: STOP("max cycles reached")
if recommendation == 'REJECT' and 'NaN' in reason: STOP("critical: NaN")
if time_state['days_to_preliminary'] <= 0: STOP("past deadline")
```

## STEP 7 — COMPOUND (Light Mode)

Every cycle:
- Append insight to `logs/insights.jsonl` only if something noteworthy happened (improvement, REJECT, surprise).
- Every 5 cycles: write a `wiki/context/snapshot-<YYYY-MM-DD>.md` snapshot.
- Every confirmed lesson: write `wiki/lessons/<id>.md`.
- Every consequential decision (model family / feature set choice): write `wiki/decisions/<id>.md`.

(At end of session, the user runs `/compound` for the full mode.)

## STEP 8 — CYCLE REPORT

```
═════════════════════════════════════════════
CYCLE <C>/<MAX> — exp_NNN_<family>
═════════════════════════════════════════════
Family tested : <family>  (tier=<X>)   roster <tried>/<total>
CV Macro-F1   : 0.XXXX ± 0.XXXX     (best ever: 0.XXXX, family=<best_family>)
Worst class   : id=<N> f1=<0.XX>    collapsed=[<ids|none>]
Inference     : <X.X> ms/sample → est full <X.X> min   (cap 10)
Model size    : <X.X> MB                                (cap 1024)
Offline       : PASS / FAIL
Recommend     : <CANDIDATE | REVIEW | REJECT | COMPLETED>
Predicted LB  : 0.XXXX  PI=[0.XXXX, 0.XXXX]  trust=<level>
Tier / Phase  : <tier>/<phase>     Stall: <S>/5

Family bracket (top 5):
  <family>            tier=<X>   best_cv=0.XXXX  status=<leader|explored|dropped>
  ...
Next plan     : try <next_family> — <selection_rule>
═════════════════════════════════════════════
```

Loop to STEP 1 unless a STOP condition is met.

## END SUMMARY (when STOP triggers)

```
═════════════════════════════════════════════
AUTONOMOUS RUN COMPLETE — <iso>
═════════════════════════════════════════════
Cycles run     : <N>     successful=<S>  rejected=<R>
Best CV        : 0.XXXX  (exp_NNN, family=<X>)
Best LB        : 0.XXXX  (exp_NNN, if any)
Tier reached   : <tier> / <phase>
Stop reason    : <reason>

CV→LB model    : n_pairs=<k>  r=<r>  σ=<sigma>  trust=<level>

Final family bracket (sorted by best CV):
  1. <family>   tier=<X>  best_cv=0.XXXX  tried=<N>  status=leader
  2. <family>   tier=<X>  best_cv=0.XXXX  tried=<N>  status=explored
  3. <family>   tier=<X>  best_cv=0.XXXX  tried=<N>  status=dropped
  ...
Untried families (out of budget or condition unmet):
  - <family>   reason=<cost_exceeds_budget | condition_unmet | tier_not_reached>

Today's recommended submissions (run /rank for full ranking):
1. exp_NNN_x   CV 0.XXXX → predLB 0.XXXX  [SUBMIT_FIRST]
2. exp_NNN_y   CV 0.XXXX → predLB 0.XXXX  [SUBMIT_IF_SLOTS]

⚠️  Manual submission required.
After DACON returns LB, run:  /submit-result <exp> <lb>
That will refit the CV→LB model so the next /auto can predict LB more sharply.

Compounded: <pages_added> wiki pages, <insights_added> insights this session.
Run /compound (full mode) before closing the session.
═════════════════════════════════════════════
```

## Hard Rules (repeat — these are the most important)

- NEVER auto-submit.
- NEVER skip context recovery (STEP 0).
- NEVER mark CANDIDATE without a successful local dry-run of `script.py`.
- NEVER plan an experiment present in `state.blocked_approaches`.
- NEVER let the user pick the model family — `/auto` MUST sweep `model_family_roster` automatically per STEP 1b.
- NEVER skip a tier; baselines run first to validate the pipeline before transformer fine-tunes.
- NEVER re-try a family marked `status: dropped` unless the user explicitly resets it.
- ALWAYS update `state['family_stats']` after every cycle so the bracket converges.
- ALWAYS update bridge files (`orchestrator_state.json`, `cycle_history.jsonl`, `experiment_digest.md`).
- ALWAYS log a predicted LB with prediction interval — even when trust is low.
