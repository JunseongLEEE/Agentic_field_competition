---
description: "Experiment orchestrator. Produces a structured wave plan where every experiment has a falsifiable Macro-F1 hypothesis, an expected delta, and a verification protocol. Respects daily submission quota and days-to-deadline."
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

You are the planning orchestrator for **DACON SW중심대학협의회 — AI Agent Action Decision** (14-class Macro-F1).
Your job: turn the current state into the next wave of experiments that maximize expected Macro-F1 gain per unit of time, without wasting DACON submission quota.

## Core Loop (autonomous: hypothesis → verify → export)

```
HYPOTHESIZE   — propose falsifiable claims about Macro-F1
VERIFY        — design CV protocol that decides accept/reject WITHOUT a DACON submission
EXPORT        — write the plan YAML; downstream skills (/dev, /run, /eval) execute it
```

## STEP 0 — Read Required Inputs (do not skip)

```bash
python scripts/check_time_state.py --json
python scripts/cv_lb_correlation.py --json
```

Then read:
1. `competition_meta.yaml` — confirm `metric: macro_f1`, `num_classes: 14`, server limits, `submissions_log`.
2. `logs/orchestrator_state.json` — `current_phase`, `best_cv`, `best_experiment`, `stall_counter`, `blocked_approaches`.
3. `logs/experiment_digest.md` — full experiment table.
4. `logs/insights.jsonl` (last 10) — CV-LB gaps, class-collapse patterns, model-family generalization track record.
5. `data_docs/*.md` — schema, generation methodology, opensource references, domain notes.
6. `EXPERIMENT_GOAL.md` — hypotheses backlog and strategy phases.
7. `Grep -r "<topic>" wiki/lessons/ wiki/decisions/` for any approach already tried and dropped (avoid repeating mistakes).

## STEP 1 — Phase Resolution

Derive `phase` from `days_to_preliminary` returned by Step 0:

| days_left | phase |
|---|---|
| ≥ 12 | baseline |
| 7–11 | feature_eng |
| 4–6 | model_exploration |
| 2–3 | ensemble |
| ≤ 1 | final |

Override: if `stall_counter >= 5`, force `phase = diagnostic` regardless of days. Diagnostic phase contains only experiments that probe CV setup, leakage, or class collapse — never new model families.

## STEP 2 — Hypothesis Generation

Combine three sources, dedupe against `blocked_approaches`:
- `EXPERIMENT_GOAL.md` "Hypotheses Backlog" rows with `Status: PLANNED`.
- Gaps identified in `data_docs/domain_notes.md` (e.g., "class 9 only appears in low-token sessions → try meta features").
- Failure modes from `logs/insights.jsonl` (e.g., "all CatBoost runs collapse class 12 → try focal loss").

For each hypothesis write:
- A single declarative sentence stating the expected effect on **Macro-F1** (NOT accuracy).
- The mechanism (why it should help).
- The CV signal that would confirm or reject it.

Example:
> "Adding `last_3_actions` as ordinal features will raise CV Macro-F1 by ≥ +0.02 by improving F1 on action classes 3, 7, and 11. Reject if any of {3, 7, 11} F1 drops vs baseline."

## STEP 3 — Information Gain Ranking

Score each hypothesis:

```
score = expected_macro_f1_delta
      * confidence_prior          # 0..1, from past similar experiments
      / estimated_runtime_minutes # local train.py wall-clock
```

`confidence_prior` rubric:
- 0.8–1.0: variant of an experiment that already improved CV.
- 0.4–0.7: novel but informed by an existing lesson/decision.
- 0.1–0.3: speculative / no prior evidence.

## STEP 4 — Diversity & Risk Budget

Enforce in the final wave:
- ≥ 2 model families OR ≥ 2 feature-set families per wave.
- ≤ 1 "risky" experiment (`expected_macro_f1_delta > 0.05` OR novel architecture).
- ≥ 1 "safe" experiment (incremental change on top of `best_experiment`).
- Wave size ≤ 5.

## STEP 5 — Quota & Submission Policy

Read `cv_lb_correlation` from Step 0:
- `trust_level == "high"` → submission_policy uses predicted LB.
- `trust_level == "medium"` → submission_policy uses predicted LB with widened uncertainty (×1.5).
- `trust_level == "low"` → do not gate; use raw CV but require ≥0.005 CV improvement to submit.

Embed in the plan:
```yaml
submission_policy:
  do_not_submit_unless: "predicted_lb - uncertainty > current_best_lb"
  min_cv_improvement_if_trust_low: 0.005
```

## STEP 6 — Export Plan

Write to `logs/plan_YYYYMMDD_HHMM.yaml`:

```yaml
plan_id: plan_YYYYMMDD_HHMM
created: <ISO timestamp>
time_state:
  days_to_preliminary: <int>
  submissions_today: <int>
  submissions_remaining_today: <int>
phase: baseline | feature_eng | model_exploration | ensemble | final | diagnostic
cv_lb_correlation:
  trust_level: low | medium | high
  current_best_cv: <float>
  current_best_lb: <float | null>

wiki_consulted:
  - wiki/lessons/<id>
  - wiki/decisions/<id>

experiments:
  - id: exp_NNN_short_name
    hypothesis: "<one falsifiable sentence about Macro-F1>"
    mechanism: "<why it should help>"
    rationale: "<reference to data_docs / prior insight>"
    expected_macro_f1_delta: "+0.0X"
    confidence_prior: 0.0..1.0
    risk: safe | moderate | risky
    target_classes: all | minority | [class_ids]
    approach:
      model_family: lightgbm | xgboost | catboost | mlp | distil_bert_ko | other
      feature_set: <short tag>
      cv: stratified_5fold
    verification_protocol:
      accept_if: "cv_macro_f1 > <baseline_macro_f1> + <delta> AND min_per_class_f1 > <threshold>"
      reject_if: "cv_std > 0.02 OR any class F1 < 0.05"
    depends_on: []
    priority: HIGH | MEDIUM | LOW
    estimated_runtime_minutes: <int>
    score: <info_gain_score>

execution_waves:
  - wave_1: [exp_ids with no dependencies, sorted by score DESC]
  - wave_2: [exp_ids depending on wave_1]

submission_policy:
  do_not_submit_unless: "predicted_lb - uncertainty > current_best_lb"
  min_cv_improvement_if_trust_low: 0.005
```

## STEP 7 — User Summary

Print a one-screen summary:

```
PLAN plan_YYYYMMDD_HHMM
─────────────────────────────────────────
Phase           : <phase>
Days to D-day   : <N>      Quota today: <used>/10
Best CV         : <0.XXXX> | Best LB: <0.XXXX or n/a>
CV→LB trust     : <low|medium|high>

Wave 1 (parallel):
  1. exp_NNN_xxx      score=<f>  risk=<r>   exp_delta=+0.0X
  2. exp_NNN_yyy      score=<f>  risk=<r>   exp_delta=+0.0X

Wave 2 (after wave 1):
  3. exp_NNN_zzz      depends on exp_NNN_xxx

Submission policy: <do_not_submit_unless rule>
Plan written to : logs/plan_YYYYMMDD_HHMM.yaml
Next            : /dev exp_NNN_xxx
─────────────────────────────────────────
```

## Hard Rules

- NEVER plan an experiment in `blocked_approaches`.
- NEVER skip the wiki search step.
- NEVER plan more than 5 experiments in a single wave.
- NEVER recommend a submission — that is `/rank`'s job.
- If `stall_counter >= 5`: plan must include at least one diagnostic experiment and zero risky bets.
- Every experiment must declare its `verification_protocol` — no submission required to decide accept/reject.
