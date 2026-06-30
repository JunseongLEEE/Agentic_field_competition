---
description: "Rank packaged CANDIDATE experiments by predicted LB (not just CV). Recommends what to submit today within the remaining daily quota. Never auto-submits."
user-invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
---

# /rank — Submission Candidate Selector

Pick today's submissions by **predicted LB net of uncertainty**, not by raw CV.
Respect the 10/day DACON quota. Never auto-submit.

## STEP 0 — Inputs

```bash
python scripts/check_time_state.py --json > /tmp/time.json
python scripts/cv_lb_correlation.py --json > /tmp/cvlb.json
ls submissions/*.zip 2>/dev/null
ls submissions/*.meta.json 2>/dev/null
```

Also read:
- `competition_meta.yaml` → `submissions_log` (what was already submitted, with which experiment, and LB returned).
- `logs/insights.jsonl` → per-model-family CV-LB track record.
- `experiments/exp_*/evaluation.json` for the OOF correlation matrix between candidates (for diversity).

## STEP 1 — Gather Today's Pool

A candidate enters the pool only if ALL of these hold:
- `evaluation.json.recommendation` ∈ {CANDIDATE, CANDIDATE_DIVERSITY}
- A packaged zip exists in `submissions/`
- It was not already submitted today (check `submissions_log` filtered to KST today)
- `submission_readiness.offline_check == "PASS"`

## STEP 2 — Composite Score

For each candidate:

```
predicted_lb        = (slope * cv) + intercept           # from cv_lb_correlation.py
uncertainty         = max(pi_high - predicted_lb, 0.003) # floor for low-trust regime
expected_gain       = predicted_lb - current_best_lb     # 0 if no LB yet
diversity_bonus     = 1 - max(corr(oof_i, oof_j))        # vs already-submitted today
generalization_prior = avg(1 - |gap|)                    # per model_family, from insights

composite =
    0.50 * normalize(predicted_lb)
  + 0.20 * (1 - normalize(uncertainty))
  + 0.15 * normalize(diversity_bonus)
  + 0.15 * normalize(generalization_prior)
```

Penalty: if `evaluation.json.per_class_summary.collapsed_classes` is non-empty, multiply composite by 0.7 (Macro-F1 will likely punish on LB).

## STEP 3 — Submission Gate

A candidate is **worth submitting** only if:

```
predicted_lb - uncertainty > current_best_lb   (when trust_level != "low")
OR
diversity_bonus > 0.30                          (it adds genuinely new signal)
OR
current_best_lb is None                         (first submission ever)
```

If none qualify and `submissions_remaining_today > 0`, recommend **HOLD** with reason "no candidate beats current best LB net of uncertainty; saving slot".

## STEP 4 — Diversity Constraints

- ≤ 3 candidates from same model_family per day.
- ≤ 2 candidates with identical feature_set per day.
- Skip a candidate whose OOF correlation with any already-recommended one is > 0.95.

## STEP 5 — Quota Cap

```
remaining = max(0, 10 - submissions_today_successful_or_runtime_error)
# install_error does NOT count against quota
top_k     = min(remaining, recommended_count)
```

Time-pressure adjustment:
- D-day or D-1: cap at 1 (the single safest pick, lowest uncertainty among those that pass the gate).
- D-2 to D-3: cap at 3.
- D-4 to D-6: cap at 5.
- D-7+: up to `remaining`.

## STEP 6 — Write SUBMISSION_CANDIDATES.md

```markdown
# Submission Candidates — <YYYY-MM-DD KST>

Quota: <used>/10  | Remaining today: <N>  | Days to D-day: <D>
CV→LB trust: <level>  (n_pairs=<k>, residual_std=<sigma>)
Current best LB: <0.XXXX>  (exp_NNN)

## Rank
| # | Experiment | CV F1 | Pred LB | PI low | PI high | Δ vs best LB | Model | Features | Composite | Priority | Reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | exp_NNN_x | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXXX | +0.XXXX | <model> | <feats> | 0.XX | SUBMIT_FIRST | <one sentence> |
| 2 | exp_NNN_y | ... |
| 3 | exp_NNN_z | ... |
```

Priority labels:
- `SUBMIT_FIRST` — top of today's recommendation, passes the gate cleanly.
- `SUBMIT_IF_SLOTS` — passes the gate, lower expected gain or higher uncertainty.
- `HOLD` — does not pass the gate today; keep packaged for a future cycle.

## STEP 7 — Console Report

```
═════════════════════════════════════════════
TODAY'S RECOMMENDATIONS — <YYYY-MM-DD>
═════════════════════════════════════════════
Quota          : <used>/10  remaining=<N>
CV→LB trust    : <level>  (n=<k>, σ=<sigma>)
Current best LB: <0.XXXX> (exp_NNN)
D-day in       : <N> days

1. exp_NNN_x   CV 0.XXXX | predLB 0.XXXX [pi 0.XXXX, 0.XXXX]  → SUBMIT_FIRST
   reason: <Δ vs best LB net of uncertainty, model family generalization>
2. exp_NNN_y   ... → SUBMIT_IF_SLOTS
3. exp_NNN_z   ... → HOLD (uncertainty wider than expected gain)

⚠️  Manual submission required — upload `submissions/exp_NNN_x.zip` to DACON.
   After DACON returns the LB, run /submit-result <exp> <lb>.
═════════════════════════════════════════════
```

## Hard Rules

- NEVER auto-submit.
- NEVER recommend more than `remaining_quota` candidates.
- NEVER recommend more than 1 candidate on D-day or D-1.
- ALWAYS show the LB prediction interval, not just point estimates.
- ALWAYS flag if all top picks share the same model_family.
