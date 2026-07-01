---
description: "Record an LB score after manual DACON submission. Refits the CV→LB correlation model, extracts patterns, updates insights and bridge files. Usage: /submit-result <exp_id> <lb_score> [--status success|runtime_error|install_error]"
user-invocable: true
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Glob
  - Grep
---

# /submit-result — Record LB Score + Refit CV→LB Model

After a manual upload to DACON, capture the LB score, refit the CV→LB correlation,
and extract actionable insight for the next planning cycle.

## Arguments
- `$ARGUMENTS` — `<experiment_id> <lb_score> [--status success|runtime_error|install_error]`
  - default `--status success`
  - `install_error` does NOT count against the daily 20-submission team quota.
  - `runtime_error` DOES count against the daily quota (and `lb_score` should be `0` or `nan`).

## STEP 0 — Parse and Validate

```bash
EXP=<first arg>
LB=<second arg>
STATUS=<--status value, default success>

# Sanity
test -d "experiments/${EXP}" || { echo "experiment dir not found"; exit 2; }
test -f "experiments/${EXP}/train_log.json" || { echo "no train_log.json"; exit 2; }
```

## STEP 1 — Persist to competition_meta.yaml

```bash
python scripts/track_submission.py \
  --exp "${EXP}" \
  --cv  "$(python -c "import json;print(json.load(open('experiments/${EXP}/train_log.json'))['cv_mean'])")" \
  --lb  "${LB}" \
  --status "${STATUS}"
```

`track_submission.py` writes a structured entry into `competition_meta.yaml.submissions_log`:
```yaml
- experiment_id: exp_NNN
  submitted_at: <iso, KST>
  cv_score: 0.XXXX
  lb_score: 0.XXXX
  status: success | runtime_error | install_error
  counted_against_quota: true | false   # false only for install_error
  package_sha256: <from submissions/exp_NNN.meta.json>
```

## STEP 2 — Refit CV→LB Correlation

```bash
python scripts/cv_lb_correlation.py --json > /tmp/cvlb.json
cat /tmp/cvlb.json
```

This produces an updated `pearson_r`, `residual_std`, `slope`, `intercept`, `trust_level`.

## STEP 3 — Extract Insight

```python
import json
log = json.load(open(f"experiments/{EXP}/train_log.json"))
ev  = json.load(open(f"experiments/{EXP}/evaluation.json"))
cvlb = json.load(open("/tmp/cvlb.json"))

cv  = log["cv_mean"]
lb  = float(LB)
gap = cv - lb
predicted = ev["lb_prediction"]["predicted_lb"]
pi_low, pi_high = ev["lb_prediction"]["pi_low"], ev["lb_prediction"]["pi_high"]
inside_interval = pi_low <= lb <= pi_high
```

Insight rules (apply all that match):

| condition | insight |
|---|---|
| `abs(gap) < 0.005` | "CV is reliable for this pipeline; trust it." |
| `gap > 0.02 and cv > lb` | "Overfit to local CV; reduce model capacity or add regularization." |
| `gap < -0.02 and lb > cv` | "CV is pessimistic; can be more aggressive next cycle." |
| `not inside_interval` | "LB fell outside 95% PI — correlation model under-fit; collect more pairs before gating." |
| `ev.per_class_summary.collapsed_classes` ≠ [] and gap > 0.01 | "Class collapse predicted by per-class F1; prioritize class_weight / focal loss next." |
| same model_family has avg gap < 0.005 | "Model family <X> generalizes well; weight it higher in /rank." |
| same model_family has avg gap > 0.02 | "Model family <X> overfits; lower its priority." |

## STEP 4 — Append to logs/insights.jsonl

```python
import json
from datetime import datetime
record = {
  "timestamp": datetime.now().astimezone().isoformat(),
  "experiment": EXP,
  "cv_macro_f1": cv,
  "cv_std": log["cv_std"],
  "lb_macro_f1": lb,
  "gap": gap,
  "predicted_lb": predicted,
  "pi_low": pi_low,
  "pi_high": pi_high,
  "inside_pi": inside_interval,
  "model_family": yaml.safe_load(open(f"experiments/{EXP}/config.yaml"))["model"]["type"],
  "feature_set": "<short tag>",
  "trust_level_after": cvlb["trust_level"],
  "pearson_r_after": cvlb["pearson_r"],
  "residual_std_after": cvlb["residual_std"],
  "insight": "<one or more rules from STEP 3>",
  "actionable": "<what /plan should do next cycle>"
}
open("logs/insights.jsonl", "a").write(json.dumps(record, ensure_ascii=False) + "\n")
```

## STEP 5 — Update SUMMARY.md and EXPERIMENT_LOG.csv

Edit `experiments/${EXP}/SUMMARY.md` Results:
```markdown
| LB Score    | 0.XXXX |
| CV-LB Gap   | 0.XXXX |
| Inside PI?  | yes / no |
| Status      | SUBMITTED |
```

Add Insight subsection:
```markdown
## Post-submission Insight
- gap=<value> (<direction>); <one-line insight from STEP 3>
- next-cycle action: <actionable>
```

Update `EXPERIMENT_LOG.csv`: set `lb_score`, `cv_lb_gap` for this experiment row.

## STEP 6 — Update orchestrator_state.json

```python
import json
state = json.load(open("logs/orchestrator_state.json"))
if STATUS == "success" and lb > (state.get("best_lb") or 0.0):
    state["best_lb"] = lb
    state["best_lb_experiment"] = EXP
state["last_updated"] = datetime.now().astimezone().isoformat()
state["last_reasoning"] = record["actionable"]
state["stall_counter"] = 0 if lb > (state.get("best_lb_prev") or 0.0) else state.get("stall_counter", 0) + 1
json.dump(state, open("logs/orchestrator_state.json", "w"), indent=2, ensure_ascii=False)
```

## STEP 7 — Rebuild Digest

```bash
python scripts/build_digest.py
```

## STEP 8 — Pattern Detection (when n_pairs ≥ 3)

After 3+ successful submissions, print a model-family generalization ranking:

```python
import json, statistics
from collections import defaultdict
gaps_by_family = defaultdict(list)
with open("logs/insights.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r.get("lb_macro_f1") is not None:
            gaps_by_family[r["model_family"]].append(abs(r["gap"]))
ranking = sorted(
    [(fam, statistics.mean(g), len(g)) for fam, g in gaps_by_family.items()],
    key=lambda x: x[1]
)
print("MODEL FAMILY GENERALIZATION RANKING (lower |gap| is better):")
for fam, avg, n in ranking:
    print(f"  {fam:<20} avg|gap|={avg:.4f}  n={n}")
```

## STEP 9 — Report

```
═════════════════════════════════════════════
SUBMISSION RECORDED: exp_NNN_name
═════════════════════════════════════════════
CV Macro-F1   : 0.XXXX ± 0.XXXX
LB Macro-F1   : 0.XXXX     (status: <success|runtime_error|install_error>)
Predicted LB  : 0.XXXX     PI=[0.XXXX, 0.XXXX]  → inside? <yes|no>
Gap (CV-LB)   : +0.XXXX    (<CV>LB | LB>CV>)

CV→LB model now:
  n_pairs       : <k>
  pearson_r     : <r>
  residual_std  : <sigma>
  trust_level   : <low|medium|high>

Insight       : <one sentence>
Action next   : <what /plan should propose next cycle>
═════════════════════════════════════════════
```

## Hard Rules

- ALWAYS pass `--status install_error` if DACON rejected at install time so the quota counter stays correct.
- ALWAYS refit the correlation model before printing insight.
- NEVER manually edit `competition_meta.yaml.submissions_log` — always go through `track_submission.py`.
- NEVER mark an experiment SUBMITTED unless LB was actually returned by DACON.
