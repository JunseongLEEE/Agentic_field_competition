# AI Competition Workflow (Swarms-Style Multi-Agent)

A structured experiment management system inspired by [am-will/swarms](https://github.com/am-will/swarms) orchestration patterns, adapted for AI/ML competitions.

## Philosophy

From swarms, we borrow:
- **Two-phase orchestration**: plan first, then execute in parallel waves
- **Dependency-aware execution**: experiments declare what they depend on
- **Context preservation**: orchestrator maintains state across iterations
- **Isolation**: each experiment is self-contained and reproducible

What we add for competitions:
- Anti-overfitting protocols (CV-first, leakage detection)
- Submission management with daily candidate limits
- Reproducibility through config files, seeds, and git tracking

## Quick Start

```bash
# 1. Create an experiment
python scripts/create_experiment.py --name "baseline_lgbm" \
  --hypothesis "LightGBM with raw features establishes baseline" \
  --model lightgbm

# 2. Implement the experiment (edit experiments/exp_001_baseline_lgbm/train.py)

# 3. Run it
python scripts/run_experiment.py --exp experiments/exp_001_baseline_lgbm

# 4. Evaluate results
python scripts/evaluate_cv.py --exp experiments/exp_001_baseline_lgbm

# 5. Package for submission
python scripts/package_submit.py --exp experiments/exp_001_baseline_lgbm

# 6. Validate the zip
python scripts/validate_submission.py --zip submissions/exp_001_baseline_lgbm.zip

# 7. Rank all candidates
python scripts/rank_candidates.py
```

## Multi-Agent Workflow with Claude Code

Use these agent prompts (in `agents/`) with Claude Code's Agent tool:

| Agent | Role | When to Use |
|-------|------|-------------|
| `orchestrator` | Plans experiments, defines dependencies | Start of day, after reviewing results |
| `model_developer` | Implements experiment code | After orchestrator creates plan |
| `experiment_runner` | Runs experiments, captures output | After code is implemented |
| `evaluator` | Checks for leakage, compares to baseline | After experiment completes |
| `packager` | Creates submission zip | For CANDIDATE experiments |
| `submission_candidate_selector` | Ranks and picks top candidates | Before manual submission |

### Typical Session

```
1. [You] → Ask orchestrator: "What experiments should we run next?"
2. [Orchestrator] → Returns plan with 2-3 experiments and dependencies
3. [Model Developer] → Implements exp_004 and exp_005 in parallel (no deps)
4. [Runner] → Runs both experiments
5. [Evaluator] → Checks results, marks exp_004 as CANDIDATE
6. [Packager] → Creates submission zip for exp_004
7. [Selector] → Ranks all candidates, recommends top picks
8. [You] → Manually submit to competition website
```

## Directory Structure

```
경진대회/
├── CLAUDE.md                    # Agent instructions
├── RULES.md                     # Competition rules & constraints
├── EXPERIMENT_GOAL.md           # Strategy & hypothesis backlog
├── EXPERIMENT_LOG.csv           # All experiments tracker
├── LEADERBOARD_LOG.md           # Submission history
├── SUBMISSION_CANDIDATES.md     # Daily candidate ranking
├── README.md                    # This file
├── agents/                      # Agent prompt files
│   ├── orchestrator.md
│   ├── model_developer.md
│   ├── experiment_runner.md
│   ├── evaluator.md
│   ├── packager.md
│   └── submission_candidate_selector.md
├── scripts/                     # Automation scripts
│   ├── create_experiment.py
│   ├── run_experiment.py
│   ├── evaluate_cv.py
│   ├── package_submit.py
│   ├── validate_submission.py
│   └── rank_candidates.py
├── data/                        # Competition data (gitignored)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── experiments/                  # One folder per experiment
│   ├── exp_001_baseline_lgbm/
│   └── exp_002_feature_eng_v1/
├── logs/                        # Evaluation and run logs
├── artifacts/                   # Shared artifacts (embeddings, etc.)
└── submissions/                 # Packaged submission zips
```

## Key Principles

1. **CV is truth** — Never trust LB over local CV
2. **Isolate experiments** — Each experiment is independent and reproducible
3. **Track everything** — Config, seed, commit hash, scores
4. **Diversity over depth** — Prefer diverse approaches for ensemble
5. **Manual submission** — Human makes the final call
