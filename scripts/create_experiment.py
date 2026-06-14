#!/usr/bin/env python3
"""Create a new experiment directory with boilerplate config and code."""

import argparse
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


def get_next_experiment_id():
    """Get the next experiment number."""
    existing = sorted(EXPERIMENTS_DIR.glob("exp_*"))
    if not existing:
        return 1
    last_num = int(existing[-1].name.split("_")[1])
    return last_num + 1


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def create_experiment(name: str, hypothesis: str, base: str = None, model_type: str = "lightgbm"):
    """Create a new experiment directory."""
    exp_num = get_next_experiment_id()
    exp_id = f"exp_{exp_num:03d}_{name}"
    exp_dir = EXPERIMENTS_DIR / exp_id

    if exp_dir.exists():
        print(f"ERROR: {exp_dir} already exists")
        return

    exp_dir.mkdir(parents=True)
    (exp_dir / "models").mkdir()

    # Copy base experiment if specified
    if base:
        base_dir = EXPERIMENTS_DIR / base
        if base_dir.exists():
            for f in ["train.py", "features.py", "model.py"]:
                src = base_dir / f
                if src.exists():
                    shutil.copy2(src, exp_dir / f)
            print(f"Copied base files from {base}")

    # Write config.yaml
    config_content = f"""experiment:
  id: {exp_id}
  hypothesis: "{hypothesis}"
  created: "{datetime.now().strftime('%Y-%m-%d %H:%M')}"
  git_commit: "{get_git_commit()}"
  status: PLANNED

data:
  train_path: ../../data/train.csv
  test_path: ../../data/test.csv
  target_col: target

cv:
  n_splits: 5
  strategy: stratified
  seed: 42

model:
  type: {model_type}
  params:
    # Add model-specific parameters here
    n_estimators: 1000
    learning_rate: 0.05
    early_stopping_rounds: 100

features:
  # Define feature engineering steps
  use_raw: true
  additional: []

output:
  oof_predictions: oof_preds.npy
  test_predictions: test_preds.npy
  model_dir: models/
  log_file: train_log.json
"""
    (exp_dir / "config.yaml").write_text(config_content)

    # Write README
    readme_content = f"""# {exp_id}

## Hypothesis
{hypothesis}

## Approach
<!-- Describe the approach -->

## Expected Outcome
<!-- What improvement do you expect and why? -->

## Results
<!-- Filled after running -->
- CV Score:
- CV Std:
- Runtime:

## Conclusion
<!-- What did we learn? -->
"""
    (exp_dir / "README.md").write_text(readme_content)

    # Write minimal train.py if not copied from base
    if not (exp_dir / "train.py").exists():
        train_content = '''#!/usr/bin/env python3
"""Training script for experiment."""

import json
import numpy as np
import yaml
from pathlib import Path

# Load config
config_path = Path(__file__).parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

SEED = config["cv"]["seed"]
np.random.seed(SEED)


def main():
    print(f"Running experiment: {config['experiment']['id']}")
    print(f"Hypothesis: {config['experiment']['hypothesis']}")

    # TODO: Implement training pipeline
    # 1. Load data
    # 2. Feature engineering
    # 3. CV loop
    # 4. Train on each fold, collect OOF predictions
    # 5. Generate test predictions (mean of fold models)
    # 6. Save outputs

    # Placeholder for results
    results = {
        "experiment_id": config["experiment"]["id"],
        "cv_scores": [],
        "cv_mean": 0.0,
        "cv_std": 0.0,
    }

    # Save results
    with open("train_log.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"CV Score: {results['cv_mean']:.6f} +/- {results['cv_std']:.6f}")


if __name__ == "__main__":
    main()
'''
        (exp_dir / "train.py").write_text(train_content)

    print(f"Created experiment: {exp_id}")
    print(f"  Directory: {exp_dir}")
    print(f"  Status: PLANNED")
    print(f"  Next: implement train.py, then run with run_experiment.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new experiment")
    parser.add_argument("--name", required=True, help="Short experiment name (no spaces)")
    parser.add_argument("--hypothesis", default="TBD", help="Experiment hypothesis")
    parser.add_argument("--base", default=None, help="Base experiment to copy from")
    parser.add_argument("--model", default="lightgbm", help="Model type")
    args = parser.parse_args()

    create_experiment(args.name, args.hypothesis, args.base, args.model)
