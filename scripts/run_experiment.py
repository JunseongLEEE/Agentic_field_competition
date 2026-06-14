#!/usr/bin/env python3
"""Run an experiment and capture outputs."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_experiment(exp_path: Path):
    """Run a single experiment and capture results."""
    exp_path = Path(exp_path).resolve()

    if not exp_path.exists():
        print(f"ERROR: Experiment directory not found: {exp_path}")
        sys.exit(1)

    config_path = exp_path / "config.yaml"
    train_path = exp_path / "train.py"

    if not config_path.exists():
        print(f"ERROR: config.yaml not found in {exp_path}")
        sys.exit(1)

    if not train_path.exists():
        print(f"ERROR: train.py not found in {exp_path}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Running experiment: {exp_path.name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Install requirements if present
    req_path = exp_path / "requirements.txt"
    if req_path.exists():
        print("Installing additional requirements...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path), "-q"],
            cwd=str(exp_path)
        )

    # Run the training script
    start_time = time.time()
    result = subprocess.run(
        [sys.executable, "train.py"],
        cwd=str(exp_path),
        capture_output=True,
        text=True
    )
    elapsed = time.time() - start_time

    # Save run log
    run_log = {
        "experiment_id": exp_path.name,
        "started_at": datetime.now().isoformat(),
        "runtime_seconds": round(elapsed, 2),
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

    log_file = exp_path / "run_log.txt"
    with open(log_file, "w") as f:
        f.write(f"=== STDOUT ===\n{result.stdout}\n")
        f.write(f"=== STDERR ===\n{result.stderr}\n")
        f.write(f"=== RETURN CODE: {result.returncode} ===\n")
        f.write(f"=== RUNTIME: {elapsed:.1f}s ===\n")

    # Print output
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"STDERR:\n{result.stderr}", file=sys.stderr)

    # Check outputs
    print(f"\n{'='*60}")
    print(f"Runtime: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Return code: {result.returncode}")

    expected_outputs = ["oof_preds.npy", "test_preds.npy"]
    for out in expected_outputs:
        out_path = exp_path / out
        if out_path.exists():
            print(f"  [OK] {out} ({out_path.stat().st_size} bytes)")
        else:
            print(f"  [MISSING] {out}")

    # Check for train_log.json with CV results
    train_log_path = exp_path / "train_log.json"
    if train_log_path.exists():
        with open(train_log_path) as f:
            train_results = json.load(f)
        cv_mean = train_results.get("cv_mean", "N/A")
        cv_std = train_results.get("cv_std", "N/A")
        print(f"\n  CV Score: {cv_mean} +/- {cv_std}")
    else:
        print("\n  [WARNING] train_log.json not found - cannot extract CV score")

    if result.returncode != 0:
        print(f"\n  [FAILED] Experiment failed with return code {result.returncode}")
        sys.exit(1)
    else:
        print(f"\n  [SUCCESS] Experiment completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an experiment")
    parser.add_argument("--exp", required=True, help="Path to experiment directory")
    args = parser.parse_args()

    run_experiment(Path(args.exp))
