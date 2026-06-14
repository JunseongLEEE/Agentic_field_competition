#!/usr/bin/env python3
"""Evaluate experiment CV results: compare against baseline, check for leakage/overfitting."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_LOG = PROJECT_ROOT / "EXPERIMENT_LOG.csv"


def load_experiment_log():
    """Load experiment log as list of dicts."""
    if not EXPERIMENT_LOG.exists():
        return []
    with open(EXPERIMENT_LOG) as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_baseline_score(log_entries):
    """Get the best baseline score from completed experiments."""
    completed = [e for e in log_entries if e.get("status") == "COMPLETED" and e.get("cv_score")]
    if not completed:
        return None
    return max(float(e["cv_score"]) for e in completed)


def check_leakage(exp_path: Path, cv_scores: list):
    """Check for potential data leakage signals."""
    flags = []

    if not cv_scores:
        return flags

    cv_std = np.std(cv_scores)
    cv_mean = np.mean(cv_scores)

    # Check fold variance
    max_deviation = max(abs(s - cv_mean) for s in cv_scores)
    if max_deviation > 3 * cv_std and cv_std > 0:
        flags.append(f"Single fold deviates {max_deviation/cv_std:.1f} sigma from mean")

    # Check prediction distributions
    oof_path = exp_path / "oof_preds.npy"
    test_path = exp_path / "test_preds.npy"

    if oof_path.exists() and test_path.exists():
        oof = np.load(oof_path)
        test = np.load(test_path)

        # Distribution shift check
        oof_mean, oof_std = np.mean(oof), np.std(oof)
        test_mean, test_std = np.mean(test), np.std(test)

        mean_shift = abs(oof_mean - test_mean) / max(oof_std, 1e-8)
        if mean_shift > 0.5:
            flags.append(f"Distribution shift: OOF mean={oof_mean:.4f}, Test mean={test_mean:.4f}")

        # Check for NaN/Inf
        if np.any(np.isnan(oof)) or np.any(np.isinf(oof)):
            flags.append("NaN/Inf detected in OOF predictions")
        if np.any(np.isnan(test)) or np.any(np.isinf(test)):
            flags.append("NaN/Inf detected in test predictions")

    return flags


def evaluate(exp_path: Path):
    """Full evaluation of an experiment."""
    exp_path = Path(exp_path).resolve()

    if not exp_path.exists():
        print(f"ERROR: {exp_path} not found")
        sys.exit(1)

    # Load training results
    train_log_path = exp_path / "train_log.json"
    if not train_log_path.exists():
        print("ERROR: train_log.json not found. Run the experiment first.")
        sys.exit(1)

    with open(train_log_path) as f:
        results = json.load(f)

    cv_scores = results.get("cv_scores", [])
    cv_mean = results.get("cv_mean", 0)
    cv_std = results.get("cv_std", 0)

    # Load log for comparison
    log_entries = load_experiment_log()
    baseline_score = get_baseline_score(log_entries)

    # Run checks
    leakage_flags = check_leakage(exp_path, cv_scores)

    # Compute improvement
    improvement = (cv_mean - baseline_score) if baseline_score else None

    # Stability grade
    if cv_std == 0:
        stability_grade = "N/A"
    elif cv_std < 0.005:
        stability_grade = "A"
    elif cv_std < 0.01:
        stability_grade = "B"
    elif cv_std < 0.02:
        stability_grade = "C"
    else:
        stability_grade = "D"

    # Recommendation
    if leakage_flags:
        recommendation = "REVIEW"
        reason = f"Leakage flags: {'; '.join(leakage_flags)}"
    elif baseline_score and cv_mean < baseline_score:
        recommendation = "REJECT"
        reason = f"CV ({cv_mean:.6f}) worse than baseline ({baseline_score:.6f})"
    elif stability_grade == "D":
        recommendation = "REVIEW"
        reason = "High CV variance"
    else:
        recommendation = "CANDIDATE"
        reason = f"Improvement: {improvement:+.6f}" if improvement else "First experiment"

    # Print report
    report = {
        "experiment_id": exp_path.name,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d"),
        "cv_score": cv_mean,
        "cv_std": cv_std,
        "cv_fold_scores": cv_scores,
        "baseline_score": baseline_score,
        "improvement": improvement,
        "stability_grade": stability_grade,
        "leakage_flags": leakage_flags,
        "recommendation": recommendation,
        "reason": reason,
    }

    print(f"\n{'='*60}")
    print(f"EVALUATION REPORT: {exp_path.name}")
    print(f"{'='*60}")
    print(f"  CV Score:     {cv_mean:.6f} +/- {cv_std:.6f}")
    print(f"  Baseline:     {baseline_score:.6f}" if baseline_score else "  Baseline:     N/A (first experiment)")
    print(f"  Improvement:  {improvement:+.6f}" if improvement else "  Improvement:  N/A")
    print(f"  Stability:    {stability_grade}")
    print(f"  Leakage:      {'CLEAN' if not leakage_flags else 'FLAGS DETECTED'}")
    for flag in leakage_flags:
        print(f"    - {flag}")
    print(f"  Recommendation: {recommendation}")
    print(f"  Reason: {reason}")
    print(f"{'='*60}")

    # Save evaluation report
    eval_path = exp_path / "evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved evaluation to: {eval_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate experiment CV results")
    parser.add_argument("--exp", required=True, help="Path to experiment directory")
    args = parser.parse_args()

    evaluate(Path(args.exp))
