#!/usr/bin/env python3
"""Rank submission candidates and update SUBMISSION_CANDIDATES.md."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
EXPERIMENT_LOG = PROJECT_ROOT / "EXPERIMENT_LOG.csv"
CANDIDATES_MD = PROJECT_ROOT / "SUBMISSION_CANDIDATES.md"

MAX_CANDIDATES = 10


def load_candidates():
    """Find all experiments with CANDIDATE status or evaluation recommendation."""
    candidates = []

    for exp_dir in sorted(EXPERIMENTS_DIR.glob("exp_*")):
        eval_path = exp_dir / "evaluation.json"
        train_log_path = exp_dir / "train_log.json"

        if not eval_path.exists():
            continue

        with open(eval_path) as f:
            evaluation = json.load(f)

        if evaluation.get("recommendation") != "CANDIDATE":
            continue

        # Get CV details
        cv_score = evaluation.get("cv_score", 0)
        cv_std = evaluation.get("cv_std", 0)

        # Get model type from config
        config_path = exp_dir / "config.yaml"
        model_type = "unknown"
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
            model_type = config.get("model", {}).get("type", "unknown")

        # Check if already submitted
        zip_path = SUBMISSIONS_DIR / f"{exp_dir.name}.zip"
        has_submission = zip_path.exists()

        candidates.append({
            "experiment_id": exp_dir.name,
            "cv_score": cv_score,
            "cv_std": cv_std,
            "model_type": model_type,
            "stability_grade": evaluation.get("stability_grade", "N/A"),
            "has_submission_zip": has_submission,
        })

    return candidates


def compute_composite_score(candidate, all_candidates):
    """Compute ranking score for a candidate."""
    if not all_candidates:
        return 0

    scores = [c["cv_score"] for c in all_candidates]
    stds = [c["cv_std"] for c in all_candidates]

    max_score = max(scores) if scores else 1
    min_score = min(scores) if scores else 0
    score_range = max_score - min_score if max_score != min_score else 1

    max_std = max(stds) if stds else 1

    # Normalize
    norm_score = (candidate["cv_score"] - min_score) / score_range
    norm_stability = 1 - (candidate["cv_std"] / max_std) if max_std > 0 else 1

    # Diversity bonus: unique model type gets bonus
    model_counts = {}
    for c in all_candidates:
        model_counts[c["model_type"]] = model_counts.get(c["model_type"], 0) + 1
    diversity = 1.0 / model_counts.get(candidate["model_type"], 1)

    # Composite
    composite = 0.5 * norm_score + 0.2 * norm_stability + 0.3 * diversity
    return composite


def rank_and_select(candidates, max_n=MAX_CANDIDATES):
    """Rank candidates and apply diversity constraints."""
    if not candidates:
        return []

    # Compute scores
    for c in candidates:
        c["composite_score"] = compute_composite_score(c, candidates)

    # Sort by composite score
    ranked = sorted(candidates, key=lambda x: x["composite_score"], reverse=True)

    # Apply diversity constraints: max 3 from same model family
    selected = []
    model_counts = {}

    for c in ranked:
        mt = c["model_type"]
        if model_counts.get(mt, 0) >= 3:
            continue
        selected.append(c)
        model_counts[mt] = model_counts.get(mt, 0) + 1
        if len(selected) >= max_n:
            break

    return selected


def update_candidates_md(selected):
    """Update SUBMISSION_CANDIDATES.md with ranked candidates."""
    today = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "# Submission Candidates\n",
        f"\n## Today's Candidates ({today}) — max {MAX_CANDIDATES}\n\n",
        "| Rank | Experiment | CV Score | CV Std | Model | Stability | Score | Priority |\n",
        "|------|-----------|----------|--------|-------|-----------|-------|----------|\n",
    ]

    for i, c in enumerate(selected, 1):
        priority = "SUBMIT_FIRST" if i <= 2 else ("SUBMIT_IF_SLOTS" if i <= 5 else "HOLD")
        lines.append(
            f"| {i} | {c['experiment_id']} | {c['cv_score']:.6f} | {c['cv_std']:.6f} "
            f"| {c['model_type']} | {c['stability_grade']} | {c['composite_score']:.3f} | {priority} |\n"
        )

    lines.extend([
        "\n## Selection Criteria\n",
        "1. **CV Score** (50%) — higher is better\n",
        "2. **Stability** (20%) — lower std preferred\n",
        "3. **Diversity** (30%) — different model families preferred\n",
        "\n## Notes\n",
        "- SUBMIT_FIRST: high confidence, submit these\n",
        "- SUBMIT_IF_SLOTS: good but not top priority\n",
        "- HOLD: keep as backup, submit only if slots remain\n",
        "- Manual submission required — do NOT auto-upload\n",
    ])

    CANDIDATES_MD.write_text("".join(lines))
    print(f"Updated: {CANDIDATES_MD}")


def main():
    print(f"Ranking submission candidates...")
    print(f"{'='*50}")

    candidates = load_candidates()
    print(f"Found {len(candidates)} candidates")

    if not candidates:
        print("No candidates found. Run evaluate_cv.py first.")
        sys.exit(0)

    selected = rank_and_select(candidates)

    print(f"\nTop {len(selected)} selections:")
    for i, c in enumerate(selected, 1):
        print(f"  {i}. {c['experiment_id']} — CV: {c['cv_score']:.6f} ({c['model_type']})")

    update_candidates_md(selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank submission candidates")
    parser.add_argument("--date", default="today", help="Date filter (unused, for future)")
    parser.add_argument("--max", type=int, default=MAX_CANDIDATES, help="Max candidates")
    args = parser.parse_args()

    MAX_CANDIDATES = args.max
    main()
