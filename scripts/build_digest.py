#!/usr/bin/env python3
"""Build experiment_digest.md — single file summarizing ALL experiments for quick agent context recovery.

Run after any experiment completes or LB score is recorded.
Usage: python scripts/build_digest.py
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DIGEST_PATH = PROJECT_ROOT / "logs" / "experiment_digest.md"
INSIGHTS_PATH = PROJECT_ROOT / "logs" / "insights.jsonl"
EXPERIMENT_LOG = PROJECT_ROOT / "EXPERIMENT_LOG.csv"


def extract_summary_table(summary_path: Path) -> dict:
    """Extract key fields from SUMMARY.md into a dict."""
    text = summary_path.read_text()
    info = {"path": str(summary_path.parent.name)}

    # Extract one-liner
    match = re.search(r"^> One-line:\s*(.+)$", text, re.MULTILINE)
    info["one_line"] = match.group(1).strip() if match else ""

    # Extract table values
    for pattern, key in [
        (r"\|\s*Model\s*\|\s*(.+?)\s*\|", "model"),
        (r"\|\s*Features\s*\|\s*(.+?)\s*\|", "features"),
        (r"\|\s*CV Mean\s*\|\s*(.+?)\s*\|", "cv_mean"),
        (r"\|\s*CV Std\s*\|\s*(.+?)\s*\|", "cv_std"),
        (r"\|\s*LB Score\s*\|\s*(.+?)\s*\|", "lb_score"),
        (r"\|\s*CV-LB Gap\s*\|\s*(.+?)\s*\|", "cv_lb_gap"),
        (r"\|\s*Status\s*\|\s*(.+?)\s*\|", "status"),
        (r"\|\s*Inference Speed\s*\|\s*(.+?)\s*\|", "inference_speed"),
    ]:
        m = re.search(pattern, text)
        info[key] = m.group(1).strip() if m else ""

    # Extract insight section
    insight_match = re.search(r"## Insight\n((?:- .+\n?)+)", text)
    info["insights"] = insight_match.group(1).strip() if insight_match else ""

    # Extract what worked
    worked_match = re.search(r"## What Worked\n((?:- .+\n?)+)", text)
    info["what_worked"] = worked_match.group(1).strip() if worked_match else ""

    # Extract what didn't
    didnt_match = re.search(r"## What Didn't Work\n((?:- .+\n?)+)", text)
    info["what_didnt"] = didnt_match.group(1).strip() if didnt_match else ""

    return info


def load_recent_insights(n=10):
    """Load last N insights."""
    if not INSIGHTS_PATH.exists():
        return []
    with open(INSIGHTS_PATH) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    return lines[-n:]


def build_digest():
    """Build the full digest file."""
    experiments = sorted(EXPERIMENTS_DIR.glob("exp_*/SUMMARY.md"))

    if not experiments:
        DIGEST_PATH.write_text("# Experiment Digest\n\nNo experiments yet.\n")
        print("No experiments found.")
        return

    summaries = [extract_summary_table(p) for p in experiments]

    # Sort: CANDIDATE/SUBMITTED first, then by CV score descending
    def sort_key(s):
        status_order = {"SUBMITTED": 0, "CANDIDATE": 1, "COMPLETED": 2, "REJECTED": 3, "PLANNED": 4}
        try:
            cv = float(s.get("cv_mean", 0))
        except (ValueError, TypeError):
            cv = 0
        return (status_order.get(s.get("status", ""), 5), -cv)

    summaries.sort(key=sort_key)

    # Build markdown
    lines = []
    lines.append("# Experiment Digest")
    lines.append(f"\n> Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Total: {len(summaries)} experiments\n")

    # === SCOREBOARD ===
    lines.append("## Scoreboard\n")
    lines.append("| # | Experiment | Model | CV | LB | Gap | Speed | Status |")
    lines.append("|---|-----------|-------|----|----|-----|-------|--------|")

    for i, s in enumerate(summaries, 1):
        lines.append(
            f"| {i} | {s['path']} | {s.get('model','')} | {s.get('cv_mean','')} "
            f"| {s.get('lb_score','')} | {s.get('cv_lb_gap','')} "
            f"| {s.get('inference_speed','')} | {s.get('status','')} |"
        )

    # === ACCUMULATED INSIGHTS ===
    lines.append("\n## Key Insights (What We've Learned)\n")

    all_worked = []
    all_didnt = []
    all_insights = []

    for s in summaries:
        if s.get("what_worked"):
            all_worked.append(f"**{s['path']}**: {s['what_worked']}")
        if s.get("what_didnt"):
            all_didnt.append(f"**{s['path']}**: {s['what_didnt']}")
        if s.get("insights"):
            all_insights.append(f"**{s['path']}**: {s['insights']}")

    if all_worked:
        lines.append("### What Worked")
        for w in all_worked:
            lines.append(f"- {w}")

    if all_didnt:
        lines.append("\n### What Didn't Work")
        for d in all_didnt:
            lines.append(f"- {d}")

    if all_insights:
        lines.append("\n### Experiment Insights")
        for ins in all_insights:
            lines.append(f"- {ins}")

    # === SUBMISSION INSIGHTS ===
    recent_insights = load_recent_insights(10)
    if recent_insights:
        lines.append("\n## Submission Insights (CV-LB Feedback Loop)\n")
        for ins in recent_insights:
            lines.append(
                f"- [{ins.get('date','')}] **{ins.get('experiment','')}**: "
                f"CV={ins.get('cv_score','')} → LB={ins.get('lb_score','')} "
                f"(gap={ins.get('gap','')}) — {ins.get('insight','')}"
            )

    # === ONE-LINE SUMMARIES ===
    lines.append("\n## Quick Reference\n")
    for s in summaries:
        if s.get("one_line"):
            lines.append(f"- **{s['path']}**: {s['one_line']}")

    lines.append("")
    DIGEST_PATH.write_text("\n".join(lines))
    print(f"Digest built: {DIGEST_PATH}")
    print(f"  {len(summaries)} experiments summarized")


if __name__ == "__main__":
    build_digest()
