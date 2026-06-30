#!/usr/bin/env python3
"""
Check time state — deadlines and daily submission quota.

Used by /auto Step 0 and /plan to inject time awareness into orchestrator.
Reads competition_meta.yaml and prints actionable urgency info.

Usage:
    python scripts/check_time_state.py
    python scripts/check_time_state.py --json   # machine-readable output
"""
import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Run: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).parent.parent
META_PATH = ROOT / "competition_meta.yaml"
KST = timezone(timedelta(hours=9))


def load_meta():
    if not META_PATH.exists():
        print(f"ERROR: {META_PATH} not found", file=sys.stderr)
        sys.exit(1)
    with open(META_PATH) as f:
        return yaml.safe_load(f)


def parse_iso(s):
    return datetime.fromisoformat(s)


def compute_urgency(meta, now=None):
    now = now or datetime.now(KST)
    deadlines = meta.get("deadlines", {})
    limits = meta.get("limits", {})
    submissions_log = meta.get("submissions_log") or []

    # Days until each deadline
    deadline_info = {}
    for name, iso in deadlines.items():
        dt = parse_iso(iso)
        delta = dt - now
        deadline_info[name] = {
            "datetime": iso,
            "days_remaining": delta.total_seconds() / 86400,
            "hours_remaining": delta.total_seconds() / 3600,
            "passed": delta.total_seconds() < 0,
        }

    # Submissions used today (KST)
    today = now.date().isoformat()
    today_subs = [
        s for s in submissions_log
        if s.get("date") == today and s.get("counts_against_daily", True)
    ]
    daily_limit = limits.get("daily_submission_limit", 10)

    return {
        "now": now.isoformat(),
        "deadlines": deadline_info,
        "submissions_today": len(today_subs),
        "submissions_remaining_today": max(0, daily_limit - len(today_subs)),
        "daily_limit": daily_limit,
        "today_submissions_detail": today_subs,
    }


def format_human(state):
    lines = []
    lines.append("━" * 60)
    lines.append("⏱  TIME STATE")
    lines.append("━" * 60)
    lines.append(f"Now: {state['now']}")
    lines.append("")
    lines.append("DEADLINES:")
    for name, info in state["deadlines"].items():
        if info["passed"]:
            status = f"🔴 PASSED ({-info['days_remaining']:.1f}d ago)"
        elif info["days_remaining"] < 1:
            status = f"🔥 {info['hours_remaining']:.1f}h remaining"
        elif info["days_remaining"] < 3:
            status = f"⚠️  {info['days_remaining']:.1f}d remaining"
        else:
            status = f"✅ {info['days_remaining']:.1f}d remaining"
        lines.append(f"  {name:20s} {info['datetime']}  {status}")
    lines.append("")
    lines.append("DAILY SUBMISSION QUOTA:")
    used = state["submissions_today"]
    remaining = state["submissions_remaining_today"]
    limit = state["daily_limit"]
    bar = "█" * used + "░" * remaining
    lines.append(f"  [{bar}] {used}/{limit} used  ({remaining} remaining today)")
    if used > 0:
        lines.append("  Today's submissions:")
        for s in state["today_submissions_detail"]:
            lines.append(f"    - {s.get('timestamp','?')} {s.get('experiment','?')} → {s.get('status','?')}")
    lines.append("━" * 60)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    meta = load_meta()
    state = compute_urgency(meta)

    if args.json:
        print(json.dumps(state, indent=2, ensure_ascii=False))
    else:
        print(format_human(state))


if __name__ == "__main__":
    main()
