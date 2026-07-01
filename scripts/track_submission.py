#!/usr/bin/env python3
"""
Record a submission event into competition_meta.yaml submissions_log.

Usage:
    python scripts/track_submission.py --exp exp_001_baseline --lb 0.8234 --status success
    python scripts/track_submission.py --exp exp_002 --status install_error  # doesn't count

Called by /submit-result skill.
"""
import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).parent.parent
META_PATH = ROOT / "competition_meta.yaml"
KST = timezone(timedelta(hours=9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name")
    ap.add_argument("--lb", type=float, default=None, help="leaderboard score")
    ap.add_argument("--cv", type=float, default=None,
                    help="CV macro-F1 (default: read cv_mean from experiments/<exp>/train_log.json)")
    ap.add_argument(
        "--status",
        choices=["success", "install_error", "runtime_error"],
        default="success",
    )
    args = ap.parse_args()

    if not META_PATH.exists():
        print(f"ERROR: {META_PATH} not found", file=sys.stderr)
        sys.exit(1)

    with open(META_PATH) as f:
        meta = yaml.safe_load(f)

    # cv_score is REQUIRED for the CV->LB correlation model to form (cv, lb) pairs.
    # Prefer --cv; otherwise read cv_mean from the experiment's train_log.json.
    cv_score = args.cv
    if cv_score is None:
        import json
        tl = ROOT / "experiments" / args.exp / "train_log.json"
        if tl.exists():
            try:
                cv_score = float(json.load(open(tl)).get("cv_mean"))
            except Exception:
                cv_score = None

    now = datetime.now(KST)
    entry = {
        "date": now.date().isoformat(),
        "timestamp": now.isoformat(),
        "experiment": args.exp,
        "cv_score": cv_score,
        "lb_score": args.lb,
        "status": args.status,
        # install_error does NOT count against daily limit per DACON rules
        "counts_against_daily": args.status != "install_error",
    }

    log = meta.get("submissions_log") or []
    log.append(entry)
    meta["submissions_log"] = log

    with open(META_PATH, "w") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)

    # Compute today's count
    today = now.date().isoformat()
    today_count = sum(
        1 for s in log if s.get("date") == today and s.get("counts_against_daily", True)
    )
    limit = meta.get("limits", {}).get("daily_submission_limit", 20)
    remaining = max(0, limit - today_count)

    print(f"✓ Recorded submission: {args.exp} ({args.status})")
    print(f"  Today: {today_count}/{limit} used | {remaining} remaining")


if __name__ == "__main__":
    main()
