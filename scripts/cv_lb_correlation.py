#!/usr/bin/env python3
"""CV → LB correlation model.

Why this exists:
  We only get 10 DACON submissions per day. We must not waste a slot on a
  candidate whose predicted LB does not beat the current best by more than
  the prediction's own uncertainty.

What it does:
  1. Loads (cv_macro_f1, lb_macro_f1) pairs from competition_meta.yaml.submissions_log
  2. Fits a 1D linear regression LB = a * CV + b (least squares)
  3. Reports correlation strength (Pearson r, MAE, residual std)
  4. Predicts LB for a given CV with an approximate prediction interval
     (PI = predicted ± 1.96 * residual_std)

Usage:
    python scripts/cv_lb_correlation.py                       # show diagnostics
    python scripts/cv_lb_correlation.py --predict 0.78         # predict LB for CV=0.78
    python scripts/cv_lb_correlation.py --predict 0.78 --json  # machine-readable

Output (--json):
{
  "n_pairs": 6,
  "pearson_r": 0.87,
  "mae": 0.012,
  "residual_std": 0.018,
  "slope": 0.92,
  "intercept": 0.04,
  "predicted_lb": 0.758,
  "pi_low": 0.722,
  "pi_high": 0.794,
  "trust_level": "medium"
}

trust_level rules:
  high   : n_pairs >= 8 AND pearson_r >= 0.7 AND residual_std <= 0.015
  medium : n_pairs >= 4 AND pearson_r >= 0.5
  low    : otherwise (fall back to raw CV ranking, do not gate submissions)

Decision helper for /rank and /submit-result:
  expected_gain  = predicted_lb - current_best_lb
  uncertainty    = pi_high - predicted_lb
  worth_submitting = expected_gain > uncertainty AND trust_level != 'low'
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("PyYAML required: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "competition_meta.yaml"


def load_pairs() -> list[tuple[float, float]]:
    """Pull (cv_macro_f1, lb_macro_f1) pairs from submissions_log."""
    if not META.exists():
        return []
    with META.open() as f:
        meta = yaml.safe_load(f) or {}
    log = meta.get("submissions_log") or []
    pairs: list[tuple[float, float]] = []
    for entry in log:
        if not isinstance(entry, dict):
            continue
        if entry.get("counted_against_quota") is False:
            # Install error — no LB returned.
            continue
        cv = entry.get("cv_score")
        lb = entry.get("lb_score")
        if isinstance(cv, (int, float)) and isinstance(lb, (int, float)):
            pairs.append((float(cv), float(lb)))
    return pairs


def fit_linear(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """Ordinary least squares y = a*x + b with diagnostics."""
    n = len(pairs)
    if n < 2:
        return {"slope": 1.0, "intercept": 0.0, "pearson_r": 0.0, "residual_std": 0.0, "mae": 0.0}
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else 1.0
    intercept = my - slope * mx
    denom = math.sqrt(sxx * syy)
    pearson = sxy / denom if denom > 0 else 0.0
    residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
    mae = sum(abs(r) for r in residuals) / n
    if n > 2:
        var = sum(r * r for r in residuals) / (n - 2)
        residual_std = math.sqrt(max(var, 0.0))
    else:
        residual_std = 0.0
    return {
        "slope": slope,
        "intercept": intercept,
        "pearson_r": pearson,
        "residual_std": residual_std,
        "mae": mae,
    }


def trust_level(n: int, pearson: float, residual_std: float) -> str:
    if n >= 8 and pearson >= 0.7 and residual_std <= 0.015:
        return "high"
    if n >= 4 and pearson >= 0.5:
        return "medium"
    return "low"


def predict(cv: float, fit: dict[str, float]) -> tuple[float, float, float]:
    """Return (predicted_lb, pi_low, pi_high) for 95% prediction interval."""
    pred = fit["slope"] * cv + fit["intercept"]
    pi = 1.96 * fit["residual_std"]
    return pred, pred - pi, pred + pi


def build_report(predict_cv: float | None) -> dict[str, Any]:
    pairs = load_pairs()
    fit = fit_linear(pairs)
    report: dict[str, Any] = {
        "n_pairs": len(pairs),
        "pearson_r": round(fit["pearson_r"], 4),
        "mae": round(fit["mae"], 4),
        "residual_std": round(fit["residual_std"], 4),
        "slope": round(fit["slope"], 4),
        "intercept": round(fit["intercept"], 4),
        "trust_level": trust_level(len(pairs), fit["pearson_r"], fit["residual_std"]),
    }
    if predict_cv is not None:
        pred, lo, hi = predict(predict_cv, fit)
        report["input_cv"] = predict_cv
        report["predicted_lb"] = round(pred, 4)
        report["pi_low"] = round(lo, 4)
        report["pi_high"] = round(hi, 4)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", type=float, default=None, help="CV macro-F1 to predict LB for")
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    args = ap.parse_args()

    report = build_report(args.predict)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"CV-LB correlation model — n={report['n_pairs']} pairs")
    print(f"  pearson_r     : {report['pearson_r']}")
    print(f"  mae           : {report['mae']}")
    print(f"  residual_std  : {report['residual_std']}")
    print(f"  fit           : LB = {report['slope']} * CV + {report['intercept']}")
    print(f"  trust_level   : {report['trust_level']}")
    if args.predict is not None:
        print()
        print(f"Prediction for CV={report['input_cv']}:")
        print(f"  predicted_lb : {report['predicted_lb']}")
        print(f"  95% PI       : [{report['pi_low']}, {report['pi_high']}]")
        if report["trust_level"] == "low":
            print("  WARNING: trust_level=low — do NOT gate submissions on this prediction yet.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
