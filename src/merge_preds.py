"""샤드 예측 JSON 병합 → 제출 CSV.

사용: python src/merge_preds.py --preds a.json b.json --out submissions/sub_x.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SUB, make_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    merged = {}
    for p in args.preds:
        merged.update(json.load(open(p)))
    path = make_submission(merged, args.out)
    print("saved:", path, f"({len(merged)} preds)")


if __name__ == "__main__":
    main()
