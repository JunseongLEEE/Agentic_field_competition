"""exp_013_engineered_char_calib — DACON server inference (SUBMITTED, OFFLINE).

Reads data/test.jsonl + data/sample_submission.csv, loads artifacts from model/,
applies per-class calibration weights (proba * class_weights), argmax -> class string,
writes output/submission.csv (id,action) in sample_submission id order.

OFFLINE ONLY. stdlib csv/json (no pandas). numpy/scipy/sklearn(pickle)/lightgbm only.
requirements.txt pins ONLY scikit-learn + lightgbm (numpy/scipy from server stack).
"""
import os
import sys
import csv
import json

import numpy as np
import joblib
from scipy import sparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import features as F  # noqa: E402

MODEL = os.path.join(ROOT, "model")


def main():
    wv = joblib.load(os.path.join(MODEL, "word_vec.pkl"))
    cv = joblib.load(os.path.join(MODEL, "char_vec.pkl"))
    cat_map = json.load(open(os.path.join(MODEL, "cat_mappings.json")))
    clf = joblib.load(os.path.join(MODEL, "lgbm.pkl"))
    class_order = json.load(open(os.path.join(MODEL, "class_order.json")))
    cw = json.load(open(os.path.join(MODEL, "class_weights.json")))
    weights = np.array([cw.get(c, 1.0) for c in class_order], dtype=np.float64)

    samples = F.load_jsonl(os.path.join("data", "test.jsonl"))
    ids, prompts, records = F.build_records(samples)

    X = F.build_X(wv, cv, cat_map, prompts, records)
    proba = clf.predict_proba(X)
    pred = (proba * weights).argmax(1)
    id2pred = {ids[i]: class_order[int(pred[i])] for i in range(len(ids))}

    with open(os.path.join("data", "sample_submission.csv"), newline="") as f:
        rows = list(csv.reader(f))
    body = rows[1:]  # skip header
    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "submission.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "action"])
        for r in body:
            sid = r[0]
            w.writerow([sid, id2pred.get(sid, class_order[0])])
    print(f"wrote output/submission.csv ({len(body)} rows)")


if __name__ == "__main__":
    main()
