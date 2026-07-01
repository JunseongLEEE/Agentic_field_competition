"""exp_010 — DACON inference, 5-FOLD BAGGED (OFFLINE).

Uses the per-fold models trained during CV (word_vec_f{k}, cat_mappings_f{k}, lgbm_f{k})
and averages their probabilities — no full-train model needed. Reads data/test.jsonl +
data/sample_submission.csv, writes output/submission.csv (id,action strings).
stdlib csv/json only (no pandas); numpy/scipy from the server's stack.
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
N_FOLDS = 5


def main():
    class_order = json.load(open(os.path.join(MODEL, "class_order.json")))
    samples = F.load_jsonl(os.path.join("data", "test.jsonl"))
    ids, prompts, records = F.build_records(samples)

    probs = None
    for k in range(N_FOLDS):
        wv = joblib.load(os.path.join(MODEL, f"word_vec_f{k}.pkl"))
        cat_map = json.load(open(os.path.join(MODEL, f"cat_mappings_f{k}.json")))
        m = joblib.load(os.path.join(MODEL, f"lgbm_f{k}.pkl"))
        Xw = wv.transform(prompts)
        Xd = sparse.csr_matrix(F.records_to_dense(records, cat_map).astype(np.float32))
        X = sparse.hstack([Xw, Xd], format="csr")
        p = m.predict_proba(X)
        probs = p if probs is None else probs + p
    probs /= N_FOLDS
    pred = probs.argmax(1)
    id2pred = {ids[i]: class_order[int(pred[i])] for i in range(len(ids))}

    with open(os.path.join("data", "sample_submission.csv"), newline="") as f:
        rows = list(csv.reader(f))
    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "submission.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "action"])
        for r in rows[1:]:
            sid = r[0]
            w.writerow([sid, id2pred.get(sid, class_order[0])])
    print(f"wrote {len(rows)-1} preds (5-fold bagged)")


if __name__ == "__main__":
    main()
