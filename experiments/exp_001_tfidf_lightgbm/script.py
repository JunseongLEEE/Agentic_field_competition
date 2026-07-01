"""exp_001_tfidf_lightgbm — DACON server inference (SUBMITTED, OFFLINE).

Reads data/test.jsonl + data/sample_submission.csv, loads artifacts from model/,
writes output/submission.csv with columns id,action (class strings).

NOTE: uses only stdlib csv/json + numpy/scipy/sklearn(via pickle)/lightgbm.
No pandas (avoids an extra C-extension import). requirements.txt pins ONLY
scikit-learn (to match the pickle) + lightgbm; numpy/scipy come from the server's
own consistent stack (pinning them caused an ABI 'dtype size changed' error).
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
    cat_map = json.load(open(os.path.join(MODEL, "cat_mappings.json")))
    clf = joblib.load(os.path.join(MODEL, "lgbm.pkl"))
    class_order = json.load(open(os.path.join(MODEL, "class_order.json")))

    samples = F.load_jsonl(os.path.join("data", "test.jsonl"))
    ids, prompts, records = F.build_records(samples)

    Xw = wv.transform(prompts)
    Xd = sparse.csr_matrix(F.records_to_dense(records, cat_map).astype(np.float32))
    X = sparse.hstack([Xw, Xd], format="csr")
    pred = clf.predict_proba(X).argmax(1)
    id2pred = {ids[i]: class_order[int(pred[i])] for i in range(len(ids))}

    # write in sample_submission.csv id order, columns id,action
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
