"""Inference script for DACON submission — exp_010_pca_engineered_lgbm.

Reads test data from data/, loads model from model/, writes output/submission.csv.
OFFLINE ONLY — no network calls, no from_pretrained.
"""
import os, json
import numpy as np
import pandas as pd
import joblib
from scipy import sparse

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, ROOT)
import features as F


def main():
    model_dir = os.path.join(ROOT, "model")
    data_dir = os.path.join(ROOT, "data")
    out_dir = os.path.join(ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)

    wv = joblib.load(os.path.join(model_dir, "word_vec.pkl"))
    cat_map = json.load(open(os.path.join(model_dir, "cat_mappings.json")))
    m = joblib.load(os.path.join(model_dir, "lgbm.pkl"))
    class_order = json.load(open(os.path.join(model_dir, "class_order.json")))

    samples = F.load_jsonl(os.path.join(data_dir, "test.jsonl"))
    ids, prompts, records = F.build_records(samples)

    Xw = wv.transform(prompts)
    Xd = sparse.csr_matrix(F.records_to_dense(records, cat_map).astype(np.float32))
    X = sparse.hstack([Xw, Xd], format="csr")

    probs = m.predict_proba(X)
    preds = probs.argmax(axis=1)
    pred_labels = [class_order[p] for p in preds]

    df = pd.DataFrame({"id": ids, "action": pred_labels})
    df.to_csv(os.path.join(out_dir, "submission.csv"), index=False)
    print(f"Wrote {len(df)} predictions to output/submission.csv")


if __name__ == "__main__":
    main()
