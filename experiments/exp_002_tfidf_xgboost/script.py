#!/usr/bin/env python3
"""exp_002_tfidf_xgboost — DACON server inference (SUBMITTED).

Offline only. Loads fitted artifacts from model/, reads data/test.jsonl +
data/sample_submission.csv, writes output/submission.csv with columns id,action
(action = predicted class STRING). Preserves sample_submission id order.
"""

import csv
import json
import os

import numpy as np
import joblib

import features as F

MODEL_DIR = "model"
DATA_DIR = "data"
TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_PATH = os.path.join("output", "submission.csv")


def load_artifacts():
    word_vec = joblib.load(os.path.join(MODEL_DIR, "word_vec.pkl"))
    char_vec = joblib.load(os.path.join(MODEL_DIR, "char_vec.pkl"))
    with open(os.path.join(MODEL_DIR, "cat_mappings.json")) as f:
        cat_mappings = json.load(f)
    with open(os.path.join(MODEL_DIR, "class_order.json")) as f:
        class_order = json.load(f)
    model = joblib.load(os.path.join(MODEL_DIR, "xgb.pkl"))
    return {
        "word_vec": word_vec,
        "char_vec": char_vec,
        "cat_mappings": cat_mappings,
        "class_order": class_order,
        "model": model,
    }


def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or fieldnames[:2] != ["id", "action"]:
        raise ValueError(f"sample_submission columns not (id, action): {fieldnames}")
    return fieldnames, rows


def write_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("Loading artifacts ...")
    art = load_artifacts()
    class_order = art["class_order"]

    print("Loading test data ...")
    samples = F.load_jsonl(TEST_PATH)
    ids, prompts, records = F.build_records(samples)
    print(f" test samples = {len(ids)}")

    print("Building features + predicting ...")
    X = F.transform_all(prompts, records, art)
    proba = art["model"].predict_proba(X)
    pred_codes = proba.argmax(1)
    pred_labels = [class_order[int(c)] for c in pred_codes]
    pred_map = dict(zip(ids, pred_labels))

    fieldnames, sub_rows = load_sample_submission(SAMPLE_SUB_PATH)
    n_missing = 0
    for row in sub_rows:
        p = pred_map.get(row["id"])
        if p is None:
            n_missing += 1
        else:
            row["action"] = p
    if n_missing:
        print(f" WARN: {n_missing} ids had no prediction (kept placeholder)")

    write_submission(OUTPUT_PATH, fieldnames, sub_rows)
    print(f"Saved {OUTPUT_PATH} (rows={len(sub_rows)})")


if __name__ == "__main__":
    main()
