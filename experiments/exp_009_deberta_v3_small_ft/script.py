#!/usr/bin/env python3
"""exp_009_deberta_v3_small_ft - DACON server inference (SUBMITTED).

OFFLINE only. Loads the fine-tuned deberta-v3-small + tokenizer from the local
model/ directory (no hub), reads data/test.jsonl + data/sample_submission.csv,
builds the SAME serialized structured input as train.py, batch-infers on
GPU-if-available else CPU, writes output/submission.csv (id, action=class STRING).
"""

import os

# Hard offline switches (belt and suspenders).
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import csv
import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import features as F

MODEL_DIR = "model"
DATA_DIR = "data"
TEST_PATH = os.path.join(DATA_DIR, "test.jsonl")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUTPUT_PATH = os.path.join("output", "submission.csv")
BATCH_SIZE = 64


def load_artifacts():
    local_dir = os.path.join(".", MODEL_DIR)
    with open(os.path.join(MODEL_DIR, "label_map.json")) as f:
        meta = json.load(f)
    tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(local_dir, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.half().to(device)
    else:
        model = model.float().to(device)
    model.eval()
    return {"tok": tok, "model": model, "device": device,
            "class_order": meta["class_order"], "max_length": int(meta["max_length"])}


def collate(batch, pad_id):
    maxlen = max(len(x) for x in batch)
    input_ids, attn = [], []
    for ids in batch:
        pad = maxlen - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long))


@torch.no_grad()
def predict(texts, art):
    tok, model, device = art["tok"], art["model"], art["device"]
    enc = tok(texts, truncation=True, max_length=art["max_length"])["input_ids"]
    preds = []
    for i in range(0, len(enc), BATCH_SIZE):
        batch = enc[i:i + BATCH_SIZE]
        ids, am = collate(batch, tok.pad_token_id)
        ids, am = ids.to(device), am.to(device)
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=am).logits
        else:
            logits = model(input_ids=ids, attention_mask=am).logits
        preds.append(logits.float().argmax(-1).cpu().numpy())
    return np.concatenate(preds, axis=0)


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
    print(f" device={art['device']}")

    print("Loading test data ...")
    samples = F.load_jsonl(TEST_PATH)
    ids, texts = F.build_texts(samples)
    print(f" test samples = {len(ids)}")

    print("Predicting ...")
    pred_codes = predict(texts, art)
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
