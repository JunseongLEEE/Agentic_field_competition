#!/usr/bin/env python3
"""Local fine-tune + StratifiedGroupKFold CV for exp_009_deberta_v3_small_ft.

NEVER submitted. Produces:
  - oof_preds.npy (70000, 14) float32
  - model/  (fine-tuned deberta-v3-small saved fp16 + tokenizer + label_map)
  - train_log.json

Modeling Protocol (LOCKED): StratifiedGroupKFold(5, shuffle, random_state=42),
group = session id = id.rsplit("-step",1)[0]. Weighted CE for Macro-F1.
Structured features serialized into the input text (see features.build_input_text).
"""

import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn.functional as Fnn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import features as F

HERE = Path(__file__).resolve().parent


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=HERE
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None, idx=None):
        self.enc = encodings
        self.labels = labels
        self.idx = idx if idx is not None else np.arange(len(encodings["input_ids"]))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        item = {"input_ids": self.enc["input_ids"][j],
                "attention_mask": self.enc["attention_mask"][j]}
        if self.labels is not None:
            item["label"] = int(self.labels[j])
        return item


def make_collate(pad_id):
    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, attn = [], []
        for b in batch:
            ids = b["input_ids"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        out = {"input_ids": torch.tensor(input_ids, dtype=torch.long),
               "attention_mask": torch.tensor(attn, dtype=torch.long)}
        if "label" in batch[0]:
            out["labels"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return out
    return collate


def build_model(base, num_labels, device):
    m = AutoModelForSequenceClassification.from_pretrained(base, num_labels=num_labels)
    return m.to(device)


@torch.no_grad()
def predict_probs(model, dataset, collate, batch_size, device):
    model.eval()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=collate, num_workers=2)
    out = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits = model(input_ids=ids, attention_mask=am).logits
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_model(model, train_ds, collate, class_weights, cfg, device, val_ds=None, y_val=None):
    tp = cfg["train"]
    bs = tp["batch_size"]
    epochs = tp["epochs"]
    loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True,
                                         collate_fn=collate, num_workers=2, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tp["lr"]), weight_decay=float(tp["weight_decay"]))
    total_steps = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * tp["warmup_ratio"]), total_steps)
    w = torch.tensor(class_weights, dtype=torch.float32, device=device)

    best_f1, best_probs, best_epoch = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        for batch in loader:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = model(input_ids=ids, attention_mask=am).logits
                loss = Fnn.cross_entropy(logits.float(), labels, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        if val_ds is not None:
            probs = predict_probs(model, val_ds, collate, bs * 2, device)
            f1 = f1_score(y_val, probs.argmax(1), average="macro")
            print(f"    epoch {ep}: val macro_f1={f1:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if f1 > best_f1:
                best_f1, best_probs, best_epoch = f1, probs, ep
        else:
            print(f"    epoch {ep}: train done ({time.time()-t0:.0f}s)", flush=True)
    return best_f1, best_probs, best_epoch


def main():
    t0 = time.time()
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["cv"]["seed"]
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = cfg["model"]["num_class"]
    base = cfg["model"]["base"]
    maxlen = cfg["train"]["max_length"]
    class_order = F.CLASS_ORDER
    class_to_code = {c: i for i, c in enumerate(class_order)}
    print(f"device={device} base={base} maxlen={maxlen}")

    # ---- Load data ----
    data = cfg["data"]
    samples = F.load_jsonl(str((HERE / data["train_path"]).resolve()))
    labels_df = pd.read_csv((HERE / data["train_labels_path"]).resolve())
    label_map = dict(zip(labels_df[data["id_col"]], labels_df[data["target_col"]]))

    ids, texts = F.build_texts(samples)
    y_str = np.array([label_map[i] for i in ids])
    assert set(np.unique(y_str)).issubset(set(class_order)), "unknown label"
    y = np.array([class_to_code[s] for s in y_str], dtype=int)
    groups = np.array([i.rsplit("-step", 1)[0] for i in ids])
    print(f"Loaded {len(ids)} rows, {len(np.unique(groups))} sessions, {len(np.unique(y))} classes")

    # ---- Tokenize once ----
    tok = AutoTokenizer.from_pretrained(base)
    print("Tokenizing ...", flush=True)
    enc = tok(texts, truncation=True, max_length=maxlen)
    encodings = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    collate = make_collate(tok.pad_token_id)

    # ---- CV ----
    sgkf = StratifiedGroupKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    oof = np.zeros((len(ids), n_classes), dtype=np.float32)
    fold_scores, best_epochs = [], []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
        assert len(set(groups[tr_idx]) & set(groups[va_idx])) == 0, "group overlap!"
        ft0 = time.time()
        cw = compute_class_weight("balanced", classes=np.arange(n_classes), y=y[tr_idx])
        model = build_model(base, n_classes, device)
        tr_ds = TextDataset(encodings, y, tr_idx)
        va_ds = TextDataset(encodings, y, va_idx)
        best_f1, best_probs, best_ep = train_model(
            model, tr_ds, collate, cw, cfg, device, val_ds=va_ds, y_val=y[va_idx])
        oof[va_idx] = best_probs
        fold_scores.append(float(best_f1))
        best_epochs.append(best_ep)
        print(f"[fold {fold}] best macro_f1={best_f1:.4f} (epoch {best_ep}) "
              f"time={time.time()-ft0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    assert np.isfinite(oof).all(), "NaN/Inf in OOF"
    oof_macro = f1_score(y, oof.argmax(1), average="macro")
    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    print(f"CV macro-f1 (fold mean)={cv_mean:.4f} +/- {cv_std:.4f} | OOF-agg={oof_macro:.4f}")

    per_class = f1_score(y, oof.argmax(1), average=None, labels=list(range(n_classes)))
    per_class_f1 = {str(i): float(per_class[i]) for i in range(n_classes)}
    per_class_named = {class_order[i]: float(per_class[i]) for i in range(n_classes)}
    worst_i = int(np.argmin(per_class))
    best_i = int(np.argmax(per_class))
    collapsed = [i for i in range(n_classes) if per_class[i] < 0.05]
    np.save(HERE / "oof_preds.npy", oof)

    # ---- Full retrain on all data ----
    full_epochs = int(round(np.mean([e for e in best_epochs if e >= 0]))) + 1
    full_epochs = max(2, min(full_epochs, cfg["train"]["epochs"]))
    print(f"Retraining on full train for {full_epochs} epochs ...", flush=True)
    cfg_full = dict(cfg)
    cfg_full["train"] = dict(cfg["train"])
    cfg_full["train"]["epochs"] = full_epochs
    cw_full = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    model = build_model(base, n_classes, device)
    full_ds = TextDataset(encodings, y, np.arange(len(y)))
    train_model(model, full_ds, collate, cw_full, cfg_full, device, val_ds=None)

    # ---- Save model/ (fp16) + tokenizer + label map ----
    model_dir = HERE / "model"
    model_dir.mkdir(exist_ok=True)
    model.half()
    model.save_pretrained(model_dir, safe_serialization=True)
    tok.save_pretrained(model_dir)
    with open(model_dir / "label_map.json", "w") as f:
        json.dump({"class_order": class_order, "max_length": maxlen}, f)

    # ---- Inference timing on 1000-row sample (GPU) ----
    n_time = min(1000, len(ids))
    time_ds = TextDataset(encodings, None, np.arange(n_time))
    if device == "cuda":
        torch.cuda.synchronize()
    it0 = time.time()
    _ = predict_probs(model, time_ds, collate, 64, device)
    if device == "cuda":
        torch.cuda.synchronize()
    infer_total = time.time() - it0
    inference_ms_per_sample = (infer_total / n_time) * 1000.0
    estimated_full_test_minutes = (inference_ms_per_sample * 30000) / 1000.0 / 60.0

    model_size_mb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / (1024 * 1024)

    runtime = time.time() - t0
    log = {
        "experiment_id": cfg["experiment"]["id"],
        "metric": "macro_f1",
        "cv_strategy": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), group=session_id",
        "cv_fold_scores": fold_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "oof_macro_f1": float(oof_macro),
        "per_class_f1": per_class_f1,
        "per_class_f1_named": per_class_named,
        "worst_class": {"id": worst_i, "name": class_order[worst_i], "f1": float(per_class[worst_i])},
        "best_class": {"id": best_i, "name": class_order[best_i], "f1": float(per_class[best_i])},
        "collapsed_classes": collapsed,
        "full_train_epochs": full_epochs,
        "runtime_seconds_train": round(runtime, 1),
        "inference_ms_per_sample": round(inference_ms_per_sample, 3),
        "estimated_full_test_minutes": round(estimated_full_test_minutes, 3),
        "model_size_mb": round(model_size_mb, 2),
        "offline_compatible": True,
        "seed": seed,
        "git_commit": git_commit(),
    }
    with open(HERE / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(json.dumps({k: log[k] for k in [
        "cv_mean", "cv_std", "oof_macro_f1", "worst_class", "collapsed_classes",
        "inference_ms_per_sample", "estimated_full_test_minutes", "model_size_mb"
    ]}, indent=2))
    print(f"Done in {runtime:.1f}s")


if __name__ == "__main__":
    main()
