#!/usr/bin/env python3
"""exp_015 — End-to-end mDeBERTa-v3-base with rich context input.

Stage: single encoder, no LightGBM. StratifiedGroupKFold CV + full-train for submission.
"""
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

_logf = open(os.path.join(ROOT, "train.log"), "a", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, d):
        for s in self.streams:
            s.write(d)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = sys.stderr = _Tee(sys.__stdout__, _logf)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import features as F
from model import ActionClassifier, focal_ce

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class-balanced weights from prescription (minority up-weighted)
CLASS_WEIGHTS = {
    "web_search": 2.373,
    "write_file": 2.061,
    "lint_or_typecheck": 1.390,
    "plan_task": 1.207,
    "ask_user": 1.198,
    "list_directory": 0.807,
    "run_tests": 0.774,
    "apply_patch": 0.741,
    "run_bash": 0.713,
    "respond_only": 0.702,
    "glob_pattern": 0.691,
    "read_file": 0.470,
    "grep_search": 0.451,
    "edit_file": 0.422,
}


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


class EncDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def run_epoch(model, loader, optimizer, scheduler, cw_tensor, log_prior,
              tau, gamma, grad_accum, train=True, use_amp=False):
    model.train(train)
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids, attention_mask, log_prior=log_prior, tau=tau)
            loss = focal_ce(logits, labels, cw_tensor, gamma=gamma)
        loss = loss / grad_accum

        if train:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        if "labels" in batch:
            all_labels.append(batch["labels"].numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels) if all_labels else None
    return probs, labels


def train_fold(train_texts, train_labels, val_texts, val_labels,
               tokenizer, cfg, log_prior, cw_tensor, fold_idx):
    tr_cfg = cfg["training"]
    max_len = cfg["model"]["max_length"]
    bs = tr_cfg["batch_size"]
    grad_accum = tr_cfg["grad_accum"]
    epochs = tr_cfg["epochs"]
    lr = tr_cfg["lr"]
    gamma = tr_cfg["focal_gamma"]
    tau = tr_cfg["logit_adjust_tau"]

    train_ds = EncDataset(train_texts, train_labels, tokenizer, max_len)
    val_ds = EncDataset(val_texts, val_labels, tokenizer, max_len)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=tr_cfg["num_workers"], pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=tr_cfg["num_workers"], pin_memory=True)

    model = ActionClassifier(
        cfg["model"]["name"], cfg["model"]["n_classes"], cfg["model"]["dropout"]
    ).to(DEVICE)
    if tr_cfg.get("gradient_checkpointing"):
        model.backbone.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=tr_cfg["weight_decay"]
    )
    total_steps = (len(train_dl) // grad_accum + 1) * epochs
    warmup = int(total_steps * tr_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
    )

    best_f1, best_state = 0.0, None
    for epoch in range(epochs):
        loss = run_epoch(
            model, train_dl, optimizer, scheduler, cw_tensor, log_prior,
            tau, gamma, grad_accum, train=True, use_amp=tr_cfg.get("use_amp", False),
        )
        val_probs, val_y = predict_probs(model, val_dl)
        f1 = f1_score(val_y, val_probs.argmax(1), average="macro")
        print(f"  fold{fold_idx} epoch{epoch} loss={loss:.4f} val_macroF1={f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_probs, _ = predict_probs(model, val_dl)

    fold_dir = os.path.join(ROOT, "models", f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(fold_dir, "weights.pt"))
    tokenizer.save_pretrained(fold_dir)

    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    return val_probs, best_f1


def save_submission_model(model, tokenizer, cfg, class_order):
    out = os.path.join(ROOT, "model")
    os.makedirs(out, exist_ok=True)
    model.backbone.save_pretrained(os.path.join(out, "backbone"))
    tokenizer.save_pretrained(os.path.join(out, "tokenizer"))
    torch.save(model.classifier.state_dict(), os.path.join(out, "head.pt"))
    meta = {
        "n_classes": cfg["model"]["n_classes"],
        "dropout": cfg["model"]["dropout"],
        "max_length": cfg["model"]["max_length"],
        "use_fp16": cfg["inference"]["use_fp16"],
        "inference_batch_size": cfg["inference"]["batch_size"],
    }
    json.dump(meta, open(os.path.join(out, "model_config.json"), "w"), indent=2)
    json.dump(class_order, open(os.path.join(out, "class_order.json"), "w"))


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    seed_everything(cfg["seed"])

    data_cfg = cfg["data"]
    build_kw = {
        "user_content_max": data_cfg["user_content_max"],
        "history_window": data_cfg["history_window"],
        "arg_hint_max": data_cfg["arg_hint_max"],
    }

    print(f"Device: {DEVICE}", flush=True)
    print(f"Model: {cfg['model']['name']}", flush=True)

    samples = F.load_jsonl(os.path.join(ROOT, data_cfg["train_path"]))
    label_map = F.load_labels(os.path.join(ROOT, data_cfg["train_labels_path"]))
    ids, texts, labels, groups = F.build_dataset(samples, label_map, **build_kw)
    labels = np.array(labels, dtype=np.int64)
    groups = np.array(groups)

    valid = labels >= 0
    ids = [i for i, v in zip(ids, valid) if v]
    texts = [t for t, v in zip(texts, valid) if v]
    labels = labels[valid]
    groups = groups[valid]
    print(f"Loaded {len(texts)} samples", flush=True)

    # Logit adjustment prior
    counts = np.bincount(labels, minlength=len(F.CLASS_ORDER)).astype(np.float64)
    prior = counts / counts.sum()
    log_prior = torch.log(torch.tensor(prior + 1e-8, dtype=torch.float32)).to(DEVICE)

    cw = torch.tensor(
        [CLASS_WEIGHTS[c] for c in F.CLASS_ORDER], dtype=torch.float32
    ).to(DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    nfolds = cfg["cv"]["n_splits"]
    skf = StratifiedGroupKFold(
        n_splits=nfolds, shuffle=True, random_state=cfg["cv"]["seed"]
    )

    oof_probs = np.zeros((len(labels), len(F.CLASS_ORDER)), dtype=np.float32)
    fold_scores = []

    print(f"\n{'='*60}", flush=True)
    print("CV: StratifiedGroupKFold", flush=True)
    print(f"{'='*60}", flush=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, labels, groups)):
        assert not (set(groups[tr_idx]) & set(groups[va_idx])), "session leak!"
        print(f"\n--- Fold {fold} (train={len(tr_idx)}, val={len(va_idx)}) ---", flush=True)
        val_probs, best_f1 = train_fold(
            [texts[i] for i in tr_idx], labels[tr_idx].tolist(),
            [texts[i] for i in va_idx], labels[va_idx].tolist(),
            tokenizer, cfg, log_prior, cw, fold,
        )
        oof_probs[va_idx] = val_probs
        fold_scores.append(best_f1)
        print(f"  fold{fold} best macroF1={best_f1:.4f} ({time.time()-t0:.0f}s)", flush=True)

    oof_preds = oof_probs.argmax(1)
    cv_mean = f1_score(labels, oof_preds, average="macro")
    per_class = f1_score(labels, oof_preds, average=None, labels=list(range(len(F.CLASS_ORDER))))
    print(f"\nOOF macroF1 = {cv_mean:.4f} (±{np.std(fold_scores):.4f})", flush=True)
    for i, c in enumerate(F.CLASS_ORDER):
        print(f"  {c:20s} F1={per_class[i]:.4f}", flush=True)

    np.save(os.path.join(ROOT, "oof_preds.npy"), oof_probs)

    # Full-train for submission
    print(f"\n{'='*60}", flush=True)
    print("FULL-TRAIN for submission", flush=True)
    print(f"{'='*60}", flush=True)

    tr_cfg = cfg["training"]
    max_len = cfg["model"]["max_length"]
    full_ds = EncDataset(texts, labels.tolist(), tokenizer, max_len)
    full_dl = DataLoader(
        full_ds, batch_size=tr_cfg["batch_size"], shuffle=True,
        num_workers=tr_cfg["num_workers"], pin_memory=True,
    )

    model = ActionClassifier(
        cfg["model"]["name"], cfg["model"]["n_classes"], cfg["model"]["dropout"]
    ).to(DEVICE)
    if tr_cfg.get("gradient_checkpointing"):
        model.backbone.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"]
    )
    total_steps = (len(full_dl) // tr_cfg["grad_accum"] + 1) * tr_cfg["epochs"]
    warmup = int(total_steps * tr_cfg["warmup_ratio"])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup, num_training_steps=total_steps
    )

    for epoch in range(tr_cfg["epochs"]):
        loss = run_epoch(
            model, full_dl, optimizer, scheduler, cw, log_prior,
            tr_cfg["logit_adjust_tau"], tr_cfg["focal_gamma"],
            tr_cfg["grad_accum"], train=True, use_amp=tr_cfg.get("use_amp", False),
        )
        print(f"  full-train epoch{epoch} loss={loss:.4f}", flush=True)

    save_submission_model(model, tokenizer, cfg, F.CLASS_ORDER)

    # Inference timing (LightGBM-free end-to-end)
    infer_ds = EncDataset(texts[:1000], None, tokenizer, max_len)
    infer_dl = DataLoader(infer_ds, batch_size=cfg["inference"]["batch_size"],
                          shuffle=False, num_workers=2)
    model.eval()
    ts = time.time()
    with torch.no_grad():
        for batch in infer_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            _ = model(input_ids, attention_mask)
    infer_ms = (time.time() - ts) / 1000 * 1000

    size_mb = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(os.path.join(ROOT, "model"))
        for fn in fns
    ) / 1048576

    worst_i = int(np.argmin(per_class))
    collapsed = [F.CLASS_ORDER[i] for i in range(len(F.CLASS_ORDER)) if per_class[i] < 0.05]

    log = {
        "experiment_id": cfg["experiment_id"],
        "metric": "macro_f1",
        "cv_strategy": f"StratifiedGroupKFold({nfolds},group=session,seed={cfg['seed']})",
        "model": cfg["model"]["name"],
        "input_format": "[CTX] turn/arch/last/result/ci/dirty [HIST] recent turns [NOW] prompt",
        "max_length": max_len,
        "cv_fold_scores": [float(s) for s in fold_scores],
        "cv_mean": float(cv_mean),
        "cv_std": float(np.std(fold_scores)),
        "per_class_f1": {F.CLASS_ORDER[i]: float(per_class[i]) for i in range(len(F.CLASS_ORDER))},
        "worst_class": F.CLASS_ORDER[worst_i],
        "collapsed_classes": collapsed,
        "runtime_seconds_train": round(time.time() - t0, 1),
        "inference_ms_per_sample": round(infer_ms, 3),
        "estimated_full_test_minutes": round(infer_ms * 30000 / 1000 / 60, 2),
        "model_size_mb": round(size_mb, 1),
        "offline_compatible": True,
        "seed": cfg["seed"],
        "git_commit": git_commit(),
    }
    json.dump(log, open(os.path.join(ROOT, "train_log.json"), "w"), indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)
    print(f"  OOF macroF1 = {cv_mean:.4f}", flush=True)
    print(f"  Model size: {size_mb:.1f} MB", flush=True)
    print(f"  Infer ms/sample: {infer_ms:.2f}", flush=True)


if __name__ == "__main__":
    main()
