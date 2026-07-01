"""exp_011 — 2-Stage DeBERTa→LightGBM (Exp A: last_action + prompt).

Stage 1: Fine-tune DeBERTa-v3-small on "[last_action] [SEP] prompt" → 14-class.
         GroupKFold OOF predictions saved as 14-dim probability vectors.
Stage 2: LightGBM on OOF probabilities + 7 structural features (MI≥0.18).
"""
import os, sys, json, time, gc
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "16")

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, "deberta"), exist_ok=True)

SEED = 42
NFOLDS = 5
NUM_CLASSES = 14
MODEL_NAME = "microsoft/deberta-v3-small"
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 5
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAT_FEATURES = [
    "action_bigram", "action_trigram", "turn_action",
    "last_action_status", "last_action", "second_last_action",
    "prompt_intent",
]


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
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
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_deberta_fold(train_texts, train_labels, val_texts, val_labels,
                       tokenizer, class_weights, fold_idx):
    train_ds = TextDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_ds = TextDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                        num_workers=4, pin_memory=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw_tensor)

    best_f1, best_state = 0, None
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
                all_labels.append(batch["labels"].numpy())

        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        preds = all_probs.argmax(1)
        f1 = f1_score(all_labels, preds, average="macro")

        avg_loss = total_loss / len(train_dl)
        print(f"  fold{fold_idx} epoch{epoch} loss={avg_loss:.4f} val_macroF1={f1:.4f}", flush=True)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in val_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    val_probs = np.concatenate(all_probs)
    model.save_pretrained(os.path.join(MODELS_DIR, f"deberta_f{fold_idx}"))
    tokenizer.save_pretrained(os.path.join(MODELS_DIR, f"deberta_f{fold_idx}"))

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return val_probs, best_f1


def build_lgbm_features(df, oof_probs, cat_mappings):
    rows = []
    for i in range(len(df)):
        row = list(oof_probs[i])
        for feat in CAT_FEATURES:
            val = str(df.iloc[i][feat])
            mapping = cat_mappings[feat]
            row.append(mapping.get(val, 0))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


def lgbm_feature_names():
    names = [f"deberta_prob_{i}" for i in range(NUM_CLASSES)]
    names += CAT_FEATURES
    return names


def main():
    seed_everything(SEED)
    t0 = time.time()

    if not os.path.exists(os.path.join(DATA_DIR, "train_processed.parquet")):
        print("Running preprocess.py first...", flush=True)
        os.system(f"python {os.path.join(ROOT, 'preprocess.py')}")

    df = pd.read_parquet(os.path.join(DATA_DIR, "train_processed.parquet"))
    class_order = json.load(open(os.path.join(DATA_DIR, "class_order.json")))
    print(f"Loaded {len(df)} samples", flush=True)

    texts = df["deberta_text"].values
    labels = df["label_id"].values
    groups = df["session_id"].values

    # Class weights
    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=labels)
    print(f"Class weights: {[f'{w:.2f}' for w in cw]}", flush=True)

    # ═══════════════════════════════════════════
    # STAGE 1: DeBERTa OOF
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("STAGE 1: DeBERTa-v3-small fine-tuning", flush=True)
    print(f"{'='*60}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    oof_probs = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    deberta_fold_scores = []

    skf = StratifiedGroupKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, labels, groups)):
        assert not (set(groups[tr_idx]) & set(groups[va_idx])), "group leak!"
        print(f"\n--- DeBERTa Fold {fold} (train={len(tr_idx)}, val={len(va_idx)}) ---", flush=True)

        val_probs, best_f1 = train_deberta_fold(
            texts[tr_idx].tolist(), labels[tr_idx],
            texts[va_idx].tolist(), labels[va_idx],
            tokenizer, cw, fold,
        )
        oof_probs[va_idx] = val_probs
        deberta_fold_scores.append(best_f1)
        print(f"  fold{fold} best macroF1={best_f1:.4f} ({time.time()-t0:.0f}s)", flush=True)

    deberta_oof_f1 = f1_score(labels, oof_probs.argmax(1), average="macro")
    deberta_per_class = f1_score(labels, oof_probs.argmax(1), average=None, labels=list(range(NUM_CLASSES)))
    print(f"\nDeBERTa OOF macroF1 = {deberta_oof_f1:.4f}", flush=True)
    for i, c in enumerate(class_order):
        print(f"  {c:20s} F1={deberta_per_class[i]:.4f}", flush=True)

    np.save(os.path.join(ROOT, "deberta_oof_probs.npy"), oof_probs)

    # ═══════════════════════════════════════════
    # STAGE 2: LightGBM on OOF probs + structural features
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("STAGE 2: LightGBM on DeBERTa OOF + 7 features", flush=True)
    print(f"{'='*60}", flush=True)

    cat_mappings = {}
    for feat in CAT_FEATURES:
        vals = sorted(df[feat].astype(str).unique())
        m = {"__UNK__": 0}
        for v in vals:
            if v not in m:
                m[v] = len(m)
        cat_mappings[feat] = m

    X_lgbm = build_lgbm_features(df, oof_probs, cat_mappings)
    feat_names = lgbm_feature_names()
    print(f"LightGBM features: {X_lgbm.shape[1]} ({feat_names})", flush=True)

    oof_lgbm = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)
    lgbm_fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(texts, labels, groups)):
        print(f"  LightGBM fold {fold} ...", flush=True)
        sw = np.ones(len(tr_idx), dtype=np.float32)
        for i, idx in enumerate(tr_idx):
            sw[i] = cw[labels[idx]]

        m = lgb.LGBMClassifier(
            objective="multiclass", num_class=NUM_CLASSES,
            n_estimators=300, learning_rate=0.08,
            num_leaves=63, min_child_samples=30,
            subsample=0.8, colsample_bytree=0.7,
            reg_lambda=1.0, n_jobs=16, seed=SEED, verbose=-1,
        )
        m.fit(X_lgbm[tr_idx], labels[tr_idx], sample_weight=sw)
        p = m.predict_proba(X_lgbm[va_idx])
        oof_lgbm[va_idx] = p
        sc = f1_score(labels[va_idx], p.argmax(1), average="macro")
        lgbm_fold_scores.append(sc)
        print(f"  fold{fold} macroF1={sc:.4f}", flush=True)
        joblib.dump(m, os.path.join(MODELS_DIR, f"lgbm_f{fold}.pkl"))

    lgbm_oof_f1 = f1_score(labels, oof_lgbm.argmax(1), average="macro")
    lgbm_per_class = f1_score(labels, oof_lgbm.argmax(1), average=None, labels=list(range(NUM_CLASSES)))
    print(f"\nFinal 2-Stage OOF macroF1 = {lgbm_oof_f1:.4f}", flush=True)
    for i, c in enumerate(class_order):
        delta = lgbm_per_class[i] - deberta_per_class[i]
        print(f"  {c:20s} DeBERTa={deberta_per_class[i]:.4f} → Final={lgbm_per_class[i]:.4f} ({delta:+.4f})", flush=True)

    np.save(os.path.join(ROOT, "oof_preds.npy"), oof_lgbm)

    # ═══════════════════════════════════════════
    # Full-train models for submission
    # ═══════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print("FULL-TRAIN for submission", flush=True)
    print(f"{'='*60}", flush=True)

    # DeBERTa full-train
    print("Training DeBERTa on full data...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES
    ).to(DEVICE)

    full_ds = TextDataset(texts.tolist(), labels, tokenizer, MAX_LEN)
    full_dl = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True)

    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=cw_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(full_dl) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in full_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            lab = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, lab)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"  full-train epoch{epoch} loss={total_loss/len(full_dl):.4f}", flush=True)

    model.save_pretrained(os.path.join(MODEL_DIR, "deberta"))
    tokenizer.save_pretrained(os.path.join(MODEL_DIR, "deberta"))
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    # LightGBM full-train (using full DeBERTa predictions as features)
    print("Getting full-train DeBERTa predictions...", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(MODEL_DIR, "deberta")
    ).to(DEVICE).eval()

    all_probs = []
    infer_dl = DataLoader(
        TextDataset(texts.tolist(), None, tokenizer, MAX_LEN),
        batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True,
    )
    with torch.no_grad():
        for batch in infer_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    full_probs = np.concatenate(all_probs)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    X_full = build_lgbm_features(df, full_probs, cat_mappings)
    sw_full = np.array([cw[l] for l in labels], dtype=np.float32)
    lgbm_full = lgb.LGBMClassifier(
        objective="multiclass", num_class=NUM_CLASSES,
        n_estimators=300, learning_rate=0.08,
        num_leaves=63, min_child_samples=30,
        subsample=0.8, colsample_bytree=0.7,
        reg_lambda=1.0, n_jobs=16, seed=SEED, verbose=-1,
    )
    lgbm_full.fit(X_full, labels, sample_weight=sw_full)
    joblib.dump(lgbm_full, os.path.join(MODEL_DIR, "lgbm.pkl"))
    json.dump(cat_mappings, open(os.path.join(MODEL_DIR, "cat_mappings.json"), "w"))
    json.dump(class_order, open(os.path.join(MODEL_DIR, "class_order.json"), "w"))

    # Timing estimate
    n_test = 2000
    ts = time.time()
    _ = lgbm_full.predict_proba(X_full[:n_test])
    lgbm_ms = (time.time() - ts) / n_test * 1000
    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(MODEL_DIR) for f in fns
    ) / 1048576

    worst_cls = class_order[np.argmin(lgbm_per_class)]
    collapsed = [class_order[i] for i in range(NUM_CLASSES) if lgbm_per_class[i] < 0.05]

    log = {
        "experiment_id": "exp_011_deberta_lgbm_A",
        "metric": "macro_f1",
        "cv_strategy": f"StratifiedGroupKFold({NFOLDS},group=session,seed={SEED})",
        "stage1_model": MODEL_NAME,
        "stage1_input": "[last_action] [SEP] prompt",
        "stage1_cv_scores": [float(s) for s in deberta_fold_scores],
        "stage1_cv_mean": float(np.mean(deberta_fold_scores)),
        "stage2_model": "LightGBM",
        "stage2_n_features": int(X_lgbm.shape[1]),
        "cv_fold_scores": [float(s) for s in lgbm_fold_scores],
        "cv_mean": float(np.mean(lgbm_fold_scores)),
        "cv_std": float(np.std(lgbm_fold_scores)),
        "per_class_f1": {class_order[i]: float(lgbm_per_class[i]) for i in range(NUM_CLASSES)},
        "deberta_per_class_f1": {class_order[i]: float(deberta_per_class[i]) for i in range(NUM_CLASSES)},
        "worst_class": worst_cls,
        "collapsed_classes": collapsed,
        "model_size_mb": round(size_mb, 1),
        "inference_ms_per_sample": round(lgbm_ms, 3),
        "estimated_full_test_minutes": round((lgbm_ms * 30000 / 1000 / 60) + 5.0, 2),
        "offline_compatible": True,
        "seed": SEED,
    }
    json.dump(log, open(os.path.join(ROOT, "train_log.json"), "w"), indent=2)
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)
    print(f"  Stage1 DeBERTa OOF macroF1 = {deberta_oof_f1:.4f}", flush=True)
    print(f"  Stage2 Final OOF macroF1   = {lgbm_oof_f1:.4f}", flush=True)
    print(f"  Model size: {size_mb:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
