"""exp_014 Multi-Modal Fusion Net — GPU training + 5-fold StratifiedGroupKFold OOF.

Towers: mdeberta text encoder + history-trajectory transformer + tabular embeddings.
Loss: logit-adjusted (class-balanced) softmax + label smoothing (targets Macro-F1).
Produces oof_preds.npy (70000,14), per-class OOF calibration, full-train model in model/.
"""
import os
import sys
import json
import time
import shutil

# ---- real-time line-buffered Tee to train.log -----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


_logf = open(os.path.join(ROOT, "train.log"), "a", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _logf)
sys.stderr = _Tee(sys.__stderr__, _logf)

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

sys.path.insert(0, ROOT)
import features as F
from model import MMFNet

DATA = os.path.join(ROOT, "..", "..", "data")
MODEL = os.path.join(ROOT, "model")
os.makedirs(MODEL, exist_ok=True)

SEED = 42
NFOLDS = 5
ENCODER_ID = "microsoft/mdeberta-v3-base"
EPOCHS = 3
BATCH = 32
INFER_BATCH = 64
LR_ENC = 2e-5
LR_HEAD = 1e-3
TAU = 1.0            # logit adjustment strength
LABEL_SMOOTH = 0.05
MAX_TEXT_LEN = F.MAX_TEXT_LEN
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
def load():
    samples = F.load_jsonl(os.path.join(DATA, "train.jsonl"))
    ids, prompts, records, text_pairs, seqs = F.build_multimodal(samples)
    id2a = {}
    with open(os.path.join(DATA, "train_labels.csv")) as f:
        next(f)
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split(",", 1)
                id2a[k] = v
    y = np.array([F.CLASS_ORDER.index(id2a[i]) for i in ids], dtype=np.int64)
    groups = np.array([i.rsplit("-step", 1)[0] for i in ids])
    return ids, prompts, records, text_pairs, seqs, y, groups


def tokenize_all(tokenizer, text_pairs):
    a = [p[0] for p in text_pairs]
    b = [p[1] for p in text_pairs]
    enc = tokenizer(a, b, truncation=True, max_length=MAX_TEXT_LEN, padding=False)
    return enc["input_ids"]  # list of lists


class MMDataset(Dataset):
    def __init__(self, idx, token_ids, seqs, cats, nums, y=None):
        self.idx = idx
        self.token_ids = token_ids
        self.seqs = seqs
        self.cats = cats
        self.nums = nums
        self.y = y

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        item = {
            "ids": self.token_ids[j],
            "seq": self.seqs[j],
            "cat": self.cats[j],
            "num": self.nums[j],
        }
        if self.y is not None:
            item["y"] = int(self.y[j])
        return item


PAD_ID = 0  # set from tokenizer at runtime


def collate(batch):
    maxt = max(len(b["ids"]) for b in batch)
    maxs = max(len(b["seq"][0]) for b in batch)
    B = len(batch)
    input_ids = np.full((B, maxt), PAD_ID, np.int64)
    attn = np.zeros((B, maxt), np.int64)
    st = np.zeros((B, maxs), np.int64)
    sr = np.zeros((B, maxs), np.int64)
    sf = np.zeros((B, maxs), np.int64)
    sm = np.zeros((B, maxs), np.int64)
    cats = np.stack([b["cat"] for b in batch]).astype(np.int64)
    nums = np.stack([b["num"] for b in batch]).astype(np.float32)
    for i, b in enumerate(batch):
        t = b["ids"]
        input_ids[i, :len(t)] = t
        attn[i, :len(t)] = 1
        toks, roles, fails = b["seq"]
        L = len(toks)
        st[i, :L] = toks
        sr[i, :L] = roles
        sf[i, :L] = fails
        sm[i, :L] = 1
    out = {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attn),
        "seq_toks": torch.from_numpy(st),
        "seq_roles": torch.from_numpy(sr),
        "seq_fails": torch.from_numpy(sf),
        "seq_mask": torch.from_numpy(sm),
        "cats": torch.from_numpy(cats),
        "nums": torch.from_numpy(nums),
    }
    if "y" in batch[0]:
        out["y"] = torch.tensor([b["y"] for b in batch], dtype=torch.long)
    return out


def build_model(cat_cards, n_num):
    encoder = AutoModel.from_pretrained(ENCODER_ID)
    hidden = encoder.config.hidden_size
    model = MMFNet(encoder, hidden, cat_cards, n_num,
                   F.N_STEP_VOCAB, F.MAX_SEQ_LEN, n_classes=14)
    return model


def make_optim(model, total_steps):
    enc_params, other_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if n.startswith("encoder.") else other_params).append(p)
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": LR_ENC, "weight_decay": 0.01},
        {"params": other_params, "lr": LR_HEAD, "weight_decay": 0.01},
    ])
    sched = get_linear_schedule_with_warmup(
        opt, int(0.06 * total_steps), total_steps)
    return opt, sched


def normalize_fit(nums):
    mean = nums.mean(0)
    std = nums.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_apply(nums, mean, std):
    z = (nums - mean) / std
    return np.clip(z, -10.0, 10.0).astype(np.float32)


def run_epoch(model, loader, opt, sched, scaler, log_prior, train=True):
    model.train(train)
    ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    tot = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            b = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(b["input_ids"], b["attention_mask"],
                               b["seq_toks"], b["seq_roles"], b["seq_fails"],
                               b["seq_mask"], b["cats"], b["nums"])
                if train:
                    adj = logits + TAU * log_prior  # logit adjustment
                    loss = ce(adj, b["y"])
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item() * b["input_ids"].size(0)
    return tot / max(1, len(loader.dataset))


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    out = []
    for batch in loader:
        b = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logits = model(b["input_ids"], b["attention_mask"],
                           b["seq_toks"], b["seq_roles"], b["seq_fails"],
                           b["seq_mask"], b["cats"], b["nums"])
        out.append(torch.softmax(logits.float(), -1).cpu().numpy())
    return np.concatenate(out, 0)


def calibrate(oof, y, n_classes=14):
    """Per-class prob multipliers by coordinate ascent to max OOF Macro-F1."""
    mult = np.ones(n_classes, np.float64)

    def score(m):
        return f1_score(y, (oof * m).argmax(1), average="macro")
    best = score(mult)
    grid = np.concatenate([np.linspace(0.5, 1.5, 21), np.linspace(1.6, 3.0, 8),
                           np.linspace(0.2, 0.45, 6)])
    for _ in range(4):
        improved = False
        for c in range(n_classes):
            base = mult[c]
            best_v, best_s = base, best
            for g in grid:
                mult[c] = g
                s = score(mult)
                if s > best_s + 1e-6:
                    best_s, best_v = s, g
            mult[c] = best_v
            if best_s > best + 1e-9:
                best = best_s
                improved = True
        if not improved:
            break
    return mult.astype(np.float32), float(best)


def main():
    t0 = time.time()
    global PAD_ID
    print(f"[exp_014] device={DEVICE} encoder={ENCODER_ID}", flush=True)
    ids, prompts, records, text_pairs, seqs, y, groups = load()
    print(f"loaded {len(ids)} rows, {len(set(groups))} sessions", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_ID)
    PAD_ID = tokenizer.pad_token_id or 0
    token_ids = tokenize_all(tokenizer, text_pairs)
    print(f"tokenized. pad_id={PAD_ID} "
          f"mean_len={np.mean([len(t) for t in token_ids]):.1f} "
          f"p95={np.percentile([len(t) for t in token_ids],95):.0f}", flush=True)

    n_num = F.num_feature_dim()
    skf = StratifiedGroupKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 14), np.float32)
    fold_scores = []
    best_epochs = []

    for k, (tr, va) in enumerate(skf.split(prompts, y, groups)):
        assert not (set(groups[tr]) & set(groups[va])), "group leak"
        # fold-specific tab encoding
        cat_map = F.build_cat_mappings([records[i] for i in tr])
        cat_cards = F.cat_cardinalities(cat_map)
        cats_all = F.records_to_cats(records, cat_map)
        nums_raw = F.records_to_nums(records)
        mean, std = normalize_fit(nums_raw[tr])
        nums_all = normalize_apply(nums_raw, mean, std)

        # log prior from train fold
        counts = np.bincount(y[tr], minlength=14).astype(np.float64)
        prior = counts / counts.sum()
        log_prior = torch.tensor(np.log(prior + 1e-12),
                                 dtype=torch.float32, device=DEVICE)

        tr_ds = MMDataset(tr, token_ids, seqs, cats_all, nums_all, y)
        va_ds = MMDataset(va, token_ids, seqs, cats_all, nums_all, y)
        tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                           collate_fn=collate, num_workers=4, pin_memory=True,
                           drop_last=False)
        va_ld = DataLoader(va_ds, batch_size=INFER_BATCH, shuffle=False,
                           collate_fn=collate, num_workers=4, pin_memory=True)

        model = build_model(cat_cards, n_num).to(DEVICE)
        total_steps = len(tr_ld) * EPOCHS
        opt, sched = make_optim(model, total_steps)
        scaler = torch.cuda.amp.GradScaler()

        best_f1, best_probs, best_ep, patience = -1.0, None, 0, 0
        for ep in range(EPOCHS):
            tl = run_epoch(model, tr_ld, opt, sched, scaler, log_prior, train=True)
            probs = predict_probs(model, va_ld)
            f1 = f1_score(y[va], probs.argmax(1), average="macro")
            print(f"fold{k} ep{ep} loss={tl:.4f} valF1={f1:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if f1 > best_f1 + 1e-4:
                best_f1, best_probs, best_ep, patience = f1, probs, ep, 0
            else:
                patience += 1
                if patience >= 1:
                    print(f"fold{k} early-stop after ep{ep}", flush=True)
                    break
        oof[va] = best_probs
        fold_scores.append(float(best_f1))
        best_epochs.append(best_ep + 1)
        print(f"fold{k} BEST valF1={best_f1:.4f} @ep{best_ep}", flush=True)
        del model, opt, sched, scaler
        torch.cuda.empty_cache()

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    raw_oof_f1 = f1_score(y, oof.argmax(1), average="macro")
    print(f"\nCV(best-epoch per fold) mean={cv_mean:.4f}+/-{cv_std:.4f}", flush=True)
    print(f"OOF raw Macro-F1 = {raw_oof_f1:.4f}", flush=True)

    mult, cal_f1 = calibrate(oof, y)
    print(f"OOF calibrated Macro-F1 = {cal_f1:.4f}", flush=True)
    print(f"calib multipliers = {np.round(mult,3).tolist()}", flush=True)

    per = f1_score(y, (oof * mult).argmax(1), average=None, labels=list(range(14)))
    per_class = {F.CLASS_ORDER[i]: float(per[i]) for i in range(14)}
    worst = min(per_class, key=per_class.get)
    collapsed = [c for c, v in per_class.items() if v < 0.05]
    print(f"per-class F1 (cal): {json.dumps({k: round(v,4) for k,v in per_class.items()})}",
          flush=True)

    np.save(os.path.join(ROOT, "oof_preds.npy"), oof)

    # ---- OOF correlation with exp_010 (fallback exp_001) GBDT ----
    corr = None
    corr_src = None
    for cand in ["exp_010_pca_engineered_lgbm", "exp_001_tfidf_lightgbm"]:
        p = os.path.join(ROOT, "..", cand, "oof_preds.npy")
        if os.path.exists(p):
            other = np.load(p)
            if other.shape == oof.shape:
                corr = float(np.corrcoef(oof.ravel(), other.ravel())[0, 1])
                corr_src = cand
            break
    if corr is not None:
        print(f"OOF corr with {corr_src} = {corr:.4f}", flush=True)

    # ---- Full-train final model for submission ----
    print("\nfull-train final model ...", flush=True)
    cat_map = F.build_cat_mappings(records)
    cat_cards = F.cat_cardinalities(cat_map)
    cats_all = F.records_to_cats(records, cat_map)
    nums_raw = F.records_to_nums(records)
    mean, std = normalize_fit(nums_raw)
    nums_all = normalize_apply(nums_raw, mean, std)
    counts = np.bincount(y, minlength=14).astype(np.float64)
    prior = counts / counts.sum()
    log_prior = torch.tensor(np.log(prior + 1e-12), dtype=torch.float32, device=DEVICE)

    full_idx = np.arange(len(y))
    ds = MMDataset(full_idx, token_ids, seqs, cats_all, nums_all, y)
    ld = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate,
                    num_workers=4, pin_memory=True)
    model = build_model(cat_cards, n_num).to(DEVICE)
    n_ep = int(round(np.mean(best_epochs)))
    n_ep = max(2, min(EPOCHS, n_ep))
    total_steps = len(ld) * n_ep
    opt, sched = make_optim(model, total_steps)
    scaler = torch.cuda.amp.GradScaler()
    for ep in range(n_ep):
        tl = run_epoch(model, ld, opt, sched, scaler, log_prior, train=True)
        print(f"full ep{ep} loss={tl:.4f} ({time.time()-t0:.0f}s)", flush=True)

    peak_mem = torch.cuda.max_memory_allocated() / 1048576 if DEVICE == "cuda" else 0.0

    # ---- inference timing (on this GPU) ----
    model.eval()
    n_time = min(2000, len(y))
    time_ds = MMDataset(np.arange(n_time), token_ids, seqs, cats_all, nums_all)
    time_ld = DataLoader(time_ds, batch_size=INFER_BATCH, shuffle=False,
                         collate_fn=collate, num_workers=4)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    ts = time.time()
    _ = predict_probs(model, time_ld)
    torch.cuda.synchronize() if DEVICE == "cuda" else None
    ms = (time.time() - ts) / n_time * 1000

    # ---- save model/ (fp16 encoder + towers + config + calibration) ----
    enc_dir = os.path.join(MODEL, "encoder")
    os.makedirs(enc_dir, exist_ok=True)
    model.encoder.half()
    model.encoder.save_pretrained(enc_dir, safe_serialization=True)
    tokenizer.save_pretrained(enc_dir)
    model.encoder.float()  # (no further use)

    tower_state = {k: v.half() for k, v in model.state_dict().items()
                   if not k.startswith("encoder.")}
    torch.save(tower_state, os.path.join(MODEL, "towers.pt"))
    json.dump(cat_map, open(os.path.join(MODEL, "cat_mappings.json"), "w"))
    np.savez(os.path.join(MODEL, "num_stats.npz"), mean=mean, std=std)
    np.save(os.path.join(MODEL, "calibration.npy"), mult)
    json.dump(F.CLASS_ORDER, open(os.path.join(MODEL, "class_order.json"), "w"))
    json.dump({
        "encoder_id": ENCODER_ID,
        "hidden_size": int(model.encoder.config.hidden_size),
        "cat_cards": cat_cards,
        "n_num": int(n_num),
        "n_step_vocab": int(F.N_STEP_VOCAB),
        "max_seq_len": int(F.MAX_SEQ_LEN),
        "max_text_len": int(MAX_TEXT_LEN),
        "n_classes": 14,
    }, open(os.path.join(MODEL, "config.json"), "w"), indent=2)

    # package feature + model code for offline inference
    shutil.copy(os.path.join(ROOT, "features.py"), os.path.join(MODEL, "features.py"))
    shutil.copy(os.path.join(ROOT, "model.py"), os.path.join(MODEL, "model.py"))

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(MODEL) for f in fs
    ) / 1048576

    est_t4_min = ms * 30000 / 1000 / 60 * 2.5  # ~2.5x slower on T4 (conservative)

    log = {
        "experiment_id": "exp_014_mmf_fusion",
        "metric": "macro_f1",
        "cv_strategy": f"StratifiedGroupKFold({NFOLDS},group=session,seed={SEED})",
        "cv_fold_scores": fold_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_mean_calibrated": cal_f1,
        "oof_raw_macro_f1": float(raw_oof_f1),
        "per_class_f1": per_class,
        "worst_class": worst,
        "collapsed_classes": collapsed,
        "oof_corr_gbdt": corr,
        "oof_corr_source": corr_src,
        "best_epochs_per_fold": best_epochs,
        "full_train_epochs": n_ep,
        "inference_ms_per_sample": round(ms, 3),
        "estimated_full_test_minutes": round(ms * 30000 / 1000 / 60, 2),
        "estimated_full_test_minutes_t4": round(est_t4_min, 2),
        "gpu_peak_mem_mb": round(peak_mem, 1),
        "model_size_mb": round(size_mb, 1),
        "offline_compatible": True,
        "seed": SEED,
        "encoder_id": ENCODER_ID,
    }
    json.dump(log, open(os.path.join(ROOT, "train_log.json"), "w"), indent=2)
    print(f"\nDONE oof_raw={raw_oof_f1:.4f} oof_cal={cal_f1:.4f} "
          f"infer={ms:.2f}ms/sample model={size_mb:.0f}MB "
          f"peak_mem={peak_mem:.0f}MB corr={corr}", flush=True)


if __name__ == "__main__":
    main()
