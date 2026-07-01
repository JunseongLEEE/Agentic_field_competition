"""exp_016 — OFFLINE geometric-blend ensemble of THREE trained models.

final_logit = 3.0*log(p_014) + 1.5*log(p_010) + 0.5*log(p_001)
prediction  = argmax over the 14 classes (shared CLASS_ORDER index space).

  p_014 = exp_014 mDeBERTa MMFNet RAW softmax (NO calibration), fp32 inference.
  p_010 = exp_010 5-fold bagged LightGBM (probabilities averaged over folds).
  p_001 = exp_001 single TF-IDF LightGBM.

The three features.py files share a name but differ in content, so each is loaded
as a SEPARATE module via importlib (feats14 / feats10 / feats01). Reads
data/test.jsonl + data/sample_submission.csv, writes output/submission.csv
(id,action) in sample_submission id order. stdlib csv/json only for I/O.
"""
import os
import sys
import csv
import json
import importlib.util

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

D14 = os.path.join(MODEL, "exp014")
D10 = os.path.join(MODEL, "exp010")
D01 = os.path.join(MODEL, "exp001")

# Blend weights (validated on group-holdout; do NOT re-tune).
W14, W10, W01 = 3.0, 1.5, 0.5
N_FOLDS_010 = 5
INFER_BATCH = 64
EPS = 1e-12


def _load_module(name, path):
    """Load a features.py under a unique module name so the three variants coexist."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_test():
    # any of the three load_jsonl impls is fine; use feats01's for I/O only.
    return None


# ---------------------------------------------------------------------------
# exp_014 — mDeBERTa MMFNet (fp32, RAW softmax, no calibration)
# ---------------------------------------------------------------------------

def probs_exp014(samples, feats14):
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(os.path.join(D14, "config.json")))
    enc_dir = os.path.join(D14, "encoder")

    # local load only (no hub)
    tokenizer = AutoTokenizer.from_pretrained(enc_dir)
    encoder = AutoModel.from_pretrained(enc_dir, torch_dtype=torch.float32)

    # import MMFNet from the packaged exp014 model.py
    mmf_mod = _load_module("mmf_model_exp014", os.path.join(D14, "model.py"))
    MMFNet = mmf_mod.MMFNet

    model = MMFNet(encoder, cfg["hidden_size"], cfg["cat_cards"], cfg["n_num"],
                   cfg["n_step_vocab"], cfg["max_seq_len"], n_classes=cfg["n_classes"])
    tower_state = torch.load(os.path.join(D14, "towers.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(tower_state, strict=False)
    assert all(m.startswith("encoder.") for m in missing), missing
    model.float()  # fp32 inference throughout
    model.to(device).eval()

    cat_map = json.load(open(os.path.join(D14, "cat_mappings.json")))
    stats = np.load(os.path.join(D14, "num_stats.npz"))
    mean, std = stats["mean"], stats["std"]
    std_safe = std.copy()
    std_safe[std_safe < 1e-6] = 1.0

    pad_id = tokenizer.pad_token_id or 0

    # build inputs
    text_pairs = [feats14.extract_text_pair(s) for s in samples]
    seqs = [feats14.extract_seq(s) for s in samples]
    records = [feats14.extract_record(s) for s in samples]
    a = [p[0] for p in text_pairs]
    b = [p[1] for p in text_pairs]
    enc = tokenizer(a, b, truncation=True, max_length=cfg["max_text_len"], padding=False)
    token_ids = enc["input_ids"]
    cats = feats14.records_to_cats(records, cat_map)
    nums_raw = feats14.records_to_nums(records)
    nums = np.clip((nums_raw - mean) / std_safe, -10, 10).astype(np.float32)

    def collate(sl_ids, sl_seqs, cts, nms):
        B = len(sl_ids)
        maxt = max(len(t) for t in sl_ids)
        maxs = max(len(s[0]) for s in sl_seqs)
        input_ids = np.full((B, maxt), pad_id, np.int64)
        attn = np.zeros((B, maxt), np.int64)
        st = np.zeros((B, maxs), np.int64)
        sr = np.zeros((B, maxs), np.int64)
        sf = np.zeros((B, maxs), np.int64)
        sm = np.zeros((B, maxs), np.int64)
        for i, t in enumerate(sl_ids):
            input_ids[i, :len(t)] = t
            attn[i, :len(t)] = 1
            toks, roles, fails = sl_seqs[i]
            L = len(toks)
            st[i, :L] = toks
            sr[i, :L] = roles
            sf[i, :L] = fails
            sm[i, :L] = 1
        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attn),
            "seq_toks": torch.from_numpy(st),
            "seq_roles": torch.from_numpy(sr),
            "seq_fails": torch.from_numpy(sf),
            "seq_mask": torch.from_numpy(sm),
            "cats": torch.from_numpy(cts.astype(np.int64)),
            "nums": torch.from_numpy(nms),
        }

    probs = np.zeros((len(samples), cfg["n_classes"]), np.float32)
    with torch.no_grad():
        for i in range(0, len(samples), INFER_BATCH):
            j = min(i + INFER_BATCH, len(samples))
            batch = collate(token_ids[i:j], seqs[i:j], cats[i:j], nums[i:j])
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"],
                           batch["seq_toks"], batch["seq_roles"], batch["seq_fails"],
                           batch["seq_mask"], batch["cats"], batch["nums"])
            probs[i:j] = torch.softmax(logits.float(), -1).cpu().numpy()
    return probs  # RAW softmax, no calibration


# ---------------------------------------------------------------------------
# exp_010 — 5-fold bagged LightGBM
# ---------------------------------------------------------------------------

def probs_exp010(samples, feats10):
    import joblib
    from scipy import sparse
    ids, prompts, records = feats10.build_records(samples)
    probs = None
    for k in range(N_FOLDS_010):
        wv = joblib.load(os.path.join(D10, f"word_vec_f{k}.pkl"))
        cat_map = json.load(open(os.path.join(D10, f"cat_mappings_f{k}.json")))
        m = joblib.load(os.path.join(D10, f"lgbm_f{k}.pkl"))
        Xw = wv.transform(prompts)
        Xd = sparse.csr_matrix(feats10.records_to_dense(records, cat_map).astype(np.float32))
        X = sparse.hstack([Xw, Xd], format="csr")
        p = m.predict_proba(X)
        probs = p if probs is None else probs + p
    probs = probs / N_FOLDS_010
    return np.asarray(probs, dtype=np.float32)


# ---------------------------------------------------------------------------
# exp_001 — single TF-IDF LightGBM
# ---------------------------------------------------------------------------

def probs_exp001(samples, feats01):
    import joblib
    from scipy import sparse
    ids, prompts, records = feats01.build_records(samples)
    wv = joblib.load(os.path.join(D01, "word_vec.pkl"))
    cat_map = json.load(open(os.path.join(D01, "cat_mappings.json")))
    clf = joblib.load(os.path.join(D01, "lgbm.pkl"))
    Xw = wv.transform(prompts)
    Xd = sparse.csr_matrix(feats01.records_to_dense(records, cat_map).astype(np.float32))
    X = sparse.hstack([Xw, Xd], format="csr")
    return np.asarray(clf.predict_proba(X), dtype=np.float32)


def main():
    # Load the three feature modules separately (same name, different content).
    feats14 = _load_module("feats14", os.path.join(D14, "features.py"))
    feats10 = _load_module("feats10", os.path.join(D10, "features.py"))
    feats01 = _load_module("feats01", os.path.join(D01, "features.py"))

    # Shared class order (verified identical across the three models).
    class_order = json.load(open(os.path.join(D14, "class_order.json")))
    co10 = json.load(open(os.path.join(D10, "class_order.json")))
    co01 = json.load(open(os.path.join(D01, "class_order.json")))
    assert class_order == co10 == co01, "CLASS_ORDER mismatch across models"

    samples = feats14.load_jsonl(os.path.join(DATA, "test.jsonl"))
    ids = [str(s.get("id", "")) for s in samples]

    p14 = probs_exp014(samples, feats14)
    p10 = probs_exp010(samples, feats10)
    p01 = probs_exp001(samples, feats01)

    assert p14.shape == p10.shape == p01.shape == (len(samples), len(class_order)), \
        (p14.shape, p10.shape, p01.shape)

    # Geometric blend in log space.
    final_logit = (W14 * np.log(p14 + EPS)
                   + W10 * np.log(p10 + EPS)
                   + W01 * np.log(p01 + EPS))
    pred_idx = final_logit.argmax(1)
    id2pred = {ids[i]: class_order[int(pred_idx[i])] for i in range(len(ids))}

    # Write in sample_submission.csv id order.
    ss_path = os.path.join(DATA, "sample_submission.csv")
    order = []
    if os.path.exists(ss_path):
        with open(ss_path, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if row:
                    order.append(row[0])
    if not order:
        order = ids
    default = class_order[0]
    with open(os.path.join(OUT, "submission.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "action"])
        for sid in order:
            w.writerow([sid, id2pred.get(sid, default)])
    print(f"wrote {len(order)} rows to output/submission.csv (blend 3/1.5/0.5, RAW exp_014)", flush=True)


if __name__ == "__main__":
    main()
