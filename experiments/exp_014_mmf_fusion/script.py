"""exp_014 OFFLINE inference. Reads data/test.jsonl + data/sample_submission.csv,
rebuilds the 3 fusion inputs, loads model/ LOCALLY, writes output/submission.csv.
"""
import os
import sys
import csv
import json

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "model")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# import packaged feature/model code from model/
sys.path.insert(0, MODEL)
import features as F           # noqa: E402
from model import MMFNet       # noqa: E402
from transformers import AutoTokenizer, AutoModel  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFER_BATCH = 64


def load_model():
    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    enc_dir = os.path.join(MODEL, "encoder")
    tokenizer = AutoTokenizer.from_pretrained(enc_dir)
    dtype = torch.float32  # fp32 everywhere at inference (avoids Float/Half mismatch in pooling)
    encoder = AutoModel.from_pretrained(enc_dir, torch_dtype=dtype)
    model = MMFNet(encoder, cfg["hidden_size"], cfg["cat_cards"], cfg["n_num"],
                   cfg["n_step_vocab"], cfg["max_seq_len"], n_classes=cfg["n_classes"])
    tower_state = torch.load(os.path.join(MODEL, "towers.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(tower_state, strict=False)
    # only encoder.* should be "missing" (loaded from from_pretrained already)
    assert all(m.startswith("encoder.") for m in missing), missing
    model.float()  # fp32 inference — consistent dtype across encoder + towers
    model.to(DEVICE).eval()
    cat_map = json.load(open(os.path.join(MODEL, "cat_mappings.json")))
    stats = np.load(os.path.join(MODEL, "num_stats.npz"))
    mean, std = stats["mean"], stats["std"]
    calib = np.load(os.path.join(MODEL, "calibration.npy"))
    class_order = json.load(open(os.path.join(MODEL, "class_order.json")))
    return model, tokenizer, cfg, cat_map, mean, std, calib, class_order


def build_inputs(samples, tokenizer, cfg):
    text_pairs = [F.extract_text_pair(s) for s in samples]
    seqs = [F.extract_seq(s) for s in samples]
    records = [F.extract_record(s) for s in samples]
    a = [p[0] for p in text_pairs]
    b = [p[1] for p in text_pairs]
    enc = tokenizer(a, b, truncation=True, max_length=cfg["max_text_len"],
                    padding=False)
    return enc["input_ids"], seqs, records


def collate(sl_ids, sl_seqs, cats, nums, pad_id, half):
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
    num_t = torch.from_numpy(nums)
    if half:
        num_t = num_t.half()
    return {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attn),
        "seq_toks": torch.from_numpy(st),
        "seq_roles": torch.from_numpy(sr),
        "seq_fails": torch.from_numpy(sf),
        "seq_mask": torch.from_numpy(sm),
        "cats": torch.from_numpy(cats.astype(np.int64)),
        "nums": num_t,
    }


@torch.no_grad()
def main():
    model, tokenizer, cfg, cat_map, mean, std, calib, class_order = load_model()
    pad_id = tokenizer.pad_token_id or 0
    half = False  # fp32 inference throughout (model is fp32)

    samples = F.load_jsonl(os.path.join(DATA, "test.jsonl"))
    ids = [str(s.get("id", "")) for s in samples]
    token_ids, seqs, records = build_inputs(samples, tokenizer, cfg)
    cats = F.records_to_cats(records, cat_map)
    nums_raw = F.records_to_nums(records)
    std_safe = std.copy()
    std_safe[std_safe < 1e-6] = 1.0
    nums = np.clip((nums_raw - mean) / std_safe, -10, 10).astype(np.float32)

    probs = np.zeros((len(samples), cfg["n_classes"]), np.float32)
    for i in range(0, len(samples), INFER_BATCH):
        j = min(i + INFER_BATCH, len(samples))
        batch = collate(token_ids[i:j], seqs[i:j], cats[i:j], nums[i:j],
                        pad_id, half)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        logits = model(batch["input_ids"], batch["attention_mask"],
                       batch["seq_toks"], batch["seq_roles"], batch["seq_fails"],
                       batch["seq_mask"], batch["cats"], batch["nums"])
        probs[i:j] = torch.softmax(logits.float(), -1).cpu().numpy()

    probs = probs * calib[None, :]
    pred_idx = probs.argmax(1)
    id2pred = {ids[i]: class_order[pred_idx[i]] for i in range(len(ids))}

    # write in sample_submission id order
    ss_path = os.path.join(DATA, "sample_submission.csv")
    order = []
    with open(ss_path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                order.append(row[0])
    if not order:
        order = ids
    default = class_order[0]
    with open(os.path.join(OUT, "submission.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "action"])
        for i in order:
            w.writerow([i, id2pred.get(i, default)])
    print(f"wrote {len(order)} rows to output/submission.csv", flush=True)


if __name__ == "__main__":
    main()
