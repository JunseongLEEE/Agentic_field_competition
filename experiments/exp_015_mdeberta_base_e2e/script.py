"""Inference for DACON submission — exp_015 end-to-end mDeBERTa.

OFFLINE ONLY — loads model from model/ directory.
"""
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import features as F
from model import ActionClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def load_model(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    class_order = json.load(open(os.path.join(model_dir, "class_order.json")))
    tok_path = os.path.join(model_dir, "tokenizer")
    backbone_path = os.path.join(model_dir, "backbone")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)

    model = ActionClassifier(
        backbone_path, cfg["n_classes"], cfg["dropout"], local_files_only=True
    )
    head = torch.load(os.path.join(model_dir, "head.pt"), map_location=DEVICE)
    model.classifier.load_state_dict(head)
    model.to(DEVICE).eval()
    return model, tokenizer, class_order, cfg


def main():
    model_dir = os.path.join(ROOT, "model")
    data_dir = os.path.join(ROOT, "data")
    out_dir = os.path.join(ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)

    model, tokenizer, class_order, cfg = load_model(model_dir)
    max_len = cfg["max_length"]
    batch_size = cfg.get("inference_batch_size", 64)

    build_kw = {
        "user_content_max": 80,
        "history_window": 8,
        "arg_hint_max": 40,
    }
    samples = F.load_jsonl(os.path.join(data_dir, "test.jsonl"))
    ids, texts, _, _ = F.build_dataset(samples, label_map=None, **build_kw)

    ds = EncDataset(texts, tokenizer, max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    all_preds = []
    use_amp = cfg.get("use_fp16") and DEVICE.type == "cuda"
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(input_ids, attention_mask)
            else:
                logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    actions = [class_order[p] for p in all_preds]
    pd.DataFrame({"id": ids, "action": actions}).to_csv(
        os.path.join(out_dir, "submission.csv"), index=False
    )
    print(f"Wrote {len(ids)} predictions to output/submission.csv")


if __name__ == "__main__":
    main()
