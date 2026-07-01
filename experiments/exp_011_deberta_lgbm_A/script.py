"""Inference for DACON submission — exp_011 DeBERTa→LightGBM (Exp A).

OFFLINE ONLY — loads all models from model/ directory.
Reads test data from data/, writes output/submission.csv.
"""
import os, sys, json, re
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = os.path.dirname(os.path.abspath(__file__))

NUM_CLASSES = 14
MAX_LEN = 128
BATCH_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NONE = "NONE"

RULE_CATEGORIES = [
    ("WRAP_UP", re.compile(r"(마무리|여기까지|이 정도면).*(요약|정리)|(wrap.?up|recap|summariz)", re.I)),
    ("ERROR_HELP", re.compile(r"(TypeError|AttributeError|ConnectionError|KeyError|AssertionError|Timeout|I keep hitting|계속 뜨는데)", re.I)),
    ("PLAN_REQ", re.compile(r"(단계.*(잡|짜|세워)|계획.*(잡|짜|세워)|lay.*out|before i (start|edit|touch)|plan (this|it|out))", re.I)),
    ("SHOW_FILE", re.compile(r"(보여줘|열어봐|열어줘|show me|open the|look at|pull up)", re.I)),
    ("SEARCH", re.compile(r"(어디|찾아|어느 파일|list what|where.*(is|are|does)|find|search for|grep)", re.I)),
    ("RUN_TEST", re.compile(r"(테스트.*돌|한번 돌려|돌려봐|run.*test|rerun|full suite|다시 빌드|build again)", re.I)),
    ("LINT_CHECK", re.compile(r"(lint|typecheck|타입체크|shellcheck|mypy|ruff|flake8)", re.I)),
    ("WEB_REF", re.compile(r"(best practice|공식.*문서|documentation|docs\b|look.*up online|web search)", re.I)),
]

CAT_FEATURES = [
    "action_bigram", "action_trigram", "turn_action",
    "last_action_status", "last_action", "second_last_action",
    "prompt_intent",
]


class TextDataset(Dataset):
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


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract_features(samples):
    rows = []
    for s in samples:
        sid = s.get("id", "")
        prompt = s.get("current_prompt", "") or ""
        history = s.get("history", []) or []
        actions = [h for h in history
                   if isinstance(h, dict) and h.get("role") == "assistant_action"]
        action_names = [a.get("name", NONE) for a in actions]

        last = action_names[-1] if len(action_names) >= 1 else NONE
        second = action_names[-2] if len(action_names) >= 2 else NONE
        third = action_names[-3] if len(action_names) >= 3 else NONE

        last_failed = 0
        if actions:
            rs = str(actions[-1].get("result_summary", ""))
            if "ERROR" in rs.upper() or "FAIL" in rs.upper():
                last_failed = 1

        meta = s.get("session_meta", {}) or {}
        turn_index = int(meta.get("turn_index", 0) or 0)

        active = [n for n, rgx in RULE_CATEGORIES if rgx.search(prompt)]
        intent = active[0] if active else "GENERAL"

        deberta_text = f"{last} [SEP] {prompt}"

        rows.append({
            "id": sid,
            "deberta_text": deberta_text,
            "last_action": last,
            "second_last_action": second,
            "action_bigram": f"{second}__{last}",
            "action_trigram": f"{third}__{second}__{last}",
            "last_action_status": f"{last}__{'FAIL' if last_failed else 'OK'}",
            "turn_action": f"t{min(turn_index, 7)}_{last}",
            "prompt_intent": intent,
        })
    return pd.DataFrame(rows)


def main():
    model_dir = os.path.join(ROOT, "model")
    data_dir = os.path.join(ROOT, "data")
    out_dir = os.path.join(ROOT, "output")
    os.makedirs(out_dir, exist_ok=True)

    class_order = json.load(open(os.path.join(model_dir, "class_order.json")))
    cat_mappings = json.load(open(os.path.join(model_dir, "cat_mappings.json")))
    lgbm_model = joblib.load(os.path.join(model_dir, "lgbm.pkl"))

    deberta_path = os.path.join(model_dir, "deberta")
    tokenizer = AutoTokenizer.from_pretrained(deberta_path)
    deberta_model = AutoModelForSequenceClassification.from_pretrained(deberta_path).to(DEVICE).eval()

    test_samples = load_jsonl(os.path.join(data_dir, "test.jsonl"))
    df = extract_features(test_samples)
    texts = df["deberta_text"].values.tolist()

    ds = TextDataset(texts, tokenizer, MAX_LEN)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    all_probs = []
    with torch.no_grad():
        for batch in dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = deberta_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    deberta_probs = np.concatenate(all_probs)

    rows = []
    for i in range(len(df)):
        row = list(deberta_probs[i])
        for feat in CAT_FEATURES:
            val = str(df.iloc[i][feat])
            mapping = cat_mappings[feat]
            row.append(mapping.get(val, 0))
        rows.append(row)
    X = np.array(rows, dtype=np.float32)

    probs = lgbm_model.predict_proba(X)
    preds = probs.argmax(axis=1)
    pred_labels = [class_order[p] for p in preds]

    out_df = pd.DataFrame({"id": df["id"].values, "action": pred_labels})
    out_df.to_csv(os.path.join(out_dir, "submission.csv"), index=False)
    print(f"Wrote {len(out_df)} predictions to output/submission.csv")


if __name__ == "__main__":
    main()
