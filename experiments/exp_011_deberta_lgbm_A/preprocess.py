"""Preprocess train.jsonl → structured dataset for exp_011.

Creates:
  data/train_processed.parquet  — all features + labels
  data/test_processed.parquet   — same features, no labels
"""
import os, json, re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "..", "data")
OUT = os.path.join(ROOT, "data")
os.makedirs(OUT, exist_ok=True)

CLASS_ORDER = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]

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
RULE_NAMES = [n for n, _ in RULE_CATEGORIES]
NONE = "NONE"


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def extract(samples, label_map=None):
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

        # DeBERTa input text (Exp A: last_action prefix)
        deberta_text = f"{last} [SEP] {prompt}"

        session_id = sid.rsplit("-step", 1)[0] if "-step" in sid else sid

        row = {
            "id": sid,
            "session_id": session_id,
            "deberta_text": deberta_text,
            "prompt": prompt,
            # MI ≥ 0.18 features only
            "last_action": last,
            "second_last_action": second,
            "action_bigram": f"{second}__{last}",
            "action_trigram": f"{third}__{second}__{last}",
            "last_action_status": f"{last}__{'FAIL' if last_failed else 'OK'}",
            "turn_action": f"t{min(turn_index, 7)}_{last}",
            "prompt_intent": intent,
        }

        if label_map is not None:
            row["label"] = label_map.get(sid, "UNKNOWN")
            row["label_id"] = CLASS_ORDER.index(row["label"]) if row["label"] in CLASS_ORDER else -1

        rows.append(row)
    return pd.DataFrame(rows)


def main():
    print("Loading train.jsonl ...")
    train_samples = load_jsonl(os.path.join(DATA, "train.jsonl"))
    label_map = {}
    with open(os.path.join(DATA, "train_labels.csv")) as f:
        next(f)
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split(",", 1)
                label_map[k] = v

    train_df = extract(train_samples, label_map)
    train_df.to_parquet(os.path.join(OUT, "train_processed.parquet"), index=False)
    print(f"Saved train_processed.parquet: {train_df.shape}")
    print(f"  Columns: {list(train_df.columns)}")
    print(f"  Label distribution:\n{train_df['label'].value_counts().to_string()}")

    print("\nLoading test.jsonl ...")
    test_samples = load_jsonl(os.path.join(DATA, "test.jsonl"))
    test_df = extract(test_samples, label_map=None)
    test_df.to_parquet(os.path.join(OUT, "test_processed.parquet"), index=False)
    print(f"Saved test_processed.parquet: {test_df.shape}")

    json.dump(CLASS_ORDER, open(os.path.join(OUT, "class_order.json"), "w"))
    print("Done.")


if __name__ == "__main__":
    main()
