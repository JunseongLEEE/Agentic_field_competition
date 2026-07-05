"""공통 유틸: 데이터 로드, 정답 파싱, 평가, 검증 분할."""
import ast
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "competition_data")
EXP = os.path.join(ROOT, "experiments")
SUB = os.path.join(ROOT, "submissions")
VAL_IDS_PATH = os.path.join(EXP, "val_ids.json")


def load_train() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    df["answer_list"] = df["Answer"].apply(ast.literal_eval)
    return df


def load_test() -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, "test.csv"), encoding="utf-8-sig")


def image_paths(row, split: str):
    """row의 4개 이미지 절대경로 (Input_1..4 순서)."""
    return [
        os.path.join(DATA, split, row["Id"], row[f"Input_{i}"]) for i in range(1, 5)
    ]


def answer_to_str(perm) -> str:
    return "[" + ", ".join(str(int(x)) for x in perm) + "]"


def exact_match(preds, golds) -> float:
    assert len(preds) == len(golds)
    hit = sum(1 for p, g in zip(preds, golds) if list(p) == list(g))
    return hit / len(preds)


def train_val_split(df: pd.DataFrame, val_ratio: float = 0.1, seed: int = 42):
    """Answer 순열 stratified 분할. val_ids는 파일로 고정해 전 실험에서 재사용."""
    if os.path.exists(VAL_IDS_PATH):
        val_ids = set(json.load(open(VAL_IDS_PATH)))
    else:
        rng_df = df.sample(frac=1.0, random_state=seed)
        val_ids = set(
            rng_df.groupby("Answer", group_keys=False)
            .apply(lambda g: g.head(max(1, int(round(len(g) * val_ratio)))))["Id"]
        )
        os.makedirs(EXP, exist_ok=True)
        json.dump(sorted(val_ids), open(VAL_IDS_PATH, "w"))
    tr = df[~df["Id"].isin(val_ids)].reset_index(drop=True)
    va = df[df["Id"].isin(val_ids)].reset_index(drop=True)
    return tr, va


def make_submission(pred_by_id: dict, out_path: str):
    """pred_by_id: {Id: [n,n,n,n]} → sample_submission 순서로 저장."""
    ss = pd.read_csv(os.path.join(DATA, "sample_submission.csv"), encoding="utf-8-sig")
    ss["Answer"] = ss["Id"].map(lambda i: answer_to_str(pred_by_id[i]))
    assert ss["Answer"].notna().all(), "누락된 Id 존재"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ss.to_csv(out_path, index=False)
    return out_path
