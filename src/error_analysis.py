"""val 오답 분석: 난이도 버킷/No_ordering/near-dup 축 분해 + CoT 게이트 판정.

사용: python src/error_analysis.py --preds experiments/preds_val_qwen_r16_ep1.json --out experiments/error_analysis_qwen_r16_ep2.md
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP, load_train, train_val_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    preds = json.load(open(args.preds))
    df = load_train()
    _, va = train_val_split(df)
    va = va[va["Id"].isin(preds)].reset_index(drop=True)

    # SigLIP 유사도 축
    z = np.load(os.path.join(EXP, "siglip_train.npz"), allow_pickle=True)
    idx = {i: k for k, i in enumerate(z["ids"])}
    img = z["img"].astype(np.float32)
    iu = np.triu_indices(4, 1)
    sim_feats = []
    for i in va["Id"]:
        s = np.einsum("id,jd->ij", img[idx[i]], img[idx[i]])[iu]
        sim_feats.append((s.mean(), s.max()))
    va["sim_mean"] = [a for a, _ in sim_feats]
    va["sim_max"] = [b for _, b in sim_feats]
    va["has_neardup_pair"] = va["sim_max"] > 0.95
    va["quartile"] = pd.qcut(va["sim_mean"], 4, labels=["Q1(쉬움)", "Q2", "Q3", "Q4(near-dup)"])

    va["pred"] = va["Id"].map(preds)
    va["correct"] = [list(p) == list(g) for p, g in zip(va["pred"], va["answer_list"])]
    va["pred_identity"] = va["pred"].apply(lambda p: list(p) == [1, 2, 3, 4])

    clean_ids = set(json.load(open(os.path.join(EXP, "val_clean_ids.json"))))
    va["clean"] = va["Id"].isin(clean_ids)

    lines = [f"# 오답 분석 — {os.path.basename(args.preds)}", ""]
    em_all = va["correct"].mean()
    em_clean = va[va["clean"]]["correct"].mean()
    lines += [f"- 전체 EM: **{em_all:.4f}** (n={len(va)}) / val_clean EM: **{em_clean:.4f}** (n={va['clean'].sum()})", ""]

    def table(group_col, title):
        g = va.groupby(group_col, observed=True).agg(
            n=("correct", "size"), EM=("correct", "mean"), 오답수=("correct", lambda x: (~x).sum())
        )
        g["오답비중"] = g["오답수"] / (~va["correct"]).sum()
        out = [f"## {title}", "", g.round(4).to_markdown(), ""]
        return out

    lines += table("quartile", "난이도 버킷 (sim_mean 4분위)")
    lines += table("No_ordering", "No_ordering 여부")
    lines += table("has_neardup_pair", "near-dup 프레임 쌍 포함 (sim_max>0.95)")

    # identity 예측 캘리브레이션
    id_rate_pred = va["pred_identity"].mean()
    id_rate_gold = va["No_ordering"].mean()
    lines += [
        "## Identity 캘리브레이션",
        "",
        f"- 정답 identity 비율: {id_rate_gold:.4f} / 예측 identity 비율: {id_rate_pred:.4f}",
        f"- identity 샘플 재현율: {va[va['No_ordering']]['correct'].mean():.4f}",
        f"- 셔플 샘플 EM: {va[~va['No_ordering']]['correct'].mean():.4f}",
        "",
    ]

    # CoT 게이트: near-dup 관련 오답 비중
    wrong = va[~va["correct"]]
    neardup_share = wrong["has_neardup_pair"].mean()
    gate = "보류 (해상도/near-dup 대응 우선)" if neardup_share >= 0.4 else "진행"
    lines += [
        "## CoT 파일럿 게이트 판정",
        "",
        f"- 오답 중 near-dup 쌍 포함 비중: **{neardup_share:.3f}** (기준 0.4)",
        f"- 판정: **CoT 파일럿 {gate}**",
        "",
    ]

    open(args.out, "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
