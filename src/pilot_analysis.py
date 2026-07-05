"""3-way 파일럿 결과 비교: EM + 캡션 길이/유사도 층화.

사용: python src/pilot_analysis.py  (experiments/preds_pilot_*.json 자동 수집)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP, load_train, train_val_split


def main():
    df = load_train()
    _, va = train_val_split(df)
    z = np.load(os.path.join(EXP, "siglip_train.npz"), allow_pickle=True)
    idx = {i: k for k, i in enumerate(z["ids"])}
    emb = z["img"].astype(np.float32)
    iu = np.triu_indices(4, 1)

    rows = []
    for path in sorted(glob.glob(os.path.join(EXP, "preds_pilot_*.json"))):
        if path.endswith("_scores.json"):
            continue
        name = os.path.basename(path).replace("preds_pilot_", "").replace(".json", "")
        preds = json.load(open(path))
        sub = va[va["Id"].isin(preds)].copy()
        gold = np.stack(sub["answer_list"].values)
        pred = np.stack([preds[i] for i in sub["Id"]])
        sub["correct"] = (gold == pred).all(1)
        sub["wc"] = sub["Sentence"].str.split().str.len()
        sub["sim_mean"] = [
            np.einsum("id,jd->ij", emb[idx[i]], emb[idx[i]])[iu].mean() for i in sub["Id"]
        ]
        rows.append({
            "exp": name,
            "n": len(sub),
            "EM": sub["correct"].mean(),
            "EM_short(<=20w)": sub[sub.wc <= 20]["correct"].mean(),
            "EM_long(>20w)": sub[sub.wc > 20]["correct"].mean(),
            "EM_neardup(sim>.8)": sub[sub.sim_mean > 0.8]["correct"].mean(),
            "EM_easy(sim<=.8)": sub[sub.sim_mean <= 0.8]["correct"].mean(),
        })
    out = pd.DataFrame(rows).set_index("exp").round(4)
    print(out.to_markdown())
    return out


if __name__ == "__main__":
    main()
