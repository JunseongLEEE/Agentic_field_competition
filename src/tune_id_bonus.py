"""identity 순열 로그보너스 튜닝 (val 스코어 파일 기반, GPU 불필요).

val을 절반(튜닝)/절반(검증)으로 나눠 과적합 없이 b* 선택.
사용: python src/tune_id_bonus.py --scores experiments/preds_val_X_scores.json
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_train, train_val_split

ID_KEY = "1,2,3,4"


def em_with_bonus(scores, gold, b):
    hit = 0
    for sc, g in zip(scores, gold):
        best, bestv = None, -1e18
        for k, v in sc.items():
            v2 = v + (b if k == ID_KEY else 0.0)
            if v2 > bestv:
                bestv, best = v2, k
        hit += int(best == ",".join(map(str, g)))
    return hit / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    args = ap.parse_args()
    sc_by_id = json.load(open(args.scores))
    _, va = train_val_split(load_train())
    va = va[va["Id"].isin(sc_by_id)].reset_index(drop=True)
    rng = np.random.RandomState(7)
    mask = rng.rand(len(va)) < 0.5

    scores = [sc_by_id[i] for i in va["Id"]]
    gold = va["answer_list"].tolist()
    s_t = [s for s, m in zip(scores, mask) if m]
    g_t = [g for g, m in zip(gold, mask) if m]
    s_h = [s for s, m in zip(scores, mask) if not m]
    g_h = [g for g, m in zip(gold, mask) if not m]

    grid = np.arange(0.0, 3.01, 0.1)
    ems = [em_with_bonus(s_t, g_t, b) for b in grid]
    b_star = float(grid[int(np.argmax(ems))])
    print(f"튜닝 절반(n={len(g_t)}): b=0 EM {ems[0]:.4f} → b*={b_star} EM {max(ems):.4f}")
    print(f"검증 절반(n={len(g_h)}): b=0 EM {em_with_bonus(s_h, g_h, 0):.4f} → b* EM {em_with_bonus(s_h, g_h, b_star):.4f}")
    print(f"전체: b=0 {em_with_bonus(scores, gold, 0):.4f} → b* {em_with_bonus(scores, gold, b_star):.4f}")


if __name__ == "__main__":
    main()
