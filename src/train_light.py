"""트랙 B: SigLIP 임베딩 + NACON식 크로스모달 포인터 헤드.

- 입력: 프레임 임베딩 4개(순서 무관), 캡션 전문 + 절 버킷 임베딩 4개(위치 임베딩 부여)
- 출력: 4x4 포인터 행렬 (frame i -> position j)
- 손실: 행/열 양방향 CE (NACON exclusive loss)
- 디코딩: Hungarian
- 증강: 매 스텝 입력 프레임 순서 랜덤 셔플

사용: python src/train_light.py --gpu 1
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP, SUB, load_train, load_test, train_val_split, exact_match, make_submission

D_IN = 1152  # siglip2-so400m


class PointerHead(nn.Module):
    def __init__(self, d=512, nhead=8, nlayers=4, dropout=0.1):
        super().__init__()
        self.img_proj = nn.Linear(D_IN, d)
        self.txt_proj = nn.Linear(D_IN, d)
        self.type_emb = nn.Embedding(3, d)  # 0=img, 1=clause, 2=caption
        self.pos_emb = nn.Embedding(4, d)   # 절 버킷 위치(순서 모달리티에만)
        enc = nn.TransformerEncoderLayer(d, nhead, d * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, nlayers)
        self.scorer = nn.Sequential(nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))
        self.d = d

    def forward(self, img, clause, txt):
        # img (B,4,D_IN) 순서 무관 / clause (B,4,D_IN) 위치 임베딩 부여 / txt (B,D_IN)
        B = img.size(0)
        pos = torch.arange(4, device=img.device)
        ti = self.img_proj(img) + self.type_emb.weight[0]
        tc = self.txt_proj(clause) + self.type_emb.weight[1] + self.pos_emb(pos)
        tt = self.txt_proj(txt).unsqueeze(1) + self.type_emb.weight[2]
        h = self.encoder(torch.cat([ti, tc, tt], dim=1))
        hi, hc = h[:, :4], h[:, 4:8]  # frame / position 표현
        # 4x4 스코어: frame i가 position j
        hi_e = hi.unsqueeze(2).expand(B, 4, 4, self.d)
        hc_e = hc.unsqueeze(1).expand(B, 4, 4, self.d)
        return self.scorer(torch.cat([hi_e, hc_e], dim=-1)).squeeze(-1)  # (B,4,4)


def exclusive_loss(scores, target):
    """scores (B,4,4) frame->pos, target (B,4) = 각 frame의 정답 pos(0-based)."""
    ce = nn.functional.cross_entropy
    row = ce(scores.reshape(-1, 4), target.reshape(-1))
    # 열방향: pos j -> frame. target_col[j] = frame index
    tcol = torch.argsort(target, dim=1)
    col = ce(scores.transpose(1, 2).reshape(-1, 4), tcol.reshape(-1))
    return row + col


def decode(scores):
    """(B,4,4) -> (B,4) 각 frame의 pos(1-based), Hungarian 최적 매칭."""
    out = []
    for s in scores.detach().cpu().numpy():
        r, c = linear_sum_assignment(-s)
        perm = np.empty(4, dtype=int)
        perm[r] = c + 1
        out.append(perm.tolist())
    return out


def load_feats(split):
    z = np.load(os.path.join(EXP, f"siglip_{split}.npz"), allow_pickle=True)
    return z["ids"], z["img"].astype(np.float32), z["clause"].astype(np.float32), z["txt"].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--submit_name", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = f"cuda:{args.gpu}"

    df = load_train()
    ids, img, clause, txt = load_feats("train")
    feat_idx = {i: k for k, i in enumerate(ids)}
    tr_df, va_df = train_val_split(df)

    def tensors(sub_df):
        k = [feat_idx[i] for i in sub_df["Id"]]
        y = np.stack(sub_df["answer_list"].values) - 1  # 0-based pos
        return (
            torch.tensor(img[k]), torch.tensor(clause[k]),
            torch.tensor(txt[k]), torch.tensor(y, dtype=torch.long),
        )

    Xi, Xc, Xt, Y = tensors(tr_df)
    Vi, Vc, Vt, Vy = [t.to(dev) for t in tensors(va_df)]

    model = PointerHead(args.d, nlayers=args.nlayers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.epochs * ((len(Xi) + args.bs - 1) // args.bs)
    )

    best_em, best_state = 0.0, None
    for ep in range(args.epochs):
        model.train()
        order = torch.randperm(len(Xi))
        tot = 0.0
        for b in range(0, len(Xi), args.bs):
            idx = order[b : b + args.bs]
            bi, bc, bt, by = Xi[idx].to(dev), Xc[idx].to(dev), Xt[idx].to(dev), Y[idx].to(dev)
            # 입력 프레임 순서 셔플 증강
            perm = torch.stack([torch.randperm(4) for _ in range(len(idx))]).to(dev)
            bi = torch.gather(bi, 1, perm.unsqueeze(-1).expand_as(bi))
            by = torch.gather(by, 1, perm)
            loss = exclusive_loss(model(bi, bc, bt), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            preds = decode(model(Vi, Vc, Vt))
        em = exact_match(preds, (Vy + 1).cpu().tolist())
        if em > best_em:
            best_em, best_state = em, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:02d} loss {tot/len(Xi):.4f} val_EM {em:.4f} best {best_em:.4f}")

    print(f"BEST val EM: {best_em:.4f}")
    torch.save(best_state, os.path.join(EXP, "track_b_best.pt"))

    if args.submit_name:
        model.load_state_dict(best_state)
        te = load_test()
        tids, timg, tclause, ttxt = load_feats("test")
        tfi = {i: k for k, i in enumerate(tids)}
        k = [tfi[i] for i in te["Id"]]
        with torch.no_grad():
            scores = model(
                torch.tensor(timg[k]).to(dev),
                torch.tensor(tclause[k]).to(dev),
                torch.tensor(ttxt[k]).to(dev),
            )
        preds = decode(scores)
        path = make_submission(
            dict(zip(te["Id"], preds)), os.path.join(SUB, f"{args.submit_name}.csv")
        )
        print("saved:", path)


if __name__ == "__main__":
    main()
