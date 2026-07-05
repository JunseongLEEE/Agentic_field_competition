"""SigLIP2 임베딩 사전 추출 (트랙 B용).

이미지 4장 + 캡션 전문 + 캡션 절 버킷 4개의 임베딩을 저장한다.
사용: python src/extract_features.py [--gpu 0]
"""
import argparse
import os
import re
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, EXP, load_test, load_train, image_paths

MODEL = "google/siglip2-so400m-patch14-384"

# 시간 접속사/구두점 기준 절 분할
_SPLIT_RE = re.compile(
    r"(?:;\s*|,\s*(?:then|and then|before|after which|followed by|finally|next|ending with|as the scene)\b\s*"
    r"|\.\s+|\b(?:then|after which|followed by|finally)\b,?\s*)",
    re.IGNORECASE,
)


def split_clauses(sentence: str, n: int = 4):
    """캡션을 절로 나눈 뒤 순서 유지한 채 n개 버킷으로 병합."""
    parts = [p.strip(" ,;.") for p in _SPLIT_RE.split(sentence) if p and p.strip(" ,;.")]
    if len(parts) == 0:
        parts = [sentence]
    # n개 버킷으로 균등 병합
    buckets = [[] for _ in range(n)]
    for i, p in enumerate(parts):
        buckets[min(i * n // len(parts), n - 1)].append(p)
    return [" ".join(b) if b else sentence for b in buckets]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}"

    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(MODEL)

    for split, df in [("train", load_train()), ("test", load_test())]:
        img_feats, txt_feats, clause_feats = [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split):
            imgs = [Image.open(p).convert("RGB") for p in image_paths(row, split)]
            px = proc(images=imgs, return_tensors="pt").to(dev)
            iv = model.get_image_features(**px)  # (4, D)
            iv = torch.nn.functional.normalize(iv, dim=-1)

            texts = [row["Sentence"]] + split_clauses(row["Sentence"])
            tx = proc(
                text=texts, padding="max_length", max_length=64,
                truncation=True, return_tensors="pt",
            ).to(dev)
            tv = model.get_text_features(**tx)  # (5, D)
            tv = torch.nn.functional.normalize(tv, dim=-1)

            img_feats.append(iv.cpu().numpy().astype(np.float16))
            txt_feats.append(tv[0].cpu().numpy().astype(np.float16))
            clause_feats.append(tv[1:].cpu().numpy().astype(np.float16))

        np.savez_compressed(
            os.path.join(EXP, f"siglip_{split}.npz"),
            ids=df["Id"].values,
            img=np.stack(img_feats),      # (N, 4, D)
            txt=np.stack(txt_feats),      # (N, D)
            clause=np.stack(clause_feats),  # (N, 4, D)
        )
        print(f"saved siglip_{split}.npz")


if __name__ == "__main__":
    main()
