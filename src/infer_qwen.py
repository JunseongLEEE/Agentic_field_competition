"""트랙 A 추론: 24개 순열 우도 스코어링 (기본) 또는 greedy 생성.

사용:
  python src/infer_qwen.py --adapter experiments/qwen_lora/ep1 --split val --gpu 0
  python src/infer_qwen.py --adapter experiments/qwen_lora/ep1 --split test --gpu 0 --submit_name sub_qwen
  (--adapter 생략 시 zero-shot)
"""
import argparse
import itertools
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SUB, exact_match, load_test, load_train, make_submission, train_val_split
from qwen_common import MODEL_ID, MAX_PIXELS, MIN_PIXELS, COT_SUFFIX, build_messages, load_images

PERMS = list(itertools.permutations([1, 2, 3, 4]))
ANS_RE = __import__("re").compile(r"\[\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*,\s*([1-4])\s*\]")


@torch.no_grad()
def generate_answer(model, proc, images, caption, dev, cot=False, max_new=240):
    """생성 기반 예측 (greedy/CoT). 유효 순열 파싱 실패 시 None."""
    msgs = build_messages(images, caption)
    if cot:
        msgs[1]["content"][-1]["text"] += COT_SUFFIX
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[prompt], images=images, return_tensors="pt").to(dev)
    out = model.generate(
        **enc, max_new_tokens=max_new if cot else 16, do_sample=False,
        pad_token_id=proc.tokenizer.pad_token_id,
    )
    text = proc.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    m = None
    for m in ANS_RE.finditer(text):  # 마지막 매치(CoT는 Answer:가 끝에)
        pass
    if m:
        perm = [int(m.group(i)) for i in range(1, 5)]
        if sorted(perm) == [1, 2, 3, 4]:
            return perm, text
    return None, text


@torch.no_grad()
def perm_scores(model, proc, images, caption, dev, chunk=4):
    """24개 순열 각각의 정답 문자열 log-likelihood 벡터 (PERMS 순서)."""
    msgs = build_messages(images, caption)
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = proc(text=[prompt], images=images, return_tensors="pt").to(dev)
    p_ids = enc["input_ids"]  # (1, P)

    cand_ids = torch.stack([
        proc.tokenizer(
            "[" + ", ".join(map(str, p)) + "]", add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        for p in PERMS
    ]).to(dev)  # (24, T) — 형식이 동일해 길이 동일
    T = cand_ids.shape[1]

    scores = []
    for s in range(0, 24, chunk):
        c = cand_ids[s : s + chunk]
        B = c.shape[0]
        ids = torch.cat([p_ids.expand(B, -1), c], dim=1)
        out = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            pixel_values=enc["pixel_values"].repeat(B, 1),
            image_grid_thw=enc["image_grid_thw"].repeat(B, 1),
        )
        tail = out.logits[:, -(T + 1) : -1]  # 타깃 구간 로짓만 사용
        logp = torch.log_softmax(tail.float(), dim=-1)  # (B, T, V)
        tok_lp = logp.gather(-1, c.unsqueeze(-1)).squeeze(-1)  # (B, T)
        scores.extend(tok_lp.sum(-1).tolist())
    return scores


def predict(model, proc, images, caption, dev, tta=1, rng=None):
    """TTA: 입력 프레임 제시 순서를 바꿔 24순열 우도를 원공간으로 매핑해 평균."""
    import random as _random

    agg = {p: 0.0 for p in PERMS}
    sigmas = [list(range(4))]
    if tta > 1:
        rng = rng or _random.Random(0)
        pool = [[x - 1 for x in p] for p in PERMS[1:]]  # 0-based 제시 순서
        rng.shuffle(pool)
        sigmas += pool[: tta - 1]
    for sigma in sigmas:
        imgs_p = [images[s] for s in sigma]
        scores = perm_scores(model, proc, imgs_p, caption, dev)
        inv = [sigma.index(k) for k in range(4)]
        for pi, p in enumerate(PERMS):
            # 제시공간 정답 p → 원공간 정답 ans_o[k] = p[inv[k]]
            ans_o = tuple(p[inv[k]] for k in range(4))
            agg[ans_o] += scores[pi]
    return list(max(agg, key=agg.get)), agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--submit_name", default="")
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--save_preds", default="", help="예측 JSON 저장 경로 (오답 분석용)")
    ap.add_argument("--pixels", type=int, default=256, help="이미지당 최대 패치 수 (기본 256=학습 설정)")
    ap.add_argument("--mode", choices=["score", "greedy", "cot"], default="score")
    ap.add_argument("--shuffled_only", action="store_true", help="val에서 No_ordering=False만 평가 (3-way 공정성)")
    ap.add_argument("--shard", default="", help="k/n: 데이터 n등분 중 k번째(0-base)만 처리 (GPU 병렬용)")
    args = ap.parse_args()
    dev = f"cuda:{args.gpu}"

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(
        MODEL_ID, max_pixels=args.pixels * 28 * 28, min_pixels=MIN_PIXELS
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model = model.to(dev).eval()

    if args.split == "val":
        _, df = train_val_split(load_train())
        img_split = "train"
        if args.shuffled_only:
            df = df[~df["No_ordering"]].reset_index(drop=True)
    else:
        df = load_test()
        img_split = "test"
    if args.limit:
        df = df.head(args.limit)
    if args.shard:
        k, n = map(int, args.shard.split("/"))
        df = df.iloc[k::n].reset_index(drop=True)

    # 증분 저장/재개: --save_preds 시 .part.jsonl에 샘플별 append, 기존 것은 스킵
    preds, all_scores, n_fallback = {}, {}, 0
    part_path = args.save_preds.replace(".json", ".part.jsonl") if args.save_preds else ""
    if part_path and os.path.exists(part_path):
        import json as _json
        for line in open(part_path):
            r = _json.loads(line)
            preds[r["Id"]] = r["pred"]
            if "scores" in r:
                all_scores[r["Id"]] = r["scores"]
        print(f"재개: 기존 {len(preds)}건 로드")
    part_f = open(part_path, "a") if part_path else None

    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row["Id"] in preds:
            continue
        images = load_images(row, img_split)
        if args.mode == "score":
            preds[row["Id"]], agg = predict(model, proc, images, row["Sentence"], dev, tta=args.tta)
            all_scores[row["Id"]] = {",".join(map(str, k)): round(v, 4) for k, v in agg.items()}
        else:
            perm, _ = generate_answer(model, proc, images, row["Sentence"], dev, cot=(args.mode == "cot"))
            if perm is None:  # 파싱 실패 → 우도 스코어링 fallback
                n_fallback += 1
                perm, _ = predict(model, proc, images, row["Sentence"], dev, tta=1)
            preds[row["Id"]] = perm
        if part_f:
            import json as _json
            rec = {"Id": row["Id"], "pred": preds[row["Id"]]}
            if row["Id"] in all_scores:
                rec["scores"] = all_scores[row["Id"]]
            part_f.write(_json.dumps(rec) + "\n")
            part_f.flush()
    if args.mode != "score":
        print(f"파싱 실패 fallback: {n_fallback}/{len(df)}")

    if args.save_preds:
        import json
        json.dump(preds, open(args.save_preds, "w"))
        if all_scores:
            json.dump(all_scores, open(args.save_preds.replace(".json", "_scores.json"), "w"))

    if args.split == "val":
        em = exact_match([preds[i] for i in df["Id"]], df["answer_list"].tolist())
        print(f"val EM ({len(df)} samples): {em:.4f}")
        clean_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "val_clean_ids.json")
        if os.path.exists(clean_path):
            import json as _json
            clean_ids = set(_json.load(open(clean_path)))
            cdf = df[df["Id"].isin(clean_ids)]
            if len(cdf):
                cem = exact_match([preds[i] for i in cdf["Id"]], cdf["answer_list"].tolist())
                print(f"val_clean EM ({len(cdf)} samples): {cem:.4f}")
    elif args.submit_name:
        path = make_submission(preds, os.path.join(SUB, f"{args.submit_name}.csv"))
        print("saved:", path)


if __name__ == "__main__":
    main()
