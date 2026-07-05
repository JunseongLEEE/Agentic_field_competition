"""[전처리 전용 — 학습/추론 파이프라인과 완전 분리]

GPT-4o-mini(vision, detail=low)로 train 샘플의 프레임별 상태 서술 rationale 생성.
정적 파일 data/cot_annotations.jsonl 생성 후 종료. openai 의존성은 이 스크립트에만 존재.

병렬화: AsyncOpenAI(세마포어 동시 24) + 이미지 인코딩 멀티쓰레드(ThreadPoolExecutor).

- 대상: No_ordering=False, sim_mean 4분위 균등 (seed 42), val 제외
- GPT 입력: 정답 순서로 정렬한 4프레임 + 캡션 (GPT는 순서 문제를 풀지 않음)
- 후처리: 셔플(Input_i) 공간으로 재매핑
- 비용 가드: 누적 단가가 추산($0.0007/샘플)의 2배 초과 시 중단

사용: python scripts/annotate_cot.py [--n 800] [--detail low] [--concurrency 24]
"""
import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from common import EXP, load_train, image_paths, VAL_IDS_PATH  # noqa: E402

OUT_PATH = os.path.join(ROOT, "data", os.environ.get("COT_OUT", "cot_annotations.jsonl"))
COST_IN_PER_M = 0.15   # gpt-4o-mini $/1M input tokens
COST_OUT_PER_M = 0.60
# 실측 기준: gpt-4o-mini는 이미지 토큰 ~33배 배율 (low detail 이미지당 ~2,833tok)
# → 4이미지+텍스트 ≈ 11.6k input tok ≈ $0.00176/샘플 (100건 실측)
UNIT_EST = 0.002       # 샘플당 예상 비용($)

PROMPT = """You see 4 frames from a video, shown in CORRECT chronological order (pos_1 to pos_4), plus the video caption.

Caption: "{caption}"

For each frame, describe its single MOST DISCRIMINATIVE visual state that distinguishes it from the other frames (hand/object position, zoom level, camera angle, or progress stage). Max 8 words each. Be specific enough that the frames could be re-ordered using only your descriptions.

STRICT RULES:
- Describe ONLY what is VISIBLE in that exact frame. Never narrate the caption's story.
- If a frame is a title/text/credits screen, transcribe its heading verbatim (e.g. 'text slide: Skills Demonstrated By').
- If two frames look nearly identical, focus on the smallest visible difference (limb position, zoom, object location). Never invent details you cannot see.

Output EXACTLY this format:
pos_1: <description>
pos_2: <description>
pos_3: <description>
pos_4: <description>"""

FMT_RE = re.compile(r"pos_([1-4]):\s*(.+)")


def encode_image(path, max_side=512):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def sample_targets(n, seed=42):
    df = load_train()
    val_ids = set(json.load(open(VAL_IDS_PATH)))
    df = df[~df["Id"].isin(val_ids) & ~df["No_ordering"]].reset_index(drop=True)
    z = np.load(os.path.join(EXP, "siglip_train.npz"), allow_pickle=True)
    idx = {i: k for k, i in enumerate(z["ids"])}
    img = z["img"].astype(np.float32)
    iu = np.triu_indices(4, 1)
    df["sim_mean"] = [
        np.einsum("id,jd->ij", img[idx[i]], img[idx[i]])[iu].mean() for i in df["Id"]
    ]
    df["q"] = pd.qcut(df["sim_mean"], 4, labels=False)
    per = n // 4
    return (
        df.groupby("q", group_keys=False)
        .apply(lambda g: g.sample(min(per, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


async def annotate_one(client, sem, pool, row, detail, model):
    ans = list(row["answer_list"])
    paths = image_paths(row, "train")
    order = [ans.index(p) + 1 for p in range(1, 5)]  # pos p에 오는 입력 슬롯(1-based)
    loop = asyncio.get_event_loop()
    b64s = await asyncio.gather(*[
        loop.run_in_executor(pool, encode_image, paths[slot - 1]) for slot in order
    ])
    content = [{"type": "text", "text": PROMPT.format(caption=row["Sentence"])}]
    for p, b64 in enumerate(b64s, 1):
        content.append({"type": "text", "text": f"pos_{p}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
        })
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=200, temperature=0.2,
                )
                break
            except Exception as e:
                if attempt == 3:
                    return {"Id": row["Id"], "error": str(e)[:200], "parse_ok": False, "cost": 0}
                await asyncio.sleep(2 ** (attempt + 1))
    u = r.usage
    cost = u.prompt_tokens / 1e6 * COST_IN_PER_M + u.completion_tokens / 1e6 * COST_OUT_PER_M
    text = r.choices[0].message.content or ""
    descs = dict(FMT_RE.findall(text))
    if len(descs) == 4:
        rationale = {
            f"Input_{i+1}": {"desc": descs[str(ans[i])].strip(), "pos": ans[i]}
            for i in range(4)
        }
        return {"Id": row["Id"], "rationale": rationale, "answer": ans,
                "raw": text, "parse_ok": True, "cost": round(cost, 6)}
    return {"Id": row["Id"], "raw": text, "parse_ok": False, "cost": round(cost, 6)}


async def main_async(args):
    from openai import AsyncOpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        for line in open(os.path.join(ROOT, ".env")):
            if line.startswith("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
    client = AsyncOpenAI()

    targets = sample_targets(args.n)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    done = set()
    if os.path.exists(OUT_PATH):
        done = {json.loads(l)["Id"] for l in open(OUT_PATH)}
    todo = [row for _, row in targets.iterrows() if row["Id"] not in done]
    print(f"대상 {len(targets)} / 신규 {len(todo)}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    pool = ThreadPoolExecutor(max_workers=16)
    tot_cost, n_done, n_fail = 0.0, 0, 0
    with open(OUT_PATH, "a") as fout:
        tasks = [annotate_one(client, sem, pool, row, args.detail, args.model) for row in todo]
        for coro in asyncio.as_completed(tasks):
            res = await coro
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()
            tot_cost += res.get("cost", 0)
            n_done += 1
            n_fail += int(not res.get("parse_ok"))
            if n_done % 100 == 0:
                unit = tot_cost / max(n_done, 1)
                print(f"{n_done}/{len(todo)} unit=${unit:.5f} total=${tot_cost:.3f} fail={n_fail}", flush=True)
                if unit > UNIT_EST * 2 and n_done >= 50:
                    print("중단: 단가 초과")
                    sys.exit(1)
    print(f"완료: {n_done}건, 파싱실패 {n_fail}, 총 ${tot_cost:.3f} (~{tot_cost*1400:.0f}원)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--detail", default="low", choices=["low", "high"])
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()
    asyncio.run(main_async(args))
