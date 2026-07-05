"""트랙 A 공통: Qwen2.5-VL 프롬프트/데이터셋."""
import os
import random
import sys

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import image_paths

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# 이미지당 최대 비주얼 토큰 ~256개 (28px 패치 기준) → 4장 합계 ~1k
MAX_PIXELS = 256 * 28 * 28
MIN_PIXELS = 64 * 28 * 28

SYSTEM = (
    "You are an expert at temporal reasoning over video frames. "
    "Given 4 shuffled frames from a video and a caption describing the video "
    "in chronological order, determine each frame's true temporal position."
)

INSTRUCTION = (
    'Caption: "{caption}"\n\n'
    "The 4 frames above are shown in shuffled order (Frame 1-4). "
    "Using the caption, which narrates events in chronological order, determine "
    "the true temporal position of each frame.\n"
    "Answer with a list of 4 numbers: the i-th number is the chronological "
    "position (1=earliest, 4=latest) of Frame i. "
    "Example: [2, 4, 3, 1] means Frame 1 is 2nd, Frame 2 is 4th, Frame 3 is 3rd, "
    "Frame 4 is 1st in time. Answer with only the list."
)


def build_messages(images, caption):
    content = []
    for i, im in enumerate(images):
        content.append({"type": "text", "text": f"Frame {i+1}:"})
        content.append({"type": "image", "image": im})
    content.append({"type": "text", "text": INSTRUCTION.format(caption=caption)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": content},
    ]


def load_images(row, split):
    return [Image.open(p).convert("RGB") for p in image_paths(row, split)]


COT_SUFFIX = (
    "\nFirst, for each frame state its most discriminative visible detail and its "
    "chronological position, then give the final answer list."
)


def build_cot_target(answer, descs):
    """descs: 셔플 후 슬롯 순서의 상태 서술 리스트."""
    lines = [
        f"Frame {i+1}: {descs[i]} -> position {answer[i]}" for i in range(4)
    ]
    return "\n".join(lines) + "\nAnswer: [" + ", ".join(map(str, answer)) + "]"


class OrderDataset(Dataset):
    """학습용: 매 호출마다 입력 프레임 순서를 랜덤 셔플(라벨 재계산).

    cot: {Id: [Input_1..4의 desc]} 제공 시 rationale+답 타깃(조건 B).
    """

    def __init__(self, df, processor, split="train", augment=True, cot=None, id_prior=0.0):
        self.df = df.reset_index(drop=True)
        self.proc = processor
        self.split = split
        self.augment = augment
        self.cot = cot
        # augment 시 identity 라벨 비율을 test prior(~0.155)에 맞춤.
        # (uniform 셔플 증강은 identity를 1/24=4.2%로 희석시켜 identity 미탐을 유발)
        self.id_prior = id_prior

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        images = load_images(row, self.split)
        answer = list(row["answer_list"])
        descs = self.cot.get(row["Id"]) if self.cot else None
        if self.augment:
            if self.id_prior and random.random() < self.id_prior:
                # 증강 후 라벨이 identity가 되도록 정렬 (자연 시간 순서 제시)
                perm = [answer.index(p) for p in range(1, 5)]
            else:
                while True:
                    perm = list(range(4))
                    random.shuffle(perm)  # perm[k] = 원래 슬롯 index
                    if not self.id_prior or [answer[k] for k in perm] != [1, 2, 3, 4]:
                        break
            images = [images[k] for k in perm]
            answer = [answer[k] for k in perm]
            if descs:
                descs = [descs[k] for k in perm]
        msgs = build_messages(images, row["Sentence"])
        if descs:
            msgs[1]["content"][-1]["text"] += COT_SUFFIX
            target = build_cot_target(answer, descs)
        else:
            target = "[" + ", ".join(map(str, answer)) + "]"

        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.proc(text=[text], images=images, return_tensors="pt", padding=False)
        prompt_ids = enc["input_ids"][0]
        # 타깃은 별도 토크나이즈 후 이어붙임 (BPE 경계 병합 방지)
        target_ids = self.proc.tokenizer(
            target + "<|im_end|>", add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]

        input_ids = torch.cat([prompt_ids, target_ids])
        labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids])

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
            "labels": labels,
        }


def collate(batch, pad_id):
    maxlen = max(x["input_ids"].size(0) for x in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for x in batch:
        n = maxlen - x["input_ids"].size(0)
        out["input_ids"].append(torch.cat([x["input_ids"], torch.full((n,), pad_id, dtype=torch.long)]))
        out["attention_mask"].append(torch.cat([x["attention_mask"], torch.zeros(n, dtype=torch.long)]))
        out["labels"].append(torch.cat([x["labels"], torch.full((n,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(out["input_ids"]),
        "attention_mask": torch.stack(out["attention_mask"]),
        "labels": torch.stack(out["labels"]),
        "pixel_values": torch.cat([x["pixel_values"] for x in batch]),
        "image_grid_thw": torch.cat([x["image_grid_thw"] for x in batch]),
    }
