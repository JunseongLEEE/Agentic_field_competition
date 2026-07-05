"""트랙 A: Qwen2.5-VL-7B-Instruct LoRA 파인튜닝 (DDP).

사용: torchrun --nproc_per_node=2 src/train_qwen.py --run_name r16_ep2
- 비전 타워 동결 (OOD 일반화 보존 + 메모리 절약)
- LLM에 LoRA, 정답 토큰만 학습
- 입력 프레임 순서 랜덤 셔플 증강
"""
import argparse
import functools
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP, load_train, train_val_split
from qwen_common import MODEL_ID, MAX_PIXELS, MIN_PIXELS, OrderDataset, collate


def log(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", default="qwen_lora")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="스모크 테스트용 샘플 수 제한")
    ap.add_argument("--ids_file", default="", help="특정 Id 목록(jsonl의 Id 또는 json 배열)으로 학습 제한")
    ap.add_argument("--cot_file", default="", help="CoT annotation jsonl (조건 B)")
    ap.add_argument("--id_prior", type=float, default=0.0, help="증강 시 identity 라벨 비율 (test prior 0.155)")
    ap.add_argument("--pixels", type=int, default=256, help="이미지당 최대 패치 수")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(
        MODEL_ID, max_pixels=args.pixels * 28 * 28, min_pixels=MIN_PIXELS
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    model.visual.requires_grad_(False)  # 비전 타워 동결
    # LLM 블록에만 LoRA 부착 (비전 타워의 동명 모듈 제외)
    proj_names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    targets = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and name.split(".")[-1] in proj_names
        and not name.startswith("visual")
        and ".visual." not in name
    ]
    log(f"LoRA target modules: {len(targets)}")
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    if rank == 0:
        model.print_trainable_parameters()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    df = load_train()
    tr_df, va_df = train_val_split(df)
    if args.limit:
        tr_df = tr_df.head(args.limit)

    import json as _json
    cot = None
    if args.cot_file:
        cot = {}
        for line in open(args.cot_file):
            r = _json.loads(line)
            if r.get("parse_ok"):
                cot[r["Id"]] = [r["rationale"][f"Input_{i}"]["desc"] for i in range(1, 5)]
    if args.ids_file:
        if args.ids_file.endswith(".jsonl"):
            keep = {_json.loads(l)["Id"] for l in open(args.ids_file) if _json.loads(l).get("parse_ok", True)}
        else:
            keep = set(_json.load(open(args.ids_file)))
        tr_df = tr_df[tr_df["Id"].isin(keep)].reset_index(drop=True)
    if cot is not None:
        tr_df = tr_df[tr_df["Id"].isin(cot)].reset_index(drop=True)

    log(f"train {len(tr_df)} / val {len(va_df)}")
    ds = OrderDataset(tr_df, proc, "train", augment=True, cot=cot, id_prior=args.id_prior)
    sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)
    dl = DataLoader(
        ds, batch_size=args.bs, sampler=sampler, num_workers=6,
        collate_fn=functools.partial(collate, pad_id=proc.tokenizer.pad_token_id),
        pin_memory=True,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_ep = len(dl) // args.grad_accum
    total = steps_per_ep * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=total, pct_start=0.03, anneal_strategy="cos"
    )

    out_dir = os.path.join(EXP, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    step = 0
    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        model.train()
        run_loss, n_acc = 0.0, 0
        for i, batch in enumerate(dl):
            batch = {k: v.to(local_rank) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            run_loss += loss.item()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                n_acc += 1
                if step % 20 == 0:
                    log(f"ep {ep} step {step}/{total} loss {run_loss/n_acc:.4f} lr {sched.get_last_lr()[0]:.2e}")
                    run_loss, n_acc = 0.0, 0
        if rank == 0:
            model.module.save_pretrained(os.path.join(out_dir, f"ep{ep}"))
            log(f"saved {out_dir}/ep{ep}")
        dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
