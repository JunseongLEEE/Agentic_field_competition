#!/bin/bash
# A_s42(실행 중) 종료 대기 후 나머지 3-way 학습 (GPU1 순차)
set -e
cd /root/kaggle
COT=data/cot_annotations_v3.jsonl
GPU=1

# A_s42 종료 대기
while pgrep -f "run_name pilot_A_s42" > /dev/null; do sleep 30; done
echo "A_s42 종료 확인"

CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=29701 \
  src/train_qwen.py --run_name pilot_B_s42 --cot_file $COT \
  --epochs 2 --grad_accum 8 --seed 42 >> experiments/pilot_B_s42.log 2>&1
echo "B_s42 완료"

CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=29702 \
  src/train_qwen.py --run_name pilot_A_s123 --ids_file $COT \
  --epochs 2 --grad_accum 8 --seed 123 >> experiments/pilot_A_s123.log 2>&1
echo "A_s123 완료"

CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=29703 \
  src/train_qwen.py --run_name pilot_B_s123 --cot_file $COT \
  --epochs 2 --grad_accum 8 --seed 123 >> experiments/pilot_B_s123.log 2>&1
echo "3-way 나머지 학습 완료"
