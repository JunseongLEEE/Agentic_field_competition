#!/bin/bash
# 3-way 파일럿 학습 (지정 GPU에서 순차): A=answer-only, B=CoT distill, seed 42/123
# 사용: bash scripts/run_3way.sh <GPU_ID>
set -e
GPU=${1:-1}
cd /root/kaggle
COT=data/cot_annotations_v3.jsonl

for SEED in 42 123; do
  echo "=== A(answer-only) seed=$SEED ==="
  CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$((29600+SEED)) \
    src/train_qwen.py --run_name pilot_A_s$SEED --ids_file $COT \
    --epochs 2 --grad_accum 8 --seed $SEED >> experiments/pilot_A_s$SEED.log 2>&1
  echo "=== B(CoT distill) seed=$SEED ==="
  CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$((29700+SEED)) \
    src/train_qwen.py --run_name pilot_B_s$SEED --cot_file $COT \
    --epochs 2 --grad_accum 8 --seed $SEED >> experiments/pilot_B_s$SEED.log 2>&1
done
echo "3-way 학습 전체 완료"
