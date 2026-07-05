# src/ 파이프라인 가이드

## 공통
- `common.py` — 데이터 로드, 고정 검증 분할(`experiments/val_ids.json`, 954개 stratified), exact match 평가, 제출 CSV 생성
- `submit.py` — Kaggle 제출:
  ```bash
  python src/submit.py submissions/xxx.csv "메시지"   # 제출 (1일 2회 제한 주의)
  python src/submit.py --list                          # 이력/점수 확인
  ```

## 트랙 B (경량: SigLIP2 + 포인터 헤드)
```bash
python src/extract_features.py --gpu 1                 # 임베딩 사전 추출 (1회)
python src/train_light.py --gpu 1 --submit_name sub_track_b
```

## 트랙 A (메인: Qwen2.5-VL-7B LoRA)
```bash
# 학습 (GPU 2장 DDP)
torchrun --nproc_per_node=2 src/train_qwen.py --run_name qwen_r16 --epochs 2

# 검증 (24순열 우도 스코어링)
python src/infer_qwen.py --adapter experiments/qwen_r16/ep1 --split val --gpu 0 --limit 300

# 테스트 추론 + 제출 파일 생성
python src/infer_qwen.py --adapter experiments/qwen_r16/ep1 --split test --gpu 0 --submit_name sub_qwen_r16
```

## 규칙 리마인더
- 앙상블 금지 → 트랙 A/B 중 하나만 최종 제출
- 최종 추론은 반드시 GPU 1장으로 검증 (제출 규정)
- `original_data/` 학습 사용 금지 (외부 데이터)
