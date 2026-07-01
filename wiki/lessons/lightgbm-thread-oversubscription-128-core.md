---
id: lightgbm-thread-oversubscription-128-core
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [performance, threads, lightgbm, xgboost, catboost]
related: [[subagent-must-run-training-foreground], [[charwb-tfidf-vocab-single-thread-stall]]]
summary: 128코어 박스에서 GBDT를 n_jobs=-1로 여러 개 동시에 돌리면 스레드 과다구독으로 수십 분씩 행(hang)한다 — 스레드 ≤16, 순차 실행.
---

# LightGBM thread oversubscription on a 128-core box

## Symptom
LightGBM/XGBoost/CatBoost 학습들이 시작만 하고 수십 분간 진행이 없거나(hang)
비정상적으로 느려졌다. CPU는 100% 근처인데 실질 진척이 없었다.

## Root cause
박스에 코어가 **128개**다. GBDT를 `n_jobs=-1` / `num_threads` / `thread_count`
기본값(=전체 코어)로 두고, 게다가 여러 학습을 동시에 팬아웃하면
스레드 수가 물리 코어를 크게 초과(oversubscription)해 컨텍스트 스위칭
thrashing이 발생한다. BLAS/OpenMP 스레드까지 겹치면 더 심해진다.

## Fix
Rule B로 명문화:
- `n_jobs` / `num_threads` / `thread_count` ≤ **16**.
- 무거운 학습은 **순차 실행**(최대 2개 병렬), 전체 코어 팬아웃 금지.
- 러너는 `OMP_NUM_THREADS=16`을 export.
- 가능하면 로컬 GPU(RTX 3090)로 학습(lightgbm device=gpu / xgboost device=cuda /
  catboost task_type=GPU)해 CPU 경합 자체를 피한다.
`/dev`, `/auto`, `/run`, 에이전트 정의에 반영.

## Generalization
코어가 아주 많은 머신에서 `n_jobs=-1`은 오히려 독이다. 동시에 여러 병렬
라이브러리(모델 내부 스레드 × 프로세스 수 × BLAS 스레드)가 곱해져 물리 코어를
초과하지 않도록 스레드 예산을 명시적으로 나눠라. "가능한 최대"가 아니라
"코어 대비 안전한 상한"을 설정한다.
