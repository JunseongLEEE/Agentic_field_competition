# Experiment Goals

> **Single source of truth**은 `competition_meta.yaml`. 이 문서는 사람이 빠르게 읽기 위한 요약 + 전략 backlog.

## Competition Objective
- **Competition**: SW중심대학협의회 — AI Agent Action Decision (DACON)
- **Task**: Multi-class classification (14 classes)
- **Metric**: Macro-F1
- **Target**: AI 코딩 에이전트의 다음 행동 예측

## Input
- `current_prompt` — 현재 사용자 발화
- `history` — 직전까지의 대화/행동 이력
- `session_meta` — 요금제, 잔여 토큰 예산, 작업공간 상태 등
- (정확한 컬럼은 데이터 공개 후 `data_docs/dataset_overview.md`에 기록)

## Output
- 14개 클래스 중 하나에 대한 예측
- 형식: 데이터 공개 후 sample_submission.csv 기준

## Constraints (평가 서버)
| 항목 | 값 |
|---|---|
| zip 크기 | ≤ 1GB |
| 설치 시간 | ≤ 10분 |
| 추론 시간 | ≤ 10분 |
| GPU | T4 (16GB VRAM) |
| CPU | 3 vCPU |
| RAM | 12GB |
| 네트워크 | offline |

## Deadlines
| 항목 | 일시 | 비고 |
|---|---|---|
| 예선 코드 제출 | 2026-07-15 10:00 KST | 매일 최대 10회 제출 |
| 본선 학습코드 + 발표자료 | 2026-07-20 10:00 KST | 상위 12팀 |
| 포스터 세션 | 2026-07-30 10:00 KST | 본선 평가 미포함 |

---

## Current Strategy

### Phase 1: Baseline (D-day ~ D-12)
- [ ] EDA + data_docs 완성 (스키마, 클래스 분포, history 길이 분포)
- [ ] 클래스 14개 정확한 정의 파악 + 불균형 정도 측정
- [ ] 단순 baseline 3종: TF-IDF + LightGBM / TF-IDF + LogReg / 짧은 history만 사용
- [ ] 5-fold StratifiedKFold 셋업 (Macro-F1 계산)
- [ ] 첫 제출로 CV-LB gap 측정

### Phase 2: Feature Engineering (D-12 ~ D-7)
- [ ] history 시퀀스 인코딩 (마지막 N action 원핫, action n-gram)
- [ ] prompt 길이, 토큰 수, 키워드 매칭 피처
- [ ] session_meta 정규화 (잔여 토큰, 요금제 인코딩)
- [ ] 도메인 피처: "테스트 직후엔 fix 확률↑" 같은 패턴

### Phase 3: Model Exploration (D-7 ~ D-4)
- [ ] LightGBM / XGBoost / CatBoost 동일 피처로 비교
- [ ] 소형 NN: MLP, 작은 BERT-distil 한국어 모델 (offline 가능 모델만)
- [ ] 추론 속도 측정 — 10분 내 전체 test 처리 가능한지

### Phase 4: Ensemble (D-4 ~ D-2)
- [ ] 다양성 있는 top-N blending (model_type 다른 것 우선)
- [ ] CV 기반 weight 최적화 (LB overfit 금지)
- [ ] 모델 크기 합산 < 1GB 확인

### Phase 5: Final (D-2 ~ D-day)
- [ ] Seed ensemble로 stability 확보
- [ ] script.py 추론 시간 측정 (≤ 10분 마진 확인)
- [ ] zip 크기 < 1GB 확인
- [ ] 최종 후보 2~3개 선정

---

## Hypotheses Backlog
| Priority | Hypothesis | Expected Impact | Status |
|----------|-----------|-----------------|--------|
| HIGH | history의 마지막 action 자체가 다음 action의 강한 predictor | +0.05 Macro-F1 | PLANNED |
| HIGH | session_meta의 잔여 토큰이 작을수록 "사용자 질문" 확률 ↑ | +0.02 | PLANNED |
| MEDIUM | 한국어 prompt embedding (offline distil 모델) | +0.03 | PLANNED |
| MEDIUM | 클래스 불균형 → focal loss or class_weight | +0.01 ~ +0.03 | PLANNED |
| LOW | history n-gram (bigram of action sequences) | +0.01 | PLANNED |
