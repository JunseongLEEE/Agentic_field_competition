# 실험 로그

> 검증: 954샘플 고정 stratified 분할 (`experiments/val_ids.json`) / 지표: Exact Match
> LB: public = test 819개 중 70% (~573개) → LB는 노이즈 있음, CV 우선 신뢰
> 제출 1일 2회 (00:00 UTC 리셋) — 제출 전 반드시 CV로 근거 확보

| Exp | 날짜 | 모델/설정 | Val EM | LB (public) | CV-LB 갭 | 제출파일 | 비고 |
|---|---|---|---|---|---|---|---|
| identity | 07-04 | 항상 [1,2,3,4] | 0.1551 | **0.15532** | -0.000 | sub_identity.csv | test identity 비율 = train과 동일 확인 |
| track_b_v1 | 07-04 | SigLIP2-so400m frozen + 포인터 헤드(d512, L4) + exclusive loss | 0.1415 | 미제출 | - | sub_track_b.csv(보류) | identity보다 낮음 → 탈락. 전역 임베딩으로는 미세 순서 신호 부족 (NACON TACoS 수치와 일치) |
| qwen_zeroshot | 07-04 | Qwen2.5-VL-7B zero-shot, 24순열 우도 | (중단) | - | - | - | GPU 확보 위해 중단. 학습 후 유휴 시간에 재측정 예정 |
| qwen_r16_ep0 | 07-04 | Qwen2.5-VL-7B LoRA r16, 24순열 우도, 1ep | 0.6033 | - | - | - | val_clean 0.6066 |
| qwen_r16_ep1 | 07-04 | 상동, 2ep | **0.6333** | **0.65794** | **-0.0246** | sub_qwen_r16_ep1.csv | val_clean 0.6324. LB > CV: 합성 29%(명시적 서사 캡션)가 쉬운 구간, OOD 붕괴 없음. 오답: adj-swap 42%, 단문캡션 EM 0.19, identity FN 17건 |
| pilot_A_s42_eval | 07-04 | A 800샘플, 24순열 score, 셔플만 200 | 0.3750 | - | - | - | val_clean 0.3846 |
| pilot_B_s42_eval | 07-04 | B CoT(v3) 800샘플, cot 생성, 셔플만 200 | 0.3900 | - | - | - | val_clean 0.4121, 파싱실패 0/200. B가 +1.5pp |
| qwen_r16_idp_ep0 | 07-04 | ep1 설정 + **id_prior 0.155**, 1ep | 0.5900 | - | - | - | |
| qwen_r16_idp_ep1 | 07-05 | 상동, 2ep | **0.6500** | **0.69284** | -0.0428 | sub_qwen_idp_ep1.csv | val_clean 0.6507. identity recall 17/25→23/25. LB +3.5pp vs 구모델. id_bonus 후처리는 기각(캘리브레이션 해결됨) |

### 남은 레버 (우선순위, 7/5 이후)
1. 해상도 px400 평가 → 고해상도 재학습 (near-dup 오답 44.5% 직접 타격)
2. 에폭 연장 (3-4ep) / TTA2 val 검증
3. 파일럿 s123 평가 (CoT seed 분산), 조건 C(greedy) 속도-정확도 트레이드오프
4. Qwen2.5-VL-32B QLoRA (디스크 확보 필요: HF 캐시 잔여 모델 정리 ~18GB)
5. train+val 합본 최종 학습 (마감 직전)
| pilot_A_s42 | 07-04 | answer-only 800샘플(4분위 균등), 2ep | 평가대기 | - | - | - | 3-way 조건 A |
| pilot_B_s42 | 07-04 | CoT distill(v3) 800샘플, 2ep | 평가대기 | - | - | - | 3-way 조건 B |

### 다음 재학습 스펙 (IMPROVEMENT_PLAN.md)
- id_prior 0.155 (identity 증강 버그 수정) + 3ep + 해상도(px400 평가로 결정) + (3-way 결과에 따라 CoT 혼합)

## CV-LB 갭 관찰
- identity: 갭 사실상 0 (0.1551 vs 0.1553) → 검증 분할이 test 실사분과 잘 정렬됨.
- **주의**: val은 전부 실사 프레임. test는 29% 합성 이미지 포함 → 파인튜닝 모델의 CV-LB 갭은 OOD 일반화 손실의 지표로 해석할 것. 갭이 크게 벌어지면 과적합(합성분 붕괴) 신호.

## 제출 이력/예산
- 07-04: 2회 중 1회 사용 (identity). **1회 남음** — qwen_r16_ep2 결과가 CV에서 유의미하면 사용.
- 07-05 (00:00 UTC~): 2회. 마감 02:00 UTC 전까지 사용 가능.
