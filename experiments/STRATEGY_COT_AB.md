# CoT Distill 파일럿 프로토콜 (합의: 2026-07-04)

## 원칙
- GPT 호출 코드는 `scripts/annotate_cot.py`에만 존재 (학습·추론 코드에 openai 의존성 금지 — 재현성 검증 대비)
- 산출물은 정적 파일 `data/cot_annotations.jsonl` (전처리 결과물로 보고서에 사용 방식·비용 명시)
- 규칙 3.2 준수: 전처리 한정, 누적 비용 ₩30,000 한도 (실측 기록 필수)

## 1단계: 오답 분석 게이트 (선행)
- qwen_r16_ep2 val 평가 후 `src/error_analysis.py` 실행
- 축: sim_mean 4분위 / No_ordering / near-dup 쌍(sim_max>0.95) / identity 캘리브레이션
- **게이트: near-dup 관련 오답 ≥ 40% → CoT 보류, 해상도·프레임 대조 대응 우선. < 40% → 2단계 진행**
- 산출: `experiments/error_analysis_qwen_r16_ep2.md`

## 2단계: 파일럿 annotation (800샘플)
- 대상: No_ordering=False, sim_mean 4분위 균등 200개씩, seed 42, val 제외
- 모델: gpt-4o-mini vision detail=low 고정 (품질 미달 시 near-dup 버킷만 detail=high 승격)
- GPT는 **정답 순서로 정렬된** 프레임을 보고 프레임별 판별적 상태만 서술 (순서 문제를 풀지 않음 → GPT 오류 전파 차단)
- 후처리로 셔플(Input_i) 공간 재매핑
- 비용 가드: 실측 단가가 추산 2배 초과 시 즉시 중단 (스크립트 내장)
- 검수: 20개 육안 (near-dup 케이스 절반 포함) — hallucination / 판별력 / 파싱 100%
- 산출: `experiments/pilot_cot_report.md`

## 3단계: 3-way 비교 (파일럿 800샘플 학습)
| 조건 | 학습 타깃 | 추론 |
|---|---|---|
| A. answer-only | `[n,n,n,n]` | 24순열 우도 (현행) |
| B. CoT distill | rationale + 답 | rationale 생성 → 답 파싱 (실패 시 24순열 우도 fallback) |
| C. 직접 생성 | `[n,n,n,n]` | greedy 생성 (스코어링 없이) — 우도 스코어링의 기여 분리 측정 |

- 조정 사항: 원안의 C(P(caption\|ordered_frames))는 answer-only 학습과 분포 불일치 → 캡션 생성 보조 태스크 포함 멀티태스크와 함께 **조건 D(후속)**로 분리
- 공정성: 동일 base(Qwen2.5-VL-7B 원본), 동일 스텝, seed {42, 123} 평균, val/val_clean 병기
- **Identity bias 통제: 3-way 평가는 val의 셔플 샘플 서브셋으로 한정** (파일럿 학습셋에 identity 없음)
- wall-clock 사전 측정: val 100샘플로 B(rationale 200tok 상한)·A(24 forward) 처리량 → test 24h 초과 예상 시 rationale 60tok 축소 or top-k 순열 부분 스코어링
- 산출: `experiments/ab_cot_results.md` (EM ± std, wall-clock, 파싱 실패율)

## 이후 결정
- B 승리 시: 전체 train(셔플분)으로 annotation 확장 (비용 재추산 후, ~8k샘플 ≈ $3~7) → 본 학습
- A/C 승리 시: CoT 폐기, 해상도·TTA·32B QLoRA 등 다른 레버로 전환
