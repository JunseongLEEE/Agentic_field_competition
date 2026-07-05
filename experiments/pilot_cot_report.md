# CoT 파일럿 annotation 검수 보고 (진행 중)

## v1 (기본 프롬프트, gpt-4o-mini, detail=low)
- 800건 생성, 파싱 성공률 100% (첫 100건 기준), 실측 단가 $0.00176/샘플 (이미지 토큰 33배 배율 반영)
- 예상 총비용: 800건 ≈ $1.4 (~2,000원)

### 직접 검수 (6건: near-dup 3, distinct 2, mid 1)
| Id | 버킷 | 판정 | 비고 |
|---|---|---|---|
| ojQTn2 | near-dup .95 | 양호 | 모래성 진행 단계 판별 가능. "shovel" 등 일부 세부 불확실 |
| hPbbe0 | near-dup .97 | 수용 | 풋살 코트, 공/선수 위치 기반 서술 대체로 부합 |
| Q2ViQw | near-dup .97 | **혼합** | 아코디언: pos_1/4 정확, pos_2/3 "smiling·adjusting notes" 근거 약함 (hallucination) |
| RKk3i1 | distinct .48 | **불량** | 텍스트 슬라이드 3장을 캡션 스토리("hands being washed")로 서술 — 시각 무근거 복창 |
| 3YwEoD | distinct .49 | 우수 | 로잉머신→클로즈업→로고→모금정보, 완벽 |
| iFuggp | mid .78 | 우수 | 진공청소기 설명→이동→시연→클로즈업, 완벽 |

### 실패 모드 (v2 프롬프트로 대응)
1. **텍스트 슬라이드 혼동**: 프레임의 실제 텍스트 대신 캡션 서사를 복창 → v2: "보이는 것만 서술, 텍스트 화면은 제목을 그대로 전사" 규칙 추가
2. **near-dup 미세 hallucination**: 안 보이는 세부("smiling") 창작 → v2: "거의 동일한 프레임은 가장 작은 가시적 차이(팔 위치/줌/객체 위치)에 집중, 창작 금지" 규칙 추가

## v2 (규칙 강화 프롬프트, gpt-4o-mini)
- Q2ViQw(near-dup hallucination) 개선 확인, **RKk3i1(텍스트 슬라이드) 여전히 실패** → 4o-mini 시각 능력 한계로 판정

## v3 (규칙 강화 프롬프트 + gpt-4.1-mini) — ✅ 채택
- 800건 전원 파싱 성공, 비용 $0.174 (4.1-mini는 이미지 토큰 배율 없음 → 4o-mini보다 8배 저렴)
- 동일 6건 재검수: **전부 우수** — 텍스트 슬라이드 정확 전사("Text slide: TALISKER WHISKY..."), near-dup 미세 구분("hand raised near chest" vs "hand lowered near vacuum handle")
- → 조건 B(CoT distill) 학습 데이터로 v3 사용. 전량 확장 시(~8k) 예상 비용 ~$1.7

## 비용 누계 (실측)
- v1 $1.405 + v2 $1.416 + v3 $0.174 + 단건 테스트 ~$0.001 = **$2.99 (~4,200원) / 한도 ₩30,000의 14%**
- 교훈: 모델 선택 시 이미지 토큰 과금 방식 확인 필수 (4o-mini 33배 배율 vs 4.1 계열 패치 기반)
