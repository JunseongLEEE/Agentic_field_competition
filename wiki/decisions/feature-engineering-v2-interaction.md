---
id: feature-engineering-v2-interaction
type: decision
created: 2026-07-01
updated: 2026-07-01
tags: [feature-engineering, interaction, lightgbm]
related: [[class-bottleneck-analysis], [deberta-2stage-underperforms-tfidf]]
summary: MI≥0.18 기준 feature 정제 + 3개 병목 그룹 타겟 interaction features 추가
---

# Feature Engineering v2: Interaction Features

## Context
exp_001 baseline (TF-IDF 25K + LightGBM) CV=0.6605에서 정체.
PCA+MI 분석으로 무의미 feature 7개 제거, 25개 신규 feature 추가 → exp_010 CV≈0.6838 (+2.3%p).
추가 개선을 위해 class별 병목 분석 수행.

## Options considered
1. DeBERTa 2-stage (soft probs → LightGBM) → 시도했으나 0.6563으로 하락
2. 더 큰 encoder (DeBERTa-base, Qwen) → zip 크기 제약, decoder는 비효율
3. **병목 class 타겟 interaction features** → 분석 기반 정밀 개선
4. class_weight / focal loss → 클래스 불균형 대응

## Decision
**Option 3 + 4 조합**. 병목 분석에서 발견한 3개 그룹 타겟 interaction features:

### Priority 1: 돌려봐 3파전 (run_bash/run_tests/lint)
- `run_verb_x_last`: "돌려봐" 동사 × last_action 조합 (categorical)
- `lint_kw_x_last`: "타입체크/lint" × last_action 조합 (categorical)
- `is_run_verb`, `is_lint_keyword`: binary
- `run_verb_x_failed`: 실행동사 × 이전실패

### Priority 2: 탐색 4형제 (read/grep/glob/list)
- `explore_bigram`: 연속 탐색 action의 bigram (precision 82~85%)
- `last_explore_type`: 마지막 탐색 action 종류
- `explore_streak`: 연속 탐색 action 수
- `explore_depth`: 총 탐색 action 수

### Priority 3: plan vs ask
- `error_x_question`: 에러언급 × 질문 4분류 (categorical)
- `has_error_in_prompt`: 에러명 언급 여부
- `question_x_no_history`: 질문 × history 없음
- `question_x_early_turn`: 질문 × 초반 턴

MI ≥ 0.18 기준으로 무의미 feature 제거:
user_tier, cnt_respond_only, prompt_has_exclaim, prompt_has_code, prompt_is_korean, n_actions, prompt_len_chars

## Consequences
- **긍정**: 병목 class 집중 개선 → Macro-F1 전체 +2~3%p 예상
- **긍정**: LightGBM은 interaction feature를 자동으로 못 찾으므로 명시적 추가가 효과적
- **부정**: feature 수 65→78 증가 (관리 복잡성)
- **미확인**: class_weight 효과는 실험 대기 중 (exp_013 classweight/focal 버전)
- **총 feature**: TF-IDF 25K + Dense 78 = 25,078개
