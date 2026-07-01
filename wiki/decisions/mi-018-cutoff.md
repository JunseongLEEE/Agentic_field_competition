---
id: mi-018-cutoff
type: decision
created: 2026-07-01
updated: 2026-07-01
tags: [feature-selection, mutual-information, pca]
related: [[feature-engineering-v2-interaction]]
summary: MI ≥ 0.18 기준으로 structural features 7개만 생존, 나머지 삭제
---

# MI ≥ 0.18 Feature Cutoff

## Context
PCA + MI 분석에서 많은 feature가 label과 무관함을 확인.
prompt_intent(MI=0.438)가 최고이나 66%가 GENERAL(신호 없음).

## Options considered
1. MI > 0.1 기준 → 15+ features 생존, 노이즈 포함 가능
2. **MI ≥ 0.18 기준** → 7개 features만 생존
3. MI > 0.25 기준 → 5개만 생존, 너무 공격적

## Decision
**MI ≥ 0.18 기준** 채택. 생존 features:

| Feature | MI |
|---------|-----|
| prompt_intent | 0.438 |
| action_trigram | 0.426 |
| action_bigram | 0.396 |
| turn_action | 0.304 |
| last_action_status | 0.261 |
| last_action | 0.252 |
| second_last_action | 0.211 |

삭제된 주요 features:
- history_len (0.187) — 근소하게 아래
- third_last_action (0.172)
- modify_ratio (0.146), result_cat (0.142)
- workspace 관련 전부 (MI < 0.05)

## Consequences
- DeBERTa 2-stage (exp_011/012)에서는 이 7개만 structural feature로 사용
- exp_010은 TF-IDF와 결합하므로 MI < 0.18 features도 일부 유지 (LightGBM이 자체 선택)
- 50% 이상 커버 안 되는 feature(prompt_intent의 GENERAL 66%)는 OOD 일반화에 불리
