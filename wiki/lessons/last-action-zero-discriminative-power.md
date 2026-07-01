---
id: last-action-zero-discriminative-power
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [eda, last-action, feature-importance]
related: [[class-bottleneck-analysis], [feature-engineering-v2-interaction]]
summary: last_action 단독은 14 class 중 0개를 40%+ precision으로 판별 불가. 조합(bigram/interaction)으로만 유효.
---

# last_action 단독 판별력 = 0

## Symptom
MI=0.252로 상위 feature이지만, 실제 class 판별에서 단독으로 강한 class가 하나도 없음.

## Root cause
모든 class의 top last_action에서 14개 경쟁 class가 섞임:
- 최고 edit_file: last=read_file일 때 29% (나머지 71%는 다른 class)
- list_directory: last=NONE일 때 20%
- web_search: last=read_file일 때 3%

| 판별력 | 개수 |
|--------|------|
| ★★★ ≥40% | 0개 |
| ★★ 25-40% | 1개 (edit_file) |
| ★ 15-25% | 5개 |
| ✗ <15% | 8개 |

## Fix
last_action을 단독으로 쓰지 말고 **interaction feature**로:
- `action_bigram` (MI=0.396): second__last 조합
- `action_trigram` (MI=0.426): third__second__last 조합
- `run_verb_x_last`: prompt 키워드 × last_action
- `explore_bigram`: 연속 탐색 action 시퀀스

## Generalization
- MI가 높다고 단독 판별력이 높은 건 아님 — MI는 전체 분포 기준, class별 precision과 다름
- 약한 feature도 **interaction으로 조합**하면 강한 신호 (bigram MI=0.40 > last_action MI=0.25)
- Feature engineering에서 "이 feature가 어떤 class를 구분하는가"를 class별로 분석해야 함
