---
id: class-bottleneck-analysis
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [eda, macro-f1, class-analysis, feature-engineering]
related: [[cv-protocol-stratifiedgroupkfold], [deberta-2stage-underperforms-tfidf]]
summary: 14 class 중 Macro-F1 병목은 3개 그룹 — 탐색4형제, 돌려봐3파전, plan/ask 혼동
---

# Class별 Macro-F1 병목 분석

## Symptom
baseline CV Macro-F1 0.66에서 정체. 일부 class는 F1>0.9인데 일부는 <0.5.

## Root cause

### last_action 단독 판별력: 사실상 0개
모든 class에서 top last_action의 precision이 30% 미만.
가장 높은 edit_file조차 29%. 14개 class가 동일 last_action을 공유.

### 3개 병목 그룹

**병목 1: 탐색 4형제 (read_file/grep_search/glob_pattern/list_directory)**
- 키워드 완전 겹침: "보여줘", "열어봐", "있는지", "찾아줘" → 4개 class 모두 상위
- prompt 길이, 파일 확장자 언급 비율도 동일 (avg 13.3 words, ext 21~24%)
- list_directory의 42%가 turn=1(history 없음) → 구분 단서 전무
- **유일한 구분**: history bigram (list→grep→read→glob 탐색 시퀀스)
  - list_directory→grep_search: read_file precision 85%
  - grep_search→grep_search: glob_pattern precision 82%
  - grep_search→glob_pattern: glob_pattern precision 82%

**병목 2: "돌려봐" 3파전 (run_bash/run_tests/lint_or_typecheck)**
- "돌려봐/돌려보자/돌려서/run" 키워드를 3개 class가 공유
- **last_action으로 분리 가능** (3파전 내):
  - last=NONE/write_file → run_bash (84%)
  - last=edit_file → run_tests (63%)
  - last=apply_patch → lint_or_typecheck (45%)
- lint 전용 키워드: "typecheck"/"타입체크" precision 100% (소수)
- turn 차이: run_bash(4.2) < run_tests(6.4) < lint(7.2)

**병목 3: plan_task vs ask_user**
- 키워드 겹침: "단계부터", "쪼개줘", "어디부터" 공유
- ask_user 고유 신호: "모르겠어"(60%), "도와줄래?"(59%), error_name 10.6%
- plan_task: no_history 41.8%, 세션 초반
- ask_user: no_history 23.3%, 세션 중반

### 노이즈
- web_search: 18.5% (동일 prompt→다른 label)
- lint_or_typecheck: 14.8%
- list_directory: 9.5%

## Fix
exp_013에서 3가지 interaction feature 그룹 추가:

1. **돌려봐 3파전**: `run_verb_x_last` (실행동사×last_action), `lint_kw_x_last`, `is_run_verb`, `is_lint_keyword`, `run_verb_x_failed`
2. **탐색 4형제**: `explore_bigram` (연속 탐색 bigram), `last_explore_type`, `explore_streak`, `explore_depth`
3. **plan vs ask**: `error_x_question` (에러×질문 4분류), `has_error_in_prompt`, `question_x_no_history`, `question_x_early_turn`

## Generalization
- Macro-F1에서는 worst class가 전체를 지배 — 상위 class 개선보다 하위 class 집중이 효율적
- last_action 단독은 무의미하나 **interaction feature (last_action × prompt keyword)**는 강력
- 합성 데이터라 class간 "경계"가 규칙적 → rule-aware interaction feature가 유효
- 14 class를 3개 병목 그룹으로 분해하면 각각 맞춤 전략 가능
