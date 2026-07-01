---
id: cv-protocol-stratifiedgroupkfold
type: decision
created: 2026-07-01
updated: 2026-07-01
tags: [cv, protocol, groupkfold, leakage]
related: [[session-grouping-requires-groupkfold], [[data-is-jsonl-not-csv]]]
summary: 모든 실험의 CV를 StratifiedGroupKFold(5, shuffle, seed=42, group=session)로 고정한다.
---

# CV protocol: StratifiedGroupKFold (group = session)

## Context
데이터는 9,429개 세션이 70,000행에 걸쳐 있고 행의 99.69%가 멀티스텝 세션 소속이다.
세션 내 step들은 session_meta/workspace를 공유하고 history prefix가 중첩된다.
기존 스킬 기본값(StratifiedKFold, 행 단위 stratify)은 같은 세션을 여러 폴드에
분산시켜 폴드 간 누수를 일으키고 CV Macro-F1를 과대평가한다.

## Options considered
- **plain KFold** — 불균형·그룹 모두 무시. 부적합.
- **StratifiedKFold** — 14-class 균형만 맞춤, 그룹 누수 방치. (기존 기본값) 부적합.
- **GroupKFold(session)** — 그룹 누수는 막지만 클래스 stratify 없음.
- **StratifiedGroupKFold(session)** — 클래스 균형 + 세션 그룹 무중복 동시 확보. 채택.

## Decision
모든 실험 CV = `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`,
`group = 세션 id = id.rsplit("-step",1)[0]`. 폴드마다 그룹 무중복 assert 필수:
`set(groups[tr]).isdisjoint(set(groups[va]))`. data_docs "Modeling Protocol
(LOCKED)"에 명시되어 있으며 `/dev` `/auto` `/plan` `/eda` `/eval` 및 에이전트
정의에 전파했다. `/eval`은 plain KFold/StratifiedKFold 사용 시 REJECT한다.

## Consequences
- (+) CV가 세션 단위 일반화를 정직하게 반영 → CV→LB 상관 신뢰도 상승.
- (+) 실험 간 비교가 일관됨(동일 seed·분할 규약).
- (−) 폴드 클래스 비율이 완벽히 균형은 아닐 수 있음(그룹 제약 때문). 허용 가능.
- (−) 기존에 StratifiedKFold로 얻은 과거 CV 수치는 낙관적이므로 그대로 비교 금지.
