---
id: session-grouping-requires-groupkfold
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [cv, leakage, groupkfold, sessions]
related: [[cv-protocol-stratifiedgroupkfold], [[data-is-jsonl-not-csv]]]
summary: 세션 단위 그룹 누수를 막으려면 StratifiedGroupKFold(group=session)가 필수 — plain StratifiedKFold는 CV를 과대평가한다.
---

# Session grouping requires StratifiedGroupKFold

## Symptom
초기 스킬 기본값이 `StratifiedKFold(n_splits=5)` (14-class stratify only) 였다.
이 방식은 같은 세션의 여러 step 행을 서로 다른 폴드에 흩뿌린다. 결과적으로
검증 폴드가 학습 폴드에 이미 등장한 세션의 workspace/meta/history prefix를
그대로 보게 되어 CV Macro-F1가 실제 일반화보다 부풀려진다.

## Root cause
데이터는 9,429개 세션이 70,000행에 걸쳐 있고, 행의 **99.69%가 멀티스텝 세션**
소속이다. 세션 id는 `id.rsplit("-step",1)[0]` 로 얻는다. 같은 세션의 step들은
- 동일한 session_meta / workspace 상태를 공유하고
- history가 이전 step의 prefix를 중첩 포함한다.
따라서 세션을 그룹으로 묶지 않으면 폴드 간 정보 누수가 발생한다.

## Fix
CV를 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`,
`group = 세션 id`로 고정한다. 폴드마다 그룹 무중복을 assert한다:
```python
for tr, va in sgkf.split(X, y, groups):
    assert set(groups[tr]).isdisjoint(set(groups[va])), "session leak across fold"
```
`/eval`은 실험이 plain KFold/StratifiedKFold를 썼거나 assert가 없으면 REJECT한다.

## Generalization
행이 상위 엔티티(세션/사용자/문서)에 속하고 그 엔티티 내부에서 특징이
공유·중첩되면, 반드시 그 엔티티를 그룹으로 하는 GroupKFold 계열을 써야 한다.
클래스 불균형까지 있으면 StratifiedGroupKFold. "행 단위 stratify"만으로는
그룹 누수를 절대 막을 수 없다.
