---
id: daily-quota-20-team
type: decision
created: 2026-07-01
updated: 2026-07-01
tags: [submission, quota, policy]
related: [[cv-protocol-stratifiedgroupkfold]]
summary: DACON 일일 제출 한도는 팀 기준 20회다 (기존 스킬의 10회 표기는 오류) — 모든 quota 계산·표시를 20으로 고정.
---

# Daily submission quota is 20 (team)

## Context
여러 스킬(`/auto` `/plan` `/rank` `/status` `/submit-result`)이 일일 제출 한도를
**10회**로 표기/계산하고 있었다(`/10`, "10 submissions", `remaining = 10 - ...`).
실제 대회의 팀 기준 일일 한도는 **20회**다. quota를 잘못 계산하면 남은 슬롯을
과소평가해 제출 기회를 낭비하거나 정책 판단이 틀어진다.
(single source of truth인 `competition_meta.yaml`의 한도는 이미 20으로 수정됨.)

## Options considered
- 10 유지 — 사실과 불일치. 기각.
- 20으로 통일 — 실제 규정과 일치. 채택.

## Decision
모든 스킬 문서의 quota 표시·계산을 **20/일(팀)**으로 통일한다:
- `/rank`: `remaining = max(0, 20 - used)`, 표시 `<used>/20`.
- `/status`: `Daily quota <used>/20`, "quota almost exhausted" 경고 임계를 18로.
- `/auto`: 상황요약 `quota=<used>/20`.
- `/plan`: `Quota today: <used>/20`.
- `/submit-result`: install_error는 20회 팀 quota에 불차감.
`competition_meta.yaml`은 이미 반영되어 있어 이 세션에서 수정하지 않는다.

## Consequences
- (+) 남은 제출 슬롯 계산이 정확 → 하루 후보 제출 스케줄링이 올바름.
- (+) 스킬 문서와 competition_meta.yaml 간 표기 불일치 제거.
- (−) 없음(순수 사실 정정). install_error 불차감 규칙은 그대로 유지.
