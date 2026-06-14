---
id: conventions
type: entity
created: 2026-06-14
updated: 2026-06-14
tags: [meta, wiki-rules]
related: [[index]]
summary: LLM Wiki 작성 규약 — Claude Code가 wiki를 읽고 쓸 때 따를 규칙
---

# Wiki Conventions

이 문서는 LLM(Claude Code)이 wiki를 읽고 쓸 때 따를 규약을 정의한다.

## Page Types & Required Sections

### Entity (entities/)
개념, 도구, 시스템, 모델 등의 정의 페이지.

필수 섹션:
- `## Definition` — 무엇인가
- `## Why it matters` — 왜 중요한가 (이 프로젝트 맥락에서)
- `## Related` — 관련 entity/decision/lesson 링크
- `## History` — 변경 이력 (날짜별)

### Decision (decisions/)
ADR(Architecture Decision Record) 스타일 결정 기록.

필수 섹션:
- `## Context` — 결정이 필요했던 상황
- `## Options considered` — 고려한 대안들
- `## Decision` — 최종 선택과 이유
- `## Consequences` — 예상되는 결과 (긍정/부정)

### Lesson (lessons/)
실수, 디버깅, 시행착오에서 얻은 교훈.

필수 섹션:
- `## Symptom` — 무엇이 잘못되었나
- `## Root cause` — 근본 원인
- `## Fix` — 어떻게 해결했나
- `## Generalization` — 다른 상황에도 적용 가능한 일반화된 교훈

### Context (context/)
프로젝트별 컨텍스트 스냅샷. 자유 형식이되 다음을 포함:
- 현재 상태 요약
- 주요 수치 (best CV, LB score 등)
- 다음 단계 방향

### Session (sessions/)
`/compound` 실행 시 자동 생성되는 세션 로그.
- 해당 세션에서 한 작업 요약
- 추출된 decisions/lessons/entities 목록
- 링크

## Conflict Resolution

새 정보가 기존 페이지와 충돌하면:
1. **새 페이지를 만들지 말 것**
2. 기존 페이지에 `## Conflict yyyy-mm-dd` 섹션 추가
3. 이전 내용과 새 내용을 모두 기록
4. 어느 쪽이 맞는지 판단 근거 명시

## Naming Convention

- 파일명: `kebab-case.md` (예: `lightgbm-baseline.md`)
- ID: 파일명과 동일 (확장자 제외)
- 링크: `[[kebab-case-id]]` 형식

## Frontmatter (필수)

모든 페이지는 다음 frontmatter를 반드시 포함:
```yaml
---
id: <kebab-case-slug>
type: entity | decision | lesson | context | session
created: <ISO date>
updated: <ISO date>
tags: [topic1, topic2]
related: [[other-page-id]]
summary: <한 줄 요약>
---
```

## Quality Rules

- 모호한 표현 금지 — 구체적 수치, 파일 경로, 실험 ID 포함
- 한 페이지에 하나의 주제만
- 양방향 링크 유지 — A가 B를 참조하면, B의 related에도 A 추가
- 날짜는 항상 절대 날짜 사용 (상대 날짜 금지)
