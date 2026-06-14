현재 세션의 결정/교훈/컨텍스트를 LLM Wiki에 compound한다.

세션 종료 전 반드시 실행하여 지식을 축적한다.

## Step 1: 추출

이 세션에서 발생한 것을 분류:

### Decisions (결정)
- 내가 명시적으로 선택한 것
- 대안 + 선택 이유 + 트레이드오프 포함

### Lessons (교훈)
- 실수했거나 잘못 알고 있었던 것
- 증상 → 원인 → 수정 → 일반화

### Context (컨텍스트)
- 이 프로젝트/도메인에 대해 새로 알게 된 사실
- best CV, LB score 등 핵심 수치 변화

### Entities (개념)
- 처음 등장한 개념/도구/시스템/모델

## Step 2: 중복 검색

각 항목에 대해 `wiki/` 디렉토리에서 Grep으로 이미 관련 페이지가 있는지 확인.
- 있으면 → 해당 페이지에 append (`updated` 날짜 갱신)
- 없으면 → 새 페이지 생성

## Step 3: 쓰기

`wiki/_meta/conventions.md`의 규약을 그대로 따라 마크다운 작성.

### 파일 위치:
- Decisions → `wiki/decisions/<kebab-case-id>.md`
- Lessons → `wiki/lessons/<kebab-case-id>.md`
- Entities → `wiki/entities/<kebab-case-id>.md`
- Context → `wiki/context/<kebab-case-id>.md`
- Session log → `wiki/sessions/session-<YYYY-MM-DD>-<NNN>.md`

### 필수사항:
- 모든 페이지에 frontmatter 포함 (conventions.md 참조)
- 모든 페이지 끝에 `related: [[...]]` 양방향 링크 추가
- 구체적 수치/파일 경로/실험 ID 포함 (모호한 표현 금지)

## Step 4: 인덱스 갱신

`wiki/_meta/index.md`의 해당 섹션에 새로 추가/수정된 페이지 목록 업데이트.

형식:
```
- [[page-id]] — 한 줄 요약 (YYYY-MM-DD)
```

## Step 5: Bridge Files 갱신

다음 bridge files도 업데이트:
1. `logs/orchestrator_state.json` — best_cv, current_phase 등 갱신
2. `logs/experiment_digest.md` — 새 실험이 있었다면 갱신
3. `logs/insights.jsonl` — 새 insight가 있었다면 추가

## Step 6: 리포트

사용자에게 다음 형식으로 보고:

```
📝 Compound 완료
─────────────────
추가된 페이지: N개
업데이트된 페이지: M개
발견된 충돌: K개
─────────────────
[상세 목록]
```
