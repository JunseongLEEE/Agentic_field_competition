# Claude Code Agentic Field 통합 셋업 프롬프트

> 아래 전체를 그대로 복사해서 Claude Code 새 세션에 붙여넣으세요.
> Superpowers가 이미 설치되어 있다면 brainstorming/planning skill을 활용하고, 아니면 Phase 1에서 같이 설치합니다.

---

# 🎯 미션

내 Claude Code 환경에 다음 4가지를 통합한 "agentic field"를 구축한다.

1. **Superpowers** — `github.com/obra/superpowers` (Brainstorm → Plan → Work → Review)
2. **Compound Engineering 80/20** — plan/review에 80%, work/compound에 20%
3. **Karpathy LLM Wiki** — Obsidian 기반 LLM이 직접 관리하는 마크다운 위키
4. **QMD** — `github.com/tobi/qmd` 로컬 하이브리드 검색 (BM25 + 벡터 + LLM 리랭킹), MCP 서버로 노출

목표는 단순히 도구를 설치하는 게 아니라, **세 가지가 서로 참조하면서 compound되는 단일 워크플로우**를 만드는 것이다. 즉:

- Superpowers의 `brainstorming` skill이 시작 시 **QMD를 통해 LLM Wiki를 먼저 검색**해서 과거 컨텍스트를 끌어옴
- `planning` skill이 LLM Wiki의 관련 결정/레슨런을 참조해 계획에 반영
- 작업 끝나면 `/compound` 커스텀 커맨드가 **결정 + 레슨런 + 컨텍스트**를 LLM Wiki에 적재
- 새 세션이 시작되면 다시 QMD를 통해 자동 참조됨

---

# 🧭 작업 원칙 (반드시 따를 것)

**이 미션 자체를 Compound Engineering 방식으로 진행한다.**

1. **코드부터 짜지 말 것.** 먼저 Phase 0에서 내 환경을 brainstorm하고, 모호한 게 있으면 한 번에 모두 묻지 말고 단계적으로 물어볼 것.
2. **Phase 단위로 게이트를 둘 것.** 각 Phase 시작 전 plan을 짧게 보여주고 내 승인(`go` 또는 수정 요청)을 받은 후 진행.
3. **각 task는 2~5분 단위로 쪼개고**, 파일 경로와 검증 방법을 미리 적어둘 것. (Superpowers 설치 후라면 subagent로 위임 가능)
4. **모든 결정과 레슨런은 마지막 Phase에서 LLM Wiki에 compound한다.** 이 셋업 자체가 첫 번째 compound 사이클이 된다.
5. **확신이 없으면 추측하지 말고 물어볼 것.** 특히 경로, 버전, 기존 설정 충돌 가능성에 대해서.

---

# Phase 0: 환경 파악 (brainstorm 단계)

다음을 순서대로 확인한다. 일부는 명령어로 확인 가능하고, 일부는 나에게 직접 물어봐야 한다.

## 0-1. 명령어로 확인할 것
```bash
uname -a                          # OS
node --version && npm --version   # QMD 설치에 필요
claude --version                  # Claude Code 버전
ls -la ~/.claude 2>/dev/null      # 기존 설정 유무
ls ~/.claude/plugins 2>/dev/null  # 기존 플러그인 유무
```

## 0-2. 나에게 물어볼 것
- **Obsidian vault**: 이미 쓰고 있는 vault가 있는가? 경로는? 없다면 어디에 새로 만들지?
- **기존 `~/.claude` 백업**: 기존 설정이 있다면 백업할까?
- **QMD 모델 다운로드**: 첫 실행 시 약 2GB GGUF 모델이 자동 다운로드된다. 진행해도 되는가? 디스크/네트워크 OK?
- **LLM Wiki 초기 시드**: 처음부터 비어있는 위키로 시작할지, 아니면 내가 가진 기존 노트/문서로 시드할지?
- **Compound Engineering 플러그인**: Every의 공식 플러그인도 같이 설치할지(Superpowers와 일부 기능 겹침), 아니면 Superpowers + 커스텀 `/compound` 커맨드만으로 갈지?

이 정보를 받기 전까지는 Phase 1로 넘어가지 말 것. **단, 한 번에 다 묻지 말고 핵심부터 차례로** (Obsidian 경로 → 백업 → 모델 다운로드 → 시드 전략 → 플러그인 선택).

---

# Phase 1: Superpowers 설치 및 검증

## 1-1. 설치
```bash
# Anthropic 공식 마켓플레이스 경유
/plugin marketplace add obra/superpowers-marketplace
/plugin install superpowers@superpowers-marketplace
```
설치 후 Claude Code 재시작 안내.

## 1-2. 검증
재시작 후 세션 시작 시 `<session-start-hook>` 부트스트랩이 주입되는지 확인. 다음 skill들이 보이면 OK:
- `brainstorming`
- `planning`
- `subagent-driven-development`
- `test-driven-development`
- `requesting-code-review`

## 1-3. 게이트
사용자에게 "Superpowers 설치 완료, 다음 Phase로 갈까요?" 묻고 승인 받음.

---

# Phase 2: LLM Wiki 구조 설계 및 초기화

## 2-1. Obsidian vault 구조
다음 디렉토리 구조를 vault 안에 만든다 (이미 vault가 있다면 충돌 피해서 `llm-wiki/` 서브폴더 안에).

```
<vault>/llm-wiki/
├── _meta/
│   ├── conventions.md          # 이 wiki의 작성 규약 (LLM이 읽음)
│   └── index.md                # entity 페이지 인덱스
├── entities/                   # 개념/시스템/사람/도구 단위 페이지
├── decisions/                  # 작업 중 내린 결정 (ADR 스타일)
├── lessons/                    # 실수/디버깅에서 얻은 교훈
├── context/                    # 프로젝트별 컨텍스트 스냅샷
└── sessions/                   # 세션 로그 (compound 입력 원본)
```

## 2-2. Frontmatter 컨벤션
모든 페이지는 다음 frontmatter를 갖는다 (QMD가 메타데이터로 활용):
```yaml
---
id: <kebab-case-slug>
type: entity | decision | lesson | context | session
created: <ISO date>
updated: <ISO date>
tags: [topic1, topic2]
related: [[other-page-id]]
summary: <한 줄 요약 — QMD 검색 결과 미리보기에 노출됨>
---
```

## 2-3. `_meta/conventions.md`
LLM(나, Claude Code)이 LLM Wiki를 읽고 쓸 때 따를 규약을 명문화한 파일을 직접 작성해 vault에 넣는다. 최소한 포함할 것:
- entity 페이지에는 `## Definition / ## Why it matters / ## Related / ## History` 섹션 강제
- decision 페이지는 ADR 포맷 (Context / Decision / Consequences)
- lesson 페이지는 (Symptom / Root cause / Fix / Generalization)
- 새 정보가 기존 페이지와 충돌하면 **새 페이지 만들지 말고** 기존 페이지에 `## Conflict yyyy-mm-dd` 섹션 추가

## 2-4. 게이트
구조 보여주고 승인 받음.

---

# Phase 3: QMD 설치 및 MCP 통합

## 3-1. 설치
```bash
npm install -g @tobi/qmd
qmd --version
```

## 3-2. 컬렉션 등록
```bash
qmd collection add <vault>/llm-wiki --name wiki --pattern "**/*.md"
qmd context add wiki "내 개인 LLM Wiki. 작업 결정, 레슨런, 엔티티 정의를 저장한다."
qmd index wiki        # 최초 인덱싱 (모델 다운로드 발생)
```

## 3-3. MCP 서버 설정
`~/.claude/mcp.json` (또는 프로젝트 `.mcp.json`)에 추가:
```json
{
  "mcpServers": {
    "qmd": {
      "command": "qmd",
      "args": ["serve", "--mcp", "--port", "8181"]
    }
  }
}
```

## 3-4. 검증
Claude Code 재시작 후 MCP 도구가 `mcp__qmd__search` 같은 형태로 보이는지 확인. 테스트 쿼리:
```
mcp__qmd__search("conventions")
```
`_meta/conventions.md`가 결과에 떠야 정상.

## 3-5. 게이트
검증 결과 보여주고 승인.

---

# Phase 4: `/compound` 커스텀 커맨드 작성

## 4-1. 파일 위치
`~/.claude/commands/compound.md` 생성. 이 커맨드는 현재 세션에서 진행된 작업을 LLM Wiki에 적재한다.

## 4-2. 커맨드 본문 (이 내용을 그대로 파일에 쓸 것)
```markdown
---
description: 현재 세션의 결정/레슨런/컨텍스트를 LLM Wiki에 compound
allowed-tools: [Read, Write, Edit, mcp__qmd__search]
---

# /compound

지금까지의 대화를 다음 순서로 처리한다.

## Step 1: 추출
이 세션에서 발생한 것을 분류:
- **Decisions**: 내가 명시적으로 선택한 것 (대안 + 선택 이유 + 트레이드오프)
- **Lessons**: 실수했거나 잘못 알고 있었던 것 (증상 → 원인 → 수정 → 일반화)
- **Context**: 이 프로젝트/도메인에 대해 새로 알게 된 사실
- **Entities**: 처음 등장한 개념/도구/시스템

## Step 2: 기존 wiki 검색 (중복 방지)
각 항목에 대해 `mcp__qmd__search`로 vault 안에 이미 관련 페이지가 있는지 확인.
- 있으면 → 해당 페이지에 append (frontmatter `updated` 갱신)
- 없으면 → 새 페이지 생성

## Step 3: 쓰기
`_meta/conventions.md`의 규약을 그대로 따라 마크다운 작성.
모든 페이지 끝에는 `related: [[...]]` 양방향 링크 추가.

## Step 4: 인덱스 갱신
```bash
qmd index wiki --incremental
```

## Step 5: 리포트
사용자에게 다음 형식으로 보고:
- 추가된 페이지: N개
- 업데이트된 페이지: M개  
- 발견된 충돌: K개 (있으면 경로 나열)
```

## 4-3. 게이트
파일 보여주고 승인 후 저장.

---

# Phase 5: Superpowers Skill 보강

## 5-1. 목표
Superpowers의 기본 `brainstorming`과 `planning` skill이 **시작 시 자동으로 LLM Wiki를 검색**하도록 보강한다.

## 5-2. 방법
직접 Superpowers 소스를 수정하지 말고, **오버라이드 skill을 추가**한다 (업스트림 업데이트 안전).

`~/.claude/skills/wiki-aware-brainstorm/SKILL.md` 생성:
```markdown
---
name: wiki-aware-brainstorm
description: brainstorming skill의 첫 단계로 QMD를 통해 LLM Wiki를 검색해 과거 컨텍스트를 끌어온다. brainstorming이 호출되기 전에 자동으로 실행되어야 함.
triggers: [brainstorm, 새 작업 시작, plan 요청]
---

# Wiki-aware brainstorm

새 작업이 들어오면 brainstorming skill을 본격 실행하기 전에:

1. 작업 설명에서 핵심 키워드 3~5개 추출
2. 각 키워드로 `mcp__qmd__search(keyword, top_k=5)` 호출  
3. 검색 결과 중 관련도 높은 것 (LLM 리랭킹 점수 0.7 이상) 요약해서 컨텍스트로 주입
4. 이 결과를 brainstorming skill의 입력에 prepend:
   > "과거 wiki에서 관련 항목을 찾았습니다:
   >  - [[decision-xxx]]: ...
   >  - [[lesson-yyy]]: ...
   > 이를 고려해 brainstorm을 진행합니다."
5. 그 후 원래의 brainstorming skill로 제어 이전
```

마찬가지로 `~/.claude/skills/wiki-aware-planning/SKILL.md`도 작성 (planning skill 시작 시 wiki 검색).

## 5-3. 게이트
두 skill 파일 보여주고 승인.

---

# Phase 6: End-to-End 검증

다음 더미 작업을 실제로 돌려서 전체 파이프라인이 흐르는지 확인한다.

## 시나리오
> "내 프로젝트의 Python 패키지 매니저를 pip에서 uv로 마이그레이션하고 싶다."

## 검증 체크리스트
- [ ] `wiki-aware-brainstorm`이 자동 실행되어 QMD로 wiki를 검색하는가? (비어있어도 "no relevant entries" 로그가 떠야 함)
- [ ] Superpowers `brainstorming`이 단계적 질문으로 요구사항을 좁히는가?
- [ ] `planning`이 task를 2~5분 단위로 쪼개고 파일 경로/검증 방법을 명시하는가?
- [ ] 더미 task 1개를 subagent에게 위임하면 메인 컨텍스트가 깨끗하게 유지되는가?
- [ ] 작업 후 `/compound` 실행 시 decisions/lessons/entities/context가 vault에 생성되는가?
- [ ] `qmd index wiki --incremental` 후 다음 세션에서 검색하면 방금 적재한 내용이 잡히는가?

## 게이트
모든 체크 통과 시에만 셋업 완료 선언.

---

# Phase 7 (이 셋업의 compound)

마지막으로, **이 셋업 작업 자체에서 얻은 교훈을 LLM Wiki에 적재한다.**

`/compound` 실행하여 다음을 wiki에 기록:
- **entities**: `superpowers`, `qmd`, `llm-wiki`, `compound-engineering` 각 1페이지
- **decisions**: Phase 0에서 내가 선택한 것들 (vault 위치, Compound Engineering 플러그인 사용 여부 등)
- **lessons**: 셋업 중 발생한 시행착오 (있었다면)
- **context**: `_meta/setup-history.md`에 이 전체 셋업의 타임라인 요약

이로써 다음 작업부터는 이 환경에 대한 모든 사실을 wiki에서 자동으로 참조 가능해진다.

---

# 🚦 시작

준비됐다면 **Phase 0-1부터 시작**해라. 명령어로 환경부터 확인하고, 그다음 0-2의 질문을 **한 번에 하나씩** 나에게 던져라. 모든 Phase 끝에 게이트가 있다는 걸 잊지 말 것.