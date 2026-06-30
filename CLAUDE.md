# Competition Agentic Field

## Philosophy: Compound Engineering 80/20

이 시스템은 **Compound Engineering** 원칙을 따른다:
- **80%** — Plan & Review (brainstorm → plan → evaluate → compound)
- **20%** — Work (implement → run)

모든 작업은 과거 지식을 먼저 검색하고, 끝나면 새 지식을 축적한다.
새 세션은 bridge files + wiki 검색으로 즉시 컨텍스트를 복구한다.

---

## Project Overview
AI competition (DACON, SW중심대학협의회) experiment management system.
**Code submission** format — submit model weights + inference script, NOT CSV predictions.
Each experiment is isolated, reproducible, and tracked.

---

## DACON Code Submission Rules (CRITICAL)

Every submission must be a zip with EXACTLY this structure:
```
submit.zip
├── model/              # Trained model weights (saved locally)
│   └── model.pt        # File names are flexible
├── script.py           # Inference ONLY — no training code
└── requirements.txt    # Extra packages (beyond server defaults)
```

### script.py constraints:
- Reads test data from `data/test.csv` (server provides)
- Writes predictions to `output/submission.csv`
- **OFFLINE ONLY** — no internet access, no API calls, no from_pretrained("model-name")
- All model files must be loaded from `model/` directory via local paths
- Must have `if __name__ == '__main__'` block

### Every experiment produces TWO scripts:
- `train.py` — local training + CV (never submitted)
- `script.py` — server-side inference only (submitted)

---

## Directory Structure
```
.claude/skills/      — Slash command skills
scripts/             — Python utility scripts
competition_meta.yaml — 대회 마감일/제출 quota/제출 로그 (single source of truth)
data_docs/           — 데이터셋 문서 (생성 방법, 오픈소스 출처, 도메인 노트)
  ├── README.md
  ├── dataset_overview.md     # 데이터 스키마/크기
  ├── generation_methodology.md  # 사용자 작성: 데이터 만든 방법
  ├── domain_notes.md         # EDA로 발견한 패턴 누적
  └── references/             # 참고 오픈소스/논문
experiments/         — One folder per experiment (exp_001/, exp_002/, ...)
  └── exp_NNN/
      ├── config.yaml
      ├── train.py         # Training + CV (local only)
      ├── script.py        # Inference (submitted to DACON)
      ├── requirements.txt
      ├── model/           # Trained weights (submitted)
      ├── models/          # Per-fold models (not submitted)
      ├── SUMMARY.md       # Experiment memory card
      └── train_log.json   # CV results
wiki/                — LLM Wiki (Compound Knowledge Base)
  ├── _meta/
  │   ├── conventions.md   # Wiki 작성 규약
  │   └── index.md         # Entity 페이지 인덱스
  ├── entities/            # 개념/도구/시스템 정의
  ├── decisions/           # ADR 스타일 결정 기록
  ├── lessons/             # 실수/디버깅 교훈
  ├── context/             # 프로젝트별 컨텍스트 스냅샷
  └── sessions/            # 세션 로그 (compound 입력 원본)
logs/                — Bridge files for agent communication
  ├── orchestrator_state.json
  ├── experiment_digest.md
  ├── insights.jsonl
  ├── cycle_history.jsonl
  └── agent_messages.jsonl
submissions/         — Packaged zip files ready for DACON upload
agents/              — Agent role definitions
```

---

## Compound Workflow (핵심 사이클)

모든 작업은 이 사이클을 따른다:

```
┌─────────────────────────────────────────────────┐
│  1. SEARCH  — wiki + bridge files에서 과거 지식 검색  │
│  2. BRAINSTORM — 과거 컨텍스트 기반으로 아이디어 탐색   │
│  3. PLAN    — 실험 계획 (80% 시간)                    │
│  4. WORK    — 구현 + 실행 (20% 시간)                  │
│  5. REVIEW  — 결과 평가 + 리키지 체크 (80% 시간)       │
│  6. COMPOUND — 결정/교훈/컨텍스트를 wiki에 적재        │
└─────────────────────────────────────────────────┘
```

### Session Start Protocol
새 세션이 시작되면 반드시:
1. `python scripts/check_time_state.py` 실행 → 마감까지 일수 + 오늘 제출 quota 파악
2. `logs/orchestrator_state.json` 읽어 현재 전략 상태 파악
3. `logs/experiment_digest.md` 읽어 전체 실험 현황 파악
4. `logs/insights.jsonl` 최근 5개 읽어 CV-LB 패턴 파악
5. `data_docs/` 모든 .md 읽어 데이터셋 도메인 컨텍스트 확보
6. `wiki/` 에서 관련 decisions/lessons 검색하여 과거 컨텍스트 주입

### Session End Protocol
작업 종료 시 `/compound` 실행하여:
1. 이 세션의 결정/교훈/새 개념을 wiki에 적재
2. bridge files 업데이트
3. experiment_digest.md 갱신

---

## Workflow Rules
1. **Never submit without local CV** — optimize CV first, leaderboard is limited validation.
2. **One experiment = one folder** — self-contained with config, code, model, and SUMMARY.md.
3. **Reproducibility** — seed, config, git commit hash, CV score in every experiment.
4. **Manual submission only** — never auto-upload to DACON.
5. **Offline check before packaging** — scan script.py for internet dependencies.
6. **Experiment memory** — every experiment has SUMMARY.md for instant context recovery.
7. **Compound before close** — 세션 종료 전 반드시 /compound로 지식 축적.
8. **Search before plan** — 새 실험 계획 전 반드시 wiki에서 과거 교훈 검색.
9. **Time-aware planning** — 모든 plan은 마감일 + 오늘 quota를 반영해야 함.
10. **Read data_docs first** — dev agent에게 위임 전 반드시 data_docs/ 컨텍스트 주입.

---

## Skills (Slash Commands)

### Compound workflow (지식 축적)
```
/compound         — 세션의 결정/교훈/컨텍스트를 wiki에 적재
```

### Manual workflow (step by step)
```
/eda              — 데이터 탐색 (대회 시작 시)
/plan             — 실험 계획 수립 (wiki 검색 → 과거 교훈 반영)
/dev baseline     — 실험 구현 (train.py + script.py 동시 생성)
/run exp_001      — 실험 실행 + SUMMARY.md 업데이트
/eval exp_001     — 결과 평가 + 리키지 체크
/pack exp_001     — DACON 제출 zip 생성 (offline 검증 포함)
/rank             — 후보 순위 매기기
/status           — 전체 현황 대시보드
/submit-result exp_001 0.82  — LB 점수 기록 + insight 추출
```

### Autonomous mode (자동 파이프라인)
```
/auto             — 5사이클 자동 실행 (search→plan→dev→run→eval→compound 반복)
/auto 10          — 10사이클
/loop 15m /auto   — 15분마다 자동 사이클 반복
```

### Guardrails
- 5회 연속 개선 없으면 자동 중단
- NaN/Inf 발생하면 즉시 중단
- 절대 자동 제출하지 않음
- 모든 실험은 EXPERIMENT_LOG.csv + SUMMARY.md에 기록
- script.py offline 검증 실패 시 CANDIDATE 불가

---

## LLM Wiki Conventions

### Frontmatter (모든 wiki 페이지 필수)
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

### Page Types
- **entity**: `## Definition / ## Why it matters / ## Related / ## History`
- **decision**: ADR 포맷 — `## Context / ## Decision / ## Consequences`
- **lesson**: `## Symptom / ## Root cause / ## Fix / ## Generalization`
- **context**: 프로젝트 스냅샷 — 자유 형식
- **session**: compound 원본 — 자동 생성

### Conflict Rule
기존 페이지와 충돌하면 새 페이지 만들지 말고 기존 페이지에 `## Conflict yyyy-mm-dd` 섹션 추가.

---

## Bridge Files (Agent Context Recovery)
새 세션에서 /auto 실행 시 이 파일들로 상태 복구:
- `logs/orchestrator_state.json` — 현재 전략, best score, stall count
- `logs/experiment_digest.md` — 모든 실험 요약 테이블
- `logs/insights.jsonl` — CV-LB 패턴 (최근 5개만 로드)
- `logs/cycle_history.jsonl` — 최근 N 사이클 reasoning

---

## Conventions
- Config files: YAML format with seed, model params, data splits
- CV: 5-fold stratified by default unless stated otherwise
- Metrics: competition metric as primary, plus auxiliary diagnostics
- Naming: exp_NNN_short_description (e.g., exp_001_baseline_lgbm)
- Inference speed: always measure ms/sample
- Model size: always report total model/ directory size in MB

---

## QMD Integration (Optional)

QMD가 설치되어 있으면 wiki를 하이브리드 검색(BM25 + 벡터 + LLM 리랭킹)으로 조회할 수 있다.

### Setup
```bash
npm install -g @tobi/qmd
qmd collection add wiki/ --name wiki --pattern "**/*.md"
qmd index wiki
```

### MCP Server (`.mcp.json`)
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

QMD 없이도 wiki는 직접 파일 읽기/Grep으로 검색 가능. QMD는 검색 품질을 높이는 선택적 레이어.
