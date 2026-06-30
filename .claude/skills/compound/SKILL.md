---
description: "Compound knowledge from this session into the LLM Wiki. Extracts decisions, lessons, entities, and context; updates bridge files and digest. Run before ending a session."
user-invocable: true
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
---

# /compound — Knowledge Capture

Turn this session's discoveries into durable wiki entries and refresh bridge files.
Run before closing the session, and at the end of every `/auto` run.

## STEP 1 — Classify

Sort everything that happened this session into four buckets:

| bucket | trigger | location |
|---|---|---|
| Decision | I deliberately picked one option over alternatives | `wiki/decisions/` |
| Lesson   | I made a mistake or discovered a wrong assumption | `wiki/lessons/` |
| Entity   | a new concept/tool/model/dataset/metric was introduced | `wiki/entities/` |
| Context  | the project state changed in a way another agent must know | `wiki/context/` |

For each entry, capture:
- exact numbers (CV, gap, model_size_mb, runtime)
- exact file paths and experiment ids
- date (ISO)

## STEP 2 — Dedupe Against Existing Wiki

For each new entry, `Grep -ri "<keyword>" wiki/`. If a page already covers the same concept:
- **append** a dated section to that page (do NOT create a duplicate)
- update its `updated:` frontmatter

For decisions that conflict with a prior decision, add a `## Conflict <YYYY-MM-DD>` section per `wiki/_meta/conventions.md`.

## STEP 3 — Write Pages

Follow `wiki/_meta/conventions.md` exactly. Required frontmatter for every page:

```yaml
---
id: <kebab-case-slug>
type: entity | decision | lesson | context | session
created: <ISO date>
updated: <ISO date>
tags: [topic1, topic2]
related: [[other-page-id]]
summary: <one-line summary>
---
```

Page bodies by type:
- **entity**: `## Definition / ## Why it matters / ## Related / ## History`
- **decision**: `## Context / ## Decision / ## Consequences` (ADR format)
- **lesson**: `## Symptom / ## Root cause / ## Fix / ## Generalization`
- **context**: free-form project snapshot
- **session**: auto-generated compound source

File locations:
- decisions → `wiki/decisions/<id>.md`
- lessons → `wiki/lessons/<id>.md`
- entities → `wiki/entities/<id>.md`
- context → `wiki/context/<id>.md`
- session log → `wiki/sessions/session-<YYYY-MM-DD>-<NNN>.md`

## STEP 4 — Update wiki/_meta/index.md

Add a single line per new/updated page:
```
- [[page-id]] — one-line summary (YYYY-MM-DD)
```
Place under the right section header (Decisions / Lessons / Entities / Context).

## STEP 5 — Refresh Bridge Files

1. `logs/orchestrator_state.json` — `best_cv`, `best_lb`, `current_phase`, `stall_counter`, `last_reasoning`, `last_updated`.
2. `python scripts/build_digest.py` — refresh `logs/experiment_digest.md`.
3. `logs/insights.jsonl` — append any new CV→LB insight (usually `/submit-result` already wrote this, but double-check).

## STEP 6 — Light vs Full Mode

- **Full compound** (default, end-of-session): walk through all four buckets, write all pages, refresh all bridge files.
- **Light compound** (each `/auto` cycle): only write lesson/decision pages when something noteworthy happened (CV improvement, failure, surprising LB result). Update orchestrator_state + cycle_history. Skip the rest.

Choose mode based on argument:
- `/compound` → full
- `/compound light` → light

## STEP 7 — Report

```
═════════════════════════════════════════════
COMPOUND — <mode> mode — <YYYY-MM-DD HH:MM>
═════════════════════════════════════════════
Pages created  : <N>
Pages updated  : <M>
Conflicts      : <K>
Bridge files   : orchestrator_state ✓ | digest ✓ | insights ✓

New decisions:
  - [[<id>]] — <summary>
New lessons:
  - [[<id>]] — <summary>
New entities:
  - [[<id>]] — <summary>

Wiki state: <X> pages total (entities <a>, decisions <b>, lessons <c>, context <d>, sessions <e>)
═════════════════════════════════════════════
```

## Hard Rules

- ALWAYS dedupe before writing a new page.
- ALWAYS include exact numbers; vague entries are worse than no entry.
- ALWAYS write bidirectional `related: [[...]]` links — index follows the graph.
- NEVER create wiki pages without frontmatter.
- NEVER overwrite existing pages without appending a dated section first.
