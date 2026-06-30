---
source_id: source_001_last_year_text_detection
url: (internal — Last_year_competition/ folder)
license: DACON terms (private)
relevance: low
added: 2026-06-30
---

# Reference: Last Year's DACON Competition (Text Detection)

## Context
The repo contains `Last_year_competition/open/` from a previous DACON competition.
This year's task is **completely different**:

| | Last year | This year |
|---|---|---|
| Task | Text detection / classification | AI Agent Action Decision |
| Classes | (different label set) | 14 actions |
| Input | text passages | current_prompt + history + session_meta |
| Metric | (varies) | Macro-F1 |

## Intent for this year
**DO NOT use last year's data as training data** for this year's task — labels and domain do not transfer.

## Possible legitimate uses
- **Framework reference only**: how the previous CV / submission pipeline was structured
- **Negative reference**: examples of pipelines we should NOT copy if they leaked
- **None** for direct modeling

## .gitignore status
`Last_year_competition/open/*.csv` is gitignored (large files, previously caused push failures).
If you need to reference structure, browse locally only.

## Action items
- [ ] Confirm with user whether to delete `Last_year_competition/` entirely or keep as local-only reference
- [ ] If kept, ensure no script under `experiments/` or `scripts/` reads from this folder
