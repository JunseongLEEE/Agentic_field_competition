# EDA Snapshot — 2026-07-01

14-class AI Agent Action Decision (Macro-F1). Data = **JSONL** (not CSV). Full train (70,000)
loaded and joined to `train_labels.csv` — **0 join failures**. All figures in
`data_docs/eda_figures/` (PNG, dpi 120, English labels). This snapshot VISUALIZES and QUANTIFIES
the prior forensic report in `domain_notes.md` and FILLS its gaps (budget_tokens, open_files,
MI ranking, session grouping).

---

## 1. Class distribution (`01_class_distribution.png`)
n=70,000. Imbalance max/min = **8.78:1** (edit_file 15.96% vs web_search 1.82%). No class <1%,
so Macro-F1 is fragile but not degenerate. Sorted %:
edit_file 15.96 · grep_search 14.16 · read_file 13.22 · glob_pattern 7.55 · respond_only 7.40 ·
run_bash 7.24 · apply_patch 6.89 · run_tests 6.52 · list_directory 6.18 · ask_user 3.86 ·
plan_task 3.83 · lint_or_typecheck 3.26 · write_file 2.12 · web_search 1.82.
→ minority classes (web_search, write_file, lint_or_typecheck) drive Macro-F1; use class_weight.

## 2. History length (`02_history_length.png`)
Even lengths only: 0→9,000 · 2→8,797 · 4→8,446 · 6→8,001 · 8→7,526 · 10→6,644 · **12→21,586**
(pile-up at the cap). Confirms user→assistant_action pairing and snapshot-at-even-step rule.

## 3. turn_index → class (`03_turnindex_class_heatmap.png`)
Row-normalized P(class|turn_index). Strong phase signal: early turns favor exploration
(list_directory, read_file, plan_task); mid turns edit_file peaks; late turns show respond_only
rising (wrap-up). MI(turn_index)=0.139.

## 4. Transition last_action → next (`04_transition_last_to_next.png`)
15×14 heatmap P(next|last_action) incl. NONE (empty history, n=9,000). Sharpest rows:
write_file→edit_file, list_directory→read_file, edit_file→run_tests, lint_or_typecheck→apply_patch.
last_action is the single strongest feature (MI=0.250).

## 5. last_action_failed effect (`05_last_action_failed.png`)
Among rows with history: not-failed n=52,527, failed n=8,473. Failure shifts mass toward
edit_file / apply_patch (repair). Effect is real but modest at the aggregate level — MI of the raw
`last_action_failed` flag alone is only **0.009** (its signal is largely conditional on last_action).

## 6. Rule categories → class (`06_rule_categories_class.png`)
Match counts: SEARCH 8,426 · SHOW_FILE 5,334 · RUN_TEST 3,830 · WRAP_UP 3,814 · PLAN_REQ 1,589 ·
ERROR_HELP 785 · LINT_CHECK 634 · WEB_REF 322. WRAP_UP is near-deterministic → respond_only
(MI(rule_WRAP_UP)=0.169, the 3rd-strongest feature). RUN_TEST → run_tests/run_bash. Rare rules
(WEB_REF, LINT_CHECK, ERROR_HELP) carry little MI (<0.013) — low coverage.

## 7. Categorical session_meta signal (`07_categorical_meta_signal.png`)
Cramer's V vs 14-class label:
- **git_dirty = 0.260** (moderate — stronger than the prior report's "weak" verdict)
- **last_ci_status = 0.120** (weak-moderate)
- user_tier = 0.022 (noise) · language_pref = 0.021 (noise)
Note: git_dirty/last_ci_status Cramer's V is inflated by class-count asymmetry; in the joint MI
model git_dirty MI=0.039 and last_ci_status MI=0.015 (both minor once last_action is present).
user_tier & language_pref are effectively noise (MI≈0.0005). See §8/§10 for language nuance.

## 8. budget_tokens_remaining — GAP FILLED (`08_budget_tokens.png`)
Lives in `session_meta` (not workspace). Distribution roughly uniform-ish, wide:
min 55 · p05 16.5k · p25 43.8k · p50 94.5k · p75 137.1k · p95 170.6k · max 199.7k · mean 92.4k.
Quintile→class heatmap: max per-class deviation from the global rate is only **0.0144** (≈1.4pp).
MI(budget_tokens_remaining)=**0.0137** (near bottom). **VERDICT: essentially NO signal** — do not
lean on it; drop or keep only as a weak raw feature.

## 9. open_files — GAP FILLED (`09_open_files.png`)
Count dist: 0→23,498 · 1→40,257 · 2→5,876 · 3→355 · 4→12 · 5→2 (0–2 = 99.6%).
`n_open_files` is a surprisingly useful feature: **MI=0.129** (7th) — largely because n=0 ⇔ empty
history / turn 1 (exploration) and n≥1 ⇔ mid-session editing. Extension breakdown (top): .py 13,024
· .ts 4,515 · .tsx 4,089 · .rs 3,326 · (noext) 2,872 · .java 2,605 · .yml 2,481 · .json 2,203 ·
.go 2,066 · .md 1,662 · .vue 1,641 · .yaml 1,634. Extension→class shows mild archetype tint but
weaker than n_open_files itself. **VERDICT: the COUNT carries signal (proxy for session phase);
the extensions add little.**

## 10. current_prompt length by language_pref (`10_prompt_length_by_lang.png`)
Char len p50/p95/max = 56/123/346. Whitespace-token p50/p95/max = 12/23/64. Prompts are SHORT.
ko/mixed have fewer whitespace tokens per char (agglutinative, no spaces) → char_wb TF-IDF matters
for Korean; word-token counts under-represent ko content.

## 11. Workspace primary language → class (`11_primary_language_class.png`)
argmax(language_mix) over the 12 archetypes. Max per-class deviation from global = 0.065 (≈6.5pp);
MI(primary_lang)=0.027 — weak-moderate archetype tint (e.g. py/ipynb workspaces lean more on
run_tests/run_bash). Keep as a low-cardinality categorical, do not over-weight.

## 12. Feature → Target Mutual Information (KEY DELIVERABLE) (`12_mutual_information.png`)
sklearn.mutual_info_classif on the dense `records_to_dense` matrix (discrete_features per type).

| Rank | Feature | MI (nats) |
|---:|---|---:|
| 1 | last_action | 0.2499 |
| 2 | second_last_action | 0.2124 |
| 3 | rule_WRAP_UP | 0.1689 |
| 4 | turn_index | 0.1386 |
| 5 | history_len | 0.1374 |
| 6 | cnt_edit_file | 0.1307 |
| 7 | n_open_files | 0.1291 |
| 8 | rule_RUN_TEST | 0.0951 |
| 9 | rule_SEARCH | 0.0766 |
| 10 | rule_SHOW_FILE | 0.0663 |
| 11 | cnt_grep_search | 0.0585 |
| 12 | elapsed_session_sec | 0.0441 |
| 13 | cnt_read_file | 0.0441 |
| 14 | cnt_apply_patch | 0.0402 |
| 15 | loc | 0.0396 |
| 16 | git_dirty | 0.0394 |
| 17 | rule_PLAN_REQ | 0.0322 |
| 18 | primary_lang | 0.0272 |
| 19 | cnt_run_tests | 0.0252 |
| 20 | cnt_lint_or_typecheck | 0.0176 |
| 21 | last_ci_status | 0.0151 |
| 22 | budget_tokens_remaining | 0.0137 |
| 23 | rule_ERROR_HELP | 0.0123 |
| 24 | cnt_glob_pattern | 0.0117 |
| 25 | cnt_run_bash | 0.0115 |
| 26 | cnt_write_file | 0.0110 |
| 27 | last_action_failed | 0.0091 |
| 28 | cnt_list_directory | 0.0086 |
| 29 | rule_WEB_REF | 0.0082 |
| 30 | rule_LINT_CHECK | 0.0067 |
| 31 | cnt_ask_user | 0.0057 |
| 32 | cnt_web_search | 0.0055 |
| 33 | cnt_plan_task | 0.0030 |
| 34 | user_tier | 0.0005 |
| 35 | language_pref | 0.0005 |
| 36 | cnt_respond_only | 0.0000 |

Note: MI here excludes the TF-IDF text block; `current_prompt` text (beyond the 8 rule flags) still
adds signal not captured above. `elapsed_session_sec` (0.044) is a mild redundant proxy for
turn_index. user_tier/language_pref/budget confirmed as near-noise.

## 13. Inference-budget forecast (`13_inference_budget_forecast.png`)
current_prompt: mean 12.8 whitespace tokens/row. Serialized history (json.dumps): mean 103.7,
p95 201, max 297 tokens/row. Combined ≈ 116 tok/row. For the hidden **30,000-row** test:
- current_prompt only ≈ **0.38M** tokens
- current_prompt + serialized history ≈ **3.49M** tokens
Tiny for TF-IDF/LightGBM (well under the 10-min T4 budget). For a transformer encoder, batching
30k short (<64-token) prompts is trivially in-budget; adding history would ~9× the token load but
still feasible if truncated. History is the token-heavy part — truncate to last K turns if used.

---

## Leakage probes
1. **last_action == label (trivial copy):** 8,473 / 61,000 rows-with-history = **13.89%**. This is
   the natural repeat rate (e.g. edit→edit, grep→grep), NOT a generation leak — it matches the
   diagonal of the transition matrix. No copy-through injection detected.
2. **label name appears anywhere in history:** 20,723 / 70,000 = 29.6% — again expected from action
   recurrence within a session; not a leak.
3. **No column trivially reveals the label.** Verdict: **no leakage.**

## Session grouping (CRITICAL — may change CV strategy)
IDs are `sess_sim_<date>_<NNNNNN>-step_XX`. Stripping `-step_XX`:
- **9,429 distinct sessions** across the 70,000 train rows.
- Steps per session: min 1 · median 7 · mean 7.42 · max 18.
- **9,213 of 9,429 sessions (97.7%) have >1 step in train; 99.69% of ROWS belong to a multi-step
  session.**
- Multiple snapshots of the SAME session share workspace, language_mix, budget, open_files and an
  overlapping history prefix. A plain StratifiedKFold will scatter steps of one session across
  folds → the model memorizes session-specific context → **optimistic, leaky CV**.
- **RECOMMENDATION: use GroupKFold (group = session id = id without `-step_XX`).** This is a
  concrete change to the default 5-fold stratified CV. (Consider StratifiedGroupKFold to also keep
  class balance across folds.)

## Test-sample note
`data/test.jsonl` has only 5 rows (format sample). Schema identical to train; value ranges in-range
(turn_index 1–6, budget 8.9k–199k, last_actions include NONE/grep_search/apply_patch/read_file).
All 5 sample session-ids happen to also exist in train — this is a **format-sample artifact** (the
sample was drawn from train sessions); the real hidden 30,000-row test is server-mounted and should
NOT overlap train. Do not tune on these 5 rows.

---

## Figure index
1. `01_class_distribution.png` — 14-class counts/%, 8.78:1 imbalance.
2. `02_history_length.png` — even-only lengths, pile-up at 12.
3. `03_turnindex_class_heatmap.png` — session-phase signal (explore→edit→wrap-up).
4. `04_transition_last_to_next.png` — 15×14 P(next|last_action); strongest single feature.
5. `05_last_action_failed.png` — failure shifts toward edit/apply_patch (modest aggregate effect).
6. `06_rule_categories_class.png` — WRAP_UP≈deterministic respond_only; rare rules weak.
7. `07_categorical_meta_signal.png` — git_dirty/last_ci moderate Cramer's V; tier/lang noise.
8. `08_budget_tokens.png` — wide distribution, ~no class signal (MI 0.014).
9. `09_open_files.png` — count is a phase proxy (MI 0.129); extensions add little.
10. `10_prompt_length_by_lang.png` — prompts short (p50 12 tok); ko needs char_wb.
11. `11_primary_language_class.png` — weak archetype tint (MI 0.027).
12. `12_mutual_information.png` — full MI ranking (KEY).
13. `13_inference_budget_forecast.png` — ~116 tok/row; 30k test ≈ 3.5M tok w/ history.
14. `14_steps_per_session.png` — 9,429 sessions, median 7 steps → GroupKFold.

## Recommended next experiments
1. Baseline TF-IDF(current_prompt word 1-2 + char_wb 2-4) + LightGBM, class_weight=balanced,
   **GroupKFold by session id**.
2. Add sequential block: last_action, second_last_action, turn_index, history_len, n_open_files,
   cnt_edit_file (top-MI dense features). Drop user_tier/language_pref/budget_tokens (noise).
3. Probe: quantify CV inflation of StratifiedKFold vs GroupKFold to confirm the session-leak fix.
