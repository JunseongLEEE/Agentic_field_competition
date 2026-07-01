# exp_009_deberta_v3_small_ft

## Hypothesis
Full fine-tuning of `microsoft/deberta-v3-small` with the dominant structured /
sequential signals (last_action, second_last_action, turn_index, user_tier,
last_ci_status, n_open_files, failed flag, history_len) **serialized into the
input text** will beat the tfidf-GBDT leader (StratifiedGroupKFold CV ~0.674).
Frozen encoder embeddings only reached ~0.62-0.635, so the structured signal must
flow through a fine-tuned encoder.

## Verification protocol
- accept_if: StratifiedGroupKFold cv_mean > 0.674 and no collapsed classes.
- reject_if: cv_mean < 0.64 or any minority class collapses (F1 < 0.05).

## Setup
- CV: StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
  group = session id = `id.rsplit("-step",1)[0]`. Group-overlap asserted per fold.
- Input: `tier=.. lang=.. turn=.. last=.. prev=.. failed=.. ci=.. nopen=.. hist=.. [SEP] {current_prompt} [SEP] prev_user: {last user msg}`.
- Loss: class-balanced weighted CrossEntropy (Macro-F1).
- Training: bf16 autocast, batch 32, max_length 160, 3 epochs, AdamW lr 2e-5,
  warmup 0.06, grad-clip 1.0. Best epoch per fold picked by val Macro-F1.
- Final model: retrained on full train (epochs ~= mean best fold epoch + 1),
  saved fp16 to `model/` with tokenizer + label_map.json.

## Files
- `features.py` — shared serialization (`build_texts`, `build_input_text`),
  reuses exp_001 `extract_seq_meta`.
- `train.py` — CV + OOF + full-train, writes `train_log.json`.
- `script.py` — OFFLINE inference from local `model/`.
