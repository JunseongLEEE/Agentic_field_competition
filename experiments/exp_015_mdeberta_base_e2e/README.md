# exp_015 — End-to-end mDeBERTa-v3-base

## Hypothesis
Rich context input (turn, workspace archetype, action history, last result) fed
directly into **microsoft/mdeberta-v3-base** at max_len=384 with Focal Loss +
Logit Adjustment should outperform exp_011's sparse 2-stage pipeline (CV 0.657).

## Key changes vs exp_011
- **Input**: `[CTX] turn/arch/last/result/ci/dirty [HIST] recent 4-6 turns [NOW] prompt`
- **Model**: mdeberta-v3-base (multilingual) instead of deberta-v3-small (English-only)
- **Architecture**: End-to-end 14-class, no LightGBM stacking
- **Loss**: Focal CE (γ=2) + class weights + logit adjustment (τ=1.0)
- **CV**: StratifiedGroupKFold(5, session_id) — unchanged

## Verification
- Accept if CV Macro-F1 > 0.68 (beats exp_010 directionally)
- Reject if CV < exp_011 (0.657) or inference > 10 min on T4

## Run
```bash
cd experiments/exp_015_mdeberta_base_e2e
python train.py          # ~several hours on GPU
python script.py         # dry-run inference
```
