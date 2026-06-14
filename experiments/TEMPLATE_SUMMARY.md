# exp_NNN_name — SUMMARY

> One-line: [한줄 요약 — 이 실험이 뭔지 즉시 파악 가능하게]

## Setup
| Item | Value |
|------|-------|
| Model | lightgbm / xgboost / catboost / nn / transformer / ensemble |
| Features | [핵심 feature 설명, 몇 개] |
| Preprocessing | [주요 전처리] |
| CV Strategy | 5-fold stratified, seed=42 |
| Key Hyperparams | lr=0.05, n_est=1000, ... |

## Inference Constraints (DACON Code Submission)
| Item | Value |
|------|-------|
| Inference Speed | XX ms/sample |
| Model Size (model/) | XX MB |
| Peak Memory | XX GB |
| Offline Compatible | Yes / No |
| requirements.txt | [핵심 패키지 목록] |
| Pretrained Weights | [로컬 포함 여부, 파일명] |

## Data
| Item | Value |
|------|-------|
| Train Rows | NNN |
| Test Rows | NNN |
| Feature Count | NN |
| Target Distribution | class0: XX%, class1: XX% |
| Special Handling | [class imbalance 처리, augmentation 등] |

## Results
| Metric | Score |
|--------|-------|
| CV Mean | 0.XXXX |
| CV Std | 0.XXXX |
| CV Fold Scores | [0.XX, 0.XX, 0.XX, 0.XX, 0.XX] |
| LB Score | 0.XXXX (or "미제출") |
| CV-LB Gap | 0.XXXX (or N/A) |
| Status | PLANNED / COMPLETED / CANDIDATE / SUBMITTED / REJECTED |

## Submission Files
| File | Status | Notes |
|------|--------|-------|
| script.py | OK/Missing | [추론 전용, data/ → output/submission.csv] |
| model/ | OK/Missing | [모델 파일 목록] |
| requirements.txt | OK/Missing | [패키지 목록] |

## What Worked
- [이 실험에서 효과가 있었던 것]

## What Didn't Work
- [시도했지만 효과 없었던 것]

## Insight
- [다음 실험에 반영할 교훈]
- [CV-LB 관계에서 배운 점]

## Diff from Previous Best
- Previous best: exp_NNN (CV: 0.XXXX)
- This experiment changed: [무엇을 바꿨는지]
- Result: [개선/하락/동일] by 0.XXXX
