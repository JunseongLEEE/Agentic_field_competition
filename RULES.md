# Competition Rules & Constraints

## Submission Rules
- Maximum submissions per day: defined by competition (default assume 5)
- We select at most 10 candidates internally, then manually choose which to submit
- Final submission selection: pick 2 (or as allowed) based on CV stability + LB correlation

## Anti-Overfitting Protocol
1. Never optimize directly for leaderboard score
2. Track CV-LB gap: if |CV - LB| > threshold, investigate before trusting
3. Use time-based or group-based splits when data has temporal/group structure
4. Ensemble diversity > ensemble size
5. If a single feature improves LB but not CV, it's likely leakage — remove it

## Data Handling
- No data leakage between train/validation/test
- Feature engineering must be fit on train only, transform on val/test
- Target encoding must use CV-aware encoding (within-fold only)

## Reproducibility Requirements
Every experiment MUST record:
- Random seed (default: 42)
- Exact data version / preprocessing steps
- Model hyperparameters (full config.yaml)
- CV fold indices
- Git commit hash at time of run
- Environment (Python version, key package versions)

## Experiment Lifecycle
1. PLANNED → experiment created with hypothesis
2. RUNNING → code executing
3. COMPLETED → CV results available
4. EVALUATED → compared against baselines, leakage checked
5. CANDIDATE → selected as submission candidate
6. SUBMITTED → actually uploaded to competition site
7. ARCHIVED → no longer active

## Code Quality
- Scripts must run end-to-end without manual intervention
- No hardcoded paths (use config or relative paths)
- Memory-efficient: clear large objects, use generators where possible
