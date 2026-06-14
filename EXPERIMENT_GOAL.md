# Experiment Goals

## Competition Objective
<!-- Fill in when competition is known -->
- **Competition**: [TBD]
- **Task**: [classification / regression / ranking / etc.]
- **Metric**: [TBD]
- **Deadline**: [TBD]
- **Max daily submissions**: [TBD]

## Current Strategy
<!-- Update as strategy evolves -->

### Phase 1: Baseline
- [ ] EDA and data understanding
- [ ] Simple baseline model (e.g., LightGBM with minimal features)
- [ ] Establish reliable CV setup
- [ ] First submission to calibrate CV-LB correlation

### Phase 2: Feature Engineering
- [ ] Domain-specific features
- [ ] Interaction features
- [ ] Aggregation features
- [ ] Feature selection

### Phase 3: Model Exploration
- [ ] Try multiple model families (LightGBM, XGBoost, CatBoost, NN)
- [ ] Hyperparameter tuning (Optuna)
- [ ] Architecture search for NN models

### Phase 4: Ensemble & Stacking
- [ ] Diverse model ensemble
- [ ] Stacking with meta-learner
- [ ] Blending weights optimization on CV

### Phase 5: Final Selection
- [ ] Stability analysis (CV variance across seeds)
- [ ] LB probing analysis
- [ ] Final 2 submissions chosen

## Hypotheses Backlog
<!-- Add experiment ideas here -->
| Priority | Hypothesis | Expected Impact | Status |
|----------|-----------|-----------------|--------|
| HIGH     | [example] Baseline LightGBM with raw features | Establish baseline | PLANNED |
