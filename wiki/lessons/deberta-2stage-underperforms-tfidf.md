---
id: deberta-2stage-underperforms-tfidf
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [deberta, tfidf, lightgbm, 2stage, encoder]
related: [[cv-protocol-stratifiedgroupkfold], [sequential-prescriptions-low-roi]]
summary: DeBERTa-v3-small→LightGBM 2-stage 파이프라인이 TF-IDF+LightGBM 대비 성능 하락 (0.6563 vs 0.6605)
---

# DeBERTa 2-stage가 TF-IDF보다 낮은 성능

## Symptom
exp_011 (DeBERTa-v3-small → 14-class soft probs → LightGBM) CV Macro-F1 = 0.6563.
exp_001 (TF-IDF 25K + LightGBM) baseline CV = 0.6605. 오히려 -0.4%p 하락.
exp_012 (prompt-only DeBERTa, last_action prefix 없음)는 DeBERTa 단독 F1=0.33~0.43으로 더 심각하여 중단.

## Root cause
1. **정보 압축 손실**: DeBERTa 14-dim soft probs에 25K TF-IDF 대비 어휘 패턴 정보가 부족
2. **max_len=128 텍스트 절단**: prompt 일부가 잘려 정보 손실
3. **합성 데이터 특성**: "의미 이해"보다 "키워드 패턴 매칭"이 유효한 데이터라 소형 encoder 이점 없음
4. **last_action이 DeBERTa에 결정적**: prefix 유무로 DeBERTa 단독 F1이 0.56 vs 0.43 차이 (Δ=0.13)

## Fix
- DeBERTa 2-stage 대신 TF-IDF + engineered features + LightGBM에 집중
- exp_010 (TF-IDF 25K + engineered 65 dense) → CV 0.6838로 현재 최고

## Generalization
- 소형 encoder(DeBERTa-v3-small, 140M)를 intermediate representation(14-dim)으로 압축하면 정보 손실이 큼
- 합성 데이터에서는 규칙 기반/TF-IDF가 encoder보다 유리할 수 있음
- encoder 사용 시 soft probs를 TF-IDF에 **추가**하는 방식이 **대체**보다 안전
- FP16 변환으로 모델 크기 절반 (541→271MB, zip 408→262MB), T4에서 성능 동일
