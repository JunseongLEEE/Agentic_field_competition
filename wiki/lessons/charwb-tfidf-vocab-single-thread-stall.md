---
id: charwb-tfidf-vocab-single-thread-stall
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [tfidf, features, performance, nlp]
related: [[lightgbm-thread-oversubscription-128-core], [[data-is-jsonl-not-csv]]]
summary: 70k 코퍼스에 char_wb 넓은 char n-gram(예 (2,4)) TfidfVectorizer는 단일 스레드로 어휘를 만들며 수 분간 멈춘 듯 보인다 — max_features를 낮추고 min_df를 올리거나 word n-gram을 써라.
---

# char_wb TF-IDF vocab build stalls (single-threaded)

## Symptom
`TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))`를 전체 70,000행
코퍼스에 fit할 때 수 분간 진행이 없어 멈춘 것처럼 보였다. 다른 학습들이
기다리며 파이프라인 전체가 지연되었다.

## Root cause
sklearn TfidfVectorizer의 어휘(vocabulary) 구축은 **단일 스레드**다. 넓은 char
n-gram 범위는 어휘가 폭발적으로 커져(수십만~수백만 항목) fit 시간이 길어진다.
멀티코어를 전혀 활용하지 못하므로 코어 수와 무관하게 느리다.

## Fix
char 피처를 꼭 써야 하면:
- `max_features`를 낮게 캡하고 `min_df`를 올려 어휘를 줄인다.
- 또는 word n-gram을 우선한다.
또한 EDA(MI) 결과 **구조·시퀀셜 피처(last_action, turn_index, rule flags 등)가
신호를 지배**하므로 char TF-IDF는 선택 사항이다. `/dev`에 Rule C로 명문화.

## Generalization
단일 스레드 전처리(특히 어휘/사전 구축)는 코어를 늘려도 빨라지지 않는다.
대형 코퍼스에서는 어휘 크기를 사전에 통제(max_features/min_df)하거나, 신호
기여가 큰 저비용 피처로 대체하라. "멈춘 것 같다"의 상당수는 단일 스레드
어휘 빌드다.
