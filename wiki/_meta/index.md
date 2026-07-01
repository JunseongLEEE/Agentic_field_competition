---
id: index
type: entity
created: 2026-06-14
updated: 2026-06-14
tags: [meta, index]
related: [[conventions]]
summary: Wiki 전체 페이지 인덱스
---

# Wiki Index

## Entities
<!-- /compound가 자동으로 업데이트 -->

## Decisions
<!-- /compound가 자동으로 업데이트 -->
- [[cv-protocol-stratifiedgroupkfold]] — CV = StratifiedGroupKFold(5, group=session, seed=42)
- [[daily-quota-20-team]] — 일일 제출 한도 20회(팀)
- [[mi-018-cutoff]] — MI ≥ 0.18 기준 structural features 7개만 생존
- [[feature-engineering-v2-interaction]] — 3개 병목 그룹 타겟 interaction features 추가

## Lessons
<!-- /compound가 자동으로 업데이트 -->
- [[session-grouping-requires-groupkfold]] — 세션 그룹 누수 → GroupKFold 필수
- [[subagent-must-run-training-foreground]] — 학습은 포그라운드로 돌리고 대기
- [[lightgbm-thread-oversubscription-128-core]] — 128코어 스레드 과다구독 방지(≤16)
- [[charwb-tfidf-vocab-single-thread-stall]] — char_wb TF-IDF 어휘 빌드 단일스레드 정체
- [[data-is-jsonl-not-csv]] — 입력은 JSONL, 라벨은 14개 문자열
- [[realtime-experiment-logging]] — train.py line-buffered Tee로 실시간 로그
- [[requirements-never-pin-numpy-scipy]] — numpy/scipy 핀 금지(서버 ABI 크래시), sklearn+lightgbm만 핀
- [[sequential-prescriptions-low-roi]] — 시퀀셜 재처리 5처방 실측 반증(+0.001~+0.02), 진짜 여지는 미세텍스트/args·캘리브레이션·앙상블
- [[deberta-2stage-underperforms-tfidf]] — DeBERTa-v3-small 2-stage가 TF-IDF보다 성능 하락 (0.6563 vs 0.6605)
- [[class-bottleneck-analysis]] — 14 class 병목 3그룹: 탐색4형제, 돌려봐3파전, plan/ask
- [[last-action-zero-discriminative-power]] — last_action 단독은 0개 class를 40%+ precision 판별 불가
- [[respond-only-write-file-keyword-solvable]] — respond_only/write_file은 키워드 precision 90-100%

## Context
<!-- /compound가 자동으로 업데이트 -->

## Sessions
<!-- /compound가 자동으로 업데이트 -->
