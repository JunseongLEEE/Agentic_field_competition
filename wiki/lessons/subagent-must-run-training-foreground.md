---
id: subagent-must-run-training-foreground
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [orchestration, subagent, training, packaging]
related: [[lightgbm-thread-oversubscription-128-core], [[realtime-experiment-logging]]]
summary: 서브에이전트/러너는 학습을 포그라운드로 돌리고 완료를 기다려야 한다 — 백그라운드 후 종료하면 학습이 고아가 되고 패키징 등 후속 단계가 누락된다.
---

# Subagent must run training in the foreground and wait

## Symptom
서브에이전트가 `train.py`를 백그라운드 잡으로 띄운 뒤 곧바로 return/exit 했다.
그 결과 학습이 orphan 프로세스로 남아 계속 돌거나 중단되었고, 러너는
`train_log.json`/`oof_preds.npy`가 아직 없는 상태에서 "완료"로 판단해
dry-run·평가·패키징(zip 생성) 같은 후속 단계를 건너뛰었다. 산출물이 없는
채로 사이클이 진행되어 상태가 꼬였다.

## Root cause
"작업 시작"과 "작업 완료"를 혼동. 비동기로 학습을 던지면 호출자는 완료 시점을
알 수 없는데, 스킬이 그 사실을 강제하지 않았다.

## Fix
Rule A로 명문화: 학습은 **FOREGROUND에서 실행하고 종료까지 BLOCK**한다.
백그라운드-후-종료 금지. 사이클은 `train_log.json` + `oof_preds.npy` +
`test_preds.npy` (패킹 단계면 zip까지) 가 실제로 디스크에 존재할 때만 "완료"다.
`/auto` STEP 3, `/dev`, `/run`, 그리고 model-developer/experiment-runner 에이전트
정의에 모두 반영했다.

## Generalization
장시간 작업을 위임할 때 에이전트는 "던지고 끝"이 아니라 "돌리고 기다려
산출물을 검증"해야 한다. 완료 판정은 프로세스 반환이 아니라 **기대 산출물의
존재**로 정의하라. 정말 백그라운드가 필요하면 완료를 폴링/대기하는 로직을
반드시 함께 둔다.
