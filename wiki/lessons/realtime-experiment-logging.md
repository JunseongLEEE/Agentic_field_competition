---
id: realtime-experiment-logging
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [logging, orchestration, observability]
related: [[subagent-must-run-training-foreground]]
summary: 학습 진행을 실시간으로 볼 수 없어 에이전트가 로그 없이 종료하던 문제 — train.py가 line-buffered Tee로 experiments/<exp>/train.log에 실시간 기록.
---

# Real-time experiment logging

## Symptom
학습이 도는 동안 진행 상황을 실시간으로 관찰할 수 없었다. 특히 에이전트가
학습을 던지고 종료하면 눈에 보이는 로그 파일이 남지 않아, 진행/정체/오류를
알 수 없었다. stdout 버퍼링 때문에 로그가 뭉텅이로 뒤늦게 나오기도 했다.

## Root cause
- 표준 로그 경로 규약이 없었다.
- 파이썬 stdout이 파이프로 나갈 때 블록 버퍼링되어 실시간이 아니었다.
- 실행 방식(포그라운드/백그라운드/에이전트)에 따라 로그가 남기도 안 남기도 했다.

## Fix
규약을 고정: 모든 `train.py`는 시작 시 **line-buffered Tee**를 설치해 표준 경로
`experiments/<exp>/train.log`에 실시간(줄 단위)으로 기록한다. 실행 방식과
무관하게 파일이 즉시 갱신되어 `tail -f`로 관찰 가능하다.
```python
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
_logf = open(os.path.join(ROOT, "train.log"), "a", buffering=1)  # line-buffered
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams:
            s.write(d); s.flush()
    def flush(self):
        for s in self.streams: s.flush()
sys.stdout = sys.stderr = _Tee(sys.__stdout__, _logf)
```
러너는 추가로 `python -u train.py 2>&1 | tee -a experiments/<exp>/train.log`
(unbuffered + tee)로 실행한다. 모든 print는 `flush=True`. `/dev` `/run`,
model-developer/experiment-runner 에이전트, data_docs Modeling Protocol에 반영.

## Generalization
장시간 작업은 표준 경로의 **실시간(line-buffered) 로그**를 남겨 관찰 가능하게
하라. 파이프 뒤 파이썬은 `-u`로 unbuffered, 스크립트 내부엔 Tee를 둬서 실행
방식과 무관하게 항상 로그가 남도록 이중으로 보장한다.
