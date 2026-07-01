---
id: data-is-jsonl-not-csv
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [data, io, jsonl, labels]
related: [[session-grouping-requires-groupkfold], [[cv-protocol-stratifiedgroupkfold]]]
summary: 입력 데이터는 CSV가 아니라 JSONL이다 — train.jsonl + train_labels.csv(id로 조인), test.jsonl. 라벨 action은 정수 0-13이 아니라 14개 snake_case 문자열.
---

# Data is JSONL, not CSV

## Symptom
스킬 스켈레톤이 `pd.read_csv('data/train.csv')` / `pd.read_csv('data/test.csv')`를
기본으로 썼다. 실제 파일은 CSV가 아니라 JSONL이라 그대로 실행하면 파일이 없거나
스키마가 맞지 않는다. 또 라벨을 정수 0-13으로 가정한 코드는 제출이 무효가 된다.

## Root cause
초기 스킬 문서가 전형적인 tabular CSV 대회를 가정하고 작성됨. 이 대회의 실제
포맷:
- `data/train.jsonl` — 70,000행, 키 `id, session_meta, history, current_prompt`
- `data/train_labels.csv` — `id,action` (train.jsonl과 `id`로 조인)
- `data/test.jsonl` — 리포지토리엔 **5행 샘플**, 실제 서버 test는 30,000 숨김행
- `data/sample_submission.csv` — 컬럼 `id,action` (이건 CSV 유지)
라벨 `action`은 14개 **정확한 snake_case 문자열**:
read_file, grep_search, list_directory, glob_pattern, edit_file, write_file,
apply_patch, run_bash, run_tests, lint_or_typecheck, ask_user, plan_task,
web_search, respond_only.

## Fix
JSONL 로더 재사용: `experiments/exp_001_tfidf_lightgbm/features.py`의
`load_jsonl` / `build_records` (+ `CLASS_ORDER`). train.jsonl을 train_labels.csv에
`id`로 조인. `script.py`는 `data/test.jsonl`을 읽고 `output/submission.csv`에
`id,action`(문자열 클래스)로 쓴다. 검증은 값이 14개 문자열 집합에 속하는지
확인(정수면 REJECT). sample_submission만 CSV. `/dev` `/eda` `/run` `/eval`에 반영.

## Generalization
새 대회/데이터셋은 스켈레톤의 IO 가정을 그대로 믿지 말고 실제 파일 포맷과
라벨 타입을 먼저 확인하라. 특히 제출 라벨이 문자열 클래스명인지 정수 인덱스인지,
test가 샘플인지 전체인지(여기선 5행 샘플 vs 30,000 실제)를 반드시 구분한다.
