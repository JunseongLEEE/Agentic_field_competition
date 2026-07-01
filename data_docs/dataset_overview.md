# Dataset Overview

> 실측 기반 (train 70,000 전량 로드 확인). 상세 도메인 분석은 `domain_notes.md` 참조.

## Files (data/ 아래)

| File | Rows | Format | Description |
|---|---|---|---|
| train.jsonl | 70,000 | JSON Lines | 학습 입력 (1줄=1샘플 JSON 객체) |
| train_labels.csv | 70,000 | CSV | 학습 정답 (`id,action`) |
| test.jsonl | 5 (샘플) | JSON Lines | 형식 확인용. **실제 평가 30,000건은 비공개**, 서버가 마운트 |
| sample_submission.csv | 5 | CSV | 제출 양식 (`id,action`) |
| baseline_submit.zip | - | zip | TF-IDF+LogReg 예시 제출본 (참고용) |

⚠️ **CSV 아님 — 입력은 JSONL.** 프레임워크 템플릿의 `pd.read_csv('test.csv')` 를 그대로 쓰면 안 됨.

## Schema — train.jsonl / test.jsonl (동일 구조, test는 정답 없음)

각 줄 = 한 세션의 특정 시점 스냅샷. 최상위 키 4개:

| Key | Type | Description |
|---|---|---|
| `id` | str | 샘플 식별자 (예: `sess_sim_20260522_028750-step_02`). train_labels.csv와 조인 키 |
| `session_meta` | dict | 세션/작업공간 메타 (아래) |
| `history` | list | 이전 대화·행동 기록 (시간순, 길이 0~12, **짝수만**) |
| `current_prompt` | str | 현재(최근) 사용자 발화 — **이 시점의 다음 action이 예측 대상** |

### session_meta
| Field | Type | Values / Note |
|---|---|---|
| `user_tier` | str | enterprise / pro / free |
| `language_pref` | str | ko / en / mixed |
| `budget_tokens_remaining` | int | 잔여 토큰 예산 |
| `turn_index` | int | 현재 턴 번호 (= history_len/2 + 1, 작을수록 초반) — **강한 신호** |
| `elapsed_session_sec` | int | 세션 경과 초 |
| `workspace.language_mix` | dict | 언어→비율 (합≈1.0). unique 조합 12종(프로젝트 아키타입) |
| `workspace.loc` | int | 코드베이스 라인 수 |
| `workspace.git_dirty` | bool | 미커밋 변경 여부 (라벨과 약상관) |
| `workspace.open_files` | list[str] | 열린 파일 경로 (0~2개가 99.5%) |
| `workspace.last_ci_status` | str | passed / failed / none (라벨과 약상관) |

### history 원소 (user → assistant_action 교대)
- **user 턴**: `{role:"user", content:<발화>}`
- **행동 턴**: `{role:"assistant_action", name:<14클래스 중 하나>, args:{...}, result_summary:<결과 요약>}`
  - `history`의 마지막 `assistant_action.name`(= `last_action`)이 **가장 강한 단일 피처**.
  - `result_summary`에 `ERROR`/`FAIL` 포함 여부(`last_action_failed`)가 부스터.

## Target — train_labels.csv

- **Column**: `action` (join key `id`)
- **Type**: multiclass classification, **14 classes** (제출 시 대소문자까지 정확히 일치 필요)
- **분포 (실측, %)**:
  edit_file 15.96 · grep_search 14.16 · read_file 13.22 · glob_pattern 7.55 ·
  respond_only 7.40 · run_bash 7.24 · apply_patch 6.89 · run_tests 6.52 ·
  list_directory 6.18 · ask_user 3.86 · plan_task 3.83 · lint_or_typecheck 3.26 ·
  write_file 2.12 · web_search 1.82
- 불균형 ≈ 8.8:1 (edit_file vs web_search). **Macro-F1** 지표라 소수 클래스 recall이 중요.

### 14 클래스 정의
read_file(파일읽기) · grep_search(패턴검색) · list_directory(디렉터리목록) · glob_pattern(글롭검색) ·
edit_file(기존파일수정) · write_file(새파일작성) · apply_patch(패치적용) · run_bash(셸실행) ·
run_tests(테스트실행) · lint_or_typecheck(린트/타입검사) · ask_user(사용자질문) · plan_task(작업계획) ·
web_search(웹검색) · respond_only(도구없이 응답만)

## Submission — sample_submission.csv

- Columns: `id,action`. test.jsonl의 각 id에 대해 예측 클래스 문자열을 채움.
- 저장 경로: `output/submission.csv` (서버 필수 경로/파일명).

## Evaluation Metric
- **Primary**: Macro-F1 (14 클래스 균등 가중)
- **Secondary diagnostics**: per-class F1, collapsed classes(F1<0.05), OOF↔test 분포 L1

## Key Constraints
- 오프라인 실행 (설치 후 인터넷 차단), `from_pretrained("hub/name")` 금지
- zip ≤ 1GB / 설치 ≤ 10분 / 추론 ≤ 10분 (T4 16GB, 3 vCPU, 12GB RAM)
- **실제 test 30,000건** 기준 추론 시간 설계 (샘플 5건에 속지 말 것)
- 데이터는 순수 합성 (오픈소스 유래 없음) → 외부 데이터 augmentation 비권장 (domain_notes 참조)

## Modeling Protocol (LOCKED — 모든 실험 준수)
> EDA(2026-07-01)와 Stage-1 bake-off 결과로 확정. `/dev`·`/auto` 에이전트는 반드시 따를 것.

1. **CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), group = 세션id (`id.rsplit("-step",1)[0]`)**.
   - 이유: 9,429개 세션 / 70,000행, 행의 99.69%가 멀티스텝 세션 소속. 같은 세션 step은 workspace/meta 공유 + history prefix 중첩 → plain StratifiedKFold는 폴드 간 누수로 CV 과대평가. **StratifiedKFold 금지.**
   - 폴드 후 반드시 assert: `set(groups[tr]) & set(groups[va]) == 0`.
2. **학습은 GPU 사용** (RTX 3090 24GB 로컬): GBDT는 GPU(xgboost device=cuda / catboost task_type=GPU / lightgbm device=gpu), 인코더는 GPU fine-tune. 추론(script.py)은 서버 T4에서 10분 내.
3. **스레드 캡**: CPU 폴백 시 n_jobs/thread_count ≤ 16 (128코어 과다구독 방지).
4. **피처**: `experiments/exp_001_tfidf_lightgbm/features.py` 재사용 (tfidf word+char ⊕ 구조/시퀀셜/rule/meta). MI 상위: last_action > second_last_action > rule_WRAP_UP > turn_index > history_len > n_open_files. budget_tokens_remaining은 신호 미미(드롭 가능).
5. **Stage-1 결과 참고치(GroupKFold)**: tfidf-GBDT ≈ 0.674(LGBM fold0) > frozen 인코더 0.62~0.635. frozen 임베딩 단독은 GBDT 미달 → 인코더는 **구조피처 직렬화 + full fine-tune** 필요.
6. **제출 한도 20/일 (팀)**. 절대 자동 제출 금지.
7. **실시간 로깅**: 모든 `train.py`는 시작 시 line-buffered Tee를 설치해 표준 경로 `experiments/<exp>/train.log`에 실시간(줄 단위)으로 로그를 남긴다(포그라운드/백그라운드/에이전트 무관, `tail -f`로 관찰 가능). runner는 `python -u train.py 2>&1 | tee -a experiments/<exp>/train.log`로 실행. 모든 print는 `flush=True`.
