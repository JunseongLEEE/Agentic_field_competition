# EDA: PCA 기반 피처 필터링 + Feature Engineering + 소수 클래스 심층 분석

> **날짜**: 2026-07-01 05:29 KST
> **분석 대상**: train.jsonl 70,000행 + train_labels.csv (14-class, Macro-F1)
> **목적**: (1) 기존 피처 중 불필요한 것을 PCA+MI로 식별·제거, (2) 새로운 교차 피처 설계·검증, (3) 소수 클래스별 ML·Encoder 관점 신호 분석, (4) 데이터 노이즈 정량화

---

## 1. PCA + Mutual Information 기반 피처 필터링

### 1.1 분석 방법

기존 `features.py`(exp_001)에서 추출되는 57개 피처(TF-IDF 제외)에 대해:
- **Mutual Information (MI)**: 각 피처와 14-class 타겟 간 상호정보량 측정
- **PCA**: StandardScaler 후 full PCA → 각 피처의 explained variance 기여도 계산
- 두 지표를 교차해서 "MI도 낮고 PCA 기여도도 낮은" 피처를 DROP 후보로 선정

### 1.2 PCA 결과 요약

| 항목 | 값 |
|---|---|
| 전체 피처 수 | 57 |
| 90% 분산 설명에 필요한 PC | 34 |
| 95% 분산 설명에 필요한 PC | 38 |
| 99% 분산 설명에 필요한 PC | 44 |

→ **전체 피처의 약 1/3이 redundant** (분산 설명에 기여하지 않음)

PC1(12.9%): n_actions, history_len, turn_index가 지배 (세션 진행도 축)
PC2(6.4%): prompt_is_korean, prompt_len_chars, language_pref (언어 축)
PC3-4(5%): primary_lang, lang_entropy (프로젝트 아키타입 축)
PC6(3.6%): user_tier 단독 — 다른 피처와 독립이지만 라벨과 무관

### 1.3 DROP 확정 피처 (7개)

| 피처 | MI | PCA 기여 | DROP 사유 |
|---|---|---|---|
| `cnt_respond_only` | 0.000 | ≈0 | history에서 respond_only가 등장하는 경우 없음 (항상 0) |
| `user_tier` | 0.001 | 0.035 | enterprise/pro/free 모두 라벨 분포 거의 동일. 독립 축이지만 무의미 |
| `prompt_has_exclaim` | 0.000 | 0.062 | 전체 1%만 해당, 라벨과 무관 |
| `prompt_has_code` | 0.001 | 0.066 | 전체 1%만 해당, 라벨과 무관 |
| `prompt_is_korean` | 0.004 | 0.067 | `language_pref`와 r=0.75 — 중복, language_pref가 더 informative |
| `n_actions` | 0.137 | 0.073 | `history_len`과 r=1.000 (완전 중복). history_len 유지 |
| `prompt_len_chars` | 0.144 | 0.081 | `prompt_len_words`와 r=0.927. words가 더 해석적 |

### 1.4 높은 상관관계 쌍 (|r| > 0.7)

| 피처 A | 피처 B | r | 조치 |
|---|---|---|---|
| history_len | n_actions | 1.000 | n_actions 제거 |
| history_len | turn_index | 0.893 | 둘 다 유지 (turn_index에 추가 정보) |
| turn_index | elapsed_session_sec | 0.712 | 둘 다 유지 |
| lang_entropy | n_languages | 0.715 | 둘 다 유지 (GBDT에서 분할점 다름) |
| prompt_len_chars | prompt_len_words | 0.927 | chars 제거 |
| prompt_is_korean | language_pref | 0.747 | prompt_is_korean 제거 |

### 1.5 기존 피처 MI 랭킹 (상위 15)

| 순위 | 피처 | MI | 유형 |
|---:|---|---:|---|
| 1 | last_action | 0.251 | 시퀀셜 |
| 2 | second_last_action | 0.211 | 시퀀셜 |
| 3 | rule_WRAP_UP | 0.170 | 규칙 |
| 4 | prompt_len_chars | 0.144 | 텍스트 |
| 5 | turn_index | 0.137 | 세션 |
| 6 | history_len | 0.139 | 세션 |
| 7 | cnt_edit_file | 0.127 | 시퀀셜 |
| 8 | n_open_files | 0.127 | 워크스페이스 |
| 9 | prompt_len_words | 0.122 | 텍스트 |
| 10 | third_last_action | 0.119 | 시퀀셜 |
| 11 | prompt_n_sentences | 0.105 | 텍스트 |
| 12 | avg_user_prompt_len | 0.101 | 텍스트 |
| 13 | rule_RUN_TEST | 0.096 | 규칙 |
| 14 | last_result_lines | 0.075 | 시퀀셜 |
| 15 | rule_SEARCH | 0.074 | 규칙 |

**시각화**: `data_docs/eda_figures/pca_feature_analysis.png` (MI 바차트, PCA scree plot, MI vs PCA scatter, 2D PCA 클래스 분포)

---

## 2. 새로 엔지니어링한 피처 (25개)

### 2.1 설계 원칙

GBDT에서 여러 tree depth가 필요한 feature interaction을 **미리 하나의 카테고리로 합쳐서**, 얕은 트리에서도 같은 패턴을 학습할 수 있게 함.

### 2.2 신규 피처 정의 + MI 측정 결과

#### 최상위 그룹 (MI > 0.25, 기존 best 초과)

| 피처 | MI | 정의 | 고유값 수 |
|---|---:|---|---:|
| `prompt_intent` | 0.438 | 8개 정규식 규칙 중 첫 매치를 단일 카테고리로. 미매치 시 "GENERAL" | 9 |
| `action_trigram` | 0.426 | `f"{third_last}__{second_last}__{last_action}"` — 3-action 시퀀스 | 1,863 |
| `action_bigram` | 0.396 | `f"{second_last}__{last_action}"` — 2-action 시퀀스 | 183 |
| `turn_action` | 0.304 | `f"t{min(turn_index,7)}_{last_action}"` — 세션 단계 × 직전 action | 79 |
| `last_action_status` | 0.261 | `f"{last_action}__{'FAIL' if last_failed else 'OK'}"` — action+결과 | 22 |

#### 중상위 그룹 (MI 0.05 ~ 0.25)

| 피처 | MI | 정의 |
|---|---:|---|
| `modify_ratio` | 0.146 | history 내 수정 action(edit/write/apply) 비율 |
| `result_cat` | 0.142 | 마지막 result_summary → NONE/OK/FAIL/OTHER 분류 |
| `steps_remaining` | 0.139 | `max(0, 7 - turn_index)` — 세션 잔여 단계 |
| `unique_actions` | 0.122 | history 내 고유 action 종류 수 |
| `most_common_action_ratio` | 0.120 | history에서 가장 빈출 action의 비율 |
| `open_ext` | 0.117 | open_files 중 첫 파일의 확장자 (28종) |
| `session_phase` | 0.113 | turn_index 기반 4단계: early(≤2)/mid(3-4)/late(5-6)/wrapup(7+) |
| `explore_ratio` | 0.104 | history 내 탐색 action(read/grep/list/glob) 비율 |
| `prompt_len_bucket` | 0.085 | short(≤8단어)/medium(9-16)/long(17+) |
| `last_is_modify` | 0.078 | 직전 action이 수정계인지 (binary) |
| `consec_same` | 0.073 | 마지막 action이 연속 몇 번째인지 |
| `has_read_edit_cycle` | 0.054 | history에 read→edit 패턴 존재 여부 |
| `verify_ratio` | 0.054 | history 내 검증 action(test/bash/lint) 비율 |

#### 중하위 그룹 (MI 0.01 ~ 0.05)

| 피처 | MI | 정의 |
|---|---:|---|
| `last_is_explore` | 0.048 | 직전 action이 탐색계인지 |
| `workspace_archetype` | 0.037 | language_mix 키를 정렬 결합 (12종 프로젝트 아키타입) |
| `last_n_files` | 0.031 | 직전 action의 args.n_files 값 |
| `meta_ratio` | 0.029 | history 내 메타 action(ask/plan/web/respond) 비율 |
| `has_path_ref` | 0.025 | 프롬프트에 파일 경로/확장자 언급 여부 |
| `last_is_verify` | 0.016 | 직전 action이 검증계인지 |
| `recent_fails` | 0.013 | 최근 3 action 중 실패 횟수 |

### 2.3 신규 vs 기존 피처 비교

신규 피처 Top-5가 기존 best(last_action MI=0.251)를 **1.0~1.7배 초과**:
- `prompt_intent` 0.438 (1.7배)
- `action_trigram` 0.426 (1.7배)
- `action_bigram` 0.396 (1.6배)
- `turn_action` 0.304 (1.2배)
- `last_action_status` 0.261 (1.0배)

**시각화**: `data_docs/eda_figures/feature_engineering_mi.png` (기존=파랑, 신규=초록 바차트)

---

## 3. prompt_intent 피처의 커버리지 한계 분석

### 3.1 intent별 라벨 분포

| intent | 샘플 수 | 비율 | Top-3 라벨 |
|---|---:|---:|---|
| **GENERAL** | **46,375** | **66.2%** | edit_file 23%, grep_search 12%, read_file 12% |
| SEARCH | 7,718 | 11.0% | grep_search 31%, read_file 22%, glob_pattern 15% |
| SHOW_FILE | 5,275 | 7.5% | read_file 35%, grep_search 31%, glob_pattern 17% |
| WRAP_UP | 3,814 | 5.4% | **respond_only 100%** |
| RUN_TEST | 3,762 | 5.4% | run_tests 41%, run_bash 38%, lint 18% |
| PLAN_REQ | 1,587 | 2.3% | plan_task 34%, ask_user 29%, web_search 14% |
| ERROR_HELP | 785 | 1.1% | ask_user 40%, plan_task 17%, web_search 9% |
| LINT_CHECK | 522 | 0.7% | lint 24%, run_bash 22%, run_tests 21% |
| WEB_REF | 162 | 0.2% | plan_task 34%, web_search 28%, ask_user 19% |

### 3.2 클래스별 GENERAL 유입률

규칙에 안 걸려서 GENERAL로 빠지는 비율:

| 클래스 | GENERAL 비율 | 규칙 커버율 | 해석 |
|---|---:|---:|---|
| write_file | **99.1%** | 0.9% | 규칙 전무 — 텍스트 키워드(만들어줘 등)로 커버 가능 |
| edit_file | **97.4%** | 2.6% | 규칙 전무 — 시퀀셜 피처 의존 |
| apply_patch | **97.4%** | 2.6% | 규칙 전무 — 시퀀셜 피처 의존 |
| run_bash | 67.5% | 32.5% | RUN_TEST 규칙이 일부 커버 |
| lint_or_typecheck | 62.9% | 37.1% | LINT_CHECK + RUN_TEST 일부 |
| run_tests | 61.5% | 38.5% | RUN_TEST 규칙 커버 |
| read_file | 59.7% | 40.3% | SEARCH + SHOW_FILE |
| list_directory | 58.3% | 41.7% | SEARCH 규칙 일부 |
| grep_search | 57.5% | 42.5% | SEARCH + SHOW_FILE |
| glob_pattern | 58.6% | 41.4% | SEARCH |
| plan_task | 53.1% | 46.9% | PLAN_REQ + ERROR_HELP + WEB_REF |
| ask_user | 51.7% | 48.3% | ERROR_HELP + PLAN_REQ |
| web_search | 51.7% | 48.3% | WEB_REF + PLAN_REQ + ERROR_HELP |
| respond_only | 26.0% | **74.0%** | WRAP_UP 규칙이 강력 |

### 3.3 GENERAL 내부 클래스 분포

GENERAL 46,375건 안의 라벨 분포 (14개 클래스 모두 존재):
```
edit_file        23.5%   ← 제일 큰 덩어리
grep_search      12.3%
read_file        11.9%
apply_patch      10.1%
run_bash          7.4%
glob_pattern      6.7%
run_tests         6.0%
list_directory    5.4%
write_file        3.2%
lint_or_typecheck 3.1%
plan_task         3.1%
ask_user          3.0%
respond_only      2.9%
web_search        1.4%
```

→ `prompt_intent`가 GENERAL일 때 이 피처는 아무 판별력이 없으며, 다른 피처(action_bigram, turn_action 등)가 구분을 담당함.

### 3.4 규칙 추가 가능한 클래스

GENERAL에서 빠져나올 수 있는 강한 키워드가 있는 클래스:

| 클래스 | 키워드 | P(class|word) | 추가 규칙 가능성 |
|---|---|---:|---|
| write_file | 만들어줘, 써줘, 새로, from scratch | 90%+ | ✅ 매우 높음 |
| run_bash | 돌려서, 빌드, 태워보자, compile | 50% | ✅ 높음 |
| list_directory | 어떻게, 뭐뭐, 구조부터, 까보자 | 25% | △ 중간 (다른 클래스와 겹침) |
| edit_file | 고쳐줘, 손봐, 박아줘 | 75% | △ apply_patch와 겹침 |
| apply_patch | 일관되게, 패치로, 쳐줘 | 35~50% | △ edit_file과 겹침 |

---

## 4. 소수 클래스 심층 분석 (Macro-F1 핵심)

### 4.1 소수 클래스별 프로파일

#### web_search (n=1,273, 1.8%) — 가장 어려운 클래스

- **ML 신호**: 매우 약함. best last_action='read_file' → P(cls|la)=3.1%밖에 안 됨
- **Encoder 신호**: 중간. 키워드 '공식'(31%), '보통'(26%), '문서'(25%) 등
- **노이즈**: 11.2% — 소수 클래스 중 irreducible error가 가장 높음
- **no_history**: 9% (시퀀셜 신호 대부분 사용 가능)
- **특징**: plan_task, ask_user와 3-way 혼동 구간. 직전 action이 read_file/grep_search일 때 주로 발생 (turn 후반)
- **과제**: 키워드가 분산적이라 정규식 규칙으로 잡기 어려움. Encoder의 문맥 이해가 유리

#### write_file (n=1,481, 2.1%) — 텍스트 신호 최강

- **ML 신호**: 약함. 48%가 turn=1(NONE)이라 시퀀셜 피처 불가
- **Encoder 신호**: 매우 강함. '만들어줘' P(cls|w)=92%, '써줘' 91%, '골격' 91%
- **노이즈**: 0% — 프롬프트만으로 완벽 구분 가능
- **no_history**: 48% (절반이 세션 첫 턴)
- **특징**: "새 파일 만들어줘" 패턴이 거의 결정론적. turn_index 평균 2.0 (매우 초반)
- **과제**: ML 모델은 시퀀셜 신호 없이 텍스트 키워드에 의존해야 함. 규칙/Encoder 필수

#### lint_or_typecheck (n=2,283, 3.3%) — 시퀀셜+텍스트 복합 신호

- **ML 신호**: 강함. apply_patch 후 17%, edit_file 후 32%. 세션 후반(turn 평균 7.2)
- **Encoder 신호**: 강함. '타입체크' P(cls|w)=100%, '정적분석' 26%
- **노이즈**: 8.0%
- **no_history**: 3% (거의 모든 샘플에 히스토리 존재)
- **특징**: git_dirty=97% (거의 항상 변경 있음), ci_status passed 43%. 코드 수정 후 검증 단계
- **과제**: run_tests, run_bash와 3-way 혼동. "다시" 같은 짧은 프롬프트에서 특히 혼동

#### ask_user (n=2,701, 3.9%) — 에러 패턴 의존

- **ML 신호**: 약함. 23%가 turn=1. best last_action=NONE → P(cls|la)=7%
- **Encoder 신호**: 강함. 에러명(AssertionError 70%, TypeError 64%, '도와줄래' 59%)
- **노이즈**: 5.0%
- **no_history**: 23%
- **특징**: 에러 관련 프롬프트에서 ask_user/plan_task/web_search가 확률적으로 분기
- **과제**: 같은 에러 프롬프트가 ask_user(60%), plan_task(25%), web_search(15%)로 분산

#### plan_task (n=2,679, 3.8%) — 세션 초반 계획

- **ML 신호**: 약함. 42%가 turn=1(NONE). ask_user 후 13%
- **Encoder 신호**: 중간. '단계부터' 49%, '손대기' 50%, '세워줘' 52%
- **노이즈**: 7.6%
- **no_history**: 42%
- **특징**: "복잡해서 단계부터 잡아줘" 패턴. 세션 초반에 집중 (turn 평균 3.4)
- **과제**: ask_user와 혼동. "건드리기 전에 계획부터" vs "막혔는데 도와줘"의 미묘한 차이

### 4.2 클래스별 구조적 특성 요약표

| 클래스 | 비율 | turn 평균 | no_hist% | git_dirty% | last_failed% | 주 선행 action |
|---|---:|---:|---:|---:|---:|---|
| edit_file | 16.0% | 5.2 | 2% | 78% | 10% | read_file 23%, grep 18% |
| grep_search | 14.2% | 5.3 | 11% | 74% | 8% | grep 17%, read 13% |
| read_file | 13.2% | 4.5 | 16% | 66% | 7% | grep 20%, NONE 16% |
| glob_pattern | 7.5% | 5.7 | 11% | 73% | 6% | grep 22%, glob 17% |
| respond_only | 7.4% | 7.3 | 2% | 88% | 1% | edit 20%, test 15% |
| run_bash | 7.2% | 4.2 | 20% | 75% | 6% | NONE 20%, bash 19% |
| apply_patch | 6.9% | 7.5 | 0% | 95% | 14% | edit 22%, read 16% |
| run_tests | 6.5% | 6.4 | 3% | 97% | 11% | **edit 54%**, apply 14% |
| list_directory | 6.2% | 2.8 | **42%** | 62% | 4% | **NONE 42%**, plan 10% |
| **ask_user** | 3.9% | 4.4 | **23%** | 72% | 6% | NONE 23%, read 14% |
| **plan_task** | 3.8% | 3.4 | **42%** | 65% | 3% | **NONE 42%**, ask 11% |
| **lint_or_typecheck** | 3.3% | 7.2 | 3% | **97%** | 13% | apply 33%, edit 32% |
| **write_file** | 2.1% | 2.0 | **48%** | 51% | 2% | **NONE 48%**, list 25% |
| **web_search** | 1.8% | 6.0 | 9% | 70% | 11% | read 21%, grep 20% |

---

## 5. 데이터 노이즈 정량 분석

### 5.1 동일 프롬프트 → 다중 라벨 현상

| 항목 | 값 |
|---|---|
| Unique 프롬프트 수 | 63,257 |
| 다중 라벨 프롬프트 수 | 2,219 |
| 해당 샘플 수 | 5,795 (전체의 8.3%) |

**원인**: 시뮬레이터가 동일 프롬프트에 대해 **확률적으로** 다른 action을 선택하기 때문. 이는 프롬프트 텍스트만으로는 해소 불가능한 **irreducible noise**.

### 5.2 대표적 다중 라벨 패턴

```
"TypeError: NoneType 자꾸 나는데..." → ask_user:10, plan_task:2, web_search:1
"AttributeError 계속 뜨는데..."     → ask_user:7, plan_task:3, web_search:2
"다시..."                          → run_tests:4, lint:3, run_bash:2
"다시 돌려봐..."                    → run_tests:5, run_bash:2
```

핵심 혼동 축:
- **ask_user ↔ plan_task ↔ web_search**: 에러/계획 프롬프트에서 3-way 확률 분기
- **run_tests ↔ lint_or_typecheck ↔ run_bash**: "다시 돌려" 프롬프트에서 3-way 분기
- **edit_file ↔ apply_patch**: "고쳐줘/두 파일 같이 손봐" 프롬프트에서 2-way 분기

### 5.3 클래스별 텍스트-only 노이즈율

동일 프롬프트 내에서 다수결 라벨이 아닌 비율:

| 클래스 | 노이즈율 | 해석 |
|---|---:|---|
| respond_only | 0.0% | 프롬프트만으로 완벽 구분 |
| write_file | 0.0% | 프롬프트만으로 완벽 구분 |
| edit_file | 1.1% | 거의 깨끗 |
| read_file | 3.4% | 양호 |
| grep_search | 3.8% | 양호 |
| apply_patch | 4.0% | 양호 |
| run_bash | 4.7% | 약간 혼동 |
| glob_pattern | 4.7% | 약간 혼동 |
| list_directory | 4.7% | 약간 혼동 |
| ask_user | 5.0% | 혼동 |
| run_tests | 5.7% | 혼동 |
| plan_task | 7.6% | 혼동 심함 |
| lint_or_typecheck | 8.0% | 혼동 심함 |
| **web_search** | **11.2%** | **혼동 매우 심함** |

### 5.4 프롬프트 + last_action 조합 시 노이즈 감소

| 클래스 | text-only 노이즈 | text+last_action 노이즈 | 감소 |
|---|---:|---:|---:|
| web_search | 11.2% | **1.1%** | -10.1pp |
| lint_or_typecheck | 8.0% | **1.8%** | -6.2pp |
| plan_task | 7.6% | **1.5%** | -6.2pp |
| run_tests | 5.7% | **1.2%** | -4.6pp |
| ask_user | 5.0% | **1.9%** | -3.2pp |

→ **last_action을 조합하면 거의 모든 노이즈가 2% 미만으로 떨어짐**. 이는 시뮬레이터가 "같은 프롬프트라도 직전 action이 다르면 다른 action을 선택"하는 규칙을 사용했다는 증거.

---

## 6. ML 모델 관점 분석 (LightGBM)

### 6.1 ML 모델이 소수 클래스를 잡기 위해 필요한 것

| 소수 클래스 | ML이 잘 잡을 수 있는가? | 핵심 신호 |
|---|---|---|
| write_file | **어려움** — 48%가 turn=1(last_action=NONE), 텍스트에 의존해야 함 | TF-IDF "만들어줘/써줘/새로" + turn_index ≤ 2 |
| web_search | **어려움** — 시퀀셜 신호 약함, 텍스트도 분산적 | TF-IDF "공식/문서/보통" + turn 후반 |
| lint_or_typecheck | **가능** — apply_patch/edit_file 후 17~32% | last_action ∈ {apply_patch, edit_file} + turn ≥ 5 |
| ask_user | **부분 가능** — 에러 키워드 있으면 잡히지만 plan_task와 혼동 | rule_ERROR_HELP=1 + TF-IDF "도와줄래/stuck" |
| plan_task | **부분 가능** — turn=1이면서 계획 키워드 | rule_PLAN_REQ=1 + turn_index ≤ 3 |

### 6.2 ML 모델의 한계

1. **텍스트 의미 파악 불가**: TF-IDF는 "새로 만들어줘"와 "새로 고쳐줘"를 "새로" 토큰 공유로 혼동. "만들어줘" vs "고쳐줘"의 의미 차이를 잡으려면 bigram이 필수
2. **GENERAL 66% 문제**: prompt_intent=GENERAL인 46K건에서 ML은 시퀀셜 + 메타 피처에만 의존
3. **다국어 처리 불가**: 한/영이 같은 의미인데 TF-IDF에서는 완전 다른 피처로 처리됨

### 6.3 ML 모델에 유효한 피처 전략

1. **교차 피처**: action_bigram/trigram, turn_action, last_action_status — 이미 검증됨
2. **TF-IDF 강화**: word bigram(1,2) + char_wb(2,4) 결합으로 "만들어줘" 같은 핵심 토큰 포착
3. **규칙 확장**: GENERAL 비율을 줄이는 추가 규칙 (write_file/run_bash/list_directory용)
4. **클래스별 맞춤 피처**: write_file용 `has_create_keyword`, web_search용 `has_reference_keyword` 등

---

## 7. Encoder 모델 관점 분석 (DeBERTa)

### 7.1 각 클래스의 텍스트 판별 키워드 (P(class|word) 기준)

**텍스트만으로 거의 확정 가능한 클래스** (P(class|word) > 80%):

| 클래스 | 키워드 | P(cls|w) |
|---|---|---:|
| respond_only | 여기까지, 마무리, 요약해줘, summarize | **100%** |
| write_file | 만들어줘, 써줘, 골격, 만들자, 작성해줘 | **87~95%** |

**텍스트로 강한 단서가 있는 클래스** (P(class|word) 40~80%):

| 클래스 | 키워드 | P(cls|w) |
|---|---|---:|
| ask_user | assertionerror, typeerror, 도와줄래, stuck | 53~70% |
| edit_file | 지워줘, 손봐, 박아줘, 감싸줘, 넘기게 | 73~80% |
| run_bash | 떠보자, 태워서, 실행해서, compile, vet | 48~56% |
| plan_task | 세워줘, 손대기, 단계부터, 계획부터 | 48~59% |
| run_tests | 돌려봐요, 통과하나, 테스트만 | 44~56% |
| lint_or_typecheck | 타입체크(100%), 정적분석 | 26~100% |

**텍스트로 구분 어려운 클래스** (P(class|word) < 40%):

| 클래스 | 최고 키워드 | P(cls|w) | 이유 |
|---|---|---:|---|
| web_search | 정석인지, 공식 | 31~43% | 키워드 분산적, ask_user/plan_task와 혼동 |
| grep_search | 어디어디야, 훑어줄래 | 47~52% | read_file, glob_pattern과 겹침 |
| read_file | 읽어봐, 열어봐요 | 46~58% | grep_search, list_directory와 겹침 |
| apply_patch | 쳐줘, 일관되게 | 35~51% | edit_file과 심하게 겹침 |
| list_directory | 뭐뭐, 어떻게 | 23~30% | 키워드가 범용적 |
| glob_pattern | 훑어봐, 컴포넌트부터 | 28~34% | grep_search와 겹침 |

### 7.2 Encoder가 ML보다 유리한 지점

1. **다국어 의미 통합**: "만들어줘" = "create a new file" = write_file. TF-IDF는 이걸 별개 토큰으로 보지만 DeBERTa는 같은 의미로 인식
2. **문맥 의존적 판별**: "고쳐줘" 앞에 "두 파일 같이"가 오면 apply_patch, "이 줄"이 오면 edit_file — 이런 문맥 의존 패턴을 학습 가능
3. **희소 키워드 일반화**: "타입체크"(39건)만 학습하고도 "타입 검사"(0건)를 유추 가능

### 7.3 작은 Encoder의 한계와 대응

대회 제약(T4, zip≤1GB, 추론≤10분, 30K test rows):
- DeBERTa-v3-small (44M params, ~170MB) → T4에서 30K rows 추론 가능 (약 5~7분)
- DeBERTa-v3-base (86M params, ~340MB) → 경계선 (8~10분, 위험)

작은 Encoder는:
- ✅ 키워드 패턴 학습 (TF-IDF 상위호환) — 충분
- ✅ 다국어 의미 매핑 — 가능
- ⚠️ 긴 history 문맥 이해 — 제한적 (max_length 512면 history 대부분 잘림)
- ❌ 복잡한 시퀀셜 추론 — 불가 (이건 ML 피처가 담당)

### 7.4 2-Stage 파이프라인 설계 근거

```
[Stage 1: DeBERTa-v3-small]
  입력: current_prompt (+ 선택적으로 last user turn)
  출력: 14-class 확률 벡터 (soft intent)
  역할: 텍스트 의미 기반 판별. respond_only(100%), write_file(92%) 확정 +
        나머지 클래스의 확률 분포 제공

[Stage 2: LightGBM]
  입력: DeBERTa 14-class 확률 + 시퀀셜 피처(bigram, trigram, turn_action 등)
        + 메타 피처(turn_index, session_phase, workspace 등)
  출력: 최종 14-class 예측
  역할: 텍스트로 애매한 것(apply_patch vs edit_file, ask_user vs plan_task)을
        구조적 맥락으로 최종 결정
```

**각 클래스에서 어떤 Stage가 주로 결정하는가**:

| 클래스 | Stage 1 (DeBERTa) | Stage 2 (LightGBM) |
|---|---|---|
| respond_only | **확정** (100%) | 패스스루 |
| write_file | **확정** (92%) | 패스스루 |
| lint_or_typecheck | 단서 제공 (타입체크 100%) | 시퀀셜로 보강 (apply_patch 후 17%) |
| ask_user | 에러 키워드 60% | plan_task와 분리 (last_action 기반) |
| plan_task | 계획 키워드 49% | ask_user와 분리 (turn_index 기반) |
| web_search | 약한 단서 (31~43%) | **핵심** — 시퀀셜+텍스트 확률 조합 |
| edit_file | 수정 키워드 73% | apply_patch와 분리 (시퀀셜) |
| apply_patch | 약한 단서 (35~51%) | **핵심** — 시퀀셜(edit 후, 세션 후반) |
| run_tests | 테스트 키워드 44~56% | run_bash와 분리 (시퀀셜) |
| run_bash | 실행 키워드 48~56% | run_tests와 분리 |
| grep_search | 검색 키워드 47~52% | read_file/glob과 분리 (시퀀셜) |
| read_file | 조회 키워드 46~58% | grep_search와 분리 |
| glob_pattern | 약한 단서 (28~34%) | grep_search와 분리 (시퀀셜) |
| list_directory | 약한 단서 (23~30%) | **핵심** — turn=1 + NONE 조합 |

---

## 8. 결론 및 다음 단계

### 8.1 데이터 노이즈 현황

- 전체 irreducible noise: **텍스트-only 8.3%, 텍스트+last_action 조합 시 ~1.5%**
- Macro-F1 이론적 ceiling (텍스트+last_action): 약 **0.98** (노이즈 제외)
- 현재 exp_001 CV: **0.666** → 개선 여지 ~0.31

### 8.2 핵심 Action Items

1. **즉시 실행**: exp_010 (PCA 필터링 + 25 신규 피처 + LightGBM) — 현재 학습 중
2. **규칙 확장**: write_file/run_bash/list_directory용 intent 규칙 추가 → GENERAL 66% → ~50%
3. **DeBERTa soft-intent 실험**: DeBERTa-v3-small로 current_prompt → 14-class 확률 벡터 생성 → LightGBM 피처로 투입
4. **history 텍스트 활용**: 마지막 user turn의 TF-IDF를 추가 피처로 (현재 current_prompt만 사용)
5. **클래스별 threshold 튜닝**: web_search 같은 소수 클래스의 예측 threshold를 낮춰서 recall 확보

### 8.3 exp_010 예상 효과

기존 exp_001 대비 변경 사항:
- 7개 노이즈 피처 제거
- 25개 교차/구조 피처 추가 (MI 상위 5개가 기존 best의 1.7배)
- 5-fold CV (기존 3-fold), num_leaves 127 (기존 63), n_estimators 500 (기존 350)

**가설**: CV Macro-F1 0.67 → **0.72+** 기대 (주로 소수 클래스 recall 개선에서)
