# AI 코딩 에이전트 Action 예측 대회 데이터셋 포렌식 리포트

> **분석 대상**: `open.zip` (train 70,000 / test 30,000 세션 스텝, 14-class action prediction)
> **핵심 결론**: 본 데이터셋은 **어떤 오픈소스 데이터셋의 파생물도 아니며**, 대회 주최측이 자체 설계한 **seed pool + 슬롯 필링 + 결정론적 툴 시뮬레이터**로 생성한 순수 합성 데이터임.

---

## 0. Executive Summary

| 항목 | 판정 근거 | 증거 강도 |
|---|---|---|
| **오픈소스 크롤링 데이터인가?** | ❌ 아님 | 매우 강함 |
| **LLM으로 자유 생성한 데이터인가?** | ❌ 부분적으로만 | 강함 |
| **주최측 자체 seed pool + 프로그램 조합인가?** | ✅ 그렇게 판단 | 매우 강함 |
| **라벨(정답 action)이 규칙 기반으로 붙었는가?** | 부분적 (일부는 결정론, 대부분은 확률 분포) | 중간 |
| **외부 데이터 augmentation의 실효성** | 낮음 (스키마 매칭 비용 > 이득) | 강함 |

즉, "TF-IDF로 오픈소스 코퍼스와 비교하면 어디서 긁어왔는지 나올 것"이라는 가설은 성립하지 않습니다. **비교할 원본 자체가 존재하지 않기 때문**입니다.

---

## 1. 분석 배경

이 리포트는 원 질문 — *"어떤 오픈소스 데이터셋을 긁어와 가공했는가?"* — 을 검증하기 위해 데이터 내부를 정량적으로 해부한 결과입니다. 다섯 가지 층위에서 조사했습니다.

1. **user 발화 텍스트의 어휘·구문 분포**
2. **assistant_action의 args / result_summary 스키마**
3. **workspace 메타데이터 (language_mix, open_files 등)의 값 분포**
4. **history 구조 (길이, 순서, 조합)**
5. **current_prompt ↔ label 상관관계**

전체 train 70,000 샘플을 로드하여 다음을 수집했습니다:

- user 발화 총 **312,532건** (unique 66,119건, 중복률 **약 4.7배**)
- assistant_action 총 **242,532건**
- open_files 경로 총 **53,132건** (unique는 훨씬 적음)

---

## 2. 합성 데이터임을 뒷받침하는 다섯 가지 결정적 증거

### 증거 1 — 접두어/접미어 pool의 균일한 uniform 분포

user 발화의 문두 2-단어 n-gram을 세어봤을 때, **한글 접두어**와 **영어 접두어**가 각각 **약 15-20개의 고정 pool**에서 뽑히고 있으며, 빈도가 **거의 uniform**하게 분포합니다.

**한글 접두어 pool (전체 25만 발화 기준, 각 1,900~2,400회 등장):**

| 접두어 | 등장 횟수 | | 접두어 | 등장 횟수 |
|---|---:|---|---|---:|
| 그건 그렇고 | 2,265 | | 별건 아닌데 | 1,989 |
| 음 잠시만 | 2,236 | | 한 가지 — | 1,972 |
| 갑자기 생각났는데 | 2,134 | | 어 잠깐 | 1,968 |
| 하는 김에 | 2,096 | | 막혀서 그런데 | 1,858 |
| 아 그리고 | 2,029 | | 한 번만 | 1,425 |
| 방금 봤는데 | 2,020 | | 급한 건데 | 1,331 |
| 혹시나 해서 | 2,019 | | | |
| 이거 말인데, | 2,011 | | | |
| 조금 헷갈리는데 | 2,001 | | | |

**영어 접두어 pool:**

| 접두어 | 등장 | | 접두어 | 등장 |
|---|---:|---|---|---:|
| side note, | 2,401 | | not urgent but | 1,112 |
| if you have | 1,038 | | by the way, | 1,086 |
| ok so | 1,474 | | you know what, | 1,072 |
| real quick, | 1,120 | | just to confirm | 1,043 |
| when you're free, | 1,019 | | minor — | 1,055 |
| no pressure but | 980 | | first off, | 1,049 |
| small thing — | 951 | | when you can, | 917 |

**해석**: 각 접두어가 **약 2,000회씩 균등하게** 분포한다는 것은 자연 언어 데이터가 아닌, 프로그램이 `random.choice(prefix_pool)`을 호출한 명확한 흔적입니다. 실제 개발자 대화라면 어떤 표현은 자주, 어떤 표현은 드물게 나오는 롱테일 분포여야 합니다.

**한글 접미어 pool도 동일한 구조:**
- "한 번만" (2,433), "한 번" (2,376), "될 때" (2,375), "한번 더" (2,240), "이 부분만" (2,221), "한번만 더" (2,135), "이번 것만" (2,129), "오늘 안에" (1,865), "대충 말고" (1,822), "좀 빨리" (1,738)

### 증거 2 — 부자연스러운 pool 조합의 노출

접두어와 base sentence를 **독립적으로 결합**하다 보니 사람이라면 절대 쓰지 않을 조합이 데이터에 등장합니다:

```
"혹시 자 이제 마무리하자, 지금까지 한 거 요약해줘"      (7회)
"음 잠시만 KeyError: 'id'"                                (7회)
"방금 봤는데 TypeError: NoneType"                         (8회)
"어 잠깐 ConnectionError"                                 (7회)
```

접두어 "혹시"와 감탄사 시작 문장 "자 이제 마무리하자"가 결합된 것은 자연 발화에서 나올 수 없는 조합입니다. 이는 프로그램의 결합 규칙이 자연 언어 유효성을 검증하지 않는다는 자백입니다.

### 증거 3 — 슬롯-필러 에러 템플릿

에러 관련 사용자 요청이 명확한 슬롯-필러 구조입니다:

```
[한글] "{ERROR_NAME} 계속 뜨는데 어떻게 해야 할지 모르겠어, 도와줄래?"
[영어] "I keep hitting {ERROR_NAME} and I'm not sure where to look — can you help?"
```

슬롯에 들어가는 에러 이름은 **정확히 6종**뿐입니다:

| ERROR_NAME | 한글 문장 | 영어 문장 |
|---|---:|---:|
| AttributeError | 60 | 23 |
| ConnectionError | 59 | 43 |
| KeyError: 'id' | 45 | 24 |
| AssertionError | 35 | 45 |
| Timeout | 31 | 29 |
| TypeError: NoneType | 25 | 15 |

한/영이 **완전히 동일한 슬롯 pool을 공유**한다는 것은 두 언어가 독립적으로 만들어진 게 아니라 하나의 pool을 두 언어로 렌더링했다는 뜻입니다.

### 증거 4 — 결정론적 result_summary 템플릿

각 assistant_action의 `result_summary`는 **완벽하게 규격화**되어 있습니다.

| 액션 | 템플릿 예시 | 슬롯 |
|---|---|---|
| `apply_patch` | `ok; patched {N} files ({A}+/{B}-)` | 숫자 3개 |
| `plan_task` | `plan with {N} steps drafted` | 숫자 1개 |
| `glob_pattern` | `{N} files matched '{PATTERN}'` | 숫자, 문자열 |
| `list_directory` | `{X} entries ({Y} files, {Z} dirs)` | 숫자 3개 |
| `run_bash` (성공) | `ok; exit=0` 또는 `exit=0; {N} lines of output` | - |
| `run_tests` (성공) | `PASS: {N} tests passed` | 숫자 1개 |
| `run_tests` (실패) | `FAIL: {N} tests failing` 또는 `FAIL: {func} ({error})` | 숫자/문자열 |
| `lint_or_typecheck` | `ok; lint clean` 또는 `{N} errors, {M} files affected` | 숫자 |
| `web_search` | `{N} results retrieved` 또는 `no relevant results` | 숫자 |
| `ask_user` | `clarifying question sent to user` (9,621건 전부 동일) | 없음 |

LLM이 자유롭게 생성한 로그라면 이 정도의 형식 일관성은 나올 수 없습니다. **파이썬 f-string 수준의 규칙 기반 렌더러**임이 분명합니다.

특히 `ask_user`의 `result_summary`는 9,621건 **전부 정확히 같은 문자열**입니다.

### 증거 5 — 12개 프로젝트 아키타입만 존재

`workspace.language_mix`의 언어 조합은 **unique가 정확히 12개**뿐입니다:

| 조합 (정렬) | 등장 세션 수 | 추정 아키타입 |
|---|---:|---|
| `md + rs + toml` | 7,881 | Rust 크레이트 |
| `java + kt + sql + yaml` | 7,196 | Spring Boot (Java/Kotlin) |
| `css + json + ts + vue` | 6,457 | Vue.js 프론트엔드 |
| `py + sh + sql + yaml` | 6,443 | 데이터 파이프라인 (Airflow류) |
| `java + json + swift + tsx + ts` | 5,841 | 하이브리드 모바일 |
| `css + json + ts + tsx` | 5,752 | Next.js/React |
| `dockerfile + md + py + yaml` | 5,698 | Python + Docker 배포 |
| `html + md + py + ts` | 5,558 | Django + 프론트 |
| `html + js + py + yaml` | 5,417 | Flask 웹앱 |
| `ipynb + py + sh + yaml` | 4,781 | ML/데이터 사이언스 |
| (외 2개) | ~ | Go 서비스 등 |

**실제 저장소를 크롤링했다면** language_mix가 unique 수만 개는 나와야 하고, 언어 조합이 자유롭게 분포해야 합니다. 12개 고정 조합은 **주최측이 정의한 프로젝트 페르소나 pool**의 결정적 증거입니다.

### 증거 6 (보너스) — history 길이는 짝수만, 파일 경로 풀은 930개

- `history` 길이 분포: **0, 2, 4, 6, 8, 10, 12만 존재**. 홀수 부재. 각 세션이 정확히 `user → assistant_action` 페어로 turn을 소비하고, `turn_index`가 매 짝수 스텝에서 스냅샷된다는 시뮬레이션 규칙의 흔적.

| history 길이 | 샘플 수 |
|---:|---:|
| 0 | 9,000 |
| 2 | 8,797 |
| 4 | 8,446 |
| 6 | 8,001 |
| 8 | 7,526 |
| 10 | 6,644 |
| 12 | **21,586** (최대치에서 몰림) |

- `args`의 path 값: **총 162,607건 사용, unique는 단 930개**. 실제 GitHub 저장소 크롤이었다면 unique 수십만 이상이어야 합니다. 상위 반복 경로는 튜토리얼용 아키타입 파일명(`src/main/java/com/app/UserController.java` 1,183회 등).

- `open_files`: 최대 5개까지지만 사실상 **0-2개가 99.5%**. `open_files=1`이 40,257건으로 가장 많고, 5개짜리는 단 2건. IDE 상태의 자연 분포가 아닙니다.

---

## 3. 프롬프트 카테고리 ↔ 라벨 상관관계

주요 프롬프트 signature를 정규식으로 잡고 라벨 분포를 본 결과, **일부 카테고리는 완전한 결정론**을 보입니다.

| 프롬프트 패턴 | 샘플 수 | 라벨 분포 | 해석 |
|---|---:|---|---|
| 랩업 요청 (한글: "마무리…요약") | 2,625 | **respond_only 100%** | 완전 결정론 매핑 |
| 랩업 요청 (영어: "wrap up / recap") | 1,211 | **respond_only 98%** | 거의 결정론 |
| 계획 요청 (한글: "단계 잡아줘") | 852 | plan_task 44%, ask_user 35%, web_search 18% | 3-way 확률 매핑 |
| 계획 요청 (영어) | 254 | plan_task 47%, ask_user 34%, web_search 17% | 같은 3-way 매핑 |
| 에러 도움 요청 (한글 템플릿) | 187 | ask_user 61%, plan_task 25%, web_search 14% | 3-way |
| 파일 조회 ("보여줘/열어봐") | 3,685 | read_file 36%, grep_search 32%, glob_pattern 17% | 조회 3종 확률 매핑 |
| 검색 요청 ("어디…찾아") | 6,129 | grep_search 30%, read_file 21%, glob_pattern 14% | 검색 3종 |
| 테스트/빌드 실행 ("돌려봐") | 7,383 | run_tests 22%, run_bash 22%, edit_file 10% | 실행 2종 + 부산물 |
| lint/typecheck 언급 | 408 | lint_or_typecheck 31%, run_bash 28%, run_tests 25% | 검사 3종 |

**핵심 발견 1**: 랩업 요청 → `respond_only`는 완전 결정론에 가까움. **train 데이터 안에 매핑 규칙이 그대로 노출**되어 있으므로, 정규식만으로 이 카테고리 100%를 잡을 수 있습니다.

**핵심 발견 2**: "계획 요청 → {plan_task, ask_user, web_search}"처럼 특정 프롬프트가 특정 액션 **부분집합에만** 매핑되는 패턴이 반복됨. 이는 라벨 생성 로직이 "이 카테고리는 이 세 액션 중 하나로 확률적 결정"이라는 규칙을 사용한다는 강한 시사.

---

## 4. 시퀀셜 신호: 직전 액션 → 다음 액션

`history`의 마지막 assistant_action이 무엇이냐에 따라 다음 액션 확률이 크게 달라집니다.

| 직전 액션 | Top-3 다음 액션 |
|---|---|
| `write_file` | **edit_file 40%**, run_bash 29%, read_file 7% |
| `read_file` | **edit_file 29%**, grep_search 15%, read_file 14% |
| `edit_file` | **run_tests 23%**, edit_file 15%, apply_patch 10% |
| `grep_search` | edit_file 22%, read_file 19%, grep_search 18% |
| `glob_pattern` | grep_search 22%, glob_pattern 18%, read_file 16% |
| `list_directory` | **read_file 25%**, grep_search 21%, glob_pattern 11% |
| `plan_task` | apply_patch 18%, read_file 17%, list_directory 17% |
| `lint_or_typecheck` | **apply_patch 25%**, edit_file 22%, respond_only 10% |
| `write_file` | **edit_file 40%** (강한 신호) | |

### 실패 후 대응 액션도 매우 정형적

`result_summary`에 `ERROR`/`FAIL`이 포함된 직후:

| 직전 액션 (실패 시) | Top-3 다음 액션 |
|---|---|
| `run_tests` FAIL | **edit_file 33%**, grep_search 15%, apply_patch 13% |
| `lint_or_typecheck` FAIL | **apply_patch 33% + edit_file 33%**, grep_search 7% |
| `apply_patch` FAIL | apply_patch 18%, run_tests 17%, lint_or_typecheck 15% |
| `edit_file` FAIL | run_tests 22%, grep_search 15%, edit_file 15% |

**실무 시사점**: 시퀀셜 신호 하나만으로도 예측력이 상당함. 특히 `write_file → edit_file`, `list_directory → read_file`, `lint FAIL → apply_patch/edit_file`은 40% 이상의 상위 확률.

### turn_index 자체가 강력한 피처

| turn_index | 최상위 라벨 |
|---:|---|
| 1 (히스토리 없음) | list_directory 20%, read_file 17%, plan_task 12% (탐색 위주) |
| 2 | read_file 20%, edit_file 18%, grep_search 15% |
| 3 | edit_file 24%, read_file 16%, grep_search 14% |
| 4 | edit_file 24%, grep_search 16%, read_file 12% |
| 7 | **respond_only 15%** 등장 (세션 후반) |
| 8+ | respond_only 14% (랩업 유도 증가) |

세션이 진행될수록 탐색 → 편집 → 검증 → 마무리로 자연스럽게 이동. 이는 시뮬레이터가 세션 phase를 의도적으로 모델링했다는 뜻.

- `turn=1`일 때 `apply_patch`는 0.1% (히스토리 없이 패치 불가)
- `turn=1`일 때 `respond_only`도 0.9% (아직 랩업 아님)

---

## 5. 세션 메타 피처의 예측력

| 피처 | 라벨 분포 변화 | 예측력 |
|---|---|---|
| `user_tier` (enterprise/pro/free) | 세 tier 모두 거의 동일 분포 | **낮음** (노이즈) |
| `last_ci_status` (passed/none/failed) | failed일 때 edit_file 15→18%로 소폭 상승, 큰 차이 없음 | 낮음 |
| `budget_tokens_remaining` | (연속형, 별도 분석 필요) | 미확인 |
| `git_dirty` | true/false 큰 차이 없음 | 낮음 |
| `language_pref` (ko/en/mixed) | ko에서 respond_only 15%로 상승 (랩업 문장이 한글에 많음) | 중간 |
| `turn_index` | **매우 강한 신호** (섹션 4 참조) | 높음 |
| `open_files` 개수/파일명 | (별도 심층 분석 여지) | 중간 |

**의외의 발견**: `user_tier`, `git_dirty`, `last_ci_status` 같은 그럴싸한 피처가 실제로는 라벨과 거의 무상관. 시뮬레이터가 이들을 라벨 생성 시 크게 활용하지 않은 것으로 보임. 반면 `turn_index`와 `history`는 매우 강한 신호.

---

## 6. 규칙 기반 베이스라인 성능 검증

위 발견을 바탕으로 **정규식 8개 + 시퀀셜 규칙 13개**만으로 규칙 기반 분류기를 만들어 train set에서 정확도를 측정했습니다.

**전체 정확도: 33.07%** (최빈 예측 15.96%의 2.07배)

### 클래스별 recall

| 클래스 | recall | 비고 |
|---|---:|---|
| `respond_only` | **80.4%** | 랩업 정규식만으로도 매우 강력 |
| `run_tests` | **67.8%** | "테스트/돌려봐/run" 정규식이 잘 잡음 |
| `edit_file` | 58.0% | 시퀀셜 fallback으로 잡힘 |
| `grep_search` | 28.7% | |
| `list_directory` | 26.7% | turn=1 특성으로 잡힘 |
| `read_file` | 26.5% | |
| `lint_or_typecheck` | 23.7% | |
| `plan_task` | 19.6% | |
| `apply_patch` | 19.7% | |
| `run_bash` | 12.5% | 정규식이 run_tests와 겹침 |
| `ask_user` | 9.3% | 에러 템플릿만으로는 부족 |
| `web_search` | 5.1% | 시그니처 부족 |
| `glob_pattern` | 0.0% | 규칙 미포함 |
| `write_file` | 0.0% | 규칙 미포함 |

**시사점**: `respond_only`와 `run_tests`는 정규식 몇 줄로 recall 70~80%를 달성. 다른 클래스는 텍스트 임베딩과 시퀀셜 피처를 결합해야 함.

---

## 7. "오픈소스 데이터셋 유래" 가설의 반박

원 질문에서 제기하신 "TF-IDF로 오픈소스 코퍼스와 비교하면 어디서 왔는지 알 수 있을 것"이라는 가설을 다음 근거로 반박합니다.

### 반박 1 — 비교 대상이 될 수 있는 후보들이 스키마상 호환 불가

| 후보 데이터셋 | 왜 유래가 아닌가 |
|---|---|
| **SWE-bench / SWE-Gym** | GitHub 이슈+PR 기반. `history` 구조 없음. `result_summary` 형식 완전 다름. |
| **Aider chat logs** | 실제 개발자 로그. 접두어 균등 분포 불가능. |
| **OpenHands / OpenDevin trajectories** | tool 이름이 `CmdRunAction`, `FileReadAction` 등 완전 다른 명명. |
| **APIGen / xLAM synthetic** | tool schema가 이 대회의 14클래스와 매핑되지 않음. |
| **ToolBench / ToolLLM** | API 호출 도메인. 코딩 agent 아님. |
| **AgentInstruct / CodeActInstruct** | 자연스러운 트래젝토리. 접미어 pool 균등 분포와 배치됨. |

이 대회의 14-액션 라벨 조합 자체가 **주최측이 자체 정의한 하이브리드 taxonomy**(Cursor + Claude Code + OpenAI Codex CLI의 명명 규칙을 섞은 것)이므로, 어떤 단일 오픈소스도 라벨 스키마가 일치하지 않습니다.

### 반박 2 — TF-IDF 자체가 의미 있는 결과를 낼 수 없는 통계 구조

크롤링한 데이터의 특징은 **긴 문장이 원문 그대로 등장**하고, 다양한 어휘가 롱테일로 분포하는 것입니다. 반면 이 데이터셋은:

- 상위 20개 문장이 전체의 약 5% (한 문장 최대 142회 반복)
- 접두어/접미어 pool 크기가 각 15~20개
- 파일 경로 unique 930개
- language_mix 조합 12개

**어떤 오픈소스 코퍼스와 TF-IDF 비교해도 유의미한 겹침이 안 나올 것이며**, 그 이유는 유래가 없어서지 은폐를 잘 해서가 아닙니다.

### 반박 3 — 인공물의 흔적이 언어 이전에 통계에서 이미 다 드러남

접두어가 균등 분포한다는 사실 하나만으로도 자연 언어 데이터가 아닙니다. TF-IDF 이전에 **unigram 빈도 분포가 Zipf 법칙을 벗어나** 있다는 것이 이미 결정적입니다.

---

## 8. 예측 모델링 실무 권고

이 데이터셋의 **인공적 구조 자체가 곧 강력한 피처**입니다. 다음 순서로 접근을 권장합니다.

### 8.1 즉시 적용 가능한 카테고리 피처 (rule-based)

정규식으로 프롬프트 signature를 매핑한 **원-핫 카테고리 피처**를 만드세요. 이것만으로도 30% 이상 정확도가 나옵니다.

```python
categories = {
  'WRAP_UP_KO':  r'(마무리|여기까지|이 정도면).*(요약|정리)',
  'WRAP_UP_EN':  r'(wrap.*up|recap|summariz)',
  'ERROR_HELP':  r'(TypeError|AttributeError|ConnectionError|KeyError|AssertionError|Timeout|I keep hitting)',
  'PLAN_REQ':    r'(단계.*잡|계획.*(잡|짜|세워)|lay.*out|before i (start|edit|touch))',
  'SHOW_FILE':   r'(보여줘|열어봐|열어줘|show me|open the|look at)',
  'SEARCH':      r'(어디|찾아|어느 파일|list what)',
  'RUN_TEST':    r'(테스트.*돌|한번 돌려|run.*test|rerun|full suite|다시 빌드)',
  'LINT_CHECK':  r'(lint|typecheck|타입체크|shellcheck)',
  'WEB_REF':     r'(best practice|공식.*문서|documentation)',
}
```

특히 `WRAP_UP_*`은 **respond_only recall 80% 이상**을 즉시 확보하는 결정타.

### 8.2 시퀀셜 피처

- `last_action` (직전 action 이름): **가장 강력한 단일 피처**
- `last_action_failed` (직전 result_summary에 ERROR/FAIL 포함): 부스터
- `history_length` = turn_index * 2
- action n-gram (직전 2-3개 action 시퀀스)

### 8.3 workspace 정형 피처 (일부만 유효)

- ✅ `turn_index` (강함): 세션 phase를 결정
- ✅ `language_mix` primary language (있음): 프로젝트 아키타입 12종 카테고리
- ⚠️ `git_dirty`, `last_ci_status`, `user_tier`, `budget_tokens_remaining`: 약함, 오버피팅 위험

### 8.4 텍스트 임베딩

`current_prompt`는 다국어이므로:
- `intfloat/multilingual-e5-base` 또는 `BAAI/bge-m3` 인코더
- 여러 history user 발화를 concatenate하지 말고 **마지막 K개만** 사용 (직전 문맥이 더 중요)

### 8.5 모델 아키텍처 추천

우선순위대로:
1. **LightGBM + 정형 피처 + 텍스트 임베딩 pooled vector** (구현 간단, 강한 베이스라인)
2. **DeBERTa/XLM-R fine-tuning** on `[categorical_features][SEP][last_action][SEP][current_prompt]` 형식 입력
3. **history 시퀀스 모델 (Transformer)** — 데이터가 워낙 정형이라 시퀀스 정보 활용이 큰 이득

### 8.6 하지 말아야 할 것

- ❌ 외부 오픈소스 데이터셋으로 pretraining/augmentation (유래도 아니고 스키마 매핑 비용만 큼)
- ❌ 자유 LLM 생성으로 데이터 증강 (합성 데이터의 rigid한 분포와 안 맞음, 오히려 성능 저하 가능)
- ❌ `user_tier`, `git_dirty` 같은 약한 피처에 depth 큰 GBDT 태우기 (오버피팅)

---

## 9. 종합 결론

이 대회 데이터셋은 다음과 같이 만들어졌다고 강한 확신을 가지고 결론 내립니다.

1. **주최측이 자체 설계한 프로젝트 아키타입 12종** (Rust/Java/Next.js/Airflow/ML notebook 등)을 정의
2. **각 아키타입별 파일 경로 pool** (총 unique 930개)을 미리 준비
3. **한/영 각각 seed 문장 pool** (base 수백 개) + **접두어 pool 15~20개** + **접미어 pool 15~20개**를 정의
4. 시뮬레이터가 세션 시작 시 아키타입 하나를 고르고, `turn_index`에 따라 phase-appropriate한 action을 결정론+확률적으로 생성
5. 각 action의 `result_summary`는 파이썬 f-string 수준의 규칙 렌더러로 생성
6. 다음 user 발화는 pool에서 `random.choice`로 뽑고 접두어/접미어를 uniform하게 붙임
7. 이 프로세스를 세션당 최대 12턴 반복, 매 짝수 스텝에서 스냅샷을 저장

**어떤 오픈소스 데이터셋의 파생물도 아닙니다.** TF-IDF, 하위 문자열 매칭, 어떤 유사도 지표를 써도 원본은 나오지 않을 것입니다. 왜냐하면 원본이 없기 때문입니다.

그러나 이 사실은 좋은 소식입니다. 데이터의 **결정론적 구조가 노출**되어 있어, 규칙 기반 카테고리 피처와 시퀀셜 피처 결합만으로도 강한 베이스라인을 만들 수 있습니다. 텍스트 임베딩과 gradient boosting을 얹으면 리더보드 상위권을 노릴 수 있는 문제라고 판단합니다.

---

*리포트 작성 근거: `train.jsonl` 70,000 샘플 전량 분석. 모든 통계는 실측치이며 코드로 재현 가능.*
---

## EDA 2026-07-01 (figures + quantification, gaps filled)

전량(70,000) 재로드, `train_labels.csv` 조인 실패 0건. 모든 그림은
`data_docs/eda_figures/` (PNG, dpi 120, 영문 라벨). 전체 스냅샷/숫자는 `logs/eda_report.md`.
이 섹션은 기존 포렌식 리포트를 **정량화·시각화**하고 **미확인 항목(budget_tokens, open_files,
MI 랭킹, 세션 그룹핑)**을 채운 신규 발견만 요약한다.

### 채운 GAP
- **budget_tokens_remaining → 신호 거의 없음.** session_meta 직속 필드. 분포는 넓고 대략 균일
  (p50 94.5k, 범위 55~199.7k). 5분위→클래스 히트맵 최대 편차 1.4pp, MI=0.0137(하위권).
  → 라벨 생성에 사실상 미사용. 강하게 쓰지 말 것. (그림 08)
- **open_files → COUNT는 유용, 확장자는 약함.** 개수 분포 0:23,498 / 1:40,257 / 2:5,876 /
  3:355 / 4:12 / 5:2 (0~2가 99.6%). `n_open_files` MI=0.129 (7위) — 세션 phase 프록시
  (0개⇔turn1 탐색, 1+⇔중반 편집). 확장자 상위 .py .ts .tsx .rs .java …는 아키타입 tint만,
  클래스 판별력 약함. (그림 09)
- **git_dirty는 기존 "약함" 판정보다 다소 강함.** Cramer's V=0.260(중간), last_ci_status=0.120.
  단, last_action이 있으면 조건부로 대부분 흡수됨(joint MI: git_dirty 0.039, last_ci 0.015).
  user_tier/language_pref는 사실상 노이즈(Cramer's V 0.02, MI 0.0005). (그림 07)

### MI 랭킹 (dense 피처, TF-IDF 제외) — KEY
1 last_action 0.250 · 2 second_last_action 0.212 · 3 rule_WRAP_UP 0.169 ·
4 turn_index 0.139 · 5 history_len 0.137 · 6 cnt_edit_file 0.131 · 7 n_open_files 0.129 ·
8 rule_RUN_TEST 0.095 · 9 rule_SEARCH 0.077 · 10 rule_SHOW_FILE 0.066 …
하위: last_action_failed 0.009(단독은 약함, last_action에 조건부), user_tier/language_pref
0.0005, cnt_respond_only 0.0. 전체 표는 `logs/eda_report.md` §12.
→ 시퀀셜(last/second_last action) + WRAP_UP 규칙 + turn/history 길이 + n_open_files가 정형
피처의 핵심. 약피처(budget/tier/lang_pref)는 drop 권장.

### 세션 그룹핑 / CV 전략 (중요 — 기존 CV 방식 변경 필요)
- id = `sess_sim_<date>_<NNNNNN>-step_XX`. `-step_XX` 제거 시 **distinct 세션 9,429개**.
  세션당 스텝 min 1 / median 7 / mean 7.42 / max 18. **97.7% 세션이 2스텝 이상,
  전체 행의 99.69%가 멀티스텝 세션 소속.**
- 같은 세션의 여러 스냅샷은 workspace/language_mix/budget/open_files 공유 + history prefix 중복.
  → **StratifiedKFold는 세션 컨텍스트를 fold 간 누수시켜 CV를 낙관 편향시킴.**
- **권고: GroupKFold(group = 세션 id) 사용** (클래스 균형까지 원하면 StratifiedGroupKFold).
  CLAUDE.md 기본값 "5-fold stratified"를 이 대회에서는 group 기준으로 대체할 것.

### 리키지 프로브
- last_action == label 13.89%(히스토리 보유 행 기준), label이 history 어디든 등장 29.6% —
  둘 다 세션 내 액션 반복의 자연 결과(전이행렬 대각선), **주입형 리키지 아님.** 리키지 없음.

### 추론 예산
- current_prompt 평균 12.8 whitespace 토큰(짧음), 직렬화 history 평균 103.7(p95 201).
  30,000 test: prompt만 ≈0.38M, prompt+history ≈3.49M 토큰. TF-IDF/LightGBM은 여유.
  history는 토큰 대부분 차지 → 인코더 사용 시 마지막 K턴만 truncate.

### test.jsonl 주의
5행 포맷 샘플. 스키마 동일, 값 범위 정상. 5개 세션 id가 train에도 존재하나 이는 샘플이
train 세션에서 추출된 **포맷 샘플 아티팩트**일 뿐(실제 30k 히든 test는 서버 마운트). 튜닝 금지.

> 그림 디렉토리: `data_docs/eda_figures/` (01~14). 상세: `logs/eda_report.md`.
