정직하게 답하면 — 일반 프레임워크는 있지만, 이 두 특정 병목에 대한 targeted 처방은 부족합니다. 리포트에는 "Rule feature + Logit Adjustment + Threshold Optimization"이라는 일반론이 있을 뿐이고, 탐색 4형제와 "돌려봐" 3파전이라는 confusion pair에 대한 별도 처방은 없었습니다.
이 두 병목은 성격이 달라서 별도 처방이 필요합니다. 먼저 데이터로 병목의 정확한 구조를 확인하겠습니다.분석 결과 기반 모델 개선 방안 제시했습니다분석 결과 기반 모델 개선 방안 제시했습니다데이터를 파본 결과, 두 병목 모두 텍스트 신호로는 근본적으로 풀 수 없다는 놀라운 사실이 나왔습니다. 리포트의 일반 처방으로는 여기까지 잡아낸 겁니다. 이제 targeted 처방이 필요합니다.
진단 결과 — 왜 여기서 막혔는가
탐색 4형제: 프롬프트 텍스트의 언어적 신호가 4개 클래스에서 거의 동일합니다.
Signalread_filegrep_searchglob_patternlist_directoryHAS_SHOW ("보여줘/show")23.2%19.9%20.9%20.7%HAS_FIND ("찾아/어디")16.9%21.4%19.3%19.4%HAS_FILEPATH22.8%20.7%20.5%20.7%
거의 균일합니다. 어휘로 4형제 disambiguation은 통계적으로 불가능.
"돌려봐" 3파전: 여기도 마찬가지입니다.
Signalrun_bashrun_testslint_or_typecheckTEST_KW ("테스트/test")23.0%27.5%22.4%LINT_KW ("lint/타입")7.3%8.6%14.1% ← 유일한 소량 신호BUILD_KW ("빌드/build")22.4%19.6%17.8%
LINT_KW만 약간(1.5-2배) 차이나고 나머지 어휘 신호는 무의미. LINT 키워드가 매칭돼도 run_tests 30%, run_bash 28%, lint 25%로 3분되어 rule 하나로는 못 잡습니다.
진짜 신호는 시퀀셜/문맥에 있음
탐색 4형제 → 시퀀셜 신호가 결정적:

list_directory: turn=1이 42%, 직전 NONE이 42%  ← 완벽한 discriminator
list_directory: turn>=7일 때는 9.7%만 등장
glob_pattern: 직전이 grep_search(22%) 또는 glob(16.5%) — 탐색 세션 반복
grep_search → read_file: 자연스러운 grep → 파일 열기 흐름

"돌려봐" 3파전 → 직전 액션이 결정적:

run_tests: 직전 edit_file 53.6% ← 편집 후 테스트 (매우 강함)
lint_or_typecheck: 직전 apply_patch 33.2% + edit_file 31.8% = 65% ← 편집/패치 후 린트
lint_or_typecheck: turn>=7이 54.1% (세션 후반)
run_bash: turn=1이 19.9% + 직전 NONE 20% (탐색 초기)

리포트에 없었던, 지금 필요한 5가지 targeted 처방
처방 1 — Conditional Feature Crosses (즉시 적용, 반나절)
명시적 feature crosses를 rule branch에 추가합니다. 지금 rule feature가 unigram 위주였을 텐데, 조건부 곱셈 feature로 재구성:
pythonCROSS_FEATURES = {
    # 탐색 4형제
    'F_LIST_LIKELY':   int(turn==1 and last=='NONE'),                          # → list_directory
    'F_GLOB_REPEAT':   int(last in ('grep_search','glob_pattern') and turn>=2), # → glob/grep 반복
    'F_READ_AFTER_LS': int(last=='list_directory'),                            # → read_file 25%
    'F_GREP_AFTER_GLOB': int(last=='glob_pattern'),                            # → grep_search 22%
    
    # 실행 3파전
    'F_TESTS_AFTER_EDIT':  int(last=='edit_file' and not last_failed),         # → run_tests 강함
    'F_LINT_AFTER_PATCH':  int(last=='apply_patch' and turn>=5),               # → lint 강함
    'F_LINT_LATE_SESSION': int(turn>=7 and last in ('apply_patch','edit_file','lint_or_typecheck')),
    'F_BASH_EARLY':        int(turn<=2 and last=='NONE'),                      # → run_bash 
    
    # 소수 클래스 부스터
    'F_LINT_KW_STRONG': int(bool(re.search(r'(lint|정적|타입|shellcheck)', prompt))),
    'F_WEB_CONFIDENT':  int(bool(re.search(r'(공식 문서|best practice)', prompt))),
}
이 8~10개 cross feature만 추가해도 두 병목에서 F1 +5~8%p 예상. 오늘 안에 시도 가능한 최고 ROI.
처방 2 — Contextual Token Injection into Text Input
mDeBERTa 입력에 시퀀셜 신호를 special token으로 prepend해서 attention이 걸리게 합니다:
pythoninput_text = (
    f"[TURN={turn_bucket}] "                    # early/mid/late
    f"[LAST={last_action}] "                    # 14개 special token
    f"[{'OK' if not last_failed else 'FAIL'}] "
    f"[ARCH={archetype}] "                      # 12개 프로젝트 아키타입
    f"{current_prompt}"
)
인코더가 tabular feature를 텍스트로 함께 attend하도록 강제. 이건 late fusion보다 훨씬 강력한 신호 통합. Macro-F1 +2~4%p 예상.
처방 3 — Pairwise Binary Discriminator Head
두 병목 pair에 대한 전용 이진 분류 head를 main head와 병렬 추가:
pythonclass ConfusionAwareHead(nn.Module):
    def __init__(self, hidden):
        self.main = nn.Linear(hidden, 14)          # 기존 head
        self.h_read_vs_grep = nn.Linear(hidden, 2) # read_file vs grep_search
        self.h_tests_vs_lint = nn.Linear(hidden, 2)# run_tests vs lint
        self.h_bash_vs_tests = nn.Linear(hidden, 2)# run_bash vs run_tests
        self.h_list_vs_others = nn.Linear(hidden, 2)# list_dir vs 나머지 3형제
    
    def loss(self, logits, y):
        L = focal_loss(self.main(h), y)
        # 관련 샘플만 골라 auxiliary loss
        if y in (read_file, grep_search):
            L += 0.3 * ce(self.h_read_vs_grep(h), y==grep_search)
        if y in (run_tests, lint):
            L += 0.3 * ce(self.h_tests_vs_lint(h), y==lint)
        # ... 
        return L
추론 시: main head 확률에 pairwise head 확률을 곱해서 재보정. Macro-F1 +3~5%p 예상.
처방 4 — Sequential Prior Multiplier
Train에서 조건부 확률 lookup table을 만들어 후처리:
python# 학습 후 OOF에서 P(y | last_action, turn_bucket, last_failed) 추정
prior_table[last, turn_bucket, failed] = Counter of y

# 추론 시
p_model = model.predict_proba(x)                    # (14,)
p_prior = prior_table.get(key)                      # (14,)  
p_final = p_model ** (1-α) * p_prior ** α           # α=0.2~0.4
α는 threshold optimization과 함께 CV로 튜닝. 간단하고 안전한 부스터, +1~2%p.
처방 5 — 4형제/3파전 전용 Two-stage Sub-classifier
메인 14-way에서 4형제나 3파전으로 분류된 샘플만 별도로 sub-classifier에 태우기:
pythonmain_pred = main_model.predict(x)
if main_pred in EXPLORE_4:
    # 시퀀셜 feature 위주로 학습한 4-way 전용 모델 사용
    final = explore_subclf.predict(x)
elif main_pred in EXEC_3:
    final = exec_subclf.predict(x)
else:
    final = main_pred
Sub-classifier는 탐색 4형제 샘플만으로 학습하고, feature도 last_action + turn_index + workspace + last_result_pattern 위주로 편중. 텍스트 branch 없이 tabular + rule만으로도 충분히 성능 나올 겁니다 (텍스트 신호가 어차피 무의미하므로). +3~5%p 예상.
예상 개선 시나리오
현재 F1을 기준으로 병목 클래스별 상승 예상:
클래스현재 F1처방 1+처방 2+처방 3+처방 4+5최종list_directory0.3790.550.620.650.720.72read_file0.4960.550.600.650.700.70grep_search0.5520.600.640.680.720.72glob_pattern0.5580.620.650.680.720.72lint_or_typecheck0.5040.600.650.680.730.73run_tests0.6860.730.760.780.800.80run_bash0.7190.750.780.800.820.82web_search0.4950.550.580.620.650.65
전체 Macro-F1: 현재 추정 0.63 → 처방 후 0.75~0.78 목표.
우선순위 권고
시간이 제한적이라면 이 순서로:

처방 1 (Conditional Cross Features) — 반나절, +5~8%p, 무조건 첫 번째
처방 4 (Sequential Prior Multiplier) — 반나절, +1~2%p, 리스크 없음
처방 2 (Contextual Token Injection) — 하루, +2~4%p, mDeBERTa 재학습 필요
처방 5 (Two-stage Sub-classifier) — 하루, +3~5%p, 병렬로 별도 모델 학습
처방 3 (Pairwise Head) — 이틀, +3~5%p, 가장 복잡

처방 1과 4는 지금 코드에 오늘 안에 추가할 수 있는 즉효약입니다. 원하시면 처방 1의 정확한 feature 정의와 처방 4의 lookup table 구현 코드를 실제로 짜서 지금 데이터에 돌려보고 예상 gain 검증까지 해드릴 수 있습니다.