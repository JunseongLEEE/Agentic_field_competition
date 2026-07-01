# Error Analysis — exp_001_tfidf_lightgbm (OOF)

_Generated 2026-07-01. Basis: out-of-fold predictions from StratifiedGroupKFold(3), shape (70000,14). OOF is the honest basis for error analysis; predicting train with the full-train model would be optimistic._

**OOF Macro-F1 = 0.6606** (matches logged CV mean 0.6605). LB 0.6753.

**Top-1 acc = 0.6726 | Top-2 acc = 0.8370 | Top-3 acc = 0.9207**


## 1. Per-class precision / recall / F1 (sorted by F1)

| class | precision | recall | f1 | support |
|---|---|---|---|---|
| list_directory | 0.348 | 0.416 | 0.379 | 4329 |
| read_file | 0.497 | 0.494 | 0.495 | 9257 |
| web_search | 0.594 | 0.425 | 0.495 | 1273 |
| lint_or_typecheck | 0.548 | 0.467 | 0.504 | 2283 |
| grep_search | 0.562 | 0.542 | 0.552 | 9912 |
| glob_pattern | 0.586 | 0.532 | 0.558 | 5284 |
| ask_user | 0.561 | 0.560 | 0.561 | 2701 |
| plan_task | 0.588 | 0.579 | 0.583 | 2679 |
| run_tests | 0.658 | 0.716 | 0.686 | 4561 |
| run_bash | 0.721 | 0.717 | 0.719 | 5068 |
| apply_patch | 0.803 | 0.886 | 0.842 | 4823 |
| edit_file | 0.908 | 0.904 | 0.906 | 11171 |
| write_file | 0.972 | 0.971 | 0.972 | 1481 |
| respond_only | 0.999 | 0.995 | 0.997 | 5178 |


Worst 3 classes: **list_directory, read_file, web_search**. Best classes (near-ceiling): respond_only, write_file, edit_file, apply_patch.


## 2. Top confusions (true -> pred)

**By count:**

| true | pred | count | rate_of_true |
|---|---|---|---|
| grep_search | read_file | 2200 | 0.222 |
| read_file | grep_search | 2150 | 0.232 |
| read_file | list_directory | 1397 | 0.151 |
| list_directory | read_file | 1173 | 0.271 |
| grep_search | list_directory | 1121 | 0.113 |
| glob_pattern | read_file | 858 | 0.162 |
| glob_pattern | grep_search | 786 | 0.149 |
| list_directory | grep_search | 781 | 0.180 |
| plan_task | ask_user | 740 | 0.276 |
| ask_user | plan_task | 713 | 0.264 |
| run_bash | run_tests | 672 | 0.133 |
| read_file | glob_pattern | 669 | 0.072 |
| lint_or_typecheck | run_tests | 656 | 0.287 |
| grep_search | glob_pattern | 632 | 0.064 |
| run_tests | run_bash | 618 | 0.135 |


**By rate (fraction of that true class sent to pred):**

| true | pred | count | rate_of_true |
|---|---|---|---|
| lint_or_typecheck | run_tests | 656 | 0.287 |
| plan_task | ask_user | 740 | 0.276 |
| list_directory | read_file | 1173 | 0.271 |
| ask_user | plan_task | 713 | 0.264 |
| web_search | ask_user | 326 | 0.256 |
| read_file | grep_search | 2150 | 0.232 |
| grep_search | read_file | 2200 | 0.222 |
| web_search | plan_task | 250 | 0.196 |
| list_directory | grep_search | 781 | 0.180 |
| lint_or_typecheck | run_bash | 388 | 0.170 |
| glob_pattern | read_file | 858 | 0.162 |
| read_file | list_directory | 1397 | 0.151 |
| glob_pattern | grep_search | 786 | 0.149 |
| run_tests | run_bash | 618 | 0.135 |
| run_bash | run_tests | 672 | 0.133 |


### Three block-diagonal confusion clusters (see err_confusion.png)
The confusion matrix is near block-diagonal with **three** mutually-confused clusters — almost all off-diagonal mass is intra-cluster:
1. **Read/search:** read_file <-> grep_search <-> list_directory <-> glob_pattern (largest error source).
2. **Execution:** run_bash <-> run_tests <-> lint_or_typecheck (e.g. lint_or_typecheck->run_tests 0.29, run_bash->run_tests 0.13).
3. **Dialogue/planning:** ask_user <-> plan_task <-> web_search (e.g. plan_task->ask_user 0.28, web_search->ask_user 0.26).

edit_file/write_file/apply_patch and respond_only are essentially solved. This block structure is the key strategic insight: a **hierarchical / 2-stage approach** (predict cluster, then disambiguate within cluster) or cluster-specific features are the natural fix.

### Read/search cluster (read_file, grep_search, list_directory, glob_pattern)

- Of all errors on cluster-true samples, **89.2%** are predicted into *another cluster member* (mutual confusion within the show/find-file cluster).

- Within-cluster errors (true and pred both in cluster) = **12691** = **55.4% of ALL errors**.

- This strongly supports the hypothesis: the generator maps similar 'show/find file' prompts to a *distribution* over these 4 actions, so they are intrinsically entangled.


## 3. Error rate by feature slice

See figures err_by_turn.png, err_by_lastaction.png, err_by_openfiles.png, err_by_langpref.png, err_by_historylen.png, err_by_promptchars.png, err_by_rulecat.png.


**Rule-category error rates:**

| rule | err | cnt |
|---|---|---|
| SEARCH | 0.482 | 8426 |
| WEB_REF | 0.475 | 322 |
| SHOW_FILE | 0.444 | 5334 |
| PLAN_REQ | 0.395 | 1589 |
| LINT_CHECK | 0.339 | 634 |
| n_rules_0 | 0.317 | 46375 |
| ERROR_HELP | 0.313 | 785 |
| RUN_TEST | 0.275 | 3830 |
| WRAP_UP | 0.000 | 3814 |


**Error rate by last_action (top by count):**

| last_action | err | cnt |
|---|---|---|
| list_directory | 0.479 | 4223 |
| NONE | 0.478 | 9000 |
| plan_task | 0.441 | 2584 |
| ask_user | 0.387 | 2192 |
| glob_pattern | 0.349 | 4967 |
| web_search | 0.316 | 1188 |
| apply_patch | 0.305 | 4417 |
| run_bash | 0.302 | 4797 |
| lint_or_typecheck | 0.296 | 2016 |
| grep_search | 0.284 | 9412 |
| edit_file | 0.278 | 10620 |
| read_file | 0.248 | 8887 |
| run_tests | 0.225 | 4251 |
| write_file | 0.216 | 1446 |


**Error rate by turn_index:**

| turn | err | cnt |
|---|---|---|
| 1.0 | 0.478 | 9000 |
| 2.0 | 0.394 | 8797 |
| 3.0 | 0.313 | 8446 |
| 4.0 | 0.301 | 8001 |
| 5.0 | 0.273 | 7526 |
| 6.0 | 0.264 | 6644 |
| 7.0 | 0.256 | 5435 |
| 8.0 | 0.301 | 4222 |
| 9.0 | 0.301 | 3240 |
| 10.0 | 0.301 | 2488 |
| 11.0 | 0.308 | 1902 |
| 12.0 | 0.307 | 1420 |
| 13.0 | 0.297 | 1035 |
| 14.0 | 0.318 | 733 |
| 15.0 | 0.309 | 499 |
| 16.0 | 0.322 | 335 |
| 17.0 | 0.259 | 193 |
| 18.0 | 0.298 | 84 |

## 4. Confidence & top-k analysis

- Mean top1-top2 margin: correct=0.718, incorrect=0.377 (errors are near-ties).

- Among **errors**, true class is in top-2 for **50.2%**, in top-3 for **75.8%**.

- Top-2 acc (0.837) and top-3 acc (0.921) are far above top-1 (0.673): the model **ranks the right answer high** but argmax loses on ambiguous multi-way classes. Ranking is good; the loss is in tie-breaking.

- See err_confidence.png: errors concentrate at LOWER confidence (ambiguity/near-ties), not high-confidence systematic failures.


## 5. Irreducible-ambiguity ceiling

> Caveat: 94.1% of prompts are unique strings and only 15% of samples share a prompt with any other sample. So the "exact-prompt Bayes ceiling" of 0.9624 is essentially a **memorization ceiling** (overfit to exact strings), NOT a realistic generalization bound — ignore it as an upper bound. The coarse (rule+last_action) ceiling of 0.36 is the opposite extreme: too weak a signature, showing that almost all class signal lives in the prompt text, not the meta/rule flags. The honest read is between them: the near-tie / top-k evidence below is the best ceiling proxy.


- Exact-prompt grouping: **85.0%** of samples have a unique prompt string; the rest share prompts.

- Groups whose identical prompt maps to >1 true label (generator stochasticity) contain **10.6% of all errors**.

- Bayes-optimal top-1 accuracy if we could perfectly predict the majority label per *exact prompt* = **0.9624** (current top-1 = 0.6726).

- Using coarser signature (rule-flags tuple + last_action): majority-vote ceiling = **0.3600**.

- **Honest interpretation:** Only ~10.6% of errors are truly irreducible (identical prompt -> multiple labels). The other ~89% are between *similar-but-distinct* prompts that DO carry distinguishing tokens (glob metachars, file extensions, "where/list/show"), i.e. **learnable**. Combined with top-2 acc 0.837 (the right answer is usually rank-2), the realistic Macro-F1 headroom is meaningful — much of the read/search confusion is a feature/tie-break problem, not a hard generator cap. Do NOT treat 0.67 as near-ceiling.


## 6. Qualitative misclassifications (worst 3 classes)


### list_directory (F1=0.379)

- conf=0.99 true=`list_directory` pred=`run_tests` :: "그나저나 manage에 useStore로 theme 잘 끌어왔는지 그 파일만 다시 확인"
- conf=0.99 true=`list_directory` pred=`glob_pattern` :: "한 가지 — internal/parser/parser.go에 abort-controller 폴리필 deps 들어가 있는지 확인 좀"
- conf=0.99 true=`list_directory` pred=`read_file` :: "프로젝트에 인증 관련 파일이 어디어디 흩어져 있는지 모르겠어요. composables 폴더 안에 뭐가 있는지부터 쭉 보여줘요"
- conf=0.99 true=`list_directory` pred=`read_file` :: "logger first. open it"
- conf=0.98 true=`list_directory` pred=`glob_pattern` :: "좋아 그럼 토큰 검증 로직부터. validateToken 같은 거 어디 있나 훑어줘"


### read_file (F1=0.495)

- conf=0.99 true=`read_file` pred=`glob_pattern` :: "actually batch 64, seq 1024, that's a 64x1024x1024 score tensor. show me how attention builds the scores"
- conf=0.99 true=`read_file` pred=`glob_pattern` :: "ok so it's in staging already, just never promoted to the mart. open the users staging model plz"
- conf=0.99 true=`read_file` pred=`glob_pattern` :: "hey did i miss any other callers? grep again"
- conf=0.99 true=`read_file` pred=`glob_pattern` :: "find the spots where Trainer reaches into main territory soon"
- conf=0.99 true=`read_file` pred=`glob_pattern` :: "로그인 자꾸 brute force 시도가 들어와서 auth 라우트에 rate limit 좀 걸어야 할 것 같은데, 일단 지금 그 라우트 어떻게 생겼는지 보자 천천히요"


### web_search (F1=0.495)

- conf=1.00 true=`web_search` pred=`plan_task` :: "그래서 PositionalEncoding 이게 absolute 방식이네. rope를 새로 넣으려면 보통 다들 어디다 끼우나... 표준적으로 어떻게들 하는지 좀 찾아봐줄래?"
- conf=1.00 true=`web_search` pred=`ask_user` :: "AttributeError 계속 뜨는데 어떻게 해야 할지 모르겠어, 도와줄래?"
- conf=1.00 true=`web_search` pred=`ask_user` :: "오케이 Timeout 계속 뜨는데 어떻게 해야 할지 모르겠어, 도와줄래?"
- conf=1.00 true=`web_search` pred=`ask_user` :: "데이터베이스 연결을 좀 리팩토링 해보고 싶은데 어디부터 손대야 할지 모르겠어요. 단계별로 계획 좀 세워줄래요?"
- conf=1.00 true=`web_search` pred=`plan_task` :: "when you're free, want to add a reusable loading spinner and use it on App while data loads. break it down for me before we touch code"


## 7. How to beat the baseline (prioritized)


1. **Second-stage / cost-sensitive model for the read/search cluster (highest leverage).** ~55% of all errors are mutual confusion among read_file/grep_search/list_directory/glob_pattern. Add discriminative features: prompt contains a glob metachar (`*`, `?`, `**`, extension pattern) -> glob_pattern; explicit path/filename with extension -> read_file; "where/which file/across the repo/pattern in content" -> grep_search; "list/what's in dir/tree" -> list_directory. Consider a dedicated binary/4-way classifier gated on cluster-predicted samples, or per-class decision thresholds tuned on OOF.
2. **Threshold / probability calibration tuned to Macro-F1.** Top-2 acc (0.837) >> top-1 (0.673) and errors are near-ties. Optimize per-class thresholds (or a small logit bias vector) on OOF to maximize Macro-F1 — cheap, no retrain of features, directly targets the metric. Low-recall classes (list_directory, web_search, read_file) benefit most.
3. **Add char n-grams / Korean-aware text features.** Prompts are Korean+English mixed. Current char_wb (2,4) helps but try (2,5)/(1,5), add a Korean subword or jamo channel, and TF-IDF sublinear_tf; boosts the confused text-driven classes (ask_user, plan_task, web_search, lint_or_typecheck).
4. **History action n-grams.** Currently only last/second-last + counts. Add ordered bigram/trigram of recent actions (e.g. edit->run_tests->lint) as categorical/hashed features — helps run_tests/lint_or_typecheck/run_bash which depend on workflow position (see err_by_lastaction / err_by_turn).
5. **Class weights / focal-style reweighting for minority-recall classes.** list_directory & web_search have low recall; mild class_weight upweighting or is_unbalance tuning can trade a little precision for recall and raise Macro-F1 (validate on OOF, watch for collapse).
6. **Don't over-invest where generator-capped.** ~11% of errors sit in identical-prompt groups mapping to multiple labels (irreducible). The read/search cluster is partly capped; target the *learnable* slice (distinct disambiguating tokens) rather than chasing every cluster error.
