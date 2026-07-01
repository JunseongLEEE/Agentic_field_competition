"""Feature engineering v2 — PCA-filtered + new engineered features.

PCA+MI analysis (2026-07-01) identified 25 new features that outperform the
original feature set.  Seven original features were dropped (MI≈0 and low PCA
contribution): cnt_respond_only, user_tier, prompt_has_exclaim, prompt_has_code,
prompt_is_korean, n_actions (duplicate of history_len), prompt_len_chars
(r=0.93 with prompt_len_words).

Feature groups
--------------
1. TF-IDF on current_prompt: word (1,2).
2. Sequential: last/second_last/third_last action, bigram, trigram,
   last_action_status, action counts, consec_same, unique_actions,
   phase ratios, read-edit cycle.
3. Rule categories: 8 regex one-hot + prompt_intent (combined).
4. Session context: turn_index, session_phase, steps_remaining,
   turn_action, result_cat, recent_fails.
5. Workspace: primary_lang, workspace_archetype, open_ext, git_dirty,
   last_ci_status, loc, n_open_files.
6. Prompt shape: prompt_len_words, prompt_has_question, prompt_n_sentences,
   has_path_ref, avg_user_prompt_len.
"""

import json
import re
from collections import Counter

import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_ORDER = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]

NONE_TOKEN = "NONE"
UNK_TOKEN = "__UNK__"

RULE_CATEGORIES = [
    ("WRAP_UP", re.compile(
        r"(마무리|여기까지|이 정도면).*(요약|정리)|(wrap.?up|recap|summariz)", re.I)),
    ("ERROR_HELP", re.compile(
        r"(TypeError|AttributeError|ConnectionError|KeyError|AssertionError|Timeout|"
        r"I keep hitting|계속 뜨는데)", re.I)),
    ("PLAN_REQ", re.compile(
        r"(단계.*(잡|짜|세워)|계획.*(잡|짜|세워)|lay.*out|before i (start|edit|touch)|"
        r"plan (this|it|out))", re.I)),
    ("SHOW_FILE", re.compile(
        r"(보여줘|열어봐|열어줘|show me|open the|look at|pull up)", re.I)),
    ("SEARCH", re.compile(
        r"(어디|찾아|어느 파일|list what|where.*(is|are|does)|find|search for|grep)", re.I)),
    ("RUN_TEST", re.compile(
        r"(테스트.*돌|한번 돌려|돌려봐|run.*test|rerun|full suite|다시 빌드|build again)", re.I)),
    ("LINT_CHECK", re.compile(
        r"(lint|typecheck|타입체크|shellcheck|mypy|ruff|flake8)", re.I)),
    ("WEB_REF", re.compile(
        r"(best practice|공식.*문서|documentation|docs\b|look.*up online|web search)", re.I)),
]
RULE_NAMES = [name for name, _ in RULE_CATEGORIES]

EXPLORE_ACTIONS = frozenset({"read_file", "grep_search", "list_directory", "glob_pattern"})
MODIFY_ACTIONS = frozenset({"edit_file", "write_file", "apply_patch"})
VERIFY_ACTIONS = frozenset({"run_tests", "run_bash", "lint_or_typecheck"})

# Categorical fields that get integer-coded.
CAT_FIELDS = [
    "last_action", "second_last_action", "third_last_action",
    "action_bigram", "action_trigram",
    "last_action_status", "turn_action",
    "session_phase", "result_cat", "prompt_intent",
    "primary_lang", "workspace_archetype", "open_ext",
    "git_dirty", "last_ci_status",
    "prompt_len_bucket",
]

# Numeric fields (kept raw).
NUM_FIELDS = [
    "last_action_failed", "history_len", "turn_index",
    "elapsed_session_sec", "loc", "n_open_files",
    "unique_actions", "consec_same",
    "explore_ratio", "modify_ratio", "verify_ratio", "meta_ratio",
    "most_common_action_ratio",
    "recent_fails", "last_is_explore", "last_is_modify", "last_is_verify",
    "has_read_edit_cycle",
    "steps_remaining", "last_n_files",
    "prompt_len_words", "prompt_has_question", "prompt_n_sentences",
    "has_path_ref", "avg_user_prompt_len",
    "last_result_lines", "last_result_matches",
]


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(x):
    if x is None:
        return ""
    return str(x) if not isinstance(x, str) else x


def get_prompt(sample):
    return _safe_str(sample.get("current_prompt", ""))


def _action_list(history):
    if not isinstance(history, list):
        return []
    return [h for h in history if isinstance(h, dict) and h.get("role") == "assistant_action"]


# ---------------------------------------------------------------------------
# Per-sample feature extraction
# ---------------------------------------------------------------------------

def extract_record(sample):
    prompt = get_prompt(sample)
    history = sample.get("history", []) or []
    actions = _action_list(history)
    action_names = [a.get("name", NONE_TOKEN) for a in actions]
    user_turns = [h for h in history if isinstance(h, dict) and h.get("role") == "user"]

    last = action_names[-1] if len(action_names) >= 1 else NONE_TOKEN
    second = action_names[-2] if len(action_names) >= 2 else NONE_TOKEN
    third = action_names[-3] if len(action_names) >= 3 else NONE_TOKEN

    # Last action result
    last_failed = 0
    last_rs = ""
    if actions:
        last_rs = _safe_str(actions[-1].get("result_summary", ""))
        if "ERROR" in last_rs.upper() or "FAIL" in last_rs.upper():
            last_failed = 1

    # Per-action counts (drop cnt_respond_only — always 0)
    counts = {}
    for c in CLASS_ORDER:
        counts["cnt_" + c] = 0
    for a in actions:
        key = "cnt_" + a.get("name", "")
        if key in counts:
            counts[key] += 1

    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    lang_mix = ws.get("language_mix", {}) or {}
    turn_index = int(meta.get("turn_index", 0) or 0)

    if isinstance(lang_mix, dict) and lang_mix:
        primary_lang = max(lang_mix.items(), key=lambda kv: kv[1])[0]
        sorted_langs = sorted(lang_mix.items(), key=lambda kv: -kv[1])
        archetype = "+".join(k for k, _ in sorted_langs)
    else:
        primary_lang = NONE_TOKEN
        archetype = NONE_TOKEN

    open_files = ws.get("open_files", []) or []
    n_open = len(open_files) if isinstance(open_files, list) else 0

    open_ext = NONE_TOKEN
    if isinstance(open_files, list):
        for fp in open_files:
            fp_s = str(fp)
            if "." in fp_s:
                open_ext = fp_s.rsplit(".", 1)[-1].lower()
                break

    # Rule flags + intent
    rflags = [1 if rgx.search(prompt) else 0 for _, rgx in RULE_CATEGORIES]
    active_rules = [RULE_NAMES[i] for i, v in enumerate(rflags) if v]
    intent = active_rules[0] if active_rules else "GENERAL"

    # Session phase
    if turn_index <= 2:
        phase = "early"
    elif turn_index <= 4:
        phase = "mid"
    elif turn_index <= 6:
        phase = "late"
    else:
        phase = "wrapup"

    # Result category
    if not last_rs:
        result_cat = "NONE"
    elif "FAIL" in last_rs.upper() or "ERROR" in last_rs.upper():
        result_cat = "FAIL"
    elif "ok" in last_rs.lower() or "PASS" in last_rs:
        result_cat = "OK"
    else:
        result_cat = "OTHER"

    # Consecutive same action
    consec = 0
    for a in reversed(action_names):
        if a == last:
            consec += 1
        else:
            break

    # Phase group ratios
    n_act = max(len(action_names), 1)
    explore_r = sum(1 for a in action_names if a in EXPLORE_ACTIONS) / n_act
    modify_r = sum(1 for a in action_names if a in MODIFY_ACTIONS) / n_act
    verify_r = sum(1 for a in action_names if a in VERIFY_ACTIONS) / n_act
    meta_r = 1.0 - explore_r - modify_r - verify_r

    # Recent failures
    recent_fails = 0
    for a in actions[-3:]:
        rs = _safe_str(a.get("result_summary", ""))
        if "ERROR" in rs.upper() or "FAIL" in rs.upper():
            recent_fails += 1

    # Read-edit cycle
    has_cycle = 0
    for i in range(len(action_names) - 1):
        if (action_names[i] in ("read_file", "grep_search") and
                action_names[i + 1] in ("edit_file", "apply_patch")):
            has_cycle = 1
            break

    # Most common action ratio
    if action_names:
        mcr = max(Counter(action_names).values()) / len(action_names)
    else:
        mcr = 0.0

    # Last action n_files from args
    last_nf = 0
    if actions:
        args = actions[-1].get("args", {}) or {}
        if isinstance(args, dict):
            last_nf = int(args.get("n_files", 0) or 0)

    # Result summary numbers
    last_lines, last_matches = 0, 0
    if actions:
        nums = re.findall(r"\d+", last_rs)
        if nums:
            last_lines = int(nums[0])
            if len(nums) > 1:
                last_matches = int(nums[1])

    prompt_words = len(prompt.split())
    prompt_bucket = "short" if prompt_words <= 8 else ("medium" if prompt_words <= 16 else "long")
    has_path = 1 if re.search(
        r"[/\\]\w+\.\w+|\.py\b|\.ts\b|\.js\b|\.rs\b|\.java\b|\.go\b|\.vue\b", prompt) else 0

    out = {
        # Categoricals
        "last_action": _safe_str(last) or NONE_TOKEN,
        "second_last_action": _safe_str(second) or NONE_TOKEN,
        "third_last_action": _safe_str(third) or NONE_TOKEN,
        "action_bigram": f"{second}__{last}",
        "action_trigram": f"{third}__{second}__{last}",
        "last_action_status": f"{last}__{'FAIL' if last_failed else 'OK'}",
        "turn_action": f"t{min(turn_index, 7)}_{last}",
        "session_phase": phase,
        "result_cat": result_cat,
        "prompt_intent": intent,
        "primary_lang": _safe_str(primary_lang) or NONE_TOKEN,
        "workspace_archetype": archetype,
        "open_ext": open_ext,
        "git_dirty": _safe_str(ws.get("git_dirty", NONE_TOKEN)),
        "last_ci_status": _safe_str(ws.get("last_ci_status", NONE_TOKEN)) or NONE_TOKEN,
        "prompt_len_bucket": prompt_bucket,
        # Numerics
        "last_action_failed": last_failed,
        "history_len": len(history) if isinstance(history, list) else 0,
        "turn_index": turn_index,
        "elapsed_session_sec": float(meta.get("elapsed_session_sec", 0) or 0),
        "loc": float(ws.get("loc", 0) or 0),
        "n_open_files": n_open,
        "unique_actions": len(set(action_names)),
        "consec_same": consec,
        "explore_ratio": explore_r,
        "modify_ratio": modify_r,
        "verify_ratio": verify_r,
        "meta_ratio": meta_r,
        "most_common_action_ratio": mcr,
        "recent_fails": recent_fails,
        "last_is_explore": 1 if last in EXPLORE_ACTIONS else 0,
        "last_is_modify": 1 if last in MODIFY_ACTIONS else 0,
        "last_is_verify": 1 if last in VERIFY_ACTIONS else 0,
        "has_read_edit_cycle": has_cycle,
        "steps_remaining": max(0, 7 - turn_index),
        "last_n_files": last_nf,
        "prompt_len_words": prompt_words,
        "prompt_has_question": 1 if "?" in prompt else 0,
        "prompt_n_sentences": len(re.split(r"[.!?]+", prompt)),
        "has_path_ref": has_path,
        "avg_user_prompt_len": (
            float(np.mean([len(h.get("content", "")) for h in user_turns]))
            if user_turns else 0.0
        ),
        "last_result_lines": last_lines,
        "last_result_matches": last_matches,
    }
    out.update(counts)
    out["_rule_flags"] = rflags
    return out


def rule_flags(prompt):
    return [1 if rgx.search(prompt) else 0 for _, rgx in RULE_CATEGORIES]


# ---------------------------------------------------------------------------
# Categorical mapping
# ---------------------------------------------------------------------------

def build_cat_mappings(records):
    mappings = {}
    for field in CAT_FIELDS:
        vals = sorted({str(r[field]) for r in records})
        m = {UNK_TOKEN: 0}
        for v in vals:
            if v not in m:
                m[v] = len(m)
        mappings[field] = m
    return mappings


def _encode_cat(value, mapping):
    return mapping.get(str(value), mapping.get(UNK_TOKEN, 0))


# ---------------------------------------------------------------------------
# Dense matrix
# ---------------------------------------------------------------------------

def records_to_dense(records, cat_mappings):
    rows = []
    for r in records:
        row = []
        for field in CAT_FIELDS:
            row.append(_encode_cat(r[field], cat_mappings[field]))
        for field in NUM_FIELDS:
            row.append(float(r.get(field, 0) or 0))
        for c in CLASS_ORDER:
            row.append(r.get("cnt_" + c, 0))
        row.extend(r["_rule_flags"])
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def dense_feature_names():
    names = list(CAT_FIELDS)
    names += list(NUM_FIELDS)
    names += ["cnt_" + c for c in CLASS_ORDER]
    names += ["rule_" + n for n in RULE_NAMES]
    return names


# ---------------------------------------------------------------------------
# Top-level builders
# ---------------------------------------------------------------------------

def build_records(samples):
    ids, prompts, records = [], [], []
    for s in samples:
        ids.append(_safe_str(s.get("id", "")))
        prompts.append(get_prompt(s))
        records.append(extract_record(s))
    return ids, prompts, records


def transform_all(prompts, records, artifacts):
    """Legacy word-only transform (kept for backwards compatibility)."""
    word_vec = artifacts["word_vec"]
    cat_mappings = artifacts["cat_mappings"]

    Xw = word_vec.transform(prompts)
    Xd = records_to_dense(records, cat_mappings)
    Xd_sparse = sparse.csr_matrix(Xd)
    return sparse.hstack([Xw, Xd_sparse], format="csr")


# ---------------------------------------------------------------------------
# exp_013: NEW char n-gram channel + combined builder
# ---------------------------------------------------------------------------
#
# Korean-aware char_wb channel.  Per lesson `charwb-tfidf-vocab-single-thread-stall`
# the TfidfVectorizer vocabulary build is single-threaded, so on 70k mixed
# Korean/English prompts we MUST keep the ngram range narrow (2,3), min_df>=5 and
# max_features<=8000 to keep the vocab build fast.  (2,4) would stall.

def make_word_vec():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=25000, min_df=2, sublinear_tf=True,
    )


def make_char_vec():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 3),
        min_df=5, max_features=8000, sublinear_tf=True,
    )


def build_X(word_vec, char_vec, cat_mappings, prompts, records):
    """Combine word TF-IDF + char TF-IDF (sparse) with engineered dense features.

    Returns a single csr_matrix: hstack([Xword, Xchar, Xdense]).
    """
    Xw = word_vec.transform(prompts)
    Xc = char_vec.transform(prompts)
    Xd = sparse.csr_matrix(records_to_dense(records, cat_mappings).astype(np.float32))
    return sparse.hstack([Xw, Xc, Xd], format="csr")
