"""Shared feature engineering for exp_001_tfidf_lightgbm.

Imported by BOTH train.py (fits vectorizers/encoders) and script.py (loads them).
Keep this module free of any training / fitting side effects at import time.

Feature groups
--------------
1. TF-IDF on current_prompt: word (1,2) + char_wb (2,4).
2. Sequential (from history): last_action, second_last_action, last_action_failed,
   history_len, per-action counts (14 features).
3. Rule categories (regex one-hot on current_prompt): 8 categories from domain_notes 8.1.
4. session_meta / workspace: user_tier, language_pref, primary language mix, git_dirty,
   last_ci_status, turn_index, elapsed_session_sec, budget_tokens_remaining,
   workspace.loc, len(open_files).

All categoricals -> integer codes via saved mappings (unknown -> code for "__UNK__").
"""

import json
import re

import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Exact 14-class order (submission strings, case-sensitive). This is also the
# canonical CLASS_ORDER used everywhere unless a saved list overrides it.
CLASS_ORDER = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

# Sentinel tokens for categorical encoding.
NONE_TOKEN = "NONE"
UNK_TOKEN = "__UNK__"

# Rule category regexes (domain_notes 8.1). Order is fixed -> stable one-hot columns.
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

# Categorical meta fields that get integer-coded (with saved mappings).
CAT_META_FIELDS = [
    "user_tier",
    "language_pref",
    "primary_lang",
    "git_dirty",
    "last_ci_status",
]

# Numeric meta fields (kept raw).
NUM_META_FIELDS = [
    "turn_index",
    "elapsed_session_sec",
    "budget_tokens_remaining",
    "loc",
    "n_open_files",
]


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Load a jsonl file into a list of dicts."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


# ---------------------------------------------------------------------------
# Raw field extraction (per sample) -- no fitting
# ---------------------------------------------------------------------------

def _safe_str(x):
    if x is None:
        return ""
    if not isinstance(x, str):
        return str(x)
    return x


def get_prompt(sample):
    return _safe_str(sample.get("current_prompt", ""))


def _last_actions(history):
    """Return list of assistant_action dicts in order."""
    if not isinstance(history, list):
        return []
    return [h for h in history if isinstance(h, dict) and h.get("role") == "assistant_action"]


def extract_seq_meta(sample):
    """Extract raw sequential + meta signals as a plain dict (strings/numbers)."""
    history = sample.get("history", []) or []
    actions = _last_actions(history)

    last_action = actions[-1].get("name", NONE_TOKEN) if len(actions) >= 1 else NONE_TOKEN
    second_last = actions[-2].get("name", NONE_TOKEN) if len(actions) >= 2 else NONE_TOKEN

    last_failed = 0
    if actions:
        rs = _safe_str(actions[-1].get("result_summary", ""))
        if "ERROR" in rs.upper() or "FAIL" in rs.upper():
            last_failed = 1

    # per-action counts in history
    counts = {c: 0 for c in CLASS_ORDER}
    for a in actions:
        name = a.get("name")
        if name in counts:
            counts[name] += 1

    history_len = len(history) if isinstance(history, list) else 0

    meta = sample.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}

    # primary language from language_mix (argmax)
    lang_mix = ws.get("language_mix", {}) or {}
    if isinstance(lang_mix, dict) and lang_mix:
        primary_lang = max(lang_mix.items(), key=lambda kv: (kv[1] if isinstance(kv[1], (int, float)) else -1))[0]
    else:
        primary_lang = NONE_TOKEN

    open_files = ws.get("open_files", []) or []
    n_open_files = len(open_files) if isinstance(open_files, list) else 0

    def _num(v, default=0):
        try:
            if v is None:
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    out = {
        "last_action": _safe_str(last_action) or NONE_TOKEN,
        "second_last_action": _safe_str(second_last) or NONE_TOKEN,
        "last_action_failed": last_failed,
        "history_len": history_len,
        # categorical meta
        "user_tier": _safe_str(meta.get("user_tier", NONE_TOKEN)) or NONE_TOKEN,
        "language_pref": _safe_str(meta.get("language_pref", NONE_TOKEN)) or NONE_TOKEN,
        "primary_lang": _safe_str(primary_lang) or NONE_TOKEN,
        "git_dirty": _safe_str(ws.get("git_dirty", NONE_TOKEN)),
        "last_ci_status": _safe_str(ws.get("last_ci_status", NONE_TOKEN)) or NONE_TOKEN,
        # numeric meta
        "turn_index": _num(meta.get("turn_index"), 0),
        "elapsed_session_sec": _num(meta.get("elapsed_session_sec"), 0),
        "budget_tokens_remaining": _num(meta.get("budget_tokens_remaining"), 0),
        "loc": _num(ws.get("loc"), 0),
        "n_open_files": n_open_files,
    }
    for c in CLASS_ORDER:
        out["cnt_" + c] = counts[c]
    return out


def rule_flags(prompt):
    """Return list of 0/1 flags for each rule category (stable order)."""
    return [1 if rgx.search(prompt) else 0 for _, rgx in RULE_CATEGORIES]


# ---------------------------------------------------------------------------
# Categorical mapping (fit on train, apply everywhere)
# ---------------------------------------------------------------------------

def build_cat_mappings(records):
    """Build {field: {value: code}} mappings. Reserve code 0 for UNK.

    Fields covered: last_action, second_last_action, and CAT_META_FIELDS.
    """
    fields = ["last_action", "second_last_action"] + CAT_META_FIELDS
    mappings = {}
    for field in fields:
        vals = sorted({str(r[field]) for r in records})
        m = {UNK_TOKEN: 0}
        for v in vals:
            if v not in m:
                m[v] = len(m)
        mappings[field] = m
    return mappings


def _encode_cat(value, mapping):
    return mapping.get(str(value), mapping.get(UNK_TOKEN, 0))


def records_to_dense(records, cat_mappings):
    """Build the dense (non-TFIDF) feature matrix from extracted records.

    Column order (must stay stable across train/infer):
      [encoded last_action, encoded second_last_action,
       encoded CAT_META_FIELDS...,
       last_action_failed, history_len,
       NUM_META_FIELDS...,
       cnt_<class> x14,
       rule_flags x8]
    """
    rows = []
    cat_fields = ["last_action", "second_last_action"] + CAT_META_FIELDS
    for r in records:
        row = []
        for field in cat_fields:
            row.append(_encode_cat(r[field], cat_mappings[field]))
        row.append(r["last_action_failed"])
        row.append(r["history_len"])
        for field in NUM_META_FIELDS:
            row.append(r[field])
        for c in CLASS_ORDER:
            row.append(r["cnt_" + c])
        row.extend(r["_rule_flags"])
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def dense_feature_names():
    cat_fields = ["last_action", "second_last_action"] + CAT_META_FIELDS
    names = list(cat_fields)
    names += ["last_action_failed", "history_len"]
    names += list(NUM_META_FIELDS)
    names += ["cnt_" + c for c in CLASS_ORDER]
    names += ["rule_" + n for n in RULE_NAMES]
    return names


# ---------------------------------------------------------------------------
# Top-level feature builders
# ---------------------------------------------------------------------------

def build_records(samples):
    """samples -> (ids, prompts, records) where records carry seq/meta + rule flags."""
    ids, prompts, records = [], [], []
    for s in samples:
        ids.append(_safe_str(s.get("id", "")))
        p = get_prompt(s)
        prompts.append(p)
        rec = extract_seq_meta(s)
        rec["_rule_flags"] = rule_flags(p)
        records.append(rec)
    return ids, prompts, records


def transform_all(prompts, records, artifacts):
    """Combine TF-IDF sparse blocks with dense features into one CSR matrix.

    artifacts must contain: 'word_vec', 'char_vec', 'cat_mappings'.
    """
    word_vec = artifacts["word_vec"]
    char_vec = artifacts["char_vec"]
    cat_mappings = artifacts["cat_mappings"]

    Xw = word_vec.transform(prompts)
    Xc = char_vec.transform(prompts)
    Xd = records_to_dense(records, cat_mappings)
    Xd_sparse = sparse.csr_matrix(Xd)
    return sparse.hstack([Xw, Xc, Xd_sparse], format="csr")
