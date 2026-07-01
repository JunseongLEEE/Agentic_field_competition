"""Shared input builder for exp_015 — rich CTX + HIST + NOW format.

Used by BOTH train.py and script.py. No fitting side effects at import.
"""
import csv
import json

CLASS_ORDER = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]

ARG_KEYS = ("path", "pattern", "target_symbol", "cmd", "scope")


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_labels(path):
    labels = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row["id"]] = row["action"]
    return labels


def session_id(sample_id):
    if "-step" in sample_id:
        return sample_id.rsplit("-step", 1)[0]
    return sample_id


def _archetype(workspace):
    mix = workspace.get("language_mix") or {}
    if not isinstance(mix, dict) or not mix:
        return "unknown"
    keys = sorted(mix.keys(), key=lambda k: (-float(mix.get(k, 0) or 0), k))
    return "_".join(keys[:3])


def _last_action_result(history):
    last_action = "NONE"
    last_result = "NONE"
    if not isinstance(history, list):
        return last_action, last_result
    for h in reversed(history):
        if not isinstance(h, dict) or h.get("role") != "assistant_action":
            continue
        last_action = h.get("name") or "NONE"
        r = str(h.get("result_summary") or "")
        ru = r.upper()
        if "FAIL" in ru or "ERROR" in ru:
            last_result = "FAIL"
        elif "PASS" in ru or ru.startswith("OK") or "ok;" in r.lower():
            last_result = "OK"
        else:
            last_result = "DONE"
        break
    return last_action, last_result


def _arg_hint(args):
    if not isinstance(args, dict):
        return ""
    for k in ARG_KEYS:
        if k in args and args[k] is not None:
            return f"({str(args[k])[:40]})"
    return ""


def build_input(sample, user_content_max=80, history_window=8, arg_hint_max=40):
    """Rich encoder input: [CTX] meta + [HIST] recent turns + [NOW] prompt."""
    meta = sample.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    history = sample.get("history") or []
    prompt = sample.get("current_prompt") or ""

    last_action, last_result = _last_action_result(history)
    ctx = (
        f"turn={int(meta.get('turn_index') or 0)} "
        f"arch={_archetype(ws)} "
        f"last={last_action} "
        f"result={last_result} "
        f"ci={ws.get('last_ci_status', 'none')} "
        f"dirty={int(bool(ws.get('git_dirty')))}"
    )

    hist_parts = []
    recent = history[-history_window:] if isinstance(history, list) else []
    for h in recent:
        if not isinstance(h, dict):
            continue
        if h.get("role") == "user":
            content = str(h.get("content") or "")[:user_content_max]
            hist_parts.append(f"U: {content}")
        elif h.get("role") == "assistant_action":
            hint = _arg_hint(h.get("args") or {})
            if hint and len(hint) > arg_hint_max + 2:
                hint = f"({hint[1:arg_hint_max + 1]})"
            hist_parts.append(f"A: {h.get('name', 'NONE')}{hint}")

    hist_text = " | ".join(hist_parts)
    return f"[CTX] {ctx} [HIST] {hist_text} [NOW] {prompt}"


def build_dataset(samples, label_map=None, **kwargs):
    """Return parallel lists: ids, texts, labels (or None), groups."""
    ids, texts, labels, groups = [], [], [], []
    for s in samples:
        sid = s.get("id", "")
        ids.append(sid)
        texts.append(build_input(s, **kwargs))
        groups.append(session_id(sid))
        if label_map is not None:
            lab = label_map.get(sid)
            labels.append(CLASS_ORDER.index(lab) if lab in CLASS_ORDER else -1)
    return ids, texts, labels if label_map is not None else None, groups
