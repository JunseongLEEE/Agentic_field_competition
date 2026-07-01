"""
DACON Competition Dashboard — Streamlit Web UI
==============================================

Records LB scores after manual DACON submission, replacing the
`/submit-result` slash command with a web form. Wraps existing scripts:
- scripts/track_submission.py
- scripts/cv_lb_correlation.py
- scripts/check_time_state.py
- scripts/build_digest.py

Run:
    streamlit run webapp/app.py --server.address 127.0.0.1 --server.port 8501
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent.parent
META_PATH = ROOT / "competition_meta.yaml"
STATE_PATH = ROOT / "logs" / "orchestrator_state.json"
INSIGHTS_PATH = ROOT / "logs" / "insights.jsonl"
SUBMISSIONS_DIR = ROOT / "submissions"
EXPERIMENTS_DIR = ROOT / "experiments"
BACKUP_DIR = ROOT / "logs" / "backups"
EXPLOG_PATH = ROOT / "EXPERIMENT_LOG.csv"
KST = timezone(timedelta(hours=9))

st.set_page_config(
    page_title="DACON Competition Dashboard",
    page_icon="🏆",
    layout="wide",
)


# ─────────────────────────────────────────────────────────────────────
# AUTH GATE — simple password from env var or .streamlit/secrets.toml
# ─────────────────────────────────────────────────────────────────────
def check_password() -> bool:
    """Return True if user entered the correct password."""
    expected = os.environ.get("DASHBOARD_PASSWORD")
    if not expected and hasattr(st, "secrets") and "password" in st.secrets:
        expected = st.secrets["password"]
    if not expected:
        st.error(
            "Server misconfiguration: DASHBOARD_PASSWORD env var not set.\n\n"
            "Run with: `DASHBOARD_PASSWORD=yourpw streamlit run webapp/app.py`\n"
            "Or create `.streamlit/secrets.toml` with `password = \"yourpw\"`"
        )
        st.stop()

    if st.session_state.get("authenticated"):
        return True

    st.title("DACON Dashboard — Login")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if pw == expected:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Wrong password")
    return False


if not check_password():
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# DATA LOADERS (cached briefly so manual record triggers refresh)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_meta() -> dict:
    with open(META_PATH) as f:
        return yaml.safe_load(f) or {}


@st.cache_data(ttl=5)
def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    with open(STATE_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=5)
def load_insights() -> list[dict]:
    if not INSIGHTS_PATH.exists():
        return []
    rows = []
    with open(INSIGHTS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


@st.cache_data(ttl=10)
def list_packaged_experiments() -> list[str]:
    """Scan submissions/ for packaged zips."""
    if not SUBMISSIONS_DIR.exists():
        return []
    return sorted(
        p.stem
        for p in SUBMISSIONS_DIR.glob("*.zip")
        if p.is_file()
    )


@st.cache_data(ttl=10)
def list_all_experiments() -> list[str]:
    """Scan experiments/ for any experiment directory."""
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        p.name
        for p in EXPERIMENTS_DIR.iterdir()
        if p.is_dir() and (p / "train_log.json").exists()
    )


def load_experiment_meta(exp: str) -> dict:
    """Read train_log.json + evaluation.json for an experiment."""
    out: dict = {}
    train_log = EXPERIMENTS_DIR / exp / "train_log.json"
    if train_log.exists():
        with open(train_log) as f:
            out["train_log"] = json.load(f)
    eval_path = EXPERIMENTS_DIR / exp / "evaluation.json"
    if eval_path.exists():
        with open(eval_path) as f:
            out["evaluation"] = json.load(f)
    return out


def get_cv_lb_pairs() -> list[tuple[float, float, str]]:
    """Extract (cv, lb, exp_id) triples from submissions_log."""
    meta = load_meta()
    log = meta.get("submissions_log") or []
    pairs = []
    for entry in log:
        if entry.get("status") != "success":
            continue
        lb = entry.get("lb_score")
        exp = entry.get("experiment")
        if lb is None or not exp:
            continue
        em = load_experiment_meta(exp)
        cv = (em.get("train_log") or {}).get("cv_mean")
        if cv is not None:
            pairs.append((cv, lb, exp))
    return pairs


# ─────────────────────────────────────────────────────────────────────
# BACKEND CALLS — wrap subprocess for existing scripts
# ─────────────────────────────────────────────────────────────────────
def backup_meta() -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"competition_meta.{ts}.yaml.bak"
    shutil.copy2(META_PATH, dest)
    # Keep only last 30 backups
    backups = sorted(BACKUP_DIR.glob("competition_meta.*.yaml.bak"))
    for old in backups[:-30]:
        old.unlink()
    return dest


def run_track_submission(exp: str, lb: float | None, status: str) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "track_submission.py"),
        "--exp", exp,
        "--status", status,
    ]
    if lb is not None:
        cmd += ["--lb", str(lb)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        ok = res.returncode == 0
        return ok, (res.stdout + res.stderr).strip()
    except Exception as e:
        return False, f"subprocess error: {e}"


def refit_correlation() -> dict:
    cmd = [sys.executable, str(ROOT / "scripts" / "cv_lb_correlation.py"), "--json"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            return json.loads(res.stdout)
    except Exception:
        pass
    return {}


def predict_lb_for_cv(cv: float) -> dict:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "cv_lb_correlation.py"),
        "--predict", str(cv),
        "--json",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            return json.loads(res.stdout)
    except Exception:
        pass
    return {}


def get_time_state() -> dict:
    cmd = [sys.executable, str(ROOT / "scripts" / "check_time_state.py"), "--json"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if res.returncode == 0:
            return json.loads(res.stdout)
    except Exception:
        pass
    return {}


def rebuild_digest() -> bool:
    cmd = [sys.executable, str(ROOT / "scripts" / "build_digest.py")]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return res.returncode == 0
    except Exception:
        return False


def append_experiment_log(
    exp_id: str,
    lb_score: float | None,
    cv_score: float | None,
    status: str,
    notes: str = "",
) -> None:
    """Append one row to EXPERIMENT_LOG.csv."""
    now_kst = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    cv_lb_gap = ""
    if cv_score is not None and lb_score is not None and lb_score > 0:
        cv_lb_gap = f"{cv_score - lb_score:.4f}"
    row = {
        "experiment_id": exp_id,
        "name": exp_id,
        "hypothesis": "",
        "status": status,
        "cv_score": f"{cv_score:.4f}" if cv_score is not None else "",
        "cv_std": "",
        "lb_score": f"{lb_score:.4f}" if lb_score is not None else "",
        "cv_lb_gap": cv_lb_gap,
        "seed": "",
        "git_commit": "",
        "created_at": now_kst,
        "completed_at": now_kst,
        "notes": notes,
    }
    df_new = pd.DataFrame([row])
    header = not EXPLOG_PATH.exists() or EXPLOG_PATH.stat().st_size == 0
    if EXPLOG_PATH.exists():
        existing = pd.read_csv(EXPLOG_PATH)
        if len(existing) > 0:
            header = False
    df_new.to_csv(EXPLOG_PATH, mode="a", header=header, index=False)


@st.cache_data(ttl=5)
def load_experiment_log() -> pd.DataFrame:
    if not EXPLOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(EXPLOG_PATH)
    return df if len(df) > 0 else pd.DataFrame()


def clear_caches() -> None:
    load_meta.clear()
    load_state.clear()
    load_insights.clear()
    list_packaged_experiments.clear()
    list_all_experiments.clear()
    load_experiment_log.clear()


# ─────────────────────────────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────────────────────────────
st.title("🏆 DACON Competition Dashboard")

with st.sidebar:
    st.markdown("### Session")
    st.write(f"Logged in")
    if st.button("Refresh data"):
        clear_caches()
        st.rerun()
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()
    st.markdown("---")
    st.caption(f"Root: `{ROOT.name}`")
    st.caption(f"Now (KST): {datetime.now(KST).strftime('%Y-%m-%d %H:%M:%S')}")

time_state = get_time_state()
meta = load_meta()
state = load_state()

c1, c2, c3, c4 = st.columns(4)
with c1:
    days = time_state.get("days_to_preliminary", "?")
    st.metric("Days to preliminary", f"D-{days}" if isinstance(days, int) else days)
with c2:
    used = time_state.get("submissions_today", 0)
    limit = time_state.get("daily_limit", 10)
    st.metric("Today's quota", f"{used}/{limit}", delta=f"{limit - used} left")
with c3:
    best_cv = state.get("best_cv")
    best_exp = state.get("best_experiment", "—")
    st.metric("Best CV", f"{best_cv:.4f}" if best_cv else "—", help=f"exp: {best_exp}")
with c4:
    best_lb = state.get("best_lb")
    best_lb_exp = state.get("best_lb_experiment", "—")
    st.metric("Best LB", f"{best_lb:.4f}" if best_lb else "—", help=f"exp: {best_lb_exp}")

# CV-LB correlation summary
cvlb = refit_correlation()
if cvlb:
    n = cvlb.get("n_pairs", 0)
    trust = cvlb.get("trust_level", "low")
    r = cvlb.get("pearson_r")
    sigma = cvlb.get("residual_std")
    badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(trust, "⚪")
    cols = st.columns([1, 1, 1, 1])
    cols[0].caption(f"{badge} CV→LB trust: **{trust}**")
    cols[1].caption(f"n_pairs: {n}")
    cols[2].caption(f"pearson_r: {r:.3f}" if isinstance(r, (int, float)) else "pearson_r: —")
    cols[3].caption(f"σ: {sigma:.4f}" if isinstance(sigma, (int, float)) else "σ: —")

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────
# UI — SUBMISSION RECORD FORM
# ─────────────────────────────────────────────────────────────────────
st.header("📤 Record DACON Submission")

packaged = list_packaged_experiments()
all_exps = list_all_experiments()
exp_options = packaged or all_exps

col_form, col_preview = st.columns([1, 1])

with col_form:
    if exp_options:
        input_mode = st.radio(
            "Experiment input",
            ["Select existing", "Type manually"],
            horizontal=True,
        )
        if input_mode == "Select existing":
            exp_choice = st.selectbox(
                "Experiment",
                options=exp_options,
                help="Packaged experiments listed first" if packaged else "All experiments",
            )
        else:
            exp_choice = st.text_input("Experiment ID", placeholder="exp_001_baseline_lgbm")
    else:
        exp_choice = st.text_input("Experiment ID", placeholder="exp_001_baseline_lgbm")

    lb_input = st.number_input(
        "LB Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.0001,
        format="%.4f",
        help="Macro-F1 score from DACON leaderboard",
    )

    status_choice = st.radio(
        "Status",
        options=["success", "runtime_error", "install_error"],
        horizontal=True,
        help=(
            "success: scored on LB | "
            "runtime_error: counts against quota, lb=0 | "
            "install_error: does NOT count against quota"
        ),
    )

    notes_input = st.text_input("Notes (optional)", placeholder="e.g. added feature X")

    # Duplicate check
    existing_log = meta.get("submissions_log") or []
    if exp_choice and any(e.get("experiment") == exp_choice for e in existing_log):
        st.warning(f"'{exp_choice}' already has a submission record")

    submit_btn = st.button("Record Submission", type="primary", use_container_width=True)

    if submit_btn:
        if not exp_choice or not exp_choice.strip():
            st.error("Experiment ID is required")
        elif status_choice == "success" and lb_input <= 0:
            st.error("Success status requires LB > 0")
        else:
            exp_choice = exp_choice.strip()
            lb_value = lb_input if status_choice == "success" else (0.0 if status_choice == "runtime_error" else None)

            # 1) Append to EXPERIMENT_LOG.csv
            append_experiment_log(
                exp_id=exp_choice,
                lb_score=lb_value,
                cv_score=None,
                status=status_choice,
                notes=notes_input,
            )

            # 2) Also update competition_meta.yaml via track_submission
            backup_path = backup_meta()
            ok, output = run_track_submission(exp_choice, lb_value, status_choice)

            if ok:
                st.success(f"Recorded `{exp_choice}` | status={status_choice} | LB={lb_value}")
                new_cvlb = refit_correlation()
                rebuild_digest()
                if new_cvlb and new_cvlb.get("n_pairs", 0) > 0:
                    st.info(
                        f"Correlation: n={new_cvlb.get('n_pairs')} | "
                        f"r={new_cvlb.get('pearson_r')} | "
                        f"trust={new_cvlb.get('trust_level')}"
                    )
                clear_caches()
            else:
                # track_submission failed but CSV was still written
                st.warning(f"CSV recorded, but meta update failed: {output}")
                clear_caches()

with col_preview:
    st.subheader("EXPERIMENT_LOG.csv")
    exp_log_df = load_experiment_log()
    if len(exp_log_df) > 0:
        show_cols = [c for c in ["experiment_id", "status", "cv_score", "lb_score", "cv_lb_gap", "notes", "created_at"] if c in exp_log_df.columns]
        st.dataframe(exp_log_df[show_cols].iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.caption("No records yet — submit your first result!")

    # Show experiment meta preview if available
    if exp_choice and exp_options and exp_choice in (packaged + all_exps):
        em = load_experiment_meta(exp_choice)
        cv_mean = (em.get("train_log") or {}).get("cv_mean")
        if cv_mean is not None:
            st.markdown("---")
            cv_std_val = (em.get("train_log") or {}).get("cv_std")
            st.write(f"**CV Macro-F1**: {cv_mean:.4f}" + (f" +/- {cv_std_val:.4f}" if cv_std_val else ""))

st.markdown("---")


# ─────────────────────────────────────────────────────────────────────
# UI — RECENT SUBMISSIONS + CV vs LB SCATTER
# ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1])

with left:
    st.header("📋 Recent Submissions")
    log = meta.get("submissions_log") or []
    if log:
        df = pd.DataFrame(log[-15:][::-1])  # latest first
        cols_show = [c for c in ["date", "experiment", "lb_score", "status", "counts_against_daily"] if c in df.columns]
        st.dataframe(df[cols_show], use_container_width=True, hide_index=True)
    else:
        st.caption("No submissions yet.")

with right:
    st.header("📊 CV vs LB")
    pairs = get_cv_lb_pairs()
    if pairs:
        scatter_df = pd.DataFrame(pairs, columns=["cv", "lb", "experiment"])
        st.scatter_chart(scatter_df, x="cv", y="lb", size=80)
        st.caption(f"{len(pairs)} successful submission(s) plotted.")
    else:
        st.caption("No successful submissions to plot yet.")


# ─────────────────────────────────────────────────────────────────────
# UI — FAMILY BRACKET (from orchestrator_state.json)
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("🏁 Model Family Bracket")
family_stats = state.get("family_stats") or {}
if family_stats:
    rows = []
    for fam, s in family_stats.items():
        rows.append({
            "family": fam,
            "tier": s.get("tier", "?"),
            "tried": s.get("tried", 0),
            "best_cv": s.get("best_cv"),
            "best_exp": s.get("best_exp", "—"),
            "avg_lb_gap": s.get("avg_lb_gap"),
            "status": s.get("status", "?"),
        })
    df = pd.DataFrame(rows).sort_values(
        by="best_cv", ascending=False, na_position="last"
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.caption("No family stats yet. Run `/auto` to start the model family sweep.")


# ─────────────────────────────────────────────────────────────────────
# UI — RECENT INSIGHTS
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.header("💡 Recent Insights")
insights = load_insights()
if insights:
    for r in insights[-5:][::-1]:
        with st.expander(
            f"{r.get('experiment', '?')} — gap {r.get('gap', 0):+.4f} — {r.get('insight', '')[:60]}"
        ):
            st.json(r)
else:
    st.caption("No insights yet.")
