#!/usr/bin/env python3
"""Local training + 5-fold stratified CV for exp_003_tfidf_catboost.

NEVER submitted. Produces:
  - oof_preds.npy (70000, 14) float32
  - test_preds.npy (n_test, 14) float32
  - model/  (word_vec, char_vec, cat_mappings, class_order, catboost full-train model)
  - models/ (per-fold catboost models for diagnostics)
  - train_log.json

Uses the IDENTICAL feature pipeline + CV folds as exp_001 so CV is comparable.
"""

import json
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import joblib
from catboost import CatBoostClassifier

import features as F

HERE = Path(__file__).resolve().parent


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=HERE
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def make_vectorizers(cfg):
    tf = cfg["features"]["tfidf"]
    w = tf["word"]
    c = tf["char"]
    word_vec = TfidfVectorizer(
        analyzer=w["analyzer"],
        ngram_range=tuple(w["ngram_range"]),
        max_features=w["max_features"],
        min_df=w["min_df"],
        lowercase=True,
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer=c["analyzer"],
        ngram_range=tuple(c["ngram_range"]),
        max_features=c["max_features"],
        min_df=c["min_df"],
        lowercase=True,
        sublinear_tf=True,
    )
    return word_vec, char_vec


def make_model(cfg, seed):
    p = cfg["model"]["params"]
    return CatBoostClassifier(
        loss_function=cfg["model"]["objective"],
        classes_count=cfg["model"]["num_class"],
        iterations=p["iterations"],
        learning_rate=p["learning_rate"],
        depth=p["depth"],
        l2_leaf_reg=p["l2_leaf_reg"],
        rsm=p["rsm"],
        border_count=p["border_count"],
        early_stopping_rounds=p["early_stopping_rounds"],
        thread_count=p["thread_count"],
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def sample_weights_for(y_codes, n_classes):
    classes = np.arange(n_classes)
    cw = compute_class_weight("balanced", classes=classes, y=y_codes)
    return cw[y_codes]


def fit_fold(all_prompts, records, y_codes, tr_idx, cfg, seed, n_classes):
    """Fit vectorizers + model on tr_idx portion. Returns (artifacts, model)."""
    prompts_tr = [all_prompts[i] for i in tr_idx]
    records_tr = [records[i] for i in tr_idx]
    y_tr_full = y_codes[tr_idx]

    word_vec, char_vec = make_vectorizers(cfg)
    word_vec.fit(prompts_tr)
    char_vec.fit(prompts_tr)
    cat_mappings = F.build_cat_mappings(records_tr)
    artifacts = {"word_vec": word_vec, "char_vec": char_vec, "cat_mappings": cat_mappings}

    Xtr_full = F.transform_all(prompts_tr, records_tr, artifacts)

    # internal split for early stopping
    idx_local = np.arange(len(tr_idx))
    tr2, es2 = train_test_split(
        idx_local, test_size=0.1, random_state=seed, stratify=y_tr_full
    )
    Xtr = Xtr_full[tr2]
    Xes = Xtr_full[es2]
    ytr = y_tr_full[tr2]
    yes = y_tr_full[es2]
    sw = sample_weights_for(ytr, n_classes)

    model = make_model(cfg, seed)
    model.fit(
        Xtr, ytr,
        sample_weight=sw,
        eval_set=(Xes, yes),
        use_best_model=True,
        verbose=False,
    )
    return artifacts, model


def main():
    t0 = time.time()
    with open(HERE / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    seed = cfg["cv"]["seed"]
    set_seed(seed)
    n_classes = cfg["model"]["num_class"]

    # ---- Load data ----
    data = cfg["data"]
    train_path = (HERE / data["train_path"]).resolve()
    labels_path = (HERE / data["train_labels_path"]).resolve()
    test_path = (HERE / data["test_path"]).resolve()

    print(f"Loading train from {train_path}")
    samples = F.load_jsonl(str(train_path))
    labels_df = pd.read_csv(labels_path)
    label_map = dict(zip(labels_df[data["id_col"]], labels_df[data["target_col"]]))

    ids, prompts, records = F.build_records(samples)
    y_str = np.array([label_map[i] for i in ids])

    class_order = F.CLASS_ORDER
    class_to_code = {c: i for i, c in enumerate(class_order)}
    assert set(np.unique(y_str)).issubset(set(class_order)), "unknown label found"
    y_codes = np.array([class_to_code[s] for s in y_str], dtype=int)
    print(f"Loaded {len(ids)} train samples, {len(np.unique(y_codes))} classes")

    # ---- CV ----
    skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    oof = np.zeros((len(ids), n_classes), dtype=np.float32)
    fold_scores = []
    models_dir = HERE / cfg["output"]["per_fold_dir"]
    models_dir.mkdir(exist_ok=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_codes)), y_codes)):
        ft0 = time.time()
        artifacts, model = fit_fold(prompts, records, y_codes, tr_idx, cfg, seed, n_classes)

        prompts_va = [prompts[i] for i in va_idx]
        records_va = [records[i] for i in va_idx]
        Xva = F.transform_all(prompts_va, records_va, artifacts)
        proba = model.predict_proba(Xva)
        oof[va_idx] = proba.astype(np.float32)

        fs = f1_score(y_codes[va_idx], proba.argmax(1), average="macro")
        fold_scores.append(float(fs))
        model.save_model(str(models_dir / f"catboost_fold{fold}.cbm"))
        print(f"[fold {fold}] macro_f1={fs:.4f} best_iter={model.get_best_iteration()} "
              f"time={time.time()-ft0:.1f}s", flush=True)

    oof_macro = f1_score(y_codes, oof.argmax(1), average="macro")
    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    print(f"CV macro-f1 (fold mean) = {cv_mean:.4f} +/- {cv_std:.4f} | OOF-agg = {oof_macro:.4f}")

    per_class = f1_score(y_codes, oof.argmax(1), average=None, labels=list(range(n_classes)))
    per_class_f1 = {str(i): float(per_class[i]) for i in range(n_classes)}
    per_class_named = {class_order[i]: float(per_class[i]) for i in range(n_classes)}
    worst_i = int(np.argmin(per_class))
    best_i = int(np.argmax(per_class))
    collapsed = [i for i in range(n_classes) if per_class[i] < 0.05]

    assert np.isfinite(oof).all(), "NaN/Inf in oof preds"
    np.save(HERE / cfg["output"]["oof_predictions"], oof)

    # ---- Full retrain on all data ----
    print("Retraining on full train ...", flush=True)
    all_idx = np.arange(len(ids))
    artifacts_full, model_full = fit_fold(prompts, records, y_codes, all_idx, cfg, seed, n_classes)

    # ---- Save model/ artifacts ----
    model_dir = HERE / cfg["output"]["weights_dir"]
    model_dir.mkdir(exist_ok=True)
    joblib.dump(artifacts_full["word_vec"], model_dir / "word_vec.pkl")
    joblib.dump(artifacts_full["char_vec"], model_dir / "char_vec.pkl")
    with open(model_dir / "cat_mappings.json", "w") as f:
        json.dump(artifacts_full["cat_mappings"], f)
    with open(model_dir / "class_order.json", "w") as f:
        json.dump(class_order, f)
    model_full.save_model(str(model_dir / "catboost.cbm"))

    # ---- Test predictions ----
    print(f"Predicting test from {test_path}")
    test_samples = F.load_jsonl(str(test_path))
    t_ids, t_prompts, t_records = F.build_records(test_samples)
    Xtest = F.transform_all(t_prompts, t_records, artifacts_full)
    test_proba = model_full.predict_proba(Xtest).astype(np.float32)
    assert np.isfinite(test_proba).all(), "NaN/Inf in test preds"
    np.save(HERE / cfg["output"]["test_predictions"], test_proba)

    # ---- Inference timing on up to 1000-row sample ----
    n_time = min(1000, len(ids))
    time_prompts = prompts[:n_time]
    time_records = records[:n_time]
    it0 = time.time()
    Xt = F.transform_all(time_prompts, time_records, artifacts_full)
    _ = model_full.predict_proba(Xt)
    infer_total = time.time() - it0
    inference_ms_per_sample = (infer_total / n_time) * 1000.0
    estimated_full_test_minutes = (inference_ms_per_sample * 30000) / 1000.0 / 60.0

    # ---- Model size ----
    model_size_mb = sum(
        f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    n_features = int(Xtest.shape[1])

    try:
        importances = model_full.get_feature_importance()
        dense_names = F.dense_feature_names()
        n_dense = len(dense_names)
        tail = importances[-n_dense:]
        order = np.argsort(tail)[::-1][:10]
        feat_imp = {dense_names[i]: float(tail[i]) for i in order}
    except Exception:
        feat_imp = {}

    runtime = time.time() - t0
    log = {
        "experiment_id": cfg["experiment"]["id"],
        "metric": "macro_f1",
        "cv_fold_scores": fold_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "oof_macro_f1": float(oof_macro),
        "per_class_f1": per_class_f1,
        "per_class_f1_named": per_class_named,
        "worst_class": {"id": worst_i, "name": class_order[worst_i], "f1": float(per_class[worst_i])},
        "best_class": {"id": best_i, "name": class_order[best_i], "f1": float(per_class[best_i])},
        "collapsed_classes": collapsed,
        "runtime_seconds_train": round(runtime, 1),
        "inference_ms_per_sample": round(inference_ms_per_sample, 3),
        "estimated_full_test_minutes": round(estimated_full_test_minutes, 3),
        "model_size_mb": round(model_size_mb, 2),
        "n_features": n_features,
        "feature_importance_top10": feat_imp,
        "offline_compatible": True,
        "seed": seed,
        "git_commit": git_commit(),
    }
    with open(HERE / cfg["output"]["log_file"], "w") as f:
        json.dump(log, f, indent=2)

    print(json.dumps({k: log[k] for k in [
        "cv_mean", "cv_std", "oof_macro_f1", "worst_class", "collapsed_classes",
        "inference_ms_per_sample", "estimated_full_test_minutes", "model_size_mb", "n_features"
    ]}, indent=2))
    print(f"Done in {runtime:.1f}s")


if __name__ == "__main__":
    main()
