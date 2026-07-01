"""exp_013 LightGBM — exp_010 engineered features + char n-gram channel + OOF calibration.

Two levers on top of exp_010 (best ~0.68 CV):
  1. Korean-aware char_wb(2,3) TF-IDF channel (narrow ngram per stall lesson).
  2. OOF per-class multiplicative probability calibration (coordinate ascent) to
     maximize Macro-F1 of argmax(oof * w). Error analysis: top-2 acc 0.84 >> top-1
     0.67 -> argmax tie-break is the bottleneck, so calibration is high-value/cheap.

CV = StratifiedGroupKFold(5, group=session, seed=42). Reuses/extends features.py.
"""
import os, sys, json, time

os.environ.setdefault("OMP_NUM_THREADS", "16")

ROOT = os.path.dirname(os.path.abspath(__file__))
# --- line-buffered Tee -> train.log (real-time logging, lesson realtime-logging) ---
_logf = open(os.path.join(ROOT, "train.log"), "a", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, d):
        for s in self.streams:
            s.write(d); s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = sys.stderr = _Tee(sys.__stdout__, _logf)

import numpy as np
import joblib
from scipy import sparse

sys.path.insert(0, ROOT)
import features as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
import lightgbm as lgb

DATA = os.path.join(ROOT, "..", "..", "data")
MODEL = os.path.join(ROOT, "model")
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODEL, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
SEED, NJOBS, NFOLDS = 42, 16, 5

READ_SEARCH = ["read_file", "grep_search", "list_directory", "glob_pattern"]


def load():
    samples = F.load_jsonl(os.path.join(DATA, "train.jsonl"))
    ids, prompts, records = F.build_records(samples)
    id2a = {}
    with open(os.path.join(DATA, "train_labels.csv")) as f:
        next(f)
        for line in f:
            line = line.strip()
            if line:
                k, v = line.split(",", 1)
                id2a[k] = v
    y = np.array([F.CLASS_ORDER.index(id2a[i]) for i in ids], dtype=np.int64)
    groups = np.array([i.rsplit("-step", 1)[0] for i in ids])
    return ids, prompts, records, y, groups


def make_model():
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=14,
        n_estimators=500, learning_rate=0.05,
        num_leaves=63, min_child_samples=20,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.6, reg_lambda=2.0, reg_alpha=0.1,
        n_jobs=NJOBS, seed=SEED, verbose=-1,
    )


def sub(seq, idx):
    return [seq[i] for i in idx]


def calibrate(oof, y, n_passes=10):
    """Coordinate-ascent per-class multiplicative weights to maximize Macro-F1.

    w[c] in a small grid; ~n_passes over classes. Returns (weights, best_f1).
    """
    grid = np.round(np.arange(0.5, 2.001, 0.1), 3)
    w = np.ones(14, dtype=np.float64)

    def macro(weights):
        return f1_score(y, (oof * weights).argmax(1), average="macro")

    best = macro(w)
    for p in range(n_passes):
        improved = False
        for c in range(14):
            base = w[c]
            best_v, best_c = base, best
            for v in grid:
                w[c] = v
                sc = macro(w)
                if sc > best_c + 1e-9:
                    best_c, best_v = sc, v
            w[c] = best_v
            if best_c > best + 1e-9:
                best = best_c
                improved = True
        print(f"  calib pass {p}: macroF1={best:.4f}", flush=True)
        if not improved:
            break
    return w, float(best)


def main():
    t0 = time.time()
    ids, prompts, records, y, groups = load()
    print(f"loaded {len(ids)} rows, {len(set(groups))} sessions", flush=True)
    print(f"dense features: {len(F.dense_feature_names())}", flush=True)

    skf = StratifiedGroupKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 14), np.float32)
    fold_scores = []

    for k, (tr, va) in enumerate(skf.split(prompts, y, groups)):
        assert not (set(groups[tr]) & set(groups[va])), "group leak"
        wv = F.make_word_vec()
        wv.fit(sub(prompts, tr))
        cv = F.make_char_vec()
        cv.fit(sub(prompts, tr))
        cat_map = F.build_cat_mappings(sub(records, tr))
        Xtr = F.build_X(wv, cv, cat_map, sub(prompts, tr), sub(records, tr))
        Xva = F.build_X(wv, cv, cat_map, sub(prompts, va), sub(records, va))
        print(f"fold{k} fitted vectorizers, X={Xtr.shape} ({time.time()-t0:.0f}s)", flush=True)
        sw = compute_sample_weight("balanced", y[tr])
        m = make_model()
        m.fit(Xtr, y[tr], sample_weight=sw)
        p = m.predict_proba(Xva)
        oof[va] = p
        sc = f1_score(y[va], p.argmax(1), average="macro")
        fold_scores.append(float(sc))
        print(f"fold{k} macroF1={sc:.4f} ({time.time()-t0:.0f}s)", flush=True)

        joblib.dump(wv, os.path.join(MODELS, f"word_vec_f{k}.pkl"))
        joblib.dump(cv, os.path.join(MODELS, f"char_vec_f{k}.pkl"))
        json.dump(cat_map, open(os.path.join(MODELS, f"cat_mappings_f{k}.json"), "w"))
        joblib.dump(m, os.path.join(MODELS, f"lgbm_f{k}.pkl"))

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))

    # ---- Raw OOF metrics ----
    raw_argmax = oof.argmax(1)
    raw_macro = float(f1_score(y, raw_argmax, average="macro"))
    per_raw = f1_score(y, raw_argmax, average=None, labels=list(range(14)))
    per_class_raw = {F.CLASS_ORDER[i]: float(per_raw[i]) for i in range(14)}
    print(f"\nRAW OOF macroF1 = {raw_macro:.4f} (fold mean {cv_mean:.4f} +/- {cv_std:.4f})", flush=True)

    # ---- Calibration (headline) ----
    print("\ncalibrating per-class multiplicative weights ...", flush=True)
    weights, calib_macro = calibrate(oof, y, n_passes=10)
    calib_argmax = (oof * weights).argmax(1)
    per_cal = f1_score(y, calib_argmax, average=None, labels=list(range(14)))
    per_class_cal = {F.CLASS_ORDER[i]: float(per_cal[i]) for i in range(14)}
    worst = min(per_class_cal, key=per_class_cal.get)
    collapsed = [c for c, v in per_class_cal.items() if v < 0.05]

    print(f"\n=== OOF Macro-F1: RAW {raw_macro:.4f} -> CALIBRATED {calib_macro:.4f} "
          f"(+{calib_macro - raw_macro:.4f}) ===", flush=True)
    print(f"class_weights = {json.dumps({F.CLASS_ORDER[i]: round(float(weights[i]),2) for i in range(14)})}", flush=True)
    print("read/search cluster F1 (raw -> calib):", flush=True)
    rs_helped = True
    for c in READ_SEARCH:
        b, a = per_class_raw[c], per_class_cal[c]
        print(f"  {c:15s} {b:.4f} -> {a:.4f} ({a-b:+.4f})", flush=True)
    rs_before = float(np.mean([per_class_raw[c] for c in READ_SEARCH]))
    rs_after = float(np.mean([per_class_cal[c] for c in READ_SEARCH]))
    rs_helped = rs_after >= rs_before
    print(f"  read/search mean {rs_before:.4f} -> {rs_after:.4f}", flush=True)

    # ---- Full-train model for submission ----
    print("\nfull-train ...", flush=True)
    wv = F.make_word_vec(); wv.fit(prompts)
    cv = F.make_char_vec(); cv.fit(prompts)
    cat_map = F.build_cat_mappings(records)
    Xall = F.build_X(wv, cv, cat_map, prompts, records)
    sw = compute_sample_weight("balanced", y)
    m = make_model()
    m.fit(Xall, y, sample_weight=sw)

    joblib.dump(wv, os.path.join(MODEL, "word_vec.pkl"))
    joblib.dump(cv, os.path.join(MODEL, "char_vec.pkl"))
    json.dump(cat_map, open(os.path.join(MODEL, "cat_mappings.json"), "w"))
    joblib.dump(m, os.path.join(MODEL, "lgbm.pkl"))
    json.dump(F.CLASS_ORDER, open(os.path.join(MODEL, "class_order.json"), "w"))
    json.dump({F.CLASS_ORDER[i]: float(weights[i]) for i in range(14)},
              open(os.path.join(MODEL, "class_weights.json"), "w"), indent=2)
    np.save(os.path.join(ROOT, "oof_preds.npy"), oof)

    # ---- Timing ----
    n = min(2000, len(prompts))
    ts = time.time()
    Xs = F.build_X(wv, cv, cat_map, sub(prompts, range(n)), sub(records, range(n)))
    _ = m.predict_proba(Xs)
    ms = (time.time() - ts) / n * 1000
    size_mb = sum(os.path.getsize(os.path.join(MODEL, f)) for f in os.listdir(MODEL)) / 1048576

    log = {
        "experiment_id": "exp_013_engineered_char_calib",
        "metric": "macro_f1",
        "cv_strategy": f"StratifiedGroupKFold({NFOLDS},group=session,seed={SEED})",
        "cv_fold_scores": fold_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_mean_raw_oof": raw_macro,
        "cv_mean_calibrated": calib_macro,
        "calibration_gain": round(calib_macro - raw_macro, 4),
        "per_class_f1_raw": per_class_raw,
        "per_class_f1": per_class_cal,
        "worst_class": worst,
        "collapsed_classes": collapsed,
        "read_search_f1_before": {c: round(per_class_raw[c], 4) for c in READ_SEARCH},
        "read_search_f1_after": {c: round(per_class_cal[c], 4) for c in READ_SEARCH},
        "calibration_helped_read_search": bool(rs_helped),
        "class_weights": {F.CLASS_ORDER[i]: round(float(weights[i]), 3) for i in range(14)},
        "inference_ms_per_sample": round(ms, 3),
        "estimated_full_test_minutes": round(ms * 30000 / 1000 / 60, 2),
        "model_size_mb": round(size_mb, 1),
        "n_features": int(Xall.shape[1]),
        "n_dense_features": len(F.dense_feature_names()),
        "offline_compatible": True,
        "seed": SEED,
        "note": ("exp_010 engineered feats + char_wb(2,3) TF-IDF channel + OOF "
                 "coordinate-ascent per-class calibration for Macro-F1"),
    }
    json.dump(log, open(os.path.join(ROOT, "train_log.json"), "w"), indent=2)
    print(f"\nDONE raw={raw_macro:.4f} calib={calib_macro:.4f} "
          f"infer={ms:.1f}ms/sample model={size_mb:.1f}MB n_feat={Xall.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
