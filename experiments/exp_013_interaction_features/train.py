"""exp_010 LightGBM — PCA-filtered + 25 new engineered features.

CV = StratifiedGroupKFold(5, group=session).
Hypothesis: new interaction features (bigram, trigram, turn_action, prompt_intent)
should lift Macro-F1 from ~0.67 (exp_001) to >0.72.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
import joblib
from scipy import sparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import features as F
from sklearn.feature_extraction.text import TfidfVectorizer
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


def word_vec():
    return TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=25000, min_df=2, sublinear_tf=True,
    )


def make_model():
    return lgb.LGBMClassifier(
        objective="multiclass", num_class=14,
        n_estimators=500, learning_rate=0.06,
        num_leaves=127, min_child_samples=20,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.6, reg_lambda=2.0, reg_alpha=0.1,
        n_jobs=NJOBS, seed=SEED, verbose=-1,
    )


def feats(wv, cat_map, prompts, records):
    Xw = wv.transform(prompts)
    Xd = sparse.csr_matrix(F.records_to_dense(records, cat_map).astype(np.float32))
    return sparse.hstack([Xw, Xd], format="csr")


def sub(seq, idx):
    return [seq[i] for i in idx]


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
        wv = word_vec()
        wv.fit(sub(prompts, tr))
        cat_map = F.build_cat_mappings(sub(records, tr))
        Xtr = feats(wv, cat_map, sub(prompts, tr), sub(records, tr))
        Xva = feats(wv, cat_map, sub(prompts, va), sub(records, va))
        sw = compute_sample_weight("balanced", y[tr])
        m = make_model()
        m.fit(Xtr, y[tr], sample_weight=sw)
        p = m.predict_proba(Xva)
        oof[va] = p
        sc = f1_score(y[va], p.argmax(1), average="macro")
        fold_scores.append(float(sc))
        print(f"fold{k} macroF1={sc:.4f} ({time.time()-t0:.0f}s)", flush=True)

        joblib.dump(wv, os.path.join(MODELS, f"word_vec_f{k}.pkl"))
        json.dump(cat_map, open(os.path.join(MODELS, f"cat_mappings_f{k}.json"), "w"))
        joblib.dump(m, os.path.join(MODELS, f"lgbm_f{k}.pkl"))

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    per = f1_score(y, oof.argmax(1), average=None, labels=list(range(14)))
    per_class = {F.CLASS_ORDER[i]: float(per[i]) for i in range(14)}
    worst = min(per_class, key=per_class.get)
    collapsed = [c for c, v in per_class.items() if v < 0.05]
    print(f"\nCV macroF1 = {cv_mean:.4f} +/- {cv_std:.4f}", flush=True)
    print(f"per-class F1: {json.dumps({k: round(v,4) for k,v in per_class.items()})}", flush=True)
    print(f"worst: {worst} ({per_class[worst]:.4f})", flush=True)
    if collapsed:
        print(f"COLLAPSED classes (<0.05): {collapsed}", flush=True)

    # Full-train model for submission
    print("\nfull-train ...", flush=True)
    wv = word_vec()
    wv.fit(prompts)
    cat_map = F.build_cat_mappings(records)
    Xall = feats(wv, cat_map, prompts, records)
    sw = compute_sample_weight("balanced", y)
    m = make_model()
    m.fit(Xall, y, sample_weight=sw)

    joblib.dump(wv, os.path.join(MODEL, "word_vec.pkl"))
    json.dump(cat_map, open(os.path.join(MODEL, "cat_mappings.json"), "w"))
    joblib.dump(m, os.path.join(MODEL, "lgbm.pkl"))
    json.dump(F.CLASS_ORDER, open(os.path.join(MODEL, "class_order.json"), "w"))
    np.save(os.path.join(ROOT, "oof_preds.npy"), oof)

    # Timing
    n = min(2000, len(prompts))
    ts = time.time()
    Xs = feats(wv, cat_map, sub(prompts, range(n)), sub(records, range(n)))
    _ = m.predict_proba(Xs)
    ms = (time.time() - ts) / n * 1000
    size_mb = sum(
        os.path.getsize(os.path.join(MODEL, f))
        for f in os.listdir(MODEL)
    ) / 1048576

    log = {
        "experiment_id": "exp_010_pca_engineered_lgbm",
        "metric": "macro_f1",
        "cv_strategy": f"StratifiedGroupKFold({NFOLDS},group=session,seed={SEED})",
        "cv_fold_scores": fold_scores,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "per_class_f1": per_class,
        "worst_class": worst,
        "collapsed_classes": collapsed,
        "inference_ms_per_sample": round(ms, 3),
        "estimated_full_test_minutes": round(ms * 30000 / 1000 / 60, 2),
        "model_size_mb": round(size_mb, 1),
        "n_features": int(Xall.shape[1]),
        "n_dense_features": len(F.dense_feature_names()),
        "offline_compatible": True,
        "seed": SEED,
        "note": "PCA-filtered + 25 new engineered features (bigram/trigram/intent/phase/ratios)",
    }
    json.dump(log, open(os.path.join(ROOT, "train_log.json"), "w"), indent=2)
    print(f"\nDONE cv={cv_mean:.4f}+/-{cv_std:.4f} "
          f"infer={ms:.1f}ms/sample model={size_mb:.1f}MB "
          f"n_feat={Xall.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
