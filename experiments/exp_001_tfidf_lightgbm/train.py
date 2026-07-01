"""exp_001 LightGBM — word TF-IDF + structured features (char dropped for speed).

CV = StratifiedGroupKFold(3, group=session). Reuses features.py.
Writes model/ (word_vec, cat_mappings, lgbm, class_order), oof_preds.npy, train_log.json.
"""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "16")
import numpy as np
import joblib
from scipy import sparse

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import features as F  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
import lightgbm as lgb  # noqa: E402

DATA = os.path.join(ROOT, "..", "..", "data")
MODEL = os.path.join(ROOT, "model")
os.makedirs(MODEL, exist_ok=True)
SEED, NJOBS = 42, 16


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
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                           max_features=20000, min_df=2, sublinear_tf=True)


def model():
    return lgb.LGBMClassifier(objective="multiclass", num_class=14, n_estimators=350,
                              learning_rate=0.08, num_leaves=63, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                              reg_lambda=1.0, n_jobs=NJOBS, seed=SEED, verbose=-1)


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

    skf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 14), np.float32)
    fold_scores = []
    for k, (tr, va) in enumerate(skf.split(prompts, y, groups)):
        assert not (set(groups[tr]) & set(groups[va])), "group leak"
        wv = word_vec(); wv.fit(sub(prompts, tr))
        cat_map = F.build_cat_mappings(sub(records, tr))
        Xtr = feats(wv, cat_map, sub(prompts, tr), sub(records, tr))
        Xva = feats(wv, cat_map, sub(prompts, va), sub(records, va))
        sw = compute_sample_weight("balanced", y[tr])
        m = model(); m.fit(Xtr, y[tr], sample_weight=sw)
        p = m.predict_proba(Xva); oof[va] = p
        sc = f1_score(y[va], p.argmax(1), average="macro")
        fold_scores.append(float(sc))
        print(f"fold{k} macroF1={sc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    cv_mean, cv_std = float(np.mean(fold_scores)), float(np.std(fold_scores))
    per = f1_score(y, oof.argmax(1), average=None, labels=list(range(14)))
    per_class = {F.CLASS_ORDER[i]: float(per[i]) for i in range(14)}
    worst = min(per_class, key=per_class.get)
    collapsed = [c for c, v in per_class.items() if v < 0.05]
    print(f"CV macroF1 = {cv_mean:.4f} +/- {cv_std:.4f}", flush=True)

    print("full-train ...", flush=True)
    wv = word_vec(); wv.fit(prompts)
    cat_map = F.build_cat_mappings(records)
    Xall = feats(wv, cat_map, prompts, records)
    sw = compute_sample_weight("balanced", y)
    m = model(); m.fit(Xall, y, sample_weight=sw)

    joblib.dump(wv, os.path.join(MODEL, "word_vec.pkl"))
    json.dump(cat_map, open(os.path.join(MODEL, "cat_mappings.json"), "w"))
    joblib.dump(m, os.path.join(MODEL, "lgbm.pkl"))
    json.dump(F.CLASS_ORDER, open(os.path.join(MODEL, "class_order.json"), "w"))
    np.save(os.path.join(ROOT, "oof_preds.npy"), oof)

    n = min(2000, len(prompts))
    ts = time.time()
    Xs = feats(wv, cat_map, sub(prompts, range(n)), sub(records, range(n)))
    _ = m.predict_proba(Xs)
    ms = (time.time() - ts) / n * 1000
    size_mb = sum(os.path.getsize(os.path.join(MODEL, f)) for f in os.listdir(MODEL)) / 1048576

    log = {
        "experiment_id": "exp_001_tfidf_lightgbm", "metric": "macro_f1",
        "cv_strategy": "StratifiedGroupKFold(3,group=session,seed=42)",
        "cv_fold_scores": fold_scores, "cv_mean": cv_mean, "cv_std": cv_std,
        "per_class_f1": per_class, "worst_class": worst, "collapsed_classes": collapsed,
        "inference_ms_per_sample": round(ms, 3),
        "estimated_full_test_minutes": round(ms * 30000 / 1000 / 60, 2),
        "model_size_mb": round(size_mb, 1), "n_features": int(Xall.shape[1]),
        "offline_compatible": True, "seed": SEED, "git_commit": "local",
        "note": "word tfidf(1,2)+structured; char_wb dropped for speed",
    }
    json.dump(log, open(os.path.join(ROOT, "train_log.json"), "w"), indent=2)
    print("DONE " + json.dumps({k: log[k] for k in
          ("cv_mean", "cv_std", "worst_class", "estimated_full_test_minutes", "model_size_mb")}),
          flush=True)


if __name__ == "__main__":
    main()
