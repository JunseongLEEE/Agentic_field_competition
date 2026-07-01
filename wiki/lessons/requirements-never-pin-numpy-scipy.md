---
id: requirements-never-pin-numpy-scipy
type: lesson
created: 2026-07-01
updated: 2026-07-01
tags: [submission, packaging, requirements, abi, offline, numpy]
related: [[data-is-jsonl-not-csv]], [[subagent-must-run-training-foreground]]
summary: Pinning numpy/scipy/pandas in requirements.txt crashes the DACON server with "numpy.dtype size changed" (ABI mismatch); pin only scikit-learn (to match the pickle) + the model lib (lightgbm).
---

# requirements.txt must NOT pin numpy/scipy/pandas

## Symptom
First real submission (exp_001_tfidf_lightgbm) failed on the DACON eval server at import time:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject   (File: /app/script.py, Line: 9)
```
Line 9 was a C-extension import (pandas/scipy). Local runs were fine — only the server crashed.

## Root cause
`requirements.txt` pinned `numpy==2.2.6`, `scipy==1.14.0`, `scikit-learn==1.5.1`, `joblib`, `pandas`.
The DACON server already ships a mutually-consistent scientific stack (Python 3.11, numpy 2.x, scipy, scikit-learn 1.8.0, pandas). Forcing specific numpy/scipy versions over that base left a C-extension compiled against one numpy ABI running against a different numpy at runtime → "dtype size changed" (96 = numpy 2.x dtype struct, 88 = numpy 1.x). The working baseline_submit.zip pinned only `scikit-learn` + `joblib` — no numpy/scipy.

## Fix
- Pin ONLY what is needed for pickle/model compatibility: `scikit-learn==<the version the artifacts were trained with>` and the model lib (`lightgbm==4.6.0`). Let numpy/scipy/joblib come transitively from the server's consistent stack — do NOT list them.
- scikit-learn wheels are built oldest-supported-numpy → forward-compatible with the server's numpy, so pinning sklearn does not force a numpy change.
- Prefer stdlib `csv`/`json` over `pandas` in script.py to drop one more C-extension import.
- Validate before submitting: fresh `python -m venv`, `pip install` ONLY the requirements, run script.py on data/test.jsonl. If numpy/scipy resolve consistently and it runs, the pins are safe. (Local dev-env dry-run does NOT catch this — it must be a clean install.)
- Local Python here is 3.10; server is 3.11+ (server sklearn 1.8.0 needs ≥3.11). Pin sklearn to a version installable on BOTH (e.g. 1.5.1) so the pickle matches exactly and it installs on the server.

## Generalization
For offline code-submission competitions: keep requirements.txt minimal and never re-pin the base scientific stack (numpy/scipy/pandas/joblib). Pin only the estimator libraries whose pickles you must load, at versions that install on the server's Python. Always test with a clean venv, not the training env.
