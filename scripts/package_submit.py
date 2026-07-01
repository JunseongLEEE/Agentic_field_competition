#!/usr/bin/env python3
"""Package experiment as DACON code submission zip.

DACON format:
    submit.zip
    ├── model/           # Trained model weights
    ├── script.py        # Inference-only code
    └── requirements.txt # Extra packages
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def package(exp_path: Path):
    """Create DACON submission zip from experiment."""
    exp_path = Path(exp_path).resolve()
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    # Validate required files
    errors = []

    script_path = exp_path / "script.py"
    model_dir = exp_path / "model"
    req_path = exp_path / "requirements.txt"

    if not script_path.exists():
        errors.append("script.py not found")
    if not model_dir.exists() or not any(model_dir.iterdir()):
        errors.append("model/ directory is empty or missing")
    if not req_path.exists():
        errors.append("requirements.txt not found")

    if errors:
        print("PACKAGING FAILED — missing required files:")
        for e in errors:
            print(f"  [X] {e}")
        sys.exit(1)

    # Offline check on script.py
    import re
    script_content = script_path.read_text()
    online_patterns = [
        (r'from_pretrained\s*\(\s*["\'](?![\./])', "HuggingFace Hub download"),
        (r'requests\.(get|post)', "HTTP request"),
        (r'urllib\.request', "urllib download"),
        (r'api_key|API_KEY|openai\.', "API key usage"),
    ]
    for pattern, desc in online_patterns:
        if re.search(pattern, script_content):
            print(f"OFFLINE VIOLATION: {desc}")
            print("Fix script.py to use only local files, then retry.")
            sys.exit(1)

    # Model size
    model_size_mb = get_dir_size_mb(model_dir)
    print(f"Model size: {model_size_mb:.1f} MB")

    # Create zip
    exp_name = exp_path.name
    zip_path = SUBMISSIONS_DIR / f"{exp_name}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add script.py
        zf.write(script_path, "script.py")

        # Add requirements.txt
        zf.write(req_path, "requirements.txt")

        # Add helper modules script.py may import (must live at zip root next to script.py)
        for extra in ("features.py", "model.py"):
            p = exp_path / extra
            if p.exists():
                zf.write(p, extra)

        # Add model/ directory
        for model_file in sorted(model_dir.rglob("*")):
            if model_file.is_file():
                arcname = str(model_file.relative_to(exp_path))
                zf.write(model_file, arcname)

    # Enforce 1GB DACON hard cap
    zip_mb = zip_path.stat().st_size / (1024 * 1024)
    if zip_mb > 1024:
        zip_path.unlink()
        print(f"PACKAGING FAILED: zip {zip_mb:.0f} MB exceeds 1 GB cap")
        sys.exit(2)

    # Hash
    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()

    # Load CV score
    train_log_path = exp_path / "train_log.json"
    cv_score = None
    inference_speed = None
    if train_log_path.exists():
        with open(train_log_path) as f:
            results = json.load(f)
        cv_score = results.get("cv_mean")
        inference_speed = results.get("inference_ms_per_sample")

    # Metadata
    meta = {
        "experiment_id": exp_name,
        "cv_score": cv_score,
        "inference_ms_per_sample": inference_speed,
        "model_size_mb": round(model_size_mb, 2),
        "git_commit": get_git_commit(),
        "sha256": sha256,
        "created_at": datetime.now().isoformat(),
        "zip_size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
        "offline_verified": True,
    }
    meta_path = SUBMISSIONS_DIR / f"{exp_name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Report
    print(f"\nSubmission packaged (DACON format):")
    print(f"  Zip:    {zip_path}")
    print(f"  Size:   {meta['zip_size_mb']} MB")
    print(f"  Model:  {model_size_mb:.1f} MB")
    print(f"  SHA256: {sha256[:16]}...")
    print(f"  CV:     {cv_score}")
    print(f"  Speed:  {inference_speed} ms/sample")
    print(f"\n  Contents: model/ + script.py + requirements.txt")
    print(f"\n  ⚠️  DACON 사이트에서 수동 업로드 필요")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package DACON code submission")
    parser.add_argument("--exp", required=True, help="Path to experiment directory")
    args = parser.parse_args()

    package(Path(args.exp))
