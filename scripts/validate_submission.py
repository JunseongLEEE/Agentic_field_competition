#!/usr/bin/env python3
"""Validate a DACON code submission zip file before manual upload.

DACON required structure:
    submit.zip
    ├── model/              # Trained model weights
    ├── script.py           # Inference-only code
    └── requirements.txt    # Extra packages

Checks:
    1. Zip structure (exactly 3 top-level entries)
    2. script.py exists and has correct structure
    3. model/ directory exists and is not empty
    4. requirements.txt exists
    5. Offline compatibility (no internet calls in script.py)
    6. script.py reads from data/, writes to output/submission.csv
"""

import argparse
import re
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Patterns that indicate online dependency (BLOCKED)
ONLINE_PATTERNS = [
    (r'from_pretrained\s*\(\s*["\'](?![\./])', "HuggingFace Hub download (use local path instead)"),
    (r'requests\.(get|post|put|delete)\s*\(', "HTTP request"),
    (r'urllib\.(request|parse)', "urllib network call"),
    (r'wget\.', "wget download"),
    (r'download\s*\(', "download() call"),
    (r'api_key|API_KEY|OPENAI|openai\.', "API key / OpenAI usage"),
    (r'anthropic\.', "Anthropic API call"),
    (r'huggingface_hub', "HuggingFace Hub import"),
]

# Patterns that script.py SHOULD have
REQUIRED_PATTERNS = [
    (r"os\.path\.join\s*\(\s*['\"]data['\"]|['\"]data/|data_path", "Reads from data/ directory"),
    (r"output/submission\.csv|os\.path\.join\s*\(\s*['\"]output['\"]", "Writes to output/submission.csv"),
    (r"if\s+__name__\s*==\s*['\"]__main__['\"]", "__main__ block"),
    (r"os\.path\.join\s*\(\s*['\"]model['\"]|['\"]model/", "Loads model from model/ directory"),
]


def validate(zip_path: Path):
    """Validate DACON submission zip."""
    zip_path = Path(zip_path).resolve()

    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found")
        sys.exit(1)

    print(f"Validating DACON submission: {zip_path.name}")
    print(f"{'=' * 60}")

    errors = []
    warnings = []

    # === 1. Zip structure ===
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
    except zipfile.BadZipFile:
        errors.append("Invalid zip file")
        _print_results(errors, warnings)
        sys.exit(1)

    print(f"\nContents ({len(names)} entries):")
    for n in names[:20]:
        print(f"  {n}")
    if len(names) > 20:
        print(f"  ... and {len(names) - 20} more")

    # Check top-level entries
    top_level = set()
    for n in names:
        parts = n.split("/")
        top_level.add(parts[0])

    print(f"\nTop-level entries: {sorted(top_level)}")

    # Required entries
    if "script.py" not in names:
        errors.append("MISSING: script.py not found in zip")

    if "requirements.txt" not in names:
        errors.append("MISSING: requirements.txt not found in zip")

    model_files = [n for n in names if n.startswith("model/") and not n.endswith("/")]
    if not model_files:
        errors.append("MISSING: model/ directory is empty or missing")
    else:
        total_model_size = sum(
            zipfile.ZipFile(zip_path).getinfo(f).file_size for f in model_files
        )
        model_size_mb = total_model_size / (1024 * 1024)
        print(f"\nModel files: {len(model_files)} files, {model_size_mb:.1f} MB")
        if model_size_mb > 800:
            warnings.append(f"Model size is {model_size_mb:.0f} MB — leaves <200MB for other zip contents (zip limit 1GB)")

    # Check for unexpected top-level entries
    allowed_top = {"model", "script.py", "requirements.txt"}
    # model/ entries show up as "model" in top_level
    unexpected = top_level - allowed_top - {"model"}
    # Also filter out __MACOSX and similar
    unexpected = {u for u in unexpected if not u.startswith("__") and not u.startswith(".")}
    if unexpected:
        warnings.append(f"Unexpected top-level entries: {unexpected} — DACON expects only model/, script.py, requirements.txt")

    # === 2. script.py validation ===
    if "script.py" in names:
        with zipfile.ZipFile(zip_path) as zf:
            script_content = zf.read("script.py").decode("utf-8")

        print(f"\nscript.py: {len(script_content)} bytes, {len(script_content.splitlines())} lines")

        # Offline check
        print("\nOffline compatibility check:")
        for pattern, desc in ONLINE_PATTERNS:
            matches = re.findall(pattern, script_content)
            if matches:
                errors.append(f"OFFLINE VIOLATION: {desc} (pattern: {pattern})")
                print(f"  [X] {desc}")
            else:
                print(f"  [OK] No {desc}")

        # Required structure check
        print("\nRequired structure check:")
        for pattern, desc in REQUIRED_PATTERNS:
            if re.search(pattern, script_content):
                print(f"  [OK] {desc}")
            else:
                warnings.append(f"script.py may be missing: {desc}")
                print(f"  [?] {desc} — not found (may be OK if implemented differently)")

    # === 3. requirements.txt ===
    if "requirements.txt" in names:
        with zipfile.ZipFile(zip_path) as zf:
            req_content = zf.read("requirements.txt").decode("utf-8").strip()

        if req_content:
            packages = [l.strip() for l in req_content.splitlines() if l.strip() and not l.startswith("#")]
            print(f"\nrequirements.txt: {len(packages)} packages")
            for p in packages:
                print(f"  - {p}")
        else:
            print("\nrequirements.txt: empty (using server defaults only)")

    # === 4. File size (DACON hard limit: 1GB) ===
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    zip_size_gb = zip_size_mb / 1024
    print(f"\nTotal zip size: {zip_size_mb:.1f} MB ({zip_size_gb:.3f} GB)")
    if zip_size_mb > 1024:
        errors.append(f"ZIP SIZE EXCEEDS 1GB LIMIT: {zip_size_mb:.1f} MB — DACON will reject")
    elif zip_size_mb > 900:
        warnings.append(f"Zip size {zip_size_mb:.0f} MB approaches 1GB limit")

    # === Results ===
    _print_results(errors, warnings)

    if errors:
        sys.exit(1)
    return True


def _print_results(errors, warnings):
    print(f"\n{'=' * 60}")
    if errors:
        print("ERRORS (must fix before submitting):")
        for e in errors:
            print(f"  [X] {e}")
    if warnings:
        print("WARNINGS (review before submitting):")
        for w in warnings:
            print(f"  [!] {w}")
    if not errors and not warnings:
        print("VALIDATION PASSED — ready to submit to DACON")
    elif not errors:
        print("VALIDATION PASSED (with warnings) — review warnings before submitting")
    else:
        print("VALIDATION FAILED — fix errors before submitting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate DACON code submission zip")
    parser.add_argument("--zip", required=True, help="Path to submission zip")
    args = parser.parse_args()

    validate(Path(args.zip))
