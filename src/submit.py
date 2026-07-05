"""Kaggle 제출 파이프라인.

사용법:
    python src/submit.py <submission.csv> "메시지"
    python src/submit.py --list          # 제출 이력 확인
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = "snuaichallenge"


def _env():
    env = dict(os.environ)
    # .env에서 자격증명 로드
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k == "Kaggle_USERNAME":
                    env["KAGGLE_USERNAME"] = v
                if k == "Kaggle_API_TOKEN":
                    env.setdefault("KAGGLE_KEY", v)
    # 최신 토큰 우선
    tok = os.path.join(os.path.expanduser("~"), ".kaggle", "access_token")
    if os.path.exists(tok):
        env["KAGGLE_KEY"] = open(tok).read().strip()
    return env


def submit(csv_path: str, message: str):
    import pandas as pd

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ["Id", "Answer"], f"잘못된 컬럼: {df.columns}"
    assert len(df) == 819, f"행 수 오류: {len(df)}"
    assert df["Answer"].str.match(r"^\[[1-4], [1-4], [1-4], [1-4]\]$").all()
    r = subprocess.run(
        ["kaggle", "competitions", "submit", "-c", COMP, "-f", csv_path, "-m", message],
        env=_env(), capture_output=True, text=True,
    )
    print(r.stdout or r.stderr)
    return r.returncode


def list_submissions():
    r = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", COMP],
        env=_env(), capture_output=True, text=True,
    )
    print(r.stdout or r.stderr)


if __name__ == "__main__":
    if sys.argv[1] == "--list":
        list_submissions()
    else:
        sys.exit(submit(sys.argv[1], sys.argv[2]))
