"""실험 결과 구조화 기록 + 제출-CV 연동 로그.

사용:
  python src/explog.py record --exp qwen_r16_ep1 --val_em 0.62 --config "lora r16 ep1"
  python src/explog.py submit_log --exp qwen_r16_ep1 --file sub_qwen.csv --val_em 0.62
  python src/explog.py sync   # kaggle 제출 이력에서 LB 점수 당겨와 갭 계산
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results.csv")
SUBLOG = os.path.join(ROOT, "submissions", "log.csv")
FIELDS_R = ["ts", "exp", "config", "val_em", "notes"]
FIELDS_S = ["ts", "exp", "file", "val_em", "lb_public", "gap"]


def _append(path, fields, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def record(args):
    _append(RESULTS, FIELDS_R, {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exp": args.exp, "config": args.config,
        "val_em": args.val_em, "notes": args.notes,
    })
    print("recorded:", args.exp, args.val_em)


def submit_log(args):
    _append(SUBLOG, FIELDS_S, {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "exp": args.exp, "file": os.path.basename(args.file),
        "val_em": args.val_em, "lb_public": "", "gap": "",
    })
    print("submission logged:", args.file)


def sync(args):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from submit import _env
    r = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", "snuaichallenge", "--csv"],
        env=_env(), capture_output=True, text=True,
    )
    lb = {}  # fileName -> publicScore (최신 우선)
    rows = list(csv.DictReader(r.stdout.strip().splitlines()))
    for row in rows:
        fn = row.get("fileName", "")
        if fn and fn not in lb and row.get("publicScore"):
            lb[fn] = row["publicScore"]

    if not os.path.exists(SUBLOG):
        print("no submission log")
        return
    out = list(csv.DictReader(open(SUBLOG)))
    for row in out:
        if row["file"] in lb:
            row["lb_public"] = lb[row["file"]]
            if row["val_em"]:
                row["gap"] = f"{float(row['val_em']) - float(lb[row['file']]):.4f}"
    with open(SUBLOG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS_S)
        w.writeheader()
        w.writerows(out)
    print(open(SUBLOG).read())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("record")
    p1.add_argument("--exp", required=True)
    p1.add_argument("--config", default="")
    p1.add_argument("--val_em", required=True)
    p1.add_argument("--notes", default="")
    p2 = sub.add_parser("submit_log")
    p2.add_argument("--exp", required=True)
    p2.add_argument("--file", required=True)
    p2.add_argument("--val_em", default="")
    p3 = sub.add_parser("sync")
    ap.set_defaults()
    args = ap.parse_args()
    {"record": record, "submit_log": submit_log, "sync": sync}[args.cmd](args)
