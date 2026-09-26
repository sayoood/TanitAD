"""The dev box's post-rank-0 chain: merge -> augmentation -> rank-0 scorer -> augmented scorer.

Each stage runs through `devbox_shards.py` (no console anywhere -- the forrtl window-close lesson)
and the next starts only when the previous one's `<name>.done` exists, i.e. EVERY shard exited 0;
a supervisor that gave up on a shard leaves no .done and the chain stops with a marker instead of
building on a partial bank. Local output names match the pod's globs once `devbox_upload.py` adds
its `_DEV10` suffix: `aug_s*` / `sc_r0_s*` / `sc_aug_s*`.

  pythonw code/devbox_chain.py --root D:/Projects/TanitAD/data/refe_navtrain10 --logs-file <dev logs> \
      --wait-done r0_dev10d --shards 8
Progress: `<root>/devbox_logs/chain.log` (ZZCHAIN_* markers).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CREATE_NO_WINDOW = 0x08000000
HERE = os.path.dirname(os.path.abspath(__file__))
REFE = os.path.join(os.path.dirname(HERE), "refe")


def log(root, msg):
    with open(os.path.join(root, "devbox_logs", "chain.log"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")


def wait_done(root, name, poll=60):
    """Block until <name>.done; return False if the supervisor finished WITHOUT it (a gave-up shard)."""
    d = os.path.join(root, "devbox_logs")
    while True:
        if os.path.exists(os.path.join(d, f"{name}.done")):
            return True
        st = os.path.join(d, f"{name}.status")
        if os.path.exists(st):
            try:
                s = json.load(open(st, encoding="utf-8"))
                if not s.get("running") and s.get("finished") and any(v != 0 for v in s["finished"].values()):
                    return False
            except Exception:
                pass
        time.sleep(poll)


def run_stage(a, name, cmd):
    pyw = os.path.join(os.path.dirname(a.python), "pythonw.exe")
    sup = os.path.join(HERE, "devbox_shards.py")
    logs = os.path.join(a.root, "devbox_logs")
    full = [pyw, sup, "--name", name, "--shards", str(a.shards), "--threads", "1", "--cwd", REFE,
            "--log-dir", logs, "--env-file", os.path.join(logs, "env10.json"), "--"] + cmd
    log(a.root, f"ZZCHAIN_STAGE_START {name}")
    rc = subprocess.call(full, creationflags=CREATE_NO_WINDOW, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = os.path.exists(os.path.join(logs, f"{name}.done"))
    log(a.root, f"ZZCHAIN_STAGE_END {name} rc={rc} done={ok}")
    return ok


def merge(a, pattern, out, key):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rc = subprocess.call([a.python, os.path.join(HERE, "merge_bank.py"), "--key", key, "--out", out,
                          os.path.join(a.root, pattern)], creationflags=CREATE_NO_WINDOW,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    n = sum(1 for _ in open(out, "rb")) if os.path.exists(out) else 0
    log(a.root, f"ZZCHAIN_MERGE {pattern} -> {out} rc={rc} rows={n}")
    return rc == 0 and n > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--logs-file", required=True)
    ap.add_argument("--wait-done", required=True, help="the rank-0 supervisor name to wait for")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--tau", default="0.3")
    ap.add_argument("--python", default=sys.executable.replace("pythonw.exe", "python.exe"))
    a = ap.parse_args()
    R = a.root
    log(R, f"ZZCHAIN_START waiting for {a.wait_done}.done")
    if not wait_done(R, a.wait_done):
        log(R, f"ZZCHAIN_ABORT {a.wait_done} finished without .done (a shard gave up)")
        return 1
    r0 = os.path.join(R, "r0", "targets_rank0.jsonl")
    if not merge(a, "r0_s*/targets_rank0.jsonl", r0, "log_name,token,step,rank"):
        log(R, "ZZCHAIN_ABORT rank-0 merge"); return 1
    py = a.python
    if not run_stage(a, "aug_dev10", [py, "augment_search.py", "--bank", os.path.join(R, "r0"),
                                      "--out", os.path.join(R, "aug_s@I@"), "--tau", a.tau, "--resume",
                                      "--resume-glob", os.path.join(R, "aug_s*"), "--device", "cpu",
                                      "--logs-file", a.logs_file]):
        log(R, "ZZCHAIN_ABORT aug"); return 1
    aug = os.path.join(R, "aug", "targets_aug.jsonl")
    merge(a, "aug_s*/targets_aug.jsonl", aug, "log_name,token,step,rank")
    if not run_stage(a, "sc_r0_dev10", [py, "build_scorer_targets.py", "--source", "navtrain",
                                        "--perframe-bank", os.path.join(R, "r0"),
                                        "--out", os.path.join(R, "sc_r0_s@I@"), "--rank", "0",
                                        "--frame-stride", "1", "--resume", "--resume-glob",
                                        os.path.join(R, "sc_r0_s*", "scorer_targets.jsonl"),
                                        "--logs-file", a.logs_file]):
        log(R, "ZZCHAIN_ABORT sc_r0"); return 1
    if os.path.exists(aug):
        if not run_stage(a, "sc_aug_dev10", [py, "build_scorer_targets.py", "--source", "navtrain",
                                             "--perframe-bank", os.path.join(R, "aug"),
                                             "--perframe-file", aug, "--out",
                                             os.path.join(R, "sc_aug_s@I@"), "--rank", "1",
                                             "--frame-stride", "1", "--resume", "--resume-glob",
                                             os.path.join(R, "sc_aug_s*", "scorer_targets_rank1.jsonl"),
                                             "--logs-file", a.logs_file]):
            log(R, "ZZCHAIN_ABORT sc_aug"); return 1
    log(R, "ZZCHAIN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
