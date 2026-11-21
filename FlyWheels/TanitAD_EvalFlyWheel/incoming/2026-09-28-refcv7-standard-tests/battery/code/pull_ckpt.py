"""READ-ONLY, md5-verified pull of ONE refcv7 checkpoint from Thor (the live training box).

    python pull_ckpt.py --step 5000                    # milestone file first, rolling ckpt.pt as fallback
    python pull_ckpt.py --step 1500 --source rolling   # a non-milestone step (only ckpt.pt carries it)

Thor is TRAINING: this tool runs ONLY `ls`, `md5sum` and `cat`-free `scp` pulls there (brief: ls /
md5sum / cat / scp). No python, no GPU, no writes, no signals on Thor.

WHICH FILE (MEASURED from the launch source, not assumed): Thor's
`/home/nvidia/refcv7_run/fec3a0dccf/stack/scripts/refc_v3_train.py` (md5 0a6fb0d8…, = blob 203b437f
at launch commit fec3a0dc) writes
  * `ckpt.pt` every `--save-every 500` steps: {model, opt, step, data_pos}, OVERWRITTEN each time
    (`refc_v3_train.py:9634-9644`);
  * `ckpt_<step>.pt` for `step in MILESTONES = (5000, 15000, 20000, 30000)` (`:225`, `:9645-9647`):
    {model, step}, NEVER rotated. It is written right after `ckpt.pt` and BEFORE that step's
    in-run eval row (the save-before-eval order, `:9608-9633`).
So the milestone file is the primary source. The rolling `ckpt.pt` is the fallback and it holds
step N only while the run's last logged step is in [N, N + 500).

Every acceptance test is on the ARTIFACT (never on an exit code through a pipe):
  1. the file exists on Thor and its (size, mtime) is unchanged over >= 20 s;
  2. `metrics.jsonl` (pulled read-only) already carries the eval row for step N -- the trainer
     writes both checkpoint files BEFORE that row, so the row proves the save finished;
  3. remote md5 BEFORE and AFTER the copy, and the local md5, are one 32-char value;
  4. the local file is read with torch (mmap, CPU) and its `step` key must equal N. A file whose
     step is not N is REFUSED loudly (renamed `*.REFUSED_step<k>`), never stamped.
Markers: ZZPULLOK<N>ZZ / ZZPULLWAIT<N>ZZ / ZZPULLFAIL<N>ZZ, plus `--out-json` (the record).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SSH = "C:/Windows/System32/OpenSSH/ssh.exe"      # MSYS ssh deadlocks under subprocess pipes
SCP = "C:/Windows/System32/OpenSSH/scp.exe"
HOST = "tanitad-thor-wifi"
RUN = "/home/nvidia/refcv7_run/runs/refcv7-r101-s0"
MILESTONES = (5000, 15000, 20000, 30000)          # refc_v3_train.py:225 at fec3a0dc
SAVE_EVERY = 500                                  # the run's argv --save-every 500


def _ssh(cmd: str, timeout: int = 120) -> str:
    r = subprocess.run([SSH, "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, cmd],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    return r.stdout


def _scp(remote: str, local: str, timeout: int) -> bool:
    r = subprocess.run([SCP, "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                        f"{HOST}:{remote}", local], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0 and os.path.exists(local)


def stat_remote(path: str) -> tuple[int, int] | None:
    """(size, mtime_epoch) via `ls` (an allowed command); None if absent/unreadable."""
    out = _ssh(f"ls -l --time-style=+%s {path} 2>/dev/null").strip().split()
    if len(out) < 7:
        return None
    try:
        return int(out[4]), int(out[5])
    except ValueError:
        return None


def md5_remote(path: str) -> str:
    out = _ssh(f"md5sum {path}", timeout=600).strip().split()
    return out[0] if out and len(out[0]) == 32 else ""


def md5_local(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def metrics_state(local_copy: str, step: int) -> dict:
    rows = []
    for ln in open(local_copy, encoding="utf-8"):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue                     # a line being appended while we copied
    steps = [int(r["step"]) for r in rows if isinstance(r.get("step"), int)]
    ev = [r for r in rows if r.get("step") == step and "eval_loss" in r]
    err = [r for r in rows if r.get("step") == step and "eval_error" in r]
    return {"n_rows": len(rows), "last_step": max(steps) if steps else -1,
            "eval_row_at_step": len(ev), "eval_error_at_step": len(err)}


def local_step(p: str) -> tuple[int | None, list]:
    import torch
    ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    st = ck.get("step") if isinstance(ck, dict) else None
    keys = sorted(ck.keys()) if isinstance(ck, dict) else []
    del ck
    return (int(st) if st is not None else None), keys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--source", choices=("auto", "milestone", "rolling"), default="auto")
    ap.add_argument("--dst", default="D:/refcv7_eval_kit/ckpt")
    ap.add_argument("--thor-reads", default="D:/refcv7_eval_kit/thor_reads")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--stable-s", type=int, default=20)
    a = ap.parse_args()
    N = int(a.step)
    os.makedirs(a.dst, exist_ok=True)
    os.makedirs(a.thor_reads, exist_ok=True)
    rec = {"tool": "pull_ckpt.py", "step": N, "host": HOST, "run": RUN,
           "t_start": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def done(marker: str, msg: str, code: int) -> int:
        rec.update(marker=marker, message=msg, t_end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if a.out_json:
            json.dump(rec, open(a.out_json, "w", encoding="utf-8"), indent=1)
        print(f"{marker} {msg}", flush=True)
        return code

    ms = f"{RUN}/ckpt_{N}.pt"
    roll = f"{RUN}/ckpt.pt"
    s_ms = stat_remote(ms) if a.source in ("auto", "milestone") else None
    rec["milestone_stat"] = s_ms
    if s_ms is not None:
        src = ms
    elif a.source == "milestone":
        return done(f"ZZPULLWAIT{N}ZZ", "no-milestone-file", 2)
    else:
        src = roll
    rec["source_path"] = src
    rec["source_kind"] = "milestone ckpt_<N>.pt (model+step)" if src == ms else \
        "ROLLING ckpt.pt (model+opt+step+data_pos), fallback"
    # 2. the eval row (the save-finished proof) + the rolling window check
    ts = time.strftime("%Y%m%dT%H%M%S")
    # one rolling poll copy (a waiter polls every few minutes); a timestamped copy is kept only when
    # this pull PROCEEDS, and that copy is the one the record names for G0
    mpoll = os.path.join(a.thor_reads, f"metrics_poll_{N}.jsonl")
    if os.path.exists(mpoll):
        os.remove(mpoll)
    if not _scp(f"{RUN}/metrics.jsonl", mpoll, 300):
        return done(f"ZZPULLFAIL{N}ZZ", "metrics-scp", 3)
    mloc = mpoll
    msta = metrics_state(mloc, N)
    rec["metrics_copy"] = mloc
    rec["metrics_state"] = msta
    if msta["eval_row_at_step"] != 1:
        return done(f"ZZPULLWAIT{N}ZZ",
                    f"no-eval-row-yet (last step {msta['last_step']}, eval rows at N "
                    f"{msta['eval_row_at_step']}, eval_error rows {msta['eval_error_at_step']})", 2)
    import shutil
    mloc = os.path.join(a.thor_reads, f"metrics_{ts}.jsonl")
    shutil.copyfile(mpoll, mloc)
    rec["metrics_copy"] = mloc
    if src == roll and not (N <= msta["last_step"] < N + SAVE_EVERY):
        return done(f"ZZPULLFAIL{N}ZZ",
                    f"rolling ckpt.pt cannot hold step {N}: last logged step {msta['last_step']}", 4)
    # 1. stability
    s1 = stat_remote(src)
    time.sleep(a.stable_s)
    s2 = stat_remote(src)
    rec["stat_1"], rec["stat_2"] = s1, s2
    if s1 is None or s1 != s2:
        return done(f"ZZPULLWAIT{N}ZZ", f"size/mtime-moving {s1} {s2}", 2)
    # 3. md5 before, copy, md5 after, local md5
    m_before = md5_remote(src)
    part = os.path.join(a.dst, f"ckpt_{N}.pt.part")
    t0 = time.time()
    ok = _scp(src, part, 3600)
    rec["scp_s"] = round(time.time() - t0, 1)
    m_after = md5_remote(src)
    s3 = stat_remote(src)
    if not ok:
        return done(f"ZZPULLFAIL{N}ZZ", "scp", 3)
    m_local = md5_local(part)
    rec.update(md5_remote_before=m_before, md5_remote_after=m_after, md5_local=m_local,
               stat_after=s3, size_local=os.path.getsize(part))
    if not (len(m_local) == 32 and m_before == m_after == m_local and s3 == s2
            and os.path.getsize(part) == s2[0]):
        os.replace(part, part + ".MISMATCH")
        return done(f"ZZPULLFAIL{N}ZZ",
                    f"md5/size-mismatch before={m_before} after={m_after} local={m_local} "
                    f"stat {s2}->{s3}", 3)
    # 4. the step key, read LOCALLY
    try:
        st, keys = local_step(part)
    except Exception as exc:                      # noqa: BLE001 -- recorded
        os.replace(part, part + ".UNREADABLE")
        return done(f"ZZPULLFAIL{N}ZZ", f"torch.load failed: {type(exc).__name__}: {exc}", 3)
    rec["ckpt_step_key"], rec["ckpt_keys"] = st, keys
    if st != N:
        os.replace(part, os.path.join(a.dst, f"ckpt_{N}.pt.REFUSED_step{st}"))
        return done(f"ZZPULLFAIL{N}ZZ", f"REFUSED: the file's step key is {st}, not {N}", 5)
    final = os.path.join(a.dst, f"ckpt_{N}.pt")
    os.replace(part, final)
    with open(os.path.join(a.dst, "MD5SUMS"), "a", encoding="utf-8") as f:
        f.write(f"{m_local}  ckpt_{N}.pt  # {rec['source_kind']}; step key {st}; "
                f"pulled {rec['t_start']}\n")
    rec["local_path"] = final
    return done(f"ZZPULLOK{N}ZZ", f"md5 {m_local} size {s2[0]} step {st} src "
                f"{os.path.basename(src)} metrics {os.path.basename(mloc)}", 0)


if __name__ == "__main__":
    sys.exit(main())
