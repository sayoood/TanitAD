"""APPEND-ONLY assembly of the growing training bank from the data-prep shards (train.py --grow).

The trainer's --grow mode re-reads its bank at every epoch boundary up to a BYTE SNAPSHOT, which is
only a complete description of a bank if its files are never rewritten. So this never rewrites:
each pass reads the keys already in the training files and APPENDS the shard rows that are new,
whole lines, one flushed + fsynced write per file.

  targets_rank0.jsonl          <- rank-0 teacher rollouts        (key log_name, token, step)
  targets_rank1.jsonl          <- goal-augmented twins, rank 1   (only when the scene's rank-0 row
                                                                   is already in the bank)
  scorer_targets.jsonl         <- rank-0 PDM candidate rows      (whole FRAMES only; each shard
  scorer_targets_rank1.jsonl   <- rank-1 PDM candidate rows       file's LAST frame is left for the
                                                                   next pass -- it may be half written)
  calib_table.json             <- copied once

⛔ A row is appended only if it parses and has the fields the trainer indexes (image x4, ego x7,
goal, traj 20x3 / candidate traj + targets); anything else is COUNTED and named, never guessed at.
A torn tail in a shard is simply not a line yet. A torn tail in an OUTPUT file (a kill mid-write)
is terminated first, so the next append cannot fuse onto it.

  python grow_assemble.py --bank /workspace/data/refe_navtrain --out /workspace/data/refe_navtrain/train_grow \
      [--loop-min 30]   # repeat until <out>/STOP exists
Prints ZZASSEMBLE_OK <counts> per pass.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time


def _terminate_torn(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            torn = f.read(1) != b"\n"
        if torn:
            with open(path, "ab") as f:
                f.write(b"\n")


def _rows(path):
    """Parsed complete rows of a file (a partial last line is not yet a row)."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        buf = f.read()
    cut = buf.rfind(b"\n")
    if cut < 0:
        return out
    for line in buf[:cut + 1].decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            out.append(None)                    # a fused/torn line: counted by the caller
    return out


def _append(path, rows):
    if not rows:
        return
    _terminate_torn(path)
    data = "".join(json.dumps(r) + "\n" for r in rows)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def _target_ok(r):
    try:
        return (isinstance(r.get("image"), list) and len(r["image"]) == 4 and len(r["ego"]) == 7
                and len(r["goal"]) >= 2 and len(r["traj"]) == 20 and len(r["traj"][0]) == 3
                and r.get("log_name") and r.get("token") is not None and "step" in r)
    except Exception:
        return False


def _scorer_ok(r):
    try:
        return (r.get("candidate") and isinstance(r.get("targets"), dict) and len(r["traj"]) >= 1
                and r.get("log_name") and r.get("token") is not None and "step" in r)
    except Exception:
        return False


def tkey(r):
    return (r.get("log_name"), r.get("token"), int(r.get("step", 0)))


def fkey(r):
    return (r.get("log_name"), r.get("token"), int(r.get("step", 0)), int(r.get("rank", 0)))


def one_pass(a) -> dict:
    os.makedirs(a.out, exist_ok=True)
    st = {"r0_new": 0, "r1_new": 0, "sc0_frames_new": 0, "sc1_frames_new": 0,
          "rejected": 0, "unparseable": 0, "r1_orphan_waiting": 0}
    # ---- targets ----------------------------------------------------------------------------------
    t0p, t1p = os.path.join(a.out, "targets_rank0.jsonl"), os.path.join(a.out, "targets_rank1.jsonl")
    have0 = {tkey(r) for r in _rows(t0p) if r}
    have1 = {tkey(r) for r in _rows(t1p) if r}
    new0 = []
    for f in sorted(glob.glob(os.path.join(a.bank, a.r0_glob))):
        for r in _rows(f):
            if r is None:
                st["unparseable"] += 1
            elif not _target_ok(r) or int(r.get("rank", 0)) != 0:
                st["rejected"] += 1
            elif tkey(r) not in have0:
                have0.add(tkey(r)); new0.append(r)
    _append(t0p, new0)
    st["r0_new"] = len(new0)
    new1 = []
    for f in sorted(glob.glob(os.path.join(a.bank, a.aug_glob))):
        for r in _rows(f):
            if r is None:
                st["unparseable"] += 1
            elif not _target_ok(r) or int(r.get("rank", 0)) != 1:
                st["rejected"] += 1
            elif tkey(r) in have1:
                continue
            elif tkey(r) not in have0:
                st["r1_orphan_waiting"] += 1      # its rank-0 row has not been assembled yet
            else:
                have1.add(tkey(r)); new1.append(r)
    _append(t1p, new1)
    st["r1_new"] = len(new1)
    # ---- scorer: whole frames only -----------------------------------------------------------------
    for tag, pattern, name in (("sc0", a.sc0_glob, "scorer_targets.jsonl"),
                               ("sc1", a.sc1_glob, "scorer_targets_rank1.jsonl")):
        outp = os.path.join(a.out, name)
        have = {fkey(r) for r in _rows(outp) if r}
        add = []
        for f in sorted(glob.glob(os.path.join(a.bank, pattern))):
            frames, order = {}, []
            for r in _rows(f):
                if r is None:
                    st["unparseable"] += 1
                    continue
                if not _scorer_ok(r):
                    st["rejected"] += 1
                    continue
                k = fkey(r)
                if k not in frames:
                    frames[k] = []
                    order.append(k)
                frames[k].append(r)
            # the LAST frame of a shard file may still be being written: leave it for the next pass
            for k in order[:-1]:
                if k not in have:
                    have.add(k); add.extend(frames[k])
                    st[f"{tag}_frames_new"] += 1
        _append(outp, add)
    # ---- calibration -------------------------------------------------------------------------------
    cal_src, cal_dst = os.path.join(a.bank, "calib_table.json"), os.path.join(a.out, "calib_table.json")
    if os.path.exists(cal_src) and not os.path.exists(cal_dst):
        shutil.copyfile(cal_src, cal_dst + ".tmp")
        os.replace(cal_dst + ".tmp", cal_dst)
    st["totals"] = {"rank0": len(have0), "rank1": len(have1)}
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True, help="the data-prep bank dir holding the shard dirs")
    ap.add_argument("--out", required=True, help="the --grow training dir (append-only)")
    ap.add_argument("--r0-glob", default="r0_s*/targets_rank0.jsonl")
    ap.add_argument("--aug-glob", default="aug_s*/targets_aug.jsonl")
    ap.add_argument("--sc0-glob", default="sc_r0_s*/scorer_targets.jsonl")
    ap.add_argument("--sc1-glob", default="sc_aug_s*/scorer_targets_rank1.jsonl")
    ap.add_argument("--loop-min", type=float, default=0.0,
                    help="repeat every N minutes until <out>/STOP exists (0 = one pass)")
    a = ap.parse_args()
    while True:
        t = time.time()
        st = one_pass(a)
        print(f"ZZASSEMBLE_OK {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
              f"{json.dumps(st)} {time.time() - t:.1f}s", flush=True)
        if a.loop_min <= 0 or os.path.exists(os.path.join(a.out, "STOP")):
            return 0
        time.sleep(a.loop_min * 60.0)


if __name__ == "__main__":
    sys.exit(main())
