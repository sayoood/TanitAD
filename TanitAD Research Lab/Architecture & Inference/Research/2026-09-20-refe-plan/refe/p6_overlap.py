#!/usr/bin/env python3
"""Helpers for P6's OVERLAP mode (eval/P6_ON_11VCPU.md section 9): P6's CPU-only stages run BESIDE the P2 seed replicate's trainer, niced and capped, and back off when
the trainer slows.  Pure standard library (+ refe/p6_cpu_probe.py for the cgroup counters).  Every subcommand prints ONE opaque marker line and returns a meaningful rc.

  trainer-rate   --metrics F [--last 5] [--max-age-s 900]    the running trainer's median seconds per optimiser step over its last N steps, from its metrics.jsonl
                 ZZTRAINER_RATE step_s=<s> n=<n> last_step=<k>   rc 0   |   ZZTRAINER_RATE unknown <why>   rc 2   (stale file, fewer than 3 steps, unreadable)
  trainer-guard  --metrics F --base B [--tol 0.05] [--last 5] the same reading against a BASELINE taken before P6 put any load on the pod
                 ZZTRAINER_GUARD ok|slow|unknown ratio=<cur/base> cur=<s> base=<s> tol=<t>     rc 0 ok | 1 slow (ratio > 1 + tol) | 2 unknown
  overlap-cap    --cores C --max M [--secs 60 --margin 1.0 --root /sys/fs/cgroup]
                 how many pool workers a CPU-only stage may use NOW: floor(C - the cgroup's MEASURED CPU burn over --secs - margin), clamped to [1, M].  The cgroup is the pod's
                 one container, so the burn IS the running trainer's (+ its loaders, + this chain's watcher).  Unreadable counters => floor(C / 2) (UNVERIFIED, said so).
                 ZZOVERLAP_CAP <n> burn=<cores> cores=<C> margin=<x> max=<M> src=census|fallback:<why>
  lab-bar        --json lab.json            did a G-LIVE-LAB instrument run already clear the gate's own bar (identity, stale refusal, throughput >= 1.5 x 4.49)?
                 ZZLAB_BAR ok|below|bad thr=<sets/s> bar=<sets/s> ...    rc 0 ok | 1 below | 2 unreadable

Tests: refe/test_p6_overlap.py (ZZTEST_PASS n).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

NEED_SETS_S, MARGIN = 4.49, 1.5                # gate_checks.LAB_NEED_SETS_S / LAB_MARGIN (pinned by test_p6_overlap.py)
BAR_GATE = round(NEED_SETS_S * MARGIN, 3)      # 6.735


# ------------------------------------------------------------------------------------------------ the running trainer
def _tail_lines(path, max_bytes=262144):
    p = Path(path)
    with open(p, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        data = fh.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[1:] if size > max_bytes and lines else lines          # the first line of a tail window may be cut


def step_times(path, last=5):
    """Per-optimiser-step seconds from a train.py metrics.jsonl: the rows that carry `traj_L1` and `elapsed_s` (event rows do not), consecutive pairs only, a step counter that goes
    backwards or an elapsed_s that does not advance (a resume) is skipped, never averaged in.  -> (list of the last `last` per-step seconds, last step)."""
    rows = []
    for ln in _tail_lines(path):
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue                                                     # a torn last line
        if "traj_L1" in r and "elapsed_s" in r and "step" in r:
            try:
                rows.append((int(r["step"]), float(r["elapsed_s"])))
            except (TypeError, ValueError):
                continue
    dts = []
    for (s0, t0), (s1, t1) in zip(rows, rows[1:]):
        if s1 > s0 and t1 > t0:
            dts.append((t1 - t0) / (s1 - s0))
    return dts[-last:], (rows[-1][0] if rows else None)


def trainer_rate(path, last=5, max_age_s=900.0, now=None):
    """-> (step_s | None, n, last_step, why)."""
    p = Path(path)
    try:
        age = (now if now is not None else time.time()) - p.stat().st_mtime
    except OSError as e:
        return None, 0, None, f"unreadable: {type(e).__name__}"
    if age > max_age_s:
        return None, 0, None, f"stale: metrics.jsonl last written {age:.0f}s ago (> {max_age_s:.0f}s): no live trainer to protect"
    try:
        dts, last_step = step_times(p, last)
    except OSError as e:
        return None, 0, None, f"unreadable: {type(e).__name__}"
    if len(dts) < 3:
        return None, len(dts), last_step, f"only {len(dts)} step interval(s) (< 3)"
    return float(statistics.median(dts)), len(dts), last_step, ""


def guard(path, base, tol=0.05, last=5, max_age_s=900.0, now=None):
    """-> (verdict ok|slow|unknown, ratio | None, cur | None). slow iff cur / base > 1 + tol (exactly 1 + tol is still ok)."""
    cur, n, _, why = trainer_rate(path, last, max_age_s, now)
    if cur is None or not base or base <= 0:
        return "unknown", None, cur
    ratio = cur / base
    return ("slow" if ratio > 1.0 + tol + 1e-12 else "ok"), ratio, cur


# ------------------------------------------------------------------------------------------------ how many cores are left
def overlap_cap(cores, max_workers, secs=60.0, margin=1.0, root="/sys/fs/cgroup", sleep=time.sleep):
    """-> dict(cap, burn, src).  The pod is ONE cgroup: its CPU burn over `secs` is the running trainer's (nothing of P6's pool runs yet)."""
    import p6_cpu_probe as CP
    max_workers = max(int(max_workers), 1)
    a = CP.cg_counters(root)
    if "usage_s" not in a:
        return {"cap": max(1, min(max_workers, int(cores // 2))), "burn": None, "src": "fallback:no readable cgroup CPU counter (UNVERIFIED)"}
    sleep(secs)
    b = CP.cg_counters(root)
    d = CP.cg_delta(a, b)
    if not d or secs <= 0 or d["cpu_s"] < 0:
        return {"cap": max(1, min(max_workers, int(cores // 2))), "burn": None, "src": "fallback:cgroup counter did not advance sanely (UNVERIFIED)"}
    burn = d["cpu_s"] / secs
    free = cores - burn - margin
    return {"cap": max(1, min(max_workers, int(math.floor(free + 1e-9)))), "burn": round(burn, 3), "src": "census"}


# ------------------------------------------------------------------------------------------------ the labeller instrument
def lab_bar(path):
    """-> (verdict ok|below|bad, detail dict)."""
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return "bad", {"why": f"unreadable: {type(e).__name__}"}
    thr = d.get("throughput_sets_s")
    ident = (bool(d.get("identity_ok")) and d.get("n_sets", 0) >= 64 and d.get("identity_equal_sets") == d.get("n_sets")      # ALL sets bit-identical, over >= 64 (gate_checks.judge_lab)
             and bool(d.get("stale_refused")))
    det = {"thr": thr, "bar": BAR_GATE, "workers": d.get("workers"), "identity": ident}
    if not ident or not isinstance(thr, (int, float)):
        return "bad", det
    return ("ok" if thr >= BAR_GATE else "below"), det


# ------------------------------------------------------------------------------------------------ CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("trainer-rate")
    p.add_argument("--metrics", required=True)
    p.add_argument("--last", type=int, default=5)
    p.add_argument("--max-age-s", type=float, default=900.0)
    p = sub.add_parser("trainer-guard")
    p.add_argument("--metrics", required=True)
    p.add_argument("--base", type=float, required=True)
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--last", type=int, default=5)
    p.add_argument("--max-age-s", type=float, default=900.0)
    p = sub.add_parser("overlap-cap")
    p.add_argument("--cores", type=float, required=True)
    p.add_argument("--max", type=int, required=True)
    p.add_argument("--secs", type=float, default=60.0)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--root", default=os.environ.get("P6_CG_ROOT", "/sys/fs/cgroup"))
    p = sub.add_parser("lab-bar")
    p.add_argument("--json", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "trainer-rate":
        cur, n, last_step, why = trainer_rate(a.metrics, a.last, a.max_age_s)
        if cur is None:
            print(f"ZZTRAINER_RATE unknown {why}")
            return 2
        print(f"ZZTRAINER_RATE step_s={cur:.3f} n={n} last_step={last_step}")
        return 0
    if a.cmd == "trainer-guard":
        v, ratio, cur = guard(a.metrics, a.base, a.tol, a.last, a.max_age_s)
        print(f"ZZTRAINER_GUARD {v} ratio={'na' if ratio is None else f'{ratio:.3f}'} cur={'na' if cur is None else f'{cur:.3f}'} base={a.base:.3f} tol={a.tol}")
        return {"ok": 0, "slow": 1, "unknown": 2}[v]
    if a.cmd == "overlap-cap":
        r = overlap_cap(a.cores, a.max, a.secs, a.margin, a.root)
        print(f"ZZOVERLAP_CAP {r['cap']} burn={'na' if r['burn'] is None else r['burn']} cores={a.cores:g} margin={a.margin:g} max={a.max} src={r['src']}")
        return 0
    if a.cmd == "lab-bar":
        v, det = lab_bar(a.json)
        print(f"ZZLAB_BAR {v} " + " ".join(f"{k}={det.get(k)}" for k in ("thr", "bar", "workers", "identity", "why") if k in det))
        return {"ok": 0, "below": 1, "bad": 2}[v]
    return 2


if __name__ == "__main__":
    sys.exit(main())
