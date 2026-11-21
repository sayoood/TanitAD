#!/usr/bin/env python3
"""Helpers for refe/p6_cpu_probe.sh -- "can P6's online PDM labeller keep up on THIS pod's cgroup cores?" (eval/P6_ON_11VCPU.md).

CPU only, no GPU, no network, no HF token. Pure standard library; numpy and (optionally) psutil. Every subcommand prints ONE opaque marker line.

  cores        the cgroup core quota (cpu.max, then cfs, then nproc = UNVERIFIED)                       ZZCORES <n> <src>
  cg           cgroup CPU usage / throttling / memory counters as JSON (v2 and v1)                       ZZCG {...}
  sample-bank  stream the training bank ONCE: pick N evenly spaced WHOLE logs (their real tokens/log),
               write a minimal targets_rank0.jsonl for them, and count the EXACT D9 workload of the bank
               (distinct logged-route tokens, distinct lane-rank twin keys)                              ZZBANK_SAMPLE {...}
  pool         the online labeller pool at --workers W on cached parts, instrumented:
                 thru     the diag_p6_labeller_identity.run_labeller loop (micro-batches of 4, collect k-1 after submit k)
                 paced    the trainer's clock: F(k) -> submit(k) -> collect(k-1) [= the GPU stall] -> B(k-1), one micro-batch = 57.0/64 s
                 latency  one micro-batch at a time, collected at once: the unqueued service latency of a 4-set micro-batch
               -> sets/s, CPU-s/set of the worker processes (steady state), RSS, threads, cgroup throttling      ZZPOOL_OK {...}
  verdict      read the stage JSONs, print ONE line ZZP6CPU_FIT|MARGINAL|SHORT|VOID with the numbers
  selftest-*   (used by test_p6_cpu_probe.py)

The probe measures the pod; it does not change a gate value. The bars are the gate's own (gate_checks.LAB_NEED_SETS_S x LAB_MARGIN, test-pinned).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "code"))
if os.environ.get("P6CPU_PKG"):                                            # the probe may live outside the shipped tree; the tree's modules are what is measured
    sys.path.insert(0, str(Path(os.environ["P6CPU_PKG"]) / "code"))
    sys.path.insert(0, str(Path(os.environ["P6CPU_PKG"]) / "refe"))

NEED_SETS_S = 4.49                 # SPEC_P6 3.3 = gate_checks.LAB_NEED_SETS_S (pinned by test_p6_cpu_probe.py)
MARGIN = 1.5                       # gate_checks.LAB_MARGIN
BAR_GATE = round(NEED_SETS_S * MARGIN, 3)    # 6.735
MICRO_S = 57.0 / 64.0              # one micro-batch of the pure training step (SPEC_P6 0.2: 57.0 s per 256-sample step = 64 micro-batches of 4)
FWD_FRAC = 1.0 / 3.0               # share of a micro-batch spent in F(k) before the proposals exist (backward is ~2x forward)
STALL_MAX = 0.05                   # gate_checks.LAB_STALL_MAX
N_PROP = 64


# ------------------------------------------------------------------------------------------------ cgroup
def _read(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except Exception:                                                      # noqa: BLE001
        return None


def cgroup_cores(root="/sys/fs/cgroup", fallback=None):
    """-> (cores float, source). v2 cpu.max, then v1 cfs, then os.cpu_count() (UNVERIFIED: no quota file)."""
    n = float(fallback if fallback is not None else (os.cpu_count() or 1))
    src = "nproc (UNVERIFIED: no cgroup quota file)"
    v = _read(f"{root}/cpu.max")
    if v:
        q, _, per = v.partition(" ")
        if q != "max" and per.isdigit() and int(per) > 0:
            return max(int(q) / int(per), 1.0), "cgroup v2 cpu.max"
    q, per = _read(f"{root}/cpu/cpu.cfs_quota_us"), _read(f"{root}/cpu/cpu.cfs_period_us")
    if q and per and re.fullmatch(r"-?\d+", q) and int(q) > 0 and int(per) > 0:
        return max(int(q) / int(per), 1.0), "cgroup v1 cfs"
    return n, src


def _kv(text):
    out = {}
    for ln in (text or "").splitlines():
        p = ln.split()
        if len(p) == 2 and re.fullmatch(r"-?\d+", p[1]):
            out[p[0]] = int(p[1])
    return out


def cg_counters(root="/sys/fs/cgroup"):
    """CPU usage (s), throttling (periods, s) and memory (bytes) of the container's cgroup; {} when nothing is readable. v2 first, then v1."""
    out = {}
    st = _kv(_read(f"{root}/cpu.stat"))
    if "usage_usec" in st:                                                 # cgroup v2
        out.update(version=2, usage_s=st["usage_usec"] / 1e6, nr_periods=st.get("nr_periods", 0), nr_throttled=st.get("nr_throttled", 0),
                   throttled_s=st.get("throttled_usec", 0) / 1e6)
        ms = _kv(_read(f"{root}/memory.stat"))
        lim = _read(f"{root}/memory.max")
        use = _read(f"{root}/memory.current")
        out.update(mem_limit=int(lim) if lim and lim.isdigit() else None, mem_usage=int(use) if use and use.isdigit() else None,
                   mem_rss=ms.get("anon"), mem_cache=ms.get("file"))
        return out
    cu = _read(f"{root}/cpuacct/cpuacct.usage") or _read(f"{root}/cpu,cpuacct/cpuacct.usage")
    cs = _kv(_read(f"{root}/cpu/cpu.stat") or _read(f"{root}/cpu,cpuacct/cpu.stat"))
    if cu and cu.isdigit():
        out.update(version=1, usage_s=int(cu) / 1e9, nr_periods=cs.get("nr_periods", 0), nr_throttled=cs.get("nr_throttled", 0),
                   throttled_s=cs.get("throttled_time", 0) / 1e9)
    ms = _kv(_read(f"{root}/memory/memory.stat"))
    lim, use = _read(f"{root}/memory/memory.limit_in_bytes"), _read(f"{root}/memory/memory.usage_in_bytes")
    if lim or use:
        out.update(version=out.get("version", 1), mem_limit=int(lim) if lim and lim.isdigit() else None,
                   mem_usage=int(use) if use and use.isdigit() else None, mem_rss=ms.get("total_rss", ms.get("rss")),
                   mem_cache=ms.get("total_cache", ms.get("cache")))
    return out


def cg_delta(a, b):
    """b - a for the counters that accumulate; {} when either side is empty."""
    if not a or not b or "usage_s" not in a or "usage_s" not in b:
        return {}
    return {"cpu_s": b["usage_s"] - a["usage_s"], "nr_throttled": b.get("nr_throttled", 0) - a.get("nr_throttled", 0),
            "nr_periods": b.get("nr_periods", 0) - a.get("nr_periods", 0), "throttled_s": b.get("throttled_s", 0.0) - a.get("throttled_s", 0.0)}


# ------------------------------------------------------------------------------------------------ process tree (psutil if present, else /proc)
try:
    import psutil as _psutil
except Exception:                                                          # noqa: BLE001
    _psutil = None


def _proc_children_rows():
    """{pid: (ppid, cpu_s, rss_bytes, threads)} for every process, from /proc (Linux)."""
    rows = {}
    tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    page = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            s = Path(f"/proc/{d}/stat").read_text()
            f = s[s.rfind(")") + 2:].split()                              # fields after "(comm) "
            rows[int(d)] = (int(f[1]), (int(f[11]) + int(f[12])) / tck, int(f[21]) * page, int(f[17]))
        except Exception:                                                  # noqa: BLE001
            continue
    return rows


def tree_snapshot(pid=None):
    """CPU seconds (user+sys), RSS and thread counts of every DESCENDANT of `pid` (default: this process), plus this process's own CPU."""
    pid = pid or os.getpid()
    kids = []
    if _psutil is not None:
        me = _psutil.Process(pid)
        for c in me.children(recursive=True):
            try:
                t = c.cpu_times()
                kids.append((c.pid, t.user + t.system, c.memory_info().rss, c.num_threads()))
            except Exception:                                              # noqa: BLE001
                continue
        t = me.cpu_times()
        own = t.user + t.system
    else:
        rows = _proc_children_rows()
        stack, seen = [pid], set()
        while stack:
            p = stack.pop()
            for q, (pp, _c, _r, _t) in rows.items():
                if pp == p and q not in seen:
                    seen.add(q)
                    stack.append(q)
        kids = [(q, rows[q][1], rows[q][2], rows[q][3]) for q in seen]
        own = rows.get(pid, (0, 0.0, 0, 0))[1]
    return {"n": len(kids), "cpu_s": sum(k[1] for k in kids), "own_cpu_s": own, "rss_max": max([k[2] for k in kids] or [0]),
            "rss_sum": sum(k[2] for k in kids), "threads_max": max([k[3] for k in kids] or [0]), "threads_sum": sum(k[3] for k in kids),
            "per_pid": {str(k[0]): k[1] for k in kids}}


class Sampler(threading.Thread):
    """Peak RSS / threads of the worker tree over the whole run (a 0.5 s poll); with psutil and `cpus`, also the system-wide busy cores inside that CPU set
    (everything running there, ours or not -- the dev-box emulation subtracts its own tree and burners to get the background)."""

    def __init__(self, every=0.5, cpus=None):
        super().__init__(daemon=True)
        self.every, self.stop, self.peak = every, threading.Event(), {"rss_max": 0, "rss_sum": 0, "threads_max": 0, "n": 0}
        self.cpus, self.sys = cpus, []

    def run(self):
        if self.cpus and _psutil is not None:
            _psutil.cpu_percent(percpu=True)
        while not self.stop.is_set():
            try:
                s = tree_snapshot()
                for k in self.peak:
                    self.peak[k] = max(self.peak[k], s[k])
                if self.cpus and _psutil is not None:
                    per = _psutil.cpu_percent(percpu=True)
                    self.sys.append((time.perf_counter(), sum(per[c] for c in self.cpus if c < len(per)) / 100.0))
            except Exception:                                              # noqa: BLE001
                pass
            self.stop.wait(self.every)


def pct(xs, q):
    if not xs:
        return None
    ys = sorted(xs)
    i = (len(ys) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return ys[lo] + (ys[hi] - ys[lo]) * (i - lo)


# ------------------------------------------------------------------------------------------------ the bank
RX = {k: re.compile(r'"%s":\s*("([^"]*)"|-?\d+|null)' % k) for k in ("log_name", "token", "rank")}
AUG_LANE = re.compile(r'"aug":\s*\{[^}]*"kind":\s*"lane_rank"[^}]*"value":\s*(-?\d+)|"aug":\s*\{[^}]*"value":\s*(-?\d+)[^}]*"kind":\s*"lane_rank"')


def head_row(line):
    """(log, token, lane_value|None) from the head+tail of one bank line, or None. Regex only: the 8 GB bank is never json-parsed."""
    h = line[:3000] + line[-600:] if len(line) > 3600 else line
    lg, tk = RX["log_name"].search(h), RX["token"].search(h)
    if not lg or not tk or lg.group(2) is None or tk.group(2) is None:
        return None
    m = AUG_LANE.search(h)
    lane = int(m.group(1) if m.group(1) is not None else m.group(2)) if m else None
    return lg.group(2), tk.group(2), lane


def sample_bank(bank, db_root, n_logs, out_dir, max_scan_s=300.0, log_names=None):
    """Two things in ONE pass: (1) the exact D9 workload (distinct plain (log, token); distinct (log, token, lane value)), (2) a minimal bank of `n_logs`
    evenly spaced whole logs. The log list comes from the DB file names (sorted), else from `log_names`. A scan that hits `max_scan_s` is reported INCOMPLETE."""
    files = sorted(Path(bank).glob("targets_rank*.jsonl"))
    if not files:
        raise SystemExit(f"no targets_rank*.jsonl under {bank}")
    if log_names is None:
        log_names = sorted(p.stem for p in Path(db_root).glob("*.db"))
    if not log_names:
        raise SystemExit(f"no *.db under {db_root}")
    k = max(len(log_names) // max(n_logs, 1), 1)
    chosen = set(log_names[::k][:n_logs])
    plain, lane, kept = set(), set(), {}
    t0, rows_seen, bad, complete = time.time(), 0, 0, True
    for fp in files:
        with open(fp, encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                rows_seen += 1
                r = head_row(ln)
                if r is None:
                    bad += 1
                    continue
                lg, tk, lv = r
                if lv is None:
                    plain.add((lg, tk))
                    if lg in chosen:
                        kept.setdefault(lg, set()).add(tk)
                else:
                    lane.add((lg, tk, lv))
                if rows_seen % 20000 == 0 and time.time() - t0 > max_scan_s:
                    complete = False
                    break
        if not complete:
            break
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_tok = 0
    with open(out / "targets_rank0.jsonl", "w", encoding="utf-8") as fh:
        for lg in sorted(kept):
            for tk in sorted(kept[lg]):
                fh.write(json.dumps({"log_name": lg, "token": tk, "rank": 0, "step": 0, "aug": None}) + "\n")
                n_tok += 1
    per_log = len(plain) / max(len({p[0] for p in plain}), 1)
    res = {"complete": complete, "scan_s": round(time.time() - t0, 1), "rows_seen": rows_seen, "unparsed_rows": bad, "n_db_logs": len(log_names),
           "bank_logs": len({p[0] for p in plain}), "plain_tokens": len(plain), "lane_twin_keys": len(lane),
           "lane_values": sorted({x[2] for x in lane}), "tokens_per_log_mean": round(per_log, 2), "sample_logs": len(kept), "sample_tokens": n_tok,
           "sample_log_names": sorted(kept)}
    return res


# ------------------------------------------------------------------------------------------------ the pool
def _items(tokens, n):
    base = [tuple(t.split(":", 1)) for t in tokens.split(",") if t]
    if not base:
        raise SystemExit("no tokens")
    return [base[i % len(base)] for i in range(n)]


def _meta(lg, tk):
    return {"log_name": lg, "token": tk, "rank": 0, "aug": None, "step": 0}


def run_pool(cache, tokens, workers, n_sets, mode="thru", python=None, backend="pdm", period=MICRO_S, seed=0, affinity=None, n_micro=None, chunks=1):
    """`chunks` k > 1 splits each 64-proposal set into k jobs of 64/k proposals (score_plans scores every proposal against PDM-Closed alone, so the labels are
    unchanged: M-LAB's chunk_probe, max |dPDMS| = 0.0). It is a HARNESS-level emulation of the latency lever; refe/online_labeller.py itself submits one job per set."""
    import numpy as np
    import online_labeller as OL
    import diag_p6_labeller_identity as DI
    cpus = set_affinity(affinity) if affinity else None
    k = max(int(chunks), 1)
    if N_PROP % k:
        raise SystemExit(f"--chunks {k} does not divide {N_PROP} proposals")
    items = _items(tokens, n_sets)
    props = DI.proposals(n_sets, seed=seed)
    if k > 1:
        m = N_PROP // k
        jitems = [x for x in items for _ in range(k)]
        jprops = props.reshape((n_sets, k, m) + props.shape[2:]).reshape((n_sets * k, m) + props.shape[2:])
    else:
        jitems, jprops = items, props
    MB = 4 * k                                                              # jobs per 4-set micro-batch
    lab = OL.OnlineLabeller(backend=backend, workers=workers, metric_cache=cache, python=python, n_prop=jprops.shape[1], timeout_s=600)
    samp = Sampler(cpus=cpus)
    samp.start()
    try:
        wm = [_meta(lg, tk) for lg, tk in jitems[:max(workers, 1)]]
        lab.collect(lab.submit(-1, 0, wm, jprops[:len(wm)]), 0)              # NOT timed: start every worker, warm its import and parts path
        lab.take_stats(reset=True)
        s0, t0, g0 = tree_snapshot(), time.perf_counter(), cg_counters()
        lat, stalls = [], []
        outs = []
        nm = 0
        if mode == "thru":                                                   # == diag_p6_labeller_identity.run_labeller's loop
            h = None
            for i in range(0, len(jitems), MB):
                prev, h = h, lab.submit(i // MB, 0, [_meta(*x) for x in jitems[i:i + MB]], jprops[i:i + MB])
                if prev is not None:
                    outs.append(lab.collect(prev, 0))
            outs.append(lab.collect(h, 0))
        elif mode == "latency":
            for i in range(0, len(jitems), MB):
                a = time.perf_counter()
                h = lab.submit(i // MB, 0, [_meta(*x) for x in jitems[i:i + MB]], jprops[i:i + MB])
                outs.append(lab.collect(h, 0))
                lat.append(time.perf_counter() - a)
        else:                                                                # paced
            nm = n_micro or max(len(items) // 4, 2)
            h_prev = None
            for kk in range(nm):
                a = time.perf_counter()
                time.sleep(period * FWD_FRAC)                                # F(k): the forward that produces micro-batch k's proposals
                j = (kk * MB) % max(len(jitems) - (MB - 1), 1)
                h = lab.submit(kk, 0, [_meta(*x) for x in jitems[j:j + MB]], jprops[j:j + MB])
                if h_prev is not None:
                    b = time.perf_counter()
                    outs.append(lab.collect(h_prev, 0))                      # the GPU waits HERE for the labels of k-1
                    stalls.append(time.perf_counter() - b)
                time.sleep(period * (1.0 - FWD_FRAC))                        # B(k-1): the deferred backward
                lat.append(time.perf_counter() - a)
                h_prev = h
            outs.append(lab.collect(h_prev, 0))
        wall = time.perf_counter() - t0
        s1, g1 = tree_snapshot(), cg_counters()
        t_end = time.perf_counter()
        stats = lab.take_stats(reset=False)
        n_done = (nm * 4) if mode == "paced" else len(items)
        labelled = int(sum(int(o[1].sum()) for o in outs))
        res = {"mode": mode, "workers": workers, "chunks": k, "n_sets": n_done, "labelled_jobs": labelled, "expected_jobs": n_done * k, "wall_s": round(wall, 3),
               "sets_per_s": round(n_done / wall, 3) if mode != "paced" else None,
               "cpu_per_set_tree_s": round((s1["cpu_s"] - s0["cpu_s"]) / max(n_done, 1), 4),
               "cpu_per_set_job_s": round(stats["cpu_s"] / max(n_done, 1), 4),
               "rss_worker_max_mb": round(max(s1["rss_max"], samp.peak["rss_max"]) / 1e6, 1), "rss_tree_sum_mb": round(max(s1["rss_sum"], samp.peak["rss_sum"]) / 1e6, 1),
               "threads_worker_max": max(s1["threads_max"], samp.peak["threads_max"]), "n_worker_procs": s1["n"],
               "own_cpu_s": round(s1["own_cpu_s"] - s0["own_cpu_s"], 3), "cg": cg_delta(g0, g1),
               "cg_bg_cores": None, "env": {kx: os.environ.get(kx) for kx in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "REFE_SIM_HZ")}}
        win = [v for (tt, v) in samp.sys if t0 <= tt <= t_end]
        if win:
            res["sys_busy_cores_in_affinity"] = round(sum(win) / len(win), 3)
            res["sys_busy_samples"] = len(win)
            res["affinity"] = sorted(cpus)
        if res["cg"]:
            ours = (s1["cpu_s"] - s0["cpu_s"]) + (s1["own_cpu_s"] - s0["own_cpu_s"])
            res["cg_bg_cores"] = round(max(res["cg"]["cpu_s"] - ours, 0.0) / max(wall, 1e-9), 3)
        if mode in ("latency", "paced"):
            res["micro_latency_s"] = {"p50": pct(lat, .5), "p95": pct(lat, .95), "max": max(lat) if lat else None, "n": len(lat)}
        if mode == "latency":
            res["service_latency_4sets_s"] = res["micro_latency_s"]
        if mode == "paced":
            tot = sum(stalls) + nm * period
            res["period_s"] = period
            res["stall_s_total"] = round(sum(stalls), 3)
            res["stall_frac"] = round(sum(stalls) / max(tot, 1e-9), 4)
            res["stall_p95_s"] = pct(stalls, .95)
            res["stall_max_s"] = max(stalls) if stalls else None
            res["sets_per_s_paced"] = round(n_done / wall, 3)
        return res
    finally:
        samp.stop.set()
        lab.close()


def set_affinity(spec):
    """'0-7,12-14' -> pin THIS process (the pool workers inherit it). psutil on Windows, sched_setaffinity on Linux."""
    cpus = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        cpus += list(range(int(a), int(b or a) + 1))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpus))
    elif _psutil is not None:
        _psutil.Process().cpu_affinity(cpus)
    else:
        raise SystemExit("cannot set CPU affinity here")
    return cpus


# ------------------------------------------------------------------------------------------------ D9
def _cg_arg(txt):
    """'ZZCG {...}' (the marker line `cg` prints) or the bare JSON -> dict."""
    txt = (txt or "").strip()
    return json.loads(txt.split(" ", 1)[1] if txt.startswith("ZZCG ") else txt or "{}")


def d9_summary(before, after, wall_s, cw, bank_json, cache, eff=0.9):
    """CPU-s per token of the D9 build on the sample, from the cgroup's own counter (a pool of spawn workers has no single process to read), and the extrapolation
    to the whole bank. `eff` is the ASSUMED pool efficiency of the full 1,192-log build (the sample has too few logs to measure it): ESTIMATED."""
    n_tok = sum(1 for _ in Path(cache).glob("*/*.pkl.xz")) if Path(cache).exists() else 0
    d = cg_delta(before, after)
    bank = _jload(bank_json) or {}
    full_tokens = (bank.get("plain_tokens") or 0) + (bank.get("lane_twin_keys") or 0)
    out = {"tokens_built": n_tok, "wall_s": wall_s, "cache_workers": cw, "cg": d, "cache_bytes": sum(f.stat().st_size for f in Path(cache).glob("*/*.pkl.xz")) if n_tok else 0}
    if d and n_tok:
        out["cpu_s_per_token"] = round(d["cpu_s"] / n_tok, 3)
        out["wall_s_per_token_at_cw"] = round(wall_s / n_tok, 3)
        out["bytes_per_token"] = round(out["cache_bytes"] / n_tok)
        out["full_tokens"] = full_tokens
        out["full_cpu_h"] = round(full_tokens * out["cpu_s_per_token"] / 3600, 2)
        out["full_wall_h"] = round(full_tokens * out["cpu_s_per_token"] / max(cw * eff, 1e-9) / 3600, 2)
        out["full_cache_gb"] = round(full_tokens * out["bytes_per_token"] / 1e9, 1)
        out["assumed_pool_efficiency"] = eff
    return out


# ------------------------------------------------------------------------------------------------ the verdict
def _num(x, nd=3):
    return "na" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def decide(cores, lw_default, per_lw, d9=None, bar_gate=BAR_GATE, bar_need=NEED_SETS_S, stall_max=STALL_MAX):
    """per_lw = {W: {"identity_ok": bool, "stale_refused": bool, "gate_sets_s": float, "paced_stall_frac": float|None, "cpu_per_set": float|None, ...}}.
    Returns (verdict, why, best_lw). The rules are PRE-REGISTERED in eval/P6_ON_11VCPU.md section 7:
      VOID      no measurement, or ANY identity / stale-refusal failure (a throughput from a labeller that is not the offline labeller is void)
      FIT       the chain's DEFAULT worker count (cores-4) passes the gate's bar (>= 6.735) AND its paced stall is <= 5 %
      MARGINAL  not FIT, but SOME measured worker count reaches the need (>= 4.49 sets/s) -- P6 would run only with a lever (more workers, a thinner trainer, chunking)
      SHORT     no measured worker count reaches 4.49 sets/s: P6 would run at the labeller's pace (step time = 256 / rate)"""
    if not per_lw:
        return "VOID", "no pool measurement", None
    bad = [w for w, r in sorted(per_lw.items()) if not (r.get("identity_ok") and r.get("stale_refused"))]
    if bad:
        return "VOID", f"identity or stale-proposal refusal failed at workers {bad}", None

    def thr(r):
        return r.get("gate_sets_s") or 0.0

    def stall_ok(r):
        s = r.get("paced_stall_frac")
        return s is not None and s <= stall_max

    passing = sorted(w for w, r in per_lw.items() if thr(r) >= bar_gate and stall_ok(r))
    d = per_lw.get(lw_default)
    if d is not None and thr(d) >= bar_gate and stall_ok(d):
        return "FIT", f"default workers {lw_default}: {thr(d):.3f} >= {bar_gate} sets/s, stall {100 * d['paced_stall_frac']:.1f} % <= {100 * stall_max:.0f} %", lw_default
    need = sorted(w for w, r in per_lw.items() if thr(r) >= bar_need)
    if need:
        best = passing[0] if passing else need[-1]
        return "MARGINAL", (f"default workers {lw_default} fails the gate or the stall bar; reaches the need (>= {bar_need}) at workers {need}; "
                            f"passes gate+stall at {passing or 'none'}"), best
    best = max(per_lw, key=lambda w: thr(per_lw[w]))
    return "SHORT", f"best {thr(per_lw[best]):.3f} sets/s at workers {best} < the need {bar_need}", best


def step_time_days(rate_sets_s, need=NEED_SETS_S, base_days=6.65, ckpt_overhead=1.02):
    """P6's wall time if the labeller sustains `rate` sets/s: the step is max(GPU 57.0 s, 256/rate)."""
    return base_days * max(1.0, need / max(rate_sets_s, 1e-9)) * ckpt_overhead


# ------------------------------------------------------------------------------------------------ CLI
def _jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:                                                      # noqa: BLE001
        return None


def cmd_verdict(a):
    d = Path(a.dir)
    cores = a.cores
    lw_default = a.lw_default
    per_lw, rows = {}, []
    for w in sorted(int(x) for x in a.workers.split(",")):
        lab = _jload(d / f"lab_w{w}.json")
        thru = _jload(d / f"pool_thru_w{w}.json")
        paced = _jload(d / f"pool_paced_w{w}.json")
        lat = _jload(d / f"pool_latency_w{w}.json")
        if lab is None:
            continue
        per_lw[w] = {"identity_ok": bool(lab.get("identity_ok")), "stale_refused": bool(lab.get("stale_refused")), "gate_sets_s": lab.get("throughput_sets_s"),
                     "identity_sets": f"{lab.get('identity_equal_sets')}/{lab.get('n_sets')}", "paced_stall_frac": (paced or {}).get("stall_frac"),
                     "cpu_per_set": (thru or {}).get("cpu_per_set_tree_s"), "lat_p95": ((lat or {}).get("micro_latency_s") or {}).get("p95"),
                     "rss_mb": (thru or {}).get("rss_worker_max_mb"), "thr_sets_s": (thru or {}).get("sets_per_s"),
                     "throttled": ((thru or {}).get("cg") or {}).get("nr_throttled"), "threads": (thru or {}).get("threads_worker_max")}
    v, why, best = decide(cores, lw_default, per_lw)
    d9 = _jload(d / "d9.json") or {}
    dd = _jload(d / "dd.json") or {}
    sm = _jload(d / "bank_sample.json") or {}
    t = {w: r["gate_sets_s"] for w, r in per_lw.items()}
    st = {w: (None if r["paced_stall_frac"] is None else round(r["paced_stall_frac"], 3)) for w, r in per_lw.items()}
    cs = {w: r["cpu_per_set"] for w, r in per_lw.items()}
    lt = {w: (None if r["lat_p95"] is None else round(r["lat_p95"], 3)) for w, r in per_lw.items()}
    rate = max([x for x in t.values() if x] or [0.0])
    line = (f"ZZP6CPU_{v} cores={_num(cores, 2)} lw_default={lw_default} best_lw={best} gate_sets_s={t} bar_gate={BAR_GATE} bar_need={NEED_SETS_S} "
            f"paced_stall={st} cpu_s_per_set={cs} svc_latency_p95_s={lt} d9_cpu_s_per_token={d9.get('cpu_s_per_token')} d9_full_wall_h={d9.get('full_wall_h')} "
            f"dd_gb_ok={dd.get('gb_ok')}/{dd.get('gb_target')} rss_worker_mb={[r['rss_mb'] for r in per_lw.values()][:1]} "
            f"p6_days_if_best_rate={_num(step_time_days(rate), 1) if rate else 'na'} why={why}")
    out = {"verdict": v, "why": why, "best_lw": best, "per_lw": per_lw, "d9": d9, "dd": dd, "bank_sample": sm, "cores": cores, "lw_default": lw_default,
           "line": line}
    (d / "verdict.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(line)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cores")
    p.add_argument("--root", default="/sys/fs/cgroup")
    p = sub.add_parser("cg")
    p.add_argument("--root", default="/sys/fs/cgroup")
    p = sub.add_parser("cg-rate")
    p.add_argument("--root", default="/sys/fs/cgroup")
    p.add_argument("--secs", type=float, default=60.0)
    p = sub.add_parser("sample-bank")
    p.add_argument("--bank", required=True)
    p.add_argument("--db-root", required=True)
    p.add_argument("--logs", type=int, default=12)
    p.add_argument("--out", required=True)
    p.add_argument("--max-scan-s", type=float, default=300.0)
    p.add_argument("--json", default=None)
    p = sub.add_parser("pool")
    p.add_argument("--cache", required=True)
    p.add_argument("--tokens", required=True)
    p.add_argument("--workers", type=int, required=True)
    p.add_argument("--n-sets", type=int, default=256)
    p.add_argument("--mode", choices=("thru", "paced", "latency"), default="thru")
    p.add_argument("--backend", choices=("pdm", "synthetic"), default="pdm")
    p.add_argument("--python", default=None)
    p.add_argument("--period", type=float, default=MICRO_S)
    p.add_argument("--n-micro", type=int, default=None)
    p.add_argument("--chunks", type=int, default=1)
    p.add_argument("--affinity", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("d9-summary")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--wall", type=float, required=True)
    p.add_argument("--cw", type=int, required=True)
    p.add_argument("--bank-json", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("verdict")
    p.add_argument("--dir", required=True)
    p.add_argument("--cores", type=float, required=True)
    p.add_argument("--lw-default", type=int, required=True)
    p.add_argument("--workers", default="7,8,9,10")
    a = ap.parse_args(argv)
    if a.cmd == "cores":
        c, src = cgroup_cores(a.root)
        print(f"ZZCORES {int(c) if c == int(c) else c} {src}")
        return 0
    if a.cmd == "cg":
        print("ZZCG " + json.dumps(cg_counters(a.root)))
        return 0
    if a.cmd == "cg-rate":
        # READ-ONLY: two reads of the cgroup counters `--secs` apart. Safe beside a training run (adds no load); on a P2-replicate pod it is the trainer + loader census.
        a0 = cg_counters(a.root)
        time.sleep(a.secs)
        a1 = cg_counters(a.root)
        d = cg_delta(a0, a1)
        if not d:
            print("ZZCGRATE unknown (no readable cgroup CPU counter)")
            return 1
        print("ZZCGRATE " + json.dumps({"secs": a.secs, "cores_used": round(d["cpu_s"] / a.secs, 3), "throttled_periods": d["nr_throttled"], "periods": d["nr_periods"],
                                        "throttled_s": round(d["throttled_s"], 3), "mem_usage_gb": round((a1.get("mem_usage") or 0) / 1e9, 2),
                                        "mem_rss_gb": round((a1.get("mem_rss") or 0) / 1e9, 2), "mem_cache_gb": round((a1.get("mem_cache") or 0) / 1e9, 2)}))
        return 0
    if a.cmd == "sample-bank":
        r = sample_bank(a.bank, a.db_root, a.logs, a.out, a.max_scan_s)
        if a.json:
            Path(a.json).write_text(json.dumps(r, indent=1), encoding="utf-8")
        print("ZZBANK_SAMPLE " + json.dumps({k: v for k, v in r.items() if k != "sample_log_names"}))
        return 0 if r["sample_tokens"] > 0 else 1
    if a.cmd == "d9-summary":
        r = d9_summary(_cg_arg(a.before), _cg_arg(a.after), a.wall, a.cw, a.bank_json, a.cache)
        Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
        print("ZZP6CPU_D9 " + json.dumps({k: v for k, v in r.items() if k != "cg"}))
        return 0 if r.get("tokens_built") else 1
    if a.cmd == "pool":
        r = run_pool(a.cache, a.tokens, a.workers, a.n_sets, a.mode, a.python, a.backend, a.period, affinity=a.affinity, n_micro=a.n_micro, chunks=a.chunks)
        Path(a.out).write_text(json.dumps(r, indent=1), encoding="utf-8")
        print("ZZPOOL_OK " + json.dumps({k: r.get(k) for k in ("mode", "workers", "n_sets", "sets_per_s", "stall_frac", "cpu_per_set_tree_s", "rss_worker_max_mb")}))
        return 0 if r["labelled_jobs"] == r["expected_jobs"] else 1
    return cmd_verdict(a)


if __name__ == "__main__":
    sys.exit(main())
