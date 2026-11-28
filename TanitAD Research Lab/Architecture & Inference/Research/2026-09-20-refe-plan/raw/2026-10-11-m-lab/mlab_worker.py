#!/usr/bin/env python3
"""M-LAB worker: ONE process, ONE shard of sets, per-phase PROCESS CPU time (user+sys, the process's own clock).

Runs in the navsim v1.1 venv (the labeller's environment: C:/Users/Admin/navsim-crun/venv, PYTHONPATH = navsim @ 3e8291b
+ refe/). Launched by mlab_driver.py, never by hand except for the smoke check.

    --mode cached   load the NAVSIM metric-cache part from disk (lzma + pickle), then executed_plans8 + score_plans
    --mode build    NuPlanScenario build (per log) -> PT.metric_cache_parts(sc) -> executed_plans8 + score_plans
                    (exactly refe/onpolicy_relabel_pdm.py:label_log's body)
    --single        build mode, but ONE scenario-build call per token (no per-log amortisation)

Phases timed per set (time.process_time deltas; quantised at the OS tick, so per-phase numbers are unbiased only as sums):
  read, decompress, unpickle (cached) | scn (scenario build share), parts (build) | e8 (executed_plans8) | score (score_plans)
The HEADLINE per-set CPU is the loop total: process cpu_times (user+sys) at loop end minus loop start, divided by sets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import pickle
import sys
import time

import numpy as np
import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PKG, "refe"))
sys.path.insert(0, os.path.join(PKG, "code"))
DB = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
MAPS = "E:/Archive/devbox-C/navsim/data/maps"

try:
    import ctypes
    _GCPN = ctypes.windll.kernel32.GetCurrentProcessorNumber
except Exception:                                                      # noqa: BLE001
    _GCPN = lambda: -1                                                 # noqa: E731

_PROC = psutil.Process()
pt = time.process_time


def cpu_times():
    c = _PROC.cpu_times()
    return c.user + c.system


def sysbusy():
    """system-wide busy CPU-seconds (user+system, all logical CPUs) -- to attribute a per-set load from OTHER processes"""
    c = psutil.cpu_times()
    return c.user + c.system


def mem():
    m = _PROC.memory_info()
    return {"rss": m.rss, "peak_wset": getattr(m, "peak_wset", 0)}


def upsample20(p8: np.ndarray) -> np.ndarray:
    """[M,8,3] (0.5..4.0 s) -> a [M,20,3] (0.2..4.0 s) array of the same path, ONLY to give executed_plans8 a realistic input
    (its CPU cost does not depend on the values)."""
    t8 = 0.5 * np.arange(1, 9)
    t20 = 0.2 * np.arange(1, 21)
    out = np.zeros((p8.shape[0], 20, 3), np.float64)
    for c in range(3):
        for m in range(p8.shape[0]):
            out[m, :, c] = np.interp(t20, t8, p8[m, :, c])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="plan name, e.g. cached_r1 (reads plan/<name>.json/.npz)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--mode", choices=["cached", "build"], required=True)
    ap.add_argument("--single", action="store_true", help="build mode: one scenario-build call per token")
    ap.add_argument("--retain", action="store_true", help="cached mode: keep every loaded part alive (memory-per-part probe)")
    ap.add_argument("--warm", type=int, default=3, help="untimed warm-up sets BEFORE the timed loop (disjoint from the timed ones)")
    ap.add_argument("--pin", type=int, default=-1, help="pin this process to ONE logical CPU (this process only; nothing else is touched)")
    ap.add_argument("--limit", type=int, default=0, help="smoke only: cap the shard at N tokens")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.pin >= 0:
        _PROC.cpu_affinity([a.pin])
    t_boot = time.perf_counter()
    cpu_boot0 = cpu_times()
    import navsim_pdm_targets as PT
    PT._setup()
    cpu_import_setup = cpu_times() - cpu_boot0
    mem_after_setup = mem()
    plan = json.load(open(os.path.join(HERE, "plan", a.plan + ".json"), encoding="utf-8"))
    npz = np.load(os.path.join(HERE, "plan", a.plan + ".npz"))
    toks = [str(t) for t in npz["token"]]
    row = {t: i for i, t in enumerate(toks)}
    props_all, census_all = npz["proposals"], npz["census_pdms"]
    logs_of = {}
    for t in toks:
        logs_of[t] = plan["paths"][t].replace("\\", "/").split("/")[-4]
    # shard: cached = round-robin tokens; build = round-robin LOGS (a log's tokens stay together), --single = round-robin tokens
    if a.mode == "build" and not a.single:
        lg_sorted = sorted(set(logs_of.values()))
        mine = {lg for k, lg in enumerate(lg_sorted) if k % a.nshards == a.shard}
        my = [t for t in toks if logs_of[t] in mine]
    else:
        my = [t for k, t in enumerate(toks) if k % a.nshards == a.shard]
    if a.limit:
        my = my[: a.limit]
    # untimed warm-up sets: taken from the END of my shard so the timed sets are the leading ones (warm-up tokens are
    # dropped from the timed list, i.e. they are NOT part of the n); build mode warms on the first log's first tokens instead
    warm_n = 0 if a.mode == "build" and not a.single else a.warm
    if a.mode == "build" and not a.single:
        timed, warm = my, []
    else:
        warm, timed = (my[:warm_n], my[warm_n:])
    ex = {"read": 0.0, "decomp": 0.0, "unpickle": 0.0, "scn": 0.0, "parts": 0.0, "e8": 0.0, "score": 0.0}
    wall = {k: 0.0 for k in ex}
    per_set, rss_series, retained = [], [], []
    check = {"n_ok": 0, "n_fail": 0, "skipped_guard": 0, "pdms_exact_1e4": 0, "pdms_total": 0, "pdms_maxabs": 0.0,
             "ctrl_cross_token_max_abs": None}
    errors = []

    def one_set(t, parts, record):
        """returns the pdms vector; times e8 and score."""
        p8 = props_all[row[t]]
        p20 = upsample20(p8)
        c0, w0 = pt(), time.perf_counter()
        _ = PT.executed_plans8(p20, repair=True)                 # the labeller's real conversion (timed, discarded)
        c1, w1 = pt(), time.perf_counter()
        out = PT.score_plans(parts, p8.astype(np.float64))        # the banked 8-pose grid == executed_plans8's output grid
        c2, w2 = pt(), time.perf_counter()
        if record is not None:
            record["e8"] = c1 - c0
            record["score"] = c2 - c1
            ex["e8"] += c1 - c0
            ex["score"] += c2 - c1
            wall["e8"] += w1 - w0
            wall["score"] += w2 - w1
        return out

    def load_cached(t):
        c0, w0 = pt(), time.perf_counter()
        with open(plan["paths"][t], "rb") as fh:
            raw = fh.read()
        c1, w1 = pt(), time.perf_counter()
        blob = lzma.decompress(raw)
        c2, w2 = pt(), time.perf_counter()
        mc = pickle.loads(blob)
        c3, w3 = pt(), time.perf_counter()
        parts = (mc.trajectory, mc.ego_state, mc.observation, mc.centerline, list(mc.route_lane_ids), mc.drivable_area_map)
        return parts, (c1 - c0, c2 - c1, c3 - c2), (w1 - w0, w2 - w1, w3 - w2), len(raw), len(blob)

    def compare(t, out):
        d = np.abs(np.asarray(out["pdms"], np.float64) - census_all[row[t]])
        check["pdms_exact_1e4"] += int((d < 1e-4).sum())
        check["pdms_total"] += int(d.size)
        check["pdms_maxabs"] = max(check["pdms_maxabs"], float(d.max()))
        check["n_ok"] += 1

    sizes = []
    if a.mode == "cached":
        for t in warm:
            parts, *_ = load_cached(t)
            one_set(t, parts, None)
        mem_after_warm = mem()
        c_loop0, w_loop0 = cpu_times(), time.perf_counter()
        for k, t in enumerate(timed):
            try:
                _s0, _c0, _w0 = sysbusy(), cpu_times(), time.perf_counter()
                parts, cc, ww, nraw, nblob = load_cached(t)
                for key, v, w in zip(("read", "decomp", "unpickle"), cc, ww):
                    ex[key] += v
                    wall[key] += w
                sizes.append((nraw, nblob))
                rec = {}
                out = one_set(t, parts, rec)
                compare(t, out)
                _s1, _c1, _w1 = sysbusy(), cpu_times(), time.perf_counter()
                per_set.append({"token": t, "read": cc[0], "decomp": cc[1], "unpickle": cc[2], "e8": rec["e8"], "score": rec["score"],
                                "cpu_set": _c1 - _c0, "wall_set": _w1 - _w0, "cpu_id": _GCPN(),
                                "bg_busy": ((_s1 - _s0) - (_c1 - _c0)) / max(_w1 - _w0, 1e-9)})
                if a.retain:
                    retained.append(parts)
                del parts
            except Exception as e:                                          # noqa: BLE001
                check["n_fail"] += 1
                errors.append(f"{t}: {type(e).__name__}: {str(e)[:160]}")
            if k % 10 == 0:
                rss_series.append((k, mem()["rss"]))
        c_loop1, w_loop1 = cpu_times(), time.perf_counter()
    else:
        c_loop0, w_loop0 = None, None
        mem_after_warm = mem_after_setup
        # group by log (or single token per call)
        groups = []
        if a.single:
            groups = [(logs_of[t], [t]) for t in timed]
            # warm-up: build+score the WARM tokens first (untimed): maps for each city get loaded here
            warm_groups = [(logs_of[t], [t]) for t in warm]
        else:
            by = {}
            for t in timed:
                by.setdefault(logs_of[t], []).append(t)
            groups = sorted(by.items())
            warm_groups = []
        import navtrain_scenarios as NS
        for lg, tl in warm_groups:
            scs = list(NS.build_scenarios_for_log(f"{DB}/{lg}.db", tl, map_root=MAPS, history_rows=40, future_rows=100))
            for sc in scs:
                parts = PT.metric_cache_parts(sc)
                one_set(sc._initial_lidar_token, parts, None)
        mem_after_warm = mem()
        c_loop0, w_loop0 = cpu_times(), time.perf_counter()
        done_logs = 0
        for lg, tl in groups:
            c0, w0 = pt(), time.perf_counter()
            try:
                scs = list(NS.build_scenarios_for_log(f"{DB}/{lg}.db", sorted(set(tl)), map_root=MAPS, history_rows=40, future_rows=100))
            except Exception as e:                                          # noqa: BLE001
                errors.append(f"{lg}: scenario build {type(e).__name__}: {str(e)[:160]}")
                check["n_fail"] += len(tl)
                continue
            c1, w1 = pt(), time.perf_counter()
            ex["scn"] += c1 - c0
            wall["scn"] += w1 - w0
            got = {sc._initial_lidar_token for sc in scs}
            check["skipped_guard"] += len([t for t in tl if t not in got])
            share = (c1 - c0) / max(len(scs), 1)
            for sc in scs:
                t = sc._initial_lidar_token
                try:
                    _s0, _c0, _w0 = sysbusy(), cpu_times(), time.perf_counter()
                    c2, w2 = pt(), time.perf_counter()
                    parts = PT.metric_cache_parts(sc)
                    c3, w3 = pt(), time.perf_counter()
                    ex["parts"] += c3 - c2
                    wall["parts"] += w3 - w2
                    rec = {}
                    out = one_set(t, parts, rec)
                    compare(t, out)
                    _s1, _c1, _w1 = sysbusy(), cpu_times(), time.perf_counter()
                    per_set.append({"token": t, "log": lg, "scn_share": share, "parts": c3 - c2, "e8": rec["e8"], "score": rec["score"],
                                    "cpu_set": _c1 - _c0, "wall_set": _w1 - _w0, "cpu_id": _GCPN(),
                                    "bg_busy": ((_s1 - _s0) - (_c1 - _c0)) / max(_w1 - _w0, 1e-9)})
                except Exception as e:                                      # noqa: BLE001
                    check["n_fail"] += 1
                    errors.append(f"{t}: {type(e).__name__}: {str(e)[:160]}")
                if len(per_set) % 10 == 0:
                    rss_series.append((len(per_set), mem()["rss"]))
            done_logs += 1
        c_loop1, w_loop1 = cpu_times(), time.perf_counter()
    n_done = check["n_ok"]
    # known-answer NEGATIVE control (cached mode only, after the timed loop): the SAME parts scored with another token's
    # proposals must NOT reproduce the census -- proves the agreement above can fail.
    if a.mode == "cached" and len(timed) >= 2:
        t0_, t1_ = timed[0], timed[1]
        parts, *_ = load_cached(t0_)
        out = PT.score_plans(parts, props_all[row[t1_]].astype(np.float64))
        check["ctrl_cross_token_max_abs"] = float(np.abs(np.asarray(out["pdms"]) - census_all[row[t0_]]).max())
    mfull = _PROC.memory_full_info()
    res = {
        "plan": a.plan, "mode": a.mode, "single": a.single, "retain": a.retain, "shard": a.shard, "nshards": a.nshards,
        "pid": os.getpid(), "pin": a.pin, "affinity_end": _PROC.cpu_affinity(), "n_timed_requested": len(timed), "n_sets": n_done,
        "loop_cpu_s": c_loop1 - c_loop0, "loop_wall_s": w_loop1 - w_loop0,
        "cpu_per_set_loop": (c_loop1 - c_loop0) / max(n_done, 1),
        "phase_cpu_s": ex, "phase_wall_s": wall,
        "phase_cpu_per_set": {k: v / max(n_done, 1) for k, v in ex.items()},
        "cpu_import_setup_s": cpu_import_setup, "setup_wall_s": time.perf_counter() - t_boot - (w_loop1 - w_loop0),
        "mem_after_setup": mem_after_setup, "mem_after_warm": mem_after_warm, "mem_end": mem(),
        "uss_end": getattr(mfull, "uss", None), "rss_series": rss_series,
        "check_vs_census": check, "errors": errors[:5], "n_errors": len(errors),
        "pkl_bytes_mean": float(np.mean([s[0] for s in sizes])) if sizes else None,
        "pickle_bytes_mean": float(np.mean([s[1] for s in sizes])) if sizes else None,
        "per_set": per_set,
        "env": {k: os.environ.get(k) for k in ("REFE_SIM_HZ", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    json.dump(res, open(a.out, "w"), indent=0)
    print(f"[{a.plan} shard {a.shard}/{a.nshards} {a.mode}] sets={n_done} fail={check['n_fail']} skip={check['skipped_guard']} "
          f"cpu/set={res['cpu_per_set_loop']:.4f} wall/set={res['loop_wall_s'] / max(n_done, 1):.4f} "
          f"pdms exact1e-4={check['pdms_exact_1e4']}/{check['pdms_total']} rss_end={res['mem_end']['rss'] / 1e6:.0f}MB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
