#!/usr/bin/env python3
"""M-LAB aggregation: runs/*/summary.json + worker JSONs (+ chunk_probe.json) -> results.json (every number the RESULT cites).
No judgement here, only arithmetic; the RESULT's wording is written from results.json. Any python with numpy.

    python mlab_summarize.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SETS_PER_S = 4.49          # SPEC_P6 section 3.3: 2,575,925 sets / 6.64 days
TOTAL_SETS = 2_575_925


def load(tag):
    d = os.path.join(RUNS, tag)
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
        return None
    S = json.load(open(p, encoding="utf-8"))
    W = [json.load(open(os.path.join(d, w), encoding="utf-8")) for w in S["worker_outputs"] if os.path.exists(os.path.join(d, w))]
    return S, W


def spread(xs):
    xs = [float(x) for x in xs]
    return {"values": [round(x, 4) for x in xs], "median": round(st.median(xs), 4), "min": round(min(xs), 4),
            "max": round(max(xs), 4), "max_over_min": round(max(xs) / min(xs), 3) if min(xs) > 0 else None}


def pct(a, q):
    return float(np.percentile(np.asarray(a, float), q))


def arm(tag):
    L = load(tag)
    if L is None:
        return None
    S, W = L
    n = sum(w["n_sets"] for w in W)
    cpu = sum(w["loop_cpu_s"] for w in W)
    wall = max(w["loop_wall_s"] for w in W)
    ph = {k: sum(w["phase_cpu_s"][k] for w in W) / max(n, 1) for k in W[0]["phase_cpu_s"]}
    keys = [k for k, v in ph.items() if v > 0 or k in ("score", "e8")]
    per_set_tot = []
    for w in W:
        for r in w["per_set"]:
            per_set_tot.append(sum(v for k, v in r.items() if k in ("read", "decomp", "unpickle", "scn_share", "parts", "e8", "score")))
    chk = {k: sum(w["check_vs_census"][k] for w in W) for k in ("n_ok", "n_fail", "skipped_guard", "pdms_exact_1e4", "pdms_total")}
    rows = S["samples"]
    bg = [r["background_logical"] for r in rows if r["workers_logical"] > 0] or [0.0]
    busy = [r["busy_logical"] for r in rows if r["workers_logical"] > 0] or [0.0]
    # per-worker steady-state RSS from the driver's 1 Hz sampler (the real child process): median over the run's second half
    rs = {}
    for r in rows:
        for pid, v in r["rss"].items():
            rs.setdefault(pid, []).append(v)
    sampler_rss = [float(np.median(v[len(v) // 2:])) for v in rs.values() if len(v) >= 4]
    return {
        "tag": tag, "plan": S["plan"], "mode": S["mode"], "nworkers": S["nworkers"], "rcs": S["rcs"], "n_sets": n,
        "cpu_per_set_loop": cpu / max(n, 1), "loop_cpu_total_s": cpu, "loop_wall_max_s": wall,
        "sets_per_s_wall": n / wall if wall > 0 else None, "busy_workers_equiv": cpu / wall if wall > 0 else None,
        "phase_cpu_per_set": {k: round(ph[k], 5) for k in keys},
        "per_set_total_pctl": {"p50": pct(per_set_tot, 50), "p90": pct(per_set_tot, 90), "p99": pct(per_set_tot, 99), "max": max(per_set_tot)} if per_set_tot else None,
        "check": chk, "n_errors": sum(w["n_errors"] for w in W), "errors": [e for w in W for e in w["errors"]][:3],
        "background_logical_cpus": {"mean": round(float(np.mean(bg)), 2), "p90": round(pct(bg, 90), 2), "max": round(max(bg), 2)},
        "machine_busy_logical_cpus": {"mean": round(float(np.mean(busy)), 2), "max": round(max(busy), 2)},
        "ncpu_logical": S["ncpu_logical"],
        "mem_available_mb_min": min((r["mem_available_mb"] for r in rows), default=None),
        "rss_end_mb_per_worker": [round(w["mem_end"]["rss"] / 1e6, 1) for w in W],
        "peak_wset_mb_per_worker": [round(w["mem_end"]["peak_wset"] / 1e6, 1) for w in W],
        "uss_end_mb_per_worker": [round((w["uss_end"] or 0) / 1e6, 1) for w in W],
        "rss_after_setup_mb": [round(w["mem_after_setup"]["rss"] / 1e6, 1) for w in W],
        "rss_after_warm_mb": [round(w["mem_after_warm"]["rss"] / 1e6, 1) for w in W],
        "sampler_rss_median_mb": [round(x / 1e6, 1) for x in sampler_rss],
        "pkl_bytes_mean": [w["pkl_bytes_mean"] for w in W if w["pkl_bytes_mean"]],
        "pickle_bytes_mean": [w["pickle_bytes_mean"] for w in W if w["pickle_bytes_mean"]],
        "cpu_import_setup_s": [round(w["cpu_import_setup_s"], 2) for w in W],
        "bg_top_before": S["bg_top_before"][:6], "bg_top_after": S["bg_top_after"][:6],
        "pin": W[0].get("pin", -1),
        "frac_sets_on_ecore": (lambda ids: round(sum(1 for i in ids if i >= 16) / len(ids), 3) if ids else None)([r["cpu_id"] for w in W for r in w["per_set"] if r.get("cpu_id", -1) >= 0]),
        "started": S["started"], "wall_s": round(S["wall_s"]), "env": W[0]["env"],
    }


def sizing(c, label):
    sustained = SETS_PER_S * c
    workers = math.ceil(1.5 * sustained)
    return {"label": label, "c_cpu_s_per_set": round(c, 3), "sustained_cpus": round(sustained, 2), "workers_1p5x": workers,
            "pod_vcpu_workers_plus_4": workers + 4, "total_cpu_hours_full_run": round(TOTAL_SETS * c / 3600.0, 1)}


def main():
    R = {}
    tags = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RUNS, "*")))
    A = {t: arm(t) for t in tags}
    A = {t: a for t, a in A.items() if a}
    R["arms"] = A
    # ---- primary c: three independent N=1 runs per mode
    def n1(prefix, pinned):
        return [a for t, a in sorted(A.items()) if a["nworkers"] == 1 and t.startswith(prefix) and t.endswith("_n1")
                and "retain" not in t and (a["pin"] >= 0) == pinned]
    cached, cached_p = n1("cached_r", False), n1("cached_r", True)
    build, build_p = n1("build_r", False), n1("build_r", True)
    single = n1("single_r", False)
    P = {}

    def prim(runs, sfx):
        if not runs:
            return
        P["c_label_only_parts_in_ram" + sfx] = spread([a["phase_cpu_per_set"]["score"] + a["phase_cpu_per_set"]["e8"] for a in runs]) if runs[0]["mode"] == "cached" else None
        if runs[0]["mode"] == "cached":
            P["c_score_plans_only" + sfx] = spread([a["phase_cpu_per_set"]["score"] for a in runs])
            P["c_executed_plans8_only" + sfx] = spread([a["phase_cpu_per_set"]["e8"] for a in runs])
            P["c_parts_load_from_disk" + sfx] = spread([a["phase_cpu_per_set"]["read"] + a["phase_cpu_per_set"]["decomp"] + a["phase_cpu_per_set"]["unpickle"] for a in runs])
            P["c_cached_load_plus_label_loop_total" + sfx] = spread([a["cpu_per_set_loop"] for a in runs])
            P["cached_runs" + sfx] = [a["tag"] for a in runs]
            P["cached_run_background_mean_logical" + sfx] = [a["background_logical_cpus"]["mean"] for a in runs]
            P["cached_run_frac_sets_on_ecore" + sfx] = [a["frac_sets_on_ecore"] for a in runs]
            P["cached_n_sets_per_run" + sfx] = [a["n_sets"] for a in runs]
            P["cached_wall_per_set_s" + sfx] = spread([a["loop_wall_max_s"] / a["n_sets"] for a in runs])
        else:
            P["c_build_parts_only" + sfx] = spread([a["phase_cpu_per_set"].get("parts", 0) + a["phase_cpu_per_set"].get("scn", 0) for a in runs])
            P["c_build_plus_label_loop_total" + sfx] = spread([a["cpu_per_set_loop"] for a in runs])
            P["build_runs" + sfx] = [a["tag"] for a in runs]
            P["build_run_background_mean_logical" + sfx] = [a["background_logical_cpus"]["mean"] for a in runs]
            P["build_run_frac_sets_on_ecore" + sfx] = [a["frac_sets_on_ecore"] for a in runs]
            P["build_n_sets_per_run" + sfx] = [a["n_sets"] for a in runs]
            P["build_wall_per_set_s" + sfx] = spread([a["loop_wall_max_s"] / a["n_sets"] for a in runs])

    prim(cached, "")
    prim(cached_p, "_pinned")
    prim(build, "")
    prim(build_p, "_pinned")
    for k in [k for k, v in P.items() if v is None]:
        del P[k]
    if single:
        P["c_single_build_plus_label_loop_total"] = spread([a["cpu_per_set_loop"] for a in single])
        P["single_n_sets_per_run"] = [a["n_sets"] for a in single]
    R["primary"] = P
    # ---- scaling
    def scal(prefix, ns, plan_tag):
        out = []
        base = None
        for n in ns:
            a = A.get(f"{prefix}_n{n}")
            if not a:
                continue
            row = {"N": n, "n_sets": a["n_sets"], "cpu_per_set": round(a["cpu_per_set_loop"], 4),
                   "loop_wall_max_s": round(a["loop_wall_max_s"], 1), "sets_per_s": round(a["sets_per_s_wall"], 3),
                   "busy_workers_equiv": round(a["busy_workers_equiv"], 2),
                   "rss_end_mb_median": round(float(np.median(a["rss_end_mb_per_worker"])), 1),
                   "peak_wset_mb_max": max(a["peak_wset_mb_per_worker"]),
                   "background_logical_cpus_mean": a["background_logical_cpus"]["mean"],
                   "machine_busy_logical_mean": a["machine_busy_logical_cpus"]["mean"]}
            if base is None:
                base = row
            row["cpu_per_set_vs_N1_pct"] = round(100.0 * (row["cpu_per_set"] / base["cpu_per_set"] - 1.0), 1)
            row["speedup_vs_N1"] = round(row["sets_per_s"] / base["sets_per_s"], 2)
            row["efficiency"] = round(row["speedup_vs_N1"] / n, 2)
            out.append(row)
        return out
    R["scaling_cached"] = scal("cached_r1", (1, 2, 4, 8), "cached_r1")
    # the N=1 baseline row of each scaling series is the matching primary run
    if "cached_r1_n1" in A:
        pass
    R["scaling_build"] = scal("build_r3", (1, 2, 4), "build_r3")
    # ---- retain probe (RAM per cached part)
    if "cached_r2_retain_n1" in A:
        L = load("cached_r2_retain_n1")[1][0]
        d = L["mem_end"]["rss"] - L["mem_after_warm"]["rss"]
        R["retain_probe"] = {"n_retained": L["n_sets"], "rss_growth_mb": round(d / 1e6, 1),
                             "bytes_per_part_in_ram": round(d / max(L["n_sets"], 1)),
                             "mb_per_part_in_ram": round(d / max(L["n_sets"], 1) / 1e6, 3)}
    allp = [x for a in cached for x in a["pkl_bytes_mean"]]
    allu = [x for a in cached for x in a["pickle_bytes_mean"]]
    if allp:
        R["part_disk"] = {"lzma_pkl_bytes_mean": round(float(np.mean(allp))), "uncompressed_pickle_bytes_mean": round(float(np.mean(allu))),
                          "navtrain_scenes_est": 103037, "est_cache_gb_navtrain_lzma": round(float(np.mean(allp)) * 103037 / 1e9, 1),
                          "evidence_class_of_estimate": "ESTIMATED (navtest mean size x D8's 103,037 scenes)"}
    if cached:
        R["rss_steady_state_cached_mb"] = spread([a["rss_end_mb_per_worker"][0] for a in cached])
        R["peak_wset_cached_mb"] = spread([a["peak_wset_mb_per_worker"][0] for a in cached])
        R["rss_after_setup_cached_mb"] = spread([a["rss_after_setup_mb"][0] for a in cached])
    if build:
        R["rss_steady_state_build_mb"] = spread([a["rss_end_mb_per_worker"][0] for a in build])
        R["peak_wset_build_mb"] = spread([a["peak_wset_mb_per_worker"][0] for a in build])
    # ---- chunk probe
    cp = os.path.join(HERE, "chunk_probe.json")
    if os.path.exists(cp):
        R["chunk_probe"] = json.load(open(cp, encoding="utf-8"))
    # ---- load dependence: per-set CPU vs the background load (other processes) during that set, N=1 runs that recorded it
    LD = {}
    for mode, runs_ in (("cached", cached), ("cached_pinned", cached_p), ("build", build), ("build_pinned", build_p)):
        xs, ys = [], []
        for a in runs_:
            L = load(a["tag"])
            for w in L[1]:
                for r in w["per_set"]:
                    if "bg_busy" in r:
                        xs.append(r["bg_busy"])
                        ys.append(r["cpu_set"])
        if len(xs) >= 50:
            x, y = np.asarray(xs), np.asarray(ys)
            k, c0 = np.polyfit(x, y, 1)
            bins = [(0, 6), (6, 10), (10, 14), (14, 18), (18, 25)]
            rows_ = []
            for lo, hi in bins:
                m = (x >= lo) & (x < hi)
                if m.sum() >= 5:
                    rows_.append({"bg_busy_logical": f"{lo}-{hi}", "n_sets": int(m.sum()), "mean_cpu_per_set": round(float(y[m].mean()), 4)})
            LD[mode] = {"n_sets": len(xs), "slope_cpu_s_per_set_per_bg_logical_cpu": round(float(k), 5), "intercept_cpu_s_per_set": round(float(c0), 4),
                        "corr": round(float(np.corrcoef(x, y)[0, 1]), 3), "bg_busy_mean": round(float(x.mean()), 2), "bins": rows_,
                        "note": "per-set windows are ~0.5-2.5 s, CPU quantised at the OS tick; the slope is an association, not a controlled experiment"}
    R["load_dependence"] = LD
    # ---- sizing by the spec's formula
    Z = []
    if cached:
        c0 = P["c_cached_load_plus_label_loop_total"]["median"]
        Z.append(sizing(c0, "ONLINE (D9 parts cached on disk, loaded per set): median of 3 runs"))
        Z.append(sizing(P["c_cached_load_plus_label_loop_total"]["max"], "ONLINE, worst of 3 runs"))
        Z.append(sizing(P["c_label_only_parts_in_ram"]["median"], "ONLINE, parts already in RAM (LRU hit) [lower bound]"))
    if cached_p:
        Z.append(sizing(P["c_cached_load_plus_label_loop_total_pinned"]["median"], "ONLINE, runs PINNED to a P-core thread: median"))
        Z.append(sizing(P["c_cached_load_plus_label_loop_total_pinned"]["max"], "ONLINE, runs PINNED to a P-core thread: worst run"))
    if build:
        Z.append(sizing(P["c_build_plus_label_loop_total"]["median"], "NO PRECOMPUTE (parts built per set, log-grouped): median of 3 runs"))
    if cached:
        for f in (1.5, 2.0):
            Z.append(sizing(P["c_cached_load_plus_label_loop_total"]["median"] * f, f"ONLINE x{f} (ESTIMATED slower pod vCPU than this dev-box core; NOT measured)"))
    R["sizing"] = Z
    R["spec_reference_rows"] = [sizing(1.78, "spec row 1.78 (INHERITED)"), sizing(5.0, "spec row 5 (pessimistic)"), sizing(18.0, "spec row 18")]
    json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    print(json.dumps({k: R[k] for k in ("primary", "scaling_cached", "scaling_build", "sizing") if k in R}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
