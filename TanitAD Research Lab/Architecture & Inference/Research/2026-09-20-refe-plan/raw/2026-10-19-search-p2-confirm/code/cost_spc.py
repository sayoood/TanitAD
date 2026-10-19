#!/usr/bin/env python3
"""SEARCH-P2-CONFIRM deliverable 2: the cost of the chain, from the banked timing lines of Step B, L2-FRESH's chain and L1 Step A.  Reads TIMING lines and token / log COUNTS only; it opens no label and no outcome.

Every projection is ESTIMATED; every rate it multiplies is MEASURED (artifact named in `sources`).
    CUDA_VISIBLE_DEVICES=-1 python cost_spc.py --out ../raw/cost.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st

STEPB = "C:/Users/Admin/par18/l2stepb/run"
FRESH_CHAIN = "C:/Users/Admin/par18/l2fresh/chain_B.log"
FRESH_STEP3 = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/step3.log"
L1_STEP3 = "C:/Users/Admin/par18/l1run/step3.log"
NAVTEST_TOKENS = "D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json"
NAVTEST_COUNTS = "D:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p2_full/refe_stage2_p2_full.counts.json"
CTX_BYTES_PER_TOKEN = 7680 * 256 * 2                     # uint16 (bf16 bits), ctx.json shape [N, 7680, 256]
N_B_REST = 905
N_CARVE_B = 400


def shard_rates(path):
    r = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"ZZSHARD \S+ (\d+)-(\d+) (\d+)s tokens=(\d+) t/token=([\d.]+)s", line)
        if m:
            r.append(float(m.group(5)))
    return r


def bundle_lines(path):
    """-> (total_s of the ZZBUNDLE_OK line, [(done_tokens, seconds)] per log)"""
    tot, per = None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"ZZBUNDLE_OK (\d+) (\d+) s", line)
        if m:
            tot = (int(m.group(1)), int(m.group(2)))
        m = re.search(r"\[(\d+)/24\] \S+ done (\d+)/\d+ bad \d+ (\d+) s", line)
        if m:
            per.append((int(m.group(2)), int(m.group(3))))
    return tot, per


def label_timing(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("ZZTIMING"):
            return json.loads(line[len("ZZTIMING"):].strip())
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rates = {arm: shard_rates(f"{STEPB}/logs/search_{arm}.log") for arm in ("P2HEAD", "L2s0", "L2s1")}
    p2 = rates["P2HEAD"]
    allr = [x for v in rates.values() for x in v]
    (nB, tB), perB = bundle_lines(FRESH_STEP3)
    (nA, tA), _ = bundle_lines(L1_STEP3)
    # per-token slope of the bundle forward from the per-log timeline (first log pays the model load; the last line to ZZBUNDLE_OK is the save)
    (t0, s0), (t1, s1) = perB[0], perB[-1]
    slope = (s1 - s0) / (t1 - t0)
    fixed = tB - slope * nB
    lt = label_timing(f"{STEPB}/logs/label_P2HEAD.log")
    nav = json.load(open(NAVTEST_TOKENS))
    n_nav = len(nav["tokens"])
    logs_nav = len(set(nav["token_log"][t] for t in nav["tokens"]))
    nav_counts = json.load(open(NAVTEST_COUNTS))

    def bundle_min(n):
        lo = min(tB / nB, tA / nA) * n / 60.0
        hi = max(tB / nB, tA / nA) * n / 60.0
        return {"low_min": lo, "high_min": hi, "slope_model_min": (fixed + slope * n) / 60.0}

    s_tok_mean, s_tok_med = st.mean(p2), st.median(p2)
    s_hi, s_lo = max(allr), min(allr)
    lab_tok = lt["seconds_per_token_all_in"]
    # label rate bound: L2-FRESH's 96-plan labeller (719 s / 400 tokens) is the upper bound for the 5-plan labeller
    lab_hi_tok = 719.0 / 400.0
    out = {
        "evidence": "rates MEASURED (artifacts in `sources`); every projection ESTIMATED (no run on B-REST or navtest exists)",
        "sources": {"search_rates": f"{STEPB}/logs/search_{{P2HEAD,L2s0,L2s1}}.log (ZZSHARD lines, CPU 4 threads, shared box, 20 tokens x 3 seeds per shard)",
                    "bundle_forward": f"{FRESH_STEP3} (ZZBUNDLE_OK {nB} {tB} s) and {L1_STEP3} (ZZBUNDLE_OK {nA} {tA} s); the dev-box GPU under the lock",
                    "labels": f"{STEPB}/logs/label_P2HEAD.log (ZZTIMING, navsim venv, 1 thread, 5 plans / token) ; upper bound = L2-FRESH chain_B.log ZZLABEL_OK 400/400 719 s at 96 plans / token",
                    "navtest_counts": f"{NAVTEST_TOKENS} ({n_nav} tokens, {logs_nav} logs); harness wall {nav_counts.get('wall_s')} s for {nav_counts.get('n_expected')} tokens ({NAVTEST_COUNTS})"},
        "sizes": {"B_REST_tokens": N_B_REST, "B_REST_logs": 24, "navtest_tokens": n_nav, "navtest_logs": logs_nav,
                  "ctx_bytes_per_token": CTX_BYTES_PER_TOKEN, "ctx_GiB_B_REST": N_B_REST * CTX_BYTES_PER_TOKEN / 2 ** 30, "ctx_GiB_navtest": n_nav * CTX_BYTES_PER_TOKEN / 2 ** 30,
                  "disk_note": "free at measurement: D: 24 GB (exFAT, 1 MiB clusters), C: 7.3 GB; the B-REST ctx.npy (3.3 GiB, ONE file) fits on D:; the navtest ctx (44.5 GiB) does NOT, so a navtest look needs a chunked bundle->search->delete pipeline (not built)"},
        "measured_rates": {"search_s_per_token_3_seeds_P2HEAD": {"mean": s_tok_mean, "median": s_tok_med, "min_shard": min(p2), "max_shard": max(p2), "n_shards": len(p2)},
                           "search_s_per_token_all_three_arms": {"min": s_lo, "max": s_hi, "n_shards": len(allr)},
                           "bundle_s_per_token_total_run": {"carve_B": tB / nB, "stepA": tA / nA, "slope_from_per_log_timeline": slope, "fixed_s": fixed},
                           "label_s_per_token_5_plans": lab_tok, "label_s_per_token_96_plans_upper_bound": lab_hi_tok,
                           "label_breakdown_s_400_tokens": {k: lt[k] for k in ("seconds_scenario_build_total", "seconds_metric_cache_parts_total", "seconds_label_plans_total")},
                           "navtest_harness_s_per_token_one_plan": nav_counts.get("wall_s") / nav_counts.get("n_expected")},
        "a_bundle_forward_B_REST": {"unit": "GPU minutes, ONE locked forward (ESTIMATED)", **bundle_min(N_B_REST),
                                    "plus": "ram_gate needs >= 8 GiB free RAM; the lock wait is NOT counted (B chain waited 2 h 20 min for the lock on 2026-10-19)", "tokens": N_B_REST},
        "b_search_B_REST": {"unit": "CPU wall minutes at 4 threads, 1 process (ESTIMATED); x4 = thread-minutes", "tokens": N_B_REST, "seeds": 3,
                            "central_min": N_B_REST * s_tok_med / 60.0, "mean_rate_min": N_B_REST * s_tok_mean / 60.0, "low_min": N_B_REST * s_lo / 60.0, "high_min": N_B_REST * s_hi / 60.0,
                            "thread_minutes_central": 4 * N_B_REST * s_tok_med / 60.0},
        "c_labels_B_REST": {"unit": "CPU wall minutes, 1 thread (ESTIMATED)", "tokens": N_B_REST, "low_min": N_B_REST * lab_tok / 60.0, "high_min": N_B_REST * lab_hi_tok / 60.0,
                            "note": "5 plans / token (identity, 3 seeds, control); the planted identity at the labeller is part of this"},
        "d_navtest": {"tokens": n_nav, "logs": logs_nav,
                      "bundle_forward_gpu_hours": {"low": bundle_min(n_nav)["low_min"] / 60.0, "high": bundle_min(n_nav)["high_min"] / 60.0,
                                                   "caveat": "UNVERIFIED: the held-out bundle builder reproduces a BANKED fan (G2b); no banked fan table exists for the 12,146 navtest tokens, so the navtest path needs the bridge (L2-FRESH `bridge_g2b.py`) or a new builder; the rate is the plan-token rate"},
                      "search_cpu_hours_1_process_4_threads": {"central": n_nav * s_tok_med / 3600.0, "low": n_nav * s_lo / 3600.0, "high": n_nav * s_hi / 3600.0, "two_processes_wall_central": n_nav * s_tok_med / 3600.0 / 2.0},
                      "labels_cpu_hours_via_label_stepB_1_process": {"low": n_nav * lab_tok / 3600.0, "high": n_nav * lab_hi_tok / 3600.0},
                      "labels_hours_via_harness_seam_route": {"per_seed_set_s": nav_counts.get("wall_s"), "three_seed_sets_hours": 3 * nav_counts.get("wall_s") / 3600.0,
                                                              "note": "ESTIMATED: the NAVSIM harness scored P2's full navtest seam in 1,106.6 s with the prebuilt navtest metric cache; a seam file built from the search outputs would reuse that route. UNVERIFIED: no builder exists, and the label-path identity (the pick reproduces its banked CSV row) must be shown first"},
                      "ctx_disk_GiB": n_nav * CTX_BYTES_PER_TOKEN / 2 ** 30,
                      "sequential_total_hours_central": bundle_min(n_nav)["low_min"] / 60.0 + n_nav * s_tok_med / 3600.0 + 3 * nav_counts.get("wall_s") / 3600.0,
                      "note": "ESTIMATED; the navtest look comes ONLY after EP-ADMISSION admits, and needs a chunked pipeline (disk) that is not built"},
        "total_B_REST_chain": {"gpu_minutes": bundle_min(N_B_REST)["low_min"], "gpu_minutes_high": bundle_min(N_B_REST)["high_min"],
                               "cpu_wall_minutes_search_plus_labels_central": N_B_REST * s_tok_med / 60.0 + N_B_REST * lab_tok / 60.0,
                               "cpu_wall_minutes_high": N_B_REST * s_hi / 60.0 + N_B_REST * lab_hi_tok / 60.0,
                               "plus_C_ID_gate_16_tokens_cpu_minutes": 16 * s_tok_med / 60.0 + 1.0, "plus_G_REACH_per_arm_minutes": 1.0},
    }
    json.dump(out, open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    t = out["total_B_REST_chain"]
    print("ZZCOST_OK gpu_min %.1f-%.1f  search_central_min %.1f  labels_min %.1f-%.1f  navtest_search_h %.1f" % (
        out["a_bundle_forward_B_REST"]["low_min"], out["a_bundle_forward_B_REST"]["high_min"], out["b_search_B_REST"]["central_min"],
        out["c_labels_B_REST"]["low_min"], out["c_labels_B_REST"]["high_min"], out["d_navtest"]["search_cpu_hours_1_process_4_threads"]["central"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
