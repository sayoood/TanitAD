#!/usr/bin/env python3
"""The MEASURED COST TABLE of SEARCH-NAVTEST-PIPE (both consumers: the navtest C2 look and the S0 bank).  CPU only; reads banked timing files and the fixture's own state; writes ../COST.md and ../raw/cost_navpipe.json.

Every rate carries its evidence class.  MEASURED = a timing in an artifact named in `sources`; ESTIMATED = a projection from a MEASURED rate (arithmetic shown in the table).  Nothing here is a navtest or S0 label, score or
search output: the sizes are token counts and the timings are wall-clock seconds of runs that already exist.

    python cost_navpipe.py [--fixture-state <work>/navpipe_state.json] [--out ../COST.md]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SPC_COST = "C:/Users/Admin/par18/spc/raw/cost.json"
HARNESS_COUNTS = "D:/Projects/TanitAD/data/refe_navtest/score/{r}/{r}.counts.json"
NAVTEST_PLAN = os.path.join(PKG, "raw", "navtest_chunk_plan_cap1000.json")
S0_COUNTS = os.path.join(PKG, "raw", "s0_chunk_plan_cap1000_counts.json")
CTX_B = 3_932_160
GiB = 2 ** 30
BRIDGE_S = 180.0                       # ESTIMATED: the certified bridge (l2fresh bridge_g2b) is ~3 min, 3 tokens x 2 arms; NOT measured for this builder
MIN_FREE = 6.0
O4_SEARCH_SCALE = 2.9                  # ESTIMATED, INHERITED: MM_RULING_DOSE.md / AMENDMENT_DOSE_DRAFT.md ("about 2.9x per token" for the O4 config: M 256 / elites 32 / widen 2.0 vs C8's 64 / 8 / 1.0); not re-measured here
T1_K = 24                              # tokens in the planted identity's control call


def lsq(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 4:                                            # a two- or three-point line is not a fit; the per-token means carry the projection
        return None
    A = np.stack([np.ones_like(x), x], 1)
    (a, b), res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = a + b * x
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(((y - yh) ** 2).sum()) / ss if ss > 0 else float("nan")
    return {"a_fixed_s": round(float(a), 2), "b_s_per_token": round(float(b), 4), "r2": round(r2, 3), "n_chunks": int(len(x)), "n_tokens_range": [int(x.min()), int(x.max())]}


def sha_throughput(mib=64, reps=8):
    buf = os.urandom(mib * 2 ** 20)
    t0 = time.time()
    for _ in range(reps):
        hashlib.sha256(buf).digest()
    return mib * reps / (time.time() - t0)            # MiB/s


def fixture_fits(state_path):
    if not state_path or not os.path.isfile(state_path):
        return None
    st = json.load(open(state_path))
    plan = json.load(open(os.path.join(os.path.dirname(state_path), "chunk_plan.json")))
    rows = []
    for cid, c in st["chunks"].items():
        s = c.get("seconds", {})
        if c.get("status") == "DELETED" and {"search", "score", "total"} <= set(s):
            rows.append((plan["chunks"][int(cid)]["n"], s["search"], s["score"], s["total"]))
    if not rows:
        return None
    n, se, sc, to = zip(*rows)
    return {"chunks_done": len(rows), "chunks_planned": len(plan["chunks"]), "tokens": list(n), "search_s": list(se), "score_s": list(sc), "total_s": list(to),
            "search_fit": lsq(n, se), "score_fit": lsq(n, sc), "total_fit": lsq(n, to), "other_s": [round(t - x - y, 1) for t, x, y in zip(to, se, sc)],
            "search_s_per_token_mean": round(sum(se) / sum(n), 3), "score_s_per_token_mean": round(sum(sc) / sum(n), 3)}


def wall(chunks, gpu_fixed, gpu_slope, cpu_fn, search_div, bridge, prefetch):
    """seconds for the whole run: sequential (GPU then CPU per chunk) or prefetched (the GPU of chunk k+1 overlaps the CPU of chunk k)"""
    g = [gpu_fixed + gpu_slope * n + (bridge if i == 0 else 0) for i, n in enumerate(chunks)]
    c = [cpu_fn(n, search_div) for n in chunks]
    if not prefetch:
        return sum(g) + sum(c)
    t = g[0]
    for i in range(len(chunks)):
        t += max(c[i], g[i + 1]) if i + 1 < len(chunks) else c[i]
    return t


def fmt_fit(f):
    return json.dumps(f) if f else "n/a (fewer than 4 chunks)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-state", default=os.path.join(PKG, "run", "fixture", "navpipe_state.json"))
    ap.add_argument("--out", default=os.path.join(PKG, "COST.md"))
    a = ap.parse_args(argv)
    sp = json.load(open(SPC_COST))
    mr = sp["measured_rates"]
    gpu_slope, gpu_fixed = mr["bundle_s_per_token_total_run"]["slope_from_per_log_timeline"], mr["bundle_s_per_token_total_run"]["fixed_s"]
    search_spc = mr["search_s_per_token_3_seeds_P2HEAD"]["mean"]
    label_spc = mr["label_s_per_token_5_plans"]
    label96 = mr["label_s_per_token_96_plans_upper_bound"]
    lb = mr["label_breakdown_s_400_tokens"]
    h = {r: json.load(open(HARNESS_COUNTS.format(r=r))) for r in ("refe_stage2_p2_R1", "refe_stage2_p2_full")}
    t1, t2 = h["refe_stage2_p2_R1"], h["refe_stage2_p2_full"]
    n1, n2, s1, s2 = int(t1["n_expected"]), int(t2["n_expected"]), float(t1["wall_s"]), float(t2["wall_s"])
    hslope = (s2 - s1) / (n2 - n1)
    hfixed = s1 - hslope * n1
    fx = fixture_fits(a.fixture_state)
    # rates used for the projections: the fixture's own search rate if it exists (MEASURED, this pipeline), else spc/cost.json; stated in the table
    search_rate = fx["search_s_per_token_mean"] if fx else search_spc
    search_src = f"fixture {fx['chunks_done']}/{fx['chunks_planned']} chunks, mean over its tokens (contended box)" if fx else "spc/cost.json mean of 20 shards"
    label_rate = fx["score_s_per_token_mean"] if fx else label_spc
    nav = json.load(open(NAVTEST_PLAN))
    nav_chunks = [c["n"] for c in nav["chunks"]]
    s0c = json.load(open(S0_COUNTS))
    s0_chunks = s0c["chunk_sizes"]
    thr = sha_throughput()
    ncpu = os.cpu_count()

    other = round(sum(fx["other_s"]) / len(fx["other_s"]), 1) if fx else 90.0
    other_src = f"fixture mean over {fx['chunks_done']} chunks (includes the fixture's bundle slicer, so slightly high)" if fx else "ESTIMATED 90 s"

    def t1_calls_s(n):
        """MM_RULING_C2AMEND.md DQ6 (b): 3 seed calls over the chunk, ONE call re-scoring P2's picks over the chunk (it also serves as the planted identity's pick rows), ONE control call of K = 24 tokens.
        Before DQ6 (b): 3 seed calls + 2 identity calls of K tokens.  The number of calls is the same (5); only the pick call grew from K to n tokens."""
        return 3 * (hfixed + hslope * n) + (hfixed + hslope * n) + (hfixed + hslope * min(T1_K, n))

    def t1_calls_before_dq6_s(n):
        return 3 * (hfixed + hslope * n) + 2 * (hfixed + hslope * min(T1_K, n))

    def cpu_navtest(n, div):
        return search_rate * n / div + t1_calls_s(n) + other

    def cpu_navtest_o4(n, div):
        return search_rate * O4_SEARCH_SCALE * n / div + t1_calls_s(n) + other

    def cpu_s0(n, div):
        return search_rate * n / div + label_rate * n + other

    out = {"sources": {"spc_cost": SPC_COST, "harness_counts": [HARNESS_COUNTS.format(r=r) for r in h], "navtest_plan": NAVTEST_PLAN, "s0_counts": S0_COUNTS, "fixture_state": a.fixture_state if fx else None},
           "rates": {"gpu_s_per_token": gpu_slope, "gpu_fixed_s_per_run": gpu_fixed, "search_s_per_token_used": search_rate, "search_source": search_src, "search_s_per_token_spc": search_spc,
                     "label_s_per_token_used": label_rate, "label_s_per_token_spc_5_plans": label_spc, "label_96_plans_upper_bound": label96, "harness_fixed_s": round(hfixed, 2), "harness_s_per_token": round(hslope, 5),
                     "sha256_MiB_per_s": round(thr), "cpu_count": ncpu}, "fixture": fx}
    rows = {}
    for name, chunks, cpu in (("navtest (C2, T1 route)", nav_chunks, cpu_navtest), ("navtest (C2, T1 route), O4 config", nav_chunks, cpu_navtest_o4), ("S0 bank (label route)", s0_chunks, cpu_s0)):
        N = sum(chunks)
        gpu_h = (gpu_fixed * len(chunks) + gpu_slope * N + BRIDGE_S) / 3600
        r = {"chunks": len(chunks), "tokens": N, "max_chunk": max(chunks), "ctx_total_GiB": round(N * CTX_B / GiB, 1), "ctx_peak_chunk_GiB": round(max(chunks) * CTX_B / GiB, 2), "gpu_h": round(gpu_h, 2),
             "search_cpu_h_1proc": round(search_rate * (O4_SEARCH_SCALE if "O4" in name else 1.0) * N / 3600, 2)}
        if "navtest" in name:
            r["score_h"] = round(sum(t1_calls_s(n) for n in chunks) / 3600, 2)
            r["score_h_before_dq6"] = round(sum(t1_calls_before_dq6_s(n) for n in chunks) / 3600, 2)
            r["dq6_extra_harness_min"] = round(sum(t1_calls_s(n) - t1_calls_before_dq6_s(n) for n in chunks) / 60, 1)
        else:
            r["score_h"] = round(label_rate * N / 3600, 2)
        for tag, div, pre in (("seq_1proc", 1, False), ("seq_2proc", 2, False), ("prefetch_1proc", 1, True), ("prefetch_2proc", 2, True)):
            r["wall_h_" + tag] = round(wall(chunks, gpu_fixed, gpu_slope, cpu, div, BRIDGE_S, pre) / 3600, 2)
        r["peak_disk_GiB_sequential"] = round(max(chunks) * CTX_B / GiB + MIN_FREE, 2)
        r["peak_disk_GiB_prefetch"] = round(2 * max(chunks) * CTX_B / GiB + MIN_FREE, 2)
        r["banked_GiB_total"] = round(N * (20_000 + (3_600 if "S0" in name else 0)) / GiB, 2)
        rows[name] = r
    out["table"] = rows
    out["fan_label_job_if_needed"] = {"tokens": sum(s0_chunks), "plans_per_token": 64, "s_per_token_upper_bound": label96, "h_upper_bound_1thread": round(label96 * sum(s0_chunks) / 3600, 2),
                                      "metric_cache_share": round(lb["seconds_metric_cache_parts_total"] / 400, 3), "s_per_plan": round(lb["seconds_label_plans_total"] / (400 * 5), 4)}
    json.dump(out, open(os.path.join(PKG, "raw", "cost_navpipe.json"), "w"), indent=1)

    L = []
    L.append("# COST TABLE, SEARCH-NAVTEST-PIPE (CPU only; generated by `code/cost_navpipe.py` from `raw/cost_navpipe.json`)\n")
    L.append("Stamps: wall-clock seconds of runs that already exist (MEASURED, artifact paths in `sources`) and arithmetic on them (ESTIMATED). No navtest or S0 label, score or search output was computed to make this table.\n")
    L.append("## 1. Rates\n")
    L.append("| quantity | value | class | source |\n|---|---|---|---|")
    L.append(f"| GPU bundle forward | {gpu_slope:.4f} s/token + {gpu_fixed:.1f} s per run (one lock hold per chunk) | MEASURED (carve B, 400 tokens, dev-box GPU) | spc/raw/cost.json `bundle_s_per_token_total_run`; step3.log ZZBUNDLE_OK 400 636 s |")
    L.append(f"| bridge (first chunk only) | {BRIDGE_S:.0f} s | ESTIMATED (certified bridge_g2b ~3 min; this builder's bridge has never run) | l2fresh/code/bridge_g2b.py |")
    L.append(f"| search P2HEAD, 3 seeds, `--threads 4`, 1 process | {search_spc:.2f} s/token | MEASURED (spc/cost.json, 20 shards, shared box) | l2stepb/run/logs/search_P2HEAD.log |")
    if fx:
        L.append(f"| search P2HEAD in THIS pipeline (carve-B fixture) | {fx['search_s_per_token_mean']:.2f} s/token mean ({fx['chunks_done']}/{fx['chunks_planned']} chunks); fit {fmt_fit(fx['search_fit'])} | MEASURED (fixture; box shared with the test suite and the mutation run) | {a.fixture_state} |")
        L.append(f"| label_stepB (5 plans/token), whole score stage | {fx['score_s_per_token_mean']:.2f} s/token mean; fit {fmt_fit(fx['score_fit'])} | MEASURED (fixture) | {a.fixture_state} |")
    L.append(f"| label_stepB (5 plans/token), 400 tokens | {label_spc:.3f} s/token; the metric cache is {lb['seconds_metric_cache_parts_total'] / 400:.2f} s/token of it, the {400 * 5} plan labels {lb['seconds_label_plans_total']:.1f} s in all | MEASURED | l2stepb/run/logs/label_P2HEAD.log ZZTIMING |")
    L.append(f"| T1 harness call | {hfixed:.1f} s fixed + {hslope:.4f} s/token | MEASURED (two banked calls: R1 and full) | D:/.../refe_stage2_p2_{{R1,full}}.counts.json |")
    L.append(f"| sha256 of ctx.npy | {thr:.0f} MiB/s ({CTX_B / 2 ** 20 * 1000 / thr:.0f} s per 1000-token chunk, twice per chunk) | MEASURED here (in memory) | this script |")
    L.append(f"| CPU threads on this box | {ncpu} | MEASURED | os.cpu_count() |\n")
    L.append("## 2. Whole-run projection (ESTIMATED: the rates above times the real chunk plans; the GPU lock wait is NOT counted, the B chain waited 2 h 20 min for it on 2026-10-19)\n")
    L.append("| | navtest C2 (T1 route), C8 | navtest C2, O4 (search x%.1f, ESTIMATED) | S0 bank (label route) |\n|---|---|---|---|" % O4_SEARCH_SCALE)
    a_, o_, b_ = rows["navtest (C2, T1 route)"], rows["navtest (C2, T1 route), O4 config"], rows["S0 bank (label route)"]
    for lab, k in (("chunks (cap 1000, whole logs)", "chunks"), ("tokens", "tokens"), ("largest chunk", "max_chunk"), ("ctx bytes, all chunks (never on disk at once) GiB", "ctx_total_GiB"), ("ctx peak, one chunk, GiB", "ctx_peak_chunk_GiB"),
                   ("GPU hours (bundle forward + bridge)", "gpu_h"), ("search CPU hours, 1 process", "search_cpu_h_1proc"), ("scoring hours (T1 harness, DQ6 (b): 3 seed calls + 1 pick re-score call over the chunk + 1 control call of K=24 per chunk / label_stepB)", "score_h"),
                   ("WALL h, sequential, 1 search process", "wall_h_seq_1proc"), ("WALL h, sequential, 2 search processes", "wall_h_seq_2proc"), ("WALL h, prefetch, 1 search process", "wall_h_prefetch_1proc"),
                   ("WALL h, prefetch, 2 search processes", "wall_h_prefetch_2proc"), ("PEAK DISK on the work volume, sequential, GiB (largest ctx + the 6 GiB guard floor)", "peak_disk_GiB_sequential"),
                   ("PEAK DISK with prefetch, GiB", "peak_disk_GiB_prefetch"), ("banked outputs kept, all chunks, GiB", "banked_GiB_total")):
        L.append(f"| {lab} | {a_[k]} | {o_[k]} | {b_[k]} |")
    L.append("\n- 2 search processes is ESTIMATED as a clean halving of the search term; it was NEVER measured, and `torch` thread contention is a known trap (TRAPS.md). The chain allows at most 2 and the fixture ran with 1.")
    L.append(f"- **DQ6 (b) (MM_RULING_C2AMEND.md): P2's picks are RE-SCORED through the same T1 route.** Extra harness time: **{a_['dq6_extra_harness_min']} min** over the 14 chunks (ESTIMATED: {hslope:.4f} s/token x (chunk tokens - {T1_K}) summed; the call count per chunk stays 5, the pick call just grows from {T1_K} tokens to the whole chunk), scoring {a_['score_h_before_dq6']} h -> {a_['score_h']} h. The draft's \"+18 min\" is the per-token term alone; no extra call overhead is added because the re-score REPLACES the planted identity's pick call.")
    L.append(f"- **O4 (MM_RULING_C2AMEND.md ADDITION): the GPU stage is unchanged (the forward is the same); the CPU search scales by about {O4_SEARCH_SCALE}x per token (ESTIMATED, INHERITED from MM_RULING_DOSE.md, not re-measured here)**: search CPU {a_['search_cpu_h_1proc']} h -> {o_['search_cpu_h_1proc']} h on 1 process, wall {a_['wall_h_seq_1proc']} h -> {o_['wall_h_seq_1proc']} h sequential. The Thor per-token latency scales the same way (UNMEASURED).")
    L.append("- The S0 row's GPU hours agree with the master plan's own figure (`srttc_s0/run/plan/NOTES.md`: 8,922 tokens at 1.44 s/token = 3.6 GPU-h).")
    L.append(f"- Per-chunk overhead outside bundle / search / score (gate = checkpoint + ctx load, identity, banking, hashing): {other} s, source: {other_src}.\n")
    f = out["fan_label_job_if_needed"]
    L.append("## 3. A cost this pipeline does NOT contain: P2's labelled FAN on the S0 tokens (the bank adapter reads `sub[N,64,6]`)\n")
    L.append("The S0 mode takes P2's fan labels as a PINNED INPUT (`fan_labels_sha256`) and refuses any token without a row. What I found (shapes and key names only): the S0 `<unit>.fan.npz` files hold P2's own fresh logits/pick but NO labels, and `<unit>.sets.npz` `labels6` belongs to the fit sets' own proposals. "
             "UNVERIFIED that a labelled fan of P2's FRESH forward exists for the 8,922 S0 tokens. If it does not, it is one more job: "
             f"{f['tokens']} tokens x 64 plans; the label route costs {f['metric_cache_share']} s/token for the metric cache plus {f['s_per_plan']} s/plan, and the L2-FRESH run labelled 96 plans/token at {f['s_per_token_upper_bound']} s/token, so at most "
             f"**{f['h_upper_bound_1thread']} h on one thread** (MEASURED rate, ESTIMATED total; CPU, no GPU).\n")
    if fx:
        L.append("## 4. The carve-B fixture, per chunk (MEASURED; the search and score are the same functions the real runs call)\n")
        L.append("| tokens | search s | score s | total s | other s |\n|---|---|---|---|---|")
        for n, se, sc, to, ot in zip(fx["tokens"], fx["search_s"], fx["score_s"], fx["total_s"], fx["other_s"]):
            L.append(f"| {n} | {se} | {sc} | {to} | {ot} |")
        L.append(f"\nFits (least squares, a + b n, shown only from 4 chunks): search {fmt_fit(fx['search_fit'])}, score {fmt_fit(fx['score_fit'])}, total {fmt_fit(fx['total_fit'])}. The per-token means above are what the projection uses.\n")
    open(a.out, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    print("ZZCOST_OK", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
