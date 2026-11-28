#!/usr/bin/env python3
"""Roundabout / junction / traffic-light prevalence arithmetic from the ONLY
in-corpus semantic labels PhysicalAI ships: `reasoning/ood_reasoning.parquet`.

⚠️ EVIDENCE CLASS. The rates below are MEASURED on the 1,740-clip OOD reasoning
subset and are an **ESTIMATED** projection when applied to the 306,152-clip
corpus. The subset is explicitly *"Out-of-Distribution driving scenarios"*
(card §Reasoning Labels) and is 49 % work-zones, so it is NOT a random sample.
Treat every projected count as an ESTIMATE with an unquantified selection bias,
and NEVER as a corpus roundabout count.

What it IS good for: (a) a same-instrument comparison of roundabout rate in
EU-collected vs US-collected clips, which is a *ratio* and so partially cancels
the OOD selection; (b) an order-of-magnitude sanity check on our own kinematic
screen (19 strict / 105 loose in 2,376).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = Path(os.environ.get(
    "PAI_PROBE_CACHE",
    r"C:\Users\Admin\AppData\Local\Temp\claude\pai_probe_cache"))

PATTERNS = {
    "roundabout": r"roundabout|traffic circle|rotary",
    "junction": r"intersection|junction|crossroad",
    "traffic_light": r"traffic light|traffic signal|red light|green light|stoplight",
    "stop_sign": r"stop sign",
    "yield": r"yield",
    "lane_change": r"lane change|change lane",
    "merge": r"merg",
    "turn": r"turn",
}
N_CORPUS = 306152
# ⚠️ The kinematic screen ran on the 3,000-clip PHASE-0 selection, not the
# 2,376-clip parity corpus (H2_SUBSTRATE_AND_LABELING.md §"all 3,000 phase-0
# clips, 485,401 windows"). Using 2,376 inflates every rate by 26 %.
OUR_N = 3000
PARITY_N = 2376
OUR_EU_FRAC = 0.916          # MEASURED, our_corpus_profile.json (n=500 local)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - s) / d, (c + s) / d)


def main():
    df = pd.read_parquet(CACHE / "reasoning" / "ood_reasoning.parquet")
    dc = pd.read_parquet(CACHE / "metadata" / "data_collection.parquet")
    for d in (df, dc):
        if d.index.names and any(n is not None for n in d.index.names):
            d.reset_index(inplace=True)
    j = df.merge(dc[["clip_id", "country", "hour_of_day"]], on="clip_id",
                 how="left")
    j["txt"] = j["events"].fillna("").astype(str)
    j["is_eu"] = j["country"].ne("United States") & j["country"].notna()

    out = {
        "EVIDENCE": "MEASURED on reasoning/ood_reasoning.parquet (n=1740); "
                    "ESTIMATED when projected to the corpus. NOT a random "
                    "sample — card calls it an OOD subset, 49% work-zones.",
        "n_reasoning_clips": int(len(j)),
        "n_matched_country": int(j["country"].notna().sum()),
        "eu_frac_in_reasoning_subset": round(float(j["is_eu"].mean()), 4),
        "event_cluster": {str(k): int(v) for k, v in
                          j["event_cluster"].value_counts().items()},
        "rates": {},
    }

    for name, pat in PATTERNS.items():
        m = j["txt"].str.contains(pat, case=False, regex=True)
        k, n = int(m.sum()), len(j)
        lo, hi = wilson(k, n)
        eu = j.loc[j["is_eu"], "txt"].str.contains(pat, case=False, regex=True)
        us = j.loc[~j["is_eu"], "txt"].str.contains(pat, case=False, regex=True)
        ek, en = int(eu.sum()), int(len(eu))
        uk, un = int(us.sum()), int(len(us))
        elo, ehi = wilson(ek, en)
        rate = {
            "k": k, "n": n, "rate": round(k / n, 5),
            "wilson95": [round(lo, 5), round(hi, 5)],
            "eu": {"k": ek, "n": en, "rate": round(ek / max(1, en), 5),
                   "wilson95": [round(elo, 5), round(ehi, 5)]},
            "us": {"k": uk, "n": un, "rate": round(uk / max(1, un), 5)},
            "eu_over_us_ratio": (round((ek / en) / (uk / un), 3)
                                 if uk and un and en else None),
            "ESTIMATED_corpus_count": int(round(k / n * N_CORPUS)),
            "ESTIMATED_count_in_our_2376_at_EU_rate":
                int(round((ek / max(1, en)) * OUR_N * OUR_EU_FRAC
                          + (uk / max(1, un)) * OUR_N * (1 - OUR_EU_FRAC))),
        }
        out["rates"][name] = rate
        print(f"{name:14s} {k:4d}/{n} = {100*k/n:5.2f}%  "
              f"[{100*lo:.2f},{100*hi:.2f}]  EU {100*ek/max(1,en):5.2f}% "
              f"US {100*uk/max(1,un):5.2f}%  ratio="
              f"{rate['eu_over_us_ratio']}  "
              f"-> est {rate['ESTIMATED_corpus_count']:6d} corpus, "
              f"{rate['ESTIMATED_count_in_our_2376_at_EU_rate']:4d} in our 2376",
              flush=True)

    # our kinematic screen, for the same-scale comparison
    out["our_kinematic_screen"] = {
        "EVIDENCE": "MEASURED (ours) — scratchpad/situ_full.py over the 3,000-clip "
                    "phase-0 selection, 485,401 windows; reported in "
                    "H2_SUBSTRATE_AND_LABELING.md. 19 strict / 105 loose.",
        "n_clips_screened": OUR_N,
        "strict": {"k": 19, "n": OUR_N, "rate": round(19 / OUR_N, 5),
                   "wilson95": [round(x, 5) for x in wilson(19, OUR_N)],
                   "thresholds": "|dpsi|>=180 deg, 6<=R<=25 m, monotone>0.90, "
                                 "3<=v<=11 m/s over 10 s"},
        "loose": {"k": 105, "n": OUR_N, "rate": round(105 / OUR_N, 5),
                  "wilson95": [round(x, 5) for x in wilson(105, OUR_N)],
                  "thresholds": "|dpsi|>=135 deg, R<=30 m, monotone>0.85 over 8 s"},
        "scaled_to_parity_2376": {
            "strict_expected": round(19 / OUR_N * PARITY_N, 1),
            "loose_expected": round(105 / OUR_N * PARITY_N, 1)},
    }

    # --- the convergence test: two independent instruments, same quantity ---
    rr = out["rates"]["roundabout"]
    out["CONVERGENCE_strict_screen_vs_reasoning_text"] = {
        "kinematic_strict_rate": round(19 / OUR_N, 5),
        "reasoning_text_rate": rr["rate"],
        "note": "Two INDEPENDENT instruments — a yaw/radius kinematic screen on "
                "3,000 clips and NVIDIA's human-verified CoC free text on 1,740 "
                "clips. Neither is a random geometric sample of the corpus. "
                "Agreement is evidence AGAINST the 'the corpus is roundabout-rich "
                "and our screen was the limiting factor' hypothesis.",
        "caveat": "The reasoning subset is OOD-curated (49% work zones), so its "
                  "roundabout rate is incidental-mention, not a geometry census. "
                  "Direction of bias is NOT established.",
    }
    # ------------------------------------------------------------------ #
    # The 4-brain program's "$0 arithmetic gate" on the AlpaSim pool.      #
    # ⚠️ The 356 figure is the number of keyframes SCREENED; only ~38      #
    # survivors were banked, and they were banked as a BALANCED design     #
    # (8/7/8/7/8). A balanced sample CANNOT estimate pool frequency — the  #
    # 8/38 = 21 % reading is inadmissible. The only usable statistic is    #
    # the screening proportion ~8 roundabouts found per 356 viewed.        #
    # ------------------------------------------------------------------ #
    POOL = 1606
    k_screen, n_screen = 8, 356
    lo_s, hi_s = wilson(k_screen, n_screen)
    out["alpasim_S4_gate"] = {
        "EVIDENCE": "MEASURED (screening note: 356 candidate keyframes viewed, "
                    "'roundabouts are rare (~2.5%)') + MEASURED (banked file "
                    "scaled_suite_labels.json = 38 entries, balanced design)",
        "banked_labels_file_n": 38,
        "banked_categories": {"roundabout": 8, "highway": 8, "straight_other": 8,
                              "traffic_light": 7, "intersection": 7},
        "WHY_8_over_38_IS_INADMISSIBLE":
            "the 38 are a deliberately BALANCED suite (~8/category), so 8/38 "
            "reflects the sampling design, not the pool. Projecting it "
            "(-> ~338 scenes) overstates the pool ~9x.",
        "screening_rate": {"k": k_screen, "n": n_screen,
                           "rate": round(k_screen / n_screen, 5),
                           "wilson95": [round(lo_s, 5), round(hi_s, 5)]},
        "pool_size": POOL,
        "projected_roundabout_scenes": {
            "point": int(round(k_screen / n_screen * POOL)),
            "ci95": [int(round(lo_s * POOL)), int(round(hi_s * POOL))]},
        "bar": 40,
        "VERDICT": None,        # filled below
    }
    g = out["alpasim_S4_gate"]
    pt = g["projected_roundabout_scenes"]["point"]
    lo_n, hi_n = g["projected_roundabout_scenes"]["ci95"]
    g["VERDICT"] = (
        f"MARGINAL / NOT POWERABLE with confidence: point estimate {pt} scenes "
        f"vs a bar of 40, 95% CI [{lo_n}, {hi_n}] — the interval STRADDLES the "
        f"bar, so the gate cannot be passed or failed on the banked evidence. "
        f"Deciding it requires classifying more of the 1,606-scene pool, which "
        f"is the cheap next step, not a corpus acquisition.")
    print("\n--- AlpaSim S4 $0 gate ---")
    print(f"screening {k_screen}/{n_screen} = {100*k_screen/n_screen:.2f}% "
          f"[{100*lo_s:.2f}, {100*hi_s:.2f}]")
    print(f"pool {POOL} -> {pt} roundabout scenes, CI [{lo_n}, {hi_n}], bar 40")
    print(g["VERDICT"])

    (HERE / "roundabout_arithmetic.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    r = out["rates"]["roundabout"]
    print("\n--- roundabout reconciliation ---")
    print(f"reasoning-label rate      {100*r['rate']:.3f}% "
          f"[{100*r['wilson95'][0]:.3f}, {100*r['wilson95'][1]:.3f}]")
    print(f"our kinematic STRICT      {100*19/OUR_N:.3f}% "
          f"[{100*wilson(19,OUR_N)[0]:.3f}, {100*wilson(19,OUR_N)[1]:.3f}]")
    print(f"our kinematic LOOSE       {100*105/OUR_N:.3f}% "
          f"[{100*wilson(105,OUR_N)[0]:.3f}, {100*wilson(105,OUR_N)[1]:.3f}]")
    print("[ra] done")


if __name__ == "__main__":
    main()
