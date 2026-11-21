#!/usr/bin/env python3
"""E-GOAL-2 -- the within-construction n = 40 -> n = 600 table, and the verdict.

⚠️ THE ONLY VALID CONTRAST IS WITHIN A CROSS-TRACK BACKGROUND. `EGOAL_2.md §4`
measures the recovery to span +13.3 % .. +29.2 % at FIXED n purely on which
learned cross-track sits in the background, so an n = 600 number under one
background may not be compared to an n = 40 number under another. This script
pairs each n = 600 cell with the n = 40 cell that differs from it ONLY in `n`,
which is what `PRE_REGISTRATION §1` actually registers.

It also reports the MEASURED CI half-width shrink factor against the
`MODEL_REGISTRY §1.2a` expectation of x2.8-3.9.

Run:  python e2_summary.py
"""
from __future__ import annotations

import json
from pathlib import Path

STREAM = Path(__file__).resolve().parent.parent
RAW = STREAM / "raw"

PAIRS = [("parent_resampled", "n40p2_parent_resampled", "n600_parent_resampled",
          "PRIMARY -- carries E-GOAL-1's cross error structure; conservative; "
          "NOT separated at n=40, so n genuinely decides"),
         ("sel", "n40p2_sel", "n600_sel",
          "secondary -- zero-fit background, provably identical at any n"),
         ("reduced", "n40p2_reduced", "n600_reduced",
          "secondary -- ridge on fan-only blocks, re-fit at each n")]
ARMS = ["E1_ego", "E1_L", "E1_L_X", "E1_L_X_D", "E1_nohist", "E1_noise_hist",
        "E0_v0", "CV", "P_ORACLE"]


def load(tag):
    p = RAW / f"e2_place_{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def hw(ci):
    return (ci["hi"] - ci["lo"]) / 2.0


def main():
    out = {"_stream": "2026-07-28-egoal-2-power",
           "_what": ("within-construction n=40 -> n=600, the contrast in which "
                     "ONLY n changes"),
           "backgrounds": {}}
    for bg, t40, t600, why in PAIRS:
        a, b = load(t40), load(t600)
        if a is None or b is None:
            continue
        blk = {"_role": why,
               "n40": {"windows": a["deployment"]["n_windows"],
                       "clusters": a["deployment"]["n_episode_clusters"],
                       "a0": a["deployment"]["a0_as_trained"],
                       "r_goal2s": a["deployment"]["r_goal2s_true_goal"],
                       "oracle_in_fan": a["deployment"]["oracle_in_fan"],
                       "headroom": a["deployment"]["headroom"]},
               "n600": {"windows": b["deployment"]["n_windows"],
                        "clusters": b["deployment"]["n_episode_clusters"],
                        "a0": b["deployment"]["a0_as_trained"],
                        "r_goal2s": b["deployment"]["r_goal2s_true_goal"],
                        "oracle_in_fan": b["deployment"]["oracle_in_fan"],
                        "headroom": b["deployment"]["headroom"]},
               "arms": {}}
        for arm in ARMS:
            if arm not in a["arms"] or arm not in b["arms"]:
                continue
            row = {}
            for m in ("iid", "by_speed"):
                pa, pb = (a["arms"][arm][m]["paired_vs_as_trained"],
                          b["arms"][arm][m]["paired_vs_as_trained"])
                row[m] = {
                    "rec40": a["arms"][arm][m]["recovery_of_headroom"],
                    "rec600": b["arms"][arm][m]["recovery_of_headroom"],
                    "d40": pa["delta"], "ci40": [pa["lo"], pa["hi"]],
                    "d600": pb["delta"], "ci600": [pb["lo"], pb["hi"]],
                    "hw40": round(hw(pa), 4), "hw600": round(hw(pb), 4),
                    "hw_shrink_x": round(hw(pa) / hw(pb), 2) if hw(pb) else None,
                    "sep40": a["arms"][arm][m]["separated"],
                    "sep600": b["arms"][arm][m]["separated"],
                    "verdict40": a["arms"][arm][m]["verdict"],
                    "verdict600": b["arms"][arm][m]["verdict"]}
            blk["arms"][arm] = row
        for k, src in (("n40", a), ("n600", b)):
            c = src["decorrelation_control"]["resampled"]
            blk[k]["decorrelation_control"] = {
                m: {"rec": c[m]["recovery_of_headroom"],
                    "verdict": c[m]["verdict"]} for m in ("iid", "by_speed")}
            if "family_matched_curve" in src:
                fc = src["family_matched_curve"]
                blk[k]["family_matched_curve"] = {
                    "sigma_0_m": fc["sigma_0_breakeven_along_rms_m"],
                    "sigma_50_m": fc["sigma_50_halfprize_along_rms_m"],
                    "monotone": fc["monotone"]}
        out["backgrounds"][bg] = blk

    # ---- the pre-registered verdict, evaluated mechanically ---------------
    prim = out["backgrounds"].get("parent_resampled")
    if prim and "E1_ego" in prim["arms"]:
        e = prim["arms"]["E1_ego"]
        better = {m: (e[m]["sep600"] and e[m]["d600"] < 0)
                  for m in ("iid", "by_speed")}
        if better["iid"] and better["by_speed"]:
            v = "CONFIRM"
        elif better["iid"]:
            v = "PARTIAL"
        else:
            v = "REFUTE"
        out["VERDICT"] = {
            "rule": ("CONFIRM = separated-better on BOTH resamplers; PARTIAL = "
                     "iid only; REFUTE = iid not separated-better"),
            "primary_background": "parent_resampled",
            "iid": {"rec": e["iid"]["rec600"], "d": e["iid"]["d600"],
                    "ci": e["iid"]["ci600"], "separated": e["iid"]["sep600"]},
            "by_speed": {"rec": e["by_speed"]["rec600"],
                         "d": e["by_speed"]["d600"],
                         "ci": e["by_speed"]["ci600"],
                         "separated": e["by_speed"]["sep600"]},
            "verdict": v}

    (RAW / "e2_summary.json").write_text(json.dumps(out, indent=1))

    for bg, blk in out["backgrounds"].items():
        print(f"\n===== background: {bg}  ({blk['_role']})")
        print(f"  n40 : {blk['n40']['windows']:6d} win / "
              f"{blk['n40']['clusters']:3d} clusters  a0={blk['n40']['a0']} "
              f"goal={blk['n40']['r_goal2s']} headroom={blk['n40']['headroom']}")
        print(f"  n600: {blk['n600']['windows']:6d} win / "
              f"{blk['n600']['clusters']:3d} clusters  a0={blk['n600']['a0']} "
              f"goal={blk['n600']['r_goal2s']} headroom={blk['n600']['headroom']}")
        print(f"  {'arm':14s} {'mode':9s} {'rec40':>7s} {'rec600':>7s} "
              f"{'hw40':>7s} {'hw600':>7s} {'x':>5s}  sep40 -> sep600")
        for arm, row in blk["arms"].items():
            for m in ("iid", "by_speed"):
                r = row[m]
                print(f"  {arm:14s} {m:9s} {100*r['rec40']:+6.1f}% "
                      f"{100*r['rec600']:+6.1f}% {r['hw40']:7.4f} "
                      f"{r['hw600']:7.4f} {str(r['hw_shrink_x']):>5s}  "
                      f"{str(r['sep40']):5s} -> {str(r['sep600']):5s} "
                      f"[{r['ci600'][0]:+.4f},{r['ci600'][1]:+.4f}] "
                      f"{r['verdict600']}")
        for k in ("n40", "n600"):
            print(f"  {k} decorrelation control:",
                  json.dumps(blk[k]["decorrelation_control"]))
            if "family_matched_curve" in blk[k]:
                print(f"  {k} family-matched curve:",
                      json.dumps(blk[k]["family_matched_curve"]))
    if "VERDICT" in out:
        print("\n*** PRE-REGISTERED VERDICT:", out["VERDICT"]["verdict"], "***")
        print(json.dumps(out["VERDICT"], indent=1))
    print(f"\n-> {RAW/'e2_summary.json'}")


if __name__ == "__main__":
    main()
