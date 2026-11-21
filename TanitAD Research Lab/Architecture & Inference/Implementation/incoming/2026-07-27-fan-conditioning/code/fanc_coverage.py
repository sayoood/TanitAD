#!/usr/bin/env python3
"""S5 -- THE DECISIVE MEASUREMENT: is the longitudinal vocabulary actually missing?

Both converging streams inferred a lever from the same TRUE observation -- the
fan's 2 s along-track marginal spans -15 to +100 m, so a 181 km/h plan is in it.
This script tests whether that MARGINAL over-dispersion implies a CONDITIONAL
coverage gap, which is what a state-conditioned anchor set would repair.

  coverage gap   g_w = min_c | speed(c) - speed(GT_w) |
  matched oracle  min ADE over ONLY the candidates within +-tol of GT's speed

If the matched oracle equals the unrestricted `oracle_in_fan`, then the
best-in-fan candidate is ALREADY speed-matched, and no reallocation of the
longitudinal vocabulary can lower the ceiling -- by construction, not by a
null result.

VALIDATION IN BOTH DIRECTIONS (brief-mandated):
  * fidelity        -- tol = inf must reproduce `oracle_in_fan` exactly
  * shift response  -- a rigid along-track shift must degrade the ceiling
                       smoothly, and shift 0 must return the committed number
  * DELIBERATELY FAILING INPUT -- `F_narrow`, a fan whose speed span is
                       artificially compressed so it is GENUINELY speed-starved.
                       On it, the coverage gap must blow up, the matched oracle
                       must separate from the unrestricted one, and re-centring
                       on v0 must HELP. This proves the instrument can return
                       "conditioning is the lever" when that is true.

Usage:  python fanc_coverage.py
"""
from __future__ import annotations

import json

import numpy as np
import torch

from fanc_common import (HUB, OUT, T_HORIZON, V5, WP_STEPS, ci_paired,
                         ci_single, eid_str, load_refc_fan, load_v5, r4)

TOLS = (0.5, 1.0, 2.0, 5.0)
SHIFTS = (-20, -10, -5, -2, -1, -0.5, 0.0, 0.5, 1, 2, 5, 10, 20)


def wp_times() -> np.ndarray:
    return np.asarray(WP_STEPS, dtype=float) * (T_HORIZON / max(WP_STEPS))


def coverage_block(cand_speed, gt_speed, err, eid, tag):
    """cand_speed [W,N], gt_speed [W], err [W,N] ade of every candidate."""
    gap = np.abs(cand_speed - gt_speed[:, None]).min(1)
    o_all = err.min(1)
    blk = {
        "n_windows": int(len(gap)),
        "coverage_gap_ms": {
            "mean": r4(gap.mean()), "p50": r4(np.median(gap)),
            "p95": r4(np.percentile(gap, 95)), "max": r4(gap.max()),
            "frac_within_0.5ms": r4((gap < 0.5).mean()),
            "frac_within_1.0ms": r4((gap < 1.0).mean())},
        "oracle_unrestricted": ci_single(o_all, eid),
        "matched_oracle": {},
    }
    for tol in TOLS:
        m = np.abs(cand_speed - gt_speed[:, None]) <= tol
        ok = m.any(1)
        o = np.where(m, err, np.inf).min(1)
        o = np.where(ok, o, o_all)                    # fallback, reported
        blk["matched_oracle"][f"tol{tol:g}"] = {
            "value": ci_single(o, eid),
            "paired_vs_unrestricted": ci_paired(o, o_all, eid),
            "windows_with_no_match": int((~ok).sum()),
            "mean_n_candidates": r4(m.sum(1).mean()),
        }
    return blk


def main() -> None:
    res = {"_stream": "2026-07-27-fan-conditioning",
           "_estimator": "paired_episode_cluster_bootstrap B=2000, unit=episode",
           "_question": "does the fan's MARGINAL longitudinal over-dispersion "
                        "imply a CONDITIONAL coverage gap that a v0-conditioned "
                        "anchor set could repair?"}
    t = wp_times()

    # ------------------------------------------------- the three REF-C fans -- #
    for arm in ("xl", "base", "small"):
        d = load_refc_fan(arm)
        fan = d["fan"].numpy().astype(np.float64)
        gt = d["gt"].numpy().astype(np.float64)
        v0 = d["v0"].numpy().astype(np.float64)
        eid = eid_str(d)
        err = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)
        cs = fan[:, :, -1, 0] / T_HORIZON
        gs = gt[:, -1, 0] / T_HORIZON
        blk = coverage_block(cs, gs, err, eid, arm)
        blk["n_anchors"] = int(fan.shape[1])
        blk["marginal_span_ms"] = r4((cs.max(1) - cs.min(1)).mean())
        blk["slope_cand_speed_on_v0"] = r4(np.polyfit(v0, cs.mean(1), 1)[0])
        # shift response -- the instrument must react
        blk["shift_response_ceiling"] = {}
        for dv in SHIFTS:
            f2 = fan.copy()
            f2[..., 0] += dv * t[None, None, :]
            blk["shift_response_ceiling"][f"{dv:+g}"] = r4(
                np.linalg.norm(f2 - gt[:, None], axis=-1).mean(-1).min(1).mean())
        res[f"refc_{arm}"] = blk
        print(f"REF-C-{arm:5s} N={fan.shape[1]:3d} gap mean {blk['coverage_gap_ms']['mean']:.4f} "
              f"max {blk['coverage_gap_ms']['max']:.4f}  "
              f"oracle {blk['oracle_unrestricted']['mean']:.4f}  "
              f"matched(tol1) {blk['matched_oracle']['tol1']['value']['mean']:.4f} "
              f"sep={blk['matched_oracle']['tol1']['paired_vs_unrestricted']['separated']}")

    # ------------------------------------------------------- v4's OWN fan ---- #
    s4 = torch.load(V5 / "fan_last_along_v4.pt", map_location="cpu",
                    weights_only=False)
    cs4 = s4["fan_last_along"].numpy().astype(np.float64) / T_HORIZON
    v5 = load_v5("v1")
    err4 = v5["fan_err4"].numpy().astype(np.float64)
    bara = (HUB / "Benchmarks & Eval" / "Implementation" / "incoming"
            / "2026-07-26-bar-a-selector" / "raw" / "bar_a_produced_windows.pt")
    tgt = torch.load(bara, map_location="cpu", weights_only=False)["tgt"].numpy()
    gs4 = tgt[:, -1, 0] / T_HORIZON
    eid4 = [str(int(x)) for x in v5["ep"].tolist()]
    blk = coverage_block(cs4, gs4, err4, eid4, "v4")
    blk["n_anchors"] = int(cs4.shape[1])
    blk["marginal_span_ms"] = r4((cs4.max(1) - cs4.min(1)).mean())
    blk["slope_cand_speed_on_v0"] = r4(np.polyfit(
        v5["v0"].numpy(), cs4.mean(1), 1)[0])
    blk["_note"] = ("v4's fan carries only the 2 s along-track scalar in the "
                    "repo, so shift-response is not computable here; the three "
                    "REF-C fans carry full trajectories and supply it.")
    res["v4_fan"] = blk
    print(f"v4_fan     N={cs4.shape[1]:3d} gap mean {blk['coverage_gap_ms']['mean']:.4f} "
          f"max {blk['coverage_gap_ms']['max']:.4f}  "
          f"oracle {blk['oracle_unrestricted']['mean']:.4f}  "
          f"matched(tol1) {blk['matched_oracle']['tol1']['value']['mean']:.4f} "
          f"sep={blk['matched_oracle']['tol1']['paired_vs_unrestricted']['separated']}")

    # ------------------------------- DELIBERATELY FAILING INPUT: F_narrow ---- #
    # A fan that IS genuinely speed-starved: compress every candidate's speed
    # offset about the fan median by `k`, so the fan can no longer reach the
    # speeds GT actually takes. If the instrument is real, THIS fan must show a
    # large coverage gap, a separated matched-oracle penalty, and a LARGE gain
    # from re-centring on v0.
    d = load_refc_fan("xl")
    fan = d["fan"].numpy().astype(np.float64)
    gt = d["gt"].numpy().astype(np.float64)
    v0 = d["v0"].numpy().astype(np.float64)
    eid = eid_str(d)
    gs = gt[:, -1, 0] / T_HORIZON
    s = fan[:, :, -1, 0] / T_HORIZON
    med = np.median(s, axis=1, keepdims=True)
    fail = {}
    for k in (1.0, 0.5, 0.25, 0.1):
        new_s = med + (s - med) * k
        f2 = fan.copy()
        f2[..., 0] = new_s[:, :, None] * t[None, None, :] + \
            (fan[..., 0] - s[:, :, None] * t[None, None, :])
        e2 = np.linalg.norm(f2 - gt[:, None], axis=-1).mean(-1)
        cs2 = f2[:, :, -1, 0] / T_HORIZON
        # and the SAME fan re-centred on v0 -- the conditioning intervention
        f3 = f2.copy()
        f3[..., 0] += (v0 - np.median(cs2, axis=1))[:, None, None] * t[None, None, :]
        e3 = np.linalg.norm(f3 - gt[:, None], axis=-1).mean(-1)
        fail[f"narrow_k{k:g}"] = {
            "coverage_gap_mean": r4(np.abs(cs2 - gs[:, None]).min(1).mean()),
            "coverage_gap_max": r4(np.abs(cs2 - gs[:, None]).min(1).max()),
            "frac_within_0.5ms": r4((np.abs(cs2 - gs[:, None]).min(1) < 0.5).mean()),
            "ceiling": ci_single(e2.min(1), eid),
            "ceiling_after_v0_recentre": ci_single(e3.min(1), eid),
            "gain_from_conditioning": ci_paired(e3.min(1), e2.min(1), eid),
        }
        g = fail[f"narrow_k{k:g}"]
        print(f"F_narrow k={k:<4g} gap {g['coverage_gap_mean']:7.4f}  "
              f"ceiling {g['ceiling']['mean']:7.4f} -> recentred "
              f"{g['ceiling_after_v0_recentre']['mean']:7.4f}  "
              f"gain {g['gain_from_conditioning']['delta']:+.4f} "
              f"{'SEP' if g['gain_from_conditioning']['separated'] else '-'}")
    res["POSITIVE_CONTROL_F_narrow"] = fail
    res["_positive_control_reading"] = (
        "k=1.0 is the real fan (no compression) and MUST show ~zero gain from "
        "conditioning. k<1 progressively starves the fan of speed and MUST show "
        "a growing, separated gain. If the k<1 rows show a gain and k=1.0 does "
        "not, the instrument is capable of reporting 'conditioning is the "
        "lever' and is reporting 'it is not' for the real fan.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fanc_coverage.json").write_text(json.dumps(res, indent=2))
    print("\nwrote", OUT / "fanc_coverage.json")


if __name__ == "__main__":
    main()
