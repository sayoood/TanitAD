#!/usr/bin/env python3
"""E-LAB-DATA-0901 -- cross-agreement of the two blind-built `g_tac` producers.

WHY THIS EXISTS. `LAB_BACKLOG` ranked row **7** (P0, D-DATA-GTAC-b): *"two producers of the
SAME label family are live with unmeasured agreement -- a silent fork in the canonical labels."*
The two are:

  A  `tanitad/data/tactical_goals.py`   (derive)
  B  `tanitad/data/g_tac_geom.py`       (g_tac_geom)

Both take the SAME contract -- poses `[T, 4]` = (x, y, yaw, v), a key/present index, and the
SAME 2-6 s tactical band -- so they are directly comparable without any adapter. That is worth
stating: it is what makes "agreement" a well-posed question at all.

WHAT THIS MEASURES, in two stages:

  S1 VOCABULARY (set arithmetic, no data needed). Which tokens can each producer EMIT?
     If the emittable sets barely intersect, "agreement" is undefined on most of the
     vocabulary and the backlog row's framing ("a silent fork") is the wrong model of the
     problem. This is mechanical and cannot be wrong.

  S2 BEHAVIOUR on the intersection (analytic trajectories). For the tokens BOTH can emit,
     do they emit the same one on the same poses?

⚠️ SCOPE, COMMITTED IN ADVANCE. S2 uses ANALYTIC trajectories, not the parity corpus. That is
enough to detect a DISAGREEMENT (if two producers differ on a clean decelerate-to-stop, they
differ) and enough to measure agreement ON THE INTERSECTION -- but it CANNOT settle a threshold
contest on real data, because the real corpus's distribution is what a threshold is tuned to.
Any threshold claim from this script would be out of scope and is not made.

CONTROLS (a probe without controls that read known values manufactures results -- CLAUDE.md):
  K1 NO-STOP    : a constant-speed straight run contains no stop. Any producer emitting
                  STOP_POINT here is WRONG, and the probe must be able to say so.
  K2 CLEAR-STOP : a clean decelerate-to-zero contains a stop at a known arc-length. A producer
                  that does NOT emit STOP_POINT here is WRONG.
  K3 DEGENERATE : sub-threshold creep (v < V_STOP_MS throughout) -- the ambiguous case; recorded,
                  not scored, so it cannot silently inflate or deflate the agreement rate.
  K1/K2 are the pair that makes the agreement number interpretable: without them, two producers
  that BOTH always abstain would score 100 % agreement.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch

from tanitad.data import g_tac_geom as G
from tanitad.data import tactical_goals as T

HZ = 10.0
DT = 1.0 / HZ
KEY = 10
N = 100


def _poses(vs, kappas):
    """Integrate a unicycle: per-step speed `vs` and curvature `kappas` -> [T,4] (x,y,yaw,v)."""
    x = y = yaw = 0.0
    out = []
    for v, k in zip(vs, kappas):
        out.append((x, y, yaw, v))
        yaw += v * k * DT
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
    return torch.tensor(out, dtype=torch.float32)


def scenarios():
    s = {}
    ones = np.ones(N)

    # K1 control: constant speed, dead straight. NO stop exists.
    s["K1_straight_cruise"] = dict(
        poses=_poses(10.0 * ones, 0.0 * ones),
        expect="no STOP_POINT from either producer", control=True)

    # K2 control: decelerate 10 -> 0, reaching zero at t = 5.0 s (index 50).
    v = np.clip(10.0 - 2.0 * (np.arange(N) * DT), 0.0, None)
    s["K2_decel_to_stop"] = dict(
        poses=_poses(v, 0.0 * ones),
        expect="STOP_POINT from any producer that claims the token", control=True)

    # K3 control: sub-threshold creep (below tactical_goals.V_STOP_MS = 0.5).
    s["K3_creep"] = dict(
        poses=_poses(0.3 * ones, 0.0 * ones),
        expect="degenerate; recorded, NOT scored", control=True)

    # Behavioural scenarios.
    s["S1_left_arc"] = dict(poses=_poses(8.0 * ones, 0.02 * ones),
                            expect="a sustained left turn", control=False)
    s["S2_sharp_junction_turn"] = dict(poses=_poses(6.0 * ones, 0.08 * ones),
                                       expect="junction-scale turn", control=False)

    # Lane change: lateral excursion ~3.2 m completed inside the band, heading returns.
    k = np.zeros(N)
    k[25:40] = 0.030
    k[40:55] = -0.030
    s["S3_lane_change"] = dict(poses=_poses(10.0 * ones, k),
                               expect="lateral offset then heading recovery", control=False)

    # Evade-and-return: smaller excursion that comes back.
    k2 = np.zeros(N)
    k2[30:38] = 0.020
    k2[38:46] = -0.020
    s["S4_evade_return"] = dict(poses=_poses(10.0 * ones, k2),
                                expect="in-corridor nudge", control=False)

    # Stop and go: decelerate to 0, hold ~1 s, accelerate away.
    v2 = np.concatenate([np.clip(10.0 - 2.5 * (np.arange(40) * DT), 0, None),
                         np.zeros(10), np.clip(2.5 * (np.arange(50) * DT), 0, 10.0)])[:N]
    s["S5_stop_and_go"] = dict(poses=_poses(v2, 0.0 * ones),
                               expect="brief hold then proceed", control=False)
    return s


def main():
    out = {"meta": {
        "what_this_is": ("cross-agreement of tactical_goals.derive (A) vs g_tac_geom (B) "
                         "on an IDENTICAL poses[T,4] contract and band. Analytic "
                         "trajectories -- detects disagreement, does NOT settle thresholds "
                         "on the real corpus."),
        "hz": HZ, "key": KEY, "n_samples": N,
        "band_s_A": [T.BAND_LO_S, T.BAND_HI_S], "band_s_B": list(G.TAC_BAND_S),
    }}

    # ---------- S1: VOCABULARY (mechanical set arithmetic) ------------------------------
    a_lat, a_lon = set(T.LAT_TOKENS), set(T.LON_TOKENS)
    refused = set(G.REFUSED)
    # B's emittable set = its declared tokens minus everything it REFUSES.
    b_emit = {"STOP_POINT", "LON_UNCONSTRAINED", "LAT_UNCONSTRAINED",
              "CORRIDOR_OFFSET", G.ABSTAIN} - refused
    out["S1_vocabulary"] = {
        "A_lat_tokens": sorted(a_lat), "A_lon_tokens": sorted(a_lon),
        "B_refuses": sorted(refused),
        "B_emittable_default_arm": sorted(b_emit),
        "A_lat_refused_by_B": sorted(a_lat & refused),
        "A_lon_refused_by_B": sorted(a_lon & refused),
        "A_lat_survivors": sorted(a_lat - refused),
        "A_lon_survivors": sorted(a_lon - refused),
    }

    # ---------- S2: BEHAVIOUR ------------------------------------------------------------
    rows = {}
    for name, sc in scenarios().items():
        p = sc["poses"]
        rec = {"expect": sc["expect"], "is_control": sc["control"]}
        try:
            ga = T.derive(p, key=KEY, hz=HZ)
            rec["A"] = {"lat": ga.lat_token, "lon": ga.lon_token,
                        "lat_reason": ga.lat_provenance,
                        "lon_reason": ga.lon_provenance,
                        "lat_args": {k: (round(float(v), 4)
                                         if isinstance(v, (int, float)) else v)
                                     for k, v in (ga.lat_args or {}).items()},
                        "truncated": bool(ga.truncated)}
        except Exception as e:
            rec["A"] = {"error": f"{type(e).__name__}: {e}"}
        try:
            gb = G.g_tac_geom(p, KEY)
            rec["B"] = {"lat": gb.lat.token, "lon": gb.lon.token,
                        "lat_reason": (gb.lat.reason or "")[:120],
                        "lon_args": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                                     for k, v in (gb.lon.args or {}).items()}}
        except Exception as e:
            rec["B"] = {"error": f"{type(e).__name__}: {e}"}

        al = rec["A"].get("lon"); bl = rec["B"].get("lon")
        rec["lon_match"] = (al == bl) if (al and bl) else None
        rec["A_stop"] = (al == "STOP_POINT")
        rec["B_stop"] = (bl == "STOP_POINT")
        rows[name] = rec
        print(f"{name:24s} A(lat={rec['A'].get('lat')}, lon={al})  "
              f"B(lat={rec['B'].get('lat')}, lon={bl})", flush=True)
    out["S2_behaviour"] = rows

    # ---------- control verdicts ----------------------------------------------------------
    k1, k2 = rows["K1_straight_cruise"], rows["K2_decel_to_stop"]
    out["controls"] = {
        "K1_no_stop": {"A_emitted_stop": k1["A_stop"], "B_emitted_stop": k1["B_stop"],
                       "pass": (not k1["A_stop"]) and (not k1["B_stop"]),
                       "note": "a constant-speed straight run has no stop; either TRUE is a defect"},
        "K2_clear_stop": {"A_emitted_stop": k2["A_stop"], "B_emitted_stop": k2["B_stop"],
                          "pass": k2["A_stop"] and k2["B_stop"],
                          "note": "a clean decel-to-zero has a stop; either FALSE is a defect"},
        "K3_degenerate_recorded_not_scored": rows["K3_creep"]["lon_match"],
    }
    scored = {n: r for n, r in rows.items() if n != "K3_creep" and r["lon_match"] is not None}
    agree = sum(1 for r in scored.values() if r["lon_match"])
    out["agreement"] = {
        "lon_axis_scored_n": len(scored), "lon_axis_agree_n": agree,
        "lon_axis_agree_frac": round(agree / len(scored), 4) if scored else None,
        "lat_axis": ("UNDEFINED -- B abstains on LAT by design (default arm), so there is no "
                     "LAT comparison to make. This is a DECLARED refusal, not a missing value."),
    }

    dst = sys.argv[1] if len(sys.argv) > 1 else "gtac_cross_agreement.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
