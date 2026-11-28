#!/usr/bin/env python3
"""GAP 2 stage 2 — the reachability-clamp ZERO-CHANGE test, on v4's OWN fan.

ZERO GPU.  Reads ``fan_v4fs_reduced.pt`` (written by ``v4_fan_dump.py``) and runs
exactly the test ``…/2026-07-27-percandidate-labels/code/t1_clip_and_fansize.py``
ran on REF-C-XL's fan, so the two are directly comparable.

⚠️ THE BAND COMES FROM PHYSICS AND FROM THE HEAD, NEVER FROM HELD-OUT ERROR.
``reach = sel_accel_max * horizons[-1] * 0.1`` = 2.5 m/s² × 2.0 s = 5.0 m/s --
the identical constant ``FlagshipV15Head.select`` already applies to the GOAL
(``flagship_v15.py:139,468``).  Nothing is tuned on ``ade``.

⛔ THE VERDICT RULE IS FIXED IN ADVANCE (PRE_REGISTRATION.md §3.1) AND IS NOT
RELAXABLE.  To flip ``V4Config.sel_reach_clamp`` to True the clamp must move the
pick on EXACTLY 0 windows, with paired Δ = 0.0000 [0.0000, 0.0000] and the
ADE-oracle surviving in 100 % of windows.  If the pick moves on even ONE window
the default STAYS OFF and this script says so.  The test is NOT re-run at a wider
band to make it pass -- a band chosen so the pick stops moving is a tuned
selector, not physics.

BOTH DIRECTIONS.  Δ = 0 is evidence only if the instrument CAN move the pick, so
the identical code is driven at ``accel_max ∈ {0.5, 0.2, 0.05}``, where the pick
MUST move and the ADE MUST get strictly worse.

⭐ It also decomposes each candidate's implied speed into VOCABULARY (its own
anchor's mean speed) + OFFSET-HEAD contribution, because the standing claim --
"the anchors are blameless, it is the unbounded offset head" -- is a claim about
REF-C-XL's fan and is testable on v4's.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                    "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
from taniteval import ci as _ci                                   # noqa: E402

SEL_ACCEL_MAX = 2.5          # V4Config.sel_accel_max
HORIZON_S = 2.0
B = 2000


def ade_of(fan_err, pick):
    return np.take_along_axis(fan_err, pick[:, None], 1)[:, 0]


def clamp_run(fe, logit, v_mean, v0, base_pick, eid, accel_max):
    reach = accel_max * HORIZON_S
    lo = np.maximum(v0 - reach, 0.0)[:, None]
    hi = (v0 + reach)[:, None]
    keep = (v_mean >= lo) & (v_mean <= hi)
    dead = ~keep.any(1)
    m = np.where(keep, logit, -np.inf)
    pick = m.argmax(1)
    pick[dead] = base_pick[dead]           # empty survivor set -> keep whole fan
    ade = ade_of(fe, pick)
    base_ade = ade_of(fe, base_pick)
    orc = fe.argmin(1)
    orc_kept = np.take_along_axis(keep, orc[:, None], 1)[:, 0]
    fe_masked = np.where(keep, fe, np.inf)
    pr = _ci.paired_episode_cluster_bootstrap(ade, base_ade, eid, n_boot=B,
                                              seed=0)
    moved = int((pick != base_pick).sum())
    return dict(
        accel_max=accel_max,
        band_mps=f"[max(0, v0-{reach}), v0+{reach}]",
        frac_candidates_removed=round(float(1.0 - keep.mean()), 4),
        frac_windows_with_empty_survivor_set=round(float(dead.mean()), 4),
        oracle_survives_frac=round(float(orc_kept.mean()), 4),
        oracle_ade_unfiltered=round(float(fe.min(1).mean()), 4),
        oracle_ade_after_clip=round(
            float(np.where(dead, fe.min(1), fe_masked.min(1)).mean()), 4),
        as_trained_ade=round(float(base_ade.mean()), 4),
        clipped_ade=round(float(ade.mean()), 4),
        windows_where_pick_moves=moved,
        n_windows=int(fe.shape[0]),
        paired_delta=round(float(pr["delta"]), 6),
        paired_ci95=[round(float(pr["lo"]), 6), round(float(pr["hi"]), 6)],
        paired_separated=bool(pr["lo"] > 0 or pr["hi"] < 0),
        miss_at_2m_as_trained=round(float((base_ade > 2.0).mean()), 4),
        miss_at_2m_clipped=round(float((ade > 2.0).mean()), 4),
        speedup_x=round(float(1.0 / max(1e-9, keep.mean())), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="fan_err4",
                    choices=("fan_err", "fan_err4"),
                    help="fan_err4 = the historical 4-waypoint (5,10,15,20) "
                         "convention every MODEL_REGISTRY row uses; fan_err = "
                         "the dense 20-waypoint ADE the head is trained on")
    a = ap.parse_args()

    d = torch.load(a.dump, map_location="cpu", weights_only=False)
    fe = d[a.metric].double().numpy()               # [W, N]
    fe_dense = d["fan_err"].double().numpy()
    logit = d["score"].double().numpy()             # [W, N]
    v_mean = d["v_mean"].double().numpy()           # [W, N]
    v0 = d["v0"].double().numpy()                   # [W]
    eid = [str(e) for e in d["eid"]]
    sel = d["sel"].numpy()
    anc_speed = d["anchor_mean_speed"].double().numpy()      # [N]
    W, N = fe.shape

    # ---- fidelity: the dump's own pick must be the argmax of its score -------
    base_pick = logit.argmax(1)
    fid = float((base_pick == sel).mean())
    if fid != 1.0:
        raise SystemExit(f"REFUSING: dumped sel_idx == argmax(score) on only "
                         f"{fid:.4%} of windows; the dumped score is not the "
                         f"score the head ranks on.")

    main_res = clamp_run(fe, logit, v_mean, v0, base_pick, eid, SEL_ACCEL_MAX)
    main_dense = clamp_run(fe_dense, logit, v_mean, v0, base_pick, eid,
                           SEL_ACCEL_MAX)

    # ---- BOTH DIRECTIONS: the instrument must be able to move the pick ------
    tight = {}
    for am in (0.5, 0.2, 0.05):
        r = clamp_run(fe, logit, v_mean, v0, base_pick, eid, am)
        r["MOVES_the_pick"] = r["windows_where_pick_moves"] > 0
        r["ADE_strictly_worse"] = r["clipped_ade"] > r["as_trained_ade"]
        tight[f"accel_max_{am}"] = r
    can_move = all(v["MOVES_the_pick"] for v in tight.values())
    gets_worse = all(v["ADE_strictly_worse"] for v in tight.values())

    # ---- vocabulary vs offset head -----------------------------------------
    off = v_mean - anc_speed[None, :]               # [W, N] m/s from refinement
    reach = SEL_ACCEL_MAX * HORIZON_S
    unreach = (v_mean > (v0 + reach)[:, None]) | (
        v_mean < np.maximum(v0 - reach, 0.0)[:, None])
    decomp = {
        "emitted_max_mean_speed_mps": round(float(v_mean.max()), 4),
        "emitted_p99_mean_speed_mps": round(float(np.quantile(v_mean, 0.99)), 4),
        "anchor_vocabulary_max_mean_speed_mps": round(float(anc_speed.max()), 4),
        "anchor_vocabulary_p99_mean_speed_mps": round(
            float(np.quantile(anc_speed, 0.99)), 4),
        "offset_head_contribution_mps": {
            "mean_abs": round(float(np.abs(off).mean()), 4),
            "p99_abs": round(float(np.quantile(np.abs(off), 0.99)), 4),
            "max_abs": round(float(np.abs(off).max()), 4)},
        "share_of_removed_candidates_whose_ANCHOR_is_already_unreachable":
            round(float(((anc_speed[None, :] > (v0 + reach)[:, None])
                         | (anc_speed[None, :]
                            < np.maximum(v0 - reach, 0.0)[:, None]))[unreach]
                        .mean()), 4),
        "note": "share ~1 means the clamp removes VOCABULARY, not offset-head "
                "excess -- the opposite of the REF-C-XL framing, and it must be "
                "reported either way",
    }

    verdict_ok = (main_res["windows_where_pick_moves"] == 0
                  and main_res["paired_delta"] == 0.0
                  and main_res["paired_ci95"] == [0.0, 0.0]
                  and main_res["oracle_survives_frac"] == 1.0
                  and can_move and gets_worse)

    out = {
        "experiment": "GAP 2 stage 2 - reachability-clamp zero-change test on "
                      "flagship-v4's OWN fan",
        "arm": d.get("arm"), "ckpt_step": int(d.get("step", -1)),
        "n_windows": W, "n_candidates": N,
        "n_episode_clusters": len(set(eid)),
        "metric": a.metric,
        "estimator": {"name": "paired episode-cluster bootstrap",
                      "module": "taniteval/taniteval/ci.py", "n_boot": B,
                      "seed": 0, "unit": "episode cluster",
                      "REFUSED": "overlapping_holdout_se"},
        "band": {"rule": "v_mean in [max(0, v0 - reach), v0 + reach], "
                         "reach = sel_accel_max * horizon = 2.5 * 2.0 = 5.0 m/s",
                 "provenance": "flagship_v15.py:139,468 - the head's OWN goal "
                               "clamp applied to the candidates; NOT tuned on "
                               "held-out error"},
        "fidelity_dumped_sel_is_argmax_of_dumped_score": fid,
        "MAIN_4wp_convention": main_res,
        "MAIN_dense_20wp": main_dense,
        "both_directions_tight_bands": tight,
        "both_directions_PASS": {"instrument_can_move_the_pick": can_move,
                                 "tight_band_makes_ADE_worse": gets_worse},
        "vocabulary_vs_offset_head": decomp,
        "PRE_REGISTERED_VERDICT": {
            "rule": "flip V4Config.sel_reach_clamp to True ONLY if pick moves "
                    "on 0 windows AND paired delta == 0.0000 [0,0] AND oracle "
                    "survives 100% AND the instrument demonstrably CAN move "
                    "the pick",
            "FLIP_THE_DEFAULT": bool(verdict_ok),
            "DEFAULT_STAYS_OFF": bool(not verdict_ok)},
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["MAIN_4wp_convention"], indent=1))
    print(json.dumps(out["vocabulary_vs_offset_head"], indent=1))
    print(json.dumps(out["PRE_REGISTERED_VERDICT"], indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
