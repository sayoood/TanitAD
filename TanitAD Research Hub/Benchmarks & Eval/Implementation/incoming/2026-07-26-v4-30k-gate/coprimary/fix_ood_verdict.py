#!/usr/bin/env python3
"""fix_ood_verdict.py — recompute the OOD/EXTRAPOLATION verdict with E1a's FULL
rule, from the persisted per-window tensors. No GPU, no re-run.

THE DEFECT THIS FIXES (found by reading the output, not by assuming it)
----------------------------------------------------------------------
``v4_corridor_cl.ood_block`` implemented only HALF of E1a's stated rule::

    "any horizon whose peak OOD ratio exceeds ~1.5x, OR WHOSE STEPS LEAVE THE
     MEASURED ENVELOPE, is EXTRAPOLATION, not measurement."   (e1a_horizon.py:28-30)

Only the ratio half was tested, so a horizon at K=185 emitted "within the
measured envelope on average" while **54.6 % of its steps had |XTE| > 3.0 m**.
That reading is not merely incomplete, it is backwards: ``OODMap.ratio_arr``
uses ``np.interp``, which **CLAMPS** at the envelope edge, so once steps leave
the envelope the ratio SATURATES and the 1.5x criterion **structurally cannot
fire**. The ratio criterion is uninformative exactly when it matters most.

The verdict below is the disjunction, with each half reported separately so the
reader can see which one fired and why the other could not.

``EXTRAPOLATION_MAJORITY_FRAC`` = 0.5 is a REPORTING CONVENTION (PROPOSED, not
measured): it only decides whether the verdict string says EXTRAPOLATION or
PARTIAL EXTRAPOLATION. Both underlying fractions are always printed, so no
verdict depends on it being right.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

for _p in ("/workspace/_v4gate", "/root/TanitAD/stack",
           "/root/TanitAD/stack/scripts", "/root/taniteval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import ci as _ci                                # noqa: E402
from taniteval import corridor as _corr                        # noqa: E402
from v4_corridor_cl import ENV_LAT_MAX, ENV_YAW_MAX, OODMap    # noqa: E402

MAJORITY = 0.5     # PROPOSED reporting convention — see the module docstring


def block(lat, yaw, eid, ood, K):
    if lat.shape[0] < 2 or len(set(eid)) < 2:
        return None
    ratio = ood.ratio_arr(lat, yaw)
    bo = lambda x: _ci.episode_cluster_bootstrap(np.asarray(x, float), eid,  # noqa: E731
                                                 n_boot=2000)
    f_steps_lat = float((lat > ENV_LAT_MAX).mean())
    f_steps_yaw = float((yaw > ENV_YAW_MAX).mean())
    f_win = float(((lat > ENV_LAT_MAX) | (yaw > ENV_YAW_MAX)).any(1).mean())
    peak = bo(ratio.max(1))
    ratio_fires = bool(peak["mean"] > 1.5)
    env_fires = bool(f_win > 0.0)
    if ratio_fires or f_win > MAJORITY:
        verdict = "EXTRAPOLATION — NOT a measurement at this horizon"
    elif env_fires:
        verdict = "PARTIAL EXTRAPOLATION — a minority of windows leave the envelope"
    else:
        verdict = "MEASUREMENT — every step stayed inside the MEASURED envelope"
    return {
        "horizon_K": int(K), "horizon_s": round(K * 0.1, 2),
        "n_windows": int(lat.shape[0]), "n_episodes": int(len(set(eid))),
        "ood_peak_ratio": peak,
        "ood_mean_ratio": bo(ratio.mean(1)),
        "frac_windows_ood_peak_under_1p16": round(float((ratio.max(1) <= 1.16).mean()), 4),
        "frac_windows_ood_peak_under_1p5": round(float((ratio.max(1) <= 1.5).mean()), 4),
        "EXTRAPOLATION_frac_steps_lat_over_3m": round(f_steps_lat, 5),
        "EXTRAPOLATION_frac_steps_yaw_over_12deg": round(f_steps_yaw, 5),
        "EXTRAPOLATION_frac_windows_any_step_out_of_envelope": round(f_win, 4),
        "criterion_1_ratio_over_1p5": {
            "fires": ratio_fires, "peak_ratio_mean": round(peak["mean"], 4),
            "_why_it_may_be_uninformative": (
                "OODMap.ratio_arr interpolates with np.interp, which CLAMPS at "
                "|dlat|=3.0 m / |dyaw|=12 deg. Once steps leave the envelope the "
                "ratio SATURATES, so this criterion structurally cannot fire "
                "there and the reported ratio is a LOWER BOUND.")},
        "criterion_2_steps_outside_measured_envelope": {
            "fires": env_fires,
            "frac_steps_lat_over_3m": round(f_steps_lat, 5),
            "frac_steps_yaw_over_12deg": round(f_steps_yaw, 5),
            "frac_windows_any_step_outside": round(f_win, 4)},
        "EXTRAPOLATION_VERDICT": verdict,
        "_rule": ("E1a's FULL disjunction (e1a_horizon.py:28-30): ratio > ~1.5x "
                  "OR steps leave the MEASURED envelope. The earlier emission in "
                  "this file's sibling driver tested only the first half; that "
                  "string is superseded by this node."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--perwindow-glob", default=None,
                    help="default: <json stem>_perwindow_K*.pt")
    ap.add_argument("--p1-json", default="/root/lanekeep/lowood_flagship_ci.json")
    ap.add_argument("--junction-deg", type=float, default=10.0)
    args = ap.parse_args()

    ood = OODMap(args.p1_json)
    jp = Path(args.json)
    d = json.loads(jp.read_text())
    stem = str(jp.with_suffix(""))
    pws = sorted(Path(jp.parent).glob(Path(stem).name + "_perwindow_K*.pt")) \
        if args.perwindow_glob is None else sorted(Path().glob(args.perwindow_glob))
    if not pws:
        raise SystemExit(f"no per-window dumps found for {jp}")
    for p in pws:
        K = int(p.name.rsplit("_K", 1)[1].split(".")[0])
        pw = torch.load(str(p), weights_only=False)
        lat = pw["lat"].numpy(); yaw = pw["yaw"].numpy()
        eid = list(pw["eid"])
        hd = pw["hd2s"].numpy(); spd = pw["speed"].numpy()
        junc = _corr.junction_mask(hd, args.junction_deg)
        long_ = (~junc) & (spd >= np.median(spd))
        strata = {"overall": np.ones(len(hd), bool), "junction": junc,
                  "longitudinal": long_, "other": (~junc) & (~long_)}
        node = {"_envelope": d.get("all_windows", {}).get(str(K), {})
                .get("ood", {}).get("_envelope", "P1 MEASURED envelope"),
                "_corrected_by": "fix_ood_verdict.py (E1a's FULL disjunction)"}
        for nm, m in strata.items():
            ix = np.flatnonzero(m)
            node[nm] = block(lat[ix], yaw[ix], [eid[i] for i in ix], ood, K)
        d.setdefault("all_windows", {}).setdefault(str(K), {})["ood"] = node
        o = node["overall"]
        print(f"[ood] {jp.name} K={K}: peak={o['ood_peak_ratio']['mean']:.4f} "
              f"stepsOutLat={o['EXTRAPOLATION_frac_steps_lat_over_3m']:.4f} "
              f"winOut={o['EXTRAPOLATION_frac_windows_any_step_out_of_envelope']:.4f}"
              f" -> {o['EXTRAPOLATION_VERDICT']}", flush=True)
    d["_ood_verdict_corrected"] = (
        "The `ood` nodes were RE-EMITTED by fix_ood_verdict.py: the driver's "
        "first emission tested only the ratio half of E1a's rule and therefore "
        "printed 'within the measured envelope on average' while a MAJORITY of "
        "steps were outside it. The per-window tensors are unchanged; only the "
        "verdict logic is. Both criteria are now reported separately.")
    jp.write_text(json.dumps(d, indent=2, default=str))
    print(f"[ood] rewrote {jp}", flush=True)


if __name__ == "__main__":
    main()
