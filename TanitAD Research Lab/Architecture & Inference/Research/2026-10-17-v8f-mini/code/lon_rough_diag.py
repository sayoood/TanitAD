"""v8F-mini: WHERE the remaining 6 s error lives -- longitudinal speed choice and path roughness. DESCRIPTIVE, no bar, decides nothing.

  python lon_rough_diag.py --eval-dir <dir with eval_<tag>.npz> --tags M1_s0 M2b_s0 [...] --out lon_rough.json

Per run, on eval-141 (ground truth REBUILT from the repo's eval inputs, as `v8f_mini_full_report.py` does, and every run's saved ade6
must equal ||pred - gt|| to 1e-5 or it is REFUSED):
  * ROUGHNESS of the path itself (origin prepended, dt 0.2 s): mean |accel|, mean |jerk|, mean |step speed change|, beside the human's
    and P10-yaw-4512's on the same windows;
  * LONGITUDINAL: final arc-length bias, and two counterfactual ADE@6s rows -- an ORACLE per-window arc scale (the path's shape kept,
    its length set to the human's: the ADE a perfect per-window speed choice would leave) and ONE global scale (what a bias
    correction could at best do). Both use the ground truth, so neither is an admissible predictor; they size the lever only;
  * the oracle-scale gain split by stratum: the human's speed change over 6 s, v0, launch (v0 < 1, dv > 3) and stop (v0 > 3, v_end < 1).
Identities (pinned by test_lon_rough_diag.py): the ground truth as a "run" reads ADE 0, oracle 0 and the human's own roughness; a path
uniformly slowed by any factor reads oracle ADE exactly 0.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
DT = 0.2
DV_BINS = ((-99, -3), (-3, -1), (-1, 1), (1, 3), (3, 99))
V0_BINS = ((0, 1), (1, 5), (5, 10), (10, 15), (15, 99))


def _report():
    spec = importlib.util.spec_from_file_location("v8f_mini_full_report", HERE.parent / "v8f_mini_full_report.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def arc(p):
    """[N,H,2] path from the origin -> [N,H] cumulative arc length."""
    q = np.concatenate([np.zeros((len(p), 1, 2)), p], 1)
    return np.cumsum(np.linalg.norm(np.diff(q, axis=1), axis=-1), 1)


def roughness(p, dt=DT):
    q = np.concatenate([np.zeros((len(p), 1, 2)), np.asarray(p, np.float64)], 1)
    v = np.diff(q, axis=1) / dt
    a = np.diff(v, axis=1) / dt
    j = np.diff(a, axis=1) / dt
    return dict(accel_mean_mps2=float(np.linalg.norm(a, axis=-1).mean()), jerk_mean_mps3=float(np.linalg.norm(j, axis=-1).mean()),
                speed_step_change_mps=float(np.abs(np.diff(np.linalg.norm(v, axis=-1), axis=1)).mean()))


def lon_block(p, gt, v0):
    ag, ap = arc(gt), arc(p)
    ade = np.linalg.norm(p - gt, axis=-1).mean(1)
    sc = (ag[:, -1] / np.maximum(ap[:, -1], 1e-3))[:, None, None]
    ado = np.linalg.norm(p * sc - gt, axis=-1).mean(1)
    glob = float(ag[:, -1].sum() / max(ap[:, -1].sum(), 1e-9))
    adg = np.linalg.norm(p * glob - gt, axis=-1).mean(1)
    vend = np.linalg.norm(gt[:, -1] - gt[:, -2], axis=-1) / DT
    dv = vend - v0
    gain = ade - ado
    tot = float(gain.sum())

    def row(m):
        return dict(n=int(m.sum()), share_windows=float(m.mean()), ade6=float(ade[m].mean()) if m.any() else None,
                    arc_bias_m=float((ap[m, -1] - ag[m, -1]).mean()) if m.any() else None,
                    share_of_oracle_gain=float(gain[m].sum() / tot) if tot > 0 and m.any() else None)
    strata = {f"dv[{lo},{hi})": row((dv >= lo) & (dv < hi)) for lo, hi in DV_BINS}
    strata.update({f"v0[{lo},{hi})": row((v0 >= lo) & (v0 < hi)) for lo, hi in V0_BINS})
    strata["launch v0<1 & dv>3"] = row((v0 < 1) & (dv > 3))
    strata["stop v0>3 & vend<1"] = row((v0 > 3) & (vend < 1))
    return dict(ade6=float(ade.mean()), ade6_oracle_arc_scale=float(ado.mean()), ade6_one_global_scale=float(adg.mean()),
                global_scale=glob, final_arc_bias_m=float((ap[:, -1] - ag[:, -1]).mean()),
                median_final_arc_ratio=float(np.median(ap[:, -1] / np.maximum(ag[:, -1], 1e-3))), strata=strata)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-npz", help="TEST INPUT: npz with gt, eid, dt, v0 in place of the rebuild")
    ap.add_argument("--p10-pred")
    a = ap.parse_args(argv)
    R = _report()
    if a.gt_npz:
        z = np.load(a.gt_npz)
        gt, eid, v0, prov = z["gt"].astype(np.float64), z["eid"], z["v0"].astype(np.float64), {"source": "gt-npz (TEST INPUT)"}
    else:
        R._paths()
        from tanitad.data import v8f_mini_data as D
        raw = R.RAW
        X = D.eval_context(json.load(open(raw / "eval_grid_windows.json", encoding="utf-8")), str(raw / "eval141_kin.npz"),
                           D.load_join(str(raw / "join_eval141.npz")), kin_mode="yaw")
        gt, eid, v0 = X["gt"].astype(np.float64), X["eid"], X["kin"][:, 0].astype(np.float64)
        prov = {"source": "v8f_mini_data.eval_context(kin_mode='yaw') over the repo's eval inputs; v0 = kin[:, 0]"}
    runs = {"GT": gt}
    p10p = a.p10_pred or str(R.P10_PRED)
    if os.path.exists(p10p):
        runs["P10"] = np.load(p10p).astype(np.float64)
    for t in a.tags:
        z = np.load(os.path.join(a.eval_dir, f"eval_{t}.npz"))
        if not np.array_equal(z["eid"], eid):
            R.refuse(f"eval_{t}.npz is not on the rebuilt eval grid (eid differs)")
        p = z["pred"].astype(np.float64)
        dev = float(np.abs(np.linalg.norm(p - gt, axis=-1).mean(1) - z["ade6"]).max())
        if dev > R.TOL_ADE:
            R.refuse(f"eval_{t}: ||pred - gt|| differs from its saved ade6 by {dev:.3g} m: the ground truth is not its grid")
        runs[t] = p
    out = dict(kind="DESCRIPTIVE diagnostic, no bar", tier="T0-open-loop own outputs, eval-141", ground_truth=prov, dt_s=DT,
               n_windows=int(len(eid)), n_episodes=int(len(np.unique(eid))),
               runs={k: dict(roughness=roughness(p), longitudinal=lon_block(p, gt, v0)) for k, p in runs.items()})
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    for k, r in out["runs"].items():
        L, Rg = r["longitudinal"], r["roughness"]
        print(f"{k:12s} ADE6 {L['ade6']:.4f}  oracle-speed {L['ade6_oracle_arc_scale']:.4f}  one-scale {L['ade6_one_global_scale']:.4f}  "
              f"arc bias {L['final_arc_bias_m']:+.3f}  |accel| {Rg['accel_mean_mps2']:.3f}  |dspeed| {Rg['speed_step_change_mps']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
