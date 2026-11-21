#!/usr/bin/env python3
"""B-OFF / B-ON / A4 — the interface repair, measured through the NEW code path.

Runs the SHIPPED integrator wrapper (`taniteval/tools/refav1_arm.paths_from_controls`
-> `refa_v1_plan.unicycle_paths` -> `kinematic.rollout_unicycle`) over the banked
step-1000 T1 dump, with `action_units` OFF ("kappa", legacy) and ON ("steer").

CROSS-CHECK, declared in SPEC §3: the Benchmarks package converted the DATA
(kappa = tan(steer)/L at read time). This converts at the INTEGRATOR. Same algebra,
different site -> the numbers must agree, or one of the two framings is wrong.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\stack")
sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\taniteval")
sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\taniteval\tools")

from tanitad.models.kinematic import STEER_WHEELBASE_M          # noqa: E402
import importlib.util                                            # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "refav1_arm_probe", r"C:\Users\Admin\tanitad-wt\taniteval\tools\refav1_arm.py")
ARM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ARM)

DUMP = Path(r"C:\Users\Admin\refav1_eval_slice\t1_dump")
EPS = Path(r"C:\Users\Admin\refav1_eval_slice\eps")
PAI = Path(r"C:\Users\Admin\tanitad-data\physicalai")
OUT = Path(sys.argv[1])
DT, K = 0.2, 10


def kin_actions(v, kap, j0, n, dt=DT):
    """refav1_loader._kin_actions, verbatim (imported shape, local tensors)."""
    idx = torch.arange(j0, j0 + n)
    f = idx * 2
    f_next = torch.clamp((idx + 1) * 2, max=v.shape[0] - 1)
    a = (v[f_next] - v[f]) / dt
    return torch.stack([a, kap[f]], dim=-1)


def main() -> None:
    man = json.load(open(DUMP / "manifest.json", encoding="utf-8"))
    names = {int(e["file_index"]): e["name"] for e in man["episodes"]}
    true_wb = {}
    p = Path(sys.argv[2]) / "true_wheelbase_per_clip_UUID.csv"
    if p.exists():
        t = pd.read_csv(p)
        true_wb = dict(zip(t.clip_id.astype(str), t.wheelbase_true_m.astype(float)))
    lenc = pd.read_csv(OUT / "lenc_exact_per_clip.csv").set_index("clip_id")

    rows = []
    for fi, f in enumerate(sorted(glob.glob(str(DUMP / "ep*.npz")))):
        z = np.load(f)
        nm = names[fi]
        o = torch.load(EPS / f"{nm}.v2ep.pt", map_location="cpu", weights_only=False)
        v_ep = o["poses"][:, 3].float()
        kap_ep = o["actions"][:, 0].float()
        yaw_ep = o["poses"][:, 2].double().numpy()
        vnp = v_ep.double().numpy()
        stnp = kap_ep.double().numpy()

        # ---- A4: per-clip trajectory fits (the estimator the previous agent used,
        #      plus a lower-variance window estimator), for THIS clip
        dyaw = np.arctan2(np.sin(np.diff(yaw_ep)), np.cos(np.diff(yaw_ep)))
        vm = 0.5 * (vnp[:-1] + vnp[1:])
        ok = vm > 3.0
        kpose = np.zeros_like(dyaw)
        kpose[ok] = dyaw[ok] / (vm[ok] * 0.1)
        tst = np.tan(0.5 * (stnp[:-1] + stnp[1:]))
        sel = ok & (np.abs(kpose) > 1e-3)
        w_pw = float((tst[sel] @ kpose[sel]) / (kpose[sel] @ kpose[sel])) if sel.sum() > 20 else np.nan
        r_pw = (float(np.corrcoef(tst[sel], kpose[sel])[0, 1])
                if sel.sum() > 20 else np.nan)

        for i in range(z["g"].shape[0]):
            t = int(z["ws"][i])
            v0 = float(z["v0"][i])
            g = z["g"][i].astype(np.float64)                     # [K,2]
            act = kin_actions(v_ep, kap_ep, t, K)                # [K,2] (a, steer)
            hold = kin_actions(v_ep, kap_ep, t - 1, 1)[0]        # [2]
            r = {"clip": nm, "file": os.path.basename(f), "w": i, "t": t, "v0": v0,
                 "curved": int(np.abs(g[:, 1]).max() >= 0.3),
                 "w_pointwise_ep": w_pw, "r_pointwise_ep": r_pw,
                 "true_wheelbase_m": true_wb.get(nm),
                 "L_enc_exact": float(lenc.loc[nm, "L_ols"]) if nm in lenc.index else None}

            # ---- window estimator: sum tan(steer)*v*dt / dyaw_pose over the window
            f0 = 2 * t
            fs = np.arange(f0, f0 + 2 * K, 2)
            fs = fs[fs + 2 < len(vnp)]
            if len(fs) >= 5:
                num = float(np.sum(np.tan(stnp[fs]) * vnp[fs] * DT))
                dy = float(np.arctan2(np.sin(yaw_ep[fs[-1] + 2] - yaw_ep[fs[0]]),
                                      np.cos(yaw_ep[fs[-1] + 2] - yaw_ep[fs[0]])))
                r["w_window"] = num / dy if abs(dy) > 0.02 else np.nan
                r["dyaw_pose_win"] = dy
            else:
                r["w_window"], r["dyaw_pose_win"] = np.nan, np.nan

            for tag, units in (("off", "kappa"), ("on", "steer")):
                for arm, ctrl in (("ol", act), ("ha", hold[None].expand(K, 2))):
                    pth = ARM.paths_from_controls(ctrl, v0, DT, K,
                                                  action_units=units
                                                  )[0].double().numpy()
                    r[f"lat_{arm}_{tag}"] = float(np.mean(np.abs(pth[:, 1] - g[:, 1])))
                    r[f"lon_{arm}_{tag}"] = float(np.mean(np.abs(pth[:, 0] - g[:, 0])))
            # ha0 (constant velocity): unit-invariant by construction
            p0 = ARM.paths_from_controls(torch.zeros(K, 2), v0, DT, K)[0].double().numpy()
            r["lat_ha0"] = float(np.mean(np.abs(p0[:, 1] - g[:, 1])))
            r["lon_ha0"] = float(np.mean(np.abs(p0[:, 0] - g[:, 0])))

            # ---- COST OF A CONSTANT: repaired ol with a per-clip L instead of 2.9
            for nmw, L in (("trueWB", true_wb.get(nm)), ("fitPW", w_pw)):
                if L is None or not np.isfinite(L) or L <= 0.1:
                    r[f"lat_ol_on_{nmw}"] = np.nan
                    r[f"lon_ol_on_{nmw}"] = np.nan
                    continue
                c = act.clone().double()
                c[:, 1] = torch.tan(c[:, 1]) / float(L)
                pth = ARM.paths_from_controls(c.float(), v0, DT, K)[0].double().numpy()
                r[f"lat_ol_on_{nmw}"] = float(np.mean(np.abs(pth[:, 1] - g[:, 1])))
                r[f"lon_ol_on_{nmw}"] = float(np.mean(np.abs(pth[:, 0] - g[:, 0])))
            rows.append(r)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "interface_per_window.csv", index=False)
    cur, st = df[df.curved == 1], df[df.curved == 0]
    print(f"n={len(df)} curved={len(cur)} straight={len(st)}   L_enc={STEER_WHEELBASE_M}\n")
    print(f"{'arm/mode':26s} {'LATcurved':>10s} {'LATstraight':>12s} {'LONall':>9s}")
    summary = {}
    for lab, la, lo in (
            ("ol  OFF (legacy kappa)", "lat_ol_off", "lon_ol_off"),
            ("ol  ON  (steer->kappa)", "lat_ol_on", "lon_ol_on"),
            ("ol  ON  @ true wheelbase", "lat_ol_on_trueWB", "lon_ol_on_trueWB"),
            ("ol  ON  @ fitted L (oracle)", "lat_ol_on_fitPW", "lon_ol_on_fitPW"),
            ("ha  OFF (legacy kappa)", "lat_ha_off", "lon_ha_off"),
            ("ha  ON  (steer->kappa)", "lat_ha_on", "lon_ha_on"),
            ("ha0 (constant velocity)", "lat_ha0", "lon_ha0")):
        vals = (float(cur[la].mean()), float(st[la].mean()), float(df[lo].mean()))
        summary[lab] = vals
        print(f"{lab:26s} {vals[0]:10.4f} {vals[1]:12.4f} {vals[2]:9.4f}")

    # ---- A4 aggregate
    per_clip = df.groupby("clip").agg(
        w_pointwise=("w_pointwise_ep", "first"),
        r_pointwise=("r_pointwise_ep", "first"),
        w_window_med=("w_window", "median"),
        true_wb=("true_wheelbase_m", "first"),
        L_enc=("L_enc_exact", "first"),
        n_win=("w", "count")).reset_index()
    per_clip.to_csv(OUT / "per_clip_fits.csv", index=False)
    pw = per_clip.w_pointwise.to_numpy(float)
    ww = df.w_window.to_numpy(float)
    ww = ww[np.isfinite(ww)]
    print(f"\n[A4] per-clip pointwise LS w: n={len(pw)} min={np.nanmin(pw):.4f} "
          f"max={np.nanmax(pw):.4f} median={np.nanmedian(pw):.4f} "
          f"sd={np.nanstd(pw):.4f}")
    print(f"[A4] per-WINDOW integral w:  n={len(ww)} p05={np.percentile(ww,5):.4f} "
          f"median={np.median(ww):.4f} p95={np.percentile(ww,95):.4f} "
          f"IQR={np.percentile(ww,75)-np.percentile(ww,25):.4f}")
    tw = per_clip.true_wb.to_numpy(float)
    m = np.isfinite(pw) & np.isfinite(tw)
    rho = float(np.corrcoef(pw[m], tw[m])[0, 1]) if m.sum() > 3 else float("nan")
    print(f"[A4] corr(per-clip fitted w, TRUE wheelbase) = {rho:.4f}  (n={int(m.sum())})")
    print(f"[A4] true wheelbase values present: {sorted(set(np.round(tw[m],4)))}")
    print(f"[A4] fitted w within 0.05 of 2.73 (47% of corpus): "
          f"{int(np.sum(np.abs(pw[m]-2.73)<0.05))}/{int(m.sum())}")

    (OUT / "interface_summary.json").write_text(json.dumps({
        "n_windows": int(len(df)), "n_curved": int(len(cur)),
        "n_straight": int(len(st)), "L_enc": STEER_WHEELBASE_M,
        "summary_lat_curved_lat_straight_lon_all": summary,
        "A4": {"per_clip_pointwise_w": {"min": float(np.nanmin(pw)),
                                        "max": float(np.nanmax(pw)),
                                        "median": float(np.nanmedian(pw)),
                                        "sd": float(np.nanstd(pw))},
               "per_window_integral_w": {"n": int(len(ww)),
                                         "p05": float(np.percentile(ww, 5)),
                                         "median": float(np.median(ww)),
                                         "p95": float(np.percentile(ww, 95)),
                                         "iqr": float(np.percentile(ww, 75)
                                                      - np.percentile(ww, 25))},
               "corr_fitted_vs_true_wheelbase": rho},
    }, indent=2), encoding="utf-8")
    print("\n[probe] wrote", OUT / "interface_per_window.csv")


if __name__ == "__main__":
    main()
