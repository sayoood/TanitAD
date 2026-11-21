"""PREREG_V8F_SMOOTH (raw/2026-10-20-smooth/, sealed at e58534a, sha256 a2f19e48...): read S(pred) = P10 + SG(pred - P10) against the arm.

  python v8f_smooth_read.py --eval-dir <dir with eval_<tag>.npz> --tags M1_s0 M2b_s0 [M1_s1 M2b_s1] --out smooth.json

Per run: S = SG(window 7, order 2, mode interp) on the residual over P10-yaw-4512's eval predictions; the dose SG(11, 2) is DESCRIPTIVE.
  * D: S - arm on accel MAE and yaw-rate MAE, each < 0 and separated (`ra1_hjepa_eval.paired_family_block`, B = 2,000);
  * P: S - arm on ADE@6s, upper CI <= +0.01 m (`taniteval.ci.paired_episode_cluster_bootstrap`);
  * C: ADE@6s of P10 + SG(GT - P10) against GT <= 0.05 m, else every reading is INVALID.
The bars decide only at BOTH seeds of an arm; an arm with one seed read is INTERIM and carries no verdict.
Ground truth is rebuilt and every run's saved ade6 checked, exactly as `v8f_mini_full_report.py` does (its guards, imported).
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
PREREG_SHA = "a2f19e4885e440fbd5308e6dd29905f038b50bf08af0c74c1e501b8117cdd822"
WINDOW, ORDER, DOSE_WINDOW = 7, 2, 11
P_BAR_M, C_BAR_M = 0.01, 0.05
D_COMPONENTS = ("accel_mae_mps2", "yaw_rate_mae_degps")


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def smooth(pred, base, window=WINDOW, order=ORDER):
    from scipy.signal import savgol_filter
    return base + savgol_filter(np.asarray(pred, np.float64) - base, window, order, axis=1, mode="interp")


def ade(p, gt):
    return np.linalg.norm(p - gt, axis=-1).mean(1)


def read_run(E, ci, R_, gt, eid, dt, p10, pred, n_boot):
    import torch
    g = torch.from_numpy(gt).float()
    fpw = lambda q: E.family_per_window(torch.from_numpy(np.asarray(q, np.float64)).float(), g, dt)      # noqa: E731
    base_pw = fpw(pred)
    out = {"residual_t0_median_m": float(np.median(np.linalg.norm(pred[:, 0] - p10[:, 0], axis=-1)))}
    for name, w in (("S", WINDOW), ("dose_w11", DOSE_WINDOW)):
        sp = smooth(pred, p10, w)
        fam = E.paired_family_block(fpw(sp), base_pw, eid, n_boot, 0)
        r = ci.paired_episode_cluster_bootstrap(ade(sp, gt), ade(pred, gt), eid, n_boot=n_boot, seed=0)
        blk = {"families_S_minus_arm": fam, "ade6_S_minus_arm": {k: r[k] for k in ("delta", "lo", "hi", "separated")},
               "roughness": R_.roughness(sp)}
        if name == "S":
            dd = {c: fam["lateral" if "yaw" in c else "longitudinal"][c] for c in D_COMPONENTS}
            blk["D"] = {c: dict(delta=v["delta"], lo=v["lo"], hi=v["hi"], separated=bool(v["separated"]),
                                ok=bool(v["delta"] < 0 and v["separated"])) for c, v in dd.items()}
            blk["D_ok"] = all(v["ok"] for v in blk["D"].values())
            blk["P_ok"] = bool(r["hi"] <= P_BAR_M)
        out[name] = blk
    out["roughness_arm"] = R_.roughness(pred)
    return out


def verdict(per_seed, c_ok):
    if not c_ok:
        return "INVALID (control C failed)"
    if len(per_seed) < 2:
        return "INTERIM (one seed read; the bars decide only at both)"
    d = all(s["S"]["D_ok"] for s in per_seed.values())
    p = all(s["S"]["P_ok"] for s in per_seed.values())
    return "SMOOTH-PASS" if d and p else ("D PASS, P FAIL" if d else "D FAIL")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-npz", help="TEST INPUT: npz with gt, eid, dt")
    ap.add_argument("--p10-pred")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    R = _mod("v8f_mini_full_report", HERE.parent / "v8f_mini_full_report.py")
    R_ = _mod("lon_rough_diag", HERE.parent / "lon_rough_diag.py")

    class G:
        gt_npz = a.gt_npz
        eval_grid, eval_kin, eval_join = (str(R.RAW / f) for f in ("eval_grid_windows.json", "eval141_kin.npz", "join_eval141.npz"))
    gt, eid, dt, prov = R.ground_truth(G)
    p10_path = a.p10_pred or str(R.P10_PRED)
    p10 = np.load(p10_path).astype(np.float64)
    if p10.shape != gt.shape:
        R.refuse(f"P10 predictions {p10.shape} are not on the eval grid {gt.shape}")
    E = R._eval_module()
    from taniteval import ci
    c_ade = float(ade(smooth(gt, p10), gt).mean())
    c_ok = c_ade <= C_BAR_M
    arms = {}
    for t in a.tags:
        z = np.load(os.path.join(a.eval_dir, f"eval_{t}.npz"))
        if not np.array_equal(z["eid"], eid):
            R.refuse(f"eval_{t}.npz is not on the rebuilt eval grid (eid differs)")
        pred = z["pred"].astype(np.float64)
        dev = float(np.abs(ade(pred, gt) - z["ade6"]).max())
        if dev > R.TOL_ADE:
            R.refuse(f"eval_{t}: ||pred - gt|| differs from its saved ade6 by {dev:.3g} m: the ground truth is not its grid")
        arm, s = t.rsplit("_s", 1)
        arms.setdefault(arm, {})[f"seed{s}"] = read_run(E, ci, R_, gt, eid, dt, p10, pred, a.n_boot)
    out = dict(prereg="PREREG_V8F_SMOOTH", prereg_sha256=PREREG_SHA, operator=f"P10 + savgol(pred - P10, {WINDOW}, {ORDER}, mode=interp)",
               tier="T0 open loop, the model's own outputs, eval-141", estimator="paired episode-cluster bootstrap (another draw of EPISODES)",
               ground_truth=prov, p10_sha256=R.sha256(p10_path), n_windows=int(len(eid)), n_episodes=int(len(np.unique(eid))),
               control_C=dict(ade6_SG_of_human_vs_human_m=c_ade, bar_m=C_BAR_M, ok=bool(c_ok)), roughness_human=R_.roughness(gt),
               arms={arm: dict(per_seed=ps, verdict=verdict(ps, c_ok)) for arm, ps in arms.items()})
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=float)
    print(f"control C: SG on the human path moves ADE@6s by {c_ade:.4f} m (bar {C_BAR_M}) -> {'ok' if c_ok else 'INVALID'}")
    for arm, r in out["arms"].items():
        for s, b in r["per_seed"].items():
            S = b["S"]
            print(f"{arm} {s}: ADE S-arm {S['ade6_S_minus_arm']['delta']:+.4f} [{S['ade6_S_minus_arm']['lo']:+.4f}, {S['ade6_S_minus_arm']['hi']:+.4f}]  "
                  + "  ".join(f"{c} {v['delta']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" for c, v in S["D"].items())
                  + f"  |accel| {b['roughness_arm']['accel_mean_mps2']:.3f} -> {S['roughness']['accel_mean_mps2']:.3f}")
        print(f"{arm}: {r['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
