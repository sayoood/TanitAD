"""STREAM D, mechanism — WHY the speed route cannot deliver ``long_accel``.

A null result is stronger when it comes with a mechanism, and there is exactly
one mechanistic claim in the prior work: *"at R² 0.72 the speed channel still has
MAE 4.04 m/s, and a 0.2 s centred difference of two such predictions carries ~5x
that error"*. That argument is only valid if the speed ERROR is **white** at the
0.2 s lag. If the error were a slowly varying per-episode bias, differencing
would CANCEL most of it and the accel channel would be recoverable through the
speed route after all — the opposite conclusion from the same R².

So measure it, do not argue it:

  1. the autocorrelation of the held-out speed ERROR at 0.2/0.4/0.8/1.6 s;
  2. the R² of the centred difference of PREDICTED speed against the centred
     difference of TRUE speed (the derived-accel route, evaluated directly);
  3. the same against the CAN ``long_accel`` label;
  4. an error-budget: how accurate would the speed track have to BE for the
     derived accel to clear R² 0.5?

Also dumps the per-window predictions to ``raw/preds_speedarm.npz`` so this is a
0-GPU re-analysis surface rather than a number in a report.

usage: python analyze_speed_error.py --out raw/speed_error_mechanism.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "taniteval"))

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def main() -> int:
    from tanitad.eval import accel_probe as AP
    from tanitad.eval import ap_ci as APCI
    import run_accel_recoverability as R

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "raw" / "speed_error_mechanism.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    sub = R.load_substrate()
    eps = sub["episodes"]
    tr_eps, ho_eps = R.split_episodes(eps)
    fit_eps = [e for i, e in enumerate(tr_eps) if i % 3 != 0]
    sel_eps = [e for i, e in enumerate(tr_eps) if i % 3 == 0]
    Zf, Yf, _ = R.stack(fit_eps)
    Zs, Ys, _ = R.stack(sel_eps)
    Ztr, Ytr, _ = R.stack(tr_eps)
    Zho, Yho, eid = R.stack(ho_eps)

    # the best speed arm from the ladder
    sp, hp, tp, keys, meta = R.ridge_arm(AP, Zf, Yf, Zs, Ztr, Ytr, Zho,
                                         feat="window", kernel="linear",
                                         device=dev)
    chosen = R.select_hparam(AP, sp, Ys, keys, meta)
    P = R.assemble(hp, chosen, meta)
    del sp, hp, tp
    gs = Yho[:, :4].numpy().astype(np.float64)
    v_hat, v_gt = P[:, 0], gs[:, 0]
    err = v_hat - v_gt
    log(f"speed arm: R2 {AP.r2_score(v_hat, v_gt):+.4f}  MAE "
        f"{np.abs(err).mean():.4f} m/s")

    # ---- per-episode sequences (windows are stride 2 => 0.2 s apart) -------- #
    out = {"_what": "why the speed route cannot deliver long_accel",
           "arm": "RIDGE_linear_window (best speed arm of the ladder)",
           "speed_r2": round(AP.r2_score(v_hat, v_gt), 5),
           "speed_mae_mps": round(float(np.abs(err).mean()), 5),
           "speed_rmse_mps": round(float(np.sqrt((err ** 2).mean())), 5),
           "window_spacing_s": 0.2}

    off, ac, dpred, dgt, dcan, eid_d = 0, [], [], [], [], []
    for e in ho_eps:
        n = e["n"]
        sl = slice(off, off + n)
        off += n
        er = err[sl]
        if er.std() > 1e-12:
            ac.append([float(np.corrcoef(er[:-L], er[L:])[0, 1])
                       if min(er[:-L].std(), er[L:].std()) > 1e-12 else np.nan
                       for L in (1, 2, 4, 8)])
        vh, vg = v_hat[sl], v_gt[sl]
        # centred difference over ONE window step each side = 0.4 s span
        if n >= 3:
            dpred.append((vh[2:] - vh[:-2]) / 0.4)
            dgt.append((vg[2:] - vg[:-2]) / 0.4)
            dcan.append(gs[sl][1:-1, 3])
            eid_d.append(np.full(n - 2, e["name"]))
    ac = np.array(ac)
    out["speed_error_autocorr_lag_0p2_0p4_0p8_1p6_s"] = \
        [round(float(x), 4) for x in np.nanmean(ac, 0)]
    out["n_episodes_in_autocorr"] = int(len(ac))
    out["reading"] = ("HIGH autocorrelation at 0.2 s means the error is a SLOW "
                      "bias that differencing would cancel; LOW means it is "
                      "white and differencing amplifies it. This is the "
                      "load-bearing fact behind the '5x amplification' argument, "
                      "which was asserted, not measured.")

    dp, dg, dc = (np.concatenate(dpred), np.concatenate(dgt),
                  np.concatenate(dcan))
    ed = np.concatenate(eid_d)
    out["derived_accel_direct"] = {
        "_what": ("the derived-accel route evaluated end to end: centred "
                  "difference of the PREDICTED speed track, scored against the "
                  "centred difference of the TRUE track and against the CAN "
                  "label."),
        "n_windows": int(len(dp)), "n_episodes": int(len(np.unique(ed))),
        "r2_vs_true_speed_difference": APCI.stat_episode_cluster_bootstrap(
            lambda s: AP.r2_score(dp[s], dg[s]), ed, n_boot=a.n_boot,
            name="r2_dspeed"),
        "r2_vs_CAN_long_accel": APCI.stat_episode_cluster_bootstrap(
            lambda s: AP.r2_score(dp[s], dc[s]), ed, n_boot=a.n_boot,
            name="r2_can_accel"),
        "ORACLE_true_difference_vs_CAN_long_accel":
            APCI.stat_episode_cluster_bootstrap(
                lambda s: AP.r2_score(dg[s], dc[s]), ed, n_boot=a.n_boot,
                name="r2_oracle"),
        "std_true_difference_mps2": round(float(dg.std()), 5),
        "std_predicted_difference_mps2": round(float(dp.std()), 5),
        "std_CAN_long_accel_mps2": round(float(dc.std()), 5),
    }
    log("derived accel: vs true-diff "
        f"{out['derived_accel_direct']['r2_vs_true_speed_difference']['point']:+.4f}"
        f"  vs CAN "
        f"{out['derived_accel_direct']['r2_vs_CAN_long_accel']['point']:+.4f}"
        f"  oracle "
        f"{out['derived_accel_direct']['ORACLE_true_difference_vs_CAN_long_accel']['point']:+.4f}")

    # ---- error budget: how good must the speed track BE? -------------------- #
    # Simulate: true speed + white noise of std s, differenced. Report the s at
    # which the derived accel would clear R2 0.5 against the CAN label, and
    # convert it to the speed R2 that implies.
    var_v = float(v_gt.var())
    budget = []
    for s_noise in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0):
        g = np.random.default_rng(11)
        noisy = v_gt + g.standard_normal(len(v_gt)) * s_noise
        off2, dn, dcc = 0, [], []
        for e in ho_eps:
            n = e["n"]
            sl = slice(off2, off2 + n)
            off2 += n
            if n >= 3:
                dn.append((noisy[sl][2:] - noisy[sl][:-2]) / 0.4)
                dcc.append(gs[sl][1:-1, 3])
        dn, dcc = np.concatenate(dn), np.concatenate(dcc)
        budget.append({
            "speed_noise_std_mps": s_noise,
            "implied_speed_r2": round(1.0 - s_noise ** 2 / var_v, 4),
            "derived_accel_r2_vs_CAN": round(AP.r2_score(dn, dcc), 4)})
    out["error_budget_white_noise"] = {
        "_what": ("what speed accuracy the derived route would NEED. True speed "
                  "plus white noise of the given std, differenced over 0.4 s, "
                  "scored against the CAN label."),
        "caveat": ("white noise is the WORST case for differencing; a real "
                   "predictor's error is partly autocorrelated, which is exactly "
                   "why the measured autocorrelation above is reported first."),
        "levels": budget}
    log("budget: " + " ".join(f"{b['speed_noise_std_mps']}->{b['derived_accel_r2_vs_CAN']:+.3f}"
                              for b in budget))

    # ---- SELF-CONSISTENCY: the scalar route vs the TRAJECTORY route --------- #
    # The arm emits the same physics twice — once as a scalar readout and once
    # implicitly through the 2 s waypoints. If the two disagree about which
    # channel is recoverable, the instrument is measuring itself. This is the
    # mandatory component-vs-family control for this stream.
    from tanitad.eval import idm_families as FF
    gt_t = Yho[:, 4:].numpy().astype(np.float64).reshape(len(Yho), 4, 2)
    pr_t = P[:, 4:].reshape(len(P), 4, 2)
    Gg, Pg = FF.geometry(gt_t, FF.IDM_DT_S), FF.geometry(pr_t, FF.IDM_DT_S)
    out["self_consistency_scalar_vs_trajectory"] = {
        "_what": ("the same arm's scalar readout and its 2 s trajectory are two "
                  "independent routes to the same physics. They must agree about "
                  "WHICH channel is recoverable."),
        "speed": {
            "r2_scalar_route_vs_CAN_speed": round(AP.r2_score(v_hat, v_gt), 5),
            "r2_traj_route_vs_CAN_speed":
                round(AP.r2_score(Pg["speed"][:, 0], v_gt), 5),
            "corr_between_routes":
                round(float(np.corrcoef(v_hat, Pg["speed"][:, 0])[0, 1]), 5)},
        "long_accel": {
            "r2_scalar_route_vs_CAN_accel":
                round(AP.r2_score(P[:, 3], gs[:, 3]), 5),
            "r2_traj_route_vs_CAN_accel":
                round(AP.r2_score(Pg["accel"].mean(1), gs[:, 3]), 5),
            "r2_TRAJ_GT_route_vs_CAN_accel_ORACLE":
                round(AP.r2_score(Gg["accel"].mean(1), gs[:, 3]), 5),
            "note": ("the trajectory route's accel is the mean of the horizon's "
                     "0.5 s speed differences — a FUTURE-window accel, not the "
                     "instantaneous label, so the GT-trajectory row is the "
                     "ceiling that route can reach and is reported beside it.")},
    }
    log("self-consistency: speed scalar "
        f"{out['self_consistency_scalar_vs_trajectory']['speed']['r2_scalar_route_vs_CAN_speed']:+.4f}"
        f" traj {out['self_consistency_scalar_vs_trajectory']['speed']['r2_traj_route_vs_CAN_speed']:+.4f}"
        f" | accel scalar "
        f"{out['self_consistency_scalar_vs_trajectory']['long_accel']['r2_scalar_route_vs_CAN_accel']:+.4f}"
        f" traj {out['self_consistency_scalar_vs_trajectory']['long_accel']['r2_traj_route_vs_CAN_accel']:+.4f}"
        f" trajGT {out['self_consistency_scalar_vs_trajectory']['long_accel']['r2_TRAJ_GT_route_vs_CAN_accel_ORACLE']:+.4f}")

    np.savez_compressed(HERE / "raw" / "preds_speedarm.npz",
                        pred=P, gt_scalars=gs,
                        gt_traj=Yho[:, 4:].numpy(), eid=eid)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    log(f"wrote {a.out} + raw/preds_speedarm.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
