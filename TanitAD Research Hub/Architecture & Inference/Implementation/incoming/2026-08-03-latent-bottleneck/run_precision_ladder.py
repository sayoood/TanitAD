"""D-LATENT part 2 — THE PRECISION LADDER: turn the `long_accel` null into an
ENGINEERING TARGET for the next latent.

WHY THIS EXISTS — what part 1 (`run_temporal_falsifier.py`) actually found
    The pre-registered pixel falsifier came back INADMISSIBLE, and the reason is
    the interesting part: on the SAME windows the frozen v1 latent reads `speed`
    at R2 +0.7145 while the best hand-built pixel motion feature reads it at
    +0.1064 — the learned latent is ~7x the BETTER motion reader. So "the
    encoder destroys motion" is not what the evidence says, and a null from the
    weaker instrument cannot falsify anything (its own positive control fails).

    That leaves ONE mechanism on the table, and it is quantitative rather than
    categorical:

      long_accel is not a DIFFERENT kind of information the latent lacks.
      It is the SAME information at a much finer resolution.

    `long_accel` has std 0.6219 m/s^2, i.e. a speed change of ~0.06 m/s per
    100 ms step. The latent's speed read has MAE 4.72 m/s. Reading a 0.06 m/s
    change off a 4.72 m/s-accurate track is the whole story — IF the numbers say
    so. This script measures whether they do, and what precision would be
    enough.

WHAT IS MEASURED, all closed-form on the banked substrate, 0 pod GPU-h
    A. WHITE-NOISE LADDER. Corrupt the TRUE speed track with white noise at
       sigma in a swept ladder and derive accel. Extends the prior panel's three
       points (0.05 / 0.10 / 0.25) across the whole range up to the latent's
       MEASURED error, so the curve — not three points — is the artifact.
    B. AUTOCORRELATED-NOISE LADDER. ⚠️ The honest version. The real latent-read
       error is autocorrelated at 0.9265 (0.2 s), so white noise UNDERSTATES how
       good a real track is: differencing cancels ~93 % of correlated error.
       An AR(1) noise matched to that autocorrelation is the fair model, and the
       two ladders BRACKET the answer.
    C. THE REAL TRACK, at full strength. The latent's OWN predicted speed
       sequence (all 9 window positions, ridge on the latent window) →
       Savitzky-Golay smoothing swept over window/order → central difference.
       This is the "unrecovered" hypothesis at its strongest: the prior panel
       differenced TWO points of a noisy prediction; a 9-point SG derivative is
       the best linear estimator of a derivative from that track.
    D. THE TARGET. The sigma at which derived accel first clears R2 0.5 and 0.7,
       read off both ladders — the precision a better latent must hit.

Every number carries the paired episode-cluster bootstrap. Positive control:
the noiseless oracle must reach ~+0.92 or the ladder is void.

usage: python run_precision_ladder.py --out results_precision_ladder.json
"""
from __future__ import annotations

import argparse
import json
import math
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

LATENTS = Path(r"C:/Users/Admin/tanitad-data/eval/idm_derived_accel_latents.pt")
K, STRIDE = 4, 2
DT = 0.1
SCALARS = ("speed", "yaw_rate", "steer", "long_accel")
ACCEL_J, SPEED_J = 3, 0
T0 = time.time()

#: measured on THESE windows by the prior panel and re-measured here
MEASURED_ERR_AUTOCORR_02S = 0.9265
SIGMA_LADDER = (0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0,
                3.0, 4.72)
TARGET_R2 = (0.5, 0.7)


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def split_episodes(eps, hold_every: int = 3):
    tr = [e for i, e in enumerate(eps) if i % hold_every != 0]
    ho = [e for i, e in enumerate(eps) if i % hold_every == 0]
    return tr, ho


def central_diff(Q, c=K, dt=DT):
    """[N, 2k+1] speed sequence -> centred difference at the centre [N]."""
    return (Q[:, c + 1] - Q[:, c - 1]) / (2.0 * dt)


def savgol_coeffs(half: int, order: int, deriv: int = 1, dt: float = DT):
    """Exact Savitzky-Golay derivative coefficients — the least-squares optimal
    linear derivative estimator over ``2*half+1`` uniformly spaced samples.

    Written out rather than imported so the instrument stays dependency-light
    (numpy only) and so the exact stencil is auditable next to its use.
    """
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, order + 1, increasing=True)          # [n, order+1]
    # pseudo-inverse row `deriv` * deriv! gives the derivative coefficients
    C = np.linalg.pinv(A)[deriv] * float(math.factorial(deriv))
    return C / (dt ** deriv)


def ar1_noise(n_rows, w, sigma, rho, rng):
    """AR(1) noise ACROSS the window positions of each row: e_j = rho*e_{j-1} +
    z. Scaled so the MARGINAL std is exactly ``sigma`` — otherwise the two
    ladders would not be comparable at the same nominal sigma, and the whole
    bracket would be meaningless.
    """
    e = np.zeros((n_rows, w), dtype=np.float64)
    e[:, 0] = rng.standard_normal(n_rows)
    for j in range(1, w):
        e[:, j] = rho * e[:, j - 1] + np.sqrt(1.0 - rho ** 2) * \
            rng.standard_normal(n_rows)
    return sigma * e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_precision_ladder.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from tanitad.eval import accel_probe as AP
    from tanitad.eval import ap_ci as APCI

    sub = torch.load(LATENTS, map_location="cpu", weights_only=False)
    eps = sub["episodes"]
    tr_eps, ho_eps = split_episodes(eps)
    fit_eps, sel_eps = split_episodes(tr_eps, hold_every=3)
    log(f"episodes {len(eps)} -> train {len(tr_eps)} / heldout {len(ho_eps)}")

    def cat(elist, key):
        return torch.cat([e[key] for e in elist]).float()

    Qho = cat(ho_eps, "Q").numpy().astype(np.float64)      # TRUE speed windows
    Sho = cat(ho_eps, "S").numpy().astype(np.float64)
    Str = cat(tr_eps, "S").numpy().astype(np.float64)
    eid = np.concatenate([np.full(e["n"], e["name"]) for e in ho_eps])
    y = Sho[:, ACCEL_J]
    w = Qho.shape[1]
    res = {"prereg": "Project Steering/PREREG_TEMPORAL_LATENT.md",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "substrate": {"latent_cache": str(LATENTS),
                         "n_heldout_windows": int(len(y)),
                         "n_heldout_episodes": int(len(ho_eps)),
                         "window_positions": int(w), "dt_s": DT},
           "estimator": "episode_cluster_bootstrap (tanitad.eval.ap_ci)",
           "target_stats": {
               "long_accel_std": round(float(y.std()), 4),
               "implied_speed_change_per_step_mps":
                   round(float(y.std() * DT), 4),
               "train_mean_null_r2": None}}

    def r2ci(pred, name):
        return APCI.stat_episode_cluster_bootstrap(
            lambda sel, p=np.asarray(pred, np.float64): AP.r2_score(p[sel],
                                                                    y[sel]),
            eid, n_boot=args.n_boot, name=name)

    # --- the empirical null and the noiseless oracle (the two anchors) ------ #
    null = np.full(len(y), float(Str[:, ACCEL_J].mean()))
    res["target_stats"]["train_mean_null_r2"] = r2ci(null, "r2_null")
    orc = central_diff(Qho)
    res["oracle_central_diff_true_speed"] = r2ci(orc, "r2_oracle")
    log(f"NULL {res['target_stats']['train_mean_null_r2']['point']:+.4f}   "
        f"ORACLE(central diff of true speed) "
        f"{res['oracle_central_diff_true_speed']['point']:+.4f}")

    # --- A/B: the two noise ladders ---------------------------------------- #
    rng = np.random.default_rng(20260803)
    ladders = {}
    for tag, rho in (("white", 0.0), ("ar1_matched", MEASURED_ERR_AUTOCORR_02S)):
        rows = []
        for sig in SIGMA_LADDER:
            noise = (np.zeros_like(Qho) if sig == 0.0
                     else ar1_noise(len(Qho), w, sig, rho, rng))
            Qn = Qho + noise
            cd = central_diff(Qn)
            # the BEST linear derivative from the same corrupted track: a
            # Savitzky-Golay derivative over the whole 9-point window. Quoting
            # only the 2-point difference would understate every rung.
            best_sg, best_cfg = None, None
            for half in (1, 2, 3, 4):
                for order in (2, 3):
                    if 2 * half + 1 <= order + 1:
                        continue
                    c = savgol_coeffs(half, order)
                    seg = Qn[:, K - half:K + half + 1]
                    p = seg @ c
                    s = AP.r2_score(p, y)
                    if best_sg is None or s > best_sg:
                        best_sg, best_cfg, best_p = s, (half, order), p
            rows.append({
                "sigma_mps": sig,
                "central_diff": r2ci(cd, f"r2_{tag}_cd_sig{sig}"),
                "best_savgol": r2ci(best_p, f"r2_{tag}_sg_sig{sig}"),
                "savgol_cfg": {"half_window": best_cfg[0], "poly_order":
                               best_cfg[1]},
                "realised_noise_autocorr_lag1": round(float(np.corrcoef(
                    noise[:, :-1].ravel(), noise[:, 1:].ravel())[0, 1]), 4)
                if sig > 0 else None})
            log(f"  {tag:>12s} sigma={sig:<5} cd={rows[-1]['central_diff']['point']:+.4f} "
                f"sg={rows[-1]['best_savgol']['point']:+.4f} "
                f"(SG {best_cfg})")
        ladders[tag] = rows
    res["noise_ladders"] = ladders

    # --- D: the precision TARGET ------------------------------------------- #
    def sigma_at(rows, key, thr):
        """Largest sigma whose R2 still clears ``thr`` (the ladder is monotone
        decreasing in sigma; the crossing is reported by linear interpolation in
        log-sigma so the answer is not quantised to the grid)."""
        xs = [(r["sigma_mps"], r[key]["point"]) for r in rows]
        xs.sort()
        prev = None
        for s, v in xs:
            if v < thr and prev is not None:
                s0, v0 = prev
                if s0 <= 0:
                    return s
                f = (v0 - thr) / max(v0 - v, 1e-9)
                return float(np.exp(np.log(max(s0, 1e-6))
                                    + f * (np.log(s) - np.log(max(s0, 1e-6)))))
            if v >= thr:
                prev = (s, v)
        return None

    res["precision_target"] = {
        f"sigma_for_r2_{t}": {
            lad: {k: sigma_at(rows, k, t) for k in ("central_diff",
                                                    "best_savgol")}
            for lad, rows in ladders.items()}
        for t in TARGET_R2}

    # --- C: the REAL latent-read track, at full strength -------------------- #
    # Fit ridge latent-window -> the WHOLE 9-position speed sequence, then take
    # the best SG derivative of the PREDICTED sequence. This is the strongest
    # form of the "the head was wrong, not the latent" hypothesis.
    Zf = cat(fit_eps, "Z")
    Zs = cat(sel_eps, "Z")
    ZF = cat(tr_eps, "Z")
    Zh = cat(ho_eps, "Z")
    Qf, Qs, QF = (cat(fit_eps, "Q"), cat(sel_eps, "Q"), cat(tr_eps, "Q"))
    Xf = AP.window_features(Zf, "window")
    Xs = AP.window_features(Zs, "window")
    XF = AP.window_features(ZF, "window")
    Xh = AP.window_features(Zh, "window")
    Xf, Xs = AP.standardize(Xf, Xs)
    XF, Xh = AP.standardize(XF, Xh)
    qmu, qsd = QF.mean(0, keepdim=True), QF.std(0, keepdim=True).clamp_min(1e-6)
    alphas = AP.DualRidge.alpha_grid(2, -4, 10)
    kw = dict(kernel="linear", matmul_device=args.device,
              matmul_dtype=torch.float32)
    inner = AP.DualRidge(Xf, ((Qf - qmu) / qsd).double(), **kw)
    Qs_np = Qs.numpy().astype(np.float64)
    best_a, best_s = None, -1e18
    for al in alphas:
        p = (inner.predict(Xs, al) * qsd + qmu).numpy()
        s = float(np.mean([AP.r2_score(p[:, j], Qs_np[:, j]) for j in range(w)]))
        if s > best_s:
            best_a, best_s = al, s
    del inner
    full = AP.DualRidge(XF, ((QF - qmu) / qsd).double(), **kw)
    Qpred = (full.predict(Xh, best_a) * qsd + qmu).numpy().astype(np.float64)
    del full
    err = Qpred - Qho
    real = {"alpha": best_a, "inner_mean_seq_r2": round(best_s, 5),
            "heldout_speed_r2_centre": round(AP.r2_score(Qpred[:, K],
                                                         Qho[:, K]), 5),
            "heldout_speed_mae_centre": round(float(np.abs(err[:, K]).mean()), 5),
            "heldout_speed_sigma_centre": round(float(err[:, K].std()), 5),
            "err_autocorr_lag1_within_window": round(float(np.corrcoef(
                err[:, :-1].ravel(), err[:, 1:].ravel())[0, 1]), 4),
            "central_diff": r2ci(central_diff(Qpred), "r2_real_cd")}
    best_sg, best_cfg, best_p = None, None, None
    per_sg = []
    for half in (1, 2, 3, 4):
        for order in (2, 3):
            if 2 * half + 1 <= order + 1:
                continue
            c = savgol_coeffs(half, order)
            p = Qpred[:, K - half:K + half + 1] @ c
            s = AP.r2_score(p, y)
            per_sg.append({"half_window": half, "poly_order": order,
                           "r2": round(s, 5)})
            if best_sg is None or s > best_sg:
                best_sg, best_cfg, best_p = s, (half, order), p
    real["savgol_sweep"] = per_sg
    real["best_savgol"] = {"half_window": best_cfg[0], "poly_order": best_cfg[1],
                           "r2_ci": r2ci(best_p, "r2_real_sg")}
    #: ⚠️ CHEATING BY CONSTRUCTION — the SG stencil above is chosen on the
    #: HELD-OUT set. Quotable ONLY as an upper bound, and it is reported
    #: precisely so "we smoothed it wrong" cannot survive as an explanation.
    real["savgol_selected_on_heldout"] = True
    res["real_latent_track"] = real
    log(f"REAL latent speed track: centre R2 {real['heldout_speed_r2_centre']:+.4f} "
        f"sigma {real['heldout_speed_sigma_centre']:.3f} m/s  ->  derived accel "
        f"cd {real['central_diff']['point']:+.4f} "
        f"sgBEST {real['best_savgol']['r2_ci']['point']:+.4f}")

    # --- the one-mechanism test -------------------------------------------- #
    # Does the AR(1) ladder AT THE REAL TRACK'S OWN SIGMA reproduce the real
    # track's derived-accel score? If yes, "the latent lacks accel" and "the
    # latent reads speed too coarsely" are ONE finding, not two.
    sig_real = real["heldout_speed_sigma_centre"]
    rho_real = real["err_autocorr_lag1_within_window"]
    noise = ar1_noise(len(Qho), w, sig_real, rho_real,
                      np.random.default_rng(4242))
    Qm = Qho + noise
    c = savgol_coeffs(best_cfg[0], best_cfg[1])
    pm = Qm[:, K - best_cfg[0]:K + best_cfg[0] + 1] @ c
    res["one_mechanism_test"] = {
        "note": "TRUE speed track corrupted with AR(1) noise matched to the "
                "REAL latent-read error's sigma AND lag-1 autocorrelation, then "
                "put through the SAME best-SG derivative. If this lands where "
                "the real track lands, the accel null is FULLY EXPLAINED by the "
                "speed-read precision — one mechanism, not two.",
        "matched_sigma_mps": sig_real, "matched_rho": rho_real,
        "simulated_r2": r2ci(pm, "r2_matched_sim"),
        "real_r2": real["best_savgol"]["r2_ci"],
        "paired_delta_sim_minus_real": APCI.paired_stat_episode_cluster_bootstrap(
            lambda sel, p=pm: AP.r2_score(p[sel], y[sel]),
            lambda sel, p=best_p: AP.r2_score(p[sel], y[sel]),
            eid, n_boot=args.n_boot, name="delta_sim_minus_real")}
    log("one-mechanism: sim "
        f"{res['one_mechanism_test']['simulated_r2']['point']:+.4f} vs real "
        f"{real['best_savgol']['r2_ci']['point']:+.4f}  delta "
        f"{res['one_mechanism_test']['paired_delta_sim_minus_real']['delta']:+.4f} "
        f"sep={res['one_mechanism_test']['paired_delta_sim_minus_real']['separated']}")

    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"-> {args.out}")


if __name__ == "__main__":
    main()
