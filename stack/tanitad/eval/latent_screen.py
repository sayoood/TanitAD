"""THE 0-GPU LATENT SCREEN — reject a temporally-collapsed latent BEFORE paying for a run.

⭐ WHY THIS IS A REPO INSTRUMENT AND NOT A ONE-OFF SCRIPT
    ``dynenc-branchB`` spent **40 000 training steps** producing a dynamics substrate that
    is WEAKER than plain flagship-v1 (in-domain rig-A speed R² **+0.039 / −0.603** vs
    flagship-v1's **+0.862 / +0.910**, ``MODEL_REGISTRY.md`` §10.1). The three numbers below
    are computable from a banked latent cache in **minutes** and would have rejected it.
    Origin: ``…/incoming/2026-08-03-latent-bottleneck/LATENT_BOTTLENECK.md`` §6, promoted here
    by the appearance-shortcut audit so that no future encoder arm can be launched unscreened.

THE FOUR NUMBERS — three gates and one diagnostic

    1  ⭐ JITTER RATIO      std( w·(z_{j+1} − z_j) ) / std( true dv/dt )     gate ≤ 2×
       Fit ONE linear speed direction ``w`` on the training rows pooled over all window
       positions, then apply it PER POSITION. Its per-position difference is the latent's own
       motion projected onto its own speed axis. Frozen v1 scores **51.0×**: the representation
       moves 51 times more along its speed direction than the car's speed actually changes,
       and that movement correlates **+0.0061** with the truth. Stacking W frames of such a
       latent stacks W near-copies whose differences are noise.

    2  DERIVATIVE CORR     corr( within-window speed derivative, true derivative )   gate > 0.50
       Frozen v1: **+0.0891** (per-position form **+0.0061**); oracle **+0.99997**.

    3  DERIVED ACCEL R²    long_accel derived from the predicted speed track            gate > +0.50
       Frozen v1 **−0.3773**, oracle **+0.9277**.
       ⚠️ EQUIVALENTLY a speed-read precision of **σ ≤ 0.28 m/s**, and that number is
       **ESTIMATOR-SPECIFIC**: it is the optimal **9-point Savitzky-Golay** derivative. The
       older programme figure *σ ≲ 0.1 m/s (~47×)* is the same requirement under a **2-point
       centred difference**. :data:`SIGMA_ESTIMATOR` travels with the number in every output
       dict on purpose — a bare σ has been mis-quoted across estimators before.

    4  COS ADJACENT (diagnostic, NOT a gate)   mean cosine between latents 100 ms apart
       Frozen v1 **0.98825** at 100 ms, **0.91398** across 800 ms, **0.32770** against a random
       other row: the representation separates SCENES far more than INSTANTS. A high cosine
       WITH a high jitter ratio means the little motion there is, is off-axis noise.

⚠️ CALIBRATION STATUS. The three thresholds were proposed from ONE encoder (flagship-v1 on
   comma2k19) plus an oracle. :func:`screen_latent` records ``calibration_n_latents`` so a
   caller can see how thin that is. They are re-calibrated by ADDING screened latents to the
   record, never by moving a threshold to make an arm pass.

⛔ WHAT THIS SCREEN DOES NOT DO. It says nothing about appearance shortcuts, about lateral or
   strategic content, or about whether a channel is in the video at all. A latent can pass
   every gate here and still read ``speed`` from static appearance — that is a different
   instrument (:func:`still_frame_ratio`-style contrasts, D-APPEAR).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .accel_probe import DualRidge, r2_score, standardize

__all__ = ["SCREEN_GATES", "SIGMA_ESTIMATOR", "REFERENCE_LATENTS", "savgol_coeffs",
           "shared_speed_direction", "screen_latent", "format_screen"]

DT = 0.1                                       # 10 Hz contract

#: The pre-flight gates. Origin LATENT_BOTTLENECK.md §6. ⛔ Never move a threshold to make an
#: arm pass; re-calibrate only by adding screened latents to :data:`REFERENCE_LATENTS`.
SCREEN_GATES: dict[str, float] = {
    "jitter_ratio_max": 2.0,
    "derivative_corr_min": 0.50,
    "derived_accel_r2_min": 0.50,
    "speed_sigma_max_mps": 0.28,
}

#: ⚠️ The estimator that the σ gate is stated against. Carried in every output dict.
SIGMA_ESTIMATOR = ("9-point Savitzky-Golay derivative (half_window=4, poly_order=2, dt=0.1 s). "
                   "The older programme figure 'sigma <= 0.1 m/s, ~47x' is the SAME requirement "
                   "under a 2-POINT CENTRED DIFFERENCE. Never quote the sigma without this.")

#: Every latent that has been screened, so the thin calibration is visible rather than implied.
REFERENCE_LATENTS: dict[str, dict[str, Any]] = {
    "flagship_v1_step29999_comma2k19": {
        "jitter_ratio": 51.0, "derivative_corr": 0.0891, "derived_accel_r2": -0.3773,
        "cos_adjacent_100ms": 0.98825, "verdict": "FAIL",
        "source": "…/incoming/2026-08-03-latent-bottleneck/results_mechanism.json"},
    "ORACLE_true_speed_window": {
        "jitter_ratio": 1.0, "derivative_corr": 0.99997, "derived_accel_r2": 0.9277,
        "verdict": "PASS",
        "source": "…/incoming/2026-08-03-latent-bottleneck/results_mechanism.json"},
}


# --------------------------------------------------------------------------- #
def savgol_coeffs(half: int, order: int, deriv: int = 1, dt: float = DT) -> np.ndarray:
    """Savitzky-Golay convolution coefficients for the ``deriv``-th derivative.

    A 2-point centred difference differentiates the noise of a noisy prediction; an SG
    derivative fits a local polynomial first and is the reason the precision requirement is
    σ ≤ 0.28 m/s rather than σ ≤ 0.1 m/s.
    """
    if 2 * half + 1 <= order:
        raise ValueError(f"window {2*half+1} too short for order {order}")
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, order + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    from math import factorial
    return pinv[deriv] * (factorial(deriv) / dt ** deriv)


def shared_speed_direction(Z_fit: torch.Tensor, Q_fit: torch.Tensor,
                           Z_sel: torch.Tensor, Q_sel: torch.Tensor,
                           Z_full: torch.Tensor, Q_full: torch.Tensor,
                           Z_hold: torch.Tensor, *, device: str = "cpu",
                           fit_stride: int = 3) -> tuple[np.ndarray, float, float]:
    """ONE linear speed readout ``w`` fitted on POOLED window positions, applied per position.

    Returns ``(v_hat [N_hold, W], alpha, inner_pooled_speed_r2)``.

    Pooling the positions is what makes ``w`` a single *speed direction* rather than a
    position-specific decoder: everything the whole-window ridge can do by borrowing other
    positions' latents is removed, so the per-position difference ``w·(z_{j+1} − z_j)`` is the
    latent's OWN motion along its OWN speed axis and nothing else.

    ``fit_stride`` subsamples the pooled rows before the eigendecomposition (the dual is
    O(n³)); 3 is the value the reference measurement used.
    """
    d = Z_fit.shape[-1]
    Xf, Xs = standardize(Z_fit.reshape(-1, d), Z_sel.reshape(-1, d))
    XF, Xh = standardize(Z_full.reshape(-1, d), Z_hold.reshape(-1, d))
    qf = Q_fit.reshape(-1, 1)
    qF = Q_full.reshape(-1, 1)
    mu, sd = qF.mean(0, keepdim=True), qF.std(0, keepdim=True).clamp_min(1e-6)
    kw = dict(kernel="linear", matmul_device=device, matmul_dtype=torch.float32)
    alphas = DualRidge.alpha_grid(1, -2, 8)
    inner = DualRidge(Xf[::fit_stride], ((qf[::fit_stride] - mu) / sd).double(), **kw)
    q_sel = Q_sel.reshape(-1).numpy().astype(np.float64)
    best_a, best_s = None, -1e18
    for al in alphas:
        p = (inner.predict(Xs, al) * sd + mu).numpy().ravel()
        s = r2_score(p, q_sel)
        if s > best_s:
            best_a, best_s = al, s
    del inner
    full = DualRidge(XF[::fit_stride], ((qF[::fit_stride] - mu) / sd).double(), **kw)
    v_hat = (full.predict(Xh, best_a) * sd + mu).numpy().reshape(Z_hold.shape[:2])
    del full
    return v_hat, float(best_a), float(best_s)


# --------------------------------------------------------------------------- #
def screen_latent(Z_fit: torch.Tensor, Q_fit: torch.Tensor,
                  Z_sel: torch.Tensor, Q_sel: torch.Tensor,
                  Z_full: torch.Tensor, Q_full: torch.Tensor,
                  Z_hold: torch.Tensor, Q_hold: torch.Tensor,
                  *, name: str = "candidate", dt: float = DT,
                  device: str = "cpu", gates: dict[str, float] | None = None,
                  fit_stride: int = 3, sg_half: int = 4, sg_order: int = 2,
                  ) -> dict[str, Any]:
    """Run the pre-flight screen on ONE candidate latent. 0 GPU required (``device='cpu'``).

    ``Z_*`` are ``[N, W, D]`` latent windows and ``Q_*`` the matching ``[N, W]`` TRUE speed
    tracks (metres/second) at the same window positions. The three splits are the programme's
    standard: an inner fit/sel pair carved out of TRAIN for hyper-parameter selection, the full
    TRAIN for the refit, and an EPISODE-DISJOINT held-out set for every quoted number.

    ⛔ The caller owns the episode-disjointness. This function cannot check it and does not
    pretend to; passing overlapping episodes produces a screen that PASSES things it should
    reject, which is the failure mode that matters.
    """
    gates = dict(SCREEN_GATES if gates is None else gates)
    for t in (Z_fit, Z_sel, Z_full, Z_hold):
        if t.ndim != 3:
            raise ValueError(f"Z must be [N, W, D], got {tuple(t.shape)}")
    if Z_hold.shape[:2] != Q_hold.shape:
        raise ValueError(f"Z_hold {tuple(Z_hold.shape)} and Q_hold "
                         f"{tuple(Q_hold.shape)} disagree on [N, W]")
    W = Z_hold.shape[1]
    if W < 3:
        raise ValueError(f"screen needs W >= 3 window positions, got {W}")
    c = W // 2

    Qh = np.asarray(Q_hold, dtype=np.float64)
    d_true = np.diff(Qh, axis=1) / dt

    # --- 1. the SHARED per-position readout -> jitter ratio + per-position corr --- #
    v_hat, alpha, inner_r2 = shared_speed_direction(
        Z_fit, Q_fit, Z_sel, Q_sel, Z_full, Q_full, Z_hold,
        device=device, fit_stride=fit_stride)
    dv = np.diff(v_hat, axis=1) / dt
    jitter = float(dv.std() / max(d_true.std(), 1e-12))
    corr_pp = (float(np.corrcoef(dv.ravel(), d_true.ravel())[0, 1])
               if dv.std() > 0 and d_true.std() > 0 else 0.0)
    speed_r2_shared = r2_score(v_hat[:, c], Qh[:, c])
    sigma = float((v_hat[:, c] - Qh[:, c]).std())

    # --- 2/3. the WHOLE-WINDOW speed-track read -> derivative corr + derived accel --- #
    Xf = Z_fit.reshape(len(Z_fit), -1)
    Xs = Z_sel.reshape(len(Z_sel), -1)
    XF = Z_full.reshape(len(Z_full), -1)
    Xh = Z_hold.reshape(len(Z_hold), -1)
    Xf, Xs = standardize(Xf, Xs)
    XF, Xh = standardize(XF, Xh)
    mu, sd = Q_full.mean(0, keepdim=True), Q_full.std(0, keepdim=True).clamp_min(1e-6)
    kw = dict(kernel="linear", matmul_device=device, matmul_dtype=torch.float32)
    inner = DualRidge(Xf, ((Q_fit - mu) / sd).double(), **kw)
    Qs_np = np.asarray(Q_sel, dtype=np.float64)
    best_a, best_s = None, -1e18
    for al in DualRidge.alpha_grid(2, -4, 10):
        p = (inner.predict(Xs, al) * sd + mu).numpy()
        s = float(np.mean([r2_score(p[:, j], Qs_np[:, j]) for j in range(p.shape[1])]))
        if s > best_s:
            best_a, best_s = al, s
    del inner
    full = DualRidge(XF, ((Q_full - mu) / sd).double(), **kw)
    Qpred = (full.predict(Xh, best_a) * sd + mu).numpy().astype(np.float64)
    del full
    d_pred = np.diff(Qpred, axis=1) / dt
    corr_win = (float(np.corrcoef(d_pred.ravel(), d_true.ravel())[0, 1])
                if d_pred.std() > 0 else 0.0)
    sigma_win = float((Qpred[:, c] - Qh[:, c]).std())

    # derived long_accel via the FIXED reference SG stencil, plus a swept upper bound
    def sg_accel(P, half, order):
        if 2 * half + 1 > W:
            return None
        cf = savgol_coeffs(half, order, 1, dt)
        return (P[:, c - half:c + half + 1] * cf[None, :]).sum(1)

    a_true = sg_accel(Qh, sg_half, sg_order)
    a_ref = sg_accel(Qpred, sg_half, sg_order)
    derived_r2 = None if a_ref is None else r2_score(a_ref, a_true)
    sweep, best_sweep, best_cfg = [], -1e18, None
    for half in range(1, W // 2 + 1):
        for order in (1, 2, 3):
            if 2 * half + 1 <= order:
                continue
            p = sg_accel(Qpred, half, order)
            if p is None:
                continue
            s = r2_score(p, a_true)
            sweep.append({"half_window": half, "poly_order": order, "r2": round(s, 5)})
            if s > best_sweep:
                best_sweep, best_cfg = s, (half, order)

    # --- 4. the diagnostic cosines --- #
    zn = torch.nn.functional.normalize(Z_hold.double(), dim=-1)
    cos_adj = float((zn[:, 1:] * zn[:, :-1]).sum(-1).mean())
    cos_ends = float((zn[:, -1] * zn[:, 0]).sum(-1).mean())
    perm = torch.randperm(len(zn), generator=torch.Generator().manual_seed(0))
    cos_rows = float((zn[:, c] * zn[perm, c]).sum(-1).mean())

    screens = {
        "jitter_ratio": {
            "value": round(jitter, 4), "gate": f"<= {gates['jitter_ratio_max']}",
            "pass": bool(jitter <= gates["jitter_ratio_max"]),
            "definition": "std(w.(z_{j+1}-z_j)) / std(true dv/dt), ONE shared linear speed "
                          "direction applied per window position",
            "pred_increment_std_mps2": round(float(dv.std()), 5),
            "true_increment_std_mps2": round(float(d_true.std()), 5)},
        "derivative_corr": {
            "value": round(corr_win, 5), "gate": f"> {gates['derivative_corr_min']}",
            "pass": bool(corr_win > gates["derivative_corr_min"]),
            "per_position_form": round(corr_pp, 5),
            "definition": "corr(within-window speed derivative, true derivative); the "
                          "per-position form uses the shared readout of screen 1"},
        "derived_accel_r2": {
            "value": None if derived_r2 is None else round(derived_r2, 5),
            "gate": f"> {gates['derived_accel_r2_min']}",
            "pass": bool(derived_r2 is not None
                         and derived_r2 > gates["derived_accel_r2_min"]),
            "upper_bound_best_stencil": round(best_sweep, 5) if best_cfg else None,
            "upper_bound_cfg": ({"half_window": best_cfg[0], "poly_order": best_cfg[1]}
                                if best_cfg else None),
            "upper_bound_note": "the stencil for the upper bound is chosen ON THE HELD-OUT "
                                "SET and is CHEATING BY CONSTRUCTION -- quotable only as an "
                                "upper bound. If even this fails the gate, the reject is safe.",
            "sweep": sweep,
            "estimator": SIGMA_ESTIMATOR},
        "speed_sigma_mps": {
            "value": round(sigma_win, 5), "gate": f"<= {gates['speed_sigma_max_mps']}",
            "pass": bool(sigma_win <= gates["speed_sigma_max_mps"]),
            "shared_readout_form": round(sigma, 5),
            "estimator": SIGMA_ESTIMATOR},
        "cos_adjacent_100ms": {
            "value": round(cos_adj, 5), "gate": "DIAGNOSTIC (not a gate)", "pass": None,
            "cos_window_ends": round(cos_ends, 5),
            "cos_random_other_row": round(cos_rows, 5),
            "note": "a high adjacent cosine WITH a high jitter ratio means the motion the "
                    "latent does have is off-axis noise"},
    }
    gated = [k for k, v in screens.items() if v["pass"] is not None]
    failed = [k for k in gated if not screens[k]["pass"]]
    return {
        "name": name,
        "verdict": "PASS" if not failed else "FAIL",
        "failed_screens": failed,
        "screens": screens,
        "fit": {"shared_readout_alpha": alpha,
                "shared_readout_inner_pooled_speed_r2": round(inner_r2, 5),
                "shared_readout_heldout_speed_r2_centre": round(speed_r2_shared, 5),
                "window_ridge_alpha": float(best_a),
                "window_ridge_inner_mean_seq_r2": round(best_s, 5),
                "window_ridge_heldout_speed_r2_centre": round(
                    r2_score(Qpred[:, c], Qh[:, c]), 5)},
        "shape": {"n_heldout_windows": int(len(Z_hold)), "W": int(W),
                  "D": int(Z_hold.shape[-1]), "dt_s": dt},
        "gates": gates,
        "sigma_estimator": SIGMA_ESTIMATOR,
        "calibration_n_latents": len(REFERENCE_LATENTS),
        "calibration_note": "thresholds proposed from ONE encoder plus an oracle "
                            "(LATENT_BOTTLENECK.md §6); re-calibrate by ADDING screened "
                            "latents, never by moving a threshold to pass an arm.",
        "does_not_test": "appearance shortcuts, lateral/strategic content, or whether a "
                         "channel is present in the video at all",
    }


def format_screen(res: dict[str, Any]) -> str:
    """One compact human-readable block — the thing a launch runbook prints."""
    lines = [f"LATENT SCREEN  {res['name']}  ->  {res['verdict']}"]
    for k, v in res["screens"].items():
        mark = "  " if v["pass"] is None else ("OK" if v["pass"] else "XX")
        lines.append(f"  {mark} {k:22s} {str(v['value']):>10s}   gate {v['gate']}")
    if res["failed_screens"]:
        lines.append(f"  FAILED: {', '.join(res['failed_screens'])}")
    lines.append(f"  sigma estimator: {res['sigma_estimator'].split('.')[0]}")
    return "\n".join(lines)
