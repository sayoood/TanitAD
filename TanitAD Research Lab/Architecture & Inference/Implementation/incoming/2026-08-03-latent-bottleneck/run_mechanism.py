"""D-LATENT part 3 — THE MECHANISM, in one number: the latent's speed track is
FLAT INSIDE THE WINDOW.

WHAT PART 2 LEFT UNEXPLAINED
    The one-mechanism test FAILED, and the failure is the finding. A synthetic
    speed track built from the TRUE speed plus AR(1) noise matched to the real
    latent-read error's sigma (5.943 m/s) AND its within-window lag-1
    autocorrelation (0.9999) yields derived accel R2 **+0.6495**; the REAL
    latent-read track yields **-0.3773** (paired delta +1.0268, SEPARATED).

    Same sigma, same lag-1 autocorrelation, opposite answer. So "the latent
    reads speed too coarsely" is NOT the mechanism — a track that coarse but
    correctly structured recovers the channel.

THE HYPOTHESIS THIS TESTS
    The latent's error is ~constant WITHIN a window (rho = 0.9999) because the
    predicted track barely MOVES: the ridge maps nine highly-similar latents to
    nine nearly-identical speeds. Then the predicted derivative is not a noisy
    version of the true derivative — it is nearly ZERO plus noise, and it
    carries no information about the true one.

    ⭐ If that is right, the defect is not PRECISION and not TEMPORAL SPAN. It
    is **within-window temporal RESOLUTION**: consecutive latents inside one
    window are too similar to distinguish. That is a statement about the
    representation that is directly actionable and is NOT what the programme
    has been assuming.

WHAT IS MEASURED (all closed-form, 0 GPU-h, on the banked latent cache)
    1. The GAIN and CORRELATION of the latent's within-window derivative against
       the true one — the decisive numbers.
    2. The LATENT's own within-window self-similarity: cosine between
       consecutive window positions vs between window centres of different
       rows. If consecutive latents are nearly identical, the encoder has no
       100 ms resolution to give a decoder, however many frames it keeps.
    3. The same two quantities for the ORACLE input, as the positive control.
    4. A DIRECT-DERIVATIVE arm: fit the ridge to predict the TRUE within-window
       speed INCREMENT (not the speed) from the latent window. If the increment
       is directly regressable the "derive it" route was simply wrong; if it is
       not, the information is absent at that timescale.

usage: python run_mechanism.py --out results_mechanism.json
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

LATENTS = Path(r"C:/Users/Admin/tanitad-data/eval/idm_derived_accel_latents.pt")
K, DT = 4, 0.1
ACCEL_J = 3
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def split_episodes(eps, hold_every=3):
    return ([e for i, e in enumerate(eps) if i % hold_every != 0],
            [e for i, e in enumerate(eps) if i % hold_every == 0])


def fit_seq(AP, Xf, Yf, Xs, Ys, XF, YF, Xh, device):
    """Ridge with alpha chosen on the inner split by the MEAN R2 over the
    predicted sequence's columns. Returns the held-out prediction."""
    Xf, Xs = AP.standardize(Xf, Xs)
    XF, Xh = AP.standardize(XF, Xh)
    mu, sd = YF.mean(0, keepdim=True), YF.std(0, keepdim=True).clamp_min(1e-6)
    kw = dict(kernel="linear", matmul_device=device, matmul_dtype=torch.float32)
    alphas = AP.DualRidge.alpha_grid(2, -4, 10)
    inner = AP.DualRidge(Xf, ((Yf - mu) / sd).double(), **kw)
    Ys_np = Ys.numpy().astype(np.float64)
    best_a, best_s = None, -1e18
    for al in alphas:
        p = (inner.predict(Xs, al) * sd + mu).numpy()
        s = float(np.mean([AP.r2_score(p[:, j], Ys_np[:, j])
                           for j in range(p.shape[1])]))
        if s > best_s:
            best_a, best_s = al, s
    del inner
    full = AP.DualRidge(XF, ((YF - mu) / sd).double(), **kw)
    out = (full.predict(Xh, best_a) * sd + mu).numpy().astype(np.float64)
    del full
    return out, float(best_a), round(best_s, 5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_mechanism.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from tanitad.eval import accel_probe as AP
    from tanitad.eval import ap_ci as APCI

    sub = torch.load(LATENTS, map_location="cpu", weights_only=False)
    eps = sub["episodes"]
    tr_eps, ho_eps = split_episodes(eps)
    fit_eps, sel_eps = split_episodes(tr_eps)

    def cat(el, k):
        return torch.cat([e[k] for e in el]).float()

    Qho = cat(ho_eps, "Q").numpy().astype(np.float64)
    Sho = cat(ho_eps, "S").numpy().astype(np.float64)
    eid = np.concatenate([np.full(e["n"], e["name"]) for e in ho_eps])
    y = Sho[:, ACCEL_J]
    res = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "substrate": {"latent_cache": str(LATENTS),
                         "n_heldout_windows": int(len(y)),
                         "n_heldout_episodes": int(len(ho_eps))},
           "estimator": "episode_cluster_bootstrap (tanitad.eval.ap_ci)"}

    def r2ci(pred, name):
        return APCI.stat_episode_cluster_bootstrap(
            lambda sel, p=np.asarray(pred, np.float64): AP.r2_score(p[sel],
                                                                    y[sel]),
            eid, n_boot=args.n_boot, name=name)

    # ---- 1/2: predict the speed SEQUENCE from the latent window ------------ #
    Xz = {t: AP.window_features(cat(e, "Z"), "window")
          for t, e in (("fit", fit_eps), ("sel", sel_eps), ("full", tr_eps),
                       ("ho", ho_eps))}
    Qs = {t: cat(e, "Q") for t, e in (("fit", fit_eps), ("sel", sel_eps),
                                      ("full", tr_eps), ("ho", ho_eps))}
    Qpred, alpha, inner = fit_seq(AP, Xz["fit"], Qs["fit"], Xz["sel"],
                                  Qs["sel"], Xz["full"], Qs["full"], Xz["ho"],
                                  args.device)
    d_true = np.diff(Qho, axis=1) / DT            # [N, 8] true increments (m/s^2)
    d_pred = np.diff(Qpred, axis=1) / DT
    err = Qpred - Qho
    d_err = np.diff(err, axis=1) / DT
    flat = {
        "speed_r2_centre": round(AP.r2_score(Qpred[:, K], Qho[:, K]), 5),
        "speed_sigma_centre_mps": round(float(err[:, K].std()), 5),
        "TRUE_within_window_increment_std_mps2": round(float(d_true.std()), 5),
        "PRED_within_window_increment_std_mps2": round(float(d_pred.std()), 5),
        "gain_pred_over_true": round(float(d_pred.std() / max(d_true.std(),
                                                              1e-9)), 4),
        "corr_pred_vs_true_increment": round(float(np.corrcoef(
            d_pred.ravel(), d_true.ravel())[0, 1]), 5),
        "ERROR_increment_std_mps2": round(float(d_err.std()), 5),
        "snr_true_over_error_increment": round(
            float(d_true.std() / max(d_err.std(), 1e-9)), 5),
        "err_autocorr_lag1_within_window": round(float(np.corrcoef(
            err[:, :-1].ravel(), err[:, 1:].ravel())[0, 1]), 5),
        "ridge_alpha": alpha, "inner_mean_seq_r2": inner}
    res["latent_speed_track"] = flat
    log(f"latent track: TRUE dv/dt std {flat['TRUE_within_window_increment_std_mps2']}  "
        f"PRED {flat['PRED_within_window_increment_std_mps2']}  "
        f"gain {flat['gain_pred_over_true']}  "
        f"corr {flat['corr_pred_vs_true_increment']}  "
        f"errdiff std {flat['ERROR_increment_std_mps2']}  "
        f"SNR {flat['snr_true_over_error_increment']}")

    # ---- the latent's OWN within-window self-similarity -------------------- #
    Zho = cat(ho_eps, "Z")                                   # [N, 9, D]
    zn = torch.nn.functional.normalize(Zho.double(), dim=-1)
    cos_adj = float((zn[:, 1:] * zn[:, :-1]).sum(-1).mean())
    cos_ends = float((zn[:, -1] * zn[:, 0]).sum(-1).mean())
    perm = torch.randperm(len(zn), generator=torch.Generator().manual_seed(0))
    cos_rows = float((zn[:, K] * zn[perm, K]).sum(-1).mean())
    res["latent_self_similarity"] = {
        "cos_adjacent_window_positions_100ms": round(cos_adj, 5),
        "cos_window_ends_800ms": round(cos_ends, 5),
        "cos_random_other_row_centre": round(cos_rows, 5),
        "note": "If adjacent-position cosine is at the ceiling while a random "
                "other row is far away, the representation has ~no 100 ms "
                "resolution: it distinguishes SCENES, not INSTANTS. Keeping W "
                "frames of such a latent adds W copies of the same vector."}
    log(f"latent cos: adjacent {cos_adj:.5f}  ends(800ms) {cos_ends:.5f}  "
        f"random-row {cos_rows:.5f}")

    # ---- 3: the ORACLE positive control on the identical quantities -------- #
    Xq = {t: AP.window_features(Qs[t].unsqueeze(-1), "window")
          for t in Qs}
    Opred, o_alpha, o_inner = fit_seq(AP, Xq["fit"], Qs["fit"], Xq["sel"],
                                      Qs["sel"], Xq["full"], Qs["full"],
                                      Xq["ho"], args.device)
    od_pred = np.diff(Opred, axis=1) / DT
    res["oracle_speed_track"] = {
        "speed_r2_centre": round(AP.r2_score(Opred[:, K], Qho[:, K]), 5),
        "PRED_within_window_increment_std_mps2": round(float(od_pred.std()), 5),
        "gain_pred_over_true": round(float(od_pred.std()
                                           / max(d_true.std(), 1e-9)), 4),
        "corr_pred_vs_true_increment": round(float(np.corrcoef(
            od_pred.ravel(), d_true.ravel())[0, 1]), 5),
        "ridge_alpha": o_alpha, "inner_mean_seq_r2": o_inner}
    log(f"oracle track: gain {res['oracle_speed_track']['gain_pred_over_true']}  "
        f"corr {res['oracle_speed_track']['corr_pred_vs_true_increment']}")

    # ---- 4: regress the INCREMENT directly, latent and oracle -------------- #
    # If the derivative is directly regressable the "derive it from speed" route
    # was simply the wrong parameterisation; if it is not, the timescale is what
    # is missing. Target = the FULL increment sequence, so the arm has the same
    # contract on both substrates.
    Dt = {t: (torch.diff(Qs[t], dim=1) / DT) for t in Qs}
    for tag, X in (("latent", Xz), ("oracle", Xq)):
        Dp, a, ii = fit_seq(AP, X["fit"], Dt["fit"], X["sel"], Dt["sel"],
                            X["full"], Dt["full"], X["ho"], args.device)
        centre_inc = Dp[:, K - 1:K + 1].mean(1)      # the centred derivative
        res[f"direct_increment_{tag}"] = {
            "ridge_alpha": a, "inner_mean_seq_r2": ii,
            "PRED_increment_std_mps2": round(float(Dp.std()), 5),
            "corr_pred_vs_true_increment": round(float(np.corrcoef(
                Dp.ravel(), d_true.ravel())[0, 1]), 5),
            "r2_vs_CAN_long_accel": r2ci(centre_inc,
                                         f"r2_direct_increment_{tag}")}
        log(f"direct-increment {tag}: corr "
            f"{res[f'direct_increment_{tag}']['corr_pred_vs_true_increment']}  "
            f"accel R2 "
            f"{res[f'direct_increment_{tag}']['r2_vs_CAN_long_accel']['point']:+.4f}")

    # ---- 5: the SHARED PER-POSITION readout — the sharpest form ------------ #
    # ⭐ Everything above fits the WHOLE 18,432-d window to each position's
    # speed, so a position's prediction may be carried by other positions'
    # latents. The cleanest statement of "does the latent MOVE along its own
    # speed direction" is a SINGLE shared readout applied per position:
    #     v_hat_j = w . z_j   ->   dv_hat_j = w . (z_{j+1} - z_j).
    # Fit `w` on the train rows pooled over all positions (so it is the same
    # linear speed direction the centre read uses), then look at what its
    # per-position application does to the derivative. If THIS is also near
    # zero correlation, the representation genuinely does not move along its own
    # speed axis at 100 ms — no head or routing can retrieve what is not there.
    def pooled(el):
        Z = cat(el, "Z")                        # [N, 9, D]
        Q = cat(el, "Q")                        # [N, 9]
        return Z.reshape(-1, Z.shape[-1]), Q.reshape(-1, 1)

    Zp_f, Qp_f = pooled(fit_eps)
    Zp_s, Qp_s = pooled(sel_eps)
    Zp_F, Qp_F = pooled(tr_eps)
    Zh_flat = cat(ho_eps, "Z")
    n_ho, wpos, dz = Zh_flat.shape
    Xp_f, Xp_s = AP.standardize(Zp_f, Zp_s)
    Xp_F, Xp_h = AP.standardize(Zp_F, Zh_flat.reshape(-1, dz))
    mu, sd = Qp_F.mean(0, keepdim=True), Qp_F.std(0, keepdim=True).clamp_min(1e-6)
    kw = dict(kernel="linear", matmul_device=args.device,
              matmul_dtype=torch.float32)
    alphas = AP.DualRidge.alpha_grid(1, -2, 8)
    inner = AP.DualRidge(Xp_f[::3], ((Qp_f[::3] - mu) / sd).double(), **kw)
    Qs_np = Qp_s.numpy().astype(np.float64).ravel()
    best_a, best_s = None, -1e18
    for al in alphas:
        p = (inner.predict(Xp_s, al) * sd + mu).numpy().ravel()
        s = AP.r2_score(p, Qs_np)
        if s > best_s:
            best_a, best_s = al, s
    del inner
    full = AP.DualRidge(Xp_F[::3], ((Qp_F[::3] - mu) / sd).double(), **kw)
    vhat = (full.predict(Xp_h, best_a) * sd + mu).numpy().reshape(n_ho, wpos)
    del full
    dv = np.diff(vhat, axis=1) / DT
    res["shared_perposition_readout"] = {
        "note": "ONE linear speed direction, applied to each window position "
                "independently. Its per-position difference is the latent's own "
                "motion projected onto its speed axis.",
        "ridge_alpha": float(best_a), "inner_pooled_speed_r2": round(best_s, 5),
        "heldout_speed_r2_centre": round(AP.r2_score(vhat[:, K], Qho[:, K]), 5),
        "PRED_increment_std_mps2": round(float(dv.std()), 5),
        "TRUE_increment_std_mps2": round(float(d_true.std()), 5),
        "corr_pred_vs_true_increment": round(float(np.corrcoef(
            dv.ravel(), d_true.ravel())[0, 1]), 5),
        "r2_vs_CAN_long_accel_centred": r2ci(
            0.5 * (dv[:, K - 1] + dv[:, K]), "r2_shared_readout_centred")}
    sp = res["shared_perposition_readout"]
    log(f"shared per-position readout: centre speed R2 "
        f"{sp['heldout_speed_r2_centre']:+.4f}  incr std {sp['PRED_increment_std_mps2']} "
        f"(true {sp['TRUE_increment_std_mps2']})  corr "
        f"{sp['corr_pred_vs_true_increment']}  accel R2 "
        f"{sp['r2_vs_CAN_long_accel_centred']['point']:+.4f}")

    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"-> {args.out}")


if __name__ == "__main__":
    main()
