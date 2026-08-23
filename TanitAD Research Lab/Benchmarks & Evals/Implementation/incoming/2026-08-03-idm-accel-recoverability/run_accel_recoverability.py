"""STREAM D — is ``long_accel`` UNRECOVERABLE from the frozen v1 latents, or
merely UNRECOVERED by the one head that was tried?

THE OPEN QUESTION (from P9 / `…/2026-08-03-idm-derived-accel/`)
    The shipped IDM head scores R² −0.15…−0.42 on ``long_accel`` while ``speed``
    scores +0.72…+0.86, and the BASELINE's ``long_accel`` is **not separated**
    from a shuffled-latent control (−0.0984 [−0.3087, +0.0179]) while ``speed``
    (+0.7187), ``yaw_rate`` (+0.2252) and ADE (−6.59 m) all ARE. One head, one
    geometry, one training recipe. That evidence cannot distinguish "the latents
    do not carry it" from "this head did not find it", and the two point at
    opposite work: fix the representation, or fix the head.

WHAT MAKES THIS DECISIVE, and why each piece is not optional
    1. A CAPACITY LADDER whose top rung is not a neural network — exact kernel
       ridge over the full regularisation path, linear AND rbf, on four feature
       bases including a DERIVATIVE basis. No learning rate, no epoch budget, no
       initialisation to blame.
    2. A DETECTION-SENSITIVITY sweep: the same probe re-run on latents with a
       KNOWN multiple of the target planted along a random direction. This turns
       an unfalsifiable "nothing is there" into "nothing above X % of latent RMS,
       at this n". ⚠️ Without it a null result is indistinguishable from a blunt
       probe.
    3. A CAPACITY CONTROL on every new head: the SAME architecture fed the TRUE
       speed window must reach the oracle. If it does, capacity is not the limit.
    4. IN-SAMPLE (train) R² for every arm: separates "cannot fit" from "cannot
       generalise" — a distinction no held-out number can make.
    5. A matched SHUFFLED-LATENT control for every arm, paired episode-cluster
       bootstrap, and — per the binding rule — the FOUR METRIC FAMILIES, never
       ADE alone.

PRE-REGISTERED READING (both outcomes committed in advance)
    * If NO arm's ``long_accel`` ΔR² vs its own control is separated, while the
      SAME arms separate on ``speed``/``yaw_rate`` and the oracle-input control
      reaches ~0.9 → **UNRECOVERABLE at this representation and this n**, and the
      programme should stop tuning IDM heads for this channel.
    * If ANY arm separates → **UNRECOVERED**, and that arm's geometry is the fix.
    * The detection sweep reports the floor either way, so the null is quotable
      with a magnitude instead of as an absence.

usage:
  python run_accel_recoverability.py --stages all --out results_accel_recoverability.json
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
K, STRIDE = 4, 2
HORIZONS = (5, 10, 15, 20)
SCALARS = ("speed", "yaw_rate", "steer", "long_accel")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# --------------------------------------------------------------------------- #
# substrate                                                                   #
# --------------------------------------------------------------------------- #
def load_substrate():
    sub = torch.load(LATENTS, map_location="cpu", weights_only=False)
    return sub


def split_episodes(eps, hold_every: int = 3):
    """EPISODE-DISJOINT, IDENTICAL to the prior panel so the contrast is on the
    same windows and the same 17 held-out episodes."""
    tr = [e for i, e in enumerate(eps) if i % hold_every != 0]
    ho = [e for i, e in enumerate(eps) if i % hold_every == 0]
    return tr, ho


def stack(eps, shuffle_seed: int | None = None):
    """-> Z [N,W,D] float32, Y [N,12], eid [N].

    ``Y`` = 4 scalars then 8 flattened waypoint coordinates, so EVERY arm —
    closed-form or neural — emits the same contract and the four families are
    computable for all of them.
    """
    Z = torch.cat([e["Z"] for e in eps]).float()
    S = torch.cat([e["S"] for e in eps]).float()
    T = torch.cat([e["T"] for e in eps]).float()
    eid = np.concatenate([np.full(e["n"], e["name"]) for e in eps])
    if shuffle_seed is not None:
        # NEGATIVE CONTROL: destroy the latent<->target link, keep the latent
        # marginal, the targets, the split and the fitting recipe untouched.
        g = np.random.default_rng(shuffle_seed)
        Z = Z[torch.from_numpy(g.permutation(len(Z)))]
    Y = torch.cat([S, T.reshape(len(T), -1)], 1)
    return Z, Y, eid


def unpack(Y):
    Y = np.asarray(Y, dtype=np.float64)
    return Y[:, :4], Y[:, 4:].reshape(len(Y), len(HORIZONS), 2)


# --------------------------------------------------------------------------- #
# arms                                                                        #
# --------------------------------------------------------------------------- #
SEL_TOL = 0.005          # R² tolerance for the shrinkage tie-break (see below)
MIN_SKILL = 0.01         # inner-val R² a channel must show to leave the mean


def ridge_arm(AP, Zfit, Yfit, Zsel, Zfull, Yfull, Zho, *, feat, kernel,
              gamma_mults=(None,), device="cpu"):
    """Closed-form arm. Hyperparameters are chosen PER OUTPUT on the inner split,
    the model is refit on the FULL train set, and (heldout, train) predictions
    come back keyed by ``(gamma_mult, alpha)``.

    Per-output selection is deliberate: the four channels have genuinely
    different signal-to-noise, and forcing one shrinkage on all of them would
    hand the weak channel the strong channel's setting and then call the result
    "capacity".
    """
    Xf = AP.window_features(Zfit, feat)
    Xs = AP.window_features(Zsel, feat)
    XF = AP.window_features(Zfull, feat)
    Xh = AP.window_features(Zho, feat)
    Xf, Xs = AP.standardize(Xf, Xs)
    XF, Xh = AP.standardize(XF, Xh)
    ymu, ysd = Yfull.mean(0, keepdim=True), Yfull.std(0, keepdim=True).clamp_min(1e-6)
    kwbase = dict(kernel=kernel, matmul_device=device, matmul_dtype=torch.float32)
    alphas = AP.DualRidge.alpha_grid(2, -4, 10)
    sel_preds, ho_preds, tr_preds, gammas = {}, {}, {}, {}
    for gm in gamma_mults:
        gamma = None
        if kernel == "rbf":
            # median-heuristic bandwidth x a multiplier, on a SEEDED subsample —
            # an unseeded one made the rbf arms irreproducible run to run
            # (steer swung 0.33 <-> mean-fallback purely on the draw).
            g = torch.Generator().manual_seed(20260803)
            idx = torch.randperm(len(Xf), generator=g)[:1000]
            d2 = torch.cdist(Xf[idx].double(), Xf[idx].double()).pow(2)
            gamma = gm / max(float(d2[d2 > 0].median()), 1e-9)
        gammas[gm] = gamma
        kw = dict(gamma=gamma, **kwbase)
        inner = AP.DualRidge(Xf, ((Yfit - ymu) / ysd).double(), **kw)
        for al in alphas:
            sel_preds[(gm, al)] = inner.predict(Xs, al)
        del inner
        full = AP.DualRidge(XF, ((Yfull - ymu) / ysd).double(), **kw)
        for al in alphas:
            ho_preds[(gm, al)] = full.predict(Xh, al)
            tr_preds[(gm, al)] = full.predict(XF, al)
        del full
    # EXACT mean model as a first-class member of the grid. alpha=1e10 is NOT the
    # mean — its residual is ~1e-3 of the target scale, which is enough to make a
    # paired bootstrap against the null report a spurious `separated` on a
    # difference that is numerically nothing. The sentinel makes the fallback
    # bit-identical to the train mean, so the degeneracy guard can actually fire.
    zs = {"sel": torch.zeros_like(sel_preds[keys0 := next(iter(sel_preds))]),
          "ho": torch.zeros_like(ho_preds[keys0]),
          "tr": torch.zeros_like(tr_preds[keys0])}
    MEAN_KEY = (None, float("inf"))
    sel_preds[MEAN_KEY] = zs["sel"]
    ho_preds[MEAN_KEY] = zs["ho"]
    tr_preds[MEAN_KEY] = zs["tr"]
    keys = list(sel_preds)
    return sel_preds, ho_preds, tr_preds, keys, {
        "feature": feat, "kernel": kernel, "gammas": gammas,
        "n_features": int(Xf.shape[1]), "ymu": ymu, "ysd": ysd,
        "alpha_grid": [alphas[0], alphas[-1], len(alphas)]}


def select_hparam(AP, sel_preds, Ysel, keys, meta, tol: float = SEL_TOL,
                  min_skill: float = MIN_SKILL):
    """Per-output (gamma, alpha) by inner-val R², with a SHRINKAGE fallback.

    ⛔ THE DEFECT THIS FIXES, MEASURED on the first pass of this very run. When a
    channel is unpredictable, every setting scores ~0 on the inner split and a
    plain ``argmax`` picks whichever tiny alpha won by numerical noise. That
    model is an unregularised fit to nothing, and on the held-out episodes it
    does NOT score ~0 — it scored **R² −4229** for ``yaw_rate`` on the shuffled
    control, because the TRAIN ``yaw_rate`` carries heading-repair outliers up to
    ±15 rad/s while the held-out episodes top out at 0.48 rad/s.

    Two rules, both decided before looking at the held-out set:
      * **skill gate** — unless the best inner-val R² clears ``min_skill``, use
        maximum shrinkage, i.e. emit the train mean. "No demonstrated skill" must
        produce the null, not a random explosion.
      * **one-SE tie-break** — among settings within ``tol`` of the best, take
        the most shrunk. Below any real effect here, so it cannot flip a verdict
        on a channel that is genuinely predictable.

    ⚠️ This is HONEST BUT LOSSY for ``yaw_rate``: its inner-val R² is corrupted
    by those same train-side outliers, so the gate sends it to the mean even
    though the held-out set would have rewarded a real model. That is why every
    arm additionally reports an ORACLE-SELECTED upper bound (selection on the
    held-out set — cheating, and labelled as such) so the loss is visible.
    """
    ymu, ysd = meta["ymu"], meta["ysd"]
    Yt = np.asarray(Ysel, dtype=np.float64)
    k_shrunk = max(keys, key=lambda k: k[1])
    chosen = []
    for j in range(Yt.shape[1]):
        scored = [(k, AP.r2_score(
            (sel_preds[k][:, j] * ysd[0, j] + ymu[0, j]).numpy(), Yt[:, j]))
            for k in keys]
        smap = dict(scored)
        best = max(smap.values())
        if best < min_skill:
            k_sel, rule = k_shrunk, "skill_gate_to_train_mean"
        else:
            ok = [(k, s) for k, s in scored if s >= best - tol]
            k_sel, rule = max(ok, key=lambda t: t[0][1])[0], "one_se_tiebreak"
        chosen.append((k_sel, float(smap[k_sel]), float(best), rule))
    return chosen


def jsonable(x):
    """``float('inf')`` (the exact-mean sentinel's alpha) is not strict JSON."""
    return "inf(exact_train_mean)" if isinstance(x, float) and np.isinf(x) else x


def assemble(preds, chosen, meta):
    """Stitch the per-output hyperparameter choices into one [N, 12] prediction."""
    ymu, ysd = meta["ymu"], meta["ysd"]
    cols = [(preds[c[0]][:, j] * ysd[0, j] + ymu[0, j]).numpy()
            for j, c in enumerate(chosen)]
    return np.stack(cols, 1)


def oracle_selected_r2(AP, ho_preds, Yho, keys, meta):
    """⚠️ CHEATING BY CONSTRUCTION — the hyperparameter is picked on the HELD-OUT
    set. Quotable ONLY as an UPPER BOUND on what this arm could have achieved.

    It exists because a null result has to be robust to the selection procedure.
    If ``long_accel`` stays at the null even when we are allowed to choose the
    best setting with full knowledge of the answer, then "the head/regulariser
    was chosen badly" is dead as an explanation.
    """
    ymu, ysd = meta["ymu"], meta["ysd"]
    Yt = np.asarray(Yho, dtype=np.float64)
    out = {}
    for j, ch in enumerate(SCALARS):
        best_k, best_s = None, -1e18
        for k in keys:
            s = AP.r2_score((ho_preds[k][:, j] * ysd[0, j] + ymu[0, j]).numpy(),
                            Yt[:, j])
            if s > best_s:
                best_k, best_s = k, s
        out[ch] = {"r2": round(float(best_s), 5),
                   "gamma_mult": best_k[0], "alpha": jsonable(best_k[1])}
    return out


# --------------------------------------------------------------------------- #
def neural_arm(AP, make, train_fit, sel, train_full, Zho, Sho, *, seeds,
               max_epochs, device, scalar_mask=None, batch=256, lr=3e-4,
               log_fn=None):
    """Two-phase neural arm: pick the epoch budget on the INNER split, then refit
    on the full train set. Ensemble = mean over seeds (the shipped recipe).

    ⛔ THE DEFECT THIS FIXES, caught in the first full run of this stream. The
    budget was selected on inner-val ``long_accel`` R² — the one channel with no
    signal — so the choice was noise and the arms came out UNDERTRAINED on
    everything else: the shipped transformer landed at ``speed`` R² ≈ −2.8 where
    the P9 panel's identical geometry reaches +0.72. That destroys the positive
    control, which is the thing that makes a null on ``long_accel`` mean anything.

    The rule now: **one uniform criterion for every arm** — the inner-val mean of
    R² clipped to [−1, 1] over all four channels. Clipping keeps a blown-up
    channel (``yaw_rate``, see the label defect) from deciding the budget;
    applying it identically to latent and oracle-input arms keeps it from being a
    per-arm choice.

    The selection can still be argued with, so the arm ALSO reports
    ``ORACLE_over_epoch_grid_heldout_long_accel_r2``: the best held-out
    ``long_accel`` over the whole budget grid. That number is selected on the
    test set and is quotable only as an UPPER BOUND — but it removes the epoch
    budget from the list of things a null could be blamed on.
    """
    Zf, Sf, Tf = train_fit
    Zs, Ss, Ts = sel
    ZF, SF, TF = train_full
    grid = sorted({max(2, max_epochs // 8), max(2, max_epochs // 4),
                   max(2, max_epochs // 2), max_epochs})
    best_e, best_s, per_budget = grid[0], -1e9, {}
    gy = Sho[:, 3].numpy().astype(np.float64)
    for e in grid:
        mod, _ = AP.fit_probe_head(make, (Zf, Sf, Tf), epochs=e, batch=batch,
                                   lr=lr, seed=0, device=device,
                                   scalar_mask=scalar_mask)
        ps, _ = AP.predict_head(mod, Zs, device=device)
        r2s = [AP.r2_score(ps[:, j].numpy(), Ss[:, j].numpy()) for j in range(4)]
        s = float(np.mean(np.clip(r2s, -1.0, 1.0)))
        hs, _ = AP.predict_head(mod, Zho, device=device)
        per_budget[e] = {"inner_r2_per_scalar": [round(x, 4) for x in r2s],
                         "inner_selection_score": round(s, 4),
                         "heldout_long_accel_r2":
                             round(AP.r2_score(hs[:, 3].numpy(), gy), 4)}
        if log_fn:
            log_fn(f"      epochs={e} inner score {s:+.4f} "
                   f"heldout accel {per_budget[e]['heldout_long_accel_r2']:+.4f}")
        if s > best_s:
            best_e, best_s = e, s
        del mod
    ho_s, ho_t, tr_s, tr_t = [], [], [], []
    n_par = 0
    for sd in range(seeds):
        mod, _ = AP.fit_probe_head(make, (ZF, SF, TF), epochs=best_e, batch=batch,
                                   lr=lr, seed=sd, device=device,
                                   scalar_mask=scalar_mask)
        hs, ht = AP.predict_head(mod, Zho, device=device)
        ts, tt = AP.predict_head(mod, ZF, device=device)
        ho_s.append(hs.numpy()); ho_t.append(ht.numpy())
        tr_s.append(ts.numpy()); tr_t.append(tt.numpy())
        n_par = sum(p.numel() for p in mod.parameters())
        del mod
    ho = np.concatenate([np.mean(ho_s, 0),
                         np.mean(ho_t, 0).reshape(len(Zho), -1)], 1)
    tr = np.concatenate([np.mean(tr_s, 0),
                         np.mean(tr_t, 0).reshape(len(ZF), -1)], 1)
    return ho, tr, ho_s, {
        "epochs_selected": int(best_e), "epoch_grid": grid,
        "inner_selection_score": round(float(best_s), 4),
        "selection_criterion": ("inner-val mean of R² clipped to [-1,1] over all "
                                "four scalars — one uniform rule for every arm"),
        "per_epoch_budget": per_budget,
        "ORACLE_over_epoch_grid_heldout_long_accel_r2":
            max(v["heldout_long_accel_r2"] for v in per_budget.values()),
        "params": int(n_par), "seeds": int(seeds)}


# --------------------------------------------------------------------------- #
# TACTICAL readout OF THE CHANNEL UNDER TEST                                   #
# --------------------------------------------------------------------------- #
ACCEL_CLASSES = ("decelerate", "cruise", "accelerate")
#: LON_DV_MPS = 1.0 m/s over the 2 s horizon is the family's longitudinal
#: threshold; the equivalent sustained acceleration is 0.5 m/s^2.
ACCEL_THRESH = 0.5


def accel_classes(a):
    a = np.asarray(a, dtype=np.float64)
    c = np.ones(len(a), dtype=np.int64)
    c[a > ACCEL_THRESH] = 2
    c[a < -ACCEL_THRESH] = 0
    return c


def accel_tactical(FF, pred_a, gt_a):
    """Confusion + per-class RECALL **and PRECISION** for the longitudinal
    decision the ``long_accel`` channel would drive.

    ⚠️ Precision is reported beside recall on purpose: a recall-only frontier
    cannot see what it is paying, and that is exactly how this programme's
    retracted "brake_stop 0.026 -> 0.503 free win" got published.
    """
    P, G = accel_classes(pred_a), accel_classes(gt_a)
    C = FF.confusion(P, G, 3)
    sup = C.sum(1)                     # per GT class
    fired = C.sum(0)                   # per predicted class
    return {
        "classes": list(ACCEL_CLASSES),
        "threshold_mps2": ACCEL_THRESH,
        "confusion_gt_rows_pred_cols": C.tolist(),
        "support_gt": sup.tolist(), "n_fired_pred": fired.tolist(),
        "recall": [round(float(C[i, i] / sup[i]), 4) if sup[i] else None
                   for i in range(3)],
        "precision": [round(float(C[i, i] / fired[i]), 4) if fired[i] else None
                      for i in range(3)],
        "balanced_accuracy": round(FF.balanced_accuracy(C), 4),
        "accuracy": round(float(np.trace(C) / max(C.sum(), 1)), 4),
        "chance_balanced_accuracy": round(1.0 / max(int((sup > 0).sum()), 1), 4),
    }


# --------------------------------------------------------------------------- #
# per-frame latent reconstruction, for the CONTEXT-LENGTH arm                  #
# --------------------------------------------------------------------------- #
def reconstruct_z(ep, k: int = K, stride: int = STRIDE):
    """Cached [N, 2k+1, D] windows -> the per-frame latent track z [M, D].

    The cache stores overlapping windows at stride 2, so the per-frame track is
    recoverable and the overlap is a free CONSISTENCY CHECK: every frame is
    written by several windows and they must agree bit-for-bit. Asserted, because
    a silent mis-indexing here would fabricate a context-length effect.
    """
    Z = ep["Z"]
    n, w, d = Z.shape
    t0 = max(k, 1)
    hi = t0 + stride * (n - 1) + k                       # last frame covered
    z = torch.full((hi + 1, d), float("nan"), dtype=Z.dtype)
    for i in range(n):
        t = t0 + stride * i
        blk = Z[i]
        sl = z[t - k:t + k + 1]
        seen = ~torch.isnan(sl[:, 0])
        if seen.any():
            assert torch.equal(sl[seen], blk[seen]), \
                f"{ep['name']}: overlapping windows disagree at centre {t}"
        z[t - k:t + k + 1] = blk
    assert not torch.isnan(z[t0 - k:]).any(), "gap in the reconstructed track"
    return z, t0, hi


def wide_windows(ep, k_wide: int):
    """-> (Zwide [M, 2k_wide+1, D], row_index into the cached window order).

    The wider window needs more frames on each side, so it is defined on a SUBSET
    of the cached centres. Returning the row index lets every k=4 arm be scored on
    exactly those rows, which is what makes the context contrast PAIRED.
    """
    z, t0, hi = reconstruct_z(ep)
    rows, blocks = [], []
    for i in range(ep["Z"].shape[0]):
        t = t0 + STRIDE * i
        if t - k_wide < t0 - K or t + k_wide > hi:
            continue
        rows.append(i)
        blocks.append(z[t - k_wide:t + k_wide + 1])
    if not rows:
        return None, None
    return torch.stack(blocks), np.asarray(rows, dtype=np.int64)


# --------------------------------------------------------------------------- #
def score_arm(AP, TCI, APCI, FF, name, pred, Yho, Ytr_pred, Ytr, eid, n_boot):
    """R² per scalar with its CI + in-sample R² + ADE + the four families."""
    gs, gt = unpack(Yho)
    ps, pt = pred[:, :4], pred[:, 4:].reshape(len(pred), len(HORIZONS), 2)
    tgs, _ = unpack(Ytr)
    tps = Ytr_pred[:, :4]

    def r2_fn(p, g):
        return lambda sel: AP.r2_score(p[sel], g[sel])

    out = {"r2": {}, "r2_train_insample": {}, "mae": {}}
    for j, ch in enumerate(SCALARS):
        out["r2"][ch] = APCI.stat_episode_cluster_bootstrap(
            r2_fn(ps[:, j], gs[:, j]), eid, n_boot=n_boot, name=f"r2_{ch}")
        out["r2_train_insample"][ch] = round(AP.r2_score(tps[:, j], tgs[:, j]), 4)
        out["mae"][ch] = round(float(np.abs(ps[:, j] - gs[:, j]).mean()), 5)
    per_win = np.linalg.norm(pt - gt, axis=-1).mean(1)
    out["ade_2s"] = TCI.episode_cluster_bootstrap(per_win, eid, n_boot=n_boot)
    out["four_families"] = FF.all_families(pt, gt, FF.IDM_DT_S,
                                           pred_scalars=ps, gt_scalars=gs)
    # TACTICAL, twice: from the trajectory (the family instrument) and from the
    # long_accel SCALAR (the channel this stream is about). The second is the one
    # that can see a decision error the scalar R² averages away.
    out["four_families"]["TACTICAL"]["from_long_accel_scalar"] = \
        accel_tactical(FF, ps[:, 3], gs[:, 3])

    def ba_traj(axis, k):
        Pm = FF.manoeuvre_classes(pt, FF.IDM_DT_S)[axis]
        Gm = FF.manoeuvre_classes(gt, FF.IDM_DT_S)[axis]
        return lambda sel: FF.balanced_accuracy(FF.confusion(Pm[sel], Gm[sel], k))

    Pa, Ga = accel_classes(ps[:, 3]), accel_classes(gs[:, 3])
    out["tactical_ba_ci"] = {
        ax: APCI.stat_episode_cluster_bootstrap(
            ba_traj(ax, k), eid, n_boot=n_boot, name=f"balanced_accuracy_{ax}")
        for ax, k in (("lateral", 3), ("longitudinal", 3), ("mixed", 5))}
    out["tactical_ba_ci"]["from_long_accel_scalar"] = \
        APCI.stat_episode_cluster_bootstrap(
            lambda sel: FF.balanced_accuracy(FF.confusion(Pa[sel], Ga[sel], 3)),
            eid, n_boot=n_boot, name="balanced_accuracy_accel_scalar")
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    from tanitad.eval import accel_probe as AP
    from tanitad.eval import idm_families as FF
    from tanitad.eval import ap_ci as APCI
    import taniteval.ci as TCI
    import idm_head as ih
    assert str(REPO) in str(Path(TCI.__file__).resolve()), TCI.__file__

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results_accel_recoverability.json"))
    ap.add_argument("--stages", default="all")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stages = ({"ridge", "neural", "sensitivity", "oracle", "context"}
              if a.stages == "all" else set(a.stages.split(",")))
    log(f"device={dev} stages={sorted(stages)} estimator={Path(TCI.__file__).name}")

    sub = load_substrate()
    eps = sub["episodes"]
    D = int(sub["state_dim"])
    tr_eps, ho_eps = split_episodes(eps)
    # inner selection split, WITHIN train only
    fit_eps = [e for i, e in enumerate(tr_eps) if i % 3 != 0]
    sel_eps = [e for i, e in enumerate(tr_eps) if i % 3 == 0]
    Ztr, Ytr, _ = stack(tr_eps)
    Zho, Yho, eid = stack(ho_eps)
    Zf, Yf, _ = stack(fit_eps)
    Zs, Ys, _ = stack(sel_eps)
    log(f"episodes  train {len(tr_eps)} (inner-fit {len(fit_eps)} / inner-sel "
        f"{len(sel_eps)})  heldout {len(ho_eps)}")
    log(f"windows   train {len(Ztr):,}  heldout {len(Zho):,}  "
        f"clusters {len(np.unique(eid))}")

    res = {
        "_what": ("STREAM D — capacity/architecture sweep answering whether "
                  "long_accel is UNRECOVERABLE from the frozen v1 latents or "
                  "merely UNRECOVERED by the head tried"),
        "substrate": {"latent_cache": str(LATENTS), "encoder": sub["encoder"],
                      "encoder_step": sub["encoder_step"], "state_dim": D,
                      "k": K, "stride": STRIDE, "horizons": list(HORIZONS),
                      "verification": "raw/substrate_verification.json"},
        "split": {"unit": "EPISODE-DISJOINT (identical to the P9 panel)",
                  "train_episodes": [e["name"] for e in tr_eps],
                  "heldout_episodes": [e["name"] for e in ho_eps],
                  "inner_select_episodes": [e["name"] for e in sel_eps],
                  "n_train_windows": int(len(Ztr)),
                  "n_heldout_windows": int(len(Zho))},
        "estimator": ("episode-cluster bootstrap; PAIRED for every arm-vs-control "
                      "contrast (taniteval/ci.py + tanitad.eval.ap_ci)"),
        "n_boot": a.n_boot, "seeds": a.seeds,
        "arms": {}, "paired_vs_control": {}, "stages_run": sorted(stages),
    }
    outp = Path(a.out)

    def bank():
        outp.write_text(json.dumps(res, indent=1, default=str))

    preds: dict[str, np.ndarray] = {}
    trpreds: dict[str, np.ndarray] = {}
    meta_by_arm: dict[str, dict] = {}

    # ------------------------------------------------------------------ #
    # STAGE 1 — closed-form capacity ladder                              #
    # ------------------------------------------------------------------ #
    # THE EMPIRICAL NULL. R²=0 is NOT the null here: the held-out episodes have a
    # different long_accel mean from train (−0.1305 vs +0.0163 m/s²), so a
    # predictor that emits the TRAIN MEAN — the best a model with no information
    # can do — already scores a NEGATIVE R². Quoting "negative R² = worse than
    # the mean" without this row is the same error class as the retracted
    # "empirical null is 1/3" (it was 0.3678).
    preds["NULL_train_mean"] = np.tile(Ytr.mean(0, keepdim=True).numpy(),
                                       (len(Yho), 1))
    trpreds["NULL_train_mean"] = np.tile(Ytr.mean(0, keepdim=True).numpy(),
                                         (len(Ytr), 1))
    meta_by_arm["NULL_train_mean"] = {
        "kind": "empirical_null",
        "_what": ("constant = TRAIN mean of every output. The floor any arm "
                  "must beat; NOT R²=0."),
        "shuffled_latents": False}

    if "ridge" in stages:
        ridge_specs = [
            ("RIDGE_linear_centre", dict(feat="centre", kernel="linear")),
            ("RIDGE_linear_window", dict(feat="window", kernel="linear")),
            ("RIDGE_linear_diff", dict(feat="diff", kernel="linear")),
            ("RIDGE_linear_centre_diff", dict(feat="centre_diff", kernel="linear")),
            ("RIDGE_rbf_centre", dict(feat="centre", kernel="rbf",
                                      gamma_mults=(0.25, 1.0, 4.0))),
            ("RIDGE_rbf_window", dict(feat="window", kernel="rbf",
                                      gamma_mults=(0.25, 1.0, 4.0))),
        ]
        for name, spec in ridge_specs:
            for tag, shuf in (("", None), ("__CTRL", 101)):
                Zf_, Yf_, _ = (Zf, Yf, None) if shuf is None else stack(fit_eps, shuf)
                ZT_, YT_, _ = (Ztr, Ytr, None) if shuf is None else stack(tr_eps, shuf)
                sp, hp, tp, keys, meta = ridge_arm(
                    AP, Zf_, Yf_, Zs, ZT_, YT_, Zho, device=dev, **spec)
                chosen = select_hparam(AP, sp, Ys, keys, meta)
                preds[name + tag] = assemble(hp, chosen, meta)
                trpreds[name + tag] = assemble(tp, chosen, meta)
                meta_by_arm[name + tag] = {
                    "kind": "closed_form_kernel_ridge",
                    "feature": meta["feature"], "kernel": meta["kernel"],
                    "n_features": meta["n_features"],
                    "gamma_mults": list(meta["gammas"]),
                    "selected_per_output": {
                        SCALARS[j]: {"gamma_mult": chosen[j][0][0],
                                     "alpha": jsonable(chosen[j][0][1]),
                                     "rule": chosen[j][3]} for j in range(4)},
                    "inner_val_r2_per_scalar":
                        {SCALARS[j]: round(chosen[j][1], 4) for j in range(4)},
                    "inner_val_r2_best_over_grid":
                        {SCALARS[j]: round(chosen[j][2], 4) for j in range(4)},
                    "ORACLE_SELECTED_heldout_r2_UPPER_BOUND_cheating":
                        oracle_selected_r2(AP, hp, Yho, keys, meta),
                    "shuffled_latents": shuf is not None,
                    "selection_rule": (
                        f"skill gate at inner-val R² {MIN_SKILL} (else train "
                        f"mean), then largest alpha within {SEL_TOL} of the best"),
                    "alpha_grid": meta["alpha_grid"]}
                log(f"  {name+tag}: inner "
                    + " ".join(f"{SCALARS[j]}={chosen[j][1]:+.3f}" for j in range(4)))
                del sp, hp, tp
        bank()

    # ------------------------------------------------------------------ #
    # STAGE 2 — neural ladder (architecture x capacity)                  #
    # ------------------------------------------------------------------ #
    if "neural" in stages:
        def mk_idm(dm, dep, nh):
            return lambda: ih.IDMHead(state_dim=D, d_model=dm, depth=dep,
                                      n_heads=nh, window=2 * K + 1,
                                      horizons=HORIZONS)
        neural_specs = [
            ("NN_transformer_d64_L1", mk_idm(64, 1, 2), None),
            ("NN_transformer_d256_L3_SHIPPED", mk_idm(256, 3, 4), None),
            ("NN_transformer_d512_L6", mk_idm(512, 6, 8), None),
            ("NN_mlp_centre_h512", lambda: AP.MLPHead(D, 512, 2, mode="centre"), None),
            ("NN_mlp_window_h1024", lambda: AP.MLPHead(D, 1024, 2, mode="window"), None),
            ("NN_gru_d128_L2", lambda: AP.GRUHead(D, 128, 2), None),
            # SINGLE-TASK: only long_accel in the scalar loss. Isolates
            # multitask interference from representation content.
            ("NN_transformer_d256_L3_ACCELONLY", mk_idm(256, 3, 4),
             [0.0, 0.0, 0.0, 1.0]),
        ]
        Sf, Tf = Yf[:, :4], Yf[:, 4:].reshape(len(Yf), len(HORIZONS), 2)
        Ss, Ts = Ys[:, :4], Ys[:, 4:].reshape(len(Ys), len(HORIZONS), 2)
        ST, TT = Ytr[:, :4], Ytr[:, 4:].reshape(len(Ytr), len(HORIZONS), 2)
        for name, make, mask in neural_specs:
            for tag, shuf in (("", None), ("__CTRL", 101)):
                Zf_ = Zf if shuf is None else stack(fit_eps, shuf)[0]
                ZT_ = Ztr if shuf is None else stack(tr_eps, shuf)[0]
                t = time.time()
                ho, tr, ho_seeds, m = neural_arm(
                    AP, make, (Zf_, Sf, Tf), (Zs, Ss, Ts), (ZT_, ST, TT), Zho,
                    Yho[:, :4], seeds=a.seeds, max_epochs=a.max_epochs,
                    device=dev, scalar_mask=mask)
                preds[name + tag] = ho
                trpreds[name + tag] = tr
                m.update({"kind": "neural", "shuffled_latents": shuf is not None,
                          "scalar_mask": mask,
                          "per_seed_heldout_long_accel_r2":
                              [round(AP.r2_score(s[:, 3], Yho[:, 3].numpy()), 4)
                               for s in ho_seeds],
                          "fit_seconds": round(time.time() - t, 1)})
                meta_by_arm[name + tag] = m
                log(f"  {name+tag}: {m['params']:,} params  epochs "
                    f"{m['epochs_selected']}  inner score "
                    f"{m['inner_selection_score']:+.4f}  best-budget heldout "
                    f"accel {m['ORACLE_over_epoch_grid_heldout_long_accel_r2']:+.4f}"
                    f"  ({m['fit_seconds']}s)")
        bank()

    # ------------------------------------------------------------------ #
    # STAGE 3 — DETECTION SENSITIVITY: how strong must the signal BE?    #
    # ------------------------------------------------------------------ #
    if "sensitivity" in stages:
        y_ho = Yho[:, 3].numpy().astype(np.float64)
        mu, sd = float(Ytr[:, 3].mean()), float(Ytr[:, 3].std())
        rms = float(Ztr.double().pow(2).mean().sqrt())

        def ystd(Y):
            return (Y[:, 3].numpy().astype(np.float64) - mu) / sd

        # the latent's leading principal direction, for the easy end of the bracket
        Xc = AP.window_features(Ztr, "centre").double()
        Xc = Xc - Xc.mean(0, keepdim=True)
        PC1 = torch.linalg.svd(Xc[torch.randperm(len(Xc))[:2000]],
                               full_matrices=False)[2][0]
        del Xc

        def planted(Z, Y, frac, rho, seed, direction=None):
            """Plant ``rho*y + sqrt(1-rho^2)*noise`` at amplitude ``frac``.

            ``rho=1`` is the amplitude sweep (how FAINT can a clean direction be
            and still be found). ``rho<1`` is the far more useful sweep: the best
            R² obtainable from that direction alone is rho², so it answers "what
            is the smallest TRUE R² this protocol would have called separated?"
            — which is the number a null result actually needs.
            """
            if frac <= 0:
                return Z
            v = ystd(Y)
            if rho < 1.0:
                g = np.random.default_rng(seed * 1000 + int(rho * 1e6))
                e = g.standard_normal(len(v))
                v = rho * v + (1.0 - rho ** 2) ** 0.5 * e
            return AP.inject_signal(Z, v, frac, seed=seed, standardize_y=False,
                                    rms=rms, direction=direction)[0]

        def probe(frac, rho, shuffle, direction=None):
            """⛔ THE CONTROL BUG THIS FIXES, caught on the first sensitivity pass.
            Shuffling the latents and THEN injecting re-establishes the very link
            the shuffle was meant to destroy — the "control" learned the planted
            direction as well as the arm (and sometimes better, having less native
            interference), which made ΔR² read NEGATIVE at a strong planted
            signal. Inject first, shuffle second.
            """
            Zf_i = planted(Zf, Yf, frac, rho, 3, direction)
            ZT_i = planted(Ztr, Ytr, frac, rho, 3, direction)
            Zs_i = planted(Zs, Ys, frac, rho, 3, direction)
            Zh_i = planted(Zho, Yho, frac, rho, 3, direction)
            if shuffle is not None:
                for name_, Zx in (("f", Zf_i), ("T", ZT_i)):
                    g = np.random.default_rng(shuffle)
                    p = torch.from_numpy(g.permutation(len(Zx)))
                    if name_ == "f":
                        Zf_i = Zx[p]
                    else:
                        ZT_i = Zx[p]
            sp, hp, tp, keys, meta = ridge_arm(
                AP, Zf_i, Yf, Zs_i, ZT_i, Ytr, Zh_i,
                feat="centre", kernel="linear", device=dev)
            chosen = select_hparam(AP, sp, Ys, keys, meta)
            P = assemble(hp, chosen, meta)
            orc = oracle_selected_r2(AP, hp, Yho, keys, meta)["long_accel"]
            del sp, hp, tp
            return P, chosen, orc

        sens = {"_what": ("the SAME probe on latents carrying a KNOWN planted "
                          "copy of long_accel. Converts an unfalsifiable "
                          "'nothing is there' into a floor with a number."),
                "latent_rms": round(rms, 5),
                "injection": ("centre frame only, one random unit direction, "
                              "seed 3; shared gain across train/select/heldout"),
                "probe": "linear ridge on the centre token, same selection rule",
                "caveat": ("a clean rank-1 LINEAR direction is the EASIEST "
                           "possible encoding. These floors bound detectability "
                           "in the easy case; a distributed or nonlinear "
                           "encoding of the same information could sit below "
                           "them and the rbf/neural arms are what cover that."),
                "amplitude_sweep_rho1": []}

        for fr in [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 1e-2]:
            P, chosen, orc = probe(fr, 1.0, None)
            r = APCI.stat_episode_cluster_bootstrap(
                (lambda s, PP=P: AP.r2_score(PP[s, 3], y_ho[s])), eid,
                n_boot=a.n_boot, name="r2_long_accel")
            sens["amplitude_sweep_rho1"].append({
                "frac_of_latent_rms": fr,
                "inner_val_r2": round(chosen[3][2], 4),
                "heldout_r2_long_accel": r, "selection_rule": chosen[3][3],
                "ORACLE_SELECTED_heldout_r2_cheating": orc})
            log(f"  SENS amp frac={fr:g}: inner {chosen[3][2]:+.4f} "
                f"heldout {r['point']:+.4f} [{r['lo']:+.4f},{r['hi']:+.4f}]")

        # SNR along the carrier direction, so the bracket is interpretable: the
        # same `frac` is a very different signal-to-noise ratio on a random
        # direction (native projection variance ~ the mean per-dim variance) than
        # on the leading principal direction (native variance is maximal there).
        Xc2 = AP.window_features(Ztr, "centre").double()
        Xc2 = Xc2 - Xc2.mean(0, keepdim=True)
        gd = torch.Generator().manual_seed(3)
        urnd = torch.randn(2048, generator=gd, dtype=torch.float64)
        urnd = urnd / urnd.norm()
        sens["native_projection_var"] = {
            "random_dir": round(float((Xc2 @ urnd).var()), 6),
            "pc1_dir": round(float((Xc2 @ PC1.double()).var()), 6),
            "note": ("injected projection variance at frac f is "
                     "(f*rms*sqrt(D))^2 * var(planted content); divide by these "
                     "to read the SNR the probe actually faces.")}
        del Xc2

        for key, direction, FRAC_FIXED, rlist in (
                ("correlation_sweep_random_dir", None, 0.05,
                 [0.0, 0.01, 0.03, 0.1, 0.3, 0.6]),
                ("correlation_sweep_random_dir_strong", None, 0.5,
                 [0.0, 0.01, 0.03, 0.1, 0.3]),
                ("correlation_sweep_pc1_dir", PC1, 0.05, [0.1, 0.6]),
                ("correlation_sweep_pc1_dir_strong", PC1, 2.0, [0.1, 0.3])):
            sens.setdefault(key, [])
            for rho2 in rlist:
                rho = rho2 ** 0.5
                P, chosen, orc = probe(FRAC_FIXED, rho, None, direction)
                Pc, _c, _o = probe(FRAC_FIXED, rho, 101, direction)
                r = APCI.stat_episode_cluster_bootstrap(
                    (lambda s, PP=P: AP.r2_score(PP[s, 3], y_ho[s])), eid,
                    n_boot=a.n_boot, name="r2_long_accel")
                dr = APCI.paired_stat_episode_cluster_bootstrap(
                    (lambda s, PP=P: AP.r2_score(PP[s, 3], y_ho[s])),
                    (lambda s, PP=Pc: AP.r2_score(PP[s, 3], y_ho[s])), eid,
                    n_boot=a.n_boot, name="d_r2_long_accel")
                if float(np.abs(P[:, 3] - Pc[:, 3]).max()) <= 1e-10 * y_ho.std():
                    dr["separated"], dr["degenerate"] = False, True
                sens[key].append({
                    "planted_true_r2_rho2": rho2, "rho": round(rho, 5),
                    "amplitude_frac_of_latent_rms": FRAC_FIXED,
                    "inner_val_r2": round(chosen[3][2], 4),
                    "heldout_r2_long_accel": r,
                    "ORACLE_SELECTED_heldout_r2_cheating": orc,
                    "paired_vs_shuffled_control": dr})
                log(f"  SENS {key[18:]} f={FRAC_FIXED} rho2={rho2:g}: heldout "
                    f"{r['point']:+.4f} oracle-sel {orc['r2']:+.4f} "
                    f"dCTRL {dr['delta']:+.4f}"
                    f"{'*' if dr.get('separated') else ''}")
        res["detection_sensitivity"] = sens
        bank()

    # ------------------------------------------------------------------ #
    # STAGE 4 — CAPACITY CONTROL: same heads, ORACLE input (true speed)  #
    # ------------------------------------------------------------------ #
    if "oracle" in stages:
        def qstack(es):
            return torch.cat([e["Q"] for e in es]).float().unsqueeze(-1)
        Qf, Qs_, QT, Qho = (qstack(fit_eps), qstack(sel_eps), qstack(tr_eps),
                            qstack(ho_eps))
        Sf, Tf = Yf[:, :4], Yf[:, 4:].reshape(len(Yf), len(HORIZONS), 2)
        Ss, Ts = Ys[:, :4], Ys[:, 4:].reshape(len(Ys), len(HORIZONS), 2)
        ST, TT = Ytr[:, :4], Ytr[:, 4:].reshape(len(Ytr), len(HORIZONS), 2)
        oracle_specs = [
            ("ORACLEIN_transformer_d256_L3",
             lambda: ih.IDMHead(state_dim=1, d_model=256, depth=3, n_heads=4,
                                window=2 * K + 1, horizons=HORIZONS)),
            ("ORACLEIN_mlp_window_h512",
             lambda: AP.MLPHead(1, 512, 2, mode="window")),
            # BLIND BY CONSTRUCTION: v(t) alone cannot contain its own derivative.
            # If this one scored, the pipeline would be leaking.
            ("ORACLEIN_mlp_centreONLY_blind",
             lambda: AP.MLPHead(1, 512, 2, mode="centre")),
        ]
        for name, make in oracle_specs:
            t = time.time()
            ho, tr, ho_seeds, m = neural_arm(
                AP, make, (Qf, Sf, Tf), (Qs_, Ss, Ts), (QT, ST, TT), Qho,
                Yho[:, :4], seeds=a.seeds, max_epochs=a.max_epochs, device=dev)
            preds[name] = ho
            trpreds[name] = tr
            m.update({"kind": "capacity_control_oracle_input",
                      "_what": ("the SAME architecture fed the TRUE speed window "
                                "instead of the latent. If it reaches the oracle "
                                "ceiling, head capacity is NOT the limit."),
                      "input": "true CAN speed at the 9 window positions [N,9,1]",
                      "shuffled_latents": False,
                      "per_seed_heldout_long_accel_r2":
                          [round(AP.r2_score(s[:, 3], Yho[:, 3].numpy()), 4)
                           for s in ho_seeds],
                      "fit_seconds": round(time.time() - t, 1)})
            meta_by_arm[name] = m
            log(f"  {name}: {m['params']:,} params inner score "
                f"{m['inner_selection_score']:+.4f} best-budget heldout accel "
                f"{m['ORACLE_over_epoch_grid_heldout_long_accel_r2']:+.4f} "
                f"({m['fit_seconds']}s)")
        bank()

    # ------------------------------------------------------------------ #
    # STAGE 5 — CONTEXT LENGTH: 2.5 s window instead of 0.8 s            #
    # ------------------------------------------------------------------ #
    if "context" in stages:
        KW = 12

        def ctxstack(es):
            """-> (Zwide, Znarrow ON THE SAME ROWS, Y, eid).

            Matching the ROWS as well as the episodes is what makes this a clean
            context-width contrast: both arms see the same centres and the same
            amount of data, and only the window width differs.
            """
            zw, zn, ys, eids = [], [], [], []
            for e in es:
                Zwi, rows = wide_windows(e, KW)
                if Zwi is None:
                    continue
                zw.append(Zwi.float())
                zn.append(e["Z"][rows].float())
                ys.append(torch.cat([e["S"][rows],
                                     e["T"][rows].reshape(len(rows), -1)], 1))
                eids.append(np.full(len(rows), e["name"]))
            return (torch.cat(zw), torch.cat(zn), torch.cat(ys),
                    np.concatenate(eids))
        ZfW, ZfN, YfW, _ = ctxstack(fit_eps)
        ZsW, ZsN, YsW, _ = ctxstack(sel_eps)
        ZTW, ZTN, YTW, _ = ctxstack(tr_eps)
        ZhoW, ZhoN, YhoW, eidW = ctxstack(ho_eps)
        log(f"  context k={KW}: heldout windows {len(ZhoW):,} "
            f"(vs {len(Zho):,} at k={K})")
        ctx = {"_what": ("a 2.5 s window (k=12) against 0.8 s (k=4) on the SAME "
                         "rows. The wide window is built from the per-frame "
                         "latent track reconstructed out of the cached "
                         "overlapping windows; the overlap is asserted "
                         "bit-identical, so a context effect cannot be a "
                         "mis-indexing artefact."),
               "k_wide": KW, "n_heldout_windows": int(len(ZhoW)),
               "n_heldout_windows_k4_full": int(len(Zho)), "arms": {}}
        pred_by_tag = {}
        for tag, name, Zfi, ZTi, Zsi, Zhoi in (
                ("wide", f"CTX_k{KW}_ridge_linear_window", ZfW, ZTW, ZsW, ZhoW),
                ("narrow", f"CTX_k{K}_ridge_linear_window_SAMEROWS", ZfN, ZTN,
                 ZsN, ZhoN)):
            sp, hp, tp, keys, meta = ridge_arm(
                AP, Zfi, YfW, Zsi, ZTi, YTW, Zhoi,
                feat="window", kernel="linear", device=dev)
            chosen = select_hparam(AP, sp, YsW, keys, meta)
            P = assemble(hp, chosen, meta)
            ctx["arms"][name] = {
                "n_features": meta["n_features"],
                "inner_val_r2_long_accel": round(chosen[3][2], 4),
                "ORACLE_SELECTED_heldout_r2_UPPER_BOUND_cheating":
                    oracle_selected_r2(AP, hp, YhoW, keys, meta),
                "heldout_r2": {SCALARS[j]: APCI.stat_episode_cluster_bootstrap(
                    (lambda s, j=j, PP=P: AP.r2_score(
                        PP[s, j], YhoW[:, j].numpy().astype(np.float64)[s])),
                    eidW, n_boot=a.n_boot, name=f"r2_{SCALARS[j]}")
                    for j in range(4)}}
            pred_by_tag[tag] = P
            log(f"  CONTEXT {name}: long_accel "
                f"{ctx['arms'][name]['heldout_r2']['long_accel']['point']:+.4f}")
            del sp, hp, tp
        Pw, Pn = pred_by_tag["wide"], pred_by_tag["narrow"]
        ctx["paired_wide_minus_narrow"] = {
            f"r2_{SCALARS[j]}": APCI.paired_stat_episode_cluster_bootstrap(
                (lambda s, j=j: AP.r2_score(
                    Pw[s, j], YhoW[:, j].numpy().astype(np.float64)[s])),
                (lambda s, j=j: AP.r2_score(
                    Pn[s, j], YhoW[:, j].numpy().astype(np.float64)[s])),
                eidW, n_boot=a.n_boot, name=f"d_r2_{SCALARS[j]}")
            for j in range(4)}
        res["context_length"] = ctx
        bank()

    # ------------------------------------------------------------------ #
    # score every arm + paired contrast against its own control          #
    # ------------------------------------------------------------------ #
    for name in sorted(preds):
        res["arms"][name] = {**meta_by_arm[name],
                             **score_arm(AP, TCI, APCI, FF, name, preds[name],
                                         Yho, trpreds[name], Ytr, eid, a.n_boot)}
        log(f"  SCORED {name}: "
            + " ".join(f"{c}={res['arms'][name]['r2'][c]['point']:+.4f}"
                       for c in SCALARS))
    bank()

    gs, gt = unpack(Yho)

    def paired_pair(a_name, b_name, tag):
        out = {}
        for j, ch in enumerate(SCALARS):
            r = APCI.paired_stat_episode_cluster_bootstrap(
                lambda s, j=j, k=a_name: AP.r2_score(preds[k][s, j], gs[s, j]),
                lambda s, j=j, k=b_name: AP.r2_score(preds[k][s, j], gs[s, j]),
                eid, n_boot=a.n_boot, name=f"d_r2_{ch}")
            # ⛔ DEGENERACY GUARD. When both arms fall back to the train mean —
            # which is exactly what the skill gate makes an uninformative channel
            # do — the two prediction columns are IDENTICAL and the bootstrap
            # returns "0.0 [0.0, 0.0] separated=true". That verdict is
            # arithmetic, not evidence, and an automated reader would act on it.
            spread = float(np.abs(preds[a_name][:, j] - preds[b_name][:, j]).max())
            scale = float(gs[:, j].std()) + 1e-12
            if spread <= 1e-10 * scale:
                r["separated"] = False
                r["degenerate"] = True
                r["degenerate_note"] = (
                    "the two arms emit IDENTICAL predictions on this channel "
                    "(both fell back to the train mean). No contrast exists; "
                    "`separated` is forced False and must not be quoted.")
            out[f"r2_{ch}"] = r
        pa = preds[a_name][:, 4:].reshape(-1, len(HORIZONS), 2)
        pb = preds[b_name][:, 4:].reshape(-1, len(HORIZONS), 2)
        out["ade_2s"] = TCI.paired_episode_cluster_bootstrap(
            np.linalg.norm(pa - gt, axis=-1).mean(1),
            np.linalg.norm(pb - gt, axis=-1).mean(1), eid, n_boot=a.n_boot)
        res["paired_vs_control"][tag] = out
        log(f"  PAIRED {tag}: " + " ".join(
            f"{c}={out['r2_'+c]['delta']:+.4f}"
            f"{'*' if out['r2_'+c]['separated'] else ''}" for c in SCALARS))

    for name in sorted(preds):
        if name.endswith("__CTRL"):
            continue
        if name + "__CTRL" in preds:
            # PRIMARY: matched capacity, matched recipe, latent<->target link cut.
            paired_pair(name, name + "__CTRL", f"{name}_minus_CTRL")
        if name != "NULL_train_mean":
            # SECONDARY: the interpretable floor. An arm that does not beat the
            # train mean has not used the latent at all.
            paired_pair(name, "NULL_train_mean", f"{name}_minus_NULLMEAN")
    bank()

    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
