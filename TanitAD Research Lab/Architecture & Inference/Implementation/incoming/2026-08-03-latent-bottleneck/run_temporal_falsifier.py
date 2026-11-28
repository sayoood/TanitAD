"""D-LATENT — THE FALSIFIER: is the LATENT the bottleneck for ``long_accel``,
or is the VIDEO?

Pre-registration: ``Project Steering/PREREG_TEMPORAL_LATENT.md`` (both outcomes
and thresholds fixed BEFORE this ran). Read it before quoting anything here.

THE QUESTION
    The programme's latent-bottleneck thesis says ``long_accel`` is missing from
    the model's decisions because the REPRESENTATION does not carry it. Every
    arm that has ever tested it probed the SAME frozen v1 latent, so the null is
    compatible with two opposite worlds:
      L  the information is in the video and the v1 encoder destroys it
         -> the lever is the representation;
      V  the information is not in monocular 10 Hz 256 px video at this n
         -> no encoder fixes it, and the thesis is FALSE for this channel.

WHY A PIXEL SUBSTRATE IS THE RIGHT FALSIFIER
    A representation that is temporal BY CONSTRUCTION, that no objective has
    trained to discard motion, and that is legal at inference under the
    vision-only rule. Raw frames average-pooled to a small grid are exactly
    that. If ridge on adjacent-frame pixel differences recovers ``long_accel``
    where a 2 048-d learned latent cannot, the encoder is the lossy step. If it
    does NOT, no encoder can be blamed and the thesis dies for this channel.

    ``pix32/centre`` is the control that makes the word "temporal" mean
    something: it is REF-C's mechanism (``refc.py:1412-1422`` keeps only the
    LAST frame's feature map) reproduced in pixel space. Temporal bases minus
    single-instant base = the quantity the architecture throws away.

PROTOCOL — deliberately IDENTICAL to the panel that produced the null
    (`…/Benchmarks & Eval/…/2026-08-03-idm-accel-recoverability/`): same 50
    episodes, same k=4 stride-2 windows, same episode-disjoint 33/17 split, same
    inner split, same DualRidge alpha path + exact-mean sentinel, same skill
    gate, same one-SE tie-break, same paired episode-cluster bootstrap. The ONLY
    thing that changes across arms is the SUBSTRATE and the FEATURE BASIS, so a
    difference between arms is attributable to the representation and to nothing
    else.

usage:
  python run_temporal_falsifier.py --stage all --out results_temporal_falsifier.json
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
REPO = HERE.parents[4]                       # …/TanitAD
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))
sys.path.insert(0, str(REPO / "taniteval"))

COMMA = Path(r"C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f")
LATENTS = Path(r"C:/Users/Admin/tanitad-data/eval/idm_derived_accel_latents.pt")
PIXCACHE = Path(r"C:/Users/Admin/tanitad-data/eval/dlatent_pixel_substrate.pt")
CLEAN_LO, CLEAN_HI = 40, 90
K, STRIDE = 4, 2
HORIZONS = (5, 10, 15, 20)
SCALARS = ("speed", "yaw_rate", "steer", "long_accel")
ACCEL_J = 3                                  # column of long_accel
SPEED_J = 0
T0 = time.time()

#: The empirical null measured on this exact split (train-mean predictor).
#: Quoted from `…/2026-08-03-idm-accel-recoverability/results_accel_recoverability.json`
#: and RE-MEASURED here as the `NULL_train_mean` arm — never inherited.
PREREG_R2_FLOOR = 0.05                       # outcome-L threshold, fixed in advance
PREREG_ORACLE_FLOOR = 0.80                   # below this the run is VOID


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# --------------------------------------------------------------------------- #
# labels — byte-identical derivation to the panel this contrasts against       #
# --------------------------------------------------------------------------- #
def heading_repair(poses: torch.Tensor, v_min: float = 0.5) -> torch.Tensor:
    """comma2k19 heading is arctan2 of ENU velocity and is UNDEFINED at
    standstill. Verbatim from `run_idm_derived_accel.py:68-84` (itself
    `idm3_labels.py:57-75`)."""
    yaw = poses[:, 2].numpy().astype(np.float64).copy()
    v = poses[:, 3].numpy().astype(np.float64)
    obs = v >= v_min
    if not obs.any():
        return torch.from_numpy(yaw).float()
    ux, uy = np.cos(yaw), np.sin(yaw)
    idx = np.where(obs, np.arange(len(yaw)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(obs))
    idx[idx < 0] = first
    return torch.from_numpy(np.arctan2(uy[idx], ux[idx])).float()


def build_targets(poses, actions, t, yaw_rep, horizons=HORIZONS):
    import idm_head as ih
    speed, steer, accel = poses[t, 3], actions[t, 0], actions[t, 1]
    yr = ih.wrap_to_pi(yaw_rep[t + 1] - yaw_rep[t - 1]) / (2.0 * ih.DT)
    scal = torch.stack([speed, yr, steer, accel], dim=-1)
    yaw0, xy0 = poses[t, 2], poses[t, :2]
    traj = torch.stack([ih.ego_frame(poses[t + h, :2] - xy0, yaw0)
                        for h in horizons], 1)
    return scal, traj


# --------------------------------------------------------------------------- #
# the PIXEL substrate                                                          #
# --------------------------------------------------------------------------- #
#: (name, grid side). ``pix1`` is whole-frame mean intensity — the parsimony
#: rung, matched in spirit to the 9-feature oracle.
PIX_GRIDS = (("pix32", 32), ("pix8", 8), ("pix1", 1))
#: The WITHIN-STACK substrate: the 3 sub-frames of the CENTRE index alone.
STK_GRIDS = (("stk32", 32), ("stk8", 8))

# ⭐ WHAT THE 9 CHANNELS ARE — MEASURED 2026-08-03, two independent probes, and
# it CORRECTS the framing this stream was briefed with.
#   `frames_u8` is [T, 9, 256, 256]: the D-015 **3-frame RGB stack**
#   (`refc.py:241` — "in_channels: int = 9  # D-015 3-frame RGB stack
#   (latest = [-3:])"), and it is a SLIDING stack — verified numerically:
#   `frames_u8[t][6:9] == frames_u8[t+1][3:6]` to max|d| = 0.0, likewise
#   `[3:6] == [t+1][0:3]`.
# ⇒ ONE model "frame" already spans 300 ms of video in the CHANNEL dimension,
#   and a conv stem over 9 channels can form a temporal difference at layer 1.
#   The claim "a single RGB frame cannot carry relative velocity" is therefore
#   NOT a description of our input. What `fmap[:, -1]` discards is everything
#   beyond that 300 ms — not motion as such.
# ⛔ It is also a defect in the FIRST pass of this very script: collapsing all 9
#   channels with `.mean(1)` box-filters 3 timesteps into each window position,
#   which attenuates exactly the signal under test. `latest_frame=True` takes
#   sub-frame [6:9] — the true frame at that index — and `stack_gray` exposes
#   the within-stack 100 ms structure as its own substrate.


def pool_gray(fr_u8: torch.Tensor, side: int, latest_frame: bool = True
              ) -> torch.Tensor:
    """frames [T, C, 256, 256] uint8 -> [T, side*side] float32 in 0..1.

    Grayscale (ITU-R 601 luma) then EXACT average pooling — 256 is divisible by
    32/8/1, so no adaptive-pool bin asymmetry can enter and the downsample is a
    pure, reproducible linear map.

    ``latest_frame`` selects the LAST RGB triplet of a D-015 9-channel stack, so
    one row is one TIMESTEP rather than a 3-frame average. See the note above.
    """
    x = fr_u8.float() / 255.0
    if x.shape[1] == 9 and latest_frame:
        x = x[:, 6:9]
    if x.shape[1] == 3:
        w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype).view(1, 3, 1, 1)
        x = (x * w).sum(1, keepdim=True)
    elif x.shape[1] != 1:
        x = x.mean(1, keepdim=True)
    k = x.shape[-1] // side
    return torch.nn.functional.avg_pool2d(x, k).reshape(x.shape[0], -1)


#: FULL-RESOLUTION motion energy, pooled AFTER the absolute difference.
MOT_GRIDS = (("mot16", 16), ("mot8", 8))


def motion_energy(fr_u8: torch.Tensor, side: int) -> torch.Tensor:
    """[T, 9or3, 256, 256] -> [T, side*side]: ``avgpool(|I_{t+1} - I_t|)`` at
    FULL 256x256 resolution, pooled only afterwards.

    ⛔⛔ THE DEFECT THIS FIXES, and it is the one that would have produced a
    FALSE "the video does not carry it".
        ``avgpool(|d|) != |avgpool(d)|``.
    Pooling BEFORE differencing — which is what a ``pix32`` + ``tdiff`` arm does
    — averages the signed brightness change over an 8x8 pixel block first, and
    the opposing-sign gradients inside that block CANCEL. On a forward-facing
    highway frame almost every edge has a matching opposite edge nearby, so the
    cancellation is close to total and the arm reads a near-zero motion field.
    Taking the magnitude FIRST and pooling second preserves motion energy, which
    is the physical quantity here: |dI| grows with optical-flow magnitude, and
    optical-flow magnitude grows with speed.

    ⚠️ FORWARD difference on purpose. ``valid_centers`` starts at index
    ``max(K,1) = 4`` and the window reaches ``t-4 = 0``; a BACKWARD difference
    would need ``gray[-1]`` at that row and would have to fabricate it. The
    forward difference is defined for every index the windows ever touch
    (max index used is ``T-17``, and the forward difference is defined to
    ``T-2``), so no row in this substrate is invented.
    """
    x = fr_u8.float() / 255.0
    if x.shape[1] == 9:
        x = x[:, 6:9]
    w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype).view(1, 3, 1, 1)
    g = (x * w).sum(1, keepdim=True) if x.shape[1] == 3 else x.mean(1, True)
    d = torch.zeros_like(g)
    d[:-1] = (g[1:] - g[:-1]).abs()
    k = g.shape[-1] // side
    return torch.nn.functional.avg_pool2d(d, k).reshape(g.shape[0], -1)


def stack_gray(fr_u8: torch.Tensor, side: int) -> torch.Tensor:
    """[T, 9, 256, 256] -> [T, 3, side*side]: the three 100 ms sub-frames of the
    D-015 stack at EACH index, as a 3-position temporal window.

    ⭐ This is the substrate that answers the architecture question directly:
    it is EXACTLY what REF-C's kept ``fmap[:, -1]`` is computed from. If the
    channel is recoverable here, the information is already inside the tensor
    REF-C keeps, and "keep W frames instead of the last" is not the fix.
    """
    if fr_u8.shape[1] != 9:
        raise ValueError(f"stack_gray needs a 9-channel D-015 stack, got "
                         f"{tuple(fr_u8.shape)}")
    x = fr_u8.float() / 255.0
    w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype).view(1, 3, 1, 1)
    subs = [(x[:, 3 * i:3 * i + 3] * w).sum(1, keepdim=True) for i in range(3)]
    k = x.shape[-1] // side
    g = [torch.nn.functional.avg_pool2d(s, k).reshape(x.shape[0], -1)
         for s in subs]
    return torch.stack(g, dim=1)                        # [T, 3, side*side]


def build_pixel_substrate(cache_path: Path, limit: int = 0):
    """Same episodes, same centres, same window offsets as the latent cache.

    ⛔ The window centres are RECOMPUTED from ``valid_centers`` rather than read
    from the latent cache (which does not store them), so the per-episode window
    count is asserted equal to the cache's ``n``. A silent misalignment here
    would make the pixel and latent arms score different rows and fabricate the
    whole result.
    """
    import idm_head as ih
    if cache_path.exists():
        log(f"loading cached pixel substrate {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    lat = torch.load(LATENTS, map_location="cpu", weights_only=False)
    lat_n = {e["name"]: int(e["n"]) for e in lat["episodes"]}
    eps = [p for p in sorted(COMMA.glob("ep_*.pt"))
           if CLEAN_LO <= int(p.stem.split("_")[1]) < CLEAN_HI]
    if limit:
        eps = eps[:limit]
    log(f"building pixel substrate over {len(eps)} clean episodes")
    out = {"episodes": [],
           "grids": {**{n: s for n, s in PIX_GRIDS},
                     **{n: s for n, s in STK_GRIDS},
                     **{n: s for n, s in MOT_GRIDS}},
           "pix_note": "pix* = the LATEST RGB triplet of the D-015 9-channel "
                       "stack at each of the 9 window indices (100 ms apart, "
                       "800 ms span). stk* = the THREE 100 ms sub-frames of the "
                       "CENTRE index only — what REF-C's kept fmap[:, -1] is "
                       "computed from.",
           "source": str(COMMA), "k": K, "stride": STRIDE}
    offs = torch.arange(-K, K + 1)
    for i, p in enumerate(eps):
        e = torch.load(p, map_location="cpu", weights_only=False)
        fr = e["frames_u8"]
        assert fr.ndim == 4 and fr.shape[2] == 256 and fr.shape[3] == 256, \
            f"{p.name}: expected [T,C,256,256], got {tuple(fr.shape)}"
        poses, actions = e["poses"].float(), e["actions"].float()
        t = ih.valid_centers(fr.shape[0], K, HORIZONS, STRIDE)
        if t.numel() == 0:
            continue
        rec = {"name": p.stem, "n": int(t.numel()), "t": t.clone()}
        # ⛔ ALIGNMENT ASSERTION — the pixel rows must be the latent rows.
        if p.stem in lat_n:
            assert lat_n[p.stem] == rec["n"], (
                f"{p.stem}: pixel windows {rec['n']} != latent cache "
                f"{lat_n[p.stem]} — the substrates are NOT aligned")
        for name, side in PIX_GRIDS:
            g = pool_gray(fr, side, latest_frame=True)  # [T, side*side]
            rec[name] = g[t[:, None] + offs[None, :]].half()
        for name, side in STK_GRIDS:
            rec[name] = stack_gray(fr, side)[t].half()  # [N, 3, side*side]
        for name, side in MOT_GRIDS:
            mo = motion_energy(fr, side)                # [T, side*side]
            rec[name] = mo[t[:, None] + offs[None, :]].half()
        S, Tj = build_targets(poses, actions, t, heading_repair(poses))
        rec["S"], rec["T"] = S, Tj
        rec["Q"] = ih.speed_seq_targets_at(poses, t, K)
        out["episodes"].append(rec)
        if (i + 1) % 10 == 0:
            log(f"  pooled {i+1}/{len(eps)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    log(f"cached pixel substrate -> {cache_path}")
    return out


# --------------------------------------------------------------------------- #
# stacking / splitting — identical rules to the contrast panel                 #
# --------------------------------------------------------------------------- #
def split_episodes(eps, hold_every: int = 3):
    tr = [e for i, e in enumerate(eps) if i % hold_every != 0]
    ho = [e for i, e in enumerate(eps) if i % hold_every == 0]
    return tr, ho


def stack_sub(eps, key, shuffle_seed: int | None = None):
    """-> X [N, W, D] float32, Y [N, 12], eid [N].

    ``shuffle_seed`` is the NEGATIVE CONTROL: destroy the substrate<->target
    link, keep the substrate marginal, the targets, the split and the fitting
    recipe untouched.
    """
    X = torch.cat([e[key] for e in eps]).float()
    S = torch.cat([e["S"] for e in eps]).float()
    Tj = torch.cat([e["T"] for e in eps]).float()
    eid = np.concatenate([np.full(e["n"], e["name"]) for e in eps])
    if shuffle_seed is not None:
        g = np.random.default_rng(shuffle_seed)
        X = X[torch.from_numpy(g.permutation(len(X)))]
    Y = torch.cat([S, Tj.reshape(len(Tj), -1)], 1)
    return X, Y, eid


def unpack(Y):
    Y = np.asarray(Y, dtype=np.float64)
    return Y[:, :4], Y[:, 4:].reshape(len(Y), len(HORIZONS), 2)


# --------------------------------------------------------------------------- #
# the closed-form arm — same selection rules as the contrast panel             #
# --------------------------------------------------------------------------- #
SEL_TOL = 0.005
MIN_SKILL = 0.01
MEAN_KEY = (None, float("inf"))


def ridge_arm(AP, Xfit, Yfit, Xsel, Xfull, Yfull, Xho, *, feat, kernel="linear",
              gamma_mults=(None,), device="cpu"):
    """Exact ridge over the whole alpha path, per-output selection on the inner
    split, refit on full train. Returns (sel_preds, ho_preds, tr_preds, keys, meta)."""
    Xf = AP.window_features(Xfit, feat)
    Xs = AP.window_features(Xsel, feat)
    XF = AP.window_features(Xfull, feat)
    Xh = AP.window_features(Xho, feat)
    Xf, Xs = AP.standardize(Xf, Xs)
    XF, Xh = AP.standardize(XF, Xh)
    ymu = Yfull.mean(0, keepdim=True)
    ysd = Yfull.std(0, keepdim=True).clamp_min(1e-6)
    kwbase = dict(kernel=kernel, matmul_device=device, matmul_dtype=torch.float32)
    alphas = AP.DualRidge.alpha_grid(2, -4, 10)
    sel_preds, ho_preds, tr_preds, gammas = {}, {}, {}, {}
    for gm in gamma_mults:
        gamma = None
        if kernel == "rbf":
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
    k0 = next(iter(sel_preds))
    sel_preds[MEAN_KEY] = torch.zeros_like(sel_preds[k0])
    ho_preds[MEAN_KEY] = torch.zeros_like(ho_preds[k0])
    tr_preds[MEAN_KEY] = torch.zeros_like(tr_preds[k0])
    keys = list(sel_preds)
    return sel_preds, ho_preds, tr_preds, keys, {
        "feature": feat, "kernel": kernel,
        "gammas": {str(k): v for k, v in gammas.items()},
        "n_features": int(Xf.shape[1]), "ymu": ymu, "ysd": ysd,
        "alpha_grid": [alphas[0], alphas[-1], len(alphas)]}


def select_hparam(AP, sel_preds, Ysel, keys, meta, tol=SEL_TOL,
                  min_skill=MIN_SKILL):
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


def assemble(preds, chosen, meta):
    ymu, ysd = meta["ymu"], meta["ysd"]
    cols = [(preds[c[0]][:, j] * ysd[0, j] + ymu[0, j]).numpy()
            for j, c in enumerate(chosen)]
    return np.stack(cols, 1)


def jsonable(x):
    return "inf(exact_train_mean)" if isinstance(x, float) and np.isinf(x) else x


def oracle_selected_r2(AP, ho_preds, Yho, keys, meta):
    """⚠️ CHEATING BY CONSTRUCTION — hyperparameter picked on the HELD-OUT set.
    Quotable ONLY as an UPPER BOUND. It exists so "the regulariser was chosen
    badly" is dead as an explanation for a null."""
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
        out[ch] = {"r2": round(float(best_s), 5), "gamma_mult": best_k[0],
                   "alpha": jsonable(best_k[1])}
    return out


# --------------------------------------------------------------------------- #
# TACTICAL readout OF THE CHANNEL UNDER TEST                                   #
# --------------------------------------------------------------------------- #
ACCEL_CLASSES = ("decelerate", "cruise", "accelerate")
ACCEL_THRESH = 0.5


def accel_classes(a):
    a = np.asarray(a, dtype=np.float64)
    c = np.ones(len(a), dtype=np.int64)
    c[a > ACCEL_THRESH] = 2
    c[a < -ACCEL_THRESH] = 0
    return c


def accel_tactical(FF, pred_a, gt_a):
    P, G = accel_classes(pred_a), accel_classes(gt_a)
    C = FF.confusion(P, G, 3)
    sup, fired = C.sum(1), C.sum(0)
    return {
        "classes": list(ACCEL_CLASSES), "threshold_mps2": ACCEL_THRESH,
        "confusion_gt_rows_pred_cols": C.tolist(),
        "support_gt": sup.tolist(), "n_fired_pred": fired.tolist(),
        "recall": [round(float(C[i, i] / sup[i]), 4) if sup[i] else None
                   for i in range(3)],
        "precision": [round(float(C[i, i] / fired[i]), 4) if fired[i] else None
                      for i in range(3)],
        "balanced_accuracy": round(FF.balanced_accuracy(C), 4),
        "accuracy": round(float(np.trace(C) / max(C.sum(), 1)), 4),
        "chance_balanced_accuracy": round(1.0 / max(int((sup > 0).sum()), 1), 4)}


# --------------------------------------------------------------------------- #
def score_arm(AP, TCI, APCI, FF, pred, Yho, tr_pred, Ytr, eid, n_boot):
    gs, gt = unpack(Yho)
    ps = pred[:, :4]
    pt = pred[:, 4:].reshape(len(pred), len(HORIZONS), 2)
    tgs, _ = unpack(Ytr)
    out = {"r2": {}, "r2_train_insample": {}, "mae": {}}
    for j, ch in enumerate(SCALARS):
        out["r2"][ch] = APCI.stat_episode_cluster_bootstrap(
            (lambda p, g: (lambda sel: AP.r2_score(p[sel], g[sel])))(
                ps[:, j], gs[:, j]),
            eid, n_boot=n_boot, name=f"r2_{ch}")
        out["r2_train_insample"][ch] = round(
            AP.r2_score(tr_pred[:, j], tgs[:, j]), 4)
        out["mae"][ch] = round(float(np.abs(ps[:, j] - gs[:, j]).mean()), 5)
    per_win = np.linalg.norm(pt - gt, axis=-1).mean(1)
    out["ade_2s"] = TCI.episode_cluster_bootstrap(per_win, eid, n_boot=n_boot)
    out["four_families"] = FF.all_families(pt, gt, FF.IDM_DT_S,
                                           pred_scalars=ps, gt_scalars=gs)
    out["four_families"]["TACTICAL"]["from_long_accel_scalar"] = \
        accel_tactical(FF, ps[:, ACCEL_J], gs[:, ACCEL_J])
    Pa, Ga = accel_classes(ps[:, ACCEL_J]), accel_classes(gs[:, ACCEL_J])
    out["tactical_ba_long_accel_ci"] = APCI.stat_episode_cluster_bootstrap(
        lambda sel: FF.balanced_accuracy(FF.confusion(Pa[sel], Ga[sel], 3)),
        eid, n_boot=n_boot, name="balanced_accuracy_from_long_accel")
    return out


def paired_delta(AP, APCI, pred, ctrl, Yho, eid, n_boot):
    """Paired ΔR² of the arm against its OWN shuffled control, per scalar.

    ⛔ Never a quadrature combination of two single-arm intervals: the arms are
    scored on the same windows and are not independent.
    """
    gs, _ = unpack(Yho)
    out = {}
    for j, ch in enumerate(SCALARS):
        a, b, g = pred[:, j], ctrl[:, j], gs[:, j]
        d = APCI.paired_stat_episode_cluster_bootstrap(
            (lambda p, gg: (lambda sel: AP.r2_score(p[sel], gg[sel])))(a, g),
            (lambda p, gg: (lambda sel: AP.r2_score(p[sel], gg[sel])))(b, g),
            eid, n_boot=n_boot, name=f"delta_r2_{ch}")
        if np.allclose(a, b, rtol=0, atol=1e-12):
            # degeneracy guard: bit-identical arms are not "separated"
            d["separated"] = False
            d["degenerate_identical_predictions"] = True
        out[ch] = d
    return out


# --------------------------------------------------------------------------- #
#: RBF arms. ⚠️ NOT decoration — a LINEAR probe on pixel differences is
#: PHYSICALLY MIS-SPECIFIED for this question. The brightness-constancy relation
#: is dI/dt = -grad(I) . v, so brightness change is linear in velocity only for a
#: FIXED image gradient; across scenes the gradient varies and the map from
#: pixel differences to speed is not linear. Running the pixel family
#: linear-only would produce a null that is a property of the PROBE, and the
#: null would point (wrongly) at "the video does not carry it". The latent family
#: already got an rbf rung in the panel this contrasts against, so this is also
#: simple parity of treatment. Kept to the SMALL feature bases where an exact
#: kernel over the whole regularisation path is cheap.
RBF_GAMMA_MULTS = (0.25, 1.0, 4.0)
RBF_ARMS: tuple[tuple[str, str, str], ...] = (
    ("pix8_abstdiff_rbf",  "pix8",  "abstdiff"),
    ("pix8_tdiff_rbf",     "pix8",  "tdiff"),
    ("pix1_window_rbf",    "pix1",  "window"),
    ("stk8_abstdiff_rbf",  "stk8",  "abstdiff"),
    ("stk8_tdiff_rbf",     "stk8",  "tdiff"),
    ("pix32_centre_rbf",   "pix32", "centre"),      # single-instant, rbf
    ("mot8_window_rbf",    "mot8",  "window"),
    ("mot8_tdiff_rbf",     "mot8",  "tdiff"),
    ("mot16_window_rbf",   "mot16", "window"),
)

ARMS: tuple[tuple[str, str, str], ...] = (
    # (arm name, substrate key, feature basis)
    ("v1_centre",        "Z",     "centre"),
    ("v1_window",        "Z",     "window"),
    ("v1_diff",          "Z",     "diff"),
    ("v1_tdiff",         "Z",     "tdiff"),
    ("v1_abstdiff",      "Z",     "abstdiff"),
    ("pix32_centre",     "pix32", "centre"),      # ⭐ single-instant control
    ("pix32_window",     "pix32", "window"),
    ("pix32_diff",       "pix32", "diff"),
    ("pix32_tdiff",      "pix32", "tdiff"),
    ("pix32_abstdiff",   "pix32", "abstdiff"),
    ("pix8_tdiff",       "pix8",  "tdiff"),
    ("pix8_abstdiff",    "pix8",  "abstdiff"),
    ("pix1_window",      "pix1",  "window"),
    ("pix1_tdiff",       "pix1",  "tdiff"),
    # ⭐ THE REF-C RUNG — the 3 sub-frames of the ONE index whose feature map the
    # anchor decoder keeps. `stk32_centre` is the middle sub-frame alone: a TRUE
    # single instant, and the only honest "one RGB frame" control in this panel.
    ("stk32_centre",     "stk32", "centre"),
    ("stk32_window",     "stk32", "window"),
    ("stk32_tdiff",      "stk32", "tdiff"),
    ("stk32_abstdiff",   "stk32", "abstdiff"),
    ("stk8_tdiff",       "stk8",  "tdiff"),
    ("stk8_abstdiff",    "stk8",  "abstdiff"),
    # ⭐ THE MOTION-ENERGY RUNG — the physically correct optical-flow proxy.
    # `centre` = motion energy NOW (a velocity proxy); `tdiff` = its change
    # (an ACCELERATION proxy); `window` = the whole 800 ms profile.
    ("mot16_centre",     "mot16", "centre"),
    ("mot16_window",     "mot16", "window"),
    ("mot16_tdiff",      "mot16", "tdiff"),
    ("mot8_centre",      "mot8",  "centre"),
    ("mot8_window",      "mot8",  "window"),
    ("mot8_tdiff",       "mot8",  "tdiff"),
)


def merge_substrates(lat, pix):
    """One episode list carrying BOTH substrates on the SAME rows.

    The join key is the episode stem and the row order is the shared
    ``valid_centers`` order; the window count is asserted equal on both sides
    (again — the build asserted it too, and this is the second, independent
    location, which is the programme's own two-probe rule applied to an
    alignment claim rather than to an absence claim).
    """
    lat_by = {e["name"]: e for e in lat["episodes"]}
    eps = []
    for pe in pix["episodes"]:
        le = lat_by.get(pe["name"])
        if le is None:
            continue
        assert int(le["n"]) == int(pe["n"]), \
            f"{pe['name']}: latent n={le['n']} vs pixel n={pe['n']}"
        assert torch.allclose(le["S"].float(), pe["S"].float(), atol=1e-5), \
            f"{pe['name']}: label mismatch between the two substrate builds"
        rec = dict(pe)
        rec["Z"] = le["Z"]
        eps.append(rec)
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_temporal_falsifier.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pixcache", default=str(PIXCACHE))
    ap.add_argument("--skip-rbf", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stage", default="all",
                    choices=("substrate", "arms", "sensitivity", "all"))
    args = ap.parse_args()

    from tanitad.eval import accel_probe as AP
    from tanitad.eval import ap_ci as APCI
    from tanitad.eval import idm_families as FF
    import taniteval.ci as TCI

    log(f"device={args.device} torch={torch.__version__}")
    pix = build_pixel_substrate(Path(args.pixcache), limit=args.limit)
    if args.stage == "substrate":
        return
    lat = torch.load(LATENTS, map_location="cpu", weights_only=False)
    eps = merge_substrates(lat, pix)
    log(f"merged substrates: {len(eps)} episodes, "
        f"{sum(e['n'] for e in eps)} windows")

    tr_eps, ho_eps = split_episodes(eps)
    fit_eps, sel_eps = split_episodes(tr_eps, hold_every=3)   # inner split
    log(f"split: train {len(tr_eps)} / heldout {len(ho_eps)} episodes; "
        f"inner fit {len(fit_eps)} / sel {len(sel_eps)}")

    res = {
        "prereg": "Project Steering/PREREG_TEMPORAL_LATENT.md",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "substrate": {
            "latent_cache": str(LATENTS), "pixel_cache": str(args.pixcache),
            "pix_note": pix.get("pix_note"),
            "encoder": lat.get("encoder"), "encoder_step": lat.get("encoder_step"),
            "state_dim": lat.get("state_dim"), "pixel_grids": pix["grids"],
            "k": K, "stride": STRIDE, "horizons": list(HORIZONS)},
        "split": {
            "train_episodes": [e["name"] for e in tr_eps],
            "heldout_episodes": [e["name"] for e in ho_eps],
            "inner_fit_episodes": [e["name"] for e in fit_eps],
            "inner_sel_episodes": [e["name"] for e in sel_eps],
            "n_train_windows": int(sum(e["n"] for e in tr_eps)),
            "n_heldout_windows": int(sum(e["n"] for e in ho_eps))},
        "estimator": "paired_episode_cluster_bootstrap (taniteval/ap_ci); "
                     "overlapping_holdout_se is NOT used anywhere",
        "n_boot": args.n_boot,
        "arms": {}, "paired_vs_control": {}, "verdict": {}}

    # targets are substrate-independent
    _, Yho, eid_ho = stack_sub(ho_eps, "pix1")
    _, Yfull, _ = stack_sub(tr_eps, "pix1")
    _, Yfit, _ = stack_sub(fit_eps, "pix1")
    _, Ysel, _ = stack_sub(sel_eps, "pix1")

    # ---- the empirical NULL (train mean) ---------------------------------- #
    mean_pred = np.repeat(np.asarray(Yfull, np.float64).mean(0, keepdims=True),
                          len(Yho), axis=0)
    mean_tr = np.repeat(np.asarray(Yfull, np.float64).mean(0, keepdims=True),
                        len(Yfull), axis=0)
    res["arms"]["NULL_train_mean"] = score_arm(
        AP, TCI, APCI, FF, mean_pred, Yho, mean_tr, Yfull, eid_ho, args.n_boot)
    res["arms"]["NULL_train_mean"]["n_features"] = 0
    log("NULL_train_mean long_accel R2 "
        f"{res['arms']['NULL_train_mean']['r2']['long_accel']['point']:+.4f}")

    # ---- ORACLE-INPUT control: the TRUE 9-position speed window ----------- #
    def qsub(elist):
        return torch.cat([e["Q"] for e in elist]).float().unsqueeze(-1)  # [N,9,1]

    sp, hp, tp, keys, meta = ridge_arm(
        AP, qsub(fit_eps), Yfit, qsub(sel_eps), qsub(tr_eps), Yfull,
        qsub(ho_eps), feat="window", device=args.device)
    ch = select_hparam(AP, sp, Ysel, keys, meta)
    orc = assemble(hp, ch, meta)
    res["arms"]["ORACLE_true_speed_window"] = score_arm(
        AP, TCI, APCI, FF, orc, Yho, assemble(tp, ch, meta), Yfull, eid_ho,
        args.n_boot)
    res["arms"]["ORACLE_true_speed_window"]["n_features"] = meta["n_features"]
    res["arms"]["ORACLE_true_speed_window"]["note"] = (
        "capacity control — the SAME protocol fed the TRUE speed track. If this "
        "recovers long_accel, the protocol/head/regulariser are all cleared and "
        "any null from the other arms is about the INPUT.")
    orc_r2 = res["arms"]["ORACLE_true_speed_window"]["r2"]["long_accel"]["point"]
    log(f"ORACLE_true_speed_window long_accel R2 {orc_r2:+.4f}")

    # ---- the ladder ------------------------------------------------------- #
    all_arms = tuple(ARMS) + tuple(
        a for a in RBF_ARMS if not args.skip_rbf)
    for name, key, feat in all_arms:
        t_arm = time.time()
        kern = "rbf" if name.endswith("_rbf") else "linear"
        gms = RBF_GAMMA_MULTS if kern == "rbf" else (None,)
        packs = {}
        for tag, shuf in (("arm", None), ("ctrl", 12345)):
            Xf, _, _ = stack_sub(fit_eps, key, shuffle_seed=shuf)
            Xs, _, _ = stack_sub(sel_eps, key, shuffle_seed=None if shuf is None
                                 else shuf + 1)
            XF, _, _ = stack_sub(tr_eps, key, shuffle_seed=shuf)
            Xh, _, _ = stack_sub(ho_eps, key, shuffle_seed=None if shuf is None
                                 else shuf + 2)
            sp, hp, tp, keys, meta = ridge_arm(
                AP, Xf, Yfit, Xs, XF, Yfull, Xh, feat=feat, kernel=kern,
                gamma_mults=gms, device=args.device)
            chosen = select_hparam(AP, sp, Ysel, keys, meta)
            packs[tag] = (assemble(hp, chosen, meta), assemble(tp, chosen, meta),
                          chosen, keys, hp, meta)
        pred, tr_pred, chosen, keys, hp, meta = packs["arm"]
        ctrl = packs["ctrl"][0]
        sc = score_arm(AP, TCI, APCI, FF, pred, Yho, tr_pred, Yfull, eid_ho,
                       args.n_boot)
        sc["n_features"] = meta["n_features"]
        sc["substrate"], sc["feature_basis"], sc["kernel"] = key, feat, kern
        sc["hparam_selection"] = [
            {"channel": (SCALARS + tuple(f"traj{i}" for i in range(8)))[j],
             "gamma_mult": c[0][0], "alpha": jsonable(c[0][1]),
             "inner_r2_at_choice": round(c[1], 5),
             "inner_r2_best": round(c[2], 5), "rule": c[3]}
            for j, c in enumerate(chosen)]
        sc["ORACLE_selected_on_heldout_UPPER_BOUND"] = oracle_selected_r2(
            AP, hp, Yho, keys, meta)
        sc["control_r2"] = {c: round(AP.r2_score(ctrl[:, j],
                                                 np.asarray(Yho)[:, j]), 5)
                            for j, c in enumerate(SCALARS)}
        res["arms"][name] = sc
        res["paired_vs_control"][name] = paired_delta(
            AP, APCI, pred, ctrl, Yho, eid_ho, args.n_boot)
        d = res["paired_vs_control"][name]
        # ASCII only: the Windows console is cp1252 and a bare Greek delta here
        # kills the run AFTER the arm has been fitted (measured 2026-08-03).
        log(f"{name:>16s} F={meta['n_features']:>6d}  "
            f"accel R2={sc['r2']['long_accel']['point']:+.4f}  "
            f"d_ctrl={d['long_accel']['delta']:+.4f} "
            f"[{d['long_accel']['lo']:+.4f},{d['long_accel']['hi']:+.4f}]"
            f"{'*' if d['long_accel']['separated'] else ' '}  |  "
            f"speed R2={sc['r2']['speed']['point']:+.4f} "
            f"d={d['speed']['delta']:+.4f}"
            f"{'*' if d['speed']['separated'] else ' '}  "
            f"({time.time()-t_arm:.0f}s)")
        Path(args.out).write_text(json.dumps(res, indent=1, default=str))

    # ---- SENSITIVITY FLOOR on the pixel substrate ------------------------- #
    # A pixel-side null is only quotable with a magnitude. Plant a known multiple
    # of the target along a random direction of the pixel substrate and re-run
    # the identical probe. ⚠️ INJECT FIRST, SHUFFLE SECOND — the reverse order
    # was a measured defect in the prior panel (the shuffle destroyed the native
    # link and the injection then rebuilt it, making the "control" better).
    if args.stage in ("all", "sensitivity"):
        sens = []
        # ⚠️ ONE gain for every block. `inject_signal` standardises `y` and takes
        # the RMS from whatever block it is handed; letting it do that per block
        # would give train and held-out DIFFERENT injection gains and the probe
        # would then be partly measuring that mismatch (the function's own
        # docstring warns about exactly this). So: standardise `y` with TRAIN
        # statistics and pass the TRAIN rms explicitly everywhere.
        X_tr_raw, _, _ = stack_sub(tr_eps, "pix32")
        train_rms = float(X_tr_raw.double().pow(2).mean().sqrt())
        yF_raw = np.asarray(Yfull)[:, ACCEL_J]
        y_mu, y_sd = float(yF_raw.mean()), float(yF_raw.std() + 1e-9)
        yh = np.asarray(Yho)[:, ACCEL_J]

        def zy(v):
            return (np.asarray(v)[:, ACCEL_J] - y_mu) / y_sd

        for frac in (0.01, 0.05):
            blocks = {}
            for tag, elist, yv in (("fit", fit_eps, Yfit), ("sel", sel_eps, Ysel),
                                   ("full", tr_eps, Yfull), ("ho", ho_eps, Yho)):
                X, _, _ = stack_sub(elist, "pix32")
                blocks[tag], _ = AP.inject_signal(
                    X, zy(yv), frac, seed=7, standardize_y=False, rms=train_rms)
            sp, hp, tp, keys, meta = ridge_arm(
                AP, blocks["fit"], Yfit, blocks["sel"], blocks["full"], Yfull,
                blocks["ho"], feat="centre", device=args.device)
            chosen = select_hparam(AP, sp, Ysel, keys, meta)
            pr = assemble(hp, chosen, meta)
            r2 = APCI.stat_episode_cluster_bootstrap(
                lambda sel, p=pr[:, ACCEL_J]: AP.r2_score(p[sel], yh[sel]),
                eid_ho, n_boot=args.n_boot, name="r2_long_accel_planted")
            sens.append({"planted_frac_of_substrate_rms": frac,
                         "basis": "pix32/centre", "train_rms": train_rms,
                         "r2_long_accel": r2})
            log(f"  sensitivity frac={frac}: accel R2 {r2['point']:+.4f}")
        res["sensitivity_pixel"] = sens

    # ---- the PRE-REGISTERED verdict -------------------------------------- #
    def sep_pos(name, ch):
        d = res["paired_vs_control"].get(name, {}).get(ch)
        return bool(d and d["separated"] and d["delta"] > 0)

    temporal = [n for n, k, f in all_arms
                if (k.startswith("pix") or k.startswith("stk")
                    or k.startswith("mot"))
                and f in ("window", "diff", "tdiff", "abstdiff")]
    #: TWO independent single-instant controls, both a genuine ONE-frame input:
    #: the true frame at the window centre, and the middle sub-frame of the
    #: D-015 stack. The programme's own two-probe rule, applied to a control.
    single_instant = ["pix32_centre", "stk32_centre",
                      "pix32_centre_rbf"]
    latent_arms = [n for n, k, _ in all_arms if k == "Z"]
    fired_temporal = [n for n in temporal
                      if sep_pos(n, "long_accel")
                      and res["arms"][n]["r2"]["long_accel"]["point"]
                      >= PREREG_R2_FLOOR]
    weak_temporal = [n for n in temporal
                     if sep_pos(n, "long_accel") and n not in fired_temporal]
    shortcut = any(sep_pos(n, "long_accel")
                   and res["arms"].get(n, {}).get("r2", {})
                   .get("long_accel", {}).get("point", -9) >= PREREG_R2_FLOOR
                   for n in single_instant if n in res["arms"])
    latent_fired = [n for n in latent_arms if sep_pos(n, "long_accel")]
    void = orc_r2 < PREREG_ORACLE_FLOOR
    if void:
        outcome = "VOID_oracle_below_floor"
    elif shortcut:
        outcome = "M_shortcut_no_verdict"
    elif fired_temporal:
        outcome = "L_latent_limited"
    else:
        outcome = "V_video_limited"
    res["verdict"] = {
        "outcome": outcome,
        "prereg_thresholds": {"r2_floor": PREREG_R2_FLOOR,
                              "oracle_floor": PREREG_ORACLE_FLOOR},
        "oracle_long_accel_r2": orc_r2,
        "temporal_pixel_arms_separated_and_over_floor": fired_temporal,
        "temporal_pixel_arms_separated_but_under_floor": weak_temporal,
        "single_instant_control_separated": shortcut,
        "single_instant_controls": {
            n: {"separated": sep_pos(n, "long_accel"),
                "r2": res["arms"].get(n, {}).get("r2", {})
                .get("long_accel", {}).get("point")}
            for n in single_instant if n in res["arms"]},
        "v1_latent_arms_separated": latent_fired,
        "speed_positive_control_separated": {
            n: sep_pos(n, "speed") for n, _, _ in all_arms}}
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    log(f"VERDICT: {outcome}  ->  {args.out}")


if __name__ == "__main__":
    main()
