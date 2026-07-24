"""TanitEval v2 — TIER-0 capability probe: every metric that is computable
TODAY, on CPU, from an already-persisted ``results/windows_<arm>.pt``.

WHY THIS FILE EXISTS
--------------------
`TANITEVAL_V2_METRIC_SUITE.md` claims a tier-0 set — metrics that need **no
GPU, no pod, no re-run, no new logging**. This script is the executable proof of
that claim, and it is where every "MEASURED (this doc)" number in the design doc
comes from. Same relationship `recompute_ci.py` has to the CI recompute: a
CPU-only back-computation over the saved window dump.

WHAT `windows_<arm>.pt` ACTUALLY CONTAINS (rollout.save_windows)
    pred/gt/cv [N,4,2] ego-frame waypoints at WP_STEPS 5/10/15/20 = 0.5/1/1.5/2 s
    eid [N] · speed [N] (= v0, m/s) · head_deg [N] (net heading change) · wp_steps
So the tier-0 surface is **4 waypoints 0.5 s apart**, not the dense 20-step
rollout — `rollout.collect` computes ``wp_full [b,20,2]`` and throws 16 of the 20
steps away at ``index_select`` time. Everything that needs 10 Hz derivatives
(jerk, comfort, a real curvature profile, decel-onset lead time) is therefore
TIER-1 and needs that one line changed, not new science.

SANITY PIN. On ``windows_flagship-30k.pt`` this reproduces MODEL_REGISTRY §1.2
exactly: full-set ade_0_2s 0.4271, CV 0.8377, long-RMSE@2s 1.042 / lat-RMSE@2s
0.360 (registry "1.04 / 0.36"), top-decile speed bias +0.659 m/s (registry
"+0.66"), top-decile long-RMSE 1.379 (registry "1.38"). If those drift, the
artifact or the geometry convention changed — fail loud rather than publish.

INTERVALS. Episode-cluster bootstrap only (``taniteval.ci`` semantics, reimplemented
here in numpy so the script runs with no package import and no torch-on-pod
assumptions). The paired form is used for every arm-vs-floor comparison; the
deprecated ``overlapping_holdout_se`` is deliberately absent.

Run:  python v2_tier0_probe.py [--arm flagship-30k] [--results DIR] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

DT_WP = 0.5          # the 4 persisted waypoints are 0.5 s apart
EPS = 1e-6
N_BOOT = 2000
BRAKE_A = 0.5        # m/s^2 — |mean realised accel| splitting brake/steady/accel
K_SIGNAL = 1e-3      # 1/m — below this GT curvature is "straight", sign is free


# ======================================================================== #
# geometry (the pathspeed.py conventions, at the 4-waypoint surface)        #
# ======================================================================== #
def prepend_origin(p):
    """[N,4,2] -> [N,5,2] with the observed pose (0,0) at t=0."""
    return torch.cat([torch.zeros(p.shape[0], 1, 2, dtype=p.dtype), p], dim=1)


def tangents(p):
    """Unit direction of travel over each 0.5 s segment. [N,4,2] -> [N,4,2]."""
    full = prepend_origin(p)
    d = full[:, 1:] - full[:, :-1]
    n = d.norm(dim=-1, keepdim=True)
    t = torch.where(n > EPS, d / n.clamp_min(EPS), torch.zeros_like(d))
    fwd = torch.tensor([1.0, 0.0])
    for i in range(t.shape[1]):                     # carry last valid forward
        bad = t[:, i].norm(dim=-1) <= EPS
        if bad.any():
            t[bad, i] = t[bad, i - 1] if i > 0 else fwd
    return t


def frenet(pred, gt):
    """Signed along/cross residual of pred vs GT on the GT tangent frame.

    along + = pred is AHEAD of GT along the path; cross + = pred is LEFT.
    Orthonormal basis => along^2 + cross^2 == ||pred-gt||^2 exactly."""
    tg = tangents(gt)
    nv = torch.stack([-tg[..., 1], tg[..., 0]], dim=-1)
    r = pred - gt
    return (r * tg).sum(-1), (r * nv).sum(-1)


def step_speed(p, dt=DT_WP):
    """Mean speed over each 0.5 s segment (m/s). NOT an instantaneous speed —
    at this surface it is a 0.5 s box-average, which is why jerk is tier-1."""
    full = prepend_origin(p)
    return (full[:, 1:] - full[:, :-1]).norm(dim=-1) / dt


def heading_deg(p):
    t = tangents(p)
    return torch.atan2(t[..., 1], t[..., 0]) * 180.0 / math.pi


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def arclength(p):
    full = prepend_origin(p)
    return (full[:, 1:] - full[:, :-1]).norm(dim=-1).sum(1)


def menger_curvature(p):
    """Signed Menger curvature (1/m) at each interior knot of the polyline.

    CAVEAT, measured: at 0.5 s knot spacing this is dominated by knot jitter at
    low speed (spacing ~1.9 m). Use the SIGN, not the magnitude, at tier-0."""
    full = prepend_origin(p)
    a, b, c = full[:, :-2], full[:, 1:-1], full[:, 2:]
    cross = ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
             - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))
    denom = ((b - a).norm(dim=-1) * (c - b).norm(dim=-1)
             * (c - a).norm(dim=-1)).clamp_min(EPS)
    return 2.0 * cross / denom


def path_geometry_crosstrack(pred, gt, m=8):
    """SPEED-DECOUPLED lateral error: resample both paths at the SAME arc
    lengths d_j = j/m * min(L_pred, L_gt), take the perpendicular deviation.

    Two paths tracing the same GEOMETRY at different speeds score ~0. This is
    ``pathspeed.path_geometry_crosstrack`` at the 4-knot surface."""
    fp, fg = prepend_origin(pred), prepend_origin(gt)

    def cum(f):
        return torch.cat([torch.zeros(f.shape[0], 1),
                          (f[:, 1:] - f[:, :-1]).norm(dim=-1).cumsum(1)], 1)
    cp, cg = cum(fp), cum(fg)
    L = torch.minimum(cp[:, -1], cg[:, -1])
    j = torch.arange(1, m + 1, dtype=torch.float32)
    q = L[:, None] * (j[None, :] / m)

    def interp(poly, c, q):
        qc = torch.minimum(q, c[:, -1:])
        idx = torch.searchsorted(c.contiguous(), qc.contiguous(),
                                 right=True).clamp(1, poly.shape[1] - 1)
        c0, c1 = torch.gather(c, 1, idx - 1), torch.gather(c, 1, idx)
        w = ((qc - c0) / (c1 - c0).clamp_min(EPS)).clamp(0, 1).unsqueeze(-1)
        g0 = torch.gather(poly, 1, (idx - 1).unsqueeze(-1).expand(-1, -1, 2))
        g1 = torch.gather(poly, 1, idx.unsqueeze(-1).expand(-1, -1, 2))
        return g0 * (1 - w) + g1 * w
    Gp, Gg = interp(fp, cp, q), interp(fg, cg, q)
    tg = torch.empty_like(Gg)
    tg[:, 1:] = Gg[:, 1:] - Gg[:, :-1]
    tg[:, 0] = Gg[:, 0]
    tn = tg / tg.norm(dim=-1, keepdim=True).clamp_min(EPS)
    nv = torch.stack([-tn[..., 1], tn[..., 0]], -1)
    cross = ((Gp - Gg) * nv).sum(-1)
    pg = cross.pow(2).mean(1).sqrt()
    return torch.where(L > EPS, pg, torch.zeros_like(pg))


# ======================================================================== #
# trivial floors constructible from the persisted meta alone               #
# ======================================================================== #
def hold_v0(v0, n=4):
    """Go straight at the observed entry speed — the floor every LONGITUDINAL
    metric must beat, and the one VTARGET provably loses to at 2 s."""
    t = torch.arange(1, n + 1, dtype=torch.float32) * DT_WP
    return torch.stack([v0[:, None] * t[None, :], torch.zeros(len(v0), n)], -1)


# ======================================================================== #
# intervals — episode-cluster bootstrap ONLY (taniteval/ci.py semantics)    #
# ======================================================================== #
def _clusters(eid):
    e = np.asarray([str(x) for x in eid])
    if e.size == 0:
        raise ValueError("episode-cluster bootstrap needs a non-empty eid")
    uniq = np.unique(e)
    return uniq, {u: np.flatnonzero(e == u) for u in uniq}


def _draws(uniq, idx, n_boot, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        yield np.concatenate([idx[p] for p in pick])


def boot(vals, eid, stat=np.nanmean, n_boot=N_BOOT, seed=0):
    """Percentile CI on any per-window statistic, resampling EPISODES.

    ``stat`` is a callable over the selected per-window values, so this covers
    means, rates, quantiles and (with a 2-column input) kappa-style statistics —
    the callable path ``taniteval/ci.py`` does not yet expose."""
    v = np.asarray(vals, dtype=float)
    uniq, idx = _clusters(eid)
    b = np.array([float(stat(v[s])) for s in _draws(uniq, idx, n_boot, seed)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"mean": round(float(stat(v)), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "n_windows": int(v.size),
            "n_episodes": int(len(uniq)), "n_boot": int(n_boot),
            "estimator": "episode_cluster_bootstrap"}


def paired(a, b, eid, n_boot=N_BOOT, seed=0):
    """CI on mean(a) - mean(b) with the SAME resampled episodes each draw.

    ``separated`` is the decision predicate. Never combine two single-arm
    intervals in quadrature — the estimates are not independent."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    uniq, idx = _clusters(eid)
    d = np.array([float(np.nanmean(a[s]) - np.nanmean(b[s]))
                  for s in _draws(uniq, idx, n_boot, seed)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": round(float(np.nanmean(a) - np.nanmean(b)), 4),
            "lo": round(float(lo), 4), "hi": round(float(hi), 4),
            "separated": bool(lo > 0 or hi < 0),
            "n_episodes": int(len(uniq)), "n_boot": int(n_boot),
            "estimator": "paired_episode_cluster_bootstrap"}


# ======================================================================== #
# the tier-0 metric block                                                  #
# ======================================================================== #
def per_window(pred, gt):
    """Every tier-0 per-window component, in ONE place. Keys are the metric
    names used in TANITEVAL_V2_METRIC_SUITE.md."""
    de = torch.linalg.norm(pred - gt, dim=-1)              # [N,4]
    al, cr = frenet(pred, gt)
    vp, vg = step_speed(pred), step_speed(gt)
    kp, kg = menger_curvature(pred), menger_curvature(gt)
    straight = kg.abs() < K_SIGNAL
    return {
        "ade_0_2s": de.mean(1),
        "fde_2s": de[:, -1],
        "miss_2m": (de[:, -1] > 2.0).float(),
        # --- LONGITUDINAL ------------------------------------------------- #
        "long_abs_2s_m": al[:, -1].abs(),
        "long_signed_2s_m": al[:, -1],
        "long_sq_2s": al[:, -1].pow(2),
        "speed_mae_mps": (vp - vg).abs().mean(1),
        "speed_bias_mps": (vp - vg).mean(1),
        "progress_abs_err_m": (arclength(pred) - arclength(gt)).abs(),
        "progress_signed_err_m": arclength(pred) - arclength(gt),
        # --- LATERAL ------------------------------------------------------ #
        "lat_abs_2s_m": cr[:, -1].abs(),
        "lat_signed_2s_m": cr[:, -1],
        "lat_sq_2s": cr[:, -1].pow(2),
        "heading_mae_2s_deg": wrap_deg(heading_deg(pred)[:, -1]
                                       - heading_deg(gt)[:, -1]).abs(),
        "pathgeom_crosstrack_m": path_geometry_crosstrack(pred, gt),
        "curv_sign_agree": ((kp.sign() == kg.sign()) | straight).float().mean(1),
    }


def regimes(gt, v0):
    """Longitudinal regime of each window from the REALISED future speed —
    the only target-speed reference that survived measurement (VTARGET at 2 s
    is refuted; see MODEL_REGISTRY §4.1)."""
    a = (step_speed(gt)[:, -1] - v0) / 2.0                 # m/s^2 over 2 s
    return np.where(a.numpy() <= -BRAKE_A, "brake",
                    np.where(a.numpy() >= BRAKE_A, "accel", "steady"))


def curv_buckets(head_deg):
    h = head_deg.abs()
    return np.where(h.numpy() < 5.0, "straight",
                    np.where(h.numpy() < 15.0, "gentle", "sharp"))


def speed_strata(v0):
    q = torch.quantile(v0, torch.tensor([1 / 3, 2 / 3, 0.90]))
    return {"low": v0 < q[0], "med": (v0 >= q[0]) & (v0 < q[1]),
            "high": v0 >= q[1], "top10pct": v0 >= q[2]}, \
        [round(float(x), 3) for x in q]


SANITY = {"ade_0_2s": 0.4271, "cv_ade_0_2s": 0.8377,
          "long_rmse_2s_m": 1.042, "lat_rmse_2s_m": 0.360,
          "top10pct_speed_bias_mps": 0.659, "top10pct_long_rmse_2s_m": 1.379}


def run(path, n_boot=N_BOOT):
    d = torch.load(path, map_location="cpu", weights_only=False)
    pred, gt, cv, eid, v0 = (d["pred"], d["gt"], d["cv"], d["eid"], d["speed"])
    hv = hold_v0(v0)
    pw = {"model": per_window(pred, gt), "cv": per_window(cv, gt),
          "holdv0": per_window(hv, gt)}
    out = {"artifact": str(path), "n_windows": int(pred.shape[0]),
           "n_episodes": len(set(eid)), "wp_steps": d.get("wp_steps"),
           "surface": "4 waypoints 0.5 s apart (tier-0); dense 20-step is tier-1",
           "estimator": "episode_cluster_bootstrap (paired for arm-vs-floor)"}

    # ---- headline block, every metric with its interval ------------------ #
    out["headline"] = {k: boot(pw["model"][k], eid, n_boot=n_boot)
                       for k in pw["model"]}
    out["floors"] = {f: {k: round(float(np.nanmean(pw[f][k])), 4)
                         for k in pw[f]} for f in ("cv", "holdv0")}
    # RMSE forms (the registry quotes RMSE, not MAE, for long/lat)
    out["rmse"] = {
        "long_rmse_2s_m": round(float(pw["model"]["long_sq_2s"].mean().sqrt()), 4),
        "lat_rmse_2s_m": round(float(pw["model"]["lat_sq_2s"].mean().sqrt()), 4),
        "long_frac_of_2s_sqerr": round(
            float(pw["model"]["long_sq_2s"].mean()
                  / (pw["model"]["long_sq_2s"].mean()
                     + pw["model"]["lat_sq_2s"].mean() + EPS)), 4)}

    # ---- the decisive test: is the win LONGITUDINAL or LATERAL? ---------- #
    out["vs_floor_paired"] = {}
    for floor in ("cv", "holdv0"):
        out["vs_floor_paired"][floor] = {
            k: paired(pw[floor][k], pw["model"][k], eid, n_boot=n_boot)
            for k in ("ade_0_2s", "long_abs_2s_m", "lat_abs_2s_m",
                      "speed_mae_mps", "pathgeom_crosstrack_m",
                      "heading_mae_2s_deg")}

    # ---- CRUISE vs TRANSIENT (the longitudinal decomposition) ------------ #
    reg = regimes(gt, v0)
    out["longitudinal_regime"] = {}
    for r in ("brake", "steady", "accel"):
        m = reg == r
        if not m.any():
            continue
        sub = [i for i, f in enumerate(m) if f]
        out["longitudinal_regime"][r] = {
            "n": int(m.sum()),
            "model_ade": round(float(pw["model"]["ade_0_2s"][sub].mean()), 4),
            "holdv0_ade": round(float(pw["holdv0"]["ade_0_2s"][sub].mean()), 4),
            "cv_ade": round(float(pw["cv"]["ade_0_2s"][sub].mean()), 4),
            "model_speed_mae": round(
                float(pw["model"]["speed_mae_mps"][sub].mean()), 4),
            "holdv0_speed_mae": round(
                float(pw["holdv0"]["speed_mae_mps"][sub].mean()), 4),
            "vs_holdv0_speed_mae_paired": paired(
                np.asarray(pw["holdv0"]["speed_mae_mps"])[m],
                np.asarray(pw["model"]["speed_mae_mps"])[m],
                [e for e, f in zip(eid, m) if f], n_boot=n_boot)}

    # ---- strata ----------------------------------------------------------- #
    masks, thr = speed_strata(v0)
    out["speed_strata_thresholds_mps"] = thr
    out["by_speed"] = {}
    for lab, m in masks.items():
        sub = m.nonzero(as_tuple=True)[0]
        sa = float(pw["model"]["long_sq_2s"][sub].mean())
        sc = float(pw["model"]["lat_sq_2s"][sub].mean())
        out["by_speed"][lab] = {
            "n": int(len(sub)),
            "model_ade": round(float(pw["model"]["ade_0_2s"][sub].mean()), 4),
            "cv_ade": round(float(pw["cv"]["ade_0_2s"][sub].mean()), 4),
            "long_rmse_2s_m": round(math.sqrt(sa), 4),
            "lat_rmse_2s_m": round(math.sqrt(sc), 4),
            "long_frac_of_2s_sqerr": round(sa / (sa + sc + EPS), 4),
            "speed_bias_mps": round(
                float(pw["model"]["speed_bias_mps"][sub].mean()), 4)}

    cb = curv_buckets(d["head_deg"])
    out["by_curvature"] = {}
    for lab in ("straight", "gentle", "sharp"):
        m = cb == lab
        if not m.any():
            continue
        sub = [i for i, f in enumerate(m) if f]
        out["by_curvature"][lab] = {
            "n": int(m.sum()),
            "model_heading_mae_deg": round(
                float(pw["model"]["heading_mae_2s_deg"][sub].mean()), 3),
            "cv_heading_mae_deg": round(
                float(pw["cv"]["heading_mae_2s_deg"][sub].mean()), 3),
            "model_curv_sign_agree": round(
                float(pw["model"]["curv_sign_agree"][sub].mean()), 4),
            "cv_curv_sign_agree": round(
                float(pw["cv"]["curv_sign_agree"][sub].mean()), 4)}

    # ---- sanity pin against MODEL_REGISTRY -------------------------------- #
    got = {"ade_0_2s": out["headline"]["ade_0_2s"]["mean"],
           "cv_ade_0_2s": out["floors"]["cv"]["ade_0_2s"],
           "long_rmse_2s_m": out["rmse"]["long_rmse_2s_m"],
           "lat_rmse_2s_m": out["rmse"]["lat_rmse_2s_m"],
           "top10pct_speed_bias_mps": out["by_speed"]["top10pct"]["speed_bias_mps"],
           "top10pct_long_rmse_2s_m": out["by_speed"]["top10pct"]["long_rmse_2s_m"]}
    out["sanity_vs_registry"] = {
        k: {"expected": SANITY[k], "got": got[k],
            "ok": bool(abs(got[k] - SANITY[k]) <= 0.005)} for k in SANITY}
    out["sanity_all_ok"] = all(v["ok"] for v in out["sanity_vs_registry"].values())
    return out


def main():
    ap = argparse.ArgumentParser("taniteval v2 tier-0 probe")
    ap.add_argument("--arm", default="flagship-30k")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    path = Path(a.results) / f"windows_{a.arm}.pt"
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Only 2 of the 10 registry arms have a window "
            f"dump in this repo; the rest live on tanitad-eval:/root/taniteval/"
            f"results (see TANITEVAL_V2_METRIC_SUITE.md §7 E1).")
    res = run(path, n_boot=a.n_boot)
    txt = json.dumps(res, indent=2, default=str)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
    if not res["sanity_all_ok"]:
        raise SystemExit("SANITY PIN FAILED vs MODEL_REGISTRY — do not publish "
                         "these numbers until the discrepancy is explained.")


if __name__ == "__main__":
    main()
