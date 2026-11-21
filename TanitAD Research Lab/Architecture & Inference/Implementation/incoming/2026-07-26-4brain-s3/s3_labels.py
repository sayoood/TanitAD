"""S3 — manoeuvre-initiation timing: target minter, decision-point miner, option
set and ordinal metrics.

WHAT S3 IS. *Given the scene, and given the manoeuvre has not yet begun, WHEN
does it begin?* The target is the ego's **realised** initiation time, derived
purely from FUTURE poses. It is never fed to the model.

WHY THIS FILE IS SEPARATE FROM ``stack/scripts/v4_labels.py``. The shipped
``strat_scalars.ttm`` is a **lateral-only** clock keyed to a junction-scale
curvature segment, minted over a 25 s lookahead on a corpus whose clips are
~20 s long -> it is silently RIGHT-CENSORED and its "no manoeuvre" mask
conflates *"nothing is coming"* with *"the clip ended"*. S3 needs a target that
is fully observable, so it:

  * pins a **decision horizon** ``H_S3`` and mines only windows that can OBSERVE
    all of it (rule M1) -> ``t_none`` becomes a real class, not censoring;
  * mints a **second, symmetric LONGITUDINAL clock** (``ttm_lon``) because a
    single mixed vocabulary is the documented mechanism behind 0/881 accelerate
    predictions (CLAUDE.md; spec §1). The two axes are never pooled;
  * excludes windows where the manoeuvre has ALREADY BEGUN (rules M2/M3), which
    the shipped label does not.

The lateral target itself calls the SHIPPED ``v4_labels.time_to_maneuver`` /
``refb_labels.route_from_future_v3`` (with the horizon overridden), so S3
inherits the audited label rather than forking it.

NON-CIRCULARITY. For a window with last observed pose index ``L``, every target
reads ONLY ``poses[L+1 : L+1+H]``. The observed range ``poses[:L+1]`` is never
touched. That disjointness is asserted by
``test_s3_labels.test_target_never_reads_the_observed_window``.

Estimator: intervals come from ``taniteval.ci.episode_cluster_bootstrap`` /
``paired_episode_cluster_bootstrap`` (B=2000, unit = episode). The deprecated
``overlapping_holdout_se`` is never used here.

CPU only. numpy + torch. No pod paths, no GPU.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

# --- shipped label machinery (the audited source of the lateral clock) -------
_STACK = Path(__file__).resolve()
for _p in _STACK.parents:
    if (_p / "stack" / "scripts" / "refb_labels.py").exists():
        sys.path.insert(0, str(_p / "stack" / "scripts"))
        sys.path.insert(0, str(_p / "stack"))
        sys.path.insert(0, str(_p / "taniteval"))
        break

import refb_labels                                             # noqa: E402
import v4_labels                                               # noqa: E402
from tanitad.lake.vtarget import savgol                        # noqa: E402

DT = refb_labels.DT_DEFAULT                       # 0.1 s (10 Hz contract)
WINDOW = v4_labels.WINDOW                         # 8 observed frames
MAX_HORIZON = v4_labels.MAX_HORIZON               # 20 (dataset window index)

# ---------------------------------------------------------------------------
# Pre-registered constants (PRE_REGISTRATION_S3.md §3, §4). Primary values.
# ---------------------------------------------------------------------------
H_S3_S = 12.0                    # decision horizon, seconds (primary)
MIN_TTM_S = 1.0                  # below this the manoeuvre has begun (M2)
MOVING_V_MS = refb_labels.MOVING_V_MS            # 1.0 m/s (M4)
CURV_TURN_PER_M = refb_labels.CURV_TURN_PER_M    # 1/60 m^-1 (junction scale)

# longitudinal manoeuvre definition (the symmetric axis; §2.1)
A_MAN_MS2 = 0.5                  # sustained |a| threshold
A_MIN_STEPS = 10                 # >= 1.0 s sustained
DV_MIN_MS = 1.5                  # total |dv| over the segment

# the option set: 5 ORDERED classes per axis (§4)
BAND_EDGES_S = (2.0, 5.0, 10.0)  # spec's own edges, adopted unchanged
BAND_NAMES = ("t_1_2", "t_2_5", "t_5_10", "t_10_H", "t_none")
N_BANDS = len(BAND_NAMES)
IX_NONE = N_BANDS - 1


# ===========================================================================
# targets
# ===========================================================================
def _obs_window_kinematics(poses: Tensor, L: int, window: int = WINDOW) -> dict:
    """Kinematics of the OBSERVED window ``poses[L-window+1 : L+1]``.

    This is the only place S3 looks at the observed range, and it is used for
    the miner's "has it already begun" rules (M3/M4) and for the blind
    baseline's causal features -- never for the target.
    """
    i0 = max(0, L - window + 1)
    seg = poses[i0:L + 1]
    if seg.shape[0] < 2:
        return {"v_mean": 0.0, "v_last": float(poses[L, 3]), "dv": 0.0,
                "kappa_mean": 0.0, "kappa_abs_mean": 0.0, "a_abs_mean": 0.0}
    d = seg[1:, :2] - seg[:-1, :2]
    ds = d.norm(dim=-1)
    dyaw = refb_labels.wrap_to_pi(seg[1:, 2] - seg[:-1, 2])
    kappa = torch.where(ds >= refb_labels.MIN_ARC_M,
                        dyaw / ds.clamp_min(refb_labels.MIN_ARC_M),
                        torch.zeros_like(ds))
    v = seg[:, 3]
    a = (v[1:] - v[:-1]) / DT
    return {"v_mean": float(v.mean()), "v_last": float(v[-1]),
            "dv": float(v[-1] - v[0]),
            "kappa_mean": float(kappa.mean()),
            "kappa_abs_mean": float(kappa.abs().mean()),
            "a_abs_mean": float(a.abs().mean())}


def ttm_lateral(poses: Tensor, L: int, horizon_s: float = H_S3_S
                ) -> tuple[float, bool]:
    """Seconds to the start of the first junction-scale curvature segment in
    ``poses[L+1 : L+1+horizon]``.

    Delegates to the SHIPPED minters with the horizon overridden, so the target
    is the audited ``strat_scalars.ttm`` restricted to an observable horizon.
    ``(t, True)`` if an event exists in range, else ``(nan, False)``.
    """
    h = int(round(horizon_s / DT))
    r3 = refb_labels.route_from_future_v3(poses, L, horizon_steps=h)
    t, ok = v4_labels.time_to_maneuver(poses, L, r3, horizon=h)
    return (float(t), True) if ok else (float("nan"), False)


def _lon_segments(a: np.ndarray, dv_cum: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous same-sign runs of ``|a| >= A_MAN_MS2`` lasting
    ``>= A_MIN_STEPS`` whose total ``|dv| >= DV_MIN_MS``."""
    hot = np.abs(a) >= A_MAN_MS2
    sign = np.sign(a)
    out, i, n = [], 0, a.shape[0]
    while i < n:
        if not hot[i]:
            i += 1
            continue
        s, j = sign[i], i + 1
        while j < n and hot[j] and sign[j] == s:
            j += 1
        if (j - i) >= A_MIN_STEPS and abs(dv_cum[j - 1] - (dv_cum[i - 1] if i else 0.0)) >= DV_MIN_MS:
            out.append((i, j))
        i = j
    return out


def ttm_longitudinal(poses: Tensor, L: int, horizon_s: float = H_S3_S,
                     v_smoothed: np.ndarray | None = None
                     ) -> tuple[float, bool]:
    """Seconds to the start of the first SUSTAINED longitudinal segment in
    ``poses[L+1 : L+1+horizon]`` -- the longitudinal twin of :func:`ttm_lateral`.

    A longitudinal manoeuvre = a same-sign run of smoothed ``|a| >= 0.5 m/s^2``
    lasting ``>= 1.0 s`` with total ``|dv| >= 1.5 m/s``. Thresholds sit inside
    the existing vocabulary (``DV_ACCEL_MS``/``DV_BRAKE_MS`` = +/-1.0 m/s over
    2 s => +/-0.5 m/s^2, ``refb_labels:58-59``).
    """
    T = poses.shape[0]
    h = min(int(round(horizon_s / DT)), T - 1 - L)
    if h < A_MIN_STEPS + 1:
        return float("nan"), False
    if v_smoothed is None:
        v_smoothed = savgol(poses[:, 3].numpy().astype(np.float64))
    v = np.asarray(v_smoothed[L:L + h + 1], dtype=np.float64)
    a = np.diff(v) / DT                                     # [h]
    dv_cum = np.cumsum(np.diff(v))
    segs = _lon_segments(a, dv_cum)
    if not segs:
        return float("nan"), False
    return (segs[0][0] + 1) * DT, True


# ---------------------------------------------------------------------------
# "has the manoeuvre already begun?" (miner rule M3)
# ---------------------------------------------------------------------------
# ⚠️ INSTRUMENT CORRECTION, 2026-07-26, made BEFORE any arm was scored and
# recorded in S3_IMPLEMENTATION.md §3.2. The first implementation tested M3 with
# an INSTANTANEOUS threshold on RAW per-step acceleration, while the TARGET is
# defined on a SUSTAINED segment of SAVGOL-SMOOTHED speed. That mismatch made
# M3_lon fire on ordinary speed noise: it admitted only windows sitting inside a
# steady-speed lull, so the next event was ALWAYS imminent and 3 of the 5
# longitudinal classes came out EMPTY (MEASURED on the 6-episode smoke:
# t_5_10 / t_10_H / t_none all 0). M3 now runs the SAME segment detector as the
# target, backwards from L, on the SAME smoothed signal.
M3_BACK_STEPS = 30                # 3 s of observed track to test for "in progress"
M3_LAT_MIN_DYAW_RAD = math.radians(5.0)   # a turn that JUST began has swept little
M3_LON_MIN_STEPS = 5                      # 0.5 s of sustained accel = begun


def lat_in_progress(poses: Tensor, L: int, back_steps: int = M3_BACK_STEPS) -> bool:
    """True iff a junction-scale curvature segment is still running at ``L``."""
    i0 = max(0, L - back_steps)
    if L - i0 < 4:
        return False
    tr = refb_labels._future_track(poses, i0, L - i0)
    if tr["h"] < 4:
        return False
    n = int(tr["ks"].shape[0])
    for a, b, _s in refb_labels._curv_segments(tr["ks"], CURV_TURN_PER_M):
        if b >= n and abs(float(tr["step_dyaw"][a:b].sum())) >= M3_LAT_MIN_DYAW_RAD:
            return True
    return False


def lon_in_progress(poses: Tensor, L: int, v_smoothed: np.ndarray,
                    back_steps: int = M3_BACK_STEPS) -> bool:
    """True iff a sustained longitudinal segment is still running at ``L``.

    Same signal (savgol speed) and same amplitude threshold as the target; the
    length gate is shorter (0.5 s) because a manoeuvre that began half a second
    ago has still begun, and the ``|dv|`` gate is dropped for the same reason.
    """
    i0 = max(0, L - back_steps)
    v = np.asarray(v_smoothed[i0:L + 1], dtype=np.float64)
    if v.size < M3_LON_MIN_STEPS + 1:
        return False
    a = np.diff(v) / DT
    hot = np.abs(a) >= A_MAN_MS2
    sign = np.sign(a)
    n, i = a.shape[0], 0
    while i < n:
        if not hot[i]:
            i += 1
            continue
        s, j = sign[i], i + 1
        while j < n and hot[j] and sign[j] == s:
            j += 1
        if j >= n and (j - i) >= M3_LON_MIN_STEPS:
            return True
        i = j
    return False


def band_of(t_s: float, ok: bool, edges=BAND_EDGES_S, horizon_s=H_S3_S) -> int:
    """Ordinal class index for an initiation time. ``ok=False`` -> ``t_none``."""
    if not ok or not np.isfinite(t_s):
        return IX_NONE
    if t_s >= horizon_s:
        return IX_NONE
    for i, e in enumerate(edges):
        if t_s < e:
            return i
    return len(edges)                                        # t_10_H


# ===========================================================================
# the miner
# ===========================================================================
def mine_episode(poses: Tensor, eid: str, horizon_s: float = H_S3_S,
                 min_ttm_s: float = MIN_TTM_S, window: int = WINDOW,
                 max_horizon: int = MAX_HORIZON) -> list[dict]:
    """Every admissible S3 decision point in one episode.

    Rules (PRE_REGISTRATION_S3.md §3), applied per axis:
      M1 full decision horizon observable   (T-1-L >= H/dt)
      M2 manoeuvre not begun                (ttm >= min_ttm_s)
      M3 observed window not already executing the axis's manoeuvre
      M4 ego moving                         (in-window mean v >= 1.0 m/s)

    ``M1``/``M4`` gate the WINDOW; ``M2``/``M3`` gate each AXIS independently
    (a window can be an admissible lateral decision point and an inadmissible
    longitudinal one). Rejections are counted, not silently dropped.
    """
    T = int(poses.shape[0])
    h_steps = int(round(horizon_s / DT))
    n_win = T - window - max_horizon                 # dataset window index
    rows: list[dict] = []
    if n_win <= 0:
        return rows
    vs = savgol(poses[:, 3].numpy().astype(np.float64))
    for t in range(n_win):
        L = t + window - 1
        obs_h = T - 1 - L
        rec = {"eid": eid, "t": t, "L": L, "obs_h_steps": obs_h,
               "m1": bool(obs_h >= h_steps)}
        k = _obs_window_kinematics(poses, L, window)
        rec.update({f"win_{a}": b for a, b in k.items()})
        rec["m4"] = bool(k["v_mean"] >= MOVING_V_MS)
        if not (rec["m1"] and rec["m4"]):
            rec.update(lat_admissible=False, lon_admissible=False)
            rows.append(rec)
            continue
        # ---- lateral axis -------------------------------------------------
        t_lat, ok_lat = ttm_lateral(poses, L, horizon_s)
        m2_lat = (not ok_lat) or (t_lat >= min_ttm_s)         # t_none passes M2
        m3_lat = not lat_in_progress(poses, L)
        rec.update(ttm_lat=t_lat, ttm_lat_ok=ok_lat, m2_lat=bool(m2_lat),
                   m3_lat=bool(m3_lat),
                   m3_lat_instantaneous=bool(
                       k["kappa_abs_mean"] < CURV_TURN_PER_M),
                   lat_admissible=bool(m2_lat and m3_lat),
                   band_lat=band_of(t_lat, ok_lat, horizon_s=horizon_s))
        # ---- longitudinal axis --------------------------------------------
        t_lon, ok_lon = ttm_longitudinal(poses, L, horizon_s, v_smoothed=vs)
        m2_lon = (not ok_lon) or (t_lon >= min_ttm_s)
        m3_lon = not lon_in_progress(poses, L, vs)
        rec.update(ttm_lon=t_lon, ttm_lon_ok=ok_lon, m2_lon=bool(m2_lon),
                   m3_lon=bool(m3_lon),
                   m3_lon_instantaneous=bool(k["a_abs_mean"] < A_MAN_MS2),
                   lon_admissible=bool(m2_lon and m3_lon),
                   band_lon=band_of(t_lon, ok_lon, horizon_s=horizon_s))
        rows.append(rec)
    return rows


# ===========================================================================
# ordinal metrics -- NEVER bare accuracy (PRE_REGISTRATION_S3.md §5)
# ===========================================================================
def quadratic_weighted_kappa(y_true, y_pred, n_classes: int = N_BANDS) -> float:
    """Quadratic-weighted Cohen's kappa over ORDERED classes.

    Chance-corrected: a constant (majority-class) predictor scores EXACTLY 0.0,
    which is the arithmetic reason ``route_acc = 1.0`` beside
    ``route_skill = 0.0`` cannot happen here. Ordinal: a 2-band miss costs 4x a
    1-band miss.
    """
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    if yt.size == 0:
        return float("nan")
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(O, (yt, yp), 1.0)
    w = (np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :]) ** 2
    w = w / max(1.0, (n_classes - 1) ** 2)
    ht = np.bincount(yt, minlength=n_classes).astype(np.float64)
    hp = np.bincount(yp, minlength=n_classes).astype(np.float64)
    E = np.outer(ht, hp)
    E *= O.sum() / max(E.sum(), 1e-12)
    den = float((w * E).sum())
    if den <= 1e-12:                     # degenerate (single class present)
        return float("nan")
    return float(1.0 - (w * O).sum() / den)


def per_band_recall(y_true, y_pred, n_classes: int = N_BANDS) -> dict:
    """Recall per ordered class. THE metric that catches a dead band -- the
    0/881-accelerate failure is exactly a class with recall 0.000."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    out = {}
    for c in range(n_classes):
        m = yt == c
        out[BAND_NAMES[c]] = (round(float((yp[m] == c).mean()), 4)
                              if m.any() else None)
    return out


def band_metrics(y_true, y_pred, n_classes: int = N_BANDS) -> dict:
    """The full reporting triple: chance-corrected + raw + baseline."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    counts = np.bincount(yt, minlength=n_classes)
    maj = int(counts.argmax())
    return {
        "qwk": round(quadratic_weighted_kappa(yt, yp, n_classes), 4),
        "band_acc": round(float((yt == yp).mean()), 4),
        "off_by_le1_acc": round(float((np.abs(yt - yp) <= 1).mean()), 4),
        "majority_class": BAND_NAMES[maj],
        "majority_rate": round(float(counts[maj] / max(1, counts.sum())), 4),
        "qwk_majority_baseline": round(
            quadratic_weighted_kappa(yt, np.full_like(yt, maj), n_classes), 4)
        if not np.isnan(quadratic_weighted_kappa(
            yt, np.full_like(yt, maj), n_classes)) else 0.0,
        "per_band_recall": per_band_recall(yt, yp, n_classes),
        "mean_signed_band_err": round(float((yp - yt).mean()), 4),
        "n": int(yt.size),
    }


def mae_skill_s(t_true, t_pred, median_const: float | None = None) -> dict:
    """``MAE(median-constant) - MAE(model)`` in seconds over EVENT windows.

    The MAE-optimal constant is the median, so this is the honest "did you beat
    always-guess-the-middle". Positive = better.
    """
    tt = np.asarray(t_true, dtype=np.float64)
    tp = np.asarray(t_pred, dtype=np.float64)
    m = np.isfinite(tt) & np.isfinite(tp)
    tt, tp = tt[m], tp[m]
    if tt.size == 0:
        return {"mae_s": float("nan"), "mae_median_baseline_s": float("nan"),
                "mae_skill_s": float("nan"), "early_late_bias_s": float("nan"),
                "n": 0}
    med = float(np.median(tt)) if median_const is None else float(median_const)
    mae = float(np.abs(tp - tt).mean())
    mae0 = float(np.abs(med - tt).mean())
    return {"mae_s": round(mae, 4), "mae_median_baseline_s": round(mae0, 4),
            "mae_skill_s": round(mae0 - mae, 4),
            "early_late_bias_s": round(float((tp - tt).mean()), 4),
            "median_const_s": round(med, 4), "n": int(tt.size)}


# --------------------------------------------------------------------------
# bootstrap wrappers -- named estimator on every interval
# --------------------------------------------------------------------------
def qwk_bootstrap(y_true, y_pred, eid, n_boot: int = 2000, seed: int = 0,
                  n_classes: int = N_BANDS) -> dict:
    """Episode-cluster bootstrap CI on QWK.

    QWK is not a per-window mean, so it goes through ``ci``'s CALLABLE-reducer
    path on resampled window INDICES -- the same estimator every other interval
    in the program uses. ``overlapping_holdout_se`` is never used.
    """
    from taniteval.ci import episode_cluster_bootstrap
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    def _red(idx):
        i = np.asarray(idx, dtype=np.int64)
        return quadratic_weighted_kappa(yt[i], yp[i], n_classes)
    _red.__name__ = "qwk"
    return episode_cluster_bootstrap(np.arange(yt.size, dtype=np.float64),
                                     eid, reduce=_red, n_boot=n_boot, seed=seed)


def paired_qwk_bootstrap(y_true, pred_a, pred_b, eid, n_boot: int = 2000,
                         seed: int = 0, n_classes: int = N_BANDS) -> dict:
    """Paired episode-cluster bootstrap on ``QWK(a) - QWK(b)``, same windows,
    same resampled episodes each draw. Never a quadrature combination."""
    from taniteval.ci import paired_episode_cluster_bootstrap
    yt = np.asarray(y_true, dtype=np.int64)
    pa = np.asarray(pred_a, dtype=np.int64)
    pb = np.asarray(pred_b, dtype=np.int64)
    n = yt.size
    # ci's paired helper applies ONE reducer to arm A and arm B under the SAME
    # resampled episodes. Arm A carries indices [0, n), arm B carries [n, 2n);
    # the reducer decodes which arm it was handed from the offset. This keeps
    # the pairing (shared per-window difficulty cancels inside each draw)
    # without reimplementing the estimator.
    idx = np.arange(n, dtype=np.float64)

    def _red(v):
        i = np.asarray(v, dtype=np.int64)
        if i.size and i[0] >= n:
            return quadratic_weighted_kappa(yt[i - n], pb[i - n], n_classes)
        return quadratic_weighted_kappa(yt[i], pa[i], n_classes)
    _red.__name__ = "qwk"
    return paired_episode_cluster_bootstrap(idx, idx + n, eid, n_boot=n_boot,
                                            seed=seed, reduce=_red)


__all__ = ["H_S3_S", "MIN_TTM_S", "BAND_EDGES_S", "BAND_NAMES", "N_BANDS",
           "IX_NONE", "A_MAN_MS2", "ttm_lateral", "ttm_longitudinal", "band_of",
           "lat_in_progress", "lon_in_progress",
           "mine_episode", "quadratic_weighted_kappa", "per_band_recall",
           "band_metrics", "mae_skill_s", "qwk_bootstrap",
           "paired_qwk_bootstrap", "_obs_window_kinematics"]
