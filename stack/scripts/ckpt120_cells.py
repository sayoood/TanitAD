"""PREREG_V8F_CKPT120 reader cells (card section 6 and 7), PURE numpy over the arrays the far-output wrappers dump. Nothing here touches a model, a file or a GPU.

Cells (NaN = not scorable, never 0):
  S_CKPT120     the far checkpoint's heading is on the announced side by >= theta (25 deg; SLIGHT 10), scorable where a TURN-class side is announced at the window's tick, d_text + COMPLETE_M <= 120 m and RC-A120 is valid
                (the form of ``strategic_cells.ckpt_cell`` at arc 120; ``arc`` is a parameter so the 30 / 50 / 80 continuity rows use the SAME function).
  balanced      the mean of the left level and the right level (a constant predictor reads exactly 0.5).
  raw           real - best constant (always the majority side among the scorable windows).
  position / heading error at 120 m against RC-A120 (``taniteval.v8f_consistency.checkpoint_errors``' formula), EXTRAP (the 80 m checkpoint carried 40 m straight along its own heading) and STRAIGHT (120, 0, 0) as paired baselines.
  strata        side reaches S by the tactical token or not; near d <= 60 m or far d in (60, 100] m; 120 / v in 6-30 s or above 30 s.
  24 driving cells per seed (band x metric), the cost / gain rule with the floor.
Controls: K1 self-pair, K7 planted HOLD / announced / mirrored far points, K8 donor on an arm that copies its nav side, K9b max |delta|.
Estimator: the paired episode-cluster bootstrap over ``taniteval.ci``'s own draws (``episode_index`` + ``_draws``), B 2,000, seed 0. A statistic that is not a plain mean (balanced) uses the SAME draws, which is why this file calls ``_draws`` itself.
"""
from __future__ import annotations

import numpy as np

FAR_ARC_M = 120.0
COMPLETE_M = 20.0                                   # STRAT1's value (HYPOTHESIS there, pre-registered by the card)
THETA_DEG = {"STD": 25.0, "SLIGHT": 10.0}
EXTRAP_M = FAR_ARC_M - 80.0
MIN_W, MIN_E = 30, 10
N_BOOT, SEED = 2000, 0
STRATA_SPLITS = {"near_d_m": 60.0, "far_d_m": 100.0, "tactical_token": True, "time_s": (6.0, 30.0)}

# ---- the 24 driving cells (STRAT1 RESULT section 1): per band (operative knots 1-10, tactical 11-30) x 12 metrics
BANDS = ("op_1_10", "tac_11_30")
DRIVING_METRICS = ("ade", "fde", "speed_mae", "accel_mae", "along_track_mae", "headway_mae", "time_gap_mae", "heading_mae", "curvature_mae", "yaw_rate_mae", "cross_track_mae", "tac_acc")
DRIVING_CELLS = tuple(f"{b}:{m}" for b in BANDS for m in DRIVING_METRICS)
HIGHER_BETTER = frozenset(c for c in DRIVING_CELLS if c.endswith(":tac_acc"))     # TAC-LAT / TAC-LON accuracy; every other cell is an error (lower is better)


def _need_taniteval():
    from taniteval import ci
    return ci


# ---------------------------------------------------------------------------------------------------------------------------------- bootstrap on the shared draws
def paired_stat(a, b, eid, stat, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """CI on ``stat(a, idx) - stat(b, idx)`` with the SAME resampled episodes each draw (``taniteval.ci._draws``). ``stat(x, idx)`` maps the values ``x[idx]`` (and may read other arrays by index) to a float.
    ``separated`` = the interval excludes zero, on the unrounded bounds."""
    ci = _need_taniteval()
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape or len(eid) != a.size:
        raise ValueError(f"paired arms must be aligned: {a.shape} vs {b.shape}, eid {len(eid)}")
    uniq, idx = ci.episode_index(eid)
    allw = np.arange(a.size)
    point = float(stat(a, allw) - stat(b, allw))
    d = np.array([float(stat(a, s) - stat(b, s)) for s in ci._draws(uniq, idx, n_boot, seed)])
    lo, hi = (float(x) for x in np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)]))
    return {"delta": point, "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0), "n_windows": int(a.size), "n_episodes": int(len(uniq))}


def _mean(x, idx):
    return float(np.nanmean(x[idx]))


def paired_mean(a, b, eid, n_boot=N_BOOT, seed=SEED):
    """Paired mean difference over windows where BOTH arms are scorable; n/a below the STRATEGIC_PLAN n rule (n printed)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    e = np.asarray(eid)[ok]
    ne = int(len(np.unique(e))) if ok.any() else 0
    if ok.sum() < MIN_W or ne < MIN_E:
        return {"status": "n/a", "n_windows": int(ok.sum()), "n_episodes": ne, "reason": f"below {MIN_W} windows or {MIN_E} episodes"}
    r = paired_stat(a[ok], b[ok], e, _mean, n_boot, seed)
    r["status"] = "OK"
    return r


def balanced_stat(side):
    """The balanced statistic on a window subset: mean of the left level and the right level (NaN windows are not scorable and are dropped by the caller)."""
    side = np.asarray(side)

    def stat(x, idx):
        s, v = side[idx], x[idx]
        lv = [float(np.mean(v[s == k])) for k in (-1, 1) if (s == k).any()]
        return float(np.mean(lv)) if len(lv) == 2 else float("nan")
    return stat


def balanced_level(cell, side):
    ok = np.isfinite(cell)
    return float(balanced_stat(side[ok])(cell[ok], np.arange(int(ok.sum()))))


# ---------------------------------------------------------------------------------------------------------------------------------- the cell
def s_ckpt120(far, side, theta_deg, d_text, text_ok, rc120_ok, complete_m=COMPLETE_M, arc=FAR_ARC_M):
    """S_CKPT at ``arc``: ``far`` [N, 3] (x, y, psi rad) in the NOW frame; side / theta_deg / d_text / text_ok / rc120_ok per window. NaN where not scorable (never 0)."""
    far = np.asarray(far, np.float64)
    side, theta_deg, d_text = np.asarray(side), np.asarray(theta_deg, np.float64), np.asarray(d_text, np.float64)
    ok = np.asarray(text_ok, bool) & np.asarray(rc120_ok, bool) & (side != 0) & np.isfinite(d_text) & (d_text + complete_m <= arc) & np.isfinite(far[:, 2])
    out = np.full(far.shape[0], np.nan)
    out[ok] = (np.degrees(far[ok, 2]) * side[ok] >= theta_deg[ok]).astype(float)
    return out


def best_constant(cell, side):
    """1 where the window's side is the majority side among the scorable windows (right on a tie), else 0; NaN elsewhere. Returns (cells, majority side)."""
    ok = np.isfinite(cell)
    out = np.full(cell.shape, np.nan)
    if not ok.any():
        return out, 0
    s = side[ok]
    maj = 1 if (s == 1).sum() >= (s == -1).sum() else -1
    out[ok] = (side[ok] == maj).astype(float)
    return out, maj


def route_following(real, donor, zero, side, eid, n_boot=N_BOOT):
    """One arm at one seed: levels (real / donor / zero), the balanced levels, and the three contrasts of card 6.1: (a) real - donor (paired), (b) balanced - 0.5, (c) raw = real - best constant."""
    bc, maj = best_constant(real, side)
    blk = {"n_windows": int(np.isfinite(real).sum()), "n_episodes": int(len(np.unique(np.asarray(eid)[np.isfinite(real)]))) if np.isfinite(real).any() else 0,
           "level": {k: (float(np.nanmean(v)) if np.isfinite(v).any() else None) for k, v in (("real", real), ("donor", donor), ("zero", zero))},
           "balanced": {k: (balanced_level(v, side) if np.isfinite(v).any() else None) for k, v in (("real", real), ("donor", donor), ("zero", zero))},
           "per_side_level": {k: {str(s): (float(np.nanmean(real[(side == s) & np.isfinite(real)])) if ((side == s) & np.isfinite(real)).any() else None) for s in (-1, 1)} for k in ("real",)},
           "best_constant": {"majority_side": int(maj), "level": float(np.nanmean(bc)) if np.isfinite(bc).any() else None}}
    blk["real_minus_donor"] = paired_mean(real, donor, eid, n_boot)
    blk["real_minus_zero"] = paired_mean(real, zero, eid, n_boot)
    blk["raw_real_minus_best_constant"] = paired_mean(real, bc, eid, n_boot)
    ok = np.isfinite(real)
    blk["balanced_minus_half"] = balanced_minus_half(real[ok], side[ok], np.asarray(eid)[ok], n_boot) if ok.sum() >= MIN_W else {"status": "n/a"}
    return blk


def balanced_minus_half(cell, side, eid, n_boot=N_BOOT, seed=SEED):
    """balanced(cell) - 0.5 with the episode-cluster interval on ``taniteval.ci``'s draws (a draw with only one side present is skipped via NaN percentiles; none occurs at the card's n)."""
    ci = _need_taniteval()
    uniq, idx = ci.episode_index(eid)
    st = balanced_stat(side)
    allw = np.arange(len(cell))
    d = np.array([st(cell, s) - 0.5 for s in ci._draws(uniq, idx, n_boot, seed)])
    d = d[np.isfinite(d)]
    lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
    return {"status": "OK", "delta": float(st(cell, allw) - 0.5), "lo": lo, "hi": hi, "separated": bool(lo > 0 or hi < 0), "n_windows": int(len(cell)), "n_episodes": int(len(uniq))}


# ---------------------------------------------------------------------------------------------------------------------------------- route setting: 120 m error against RC-A120, EXTRAP, STRAIGHT
def extrap_from_80(ck):
    """EXTRAP: S's own 80 m checkpoint (``ck`` [N, 3, 3] at 30 / 50 / 80 m, NOW frame) carried EXTRAP_M straight along its own heading. [N, 3]."""
    c = np.asarray(ck, np.float64)[:, 2]
    return np.stack([c[:, 0] + EXTRAP_M * np.cos(c[:, 2]), c[:, 1] + EXTRAP_M * np.sin(c[:, 2]), c[:, 2]], -1)


def straight_far(n):
    return np.tile(np.array([FAR_ARC_M, 0.0, 0.0]), (n, 1))


def far_errors(far, rc120, rc120_ok):
    """Position error (m) and heading error (deg, wrapped) against RC-A120, NaN where the label is invalid: ``taniteval.v8f_consistency.checkpoint_errors``' formula on one checkpoint."""
    far, rc = np.asarray(far, np.float64), np.asarray(rc120, np.float64)
    pe = np.linalg.norm(far[:, :2] - rc[:, :2], axis=-1)
    dh = np.degrees(np.abs((far[:, 2] - rc[:, 2] + np.pi) % (2 * np.pi) - np.pi))
    ok = np.asarray(rc120_ok, bool)
    return {"pos_err_m": np.where(ok, pe, np.nan), "head_err_deg": np.where(ok, dh, np.nan)}


def route_setting_120(far, ck80, rc120, rc120_ok, eid, n_boot=N_BOOT):
    """Card 6.2 at 120 m: the arm's position / heading error, EXTRAP's and STRAIGHT's, and the paired cells arm - EXTRAP (and arm - STRAIGHT) with n printed. ``ck80`` None = the arm has no 80 m checkpoint to extrapolate (never silently dropped)."""
    e_arm = far_errors(far, rc120, rc120_ok)
    out = {"arm": {k: float(np.nanmean(v)) if np.isfinite(v).any() else None for k, v in e_arm.items()}, "n_windows": int(np.isfinite(e_arm["pos_err_m"]).sum())}
    for name, base in (("extrap", None if ck80 is None else extrap_from_80(ck80)), ("straight", straight_far(len(e_arm["pos_err_m"])))):
        if base is None:
            out[name] = {"status": "n/a", "reason": "no 80 m checkpoint"}
            continue
        e_b = far_errors(base, rc120, rc120_ok)
        out[name] = {"level": {k: float(np.nanmean(v)) if np.isfinite(v).any() else None for k, v in e_b.items()},
                     "arm_minus_baseline": {k: paired_mean(e_arm[k], e_b[k], eid, n_boot) for k in ("pos_err_m", "head_err_deg")}}
    return out


def strata(token_on, d_text, v, far_arc=FAR_ARC_M):
    """The fixed strata of card 6.1, as boolean masks over windows: side reaches S by the tactical token or not; near d <= 60 m or far d in (60, 100] m; 120 / v in 6-30 s or above 30 s."""
    d, v, tk = np.asarray(d_text, np.float64), np.asarray(v, np.float64), np.asarray(token_on, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(v > 0, far_arc / v, np.inf)
    lo, hi = STRATA_SPLITS["time_s"]
    return {"token_reaches_S": tk, "token_does_not": ~tk, "near_d<=60": np.isfinite(d) & (d <= 60.0), "far_d_60_100": np.isfinite(d) & (d > 60.0) & (d <= 100.0),
            "t_6_30_s": (t >= lo) & (t <= hi), "t_above_30_s": t > hi}


# ---------------------------------------------------------------------------------------------------------------------------------- driving cells: the cost / gain rule
def cost_gain(cell, delta_row, floor):
    """STRAT1's rule: a cost is separated on the worse side with |delta| > floor; a gain is separated on the better side with |delta| > floor. ``delta_row`` = the paired row (arm - reference). ``floor`` None = not available:
    the read says so and the cell is reported without a verdict (never a silent pass)."""
    if delta_row.get("status") == "n/a" or floor is None:
        return "n/a"
    worse_is_up = cell not in HIGHER_BETTER
    d = delta_row["delta"]
    if not delta_row["separated"] or abs(d) <= floor:
        return "none"
    return "cost" if (d > 0) == worse_is_up else "gain"


def driving_rows(arm, ref, eid, floors=None, n_boot=N_BOOT):
    """Every one of the 24 cells, arm - reference, paired, with the verdict. ``arm`` / ``ref`` map a cell name to a per-window array; a cell missing from either side is reported ``absent`` with its name, never dropped."""
    rows = {}
    for c in DRIVING_CELLS:
        if c not in arm or c not in ref:
            rows[c] = {"status": "absent", "reason": "the arm / reference has no per-window values for this cell"}
            continue
        r = paired_mean(arm[c], ref[c], eid, n_boot)
        r["verdict"] = cost_gain(c, r, None if floors is None else floors.get(c))
        rows[c] = r
    return rows


# ---------------------------------------------------------------------------------------------------------------------------------- controls
def k1_self_pair(cells, eid, n_boot=200):
    """K1: an arm paired with itself reads delta 0.0 and CI [0, 0] on every cell. Returns the cells that do not."""
    bad = []
    for c, v in cells.items():
        r = paired_mean(v, v, eid, n_boot)
        if r.get("status") == "OK" and not (r["delta"] == 0.0 and r["lo"] == 0.0 and r["hi"] == 0.0):
            bad.append(c)
    return bad


def planted(side, kind, deg=40.0):
    """A planted far-point array [N, 3]: ``hold`` psi = 0, ``announced`` psi = side x deg, ``mirrored`` psi = -side x deg."""
    s = np.asarray(side, np.float64)
    psi = {"hold": np.zeros_like(s), "announced": s * np.radians(deg), "mirrored": -s * np.radians(deg)}[kind]
    return np.stack([np.full_like(s, FAR_ARC_M), np.zeros_like(s), psi], -1)


def k7(side, theta_deg, d_text, text_ok, rc120_ok):
    """K7: S_CKPT120 on HOLD / announced side x 40 deg / mirrored must read 0.0 / 1.0 / 0.0 (mean over the scorable windows)."""
    return {k: float(np.nanmean(s_ckpt120(planted(side, k), side, theta_deg, d_text, text_ok, rc120_ok))) for k in ("hold", "announced", "mirrored")}


def k8(side, donor_idx, theta_deg, d_text, text_ok, rc120_ok):
    """K8: an arm that copies the announced side of ITS nav. With the real nav it reads 1.0; fed the donor's nav (``donor_idx`` per window) it must read the donors' side agreement computed from labels. Returns (real, donor, expected)."""
    side = np.asarray(side)
    real = float(np.nanmean(s_ckpt120(planted(side, "announced"), side, theta_deg, d_text, text_ok, rc120_ok)))
    cell = s_ckpt120(planted(side[np.asarray(donor_idx)], "announced"), side, theta_deg, d_text, text_ok, rc120_ok)
    ok = np.isfinite(cell)
    expected = float((side[np.asarray(donor_idx)][ok] == side[ok]).mean())
    return real, float(np.nanmean(cell)), expected


def k9b(a, b, tol=1e-6):
    """K9b: the T1 tick-0 arrays against the T0 arrays (S's 30 / 50 / 80 checkpoints and the far checkpoint): max |delta| <= 1e-6 m on every key both sides carry. A key on one side only is a failure."""
    res, ok = {}, True
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            res[k], ok = "missing on one side", False
            continue
        d = float(np.max(np.abs(np.asarray(a[k], np.float64) - np.asarray(b[k], np.float64)))) if np.asarray(a[k]).size else 0.0
        res[k] = d
        ok = ok and d <= tol
    return {"pass": ok, "max_abs_delta": res, "tol": tol}
