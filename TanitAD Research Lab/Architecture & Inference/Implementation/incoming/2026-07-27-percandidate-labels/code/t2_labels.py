"""T2 ⭐ — mint PER-CANDIDATE rule labels and test the PRE-REGISTERED KILL.

THE POINT OF THE LADDER. Every strong published scorer (Hydra-MDP/++, WoTE,
GTRS, GoalFlow, PDM, PLUTO) is trained against a rule-computed, deterministic
per-candidate verdict — never against distance-to-the-one-realised-future.
Before spending a GPU-minute on such a head we ask, at zero GPU, whether the
verdict DISCRIMINATES on OUR corpus.

⛔ PRE-REGISTERED KILL, taken verbatim from the research that proposed it:
   "if the NC/TTC labels have < 5 % positive rate AND their Spearman rho with
    fan_err is < 0.15, R1 is REFUTED on this corpus."
Honoured exactly. Both legs must fail for the kill to fire.

WHAT IS BUILT (and what is not)
  NC   at-fault collision   <- obstacle.offline log-replay tracks. BUILDABLE.
  TTC  time-to-collision    <- same + ego kinematics.             BUILDABLE.
  C    comfort              <- candidate geometry only.           BUILDABLE.
  EP   ego progress         <- along-heading proxy (no route).    PARTIAL.
  DAC  drivable area        <- needs an HD map.            ⛔ NOT BUILDABLE:
       PhysicalAI-AV ships no map (settled at five probes; the card says
       verbatim "we do not include open maps data"). Reported as absent, not
       silently dropped — DAC is one of PDMS's two multiplicative terms.

AGENT FUTURES: log replay. The tracks ARE the future (exactly how NAVSIM's
NC/TTC teachers work). No prediction model is built — PLUTO measured that a
learned predictor is worth only +0.75 over constant velocity FOR SCORING.

⚠️ CORPUS: dev-box cache + PhysicalAI-AV chunk 0000. NOT the parity corpus.
This measures a property of LABELS, not of any model, so parity is untouched.
🔒 No clip UUID or raw content reaches any artifact — aliases only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcl_common import (DT, N_STEPS, WP_STEPS, ego_at, load_scenes,  # noqa: E402
                        obb_corners, obb_overlap, quaternion_yaw)

# --- comfort bounds ---------------------------------------------------------
# `PUBLISHED (nuPlan/NAVSIM comfort bounds) — INHERITED, not re-verified from a
#  PDF in this session.` The DISCRIMINATION verdict must not rest on them, so
#  the continuous distributions are reported alongside every binary rate and a
#  threshold sweep is emitted.
A_LON_MAX, A_LON_MIN = 4.89, -4.05
A_LAT_MAX = 4.89
JERK_MAX = 8.37
YAW_RATE_MAX = 0.95

# ⚠️ Chosen from PHYSICS and the GT distribution ONLY — never from held-out
# error. Both are CLI-swept and the kill verdict is reported at BOTH operating
# points, so no conclusion rests on a threshold choice.
EGO_RAD_PAD = 0.35          # collision inflation [m]
TTC_HORIZON_S = 1.0         # constant-velocity projection horizon [s]


# --------------------------------------------------------------------------- #
def candidate_kinematics(fan: np.ndarray, v0: np.ndarray):
    """fan [C, T, 2] ego-frame dense waypoints -> heading, speed, accel, jerk.

    Uses the corpus's OWN action convention (tanitad.data.physicalai /
    v5 §0.3): p = [(0,0), w_1..w_T]; v_j = |dp_j|/dt with v_0 := observed v0.
    """
    C, T, _ = fan.shape
    p = np.concatenate([np.zeros((C, 1, 2)), fan], axis=1)        # [C, T+1, 2]
    d = np.diff(p, axis=1)                                        # [C, T, 2]
    seg = np.linalg.norm(d, axis=-1)
    v = seg / DT                                                  # [C, T]
    psi = np.arctan2(d[..., 1], d[..., 0])
    psi = np.where(seg < 1e-3, 0.0, psi)
    psi = np.unwrap(np.concatenate([np.zeros((C, 1)), psi], axis=1), axis=1)[:, 1:]
    vfull = np.concatenate([np.full((C, 1), float(v0)), v], axis=1)
    a_lon = np.diff(vfull, axis=1) / DT                           # [C, T]
    yaw_rate = np.concatenate([psi[:, :1] / DT,
                               np.diff(psi, axis=1) / DT], axis=1)
    a_lat = v * yaw_rate
    jerk = np.concatenate([np.zeros((C, 1)), np.diff(a_lon, axis=1) / DT], axis=1)
    return psi, v, a_lon, a_lat, jerk, yaw_rate


def track_table(scene, t0_us: float, times_us: np.ndarray, ego0: np.ndarray):
    """Log-replay agent boxes in the WINDOW's ego frame at each future step.

    Returns (xy [T, A, 2], yaw [T, A], size [A, 2], valid [T, A]) or None.
    """
    o = scene.obst
    t = o["timestamp_us"].to_numpy(np.float64)
    lo, hi = times_us.min() - 3e5, times_us.max() + 3e5
    m = (t >= lo) & (t <= hi)
    if m.sum() == 0:
        return None
    t, o = t[m], o[m]
    rig = np.stack([o["center_x"].to_numpy(np.float64),
                    o["center_y"].to_numpy(np.float64)], axis=1)
    ayaw_rig = quaternion_yaw(*(o[c].to_numpy(np.float64) for c in
                                ("orientation_x", "orientation_y",
                                 "orientation_z", "orientation_w")))
    pose = ego_at(scene.ego, t)
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    wx = rig[:, 0] * c - rig[:, 1] * s + pose[:, 0]
    wy = rig[:, 0] * s + rig[:, 1] * c + pose[:, 1]
    wyaw = ayaw_rig + pose[:, 2]
    sx = o["size_x"].to_numpy(np.float64)
    sy = o["size_y"].to_numpy(np.float64)
    tid = o["track_id"].to_numpy()

    uniq, inv = np.unique(tid, return_inverse=True)
    A, T = len(uniq), len(times_us)
    XY = np.zeros((T, A, 2)); YAW = np.zeros((T, A))
    SZ = np.zeros((A, 2)); VAL = np.zeros((T, A), bool)
    for k in range(A):
        sel = inv == k
        tk = t[sel]
        if len(tk) < 2:
            continue
        order = np.argsort(tk)
        tk = tk[order]
        XY[:, k, 0] = np.interp(times_us, tk, wx[sel][order])
        XY[:, k, 1] = np.interp(times_us, tk, wy[sel][order])
        yk = np.unwrap(wyaw[sel][order])
        YAW[:, k] = np.interp(times_us, tk, yk)
        SZ[k] = [np.median(sx[sel]), np.median(sy[sel])]
        VAL[:, k] = (times_us >= tk.min() - 1.5e5) & (times_us <= tk.max() + 1.5e5)
    # world -> window ego frame
    d = XY - ego0[:2]
    cc, ss = np.cos(-ego0[2]), np.sin(-ego0[2])
    XY = np.stack([d[..., 0] * cc - d[..., 1] * ss,
                   d[..., 0] * ss + d[..., 1] * cc], axis=-1)
    YAW = YAW - ego0[2]
    return XY, YAW, SZ, VAL


def collide(fan, psi, XY, YAW, SZ, VAL, ego_len, ego_wid):
    """[C] any-collision, [C] at-fault(front-sector), [C] min surface gap [m]."""
    C, T, _ = fan.shape
    A = XY.shape[1]
    er = 0.5 * np.hypot(ego_len, ego_wid) + EGO_RAD_PAD
    ar = 0.5 * np.hypot(SZ[:, 0], SZ[:, 1])                       # [A]
    d = np.linalg.norm(fan[:, :, None, :] - XY[None], axis=-1)    # [C, T, A]
    thr = er + ar[None, None, :]
    near = (d < thr) & VAL[None]
    gap = np.where(VAL[None], d - thr, np.inf).min(axis=(1, 2))   # [C]
    hit = np.zeros(C, bool); fault = np.zeros(C, bool)
    ci, ti, ai = np.nonzero(near)
    if len(ci):
        ec = obb_corners(fan[ci, ti, 0], fan[ci, ti, 1], psi[ci, ti],
                         np.full(len(ci), ego_len + 2 * EGO_RAD_PAD),
                         np.full(len(ci), ego_wid + 2 * EGO_RAD_PAD))
        ac = obb_corners(XY[ti, ai, 0], XY[ti, ai, 1], YAW[ti, ai],
                         SZ[ai, 0], SZ[ai, 1])
        ov = obb_overlap(ec, ac)
        # at-fault proxy: the agent is AHEAD of the ego's rear axle in the ego's
        # own instantaneous frame at the moment of contact (a rear-end from
        # behind is not our infraction). Simplification, stated not hidden.
        rel = XY[ti, ai] - fan[ci, ti]
        cph, sph = np.cos(psi[ci, ti]), np.sin(psi[ci, ti])
        ahead = (rel[:, 0] * cph + rel[:, 1] * sph) > -0.5 * ego_len
        np.logical_or.at(hit, ci[ov], True)
        np.logical_or.at(fault, ci[ov & ahead], True)
    return hit, fault, gap


def _footprint_hit(ex, ey, epsi, XY_, YAW_, SZ, VAL, ego_len, ego_wid):
    """Exact OBB overlap with a circular pre-filter. All ego args [C, T]."""
    C, T = ex.shape
    er = 0.5 * np.hypot(ego_len, ego_wid) + EGO_RAD_PAD
    ar = 0.5 * np.hypot(SZ[:, 0], SZ[:, 1])
    d = np.hypot(ex[:, :, None] - XY_[None, ..., 0],
                 ey[:, :, None] - XY_[None, ..., 1])
    near = (d < (er + ar[None, None, :])) & VAL[None]
    hit = np.zeros(C, bool)
    ci, ti, ai = np.nonzero(near)
    if not len(ci):
        return hit, ci, ti, ai, np.zeros(0, bool)
    ec = obb_corners(ex[ci, ti], ey[ci, ti], epsi[ci, ti],
                     np.full(len(ci), ego_len + 2 * EGO_RAD_PAD),
                     np.full(len(ci), ego_wid + 2 * EGO_RAD_PAD))
    ac = obb_corners(XY_[ti, ai, 0], XY_[ti, ai, 1], YAW_[ti, ai],
                     SZ[ai, 0], SZ[ai, 1])
    ov = obb_overlap(ec, ac)
    np.logical_or.at(hit, ci[ov], True)
    return hit, ci, ti, ai, ov


def ttc(fan, psi, v, XY, YAW, SZ, VAL, ego_len, ego_wid):
    """NAVSIM-shaped TTC: from every step, project the ego AND the agents at
    CONSTANT VELOCITY for TTC_HORIZON_S and test the real footprints.
    Returns [C] infraction flag, [C] earliest tau at which it occurs."""
    C, T, _ = fan.shape
    dt_probe = 0.2
    agv = np.zeros_like(XY)
    if T > 1:
        agv[1:] = (XY[1:] - XY[:-1]) / DT
        agv[0] = agv[1]
    flag = np.zeros(C, bool)
    best = np.full(C, np.inf)
    for k in range(1, int(round(TTC_HORIZON_S / dt_probe)) + 1):
        tau = k * dt_probe
        ex = fan[..., 0] + v * np.cos(psi) * tau
        ey = fan[..., 1] + v * np.sin(psi) * tau
        XY_ = XY + agv * tau
        h, _, _, _, _ = _footprint_hit(ex, ey, psi, XY_, YAW, SZ, VAL,
                                       ego_len, ego_wid)
        best = np.where(h & np.isinf(best), tau, best)
        flag |= h
    return flag, best


def main(argv=None):
    global EGO_RAD_PAD, TTC_HORIZON_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--pad", type=float, default=0.35)
    ap.add_argument("--ttc-horizon", type=float, default=1.0)
    ap.add_argument("--clips", type=int, default=96)
    ap.add_argument("--stride-s", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", required=True)
    a = ap.parse_args(argv)
    EGO_RAD_PAD, TTC_HORIZON_S = a.pad, a.ttc_horizon

    anc = torch.load(a.anchors, map_location="cpu", weights_only=False)["anchors"]
    FAN = anc.numpy().astype(np.float64)                          # [C, 20, 2]
    C = FAN.shape[0]
    wp_idx = np.array(WP_STEPS) - 1

    rows = []
    scenes = load_scenes(limit=a.clips)
    for sc in scenes:
        et = sc.ego["timestamp"].to_numpy(np.float64)
        t_lo, t_hi = max(0.0, et.min()), min(20e6, et.max())
        starts = np.arange(t_lo + 5e5, t_hi - 2.1e6, a.stride_s * 1e6)
        for t0 in starts:
            times = t0 + (np.arange(1, N_STEPS + 1) * DT) * 1e6
            ego0 = ego_at(sc.ego, np.array([t0]))[0]
            v0 = float(ego0[3])
            fut = ego_at(sc.ego, times)
            d = fut[:, :2] - ego0[:2]
            cc, ss = np.cos(-ego0[2]), np.sin(-ego0[2])
            gt = np.stack([d[:, 0] * cc - d[:, 1] * ss,
                           d[:, 0] * ss + d[:, 1] * cc], axis=-1)   # [20, 2]
            fan_err = np.linalg.norm(FAN[:, wp_idx] - gt[None, wp_idx],
                                     axis=-1).mean(1)               # [C]
            psi, v, a_lon, a_lat, jerk, yr = candidate_kinematics(FAN, v0)

            # ⭐ INSTRUMENT VALIDATION, the other direction: label the REALISED
            # human future with the identical labeler. PARA-Drive measured that
            # GT trajectories "collide" at 0.384 % under a sloppy metric — if
            # OUR labeler flags the human at a high rate it is measuring its own
            # padding, not driving. Reported, never tuned away.
            gpsi, gv, ga_lon, ga_lat, gjerk, gyr = candidate_kinematics(
                gt[None], v0)

            tt = track_table(sc, t0, times, ego0)
            if tt is None:
                XY = np.zeros((N_STEPS, 0, 2)); YAW = np.zeros((N_STEPS, 0))
                SZ = np.zeros((0, 2)); VAL = np.zeros((N_STEPS, 0), bool)
            else:
                XY, YAW, SZ, VAL = tt
            n_ag = int(VAL.any(axis=0).sum())
            if XY.shape[1]:
                hit, fault, gap = collide(FAN, psi, XY, YAW, SZ, VAL,
                                          sc.ego_len, sc.ego_wid)
                tflag, tmin = ttc(FAN, psi, v, XY, YAW, SZ, VAL,
                                  sc.ego_len, sc.ego_wid)
                ghit, gfault, _ = collide(gt[None], gpsi, XY, YAW, SZ, VAL,
                                          sc.ego_len, sc.ego_wid)
                gtf, _ = ttc(gt[None], gpsi, gv, XY, YAW, SZ, VAL,
                             sc.ego_len, sc.ego_wid)
            else:
                hit = fault = tflag = np.zeros(C, bool)
                gap = np.full(C, np.inf); tmin = np.full(C, np.inf)
                ghit = gfault = gtf = np.zeros(1, bool)
            gcomf = bool((ga_lon.max() <= A_LON_MAX) and
                         (ga_lon.min() >= A_LON_MIN) and
                         (np.abs(ga_lat).max() <= A_LAT_MAX) and
                         (np.abs(gjerk).max() <= JERK_MAX) and
                         (np.abs(gyr).max() <= YAW_RATE_MAX))

            comf = ((a_lon.max(1) <= A_LON_MAX) & (a_lon.min(1) >= A_LON_MIN) &
                    (np.abs(a_lat).max(1) <= A_LAT_MAX) &
                    (np.abs(jerk).max(1) <= JERK_MAX) &
                    (np.abs(yr).max(1) <= YAW_RATE_MAX))
            prog = FAN[:, -1, 0]
            rows.append(dict(
                alias=sc.alias, t0_s=float(t0 / 1e6), v0=v0, n_agents=n_ag,
                fan_err=fan_err, nc_any=hit, nc_fault=fault, gap=gap,
                ttc_flag=tflag, ttc_min=tmin, comfort_ok=comf,
                progress=prog, a_lon_max=a_lon.max(1), a_lon_min=a_lon.min(1),
                a_lat_absmax=np.abs(a_lat).max(1),
                jerk_absmax=np.abs(jerk).max(1), yr_absmax=np.abs(yr).max(1),
                gt_nc_fault=bool(gfault[0]), gt_nc_any=bool(ghit[0]),
                gt_ttc=bool(gtf[0]), gt_comfort_ok=gcomf))

    # ---------------------------------------------------------------- stats --
    def st(key):
        return np.stack([r[key] for r in rows])                   # [W, C]
    W = len(rows)
    FE, NCA, NCF = st("fan_err"), st("nc_any"), st("nc_fault")
    TF, CO, PR = st("ttc_flag"), st("comfort_ok"), st("progress")
    GAP, TMIN = st("gap"), st("ttc_min")
    nag = np.array([r["n_agents"] for r in rows])

    def spearman_rows(lab):
        """mean/median per-window Spearman rho(label, fan_err), non-degenerate
        windows only; plus the pooled rho over all (window, candidate) pairs."""
        from scipy.stats import rankdata
        rs = []
        for i in range(W):
            x = lab[i].astype(float)
            if x.std() == 0:
                continue
            rx, ry = rankdata(x), rankdata(FE[i])
            rs.append(float(np.corrcoef(rx, ry)[0, 1]))
        pooled = float(np.corrcoef(rankdata(lab.ravel().astype(float)),
                                   rankdata(FE.ravel()))[0, 1])
        return (dict(n_nondegenerate_windows=len(rs),
                     frac_nondegenerate=round(len(rs) / W, 4),
                     mean_rho=round(float(np.mean(rs)), 4) if rs else None,
                     median_rho=round(float(np.median(rs)), 4) if rs else None,
                     pooled_rho=round(pooled, 4)))

    def binrate(lab, name):
        return dict(
            label=name,
            positive_rate=round(float(lab.mean()), 5),
            frac_windows_with_any_positive=round(float(lab.any(1).mean()), 4),
            frac_windows_label_varies=round(
                float((lab.any(1) & ~lab.all(1)).mean()), 4),
            mean_positives_per_window=round(float(lab.sum(1).mean()), 2),
            rho_vs_fan_err=spearman_rows(lab))

    nc = binrate(NCF, "NC (at-fault collision)")
    nc_any = binrate(NCA, "collision (any, incl. rear-end)")
    tt_ = binrate(TF, "TTC infraction (<=1 s const-vel projection)")
    cf = binrate(~CO, "comfort VIOLATION")

    kill_nc = (nc["positive_rate"] < 0.05 and
               abs(nc["rho_vs_fan_err"]["pooled_rho"]) < 0.15)
    kill_tt = (tt_["positive_rate"] < 0.05 and
               abs(tt_["rho_vs_fan_err"]["pooled_rho"]) < 0.15)

    # ------------------------------------------------- the composite metric --
    # PDM's own term list and weights (PDF-verbatim in CITATIONS.md #11):
    # NC and DAC multiplicative; EP 5, TTC 5, Comfort 2 in the weighted sum.
    # DAC is DROPPED because PhysicalAI-AV has no map — so this is PDMS-lite
    # (no-map) and it is NOT comparable to a published PDMS number.
    pmax = np.maximum(PR.max(1, keepdims=True), 1e-6)
    EP = np.clip(PR / pmax, 0.0, 1.0)
    EP = np.where(PR <= 0.0, 0.0, EP)                       # no-progress veto
    PDMS = (~NCF) * (5.0 * EP + 5.0 * (~TF) + 2.0 * CO) / 12.0

    rng = np.random.default_rng(0)
    arms = dict(
        rule_pdms_lite_argmax=PDMS.argmax(1),
        rule_nc_ttc_only=((~NCF) * (5.0 * (~TF) + 2.0 * CO)).argmax(1),
        random=rng.integers(0, C, size=W),
        oracle_in_fan=FE.argmin(1))
    sel = {k: dict(
        ade_0_2s=round(float(np.take_along_axis(FE, v[:, None], 1).mean()), 4),
        mean_pdms_lite=round(float(np.take_along_axis(PDMS, v[:, None], 1).mean()), 4),
        at_fault_collision_rate=round(
            float(np.take_along_axis(NCF, v[:, None], 1).mean()), 4))
        for k, v in arms.items()}
    # does a rule VETO destroy the fan's own ceiling? (T1's question, free here)
    orc = FE.argmin(1)
    ok_nc = ~NCF
    ok_all = ok_nc & (~TF) & CO
    def _oracle_within(mask):
        m = np.where(mask, FE, np.inf)
        alive = mask.any(1)
        best = m.min(1)
        return dict(
            frac_windows_with_any_survivor=round(float(alive.mean()), 4),
            oracle_ade_among_survivors=round(
                float(best[alive & np.isfinite(best)].mean()), 4),
            frac_windows_where_the_ade_oracle_is_vetoed=round(
                float((~np.take_along_axis(mask, orc[:, None], 1)[:, 0]).mean()), 4))
    oracle_survival = dict(
        unfiltered_oracle_ade=round(float(FE.min(1).mean()), 4),
        after_NC_veto=_oracle_within(ok_nc),
        after_NC_TTC_comfort_veto=_oracle_within(ok_all))

    gtc = dict(
        note="the SAME labeler applied to the REALISED human future — an "
             "instrument check in the direction that can only embarrass it",
        gt_at_fault_collision_rate=round(
            float(np.mean([r["gt_nc_fault"] for r in rows])), 5),
        gt_any_collision_rate=round(
            float(np.mean([r["gt_nc_any"] for r in rows])), 5),
        gt_ttc_infraction_rate=round(
            float(np.mean([r["gt_ttc"] for r in rows])), 5),
        gt_comfort_ok_rate=round(
            float(np.mean([r["gt_comfort_ok"] for r in rows])), 5),
        published_reference="PARA-Drive: GT trajectories 'collide' at 0.384 % "
                            "under axis-aligned boxes (PDF-VERBATIM)")

    composite = dict(
        definition="PDMS-lite (no-map) = NC * (5*EP + 5*TTC + 2*C) / 12, "
                   "PDM's weights; DAC/DDC/LK/TL absent (no map in corpus)",
        selection_arms=sel,
        oracle_survival_under_rule_veto=oracle_survival,
        ground_truth_self_label_control=gtc,
        selftest_failing_input=dict(
            note="a uniform-random pick over the same fan must score WORSE on "
                 "the composite than the rule pick",
            passes=bool(sel["random"]["mean_pdms_lite"]
                        < sel["rule_pdms_lite_argmax"]["mean_pdms_lite"])),
        note=("⚠️ ade_0_2s CANNOT ADJUDICATE A SCORER — reported here only to "
              "show the two surfaces disagree, which is the point. "
              "L2/ADE vs closed-loop Driving Score measured at rho = -0.36, "
              "p = 0.43 by the same-day closed-loop research."))

    res = dict(
        what="T2 — per-candidate rule-label discrimination on PhysicalAI-AV",
        thresholds=dict(ego_pad_m=a.pad, ttc_horizon_s=a.ttc_horizon,
                        a_lon=[A_LON_MIN, A_LON_MAX], a_lat_max=A_LAT_MAX,
                        jerk_max=JERK_MAX, yaw_rate_max=YAW_RATE_MAX),
        corpus=dict(clips=len(scenes), windows=W, candidates=int(C),
                    fan="256 FPS anchors over dev-box train waypoint targets "
                        "(build_refc_anchors real-data recipe); NO offset head",
                    note="NOT the parity corpus — a label property, not a model "
                         "comparison"),
        agents=dict(mean_agents_per_window=round(float(nag.mean()), 2),
                    median=int(np.median(nag)), max=int(nag.max()),
                    frac_windows_with_zero_agents=round(
                        float((nag == 0).mean()), 4)),
        labels=[nc, nc_any, tt_, cf],
        continuous=dict(
            surface_gap_m=dict(
                p1=round(float(np.percentile(GAP[np.isfinite(GAP)], 1)), 3),
                p50=round(float(np.percentile(GAP[np.isfinite(GAP)], 50)), 3),
                mean_within_window_std=round(
                    float(np.nanmean(np.where(np.isfinite(GAP), GAP,
                                              np.nan).std(1))), 4)),
            ttc_min_s_finite_frac=round(float(np.isfinite(TMIN).mean()), 5),
            progress_m=dict(
                mean_within_window_std=round(float(PR.std()), 4))),
        composite_metric=composite,
        DAC=dict(buildable=False,
                 why="PhysicalAI-AV ships no map / lane graph / drivable-area "
                     "polygon (settled at five probes). DAC is one of PDMS's "
                     "two multiplicative terms — this is a real ceiling."),
        PRE_REGISTERED_KILL=dict(
            rule="positive rate < 5 % AND |Spearman rho with fan_err| < 0.15 "
                 "=> R1 REFUTED on this corpus",
            NC=dict(positive_rate=nc["positive_rate"],
                    pooled_rho=nc["rho_vs_fan_err"]["pooled_rho"],
                    fires=bool(kill_nc)),
            TTC=dict(positive_rate=tt_["positive_rate"],
                     pooled_rho=tt_["rho_vs_fan_err"]["pooled_rho"],
                     fires=bool(kill_tt)),
            verdict=("KILL — R1 REFUTED" if (kill_nc and kill_tt)
                     else "SURVIVES")),
    )
    Path(a.out).write_text(json.dumps(res, indent=1))
    torch.save(dict(
        alias=[r["alias"] for r in rows], t0_s=[r["t0_s"] for r in rows],
        v0=np.array([r["v0"] for r in rows]), n_agents=nag,
        fan_err=FE.astype(np.float32), nc_fault=NCF, nc_any=NCA,
        ttc_flag=TF, comfort_ok=CO, progress=PR.astype(np.float32),
        gap=GAP.astype(np.float32), ttc_min=TMIN.astype(np.float32),
        a_lon_max=st("a_lon_max").astype(np.float32),
        a_lon_min=st("a_lon_min").astype(np.float32),
        a_lat_absmax=st("a_lat_absmax").astype(np.float32),
        jerk_absmax=st("jerk_absmax").astype(np.float32),
        yr_absmax=st("yr_absmax").astype(np.float32),
        pdms_lite=PDMS.astype(np.float32),
        _note="per-window x per-candidate rule labels; every rate/rho/arm in "
              "t2_labels.json recomputes from this with no GPU"), a.dump)
    print(json.dumps({k: res[k] for k in
                      ("corpus", "agents", "labels", "composite_metric",
                       "PRE_REGISTERED_KILL")}, indent=1))


if __name__ == "__main__":
    main()
