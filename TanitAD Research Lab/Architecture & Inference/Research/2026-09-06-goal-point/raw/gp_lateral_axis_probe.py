"""GP-LATERAL — the CONFOUND-FREE version: hold the LONGITUDINAL choice fixed and ask
which route signal picks the better LATERAL index.

WHY THIS SHAPE. refcv4b's 117-anchor vocabulary is a 13 x 9 GRID over
(a_long m/s^2, a_lat m/s^2) — MEASURED from the live `anchors.pt` controls. Ranking the
WHOLE fan by goal proximity is dominated by the longitudinal axis (2 s ADE is
longitudinal-dominated: 88.7 % of the programme's oracle gap), so a whole-fan goal
ranking answers a question nobody asked. Fixing `a_long` at the model's OWN choice and
varying only `a_lat` isolates exactly the decision a route signal is supposed to make.

Every arm is scored ONLY on windows that HAVE a goal-point label at the stated arc
length; the live pick stands elsewhere and those windows are excluded from the margin.

⛔ SELECTION-PATH ONLY, on the BANK scale (unrefined anchors). Not a trained-arm result.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

from taniteval import four_families as ff
from tanitad.models.kinematic import rollout_unicycle
from tanitad.refs.refc_tactical import LAT_CLASSES, factor_from_kinematics
from tanitad.data.lan import LanConfig, horizon_lead_m

SLOTS = [4, 9, 14, 19, 29, 39, 49, 59]
N_ANCH, ROLL_H = 117, 60
ALAT_V_FLOOR, KAPPA_CAP, ANCHOR_DT = 4.0, 0.12, 0.1


def ego_frame(p, pose0):
    c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    dx, dy = p[..., 0] - float(pose0[0]), p[..., 1] - float(pose0[1])
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


def roll_bank(v, ctrl0):
    b = v.shape[0]
    vv = v.clamp_min(ALAT_V_FLOOR) ** 2
    kap = (ctrl0[None, :, 1] / vv[:, None]).clamp(-KAPPA_CAP, KAPPA_CAP)
    c = torch.stack([ctrl0[None, :, 0].expand(b, N_ANCH), kap], -1)
    c = c[:, :, None, :].expand(b, N_ANCH, ROLL_H, 2).reshape(-1, ROLL_H, 2)
    s0 = torch.zeros(b * N_ANCH, 4)
    s0[:, 3] = v[:, None].expand(b, N_ANCH).reshape(-1)
    p = rollout_unicycle(s0, c, dt=ANCHOR_DT)[..., :2]
    return p[:, SLOTS].reshape(b, N_ANCH, 8, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arc", type=float, default=40.0)
    ap.add_argument("--adaptive", action="store_true",
                    help="place the goal at the PER-WINDOW leak guard "
                         "S_w = max(2s GT arc, v0*2.0) + min_lead_m — the shortest ADMISSIBLE arc, which the fixed-arc sweep measures to be where the signal is")
    ap.add_argument("--leak-guard", action="store_true",
                    help="score ONLY windows where the goal arc is beyond the "
                         "LAN leak guard max(2s GT arc, v0*2.0) + min_lead_m")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    d = torch.load(args.anchors, map_location="cpu", weights_only=False)
    ctrl0 = d["controls"].float()
    a_long, a_lat = ctrl0[:, 0].numpy(), ctrl0[:, 1].numpy()
    long_lvl = np.unique(np.round(a_long, 4))
    lat_lvl = np.unique(np.round(a_lat, 4))
    i_long = np.searchsorted(long_lvl, np.round(a_long, 4))
    i_lat = np.searchsorted(lat_lvl, np.round(a_lat, 4))

    keys_top = ("g", "v0")
    keys_dec = ("sel_idx_nav_true", "reach_keep_nav_true", "sel_bank_nav_true",
                "gt_future_ext", "gt_future_valid_ext", "pose_last", "route_label")
    acc = {k: [] for k in keys_top + keys_dec}
    eid = []
    for f in sorted(glob.glob(os.path.join(args.dump, "ep*.npz"))):
        a = np.load(f)
        b = np.load(os.path.join(args.dump, "decisions", os.path.basename(f)))
        for k in keys_top:
            acc[k].append(a[k])
        for k in keys_dec:
            acc[k].append(b[k])
        eid.append(np.full(len(a["ws"]), int(os.path.basename(f)[2:-4]), dtype=np.int64))
    X = {k: np.concatenate(v, 0) for k, v in acc.items()}
    eid = np.concatenate(eid)
    n = X["g"].shape[0]
    g = X["g"]
    BANK = roll_bank(torch.from_numpy(X["v0"]).float(), ctrl0).numpy()
    sel = X["sel_idx_nav_true"]
    reach = X["reach_keep_nav_true"] > 0

    k2 = float(np.abs(BANK[np.arange(n), sel] - X["sel_bank_nav_true"]).max())

    gt_ego = np.stack([ego_frame(X["gt_future_ext"][i, :, :2], X["pose_last"][i])
                       for i in range(n)])
    gtv = X["gt_future_valid_ext"]
    # ⛔⛔ ARC IS MEASURED FROM THE CAR. `gt_future_ext[0]` is already 0.1 s ahead and
    # the anchor bank's slot 0 is 0.5 s ahead, so measuring each polyline's arc from
    # its own first sample would compare the goal at arc S to the anchor at arc S+~4.6 m
    # — a systematic several-metre mismatch between the two quantities being matched.
    # The ego origin is PREPENDED to both, exactly as `lan.horizon_lead_m` does it.
    gt_ego = np.concatenate([np.zeros((n, 1, 2)), gt_ego], axis=1)          # [n, 61, 2]
    gtv = np.concatenate([np.ones((n, 1)), gtv], axis=1)                    # [n, 61]
    step = np.linalg.norm(np.diff(gt_ego, axis=1), axis=-1)
    step = np.where((gtv[:, 1:] > 0) & (gtv[:, :-1] > 0), step, 0.0)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(step, 1)], 1)
    vp = np.cumprod(gtv > 0, axis=1).astype(bool)
    reach_len = np.where(vp, cum, -1.0).max(1)
    S = args.arc
    have = reach_len >= S
    # ⛔⛔ THE LEAK GUARD. A goal point INSIDE the scored 2 s horizon is the ANSWER,
    # not a route signal. `tanitad.data.lan.horizon_lead_m` is the programme's own
    # guard and is IMPORTED, never re-derived: max(2 s GT arc length, v0 * 2.0) +
    # cfg.min_lead_m. Applying it is what separates an admissible design point from
    # an oracle.
    lan_cfg = LanConfig()
    lead = np.array([horizon_lead_m(gt_path_ego=g[i], v0=float(X["v0"][i]),
                                    t_pred_s=2.0, cfg=lan_cfg) for i in range(n)])
    Sw = np.full(n, float(S))
    if args.adaptive:
        Sw = lead.copy()
        have = reach_len >= Sw
    guarded = Sw >= lead
    R_leak = {"lead_m_mean": round(float(lead.mean()), 2),
              "lead_m_p95": round(float(np.percentile(lead, 95)), 2),
              "frac_arc_beyond_guard": round(float(guarded.mean()), 4),
              "min_lead_m": lan_cfg.min_lead_m}
    if args.leak_guard:
        have = have & guarded
    gp = np.zeros((n, 2))
    for i in np.where(have)[0]:
        Si = float(Sw[i])
        j = min(max(int(np.searchsorted(cum[i], Si)), 1), 60)
        c0, c1 = cum[i, j - 1], cum[i, j]
        w = 0.0 if c1 <= c0 else (Si - c0) / (c1 - c0)
        gp[i] = gt_ego[i, j - 1] * (1 - w) + gt_ego[i, j] * w

    # anchor point at the same ARC LENGTH along each anchor (per window, v0-conditioned)
    BANK0 = np.concatenate([np.zeros((n, N_ANCH, 1, 2)), BANK], axis=2)   # from the CAR
    astep = np.linalg.norm(np.diff(BANK0, axis=2), axis=-1)
    acum = np.concatenate([np.zeros((n, N_ANCH, 1)), np.cumsum(astep, 2)], 2)
    jj = np.clip((acum < Sw[:, None, None]).sum(2), 1, 8)
    ii, kk = np.meshgrid(np.arange(n), np.arange(N_ANCH), indexing="ij")
    c0, c1 = acum[ii, kk, jj - 1], acum[ii, kk, jj]
    w = np.clip(np.where(c1 > c0, (Sw[:, None] - c0) / np.maximum(c1 - c0, 1e-9),
                        0.0), 0, 1)[..., None]
    APT = BANK0[ii, kk, jj - 1] * (1 - w) + BANK0[ii, kk, jj] * w    # [n, 117, 2]
    # ⚠️ an anchor SHORTER than S is clamped to its endpoint; count it, do not hide it
    short = (acum[:, :, -1] < Sw[:, None])

    # candidate set: same a_long row as the LIVE pick, reachable
    same_long = (i_long[None, :] == i_long[sel][:, None]) & reach
    err2 = np.linalg.norm(BANK[:, :, :4, :] - g[:, None], axis=-1).mean(-1)

    def pick(score, mask):
        return np.where(mask, score, -np.inf).argmax(1)

    dist = np.linalg.norm(APT - gp[:, None], axis=-1)
    gpm = gp.copy(); gpm[:, 1] *= -1.0
    dist_m = np.linalg.norm(APT - gpm[:, None], axis=-1)
    # ⭐ the NO-INFORMATION goal: straight ahead at the SAME arc length. With a
    # per-window arc this is the honest constant control — it removes the route
    # content while keeping the goal's presence, distance and validity identical.
    gpc = np.stack([Sw, np.zeros(n)], -1)
    dist_c = np.linalg.norm(APT - gpc[:, None], axis=-1)
    bt = gp / np.maximum(np.linalg.norm(gp, axis=-1, keepdims=True), 1e-6)
    ab = APT / np.maximum(np.linalg.norm(APT, axis=-1, keepdims=True), 1e-6)
    cosb = (ab * bt[:, None]).sum(-1)

    surf = {
        "live": sel,
        "lat_ORACLE": pick(-err2, same_long),
        "point_ORACLE": pick(-dist, same_long),
        "point_MIRRORED": pick(-dist_m, same_long),
        "point_STRAIGHT": pick(-dist_c, same_long),
        "bearing_ORACLE": pick(cosb, same_long),
    }
    # categorical ceiling ON THE LATERAL AXIS: one a_lat level per route class,
    # chosen with the TRUE label by class-mean 2 s error inside the fixed a_long row
    cat = sel.copy()
    for c in np.unique(X["route_label"]):
        m = (X["route_label"] == c) & have
        if not m.any():
            continue
        e = np.where(same_long[m], err2[m], np.nan)
        best_lat = int(np.nanargmin([np.nanmean(np.where(i_lat[None, :] == L, e, np.nan))
                                     for L in range(len(lat_lvl))]))
        rows = np.where(m)[0]
        cand = same_long[rows] & (i_lat[None, :] == best_lat)
        ok = cand.any(1)
        cat[rows[ok]] = np.where(cand[ok], 1.0, -np.inf).argmax(1)
    surf["categorical_ORACLE_3way"] = cat
    # ⭐ THE STRONGEST DETERMINISTIC CATEGORICAL DECODER, so the categorical arm is
    # not straw-manned: per route class, the MODAL oracle lateral level (not the
    # class-mean-error one). A 3-way token cannot do better than its best 3 -> 9 map.
    lat_star = i_lat[pick(-err2, same_long)]
    cat2 = sel.copy()
    modal = {}
    for c in np.unique(X["route_label"]):
        m = (X["route_label"] == c) & have
        if not m.any():
            continue
        L = int(np.bincount(lat_star[m], minlength=len(lat_lvl)).argmax())
        modal[int(c)] = {"modal_oracle_lat_level": L,
                         "share": round(float((lat_star[m] == L).mean()), 4),
                         "n": int(m.sum()),
                         "oracle_lat_hist": np.bincount(
                             lat_star[m], minlength=len(lat_lvl)).tolist()}
        rows = np.where(m)[0]
        cand = same_long[rows] & (i_lat[None, :] == L)
        ok = cand.any(1)
        cat2[rows[ok]] = np.where(cand[ok], 1.0, -np.inf).argmax(1)
    surf["categorical_MODAL_3way"] = cat2

    # ⭐⭐ THE HEAD-ACCURACY REQUIREMENT CURVE. `point_ORACLE` is oracle-fed; a real
    # arm must PREDICT the point. Injecting LATERAL noise of a known sigma converts
    # "how good must the goal-point head be?" from an opinion into a number, and it
    # is what sizes the head's own pre-registered bar. Noise is applied in the ego
    # frame's lateral axis only (the axis the lateral index answers to).
    rng_n = np.random.default_rng(0)
    for sig in (0.5, 1.0, 2.0, 4.0, 8.0):
        gpn = gp.copy()
        gpn[:, 1] = gpn[:, 1] + rng_n.normal(0.0, sig, size=n)
        surf["point_NOISE_lat%.1fm" % sig] = pick(
            -np.linalg.norm(APT - gpn[:, None], axis=-1), same_long)

    sub = np.where(have)[0]
    gt_t = torch.from_numpy(g[sub].astype(np.float32))
    dyg, dvg, v0g, v1g, _ = ff.maneuver_kinematics(gt_t, 0.5)
    lat_gt = factor_from_kinematics(dyg, dvg, v0g, v1g)[0].numpy()

    eps = np.unique(eid[sub])
    rows_by_ep = {e: np.where(eid[sub] == e)[0] for e in eps}
    rg = np.random.default_rng(0)
    draws = [np.concatenate([rows_by_ep[eps[j]] for j in rg.integers(0, len(eps), len(eps))])
             for _ in range(args.n_boot)]

    R = {
        "_is": ("GP-LATERAL: hold the LONGITUDINAL anchor choice at the model's own and "
                "ask which route signal picks the better LATERAL index. Selection path "
                "only, bank scale, ORACLE-FED goal signals = UPPER BOUNDS."),
        "evidence_class": "MEASURED (ours)", "tier": "T1 (dump's tier; re-ranked here)",
        "arc_length_m": ("PER-WINDOW leak guard" if args.adaptive else S),
        "arc_length_m_stats": {"mean": round(float(Sw[have].mean()), 2),
                               "p05": round(float(np.percentile(Sw[have], 5)), 2),
                               "p95": round(float(np.percentile(Sw[have], 95)), 2)},
        "leak_guard_applied": bool(args.leak_guard),
        "n_windows_scored": int(len(sub)), "n_windows_total": int(n),
        "n_episodes_scored": int(len(eps)),
        "estimator": "paired episode-cluster bootstrap, n_boot=%d, seed=0" % args.n_boot,
        "anchor_grid": {"a_long_levels": [round(float(x), 4) for x in long_lvl],
                        "a_lat_levels": [round(float(x), 4) for x in lat_lvl],
                        "units": "m/s^2 (control_units='alat'; kappa = a_lat / v^2, "
                                 "clamped at 0.12 with a 4.0 m/s speed floor)"},
        "controls": {
            "K2_rebuilt_bank_equals_banked_sel_bank_max_abs_m": k2,
            "K2_verdict": "PASS" if k2 < 1e-4 else "FAIL",
            "candidates_per_window_same_a_long": {
                "mean": round(float(same_long[sub].sum(1).mean()), 3),
                "min": int(same_long[sub].sum(1).min())},
            "anchors_SHORTER_than_arc_clamped_to_endpoint_frac":
                round(float(short[sub].mean()), 4),
        },
        "leak_guard": R_leak,
        "lat_recall_support": {cn: int((lat_gt == ci).sum())
                               for ci, cn in enumerate(LAT_CLASSES)},
        "categorical_modal_map": modal,
        "surfaces": {}, "paired_vs_live": {},
    }

    A2 = {}
    for name, idx in surf.items():
        traj = BANK[np.arange(n), idx][sub]
        pt = torch.from_numpy(traj[:, :4].astype(np.float32))
        A2[name] = np.linalg.norm(traj[:, :4] - g[sub], axis=-1).mean(-1)
        lat = ff.lateral(pt, gt_t, dt=0.5, eid=None, n_boot=0)
        dyp, dvp, v0p, v1p, _ = ff.maneuver_kinematics(pt, 0.5)
        lp = factor_from_kinematics(dyp, dvp, v0p, v1p)[0].numpy()
        R["surfaces"][name] = {
            "bank_ADE_2s_m": round(float(A2[name].mean()), 4),
            "curvature_mae_1pm": lat["curvature_mae_1pm"],
            "heading_mae_deg": lat["heading_mae_deg"],
            "cross_mae_m": lat["cross_mae_m"],
            "lat_recall": {cn: (round(float((lp[lat_gt == ci] == ci).mean()), 4)
                                if (lat_gt == ci).any() else None)
                           for ci, cn in enumerate(LAT_CLASSES)},
            "changed_vs_live_frac": round(float((idx[sub] != sel[sub]).mean()), 4)}

    def pair(a, b, vec_a, vec_b):
        d = vec_a - vec_b
        bs = np.array([d[r].mean() for r in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(delta=round(float(d.mean()), 5),
                    ci95=[round(float(lo), 5), round(float(hi), 5)],
                    separated=bool(lo > 0 or hi < 0))

    turn = np.isin(lat_gt, [1, 2])
    for name in surf:
        if name == "live":
            continue
        idx = surf[name]
        traj = BANK[np.arange(n), idx][sub]
        pt = torch.from_numpy(traj[:, :4].astype(np.float32))
        dyp, dvp, v0p, v1p, _ = ff.maneuver_kinematics(pt, 0.5)
        lp = factor_from_kinematics(dyp, dvp, v0p, v1p)[0].numpy()
        lv = factor_from_kinematics(*ff.maneuver_kinematics(
            torch.from_numpy(BANK[np.arange(n), sel][sub][:, :4].astype(np.float32)),
            0.5)[:4])[0].numpy()
        hit_a = (lp == lat_gt).astype(float)
        hit_b = (lv == lat_gt).astype(float)
        R["paired_vs_live"][name] = {
            "bank_ADE_2s_m": pair(name, "live", A2[name], A2["live"]),
            "lat_class_accuracy": pair(name, "live", hit_a, hit_b),
        }
    # the turn-subset pairing needs its own draws (different row set)
    tsub = np.where(turn)[0]
    rows_t = {e: np.where(eid[sub][tsub] == e)[0] for e in np.unique(eid[sub][tsub])}
    epst = np.unique(eid[sub][tsub])
    rg2 = np.random.default_rng(0)
    draws_t = [np.concatenate([rows_t[epst[j]] for j in rg2.integers(0, len(epst), len(epst))])
               for _ in range(args.n_boot)]
    lv = factor_from_kinematics(*ff.maneuver_kinematics(
        torch.from_numpy(BANK[np.arange(n), sel][sub][:, :4].astype(np.float32)), 0.5)[:4])[0].numpy()
    for name in surf:
        if name == "live":
            continue
        traj = BANK[np.arange(n), surf[name]][sub]
        lp = factor_from_kinematics(*ff.maneuver_kinematics(
            torch.from_numpy(traj[:, :4].astype(np.float32)), 0.5)[:4])[0].numpy()
        d = (lp == lat_gt).astype(float)[tsub] - (lv == lat_gt).astype(float)[tsub]
        bs = np.array([d[r].mean() for r in draws_t])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        R["paired_vs_live"][name]["turn_windows_lat_accuracy"] = dict(
            delta=round(float(d.mean()), 5), ci95=[round(float(lo), 5), round(float(hi), 5)],
            separated=bool(lo > 0 or hi < 0), n_turn_windows=int(len(tsub)),
            n_episodes=int(len(epst)))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(json.dumps({k: R[k] for k in ("controls", "lat_recall_support")}, indent=2))
    print(json.dumps(R["surfaces"], indent=2))
    print(json.dumps(R["paired_vs_live"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
