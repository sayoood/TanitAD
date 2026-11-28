"""GP-LONGITUDINAL — the axis that owns 88.7 % of the oracle gap, and the ONE thing a
metric goal POINT can carry that a bearing and a categorical command structurally cannot.

WHY THIS RUN EXISTS. The lateral-axis probe (`GP_LATERAL_ADAPTIVE.json`) found that at a
FIXED ARC LENGTH a metric goal point and a bearing TIE EXACTLY (-0.0414 / -0.0414 bank
ADE, +0.1115 / +0.1115 turn accuracy) — and that is not a coincidence, it is an identity:
all anchors evaluated at the same arc S lie at ~the same range from the car, so distance
and cosine are monotonically related. ⇒ **at a fixed arc, the "metric" half of a goal
point is degenerate.** The point's extra content is RANGE, and range is a LONGITUDINAL
quantity. This probe tests it where it can act: hold the LATERAL index at the model's own
and vary only ``a_long``, with the goal defined at a fixed TIME beyond the scored horizon.

⛔ SELECTION PATH ONLY, BANK SCALE, ORACLE-FED = UPPER BOUNDS. Not a trained-arm result.
⛔ LEAK: a goal at t >= 3.0 s is strictly outside the scored 2 s grid. Stated, not assumed.
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

SLOTS = [4, 9, 14, 19, 29, 39, 49, 59]
SLOT_SEC = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
N_ANCH, ROLL_H = 117, 60
ALAT_V_FLOOR, KAPPA_CAP, ANCHOR_DT = 4.0, 0.12, 0.1


def ego_frame(p, pose0):
    c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    dx, dy = p[..., 0] - float(pose0[0]), p[..., 1] - float(pose0[1])
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


def roll_bank(v, ctrl0):
    b = v.shape[0]
    kap = (ctrl0[None, :, 1] / (v.clamp_min(ALAT_V_FLOOR) ** 2)[:, None]).clamp(
        -KAPPA_CAP, KAPPA_CAP)
    c = torch.stack([ctrl0[None, :, 0].expand(b, N_ANCH), kap], -1)
    c = c[:, :, None, :].expand(b, N_ANCH, ROLL_H, 2).reshape(-1, ROLL_H, 2)
    s0 = torch.zeros(b * N_ANCH, 4)
    s0[:, 3] = v[:, None].expand(b, N_ANCH).reshape(-1)
    return rollout_unicycle(s0, c, dt=ANCHOR_DT)[..., :2][:, SLOTS].reshape(
        b, N_ANCH, 8, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--goal-t", type=float, default=4.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.goal_t <= 2.0:
        raise SystemExit("goal-t must be beyond the scored 2 s horizon (leak guard)")
    slot = SLOT_SEC.index(args.goal_t)
    k_gt = int(round(args.goal_t * 10))

    d = torch.load(args.anchors, map_location="cpu", weights_only=False)
    ctrl0 = d["controls"].float()
    a_long, a_lat = ctrl0[:, 0].numpy(), ctrl0[:, 1].numpy()
    i_long = np.searchsorted(np.unique(np.round(a_long, 4)), np.round(a_long, 4))
    i_lat = np.searchsorted(np.unique(np.round(a_lat, 4)), np.round(a_lat, 4))

    kt = ("g", "v0")
    kd = ("sel_idx_nav_true", "reach_keep_nav_true", "sel_bank_nav_true",
          "gt_future_ext", "gt_future_valid_ext", "pose_last")
    acc = {k: [] for k in kt + kd}
    eid = []
    for f in sorted(glob.glob(os.path.join(args.dump, "ep*.npz"))):
        a = np.load(f)
        b = np.load(os.path.join(args.dump, "decisions", os.path.basename(f)))
        for k in kt:
            acc[k].append(a[k])
        for k in kd:
            acc[k].append(b[k])
        eid.append(np.full(len(a["ws"]), int(os.path.basename(f)[2:-4]), dtype=np.int64))
    X = {k: np.concatenate(v, 0) for k, v in acc.items()}
    eid = np.concatenate(eid)
    n = X["g"].shape[0]
    g = X["g"]
    BANK = roll_bank(torch.from_numpy(X["v0"]).float(), ctrl0).numpy()
    sel, reach = X["sel_idx_nav_true"], X["reach_keep_nav_true"] > 0
    k2 = float(np.abs(BANK[np.arange(n), sel] - X["sel_bank_nav_true"]).max())

    gt_ego = np.stack([ego_frame(X["gt_future_ext"][i, :, :2], X["pose_last"][i])
                       for i in range(n)])
    gtv = X["gt_future_valid_ext"]
    have = gtv[:, k_gt - 1] > 0
    gp = np.where(have[:, None], gt_ego[:, k_gt - 1], 0.0)          # [n, 2] at goal_t

    APT = BANK[:, :, slot]                                          # [n, 117, 2]
    same_lat = (i_lat[None, :] == i_lat[sel][:, None]) & reach
    err2 = np.linalg.norm(BANK[:, :, :4, :] - g[:, None], axis=-1).mean(-1)

    def pick(sc):
        return np.where(same_lat, sc, -np.inf).argmax(1)

    rng = np.random.default_rng(0)
    rng_true = np.linalg.norm(gp, axis=-1)
    bt = gp / np.maximum(rng_true[:, None], 1e-6)
    ab = APT / np.maximum(np.linalg.norm(APT, axis=-1, keepdims=True), 1e-6)
    surf = {
        "live": sel,
        "long_ORACLE": pick(-err2),
        "point_TIME_ORACLE": pick(-np.linalg.norm(APT - gp[:, None], axis=-1)),
        # ⭐ THE DISCRIMINATOR: a BEARING at the same time carries DIRECTION only. On the
        # longitudinal axis every candidate points nearly straight ahead, so a bearing
        # is near-blind here BY CONSTRUCTION — this row measures how blind.
        "bearing_TIME_ORACLE": pick((ab * bt[:, None]).sum(-1)),
        # ⛔ THE DELIBERATE-REGRESSION ARM for this axis is a RANGE corruption, not a
        # mirror: mirroring y is a lateral intervention and would be inert here.
        "point_TIME_RANGE_x0.5": pick(-np.linalg.norm(APT - (gp * 0.5)[:, None], axis=-1)),
        "point_TIME_RANGE_x2.0": pick(-np.linalg.norm(APT - (gp * 2.0)[:, None], axis=-1)),
        # the no-information goal: the corpus-mean RANGE, straight ahead
        "point_TIME_CONSTANT": pick(-np.linalg.norm(
            APT - np.stack([np.full(n, float(rng_true[have].mean())),
                            np.zeros(n)], -1)[:, None], axis=-1)),
    }
    for sig in (0.5, 1.0, 2.0, 4.0):
        gpn = gp.copy()
        gpn[:, 0] = gpn[:, 0] + rng.normal(0.0, sig, size=n)
        surf["point_TIME_NOISE_rng%.1fm" % sig] = pick(
            -np.linalg.norm(APT - gpn[:, None], axis=-1))

    sub = np.where(have)[0]
    gt_t = torch.from_numpy(g[sub].astype(np.float32))
    eps = np.unique(eid[sub])
    rows = {e: np.where(eid[sub] == e)[0] for e in eps}
    rg = np.random.default_rng(0)
    draws = [np.concatenate([rows[eps[j]] for j in rg.integers(0, len(eps), len(eps))])
             for _ in range(args.n_boot)]

    R = {"_is": ("GP-LONGITUDINAL: hold the LATERAL anchor index at the model's own and "
                 "ask which route signal picks the better LONGITUDINAL index, with the "
                 "goal defined at a fixed TIME beyond the scored horizon. Selection path "
                 "only, bank scale, ORACLE-FED = UPPER BOUNDS."),
         "evidence_class": "MEASURED (ours)", "tier": "T1 (dump's tier; re-ranked here)",
         "goal_time_s": args.goal_t, "anchor_slot": slot,
         "n_windows_scored": int(len(sub)), "n_windows_total": int(n),
         "n_episodes_scored": int(len(eps)),
         "estimator": "paired episode-cluster bootstrap, n_boot=%d, seed=0" % args.n_boot,
         "controls": {"K2_rebuilt_bank_max_abs_m": k2,
                      "K2_verdict": "PASS" if k2 < 1e-4 else "FAIL",
                      "candidates_per_window_same_a_lat": {
                          "mean": round(float(same_lat[sub].sum(1).mean()), 3),
                          "min": int(same_lat[sub].sum(1).min())},
                      "goal_beyond_scored_horizon": bool(args.goal_t > 2.0)},
         "surfaces": {}, "paired_vs_live": {}}

    A2, SP = {}, {}
    for name, idx in surf.items():
        traj = BANK[np.arange(n), idx][sub]
        A2[name] = np.linalg.norm(traj[:, :4] - g[sub], axis=-1).mean(-1)
        lon = ff.longitudinal(torch.from_numpy(traj[:, :4].astype(np.float32)), gt_t,
                              dt=0.5, eid=None, n_boot=0)
        SP[name] = np.abs(np.linalg.norm(np.diff(np.concatenate(
            [np.zeros((len(sub), 1, 2)), traj[:, :4]], 1), axis=1), axis=-1) / 0.5
            - np.linalg.norm(np.diff(np.concatenate(
                [np.zeros((len(sub), 1, 2)), g[sub]], 1), axis=1), axis=-1) / 0.5
            ).mean(-1)
        R["surfaces"][name] = {
            "bank_ADE_2s_m": round(float(A2[name].mean()), 4),
            "speed_mae_ms": lon.get("speed_mae_mps"),
            "along_track_mae_m": lon.get("along_mae_m"),
            "recomputed_speed_mae_ms": round(float(SP[name].mean()), 4),
            "n_distinct": int(len(np.unique(idx[sub]))),
            "changed_vs_live_frac": round(float((idx[sub] != sel[sub]).mean()), 4)}

    def pair(a, b):
        dd = a - b
        bs = np.array([dd[r].mean() for r in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(delta=round(float(dd.mean()), 5),
                    ci95=[round(float(lo), 5), round(float(hi), 5)],
                    separated=bool(lo > 0 or hi < 0))

    for name in surf:
        if name == "live":
            continue
        R["paired_vs_live"][name] = {"bank_ADE_2s_m": pair(A2[name], A2["live"]),
                                     "speed_mae_ms": pair(SP[name], SP["live"])}

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(json.dumps(R["controls"], indent=2))
    print(json.dumps(R["surfaces"], indent=2))
    print(json.dumps(R["paired_vs_live"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
