"""GP-PROBE — what a GEOMETRIC GOAL POINT can buy on refcv4b's LIVE anchor fan.

ZERO GPU. Reads the banked `navflip_dump` (the dump E13 was read off) and rebuilds the
v0-CONDITIONED 117-anchor bank each window is actually ranked over, then re-ranks it
under different goal surfaces.

⛔ SCOPE, stated up front so nothing is over-read:
  * this bounds what a goal signal can buy through the SELECTION path ONLY, with the
    emitted fan held fixed. It is NOT a trained-arm result and it does not speak for
    the GENERATION path. It exists to RANK levers and to SIZE a pre-registered bar.
  * every oracle-fed row is an UPPER BOUND, not a prediction of a trained arm.
  * a counterfactual pick is scored on its BANK path (the unrefined anchor). The
    decoder's per-candidate offset is only known for the candidate that was actually
    selected, so a re-ranked arm's refined path cannot be reconstructed offline.
    Reported on the bank scale throughout; never mixed with emitted-plan ADE.

⛔ `a_star`, `anchor_acc` and `sel_agrees_oracle` are NOT read — D-REFCV4B-ASTAR-GEOMETRY
is unrepaired. The oracle used here is computed in this file from `g` with its geometry
stated, and its bank reconstruction is validated against `sel_bank_nav_true` (K2).
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

SLOTS = [4, 9, 14, 19, 29, 39, 49, 59]          # V3_HORIZONS - 1, at anchor_dt 0.1 s
HZ_SEC = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
N_ANCH = 117
ROLL_H = 60
# refcv4b's live anchor settings (manifest_9500.json model_checks / cfg.core.anchors)
REF_SPEED, ALAT_V_FLOOR, KAPPA_CAP, ANCHOR_DT = 10.0, 4.0, 0.12, 0.1


def ego_frame(p_abs: np.ndarray, pose0: np.ndarray) -> np.ndarray:
    c, s = math.cos(float(pose0[2])), math.sin(float(pose0[2]))
    dx, dy = p_abs[..., 0] - float(pose0[0]), p_abs[..., 1] - float(pose0[1])
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=-1)


def roll_bank(v: torch.Tensor, ctrl0: torch.Tensor) -> torch.Tensor:
    """[n, 117, 8, 2] — refc.AnchorDecoder.roll_bank for control_units='alat',
    anchor_v0_cond=True, ego kept. Reproduced here because the probe needs the WHOLE
    fan and the arm only banks the selected row; validated bit-close by K2."""
    b = v.shape[0]
    vv = v.clamp_min(ALAT_V_FLOOR) ** 2
    kap = (ctrl0[None, :, 1] / vv[:, None]).clamp(-KAPPA_CAP, KAPPA_CAP)
    c = torch.stack([ctrl0[None, :, 0].expand(b, N_ANCH), kap], -1)
    c = c[:, :, None, :].expand(b, N_ANCH, ROLL_H, 2).reshape(-1, ROLL_H, 2)
    s0 = torch.zeros(b * N_ANCH, 4)
    s0[:, 3] = v[:, None].expand(b, N_ANCH).reshape(-1)
    p = rollout_unicycle(s0, c, dt=ANCHOR_DT)[..., :2]
    return p[:, SLOTS].reshape(b, N_ANCH, 8, 2)


def ade(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1).mean(axis=-1)


def boot_paired(a, b, eid, draws):
    d = a - b
    bs = np.array([d[r].mean() for r in draws])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(delta=round(float(d.mean()), 4),
                ci95=[round(float(lo), 4), round(float(hi), 4)],
                separated=bool(lo > 0 or hi < 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    d = torch.load(args.anchors, map_location="cpu", weights_only=False)
    ctrl0, anch_fixed = d["controls"].float(), d["anchors"].float()

    keys_top = ("g", "os", "ha0", "ha0_ext", "ha", "v0")
    keys_dec = ("sel_bank_nav_true", "sel_idx_nav_true", "reach_keep_nav_true",
                "plan_full_nav_true", "gt_future_ext", "gt_future_valid_ext",
                "pose_last", "route_label", "nav_cmd")
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

    R = {
        "_is": ("GP-PROBE: the SELECTION-surface ceiling of a geometric goal point on "
                "refcv4b's live, v0-conditioned 117-anchor fan. ZERO GPU, from the "
                "banked navflip dump. UPPER BOUNDS, not a trained-arm result."),
        "evidence_class": "MEASURED (ours)",
        "tier": "T1 (the dump's tier; this probe re-ranks within it)",
        "n_windows": int(n), "n_episodes": int(len(np.unique(eid))),
        "n_anchors": N_ANCH,
        "estimator": "paired episode-cluster bootstrap, n_boot=%d, seed=0" % args.n_boot,
        "variance_answered": ("would another draw of EPISODES say this? — the fan and the "
                              "checkpoint are held fixed, so no training-run and no "
                              "inference-sampling variance enters"),
        "anchor_bank": {
            "v0_conditioned": True, "control_units": "alat", "ref_speed_ms": REF_SPEED,
            "alat_v_floor_ms": ALAT_V_FLOOR, "kappa_cap": KAPPA_CAP,
            "integrator": "tanitad.models.kinematic.rollout_unicycle (imported)"},
    }

    v0 = torch.from_numpy(X["v0"]).float()
    BANK = roll_bank(v0, ctrl0).numpy()                      # [n, 117, 8, 2]
    g = X["g"]
    gt_ego = np.stack([ego_frame(X["gt_future_ext"][i, :, :2], X["pose_last"][i])
                       for i in range(n)])
    gtv = X["gt_future_valid_ext"]

    # ------------------------- CONTROLS -------------------------------------
    k1 = float(np.abs(gt_ego[:, [4, 9, 14, 19], :] - g).max())  # pre-prepend indexing
    got = BANK[np.arange(n), X["sel_idx_nav_true"]]
    k2 = float(np.abs(got - X["sel_bank_nav_true"]).max())
    k2b = float(np.abs(anch_fixed.numpy()[X["sel_idx_nav_true"]]
                       - X["sel_bank_nav_true"]).max())
    k3 = float(ade(X["os"], g).mean())
    k4 = (float(ade(X["ha0"], g).mean()), float(ade(X["ha0_ext"], g).mean()),
          float(ade(X["ha"], g).mean()))
    R["controls"] = {
        "K1_gt_future_ext_transformed_equals_scored_grid": {
            "max_abs_m": k1, "must_be": "< 1e-3", "verdict": "PASS" if k1 < 1e-3 else "FAIL"},
        "K2_rebuilt_v0_bank_equals_banked_sel_bank": {
            "max_abs_m": k2, "must_be": "< 1e-4 (float32 roll noise)",
            "verdict": "PASS" if k2 < 1e-4 else "FAIL",
            "_same_breath_NEGATIVE_control_fixed_anchors_table": {
                "max_abs_m": round(k2b, 4),
                "_is": ("the FIXED anchors[sel] reading, which must NOT match — it is "
                        "157 m off. A probe whose control and its negative both passed "
                        "would be measuring nothing. This is also an independent "
                        "re-validation of the `alat` control units.")}},
        "K3_os_ADE_reproduces_banked": {
            "measured": round(k3, 4), "known": 0.2965,
            "verdict": "PASS" if abs(k3 - 0.2965) < 5e-4 else "FAIL"},
        "K4_model_free_known_values": {
            "ha0": round(k4[0], 4), "ha0_ext": round(k4[1], 4), "ha": round(k4[2], 4),
            "known": [0.6723, 0.2874, 0.2996],
            "verdict": "PASS" if (abs(k4[0] - 0.6723) < 5e-4 and abs(k4[1] - 0.2874) < 5e-4
                                  and abs(k4[2] - 0.2996) < 5e-4) else "FAIL"},
    }

    reach = X["reach_keep_nav_true"] > 0
    sel_live = X["sel_idx_nav_true"]
    R["controls"]["K5_reachable_anchors_per_window"] = {
        "mean": round(float(reach.sum(1).mean()), 2), "min": int(reach.sum(1).min())}

    bank_ade = ade(BANK[np.arange(n), sel_live][:, :4], g)
    R["scope_selection_vs_generation"] = {
        "_is": ("the emitted plan is the DECODER's refinement of the picked candidate. "
                "Everything below is on the BANK scale and is never mixed with it."),
        "ADE_2s_of_the_SELECTED_BANK_PATH": round(float(bank_ade.mean()), 4),
        "ADE_2s_of_the_EMITTED_PLAN(os)": round(k3, 4),
        "mean_decoder_offset_m": round(float(np.linalg.norm(
            X["plan_full_nav_true"][:, :4] - X["sel_bank_nav_true"][:, :4],
            axis=-1).mean()), 4),
    }

    # ------------------------- GOAL-POINT AVAILABILITY -----------------------
    # ⛔ arc is measured FROM THE CAR on both polylines (see goal_point.py's
    # anchor_point_at_arc): gt_future_ext[0] is already 0.1 s ahead and the bank's
    # slot 0 is 0.5 s ahead, so measuring each from its own first sample compares the
    # goal at arc S to the anchor at arc S + ~4.6 m.
    gt_ego = np.concatenate([np.zeros((n, 1, 2)), gt_ego], axis=1)
    gtv = np.concatenate([np.ones((n, 1)), gtv], axis=1)
    step = np.linalg.norm(np.diff(gt_ego, axis=1), axis=-1)
    okstep = (gtv[:, 1:] > 0) & (gtv[:, :-1] > 0)
    step = np.where(okstep, step, 0.0)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(step, 1)], 1)
    valid_prefix = np.cumprod(gtv > 0, axis=1).astype(bool)
    reach_len = np.where(valid_prefix, cum, -1.0).max(1)
    R["goal_point_availability"] = {
        "_is": ("what fraction of windows HAS a goal-point label. A TIME horizon loses "
                "windows to clip end; an ARC-LENGTH horizon loses them to low speed. "
                "A goal point therefore needs a validity bit (the X15 rule: withheld "
                "must not be byte-identical to a genuine zero)."),
        "by_TIME": {"t_%.1fs" % s: round(float((gtv[:, k - 1] > 0).mean()), 4)
                    for k, s in zip((5, 10, 15, 20, 30, 40, 50, 60), HZ_SEC)},
        "by_ARC_LENGTH": {"s_%dm" % int(S): round(float((reach_len >= S).mean()), 4)
                          for S in (20, 40, 60, 80, 100)},
        "arc_length_available_m": {"mean": round(float(reach_len.mean()), 2),
                                   "p05": round(float(np.percentile(reach_len, 5)), 2),
                                   "median": round(float(np.median(reach_len)), 2)},
    }

    def goal_at_arc(S: float):
        p = np.zeros((n, 2)); ok = reach_len >= S
        for i in np.where(ok)[0]:
            j = min(max(int(np.searchsorted(cum[i], S)), 1), 60)
            c0, c1 = cum[i, j - 1], cum[i, j]
            w = 0.0 if c1 <= c0 else (S - c0) / (c1 - c0)
            p[i] = gt_ego[i, j - 1] * (1 - w) + gt_ego[i, j] * w
        return p, ok

    def anchor_at_arc(S: float):
        """[n, 117, 2] — each window's anchor positions at arc length S along the
        anchor. v0-conditioned, so this is per-window, not a fixed table."""
        # ⛔ arc measured FROM THE CAR — the ego origin is prepended, exactly as
        # goal_point.anchor_point_at_arc does it. Without this the anchor's arc
        # origin sits ~5 m down the road while the goal's sits at the car.
        B0 = np.concatenate([np.zeros((n, N_ANCH, 1, 2)), BANK], axis=2)
        astep = np.linalg.norm(np.diff(B0, axis=2), axis=-1)             # [n,117,8]
        acum = np.concatenate([np.zeros((n, N_ANCH, 1)), np.cumsum(astep, 2)], 2)
        j = np.clip((acum < S).sum(2), 1, 8)                             # [n,117]
        ii, kk = np.meshgrid(np.arange(n), np.arange(N_ANCH), indexing="ij")
        c0, c1 = acum[ii, kk, j - 1], acum[ii, kk, j]
        w = np.where(c1 > c0, (S - c0) / np.maximum(c1 - c0, 1e-9), 0.0)
        w = np.clip(w, 0.0, 1.0)[..., None]
        return B0[ii, kk, j - 1] * (1 - w) + B0[ii, kk, j] * w

    def pick(score):
        return np.where(reach, score, -np.inf).argmax(1)

    err2 = np.linalg.norm(BANK[:, :, :4, :] - g[:, None], axis=-1).mean(-1)   # [n,117]
    surfaces = {"live_selection": sel_live, "oracle_anchor_2s": pick(-err2)}

    for S in (40.0, 60.0):
        p, ok = goal_at_arc(S)
        apt = anchor_at_arc(S)
        tag = "s%d" % int(S)
        surfaces["point_ORACLE_" + tag] = pick(
            -np.linalg.norm(apt - p[:, None], axis=-1))
        pm = p.copy(); pm[:, 1] *= -1.0
        surfaces["point_MIRRORED_" + tag] = pick(
            -np.linalg.norm(apt - pm[:, None], axis=-1))
        pc = np.repeat(p[ok].mean(0)[None], n, 0)
        surfaces["point_CONSTANT_" + tag] = pick(
            -np.linalg.norm(apt - pc[:, None], axis=-1))
        bt = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-6)
        ab = apt / np.maximum(np.linalg.norm(apt, axis=-1, keepdims=True), 1e-6)
        surfaces["bearing_ORACLE_" + tag] = pick((ab * bt[:, None]).sum(-1))
        R.setdefault("goal_specs", {})[tag] = {
            "arc_length_m": S, "n_with_label": int(ok.sum()),
            "frac_with_label": round(float(ok.mean()), 4),
            "constant_goal_point_m": [round(float(x), 3) for x in p[ok].mean(0)]}

    cat_idx = np.zeros(n, dtype=np.int64)
    for c in np.unique(X["route_label"]):
        m = X["route_label"] == c
        if not m.any():
            continue
        cat_idx[m] = int(np.nanargmin(np.where(reach[m], err2[m], np.nan).mean(0)))
    bad = ~reach[np.arange(n), cat_idx]
    cat_idx[bad] = sel_live[bad]
    surfaces["categorical_ORACLE_3way"] = cat_idx
    R["categorical_ceiling_note"] = {
        "_is": ("a categorical command can only name a CLASS, so the strongest thing it "
                "can encode is ONE anchor per class. Chosen with the TRUE route label "
                "and by class-mean 2 s bank error — an ORACLE token used optimally."),
        "n_unreachable_fallback": int(bad.sum()),
        "per_class_anchor": {int(c): int(cat_idx[X["route_label"] == c][0])
                             for c in np.unique(X["route_label"])}}
    rng = np.random.default_rng(0)
    surfaces["random_reachable"] = np.array(
        [rng.choice(np.where(reach[i])[0]) for i in range(n)])

    # ------------------------- SCORE ----------------------------------------
    gt_t = torch.from_numpy(g.astype(np.float32))
    dyg, dvg, v0g, v1g, _ = ff.maneuver_kinematics(gt_t, 0.5)
    lat_gt = factor_from_kinematics(dyg, dvg, v0g, v1g)[0].numpy()
    rows, A2 = {}, {}
    for name, idx in surfaces.items():
        traj = BANK[np.arange(n), idx]
        pt = torch.from_numpy(traj[:, :4].astype(np.float32))
        A2[name] = ade(traj[:, :4], g)
        lat = ff.lateral(pt, gt_t, dt=0.5, eid=None, n_boot=0)
        dyp, dvp, v0p, v1p, _ = ff.maneuver_kinematics(pt, 0.5)
        lat_p = factor_from_kinematics(dyp, dvp, v0p, v1p)[0].numpy()
        rows[name] = {
            "bank_ADE_2s_m": round(float(A2[name].mean()), 4),
            "curvature_mae_1pm": lat["curvature_mae_1pm"],
            "heading_mae_deg": lat["heading_mae_deg"],
            "cross_mae_m": lat["cross_mae_m"],
            "lat_recall": {cn: (round(float((lat_p[lat_gt == ci] == ci).mean()), 4)
                                if (lat_gt == ci).any() else None)
                           for ci, cn in enumerate(LAT_CLASSES)},
            "n_distinct_anchors": int(len(np.unique(idx))),
            "agrees_with_live_selection": round(float((idx == sel_live).mean()), 4)}
    R["selection_surfaces"] = rows
    R["lat_recall_support"] = {cn: int((lat_gt == ci).sum())
                               for ci, cn in enumerate(LAT_CLASSES)}

    eps = np.unique(eid)
    rows_by_ep = {e: np.where(eid == e)[0] for e in eps}
    rg = np.random.default_rng(0)
    draws = [np.concatenate([rows_by_ep[eps[j]] for j in
                             rg.integers(0, len(eps), len(eps))])
             for _ in range(args.n_boot)]
    pairs = [("point_ORACLE_s60", "bearing_ORACLE_s60"),
             ("point_ORACLE_s40", "bearing_ORACLE_s40"),
             ("point_ORACLE_s60", "categorical_ORACLE_3way"),
             ("point_ORACLE_s60", "live_selection"),
             ("bearing_ORACLE_s60", "live_selection"),
             ("categorical_ORACLE_3way", "live_selection"),
             ("point_MIRRORED_s60", "point_ORACLE_s60"),
             ("point_MIRRORED_s60", "live_selection"),
             ("point_CONSTANT_s60", "point_ORACLE_s60"),
             ("oracle_anchor_2s", "live_selection"),
             ("random_reachable", "live_selection")]
    R["contrasts_bank_ADE_2s"] = {
        "%s__minus__%s" % (a, b): boot_paired(A2[a], A2[b], eid, draws)
        for a, b in pairs}

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(json.dumps(R["controls"], indent=2))
    print(json.dumps(R["goal_point_availability"], indent=2))
    print(json.dumps(R["selection_surfaces"], indent=2))
    print(json.dumps(R["contrasts_bank_ADE_2s"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
