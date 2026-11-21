"""NP-2 — PAIRED episode-cluster margins for the LATERAL and TACTICAL families.

The navpred RESULT reports curvature / heading / turn-recall per arm with
``_intervals_complete: false`` and says in terms: *"Do not quote any of these as
separated until NP-2 lands."* This lands it, ZERO GPU, off the banked navflip dump.

⛔ ESTIMATOR NOTE — why this is not a mean-of-window-means. ``curvature_mae_1pm`` is a
STEP-POOLED masked mean, so a window-level average of it would be a different statistic
(the `heldout`-vs-`full_set` error in miniature). Each bootstrap draw here resamples
EPISODES and RE-POOLS the steps of the drawn episodes, so the resampled statistic is the
same arithmetic as the published point estimate.

⛔ WHICH VARIANCE THIS ANSWERS: *"would another draw of EPISODES say this?"* — nothing
else. Every arm here is ONE checkpoint under an input intervention, and refcv4b's decoder
is deterministic at inference (D-REFCV4B-SEED-SCOPE), so neither the training-run variance
(H-ESTIM-SEED-1) nor an inference-sampling variance enters. It carries no licence to
compare refcv4b to another trained model.
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

ARMS = ("os", "os_navflip", "os_navpred", "os_navshuf", "os_navzero",
        "ha", "ha0", "ha0_ext")
LAT_CLASSES = ("lane_keep", "turn_left", "turn_right")
PRED_KEY = {"os": "lat_pred_nav_true", "os_navflip": "lat_pred_nav_flipped",
            "os_navpred": "lat_pred_nav_predicted", "os_navshuf": "lat_pred_nav_shuffled",
            "os_navzero": "lat_pred_nav_zero"}


def load(dump: str):
    out = {"g": [], "eid": [], "lat_label": []}
    for a in ARMS:
        out[a] = []
    for k in PRED_KEY:
        out["pred_" + k] = []
    for f in sorted(glob.glob(os.path.join(dump, "ep*.npz"))):
        a = np.load(f)
        b = np.load(os.path.join(dump, "decisions", os.path.basename(f)))
        n = len(a["ws"])
        out["g"].append(a["g"])
        out["eid"].append(np.full(n, int(os.path.basename(f)[2:-4]), dtype=np.int64))
        out["lat_label"].append(b["lat_label"])
        for arm in ARMS:
            out[arm].append(a[arm])
        for arm, key in PRED_KEY.items():
            out["pred_" + arm].append(b[key])
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def step_errors(pred: np.ndarray, gt: np.ndarray, dt: float):
    """Per-step |curvature err|, |heading err| and their masks, via the SHIPPED
    geometry (`ff._seq_geometry`) — never a re-implementation."""
    P = ff._seq_geometry(torch.from_numpy(pred.astype(np.float32)), dt)
    G = ff._seq_geometry(torch.from_numpy(gt.astype(np.float32)), dt)
    both = (P["valid"] & G["valid"]).numpy()
    both_pair = (P["pair_valid"] & G["pair_valid"]).numpy()
    dh = (P["heading"] - G["heading"]).numpy()
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    dk = (P["curvature"] - G["curvature"]).numpy()
    ct = (P["cross"] - G["cross"]).numpy()
    return dict(head=np.abs(dh), head_m=both, curv=np.abs(dk), curv_m=both_pair,
                cross=np.abs(ct))


def pooled(err: np.ndarray, m: np.ndarray, rows: np.ndarray) -> float:
    e, mm = err[rows], m[rows]
    d = mm.sum()
    return float((e * mm).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    D = load(args.dump)
    g, eid, lat_label = D["g"], D["eid"], D["lat_label"]
    n = g.shape[0]
    E = {a: step_errors(D[a], g, args.dt) for a in ARMS}

    R = {
        "_is": ("NP-2: PAIRED episode-cluster margins for the LATERAL family (curvature, "
                "heading, cross-track) and the TACTICAL turn recalls, on the banked "
                "navflip dump. The navpred RESULT reported these per-arm with NO paired "
                "margins and forbade quoting them as separated until this landed."),
        "evidence_class": "MEASURED (ours)",
        "tier": "T1",
        "dump": os.path.abspath(args.dump),
        "n_windows": int(n), "n_episodes": int(len(np.unique(eid))),
        "dt_s": args.dt,
        "estimator": ("paired episode-cluster bootstrap, steps RE-POOLED inside each draw "
                      "(n_boot=%d, seed=0)" % args.n_boot),
        "variance_answered": ("would another draw of EPISODES say this? — NOT another "
                              "training run (H-ESTIM-SEED-1) and NOT another inference "
                              "sample (refcv4b's decoder is deterministic at inference)"),
        "geometry_source": "taniteval.four_families._seq_geometry (imported, not re-derived)",
    }

    # per-arm point estimates, recomputed by the SAME arithmetic as ff.lateral
    allrows = np.arange(n)
    R["per_arm"] = {}
    for a in ARMS:
        lat = ff.lateral(torch.from_numpy(D[a].astype(np.float32)),
                         torch.from_numpy(g.astype(np.float32)),
                         dt=args.dt, eid=None, n_boot=0)
        R["per_arm"][a] = {
            "curvature_mae_1pm": lat["curvature_mae_1pm"],
            "heading_mae_deg": lat["heading_mae_deg"],
            "cross_mae_m": lat["cross_mae_m"],
            "_recomputed_curv_pooled": round(
                pooled(E[a]["curv"], E[a]["curv_m"], allrows), 6),
            "n_steps_curvature": lat["n_steps_curvature"],
        }
        if abs(R["per_arm"][a]["_recomputed_curv_pooled"]
               - lat["curvature_mae_1pm"]) > 1e-6:
            R["per_arm"][a]["POOLING_CONTROL"] = "FAIL — pooling differs from ff.lateral"
        else:
            R["per_arm"][a]["POOLING_CONTROL"] = "PASS"

    eps = np.unique(eid)
    rows_by_ep = {e: np.where(eid == e)[0] for e in eps}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([rows_by_ep[eps[j]] for j in
                             rng.integers(0, len(eps), len(eps))])
             for _ in range(args.n_boot)]

    def margin(a: str, b: str, field: str, mask: str) -> dict:
        pt = pooled(E[a][field], E[a][mask], allrows) - pooled(E[b][field], E[b][mask], allrows)
        bs = np.array([pooled(E[a][field], E[a][mask], r)
                       - pooled(E[b][field], E[b][mask], r) for r in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(delta=round(float(pt), 6), ci95=[round(float(lo), 6), round(float(hi), 6)],
                    separated=bool(lo > 0 or hi < 0))

    def cross_margin(a: str, b: str) -> dict:
        ea = E[a]["cross"].mean(axis=1)
        eb = E[b]["cross"].mean(axis=1)
        d = ea - eb
        bs = np.array([d[r].mean() for r in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(delta=round(float(d.mean()), 6),
                    ci95=[round(float(lo), 6), round(float(hi), 6)],
                    separated=bool(lo > 0 or hi < 0))

    def margin_isect(a: str, b: str, field: str, mask: str) -> dict:
        """Same margin on the INTERSECTION mask — a strictly paired comparison.
        `ff.lateral`'s published per-arm value masks with the ARM'S OWN path, so the
        two operands of a difference can rest on slightly different step sets
        (13,545-13,559 here). Reported beside the own-mask form, never instead of it."""
        m = E[a][mask] & E[b][mask]
        pt = pooled(E[a][field], m, allrows) - pooled(E[b][field], m, allrows)
        bs = np.array([pooled(E[a][field], m, r) - pooled(E[b][field], m, r)
                       for r in draws])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return dict(delta=round(float(pt), 6), ci95=[round(float(lo), 6), round(float(hi), 6)],
                    separated=bool(lo > 0 or hi < 0), n_steps=int(m.sum()))

    pairs = [("os_navflip", "os"), ("os_navpred", "os"), ("os_navshuf", "os"),
             ("os_navzero", "os"), ("os_navflip", "os_navzero"),
             ("os_navflip", "os_navpred"), ("os_navpred", "os_navshuf"),
             ("os_navflip", "os_navshuf"), ("os", "ha0_ext"), ("os", "ha0"),
             ("os_navpred", "ha0"), ("os_navpred", "ha0_ext"),
             ("ha0_ext", "ha0")]
    R["paired_LATERAL"] = {}
    for a, b in pairs:
        R["paired_LATERAL"]["%s__minus__%s" % (a, b)] = {
            "curvature_mae_1pm": margin(a, b, "curv", "curv_m"),
            "heading_mae_rad": margin(a, b, "head", "head_m"),
            "cross_mae_m": cross_margin(a, b),
            "curvature_mae_1pm_INTERSECTION_MASK": margin_isect(a, b, "curv", "curv_m"),
        }
    for k, v in R["paired_LATERAL"].items():
        h = v["heading_mae_rad"]
        v["heading_mae_deg"] = dict(
            delta=round(math.degrees(h["delta"]), 4),
            ci95=[round(math.degrees(h["ci95"][0]), 4),
                  round(math.degrees(h["ci95"][1]), 4)],
            separated=h["separated"])

    # ---- TACTICAL: per-class LAT recall, paired ------------------------------
    # ⛔ TRAJECTORY-DERIVED, via the programme's OWN canonical gate, exactly as
    # `ff.tactical_from_trajectory` does it. The `lat_label` field in the dump is the
    # v7.2 9-wide LATMANEUVER vocabulary with IGNORE_ID -100 on 3,666/4,823 windows —
    # a DIFFERENT vocabulary from the 3-way turnL/turnR row the RESULT reports, and
    # reading one as the other is the label-space error §8 of that RESULT retracted.
    from tanitad.refs.refc_tactical import LAT_CLASSES as LATC, factor_from_kinematics

    def lat_of(arm: str) -> np.ndarray:
        dy, dv, v0_, v1_, _ = ff.maneuver_kinematics(
            torch.from_numpy(D[arm].astype(np.float32)), args.dt)
        return factor_from_kinematics(dy, dv, v0_, v1_)[0].numpy()

    dyg, dvg, v0g, v1g, _ = ff.maneuver_kinematics(
        torch.from_numpy(g.astype(np.float32)), args.dt)
    lat_gt = factor_from_kinematics(dyg, dvg, v0g, v1g)[0].numpy()
    lat_pred = {a: lat_of(a) for a in ARMS}
    R["tactical_label_source"] = (
        "trajectory-derived via tanitad.refs.refc_tactical.factor_from_kinematics "
        "(the same gate ff.tactical_from_trajectory uses); NOT the dump's 9-wide "
        "v7.2 `lat_label` field")
    R["paired_TACTICAL_lat_recall"] = {}
    for ci, cname in enumerate(LATC):
        sub = np.where(lat_gt == ci)[0]
        blk = {"n": int(len(sub)), "per_arm": {}, "paired": {}}
        if len(sub) == 0:
            R["paired_TACTICAL_lat_recall"][cname] = blk
            continue
        for arm in ARMS:
            blk["per_arm"][arm] = round(float((lat_pred[arm][sub] == ci).mean()), 4)
        draws_c = [r[np.isin(r, sub)] for r in draws]
        draws_c = [r for r in draws_c if len(r)]
        for a, b in [("os_navflip", "os"), ("os_navpred", "os"), ("os_navpred", "os_navshuf"),
                     ("os_navshuf", "os"), ("os_navzero", "os"),
                     ("os", "ha0_ext"), ("os_navpred", "ha0_ext")]:
            ha_ = (lat_pred[a] == ci).astype(float)
            hb_ = (lat_pred[b] == ci).astype(float)
            pt = ha_[sub].mean() - hb_[sub].mean()
            bs = np.array([ha_[r].mean() - hb_[r].mean() for r in draws_c])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            blk["paired"]["%s__minus__%s" % (a, b)] = dict(
                delta=round(float(pt), 4), ci95=[round(float(lo), 4), round(float(hi), 4)],
                separated=bool(lo > 0 or hi < 0))
        R["paired_TACTICAL_lat_recall"][cname] = blk

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(json.dumps(R["per_arm"], indent=2))
    print(json.dumps(R["paired_LATERAL"], indent=2))
    print(json.dumps(R["paired_TACTICAL_lat_recall"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
