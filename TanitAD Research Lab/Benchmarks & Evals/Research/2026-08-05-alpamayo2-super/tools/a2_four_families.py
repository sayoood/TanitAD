"""Alpamayo 2 Super vs TanitAD flagship on the FOUR BINDING METRIC FAMILIES.

⛔ WHY THIS EXISTS. `a2_compare.py` reported ADE plus a speed bias. Under the
binding rule of 2026-08-02 that is **one row of four families**, and an eval that
leads with ADE is INCOMPLETE. This scores the same 39 paired clips on
LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC, per family, never pooled.

⭐ IT REUSES `taniteval.four_families`, it does not re-implement it. Re-deriving
`speed_bias` here would produce a number that is not comparable to the banked
v1arch block — and the dt contract (`_DT_CONTRACT`) is exactly the kind of detail
a re-implementation gets wrong. Both arms are on a DENSE 0.1 s grid of 20
waypoints, so `dt=0.1` is correct and is carried in the output.

⚠️ EACH ARM IS SCORED AGAINST ITS OWN GROUND TRUTH AT ITS OWN t0. Alpamayo ran at
clip t0 = 5.1 s; our window origins sit on an 0.8 s stride, so |dt| <= 0.4 s. A
RATE or a BIAS is robust to that offset; a per-clip positional difference is not.

⛔ WHAT IS NOT LIKE-FOR-LIKE is carried in the output, not in a footnote:
34.3 B vs < 0.3 B params; 6 cameras x 4 frames at 1920x1080 vs ONE 256x256 front
crop; Alpamayo truncated from its native 6.4 s / 64 wp to 2.0 s / 20 wp;
NF4-quantised, which is not an NVIDIA-validated configuration; and the clips are
PhysicalAI-AV, which Alpamayo lists as TRAINING data — contamination UNRESOLVED.
"""
from __future__ import annotations

import argparse
import json
import lzma
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "taniteval"))

DIR_YAW_RAD = 0.15                    # hierarchy.DIR_YAW_RAD — 2 s net-heading gate
R_LEFT, R_STRAIGHT, R_RIGHT = 0, 1, 2
DIRNAME = {R_LEFT: "left", R_STRAIGHT: "straight", R_RIGHT: "right"}
MAN2DIR = {0: R_STRAIGHT, 1: R_LEFT, 2: R_RIGHT, 3: R_STRAIGHT, 4: R_STRAIGHT}
MAN_NAME = ["lane_keep", "turn_left", "turn_right", "accelerate", "brake_stop"]


def net_yaw(path: np.ndarray) -> np.ndarray:
    """[n,H,2] ego-frame path -> net heading change over the horizon, radians (+ = left).

    Sums the WRAPPED per-step heading deltas rather than differencing the first and
    last tangent, so a path that swings through +/-pi does not register a fake
    reversal. This is the same net-yaw quantity `hierarchy._dir_of` gates on."""
    p = np.concatenate([np.zeros((path.shape[0], 1, 2)), path], axis=1)
    d = p[:, 1:] - p[:, :-1]
    h = np.arctan2(d[..., 1], d[..., 0])
    dh = (h[:, 1:] - h[:, :-1] + math.pi) % (2 * math.pi) - math.pi
    return dh.sum(axis=1)


def dir_of(ny: np.ndarray) -> np.ndarray:
    d = np.full(ny.shape, R_STRAIGHT, dtype=int)
    d[ny > DIR_YAW_RAD] = R_LEFT
    d[ny < -DIR_YAW_RAD] = R_RIGHT
    return d


def kappa(x: np.ndarray, y: np.ndarray) -> float | None:
    """Cohen's kappa. ⛔ On a straight-dominated corpus a raw agreement RATE is
    ~the base rate and says nothing; kappa is the honest read and is what
    `hierarchy.agree` reports alongside it."""
    labs = sorted(set(x.tolist()) | set(y.tolist()))
    n = len(x)
    if n == 0:
        return None
    po = float((x == y).mean())
    pe = sum((x == c).mean() * (y == c).mean() for c in labs)
    if abs(1.0 - pe) < 1e-9:
        return None                    # degenerate: one class only, kappa undefined
    return round((po - pe) / (1.0 - pe), 4)


def decision_block(pred: np.ndarray, gt: np.ndarray, names: dict) -> dict:
    """accuracy + kappa + per-class recall + never-predicted, the shape
    `four_families._decision_family` publishes."""
    out = {"status": "OK", "n": int(len(gt)),
           "accuracy": round(float((pred == gt).mean()), 4),
           "kappa": kappa(pred, gt), "per_class": {}}
    for c, nm in names.items():
        sel = gt == c
        out["per_class"][nm] = {
            "n_true": int(sel.sum()),
            "recall": round(float((pred[sel] == gt[sel]).mean()), 4) if sel.any() else None,
            "n_pred": int((pred == c).sum()),
        }
    out["never_predicted"] = [nm for nm, v in out["per_class"].items()
                              if v["n_true"] > 0 and v["n_pred"] == 0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpamayo-traj", required=True, help="a2t20.json.xz, [20,2] per index")
    ap.add_argument("--alpamayo-gt", required=True, help="VALIDATED egomotion GT")
    ap.add_argument("--flagship-json", required=True)
    ap.add_argument("--compare-json", required=True, help="defines the paired set")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from taniteval import four_families as ff

    A = {int(k): np.asarray(v, dtype=np.float64)
         for k, v in json.loads(lzma.open(a.alpamayo_traj).read()).items()}
    agt_j = json.load(open(a.alpamayo_gt))
    AG = {int(k): np.asarray(v, dtype=np.float64)
          for k, v in agt_j["gt_xy_by_index"].items()}
    FS = {r["i"]: r for r in json.load(open(a.flagship_json)) if "ade_2s_m" in r}
    paired = [r["i"] for r in json.load(open(a.compare_json))["rows"]]
    paired = sorted(i for i in paired if i in A and i in AG and i in FS)
    H = 20
    print(f"paired clips: {len(paired)}  horizon {H} steps @ 0.1 s = 2.0 s")

    aP = torch.tensor(np.stack([A[i][:H] for i in paired]))
    aG = torch.tensor(np.stack([AG[i][:H, :2] for i in paired]))
    fP = torch.tensor(np.stack([np.asarray(FS[i]["pred"])[:H] for i in paired]))
    fG = torch.tensor(np.stack([np.asarray(FS[i]["gt"])[:H] for i in paired]))

    DK_REASON = (
        "no lead-agent track was built for these t0-ALIGNED windows. The banked "
        "OOD-val lead block is keyed on the 0.8 s rollout window grid; Alpamayo "
        "ran at clip t0 = 5.1 s, so the two do not join. Building it needs a "
        "`build_lead_block.py` pass over obstacle.offline at these 39 origins. "
        "⛔ WORK ITEM, NOT A PASS — this is half of the LONGITUDINAL family.")

    fam = {}
    for arm, P, G in (("alpamayo2-super", aP, aG), ("flagship-v1arch-v2bal-30k", fP, fG)):
        lon = ff.longitudinal(P, G, dt=0.1)
        lon["distance_keeping"] = {"status": "UNAVAILABLE", "reason": DK_REASON, "n": 0}
        lat = ff.lateral(P, G, dt=0.1)

        # --- TACTICAL, trajectory-derived so BOTH arms are on the same instrument ---
        d_pred, d_gt = dir_of(net_yaw(P.numpy())), dir_of(net_yaw(G.numpy()))
        tac = {"status": "OK",
               "source": "trajectory-derived direction (net yaw over 2 s, gate "
                         f"{DIR_YAW_RAD} rad) — the SAME instrument for both arms, so "
                         "the comparison does not depend on either arm's declared head",
               "executed_maneuver_vs_gt": decision_block(d_pred, d_gt, DIRNAME)}
        fam[arm] = {"LONGITUDINAL": lon, "LATERAL": lat, "TACTICAL": tac}

    # --- TACTICAL, DECLARED head: only our arm emits one in this pass ---
    man = np.array([FS[i]["man"] for i in paired
                    if FS[i].get("man") is not None])
    if len(man) == len(paired):
        d_man = np.array([MAN2DIR[int(m)] for m in man])
        d_driven = dir_of(net_yaw(fP.numpy()))
        d_gt_fs = dir_of(net_yaw(fG.numpy()))
        k = kappa(d_man, d_driven)
        fam["flagship-v1arch-v2bal-30k"]["TACTICAL"]["declared_maneuver_head"] = {
            "status": "OK",
            "distribution": {MAN_NAME[c]: int((man == c).sum()) for c in range(5)},
            "never_predicted": [MAN_NAME[c] for c in range(5) if (man == c).sum() == 0],
            "declared_vs_driven_kappa": k,
            # ⛔ kappa ~ 0 means the declared manoeuvre and the path actually driven are
            # unrelated. That is a decision error a scalar ADE cannot see, which is the
            # whole reason this family is binding.
            "verdict": (None if k is None else
                        "DECORATIVE — declared manoeuvre is ~unrelated to the driven path"
                        if k < 0.1 else "WEAK" if k < 0.4 else "SUBSTANTIAL"),
            "declared_vs_gt": decision_block(d_man, d_gt_fs, DIRNAME),
        }
    fam["alpamayo2-super"]["TACTICAL"]["declared_maneuver_head"] = {
        "status": "UNAVAILABLE",
        "reason": ("Alpamayo ships a META-ACTION task head, but this pass ran the "
                   "TRAJECTORY task only. Its declared manoeuvre therefore does not "
                   "exist in these artifacts. ⛔ WORK ITEM: re-run the same 39 clips "
                   "under the meta-action task. Its Chain-of-Causation text was "
                   "captured and is a natural-language tactical rationale, but a "
                   "free-text string is NOT a scored decision and is not counted here."),
        "n": 0}

    # --- STRATEGIC ---
    route = [FS[i].get("route") for i in paired]
    fam["flagship-v1arch-v2bal-30k"]["STRATEGIC"] = {
        "status": "UNAVAILABLE",
        "reason": ("the arm emits a route class, but the only route LABEL available on "
                   "PhysicalAI-AV is derived from the ego's own future path — which "
                   "cannot separate 'took the left branch' from 'drifted left on a "
                   "curving road', and cannot see whether a branch existed at all. "
                   "MEASURED 2026-08-03: flagship v1's route head is an exact bijection "
                   "of the nav it is fed (369/369, 81/81) and scored 1.0000 — an ECHO of "
                   "its own input read as skill. ⛔ Scoring it against a future-derived "
                   "label would republish that artifact. The admissible instrument is "
                   "`taniteval.strategic_optionset` over MAP-DERIVED option sets, and "
                   "PhysicalAI-AV ships no map (five independent probes; the card says "
                   "verbatim 'we do not include open maps data'). ⇒ needs AlpaSim or an "
                   "external corpus. WORK ITEM."),
        "n": 0,
        "route_class_distribution": {DIRNAME[c]: int(sum(1 for r in route if r == c))
                                     for c in DIRNAME} if all(r is not None for r in route) else None,
    }
    fam["alpamayo2-super"]["STRATEGIC"] = {
        "status": "UNAVAILABLE",
        "reason": ("Alpamayo emits no route/goal class in the trajectory task. Its "
                   "Chain-of-Causation string IS a strategic rationale and was captured "
                   "for all 39 clips, but there is no reference label to score it "
                   "against on this corpus — same map-absence blocker as our arm. "
                   "⛔ WORK ITEM, and it is the SAME work item, which is itself the "
                   "finding: the strategic level is unmeasured for BOTH arms."),
        "n": 0}

    out = {
        "_binding_rule": ("Sayed 2026-08-02 — every eval reports LONGITUDINAL, LATERAL, "
                          "TACTICAL and STRATEGIC. ADE alone is INCOMPLETE. Families are "
                          "reported PER FAMILY and never pooled into one score."),
        "n_paired": len(paired), "horizon_s": H * 0.1, "dt_s": 0.1,
        "_grid": ("both arms on a DENSE 0.1 s grid of 20 waypoints, so dt=0.1 is the "
                  "true spacing — NOT the sparse 4-waypoint (0.5 s) view whose misread "
                  "inflated every published speed by 5x and accel by 25x"),
        "_alignment": ("each arm scored against ITS OWN ground truth at ITS OWN t0; "
                       "Alpamayo at clip t0 5.1 s, ours on an 0.8 s stride, |dt| <= 0.4 s. "
                       "Biases and rates are robust to that; per-clip positional "
                       "differences are not."),
        "_not_like_for_like": ("34.3B vs <0.3B params; 6 cameras x 4 frames at 1920x1080 "
                               "vs ONE 256x256 front crop; Alpamayo truncated 6.4s/64wp "
                               "-> 2.0s/20wp; NF4-quantised, NOT an NVIDIA-validated "
                               "configuration"),
        "_contamination": ("clips are PhysicalAI-AV, which Alpamayo lists as TRAINING "
                           "data. Overlap with its training split is UNRESOLVED, so any "
                           "Alpamayo advantage may be contamination rather than capability."),
        "_estimator": ("unweighted mean over 39 paired clips. ⛔ This is NOT the "
                       "decision-grade estimator — with one window per clip the "
                       "episode-cluster bootstrap degenerates to the i.i.d. case, so no "
                       "CI is quoted rather than quoting a wrong one. A decision-grade "
                       "read needs many windows per episode."),
        "_evidence_class": "MEASURED (ours) — artifacts listed in _artifacts",
        "families": fam,
        "clip_indices": paired,
    }
    json.dump(out, open(a.out, "w"), indent=1)

    # ---- console read, per family, never pooled ----
    AA, FF_ = fam["alpamayo2-super"], fam["flagship-v1arch-v2bal-30k"]
    print(f"\n{'':30s} {'ALPAMAYO 2 SUPER':>18s} {'TANITAD FLAGSHIP':>18s}")
    print("-- LONGITUDINAL " + "-" * 52)
    for lab, k in (("speed MAE (m/s)", "speed_mae_mps"),
                   ("speed BIAS (m/s)  +=too fast", "speed_bias_mps"),
                   ("accel MAE (m/s^2)", "accel_mae_mps2"),
                   ("along BIAS (m)  +=ahead", "along_bias_m"),
                   ("along final bias (m)", "along_final_bias_m")):
        print(f"{lab:30s} {AA['LONGITUDINAL'][k]:>18} {FF_['LONGITUDINAL'][k]:>18}")
    for lab, k in (("ego progress ratio (GT=1.0)", "progress_ratio_mean"),
                   ("ego progress |error|", "progress_error_mean"),
                   ("under-progress rate", "under_progress_rate")):
        print(f"{lab:30s} {str(AA['LONGITUDINAL']['ego_progress'].get(k)):>18} "
              f"{str(FF_['LONGITUDINAL']['ego_progress'].get(k)):>18}")
    print(f"{'distance keeping':30s} {'UNAVAILABLE':>18} {'UNAVAILABLE':>18}   <- WORK ITEM")
    print("-- LATERAL " + "-" * 57)
    for lab, k in (("heading MAE (deg)", "heading_mae_deg"),
                   ("yaw-rate MAE (deg/s)", "yaw_rate_mae_degps"),
                   ("curvature MAE (1/m)", "curvature_mae_1pm"),
                   ("curvature BIAS (1/m)", "curvature_bias_1pm"),
                   ("cross-track MAE (m)", "cross_mae_m"),
                   ("cross-track BIAS (m) +=left", "cross_bias_m")):
        print(f"{lab:30s} {str(AA['LATERAL'][k]):>18} {str(FF_['LATERAL'][k]):>18}")
    print("-- TACTICAL " + "-" * 56)
    ea, ef = (AA["TACTICAL"]["executed_maneuver_vs_gt"],
              FF_["TACTICAL"]["executed_maneuver_vs_gt"])
    print(f"{'executed manoeuvre acc':30s} {ea['accuracy']:>18} {ef['accuracy']:>18}")
    print(f"{'executed manoeuvre kappa':30s} {str(ea['kappa']):>18} {str(ef['kappa']):>18}")
    dm = FF_["TACTICAL"].get("declared_maneuver_head", {})
    print(f"{'declared head kappa vs driven':30s} {'UNAVAILABLE':>18} "
          f"{str(dm.get('declared_vs_driven_kappa')):>18}")
    if dm.get("verdict"):
        print(f"{'  -> flagship verdict':30s} {dm['verdict']}")
        print(f"{'  -> never predicted':30s} {dm.get('never_predicted')}")
    print("-- STRATEGIC " + "-" * 55)
    print(f"{'':30s} {'UNAVAILABLE':>18} {'UNAVAILABLE':>18}   <- same blocker, BOTH arms")
    print(f"\n[out] {a.out}")


if __name__ == "__main__":
    main()
