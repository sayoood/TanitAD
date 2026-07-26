"""H2 classifier — STEP 4 (dev box, CPU): the HELD-OUT evaluation.

Runs on the dev box on purpose: every interval is produced by the repo's own
`taniteval/taniteval/ci.py`, not by a pod copy, so the estimator provenance is exact and the whole
evaluation reproduces from the repo with no pod.

Metric space, stated exactly (C1 — a metric NAME is not a metric DEFINITION)
---------------------------------------------------------------------------
The primary unit is the **(camera, frame) pair**: 2 side cameras x N held-out frames. A positive is
a pair `(X, t)` with `L2_trigger(X, t) = 1`. Firing the LEFT camera when the RIGHT one was needed is
a MISS and a wasted activation, and this unit is the only one that scores it that way.

    B  = the compute budget = EXTRA CAMERA ACTIVATIONS PER FRAME.
         B = 0.05 means 1.05 camera passes per frame on average.
         In (camera, frame) space the corresponding firing rate is B / 2.

Operating point: `theta*(B)` is the threshold at which the **TRAIN out-of-fold** score achieves
exactly B, and it is applied to the held-out side UNCHANGED (`PRE_REGISTRATION.md §5`). Nothing in
the selection path reads a held-out metric.

usage:  python h2c_eval.py --run <dir with scores_*.npz> --out <artifacts dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from h2c_stats import (ESTIMATOR, average_precision, boot_stat,  # noqa: E402
                       paired_stat, roc_auc)

BUDGETS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
B_STAR = 0.05
N_RANDOM_SEEDS = 200


# ---------------------------------------------------------------------------- reducers
def _ap(y, s, **_):
    return average_precision(y, s)


def _rate(fire, **_):
    return float(np.mean(fire))


def _recall(y, fire, **_):
    return float((y * fire).sum() / max(y.sum(), 1e-12))


def _precision(y, fire, **_):
    return float((y * fire).sum() / max(fire.sum(), 1e-12))


def _lift(y, fire, **_):
    p = (y * fire).sum() / max(fire.sum(), 1e-12)
    b = y.mean()
    return float(p / b) if b > 0 else float("nan")


def opt(res, k):
    return {kk: res[kk] for kk in ("point", "lo", "hi") if kk in res} | {"n_ep": res["n_episodes"],
                                                                        "what": k}


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--run", required=True)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--boot", type=int, default=2000)
    ap_.add_argument("--cost", default=None,
                     help="cost_model.json from h2c_cost.py (pod2) — makes the compute saving "
                          "MEASURED rather than a camera-count proxy")
    args = ap_.parse_args()
    os.makedirs(args.out, exist_ok=True)
    HO = np.load(os.path.join(args.run, "scores_heldout.npz"))
    OO = np.load(os.path.join(args.run, "scores_oof_train.npz"))

    def cf(d, key=None, arr=None):
        """frame-level -> (camera, frame) pair space: left rows then right rows."""
        a = d[key] if arr is None else arr
        return np.concatenate([a[:, 0], a[:, 1]]) if a.ndim == 2 else np.concatenate([a, a])

    out = {"metric_space": "(camera, frame) pairs — 2 side cameras x N frames; a positive is "
                           "L2_trigger(X,t)=1; firing the wrong camera is a MISS",
           "budget_definition": "B = extra camera activations per frame; camera-frame firing "
                                "rate = B/2", "estimator": ESTIMATOR, "B_star": B_STAR}

    # ------------------------------------------------------------------ arrays
    eid = cf(HO, arr=HO["clip"])
    y = cf(HO, "Y").astype(float)
    eid_tr = cf(OO, arr=OO["clip"])
    y_tr = cf(OO, "Y").astype(float)
    EX, EXtr = HO["EX"], OO["EX"]
    r2 = cf(HO, arr=EX[:, 2]).astype(bool)                 # behavioural response
    junction = cf(HO, arr=EX[:, 3]).astype(bool)
    enc_seen = cf(HO, arr=EX[:, 7]).astype(bool)
    v_ho, a_ho = cf(HO, arr=HO["ego_v"]), cf(HO, arr=HO["alon_pre"])
    v_tr, a_tr = cf(OO, arr=OO["ego_v"]), cf(OO, arr=OO["alon_pre"])
    n_frames_ho = HO["Y"].shape[0]
    n_frames_tr = OO["Y"].shape[0]

    arms = {}
    for k in HO.files:
        if k.startswith("s__") and k.endswith("__trigger"):
            arms[k[3:].replace("__trigger", "")] = (cf(HO, k), cf(OO, k))
    out["arms_trained"] = sorted(arms)
    out["universe"] = {
        "heldout_frames": int(n_frames_ho), "heldout_camera_frames": int(y.size),
        "heldout_positives": int(y.sum()), "heldout_base_rate": float(y.mean()),
        "heldout_clips": int(len(np.unique(HO["clip"]))),
        "heldout_positive_clips": int(len(np.unique(HO["clip"][HO["Y"].max(1) > 0]))),
        "train_frames": int(n_frames_tr), "train_positives": int(y_tr.sum()),
        "train_positive_clips": int(len(np.unique(OO["clip"][OO["Y"].max(1) > 0]))),
        "behavioural_slice_positives": int((y * r2).sum()),
    }

    # ------------------------------------------------------------------ AP / AUC, all arms
    disc = {}
    for a, (s_ho, _s_tr) in arms.items():
        disc[a] = {"AP": boot_stat(_ap, eid, n_boot=args.boot, y=y, s=s_ho),
                   "AUROC": float(roc_auc(y, s_ho))}
    # non-learned ego arms as SCORES for the same curve (simplest monotone ego signals)
    ego_scores = {"heur_speed": v_ho, "heur_decel": -a_ho}
    for a, s in ego_scores.items():
        disc[a] = {"AP": boot_stat(_ap, eid, n_boot=args.boot, y=y, s=s),
                   "AUROC": float(roc_auc(y, s))}
    disc["base_rate"] = float(y.mean())
    out["discrimination"] = disc

    # AP vs CHANCE, done properly. Comparing an AP interval to the FULL-SAMPLE base rate is not a
    # test: the base rate is itself a random quantity under episode resampling. A constant score
    # has AP exactly equal to the base rate WITHIN EACH DRAW, so the paired delta against it is
    # the correct "is this above chance?" statistic.
    chance = np.zeros_like(y)
    out["paired_AP_vs_chance"] = {
        a: _paired_ap(y, s, chance, eid, args.boot)
        for a, s in [(k, v[0]) for k, v in arms.items()] + list(ego_scores.items())}

    # paired AP deltas — the pre-registered secondary
    prim = "head_img_ego"
    out["paired_AP_deltas"] = {}
    for b in [a for a in arms if a != prim] + list(ego_scores):
        s_b = arms[b][0] if b in arms else ego_scores[b]
        out["paired_AP_deltas"][f"{prim} - {b}"] = _paired_ap(
            y, arms[prim][0], s_b, eid, args.boot)

    # ------------------------------------------------------------------ heuristic rule family
    vq = np.quantile(v_tr, np.linspace(0.0, 0.98, 25))
    aq = np.quantile(a_tr, np.linspace(0.02, 1.0, 25))

    def heur_fire(vt, at, v, a):
        return (v >= vt) & (a <= at)

    def pick_heuristic(frame_budget):
        """Best (v_t, a_t) on TRAIN: maximise TRAIN recall s.t. TRAIN frame firing rate <= budget.

        Fitted on TRAIN only and to the SAME objective the head optimises, so this is the
        strongest honest non-learned opponent, not a straw man. Falls back to the tightest rule in
        the grid when no rule fits the budget (rather than returning None and hiding the case)."""
        best, tightest = (None, -1.0), (None, 1e9)
        for vt in vq:
            for at in aq:
                f = heur_fire(vt, at, v_tr, a_tr)
                rate = float(f.mean())
                if rate < tightest[1]:
                    tightest = ((float(vt), float(at)), rate)
                if rate > frame_budget + 1e-12:
                    continue
                r = _recall(y_tr, f.astype(float))
                if r > best[1]:
                    best = ((float(vt), float(at)), r)
        return best if best[0] is not None else (tightest[0], float("nan"))

    # ------------------------------------------------------------------ the trade-off curve
    curve, oppt = [], {}
    for B in BUDGETS:
        row = {"B_extra_cams_per_frame": B, "camera_frame_rate_target": B / 2.0}
        # --- learned arms: theta* fixed on TRAIN OOF, applied unchanged
        for a, (s_ho, s_tr) in arms.items():
            th = float(np.quantile(s_tr, 1.0 - B / 2.0))
            fire = (s_ho >= th).astype(float)
            row[a] = {
                "theta_star_from_TRAIN": round(th, 6),
                "realised_camera_frame_rate": float(fire.mean()),
                "realised_extra_cams_per_frame": float(fire.sum() / n_frames_ho),
                "recall": _recall(y, fire), "precision": _precision(y, fire),
                "precision_lift_over_base": _lift(y, fire),
                "recall_behavioural_slice": float((y * r2 * fire).sum()
                                                  / max((y * r2).sum(), 1e-12)),
            }
        # --- ego heuristic, both cameras (an ego rule has NO direction, so it must wake both)
        (vt, at), _ = pick_heuristic(B / 2.0)
        fh = heur_fire(vt, at, v_ho, a_ho).astype(float)
        row["heur_ego_both"] = {"v_thresh": vt, "a_pre_thresh": at,
                                "realised_camera_frame_rate": float(fh.mean()),
                                "realised_extra_cams_per_frame": float(fh.sum() / n_frames_ho),
                                "recall": _recall(y, fh), "precision": _precision(y, fh),
                                "precision_lift_over_base": _lift(y, fh),
                                "recall_behavioural_slice": float((y * r2 * fh).sum()
                                                                  / max((y * r2).sum(), 1e-12))}
        # --- ego heuristic, one camera at random (half the cost, half the expected recall)
        (vt1, at1), _ = pick_heuristic(B)
        fh1 = heur_fire(vt1, at1, v_ho, a_ho).astype(float)
        row["heur_ego_one"] = {"v_thresh": vt1, "a_pre_thresh": at1,
                               "realised_camera_frame_rate": float(fh1.mean() / 2.0),
                               "realised_extra_cams_per_frame": float(fh1.sum() / 2 / n_frames_ho),
                               "recall_expected": 0.5 * _recall(y, fh1),
                               "note": "fires ONE side camera uniformly at random; recall is the "
                                       "expectation over that coin"}
        # --- random at matched rate
        rs = []
        for sd in range(N_RANDOM_SEEDS):
            rng = np.random.default_rng(1000 + sd)
            u = rng.random(y.size)
            fr = (u >= np.quantile(u, 1.0 - B / 2.0)).astype(float)
            rs.append(_recall(y, fr))
        row["random_at_rate"] = {"recall_mean": float(np.mean(rs)),
                                 "recall_p2.5": float(np.percentile(rs, 2.5)),
                                 "recall_p97.5": float(np.percentile(rs, 97.5)),
                                 "n_seeds": N_RANDOM_SEEDS,
                                 "analytic_expectation": B / 2.0,
                                 "precision_expected": float(y.mean())}
        row["always"] = {"realised_extra_cams_per_frame": 2.0, "recall": 1.0,
                         "precision": float(y.mean())}
        row["never"] = {"realised_extra_cams_per_frame": 0.0, "recall": 0.0, "precision": None}
        curve.append(row)
        if abs(B - B_STAR) < 1e-12:
            oppt = row
    out["tradeoff_curve"] = curve

    # ------------------------------------------------------------------ the operating point
    # theta* is the PRE-REGISTERED value fixed on TRAIN out-of-fold (§5) and is NOT re-chosen here.
    # The BASELINES are then matched to the head's REALISED held-out firing rate, so the comparison
    # is compute-matched. Matching a firing rate reads the held-out SCORE distribution only —
    # never the held-out TARGETS — so no test metric enters any selection path.
    B = B_STAR
    th = float(np.quantile(arms[prim][1], 1.0 - B / 2.0))
    fire_h = (arms[prim][0] >= th).astype(float)
    rate_h = float(fire_h.mean())
    (vt, at), _ = pick_heuristic(B / 2.0)
    fire_e = heur_fire(vt, at, v_ho, a_ho).astype(float)
    # rate-matched heuristic: same held-out camera-frame rate as the head, chosen on TRAIN among
    # rules whose TRAIN rate is admissible, then reported with its own realised rate
    (vt_m, at_m), _ = pick_heuristic(rate_h)
    fire_em = heur_fire(vt_m, at_m, v_ho, a_ho).astype(float)
    rng = np.random.default_rng(1000)
    u = rng.random(y.size)
    fire_r = (u >= np.quantile(u, 1.0 - max(rate_h, 1e-9))).astype(float)

    op = {"B_preregistered": B, "theta_star_from_TRAIN_oof": th,
          "realised_head_camera_frame_rate": rate_h,
          "realised_head_extra_cams_per_frame": float(fire_h.sum() / n_frames_ho),
          "calibration_transfer": {"target_camera_frame_rate": B / 2.0,
                                   "realised": rate_h,
                                   "ratio_realised_over_target": rate_h / (B / 2.0)},
          "heuristic_thresholds_preregistered": {"v": vt, "a_pre": at},
          "heuristic_thresholds_rate_matched": {"v": vt_m, "a_pre": at_m}}
    # descriptive companions: the other learned arms at the SAME pre-registered budget. They do
    # not enter the verdict (the PRIMARY is fixed at head_img_ego) but the reader must see them.
    others = {}
    for a in arms:
        if a == prim:
            continue
        th_a = float(np.quantile(arms[a][1], 1.0 - B / 2.0))
        others[a] = (arms[a][0] >= th_a).astype(float)
        op[f"theta_star_{a}"] = th_a
    for nm, f in ([("head_img_ego", fire_h)] + sorted(others.items())
                  + [("heur_ego_both", fire_e), ("heur_ego_both_rate_matched", fire_em),
                     ("random_at_rate", fire_r)]):
        op[nm] = {
            "rate": boot_stat(_rate, eid, n_boot=args.boot, fire=f),
            "extra_cams_per_frame": float(f.sum() / n_frames_ho),
            "recall": boot_stat(_recall, eid, n_boot=args.boot, y=y, fire=f),
            "precision": boot_stat(_precision, eid, n_boot=args.boot, y=y, fire=f),
            "precision_lift_over_base": boot_stat(_lift, eid, n_boot=args.boot, y=y, fire=f),
            "recall_behavioural_slice": float((y * r2 * f).sum() / max((y * r2).sum(), 1e-12)),
            "missed_positives": int(y.sum() - (y * f).sum()),
            "missed_behavioural_positives": int((y * r2).sum() - (y * r2 * f).sum()),
        }
    op["paired_recall_deltas"] = {
        "head_img_ego - heur_ego_both": _paired_recall(y, fire_h, fire_e, eid, args.boot),
        "head_img_ego - heur_ego_both_rate_matched":
            _paired_recall(y, fire_h, fire_em, eid, args.boot),
        "head_img_ego - random_at_rate": _paired_recall(y, fire_h, fire_r, eid, args.boot),
    }
    for a, f in others.items():                       # descriptive, not part of the verdict
        op["paired_recall_deltas"][f"{a} - heur_ego_both_rate_matched"] = \
            _paired_recall(y, f, fire_em, eid, args.boot)
        op["paired_recall_deltas"][f"{a} - random_at_rate"] = \
            _paired_recall(y, f, fire_r, eid, args.boot)
    op["random_seed_spread"] = oppt["random_at_rate"] if oppt else None

    # ---- the EFFICIENCY claim, with an interval, NET of the gate's own cost -----------------
    # cost(gated) = 1 encoder pass (front, always) + head (always) + extra camera passes;
    # cost(always-on-K) = K encoder passes. The head's cost is INSIDE the numerator on purpose.
    ho = None
    if args.cost and os.path.exists(args.cost):
        cm = json.load(open(args.cost))
        ho = {"macs": cm["analytic_macs"]["head_over_encoder"],
              "wallclock": cm["head_over_encoder_wallclock"]}
    rate_ci = op[prim]["rate"]

    def sav(rate_cf, K, overhead):
        return 1.0 - (1.0 + 2.0 * rate_cf + overhead) / K

    eff = {"head_over_encoder": ho,
           "extra_cams_per_frame": {"point": 2 * rate_ci["point"], "lo": 2 * rate_ci["lo"],
                                    "hi": 2 * rate_ci["hi"]},
           "note": "interval propagated from the episode-cluster bootstrap on the firing rate; "
                   "the saving is monotone decreasing in the rate so the CI maps directly",
           # the ORACLE: a gate that fires exactly on the label and nowhere else. It bounds what
           # ANY classifier can save, and it is the row that shows the saving is set by the POLICY
           # SHAPE, not by classifier quality.
           "oracle_extra_cams_per_frame": float(y.sum() / n_frames_ho),
           "oracle_saving_vs_always_on_3": 1 - (1 + float(y.sum() / n_frames_ho)) / 3,
           "oracle_saving_vs_always_on_7": 1 - (1 + float(y.sum() / n_frames_ho)) / 7}
    for K in (3, 7):
        for kind, o in (("macs", ho["macs"] if ho else 0.0),
                        ("wallclock", ho["wallclock"] if ho else 0.0),
                        ("camera_count_proxy_head_free", 0.0)):
            eff[f"saving_vs_always_on_{K}__{kind}"] = {
                "point": sav(rate_ci["point"], K, o),
                "lo": sav(rate_ci["hi"], K, o), "hi": sav(rate_ci["lo"], K, o)}
        eff[f"saving_vs_always_on_{K}__never_escalate"] = 1.0 - 1.0 / K
        eff[f"saving_vs_always_on_{K}__always_escalate"] = 1.0 - 3.0 / K
    op["efficiency"] = eff
    out["operating_point"] = op

    # ------------------------------------------------------------------ C12 2x2 decomposition
    # FIRST the LABEL's own structure, before any model: which conjunct actually carries the
    # composite? C12 is binding precisely because a null on the composite says nothing about
    # which half failed — but here the answer is available from the label geometry alone.
    t_off_f = EX[:, 0].astype(bool)
    t_seen_f = EX[:, 1].astype(bool)
    trig_f = HO["Y"].max(1) > 0
    c12_struct = {
        "frames": int(len(trig_f)),
        "T_off_rate": float(t_off_f.mean()), "T_off_n": int(t_off_f.sum()),
        "T_seen_rate": float(t_seen_f.mean()), "T_seen_n": int(t_seen_f.sum()),
        "trigger_rate_frame_level": float(trig_f.mean()), "trigger_n": int(trig_f.sum()),
        "P_trigger_given_T_off": float(trig_f[t_off_f].mean()) if t_off_f.sum() else None,
        "P_trigger_given_T_seen": float(trig_f[t_seen_f].mean()) if t_seen_f.sum() else None,
        "reading": "the composite is carried by whichever conjunct has P(trigger|conjunct) ~ 1 "
                   "AND a comparable rate; the other clause is then near-vacuous on this corpus",
    }
    out["c12_label_structure"] = c12_struct
    c12 = {}
    for tgt, col in (("T_off", 0), ("T_seen", 1)):
        k = f"s__head_img_ego__{tgt}"
        if k not in HO.files:
            continue
        s = HO[k][:, 0]
        yy = EX[:, col].astype(float)
        c12[tgt] = {"base_rate": float(yy.mean()),
                    "AP": boot_stat(_ap, HO["clip"], n_boot=args.boot, y=yy, s=s),
                    "AUROC": float(roc_auc(yy, s)),
                    "AP_over_base": float(average_precision(yy, s) / max(yy.mean(), 1e-12))}
        # For a high-base-rate conjunct (T_seen fires on ~95 % of frames) AP is uninformative —
        # score the RARE complement, which is the event that actually carries information.
        if yy.mean() > 0.5:
            c12[tgt]["complement"] = {
                "name": f"NOT_{tgt}", "base_rate": float(1 - yy.mean()),
                "AP": boot_stat(_ap, HO["clip"], n_boot=args.boot, y=1 - yy, s=-s),
                "AP_over_base": float(average_precision(1 - yy, -s)
                                      / max(1 - yy.mean(), 1e-12))}
    out["c12_conjuncts"] = c12

    # ------------------------------------------------------------------ sensitivities
    sens = {}
    m = ~enc_seen
    if m.sum() and y[m].sum() > 0:
        sens["encoder_unseen_clips"] = {
            "n_camera_frames": int(m.sum()), "n_positives": int(y[m].sum()),
            "n_clips": int(len(np.unique(HO["clip"][EX[:, 7] == 0]))),
            "AP": boot_stat(_ap, eid[m], n_boot=args.boot, y=y[m], s=arms[prim][0][m]),
            "base_rate": float(y[m].mean()),
            "AP_heur_speed": boot_stat(_ap, eid[m], n_boot=args.boot, y=y[m], s=v_ho[m])}
    for nm, mm in (("junction_in", junction), ("junction_out", ~junction)):
        if mm.sum() and y[mm].sum() > 0:
            sens[nm] = {"n_positives": int(y[mm].sum()), "base_rate": float(y[mm].mean()),
                        "AP": boot_stat(_ap, eid[mm], n_boot=args.boot, y=y[mm],
                                        s=arms[prim][0][mm])}
    # residual scope (E0's genuine off-front 36.4 %) — descriptive, the label is NOT separated there
    y_res = cf(HO, arr=np.stack([EX[:, 5], EX[:, 6]], 1)).astype(float)
    if y_res.sum() > 0:
        sens["residual_scope_target"] = {
            "n_positives": int(y_res.sum()), "base_rate": float(y_res.mean()),
            "AP_of_primary_head": boot_stat(_ap, eid, n_boot=args.boot, y=y_res,
                                            s=arms[prim][0])}
    out["sensitivities"] = sens

    # ------------------------------------------------------------------ THE VERDICT, mechanically
    # Evaluated by code against `PRE_REGISTRATION.md §7`, so it is not a judgement made after
    # seeing the numbers. Both outcomes were committed in advance.
    d_rand = op["paired_recall_deltas"]["head_img_ego - random_at_rate"]
    d_heur = op["paired_recall_deltas"]["head_img_ego - heur_ego_both_rate_matched"]
    ap_ego_best = max((k for k in ("heur_speed", "heur_decel")),
                      key=lambda k: disc[k]["AP"]["point"])
    d_ap = _paired_ap(y, arms[prim][0], ego_scores[ap_ego_best], eid, args.boot)
    beats_rand = bool(d_rand["favours_a"])
    beats_heur = bool(d_heur["favours_a"])
    ap_sep = bool(d_ap["favours_a"])
    hw_exceeds = all((not d["separated"]) and
                     ((d["hi"] - d["lo"]) / 2.0) > abs(d["delta"]) for d in (d_rand, d_heur))
    if beats_rand and beats_heur and ap_sep:
        verdict = "A"
    elif hw_exceeds:
        verdict = "UNDERPOWERED"
    else:
        verdict = "B"
    out["verdict"] = {
        "outcome": verdict,
        "rule": "PRE_REGISTRATION.md 7, evaluated in code",
        "beats_random_at_matched_rate": beats_rand,
        "beats_ego_heuristic_at_matched_rate": beats_heur,
        "AP_separated_from_best_nonlearned_ego_score": ap_sep,
        "best_nonlearned_ego_score": ap_ego_best,
        "delta_vs_random": d_rand, "delta_vs_heuristic": d_heur, "delta_AP_vs_ego": d_ap,
        "power_note": ("held-out positive CLUSTERS = %d; the label's OWN effect is not separated "
                       "on this subset (see subset_lift.json) — a non-separation here is "
                       "UNPOWERED, not refuted" % out["universe"]["heldout_positive_clips"]),
    }
    json.dump(out, open(os.path.join(args.out, "h2c_results.json"), "w"), indent=2, default=float)
    print("VERDICT:", json.dumps(out["verdict"], indent=2, default=float))
    print(json.dumps({"universe": out["universe"],
                      "AP": {a: d["AP"]["point"] if isinstance(d, dict) and "AP" in d else d
                             for a, d in disc.items()},
                      "operating_point": {k: v for k, v in op.items()
                                          if k in ("head_img_ego", "heur_ego_both",
                                                   "random_at_rate", "paired_recall_deltas")},
                      "c12": c12}, indent=2, default=float))


def _paired_ap(y, sa, sb, eid, boot):
    from h2c_stats import _draws, episode_index
    uniq, idx = episode_index(eid)
    pa, pb = average_precision(y, sa), average_precision(y, sb)
    d = []
    for sel in _draws(uniq, idx, boot, 0):
        va, vb = average_precision(y[sel], sa[sel]), average_precision(y[sel], sb[sel])
        if np.isfinite(va) and np.isfinite(vb):
            d.append(va - vb)
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5]) if d.size > 50 else (np.nan, np.nan)
    return {"AP_a": round(pa, 6), "AP_b": round(pb, 6), "delta": round(pa - pb, 6),
            "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "separated": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
            "favours_a": bool(np.isfinite(lo) and lo > 0),
            "n_episodes": int(len(uniq)), "n_draws_used": int(d.size), "estimator": ESTIMATOR}


def _paired_recall(y, fa, fb, eid, boot):
    from h2c_stats import _draws, episode_index
    uniq, idx = episode_index(eid)
    pa, pb = _recall(y, fa), _recall(y, fb)
    d = []
    for sel in _draws(uniq, idx, boot, 0):
        ys = y[sel]
        if ys.sum() <= 0:
            continue
        d.append(_recall(ys, fa[sel]) - _recall(ys, fb[sel]))
    d = np.asarray(d)
    lo, hi = np.percentile(d, [2.5, 97.5]) if d.size > 50 else (np.nan, np.nan)
    return {"recall_a": round(pa, 6), "recall_b": round(pb, 6), "delta": round(pa - pb, 6),
            "lo": round(float(lo), 6), "hi": round(float(hi), 6),
            "separated": bool(np.isfinite(lo) and (lo > 0 or hi < 0)),
            "favours_a": bool(np.isfinite(lo) and lo > 0),
            "n_episodes": int(len(uniq)), "n_draws_used": int(d.size), "estimator": ESTIMATOR}


if __name__ == "__main__":
    main()
