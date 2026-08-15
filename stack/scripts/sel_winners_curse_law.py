"""E-WC — the winner's-curse LAW, measured 0-GPU on a banked fan.

WHY THIS EXISTS
---------------
W7-FULL (registry §1.14) refuted argmin-over-a-large-fan absolutely: selected
**3.3348** against a **0.4505** gate over a fan whose oracle is **0.1273**, with
the argmin's error-rank at **132 of 256 — the median**. The registry's binding
consequence names the remedy as *"a noise-robust rule (top-m aggregation /
sharpened cost), pre-registered before it is used"*, and W7-PROG's binding
consequence adds *"the cost needs a GOAL-CONDITIONED component"*.

Neither consequence has been MEASURED as a rule. W7's own banked per-window
arrays (`w7_eval_windows.pt`) live on pod4/pod5, which are gone — so the
`w7_selection_rules.json` sweep could only compute top-m CEILINGS and MEANS, not
the medoid, and no goal rule was ever tried.

This module re-opens the question on a fan that IS in the repo:
``…/incoming/2026-08-03-esel-verdict/raw/fan_refined_refc-xl-30k.pt`` —
881 windows x 256 candidates x 4 waypoints (steps 5/10/15/20 = 0.5-2.0 s), with
GT, CV, v0, per-window episode ids, and THREE scores, one of which
(``cons_score``) is a world-model roll-consistency score by its own provenance
string. That makes it an INDEPENDENT replication surface for the W7 mechanism.

WHAT IT MEASURES (four arms, all on the same windows)
-----------------------------------------------------
A. **The N-law.** Sub-sample the fan to N candidates and re-select. If the
   winner's curse is real, selection degrades with N while the oracle improves.
   This is the measurement that licenses (or refutes) v6's ``n_candidates=8``.
B. **The goal-requirement curve.** Select by distance from the candidate's
   endpoint to a GOAL POINT of controlled accuracy (GT endpoint + isotropic
   noise sigma). Answers the design question *"how accurate must the goal head
   be for goal-conditioned selection to be worth having?"*
   ⛔ EVIDENCE CLASS: a REQUIREMENT CURVE, not a capability number — the sigma=0
   end is GT-derived and is NOT deployable. The deployable point on the same
   axis is the CV-goal arm, which uses only v0 + heading.
C. **The aggregators.** argmin/argmax vs top-m centroid vs top-m medoid — the
   half of the registry's remedy that has never been computed on trajectories.
D. **The composition.** goal-prefilter THEN learned score (and the reverse) —
   the shape v6 actually proposes.

NEGATIVE CONTROLS, reported alongside and never omitted
-------------------------------------------------------
* ``random`` — uniform pick. The null every rule must beat.
* ``goal_echo`` — the goal replaced by the CORPUS-MEAN endpoint, i.e. a goal
  carrying zero per-window information. This is the nav-echo test applied to
  selection: if a goal with no information selects nearly as well, the goal term
  is not doing the work its name claims.

Intervals are the **paired episode-cluster bootstrap** (`taniteval.ci`);
``overlapping_holdout_se`` is used nowhere.

USAGE (0 GPU, ~1 min)
    python stack/scripts/sel_winners_curse_law.py --fan <fan.pt> --out <out.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

#: waypoint index used as the "endpoint" a goal point refers to (last wp).
ENDPOINT = -1


def _load_ci():
    """`taniteval.ci` lives in the sibling `taniteval/` package, not in `stack/`."""
    here = Path(__file__).resolve()
    for up in here.parents:
        cand = up / "taniteval" / "taniteval" / "ci.py"
        if cand.exists():
            sys.path.insert(0, str(up / "taniteval"))
            from taniteval import ci  # noqa: E402
            return ci
    raise ImportError("taniteval/taniteval/ci.py not found above " + str(here))


def per_candidate_err(fan: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """[W, C, T, 2] vs [W, T, 2] -> [W, C] mean displacement error."""
    return np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)


def norm_rank_of(err: np.ndarray, chosen: np.ndarray) -> float:
    """Mean error-RANK of the chosen candidate, normalised to [0, 1].

    0.0 = always the true best, 0.5 = indistinguishable from a random pick,
    1.0 = always the true worst. THIS, not ADE, is the primary endpoint for a
    ranking claim — W7-PROG's precedent (*"the mechanism under test is a ranking
    claim"*). Normalising by C-1 is what makes N=8 and N=256 comparable at all.
    """
    w, c = err.shape
    rank = np.argsort(np.argsort(err, axis=1), axis=1)
    return float(rank[np.arange(w), chosen].mean() / max(c - 1, 1))


def lower_tail_hit(err: np.ndarray, chosen: np.ndarray, q: float = 0.1) -> float:
    """P(chosen candidate is in the true-best q-quantile of its own window).

    The paper's §3.12 names LOWER-TAIL DEPENDENCE as the quantity that governs
    selection (*"rank correlation is a bulk statistic; argmin is an extreme
    one"*). Under a random pick this is exactly q, so the null is known and does
    not have to be simulated.
    """
    w, c = err.shape
    k = max(int(round(q * c)), 1)
    thresh = np.partition(err, k - 1, axis=1)[:, k - 1]
    return float((err[np.arange(w), chosen] <= thresh).mean())


def _rule_stats(err: np.ndarray, chosen: np.ndarray) -> dict:
    w = err.shape[0]
    return {"ade": float(err[np.arange(w), chosen].mean()),
            "norm_err_rank": norm_rank_of(err, chosen),
            "lower_tail_p10": lower_tail_hit(err, chosen, 0.1)}


# --------------------------------------------------------------------------- #
# selection rules                                                             #
# --------------------------------------------------------------------------- #
def sel_score_max(score: np.ndarray) -> np.ndarray:
    return score.argmax(axis=1)


def sel_score_min(score: np.ndarray) -> np.ndarray:
    return score.argmin(axis=1)


def sel_goal(fan: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """argmin over candidates of ||endpoint_c - goal||.

    ⭐ The structural property that matters: ``goal`` does NOT depend on which
    candidate is being scored. A self-consistency cost's minimiser is a
    near-stationary candidate (inaction minimises it); a candidate-INDEPENDENT
    reference has no such degenerate minimiser. That is the whole reason the
    published loops we copied (V-JEPA 2-AC, DINO-WM) minimise distance to a
    goal, and it is the term W7 dropped.
    """
    return np.linalg.norm(fan[:, :, ENDPOINT] - goal[:, None], axis=-1).argmin(1)


def sel_topm_medoid(fan: np.ndarray, order: np.ndarray, m: int) -> np.ndarray:
    """Among the m best-scoring candidates, the one closest to their centroid.

    A REAL plan (unlike the centroid, which is an average of trajectories and
    need not be dynamically realisable), and robust to a single score outlier —
    the aggregation half of the registry's remedy, which has never been computed
    on trajectories because the arrays it needed died with the pods.
    """
    w = fan.shape[0]
    idx = order[:, :m]                                          # [W, m]
    sub = np.take_along_axis(fan, idx[:, :, None, None], axis=1)
    cen = sub.mean(1)                                           # [W, T, 2]
    dist = np.linalg.norm(sub - cen[:, None], axis=-1).mean(-1)  # [W, m]
    return idx[np.arange(w), dist.argmin(1)]


def sel_topm_centroid_err(fan: np.ndarray, gt: np.ndarray,
                          order: np.ndarray, m: int) -> np.ndarray:
    """Per-window error of the top-m CENTROID trajectory (not a candidate)."""
    idx = order[:, :m]
    sub = np.take_along_axis(fan, idx[:, :, None, None], axis=1)
    cen = sub.mean(1)
    return np.linalg.norm(cen - gt, axis=-1).mean(-1)


# --------------------------------------------------------------------------- #
def arm_a_n_law(err, scores, rng, ns, repeats) -> dict:
    """A — selection quality as a function of the candidate-set size N."""
    w, c = err.shape
    out = {}
    for n in ns:
        if n > c:
            continue
        acc = {k: {"ade": [], "norm_err_rank": [], "lower_tail_p10": []}
               for k in list(scores) + ["random"]}
        orc, fmean = [], []
        for _ in range(repeats):
            cols = np.stack([rng.choice(c, size=n, replace=False)
                             for _ in range(w)])                # [W, n]
            e = np.take_along_axis(err, cols, axis=1)
            orc.append(float(e.min(1).mean()))
            fmean.append(float(e.mean()))
            for name, (score, sense) in scores.items():
                s = np.take_along_axis(score, cols, axis=1)
                ch = s.argmax(1) if sense == "max" else s.argmin(1)
                for k, v in _rule_stats(e, ch).items():
                    acc[name][k].append(v)
            ch = rng.integers(0, n, size=w)
            for k, v in _rule_stats(e, ch).items():
                acc["random"][k].append(v)
        out[str(n)] = {
            "oracle_ade": float(np.mean(orc)),
            "fan_mean_ade": float(np.mean(fmean)),
            "rules": {k: {kk: float(np.mean(vv)) for kk, vv in d.items()}
                      for k, d in acc.items()},
            "repeats": repeats}
    return out


def arm_b_goal_curve(err, fan, gt, cv, rng, sigmas, ns) -> dict:
    """B — the goal-requirement curve, plus its two negative controls."""
    w, c = err.shape
    gt_end = gt[:, ENDPOINT]
    echo = np.repeat(gt_end.mean(0, keepdims=True), w, axis=0)   # zero info
    out = {"_note": "sigma is the isotropic 1-sigma error, in METRES, of a goal "
                    "point referred to the last waypoint (2.0 s on this fan). "
                    "sigma=0 is GT-derived and NOT deployable — this is a "
                    "REQUIREMENT CURVE, not a capability number.",
           "cv_goal": {}, "goal_echo": {}, "sigma": {}}
    for n in ns:
        if n > c:
            continue
        cols = np.stack([rng.choice(c, size=n, replace=False) for _ in range(w)])
        e = np.take_along_axis(err, cols, axis=1)
        f = np.take_along_axis(fan, cols[:, :, None, None], axis=1)
        out["cv_goal"][str(n)] = _rule_stats(e, sel_goal(f, cv[:, ENDPOINT]))
        out["goal_echo"][str(n)] = _rule_stats(e, sel_goal(f, echo))
        for s in sigmas:
            g = gt_end + rng.normal(0.0, s, size=gt_end.shape)
            out["sigma"].setdefault(f"{s:g}", {})[str(n)] = \
                _rule_stats(e, sel_goal(f, g))
    return out


def arm_c_aggregators(err, fan, gt, scores, ms) -> dict:
    """C — argmin/argmax vs top-m medoid vs top-m centroid, on the full fan."""
    out = {}
    for name, (score, sense) in scores.items():
        order = np.argsort(-score if sense == "max" else score, axis=1)
        d = {"argbest": _rule_stats(err, order[:, 0]), "topm_medoid": {},
             "topm_centroid_ade": {}, "topm_ceiling_ade": {},
             "topm_mean_ade": {}}
        for m in ms:
            if m > err.shape[1]:
                continue
            d["topm_medoid"][str(m)] = _rule_stats(
                err, sel_topm_medoid(fan, order, m))
            d["topm_centroid_ade"][str(m)] = float(
                sel_topm_centroid_err(fan, gt, order, m).mean())
            sub = np.take_along_axis(err, order[:, :m], axis=1)
            d["topm_ceiling_ade"][str(m)] = float(sub.min(1).mean())
            d["topm_mean_ade"][str(m)] = float(sub.mean())
        out[name] = d
    return out


def arm_d_composition(err, fan, cv, scores, ms) -> dict:
    """D — goal PREFILTER then learned score, and the reverse ordering.

    This is the shape v6 proposes, and the reverse arm is what distinguishes
    "the goal supplies admissibility" from "the goal supplies the ranking".
    """
    w, c = err.shape
    gd = np.linalg.norm(fan[:, :, ENDPOINT] - cv[:, ENDPOINT][:, None], axis=-1)
    g_order = np.argsort(gd, axis=1)
    out = {}
    for name, (score, sense) in scores.items():
        s = -score if sense == "max" else score          # lower == better
        s_order = np.argsort(s, axis=1)
        fwd, rev = {}, {}
        for m in ms:
            if m > c:
                continue
            keep = g_order[:, :m]
            sub = np.take_along_axis(s, keep, axis=1)
            fwd[str(m)] = _rule_stats(err, keep[np.arange(w), sub.argmin(1)])
            keep2 = s_order[:, :m]
            sub2 = np.take_along_axis(gd, keep2, axis=1)
            rev[str(m)] = _rule_stats(err, keep2[np.arange(w), sub2.argmin(1)])
        out[name] = {"goal_prefilter_then_score": fwd,
                     "score_prefilter_then_goal": rev}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("sel_winners_curse_law")
    ap.add_argument("--fan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="4,8,16,32,64,128,256")
    ap.add_argument("--ms", default="2,3,4,8,16,32")
    ap.add_argument("--sigmas", default="0,0.25,0.5,1,2,4,8,16")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)

    import torch
    ci = _load_ci()
    d = torch.load(a.fan, map_location="cpu", weights_only=False)
    fan = d["fan"].float().numpy()
    gt = d["gt"].float().numpy()
    cv = d["cv"].float().numpy()
    eid = list(d["eid"])
    err = per_candidate_err(fan, gt)
    w, c = err.shape

    scores = {}
    for key, sense, alias in (("logits", "max", "shipped_selector"),
                              ("refined_logits", "max", "refined_conf"),
                              ("cons_score", "max", "wm_roll_consistency")):
        if key in d:
            scores[alias] = (d[key].float().numpy(), sense)

    ns = [int(x) for x in a.ns.split(",") if x]
    ms = [int(x) for x in a.ms.split(",") if x]
    sigmas = [float(x) for x in a.sigmas.split(",") if x != ""]
    rng = np.random.default_rng(a.seed)

    res = {
        "item": "E-WC — the winner's-curse law, 0-GPU on a banked fan",
        "_evidence_class": "MEASURED (ours) — direct re-analysis of a banked "
                           "in-repo fan; NO model, NO GPU, NO re-inference",
        "_class": "EXPLORATORY — a single banked fan (REF-C-XL, 2 s / 4 wp). "
                  "The N-law and the rank statistics are structural claims; "
                  "the ABSOLUTE ADE values are this fan's, not v6's.",
        "fan": {"path": str(a.fan), "n_windows": w, "n_candidates": c,
                "wp_steps": list(d.get("wp_steps", [])),
                "ckpt": str(d.get("ckpt", "")),
                "ckpt_step": int(d.get("ckpt_step", -1)),
                "nav_mode": str(d.get("nav_mode", "")),
                "n_episodes": len(set(eid)),
                "cons_provenance": str(d.get("cons_provenance", ""))[:400]},
        "headline": {
            "oracle_ade": float(err.min(1).mean()),
            "fan_mean_ade": float(err.mean()),
            "shipped_selector_ade": float(err[np.arange(w), d["sel"].numpy()].mean()),
            "cv_ade": float(np.linalg.norm(cv - gt, axis=-1).mean()),
        },
        "A_n_law": arm_a_n_law(err, scores, np.random.default_rng(a.seed),
                               ns, a.repeats),
        "B_goal_curve": arm_b_goal_curve(err, fan, gt, cv,
                                         np.random.default_rng(a.seed + 1),
                                         sigmas, ns),
        "C_aggregators": arm_c_aggregators(err, fan, gt, scores, ms),
        "D_composition": arm_d_composition(err, fan, cv, scores, ms),
    }

    # ---- paired episode-cluster bootstrap on the headline deltas ------------
    def perwin(chosen):
        return err[np.arange(w), chosen]

    order_ship = np.argsort(-scores["shipped_selector"][0], axis=1)
    ship = perwin(order_ship[:, 0])
    goal_cv = perwin(sel_goal(fan, cv[:, ENDPOINT]))
    med8 = perwin(sel_topm_medoid(fan, order_ship, min(8, c)))
    comp = None
    gd = np.linalg.norm(fan[:, :, ENDPOINT] - cv[:, ENDPOINT][:, None], axis=-1)
    keep = np.argsort(gd, axis=1)[:, :min(8, c)]
    sub = np.take_along_axis(-scores["shipped_selector"][0], keep, axis=1)
    comp = perwin(keep[np.arange(w), sub.argmin(1)])
    roll = perwin(scores["wm_roll_consistency"][0].argmax(1)) \
        if "wm_roll_consistency" in scores else None
    rng2 = np.random.default_rng(a.seed + 7)
    rnd = perwin(rng2.integers(0, c, size=w))
    gsig = {}
    for s in (0.5, 1.0, 2.0):
        g = gt[:, ENDPOINT] + rng2.normal(0.0, s, size=gt[:, ENDPOINT].shape)
        gsig[f"goal_sigma{s:g}_minus_shipped"] = \
            ci.paired_episode_cluster_bootstrap(perwin(sel_goal(fan, g)), ship,
                                                eid, n_boot=a.n_boot)
    res["paired_ci"] = {
        "_estimator": "paired episode-cluster bootstrap (taniteval.ci); "
                      "overlapping_holdout_se used NOWHERE",
        "_sign": "delta = ARM - shipped_selector; POSITIVE means the arm is "
                 "WORSE (higher ADE)",
        "goal_cv_minus_shipped": ci.paired_episode_cluster_bootstrap(
            goal_cv, ship, eid, n_boot=a.n_boot),
        "top8medoid_minus_shipped": ci.paired_episode_cluster_bootstrap(
            med8, ship, eid, n_boot=a.n_boot),
        "goalprefilter8_then_score_minus_shipped":
            ci.paired_episode_cluster_bootstrap(comp, ship, eid,
                                                n_boot=a.n_boot),
        **gsig,
        "shipped_ade": ci.episode_cluster_bootstrap(ship, eid,
                                                    n_boot=a.n_boot),
    }
    if roll is not None:
        res["paired_ci"]["wm_rollcost_minus_shipped"] = \
            ci.paired_episode_cluster_bootstrap(roll, ship, eid,
                                                n_boot=a.n_boot)
        res["paired_ci"]["wm_rollcost_minus_random"] = \
            ci.paired_episode_cluster_bootstrap(roll, rnd, eid,
                                                n_boot=a.n_boot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("headline", "paired_ci")}, indent=1))
    print("EWC_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
