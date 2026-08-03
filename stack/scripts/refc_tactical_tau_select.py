"""D-TAC1b — choose the prior-corrected decode's tau WITHOUT fitting on the set
it is reported on. 0 GPU: a pure re-analysis of the banked probe substrate.

Pre-registration: ``Project Steering/PREREG_D-TAC1B_TAU_SELECTION_AND_F1_ARM.md``
(criteria, tau grid, denominators and both outcomes fixed BEFORE this ran).

THE DEFECT
============================================================================
``DTAC1_RESULTS.md`` section 2.3 reports ``brake_stop`` recall 0.026 -> 0.503 at
tau = 0.5 with no retrain. The report flags the problem itself: **tau was read
off the val frontier**, which is fitting on the eval set. The adversarial pass
(R6) found the second half of the same defect: the *prior* the adjustment
divides by is ``label_marginal`` — literally the VAL label marginal — and the
frontier is not robust to it (a +-25 % brake-prior perturbation swings
accelerate recall 5.6x).

So BOTH halves of the decode rule ``argmax(log P_lon - tau * log pi)`` were
estimated on the reported set: ``tau`` AND ``pi``.

WHAT THIS SCRIPT DOES INSTEAD
============================================================================
LEAVE-ONE-EPISODE-OUT selection. For each val episode e, fit ``(pi_e, tau_e)``
on the windows NOT in e and decode only e's windows with it. Every reported
window is decoded by a rule that never saw it, nor any window from its clip —
the episode is the independent unit (windows inside one clip are strongly
dependent, which is the whole reason ``taniteval.ci`` resamples episodes).

⚠️ THIS IS NOT A TRAIN-SELECTED TAU, and the output never calls it one. A
train-selected tau needs REF-C's posteriors on TRAIN windows; MEASURED
2026-08-03, no train episode cache is reachable (5 probes, see the
pre-registration section 1.1), and ``refc-base`` predates ``factored_maneuver``
so it carries no EMA prior buffer either. ``--train-substrate`` implements that
path for the day one exists; without it the script reports LOEO and says so.

What LOEO removes: "tau was read off the number being reported".
What it does NOT remove: selection and reporting share a distribution.

REPORTING RULES CARRIED FROM THE ADVERSARIAL RECORD
============================================================================
* **R3** every class row carries recall AND PRECISION AND F1. A decision rule
  whose entire mechanism is moving the boundary toward the rare class cannot be
  judged on recall.
* **R2 / R10** every table is emitted on BOTH denominators: ALL windows, and the
  REPRESENTABLE subset excluding the windows whose longitudinal class the 5-way
  LABEL destroys into a turn. At tau=0 every correct ``accelerate`` prediction
  fell on a destroyed window, so the ALL number is not a recoverability claim.
* **R7** intervals are the PAIRED episode-cluster bootstrap, never unpaired,
  never ``overlapping_holdout_se``.
* **R8** the lateral readout is CLASSIFICATION and is labelled as such; it is not
  the LATERAL kinematics family.

Usage (dev box or any non-training box; CPU, seconds):
  PYTHONPATH=<repo>/stack python scripts/refc_tactical_tau_select.py \\
      --substrate dtac1_substrate_refc-base-30k.pt \\
      --out dtac1_tau_selection_refc-base-30k.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from tanitad.refs import refc_tactical as tac

#: The parent's published grid, unchanged (PREREG D-TAC1b section 1.3).
TAU_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)

#: The parent's published tau frontier (DTAC1_RESULTS.md section 2.3), hard-coded
#: so the re-derivation is a CHECK and not a copy. Quote a run directory, not a
#: number: if these do not reproduce from the substrate, the substrate and the
#: published table are not the same run and nothing here is quotable.
PUBLISHED_FRONTIER = {
    0.0: {"accuracy": 0.7045, "macro_recall": 0.3621},
    0.25: {"accuracy": 0.6782, "macro_recall": 0.4125},
    0.5: {"accuracy": 0.5953, "macro_recall": 0.4588},
    0.75: {"accuracy": 0.5132, "macro_recall": 0.4709},
    1.0: {"accuracy": 0.4267, "macro_recall": 0.4761},
    1.25: {"accuracy": 0.3644, "macro_recall": 0.4666},
    2.0: {"accuracy": 0.1510, "macro_recall": 0.3624},
}


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------

def _paired_bootstrap():
    """The ONE admissible interval. ``taniteval`` is a sibling of ``stack``, so a
    bare ``stack`` PYTHONPATH cannot import it — extend the path rather than
    degrade to a weaker estimator. A missing estimator FAILS LOUD: an interval
    without its estimator is not quotable (CLAUDE.md)."""
    try:
        from taniteval.ci import paired_episode_cluster_bootstrap
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "taniteval"))
        from taniteval.ci import paired_episode_cluster_bootstrap
    return paired_episode_cluster_bootstrap


def _packed(pred, true) -> np.ndarray:
    """Pack (true, pred) into ONE float per window: ``true * N_LON + pred``.

    ``paired_episode_cluster_bootstrap`` hands its reducer a 1-D array of the
    resampled windows' values, so a statistic like macro-recall — which is not a
    per-window mean — needs both labels to survive the resample. Integers 0..8
    are exact in float64, so this is lossless, and it lets macro-recall and
    macro-F1 use the SAME estimator as an accuracy rather than inventing an
    interval of their own (the mistake ``overlapping_holdout_se`` was)."""
    return (np.asarray(true, dtype=np.float64) * tac.N_LON
            + np.asarray(pred, dtype=np.float64))


def _unpack(v):
    v = np.asarray(v, dtype=np.int64)
    return v % tac.N_LON, v // tac.N_LON          # (pred, true)


def _macro_recall(v) -> float:
    pred, true = _unpack(v)
    rs = [float((pred[true == k] == k).mean()) for k in range(tac.N_LON)
          if int((true == k).sum())]
    return float(np.mean(rs)) if rs else float("nan")


def _macro_f1(v) -> float:
    pred, true = _unpack(v)
    fs = []
    for k in range(tac.N_LON):
        tp = int(((pred == k) & (true == k)).sum())
        fs.append(0.0 if tp == 0 else
                  2 * tp / float(int((pred == k).sum()) + int((true == k).sum())))
    return float(np.mean(fs))


def _accuracy(v) -> float:
    pred, true = _unpack(v)
    return float((pred == true).mean())


CRITERIA = {"macro_recall": _macro_recall, "macro_f1": _macro_f1}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def class_report(pred, true, names=tac.LON_CLASSES) -> dict:
    """Per class: n_true, n_pred, recall, PRECISION, f1 (R3 — precision is never
    optional for a rule that works by moving the boundary toward a rare class)."""
    pred, true = np.asarray(pred), np.asarray(true)
    n = len(names)
    conf = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred):
        conf[int(t), int(p)] += 1
    per = {}
    for k, nm in enumerate(names):
        n_true, n_pred, tp = int(conf[k].sum()), int(conf[:, k].sum()), int(conf[k, k])
        rec = tp / n_true if n_true else None
        prec = tp / n_pred if n_pred else None
        f1 = (2 * tp / (n_true + n_pred)) if (n_true + n_pred) else None
        per[nm] = {"n_true": n_true, "n_pred": n_pred,
                   "recall": None if rec is None else round(rec, 4),
                   "precision": None if prec is None else round(prec, 4),
                   "f1": None if f1 is None else round(f1, 4)}
    v = _packed(pred, true)
    return {"n": int(len(true)), "per_class": per, "confusion": conf.tolist(),
            "accuracy": round(_accuracy(v), 4),
            "macro_recall": round(_macro_recall(v), 4),
            "macro_f1": round(_macro_f1(v), 4),
            "never_predicted": [nm for nm, d in per.items() if d["n_pred"] == 0]}


def decode(log_lon: torch.Tensor, log_prior: torch.Tensor, tau: float):
    return tac.logit_adjust(log_lon, log_prior, tau).argmax(-1).numpy()


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def select_tau(log_lon, lon, mask, criterion, grid=TAU_GRID):
    """Fit (log_prior, tau) on the windows where ``mask`` is True. Returns
    ``(log_prior, tau, score_at_tau, per_tau_scores)``. Ties -> the SMALLEST tau
    (the least aggressive rule that attains the optimum)."""
    sel_lon = lon[mask]
    prior = tac.class_log_prior(sel_lon, tac.N_LON)
    scores = {}
    for t in grid:
        p = decode(log_lon[mask], prior, t)
        scores[t] = criterion(_packed(p, sel_lon.numpy()))
    best = max(grid, key=lambda t: (scores[t], -t))
    return prior, float(best), scores[best], scores


def loeo(log_lon, lon, eid, criterion, grid=TAU_GRID) -> dict:
    """Leave-one-EPISODE-out selection. Each window is decoded by a rule fitted
    on every episode except its own."""
    eid = np.asarray(eid)
    episodes = sorted(set(eid.tolist()))
    pred = np.zeros(len(lon), dtype=np.int64)
    chosen, priors = {}, {}
    for e in episodes:
        te = eid == e
        tr = ~te
        # FOLD-DISJOINTNESS CONTROL: the whole claim rests on this, so assert it.
        assert not set(eid[tr].tolist()) & {e}, f"fold leak on {e}"
        assert tr.sum() > 0 and te.sum() > 0, f"degenerate fold {e}"
        prior, tau, _, _ = select_tau(log_lon, lon, torch.as_tensor(tr),
                                      criterion, grid)
        pred[te] = decode(log_lon[torch.as_tensor(te)], prior, tau)
        chosen[e] = tau
        priors[e] = [round(float(x), 6) for x in prior.exp()]
    taus = np.array(list(chosen.values()), dtype=float)
    return {"pred": pred, "tau_per_fold": chosen,
            "tau_stability": {
                "n_folds": len(episodes),
                "modal_tau": float(max(set(taus.tolist()),
                                       key=taus.tolist().count)),
                "unique_taus": sorted(set(taus.tolist())),
                "frac_at_modal": round(float((taus == max(
                    set(taus.tolist()), key=taus.tolist().count)).mean()), 4),
                "min": float(taus.min()), "max": float(taus.max())},
            "prior_spread": {
                "min": [round(float(x), 6) for x in
                        np.array(list(priors.values())).min(0)],
                "max": [round(float(x), 6) for x in
                        np.array(list(priors.values())).max(0)],
                "_classes": list(tac.LON_CLASSES)}}


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def controls(sub, log_lon, lon, lat, man5, eid) -> dict:
    """⛔ RUN AND REPORTED FIRST. A metric that cannot separate the right answer
    from a wrong one certifies nothing."""
    out = {}

    # (1) COMPONENT vs FAMILY self-consistency — mandatory by brief.
    out["self_consistency_collapse"] = {
        "_what": "collapse(lat, lon) must equal the 5-way label elementwise",
        "agreement": round(float((tac.collapse(lat, lon) == man5).float().mean()), 4),
        "n": int(len(man5))}
    banked = sub.get("man_banked")
    if banked is not None:
        have = banked >= 0
        out["self_consistency_label_source"] = {
            "_what": "derived 5-way vs the epcache's banked `maneuvers`",
            "n_with_banked": int(have.sum()),
            "agreement": round(float((man5[have] == banked[have]).float().mean()), 4)
            if bool(have.any()) else None}

    # (2) Re-derive the PUBLISHED frontier from the substrate (not copy it).
    val_prior = tac.class_log_prior(lon, tac.N_LON)
    rederived, worst = {}, 0.0
    for t, want in PUBLISHED_FRONTIER.items():
        rep = class_report(decode(log_lon, val_prior, t), lon.numpy())
        rederived[str(t)] = {"accuracy": rep["accuracy"],
                             "macro_recall": rep["macro_recall"]}
        worst = max(worst, abs(rep["accuracy"] - want["accuracy"]),
                    abs(rep["macro_recall"] - want["macro_recall"]))
    out["frontier_rederivation"] = {
        "_what": ("the parent's published 8-row frontier recomputed FROM THE "
                  "SUBSTRATE with prior = the val label marginal (R6: that is "
                  "what it used). A mismatch means the substrate and the "
                  "published table are different runs."),
        "max_abs_deviation": round(float(worst), 4),
        "matches_published_to_4dp": bool(worst <= 1.5e-4),
        "rederived": rederived,
        "val_label_marginal": {nm: round(float(val_prior.exp()[k]), 4)
                               for k, nm in enumerate(tac.LON_CLASSES)}}

    # (3) UNIFORM prior is inert at every tau.
    uni = torch.full((tac.N_LON,), -math.log(tac.N_LON))
    raw = decode(log_lon, uni, 0.0)
    out["uniform_prior_is_inert"] = {
        "_what": "subtracting a constant from every logit cannot move an argmax",
        "identical_at_every_tau": bool(all(
            np.array_equal(decode(log_lon, uni, t), raw) for t in TAU_GRID))}

    # (4) tau = 0 is the identity.
    out["tau0_is_the_raw_argmax"] = {
        "identical": bool(np.array_equal(decode(log_lon, val_prior, 0.0),
                                         log_lon.argmax(-1).numpy()))}

    # (5) SHUFFLED — the whole LOEO pipeline on permuted logits must be chance.
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(log_lon.shape[0], generator=g)
    sh = loeo(log_lon[perm], lon, eid, _macro_recall)
    rep = class_report(sh["pred"], lon.numpy())
    out["shuffled_pipeline"] = {
        "_what": ("logits permuted across windows, then the FULL out-of-fold "
                  "selection. Must land at chance; if it does not, the selection "
                  "is reading the class prior, not the model."),
        "macro_recall": rep["macro_recall"], "macro_f1": rep["macro_f1"],
        "accuracy": rep["accuracy"], "chance_macro_recall": round(1 / tac.N_LON, 4),
        "modal_tau": sh["tau_stability"]["modal_tau"]}
    return out


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--substrate", required=True,
                    help="banked probe substrate (.pt) for the REPORTED set")
    ap.add_argument("--train-substrate", default=None,
                    help="banked substrate over TRAIN windows. When present the "
                         "honest selection is train-selected and the LOEO run "
                         "becomes a secondary. MEASURED 2026-08-03: no train "
                         "epcache is reachable, so this is normally absent.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=4000)
    args = ap.parse_args(argv)

    sub = torch.load(args.substrate, map_location="cpu", weights_only=False)
    log5, lon, lat = sub["log5"].float(), sub["lon"], sub["lat"]
    man5, eid = sub["man5"], list(sub["eid"])
    _, log_lon = tac.invert_man5(log5)
    lon_np = lon.numpy()

    # windows the 5-way LABEL can represent at all (R2 / R10)
    turning = (man5 == tac.TURN_LEFT) | (man5 == tac.TURN_RIGHT)
    destroyed = (turning & (lon != tac.LON_STEADY)).numpy()
    representable = ~destroyed

    res = {
        "experiment": "D-TAC1b — out-of-fold tau selection for the "
                      "prior-corrected LON decode",
        "prereg": "Project Steering/PREREG_D-TAC1B_TAU_SELECTION_AND_F1_ARM.md",
        "evidence_class": "MEASURED (ours)",
        "substrate": {"path": str(args.substrate),
                      "ckpt": str(sub.get("ckpt")),
                      "val_dir": str(sub.get("val_dir")),
                      "stride": sub.get("stride"),
                      "decoder_steps": sub.get("decoder_steps"),
                      "n_windows": int(log5.shape[0]),
                      "n_episodes": len(set(eid))},
        "estimator": ("taniteval.ci.paired_episode_cluster_bootstrap, unit = "
                      "episode, callable reducers; overlapping_holdout_se is "
                      "NEVER used"),
        "denominators": {
            "ALL": int(len(lon_np)),
            "REPRESENTABLE": int(representable.sum()),
            "label_destroyed_excluded": int(destroyed.sum()),
            "_why": ("R2: at tau=0 every correct `accelerate` prediction fell on "
                     "a label-destroyed window, so the ALL-denominator number is "
                     "not a claim about recoverability")},
    }

    # ---- CONTROLS FIRST ---------------------------------------------------
    res["CONTROLS"] = controls(sub, log_lon, lon, lat, man5, eid)

    # ---- the honest selection --------------------------------------------
    res["selection_protocol"] = {
        "kind": "leave-one-EPISODE-out within val",
        "⛔_this_is_not_train_selected": (
            "a train-selected tau needs REF-C posteriors on TRAIN windows. "
            "MEASURED 2026-08-03: no train episode cache is reachable (5 probes, "
            "PREREG D-TAC1b section 1.1) and refc-base predates factored_maneuver "
            "so it carries no EMA prior buffer. LOEO removes 'tau was read off "
            "the number being reported'; it does NOT remove 'selection and "
            "reporting share a distribution'."),
        "tau_grid": list(TAU_GRID),
        "criteria_fixed_in_advance": list(CRITERIA),
        "train_substrate_supplied": bool(args.train_substrate)}

    boot = _paired_bootstrap()
    baseline = decode(log_lon, tac.class_log_prior(lon, tac.N_LON), 0.0)
    res["arms"] = {}
    for cname, crit in CRITERIA.items():
        fold = loeo(log_lon, lon, eid, crit)
        pred = fold["pred"]
        # ORACLE: (pi, tau) fitted on ALL of val — what the parent published.
        all_mask = torch.ones(len(lon_np), dtype=torch.bool)
        o_prior, o_tau, _, o_scores = select_tau(log_lon, lon, all_mask, crit)
        o_pred = decode(log_lon, o_prior, o_tau)

        arm = {"criterion": cname,
               "out_of_fold": {
                   "ALL": class_report(pred, lon_np),
                   "REPRESENTABLE": class_report(pred[representable],
                                                 lon_np[representable])},
               "tau_stability": fold["tau_stability"],
               "prior_spread_across_folds": fold["prior_spread"],
               "val_optimal_oracle": {
                   "tau": o_tau,
                   "_what": "(pi, tau) fitted on ALL of val — the parent's rule",
                   "ALL": class_report(o_pred, lon_np),
                   "REPRESENTABLE": class_report(o_pred[representable],
                                                 lon_np[representable]),
                   "score_per_tau": {str(k): round(float(v), 4)
                                     for k, v in o_scores.items()}},
               "tau0_baseline": {
                   "ALL": class_report(baseline, lon_np),
                   "REPRESENTABLE": class_report(baseline[representable],
                                                 lon_np[representable])}}

        # COST OF HONESTY, pre-committed as a reported quantity
        for dn, msk in (("ALL", slice(None)), ("REPRESENTABLE", representable)):
            oof = arm["out_of_fold"][dn][cname]
            orc = arm["val_optimal_oracle"][dn][cname]
            arm.setdefault("cost_of_honesty", {})[dn] = {
                "val_optimal": orc, "out_of_fold": oof,
                "cost": round(orc - oof, 4),
                "cost_pct": (round(100.0 * (orc - oof) / orc, 2)
                             if orc else None)}

        # PAIRED intervals on the SAME windows (R7).
        # BOTH criteria are reported for EVERY arm, not just the one that
        # selected it: a rule chosen to maximise macro-recall must still be shown
        # its macro-F1, or the trade it made is invisible — and the
        # pre-registration's own trigger is phrased on macro-recall.
        reducers = {**CRITERIA, "accuracy": _accuracy}
        arm["ci_paired"] = {}
        for label, a, b in (("oof_vs_tau0", pred, baseline),
                            ("oof_vs_val_optimal", pred, o_pred)):
            for dn, msk in (("ALL", slice(None)),
                            ("REPRESENTABLE", representable)):
                e = list(np.asarray(eid)[msk]) if dn != "ALL" else eid
                pa = _packed(a[msk], lon_np[msk])
                pb = _packed(b[msk], lon_np[msk])
                for rname, red in reducers.items():
                    arm["ci_paired"][f"{label}::{dn}::{rname}"] = boot(
                        pa, pb, e, n_boot=args.n_boot, reduce=red)
        res["arms"][cname] = arm

    # ---- LATERAL readout: CLASSIFICATION, explicitly not kinematics (R8) ---
    log_lat, _ = tac.invert_man5(log5)
    res["lateral_readout_CLASSIFICATION_not_kinematics"] = {
        "_what": ("the lateral axis recovered from the same 5-way head. This is "
                  "a CLASSIFICATION readout and is NOT the LATERAL metric family "
                  "(heading / curvature / yaw-rate / cross-track), which needs a "
                  "predicted trajectory the substrate does not carry."),
        **class_report(log_lat.argmax(-1).numpy(), lat.numpy(), tac.LAT_CLASSES)}

    # ---- the binding four-family statement, per family, with reasons ------
    res["FOUR_METRIC_FAMILIES"] = {
        "_rule": "CLAUDE.md — every eval reports four families IN ADDITION to ADE",
        "TACTICAL": "MEASURED in full above (recall + precision + F1 + confusion, "
                    "both denominators, paired episode-cluster bootstrap).",
        "LONGITUDINAL": {
            "delta": "EXACTLY ZERO — the tau patch is a post-hoc argmax on "
                     "already-emitted logits and cannot reach the trajectory. "
                     "Pinned by tests/test_refc_tactical.py::"
                     "test_man_prior_tau_cannot_move_the_trajectory (traj / "
                     "anchor_logits / sel_idx / ... bit-identical across tau).",
            "level_not_computable": ("target-speed accuracy and headway / "
                                     "time-gap / TTC need a predicted speed and "
                                     "lead-agent state; the substrate banks "
                                     "logits, pooled, v0, labels and eid only"),
            "n": 0},
        "LATERAL": {
            "delta": "EXACTLY ZERO (same proof).",
            "level_not_computable": ("heading / curvature / yaw-rate / "
                                     "cross-track need a predicted trajectory, "
                                     "not banked. The lateral CLASSIFICATION "
                                     "readout is reported separately and is not "
                                     "a substitute (R8)."),
            "n": 0},
        "STRATEGIC": {
            "delta": "EXACTLY ZERO (same proof).",
            "level_not_computable": "route_logits are not banked in the substrate",
            "n": 0},
        "ADE": {"delta": "EXACTLY ZERO at every horizon (same proof).", "n": 0},
    }

    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({"controls": res["CONTROLS"]["frontier_rederivation"][
        "matches_published_to_4dp"],
        "shuffled_macro_recall": res["CONTROLS"]["shuffled_pipeline"][
            "macro_recall"],
        "oof_modal_tau": res["arms"]["macro_recall"]["tau_stability"][
            "modal_tau"],
        "oof_macro_recall_ALL": res["arms"]["macro_recall"]["out_of_fold"][
            "ALL"]["macro_recall"],
        "cost_of_honesty_ALL": res["arms"]["macro_recall"]["cost_of_honesty"][
            "ALL"]}, indent=2), flush=True)
    print(f"[dtac1b] wrote {args.out}", flush=True)
    return res


if __name__ == "__main__":                       # pragma: no cover
    main()
