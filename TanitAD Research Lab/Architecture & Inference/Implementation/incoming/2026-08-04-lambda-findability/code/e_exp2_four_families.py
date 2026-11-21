"""E-EXP-2 P2 — the FOUR METRIC FAMILIES for the λ operator, per family, never pooled.

Pre-registration: ../PREREG_E_EXP2.md §7 (blob 7d09ee5f8e9d71b1d465a0205c196b5ca7b6757f)
0 GPU. Binding rule: Sayed 2026-08-02 — "Any future eval must include these metrics."

Three trajectory sets on IDENTICAL windows:
    SEL      the shipped selected trajectory (λ = 1)
    PRED     λ̂ · sel, λ̂ from the pre-registered LOEO ridge on the named feature set
    ORACLE   λ* · sel, the E-EXP-1 ceiling — an upper bound, never a result

⛔ dt = 0.5 s, DERIVED from the dump's own `wp_steps=(5,10,15,20)` through
`four_families.infer_dt`. The 0.1 s default inflated every published speed by x5 and every
accel by x25 (R-2026-08-03) — that defect is the reason the grid travels with the number.

⭐ THE PRE-REGISTERED ANALYTIC PREDICTION, verified here rather than asserted: a radial
scale about t0 multiplies every displacement uniformly, so heading and yaw-rate are EXACTLY
invariant, curvature scales as 1/λ and cross-track as λ. A measured contradiction is
INSTRUMENT-FAIL, not a finding.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "taniteval"))
sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, HERE)

from taniteval import four_families as FF                          # noqa: E402
from taniteval.ci import paired_episode_cluster_bootstrap as PB     # noqa: E402
from taniteval.lead_metrics import (distance_keeping,               # noqa: E402
                                    distance_keeping_by_speed,
                                    paired_distance_keeping)
from tanitad.refs import refc_tactical as tac                       # noqa: E402
from e_exp2_findability import (LAMBDAS, ade, build_features, loeo)  # noqa: E402

N_BOOT, SEED = 2000, 0


def _pw_long_lat(pred, gt, dt):
    """Per-window arrays for the LONGITUDINAL/LATERAL metrics, using the CANONICAL
    `four_families._seq_geometry` so the per-window values and the aggregate block cannot
    drift apart. Returns numpy arrays of length W."""
    P = FF._seq_geometry(torch.as_tensor(pred), dt)
    G = FF._seq_geometry(torch.as_tensor(gt), dt)
    both = P["valid"] & G["valid"]
    both_pair = P["pair_valid"] & G["pair_valid"]
    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi

    def m(x, mask=None):
        x = x.double()
        if mask is None:
            return x.mean(dim=1).numpy()
        w = mask.double()
        n = w.sum(1)
        v = (x * w).sum(1) / n.clamp_min(1)
        return torch.where(n > 0, v, torch.full_like(v, float("nan"))).numpy()

    return {
        "speed_mae_mps": m((P["speed"] - G["speed"]).abs()),
        "speed_bias_mps": m(P["speed"] - G["speed"]),
        "along_mae_m": m((P["along"] - G["along"]).abs()),
        "along_bias_m": m(P["along"] - G["along"]),
        "along_final_bias_m": (P["along"][:, -1] - G["along"][:, -1]).double().numpy(),
        "accel_mae_mps2": m((P["accel"] - G["accel"]).abs()),
        "heading_mae_deg": np.degrees(m(dh.abs(), both)),
        "yaw_rate_mae_degps": np.degrees(
            m((P["yaw_rate"] - G["yaw_rate"]).abs(), both_pair)),
        "curvature_mae_1pm": m((P["curvature"] - G["curvature"]).abs(), both_pair),
        "cross_mae_m": m((P["cross"] - G["cross"]).abs()),
        "cross_bias_m": m(P["cross"] - G["cross"]),
        "cross_final_mae_m": (P["cross"][:, -1] - G["cross"][:, -1]).abs()
        .double().numpy(),
    }


def _tactical_labels(wp, v0, dt):
    """Trajectory-derived FACTORED (lat, lon) manoeuvre through the canonical labeller
    `refc_tactical.factor_from_kinematics`, v2 (curvature) gate.

    ⚠️ CAVEAT THAT TRAVELS WITH EVERY NUMBER: this reads the PATH TANGENT where the trained
    label reads pose YAW, on a 0.5 s grid rather than 0.1 s. It is therefore a PROXY. It is
    identical across the three trajectory sets — which is what makes the COMPARISON valid —
    and it is not quotable as the trained tactical head's accuracy.
    """
    wp = torch.as_tensor(wp).double()
    v0 = torch.as_tensor(v0).double()
    p = torch.cat([torch.zeros_like(wp[:, :1]), wp], dim=1)
    d = p[:, 1:] - p[:, :-1]
    seg = torch.linalg.norm(d, dim=-1)
    v1 = seg[:, -1] / dt                       # chord speed over the final step
    dyaw = torch.atan2(d[:, -1, 1], d[:, -1, 0])
    kappa = dyaw / seg.sum(1).clamp_min(0.10)
    return tac.factor_from_kinematics(dyaw, v1 - v0, v0, v1, kappa=kappa)


def _decision_block(pred_lat, pred_lon, gt_lat, gt_lon):
    out = {}
    for nm, p, g, classes in (("lat", pred_lat, gt_lat, tac.LAT_CLASSES),
                              ("lon", pred_lon, gt_lon, tac.LON_CLASSES)):
        p, g = p.numpy(), g.numpy()
        per = {}
        for c, cname in enumerate(classes):
            sel = g == c
            per[cname] = {"n_true": int(sel.sum()),
                          "n_pred": int((p == c).sum()),
                          "recall": (round(float((p[sel] == c).mean()), 4)
                                     if sel.any() else None)}
        out[nm] = {"accuracy": round(float((p == g).mean()), 4), "per_class": per,
                   "never_predicted": [k for k, v in per.items()
                                       if v["n_true"] > 0 and v["n_pred"] == 0]}
    # the shipped 5-way collapse, so the lat/lon MIXING defect is visible in one place
    m5p = tac.collapse(pred_lat, pred_lon).numpy()
    m5g = tac.collapse(gt_lat, gt_lon).numpy()
    out["man5_accuracy"] = round(float((m5p == m5g).mean()), 4)
    return out


def _strategic_block(wp, gt):
    """λ is a RADIAL scale, so the 2 s bearing is invariant by construction. Verified, not
    assumed — and reported beside the goal DISTANCE error, which λ does move."""
    b_p = np.arctan2(wp[:, -1, 1], wp[:, -1, 0])
    b_g = np.arctan2(gt[:, -1, 1], gt[:, -1, 0])
    db = (b_p - b_g + np.pi) % (2 * np.pi) - np.pi
    r_p = np.linalg.norm(wp[:, -1], axis=-1)
    r_g = np.linalg.norm(gt[:, -1], axis=-1)
    return {"goal_bearing_err_deg": np.degrees(np.abs(db)),
            "goal_dist_err_m": np.abs(r_p - r_g),
            "goal_dist_bias_m": r_p - r_g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--lead", required=True)
    ap.add_argument("--featset", default="F4_pooled")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = torch.load(a.fan, map_location="cpu", weights_only=False)
    L = torch.load(a.latents, map_location="cpu", weights_only=False)
    fan = d["fan"].double().numpy()
    gt = d["gt"].double().numpy()
    sel = d["sel"].numpy()
    eids = list(d["eid"])
    v0 = d["v0"].double().numpy()
    W = len(gt)
    dt, dt_prov = FF.infer_dt({"wp_steps": list(d["wp_steps"]), "dt_s": 0.1})

    fails = list(L.get("instrument_fail", []))
    if not torch.equal(d["fan"].float(), L["fan"].float()):
        fails.append("latent dump's fan is not bit-identical to the bank")

    # ---- the lead block, and the registration check that makes it usable ----
    z = np.load(a.lead, allow_pickle=True)
    lead = {"leads": z["leads"], "lead_lens": z["lead_lens"], "speeds": z["speeds"],
            "state": z["state"], "eid": [str(x) for x in z["eid"]]}
    reg = {"n": int(len(z["speeds"])),
           "speeds_match_v0_max_abs": float(np.abs(z["speeds"] - v0).max()),
           "state_counts": {s: int((z["state"] == s).sum())
                            for s in ("LEAD", "NO_LEAD", "NO_LABEL")},
           "episodes_agree": [str(x) for x in z["eid"]] ==
                             [f"ep_{e:05d}" for e in eids]}
    # ⛔ the lead block is only usable if its rows ARE our rows. `speeds` is the ego speed at
    # t0 — the same quantity the fan dump banks as `v0` — so an exact match is the content
    # check, not an arithmetic assumption about window ordering.
    if reg["n"] != W or reg["speeds_match_v0_max_abs"] > 1e-4:
        fails.append(f"lead block does not register to the fan windows: {reg}")

    # ---- λ̂ from the pre-registered pipeline --------------------------------
    feats, fsel = build_features(
        fan, gt, sel, v0, d["logits"].double().numpy(),
        d["emitted_logits"].double().numpy(), d["prefinal_logits"].double().numpy(),
        {k: L[k].double().numpy() for k in
         ("pooled", "pooled_seq", "ctx", "measurement")}, eids)
    err = np.stack([ade(fsel * l, gt) for l in LAMBDAS])
    lam_star = LAMBDAS[err.argmin(0)]
    num = (fsel * gt).sum(axis=(1, 2))
    den = (fsel * fsel).sum(axis=(1, 2))
    lam_hat, _ = loeo(feats[a.featset], num / np.maximum(den, 1e-12), fsel, gt, eids,
                      mode="ridge")

    sets = {"SEL": fsel,
            "PRED": fsel * lam_hat[:, None, None],
            "ORACLE": fsel * lam_star[:, None, None]}

    out = {"arm": os.path.basename(a.fan), "featset_for_lambda_hat": a.featset,
           "n_windows": W, "n_episodes": len(set(eids)),
           "dt_s": dt, "dt_provenance": dt_prov, "instrument_fail": fails,
           "lead_registration": reg,
           "lambda_hat": {"mean": float(lam_hat.mean()), "std": float(lam_hat.std()),
                          "min": float(lam_hat.min()), "max": float(lam_hat.max())},
           "ade": {k: float(ade(v, gt).mean()) for k, v in sets.items()},
           "FAMILIES": {}, "PAIRED_vs_SEL": {}}

    pw = {k: _pw_long_lat(v, gt, dt) for k, v in sets.items()}
    strat = {k: _strategic_block(v, gt) for k, v in sets.items()}
    dk = {k: distance_keeping(v, lead["leads"], lead["lead_lens"], lead["speeds"], dt)
          for k, v in sets.items()}
    dk["GT_HUMAN"] = distance_keeping(gt, lead["leads"], lead["lead_lens"],
                                      lead["speeds"], dt)
    tl = {k: _tactical_labels(v, v0, dt) for k, v in sets.items()}
    gl_lat, gl_lon = _tactical_labels(gt, v0, dt)

    # ⭐ the pre-registered analytic prediction, VERIFIED
    inv = {
        "heading_mae_deg_SEL_vs_PRED_max_abs": float(np.nanmax(np.abs(
            pw["SEL"]["heading_mae_deg"] - pw["PRED"]["heading_mae_deg"]))),
        "yaw_rate_mae_degps_SEL_vs_PRED_max_abs": float(np.nanmax(np.abs(
            pw["SEL"]["yaw_rate_mae_degps"] - pw["PRED"]["yaw_rate_mae_degps"]))),
        "goal_bearing_err_deg_SEL_vs_PRED_max_abs": float(np.max(np.abs(
            strat["SEL"]["goal_bearing_err_deg"]
            - strat["PRED"]["goal_bearing_err_deg"]))),
        "tactical_lat_SEL_vs_PRED_disagree": int((tl["SEL"][0] != tl["PRED"][0]).sum()),
        "note": ("λ is a radial scale about t0: heading, yaw-rate and the 2 s goal BEARING "
                 "are invariant by construction. A non-zero value here is INSTRUMENT-FAIL, "
                 "not a finding. The lateral MANOEUVRE can still move, because its v2 gate "
                 "reads curvature, which scales as 1/λ."),
    }
    if max(inv["heading_mae_deg_SEL_vs_PRED_max_abs"],
           inv["yaw_rate_mae_degps_SEL_vs_PRED_max_abs"],
           inv["goal_bearing_err_deg_SEL_vs_PRED_max_abs"]) > 1e-6:
        fails.append("λ-invariance of heading/yaw/bearing VIOLATED — instrument bug")
    out["INVARIANCE_CHECK"] = inv

    for k in sets:
        out["FAMILIES"][k] = {
            "LONGITUDINAL": {
                **{m: round(float(np.nanmean(pw[k][m])), 4) for m in
                   ("speed_mae_mps", "speed_bias_mps", "along_mae_m", "along_bias_m",
                    "along_final_bias_m", "accel_mae_mps2")},
                "ego_progress": FF._ego_progress(torch.as_tensor(sets[k]),
                                                 torch.as_tensor(gt)),
                "distance_keeping": {
                    m: dk[k][m] for m in dk[k]
                    if not m.startswith("_") and m != "by_speed"},
                "distance_keeping_by_speed": distance_keeping_by_speed(
                    dk[k], lead["speeds"], lead["eid"], states=lead["state"],
                    n_boot=N_BOOT, seed=SEED),
                "dt_s": dt,
            },
            "LATERAL": {m: (round(float(np.nanmean(pw[k][m])), 6) if
                            np.isfinite(pw[k][m]).any() else None) for m in
                        ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm",
                         "cross_mae_m", "cross_bias_m", "cross_final_mae_m")},
            "TACTICAL": {
                **_decision_block(tl[k][0], tl[k][1], gl_lat, gl_lon),
                "instrument": ("trajectory-derived proxy through "
                               "refc_tactical.factor_from_kinematics (v2 curvature gate), "
                               "path tangent not pose yaw, 0.5 s grid. Comparable ACROSS "
                               "these three sets; NOT the trained head's accuracy."),
            },
            "STRATEGIC": {
                "option_set_path": {
                    "status": "NOT-APPLICABLE", "n_junctions_scorable": 0,
                    "reason": ("PhysicalAI-AV ships no map, lane graph, junction "
                               "annotation or route signal — settled at five independent "
                               "probes; the dataset card says verbatim 'we do not include "
                               "open maps data'. The map-derived option-set scorer "
                               "(taniteval.strategic_optionset) therefore has a "
                               "denominator of 0 on this corpus."),
                },
                "goal_bearing_err_deg": round(
                    float(np.mean(strat[k]["goal_bearing_err_deg"])), 4),
                "goal_dist_err_m": round(float(np.mean(strat[k]["goal_dist_err_m"])), 4),
                "goal_dist_bias_m": round(
                    float(np.mean(strat[k]["goal_dist_bias_m"])), 4),
                "note": ("λ preserves bearing exactly (verified in INVARIANCE_CHECK), so "
                         "the strategic DIRECTION is untouched by this operator and only "
                         "the goal DISTANCE can move."),
            },
        }

    # ---- paired deltas, SEL minus arm (positive = the arm is better) --------
    for k in ("PRED", "ORACLE"):
        blk = {}
        for m in pw["SEL"]:
            a_, b_ = pw["SEL"][m], pw[k][m]
            ok = np.isfinite(a_) & np.isfinite(b_)
            sgn = 1.0 if not m.endswith("bias_m") and not m.endswith("bias_mps") else 0.0
            base = (np.abs(a_) if sgn == 0 else a_)[ok]
            arm = (np.abs(b_) if sgn == 0 else b_)[ok]
            blk[m] = {**PB(base, arm, list(np.asarray(eids)[ok]), n_boot=N_BOOT,
                           seed=SEED), "n_used": int(ok.sum()),
                      "note": ("|bias| compared, so 'positive delta = better' holds for "
                               "a two-sided quantity" if sgn == 0 else None)}
        for m in ("goal_dist_err_m", "goal_bearing_err_deg"):
            blk[m] = PB(strat["SEL"][m], strat[k][m], eids, n_boot=N_BOOT, seed=SEED)
        blk["distance_keeping"] = paired_distance_keeping(
            dk[k], dk["SEL"], lead["eid"], names=(k, "SEL"), n_boot=N_BOOT, seed=SEED)
        # ⭐ the read that matters: does the operator move distance-keeping TOWARDS the human?
        toward = {}
        for key in ("headway_min_m", "time_gap_min_s", "min_ttc_s"):
            g = np.asarray(dk["GT_HUMAN"]["_per_window"][key], dtype=float)
            s = np.asarray(dk["SEL"]["_per_window"][key], dtype=float)
            x = np.asarray(dk[k]["_per_window"][key], dtype=float)
            ok = np.isfinite(g) & np.isfinite(s) & np.isfinite(x)
            toward[key] = {**PB(np.abs(s - g)[ok], np.abs(x - g)[ok],
                                list(np.asarray(lead["eid"])[ok]), n_boot=N_BOOT,
                                seed=SEED),
                           "n_used": int(ok.sum()),
                           "_what": f"|SEL-human| - |{k}-human|; positive = closer to the "
                                    f"human driver"}
        blk["distance_keeping_toward_human"] = toward
        out["PAIRED_vs_SEL"][k] = blk

    out["GT_HUMAN_distance_keeping"] = {
        m: dk["GT_HUMAN"][m] for m in dk["GT_HUMAN"]
        if not m.startswith("_") and m != "by_speed"}
    out["instrument_fail"] = fails
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(json.dumps({"arm": out["arm"], "dt_s": dt, "ade": out["ade"],
                      "instrument_fail": fails,
                      "invariance": out["INVARIANCE_CHECK"],
                      "lead_state_counts": reg["state_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
