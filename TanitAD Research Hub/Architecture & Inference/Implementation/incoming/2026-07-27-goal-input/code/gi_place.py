#!/usr/bin/env python3
"""S3 -- THE LEARNED HEAD, PLACED ON THE CURVE (priority 3).

    requirement from S2   vs   achievement from S1/S3

Every learned goal is fed through EXACTLY the S2 pipeline (`goal_reference` ->
`pick_nearest_to` -> `ade_0_2s`), so the head arm and the noise curve are the
same instrument. The head's recovery is MEASURED directly, never read off the
curve; the curve is then used only to express the gap in metres.

Extra arms, each answering an objection in advance:

  R_selend            the selector's OWN endpoint used as the goal -- the pure
                      circularity control
  R_head_lat_ego      a head that never sees the selector's answer (what a
                      STRATEGIC brain would actually have)
  R_head_insample     fitted on the evaluation windows -- the bound; discharges
                      "n = 881 is thin" in advance
  R_head_along_oracle the head's cross-track with the TRUE along-track (and vice
                      versa) -- localises which axis fails
  R_head_deshrunk     EXPLORATORY (not pre-registered): variance-matching the
                      head's predictions to the GT endpoint distribution, since
                      SHRINK is measured to be the harshest error family

Usage:  python gi_place.py
"""
from __future__ import annotations

import json

import numpy as np

from gi_common import (OUT, ci_paired, ci_single, eid_str, goal_reference,
                       load_refc_fan, pick_nearest_to, r4)

BASE_HALFWIDTH_SHRINK = (2.8, 3.9)     # 40 -> 600 episodes, MODEL_REGISTRY 1.2a


def n_to_separate(delta: float, halfwidth: float, n0: int = 40) -> float:
    """Episodes needed for the CI half-width to fall below |delta|, assuming the
    half-width scales as n^-1/2 (consistent with the registry's measured
    x2.8-3.9 over a x15 increase in n)."""
    if delta == 0:
        return float("inf")
    return float(n0 * (halfwidth / abs(delta)) ** 2)


def main() -> None:
    d = load_refc_fan("xl")
    fan = d["fan"].numpy().astype(np.float64)
    gt = d["gt"].numpy().astype(np.float64)
    cv = d["cv"].numpy().astype(np.float64)
    sel = d["sel"].numpy()
    eid = eid_str(d)
    W = len(gt)
    g = gt[:, -1]
    err4 = np.linalg.norm(fan - gt[:, None], axis=-1).mean(-1)
    a0 = err4[np.arange(W), sel]
    A0 = float(a0.mean())

    taut = json.loads((OUT / "gi_tautology.json").read_text())
    sw = json.loads((OUT / "gi_sweep.json").read_text())
    R0 = sw["R_goal2s_oracle"]
    HEADROOM = A0 - R0
    S50 = sw["families"]["ISO"]["sigma_50_radial_rms_m"]
    S0 = sw["families"]["ISO"]["sigma_0_radial_rms_m"]
    S50_SH = sw["families"]["SHRINK"]["sigma_50_radial_rms_m"]
    S0_SH = sw["families"]["SHRINK"]["sigma_0_radial_rms_m"]

    best_arm = taut["_best_oof"]["arm"]
    P = {k: np.asarray(v, float) for k, v in taut["preds"].items()}
    sel_end = fan[np.arange(W), sel][:, -1, :]
    cv_end = cv[:, -1, :]

    goals = {
        "R_goal2s_ORACLE": g,
        f"R_head_oof [{best_arm}]": P[best_arm],
        "R_head_oof_lat_ego [H_ridge_lat_ego_resid_sel]":
            P["H_ridge_lat_ego_resid_sel"],
        "R_head_insample": P["H7_insample"],
        "R_selend (circularity control)": sel_end,
        "R_cvend": cv_end,
        "R_head_along_oracle": np.stack([g[:, 0], P[best_arm][:, 1]], 1),
        "R_head_cross_oracle": np.stack([P[best_arm][:, 0], g[:, 1]], 1),
        "R_head_deshrunk_EXPLORATORY": None,      # filled below
        "NC3_shuffled_feats_head": P["nc3"] if "nc3" in P else None,
    }
    # de-shrink: match the per-axis SD of the prediction to the GT endpoint SD
    hp = P[best_arm]
    mu = hp.mean(0)
    scale = g.std(0) / np.maximum(hp.std(0), 1e-9)
    goals["R_head_deshrunk_EXPLORATORY"] = mu + (hp - mu) * scale

    res = {"_stream": "2026-07-27-goal-input",
           "_stage": "S3 learned head placed on the S2 curve",
           "_estimator": "paired_episode_cluster_bootstrap B=2000, unit=episode",
           "A0_as_trained": r4(A0), "R_goal2s_oracle": r4(R0),
           "headroom_m": r4(HEADROOM),
           "requirement_ISO": {"sigma_50_radial_rms_m": r4(S50),
                               "sigma_0_radial_rms_m": r4(S0)},
           "requirement_SHRINK": {"sigma_50_radial_rms_m": r4(S50_SH),
                                  "sigma_0_radial_rms_m": r4(S0_SH)},
           "arms": {}}

    print("=== S3 -- REQUIREMENT vs ACHIEVEMENT ===")
    print(f"requirement (ISO):    sigma_50 = {S50:.4f} m   "
          f"sigma_0 (break-even) = {S0:.4f} m   [radial RMS goal error]")
    print(f"requirement (SHRINK): sigma_50 = {S50_SH:.4f} m   "
          f"sigma_0 = {S0_SH:.4f} m   <- the realistic regressor error family")
    print(f"\n{'arm':48s} {'rmsE':>6s} {'meanE':>6s} {'realised':>8s} "
          f"{'recov':>7s} {'vs A0':>8s} sep")
    for name, gh in goals.items():
        if gh is None:
            continue
        e = np.linalg.norm(gh - g, axis=-1)
        rms, mean_e = float(np.sqrt((e ** 2).mean())), float(e.mean())
        idx = pick_nearest_to(goal_reference(gh), fan)
        v = err4[np.arange(W), idx]
        rec = (A0 - v.mean()) / HEADROOM
        cip = ci_paired(v, a0, eid)
        res["arms"][name] = {
            "goal_err_radial_rms_m": r4(rms),
            "goal_err_radial_mean_m": r4(mean_e),
            "goal_err_along_rms_m": r4(np.sqrt(((gh[:, 0] - g[:, 0]) ** 2).mean())),
            "goal_err_cross_rms_m": r4(np.sqrt(((gh[:, 1] - g[:, 1]) ** 2).mean())),
            "realised": ci_single(v, eid),
            "recovery": r4(rec),
            "vs_as_trained": cip,
            "beats_CONFIRM_0.4907": bool(v.mean() < 0.4907),
            "beats_STRONG_0.4271": bool(v.mean() < 0.4271),
            "gap_to_sigma0_ISO_m": r4(rms - S0),
            "gap_to_sigma50_ISO_m": r4(rms - S50),
            "accuracy_factor_needed_vs_sigma0": r4(rms / S0),
        }
        print(f"{name:48s} {rms:6.3f} {mean_e:6.3f} {v.mean():8.4f} "
              f"{rec:+7.3f} {cip['delta']:+8.4f} "
              f"{'SEP' if cip['separated'] else '-'}")

    # --- THE OPERATIVE SPEC: the along-track requirement CONDITIONAL on the
    #     head's realistic cross-track (and the mirror image). The pure LONG /
    #     LAT families in S2 hold the OTHER axis EXACT, which no supplier will.
    from gi_sweep import interp_x_at
    hp_cross, hp_along = hp[:, 1], hp[:, 0]
    for tag, build in (
        ("LONG_given_head_cross",
         lambda dv: np.stack([g[:, 0] + dv, hp_cross], 1)),
        ("LAT_given_head_along",
         lambda dv: np.stack([hp_along, g[:, 1] + dv], 1)),
    ):
        xs, ys, rows = [], [], []
        for s in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
            acc = np.zeros(W)
            for k in range(16):
                rng = np.random.default_rng(20260727 + 1000 * k + 7)
                gh = build(rng.normal(0.0, s, W))
                acc += err4[np.arange(W), pick_nearest_to(goal_reference(gh), fan)]
            v = acc / 16
            rec = (A0 - v.mean()) / HEADROOM
            xs.append(s)
            ys.append(rec)
            rows.append({"sigma_axis_m": s, "realised": ci_single(v, eid),
                         "recovery": r4(rec),
                         "vs_as_trained": ci_paired(v, a0, eid)})
        res[tag] = {"rows": rows,
                    "sigma_50_axis_m": interp_x_at(0.5, xs, ys),
                    "sigma_0_axis_m": interp_x_at(0.0, xs, ys),
                    "recovery_at_sigma0_ie_other_axis_only": r4(ys[0])}
        s50, s0 = res[tag]["sigma_50_axis_m"], res[tag]["sigma_0_axis_m"]
        print(f"\n{tag}: recovery with this axis EXACT = {ys[0]:+.3f}; "
              f"sigma_50 = {s50} m, sigma_0 = {s0} m")
        if tag.startswith("LONG"):
            res[tag]["spec_mean_speed_2s_m_per_s"] = {
                "for_50pct": r4(s50 / 2.0) if s50 else None,
                "for_break_even": r4(s0 / 2.0) if s0 else None,
                "achieved_by_head": r4(
                    float(np.sqrt(((hp[:, 0] - g[:, 0]) ** 2).mean())) / 2.0),
                "_note": ("2 s along-track displacement = 2 s x mean speed, so "
                          "metres of along-track goal error / 2 = m/s of "
                          "2-s-mean-speed error. Units, not correlations."),
            }
            print(f"  -> spec in m/s of 2-s-mean speed: "
                  f"{res[tag]['spec_mean_speed_2s_m_per_s']}")

    # --- curve-vs-actual consistency ---------------------------------------
    rows = sw["families"]["ISO"]["rows"]
    xs = [r["radial_rms_m"] for r in rows]
    ys = [r["recovery"] for r in rows]
    key = f"R_head_oof [{best_arm}]"
    rms_h = res["arms"][key]["goal_err_radial_rms_m"]
    pred_rec = float(np.interp(rms_h, xs, ys))
    res["curve_vs_actual"] = {
        "head_radial_rms_m": rms_h,
        "recovery_predicted_by_ISO_curve": r4(pred_rec),
        "recovery_measured": res["arms"][key]["recovery"],
        "_note": ("a real head's error is BIASED and heteroscedastic; a gap "
                  "between these two numbers is the bias structure, and is "
                  "reported as a finding rather than smoothed away."),
    }

    # --- power statements for every NON-separated headline delta ------------
    res["power"] = {}
    for name, blk in res["arms"].items():
        c = blk["vs_as_trained"]
        if not c["separated"]:
            res["power"][name] = {
                "delta_m": c["delta"], "half_width_m": c["ci95"],
                "n_episodes_to_separate": r4(n_to_separate(c["delta"], c["ci95"]))}

    # --- the tautology delta's power statement ------------------------------
    t = taut["TAUTOLOGY"]["delta_head_minus_sel"]
    res["tautology_power"] = {
        "delta_m": t["delta"], "ci": [t["lo"], t["hi"]],
        "half_width_m": t["ci95"],
        "n_episodes_to_separate": r4(n_to_separate(t["delta"], t["ci95"])),
        "_note": ("UNPOWERED, NOT REFUTED is the correct label ONLY if the "
                  "point estimate is materially non-zero. Here it is 0.3 % of "
                  "e_sel and the n required is astronomical, which is the "
                  "signature of a NULL, not of a resolution limit."),
    }

    (OUT / "gi_place.json").write_text(json.dumps(res, indent=2))
    print(f"\ncurve predicts recovery {pred_rec:+.3f} at rms {rms_h:.3f} m; "
          f"measured {res['arms'][key]['recovery']:+.3f}")
    print(f"tautology delta {t['delta']:+.4f} m, half-width {t['ci95']:.4f} -> "
          f"n_episodes to separate ~ "
          f"{res['tautology_power']['n_episodes_to_separate']:.0f}")
    print("wrote", OUT / "gi_place.json")


if __name__ == "__main__":
    main()
