#!/usr/bin/env python3
"""FINISH THE BOUNDED-TERM REPAIR: ``lat_heading``, the gate guard, ``ego_progress``, ``comfort``.

**No GPU. No model. No checkpoint. No corpus.** Every number is arithmetic over
the committed ``pw_<arm>.npz`` per-window pseudo-simulation dumps.
``taniteval.pseudosim`` / ``taniteval.control`` are IMPORTED, never reimplemented.

⛔ pod1 (TRAINING, 23,000/30,000) and pod2 (small validation) are not contacted.

WHAT IT PRODUCES (banked incrementally, in the brief's PRIORITY ORDER)
---------------------------------------------------------------------
``density.json``          ⭐ FIRST, because C47 says choose the shape against the
                          DENSITY OF THE DATA and not the algebra. The raw
                          quantity behind every bounded term, per arm: quantiles,
                          the fraction living in each clamped region, and — for
                          ``ego_progress`` — WHICH SIDE the floor comes from.
``injections_lat_heading.json``
                          ⭐⭐ THE ACCEPTANCE TEST for priority 1. Heading
                          degradations, both signs, plus the ZERO-MEAN
                          ``yaw_jitter``, on both real arms, against every
                          candidate term. Carries the PRE-REGISTERED rule and
                          the pre-registered C47 prediction.
``guard.json``            ⛔ priority 2: what ``FLOOR_FRAC_MAX = 0.95`` admits and
                          what gate ``v2``'s ``live_frac`` clause refuses, on all
                          20 arms and every bounded term.
``repro_gate.json``       ✅ the 16 published ``@clamp_v1`` composites must still
                          reproduce at ``max|diff| = 0.000000``, the published
                          ``lat_heading`` expression must be BIT-identical, and
                          the published ids must be unchanged.
``panel.json``            the 20-arm panel + the RANKING STATEMENT + the two
                          instrument guards, under the fixed term.
``injections_ego_progress.json``
                          ⚠️ priority 3: can ``twosided_v2`` still charge
                          over-travel — including ON TOP OF an arm that is
                          already over-travelling?
``comfort.json``          ⚠️ priority 4: the evidence behind the publish/withdraw
                          decision for a 100 %-saturated diagnostic.
``audit.json``            the full bounded-term audit, re-run with ``live_frac``.

ESTIMATOR: ``taniteval.ci.paired_episode_cluster_bootstrap`` (B = 2000, unit =
val episode). ⛔ ``overlapping_holdout_se`` appears nowhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[6]
for _p in (str(_REPO / "taniteval"), str(_REPO / "stack")):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

from taniteval import ci as _ci            # noqa: E402
from taniteval import control as C         # noqa: E402
from taniteval import pseudosim as PS      # noqa: E402


# --------------------------------------------------------------------------- #
# ⛔ C44: ASSERT THE MD5 OF THE MODULES ACTUALLY LOADED, not the ones intended.  #
# --------------------------------------------------------------------------- #
def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


IMPORT_PROVENANCE = {m.__name__: {"file": m.__file__, "md5": _md5(m.__file__)}
                     for m in (_ci, C, PS)}

#: probes: scored and reported, but they do NOT vote on any panel-wide gate.
PROBES = ("stand_still", "v1_ego_half", "v1_ego_double", "oracle_lon_straight")
#: arms that are not realisable planners.
ORACLES = ("v1_ego_oracle_lon", "v4_oracle", "v1_tactical_oracle",
           "nospeed_tactical_oracle", "oracle_lon_straight")

#: PUBLISHED `@clamp_v1` composites — the reproduction gate, the same 16 values
#: the progress-term and recovery-term fixes each reproduced at 0.000000.
PUBLISHED_CLAMP_V1 = {
    "cv_holdv0": 0.5705, "v4_oracle": 0.5622, "refc_xl_produced": 0.5499,
    "v1_tactical_follow": 0.5471, "v1_tactical_oracle": 0.5467,
    "refc_small_produced": 0.5444, "refc_base_produced": 0.5439,
    "nospeed_tactical_oracle": 0.5394, "v4_blind": 0.3749,
    "v1_ego_v0": 0.5608, "v1_ego_oracle_lon": 0.5946, "v1_lat_straight": 0.5460,
    "refc_base_v0on": 0.5439, "refc_base_v0off": 0.4980,
    "refc_xl_v0on": 0.5499, "v1_ego_half": 0.3117,
}

# =========================================================================== #
# ⭐ THE ACCEPTANCE SUITE FOR `lat_heading`                                     #
# =========================================================================== #
#: ⛔ THE SIBLING'S 8 INJECTIONS CANNOT TEST THIS AXIS AND THAT IS NOT A CHOICE.
#: `lat_shift` and `lat_jitter` TRANSLATE the plan, and a translation leaves
#: every plan tangent exactly unchanged — `control._ctl_yaw_jitter`'s own
#: docstring says so: *"a per-row translation leaves the terminal heading
#: exactly unchanged, so it is structurally unable to move that axis."* Reusing
#: them here would produce a suite the candidate CANNOT FAIL, which is the one
#: thing the brief forbids. What is reused is the MACHINERY: the same two real
#: arms, the same paired episode-cluster bootstrap, the same requirement of both
#: signs and of a zero-mean control that cannot re-centre a bias. The controls
#: are the ones that move heading, and the 4 translation controls are ALSO run,
#: as the CONTAMINATION panel (§ `null_panel`) — they must move it least.
LATH_INJECTIONS = (("yaw_bias", 5.0), ("yaw_bias", -5.0),
                   ("lat_drift", 0.05), ("lat_drift", -0.05),
                   ("yaw_jitter", 5.0))          # ⭐ the ZERO-MEAN one
LATH_NULL_PANEL = (("lat_shift", 2.0), ("lat_shift", -2.0),
                   ("lat_jitter", 1.0), ("lon_retime", 0.5))
INJECTION_ARMS = ("cv_holdv0", "v1_tactical_follow")
#: ⚠️ reported, NOT part of the pass rule. `v4_blind`'s median u is 6.07 — its
#: plans point ~70 deg off — so requiring a shape to still have gradient there
#: would force q > 0.835 and crush the in-tolerance band to 16 % of the range.
#: It is the STRESS substrate: it shows how far past the floor a term can still
#: charge, which is the C45 mechanism itself.
STRESS_ARM = "v4_blind"

#: the candidate terms. Both RESOLUTIONS x a shape grid spanning both families
#: plus the angle-native one. `term_lin_q0` is the PUBLISHED term and is in the
#: list so the acceptance test is shown to be able to fail.
LATH_CANDIDATES = tuple(
    f"{r}_{s}" for r in ("term", "mean")
    for s in ("lin_q0", "lin_q0p5", "lin_q0p6667", "lin_q0p85", "lin_q0p9",
              "share_q0p5", "cos_q0p5", "cos_q0p75"))

#: ⭐ THE PRE-REGISTERED SELECTION RULE. Banked into `injections_lat_heading.json`
#: BEFORE any panel number is computed, and mirrored verbatim in
#: `control.LAT_HEADING_TERMS`' docstring.
LATH_RULE = {
    "H1": ("DISQUALIFY any term whose injected HEADING degradations are not ALL "
           "separated in the CORRECT (negative) direction on BOTH real arms, "
           "across both signs of every two-sided control AND the ZERO-MEAN "
           "yaw_jitter. A constant-sign control can re-centre a biased arm; the "
           "zero-mean cell cannot, and it is the load-bearing one."),
    "H2": ("DISQUALIFY any term whose lat_heading live_frac < 0.50 on any "
           "scorable (non-probe) arm — equivalently floor OR ceiling >= 0.50. "
           "⭐ Written TWO-SIDED on purpose: the one-sided pair "
           "(FLOOR_FRAC_MAX, CEIL_FRAC_MAX) is exactly what missed this term."),
    "H3": ("Among survivors PREFER THE SMALLEST DEPARTURE FROM THE PUBLISHED "
           "TERM, counted as FREE PARAMETERS FIRST and family second. A term "
           "that changes only the RESOLUTION introduces no parameter at all and "
           "keeps the published per-step expression verbatim, so it beats any "
           "shape change. Within a resolution prefer the LINEAR family (affine-"
           "equivalent on u <= 1) and the SMALLEST surviving q."),
    "H4": ("If no member of H3's preferred class survives, take the family with "
           "the LOWEST C47 reward bias among survivors, and record that C47's "
           "law decided it rather than a preference."),
    "_why_a_rule_at_all": (
        "A shape chosen after seeing the panel is a shape fitted to the panel. "
        "The sibling term's own author pre-registered a preference for the "
        "unsaturating share form and its acceptance test scored that form 0/8."),
    "_PRE_REGISTERED_PREDICTION": (
        "⚠️ From C47, the pass rate should be MONOTONE IN THE REWARD BIAS: cos "
        "(bias << 1) >= lin (bias = 1) >= share (bias > 1). ⛔ BOTH OUTCOMES "
        "ARE COMMITTED IN ADVANCE: if `share` passes here, on a density whose "
        "median u is 0.910 rather than recovery's 1.181, then C47 is BOUNDED "
        "rather than confirmed and that is the finding. If it fails again, C47 "
        "stops being an anecdote about one term."),
    "_why_the_sibling_injections_are_not_reused_verbatim": (
        "⛔ `lat_shift` and `lat_jitter` TRANSLATE the plan and a translation "
        "leaves every plan tangent unchanged, so they are STRUCTURALLY UNABLE "
        "to move this axis — a suite built from them is a suite the candidate "
        "cannot fail. They are run anyway, as the CONTAMINATION panel: a "
        "translation and a pure re-timing must move lat_heading LEAST."),
}

#: ⚠️ over-travel controls for `ego_progress`. `lon_retime` preserves the PATH
#: and scales only the schedule, so it is the axis-pure one; `lon_scale` is the
#: panel's own v1_ego_half / v1_ego_double construction and is kept comparable.
PROG_INJECTIONS = (("lon_retime", 1.5), ("lon_retime", 2.0),
                   ("lon_scale", 1.5), ("lon_jitter", 2.0))
#: ⭐ THE DECISIVE SUBSTRATE for E-3: an arm ALREADY over-travelling (median
#: ratio 1.956, 23.42 % of its rows already at the floor). If the term cannot
#: charge a further over-travel HERE, the residual is consequential; if it can,
#: it is not. A probe arm is the right substrate for testing a METRIC.
PROG_ARMS = ("cv_holdv0", "v1_tactical_follow", "v1_ego_double")
PROG_TERMS = ("clamp_v1", "twosided_v2", "twosided_asym_w0p5",
              "twosided_asym_w2")


def load_pw(path):
    z = np.load(path, allow_pickle=False)
    return {"traj": torch.as_tensor(z["traj"]),
            "ref_path": torch.as_tensor(z["ref_path"]),
            "ref_yaw": torch.as_tensor(z["ref_yaw"]),
            "v0": torch.as_tensor(z["v0"]),
            "pt_dlat": torch.as_tensor(z["pt_dlat"]),
            "pt_dyaw": torch.as_tensor(z["pt_dyaw"]),
            "pt_dlon": torch.as_tensor(z["pt_dlon"]),
            "anchor": torch.as_tensor(z["anchor"]),
            "ep_i": torch.as_tensor(z["ep_i"]),
            "eid": [str(x) for x in z["eid"]]}


def key_of(pw):
    """Row identity. A paired bootstrap over misaligned rows is a fabricated
    number, so this is ASSERTED, never assumed."""
    return np.stack([pw["ep_i"].numpy(), pw["anchor"].numpy(),
                     np.round(pw["pt_dlat"].numpy(), 6) * 1e3,
                     np.round(pw["pt_dyaw"].numpy(), 6) * 1e3,
                     pw["pt_dlon"].numpy()], axis=1).astype(np.int64)


def human_replay(pw):
    """⭐ THE ZERO-BIAS REFERENCE ARM — the plan IS the logged future path.

    ⛔ IT IS NEVER PUT INTO `arms`. The control-suite stream's own reproduction
    gate caught it voting on the PANEL-WIDE gate: it is ceiling-saturated on
    `ego_progress` by construction, so it made that term inadmissible for EVERY
    arm and silently redefined the composite (max|diff| 0.000000 -> 0.393900)."""
    x, y, ref_x, ref_y = PS._cross_and_along(pw)
    Hh = x.shape[1]
    gx = ref_x[:, 1:Hh + 1]
    gy = ref_y[:, 1:Hh + 1] - pw["pt_dlat"][:, None]
    dpsi = torch.deg2rad(pw["pt_dyaw"])
    c, s = torch.cos(dpsi)[:, None], torch.sin(dpsi)[:, None]
    tj = torch.stack([c * gx + s * gy, -s * gx + c * gy], dim=-1)
    out = dict(pw)
    out["traj"] = tj.to(pw["traj"].dtype)
    return out


def _composite_values(sc, weights=None):
    """The PSS composite under the shipped weights, per row."""
    w = PS.COMPONENT_WEIGHTS if weights is None else weights
    ref = np.asarray(sc["ego_progress"], float)
    num, den = np.zeros_like(ref), np.zeros_like(ref)
    for nm, wt in w.items():
        if float(wt) == 0.0:
            continue
        v = np.asarray(sc[nm], float)
        fin = np.isfinite(v)
        num = num + np.where(fin, v * wt, 0.0)
        den = den + np.where(fin, wt, 0.0)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def _control_composite(ax, lat_heading):
    """The CONTROL composite under ``CONTROL_WEIGHTS`` — the object the source
    stream's E-2 asks the PI to adopt as the gate primary, with the candidate
    ``lat_heading`` substituted in. Reported, never the pass criterion: at
    weight 1 of 18 the axis is diluted and yaw controls move `lat_track` too, so
    keying the rule on it would confound the term with its neighbours."""
    parts = {"ego_progress": ax["ego_progress"], "recovery": ax["recovery"],
             "lon_track": ax["lon_track"], "lat_track": ax["lat_track"],
             "lat_heading": lat_heading}
    ref = np.asarray(parts["ego_progress"], float)
    num, den = np.zeros_like(ref), np.zeros_like(ref)
    for nm, wt in C.CONTROL_WEIGHTS.items():
        v = np.asarray(parts[nm], float)
        fin = np.isfinite(v)
        num = num + np.where(fin, v * float(wt), 0.0)
        den = den + np.where(fin, float(wt), 0.0)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def _plan_state(pw):
    """residuals + axes + score_windows ONCE per plan, so every candidate term is
    evaluated from the SAME geometry rather than re-deriving it 16 times."""
    ax = C.axes(pw)
    r = C.residuals(pw)
    return {"ax": ax, "dpsi_end": r["heading_err_rad"],
            "dpsi_steps": r["heading_err_rad_steps"],
            "lat_mask": r["lat_mask"]}


def _lath(state, term, psi_tol=None):
    v = C.lat_heading_from_err(state["dpsi_end"], state["dpsi_steps"],
                               psi_tol=psi_tol, lat_heading_term=term)
    return np.where(state["lat_mask"], v, np.nan)


# =========================================================================== #
# PRIORITY 0 — THE DENSITY. C47: choose the shape against the DATA.            #
# =========================================================================== #
def job_density(arms, out):
    """⭐ FIRST, and it changed two design decisions before a shape was written.

    (a) the psi_tol refutation: widening the tolerance does NOT remove the
        one-sidedness, and the number that proves it is `v4_blind` at 4x.
    (b) the `ego_progress` floor DECOMPOSITION: the residual E-3 named is
        overwhelmingly the UNDER-travel side, not "one-sided above ratio 2"."""
    lath, prog = {}, {}
    pool = []
    for n, pw in arms.items():
        st = _plan_state(pw)
        d = np.abs(np.asarray(st["dpsi_end"], float))
        d = d[np.isfinite(d) & np.isfinite(np.asarray(st["ax"]["lat_heading"],
                                                      float))]
        node = {"n_defined": int(d.size)}
        if d.size:
            u = d / C.PSI_TOL_RAD
            if n not in PROBES:
                pool.append(u)
            node.update({
                "u_median": round(float(np.median(u)), 4),
                "u_p75": round(float(np.percentile(u, 75)), 4),
                "u_p90": round(float(np.percentile(u, 90)), 4),
                "u_p99": round(float(np.percentile(u, 99)), 4),
                "u_max": round(float(u.max()), 4),
                "median_abs_dpsi_deg": round(float(np.degrees(np.median(d))), 3),
                "frac_u_gt_1": round(float((u > 1).mean()), 4),
                "frac_u_gt_2": round(float((u > 2).mean()), 4),
                "frac_u_gt_3": round(float((u > 3).mean()), 4),
                # ⛔ THE REFUTATION OF THE OBVIOUS ALTERNATIVE
                "published_floor_frac_at_psi_tol": {
                    f"{t:g}": round(float((d >= t).mean()), 4)
                    for t in (0.1, 0.2, 0.4, 0.8)},
                # ⭐ THE RESOLUTION LEVER, measured with no shape change at all
                "floor_frac_terminal_resolution": round(float(PS.saturation(
                    _lath(st, "term_lin_q0"))["floor_frac_le_0p001"]), 4),
                "floor_frac_mean_resolution": round(float(PS.saturation(
                    _lath(st, "mean_lin_q0"))["floor_frac_le_0p001"]), 4),
            })
        lath[n] = node

        sc = PS.score_windows(pw, progress_term="twosided_v2")
        g = np.asarray(sc["ego_progress"], float)
        r = np.asarray(sc["ego_progress_raw_ratio"], float)
        fin = np.isfinite(g)
        if fin.any():
            gf, rf, fl = g[fin], r[fin], g[fin] <= 1e-3
            prog[n] = {
                "floor_frac": round(float(fl.mean()), 4),
                # ⛔ WHICH SIDE the floor comes from — the question E-3 never
                # asked and the reason its headline number is misattributed.
                "floor_from_UNDER_r_le_0": round(float((fl & (rf <= 0)).mean()), 4),
                "floor_from_OVER_r_ge_2": round(float((fl & (rf >= 2)).mean()), 4),
                "ceiling_frac": round(float((gf >= 0.999).mean()), 4),
                "median_ratio": round(float(np.median(rf)), 4),
                "p99_ratio": round(float(np.percentile(rf, 99)), 4),
                "max_ratio": round(float(rf.max()), 4),
                "frac_r_gt_1": round(float((rf > 1).mean()), 4),
            }
    allu = np.concatenate(pool)
    out["density"] = {
        "_why_first": ("C47: choose the shape against the DENSITY OF THE DATA, "
                       "not the algebra. Two design decisions were settled here "
                       "before any shape was written."),
        "lat_heading_raw_u": lath,
        "lat_heading_pooled_non_probe": {
            "u_median": round(float(np.median(allu)), 4),
            "u_p90": round(float(np.percentile(allu, 90)), 4),
            "u_p99": round(float(np.percentile(allu, 99)), 4),
            "u_max": round(float(allu.max()), 4),
            "frac_u_gt_1": round(float((allu > 1).mean()), 4),
            "_u_max_is_structural": (
                "u_max = pi / PSI_TOL_RAD = 15.708 is a plan pointing EXACTLY "
                "BACKWARDS; the raw quantity is a wrapped angle and is bounded, "
                "unlike recovery's ratio which reaches 34.1."),
        },
        "ego_progress_floor_decomposition": prog,
        "_psi_tol_widening_refuted": (
            "Widening PSI_TOL_RAD moves the floor and does not remove the "
            "one-sidedness: at FOUR TIMES the published tolerance (0.8 rad = "
            "45.8 deg, at which point 'tolerance' means nothing) `v4_blind` is "
            "still floored on 61.20 % of its rows."),
    }
    print("[bt] density banked", flush=True)
    return out


# =========================================================================== #
# PRIORITY 1 — THE ACCEPTANCE TEST FOR `lat_heading`                           #
# =========================================================================== #
def _charge_rate(arms, terms, probe=1e-4):
    """⭐⭐ C47's DISCRIMINATOR, measured on THIS term's own density.

    ``|dg/du|`` at the ratios the data occupies, plus the reward bias
    ``|dg/du|(near-ideal) / |dg/du|(median row)``. ⛔ > 1 means the shape rewards
    polishing an already-good row harder than it charges a typical one."""
    pool = []
    for n, pw in arms.items():
        if n in PROBES:
            continue
        st = _plan_state(pw)
        d = np.abs(np.asarray(st["dpsi_end"], float))
        d = d[np.isfinite(d)]
        pool.append(d / C.PSI_TOL_RAD)
    allu = np.concatenate(pool)
    med = float(np.median(allu))
    out = {"_what": ("|dg/du|, the rate at which a shape charges one more unit "
                     "of heading error, at the u the panel actually occupies."),
           "panel_u_median": round(med, 4),
           "panel_u_frac_gt_1": round(float((allu > 1).mean()), 4),
           "shapes": {}}
    for shp in sorted({C.LAT_HEADING_TERMS[t][1] for t in terms}):
        g = C.LAT_HEADING_SHAPES[shp]

        def slope(x):
            a, b = max(x - probe, 0.0), x + probe
            return float((g(np.array([a])) - g(np.array([b])))[0] / (b - a))
        s_med, s_near = slope(med), slope(0.05)
        out["shapes"][shp] = {
            "slope_at_u0": round(slope(0.0), 4),
            "slope_at_u0p5": round(slope(0.5), 4),
            "slope_at_u1": round(slope(1.0), 4),
            "slope_at_u2": round(slope(2.0), 4),
            "slope_at_u3": round(slope(3.0), 4),
            "reward_bias_near_perfect_over_median": round(
                s_near / max(s_med, 1e-12), 3),
        }
    return out


def job_inject_lat_heading(arms, out, n_boot, terms=LATH_CANDIDATES):
    """⭐⭐ THE ACCEPTANCE TEST. All 10 cells must move the CORRECT way.

    ``Delta > 0`` means the DEGRADED plan scores HIGHER — the defect."""
    res = {"_selection_rule_PRE_REGISTERED": LATH_RULE,
           "_read": ("delta > 0 means the DEGRADED plan scores HIGHER (the "
                     "defect). PASS requires delta < 0 AND separated, on all "
                     "10 cells (5 injections x 2 real arms)."),
           "_injections": [f"{c}({l:+g})" for c, l in LATH_INJECTIONS],
           "_zero_mean": [c for c, _ in LATH_INJECTIONS
                          if c in C.ZERO_MEAN_CONTROLS],
           "_arms": list(INJECTION_ARMS), "_stress_arm": STRESS_ARM,
           "n_boot": n_boot,
           "charge_rate": _charge_rate(arms, terms), "terms": {}}

    # ---- every plan's geometry ONCE ---------------------------------------- #
    states, eids = {}, {}
    for arm in INJECTION_ARMS + (STRESS_ARM,):
        if arm not in arms:
            continue
        pw = arms[arm]
        eids[arm] = np.asarray(arms[arm]["eid"])
        states[(arm, None)] = _plan_state(pw)
        for ctl, lv in LATH_INJECTIONS + LATH_NULL_PANEL:
            states[(arm, (ctl, lv))] = _plan_state(
                C.apply_control(pw, ctl, lv))

    for term in terms:
        cells, n_ok, n_tot = {}, 0, 0
        stress, nulls = {}, {}
        for arm in INJECTION_ARMS + (STRESS_ARM,):
            if arm not in arms:
                continue
            base_st = states[(arm, None)]
            base = _lath(base_st, term)
            base_cc = _control_composite(base_st["ax"], base)
            for ctl, lv in LATH_INJECTIONS + LATH_NULL_PANEL:
                st = states[(arm, (ctl, lv))]
                v = _lath(st, term)
                m = np.isfinite(v) & np.isfinite(base)
                blk = {"lat_heading": _ci.paired_episode_cluster_bootstrap(
                    v[m], base[m], list(eids[arm][m]), n_boot=n_boot)}
                cc = _control_composite(st["ax"], v)
                mc = np.isfinite(cc) & np.isfinite(base_cc)
                blk["CONTROL_composite"] = _ci.paired_episode_cluster_bootstrap(
                    cc[mc], base_cc[mc], list(eids[arm][mc]), n_boot=n_boot)
                blk["zero_mean_control"] = ctl in C.ZERO_MEAN_CONTROLS
                ok = bool(blk["lat_heading"]["separated"]
                          and blk["lat_heading"]["delta"] < 0)
                blk["CORRECT_DIRECTION"] = ok
                tag = f"{arm}|{ctl}({lv:+g})"
                if (ctl, lv) in LATH_NULL_PANEL:
                    blk["_role"] = ("CONTAMINATION panel — a translation or a "
                                    "pure re-timing must move this axis LEAST; "
                                    "it is NOT part of the pass rule")
                    nulls[tag] = blk
                elif arm == STRESS_ARM:
                    blk["_role"] = ("STRESS substrate (median u = 6.07) — "
                                    "reported, not part of the pass rule")
                    stress[tag] = blk
                else:
                    cells[tag] = blk
                    n_ok += int(ok)
                    n_tot += 1
        res["terms"][term] = {
            "n_correct": n_ok, "n_total": n_tot,
            "ALL_CORRECT": bool(n_tot == 10 and n_ok == 10),
            "resolution": C.LAT_HEADING_TERMS[term][0],
            "shape": C.LAT_HEADING_TERMS[term][1],
            "axis_id": C.lat_heading_axis_id(term),
            "cells": cells, "stress_cells": stress,
            "contamination_cells": nulls}
        print(f"  [lath] {term:18s} {n_ok}/{n_tot} correct direction",
              flush=True)
    out["injections_lat_heading"] = res
    return out


def job_lath_saturation(arms, out, terms=LATH_CANDIDATES):
    """H2's input: live_frac of every candidate on every scorable arm."""
    per = {}
    for term in terms:
        node = {}
        for n, pw in arms.items():
            st = _plan_state(pw)
            s = PS.saturation(
                _lath(st, term),
                n_sub=1 if C.LAT_HEADING_TERMS[term][0] == "term" else 20)
            node[n] = {"floor": s["floor_frac_le_0p001"],
                       "ceiling": s["ceiling_frac_ge_0p999"],
                       "live_frac": s["live_frac"],
                       "row_resolution": s["row_resolution"]}
        live = [v["live_frac"] for k, v in node.items()
                if k not in PROBES and v["live_frac"] is not None]
        per[term] = {"per_arm": node,
                     "min_live_frac_over_scorable_arms": round(min(live), 6),
                     "max_floor_over_scorable_arms": round(max(
                         v["floor"] for k, v in node.items()
                         if k not in PROBES and v["floor"] is not None), 6),
                     "H2_PASS": bool(min(live) >= PS.LIVE_FRAC_MIN)}
        print(f"  [lath-sat] {term:18s} min live_frac "
              f"{per[term]['min_live_frac_over_scorable_arms']:.4f} "
              f"H2={per[term]['H2_PASS']}", flush=True)
    out["lat_heading_saturation"] = {
        "_rule": ("H2: live_frac = 1 - floor - ceiling must be >= "
                  f"{PS.LIVE_FRAC_MIN} on EVERY scorable (non-probe) arm."),
        "live_frac_min": PS.LIVE_FRAC_MIN, "terms": per}
    return out


def job_select(out):
    """⭐ APPLY THE PRE-REGISTERED RULE, mechanically, and show the working."""
    inj = out["injections_lat_heading"]["terms"]
    sat = out["lat_heading_saturation"]["terms"]
    cr = out["injections_lat_heading"]["charge_rate"]["shapes"]
    rows = {}
    for t in inj:
        res, shp = C.LAT_HEADING_TERMS[t]
        rows[t] = {
            "resolution": res, "shape": shp,
            "H1_all_correct": inj[t]["ALL_CORRECT"],
            "H1_n_correct": inj[t]["n_correct"],
            "H2_pass": sat[t]["H2_PASS"],
            "min_live_frac": sat[t]["min_live_frac_over_scorable_arms"],
            "reward_bias": cr[shp]["reward_bias_near_perfect_over_median"],
            "survives": bool(inj[t]["ALL_CORRECT"] and sat[t]["H2_PASS"]),
        }
    surv = [t for t, v in rows.items() if v["survives"]]
    # H3: fewest free parameters first. A pure resolution change keeps the
    # published shape (lin_q0) verbatim and introduces no parameter at all.
    zero_param = [t for t in surv if C.LAT_HEADING_TERMS[t][1] == "lin_q0"]

    def _q_of(t):
        s = C.LAT_HEADING_TERMS[t][1]
        return float(s.split("_q")[1].replace("p", ".")) if "_q" in s else 1.0
    if zero_param:
        pick = sorted(zero_param, key=lambda t: (
            C.LAT_HEADING_TERMS[t][0] != "term", t))[0]
        why = ("H3 — a term that changes only the RESOLUTION keeps the "
               "published per-step expression verbatim and introduces NO free "
               "parameter, so it beats every shape change.")
    else:
        lin = [t for t in surv if C.LAT_HEADING_TERMS[t][1].startswith("lin_")]
        if lin:
            pick, why = sorted(lin, key=_q_of)[0], "H3 — linear family, smallest q"
        elif surv:
            pick = sorted(surv, key=lambda t: cr[C.LAT_HEADING_TERMS[t][1]][
                "reward_bias_near_perfect_over_median"])[0]
            why = "H4 — lowest C47 reward bias among survivors"
        else:
            pick, why = None, "NO TERM SURVIVED — the axis must be dropped"
    # ⚠️ the pre-registered C47 prediction, scored
    by_fam = {}
    for t, v in rows.items():
        by_fam.setdefault(v["shape"].split("_q")[0], []).append(
            (v["reward_bias"], v["H1_n_correct"]))
    pred = {f: {"mean_reward_bias": round(float(np.mean([b for b, _ in xs])), 3),
                "mean_n_correct": round(float(np.mean([c for _, c in xs])), 2)}
            for f, xs in by_fam.items()}
    fams = sorted(pred, key=lambda f: pred[f]["mean_reward_bias"])
    out["selection"] = {
        "_rule": LATH_RULE,
        "per_term": rows, "survivors": surv,
        "SELECTED": pick, "why": why,
        "C47_prediction": {
            "_stated_before_the_numbers": LATH_RULE["_PRE_REGISTERED_PREDICTION"],
            "per_family": pred,
            "families_by_ascending_reward_bias": fams,
            "pass_rate_is_monotone_in_reward_bias": bool(all(
                pred[a]["mean_n_correct"] >= pred[b]["mean_n_correct"]
                for a, b in zip(fams, fams[1:]))),
        }}
    print(f"[bt] SELECTED {pick}  ({why})", flush=True)
    return out


# =========================================================================== #
# PRIORITY 2 — THE GUARD                                                       #
# =========================================================================== #
def job_guard(arms, out, lath_term):
    """⛔ WHAT `FLOOR_FRAC_MAX = 0.95` ADMITS AND WHAT GATE v2 REFUSES.

    The test of a guard is that it refuses the broken term and passes the fixed
    one. Both are run here, on all 20 arms, side by side."""
    def _axes_for(term):
        return {n: dict(C.axes(pw), lat_heading=_lath(_plan_state(pw), term))
                for n, pw in arms.items()}

    rep = {}
    for tag, term in (("PUBLISHED", C.LAT_HEADING_TERM_PUBLISHED),
                      ("FIXED", lath_term)):
        by = _axes_for(term)
        for n in by:
            by[n]["_lat_heading_term"] = term
        g1 = C.panel_gate(by, probes=PROBES, gate_version="v1")
        g2 = C.panel_gate(by, probes=PROBES, gate_version="v2")
        rep[tag] = {
            "lat_heading_term": term,
            "admitted_v1": {k: bool(k in g1["admitted"]) for k in C.AXES},
            "admitted_v2": {k: bool(k in g2["admitted"]) for k in C.AXES},
            "dropped_v1": g1["dropped"], "dropped_v2": g2["dropped"],
            "per_arm_lat_heading_v1": {
                a: g1["detail"][a]["lat_heading"].get("admissible_v1")
                for a in g1["detail"]},
            "per_arm_lat_heading_v2": {
                a: g1["detail"][a]["lat_heading"].get("admissible_v2")
                for a in g1["detail"]},
            "per_arm_lat_heading_live_frac": {
                a: g1["detail"][a]["lat_heading"].get("live_frac")
                for a in g1["detail"]},
        }
    out["guard"] = {
        "_defect": ("FLOOR_FRAC_MAX = 0.95 is a DEAD-COMPONENT tripwire, not a "
                    "discrimination test. Two structural reasons it missed "
                    "lat_heading: (1) RESOLUTION — 0.95 only makes sense for a "
                    "term that is a MEAN over 20 steps, where a row saturates "
                    "only if all 20 do; applied to a SINGLE-VALUE-PER-ROW term "
                    "it is ~20x too loose. (2) IT TESTS EACH END SEPARATELY — "
                    "49 % floored plus 49 % ceilinged clears both one-sided "
                    "thresholds while having gradient on 2 % of rows."),
        "_fix": ("gate v2 adds live_frac = 1 - floor - ceiling >= "
                 f"{PS.LIVE_FRAC_MIN}, the TWO-SIDED statistic the one-sided "
                 "pair cannot express, and `saturation` now PUBLISHES the row "
                 "resolution (n_sub) beside the fraction so the next reader "
                 "does not have to rediscover why the threshold was wrong."),
        "_why_the_published_constant_is_not_simply_lowered": (
            "⛔ recovery@clamp_v1 floors on 55.65-92.19 % of rows. Lowering "
            "FLOOR_FRAC_MAX to 0.50 would make it INADMISSIBLE panel-wide, drop "
            "it from the composite and change every published PSS value — the "
            "silent redefinition this program keeps logging. The gate is "
            "VERSIONED instead: v1 stays exactly what published numbers were "
            "gated under, v2 is strictly stronger and is what `control` uses."),
        "thresholds": {"floor_frac_max_v1": PS.FLOOR_FRAC_MAX,
                       "ceil_frac_max_v1": PS.CEIL_FRAC_MAX,
                       "live_frac_min_v2": PS.LIVE_FRAC_MIN,
                       "saturation_warn_frac": PS.SATURATION_WARN_FRAC,
                       "range_min": PS.RANGE_MIN},
        "control_gate_version": C.CONTROL_GATE_VERSION,
        "panels": rep,
        "GUARD_REFUSES_THE_BROKEN_TERM": bool(
            not rep["PUBLISHED"]["admitted_v2"]["lat_heading"]),
        "GUARD_ADMITS_THE_FIXED_TERM": bool(
            rep["FIXED"]["admitted_v2"]["lat_heading"]),
        "OLD_GUARD_ADMITTED_THE_BROKEN_TERM": bool(
            rep["PUBLISHED"]["admitted_v1"]["lat_heading"]),
    }
    print(f"  [guard] v1 admits broken="
          f"{out['guard']['OLD_GUARD_ADMITTED_THE_BROKEN_TERM']} | "
          f"v2 refuses broken="
          f"{out['guard']['GUARD_REFUSES_THE_BROKEN_TERM']} | "
          f"v2 admits fixed={out['guard']['GUARD_ADMITS_THE_FIXED_TERM']}",
          flush=True)
    return out


# =========================================================================== #
# PRIORITY 2b — THE REPRODUCTION GATE                                          #
# =========================================================================== #
def job_repro(arms, out, n_boot):
    """✅ Every published number must still reproduce EXACTLY."""
    scores = {}
    for n, pw in arms.items():
        sc = PS.score_windows(pw, progress_term="clamp_v1",
                              recovery_term="clamp_v1")
        scores[n] = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
        scores[n]["no_collision"] = None
        scores[n]["ttc"] = None
    gate_arms = [n for n in arms if n not in PROBES]
    per_arm = {n: PS.discriminative_range(scores[n], by_arm=scores) for n in arms}
    admissible = {}
    for comp in PS.COMPONENT_WEIGHTS_PUBLISHED_V1:
        admissible[comp] = not [n for n in gate_arms
                                if not per_arm[n].get(comp, {}).get("admissible")]
    rep, worst = {}, 0.0
    for n, want in PUBLISHED_CLAMP_V1.items():
        if n not in arms:
            rep[n] = {"published": want, "status": "ARM ABSENT"}
            continue
        pr = dict(per_arm[n])
        for c, ok in admissible.items():
            if not ok and c in pr:
                pr[c] = dict(pr[c], admissible=False,
                             reason="dropped by the PANEL-WIDE gate")
        comp = PS.composite(scores[n], pr, progress_term="clamp_v1",
                            recovery_term="clamp_v1")
        v = comp.pop("value")
        got = PS._boot(v, arms[n]["eid"], n_boot, 0)["mean"]
        d = abs(got - want)
        worst = max(worst, d)
        rep[n] = {"published": want, "got": got, "abs_diff": round(d, 6),
                  "metric_id": comp["name"],
                  "status": "OK" if d <= 5e-4 else "MISMATCH"}
    # ⭐ the lat_heading half of the gate: the PUBLISHED expression must be
    # BIT-identical, not merely close, and its id must not have moved.
    bit = {}
    for n, pw in arms.items():
        st = _plan_state(pw)
        old = np.where(st["lat_mask"], np.clip(
            1.0 - np.abs(np.asarray(st["dpsi_end"], float)) / C.PSI_TOL_RAD,
            0.0, 1.0), np.nan)
        new = _lath(st, C.LAT_HEADING_TERM_PUBLISHED)
        m = np.isfinite(old) & np.isfinite(new)
        bit[n] = bool((old[m] == new[m]).all()
                      and np.isfinite(old).sum() == np.isfinite(new).sum())
    out["repro_gate"] = {
        "rule": ("Every PUBLISHED @clamp_v1 composite must still reproduce to "
                 "4 dp with the gate VERSIONED, `saturation` extended and the "
                 "lat_heading term versioned with its default flipped."),
        "progress_term": "clamp_v1", "recovery_term": "clamp_v1",
        "gate_version_used": PS.GATE_VERSION_DEFAULT,
        "metric_id_must_be_unchanged": PS.metric_id("clamp_v1", "clamp_v1"),
        "panel_gate_admissible": admissible,
        "max_abs_diff": round(worst, 6),
        "PASS": bool(worst <= 5e-4),
        "n_checked": sum(1 for v in rep.values() if v.get("status") == "OK"),
        "per_arm": rep,
        "lat_heading_published_is_BIT_IDENTICAL_per_arm": bit,
        "lat_heading_published_is_BIT_IDENTICAL": bool(all(bit.values())),
        "published_lat_heading_axis_id_unchanged": bool(
            C.lat_heading_axis_id(C.LAT_HEADING_TERM_PUBLISHED) == "lat_heading"),
        "published_suite_id_unchanged": C.SUITE_ID(
            lat_heading_term=C.LAT_HEADING_TERM_PUBLISHED),
        "new_suite_id": C.SUITE_ID(),
    }
    print(f"  [repro] max|diff| = {worst:.6f} PASS={worst <= 5e-4} | "
          f"lat_heading bit-identical="
          f"{out['repro_gate']['lat_heading_published_is_BIT_IDENTICAL']}",
          flush=True)
    return out


# =========================================================================== #
# PRIORITY 2c — THE PANEL AND THE RANKING STATEMENT                            #
# =========================================================================== #
CONTRASTS = (
    ("v1_ego_v0", "cv_holdv0"),
    ("v1_tactical_follow", "cv_holdv0"),
    ("refc_xl_produced", "cv_holdv0"),
    ("v1_ego_v0", "v1_tactical_follow"),
    ("v1_lat_straight", "v1_tactical_follow"),
    ("refc_xl_v0on", "refc_xl_v0off"),
    ("v4_oracle", "v4_blind"),
    ("v1_ego_half", "v1_tactical_follow"),
    ("v1_tactical_follow", "refc_xl_produced"),
    ("v1_tactical_follow", "refc_base_v0off"),
)
V1_TACTICAL_FAMILY = ("v1_tactical_follow", "v1_tactical_oracle",
                      "v1_lat_straight", "nospeed_tactical_oracle")


def _pss_panel(arms, n_boot):
    """⛔ THE PSS PANEL IS UNCHANGED BY THIS WORK and is recomputed to prove it:
    `lat_heading` is not in `pseudosim.COMPONENT_WEIGHTS`."""
    scores, vals = {}, {}
    for n, pw in arms.items():
        sc = PS.score_windows(pw)
        scores[n] = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
        scores[n]["no_collision"] = scores[n]["ttc"] = None
    gate_arms = [n for n in arms if n not in PROBES]
    per_arm = {n: PS.discriminative_range(scores[n], by_arm=scores) for n in arms}
    adm = {c: not [n for n in gate_arms
                   if not per_arm[n].get(c, {}).get("admissible")]
           for c in PS.COMPONENT_WEIGHTS}
    w = {k: v for k, v in PS.COMPONENT_WEIGHTS.items()
         if adm.get(k) and float(v) > 0}
    levels = {}
    for n in arms:
        vals[n] = _composite_values(scores[n], weights=w)
        levels[n] = PS._boot(vals[n], arms[n]["eid"], n_boot, 0)["mean"]
    return levels, vals, adm, PS.metric_id(PS.PROGRESS_TERM_DEFAULT,
                                           PS.RECOVERY_TERM_DEFAULT)


def _control_panel(arms, n_boot, term):
    levels, vals = {}, {}
    for n, pw in arms.items():
        st = _plan_state(pw)
        v = _control_composite(st["ax"], _lath(st, term))
        vals[n] = v
        levels[n] = PS._boot(v, arms[n]["eid"], n_boot, 0)["mean"]
    return levels, vals


def _ranked(levels):
    real = [n for n in levels if n not in PROBES and n not in ORACLES]
    order = sorted(levels, key=lambda n: -levels[n])
    return {"all_non_probe": [{"rank": i + 1, "arm": n, "value": levels[n],
                               "oracle": n in ORACLES}
                              for i, n in enumerate(
                                  [x for x in order if x not in PROBES])],
            "realisable_only": [{"rank": i + 1, "arm": n, "value": levels[n]}
                                for i, n in enumerate(
                                    sorted(real, key=lambda n: -levels[n]))]}


def job_panel(arms, out, n_boot, lath_term):
    pss_lv, pss_v, pss_adm, pss_id = _pss_panel(arms, n_boot)
    pub_lv, pub_v = _control_panel(arms, n_boot, C.LAT_HEADING_TERM_PUBLISHED)
    new_lv, new_v = _control_panel(arms, n_boot, lath_term)

    def _contrasts(vv, tag):
        o = {}
        for a, b in CONTRASTS:
            if a not in vv or b not in vv:
                continue
            x, y = vv[a], vv[b]
            m = np.isfinite(x) & np.isfinite(y)
            o[f"{a} - {b}"] = _ci.paired_episode_cluster_bootstrap(
                x[m], y[m], list(np.asarray(arms[a]["eid"])[m]), n_boot=n_boot)
        return o

    cpub, cnew, cpss = (_contrasts(pub_v, "pub"), _contrasts(new_v, "new"),
                        _contrasts(pss_v, "pss"))
    flips = {k: {"published": cpub[k], "new": cnew[k]}
             for k in cnew
             if cpub[k]["separated"] != cnew[k]["separated"]
             or (np.sign(cpub[k]["delta"]) != np.sign(cnew[k]["delta"])
                 and cnew[k]["separated"])}
    rk_pub, rk_new, rk_pss = _ranked(pub_lv), _ranked(new_lv), _ranked(pss_lv)

    def _rk(rk, arm):
        return next((r["rank"] for r in rk["realisable_only"]
                     if r["arm"] == arm), None)

    def _below_all_refc(rk):
        order = [r["arm"] for r in rk["all_non_probe"]]
        fam = [order.index(n) for n in V1_TACTICAL_FAMILY if n in order]
        rc = [order.index(n) for n in order if n.startswith("refc_")]
        return bool(fam and rc and min(fam) > max(rc))

    out["panel"] = {
        "PSS_composite": {
            "metric_id": pss_id, "panel_gate_admissible": pss_adm,
            "levels": pss_lv, "ranking": rk_pss, "contrasts": cpss,
            "_unchanged_by_this_work": (
                "`lat_heading` is not in pseudosim.COMPONENT_WEIGHTS, so the "
                "PSS panel is recomputed only to PROVE this work did not move "
                "it. Compare against the recovery-fix panel."),
        },
        "CONTROL_composite": {
            "weights": C.CONTROL_WEIGHTS,
            "suite_id_published": C.SUITE_ID(
                lat_heading_term=C.LAT_HEADING_TERM_PUBLISHED),
            "suite_id_new": C.SUITE_ID(lat_heading_term=lath_term),
            "levels_published_lat_heading": pub_lv,
            "levels_new_lat_heading": new_lv,
            "ranking_published": rk_pub, "ranking_new": rk_new,
            "contrasts_published": cpub, "contrasts_new": cnew,
            "verdicts_that_flipped": flips,
        },
        "RANKING_STATEMENT": {
            "cv_holdv0_rank_realisable_PSS": _rk(rk_pss, "cv_holdv0"),
            "cv_holdv0_rank_realisable_CONTROL_published": _rk(rk_pub,
                                                               "cv_holdv0"),
            "cv_holdv0_rank_realisable_CONTROL_new": _rk(rk_new, "cv_holdv0"),
            "cv_holdv0_STILL_RANKS_FIRST_AMONG_REALISABLE_PSS":
                bool(_rk(rk_pss, "cv_holdv0") == 1),
            "cv_holdv0_STILL_RANKS_FIRST_AMONG_REALISABLE_CONTROL":
                bool(_rk(rk_new, "cv_holdv0") == 1),
            "_v1_family_definition": (
                "the v1 TACTICAL family = " + ", ".join(V1_TACTICAL_FAMILY)
                + ". `v1_ego_v0` / `v1_ego_oracle_lon` are ego-SCHEDULE "
                  "transforms and rank ABOVE every REF-C arm under the "
                  "published term as well, so they are not part of the claim."),
            "whole_v1_tactical_family_below_every_REFC_PSS":
                _below_all_refc(rk_pss),
            "whole_v1_tactical_family_below_every_REFC_CONTROL_published":
                _below_all_refc(rk_pub),
            "whole_v1_tactical_family_below_every_REFC_CONTROL_new":
                _below_all_refc(rk_new),
            "realisable_order_CONTROL_published": [
                r["arm"] for r in rk_pub["realisable_only"]],
            "realisable_order_CONTROL_new": [
                r["arm"] for r in rk_new["realisable_only"]],
        },
        "INSTRUMENT_GUARDS": {
            "_rule": ("⚠️ A fix that makes the metric MORE PERMISSIVE is not a "
                      "fix. Both guards must survive and should strengthen."),
            "G1_v4_oracle_minus_v4_blind": {
                "PSS": cpss.get("v4_oracle - v4_blind"),
                "CONTROL_published": cpub.get("v4_oracle - v4_blind"),
                "CONTROL_new": cnew.get("v4_oracle - v4_blind")},
            "G2_v1_ego_half_minus_v1_tactical_follow": {
                "PSS": cpss.get("v1_ego_half - v1_tactical_follow"),
                "CONTROL_published": cpub.get(
                    "v1_ego_half - v1_tactical_follow"),
                "CONTROL_new": cnew.get("v1_ego_half - v1_tactical_follow")},
        },
    }
    for k in ("v4_oracle - v4_blind", "v1_ego_half - v1_tactical_follow"):
        if k in cnew:
            print(f"  [guard-metric] {k}: {cpub[k]['delta']:+.4f} -> "
                  f"{cnew[k]['delta']:+.4f}", flush=True)
    return out


# =========================================================================== #
# PRIORITY 3 — `ego_progress` ABOVE r = 2                                      #
# =========================================================================== #
def job_inject_progress(arms, out, n_boot):
    """⚠️ CAN `twosided_v2` STILL CHARGE OVER-TRAVEL — including ON TOP OF an
    arm that is ALREADY over-travelling? That is what E-3 actually asks."""
    res = {"_read": ("delta > 0 means the OVER-TRAVELLING plan scores HIGHER "
                     "(the C45 defect). PASS requires delta < 0 AND separated."),
           "_arms": list(PROG_ARMS),
           "_why_v1_ego_double": (
               "⭐ THE DECISIVE SUBSTRATE. Its median ratio is already 1.956 "
               "and 23.42 % of its rows are already at the floor, so a further "
               "over-travel lands almost entirely in the zero-gradient half. If "
               "the term can still charge it, the residual is not "
               "consequential; if it cannot, it is. A probe arm is the right "
               "substrate for testing a METRIC — it is not being ranked."),
           "n_boot": n_boot, "terms": {}}
    cache = {}
    for arm in PROG_ARMS:
        if arm not in arms:
            continue
        cache[(arm, None)] = arms[arm]
        for ctl, lv in PROG_INJECTIONS:
            cache[(arm, (ctl, lv))] = C.apply_control(arms[arm], ctl, lv)
    for term in PROG_TERMS:
        cells, n_ok, n_tot = {}, 0, 0
        for arm in PROG_ARMS:
            if arm not in arms:
                continue
            eid = np.asarray(arms[arm]["eid"])
            base = np.asarray(PS.score_windows(
                cache[(arm, None)], progress_term=term)["ego_progress"], float)
            for ctl, lv in PROG_INJECTIONS:
                v = np.asarray(PS.score_windows(
                    cache[(arm, (ctl, lv))],
                    progress_term=term)["ego_progress"], float)
                m = np.isfinite(v) & np.isfinite(base)
                b = _ci.paired_episode_cluster_bootstrap(
                    v[m], base[m], list(eid[m]), n_boot=n_boot)
                ok = bool(b["separated"] and b["delta"] < 0)
                b["CORRECT_DIRECTION"] = ok
                b["zero_mean_control"] = ctl in C.ZERO_MEAN_CONTROLS
                cells[f"{arm}|{ctl}({lv:+g})"] = b
                n_ok += int(ok)
                n_tot += 1
        floor_over = {}
        for n, pw in arms.items():
            sc = PS.score_windows(pw, progress_term=term)
            g = np.asarray(sc["ego_progress"], float)
            r = np.asarray(sc["ego_progress_raw_ratio"], float)
            fin = np.isfinite(g)
            if fin.any():
                floor_over[n] = round(
                    float(((g[fin] <= 1e-3) & (r[fin] >= 1)).mean()), 4)
        res["terms"][term] = {
            "n_correct": n_ok, "n_total": n_tot,
            "ALL_CORRECT": bool(n_ok == n_tot and n_tot > 0),
            "over_side_floor_frac_per_arm": floor_over,
            "over_side_floor_frac_max_scorable": round(float(max(
                v for k, v in floor_over.items() if k not in PROBES)), 4),
            "cells": cells}
        print(f"  [prog] {term:20s} {n_ok}/{n_tot} correct; over-side floor max "
              f"{res['terms'][term]['over_side_floor_frac_max_scorable']:.4f}",
              flush=True)
    out["injections_ego_progress"] = res
    return out


# =========================================================================== #
# PRIORITY 4 — THE `comfort` DECISION                                          #
# =========================================================================== #
def job_comfort(arms, out):
    """⚠️ SHOULD A 100 %-SATURATED DIAGNOSTIC BE PUBLISHED AT ALL?

    The evidence, not the opinion: it is a {0,1} indicator, so `observed_range`
    = 1.0 records only that both values OCCUR; and the continuous margin
    underneath it — which is what a diagnostic would actually need — is
    computable from the identical quantities and is NOT emitted today."""
    per = {}
    for n, pw in arms.items():
        sc = PS.score_windows(pw)
        c = np.asarray(sc["comfort"], float)
        fin = c[np.isfinite(c)]
        s = PS.saturation(c, n_sub=1)
        per[n] = {
            "mean_pass_rate": round(float(fin.mean()), 6) if fin.size else None,
            "n_distinct_values": int(len(np.unique(fin))) if fin.size else 0,
            "observed_range": round(float(fin.max() - fin.min()), 6)
            if fin.size else None,
            "floor_frac": s["floor_frac_le_0p001"],
            "ceiling_frac": s["ceiling_frac_ge_0p999"],
            "live_frac": s["live_frac"],
        }
    lv = [v["live_frac"] for v in per.values() if v["live_frac"] is not None]
    out["comfort"] = {
        "per_arm": per,
        "max_live_frac_over_all_arms": round(float(max(lv)), 6),
        "every_arm_is_100pct_saturated": bool(max(lv) <= 1e-9),
        "observed_range_is_an_ARTEFACT": (
            "⛔ `observed_range = 1.0` is the statistic that let this term clear "
            "RANGE_MIN for its whole life. For a {0,1} indicator it records "
            "ONLY THAT BOTH VALUES OCCUR — max - min = 1 for any binary array "
            "containing at least one 0 and one 1, whatever the mixture. It is "
            "not evidence of range and RANGE_MIN cannot see the difference."),
        "gate_v1_verdict": "ADMISSIBLE on range; dropped only per-arm",
        "gate_v2_verdict": ("REFUSED on every arm: live_frac = 0.0000 < "
                            f"{PS.LIVE_FRAC_MIN}"),
        "DECISION": (
            "PUBLISH THE MEASUREMENT, RETIRE THE NAME AND THE SCORE-SHAPED "
            "NODE. (a) It must not be emitted as `components.comfort` beside "
            "two real scores in [0,1]: a reader comparing 0.5492 (recovery) "
            "with 1.0000 (comfort) is comparing a mean score with a PASS RATE, "
            "and the program has already published `observed_range = 1.0` for "
            "it 20 times as if it were range. (b) It must not be silently "
            "deleted either: C46 was found BECAUSE the number was still on the "
            "page, and a measurement that refutes its own term is the cheapest "
            "instrument in the suite. ⇒ it is emitted as "
            "`diagnostics.plan_smoothness_pass_rate`, a RATE with its own "
            "units, carrying COMFORT_STATUS and its live_frac = 0.0000, and "
            "gate v2 refuses it automatically so no future weight can be "
            "attached without the refusal being visible."),
        "_what_would_make_it_a_score": (
            "The AND of four bounds discards a continuous margin that is "
            "already computed: max over the four clauses of (observed / limit). "
            "That quantity has real range and would be a usable diagnostic. It "
            "is NAMED here and NOT shipped: C46's finding is that the LIMITS "
            "themselves fail the human's own logged path on 16.60 % of windows, "
            "so a margin against them would be a margin against a bound we know "
            "is wrong. Fix the bound first."),
    }
    print("[bt] comfort decision banked", flush=True)
    return out


# =========================================================================== #
# THE FULL BOUNDED-TERM AUDIT, re-run with live_frac                           #
# =========================================================================== #
def job_audit(arms, out, lath_term):
    terms = {}

    def _collect(name, fn, n_sub, expr, sides):
        per = {}
        for n, pw in arms.items():
            s = PS.saturation(fn(pw), n_sub=n_sub)
            per[n] = {"floor": s["floor_frac_le_0p001"],
                      "ceiling": s["ceiling_frac_ge_0p999"],
                      "live_frac": s["live_frac"]}
        sc = [v for k, v in per.items() if k not in PROBES
              and v["live_frac"] is not None]
        terms[name] = {
            "expr": expr, "clamped_sides": sides, "n_sub_per_row": n_sub,
            "min_live_frac_scorable": round(min(v["live_frac"] for v in sc), 4),
            "max_floor_scorable": round(max(v["floor"] for v in sc), 4),
            "max_ceiling_scorable": round(max(v["ceiling"] for v in sc), 4),
            "GATE_V1_ADMITS": bool(max(v["floor"] for v in sc) < PS.FLOOR_FRAC_MAX
                                   and max(v["ceiling"] for v in sc)
                                   < PS.CEIL_FRAC_MAX),
            "GATE_V2_ADMITS": bool(min(v["live_frac"] for v in sc)
                                   >= PS.LIVE_FRAC_MIN),
            "per_arm": per}

    _collect("ego_progress@clamp_v1",
             lambda pw: PS.score_windows(pw, progress_term="clamp_v1")[
                 "ego_progress"], 1,
             "clamp(r, 0, 1), r = plan_along / human_along",
             "BOTH (floor at r<=0, ceiling at r>=1)")
    _collect("ego_progress@twosided_v2",
             lambda pw: PS.score_windows(pw, progress_term="twosided_v2")[
                 "ego_progress"], 1,
             "clamp(clamp_v1(r) - w*max(r-1,0), 0, 1), w = 1",
             "FLOOR at r<=0 AND at r>=2 — two-sided, both ends floor")
    _collect("recovery@clamp_v1",
             lambda pw: PS.score_windows(pw, recovery_term="clamp_v1")[
                 "recovery"], 1,
             "clamp(1 - r, 0, 1), r = |xt_end| / |xt_hold|",
             "⛔ THE DEFECT (C45) — floor at r>=1, ceiling never active")
    _collect("recovery@twosided_v2",
             lambda pw: PS.score_windows(pw)["recovery"], 1,
             "clamp(1 - r/3, 0, 1) (lin_q0p6667)", "FLOOR at r >= 3")
    _collect("comfort", lambda pw: PS.score_windows(pw)["comfort"], 1,
             "AND of four bounds — a {0,1} INDICATOR, not a score",
             "⛔ NOT A CLAMP: 100 % saturated BY CONSTRUCTION")
    _collect("lon_track", lambda pw: C.axes(pw)["lon_track"], 20,
             "mean_k clamp(1 - |t_err_k| / T_TOL, 0, 1)",
             "per-step floor, MEANED over 20 steps")
    _collect("lat_track", lambda pw: C.axes(pw)["lat_track"], 20,
             "mean_k clamp(1 - |XTE_k| / corridor(s_k), 0, 1)",
             "per-step floor, MEANED over 20 steps")
    _collect("lat_heading@term_lin_q0 (PUBLISHED)",
             lambda pw: _lath(_plan_state(pw), "term_lin_q0"), 1,
             "clamp(1 - |dpsi_end| / PSI_TOL, 0, 1) — a SINGLE value per row",
             "⛔ FLOOR at |dpsi| >= PSI_TOL; ceiling never active")
    _collect(f"lat_heading@{lath_term} (SHIPPED)",
             lambda pw: _lath(_plan_state(pw), lath_term), 20,
             "mean_k clamp(1 - |dpsi_k| / PSI_TOL, 0, 1) over 20 steps",
             "per-step floor, MEANED over 20 steps")
    out["audit"] = {
        "_rule": ("Three of three audited bounded terms were one-sidedly "
                  "clamped and the fourth is saturated by construction. Every "
                  "term is re-audited here on the TWO-SIDED live_frac, and the "
                  "v1/v2 verdicts are reported side by side so the guard's "
                  "effect is a measured fact rather than a claim."),
        "_detection_heuristic": (
            "For any bounded score: (1) compute live_frac = 1 - floor - ceiling "
            "on every arm — NOT the two ends separately; (2) state n_sub, the "
            "sub-samples averaged per row, because a per-row term saturates "
            "~20x more readily than a mean-over-20 one; (3) compute |dg/du| at "
            "the MEDIAN of the term's own raw quantity and divide by |dg/du| "
            "near the ideal — > 1 means the term rewards polishing good rows "
            "more than it charges bad ones (C47)."),
        "terms": terms}
    return out


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=_ci.DEFAULT_N_BOOT)
    ap.add_argument("--ref-arm", default="cv_holdv0")
    ap.add_argument("--lat-heading-term", default=None)
    ap.add_argument("--jobs",
                    default="density,lath,select,guard,repro,panel,prog,"
                            "comfort,audit")
    a = ap.parse_args()

    t0 = time.time()
    arms, seen = {}, {}
    for d in a.in_dir:
        for f in sorted(Path(d).glob("pw_*.npz")):
            name = f.name[len("pw_"):-len(".npz")]
            if name in arms:
                raise SystemExit(f"duplicate arm {name}: {seen[name]} vs {f}")
            arms[name] = load_pw(f)
            seen[name] = str(f)
    print(f"[bt] {len(arms)} arms: {sorted(arms)}", flush=True)

    ref = a.ref_arm if a.ref_arm in arms else sorted(arms)[0]
    kref = key_of(arms[ref])
    row_id, refused = {}, {}
    for n in list(arms):
        k = key_of(arms[n])
        ok = (k.shape == kref.shape) and bool((k == kref).all())
        row_id[n] = {"n_rows": int(k.shape[0]), "identical_to_reference": ok}
        if not ok:
            refused[n] = f"row keys differ from {ref}"
            arms.pop(n)
    assert arms, f"every arm refused on row identity: {refused}"

    # ⭐ the zero-bias reference, built and ASSERTED — but kept OUT of `arms`.
    hr = human_replay(arms[ref])
    _r = C.residuals(hr)
    _worst = max(float(np.abs(_r["along_err_m"]).max()),
                 float(np.abs(_r["cross_err_m"]).max()),
                 float(np.abs(_r["xte_m"]).max()))
    assert _worst < 1e-3, (
        f"human_replay is NOT on the logged path (max |residual| = {_worst})")
    _hpsi = float(np.nanmax(np.abs(_r["heading_err_rad_steps"])))
    print(f"[bt] human_replay verified at max|residual| = {_worst:.3e} m, "
          f"max|per-step dpsi| = {_hpsi:.3e} rad (kept OUT of the gate vote)",
          flush=True)

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    head = {
        "_experiment": ("FINISH THE BOUNDED-TERM REPAIR: lat_heading (C45's "
                        "third clamp), the FLOOR_FRAC_MAX guard, "
                        "ego_progress above r = 2, and the comfort decision. "
                        "0 GPU-h."),
        "_evidence_class": "MEASURED (ours; artifact = these JSONs + pw_*.npz)",
        "_protocol": PS.PROTOCOL,
        "_estimator": (f"taniteval.ci.episode_cluster_bootstrap / paired form "
                       f"(B={a.n_boot}, unit = val episode)"),
        "_refused_estimator": ("overlapping_holdout_se — it biases the POINT "
                               "ESTIMATE as well as the interval"),
        "_gate": "PANEL-WIDE. The per-arm gate is REFUSED, not offered.",
        "_hosts": ("dev box only. pod1 (TRAINING 23,000/30,000) and pod2 "
                   "(small validation) were NOT contacted."),
        "_import_provenance_md5": IMPORT_PROVENANCE,
        "_row_identity": row_id, "_refused_arms": refused,
        "_human_replay_max_residual_m": _worst,
        "_zero_bias_reference_kept_out_of_the_gate_vote": True,
        "_parity": ("No episode is re-selected. All arms share the identical "
                    "row set and row identity is ASSERTED, not assumed. "
                    "Counts only — no clip UUID appears in any artifact."),
    }
    jobs = [j.strip() for j in a.jobs.split(",") if j.strip()]
    out = dict(head)

    def bank(name, payload_key):
        p = od / f"{name}.json"
        p.write_text(json.dumps({**head, name: out[payload_key]}, indent=1,
                                default=str), encoding="utf-8")
        print(f"[bt] wrote {p}", flush=True)

    if "density" in jobs:
        job_density(arms, out)
        bank("density", "density")
    if "lath" in jobs:
        job_inject_lat_heading(arms, out, a.n_boot)
        job_lath_saturation(arms, out)
    if "select" in jobs:
        job_select(out)
        out["injections_lat_heading"]["selection"] = out["selection"]
        out["injections_lat_heading"]["saturation"] = \
            out["lat_heading_saturation"]
        bank("injections_lat_heading", "injections_lat_heading")

    lath_term = (a.lat_heading_term or out.get("selection", {}).get("SELECTED")
                 or C.LAT_HEADING_TERM_DEFAULT)
    print(f"[bt] lat_heading term in use: {lath_term}", flush=True)
    out["_lat_heading_term_in_use"] = lath_term

    if "guard" in jobs:
        job_guard(arms, out, lath_term)
        bank("guard", "guard")
    if "repro" in jobs:
        job_repro(arms, out, a.n_boot)
        bank("repro_gate", "repro_gate")
    if "panel" in jobs:
        job_panel(arms, out, a.n_boot, lath_term)
        bank("panel", "panel")
    if "prog" in jobs:
        job_inject_progress(arms, out, a.n_boot)
        bank("injections_ego_progress", "injections_ego_progress")
    if "comfort" in jobs:
        job_comfort(arms, out)
        bank("comfort", "comfort")
    if "audit" in jobs:
        job_audit(arms, out, lath_term)
        bank("audit", "audit")

    print(f"[bt] done in {time.time() - t0:.1f} s", flush=True)
    print("BOUNDED_TERMS_DONE", flush=True)


if __name__ == "__main__":
    main()
