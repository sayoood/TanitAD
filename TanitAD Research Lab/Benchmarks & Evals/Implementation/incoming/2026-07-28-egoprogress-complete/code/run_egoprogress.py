#!/usr/bin/env python3
"""THE `ego_progress` UNDER SIDE (never audited) AND THE OVER-SIDE ZERO-MEAN DEFECT.

**No GPU. No model. No checkpoint. No corpus.** Every number is arithmetic over
the committed ``pw_<arm>.npz`` per-window pseudo-simulation dumps.
``taniteval.pseudosim`` / ``taniteval.control`` are IMPORTED, never reimplemented.

⛔ pod1 (TRAINING 23,000/30,000) and pod2 (small validation) are not contacted.

WHAT IT PRODUCES — in the brief's PRIORITY ORDER, each banked as it completes
---------------------------------------------------------------------------
``under_density.json``    the under side's density: how far below r = 0 the
                          panel actually goes, per arm, and the demonstration
                          that a plan reversing 1 m and one reversing 10 m score
                          IDENTICALLY. Plus the RESOLUTION probe (n_sub = 1).
``inject_under.json``     ⭐⭐ **PRIORITY 1 — THE UNDER-SIDE AUDIT.** 6 injections
                          x 3 real arms x every progress term, each control's
                          direction VERIFIED against a metric-independent ground
                          truth first, plus a reversing stress substrate and the
                          ZERO-MEAN cell that cannot re-centre a bias.
``under_fix.json``        the under-side range budget, PRICED on stand_still.
``inject_over.json``      ⭐ **PRIORITY 2** — the over-side grid: reproduce the
                          inherited failing cell exactly, then sweep the three
                          tail classes (convex / linear / concave).
``contamination.json``    ⭐ **PRIORITY 3** — purely LATERAL degradations must
                          move this longitudinal axis LEAST.
``repro_gate.json``       ✅ the 16 published ``@clamp_v1`` composites at
                          ``max|diff| = 0.000000``.
``panel.json``            the 20-arm ranking statement + the two instrument
                          guards under every candidate term.
``lat_heading_weight.json`` ⚠️ **PRIORITY 4** — CONTROL_WEIGHTS' 1.0 for
                          ``lat_heading`` was derived for the PUBLISHED term,
                          whose live range is twice the shipped one's.

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
ORACLES = ("v1_ego_oracle_lon", "v4_oracle", "v1_tactical_oracle",
           "nospeed_tactical_oracle", "oracle_lon_straight")

#: PUBLISHED `@clamp_v1` composites — the reproduction gate, the same 16 values
#: three consecutive repairs have each reproduced at 0.000000.
PUBLISHED_CLAMP_V1 = {
    "cv_holdv0": 0.5705, "v4_oracle": 0.5622, "refc_xl_produced": 0.5499,
    "v1_tactical_follow": 0.5471, "v1_tactical_oracle": 0.5467,
    "refc_small_produced": 0.5444, "refc_base_produced": 0.5439,
    "nospeed_tactical_oracle": 0.5394, "v4_blind": 0.3749,
    "v1_ego_v0": 0.5608, "v1_ego_oracle_lon": 0.5946, "v1_lat_straight": 0.5460,
    "refc_base_v0on": 0.5439, "refc_base_v0off": 0.4980,
    "refc_xl_v0on": 0.5499, "v1_ego_half": 0.3117,
}

#: ⛔ INHERITED, and asserted rather than trusted: the over-side cells published
#: by `…/2026-07-28-bounded-terms-complete/raw/injections_ego_progress.json`.
#: If my run does not reproduce these EXACTLY the two runs disagree and nothing
#: downstream is quotable.
INHERITED_OVER_CELLS = {
    "twosided_v2": {
        "v1_ego_double|lon_jitter(+2)": 0.0104,
    },
}

# =========================================================================== #
# THE TERMS UNDER TEST                                                         #
# =========================================================================== #
#: every one is a STRICT REFINEMENT of the published term (bit-identical for
#: r <= 1); they differ ONLY in the over-side tail, and they span the three
#: curvature classes the trichotomy names.
PROG_TERMS = ("clamp_v1", "twosided_v2",
              "twosided_asym_w0p5", "twosided_asym_w0p3333", "twosided_asym_w2",
              "hyp_w1", "hyp_w0p5", "hyp_w2", "exp_w1", "exp_w0p5",
              "sqrtlin_w1", "sqrtlin_w0p5", "sqrtlin_w0p3333", "sqrtlin_w0p25")
#: the RESOLUTION lever (the brief's ⭐ free lever) applied to the two terms it
#: could plausibly rescue. `term` = published terminal resolution (n_sub = 1);
#: `mean` = mean over the per-step ratios with human_k > PROGRESS_HUMAN_MIN_M.
PROG_RESOLUTIONS = ("term", "mean")
RESOLUTION_TERMS = ("clamp_v1", "twosided_v2", "sqrtlin_w0p3333", "hyp_w1")
#: every candidate, resolution-qualified. ``"mean|twosided_v2"`` reads
#: "the twosided_v2 shape at the 20-step mean resolution".
ALL_SPECS = PROG_TERMS + tuple(f"mean|{t}" for t in RESOLUTION_TERMS)

# =========================================================================== #
# ⭐⭐ THE UNDER-SIDE SUITE — the half nobody has ever probed                    #
# =========================================================================== #
#: ⛔ `lon_shift` is NEW (added to control.py by this stream). Without it the
#: under floor CANNOT be probed: `lon_scale` MULTIPLIES, so it moves a row at
#: r = 0 not at all, and `lon_retime` re-samples along the plan's own arc.
UNDER_INJECTIONS = (("lon_retime", 0.7), ("lon_retime", 0.5),
                    ("lon_scale", 0.5),
                    ("lon_shift", -2.0), ("lon_shift", -5.0),
                    ("lon_jitter", 2.0))         # ⭐ the ZERO-MEAN one
#: cv_holdv0 and v1_tactical_follow are the two real arms every sibling suite
#: used. v4_blind is THE DECISIVE SUBSTRATE: 31.78 % of its defined rows sit at
#: ego_progress = 0 with r <= 0 and no injection suite has ever touched them.
UNDER_ARMS = ("cv_holdv0", "v1_tactical_follow", "v4_blind")
#: ⛔ SYNTHETIC. Never inserted into `arms`, never votes on a gate or a ranking.
REVERSE_SUBSTRATE = ("v1_tactical_follow", -0.5)
REVERSE_INJECTIONS = (("lon_shift", -5.0), ("lon_scale", 2.0),
                      ("lon_jitter", 2.0))

# =========================================================================== #
# THE OVER-SIDE SUITE — inherited VERBATIM so the failing cell reproduces       #
# =========================================================================== #
OVER_INJECTIONS = (("lon_retime", 1.5), ("lon_retime", 2.0),
                   ("lon_scale", 1.5), ("lon_jitter", 2.0))
OVER_ARMS = ("cv_holdv0", "v1_tactical_follow", "v1_ego_double")

#: ⚠️ THE CONTAMINATION PANEL. `ego_progress` is a LONGITUDINAL axis; a purely
#: lateral degradation must move it LEAST. The plain-mean `lat_heading`
#: candidate scored 10/10 on its own injections and was killed here.
CONTAM_CONTROLS = (("lat_shift", 2.0), ("lat_shift", -2.0),
                   ("lat_jitter", 1.0), ("yaw_bias", 5.0))
CONTAM_PURE = (("lat_shift", 2.0), ("lat_shift", -2.0), ("lat_jitter", 1.0))


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

    ⛔ IT IS NEVER PUT INTO `arms`. A sibling stream's reproduction gate caught
    it voting on the PANEL-WIDE gate and silently redefining the composite for
    every arm (max|diff| 0.000000 -> 0.393900)."""
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


# --------------------------------------------------------------------------- #
# the quantities                                                               #
# --------------------------------------------------------------------------- #
#: ⛔⛔ A NEAR-MISS WORTH RECORDING. This memo was first keyed on ``id(pw)``
#: alone. CPython REUSES an object's id once it is garbage collected, so a
#: temporary plan built inside one job could hand its cached scores to a
#: DIFFERENT plan built later at the same address. It was caught because the
#: over-side cells stopped reproducing the inherited ones — i.e. by the
#: reproduction assertion, not by inspection. ⇒ ``_VKEEP`` holds a strong
#: reference to every keyed object, which makes id reuse impossible by
#: construction. **A cache keyed on a REUSABLE identity is a correctness bug,
#: not a performance detail.**
_VCACHE: dict = {}
_VKEEP: list = []
_VSEEN: set = set()


def prog_values(pw, term, resolution="term"):
    """`ego_progress` under a term AND a RESOLUTION.

    ``term``  — the published terminal value, straight out of `score_windows`,
                so the published path is exercised rather than re-derived.
    ``mean``  — the mean over the per-step ratios (`PS.progress_ratios_per_step`)
                restricted to steps with ``human_k > PROGRESS_HUMAN_MIN_M``, the
                PUBLISHED constant reused verbatim. Row definedness is the
                published one, bit-for-bit."""
    ck = (id(pw), term, resolution)
    if ck in _VCACHE:
        return _VCACHE[ck]
    if id(pw) not in _VSEEN:               # ⛔ pin it so the id cannot be reused
        _VSEEN.add(id(pw))
        _VKEEP.append(pw)
    if resolution == "term":
        v = np.asarray(PS.score_windows(pw, progress_term=term)
                       ["ego_progress"], float)
        _VCACHE[ck] = v
        return v
    if resolution != "mean":
        raise ValueError(f"unknown resolution {resolution!r}")
    r, step_def, row_def = PS.progress_ratios_per_step(pw)
    g = PS.progress_from_ratio(r, term)
    g = torch.where(step_def, g, torch.full_like(g, float("nan")))
    v = torch.nanmean(g, dim=1)
    v = torch.where(row_def, v, torch.full_like(v, float("nan")))
    out = np.asarray(v, float)
    _VCACHE[ck] = out
    return out


def ground_truth_along_err(pw):
    """⭐⭐ THE METRIC-INDEPENDENT GROUND TRUTH — |s_plan − s_human| in metres.

    A control's DIRECTION is verified against this and never against its name.
    It contains no clamp, no term, no bound: it is the along-track endpoint
    error the score is supposed to be a bounded proxy for."""
    x, _y, ref_x, ref_y = PS._cross_and_along(pw)
    human = torch.sqrt((ref_x[:, -1] - ref_x[:, 0]) ** 2
                       + (ref_y[:, -1] - ref_y[:, 0]) ** 2)
    err = (x[:, -1] - human).abs()
    out = np.asarray(err, float)
    return np.where(np.asarray(human, float) > PS.PROGRESS_HUMAN_MIN_M,
                    out, np.nan)


def _pss_composite(sc, ego_progress=None):
    """The PSS composite under the shipped weights, per row.

    ``ego_progress`` may be substituted so a RESOLUTION change is scored through
    the identical composite arithmetic rather than a parallel one."""
    parts = dict(sc)
    if ego_progress is not None:
        parts["ego_progress"] = ego_progress
    ref = np.asarray(parts["ego_progress"], float)
    num, den = np.zeros_like(ref), np.zeros_like(ref)
    for nm, wt in PS.COMPONENT_WEIGHTS.items():
        if float(wt) == 0.0:
            continue
        v = np.asarray(parts[nm], float)
        fin = np.isfinite(v)
        num = num + np.where(fin, v * wt, 0.0)
        den = den + np.where(fin, wt, 0.0)
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def _split(spec):
    """``"mean|twosided_v2"`` -> ``("mean", "twosided_v2")``; bare -> term."""
    if "|" in spec:
        res, term = spec.split("|", 1)
        return res, term
    return "term", spec


def composite_of(pw, spec):
    res, term = _split(spec)
    sc = PS.score_windows(pw, progress_term=term)
    ego = None if res == "term" else prog_values(pw, term, res)
    return _pss_composite(sc, ego)


def _paired(v, base, eid, n_boot):
    m = np.isfinite(v) & np.isfinite(base)
    b = _ci.paired_episode_cluster_bootstrap(v[m], base[m], list(eid[m]),
                                             n_boot=n_boot)
    b["defined_frac"] = round(float(np.isfinite(v).mean()), 6)
    b["defined_frac_base"] = round(float(np.isfinite(base).mean()), 6)
    b["defined_frac_delta_pp"] = round(
        100.0 * (b["defined_frac"] - b["defined_frac_base"]), 4)
    return b


def _tag(arm, ctl, lv):
    return f"{arm}|{ctl}({lv:+g})"


# =========================================================================== #
# PRIORITY 0 — THE UNDER-SIDE DENSITY                                          #
# =========================================================================== #
def job_under_density(arms, out):
    """How far below r = 0 does the panel actually go, and does anything change
    when it goes further? ⛔ The question no report has asked."""
    per = {}
    for n, pw in arms.items():
        sc = PS.score_windows(pw, progress_term="twosided_v2")
        r = np.asarray(sc["ego_progress_raw_ratio"], float)
        g = np.asarray(sc["ego_progress"], float)
        fin = np.isfinite(g)
        rf = r[fin]
        under = rf[rf <= 0]
        # the RESOLUTION probe: floor fraction at n_sub = 1 vs a 20-step mean
        node = {
            "n_defined": int(fin.sum()),
            "frac_r_le_0": round(float((rf <= 0).mean()), 4),
            "frac_r_lt_0_STRICT": round(float((rf < 0).mean()), 4),
            "frac_r_eq_0_EXACT": round(float((rf == 0).mean()), 4),
            "frac_0_lt_r_lt_1": round(float(((rf > 0) & (rf < 1)).mean()), 4),
            "median_r": round(float(np.median(rf)), 4),
            "min_r": round(float(rf.min()), 4),
            "p01_r": round(float(np.percentile(rf, 1)), 4),
        }
        if under.size:
            node.update({
                "under_median_r": round(float(np.median(under)), 4),
                "under_p05_r": round(float(np.percentile(under, 5)), 4),
                "under_min_r": round(float(under.min()), 4),
                "under_frac_strictly_below_m0p10": round(
                    float((under < -0.10).mean()), 4),
            })
        for res in PROG_RESOLUTIONS:
            v = prog_values(pw, "twosided_v2", res)
            s = PS.saturation(v, n_sub=1 if res == "term" else 20)
            node[f"{res}_floor"] = s["floor_frac_le_0p001"]
            node[f"{res}_ceiling"] = s["ceiling_frac_ge_0p999"]
            node[f"{res}_live_frac"] = s["live_frac"]
        per[n] = node

    # ⭐ THE DEMONSTRATION the brief asks for, on real rows and on a FIXED row
    # set: the SAME rows, pushed further and further backwards, keep the SAME
    # score while their ground-truth error grows without bound.
    ref = arms["v1_tactical_follow"]
    r0 = np.asarray(PS.score_windows(ref, progress_term="clamp_v1")
                    ["ego_progress_raw_ratio"], float)
    g0 = np.asarray(PS.score_windows(ref, progress_term="clamp_v1")
                    ["ego_progress"], float)
    # the rows that are ALREADY at r <= 0 once a 1 m back-shift is applied —
    # held FIXED across every deeper shift, so nothing is re-selected.
    r1 = np.asarray(PS.score_windows(C.apply_control(ref, "lon_shift", -1.0),
                                     progress_term="clamp_v1")
                    ["ego_progress_raw_ratio"], float)
    fixed = np.isfinite(g0) & (r1 <= 0)
    demo = {"_fixed_row_set": {
        "definition": ("rows with r <= 0 after lon_shift(-1 m), on "
                       "v1_tactical_follow — HELD FIXED at every deeper shift "
                       "so no row is re-selected"),
        "n_rows": int(fixed.sum()),
        "frac_of_defined": round(float(fixed.sum() / np.isfinite(g0).sum()), 6)}}
    for d in (-1.0, -3.0, -10.0, -30.0):
        pwd = C.apply_control(ref, "lon_shift", d)
        gt = ground_truth_along_err(pwd)
        rr = np.asarray(PS.score_windows(pwd, progress_term="clamp_v1")
                        ["ego_progress_raw_ratio"], float)
        row = {"mean_ground_truth_along_err_m_ON_THE_FIXED_ROWS": round(
                   float(np.nanmean(gt[fixed])), 4),
               "mean_ratio_r_ON_THE_FIXED_ROWS": round(
                   float(np.nanmean(rr[fixed])), 4),
               "arm_wide_frac_r_le_0": round(
                   float((rr[np.isfinite(g0)] <= 0).mean()), 4),
               "arm_wide_mean_ground_truth_along_err_m": round(
                   float(np.nanmean(gt)), 4)}
        for term in PROG_TERMS:
            v = prog_values(pwd, term)
            row[f"mean_score_ON_THE_FIXED_ROWS@{term}"] = round(
                float(np.nanmean(v[fixed])), 10)
        for term in ("clamp_v1", "twosided_v2"):
            row[f"arm_wide_mean_score@{term}"] = round(
                float(np.nanmean(prog_values(pwd, term))), 6)
            row[f"arm_wide_mean_score@mean|{term}"] = round(
                float(np.nanmean(prog_values(pwd, term, "mean"))), 6)
        demo[f"lon_shift({d:+g})"] = row
    out["under_density"] = {
        "_what": ("The under side of ego_progress (r <= 0). It is the published "
                  "clamp_v1 floor, it predates both 2026-07-28 repairs, and no "
                  "injection suite in this programme has ever probed it."),
        "per_arm": per,
        "_identical_score_demonstration": demo,
        "_read_the_demonstration": (
            "⛔ The mean score over the r <= 0 rows is EXACTLY 0.0 at every "
            "back-shift while the ground-truth along-track error grows without "
            "bound. A plan reversing 1 m and one reversing 30 m are the same "
            "number. That is the defect, shown rather than argued."),
    }
    print("[ep] under-side density banked", flush=True)
    return out


# =========================================================================== #
# PRIORITY 1 — ⭐⭐ THE UNDER-SIDE AUDIT                                         #
# =========================================================================== #
def job_inject_under(arms, out, n_boot):
    """Every control's DIRECTION is verified against |s_plan − s_human| BEFORE
    any metric verdict is read off it. A control that fails verification on an
    arm is REFUSED for that arm and does not count as a metric failure."""
    res = {"_read": ("delta > 0 means the DEGRADED plan scores HIGHER (the C45 "
                     "defect). A cell PASSES only if delta < 0 AND separated, "
                     "and only if its control's direction was VERIFIED."),
           "_arms": list(UNDER_ARMS),
           "_injections": [f"{c}({l:+g})" for c, l in UNDER_INJECTIONS],
           "_zero_mean": [c for c, _ in UNDER_INJECTIONS
                          if c in C.ZERO_MEAN_CONTROLS],
           "n_boot": n_boot,
           "_control_direction_rule": (
               "⭐ A control counts as an UNDER-TRAVEL DEGRADATION on an arm "
               "only if it raises the METRIC-INDEPENDENT along-track endpoint "
               "error |s_plan - s_human| (paired bootstrap, SEPARATED). "
               "lon_shift and lon_scale have a CONSTANT SIGN and can RE-CENTRE "
               "an arm biased the other way — control._ctl_lat_shift's own "
               "docstring warns of exactly this."),
           "direction_check": {}, "terms": {}}

    # ---- geometry once ----------------------------------------------------- #
    cache, eids = {}, {}
    for arm in UNDER_ARMS:
        eids[arm] = np.asarray(arms[arm]["eid"])
        cache[(arm, None)] = arms[arm]
        for ctl, lv in UNDER_INJECTIONS:
            cache[(arm, (ctl, lv))] = C.apply_control(arms[arm], ctl, lv)

    # ---- ⭐ STEP 1: VERIFY EVERY CONTROL'S DIRECTION ------------------------ #
    verified = {}
    for arm in UNDER_ARMS:
        gt0 = ground_truth_along_err(cache[(arm, None)])
        for ctl, lv in UNDER_INJECTIONS:
            gt = ground_truth_along_err(cache[(arm, (ctl, lv))])
            b = _paired(gt, gt0, eids[arm], n_boot)
            ok = bool(b["separated"] and b["delta"] > 0)
            b["DEGRADES_GROUND_TRUTH"] = ok
            b["units"] = "m (|s_plan - s_human|); delta > 0 means MORE error"
            res["direction_check"][_tag(arm, ctl, lv)] = b
            verified[(arm, ctl, lv)] = ok
            if not ok:
                b["_refused"] = (
                    "⛔ REFUSED AS A DEGRADATION ON THIS ARM — it does not "
                    "raise the ground-truth along-track error here, so a "
                    "metric moving up under it is NOT evidence of a defect. "
                    "This is the re-centring hazard, caught by measurement.")
    n_ver = sum(verified.values())
    res["_n_verified_cells"] = int(n_ver)
    res["_n_cells"] = int(len(verified))
    print(f"  [under] {n_ver}/{len(verified)} controls verified as degradations",
          flush=True)

    # ---- STEP 2: the metric, on the verified cells -------------------------- #
    for spec in ALL_SPECS:
        resn, term = _split(spec)
        cells, n_ok, n_tot, n_wrong = {}, 0, 0, 0
        for arm in UNDER_ARMS:
            base = prog_values(cache[(arm, None)], term, resn)
            base_cc = composite_of(cache[(arm, None)], spec)
            for ctl, lv in UNDER_INJECTIONS:
                pwv = cache[(arm, (ctl, lv))]
                v = prog_values(pwv, term, resn)
                blk = {"ego_progress": _paired(v, base, eids[arm], n_boot)}
                cc = composite_of(pwv, spec)
                blk["PSS_composite"] = _paired(cc, base_cc, eids[arm], n_boot)
                blk["zero_mean_control"] = ctl in C.ZERO_MEAN_CONTROLS
                blk["control_direction_VERIFIED"] = verified[(arm, ctl, lv)]
                # ⛔ U-4: ego_progress definedness is a property of the LOGGED
                # path, so it MUST be invariant. Asserted, not assumed.
                blk["ego_progress_defined_frac_INVARIANT"] = bool(
                    blk["ego_progress"]["defined_frac"]
                    == blk["ego_progress"]["defined_frac_base"])
                d = blk["ego_progress"]
                ok = bool(d["separated"] and d["delta"] < 0)
                wrong = bool(d["separated"] and d["delta"] > 0)
                blk["CORRECT_DIRECTION"] = ok
                blk["SEPARATED_THE_WRONG_WAY"] = wrong
                cells[_tag(arm, ctl, lv)] = blk
                if verified[(arm, ctl, lv)]:
                    n_tot += 1
                    n_ok += int(ok)
                    n_wrong += int(wrong)
        res["terms"][spec] = {"resolution": resn, "shape": term,
                              "n_correct": n_ok, "n_verified_cells": n_tot,
                              "n_separated_WRONG_WAY": n_wrong,
                              "ALL_CORRECT": bool(n_tot and n_ok == n_tot),
                              "NO_CELL_WRONG_WAY": bool(n_wrong == 0),
                              "cells": cells}
        print(f"  [under] {spec:24s} {n_ok}/{n_tot} correct, "
              f"{n_wrong} WRONG WAY", flush=True)

    # ---- STEP 3: the reversing stress substrate ----------------------------- #
    src, k = REVERSE_SUBSTRATE
    rev = C.apply_control(arms[src], "lon_scale", k)
    eid = np.asarray(arms[src]["eid"])
    scr = PS.score_windows(rev, progress_term="twosided_v2")
    rr = np.asarray(scr["ego_progress_raw_ratio"], float)
    gg = np.asarray(scr["ego_progress"], float)
    fin = np.isfinite(gg)
    stress = {"_what": (f"SYNTHETIC substrate: lon_scale({k}) applied to {src} "
                        "— a plan that travels BACKWARDS at half the human's "
                        "speed. ⛔ Never inserted into `arms`, never votes on a "
                        "gate or a ranking. Same role v1_ego_double played for "
                        "the over side."),
              "frac_r_le_0": round(float((rr[fin] <= 0).mean()), 4),
              "median_r": round(float(np.median(rr[fin])), 4),
              "base_mean_ground_truth_along_err_m": round(
                  float(np.nanmean(ground_truth_along_err(rev))), 4),
              "base_levels": {}, "injected_levels": {}, "cells": {}}
    for spec in ALL_SPECS:
        resn, term = _split(spec)
        stress["base_levels"][spec] = round(
            float(np.nanmean(prog_values(rev, term, resn))), 6)
    for ctl, lv in REVERSE_INJECTIONS:
        pwd = C.apply_control(rev, ctl, lv)
        stress["injected_levels"][f"{ctl}({lv:+g})"] = {
            "mean_ground_truth_along_err_m": round(
                float(np.nanmean(ground_truth_along_err(pwd))), 4),
            "mean_score@twosided_v2": round(
                float(np.nanmean(prog_values(pwd, "twosided_v2"))), 6),
            "mean_score@clamp_v1": round(
                float(np.nanmean(prog_values(pwd, "clamp_v1"))), 6)}
    for spec in ALL_SPECS:
        resn, term = _split(spec)
        base = prog_values(rev, term, resn)
        for ctl, lv in REVERSE_INJECTIONS:
            v = prog_values(C.apply_control(rev, ctl, lv), term, resn)
            b = _paired(v, base, eid, n_boot)
            b["zero_mean_control"] = ctl in C.ZERO_MEAN_CONTROLS
            b["CORRECT_DIRECTION"] = bool(b["separated"] and b["delta"] < 0)
            b["SEPARATED_THE_WRONG_WAY"] = bool(b["separated"]
                                                and b["delta"] > 0)
            stress["cells"][f"{spec} @ {ctl}({lv:+g})"] = b
    gt0 = ground_truth_along_err(rev)
    stress["direction_check"] = {}
    for ctl, lv in REVERSE_INJECTIONS:
        gt = ground_truth_along_err(C.apply_control(rev, ctl, lv))
        b = _paired(gt, gt0, eid, n_boot)
        b["DEGRADES_GROUND_TRUTH"] = bool(b["separated"] and b["delta"] > 0)
        stress["direction_check"][f"{ctl}({lv:+g})"] = b
    res["reversing_stress_substrate"] = stress
    print("[ep] under-side injections banked", flush=True)
    out["inject_under"] = res
    return out


# =========================================================================== #
# THE UNDER-SIDE FIX — named, PRICED, and refused on a number                  #
# =========================================================================== #
def job_under_fix(arms, out):
    """⛔ A strict refinement of the under side is arithmetically impossible; the
    only lever is a RANGE BUDGET g(0) = p > 0, and it pays a plan that does not
    move. Here is what it costs, per arm, rather than an argument."""
    def budget(ratio, p):
        """g(r) = p + (1-p) * clamp_v1(r) for r >= 0; below 0 it decays to 0 at
        r = -1 with slope p. The closest analogue of recovery's `q`."""
        base = ratio.clamp(0.0, 1.0)
        up = float(p) + (1.0 - float(p)) * base
        down = (float(p) * (1.0 + ratio)).clamp_min(0.0)
        return torch.where(ratio >= 0.0, up, down)

    per = {}
    for p in (0.0, 0.05, 0.10, 0.25):
        node = {}
        for n, pw in arms.items():
            sc = PS.score_windows(pw, progress_term="clamp_v1")
            r = torch.as_tensor(sc["ego_progress_raw_ratio"])
            g = np.asarray(budget(r, p), float)
            g = np.where(np.isfinite(np.asarray(sc["ego_progress"], float)),
                         g, np.nan)
            node[n] = round(float(np.nanmean(g)), 6)
        per[f"p={p:g}"] = node
    ss = {k: v.get("stand_still") for k, v in per.items()}
    out["under_fix"] = {
        "_impossibility": PS.PROGRESS_UNDER_FLOOR_IS_UNFIXABLE,
        "_family": ("g(r) = p + (1-p)*clamp_v1(r) for r >= 0; p*(1+r) clipped "
                    "at 0 below. p = 0 IS the published term, bit-identically."),
        "mean_ego_progress_per_arm_at_each_p": per,
        "⛔ stand_still_price": ss,
        "DECISION": (
            "REFUSED, on the measurement and not on the argument. stand_still "
            "does not move on any row and its ego_progress rises 0.000000 -> p "
            "for free at every p > 0. This programme has already paid for that "
            "trade once: the naive v0-based recovery denominator scored the "
            "BLIND arm +0.597 ABOVE the sighted one because a planner that "
            "barely moves has a small cross-track error. 'Standing still is "
            "not progress.' ⇒ the under floor is NAMED, PRICED and LEFT OPEN, "
            "and the honest instrument is to PUBLISH the under-floor fraction "
            "beside every ego_progress value — which `saturation` now does."),
    }
    print("[ep] under-side fix priced and refused", flush=True)
    return out


# =========================================================================== #
# PRIORITY 2 — THE OVER SIDE                                                   #
# =========================================================================== #
def job_inject_over(arms, out, n_boot, resolutions=("term",)):
    res = {"_read": ("delta > 0 means the OVER-TRAVELLING plan scores HIGHER "
                     "(the C45 defect). PASS requires delta < 0 AND separated."),
           "_arms": list(OVER_ARMS),
           "_injections": [f"{c}({l:+g})" for c, l in OVER_INJECTIONS],
           "_inherited_cells_asserted": INHERITED_OVER_CELLS,
           "n_boot": n_boot, "terms": {}}
    cache, eids = {}, {}
    for arm in OVER_ARMS:
        eids[arm] = np.asarray(arms[arm]["eid"])
        cache[(arm, None)] = arms[arm]
        for ctl, lv in OVER_INJECTIONS:
            cache[(arm, (ctl, lv))] = C.apply_control(arms[arm], ctl, lv)
    # direction check for the over side too — same rule, no exceptions
    dirchk = {}
    for arm in OVER_ARMS:
        gt0 = ground_truth_along_err(cache[(arm, None)])
        for ctl, lv in OVER_INJECTIONS:
            gt = ground_truth_along_err(cache[(arm, (ctl, lv))])
            b = _paired(gt, gt0, eids[arm], n_boot)
            b["DEGRADES_GROUND_TRUTH"] = bool(b["separated"] and b["delta"] > 0)
            dirchk[_tag(arm, ctl, lv)] = b
    res["direction_check"] = dirchk

    for spec in ALL_SPECS:
        resn, term = _split(spec)
        cells, n_ok, n_tot, n_wrong = {}, 0, 0, 0
        for arm in OVER_ARMS:
            base = prog_values(cache[(arm, None)], term, resn)
            for ctl, lv in OVER_INJECTIONS:
                v = prog_values(cache[(arm, (ctl, lv))], term, resn)
                b = _paired(v, base, eids[arm], n_boot)
                b["zero_mean_control"] = ctl in C.ZERO_MEAN_CONTROLS
                ok = bool(b["separated"] and b["delta"] < 0)
                wrong = bool(b["separated"] and b["delta"] > 0)
                b["CORRECT_DIRECTION"] = ok
                b["SEPARATED_THE_WRONG_WAY"] = wrong
                cells[_tag(arm, ctl, lv)] = b
                n_tot += 1
                n_ok += int(ok)
                n_wrong += int(wrong)
        # over-side floor fraction, every arm
        fl = {}
        for n, pw in arms.items():
            sc = PS.score_windows(pw, progress_term=term)
            g = prog_values(pw, term, resn)
            r = np.asarray(sc["ego_progress_raw_ratio"], float)
            fin = np.isfinite(g)
            if fin.any():
                fl[n] = round(float(((g[fin] <= 1e-3)
                                     & (r[fin] >= 1)).mean()), 4)
        res["terms"][spec] = {
            "resolution": resn, "shape": term,
            "n_correct": n_ok, "n_total": n_tot,
            "n_separated_WRONG_WAY": n_wrong,
            "ALL_CORRECT": bool(n_ok == n_tot and n_tot),
            "NO_CELL_WRONG_WAY": bool(n_wrong == 0),
            "over_side_floor_frac_max_scorable": round(float(max(
                v for k, v in fl.items() if k not in PROBES)), 4),
            "over_side_floor_frac_per_arm": fl,
            "cells": cells}
        print(f"  [over] {spec:24s} {n_ok}/{n_tot} correct, "
              f"{n_wrong} WRONG WAY", flush=True)
    # ⛔ assert the inherited failing cell reproduces
    chk = {}
    for term, cellmap in INHERITED_OVER_CELLS.items():
        for tag, want in cellmap.items():
            got = res["terms"][term]["cells"][tag]["delta"]
            chk[f"{term}|{tag}"] = {"inherited": want, "measured": got,
                                    "abs_diff": round(abs(got - want), 6),
                                    "REPRODUCES": bool(abs(got - want) < 5e-5)}
    res["_inherited_cell_reproduction"] = chk
    out["inject_over"] = res
    print("[ep] over-side injections banked", flush=True)
    return out


# =========================================================================== #
# PRIORITY 3 — THE CONTAMINATION PANEL                                         #
# =========================================================================== #
def job_contamination(arms, out, n_boot):
    """⚠️ A term can pass its injections and still be wrong. `ego_progress` is a
    LONGITUDINAL axis; a purely lateral degradation must move it LEAST."""
    res = {"_rule": ("PURITY: max|delta| over the pure lateral controls "
                     "(lat_shift +-2 m, lat_jitter 1 m) must be < min|delta| "
                     "over the VERIFIED longitudinal injections, on both real "
                     "arms. yaw_bias is reported but does NOT carry the rule: "
                     "rotating a plan genuinely shortens its along-track "
                     "projection, so it is expected to move this axis."),
           "n_boot": n_boot, "terms": {}}
    arms_ = ("cv_holdv0", "v1_tactical_follow")
    cache, eids = {}, {}
    for arm in arms_:
        eids[arm] = np.asarray(arms[arm]["eid"])
        cache[(arm, None)] = arms[arm]
        for ctl, lv in CONTAM_CONTROLS:
            cache[(arm, (ctl, lv))] = C.apply_control(arms[arm], ctl, lv)
        for ctl, lv in OVER_INJECTIONS + UNDER_INJECTIONS:
            if (arm, (ctl, lv)) not in cache:
                cache[(arm, (ctl, lv))] = C.apply_control(arms[arm], ctl, lv)
    for spec in ALL_SPECS:
        resn, term = _split(spec)
        node = {"resolution": resn, "shape": term,
                "contamination_cells": {}, "purity": {}}
        for arm in arms_:
            base = prog_values(cache[(arm, None)], term, resn)
            lat = []
            for ctl, lv in CONTAM_CONTROLS:
                b = _paired(prog_values(cache[(arm, (ctl, lv))], term, resn),
                            base, eids[arm], n_boot)
                node["contamination_cells"][_tag(arm, ctl, lv)] = b
                if (ctl, lv) in CONTAM_PURE:
                    lat.append(abs(b["delta"]))
            lon = []
            for ctl, lv in OVER_INJECTIONS + UNDER_INJECTIONS:
                b = _paired(prog_values(cache[(arm, (ctl, lv))], term, resn),
                            base, eids[arm], n_boot)
                lon.append(abs(b["delta"]))
            node["purity"][arm] = {
                "max_abs_delta_pure_lateral": round(float(max(lat)), 4),
                "min_abs_delta_longitudinal": round(float(min(lon)), 4),
                "PURE": bool(max(lat) < min(lon)),
                "margin_x": round(float(min(lon) / max(max(lat), 1e-9)), 2)}
        node["PURE_BOTH_ARMS"] = bool(all(v["PURE"]
                                          for v in node["purity"].values()))
        res["terms"][spec] = node
        print(f"  [contam] {spec:24s} pure={node['PURE_BOTH_ARMS']} "
              f"margin={min(v['margin_x'] for v in node['purity'].values()):.2f}x",
              flush=True)
    out["contamination"] = res
    return out


# =========================================================================== #
# THE REPRODUCTION GATE                                                        #
# =========================================================================== #
def job_repro(arms, out, n_boot):
    """✅ Every published `@clamp_v1` composite must still reproduce EXACTLY."""
    rows, worst = {}, 0.0
    for n, want in PUBLISHED_CLAMP_V1.items():
        if n not in arms:
            rows[n] = {"published": want, "measured": None,
                       "note": "arm not present in these dumps"}
            continue
        sc = PS.score_windows(arms[n], progress_term="clamp_v1",
                              recovery_term="clamp_v1")
        got = float(np.nanmean(_pss_composite(sc)))
        d = abs(round(got, 4) - want)
        worst = max(worst, d)
        rows[n] = {"published": want, "measured": round(got, 6),
                   "abs_diff": round(d, 6)}
    # the published TERM must be bit-identical under every new strict refinement
    bits = {}
    for term in PROG_TERMS:
        same = []
        for n, pw in arms.items():
            sc0 = PS.score_windows(pw, progress_term="clamp_v1")
            r = torch.as_tensor(sc0["ego_progress_raw_ratio"])
            a = PS.progress_from_ratio(r, "clamp_v1")
            b = PS.progress_from_ratio(r, term)
            und = r <= 1.0
            same.append(bool(torch.equal(a[und], b[und])))
        bits[term] = {"bit_identical_to_clamp_v1_on_every_r_le_1_row":
                      bool(all(same)), "n_arms": len(same)}
    out["repro_gate"] = {
        "n_published_composites_checked": sum(
            1 for v in rows.values() if v.get("measured") is not None),
        "max_abs_diff": round(worst, 6),
        "PASS": bool(worst < 5e-5),
        "per_arm": rows,
        "strict_refinement_bit_identity": bits,
        "_published_metric_id": PS.metric_id(progress_term="clamp_v1",
                                             recovery_term="clamp_v1"),
        "_default_metric_id": PS.metric_id(),
        "_note": ("The published values are computed by NAMING clamp_v1 "
                  "explicitly, so they are immune to any default change — that "
                  "is the whole point of the versioned term registry."),
    }
    print(f"[ep] repro gate max|diff| = {worst:.6f} "
          f"PASS={out['repro_gate']['PASS']}", flush=True)
    return out


# =========================================================================== #
# THE PANEL — the ranking statement and the two instrument guards              #
# =========================================================================== #
def _ranked(levels):
    order = sorted(levels, key=lambda k: -levels[k])
    return {n: i + 1 for i, n in enumerate(order)}


def job_panel(arms, out, n_boot, terms):
    res = {"_gate": "PANEL-WIDE. The per-arm gate is REFUSED, not offered.",
           "n_boot": n_boot, "terms": {}}
    eid = {n: np.asarray(pw["eid"]) for n, pw in arms.items()}
    for term in terms:
        lv, cc, ego = {}, {}, {}
        for n, pw in arms.items():
            v = composite_of(pw, term)
            cc[n] = v
            lv[n] = round(float(np.nanmean(v)), 4)
            res_, t_ = _split(term)
            e = prog_values(pw, t_, res_)
            s = PS.saturation(e, n_sub=1 if res_ == "term" else 20)
            ego[n] = {"mean": round(float(np.nanmean(e)), 4),
                      "floor": s["floor_frac_le_0p001"],
                      "ceiling": s["ceiling_frac_ge_0p999"],
                      "live_frac": s["live_frac"]}
        realis = {k: v for k, v in lv.items()
                  if k not in PROBES and k not in ORACLES}
        node = {"levels": lv, "rank_all": _ranked(lv),
                "rank_realisable": _ranked(realis),
                "cv_holdv0_rank_among_realisable": _ranked(realis)["cv_holdv0"],
                "ego_progress_axis": ego,
                "ego_progress_min_live_frac_scorable": round(float(min(
                    v["live_frac"] for k, v in ego.items()
                    if k not in PROBES and v["live_frac"] is not None)), 4),
                "guards": {}}
        node["GATE_V2_ADMITS_ego_progress"] = bool(
            node["ego_progress_min_live_frac_scorable"] >= PS.LIVE_FRAC_MIN)
        for a, b in (("v4_oracle", "v4_blind"),
                     ("v1_ego_half", "v1_tactical_follow")):
            m = np.isfinite(cc[a]) & np.isfinite(cc[b])
            node["guards"][f"{a} - {b}"] = _ci.paired_episode_cluster_bootstrap(
                cc[a][m], cc[b][m], list(eid[a][m]), n_boot=n_boot)
        # v1 TACTICAL family vs every REF-C arm
        v1t = ("v1_tactical_follow", "v1_tactical_oracle", "v1_lat_straight",
               "nospeed_tactical_oracle")
        refc = [k for k in lv if k.startswith("refc_")]
        node["v1_tactical_family_below_every_REFC"] = bool(
            max(lv[k] for k in v1t if k in lv) < min(lv[k] for k in refc))
        res["terms"][term] = node
        print(f"  [panel] {term:22s} cv_holdv0 realisable rank "
              f"{node['cv_holdv0_rank_among_realisable']}, "
              f"v4_oracle-v4_blind "
              f"{node['guards']['v4_oracle - v4_blind']['delta']:+.4f}",
              flush=True)
    # 10 paired contrasts, published vs candidate
    pairs = [("cv_holdv0", "v1_tactical_follow"),
             ("cv_holdv0", "refc_xl_produced"),
             ("cv_holdv0", "v4_blind"),
             ("refc_xl_produced", "v1_tactical_follow"),
             ("refc_base_produced", "v1_tactical_follow"),
             ("refc_small_produced", "v1_tactical_follow"),
             ("refc_xl_v0on", "refc_xl_v0off"),
             ("refc_base_v0on", "refc_base_v0off"),
             ("v4_oracle", "v4_blind"),
             ("v1_ego_half", "v1_tactical_follow")]
    contrasts = {}
    ref_term = terms[0]
    for a, b in pairs:
        row = {}
        for term in terms:
            ca = composite_of(arms[a], term)
            cb = composite_of(arms[b], term)
            m = np.isfinite(ca) & np.isfinite(cb)
            r = _ci.paired_episode_cluster_bootstrap(ca[m], cb[m],
                                                     list(eid[a][m]),
                                                     n_boot=n_boot)
            row[term] = {"delta": r["delta"], "separated": r["separated"]}
        s0 = np.sign(row[ref_term]["delta"])
        row["FLIPS"] = {t: bool(np.sign(row[t]["delta"]) != s0)
                        for t in terms if t != ref_term}
        contrasts[f"{a} - {b}"] = row
    res["paired_contrasts"] = contrasts
    res["_reference_term_for_flips"] = ref_term
    res["n_flipped"] = {t: int(sum(v["FLIPS"][t] for v in contrasts.values()))
                        for t in terms if t != ref_term}
    out["panel"] = res
    print("[ep] panel banked", flush=True)
    return out


# =========================================================================== #
# PRIORITY 4 — THE `lat_heading` WEIGHT                                        #
# =========================================================================== #
def job_lat_heading_weight(arms, out):
    """⚠️ CONTROL_WEIGHTS puts lat_heading at 1.0. That number was derived for
    the PUBLISHED term. The shipped term's live range is HALF (q = 0.5), so
    weight 1.0 now buys half the influence the proposal assumed."""
    def lath(pw, term):
        r = C.residuals(pw)
        v = C.lat_heading_from_err(r["heading_err_rad"],
                                   r["heading_err_rad_steps"],
                                   lat_heading_term=term)
        return np.where(r["lat_mask"], v, np.nan)

    stats = {}
    for term in ("term_lin_q0", C.LAT_HEADING_TERM_DEFAULT_TARGET):
        per, pool = {}, []
        for n, pw in arms.items():
            v = lath(pw, term)
            f = v[np.isfinite(v)]
            if not f.size:
                continue
            if n not in PROBES:
                pool.append(f)
            per[n] = {"mean": round(float(f.mean()), 4),
                      "sd": round(float(f.std()), 4),
                      "p05": round(float(np.percentile(f, 5)), 4),
                      "p95": round(float(np.percentile(f, 95)), 4)}
        allv = np.concatenate(pool)
        # ⭐ the quantity a weight should be proportional to: the axis's
        # BETWEEN-ARM spread, which is what a composite actually aggregates.
        means = np.array([v["mean"] for k, v in per.items() if k not in PROBES])
        stats[term] = {
            "per_arm": per,
            "pooled_sd_within": round(float(allv.std()), 4),
            "pooled_p05_p95_span": round(float(np.percentile(allv, 95)
                                               - np.percentile(allv, 5)), 4),
            "BETWEEN_ARM_sd_of_means": round(float(means.std()), 4),
            "BETWEEN_ARM_span_of_means": round(
                float(means.max() - means.min()), 4)}
    a, b = "term_lin_q0", C.LAT_HEADING_TERM_DEFAULT_TARGET
    ratio_between = (stats[b]["BETWEEN_ARM_span_of_means"]
                     / max(stats[a]["BETWEEN_ARM_span_of_means"], 1e-9))
    out["lat_heading_weight"] = {
        "_why": ("A composite weight is only meaningful relative to the range "
                 "the axis actually exercises. Halving an axis's live range at "
                 "constant weight halves its influence on the composite, "
                 "silently."),
        "terms": stats,
        "shipped_over_published_BETWEEN_ARM_span": round(
            float(ratio_between), 4),
        "weight_that_PRESERVES_the_published_influence": round(
            float(C.CONTROL_WEIGHTS["lat_heading"] / max(ratio_between, 1e-9)),
            3),
        "CONTROL_WEIGHTS_today": dict(C.CONTROL_WEIGHTS),
        "STATUS": ("MEASURED and ESCALATED, not changed. CONTROL_WEIGHTS is a "
                   "PI decision and the control composite is being proposed as "
                   "the gate primary; a weight change made by an agent inside "
                   "an unrelated repair is exactly the silent redefinition "
                   "this programme keeps logging."),
    }
    print("[ep] lat_heading weight re-derived", flush=True)
    return out


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=_ci.DEFAULT_N_BOOT)
    ap.add_argument("--ref-arm", default="cv_holdv0")
    ap.add_argument("--panel-terms", default="clamp_v1,twosided_v2")
    ap.add_argument("--jobs", default=("udensity,under,underfix,over,contam,"
                                       "repro,panel,lathw"))
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
    print(f"[ep] {len(arms)} arms: {sorted(arms)}", flush=True)

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

    hr = human_replay(arms[ref])
    _r = C.residuals(hr)
    _worst = max(float(np.abs(_r["along_err_m"]).max()),
                 float(np.abs(_r["cross_err_m"]).max()),
                 float(np.abs(_r["xte_m"]).max()))
    assert _worst < 1e-3, (
        f"human_replay is NOT on the logged path (max |residual| = {_worst})")
    _hp = float(np.nanmean(prog_values(hr, "twosided_v2")))
    print(f"[ep] human_replay verified at max|residual| = {_worst:.3e} m, "
          f"ego_progress = {_hp:.4f} (kept OUT of the gate vote)", flush=True)

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    head = {
        "_experiment": ("THE ego_progress UNDER SIDE (never audited by any "
                        "injection suite) AND THE OVER-SIDE ZERO-MEAN DEFECT. "
                        "0 GPU-h."),
        "_evidence_class": "MEASURED (ours; artifact = these JSONs + pw_*.npz)",
        "_protocol": PS.PROTOCOL,
        "_estimator": (f"taniteval.ci.paired_episode_cluster_bootstrap "
                       f"(B={a.n_boot}, unit = val episode)"),
        "_refused_estimator": ("overlapping_holdout_se — it biases the POINT "
                               "ESTIMATE as well as the interval"),
        "_gate": "PANEL-WIDE. The per-arm gate is REFUSED, not offered.",
        "_hosts": ("dev box only. pod1 (TRAINING 23,000/30,000) and pod2 "
                   "(small validation) were NOT contacted."),
        "_import_provenance_md5": IMPORT_PROVENANCE,
        "_row_identity": row_id, "_refused_arms": refused,
        "_human_replay_max_residual_m": _worst,
        "_human_replay_ego_progress": round(_hp, 6),
        "_zero_bias_reference_kept_out_of_the_gate_vote": True,
        "_synthetic_substrates_kept_out_of_the_gate_vote": [
            "human_replay", "reverse_half = lon_scale(-0.5) o v1_tactical_follow"],
        "_parity": ("No episode is re-selected. All arms share the identical "
                    "row set and row identity is ASSERTED, not assumed. "
                    "Counts only — no clip UUID appears in any artifact."),
        "_preregistration": "raw/PREREGISTRATION.json (staged before any "
                            "under-side number was computed)",
    }
    jobs = [j.strip() for j in a.jobs.split(",") if j.strip()]
    out = dict(head)

    def bank(name, key=None):
        key = key or name
        p = od / f"{name}.json"
        p.write_text(json.dumps({**head, name: out[key]}, indent=1,
                                default=str), encoding="utf-8")
        print(f"[ep] wrote {p}  (+{time.time() - t0:.0f}s)", flush=True)

    if "udensity" in jobs:
        job_under_density(arms, out)
        bank("under_density")
    if "under" in jobs:
        job_inject_under(arms, out, a.n_boot)
        bank("inject_under")
    if "underfix" in jobs:
        job_under_fix(arms, out)
        bank("under_fix")
    if "over" in jobs:
        job_inject_over(arms, out, a.n_boot,
                        resolutions=PROG_RESOLUTIONS)
        bank("inject_over")
    if "contam" in jobs:
        job_contamination(arms, out, a.n_boot)
        bank("contamination")
    if "repro" in jobs:
        job_repro(arms, out, a.n_boot)
        bank("repro_gate")
    if "panel" in jobs:
        job_panel(arms, out, a.n_boot,
                  [t.strip() for t in a.panel_terms.split(",") if t.strip()])
        bank("panel")
    if "lathw" in jobs:
        job_lat_heading_weight(arms, out)
        bank("lat_heading_weight")
    print(f"[ep] DONE in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
