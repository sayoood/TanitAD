#!/usr/bin/env python3
"""FIX THE SECOND ONE-SIDED CLAMP: ``recovery``. C45, step 2 of the control-suite plan.

**No GPU. No model. No checkpoint. No corpus.** Every number is arithmetic over
the committed ``pw_<arm>.npz`` per-window pseudo-simulation dumps.
``taniteval.pseudosim`` / ``taniteval.control`` are IMPORTED, never reimplemented.

⛔ pod1 (TRAINING) and pod2 (small validation) are not contacted. Nothing here
needs a GPU.

WHAT IT PRODUCES (banked incrementally, in PRIORITY ORDER)
----------------------------------------------------------
``injections.json``     ⭐ **THE ACCEPTANCE TEST.** The 8 injected lateral
                        degradations that exposed the defect, re-run under the
                        published term AND under every member of the shape
                        family. All 8 must move the CORRECT way (negative and
                        separated) on BOTH arms, or the shape is disqualified.
``repro_gate.json``     ✅ every published ``@clamp_v1`` composite must still
                        reproduce at ``max|diff| = 0.000000`` with the new,
                        versioned code in place.
``panel_both_terms.json`` the 20-arm panel under the published and the new
                        recovery term, side by side, with the explicit ranking
                        statement and the paired contrasts.
``shape_sensitivity.json`` the ranking at every point of the ``q`` grid, in both
                        families — the ``w``-sweep treatment applied to ``q``.
``saturation_census.json`` ⚠️ C45's standing consequence: the floor/ceiling
                        fraction of EVERY bounded term on EVERY arm.
``term_audit.json``     ⛔ the audit of the REMAINING terms — is a third term
                        one-sidedly clamped? ``comfort`` is included **because it
                        is still measured**, and a saturating diagnostic misleads
                        exactly as a saturating score does.

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
# A stale tree on the import path has already supplied a DIFFERENT statistical  #
# estimator to a published interval in this program.                            #
# --------------------------------------------------------------------------- #
def _md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


IMPORT_PROVENANCE = {m.__name__: {"file": m.__file__, "md5": _md5(m.__file__)}
                     for m in (_ci, C, PS)}

#: probes: scored and reported, but they do NOT vote on the panel-wide gate.
PROBES = ("stand_still", "v1_ego_half", "v1_ego_double", "oracle_lon_straight")

#: PUBLISHED `@clamp_v1` composites — the reproduction gate. Same 16 values the
#: 2026-07-28 progress-term fix reproduced at max|diff| = 0.000000.
PUBLISHED_CLAMP_V1 = {
    "cv_holdv0": 0.5705, "v4_oracle": 0.5622, "refc_xl_produced": 0.5499,
    "v1_tactical_follow": 0.5471, "v1_tactical_oracle": 0.5467,
    "refc_small_produced": 0.5444, "refc_base_produced": 0.5439,
    "nospeed_tactical_oracle": 0.5394, "v4_blind": 0.3749,
    "v1_ego_v0": 0.5608, "v1_ego_oracle_lon": 0.5946, "v1_lat_straight": 0.5460,
    "refc_base_v0on": 0.5439, "refc_base_v0off": 0.4980,
    "refc_xl_v0on": 0.5499, "v1_ego_half": 0.3117,
}

#: ⛔ THE 8 INJECTIONS THAT EXPOSED THE DEFECT — the identical set, so the
#: acceptance test is the same experiment, not a friendlier one.
INJECTIONS = (("lat_shift", 2.0), ("lat_shift", -2.0),
              ("lat_jitter", 1.0), ("yaw_bias", 5.0))
INJECTION_ARMS = ("cv_holdv0", "v1_tactical_follow")

#: the shapes under test. `clamp_v1` is the incumbent and is expected to FAIL —
#: it is in the list so the acceptance test is shown to be able to fail.
SHAPES = ("clamp_v1", "lin_q0p25", "lin_q0p5", "lin_q0p6667", "lin_q0p75",
          "share_q0p25", "share_q0p5", "share_q0p6667", "share_q0p75")

#: ⭐ THE PRE-REGISTERED SELECTION RULE. Banked into `injections.json` BEFORE any
#: panel number is computed, so the choice cannot be fitted to the ranking it
#: produces. Mirrored verbatim in `pseudosim.RECOVERY_TERMS`' docstring.
SELECTION_RULE = {
    "R1": ("DISQUALIFY any shape for which the 8 injected lateral degradations "
           "are not ALL separated in the CORRECT (negative) direction on BOTH "
           "real arms. A shape failing this does not fix the defect."),
    "R2": ("DISQUALIFY any shape whose recovery FLOOR fraction is >= 0.50 on "
           "any scorable arm. C45: 'a term saturating on the majority of rows "
           "is not a metric; it is a constant with noise.'"),
    "R3": ("Among survivors PREFER the LINEAR family — it is affine-equivalent "
           "to the published term on r <= 1, so no pair of under-recovering "
           "rows changes order and the published value stays exactly "
           "invertible. Within it prefer the SMALLEST q."),
    "R4": ("If no LINEAR member survives, take the SHARE family at q = 0.5 — "
           "the equal-budget, parameter-free member "
           "(xt_hold / (xt_hold + xt_end))."),
    "_why_a_rule_at_all": (
        "The `w` decision published its sensitivity instead of choosing on the "
        "ranking it liked. A shape chosen after seeing the panel is a shape "
        "fitted to the panel."),
    "_impossibility": (
        "⛔ A STRICT REFINEMENT IS PROVABLY IMPOSSIBLE, which is why this fix "
        "cannot simply copy `twosided_v2`. Any g: [0, inf) -> [0, 1] agreeing "
        "with the published 1 - r on [0, 1] has g(1) = 0 and, being bounded "
        "below by 0 and non-increasing, must be CONSTANT on [1, inf) — i.e. it "
        "IS the defect. The published term spent its entire range on the half "
        "of the domain where the plan beats hold. ⇒ the fix is a RANGE-BUDGET "
        "choice, not a slope choice, and q is that budget."),
}


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
    arm and silently redefined the composite (max|diff| 0.000000 -> 0.393900).
    A synthetic reference must never vote."""
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
    """The composite under the shipped weights, per row. Identical arithmetic to
    ``pseudosim.composite`` on the admitted set; the panel-wide gate admits
    exactly {ego_progress, recovery} and zero-weights comfort."""
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


# =========================================================================== #
# PRIORITY 1 — THE 8 INJECTIONS                                                #
# =========================================================================== #
def _charge_rate(arms, shapes, probe=1e-4):
    """⭐⭐ THE MECHANISM, MEASURED — why an UNSATURATING shape still failed.

    ``|dg/dr|`` at the ratios the data actually occupies. A term does not need a
    hard floor to stop charging: it only needs its charge rate to COLLAPSE where
    the rows are. The share family never floors and still loses, because its
    slope decays like ``r^-2`` while the median row sits at ``r >= 1.004``.

    Reported as the slope at r = 0 / 0.5 / 1 / 2 / 3, plus the ROW-WEIGHTED mean
    slope over the panel's own ratio distribution, and the ratio of the slope at
    the median row to the slope at a near-perfect row — ⛔ **that last number is
    the discriminator: > 1 favours degradation.**"""
    rr = []
    for n, pw in arms.items():
        sc = PS.score_windows(pw)
        r = np.asarray(sc["recovery_raw_ratio"], float)
        fin = np.isfinite(np.asarray(sc["recovery"], float))
        if fin.any():
            rr.append(r[fin])
    allr = np.concatenate(rr) if rr else np.zeros(1)
    med = float(np.median(allr))
    out = {"_what": ("|dg/dr|, the rate at which a shape charges one more unit "
                     "of remaining error, at the ratios the panel occupies."),
           "panel_ratio_median": round(med, 4),
           "panel_ratio_p90": round(float(np.percentile(allr, 90)), 4),
           "panel_ratio_frac_gt_1": round(float((allr > 1).mean()), 4),
           "shapes": {}}
    t_all = torch.as_tensor(allr)
    for s in shapes:
        def slope(x):
            a = torch.tensor([max(x - probe, 0.0)], dtype=torch.float64)
            b = torch.tensor([x + probe], dtype=torch.float64)
            return float((PS.recovery_from_ratio(a, s)
                          - PS.recovery_from_ratio(b, s)).item()
                         / (b.item() - a.item()))
        g = PS.recovery_from_ratio(t_all, s).numpy()
        gp = PS.recovery_from_ratio(t_all + probe, s).numpy()
        w = float(np.mean((g - gp) / probe))
        s_med, s_near = slope(med), slope(0.05)
        out["shapes"][s] = {
            "slope_at_r0": round(slope(0.0), 4),
            "slope_at_r0p5": round(slope(0.5), 4),
            "slope_at_r1": round(slope(1.0), 4),
            "slope_at_r2": round(slope(2.0), 4),
            "slope_at_r3": round(slope(3.0), 4),
            "row_weighted_mean_slope": round(w, 4),
            # ⛔ THE DISCRIMINATOR. > 1 means a near-perfect row is charged
            # (equivalently, rewarded) HARDER than the typical row, so moving a
            # few rows toward 0 outweighs moving many rows further out.
            "reward_bias_near_perfect_over_median": round(
                s_near / max(s_med, 1e-12), 3),
        }
    return out



def job_injections(arms, out, n_boot, shapes=SHAPES):
    """⭐⭐ THE ACCEPTANCE TEST. All 8 must move the CORRECT way, on both arms.

    ``Delta > 0`` means the DEGRADED plan scores HIGHER — the defect. The
    published term produces 8 of 8 positive-and-separated; a shape that fixes
    the defect must produce 8 of 8 NEGATIVE-and-separated.

    ⚠️ The zero-mean control (`lat_jitter`) is the load-bearing row: a per-row
    Gaussian offset with expectation 0 cannot re-centre a one-sided bias, so its
    movement is the metric and not the re-centring artefact this program already
    knows about."""
    res = {"_selection_rule_PRE_REGISTERED": SELECTION_RULE,
           "_read": ("delta > 0 means the DEGRADED plan scores HIGHER (the "
                     "defect). PASS requires delta < 0 AND separated, on all 8."),
           "_injections": [f"{c}({l:+g})" for c, l in INJECTIONS],
           "_arms": list(INJECTION_ARMS), "n_boot": n_boot,
           "charge_rate": _charge_rate(arms, shapes), "shapes": {}}
    for shape in shapes:
        cells, n_ok, n_tot = {}, 0, 0
        for arm in INJECTION_ARMS:
            if arm not in arms:
                continue
            pw, eid = arms[arm], arms[arm]["eid"]
            base = PS.score_windows(pw, recovery_term=shape)
            cb = _composite_values(base)
            for ctl, lv in INJECTIONS:
                p = C.apply_control(pw, ctl, lv)
                s = PS.score_windows(p, recovery_term=shape)
                blk = {}
                for comp in ("recovery", "ego_progress"):
                    a = np.asarray(s[comp], float)
                    b0 = np.asarray(base[comp], float)
                    m = np.isfinite(a) & np.isfinite(b0)
                    blk[comp] = _ci.paired_episode_cluster_bootstrap(
                        a[m], b0[m], list(np.asarray(eid)[m]), n_boot=n_boot)
                ca = _composite_values(s)
                m = np.isfinite(ca) & np.isfinite(cb)
                pss = _ci.paired_episode_cluster_bootstrap(
                    ca[m], cb[m], list(np.asarray(eid)[m]), n_boot=n_boot)
                blk["PSS_composite"] = pss
                ok = bool(pss["separated"] and pss["delta"] < 0)
                blk["CORRECT_DIRECTION"] = ok
                blk["zero_mean_control"] = ctl in C.ZERO_MEAN_CONTROLS
                n_ok += int(ok)
                n_tot += 1
                cells[f"{arm}|{ctl}({lv:+g})"] = blk
        res["shapes"][shape] = {
            "n_correct": n_ok, "n_total": n_tot,
            "ALL_8_CORRECT": bool(n_tot == 8 and n_ok == 8),
            "cells": cells}
        print(f"  [inject] {shape:14s} {n_ok}/{n_tot} correct direction",
              flush=True)
    out["injections"] = res
    return out


# =========================================================================== #
# PRIORITY 2 — THE REPRODUCTION GATE                                           #
# =========================================================================== #
def job_repro(arms, out, n_boot):
    """✅ Every published ``@clamp_v1`` number must still reproduce EXACTLY.

    ⛔ This is not paperwork. The sibling stream's own gate caught a synthetic
    reference arm silently redefining the composite for every arm
    (max|diff| = 0.393900); nothing else in that pipeline noticed."""
    scores = {}
    for n, pw in arms.items():
        sc = PS.score_windows(pw, progress_term="clamp_v1",
                              recovery_term="clamp_v1")
        scores[n] = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
        scores[n]["no_collision"] = None
        scores[n]["ttc"] = None
    gate_arms = [n for n in arms if n not in PROBES]
    per_arm = {n: PS.discriminative_range(scores[n], by_arm=scores)
               for n in arms}
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
    out["repro_gate"] = {
        "rule": ("Every PUBLISHED @clamp_v1 composite must still reproduce to "
                 "4 dp with the recovery term VERSIONED and its default "
                 "flipped. If it does not, the old value is no longer "
                 "computable and the change is a silent redefinition."),
        "progress_term": "clamp_v1", "recovery_term": "clamp_v1",
        "metric_id_must_be_unchanged": PS.metric_id("clamp_v1", "clamp_v1"),
        "panel_gate_admissible": admissible,
        "max_abs_diff": round(worst, 6),
        "PASS": bool(worst <= 5e-4),
        "n_checked": sum(1 for v in rep.values() if v.get("status") == "OK"),
        "per_arm": rep,
        # ⭐ the SECOND half of the gate: the published term must still be
        # BIT-identical to the published expression, not merely close.
        "clamp_v1_is_bit_identical_to_lin_q0": bool(
            (PS.recovery_from_ratio(torch.linspace(0, 40, 4001), "clamp_v1")
             == PS.recovery_from_ratio(torch.linspace(0, 40, 4001), "lin_q0")
             ).all()),
    }
    print(f"  [repro] max|diff| = {worst:.6f} PASS={worst <= 5e-4}", flush=True)
    return out


# =========================================================================== #
# PRIORITY 3 — THE PANEL, BOTH TERMS, AND THE RANKING STATEMENT                #
# =========================================================================== #
#: arms that are NOT realisable planners — oracles and plan-transform probes.
#: The ranking statement is about the REALISABLE arms; quoting an oracle as
#: rank 1 without saying so is the over-claim the panel already guards against.
ORACLES = ("v1_ego_oracle_lon", "v4_oracle", "v1_tactical_oracle",
           "nospeed_tactical_oracle", "oracle_lon_straight")

CONTRASTS = (
    ("v1_ego_v0", "cv_holdv0"),
    ("v1_tactical_follow", "cv_holdv0"),
    ("refc_xl_produced", "cv_holdv0"),
    ("refc_base_produced", "cv_holdv0"),
    ("v4_oracle", "cv_holdv0"),
    ("v1_ego_oracle_lon", "cv_holdv0"),
    ("v1_ego_v0", "v1_tactical_follow"),
    ("v1_lat_straight", "v1_tactical_follow"),
    ("refc_xl_v0on", "refc_xl_v0off"),
    ("refc_base_v0on", "refc_base_v0off"),
    ("v4_oracle", "v4_blind"),
    ("v1_ego_half", "v1_tactical_follow"),
    ("v1_tactical_follow", "refc_xl_produced"),
    ("v1_tactical_follow", "refc_base_v0off"),
)


def _panel_under(arms, n_boot, *, progress_term, recovery_term):
    """Score every arm under one (progress, recovery) term pair, PANEL-WIDE gate."""
    scores, vals = {}, {}
    for n, pw in arms.items():
        sc = PS.score_windows(pw, progress_term=progress_term,
                              recovery_term=recovery_term)
        scores[n] = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
        scores[n]["no_collision"] = None
        scores[n]["ttc"] = None
    gate_arms = [n for n in arms if n not in PROBES]
    per_arm = {n: PS.discriminative_range(scores[n], by_arm=scores)
               for n in arms}
    admissible = {c: not [n for n in gate_arms
                          if not per_arm[n].get(c, {}).get("admissible")]
                  for c in PS.COMPONENT_WEIGHTS}
    levels, refused = {}, {}
    for n in arms:
        if not admissible.get("ego_progress") and not admissible.get("recovery"):
            refused[n] = "no weighted component admitted PANEL-WIDE"
            continue
        w = {k: v for k, v in PS.COMPONENT_WEIGHTS.items()
             if admissible.get(k) and float(v) > 0}
        vals[n] = _composite_values(scores[n], weights=w)
        ci = PS._boot(vals[n], arms[n]["eid"], n_boot, 0)
        levels[n] = {
            "mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"],
            "ego_progress": round(float(np.nanmean(scores[n]["ego_progress"])), 6),
            "recovery": round(float(np.nanmean(scores[n]["recovery"])), 6),
            "recovery_saturation": PS.saturation(scores[n]["recovery"]),
            "ego_progress_saturation": PS.saturation(scores[n]["ego_progress"]),
        }
    return {"metric_id": PS.metric_id(progress_term, recovery_term),
            "progress_term": progress_term, "recovery_term": recovery_term,
            "panel_gate_admissible": admissible,
            "weights_used": {k: v for k, v in PS.COMPONENT_WEIGHTS.items()
                             if admissible.get(k) and float(v) > 0},
            "levels": levels, "refused": refused}, vals


def _ranked(levels, arms_all):
    real = [n for n in levels if n not in PROBES and n not in ORACLES]
    order = sorted(levels, key=lambda n: -levels[n]["mean"])
    ro = sorted(real, key=lambda n: -levels[n]["mean"])
    return {"all_non_probe": [{"rank": i + 1, "arm": n,
                               "value": levels[n]["mean"],
                               "oracle": n in ORACLES}
                              for i, n in enumerate(order) if n not in PROBES],
            "realisable_only": [{"rank": i + 1, "arm": n,
                                 "value": levels[n]["mean"]}
                                for i, n in enumerate(ro)]}


def job_panel(arms, out, n_boot, recovery_term):
    """⭐ The panel under BOTH terms, the ranking statement, the contrasts."""
    pub, vpub = _panel_under(arms, n_boot, progress_term="twosided_v2",
                             recovery_term="clamp_v1")
    new, vnew = _panel_under(arms, n_boot, progress_term="twosided_v2",
                             recovery_term=recovery_term)
    rk_pub = _ranked(pub["levels"], arms)
    rk_new = _ranked(new["levels"], arms)

    def _rank_of(rk, arm):
        return next((r["rank"] for r in rk["realisable_only"]
                     if r["arm"] == arm), None)

    contrasts = {}
    for a, b in CONTRASTS:
        if a not in vnew or b not in vnew:
            continue
        row = {}
        for tag, vv in (("published_rec_clamp_v1", vpub),
                        (f"new_rec_{recovery_term}", vnew)):
            x, y = vv[a], vv[b]
            m = np.isfinite(x) & np.isfinite(y)
            row[tag] = _ci.paired_episode_cluster_bootstrap(
                x[m], y[m], list(np.asarray(arms[a]["eid"])[m]), n_boot=n_boot)
        row["verdict_flipped"] = bool(
            row["published_rec_clamp_v1"]["separated"]
            != row[f"new_rec_{recovery_term}"]["separated"]
            or (np.sign(row["published_rec_clamp_v1"]["delta"])
                != np.sign(row[f"new_rec_{recovery_term}"]["delta"])
                and row[f"new_rec_{recovery_term}"]["separated"]))
        contrasts[f"{a} - {b}"] = row
        print(f"  [contrast] {a} - {b}: "
              f"{row['published_rec_clamp_v1']['delta']:+.4f} -> "
              f"{row[f'new_rec_{recovery_term}']['delta']:+.4f}"
              f"{' FLIP' if row['verdict_flipped'] else ''}", flush=True)

    real_pub = [r["arm"] for r in rk_pub["realisable_only"]]
    real_new = [r["arm"] for r in rk_new["realisable_only"]]
    # ⚠️ "the v1 family" in the source claim is the TACTICAL family — the arms
    # built on v1's tactical head. `v1_ego_v0` / `v1_ego_oracle_lon` are ego-
    # SCHEDULE transforms of v1 and ranked ABOVE every REF-C arm under the
    # published term too, so folding them in would refute a claim nobody made.
    # The set is written out rather than pattern-matched, for that reason.
    V1_TACTICAL_FAMILY = ("v1_tactical_follow", "v1_tactical_oracle",
                          "v1_lat_straight", "nospeed_tactical_oracle")
    allp = [r["arm"] for r in rk_pub["all_non_probe"]]
    alln = [r["arm"] for r in rk_new["all_non_probe"]]

    def _below_all_refc(order):
        fam = [order.index(n) for n in V1_TACTICAL_FAMILY if n in order]
        rc = [order.index(n) for n in order if n.startswith("refc_")]
        return bool(fam and rc and min(fam) > max(rc))

    out["panel_both_terms"] = {
        "published": pub, "new": new,
        "ranking_published": rk_pub, "ranking_new": rk_new,
        "contrasts": contrasts,
        "RANKING_STATEMENT": {
            "cv_holdv0_rank_realisable_published": _rank_of(rk_pub, "cv_holdv0"),
            "cv_holdv0_rank_realisable_new": _rank_of(rk_new, "cv_holdv0"),
            "cv_holdv0_STILL_RANKS_FIRST_AMONG_REALISABLE":
                bool(_rank_of(rk_new, "cv_holdv0") == 1),
            "_v1_family_definition": (
                "the v1 TACTICAL family = " + ", ".join(V1_TACTICAL_FAMILY)
                + ". `v1_ego_v0` / `v1_ego_oracle_lon` are ego-SCHEDULE "
                  "transforms and rank ABOVE every REF-C arm under the "
                  "published term as well, so they are not part of the claim."),
            "whole_v1_tactical_family_below_every_REFC_published":
                _below_all_refc(allp),
            "whole_v1_tactical_family_below_every_REFC_new":
                _below_all_refc(alln),
            "realisable_order_published": real_pub,
            "realisable_order_new": real_new,
            "all_non_probe_order_published": allp,
            "all_non_probe_order_new": alln,
            "n_realisable_arms_that_moved": int(sum(
                1 for i, n in enumerate(real_new)
                if n in real_pub and real_pub.index(n) != i)),
            "n_non_probe_arms_that_moved": int(sum(
                1 for i, n in enumerate(alln)
                if n in allp and allp.index(n) != i)),
            "level_shift_min": round(float(min(
                new["levels"][n]["mean"] - pub["levels"][n]["mean"]
                for n in new["levels"] if n in pub["levels"])), 6),
            "level_shift_max": round(float(max(
                new["levels"][n]["mean"] - pub["levels"][n]["mean"]
                for n in new["levels"] if n in pub["levels"])), 6),
            "_levels_are_not_comparable": (
                "⛔ These are levels on a NEW METRIC ID "
                f"({new['metric_id']}) and are NOT comparable to any published "
                "PSS value. Every arm's level rises by 0.14-0.24 because the "
                "term now awards 2/3 rather than 0 to a plan that recovers "
                "nothing. Only ORDERINGS and PAIRED DELTAS carry across."),
        },
    }
    return out


# =========================================================================== #
# PRIORITY 4 — SATURATION EVERYWHERE + THE REMAINING-TERM AUDIT                 #
# =========================================================================== #
def job_saturation(arms, out, recovery_term):
    """⚠️ C45's STANDING CONSEQUENCE, made permanent: the floor/ceiling fraction
    of EVERY bounded term on EVERY arm, published beside its level."""
    rows = {}
    for n, pw in arms.items():
        a = C.axes(pw, recovery_term=recovery_term)
        sc_pub = PS.score_windows(pw, progress_term="clamp_v1",
                                  recovery_term="clamp_v1")
        sc_new = PS.score_windows(pw, progress_term="twosided_v2",
                                  recovery_term=recovery_term)
        node = {}
        for tag, arr in (
                ("ego_progress@clamp_v1", sc_pub["ego_progress"]),
                ("ego_progress@twosided_v2", sc_new["ego_progress"]),
                ("recovery@clamp_v1", sc_pub["recovery"]),
                (f"recovery@{recovery_term}", sc_new["recovery"]),
                ("comfort", sc_new["comfort"]),
                ("lon_track", a["lon_track"]),
                ("lat_track", a["lat_track"]),
                ("lat_heading", a["lat_heading"])):
            s = PS.saturation(arr)
            s["mean"] = (round(float(np.nanmean(arr)), 6)
                         if np.isfinite(np.asarray(arr, float)).any() else None)
            node[tag] = s
        rows[n] = node
    out["saturation_census"] = {
        "_rule": ("⚠️ C45 STANDING CONSEQUENCE — report the floor/ceiling "
                  "fraction BESIDE every bounded term, permanently. "
                  "`discriminative_range` COMPUTED floor_frac and never used "
                  "it; that is how a term floored on 55-92 % of its rows was "
                  "published 20 times."),
        "_warn_frac": PS.SATURATION_WARN_FRAC,
        "_gate_thresholds": {"floor_frac_max": PS.FLOOR_FRAC_MAX,
                             "ceil_frac_max": PS.CEIL_FRAC_MAX,
                             "range_min": PS.RANGE_MIN},
        "per_arm": rows}
    return out


def job_term_audit(arms, out, n_boot, recovery_term):
    """⛔ AUDIT THE REMAINING TERMS. Two of two audited terms were one-sidedly
    clamped; the composite is not sound until the rest are checked — and
    ``comfort`` counts, because it is still MEASURED and a saturating diagnostic
    misleads exactly as a saturating score does."""
    ref = "cv_holdv0" if "cv_holdv0" in arms else sorted(arms)[0]
    pw = arms[ref]
    audit = {}

    # --- 1. the ONE-SIDEDNESS SCAN, on the actual expressions ---------------- #
    # A term is at risk iff (a) it is a clamp/max/min of an unbounded quantity
    # AND (b) a non-trivial fraction of rows lands on the clamped side.
    a = C.axes(pw, recovery_term=recovery_term)
    scn = PS.score_windows(pw, progress_term="twosided_v2",
                           recovery_term=recovery_term)
    scp = PS.score_windows(pw, progress_term="clamp_v1",
                           recovery_term="clamp_v1")
    terms = {
        "ego_progress@clamp_v1": {
            "expr": "clamp(r, 0, 1), r = plan_along / human_along",
            "clamped_sides": "BOTH (floor at r<=0, ceiling at r>=1)",
            "arr": scp["ego_progress"],
            "status": "⛔ FIXED 2026-07-28 by twosided_v2 (ceiling side)"},
        "ego_progress@twosided_v2": {
            "expr": "clamp(clamp_v1(r) - w*max(r-1,0), 0, 1), w = 1",
            "clamped_sides": "FLOOR at r>=2 — ⚠️ STILL ONE-SIDED ABOVE r = 2",
            "arr": scn["ego_progress"],
            "status": "audited here"},
        "recovery@clamp_v1": {
            "expr": "clamp(1 - r, 0, 1), r = |xt_end| / |xt_hold|",
            "clamped_sides": "FLOOR at r>=1 (the ceiling is never active: r>=0)",
            "arr": scp["recovery"],
            "status": "⛔ THE DEFECT — C45"},
        f"recovery@{recovery_term}": {
            "expr": PS.RECOVERY_TERMS and "see RECOVERY_TERMS",
            "clamped_sides": ("NONE — the share family is strictly decreasing "
                              "on [0, inf) and attains neither bound"
                              if recovery_term.startswith("share")
                              else "FLOOR at r >= 1/(1-q)"),
            "arr": scn["recovery"], "status": "the fix"},
        "comfort": {
            "expr": "AND of four bounds — a BINARY indicator, not a score",
            "clamped_sides": ("⛔ NOT A CLAMP AT ALL: it is {0,1}-valued, so it "
                              "is 100 % saturated BY CONSTRUCTION on every row. "
                              "Its 'range' is an artefact of averaging."),
            "arr": scn["comfort"],
            "status": "weight 0.0 (C46) — still MEASURED, so still audited"},
        "lon_track": {
            "expr": "mean_k clamp(1 - |t_err_k| / T_TOL, 0, 1)",
            "clamped_sides": "FLOOR per-step at |t_err| >= T_TOL; the MEAN over "
                             "20 steps is what is scored, so row-level "
                             "saturation needs all 20 steps clamped",
            "arr": a["lon_track"], "status": "audited here"},
        "lat_track": {
            "expr": "mean_k clamp(1 - |XTE_k| / corridor(s_k), 0, 1)",
            "clamped_sides": "FLOOR per-step; mean over 20 steps",
            "arr": a["lat_track"], "status": "audited here"},
        "lat_heading": {
            "expr": "clamp(1 - |dpsi| / PSI_TOL, 0, 1) — a SINGLE value, not a "
                    "mean, so it saturates at the ROW level",
            "clamped_sides": "FLOOR at |dpsi| >= PSI_TOL",
            "arr": a["lat_heading"], "status": "⚠️ audited here — see verdict"},
    }
    for name, node in terms.items():
        arr = node.pop("arr")
        s = PS.saturation(arr)
        panel = {n: PS.saturation(
            (C.axes(arms[n], recovery_term=recovery_term)[name]
             if name in ("lon_track", "lat_track", "lat_heading")
             else PS.score_windows(
                 arms[n],
                 progress_term=("clamp_v1" if name.endswith("@clamp_v1")
                                else "twosided_v2"),
                 recovery_term=("clamp_v1" if name == "recovery@clamp_v1"
                                else recovery_term))[
                     name.split("@")[0]]))["floor_frac_le_0p001"]
            for n in arms if n not in PROBES}
        node["saturation_on_" + ref] = s
        node["floor_frac_max_over_panel"] = round(
            float(max(v for v in panel.values() if v is not None)), 6)
        node["floor_frac_per_arm"] = panel
        node["ONE_SIDED_RISK"] = bool(
            node["floor_frac_max_over_panel"] >= PS.SATURATION_WARN_FRAC)
        audit[name] = node
    out["term_audit"] = {
        "_rule": ("Two of two audited terms were one-sidedly clamped. A term is "
                  "AT RISK iff it clamps an unbounded quantity AND a "
                  "non-trivial fraction of rows lands on the clamped side. "
                  "`comfort` is audited despite carrying weight 0.0 — it is "
                  "still MEASURED, and a saturating diagnostic misleads exactly "
                  "as a saturating score does."),
        "_reference_arm": ref,
        "_detection_heuristic": (
            "grep any metric for a one-sided clamp/max/min and ENUMERATE what "
            "lives in the zero-gradient half before adopting it (C45's own)."),
        "terms": audit}
    return out


# =========================================================================== #
# PRIORITY 5 — THE SHAPE SENSITIVITY                                           #
# =========================================================================== #
def job_shape_sensitivity(arms, out, n_boot, shapes=SHAPES):
    """⚠️ THE `w` TREATMENT APPLIED TO `q`: publish the ranking at every point of
    the grid rather than choosing one and moving on."""
    res = {"_read": ("q = the score a plan that recovers NOTHING receives. "
                     "q = 0 (clamp_v1) is the published, defective term. Two "
                     "FAMILIES at the same anchors: `lin` is affine-equivalent "
                     "to the published term on r <= 1 but still floors at "
                     "r = 1/(1-q); `share` never saturates but re-spaces the "
                     "under-side."),
           "shapes": {}}
    for shape in shapes:
        p, v = _panel_under(arms, n_boot, progress_term="twosided_v2",
                            recovery_term=shape)
        rk = _ranked(p["levels"], arms)
        res["shapes"][shape] = {
            "metric_id": p["metric_id"],
            "ranking_realisable": rk["realisable_only"],
            "ranking_all_non_probe": rk["all_non_probe"],
            "recovery_floor_frac_max": round(float(max(
                p["levels"][n]["recovery_saturation"]["floor_frac_le_0p001"]
                for n in p["levels"] if n not in PROBES)), 6),
            "levels": {n: p["levels"][n]["mean"] for n in p["levels"]},
        }
        print(f"  [shape] {shape:14s} rank1(realisable)="
              f"{rk['realisable_only'][0]['arm']} "
              f"floor_max={res['shapes'][shape]['recovery_floor_frac_max']:.4f}",
              flush=True)
    real = {s: [r["arm"] for r in res["shapes"][s]["ranking_realisable"]]
            for s in res["shapes"]}
    res["ROBUSTNESS"] = {
        "rank1_realisable_per_shape": {s: real[s][0] for s in real},
        "rank1_is_the_same_under_every_shape":
            len({real[s][0] for s in real}) == 1,
        "full_order_identical_under_every_shape":
            len({tuple(real[s]) for s in real}) == 1,
    }
    out["shape_sensitivity"] = res
    return out


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", action="append", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=_ci.DEFAULT_N_BOOT)
    ap.add_argument("--ref-arm", default="cv_holdv0")
    ap.add_argument("--recovery-term", default=PS.RECOVERY_TERM_DEFAULT)
    ap.add_argument("--jobs", default="inject,repro,panel,sat,audit,shape")
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
    print(f"[recov] {len(arms)} arms: {sorted(arms)}", flush=True)

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
    print(f"[recov] human_replay verified at max|residual| = {_worst:.3e} m "
          f"(kept OUT of the gate vote)", flush=True)

    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    head = {
        "_experiment": ("FIX THE SECOND ONE-SIDED CLAMP: the `recovery` term "
                        "(C45). 0 GPU-h."),
        "_evidence_class": "MEASURED (ours; artifact = these JSONs + pw_*.npz)",
        "_protocol": PS.PROTOCOL,
        "_estimator": (f"taniteval.ci.episode_cluster_bootstrap / paired form "
                       f"(B={a.n_boot}, unit = val episode)"),
        "_refused_estimator": ("overlapping_holdout_se — it biases the POINT "
                               "ESTIMATE as well as the interval"),
        "_gate": "PANEL-WIDE. The per-arm gate is REFUSED, not offered.",
        "_import_provenance_md5": IMPORT_PROVENANCE,
        "traffic_mode": PS.TRAFFIC_MODE_LOG_REPLAY,
        "traffic_mode_note": PS.TRAFFIC_MODE_NOTE,
        "missing_gates": C.MISSING_GATES,
        "recovery_term_under_test": a.recovery_term,
        "recovery_term_published": PS.RECOVERY_TERM_PUBLISHED,
        "recovery_terms_available": sorted(PS.RECOVERY_TERMS),
        "arm_sources": seen, "row_identity": row_id,
        "arms_refused_on_row_identity": refused,
        "reference_arm": ref, "probes_excluded": list(PROBES),
        "oracles_excluded_from_realisable_ranking": list(ORACLES),
        "human_replay_max_residual_m": _worst,
        "human_replay_votes_on_the_gate": False,
        "selected_frac_note": (
            "selected_frac = 1.000 for every arm BY CONSTRUCTION: there is no "
            "selection step on this surface."),
    }

    def bank(name, node):
        p = od / f"{name}.json"
        p.write_text(json.dumps(dict(head, **{name: node}), indent=2,
                                default=str), encoding="utf-8")
        print(f"[recov] wrote {p}", flush=True)

    jobs = set(a.jobs.split(","))
    acc = {}
    if "inject" in jobs:                       # PRIORITY 1
        job_injections(arms, acc, a.n_boot)
        bank("injections", acc["injections"])
    if "repro" in jobs:                        # PRIORITY 2
        job_repro(arms, acc, a.n_boot)
        bank("repro_gate", acc["repro_gate"])
    if "panel" in jobs:                        # PRIORITY 3
        job_panel(arms, acc, a.n_boot, a.recovery_term)
        bank("panel_both_terms", acc["panel_both_terms"])
    if "sat" in jobs:                          # PRIORITY 4
        job_saturation(arms, acc, a.recovery_term)
        bank("saturation_census", acc["saturation_census"])
    if "audit" in jobs:
        job_term_audit(arms, acc, a.n_boot, a.recovery_term)
        bank("term_audit", acc["term_audit"])
    if "shape" in jobs:                        # PRIORITY 5
        job_shape_sensitivity(arms, acc, a.n_boot)
        bank("shape_sensitivity", acc["shape_sensitivity"])

    print(f"[recov] done in {time.time() - t0:.1f} s")
    print("RECOVERY_TWOSIDED_DONE", flush=True)


if __name__ == "__main__":
    main()
