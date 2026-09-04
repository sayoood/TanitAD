#!/usr/bin/env python3
"""refav1_openloop_report.py — the four-family OPEN-LOOP report for a banked refav1 record.

⛔ VOCABULARY (PI ruling 2026-09-02, ``Project Steering/VOCABULARY.md``). Every arm a
``refav1_arm.py`` dump contains is **OPEN LOOP**: the model is not controlling a vehicle and
its trajectory never affects the ego data it is next fed. The record's internal ``T0``/``T1``
stamps are reproduced verbatim — they are the doctrine's *conditioning* labels — and the
words "closed loop" never appear in this tool's output.

WHAT IT COMPUTES ITSELF — exactly one thing, and it is the headline
------------------------------------------------------------------
``identity_probe()`` reads the raw ``ep*.npz`` arrays and ``decisions/<arm>_controls``:

* is the planner arm ``cl`` **bit-identical** to the constant-velocity floor ``ha0``
  (``max |cl - ha0| < 1e-9 m``, and separately ``np.array_equal``) — per window, not on
  average;
* are the planner's OWN emitted ``cl_controls`` **exactly zero** — the MECHANISM behind such
  an identity, and a fact no family table can ever show, because a family table answers
  *how far off* and never *what shape*.

⭐ THE CONTROL, AND WHY IT IS THE POINT. The record already carries a ``trivial_profile``
with its own ``identical_to`` count. This tool recomputes that number through a **different
code path** and **REFUSES** to render when the two disagree. Two reads of one array through
one function are one probe; the operating standard's "a second probe must differ in
path-binding" applies to a *presence* claim as hard as to an absence, because the headline
here — *the planner emits the trivial baseline* — is an indictment.

⛔ It NEVER recomputes a family metric. Every level, interval and paired delta is READ from
the record, which computed them with ``taniteval.ci``'s episode-cluster bootstrap;
``overlapping_holdout_se`` appears nowhere in that chain.

Usage::

    python taniteval/tools/refav1_openloop_report.py \
        --record refav1_t1.json --dump t1_dump \
        --title "refav1 @ 21,109 — OPEN-LOOP four-family read" \
        --out RESULT.md --ident-out identity_probe.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

IDENT_M = 1e-9
ARMS_DEFAULT = ["cl", "ha", "ha0", "ol"]
PAIRS_DEFAULT = ["paired_cl_minus_ha0", "paired_cl_minus_ha",
                 "paired_closed_minus_open"]
OPEN_LOOP_NOTE = (
    "**OPEN LOOP** (PI ruling 2026-09-02). The model is not controlling a vehicle; a "
    "predictor consuming its own planner's actions is still open loop. The `T0`/`T1` "
    "stamps below are the doctrine's CONDITIONING labels, not loop labels.")


# --------------------------------------------------------------------------- #
# the one computation this tool does itself                                    #
# --------------------------------------------------------------------------- #
def identity_probe(dump: str, arm: str = "cl", floor: str = "ha0") -> dict:
    """Per-window bit-identity of ``arm`` against ``floor``, plus the SHAPE of the
    controls the arm actually emitted.

    ⭐ WHY THE CONTROL SHAPE IS REPORTED SEPARATELY FROM THE IDENTITY, and it is not a
    refinement — it is the difference between a true and a false headline. MEASURED
    2026-09-04 on the 141-clip split: `cl` is bit-identical to `ha0` on **94.8 %** of
    windows, not 100 % as two earlier 20-clip reads found. Every one of the exceptions is
    ALSO an injected trivial baseline — a constant −1.5 m/s² brake with κ ≡ 0 — so
    "identical to `ha0`" UNDERSTATES the defect while "not identical on 5 %" would have
    read as evidence of planning. The honest quantities are: how often κ is identically
    zero (the planner never turns), how often the acceleration is constant in time (it
    never modulates), and **how many DISTINCT plans it emitted in total**.
    """
    n = n_bit = n_exact = 0
    resid, ctrl_absmax = [], []
    ctrl_n = ctrl_zero = ctrl_kzero = ctrl_aconst = ctrl_straight_const = 0
    accels: dict[str, int] = {}
    per_ep: dict[str, dict] = {}
    for f in sorted(glob.glob(os.path.join(dump, "ep*.npz"))):
        d = np.load(f)
        if arm not in d.files or floor not in d.files:
            continue
        a, b = d[arm], d[floor]
        m = np.abs(a - b).reshape(len(a), -1).max(axis=1)
        n += len(a)
        n_bit += int((m < IDENT_M).sum())
        n_exact += sum(int(np.array_equal(a[i], b[i])) for i in range(len(a)))
        resid.append(m)
        per_ep[os.path.basename(f)] = {
            "n": int(len(a)),
            "n_bit_identical": int((m < IDENT_M).sum()),
            "max_resid_m": float(m.max()),
        }
        g = os.path.join(dump, "decisions", os.path.basename(f))
        if os.path.exists(g):
            dd = np.load(g)
            key = f"{arm}_controls"
            if key in dd.files:
                c = dd[key]
                am = np.abs(c).reshape(len(c), -1).max(axis=1)
                ctrl_n += len(c)
                ctrl_zero += int((am == 0.0).sum())
                ctrl_absmax.append(am)
                kz = (c[..., 1] == 0.0).all(axis=1)                 # never turns
                ac = (c[..., 0] == c[:, :1, 0]).all(axis=1)         # a constant in time
                ctrl_kzero += int(kz.sum())
                ctrl_aconst += int(ac.sum())
                both = kz & ac
                ctrl_straight_const += int(both.sum())
                for v in c[both, 0, 0]:
                    accels[f"{float(v):.6g}"] = accels.get(f"{float(v):.6g}", 0) + 1
    resid = np.concatenate(resid) if resid else np.zeros(0)
    ctrl_absmax = np.concatenate(ctrl_absmax) if ctrl_absmax else np.zeros(0)
    return {
        "arm": arm, "floor": floor,
        "n_windows": n,
        "n_episodes": len(per_ep),
        "n_bit_identical": n_bit,
        "frac_bit_identical": (n_bit / n) if n else None,
        "n_exact_array_equal": n_exact,
        "max_residual_m": float(resid.max()) if n else None,
        "mean_residual_m": float(resid.mean()) if n else None,
        "n_windows_with_controls": ctrl_n,
        "n_windows_controls_exactly_zero": ctrl_zero,
        "frac_controls_exactly_zero": (ctrl_zero / ctrl_n) if ctrl_n else None,
        "n_windows_kappa_identically_zero": ctrl_kzero,
        "frac_kappa_identically_zero": (ctrl_kzero / ctrl_n) if ctrl_n else None,
        "n_windows_accel_constant_in_time": ctrl_aconst,
        "n_windows_straight_and_constant_accel": ctrl_straight_const,
        "frac_straight_and_constant_accel": (ctrl_straight_const / ctrl_n) if ctrl_n else None,
        "distinct_plans_emitted": accels,
        "n_distinct_plans_emitted": len(accels),
        "max_abs_control": float(ctrl_absmax.max()) if ctrl_n else None,
        "tolerance_m": IDENT_M,
        "per_episode": per_ep,
        "_probe": ("raw ep*.npz arrays + decisions/<arm>_controls, computed by "
                   "refav1_openloop_report.identity_probe — a SECOND code path over the "
                   "record's own trivial_profile.identical_to"),
        "_distinct_plans_is": ("the constant acceleration (m/s^2) of each window whose "
                               "controls are straight (kappa identically 0) AND constant "
                               "in time, with its window count. A planner that emits a "
                               "handful of distinct scalars over hundreds of windows has "
                               "a collapsed action space, whatever its ADE says."),
    }


def cross_check(rec: dict, ident: dict) -> dict:
    """The record's own ``trivial_profile`` against the independent probe.

    ``status: REFUSED`` means the two disagree: one of them is wrong and NEITHER may be
    quoted. This is the whole reason the second probe exists.
    """
    tp = (rec.get("refav1") or {}).get("trivial_profile") or {}
    row = (tp.get("arms") or {}).get(ident["arm"]) or {}
    rec_n = int(((row.get("identical_to") or {}).get(ident["floor"]) or {}).get("n", 0))
    out = {
        "record_trivial_profile_identical_n": rec_n,
        "record_trivial_frac": row.get("trivial_frac"),
        "record_n_windows": tp.get("n_windows"),
        "independent_probe_identical_n": ident["n_bit_identical"],
        "independent_n_windows": ident["n_windows"],
        "_rule": ("both the window count and the identical count must agree, or the "
                  "report REFUSES to render"),
    }
    ok = (rec_n == ident["n_bit_identical"]
          and int(tp.get("n_windows") or -1) == ident["n_windows"])
    out["status"] = "OK" if ok else "REFUSED"
    if not ok:
        out["reason"] = ("the record's trivial_profile and the independent raw-array probe "
                         "disagree; one is wrong and neither may be quoted")
    return out


# --------------------------------------------------------------------------- #
# formatting — reads only                                                      #
# --------------------------------------------------------------------------- #
LON_ROWS = [("speed_mae_mps", "target-speed MAE (m/s)"),
            ("speed_bias_mps", "speed bias (m/s)"),
            ("speed_rmse_mps", "speed RMSE (m/s)"),
            ("along_mae_m", "along-track MAE (m)"),
            ("along_final_bias_m", "along-track final bias (m)"),
            ("accel_mae_mps2", "accel MAE (m/s^2)")]
LAT_ROWS = [("heading_mae_deg", "heading MAE (deg)"),
            ("yaw_rate_mae_degps", "yaw-rate MAE (deg/s)"),
            ("curvature_mae_1pm", "curvature MAE (1/m)"),
            ("cross_mae_m", "cross-track MAE (m)"),
            ("cross_final_mae_m", "cross-track final MAE (m)")]
TAC_ROWS = [("lateral_decision", "LAT decision acc / kappa"),
            ("longitudinal_decision", "LON decision acc / kappa"),
            ("maneuver_5way_collapsed", "5-way collapsed acc / kappa")]


def _f(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _ci(m):
    if not isinstance(m, dict):
        return "n/a"
    return f"{_f(m.get('mean', m.get('delta')))} [{_f(m.get('lo'))}, {_f(m.get('hi'))}]"


def _dk_cell(dk: dict, arm: str) -> str:
    if dk.get("status") not in ("PRESENT", "OK"):
        return str(dk.get("status", "UNAVAILABLE"))
    row = ((dk.get("per_arm") or {}).get(arm)) or {}
    if row.get("status") != "OK":
        return str(row.get("status", "n/a"))
    return (f"{_f(row.get('mean_headway_min_m'))} / "
            f"{_f(row.get('mean_time_gap_min_s'))} / "
            f"{_f(row.get('mean_min_ttc_s'))}")


def family_tables(rec: dict, arms=None) -> str:
    A, R = rec["arms"], rec["refav1"]
    arms = [a for a in (arms or ARMS_DEFAULT) if a in A]
    head = "| | " + " | ".join(f"`{a}`" for a in arms) + " |"
    rule = "|---|" + "---|" * len(arms)
    L: list[str] = []

    L.append("**LONGITUDINAL**\n")
    L += [head, rule]
    for k, lab in LON_ROWS:
        L.append(f"| {lab} | " + " | ".join(
            _f(A[a]["four_families"]["longitudinal"].get(k)) for a in arms) + " |")
    for k, lab in [("LON_speed_mae_mps", "speed MAE, 95 % CI"),
                   ("LON_along_mae_m", "along MAE, 95 % CI")]:
        L.append(f"| {lab} | " + " | ".join(
            _ci(A[a]["intervals"]["metrics"].get(k)) for a in arms) + " |")
    dk = R.get("distance_keeping") or {}
    L.append("| distance-keeping: headway (m) / time-gap (s) / min-TTC (s) | "
             + " | ".join(_dk_cell(dk, a) for a in arms) + " |")
    L.append("")

    L.append("**LATERAL**\n")
    L += [head, rule]
    for k, lab in LAT_ROWS:
        L.append(f"| {lab} | " + " | ".join(
            _f(A[a]["four_families"]["lateral"].get(k)) for a in arms) + " |")
    for k, lab in [("LAT_cross_mae_m", "cross MAE, 95 % CI"),
                   ("LAT_heading_mae_deg", "heading MAE, 95 % CI")]:
        L.append(f"| {lab} | " + " | ".join(
            _ci(A[a]["intervals"]["metrics"].get(k)) for a in arms) + " |")
    L.append("")

    L.append("**TACTICAL — executed (trajectory-derived, refc 3-way labeller)**\n")
    L += [head, rule]
    for key, lab in TAC_ROWS:
        cells = []
        for a in arms:
            b = (A[a]["four_families"]["tactical"] or {}).get(key) or {}
            cells.append("n/a" if b.get("status") != "OK"
                         else f"{_f(b.get('accuracy'))} / {_f(b.get('kappa'))}")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("| tactical goal-point error (m) | " + " | ".join(
        _f(((A[a]["four_families"]["tactical"] or {}).get("goal_setting") or {})
           .get("goal_point_error_m")) for a in arms) + " |")
    L.append("")
    return "\n".join(L)


def _majority_floor(v: dict):
    """The no-information value for a categorical row: the majority-class rate.

    Preferred straight from the record (``majority_class_rate``). When a block does not
    carry it, it is DERIVED from that same block's own ``per_class[*].n_true`` counts —
    an arithmetic restatement of the record's confusion matrix, never a re-estimation.
    """
    if v.get("majority_class_rate") is not None:
        return v["majority_class_rate"], "record"
    pc = v.get("per_class") or {}
    n = v.get("n") or 0
    tot = sum(int((c or {}).get("n_true") or 0) for c in pc.values())
    if not pc or not n or tot != n:
        return None, "n/a"
    return max(int((c or {}).get("n_true") or 0) for c in pc.values()) / n, "derived"


def _cond_table(block: dict, keys, title: str) -> str:
    L = [f"**{title}**\n",
         "| conditioning | n | accuracy | kappa | majority-class floor | acc 95 % CI |",
         "|---|---|---|---|---|---|"]
    conds = block.get("conditionings") or {}
    for k in keys:
        v = conds.get(k)
        if not isinstance(v, dict):
            L.append(f"| `{k}` | — | n/a | n/a | n/a | n/a |")
            continue
        if v.get("status") != "OK":
            L.append(f"| `{k}` | {v.get('n')} | {v.get('status')} | | | |")
            continue
        acc_ci = ((v.get("ci") or {}).get("accuracy"))
        floor, src = _majority_floor(v)
        floor_s = _f(floor) + ("" if src in ("record", "n/a") else " *(derived)*")
        L.append(f"| `{k}` | {v.get('n')} | {_f(v.get('accuracy'))} | "
                 f"{_f(v.get('kappa'))} | {floor_s} | {_ci(acc_ci)} |")
    L.append("")
    return "\n".join(L)


def strategic_section(rec: dict) -> str:
    s = (rec["refav1"].get("strategic") or {})
    L = [f"* windows {s.get('n_windows')} · route-labelled **{s.get('n_route_labeled')}** · "
         f"nav-valid frac {_f(s.get('nav_valid_frac'))} · "
         f"excluded (no route label) {s.get('n_excluded_no_route_label')}",
         "* ⛔ `nav_true` accuracy is a **nav ECHO index**, not skill — `route_label` and "
         "`nav_cmd` derive from the same field. Read `nav_shuffled` / `nav_zero` and the "
         "changed subset.", ""]
    L.append(_cond_table(s, ["nav_true", "nav_shuffled", "nav_zero"],
                         "STRATEGIC — route head vs `route_label`"))
    p = s.get("paired_true_minus_shuffled_accuracy")
    if isinstance(p, dict):
        L.append(f"paired `true − shuffled` accuracy: **{_f(p.get('delta'))}** "
                 f"[{_f(p.get('lo'))}, {_f(p.get('hi'))}], separated "
                 f"**{_f(p.get('separated'))}**, n_windows {p.get('n_windows')}, "
                 f"n_episodes {p.get('n_episodes')}\n")
    cs = s.get("changed_subset")
    if isinstance(cs, dict) and cs.get("n"):
        a = cs.get("route_follows_LABEL_under_shuffle") or {}
        b = cs.get("route_follows_SHUFFLED_NAV_under_shuffle") or {}
        L.append(f"changed subset (n = {cs.get('n')}): follows the LABEL "
                 f"**{_f(a.get('mean'))}**, follows the SHUFFLED NAV **{_f(b.get('mean'))}** "
                 f"— mutually exclusive by construction\n")
    return "\n".join(L)


def tactical_declared_section(rec: dict) -> str:
    t = (rec["refav1"].get("tactical_declared") or {})
    L = [f"* vocabulary `{t.get('vocabulary')}` · windows {t.get('n_windows')} · "
         "the DECLARED (selected) half of the TACTICAL family; the executed half is the "
         "trajectory-derived table above, in a DIFFERENT vocabulary — no mapping is "
         "invented", ""]
    L.append(_cond_table(t, ["lat_nav_true", "lat_nav_shuffled", "lat_nav_zero"],
                         "TACTICAL declared — LAT head"))
    L.append(_cond_table(t, ["lon_nav_true", "lon_nav_shuffled", "lon_nav_zero"],
                         "TACTICAL declared — LON head"))
    for k, lab in [("lat_paired_true_minus_shuffled_accuracy", "LAT"),
                   ("lon_paired_true_minus_shuffled_accuracy", "LON")]:
        p = t.get(k)
        if isinstance(p, dict):
            L.append(f"paired `true − shuffled` {lab} accuracy: **{_f(p.get('delta'))}** "
                     f"[{_f(p.get('lo'))}, {_f(p.get('hi'))}], separated "
                     f"**{_f(p.get('separated'))}**, n {p.get('n_windows')}\n")
    return "\n".join(L)


def paired_table(rec: dict, pair: str) -> str:
    fp = ((rec["refav1"].get("families_paired")) or {}).get(pair)
    if not fp:
        return f"_(pair `{pair}` absent from this record)_\n"
    L = [f"**{fp['direction']}** — estimator `{fp['estimator']}`, conditioning "
         f"`{fp['tier']}`; both arms OPEN LOOP\n",
         "| family | metric | delta | 95 % CI | separated |",
         "|---|---|---|---|---|"]
    for fam, mets in fp["families"].items():
        for m, v in mets.items():
            if not isinstance(v, dict):
                continue
            L.append(f"| {fam} | `{m}` | {_f(v.get('delta'))} | "
                     f"[{_f(v.get('lo'))}, {_f(v.get('hi'))}] | "
                     f"{'**yes**' if v.get('separated') else 'no'} |")
    L.append("")
    return "\n".join(L)


def planner_section(rec: dict) -> str:
    p = (rec["refav1"].get("planner") or {})
    L = ["| arm | n | plan sources | baseline_won_frac | goal source | "
         "selected LON goal | mean cost |", "|---|---|---|---|---|---|---|"]
    for arm, v in p.items():
        L.append(f"| `{arm}` | {v.get('n')} | {v.get('source_fractions')} | "
                 f"{_f(v.get('baseline_won_frac'))} | {v.get('goal_source_fractions')} | "
                 f"{v.get('goal_action_lon')} | {v.get('cost_mean')} |")
    L.append("")
    return "\n".join(L)


def wm_section(rec: dict) -> str:
    w = (rec["refav1"].get("wm_diagnostic_T0") or {})
    iv = w.get("intervals") or {}
    L = [f"* tier **{w.get('tier')}** — {w.get('tier_note')}",
         f"* n {w.get('n')} · K {w.get('k_wm')} · space `{w.get('space','')[:80]}` · "
         f"tgt_std_mean {_f(w.get('tgt_std_mean'))}", ""]
    if iv:
        L += ["| quantity | mean | 95 % CI |", "|---|---|---|"]
        for k, v in iv.items():
            if isinstance(v, dict):
                L.append(f"| `{k}` | {_f(v.get('mean'))} | "
                         f"[{_f(v.get('lo'))}, {_f(v.get('hi'))}] |")
        L.append("")
    for k in ("paired_const_minus_model", "paired_zero_minus_model_mean_over_steps"):
        v = w.get(k)
        if not isinstance(v, dict):
            continue
        if "delta" in v:                       # a single paired block
            L.append(f"`{k}`: **{_f(v.get('delta'))}** [{_f(v.get('lo'))}, "
                     f"{_f(v.get('hi'))}], separated {_f(v.get('separated'))}\n")
            continue
        # ⭐ a PER-HORIZON block. It is rendered row by row rather than averaged,
        # because the sign is not constant across the horizon: MEASURED 2026-09-04,
        # the model LOSES to persist-last-field at step 1 and wins from step 5 on,
        # and a single mean would hide the one horizon where the control wins.
        L.append(f"`{k}` — per horizon (positive = the model beats the control)\n")
        L += ["| horizon | delta | 95 % CI | separated |", "|---|---|---|---|"]
        for h, hv in v.items():
            if isinstance(hv, dict) and "delta" in hv:
                L.append(f"| `{h}` | {_f(hv.get('delta'))} | [{_f(hv.get('lo'))}, "
                         f"{_f(hv.get('hi'))}] | "
                         f"{'**yes**' if hv.get('separated') else 'no'} |")
        L.append("")
    return "\n".join(L)


def render(rec: dict, ident: dict, xc: dict, title: str, pairs=None, arms=None) -> str:
    R = rec["refav1"]
    tp = R.get("trivial_profile") or {}
    cl_tp = (tp.get("arms") or {}).get(ident["arm"]) or {}
    L = [f"# {title}", "", f"> {OPEN_LOOP_NOTE}", "",
         "## 1. Shape before metrics — is the planner arm the trivial floor?", "",
         f"* **n = {ident['n_windows']} windows over {ident['n_episodes']} episode "
         f"clusters** (the bootstrap's resampling unit is the episode)",
         f"* `{ident['arm']}` **bit-identical** to `{ident['floor']}` "
         f"(< {IDENT_M:g} m) on **{ident['n_bit_identical']}/{ident['n_windows']} = "
         f"{_f(ident['frac_bit_identical'])}**; bitwise-equal arrays on "
         f"{ident['n_exact_array_equal']}/{ident['n_windows']}; max residual "
         f"**{_f(ident['max_residual_m'], 9)} m**",
         f"* the planner's OWN emitted `{ident['arm']}_controls` are **exactly zero** on "
         f"**{ident['n_windows_controls_exactly_zero']}/"
         f"{ident['n_windows_with_controls']}** windows "
         f"(max |control| = {_f(ident['max_abs_control'], 9)})",
         f"* ⭐ **the control SHAPE, which is the sharper statement**: κ is *identically "
         f"zero* on **{ident['n_windows_kappa_identically_zero']}/"
         f"{ident['n_windows_with_controls']}** windows (the planner never turns), the "
         f"acceleration is *constant in time* on "
         f"**{ident['n_windows_accel_constant_in_time']}/"
         f"{ident['n_windows_with_controls']}**, and both together on "
         f"**{ident['n_windows_straight_and_constant_accel']}/"
         f"{ident['n_windows_with_controls']}** = "
         f"{_f(ident['frac_straight_and_constant_accel'])}",
         f"* over those windows it emitted **{ident['n_distinct_plans_emitted']} distinct "
         f"plans in total**, as constant accelerations (m/s²) with counts: "
         f"`{ident['distinct_plans_emitted']}`",
         f"* record's own `trivial_profile`: constant-velocity frac "
         f"**{_f(cl_tp.get('trivial_frac'))}**, degenerate arms "
         f"**{tp.get('degenerate_arms')}**",
         f"* the two probes cross-check **{xc['status']}** "
         f"({xc['record_trivial_profile_identical_n']} vs "
         f"{xc['independent_probe_identical_n']} identical windows)", "",
         "## 2. The four families, per arm, never pooled", "",
         family_tables(rec, arms), "",
         "### 2d. STRATEGIC", "", strategic_section(rec), "",
         "### 2e. TACTICAL — declared (selected) half", "",
         tactical_declared_section(rec), "",
         "### 2f. ADE / FDE — one row of four families, ⛔ never \"the result\"", ""]
    A = rec["arms"]
    a_list = [a for a in (arms or ARMS_DEFAULT) if a in A]
    L += ["| | " + " | ".join(f"`{a}`" for a in a_list) + " |",
          "|---|" + "---|" * len(a_list)]
    for k, lab in [("ade_dense_m", "ADE (m), 95 % CI"),
                   ("fde_last_m", "FDE (m), 95 % CI")]:
        L.append(f"| {lab} | " + " | ".join(
            _ci(A[a]["intervals"]["metrics"].get(k)) for a in a_list) + " |")
    L += ["", "## 3. Paired margins on the same windows", ""]
    for p in (pairs or PAIRS_DEFAULT):
        L.append(paired_table(rec, p))
    L += ["## 4. What the planner did", "", planner_section(rec), "",
          "## 5. World-model diagnostic (teacher-forced)", "", wm_section(rec)]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="four-family OPEN-LOOP report for a refav1 record")
    ap.add_argument("--record", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--title", default="refav1 OPEN-LOOP four-family read")
    ap.add_argument("--out", default=None, help="write markdown here (else stdout)")
    ap.add_argument("--ident-out", default=None, help="write the identity-probe JSON here")
    ap.add_argument("--arm", default="cl")
    ap.add_argument("--floor", default="ha0")
    ap.add_argument("--allow-probe-disagreement", action="store_true",
                    help="⛔ do not use: renders even when the two probes disagree")
    a = ap.parse_args()

    rec = json.load(open(a.record, encoding="utf-8"))
    ident = identity_probe(a.dump, a.arm, a.floor)
    xc = cross_check(rec, ident)
    if a.ident_out:
        with open(a.ident_out, "w", encoding="utf-8") as fh:
            json.dump({"identity_probe": ident, "cross_check": xc}, fh, indent=1)
    if xc["status"] != "OK" and not a.allow_probe_disagreement:
        print(json.dumps({"status": "REFUSED", "cross_check": xc}, indent=1))
        return 2
    md = render(rec, ident, xc, a.title)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(md)
        print(f"[out] {a.out}  ({len(md)} chars)")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
