#!/usr/bin/env python3
"""RUNG 1 — render every table in ``TBLIND_RUNG1.md`` FROM THE RAW JSON.

The program's first working agreement exists because prose lied to us: a number
that is hand-transcribed from a run into a report is a number nobody can audit.
Every table below is generated from ``artifacts/*.json`` so the report and the
artifacts cannot drift.

Usage:
    python tb_rung1_tables.py --art artifacts > artifacts/_tables.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ci(b):
    return "[%.1f, %.1f]" % (b[0], b[1])


def _sep(d):
    return "✅" if d.get("separated") else "⛔"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--art", required=True)
    a = ap.parse_args()
    A = Path(a.art)
    J = {p.stem: json.loads(p.read_text(encoding="utf-8"))
         for p in A.glob("rung1_*.json")}
    o = []

    g = J["rung1_gates"]
    o.append("## Gates\n")
    o.append("| check | result |")
    o.append("|---|---|")
    ident = g["window_set_identity_gate"]
    o.append(f"| windows | **{ident['n_windows_new']} new vs "
             f"{ident['n_windows_committed']} committed** |")
    o.append(f"| episode clusters | {ident['n_episode_clusters']} |")
    o.append(f"| `eid` ordering identical (both dumps) | "
             f"{ident['eid_identical_vs_bi'] and ident['eid_identical_vs_matched']} |")
    o.append(f"| `t0` ordering identical | {ident['t0_identical_vs_bi']} |")
    for arm, v in ident["anchors"].items():
        o.append(f"| anchor `{arm}` (in `{v['committed_in']}`) | max \\|Δ\\| = "
                 f"**{v['max_abs_diff_m']:.3g} m** (tol {v['tol_m']:g}) |")
    o.append(f"| **WINDOW-SET IDENTITY GATE** | **"
             f"{'PASS' if ident['GATE_PASS'] else 'FAIL'}** |")
    st = g["plumbing_selftest"]
    for k, v in st.items():
        if isinstance(v, dict) and "must_equal" in v:
            o.append(f"| self-test `{k}` == `{v['must_equal']}` | max \\|Δ\\| = "
                     f"**{v['max_abs_diff_m']:g}** — "
                     f"{'BIT-IDENTICAL' if v['bit_identical'] else 'DIFFERS'} |")
    o.append(f"| anti-no-op: smallest \\|Δ\\| vs `own` over all filter arms | "
             f"**{st.get('_anti_noop_min_abs_diff_vs_own_m')} m** |")
    o.append(f"| arms identical to `own` (must be empty) | "
             f"`{st.get('_anti_noop_arms_identical_to_own')}` |")
    p = g["diagnostic_vacuity_audit"]["failing_value_probe"]
    o.append(f"| failing-value probe — identical arms / swapped arms | "
             f"**{p['identical_arms_steps']} / {p['swapped_arms_steps']} steps** "
             f"(both must be 1) |")
    o.append(f"| **ALL GATES** | **"
             f"{'PASS' if g['ALL_GATES_PASS'] else 'FAIL'}** |")
    f = g["fidelity_vs_ladder"]
    o.append("\n### Fidelity against the ladder's committed numbers\n")
    o.append("| quantity | committed | recomputed here |")
    o.append("|---|---:|---:|")
    for k, v in f.items():
        if isinstance(v, dict) and "committed" in v:
            o.append(f"| `{k}` | {v['committed']} | **{v['recomputed']}** |")
    o.append(f"\n`LEVEL_FIDELITY_PASS = {f['LEVEL_FIDELITY_PASS']}` · "
             f"`T_BLIND_EXACT_REPRODUCTION = {f['T_BLIND_EXACT_REPRODUCTION']}` · "
             f"`T_useful_reproduces = {f['T_useful_reproduces']}`\n")

    b = J["rung1_blend_curve"]
    o.append("\n## The blend curve — own actions <-> hold-last\n")
    o.append("| α | eligible | `T_blind` | CI95 (s) | Δ vs own (steps) | `de@2s` | "
             "`ade_0_2s` | paired Δ@2s vs comparator | beats CV | `T_useful@1m` |")
    o.append("|---:|---|---:|---|---|---:|---:|---|---:|---:|")
    for al in b["alphas"]:
        c = b["curve"][f"{al:g}"]
        gv = c.get("gain_vs_own_baseline", {})
        gtxt = ("%+.0f %s" % (gv.get("median_gain_steps", 0),
                              "✅" if gv.get("separated_better") else "⛔")
                if gv else "—")
        pd = c["paired_delta_2s_vs_comparator"]
        o.append("| %g | %s | **%d** (%.1f s) | %s | %s | %.4f | %.4f | "
                 "%+.4f [%+.4f, %+.4f] %s | %d/185 | %.1f s |"
                 % (al, "✅" if c["eligible"] else "⛔ endpoint",
                    c["T_blind_steps"], c["T_blind_s"], _ci(c["T_blind_ci95_s"]),
                    gtxt, c["de_at_2s"]["mean"], c["ade_0_2s"]["mean"],
                    pd["delta_b_minus_a"], pd["lo"], pd["hi"], _sep(pd),
                    c["beats_cv"]["n_steps"], c["T_useful_s"]["1m"]))
    o.append(f"\nmonotone non-decreasing in α: **{b['monotone_nondecreasing']}** · "
             f"Spearman(α, `T_blind`) = **{b['spearman_alpha_vs_T']}**\n")

    iv = J["rung1_interventions"]
    o.append("\n## The other intervention families\n")
    o.append("| family | config | eligible | `T_blind` | CI95 (s) | Δ vs own | "
             "`de@2s` | `ade_0_2s` | beats CV | `T_useful@1m` |")
    o.append("|---|---|---|---:|---|---|---:|---:|---:|---:|")
    for fam, blk in iv.items():
        for tag, c in blk.items():
            if c.get("MISSING"):
                o.append(f"| {fam} | `{tag}` | — | MISSING | | | | | | |")
                continue
            gv = c.get("gain_vs_own_baseline", {})
            gtxt = ("%+.0f %s" % (gv.get("median_gain_steps", 0),
                                  "✅" if gv.get("separated_better") else "⛔"))
            o.append("| %s | `%s` | %s | **%d** (%.1f s) | %s | %s | %.4f | "
                     "%.4f | %d/185 | %.1f s |"
                     % (fam, tag, "✅" if c["eligible"] else "⛔ diagnostic",
                        c["T_blind_steps"], c["T_blind_s"],
                        _ci(c["T_blind_ci95_s"]), gtxt,
                        c["de_at_2s"]["mean"], c["ade_0_2s"]["mean"],
                        c["beats_cv"]["n_steps"], c["T_useful_s"]["1m"]))

    m = J["rung1_mechanism"]
    pen = m["penalty_own_minus_hold"]
    o.append("\n## Mechanism\n### The penalty curve (own − hold), comparator-free\n")
    o.append("| step | penalty (m) | own / hold |")
    o.append("|---:|---:|---:|")
    for k in pen["at_steps"]:
        o.append(f"| {k} | {pen['at_steps'][k]} | "
                 f"{pen['ratio_own_over_hold'][k]} |")
    for key in ("loglog_fit_2_20", "loglog_fit_20_185", "loglog_fit_full"):
        fit = pen.get(key)
        if fit:
            o.append(f"\n`{key}`: exponent **{fit['exponent']}**, R² "
                     f"{fit['r2']}, n {fit['n']}, window "
                     f"{fit['fit_window_steps']} — admissible: "
                     f"{fit['admissible_per_CLAUDE_md']}")
    act = m["action_statistics_599_windows"]
    o.append("\n### Action statistics (reconstructed over all 599 windows)\n")
    o.append("| arm | step | mean \\|steer\\| | mean \\|accel\\| | frac steer at "
             "clamp | frac accel at clamp | jitter(steer) | jitter(accel) | "
             "mean speed |")
    o.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm, s in act.items():
        for n in ("5", "20", "60", "185"):
            if n not in s["mean_abs_steer_rad"]:
                continue
            o.append("| `%s` | %s | %.5f | %.4f | %.4f | %.4f | %.5f | %.4f | "
                     "%.2f |"
                     % (arm, n, s["mean_abs_steer_rad"][n],
                        s["mean_abs_accel_ms2"][n], s["frac_steer_at_clamp"][n],
                        s["frac_accel_at_clamp"][n],
                        s["mean_step_jitter_steer"][n],
                        s["mean_step_jitter_accel"][n],
                        s["mean_pred_speed_ms"][n]))
    on = m["onset_switch_sweep"]
    o.append("\n### The onset sweep — recovery as a fraction of the own→hold gap\n")
    o.append("| arm | " + " | ".join(f"@{n}" for n in ("5", "20", "60", "185"))
             + " |")
    o.append("|---|" + "---:|" * 4)
    for tag, v in on.items():
        o.append("| `%s` | " % tag + " | ".join(
            "%.3f" % v["recovery_frac_of_own_hold_gap"].get(n, float("nan"))
            for n in ("5", "20", "60", "185")) + " |")
    if "fed_action_stats_audit_subset" in m:
        o.append("\n### Fed actions, dense (audit subset)\n")
        o.append("| arm | mean \\|steer\\| | mean \\|accel\\| | frac steer at "
                 "clamp | frac accel at clamp | jitter(steer) | jitter(accel) |")
        o.append("|---|---:|---:|---:|---:|---:|---:|")
        for arm, s in m["fed_action_stats_audit_subset"].items():
            o.append("| `%s` | %.5f | %.4f | %.4f | %.4f | %.5f | %.4f |"
                     % (arm, s["mean_abs_steer_rad"], s["mean_abs_accel_ms2"],
                        s["frac_steer_at_clamp"], s["frac_accel_at_clamp"],
                        s["mean_step_jitter_steer"], s["mean_step_jitter_accel"]))

    v = J["rung1_verdict"]
    o.append("\n## Verdict\n")
    o.append("```json")
    o.append(json.dumps({k: x for k, x in v.items() if k != "ranking"}, indent=1))
    o.append("```")
    o.append("\n### Ranking of ELIGIBLE arms by `T_blind`\n")
    o.append("| rank | arm | steps | s |")
    o.append("|---:|---|---:|---:|")
    for i, (nm, steps, secs) in enumerate(v["ranking"], 1):
        o.append(f"| {i} | `{nm}` | {steps} | {secs} |")
    print("\n".join(o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
