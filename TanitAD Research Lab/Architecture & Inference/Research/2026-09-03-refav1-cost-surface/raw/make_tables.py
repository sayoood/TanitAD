#!/usr/bin/env python3
"""Render the two cost-surface JSONs into the RESULT.md tables. Analysis only."""
import json
import os
import sys

import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
RUNS = [("incumbent", "cost_surface_incumbent.json"),
        ("clean epoch (ep2)", "cost_surface_ep2.json")]


def load():
    out = {}
    for name, f in RUNS:
        p = os.path.join(OUT, f)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                out[name] = json.load(fh)
    return out


def ci(d, pct=False, prec=3):
    if d is None or d.get("point") is None:
        return "—"
    m = 100.0 if pct else 1.0
    s = "%" if pct else ""
    return (f"{d['point'] * m:.{prec}f}{s} "
            f"[{d['lo'] * m:.{prec}f}, {d['hi'] * m:.{prec}f}]")


def main():
    J = load()
    if not J:
        print("no JSON yet"); return 1
    L = []
    P = L.append

    # ---- controls -----------------------------------------------------------
    P("### Controls — every one must read its known value\n")
    P("| control | reads | " + " | ".join(J) + " |")
    P("|---|---|" + "---|" * len(J))
    rows = [
        ("C0 decomposition identity", "terms sum to the total, ≤ 1e-6",
         lambda c: f"{c['C0_decomposition_identity']['max_abs']:.2e} "
                   f"{'PASS' if c['C0_decomposition_identity']['pass'] else 'FAIL'}"),
        ("C1 shipped-cost gate", "our re-scoring == PlanResult.baseline_costs",
         lambda c: f"rel err {c['C1_shipped_cost_gate']['worst_rel_err']:.2e} "
                   f"(oracle-goal arm scale "
                   f"{c['C1_shipped_cost_gate']['max_cost_scale_oracle']:.3g}, n="
                   f"{c['C1_shipped_cost_gate']['n']}) "
                   f"{'PASS' if c['C1_shipped_cost_gate']['pass'] else 'FAIL'}"),
        ("C2 banked-winner gate", "plan() reproduces the banked cl path < 3.8e-6 m",
         lambda c: f"{c['C2_banked_winner_gate']['worst_max_abs_m']:.2e} m "
                   f"(n={c['C2_banked_winner_gate']['n']}) "
                   f"{'PASS' if c['C2_banked_winner_gate']['pass'] else 'FAIL'}"),
        ("C3 zero-model", "action-blind predictor ⇒ goal term flat; "
                          "total(κ)−total(0) == 0.05κ² exactly",
         lambda c: f"ptp {c['C3_zero_model']['goal_ptp_max']:.1e}, "
                   f"penalty recovered to "
                   f"{c['C3_zero_model']['penalty_recovery_max_abs']:.1e}, "
                   f"argmin κ=0 on all "
                   f"{'PASS' if c['C3_zero_model']['pass'] else 'FAIL'}"),
        ("C4 constant-cost", "all weights zeroed ⇒ total ≡ 0, argmin = index 0",
         lambda c: 'PASS' if c['C4_constant_cost']['pass'] else 'FAIL'),
        ("C5 channel-0 invariance", "A and B agree EXACTLY on channel 0",
         lambda c: f"jerk {c['C5_channel0_invariance']['jerk_max_abs_diff']:.1e}, "
                   f"a-marginal "
                   f"{c['C5_channel0_invariance']['a_marginal_max_abs_diff']:.1e} "
                   f"{'PASS' if c['C5_channel0_invariance']['pass'] else 'FAIL'}"),
        ("C6 n and d", "printed",
         lambda c: f"n={c['C6_n_and_d']['n_windows']} windows, "
                   f"d={c['C6_n_and_d']['n_candidates']} candidates"),
        ("C7 conversion agreement", "our arctan(L·κ) == kinematic.as_command",
         lambda c: f"max |Δ| "
                   f"{c['C7_as_command_agreement'].get('max_abs_diff', float('nan')):.1e}"
                   f", L={c['C7_as_command_agreement'].get('kinematic_STEER_WHEELBASE_M')}"
                   f" {'PASS' if c['C7_as_command_agreement'].get('pass') else 'n/a'}"),
        ("**PANEL_VOID**", "any must-pass control failed",
         lambda c: "**" + str(c["PANEL_VOID"]) + "**"),
    ]
    for label, reads, fn in rows:
        cells = []
        for n in J:
            try:
                cells.append(fn(J[n]["summary"]["controls"]))
            except Exception as ex:                             # noqa: BLE001
                cells.append(f"ERR {type(ex).__name__}")
        P(f"| {label} | {reads} | " + " | ".join(cells) + " |")

    # ---- saturation ---------------------------------------------------------
    P("\n### The goal term's resolution — the read that reframes the question\n")
    P("| quantity | " + " | ".join(J) + " |")
    P("|---|" + "---|" * len(J))
    sat_rows = [
        ("goal-term range across the WHOLE grid, in float32 ULPs (median)",
         lambda s: f"{s['median_goal_ptp_in_ulps_A']:.1f}"),
        ("… along the κ axis at a = 0, in ULPs (median) — convention A",
         lambda s: f"**{s['median_goal_KAPPA_range_in_ulps_A']:.2f}**"),
        ("… along the κ axis at a = 0, in ULPs (median) — convention B",
         lambda s: f"**{s['median_goal_KAPPA_range_in_ulps_B']:.2f}**"),
        ("… along the a axis at κ = 0, in ULPs (median)",
         lambda s: f"{s['median_goal_A_range_in_ulps_A']:.1f}"),
        ("distinct float32 values the goal term takes over 189 cells (median)",
         lambda s: f"{s['median_n_distinct_f32_A']:.0f}"),
        ("goal-term values are exact ULP multiples (fraction of windows)",
         lambda s: f"{s['frac_goal_ulp_quantized_A'] * 100:.0f}%"),
        ("goal-term range in float64 (median)",
         lambda s: f"{s['median_goal_ptp_f64_A']:.3e}"),
        ("the explicit κ² penalty at GOAL_KAPPA_TURN = 0.08",
         lambda s: f"{s['median_penalty_at_kturn']:.3e}"),
    ]
    for label, fn in sat_rows:
        cells = []
        for n in J:
            try:
                cells.append(fn(J[n]["summary"]["f32_saturation"]))
            except Exception as ex:                             # noqa: BLE001
                cells.append(f"ERR {type(ex).__name__}")
        P(f"| {label} | " + " | ".join(cells) + " |")

    # ---- the chord-distance headroom ---------------------------------------
    P("\n### Why the goal term loses its own signal: `1 - cos` is QUADRATIC "
      "near its optimum\n")
    P("`1 - cos(z, g) = d^2/2` where `d = ||z_hat - g_hat||` is the chord distance "
      "between the L2-normalised fields. Near `cos = 1` the cost therefore SQUARES "
      "the displacement it is trying to measure, and the square lands under float32's "
      "resolution. The chord distance is the SAME ordering (monotone), computed "
      "without a cancellation.\n")
    P("| quantity | " + " | ".join(J) + " |")
    P("|---|" + "---|" * len(J))
    def chord_stats(j):
        """Computed from the FULL banked surfaces (`--save-surfaces-n`), so both the
        level and the range come from the SAME exact float64 cosine — never from a
        float32 origin mixed with a float64 range."""
        c0s, d0s, drs, ress, steps = [], [], [], [], []
        for s in j.get("surfaces", []):
            G = np.array(s["surface"]["A"]["c_goal_f64"], dtype=np.float64)
            n_a, n_k = G.shape
            row0 = n_a // 2                       # the a = 0 row (odd n_a, exact 0)
            gk = np.maximum(G[row0], 0.0)         # 1-cos is non-negative in exact math
            d = np.sqrt(2.0 * gk)
            c0 = float(gk[n_k // 2])
            d0 = float(d[n_k // 2])
            dr = float(d.max() - d.min())
            res = float(np.spacing(np.float32(max(d0, d.max()))))
            c0s.append(c0); d0s.append(d0); drs.append(dr); ress.append(res)
            steps.append(dr / max(res, 1e-300))
        if not c0s:
            return {k: float("nan") for k in
                    ("c0", "d0", "drange", "res", "steps", "n")}
        return {"c0": float(np.median(c0s)), "d0": float(np.median(d0s)),
                "drange": float(np.median(drs)), "res": float(np.median(ress)),
                "steps": float(np.median(steps)), "n": len(c0s)}
    S = {n: chord_stats(J[n]) for n in J}
    for label, key, fmt in [
            ("goal term `1 - cos` at the (a=0, kappa=0) cell (median)", "c0", "{:.3e}"),
            ("the same as a CHORD DISTANCE `d = sqrt(2(1-cos))` (median)", "d0", "{:.3e}"),
            ("kappa-axis range of `d` (median)", "drange", "{:.3e}"),
            ("float32 resolution AT that `d`", "res", "{:.3e}"),
            ("**representable float32 steps the kappa axis spans, as a chord "
             "distance**", "steps", "{:,.0f}"),
            ("_n windows with a banked full surface_", "n", "{:.0f}")]:
        P(f"| {label} | " + " | ".join(fmt.format(S[n][key]) for n in J) + " |")
    P("\n⇒ the same information, in the same float32, spans **~2 steps as "
      "`1 - cos`** and the number above **as a chord distance**. The loss is "
      "catastrophic cancellation in `1 - cos`, not a limit of the model.\n")

    # ---- goal degeneracy ----------------------------------------------------
    P("\n### Factor (0) — the imagined goal itself\n")
    P("| quantity | " + " | ".join(J) + " |")
    P("|---|" + "---|" * len(J))
    for label, fn in [
        ("windows whose canonical goal controls are EXACTLY zero",
         lambda g: f"**{g['n_zero_goal']} / {g['n']}** "
                   f"({g['frac_zero_goal'] * 100:.1f}%)"),
        ("decoded tactical LATERAL token", lambda g: str(g["goal_lat_counts"])),
        ("decoded tactical LONGITUDINAL token",
         lambda g: str(g["goal_lon_counts"])),
    ]:
        cells = [fn(J[n]["summary"]["goal_degeneracy"]) for n in J]
        P(f"| {label} | " + " | ".join(cells) + " |")

    # ---- does the decoded goal fall where the human turns? -------------------
    P("\n### Does the imagined goal ask for a turn where the human turns?\n")
    P("The 'human turns' stratum is GT heading change ≥ 5° over the 2.0 s plan "
      "horizon, read from the raw 10 Hz poses. Ground truth SELECTS windows here; "
      "it never enters a cost.\n")
    P("| quantity | " + " | ".join(J) + " |")
    P("|---|" + "---|" * len(J))

    def xt(j):
        r = j["rows"]
        deg = np.array([float(x.get("gt_turn_deg") or 0.0) for x in r])
        v0 = np.array([float(x["v0"]) for x in r])
        gt = deg >= 5.0
        gk = np.array([float(x.get("goal_kappa_max") or 0.0) > 0.0 for x in r])
        kimp = np.deg2rad(deg) / np.maximum(v0 * 2.0, 1e-6)
        return {"n": len(r), "gt": int(gt.sum()), "gk": int(gk.sum()),
                "gt20": int((deg >= 20.0).sum()),
                "gt_hard": int((gt & (kimp >= 0.04) & (v0 > 1.0)).sum()),
                "deg_p95": float(np.nanpercentile(deg, 95)),
                "deg_max": float(np.nanmax(deg)),
                "both": int((gt & gk).sum()),
                "gt_not_gk": int((gt & ~gk).sum()),
                "gk_not_gt": int((~gt & gk).sum()),
                "recall": (float((gt & gk).sum()) / max(int(gt.sum()), 1)),
                "prec": (float((gt & gk).sum()) / max(int(gk.sum()), 1))}
    X = {n: xt(J[n]) for n in J}
    for label, k, f in [
            ("GT heading change over 2.0 s — p95 / max (deg)", "deg_p95", "{:.1f}"),
            ("… max (deg)", "deg_max", "{:.1f}"),
            ("windows where the HUMAN turns ≥ 5°/2 s", "gt", "{}"),
            ("… of those, ≥ 20°/2 s (an unambiguous turn)", "gt20", "{}"),
            ("… of those, implied curvature ≥ 0.04 1/m — the scale "
             "`GOAL_KAPPA_TURN` addresses", "gt_hard", "{}"),
            ("windows where the imagined GOAL carries curvature", "gk", "{}"),
            ("both", "both", "{}"),
            ("human turns but the goal is straight", "gt_not_gk", "**{}**"),
            ("goal turns but the human does not", "gk_not_gt", "{}"),
            ("**recall** — the goal turns GIVEN the human turns", "recall", "**{:.3f}**"),
            ("precision — the human turns GIVEN the goal turns", "prec", "{:.3f}")]:
        P(f"| {label} | " + " | ".join(f.format(X[n][k]) for n in J) + " |")

    # ---- the attribution panel ---------------------------------------------
    for n in J:
        s = J[n]["summary"]
        P(f"\n### The attribution panel — {n}\n")
        P("`turn_frac` = fraction of windows whose grid argmin has |κ*| ≥ 0.04 "
          "(= ½ GOAL_KAPPA_TURN). Paired episode-cluster bootstrap, 10,000 "
          "resamples, [2.5, 97.5] pct.\n")
        P("| stratum | n (eps) | A_full (shipped) | A_nopen (−penalty) | "
          "B_full (+conversion) | B_nopen (both) | A_full_f64 (exact cosine) | "
          "B_nopen_f64 (all three) |")
        P("|---|---|---|---|---|---|---|---|")
        for st, blk in s["strata"].items():
            if not blk["n"]:
                P(f"| `{st}` | 0 | — | — | — | — | — | — |")
                continue
            cells = [ci(blk[f"turn_frac_{a}"], pct=True, prec=1) for a in
                     ("A_full", "A_nopen", "B_full", "B_nopen", "A_full_f64",
                      "B_nopen_f64")]
            P(f"| `{st}` | {blk['n']} ({blk['n_episodes']}) | " +
              " | ".join(cells) + " |")
        P("\n**Factor shares** (Δ turn_frac from the shipped arm, same estimator):\n")
        P("| stratum | n | (iii) penalty | (ii) boundary | (ii)+(iii) | "
          "(iv) f32 saturation | all three |")
        P("|---|---|---|---|---|---|---|")
        for st, blk in s["strata"].items():
            if not blk["n"]:
                continue
            P(f"| `{st}` | {blk['n']} | " + " | ".join(
                ci(blk[k], pct=True, prec=1) for k in
                ("share_iii_penalty", "share_ii_boundary", "share_ii_plus_iii",
                 "share_iv_f32_saturation", "share_ii_iii_iv_all")) + " |")
        P("\n**The surface itself** (medians over the stratum):\n")
        P("| stratum | n | κ-range of T1, f64 | a-range of T1, f64 | "
          "lateral/longitudinal potency (f64) | B/A κ-range | "
          "T1 gain at κ_turn (f64) | κ² charge at κ_turn | "
          "windows where gain > charge (A / B) |")
        P("|---|---|---|---|---|---|---|---|---|")
        for st, blk in s["strata"].items():
            if not blk["n"]:
                continue
            P(f"| `{st}` | {blk['n']} | "
              f"{blk['median_A_goal_kappa_range_f64']:.3e} | "
              f"{blk['median_A_goal_a_range_f64']:.3e} | "
              f"{blk['median_A_potency_ratio_f64']:.3e} | "
              f"{blk['median_B_over_A_kappa_range']:.3f} | "
              f"{blk['median_A_goal_gain_at_kturn_f64']:.3e} | "
              f"{blk['penalty_at_kturn']:.3e} | "
              f"{blk['frac_gain_exceeds_penalty_A_f64'] * 100:.0f}% / "
              f"{blk['frac_gain_exceeds_penalty_B_f64'] * 100:.0f}% |")
        # named candidates
        P("\n**The named candidates** (the ones `icem_plan` injects, plus the "
          "canonical-goal seed `plan()` adds):\n")
        P("| candidate | A c_total (median) | of which T1 | T2 jerk | T3 κ² | "
          "beats `cv` (A) | beats `cv` (B) |")
        P("|---|---|---|---|---|---|---|")
        for nm, blk in s["named_candidates"].items():
            P(f"| `{nm}` | {blk.get('A_c_total_median', float('nan')):.3e} | "
              f"{blk.get('A_c_goal_median', float('nan')):.3e} | "
              f"{blk.get('A_c_jerk_median', float('nan')):.3e} | "
              f"{blk.get('A_c_kappa_median', float('nan')):.3e} | "
              f"{blk.get('A_beats_cv_frac', float('nan')) * 100:.0f}% | "
              f"{blk.get('B_beats_cv_frac', float('nan')) * 100:.0f}% |")
        ws = s.get("wheelbase_sensitivity")
        if ws:
            P("\n**Wheelbase sensitivity** (turn_frac of the repaired arm at other L):\n")
            P("| L | turn_frac |")
            P("|---|---|")
            for k, v in ws.items():
                if v is None:
                    continue
                P(f"| {k} | {v * 100:.1f}% |")
        P(f"\n_source: `raw/cost_surface_{'incumbent' if 'incumbent' in n else 'ep2'}"
          f".json`; device {J[n]['device']}; {J[n]['n_windows']} windows; "
          f"{J[n]['wallclock_s']} s; ckpt step {J[n]['model']['step']}_\n")
        P("**Source md5 of every file this run imported:**\n")
        for k, v in J[n]["source_md5"].items():
            P(f"* `{k}` = `{v}`")

    txt = "\n".join(L)
    with open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
