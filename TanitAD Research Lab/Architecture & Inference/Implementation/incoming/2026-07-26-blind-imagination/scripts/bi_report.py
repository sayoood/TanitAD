#!/usr/bin/env python3
"""Render every table in `BLIND_IMAGINATION.md` from the artifact JSON.

No number in the report is typed by hand — the `h2c_report.py` pattern. Writes
`artifacts/_tables.md`; `bi_splice.py` places the blocks at the markers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GRID_LABEL = ["0.5s", "1s", "2s", "3s", "4.5s", "6s", "9s", "12s", "18.5s"]
PRETTY = {
    "a_imagination__true": "(a) IMAGINATION",
    "b_frozenlast__true": "(b) FROZEN-LAST-FRAME",
    "c_fullobs__true": "(c) FULL OBSERVATION",
    "c2_observedpair__true": "(c2) observed-pair odometry",
    "a_imagination__own": "(a) IMAGINATION",
    "b_frozenlast__own": "(b) FROZEN-LAST-FRAME",
    "c_fullobs__own": "(c) FULL OBS (teacher-forced percept)",
    "a_imagination__hold": "(a) IMAGINATION",
    "b_frozenlast__hold": "(b) FROZEN-LAST-FRAME",
    "a_imagination__gtkin": "(a) IMAGINATION",
    "b_frozenlast__gtkin": "(b) FROZEN-LAST-FRAME",
    "c_fullobs__gtkin": "(c) FULL OBSERVATION",
    "d_constant_velocity": "(d) CONSTANT VELOCITY — the floor",
    "d2_hold_v0": "(d2) hold-v0 go-straight",
    "a_imagination__own_vupd": "(a) imagination, speed channel updated",
    "a_imagination__true__roTAC": "(a) imagination, readout=tac (16-step)",
    "a_imagination__true__roSTR": "(a) imagination, readout=str (20-step)",
    "a_imagination__own__roSTR": "(a) imagination, readout=str (20-step)",
    "a_imagination__hold__roSTR": "(a) imagination, readout=str (20-step)",
    "b_frozenlast__true__roSTR": "(b) frozen-last, readout=str (20-step)",
    "c2_observedpair__true__roSTR": "(c2) observed-pair, readout=str",
}
REGIME_TITLE = {
    "privileged_true_actions":
        "REGIME (i) — TRUE FUTURE ACTIONS  ⚠️ PRIVILEGED UPPER BOUND, not deployable",
    "deployable_own_actions":
        "REGIME (ii) — THE MODEL'S OWN ACTIONS  ⭐ THE DEPLOYABLE CONDITION",
    "deployable_held_action":
        "REGIME (ii-0) — HELD LAST ACTION (deployable, no policy)",
    "convention_control_gt_actions":
        "CONVENTION CONTROL — actions from the TRUE motion through the same inverse",
    "A2_readout_str_true_actions":
        "A2 SENSITIVITY — readout = step['str'] (20-step-calibrated), true actions",
    "A2_readout_str_own_actions":
        "A2 SENSITIVITY — readout = step['str'], own actions",
}


def _f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def curve_table(curve, arms, metric="de"):
    ks = list(curve["arms"][arms[0]][metric].keys())
    head = "| arm | " + " | ".join(ks) + " |"
    sep = "|---|" + "---|" * len(ks)
    rows = [head, sep]
    for a in arms:
        if a not in curve["arms"]:
            continue
        cells = [f"**{curve['arms'][a][metric][k]['mean']:.3f}**" for k in ks]
        rows.append(f"| {PRETTY.get(a, a)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def paired_table(block, title):
    ks = list(block.keys())
    rows = [f"| {title} | " + " | ".join(ks) + " |",
            "|---|" + "---|" * len(ks)]
    d = [f"{block[k]['delta_b_minus_a']:+.3f}" for k in ks]
    lo = [f"[{block[k]['lo']:+.3f}, {block[k]['hi']:+.3f}]" for k in ks]
    sep = ["✅" if block[k]["a_better"] else
           ("⛔" if block[k]["separated"] else "—") for k in ks]
    rows.append("| Δ (b − a), + = imagination better | " + " | ".join(d) + " |")
    rows.append("| CI95 | " + " | ".join(lo) + " |")
    rows.append("| imagination separated-better? | " + " | ".join(sep) + " |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts")
    ap.add_argument("out")
    a = ap.parse_args()
    A = Path(a.artifacts)
    curve = json.loads((A / "horizon_curve.json").read_text(encoding="utf-8"))
    tb = json.loads((A / "t_blind.json").read_text(encoding="utf-8"))
    dec = json.loads((A / "decomposition.json").read_text(encoding="utf-8"))
    duty = json.loads((A / "duty_cycle.json").read_text(encoding="utf-8"))
    gate = json.loads((A / "gate_reproduction.json").read_text(encoding="utf-8"))
    out = []

    # ---- gate ------------------------------------------------------------- #
    out.append("<!-- TABLES:GATE -->")
    out.append("| deployment | n windows | n episode clusters | `ade_0_2s` "
               "(new instrument) | committed | max abs diff vs unmodified "
               "`rollout.collect` |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for k, lbl in (("40ep_canonical", "40 eps — CANONICAL"),
                   ("600ep", "600 eps")):
        g = gate[k]
        md = g["max_abs_diff_pred_vs_rollout_collect_m"]
        out.append(
            f"| **{lbl}** | {g['n_windows']} | {g['n_episode_clusters']} | "
            f"**{g['ade_0_2s_new_instrument']:.6f}** | "
            f"{g['ade_0_2s_expected_committed']} | "
            f"{'%.2e m' % md if md is not None else 'n/a (cached-encode path)'} |")
    out.append("")
    out.append(f"**`GATE_PASS = {gate['GATE_PASS']}`** · ckpt step "
               f"{gate['ckpt_step']} · torch {gate['torch']} · "
               f"python {gate['python']}")
    out.append("<!-- /TABLES:GATE -->\n")

    # ---- the horizon curve, per regime ------------------------------------ #
    out.append("<!-- TABLES:CURVE -->")
    m = curve["meta"]
    out.append(f"*One fixed window set: **{m['n_windows']} windows / "
               f"{m['n_episode_clusters']} episode clusters**, `K_max = "
               f"{m['kmax']}`, stride {m['stride']}, {m['episodes_requested']}-"
               f"episode clean val. Every horizon and every arm is scored on "
               f"the SAME windows, so the whole curve is paired.*\n")
    for reg, arms in (("privileged_true_actions",
                       ["a_imagination__true", "b_frozenlast__true",
                        "c_fullobs__true", "c2_observedpair__true",
                        "d_constant_velocity", "d2_hold_v0"]),
                      ("deployable_own_actions",
                       ["a_imagination__own", "b_frozenlast__own",
                        "c_fullobs__own", "d_constant_velocity"]),
                      ("deployable_held_action",
                       ["a_imagination__hold", "b_frozenlast__hold",
                        "d_constant_velocity"]),
                      ("convention_control_gt_actions",
                       ["a_imagination__gtkin", "b_frozenlast__gtkin",
                        "c_fullobs__gtkin"])):
        out.append(f"### {REGIME_TITLE[reg]}\n")
        out.append("**`de_N` — displacement error AT horizon N (m)**\n")
        out.append(curve_table(curve, arms, "de"))
        out.append("")
        if reg in tb["regimes"]:
            out.append(paired_table(tb["regimes"][reg]["paired_de_at_grid"],
                                    "paired contrast"))
            out.append("")
    out.append("<!-- /TABLES:CURVE -->\n")

    # ---- T_blind ----------------------------------------------------------- #
    out.append("<!-- TABLES:TBLIND -->")
    out.append("| regime | `T_blind` | CI95 | draws with `T_blind = 0` | "
               "first step where (b) is separated-BETTER | C14 saturated? |")
    out.append("|---|---:|---|---:|---:|---|")
    for reg, blk in tb["regimes"].items():
        t = blk["t_blind"]
        fs = t["first_step_where_b_separated_better"]
        out.append(
            f"| {REGIME_TITLE.get(reg, reg)} | **{t['T_blind_s']:.1f} s** "
            f"({t['T_blind_steps']} steps) | "
            f"[{t['T_blind_ci95_s'][0]:.1f}, {t['T_blind_ci95_s'][1]:.1f}] s | "
            f"{t['frac_draws_T_blind_is_zero']:.3f} | "
            f"{('%.1f s' % (fs * 0.1)) if fs else '—'} | "
            f"{'⚠️ YES — LOWER BOUND' if t['C14_saturated_at_grid_terminus'] else 'no'} |")
    out.append("")
    out.append("**Usefulness horizons for arm (a), and the CV floor crossing**\n")
    out.append("| regime | `de_N` < 1.0 m | < 1.391 m (corridor) | < 2.0 m "
               "(miss@2m) | beats CV floor until |")
    out.append("|---|---|---|---|---|")
    for reg, blk in tb["regimes"].items():
        b = blk["t_useful_bars"]
        cv = blk["t_beats_cv"]
        out.append(
            f"| {REGIME_TITLE.get(reg, reg)} | "
            f"{b['lane_half_1m']['T_s']:.1f} s | "
            f"{b['corridor_1p391m']['T_s']:.1f} s | "
            f"{b['miss_2m']['T_s']:.1f} s | "
            f"**{cv['T_blind_s']:.1f} s** "
            f"[{cv['T_blind_ci95_s'][0]:.1f}, {cv['T_blind_ci95_s'][1]:.1f}] |")
    out.append("<!-- /TABLES:TBLIND -->\n")

    # ---- the action-inverse control ---------------------------------------- #
    out.append("<!-- TABLES:CONTROL -->")
    if "action_inverse_fidelity" in tb:
        out.append("*Positive = the TRUE-action arm is better, i.e. the cost of "
                   "routing the true motion through my kinematic inverse. A "
                   "value indistinguishable from 0 means the inverse is "
                   "faithful and any own-action penalty belongs to the model.*\n")
        out.append(paired_table(tb["action_inverse_fidelity"],
                                "true_future − gt_kinematic, arm (a)"))
    out.append("<!-- /TABLES:CONTROL -->\n")

    # ---- A2 readout lever --------------------------------------------------- #
    out.append("<!-- TABLES:LEVER -->")
    lv = tb.get("A2_readout_level_lever", {})
    if lv.get("contrasts"):
        gk = list(next(iter(lv["contrasts"].values())).keys())
        out.append("| contrast (positive = the ALTERNATE readout is better) | " +
                   " | ".join(gk) + " |")
        out.append("|---|" + "---|" * len(gk))
        for name, blk in lv["contrasts"].items():
            cells = []
            for k, v in blk.items():
                mark = "✅" if v["a_better"] else ("⛔" if v["separated"] else "")
                cells.append(f"{v['delta_b_minus_a']:+.3f}{mark}")
            out.append(f"| `{name}` | " + " | ".join(cells) + " |")
    out.append("<!-- /TABLES:LEVER -->\n")

    # ---- decomposition ------------------------------------------------------ #
    out.append("<!-- TABLES:DECOMP -->")
    for arm in ("a_imagination__true", "a_imagination__own",
                "b_frozenlast__true", "c2_observedpair__true"):
        if arm not in dec["arms"]:
            continue
        out.append(f"**{PRETTY.get(arm, arm)}** — `{arm}`\n")
        ph = dec["arms"][arm]["per_horizon"]
        ks = list(ph.keys())
        out.append("| quantity | " + " | ".join(ks) + " |")
        out.append("|---|" + "---|" * len(ks))
        for key, lbl, nd in (
                ("de_mean", "`de_N` (m)", 3),
                ("ego_along_abs_mean", "along-track \\|err\\| (m)", 3),
                ("ego_cross_abs_mean", "cross-track \\|err\\| (m)", 3),
                ("ego_longitudinal_energy_share",
                 "longitudinal share of squared error", 3),
                ("drift_along_signed_mean", "DRIFT: signed along mean (m)", 3),
                ("variance_along_std", "VARIANCE: along std (m)", 3),
                ("drift_cross_signed_mean", "DRIFT: signed cross mean (m)", 3),
                ("variance_cross_std", "VARIANCE: cross std (m)", 3),
                ("drift_share_along", "drift share of along energy", 3),
                ("frenet_cross_p90", "Frenet cross-track p90 (m)", 2),
                ("pred_speed_mean_mps", "model's own predicted speed (m/s)", 2),
                ("frac_steps_out_of_envelope",
                 "frac steps OUTSIDE the measured envelope", 3)):
            out.append(f"| {lbl} | " +
                       " | ".join(_f(ph[k].get(key), nd) for k in ks) + " |")
        out.append("")
    out.append(f"*GT ego speed at window end: "
               f"**{dec['gt_speed_at_window_end_mps']:.2f} m/s**. "
               f"Envelope: |dlat| ≤ {dec['meta']['envelope']['ENV_LAT_MAX_m']} m, "
               f"|dyaw| ≤ {dec['meta']['envelope']['ENV_YAW_MAX_deg']}°. "
               f"⚠️ {dec['meta']['envelope']['extrapolation_note']}*")
    out.append("<!-- /TABLES:DECOMP -->\n")

    # ---- duty cycle --------------------------------------------------------- #
    out.append("<!-- TABLES:DUTY -->")
    pol = duty["policies"]
    uni = sorted([(v["duty_cycle_realised"], k, v) for k, v in pol.items()
                  if v["config"].get("uniform_period")
                  and v["config"].get("base_action_source") == "own_kinematic"])
    orc = sorted([(v["duty_cycle_realised"], k, v) for k, v in pol.items()
                  if v["config"].get("oracle_bar_m") is not None
                  and v["config"].get("base_action_source") == "own_kinematic"])
    for label, rows in (("UNIFORM peek-every-T′ (deployable)", uni),
                        ("ORACLE peek ⚠️ privileged — reads the true error", orc)):
        if not rows:
            continue
        out.append(f"**{label}** — base arm: imagination + own actions\n")
        ks = list(rows[0][2]["de"].keys())
        out.append("| policy | front-camera duty cycle | " + " | ".join(
            f"de@{k}" for k in ks) + " |")
        out.append("|---|---:|" + "---:|" * len(ks))
        for duty_c, name, v in rows:
            out.append(f"| `{name}` | **{duty_c:.4f}** | " + " | ".join(
                f"{v['de'][k]['mean']:.3f}" for k in ks) + " |")
        out.append("")
    if duty.get("baselines"):
        bk = list(next(iter(duty["baselines"].values()))["de"].keys())
        out.append("**Anchors — the two ends of the duty-cycle axis**\n")
        out.append("| baseline | duty cycle | " +
                   " | ".join(f"de@{k}" for k in bk) + " |")
        out.append("|---|---:|" + "---:|" * len(bk))
        for name, base in duty["baselines"].items():
            out.append(f"| `{name}` | {base['duty_cycle']:.1f} | " + " | ".join(
                f"{base['de'][k]['mean']:.3f}" for k in bk) + " |")
        out.append("")
    ovu = duty.get("oracle_vs_uniform", {})
    if ovu:
        out.append("**ORACLE vs UNIFORM at matched duty cycle — the informative "
                   "version of H2's efficiency claim**\n")
        ks = list(next(iter(ovu.values()))["relative_de_reduction"].keys())
        out.append("| oracle policy | oracle duty | matched uniform | uniform "
                   "duty | " + " | ".join(f"rel. Δde@{k}" for k in ks) + " |")
        out.append("|---|---:|---|---:|" + "---:|" * len(ks))
        for name, v in ovu.items():
            out.append(
                f"| `{name}` | {v['oracle_duty']:.4f} | "
                f"`{v['matched_uniform']}` | {v['matched_uniform_duty']:.4f} | "
                + " | ".join(f"{v['relative_de_reduction'][k]:+.1%}"
                             for k in ks) + " |")
    out.append("<!-- /TABLES:DUTY -->\n")

    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
