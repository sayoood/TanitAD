#!/usr/bin/env python3
"""Render every table in ``LATENT_ABLATION.md`` FROM THE RAW JSON. Zero GPU.

Nothing in the report is transcribed by hand; this writes ``artifacts/_tables.md``
and the report quotes it. A transcription error is a retraction class this
program has already paid for.

Usage:  python la_tables.py --art ../artifacts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC_LABEL = {
    "imagination": "**INTACT** — the model's own imagined latent",
    "frozen_last": "FROZEN — last real percept, held",
    "frozen_other": "FROZEN-OTHER — a *different* window's percept, held",
    "shuffled": "SHUFFLED — a different window's imagined latent, per step",
    "shuffled_obs": "SHUF-REAL — a different window's *real* latent, per step",
    "mean_latent": "MEAN — the batch-mean percept",
    "zero_latent": "ZERO — all zeros",
    "full_obs": "*FULL-OBS (privileged ceiling)*",
}
ORDER = ("imagination", "full_obs", "frozen_last", "frozen_other", "shuffled",
         "shuffled_obs", "mean_latent", "zero_latent")


def _f(x, n=4):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--art", default=str(Path(__file__).resolve().parent.parent
                                         / "artifacts"))
    a = ap.parse_args()
    art = Path(a.art)
    L = []
    A = json.loads((art / "la_stage_a_frozen.json").read_text(encoding="utf-8"))

    # ---------------- STAGE A ------------------------------------------- #
    L += ["# Rendered tables — every number read from the raw JSON", "",
          "## A. STAGE A — the FROZEN-LATENT ablation (zero GPU, committed dump)",
          "",
          "| α | model share | INTACT `de@2s` | FROZEN `de@2s` | `R_FROZEN` | paired Δ (CI95) | sep | `T_blind` | beats-CV int/frz | `T_useful@1m` int/frz |",
          "|---:|---:|---:|---:|---:|---|---|---:|---|---|"]
    for al, b in A["table"].items():
        p = b["paired_2s_intact_minus_frozen"]
        c = b["capability"]
        L.append(
            f"| {al.split('=')[1]} | {b['model_share_of_command_pct']:.0f} % | "
            f"{b['de_2s']['INTACT']['mean']:.4f} | {b['de_2s']['FROZEN']['mean']:.4f} | "
            f"**{b['R_FROZEN_at_2s']:+.4f}** | {p['delta_b_minus_a']:.4f} "
            f"[{p['lo']:.4f}, {p['hi']:.4f}] | {'✅' if p['separated'] else '⛔'} | "
            f"{b['T_blind_INTACT_vs_FROZEN']['steps']} "
            f"({b['T_blind_INTACT_vs_FROZEN']['s']:.1f} s) | "
            f"{c['beats_cv']['INTACT']['n_steps']}/{c['beats_cv']['FROZEN']['n_steps']} | "
            f"{c['T_useful_s']['INTACT']['1m']} / {c['T_useful_s']['FROZEN']['1m']} s |")
    L += ["", "### A.2 `R_FROZEN` across the horizon (positive = the frozen "
          "latent is WORSE)", ""]
    hdr = list(next(iter(A["table"].values()))["grid"].keys())
    L += ["| α | " + " | ".join(hdr) + " |", "|---|" + "---|" * len(hdr)]
    for al, b in A["table"].items():
        L.append("| " + al.split("=")[1] + " | "
                 + " | ".join(f"{b['grid'][h]['R_x']:+.3f}" for h in hdr) + " |")
    L += ["", "### A.3 the ABSOLUTE deviations behind those ratios (m), "
          "INTACT / FROZEN", ""]
    L += ["| α | " + " | ".join(hdr) + " |", "|---|" + "---|" * len(hdr)]
    for al, b in A["table"].items():
        L.append("| " + al.split("=")[1] + " | "
                 + " | ".join(f"{b['grid'][h]['de_intact_m']:.2f} / "
                              f"{b['grid'][h]['de_ablated_m']:.2f}" for h in hdr)
                 + " |")
    fl = A["floors"]
    L += ["", f"Comparator-free floors: constant velocity `de@2s` "
          f"**{fl['constant_velocity']['de_2s']['mean']:.4f}**, "
          f"`hold_v0` **{fl['hold_v0']['de_2s']['mean']:.4f}** "
          f"(`T_useful@1m` {fl['hold_v0']['T_useful_1m_s']} s).", ""]

    # ---------------- the sweep ------------------------------------------ #
    tp, vp = art / "la_table.json", art / "la_verdict.json"
    if tp.exists():
        T = json.loads(tp.read_text(encoding="utf-8"))
        V = json.loads(vp.read_text(encoding="utf-8"))
        for al in ("1", "0.25", "0.75", "0"):
            k = f"alpha={al}"
            if k not in T:
                continue
            blk = T[k]
            star = " ⭐ **UNCONFOUNDED**" if blk["unconfounded"] else ""
            L += ["", f"## B. THE FULL ABLATION TABLE — α = {al} "
                  f"(model supplies {blk['model_share_of_command_pct']:.0f} % "
                  f"of the steer/accel command){star}", "",
                  "| latent channel | class | `de@2s` [CI95] | `R` | sep | `de@6s` | `ade_0_2s` | `T_blind` vs FROZEN | cost | beats-CV | `T_useful@1m` |",
                  "|---|---|---|---:|---|---:|---:|---:|---:|---|---:|"]
            for s in ORDER:
                r = blk["rows"].get(s)
                if r is None:
                    continue
                d2, tb = r["de_2s"], r["T_blind_vs_FROZEN"]
                tbs = ("⛔ VACUOUS" if tb.get("VACUOUS")
                       else f"{tb['steps']} ({tb['s']:.1f} s)")
                L.append(
                    f"| {SRC_LABEL[s]} | {r['class']} | "
                    f"{d2['mean']:.4f} [{d2['lo']:.4f}, {d2['hi']:.4f}] | "
                    f"{r['R_at_2s']:+.4f} | "
                    f"{'✅' if r['separated_from_intact_at_2s'] else '⛔'} | "
                    f"{r['de_6s']['mean']:.4f} | {r['ade_0_2s']['mean']:.4f} | "
                    f"{tbs} | {_f(r['cost_vs_intact'], 3)} | "
                    f"{r['capability']['beats_cv']['n_steps']}/185 | "
                    f"{r['capability']['T_useful_s']['1m']} s |")
            sd = blk["rows"]["imagination"].get("speed_diagnostic")
            if sd:
                L += ["", f"### B.{al} — the DECODED-SPEED diagnostic "
                      f"(true mean `v0` = {sd['mean_true_v0_mps']:.4f} m/s)", "",
                      "| latent channel | mean decoded speed 0–2 s | R² with true `v0` | mean abs speed error |",
                      "|---|---:|---:|---:|"]
                for s in ORDER:
                    r = blk["rows"].get(s)
                    q = r and r.get("speed_diagnostic")
                    if q:
                        L.append(f"| {SRC_LABEL[s]} | "
                                 f"{q['mean_decoded_speed_0_2s_mps']:.4f} m/s | "
                                 f"{q['r2_with_true_v0']:.4f} | "
                                 f"{q['mean_abs_speed_error_mps']:.4f} m/s |")
        L += ["", "## C. THE VERDICT, applied mechanically", "",
              "| | α = 0.25 (deployable) | α = 1 (⭐ unconfounded) |", "|---|---|---|"]
        v25, v1 = V["alpha=0.25"], V["alpha=1"]
        L += [f"| model's share of the command | "
              f"{v25['model_share_of_command_pct']:.0f} % | "
              f"{v1['model_share_of_command_pct']:.0f} % |",
              f"| `R` over the destructive ablations | "
              f"{v25['R_min']:+.4f} … {v25['R_max']:+.4f} | "
              f"{v1['R_min']:+.4f} … {v1['R_max']:+.4f} |",
              f"| `cost` over the destructive ablations | "
              f"{v25['cost_min']:+.4f} … {v25['cost_max']:+.4f} | "
              f"{v1['cost_min']:+.4f} … {v1['cost_max']:+.4f} |",
              f"| PRIMARY (`de@2s`) | **{v25['PRIMARY_de2s']}** | "
              f"**{v1['PRIMARY_de2s']}** |",
              f"| CO-PRIMARY (`T_blind`) | **{v25['CO_PRIMARY_T_blind']}** | "
              f"**{v1['CO_PRIMARY_T_blind']}** |",
              f"| agree? | {'✅' if v25['agree'] else '⛔'} | "
              f"{'✅' if v1['agree'] else '⛔'} |",
              f"| **VERDICT** | ⭐ **{v25['VERDICT']}** | ⭐ **{v1['VERDICT']}** |",
              "", "Per-ablation detail:", "",
              "| ablation | `R` @0.25 | sep | `cost` @0.25 | `R` @1 | sep | `cost` @1 |",
              "|---|---:|---|---:|---:|---|---:|"]
        for s in v1["R_per_destructive_ablation"]:
            L.append(f"| {SRC_LABEL[s]} | "
                     f"{v25['R_per_destructive_ablation'][s]:+.4f} | "
                     f"{'✅' if v25['separated_per_ablation'][s] else '⛔'} | "
                     f"{_f(v25['cost_per_destructive_ablation'][s], 3)} | "
                     f"{v1['R_per_destructive_ablation'][s]:+.4f} | "
                     f"{'✅' if v1['separated_per_ablation'][s] else '⛔'} | "
                     f"{_f(v1['cost_per_destructive_ablation'][s], 3)} |")
        sr = T.get("seed_robustness")
        if sr:
            L += ["", "### C.2 seed robustness of the derangement (`de@2s`)", "",
                  "| α | " + " | ".join(sorted(next(iter(sr.values())))) + " |",
                  "|---|" + "---|" * len(next(iter(sr.values())))]
            for al, blk in sr.items():
                L.append("| " + al.split("=")[1] + " | "
                         + " | ".join(f"{blk[k]:.4f}" for k in sorted(blk)) + " |")

    fp = art / "la_fixedpoint.json"
    if fp.exists():
        F = json.loads(fp.read_text(encoding="utf-8"))
        L += ["", "## D. THE FIXED-POINT PROBE (corroborative, never adjudicating)",
              "", "| arm | rel. step @0.1 s | rel. step @4 s | ratio | "
              "‖z−z₀‖ change 10 s→18.5 s | cos(z,z₀) @18.5 s | **reading** |",
              "|---|---:|---:|---:|---:|---:|---|"]
        for k, b in sorted(F.items()):
            if not isinstance(b, dict) or "READING" not in b:
                continue
            L.append(f"| `{k}` | {b['rel_step_at_0.1s']:.4f} | "
                     f"{b['rel_step_at_4s']:.4f} | "
                     f"{b['rel_step_ratio_4s_over_0.1s']:.3f} | "
                     f"{b['d0_change_10s_to_18.5s_frac']:+.3f} | "
                     f"{b['cos_with_z0']['18.5s']:.4f} | "
                     f"**{b['READING']}** |")
        k0 = "imagination__a1"
        if k0 in F:
            hdr2 = list(F[k0]["rel_step_size"].keys())
            L += ["", f"### D.2 `{k0}` — the trajectory in full", "",
                  "| quantity | " + " | ".join(hdr2) + " |",
                  "|---|" + "---|" * len(hdr2)]
            for q, lab in (("rel_step_size", "‖Δz‖/‖z‖"),
                           ("dist_from_z0", "‖z−z₀‖"),
                           ("cos_with_z0", "cos(z,z₀)"),
                           ("latent_norm", "‖z‖")):
                L.append(f"| {lab} | " + " | ".join(
                    f"{F[k0][q][h]:.4f}" for h in hdr2) + " |")

    (art / "_tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[tables] wrote {art / '_tables.md'} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
