"""Render the STREAM D result JSON as the tables the report quotes.

Exists so every number in `ACCEL_RECOVERABILITY.md` is produced by a script from
the raw JSON rather than transcribed by hand — the transcription step is where
this programme has repeatedly lost numbers.

usage: python summarize.py [--res results_accel_recoverability.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SC = ("speed", "yaw_rate", "steer", "long_accel")


def ci(d, k="point"):
    return f"{d[k]:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"


def dci(d):
    star = "*" if d.get("separated") else ""
    deg = " (DEGENERATE)" if d.get("degenerate") else ""
    return f"{d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]{star}{deg}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(HERE / "results_accel_recoverability.json"))
    a = ap.parse_args()
    r = json.loads(Path(a.res).read_text())
    arms = r["arms"]
    real = [k for k in sorted(arms) if not k.endswith("__CTRL")]

    print("\n### A. HELD-OUT R2 PER ARM (point [lo, hi]), 17 episodes / "
          f"{r['split']['n_heldout_windows']} windows\n")
    print("| arm | params/feats | speed | steer | yaw_rate | long_accel | "
          "long_accel TRAIN in-sample |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for k in real:
        m = arms[k]
        size = m.get("params") or m.get("n_features") or 0
        print(f"| {k} | {size:,} | {arms[k]['r2']['speed']['point']:+.4f} | "
              f"{arms[k]['r2']['steer']['point']:+.4f} | "
              f"{arms[k]['r2']['yaw_rate']['point']:+.4f} | "
              f"**{ci(arms[k]['r2']['long_accel'])}** | "
              f"{arms[k]['r2_train_insample']['long_accel']:+.4f} |")

    print("\n### B. PAIRED dR2 vs the SHUFFLED-LATENT CONTROL (matched capacity)\n")
    print("| arm | d speed | d steer | d long_accel |")
    print("|---|---|---|---|")
    for k in real:
        t = f"{k}_minus_CTRL"
        if t not in r["paired_vs_control"]:
            continue
        p = r["paired_vs_control"][t]
        print(f"| {k} | {dci(p['r2_speed'])} | {dci(p['r2_steer'])} | "
              f"**{dci(p['r2_long_accel'])}** |")

    print("\n### B2. PAIRED dR2 vs the EMPIRICAL NULL (train-mean predictor)\n")
    print("| arm | d long_accel |")
    print("|---|---|")
    for k in real:
        t = f"{k}_minus_NULLMEAN"
        if t in r["paired_vs_control"]:
            print(f"| {k} | {dci(r['paired_vs_control'][t]['r2_long_accel'])} |")

    print("\n### C. ORACLE-SELECTED UPPER BOUND (hyperparameter / epoch budget "
          "picked ON the held-out set — cheating, an upper bound only)\n")
    print("| arm | long_accel | speed |")
    print("|---|---:|---:|")
    for k in real:
        o = arms[k].get("ORACLE_SELECTED_heldout_r2_UPPER_BOUND_cheating")
        if o:
            print(f"| {k} | {o['long_accel']['r2']:+.4f} | {o['speed']['r2']:+.4f} |")
        elif "ORACLE_over_epoch_grid_heldout_long_accel_r2" in arms[k]:
            print(f"| {k} | "
                  f"{arms[k]['ORACLE_over_epoch_grid_heldout_long_accel_r2']:+.4f} "
                  f"| (epoch-grid bound) |")
    print("\n### C2. NEURAL ARMS — epoch budget actually selected, and what the "
          "arm scored\n")
    print("| arm | epochs | inner selection score | per-seed heldout long_accel R2 |")
    print("|---|---:|---:|---|")
    for k in real:
        m = arms[k]
        if m.get("kind") not in ("neural", "capacity_control_oracle_input"):
            continue
        print(f"| {k} | {m['epochs_selected']} | {m['inner_selection_score']:+.4f} "
              f"| {m.get('per_seed_heldout_long_accel_r2')} |")

    if "detection_sensitivity" in r:
        s = r["detection_sensitivity"]
        print("\n### D. DETECTION SENSITIVITY — amplitude sweep (clean signal, "
              "rho=1)\n")
        print("| planted amplitude (frac of latent RMS) | heldout R2 long_accel |")
        print("|---:|---|")
        for lv in s["amplitude_sweep_rho1"]:
            print(f"| {lv['frac_of_latent_rms']:g} | "
                  f"{ci(lv['heldout_r2_long_accel'])} |")
        for key in [k for k in s if k.startswith("correlation_sweep")]:
            print(f"\n### D2. {key}\n")
            print("| planted TRUE R2 | measured heldout R2 | oracle-selected | "
                  "paired d vs control |")
            print("|---:|---|---:|---|")
            for lv in s[key]:
                print(f"| {lv['planted_true_r2_rho2']:g} | "
                      f"{ci(lv['heldout_r2_long_accel'])} | "
                      f"{lv['ORACLE_SELECTED_heldout_r2_cheating']['r2']:+.4f} | "
                      f"{dci(lv['paired_vs_shuffled_control'])} |")
        print("\nnative projection variance:", s.get("native_projection_var"))

    if "context_length" in r:
        c = r["context_length"]
        print(f"\n### E. CONTEXT LENGTH (k={c['k_wide']} vs k=4, same "
              f"{c['n_heldout_windows']} rows)\n")
        print("| arm | speed | steer | long_accel |")
        print("|---|---:|---:|---|")
        for k, v in c["arms"].items():
            print(f"| {k} | {v['heldout_r2']['speed']['point']:+.4f} | "
                  f"{v['heldout_r2']['steer']['point']:+.4f} | "
                  f"{ci(v['heldout_r2']['long_accel'])} |")
        print("\npaired wide - narrow:",
              {k: dci(v) for k, v in c["paired_wide_minus_narrow"].items()})

    print("\n### F. FOUR FAMILIES (headline arms)\n")
    for k in real:
        f = arms[k].get("four_families")
        if not f:
            continue
        lo, la, ta = f["LONGITUDINAL"], f["LATERAL"], f["TACTICAL"]
        acc = ta.get("from_long_accel_scalar", {})
        print(f"* **{k}** — LONG scalar-speed MAE {lo.get('scalar_speed_mae_mps')} "
              f"m/s, traj-speed MAE {lo['traj_speed_mae_mps']}, along MAE "
              f"{lo['along_mae_m']} m, distance-keeping {lo['distance_keeping']['status']}"
              f" | LAT heading {la['heading_mae_rad']} rad, curvature "
              f"{la['curvature_mae_inv_m']} 1/m, yaw-rate {la['yaw_rate_mae_rad_s']} "
              f"rad/s, cross-track {la['cross_track_mae_m']} m"
              f" | TACT lateral BA {ta['lateral']['balanced_accuracy']} (chance "
              f"{ta['lateral']['chance_balanced_accuracy']}), longitudinal BA "
              f"{ta['longitudinal']['balanced_accuracy']}, accel-scalar BA "
              f"{acc.get('balanced_accuracy')} recall {acc.get('recall')} "
              f"precision {acc.get('precision')} support {acc.get('support_gt')}"
              f" | STRAT {f['STRATEGIC']['status']}"
              f" | ADE {arms[k]['ade_2s']['mean']} m "
              f"[{arms[k]['ade_2s']['lo']}, {arms[k]['ade_2s']['hi']}]")


if __name__ == "__main__":
    main()
