"""Tabulate paired_families_navsim outputs: delta [lo, hi] per component, * = separated (paired log-cluster bootstrap)."""
import glob, json, os, sys
d = sys.argv[1]; order = sys.argv[2].split(",")
M = [("ADE", "ade_m", "ADE m"), ("longitudinal", "LON_speed_mae_mps", "speed MAE m/s"), ("longitudinal", "LON_along_mae_m", "along MAE m"),
     ("lateral", "LAT_cross_mae_m", "cross MAE m"), ("lateral", "LAT_heading_mae_deg", "heading MAE deg"),
     ("lateral", "LAT_yaw_rate_mae_radps", "yaw-rate MAE rad/s (unmasked)"), ("lateral", "LAT_yaw_rate_mae_masked_radps", "yaw-rate MAE rad/s (pair-masked)"), ("lateral", "LAT_curvature_mae_1pm", "curvature MAE 1/m"), ("tactical", "TAC_traj_lat_correct", "TAC lat agree"),
     ("tactical", "TAC_traj_lon_correct", "TAC lon agree")]
rows = ["| comparison (B - A) | n / logs | " + " | ".join(x[2] for x in M) + " |", "|---|---|" + "---|" * len(M)]
out = {}
for stem in order:
    r = json.load(open(os.path.join(d, stem + ".json"), encoding="utf-8"))
    assert r["controls"]["all_pass"], stem
    cells = []
    out[stem] = {"direction": r["direction"], "n": r["n_tokens"], "logs": r["n_log_clusters"], "cells": {}}
    for fam, m, _ in M:
        x = r["paired"]["families"][fam][m]
        out[stem]["cells"][m] = {k: x[k] for k in ("delta", "lo", "hi", "separated")}
        cells.append(f"{x['delta']:+.4f}{'*' if x['separated'] else ''}")
    rows.append(f"| {r['direction']} | {r['n_tokens']} / {r['n_log_clusters']} | " + " | ".join(cells) + " |")
print("\n".join(rows))
json.dump(out, open(os.path.join(d, "FAMILIES_TABLE.json"), "w"), indent=1)
