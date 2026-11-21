#!/usr/bin/env python3
"""make_results2.py -- RESULTS2.md from raw/LEVELPROBE2.json (H-LEVELPROBE-2).  The committed outcome is printed as the reader's `card_outcome` gives it.  python -I code/make_results2.py"""
import json

PK = "E:/Projects/TanitAD-artifacts/dagland/levelprobe"
R = json.load(open(f"{PK}/raw/LEVELPROBE2.json", encoding="utf-8"))
HB = ("TAC-LAT", "TAC-LON")
FAM = {"ADE6": "PRIMARY", "M1 crawl share": "crawl"}
for q in ("speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG"):
    FAM[q] = "LONGITUDINAL"
for q in ("heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)"):
    FAM[q] = "LATERAL"
for q in HB:
    FAM[q] = "TACTICAL"
for q in ("S-POS@30", "S-POS@50", "S-POS@80"):
    FAM[q] = "STRATEGIC (PAUSED, ITEM 25; S-POS UNVERIFIED)"
ARMS = {"MEAN6": "mean logged speed over (t0, t0 + 6 s]", "FUT3": "logged speed at t0 + 3 s", "FUT1": "logged speed at t0 + 1 s"}
E = "tier **T1-kin** / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** (answers another draw of EPISODES only; X and Y share the checkpoint) / evidence class **MEASURED**"


def mk(lab, r):
    sep = bool(r["lo"] > 0 or r["hi"] < 0)
    better = (r["delta"] > 0) if lab in HB else (r["delta"] < 0)
    return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] " + ("ns" if not sep else ("SB" if better else "SW"))


co, CTX = R["card_outcome"], R["context_histswap_lf"]
rng = lambda m: "%.3f to %.3f" % (min(x["share_closed"] for x in CTX[m].values()), max(x["share_closed"] for x in CTX[m].values()))  # noqa: E731
t0 = R["context_hlvl_t0_from_levelprobe1"]
L = ["# RESULTS2: H-LEVELPROBE-2 (zero training; ORACLE diagnostics)\n",
     f"Card `PREREG_LEVELPROBE2.md` sha256 `{R['prereg_sha256']}` (frozen before any level file and any rollout); reader `code/levelprobe2_read.py`; raw `raw/LEVELPROBE2.json`; level files `raw/levels_meta.json`. Every number: {E}.\n",
     f"## 0. The committed outcome: **{co['outcome']}**\n",
     "| arm | level | mean share of the M1 excess closed | per-cell shares | cells >= 0.30 | M1 reading |\n|---|---|---|---|---|---|"]
for m, d in ARMS.items():
    L.append(f"| {m} | {d} | {co['mean_share_closed'][m]:.3f} | {', '.join(f'{x:.3f}' for x in co['shares'][m])} | {co['n_cells_ge_0p30'][m]} of 4 | {R['m1_reading'][m]['reading']} ({R['m1_reading'][m]['n_cells_separated_better']} of 4 cells separated-better) |")
L.append(f"\nHorizon pattern (mean share closed): FUT1 {co['horizon_pattern_mean_share']['FUT1']:.3f}, FUT3 {co['horizon_pattern_mean_share']['FUT3']:.3f}, MEAN6 {co['horizon_pattern_mean_share']['MEAN6']:.3f}. TARGET-SPEED SUPPORTED: {co['target_speed_supported']}; PROFILE: {co['profile']}.")
L.append(f"Reference rows read from files: HISTSWAP-LF HLVL (per-tick logged speed) {rng('HLVL')} per cell, HCV {rng('HCV')} (`histswap_lf/raw/HISTSWAP_LF_T1.json`); HLVL-T0 (logged speed at t0) mean {t0['mean_share_closed']:.3f} (`raw/LEVELPROBE.json`).\n")
L.append("## 1. Per arm and cell: M1 X - Y, the excess, ADE6 X - Y, HARMS flag, outcome label\n")
L.append("| arm | cell | M1 X - Y | Y - P10 (the excess) | share closed | ADE6 X - Y (m) | outcome label |\n|---|---|---|---|---|---|---|")
for m in ARMS:
    for k, a in R["arms"].items():
        if k.endswith(":" + m):
            c, arm = a["m1_cell"], {"CTRL": "DAG-OACC", "SOUP": "SOUP"}[k.split("_")[0]]
            L.append(f"| {m} | {k.split(':')[0]} | {mk('M1 crawl share', a['rows']['X_minus_Y']['M1 crawl share'])} | {c['yp_delta']:+.4f} | {'n/a' if c['share_closed'] is None else format(c['share_closed'], '.4f')} | {mk('ADE6', a['rows']['X_minus_Y']['ADE6'])} | {R['outcome'][m][arm]['outcome']} |")
L.append(f"\nHARMS flag per arm and base (outcome label == HARMS): {R['harms_flag']}.\n")
L.append("## 2. All 19 rows, X - Y per cell, by family (S-POS rows UNVERIFIED; STRATEGIC family PAUSED, ITEM 25; oracle absolute rows are not performance)\n")
for m in ARMS:
    cells = [k for k in R["arms"] if k.endswith(":" + m)]
    L.append(f"### {m}\n\n| row | family | " + " | ".join(c.split(":")[0] for c in cells) + " |\n|---|---|" + "---|" * len(cells))
    for lab in R["arms"][cells[0]]["rows"]["X_minus_Y"]:
        L.append(f"| {lab} | {FAM.get(lab, '')} | " + " | ".join(mk(lab, R["arms"][c]["rows"]["X_minus_Y"][lab]) for c in cells) + " |")
    L.append("")
L.append("## 3. Controls (computed before any row; a failure refuses the read)\n\n| control | PASS |\n|---|---|")
for k, v in R["controls"].items():
    L.append(f"| {k} | {v['PASS']} |")
L.append("\n## 4. Scope\n* Oracles are diagnostics, not deployable; the arms hold a per-window constant level computed from the logged speed after t0.\n* One checkpoint per cell; the 4 cells are the rig's replicates, not a variance estimate (`H-ESTIM-SEED-1`). T1-kin cameras follow the logged path.\n"
         "* A SUPPORTED outcome opens a separately pre-registered predicted mean-speed / goal input (vision + nav, not derived from the situation classifier); a PROFILE outcome closes the scalar target-speed family for M1.\n")
B = chr(92)
open(f"{PK}/RESULTS2.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
print("wrote RESULTS2.md")
