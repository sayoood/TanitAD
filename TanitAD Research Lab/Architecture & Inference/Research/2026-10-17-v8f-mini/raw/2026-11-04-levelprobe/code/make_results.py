#!/usr/bin/env python3
"""make_results.py -- RESULTS.md from raw/LEVELPROBE.json (+ raw/STAGE1.json).  The card's outcome is printed as the reader's `card_outcome` gives it (NOT DEFINED while HLVL-PRED has not run: F1).
python -I code/make_results.py"""
import json
import sys

PK = "E:/Projects/TanitAD-artifacts/dagland/levelprobe"
R = json.load(open(f"{PK}/raw/LEVELPROBE.json", encoding="utf-8"))
S1 = json.load(open(f"{PK}/raw/STAGE1.json", encoding="utf-8"))
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


def mk(lab, r):
    sep = bool(r["lo"] > 0 or r["hi"] < 0)
    better = (r["delta"] > 0) if lab in HB else (r["delta"] < 0)
    return f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] " + ("ns" if not sep else ("SB" if better else "SW"))


E = "tier **T1-kin** / estimator **paired episode-cluster bootstrap, B 2,000, seed 0** (answers another draw of EPISODES only; X and Y share the checkpoint) / evidence class **MEASURED**"
co = R["card_outcome"]
CTX = R["context_histswap_lf"]


def rng(m):
    v = [x["share_closed"] for x in CTX[m].values()]
    return f"{min(v):.3f} to {max(v):.3f}"


L = ["# RESULTS: H-LEVELPROBE-1 (zero training)\n",
     f"Card `PREREG_LEVELPROBE.md` sha256 `{R['prereg_sha256']}`; reader `code/levelprobe_read.py`; raw `raw/LEVELPROBE.json`, `raw/STAGE1.json`. Modes read: {R['modes_read']}.\n",
     "## 0. The committed outcome\n",
     f"**Card outcome as the reader gives it: {co['outcome']}.** " + (co.get("reason", "") if co["outcome"] == "NOT DEFINED" else f"HLVL-PRED shares {co.get('pred_shares')}, mean {co.get('pred_mean_share_closed')}, cells >= 0.30: {co.get('n_cells_ge_0p30')}, ratio to HLVL-T0 mean {co.get('ratio_pred_over_t0_mean')}."),
     f"Stage 1 beside it (T0-class, v0 < 6 windows only, `STAGE1.md`): eval RMSE {S1['main']['eval']['all']['rmse']:.3f} m/s, R^2 {S1['main']['eval']['all']['r2']:.3f} (608 windows); on the 313 M1 windows with v0 < 6 RMSE {S1['main']['eval']['M1_windows']['rmse']:.3f}, R^2 {S1['main']['eval']['M1_windows']['r2']:.3f}.\n",
     f"## 1. HLVL-T0 (ORACLE, the logged speed at t0 held constant): {E}\n",
     f"HLVL-T0 mean share of the M1 excess closed **{co['t0_mean_share_closed']:.3f}** (cells {', '.join(f'{x:.3f}' for x in co['t0_shares'])}); M1 reading `{R['m1_reading']['HLT0']['reading']}` ({R['m1_reading']['HLT0']['n_cells_separated_better']} of 4 cells separated-better). "
     "Context (HISTSWAP-LF, per-tick logged speed, MEASURED in `histswap_lf/raw/HISTSWAP_LF_T1.json`): HLVL @@HLVL@@ per cell; HCV @@HCV@@. HLVL-T0 differs from HLVL by holding the t0 speed instead of re-reading the logged speed at each tick (PREREG B7).\n",
     "| cell | M1 X - Y | Y - P10 (the excess) | share closed | ADE6 X - Y (m) | outcome label |\n|---|---|---|---|---|---|"]
for k, a in R["arms"].items():
    if not k.endswith(":HLT0"):
        continue
    c = a["m1_cell"]
    arm = {"CTRL": "DAG-OACC", "SOUP": "SOUP"}[k.split("_")[0]]
    L.append(f"| {k.split(':')[0]} | {mk('M1 crawl share', a['rows']['X_minus_Y']['M1 crawl share'])} | {c['yp_delta']:+.4f} | {'n/a' if c['share_closed'] is None else format(c['share_closed'], '.4f')} | {mk('ADE6', a['rows']['X_minus_Y']['ADE6'])} | {R['outcome']['HLT0'][arm]['outcome']} |")
L.append("\n## 2. All 19 rows, X - Y per cell for HLVL-T0 (S-POS rows UNVERIFIED; STRATEGIC family PAUSED; HLVL-T0 absolute rows are not performance)\n")
cells = [k for k in R["arms"] if k.endswith(":HLT0")]
L.append("| row | family | " + " | ".join(c.split(":")[0] for c in cells) + " |\n|---|---|" + "---|" * len(cells))
for lab in R["arms"][cells[0]]["rows"]["X_minus_Y"]:
    L.append(f"| {lab} | {FAM.get(lab, '')} | " + " | ".join(mk(lab, R["arms"][c]["rows"]["X_minus_Y"][lab]) for c in cells) + " |")
L.append("\n## 3. Controls (computed before any row; a failure REFUSES the read)\n")
L.append("| control | PASS |\n|---|---|")
for k, v in R["controls"].items():
    L.append(f"| {k} | {v['PASS']} |")
L.append(f"\nHARMS flag (outcome label == HARMS, per base): {R['harms_flag']}. Outcome labels: {R['outcome']['HLT0']}.\n")
L.append("## 4. Scope\n* HLVL-T0 is an ORACLE (the human's speed at t0); its absolute rows are not performance. HLVL-PRED is not read here (F1; `AMEND_A1_PROPOSED.md`).\n* One checkpoint per cell; the 4 cells are the rig's replicates, not a variance estimate (`H-ESTIM-SEED-1`). T1-kin cameras follow the logged path.\n")
L.append("## 5. What the HLVL-T0 reading means (MEASURED numbers above; the inferences are labelled)\n"
         "* **MEASURED: the logged speed AT t0, held constant, does not close the M1 excess** (mean share closed %.3f, 0 of 4 cells separated-better, ADE6 X - Y over the +0.0604 m guard at all 4 cells). HISTSWAP-LF's HLVL (the logged speed re-read at every tick) closed @@HLVL@@ at 4 of 4 cells on the same cells and comparator; HCV (the executed speed held) read @@HCV@@. HLVL-T0 reads like HCV, not like HLVL.\n"
         "* **INFERENCE (not separately tested):** what carried HLVL's closure is not the speed level at t0 but the logged speed AFTER t0 (the human's speed profile over the rollout), which no t0 frame supplies. A level-from-image auxiliary that only reproduces the t0 level has, on this reading, nothing to gain here.\n"
         "* **HYPOTHESIS, unmeasured (F1 blocks it):** HLVL-PRED is expected not to close either. It is NOT bounded by HLVL-T0: the image can carry scene context that predicts speed beyond v(t0) (a green light, a clearing lead), so a prediction could differ from v(t0) in a helpful direction. Only the amended run can say.\n"
         "* **A hole in the card (reported, the rule is not changed):** SUPPORTED requires HLVL-PRED's mean closure >= 0.5 x HLVL-T0's mean closure. With HLVL-T0's mean at %.3f that threshold is %.3f, so the second clause is satisfied by any prediction above it; only the first clause (>= 0.30 at >= 3 of 4 cells) binds. If A1 is approved this clause should be re-stated before it is frozen.\n"
         "* **Next lever (Rule Zero, proposals for a card, none run):** (i) HLVL-FUT, an oracle holding the logged speed at t0 + 1 s (and t0 + 3 s) constant, to see how much of the profile is needed; (ii) HLVL-MEAN6, the mean logged speed over the 6 s horizon held constant. Both are zero-training, use the same tool path (a per-window constant via `--level-file`), and tell whether the closure is a short-horizon or a profile effect before any architecture decision.\n" % (co["t0_mean_share_closed"], co["t0_mean_share_closed"], co["t0_mean_share_closed"] * 0.5))
open(f"{PK}/RESULTS.md", "w", encoding="utf-8", newline="\n").write(("\n".join(L) + "\n").replace("@@HLVL@@", rng("HLVL")).replace("@@HCV@@", rng("HCV")))
print("wrote RESULTS.md")
