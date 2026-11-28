"""Report v32: SPEEDREAD (7.24, NOT IN INPUT), DAGGER2 (7.25, HARMS) and DAG-ORC (7.26, HARMS, arm A); the abstract, 8.1, 8.3, 8.4 rows 1w-1y; the version line.
Builds the v32 .md and .html from the v31 blobs of one repository tip. Nothing is read from a working tree.

python patch_v32.py [tip [out_dir]]
  tip       repository tip whose blobs are the inputs (default ea5e3d3b, the tip that carries RESULT_V8F_DAG_ORC.md); its first 8 hex digits go to the status line
  out_dir   folder to write V8F_SCIENTIFIC_REPORT.md and v8f_report.html into (default: out/ beside this script)

Inputs (all `git show <tip>:<path>` from the mirror, never the mirror's HEAD):
  the v31 V8F_SCIENTIFIC_REPORT.md, v8f_report.html and page_template_v30.html; RESULT_V8F_SPEEDREAD.md, RESULT_V8F_DAGGER2.md and RESULT_V8F_DAG_ORC.md
  (their tables are COPIED LINE FOR LINE by anchored extraction, each line asserted to be a line of the source);
  PREREG_V8F_SPEEDREAD.md, PREREG_V8F_DAGGER2.md, PREREG_V8F_DAG_ORC.md, R2A_NOTE.md and R2C_NOTE.md (every claim taken from them is asserted
  to be present in the source text, see PHRASES below), and the register row H-V8F-DAG-ORC-1 of GOALS_AND_CLAIMS.md (asserted to read HARMS).
Every insertion is anchored, and its anchor is asserted to occur exactly once. The .html is produced twice, by the same anchored edits
applied to the v31 .html and by the template with the new .md, and the two must be equal.
Section 7.26 carries the DAG-ORC read (outcome HARMS, arm A); the patch asserts that no DAG-ORC placeholder is left in either file."""
import os
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TIP = (sys.argv[1] if len(sys.argv) > 1 else "ea5e3d3b")
TP = TIP[:8]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = (sys.argv[2] if len(sys.argv) > 2 else HERE + "/out").rstrip("/\\") + "/"
MIRROR = "C:/Users/Admin/tanitad-push"
R = "TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/"
RP = R + "report/2026-10-26/"


def blob(path, strict=True):
    r = subprocess.run(["git", "-C", MIRROR, "-c", "safe.directory=*", "show", f"{TIP}:{path}"], capture_output=True)
    assert r.returncode == 0 and len(r.stdout) > 0, (path, r.stderr[:200])
    t = r.stdout.decode("utf-8")
    if strict:
        assert "\r" not in t and "\ufeff" not in t, path
    return t.replace("\r\n", "\n")


s = blob(RP + "V8F_SCIENTIFIC_REPORT.md")
h = blob(RP + "v8f_report.html")
tpl = blob(RP + "page_template_v30.html")
assert tpl.count("%%MD%%") == 1 and h == tpl.replace("%%MD%%", s), "v31 html is not the template with the v31 md"
assert s.count("**Status:** research report, v31 of 2026-10-27") == 1

SR = blob(R + "RESULT_V8F_SPEEDREAD.md")
D2 = blob(R + "RESULT_V8F_DAGGER2.md")
PSR = blob("Project Steering/PREREG_V8F_SPEEDREAD.md")
PD2 = blob("Project Steering/PREREG_V8F_DAGGER2.md")
POR = blob("Project Steering/PREREG_V8F_DAG_ORC.md")
R2A = blob(R + "raw/2026-10-28-dag-orc-r2a/R2A_NOTE.md")
R1S = blob(R + "raw/2026-10-28-dag-orc-a2/r1_refused/M2b_s0_dagoaccorc/summary.json", strict=False)  # a JSON from box 2 (CRLF there)
OR = blob(R + "RESULT_V8F_DAG_ORC.md")
R2C = blob(R + "raw/2026-10-28-dag-orc-result/notes/R2C_NOTE.md")
REG = blob("Project Steering/GOALS_AND_CLAIMS.md")
PSS = blob("Project Steering/PREREG_V8F_SPEEDSUB.md")  # landed at 94a47f4b, after the SPEEDREAD result; read only to keep 7.24 from saying "needs its own PREREG"


# ---- tables, copied line for line -------------------------------------------------------------------------------------------
def grab(text, anchor, nrows):
    assert text.count(anchor) == 1, (text.count(anchor), anchor[:70])
    lines = text[text.index(anchor):].split("\n")
    first = next(k for k, l in enumerate(lines) if k > 0 and l.startswith("|"))
    assert first <= 3, (anchor[:60], first)
    rows = []
    for l in lines[first:]:
        if not l.startswith("|"):
            break
        rows.append(l)
    assert len(rows) == nrows, (anchor[:60], len(rows), nrows)
    return rows


def ncols(line):
    body = re.sub(r"`[^`\n]*`", "x", line.replace("\\|", "x"))
    return body.count("|") - 1


TABLES = {
    "SR_OUT": grab(SR, "## 1. The outcome, as the pre-registered table decides it", 11),
    "SR_COND": grab(SR, "The pre-registered conditions (PREREG §5, first match wins), restated against the rows.", 6),
    "SR_CTRL": grab(SR, "## 2. Controls: each must read its known value before any row is read", 13),
    "SR_FOLD0": grab(SR, "Fold-0 hold-out RMSE, train-SD units", 6),
    "SR_RO": grab(SR, "### read − own (the PRIMARY contrast; point [95 % CI])", 33),
    "SR_GT": grab(SR, "### gt − own (T0 ceiling, a control only; point [95 % CI])", 33),
    "SR_CELLS": grab(SR, "Per-cell eval RMSE to the GT, m/s, on the eval-141 windows where the GT cell is present", 7),
    "D2_OUT": grab(D2, "## 1. The outcome, as the table decides it", 9),
    "D2_FAM": grab(D2, "## 2. The four families (T1-kin; point [95 % CI]; B / W = separated-better / -worse)", 19),
    "OR_OUT": grab(OR, "## 1. The outcome, as the table decides it", 12),
    "OR_FLOOR": grab(OR, "(`RF_REP.json`, written before any A contrast).", 21),
    "OR_AC": grab(OR, "### 2.2 A − CTRL (the one-lever contrast; = A − DAG-OACC) and REP − CTRL (the replicate)", 21),
    "OR_CRAWL": grab(OR, "R-MOVE: A − CTRL separated-better AND the improvement (−delta) ≥ bar. R-CRAWL: A − P10 not separated-worse.", 10),
    "OR_NEW": grab(OR, "R-NOBASE: (1) and (3).", 4),
    "OR_FAM": grab(OR, "Source `four_families.A.<seed>.<family>.rows` (each family block carries this same tier and estimator string", 20),
    "OR_SHARE": grab(OR, "### (b) The share of the CTRL − P10 crawl gap that A closes", 10),
    "OR_REPS": grab(OR, "The A − CTRL effects, set beside the floor they had to clear (size = |A − CTRL|; the bar is 2 × RF on all four crawl rows):", 7),
    "OR_LONG": grab(OR, "### (d) The longitudinal cost at s0 (A − CTRL)", 5),
    "OR_FACT": grab(OR, "1. **What B must clear.**", 6),
    "OR_LAUNCH": grab(OR, "**Launches and stops, from the box-2 chain logs (first to last timestamped line, minute resolution).**", 7),
    "OR_FINAL": grab(OR, "> `10-28 07:44 summary: M2b_s0_dagoaccorc=380b230d0603", 6),
}
for name, rows in TABLES.items():
    src = SR if name.startswith("SR") else D2 if name.startswith("D2") else OR
    assert all(r in src.split("\n") for r in rows), name
    assert len({ncols(r) for r in rows}) == 1, (name, sorted({ncols(r) for r in rows}))

# ---- claims taken from the sources, asserted present --------------------------------------------------------------------------
PHRASES = [
    (SR, "Outcome **NOT IN INPUT**"), (SR, "all **19 control flags PASS**"), (SR, "46543c392ee412c76edab48ec95c993e11ba621cc43188ee8e25f7b966d4579f"),
    (SR, "T1-class open loop (X_s's own held S / T outputs; only the five speed cells O reads are replaced)"),
    (SR, "T0 (reads the eval GT): ceiling / control only, never a result"), (SR, "T0-open-loop-own-held-outputs (trunk-probe grid)"),
    (SR, "B=2000, seed 0"), (SR, "141 episodes / 2,399 windows per seed"), (SR, "`readout_sha256` `a205609a…` in all 14 dumps"),
    (SR, "Checkpoint sha256 s0 `c6a62876…`, s1 `68a0f335…`"), (SR, "F = 0.1207 m is the cross-seed spread of M2b (INHERITED, `RESULT_V8F_GRID.md`)"),
    (SR, "perception stays the lever: PREREG_V8F_LORA on the A40 (D-V8F-SPEED-PERCEPTION, the PI's go); nothing on box 2 on this line"),
    (SR, "the full 40-token attention view T itself has; T fails on that view too (SPEEDCAL, INHERITED)"),
    (SR, "6 of 49 cells separated (read − own)"), (SR, "8 of 49 cells separated (read − own)"),
    (SR, "LORA projects 494 h per seed on box 2 against its venue rule of ≤ 36 h per seed (INHERITED, LORA Amendment 3)"),
    (SR, "about 100+ GPU-h per seed on the A40 (ESTIMATED, LORA Amendment 3)"),
    (SR, "PI: box 2 and dev-box CPU only; INHERITED from the Master Mind's brief of 2026-10-28"),
    (SR, "DAG-ORC, running on box 2 since 2026-10-28 01:16"),
    (SR, "**Its ADE6 substitution was NOT run**"), (SR, "ZZSPEEDREAD-PULL-DONE"), (SR, "no temperature check"),
    (SR, "extract 23:21:49–00:17:14; fit 00:17:14–00:54:18; evals_s0 00:56:19–01:03:22; evals_s1 01:03:22–01:09:19; read 01:09:19–01:09:25"),
    (SR, "Self-test 27 passed and 19 / 19 mutants killed (INHERITED, PREREG Amendment 1 §3, `raw/2026-10-27-speedread-a1/`)"),
    (SR, "118,851,106 B"), (SR, "94,730,100 B"), (SR, "8,789,860,480 B"),
    (SR, "4 s after the fit stage finished"), (SR, "the 12 files now there are byte-identical"),
    (SR, "max abs delta over all cells 8.0e-07 (bar: at most 1e-6)"), (SR, "max abs delta over all cells 7.0e-07 (bar: at most 1e-6)"),
    (SR, "true (max abs dev 5.0e-05 over 42 checks; bar 1e-4)"), (SR, "`perm_holdout_skill_max` 0.0069 (≤ 0.02)"),
    (PSR, "(p10f 8, nav 5, limit 4, limit_ok 1, frame_ok 4)"), (PSR, "`d5545751…` to **`fd5ce0be"), (PSR, "Promoted 2026-10-27 by the Master Mind"),
    (PSR, "randomised PCA with seed 20261027"), (PSR, "PCA 512"),
    (PSR, "Nothing was summarised, fitted, selected or printed, and no target value was looked at."),
    (D2, "Outcome **HARMS**"), (D2, "all **199 reader controls PASS** (23 control groups in `controls_PASS`)"), (D2, "056ecba5abf029966ad96e367b4051767449d0e812014ca3e827d0676c959cf9"),
    (D2, "Banks: s0 `90c43c66…`, s1 `a06c8289…`, COVERAGE\n  0.773 / 0.764"), (D2, "0.10085 (DAG2 s0), 0.10062 (DAG2 s1)"),
    (D2, "Checkpoints `0d0a800b…`, `603b9151…`,\n  `2e03b141…`, `f4234ac1…`"), (D2, "+0.0426 [+0.0011, +0.0896] m at s0 (separated-worse) and −0.0309 [−0.085, +0.026] at s1"),
    (D2, "DAG2-0 − P10 ADE6 is −0.898 /\n   −0.855 m (points, by additivity of paired means), against DAG's −1.166 / −0.987"),
    (D2, "8 / 8 rollouts plus 4 fresh-T0 evals, 15:19–15:56"), (D2, "199 / 199 checks, 68 / 68 mutants killed"), (D2, "`ZZDAG2T1-DRY-OK`"),
    (D2, "(1.3 GB)"),
    (PD2, "Promoted 2026-10-27 by the Master Mind from the draft banked at `5eaa0232`"), (PD2, "`A1-2026-10-27`"),
    (PD2, "**4 cells match no row**"), (PD2, "FIXED takes 1 cell;\n* PARTIAL takes 7;\n* NO EFFECT takes 4;"),
    (PD2, "R-T0 point ≤ +F/2 = 0.0604 m"),
    (PD2, "* s0: curvature MAE, L2, M1;\n     * s1: curvature MAE, L2, M1, yaw-rate MAE."),
    (POR, "Claim row `H-V8F-DAG-ORC-1`"), (POR, "sha256 `1608d70b…`"), (POR, "**F = 0.1207 m**"), (POR, "**2 × max(F_r, RF0_r, RF_r)**"),
    (POR, "There are 524,288 cells"), (POR, "| REFUSED | 262,144 |"), (POR, "| HARMS | 261,888 |"), (POR, "| FIXED / ADOPTED | 1 |"),
    (POR, "| FIXED / NOT ADOPTED | 15 |"), (POR, "| PARTIAL / ADOPTED | 3 |"), (POR, "| PARTIAL / NOT ADOPTED | 45 |"),
    (POR, "| PARTIAL / ONE SEED | 128 |"), (POR, "| NO EFFECT | 64 |"), (POR, "13 / 13 mutants exit 1"),
    (POR, "Written 2026-10-28 ~02:30 by the Master Mind."), (POR, "19,161 / 76,898 = **0.2492**"), (POR, "(`ZZDAGOACCORC-CONTROL-FAIL`)"),
    (POR, "`(seed × 1,000,003 + step) mod 2^62`"), (POR, "0.2 / 0.9 = 0.222"), (POR, "`EGO_PERTURB_COMPOSED_STREAM = 1`"),
    (POR, "0.0000 at 0.5 / 0.5 and 0.221 / 0.224 at 0.1 / 0.3"), (POR, "1,561 / 6,345 = 0.2460"), (POR, "1,860 / 6,345 = 0.2931, swap 720 / 7,380"),
    (POR, "**27 / 27 killed**"), (POR, "66 / 66 tests, 223 / 223 mutants killed"), (POR, "reader (`2e5f9f17…`)"),
    (POR, "r1 used box 2 from 01:16 to 02:10 (0.9 h)"), (POR, "about 3.7 h in all"), (POR, "N0 already read NOT NAVMOVE."),
    (POR, "r1 (`chain_v8f_dagoaccorc.sh` at `e7ea20d1`, started 01:16"),
    (POR, "Checkpoints `67265ec1…` (s0) / `90b01474…` (s1)"), (POR, "`--ego-perturb-p 0.3 --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed <s>`"),
    (POR, "`--ego-perturb-with-dagger`"), (POR, "A + REP is 4 runs, ≈ 2.8 h"),
    (POR, "knots 1–12 at m = 10"), (POR, "v0 < 6 m/s and GT v(6 s) ≥ v0 + 1 m/s"), (POR, "+0.1103 / +0.0849 m, above the +F/2 bar at both seeds"),
    (R2A, "r2_aside_done.txt"), (R2A, "written 2026-10-27 20:21"), (R2A, "(`6ea4d6baf3f806b3…`"), (R2A, "r2_aside_done_dagoaccorc.txt"),
    (R2A, "Box 2 03:27–03:40 (0.2 h). The total is now about 3.9 h"), (R2A, "ASIDE-ALREADY-DONE"), (R2A, "`ZZV8FMINI-STOP"),
    (R2A, "*Two chains that write the same marker into one directory share one marker.*"),
    (R1S, '"ego_perturb_n": 19161'), (R1S, '"ego_perturb_eligible": 76898'), (R1S, '"ego_dagger_overlap_n": 0'),
    # RESULT_V8F_DAG_ORC.md
    (OR, "Outcome **HARMS** (arm A, p = 0.3, not adopted)"), (OR, "all **162 reader controls PASS**"), (OR, "Read 2026-10-28 07:58 by the pre-registered reader"),
    (OR, "started 07:54:18, rc 0 at 07:58:54, 275.4 s"), (OR, "sha256 `2e5f9f17…`"), (OR, "(sha256 `70a7a3a4…`)"),
    (OR, "(sha256 `64c9444d…`, written before any A contrast was computed)"),
    (OR, "four gates PASS, four run records PASS, four K-REACH PASS"), (OR, "**Arm B (p = 0.1) was not trained and is not read.**"),
    (OR, "A `380b230d…` (s0) / `870dded7…` (s1); REP `b03fd6ef…` / `77853e56…`; CTRL `67265ec1…` / `90b01474…`"),
    (OR, "T1-kin (self-action OPEN loop; m=10, causal bucket, dead-reckoned nav, logged scene, periodic, executor plan; plain arms)"),
    (OR, "T0 (teacher-forced own-code, eval-141, ages cycle: box 2's standard eval)"),
    (OR, "`taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed, paired, on the same windows and matched cells"),
    (OR, "so the training-run question is carried by the two seeds and the replicate floors RF0 / RF (`H-ESTIM-SEED-1`)"),
    (OR, "The planner is deterministic, so no inference-seed question arises."),
    (OR, "A_s = DAG-OACC_s's training argv verbatim plus the six perturbation flags at the registered law (`--ego-perturb-p 0.3`) and `--ego-perturb-with-dagger`"),
    (OR, "REP_s = DAG-OACC_s's argv with `--seed 10` / `--seed 11` (same init M2b_s). CTRL = DAG-OACC_s, banked (no control was retrained)"),
    (OR, "R-ADE and R-KEEP read one contrast, A − DAG-OACC (Amendment 1 section 2: DAG = CTRL = DAG-OACC)"),
    (OR, "*\"controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at ≥ 1 seed\"*"),
    (OR, "Three guard cells fail: R-ADE at s0, and R-T0 at s0 and at s1."),
    (OR, "REFUSED is decided first and HARMS second (PREREG section 7, *Precedence*), so the R-MOVE flags do not change the word."),
    (OR, "*\"A: not adopted; read B (p 0.1) under this same table (B is NOT queued: it needs its own PI go, A1 section 4).\"*"),
    (OR, "R-MOVE is false at both seeds, with 0 of 4 crawl rows clearing the bar at either seed (section 2.4). R-CRAWL is false at both seeds."),
    (OR, "the table does not test whether the cost is the lever's (section 4 (c))"),
    (OR, "Source `four_families.A.<seed>.<family>.rows` (each family block carries this same tier and estimator string, and its leaves equal `read_a` cell for cell"),
    (OR, "**A − DAG is identical to A − CTRL, cell for cell**"),
    (OR, "P10 is the kinematic floor. Row units are the reader's own."),
    (OR, "the JSON labels its `n` *\"ticks>=1, unpaired vs CTRV (DAG_READ.json)\"*: those cells are not paired across runs"),
    (OR, "the A − CTRL leaf is the difference of the two runs' (S − CTRV, ticks ≥ 1) contrasts on the same windows"),
    (OR, "*\"NOT paused for v8F (PI ruling 2026-10-17)\"*"),
    (OR, "The T-head macro-F1 values behind R-TLAT / R-TLON are not carried in the JSON, only the booleans (UNVERIFIED as numbers)."),
    (OR, "`t0_four_families.status` reads **NOT RUN**."),
    (OR, "*\"DAGGER2 left the T0 families NOT RUN. Here they are a build item of the reader\"*"), (OR, "the reader did not carry it"),
    (OR, "orc_families.py pointed at A / CTRL (section 5)"),
    (OR, "The only T0 numbers in this read are the R-T0 guard points, T0 own-code ADE6 A − CTRL +0.1106 m (s0) and +0.0722 m (s1), with no interval in the JSON."),
    (OR, "Inputs exist: the pulled T0 evals (`eval_M2b-dagoaccorc_s*.json` / `.npz`) of A and of CTRL, on E:."),
    (OR, "**Nothing is claimed for T0 LONGITUDINAL, LATERAL or TACTICAL here.**"),
    (OR, "`M-LAUNCH`: X - CTRL of the mean (v_own - v_GT) at ticks 1-2 in the GT-launch windows, from the tick dumps: NOT BUILT"),
    (OR, "`anti_echo_block`: longitudinal_claim_admissible, hold-v0, copy detector (RESULT_V8F_ORECOVER 3.1): NOT BUILT"),
    (OR, "`M2_and_L2_launch_subset`: NOT BUILT"),
    (OR, "Not computable by design (PREREG section 5): T1-kin goal-head F1 and head accuracy (the paired tool reads eval `run` dumps, not tick dumps); anchor selection (one path per window)."),
    (OR, "The REP pair sets the bar on 15 of 19 rows (every row of C among them); RF0 sets it on 4 rows. REP s0 sets RF on 17 of 19 rows; REP s1 sets it on cross-track MAE and S-POS@80."),
    (OR, "REP differs from CTRL in the trainer seed alone. Even so, REP − CTRL reads separated on 15 of 19 rows at s0 and 2 of 19 at s1."),
    (OR, "`bar_r = 2 × max(F_r, RF0_r, RF_r)`, one value per row, applied at both seeds, inclusive. F is 0.1207 m on ADE6 and 0 elsewhere."),
    (OR, "Every one of the eight cells is separated-better against CTRL, and every one is below its bar: the largest is 33.8 % of the bar (M1 at s0) and the smallest 8.8 % (yaw-rate at s0)."),
    (OR, "At s0 the yaw-rate row ends not separated-worse against P10; the other 7 rows stay separated-worse."),
    (OR, "A closes 27.6 % to 79.9 % of the gap that DAG-OACC leaves to P10 on the crawl rows (largest: yaw-rate at s0; smallest: M1 at s1)."),
    (OR, "The two forms differ by at most 0.9 points, because the CTRL − P10 and A − P10 cells differ by one window on 6 of the 8 rows (the `n` fields)."),
    (OR, "(i) On the four crawl rows A − CTRL is 17.5 % to 67.5 % of RF, so each is smaller than the REP s0 draw on the same row."),
    (OR, "(ii) REP s0 − CTRL on ADE6 is +0.4025 m, 3.33 × F (0.1207 m); the s0 cost A − CTRL on ADE6 that failed R-ADE, +0.1434 m, is 36 % of that replicate draw. REP s1 − CTRL on ADE6 is +0.0116 m."),
    (OR, "The bar is one value per row from the larger draw by registration (PREREG section 3, *Two choices*), so a small s1 draw does not lower it."),
    (OR, "The closest approach is DK-VIOL at s0: A − CTRL +0.0185 is separated-worse and above its bar (0.0100)"),
    (OR, "R-NEW does not fire there because A − P10 on DK-VIOL is not separated-worse (+0.0202 [−0.0080, +0.0536] –)."),
    (OR, "At s0 all three are separated-worse against CTRL; at s1 none is."),
    (OR, "ADE6 +0.1434 is below its RF (0.4025) and its bar (0.8050); along-track +0.1188 is below its RF (0.4596)"),
    (OR, "DK-VIOL +0.0185 is above its RF (0.0017) and above its bar (0.0100, which RF0 sets), and R-NEW did not fire on it (section 2.5)."),
    (OR, "At s1 the longitudinal family carries one separated-worse row, accel MAE (**+0.1064 [+0.0031, +0.2287] W**)."),
    (OR, "**One training run per arm and seed.** The two seeds and the REP pair carry the training-run question; no row here is an effect of the lever on its own (`H-ESTIM-SEED-1`)."),
    (OR, "**No lateral / longitudinal split.** The perturbation has two components and neither half is credited."),
    (OR, "**B is not read.** Nothing here says what p = 0.1 does."),
    (OR, "K-PAIR (reader control) PASS at both seeds: argv(REP_s) equals argv(DAG-OACC_s) except `--out` and `--seed` (10 vs 0, 11 vs 1)."),
    (OR, "So REP − CTRL is a measured draw of **training variance** on this rig (`H-ESTIM-SEED-1`), the one thing the episode-cluster bootstrap does not see."),
    (OR, "The same gate tokens (same commit, same argv sha256) were written in the second launch (r2b) and the final one (r2c)."),
    (OR, "The K-REACH bands are 0.3 ± 0.02 for A, exactly 0 for REP, and 0.1 ± 0.02 for the swap; the overlap of swapped and perturbed samples is exactly 0 on A. All four fall inside."),
    (OR, "37 changed hunks in 2 files (`v8f_mini_train.py` 25, `v8f_mini_data.py` 12), every one behind `--ego-perturb-with-dagger`, default off, bit-identity test PASS; `v8f_mini.py` and `v8f_dagger.py` byte-identical."),
    (OR, "Record: `kbuild/kbuild_r2_launch_8c24823f.json` (sha256 `b1c5c36b…`)."),
    (OR, "`10-28 07:44 summary: M2b_s0_dagoaccorc=380b230d0603 M2b_s0_dagoaccorcrep=b03fd6efebec M2b_s1_dagoaccorc=870dded74c71 M2b_s1_dagoaccorcrep=77853e564f52`"),
    (OR, "checkpoint sha256 from `SHA256SUMS.txt`, verified by `sha256sum -c`; the T1-kin rollout done-markers carry the same sha256"),
    (OR, "Summed over the four launches the logs span **296 min = 4.93 h** (53 + 12 + 136 + 95), against the **~3 h** the PI's go was asked as (Amendment 1 section 4 planned 2.8 h)"),
    (OR, "against the **about 4.8 h** that `R2C_NOTE.md` states"), (OR, "`R2A_NOTE.md` 3.9 h after the first r2 launch, plus the 0.2 h and the 0.7 h that `R2C_NOTE.md` adds"),
    (OR, "The overrun beyond the ~3 h is 1.9 h by the logs. The four kept trainings account for 2.52 h of it (A s0 2286 s, REP s0 2240 s, A s1 2293 s, REP s1 2237 s; `summary.json`)."),
    (OR, "The rest is the refused r1 A s0 training, the r2 first launch, the refused A s1 training and the gates, evals and pulls of each launch."),
    (OR, "A s0 and REP s0 were skipped by the queue (already trained, jobs ended 04:35 and 05:17 in r2b, checkpoint sha256 re-verified); A s1 and REP s1 trained 06:21–07:00 and 07:03–07:42"),
    (OR, "pulling 805,306,368 B would leave less than 1,073,741,824 B (free 1,858,076,672 B)"),
    (OR, "two aside dirs and two dead lander holds (414 MB) were moved to F:"),
    (OR, "Pinned to `8c24823f` through `OAF_BR=agent/dagorc-r2-pin` (`run_chain_v8f_dagoaccorc_r2c_pin.cmd`;"),
    (OR, "*\"summary.json init_ckpt not applied / read back: False / None\"*"),
    (OR, "a second `train M2b_s1_dagoaccorc` at 05:59:36"),
    (OR, "*\"A: not adopted; read B (p 0.1) under this same table (B is NOT queued: it needs its own PI go, A1 section 4).\"* `cascade.b_status`: *\"PENDING: train B (p = 0.1) through its gate and read it under this same table\"*."),
    (OR, "ESTIMATED ≈ 1.4 h of box 2: A's two trainings took 2286 s and 2293 s (`summary.json`), plus about 4 min per run for gate and eval (log minutes). The T1-kin rollouts and the read are dev-box CPU."),
    (OR, "**The Master Mind's ranking, beside the table (arithmetic, not a forecast).**"),
    (OR, "the bar divided by A's improvement is the factor B's improvement would have to exceed A's by:"),
    (OR, "B's dose is a third of A's (p = 0.1 against 0.3). **So B cannot reach R-MOVE unless its crawl effect exceeds A's by 3.0–11.4 ×,** at a third of the dose."),
    (OR, "(PREREG section 7 also says B's claim *\"carries B's own floors and seeds\"*; if the reader recomputes floors for B, the factors change)"),
    (OR, "HARMS of B, NO EFFECT and PARTIAL / ONE SEED all route to NEXT-CRAWL, which on DAG-OACC's states is LAUNCH-INTENT (N0 read NOT NAVMOVE; COMPOSE is not met, since DAG-OACC − DAG is not separated-better on the rows of C, `RESULT_V8F_DAG_OACC.md` section 3)."),
    (OR, "Only R-MOVE at both seeds routes elsewhere. So B changes the route only if it clears the factors above."),
    (OR, "(draft: `dagland/launch_intent/PREREG_V8F_LAUNCH_INTENT.md`, on E:)"),
    (OR, "Its draft reuses this REP pair, and therefore these floors"),
    (OR, "this read shows the REP run records PASS (swap share 0.1013 / 0.0999, perturbation exactly 0, composition flag absent, stream 0). Its other conditions were not checked here (UNVERIFIED)."),
    (OR, "If R-MOVE fails while M-LAUNCH moves, the speed deficit was repaired and the crawl rows did not follow. That outcome points away from the state channel."),
    (OR, "**M-LAUNCH** (PREREG section 4)"), (OR, "(`F:/TanitAD-scratch/dagorc_t1_out/ticks/`)"),
    (OR, "**Single copy.** The per-episode dumps and tick arrays are on F: scratch only (`F:/TanitAD-scratch/dagorc_t1_out/`); the checkpoints and T0 evals are on E:. None is in the repo."),
    (OR, "the PREREG sha256 it read (`38bc688b…`, the file at `3b7b404e`) and Amendment 1's (`f2f96157…`)"),
    (OR, "The PREREG read for this document is the file at `4c412181` (sha256 `d8b8885b…`, Amendments 1 and 2 included)."),
    (OR, "Amendment 2 changes the build and the re-run, not a row, a bar or a floor"),
    (OR, "| M2b_s0_dagoaccorc | r2b | 03:56 → 04:35 | 40 min | yes |"),
    # R2C_NOTE.md
    (R2C, "A s1 re-run with the code pinned"), (R2C, "05:56:16: a dry-run self-test, run by the agent building the LAUNCH-INTENT box-2 chain, reached the REAL box 2."),
    (R2C, "reached the REAL box 2. Git bash put `/usr/bin` ahead"), (R2C, "Of 65 paths, 62 were byte-identical to r2's. Three were not:"),
    (R2C, "(`0d708b3c…`, the unlanded launch-intent"), (R2C, "r2's queue plus a `:launchintentarm` stage"),
    (R2C, "A s1 was training (launched 05:20; its modules were imported at 05:20, from r2's code). It finished at about 06:01 (3,000 steps, 2,262 s,"),
    (R2C, "cmd.exe then resumed the queue `.cmd` by byte offset in the replaced file and started the trainer again."),
    (R2C, "launch-intent build: `[resume] step 3000"), (R2C, "It rewrote `summary.json`, `selfcheck.json` and the checkpoint."),
    (R2C, "`init_ckpt.applied = false, skipped = \"resume\"`"), (R2C, "The A s1 T0 eval (06:01) imported the replaced model file."),
    (R2C, "*\"summary.json init_ckpt not applied / read back: False / None\"*"),
    (R2C, "The r2 chain, the T1-kin chain and"), (R2C, "the SPEEDSUB waiter all stopped as designed."),
    (R2C, "A s0 and REP s0 trained and evaluated before 05:56."), (R2C, "A s1's refused artifacts are never rolled out or read."),
    (R2C, "06:04:09: r2's own tar (`repo_v8fdagoaccorc_r2_tip.tar`) was re-extracted on box 2."),
    (R2C, "trainer\n  `1c888280…`, model `8ec5e190…`, queue `d8aa4f83…`"),
    (R2C, "A s1's box-2 items were renamed `*_r2b_queue_edit_stop`: run dir, gate dir + log, three argv pins, four eval files."),
    (R2C, "Nothing was deleted."),
    (R2C, "The remote tip had moved to `94a47f4b` (SPEEDSUB; no trainer, model, data or queue change)."),
    (R2C, "branch `agent/dagorc-r2-pin`, created at `8c24823f`, through `OAF_BR` (`run_chain_v8f_dagoaccorc_r2c_pin.cmd`). The chain code is"),
    (R2C, "The first relaunch stopped on the E: disk floor before touching anything. Two aside dirs and two dead lander holds (414 MB) moved to F:."),
    (R2C, "r2c relaunched at 06:09"), (R2C, "A s0 and REP s0 skipped by the queue"),
    (R2C, "The box-2 total is now about 4.8 h, against the ~3 h the PI's go was\nasked as."),
    (R2C, "*A batch file edited while cmd.exe runs it executes the new bytes from the old offset.* Never rewrite a `.cmd` that a live job may be\n   executing. Ship a new file name, and switch only between jobs."),
    (R2C, "*A stub that relies on PATH order is not a stub on Git bash.* `/usr/bin` wins. A self-test that must not reach a host fails closed:\n   check that the stub resolves first and that the real host alias does not resolve, before any step runs."),
    # the PREREG: controls, the M-LAUNCH row, the risk, the arms
    (POR, "As in ORC and DAGGER2, the HMAC is not verified."),
    (POR, "argv(A_s) equals argv(DAG-OACC_s) except `--out`, the six perturbation flags at §1's values and `--ego-perturb-with-dagger`."),
    (POR, "RF0's frozen sha256 predates every A / REP / B\n  rollout's mtime."),
    (POR, "REP − REP reads exactly 0.0 with CI [0, 0] on every row. Feeding REP's dump as CTRL must make RF exactly 0."),
    (POR, "the reader, with the floors at 0 and DAG-OACC in the base slots, reproduces `DAGOACC_READ.json`'s Read-A table"),
    (POR, "| **R-ADE** | T1-kin plain ADE6, X − CTRL: not separated-worse | guard |"),
    # the register
    (REG, "**HARMS (MEASURED 2026-10-28, arm A, Amendments 1 and 2 in force; all 162 reader controls PASS).**"),
    (PSS, "Promoted by the Master Mind from the draft"), (PSS, "No SPEEDSUB number exists"), (PSS, "**Status: PRE-REGISTERED, NOT RUN (2026-10-28).**"),
    (PSS, "The ego-only MLP `mlp_ego` has the lowest fold-0 hold-out RMSE of the four fitted read-outs in all five cells"),
]
_missing = [ph[:100] for src, ph in PHRASES if src.count(ph) < 1]
assert not _missing, _missing

# ---- 7.24 SPEEDREAD -----------------------------------------------------------------------------------------------------------
SEC_SR = """### 7.24 SPEEDREAD: a read-out of T's five speed cells from its own input does not recover what O needs

* **Setup** (`PREREG_V8F_SPEEDREAD` with **Amendment 1**, `A1-2026-10-27`, landed `569a9d9b` before any SPEEDREAD number; claim row `H-V8F-SPEEDREAD-1`).
  The question, commissioned by FLAT-DAG's committed next (§7.22): is the per-window speed information O needs present in T's own input? Two
  readings were left, and they call for different compute. *The speed is in T's input but T does not extract it in a way that generalises*: the
  next lever is then T-side and fits on box 2. *The speed is not in T's input*: perception is then the lever, and it waits for the A40. A
  read-out fitted on frozen inputs, with no training of the driving model, separates the two.
  * **Base.** X_s = `M2b-mixo_s` (MIXO's checkpoints, the SPEEDCAL / KINDX base; §7.10), sha256 `c6a62876…` (s0) and `68a0f335…` (s1), each checked
    at launch and recorded in every dump. No model is trained.
  * **The read-out.** Input: what T reads, minus S's checkpoints. The four scene frames ending at T's tick (each frame's token mean, and the tick
    frame pooled 2 × 5) are reduced to PCA-512 (14,336 raw scene dims), joined to the 22 ego dims T reads (p10f 8, nav 5, limit 4, limit_ok 1,
    frame_ok 4) and standardised on TRAIN. No label, future pose, steering angle or situation-classifier output enters. Targets: the five GT
    speed cells at T's tick (`lon_v_target_ms`, `v_a_ms`, `v_end_ms`, `v_lo_ms`, `v_hi_ms`). Fitted on 306,566 windows of 4,512 TRAIN clips, in 5
    folds by clip, and never on eval-141. LIN is a ridge per cell; MLP is 534 → 256 → 256 → 5 and is the ALT. **PRIMARY** is the one of the two
    with the lower fold-0 hold-out RMSE (train-SD units, mean over cells), selected and written to `fit.json` before any eval batch was opened.
  * **The substitution (the decision instrument).** Per seed, a hook replaces only the five speed cells of the code O reads by the read-out's
    prediction, computed inside the hook from the same batch O sees, so window identity is exact and the substitution is a function of inference
    inputs. S, T's decisions, every other cell and O are X_s's own. Modes: `own`, `identity`, `gt`, `read`, `alt`, `const`, `perm`.
  * **Tier and estimator.** The instrument's tier stamp (READ.json `tier`) is *T1-class open loop (X_s's own held S / T outputs; only the five speed
    cells O reads are replaced)*, the tier its PREREG gives the SPEEDCAL and KINDX ADE6 rows so that CAPTURE is comparable to them. The unmodified
    harness stamps each dump `T0-open-loop-own-held-outputs (trunk-probe grid)`, and §7.10 stamps KINDX's oracle gains T0; both stamps are on record,
    and the quotable tier of this read is the instrument's. The `gt` arm reads the eval GT, so it is a **T0 ceiling and a control, never a
    result**. Paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes / 2,399 windows per seed. It answers *"would
    another draw of EPISODES say this?"* and is blind to training variance. One read-out fit (sha256 `a205609a…`, in all 14 dumps) serves both
    seeds, so the two seeds are O's two training runs, the replicate `H-ESTIM-SEED-1` asks for at this stage. The planner is deterministic, so
    there is no inference-seed term. F = 0.1207 m is the cross-seed spread of M2b (**[I]**, `RESULT_V8F_GRID.md`).
  * **Amendment 1 and the three stops.** The PREREG counted the ego vector as 23 dims; on the real cache `limit` is 4 wide, so it is 22. The
    extract stopped on that broadcast error after one batch of 128 TRAIN windows, before any eval batch was opened and before anything was
    summarised, fitted, selected or printed. Amendment 1 (instrument sha256 `d5545751…` → `fd5ce0be…`) changes how the vector's length is counted
    and nothing that goes into it; the decision table, keys, features, PCA, λ grid, folds and controls are as written. Two further stops were
    run mechanics, and none wrote a number (provenance below).
* **Outcome [M]** (`RESULT_V8F_SPEEDREAD.md`, landed `c7963d72`; decided by the pre-registered `decide()` in `stack/scripts/v8f_mini_eval_speedread.py`,
  stage `read`; all 19 control flags PASS; output `raw/2026-10-28-speedread/read/READ.json`, sha256 `46543c39…`):

<<SR_OUT>>

  The means are the dumps' `raw.traj.ade6_m.mean` (4 decimals), so a delta can differ from the difference of two printed means by 0.0001.
  "Separated-better" is the CI's upper end below 0 and "separated-worse" its lower end above 0. CAPTURE is PRIMARY / (gt − own). The
  pre-registered conditions (first match wins), against the rows:

<<SR_COND>>

  The outcome is **NOT IN INPUT**: PRIMARY is not separated-better at either seed, and ALT is separated-worse at both. Committed next, as written
  (READ.json `next`): *"perception stays the lever: PREREG_V8F_LORA on the A40 (D-V8F-SPEED-PERCEPTION, the PI's go); nothing on box 2 on this
  line"*.
  ⚠ **Scope of the negative.** NOT IN INPUT means *not recoverable by a ridge or a 2 × 256 MLP from the pooled four-frame summary, on unseen
  clips*. It does not cover the full 40-token attention view T itself has; T fails on that view too (SPEEDCAL, **[I]**), and the two negatives
  together are what route to perception. Disclosed design limits: the PCA-512 summary is lossy; the 2 × 5 pooling assumes the `pool4x10` tokens
  are row-major (if they are not, it is a different fixed linear summary, not a leak); S's checkpoints are not read-out inputs; the fit uses the
  eval's age cycle, not the trainer's random ages.
* **Controls** (each ran before its rows were read; 11 controls, 19 flags, all PASS [M]):

<<SR_CTRL>>

  The substitute control (the SPEEDREAD `own` dump against the banked X_s dump) reads a maximum absolute delta over all cells of 8.0e-07 (s0)
  and 7.0e-07 (s1), against a bar of 1e-6. The longitudinal / lateral row tool's own checks (identity, agreement of its ADE6 row with
  READ.json, per-arm values equal to the dump's) pass at both seeds, with a maximum absolute deviation of 5.0e-05 over 42 checks against a bar
  of 1e-4.
* **The read-outs.** PRIMARY is `lin`: its fold-0 hold-out RMSE (train-SD units, mean over the five cells, derived here) is 0.2724 against 0.2920
  for `mlp`, so the MLP is the ALT (`fit.json` `primary`). PERM, the PRIMARY's class on window-permuted targets, reads a held-out TRAIN skill of
  at most 0.0069 (bar 0.02). `lin_ego` and `mlp_ego` are ego-only read-outs (no scene), fitted the same way and read beside.

<<SR_FOLD0>>

* **The four families [M]** (beside: they decide nothing). READ.json carries ADE6 only, so the four families were computed afterwards on the
  banked dumps, read − own at each seed (the PRIMARY contrast) and gt − own as the T0 ceiling. TACTICAL and STRATEGIC come from
  `v8f_paired_families.py`, run as SPEEDCAL's `pairfam` stage runs it. LONGITUDINAL and LATERAL come from `gen/families_longlat.py`, which reuses
  the committed per-window row definitions (`comps` of `t1k_read.py` and `lat_rows` of `t1klc_read.py` at tip `e7ea20d1`) and writes none.
  Tier: T1-class open loop for read − own; the gt − own table is the T0 ceiling and a control. Estimator: paired episode-cluster bootstrap,
  B = 2,000, seed 0, 141 episodes, which answers *another draw of EPISODES* only; there is one training run per seed, so a separated cell is a
  pointer (`H-ESTIM-SEED-1`). **STRATEGIC is reported, not paused, for v8F.** B / W = separated-better / -worse, – = not separated, S+ / S− =
  separated on a row with no better direction (a bias or a ratio), bold = separated. Error rows are lower-better; accuracy and F1 rows are
  higher-better.

  **read − own** (the PRIMARY contrast; point [95 % CI]):

<<SR_RO>>

  **gt − own** (T0 ceiling, a control only; point [95 % CI]):

<<SR_GT>>

* **The read-outs ranked [M] (beside; none of this moves the outcome).** Per-cell eval RMSE to the GT, m/s, on the eval-141 windows where the GT
  cell is present. Every column except T_own is identical at the two seeds, because the read-outs do not depend on the seed. `lin` is the
  PRIMARY (the `read` mode) and `mlp` the ALT.

<<SR_CELLS>>

  1. **By accuracy, the ego-only MLP is first on every beside measure.** `mlp_ego` has the lowest fold-0 hold-out RMSE of the four read-outs in
     all five cells (mean over cells 0.2475, derived, against 0.2724 for `lin`, 0.2807 for `lin_ego` and 0.2920 for `mlp`) and the lowest eval RMSE
     in all five cells, and its eval RMSE is below T_own's at both seeds in all five cells. **Its ADE6 substitution was NOT run**: the
     pre-registered modes are own, identity, gt, read, alt, const and perm, and `mlp_ego` appears only in the identity run's cell dump. No claim is
     made here about what it would do to ADE6.
  2. **Vision plus ego is worse than ego alone for the MLP, and slightly better for the ridge.** `mlp` is above `mlp_ego` in all five cells, on
     hold-out and on eval, and the ALT substitution costs +0.2122 / +0.2040 m of ADE6, separated-worse. For the ridge, `lin` is below `lin_ego` in
     all five cells on both measures (hold-out mean 0.2724 against 0.2807, derived). **[H]** One reading, with no bar: the nonlinear read-out gains
     from the ego channel and loses when the lossy PCA-512 scene summary is added.
  3. **A lower RMSE in one cell did not move ADE6.** Against T's own cells the `lin` read-out is lower in `v_a_ms` at both seeds (0.7796 against
     1.9190 at s0 and 1.3020 at s1), lower in `v_hi_ms` at s0 only, higher in `v_end_ms` and `v_lo_ms` at both seeds, level in `lon_v_target_ms`
     at s0 (2.8934 against 2.8940) and higher at s1 (T_own 2.7470). ADE6 read − own is −0.0097 / +0.0022 m.
  4. **O does respond to these five cells; the ridge read-out's cells do not carry that response.** ADE6 moves by −0.7164 / −0.6676 m under the
     `gt` ceiling, by +5.3455 / +5.7670 m under the constant substitute and by +5.3612 / +5.7569 m under the shuffled one (s0 / s1), and by
     −0.0097 / +0.0022 m under the read. CAPTURE is 0.0135 at s0 and −0.0033 at s1.
* **What the numbers say** (beside the outcome; none of this moves it):
  1. **The read-out recovers none of the headroom.** The five GT cells are worth 0.7164 / 0.6676 m of ADE6 at the T0 ceiling (separated-better,
     a control). The PRIMARY read-out moves ADE6 by −0.0097 [−0.0871, +0.0620] / +0.0022 [−0.0534, +0.0564] m against a bar of −F = −0.1207 m, and
     its controls bracket it (const and perm +5.3455 to +5.7670 m; perm not separated-better).
  2. **No LONGITUDINAL row moves.** Speed, along-track and acceleration MAE are not separated for the read at either seed (+0.0035 / +0.0069 m/s,
     −0.0188 / −0.0103 m, −0.0215 / −0.0159 m/s²). They do move at the ceiling: speed MAE −0.3544 / −0.3424 m/s, along-track MAE −0.7644 /
     −0.7122 m, accel MAE −0.0536 / −0.0303 m/s² (T0, separated-better).
  3. **The hierarchy's own outputs are untouched, as they must be.** The substitution acts on the code O reads, after S and T have run, so the
     three head macro-F1 rows and all six S rows read exactly +0.0000 [+0.0000, +0.0000] at both seeds, for the read and for the gt arm. That is
     a structural zero, not an estimate.
  4. **UNAVAILABLE rows are work items, not passes.** DK-VIOL / DK-HW / DK-TG and the anti-echo block have status UNAVAILABLE with n = 0 in all 14
     dumps: the scoring call was given no lead-agent track and no t0 speed. Every longitudinal number beside the anti-echo block is therefore
     UNVERIFIED against a hold-v0 copy.
  5. **The separated cells are pointers.** 6 of 49 cells are separated for the read at s0 and 8 of 49 at s1. Two rows are separated at both
     seeds: L3 yaw-rate MAE (GT ≥ 5.2 m/s) is separated-better (−0.2536 / −0.2335 deg/s), and the goal longitudinal bias moves by +0.5985 /
     +0.4706 m (S+, a bias has no better direction). The separated-worse rows are cross-track MAE at s0 (+0.0193 m) and ADE2 at s1 (+0.0060 m),
     so no row is separated-worse at both seeds. Each separated cell is one of the 49, tested at one nominal level, with no replicate arm.
  6. **What the ceiling would move, and what it would not.** At the T0 ceiling, TAC-LON accuracy (+0.1159 / +0.1446), the 5-way manoeuvre
     accuracy (+0.0805 / +0.0988), goal-point error (−2.2239 / −2.0637 m) and the fast-window L2 and L3 rows are separated-better at both
     seeds. The all-speed yaw-rate MAE (+0.0570 [−0.4042, +0.4881] / −0.0760 [−0.3837, +0.1851] deg/s) and curvature MAE (+0.0005 [−0.0122,
     +0.0122] / −0.0032 [−0.0099, +0.0027] 1/m) are not separated at either seed.
* **Next (Rule Zero).**
  * **The committed next is blocked, and here is exactly what blocks it.** READ.json `next` names perception: `PREREG_V8F_LORA` (DINOv3-L/16 plus
    LoRA on the fly) on the A40. The A40 is not allocated (PI: box 2 and dev-box CPU only; **[I]**, the Master Mind's brief of 2026-10-28). What
    unblocks it is three PI decisions: (1) the A40 provisioned (`D-V8F-SPEED-PERCEPTION`); (2) the frames on HF private or a disk provision (LORA
    §7, AR-4); (3) the go for about 200 GPU-h (2 seeds). LORA projects 494 h per seed on box 2 against its venue rule of ≤ 36 h per seed (**[I]**,
    LORA Amendment 3; §7.22 and §8.4 row 2 carry 470–494 h), and about 100+ GPU-h per seed on the A40 (**[E]**, LORA Amendment 3).
  * **Nothing on box 2 on this line.** The box-2 streams already queued are unaffected.
  * **How the Master Mind ranks the remaining levers** (beside; **[I]**, `RESULT_V8F_SPEEDREAD.md` §6; scoped to M2b-mixo at T1-class open loop, not
    measured on DAG-OACC under T1-kin). Even the GT speed cells (gt − own, T0 ceiling) leave the all-speed yaw-rate MAE and curvature MAE not
    separated at either seed, and only the fast-window L2 and L3 rows move. So the speed line reaches L2 of DAG-OACC's open crawl set and not
    yaw-rate or curvature. **M1 has no cell in this read, so nothing is measured about it here (UNVERIFIED).** The lever aimed at the crawl rows is
    DAG-ORC (§7.26), which the RESULT records as running on box 2 since 2026-10-28 01:16 (that is attempt r1, which stopped; §7.26 has the history and the read, which is HARMS).
    An ego-only speed-cell arm (the `mlp_ego` read-out, not yet run as a substitution) is ranked below it; `PREREG_V8F_SPEEDSUB` registers it (promoted at `94a47f4b`, before any SPEEDSUB number; no SPEEDSUB number exists in this version).
* **Provenance.** Dev-box CPU (`CUDA_VISIBLE_DEVICES=-1`; `logs/run.out`): extract 23:21:49–00:17:14, fit 00:17:14–00:54:18, evals s0
  00:56:19–01:03:22, evals s1 01:03:22–01:09:19, read 01:09:19–01:09:25 (2026-10-27 to 2026-10-28). Data parity: 4,512 clips, 306,566 windows,
  excluded-holdout sha256 `f6a1dee0…`, grid `pool4x10`, join sha256 `978f41bf…` (equal to the trainer's `--join-train`), and the eval-141 clips
  and the fit clips do not intersect. Instrument `stack/scripts/v8f_mini_eval_speedread.py`, sha256 `fd5ce0be…`, pinned in `run/run_speedread.sh`;
  its self-test is 27 passed and 19 / 19 mutants killed (**[I]**, `PREREG_V8F_SPEEDREAD` Amendment 1 §3, `raw/2026-10-27-speedread-a1/`).
  * **Three stops, each before any SPEEDREAD number** (none wrote or displayed a result): (1) 2026-10-27 23:09, `extract`,
    `ModuleNotFoundError: No module named 'taniteval'`, run mechanics (a wrapper now puts the exported tree's `taniteval/` on `sys.path`);
    (2) 2026-10-27 23:10, `extract`, a (128, 22) array into a (128, 23) slot, which is **Amendment 1**; (3) 2026-10-28 00:54, `evals_s0`,
    `FileNotFoundError` on `eval_grid_windows.json`, 4 s after the fit finished, run mechanics (the 12 eval-input files were exported from the same
    commit `f3971165` and compared byte-identical by sha256, and the run restarted at 00:56:18 with extract and fit skipped as done). The logs are
    banked under `stops/`.
  * **The pull.** 4,572 TRAIN clips were md5-verified against the manifest (`ZZSPEEDREAD-PULL-DONE` 23:08:43; 94.27 GB for the 4,512 the parity rule
    keeps). It read box 2's USB HDD (per the Master Mind) and ran without the temperature guard (`run/pull_speedread.sh` has no temperature check).
  * **Banked** in `R/raw/2026-10-28-speedread/` (`read/`, `fit/fit.json`, `train/extract.json`, the 14 per-mode eval JSONs, `families/`, `logs/`,
    `pull/`, `stops/`, `run/`, `landing/`). **Single copy on F: (dev-box scratch), not in the repo:** `fit/readout.pt` (118,851,106 B, sha256
    `a205609a…`), `train/train_arrays.npz` (94,730,100 B), `train/scene_raw.npy` (8,789,860,480 B), the per-mode `eval_*.npz` and
    `speedread_cells_*.npz` dumps, the train-cache pull and `stop2/train_partial_unread/`.
"""

# ---- 7.25 DAGGER2 -------------------------------------------------------------------------------------------------------------
SEC_D2 = """### 7.25 DAgger iteration 2: DAG's own states used a second time do not move the crawl rows

* **Setup** (`PREREG_V8F_DAGGER2` with **Amendment 1**, `A1-2026-10-27`, landed `1446333d` before any DAGGER2 row existed; claim row `H-V8F-DAGGER2-1`;
  promoted 2026-10-27 from the draft banked at `5eaa0232`). This section is placed here for the sequence of §7; the read was made on 2026-10-27
  15:56, before FLAT-DAG (§7.22) and DAG-OACC (§7.23), which cite it. The question: does a second DAgger iteration, on the states DAG itself
  reaches, remove the launch crawl DAG leaves (§7.20)? DAgger's ROBUST row, registered before any DAgger data, committed it (*"If the crawl beside
  rows (M1, L2) still fail, next is DAgger iteration 2 (DAG's own states), under its own read"*), and Read A (§7.20) measured M1 and L2 separated-worse at both seeds.
  * **Arms** (box 2, all four through their launch gates). DAG2 = DAG_s fine-tuned 3,000 steps at `--dagger-p 0.1` on the iteration-2 bank, which
    DAG_s generated on its own T1-kin states (m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, executor `plan`; the same 4,512
    TRAIN clips and 306,566 windows). DAG2-0 = the same argv at `--dagger-p 0`. Banks: s0 `90c43c66…`, s1 `a06c8289…`, COVERAGE 0.773 / 0.764.
    The swap share reached in training is 0.10085 (DAG2 s0) and 0.10062 (DAG2 s1), and 0 for both DAG2-0. Checkpoints `0d0a800b…`,
    `603b9151…`, `2e03b141…`, `f4234ac1…`.
  * **Rows and guards, committed before any DAGGER2 row.** The crawl set C is the rows that read separated-worse for DAG − P10 (`RESULT_V8F_DAG_REREAD.md`
    §1): curvature MAE, L2 and M1 at s0, and those plus yaw-rate MAE at s1. **R-CRAWL**: every row of C is not separated-worse for DAG2 − P10.
    **R-MOVE**: every row of C is separated-better for DAG2 − DAG2-0. Guards, each at both seeds: T's lateral and longitudinal macro-F1 (DAG2 −
    DAG2-0) not separated-worse; T1-kin ADE6 (DAG2 − DAG2-0) not separated-worse; T0 ADE6 (DAG2 − DAG2-0) point ≤ +F/2 = 0.0604 m; and no Read-A row
    that was not separated-worse for DAG − P10 becomes separated-worse for DAG2 − P10 (accel MAE and DK-VIOL are exempt, because they were
    already worse and belong to DAG-OACC). The outcome table is FIXED / PARTIAL / NO EFFECT / HARMS / BLOCKED / REFUSED: REFUSED and BLOCKED are
    decided first, then the guards (any failure is HARMS), then R-MOVE and R-CRAWL.
  * **Amendment 1** found that the table, though it said it was complete, left 4 of its 16 (R-CRAWL, R-MOVE) × 2-seed cells unplaced (R-MOVE at
    exactly one seed with R-CRAWL failing there); they become PARTIAL, and each of the 12 placed cells keeps its outcome. It was written after the
    four box-2 arms were trained and before any DAGGER2 row existed, and it moves no guard. A one-seed R-MOVE is reported as PARTIAL (one seed) and
    is never called an effect of the lever, because no crawl row has a training-replicate floor (`H-ESTIM-SEED-1`).
  * **Tier and estimator.** T1-kin (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, plain arms); R-T0 is at
    T0 (box 2's standard eval). Paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed. It answers *"would
    another draw of EPISODES say this?"*; the training-run question is carried by the two seeds, and the planner is deterministic, so there is
    no inference-seed term.
* **Outcome [M]** (`RESULT_V8F_DAGGER2.md`; read 2026-10-27 15:56 by the pre-registered reader `raw/2026-10-27-dagger2-t1read/dag2_read.py`, sha256
  `1425e8a3…`; all 199 reader controls PASS (23 control groups); output `raw/2026-10-27-dagger2-result/read/DAG2_READ.json`, sha256 `056ecba5…`):

<<D2_OUT>>

  The outcome is **HARMS** (a guard fails at ≥ 1 seed). Committed next, as written: *DAG2 is not adopted; DAG stays the base. Next: as PARTIAL, on
  DAG states* (the launch-recovery target of `PREREG_V8F_HACC_MC` §7 on DAG states, under its own PREREG). With the guards held the table would
  have read NO EFFECT (R-MOVE fails at both seeds): the crawl rows did not move.
* **The four families [M]** (T1-kin; point [95 % CI]; B / W = separated-better / -worse):

<<D2_FAM>>

  STRATEGIC is reported, not paused, for v8F (S-POS is read against CTRV within a run; paired S-POS across arms is not computed). **Not computed,
  with the reason:** the T0 four-family beside rows and the HACC re-fit beside row were not built into this reader (declared NOT RUN in its
  output), so T0 carries only R-T0. Full CIs for every cell are in `read/readA_compact.txt`.
* **What the numbers say** (beside the outcome; none of this moves it):
  1. **DAG2 does not beat DAG.** DAG2 − DAG ADE6 is +0.0426 [+0.0011, +0.0896] m at s0 (separated-worse) and −0.0309 [−0.085, +0.026] m at s1. The
     lever's win over DAG2-0 (−0.22 / −0.16 m) is mostly recovery of what 3,000 more GT-only steps cost: DAG2-0 − P10 ADE6 is −0.898 / −0.855 m
     (points, by additivity of paired means), against DAG's −1.166 / −0.987 m.
  2. **The crawl rows did not move.** M1, L2 and curvature stay separated-worse than P10 at both seeds, and R-MOVE fails at both. Iterating the
     state distribution once more does not remove the launch crawl. This is the second training-distribution lever on the crawl to read no
     effect (O-RECOVER read HARMS on M2b, §7.18).
  3. **The swap costs acceleration.** DAG2 − DAG2-0 accel MAE is +0.5846 / +0.6034 m/s², separated-worse at both seeds. That is the row DAG-OACC
     was built to fix (§7.23).
  4. **The guard that fired rides on run-to-run spread as much as on the lever.** Yaw-rate MAE against P10 reads −0.04 / +8.66 for DAG's two
     seeds and +6.25 / +5.63 for DAG2's; DAG2 − DAG is +6.29 (separated-worse) at s0 and −3.03 (separated-better) at s1, and DAG2 − DAG2-0 is not
     separated at either seed. The newly-worse clause compares two different training runs of the same recipe (DAG against DAG2) on a row
     with no measured replicate floor. The outcome is still HARMS: the clause was registered, and it is reported as written.
* **Next (Rule Zero).**
  * **The committed next** is the launch-recovery target on DAG states, under its own PREREG. It became `PREREG_V8F_DAG_ORC`, written after this
    read, and was then re-based on DAG-OACC (§7.23, §7.26).
  * **The instrument gap this read exposed:** the newly-worse clause needs a replicate floor per row before it can carry weight on rows like
    yaw-rate MAE (cheapest form: one same-argv replicate of the base, read on Read A). `PREREG_V8F_DAG_ORC` registers it as the replicate arm REP and
    the floors RF0 and RF (§7.26).
  * **Context at the read:** FLAT-DAG was running on box 2 (since 15:37) and DAG-OACC was re-queued behind it (§7.22, §7.23).
* **Provenance.** Dev-box CPU: chain `raw/2026-10-27-dagger2-t1read/dag2_t1_chain.sh`, 8 / 8 rollouts plus 4 fresh-T0 evals, 15:19–15:56
  (`read/_dag2_work/chain.out`). Before launch: a real-pull self-test of 199 / 199 checks with 68 / 68 mutants killed (`selftest_real_pulls.txt`)
  and a dry run on the real config, `ZZDAG2T1-DRY-OK` (`dry_run_real_config_amend1.txt`). Per-episode dumps and tick arrays (1.3 GB) stay at
  `E:/Projects/TanitAD-artifacts/v8f-dagger2-t1/`, single copy, not in the repo.
"""

# ---- 7.26 DAG-ORC -------------------------------------------------------------------------------------------------------------
SEC_OR = """### 7.26 DAG-ORC: O-RECOVER's perturbation composed onto DAG-OACC reads HARMS, and its crawl effects are smaller than the replicate floor

* **Status of this section (v32).** DAG-ORC (arm A, p = 0.3) was read on 2026-10-28 at 07:58 and its outcome is **HARMS: A is not adopted**
  (`RESULT_V8F_DAG_ORC.md`, landed `ea5e3d3b`; the register row `H-V8F-DAG-ORC-1` reads HARMS). Arm B (p = 0.1) was not trained and is not read.
  The order below is the design, the run history (the launch that completed was the fifth; the four before it stopped, each for a recorded
  reason), and then the outcome, the controls, the floors, the four families, the numbers, the next step and the provenance. What was NOT RUN or
  NOT BUILT is stated per family in the families item, and none of it enters the outcome.
* **Design** (`PREREG_V8F_DAG_ORC`, claim row `H-V8F-DAG-ORC-1`, promoted 2026-10-27 before any DAG-ORC gate, run or row; **Amendment 1**,
  `A1-2026-10-27`, landed `6cd741b6`, re-based it on DAG-OACC before any DAG-ORC run; **Amendment 2**, `A2-2026-10-28`, landed `8c24823f`,
  before any DAG-ORC row is read).
  * **Why it runs.** DAGGER2 read HARMS (§7.25), and its committed next is the launch-recovery target on the base's states. The PREREG's case
    (**[I]**) is that O-RECOVER's perturbation (§7.18) is the only training lever measured to move two crawl rows at both seeds (curvature and
    yaw-rate MAE, separated-better at T1-kin on M2b), and that it puts its samples where the crawl is, in windows with logged v0 < 8 m/s.
    Amendment 1 re-based the arm because DAG-OACC read LONGITUDINAL FIXED (§7.23): its open rows at both seeds, DAG-OACC − P10, all
    separated-worse, are yaw-rate MAE, curvature MAE, L2 and M1.
  * **Arms** (box 2; each through the trainer's own launch gate; both seeds):

| arm | what | role |
|---|---|---|
| **A** (DAG-ORC) | DAG-OACC_s's training argv verbatim (init M2b_s, 3,000 steps, `--dagger-p 0.1` on the iteration-1 bank, `--w-acc 49.6969`), plus the perturbation `--ego-perturb-p 0.3 --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed <s>`, plus `--ego-perturb-with-dagger`, with its own `--out` | decisive |
| **CTRL** = DAG-OACC_s | banked, reused: checkpoints `67265ec1…` (s0) / `90b01474…` (s1); T1-kin dumps `v8f-dagoacc-t1/` | the one-lever control: A minus the perturbation |
| **REP** | DAG-OACC_s's argv with `--seed 10` / `--seed 11` (init M2b_s0 / M2b_s1), with its own `--out`; trained new | the replicate that sets the training-run floor RF |
| **B** | A with `--ego-perturb-p 0.1` | the dose fallback, trained only on A's HARMS and with its own PI go; not queued |

  * **The composition.** The trainer refused `--dagger-p > 0` together with `--ego-perturb-p > 0`. A new default-off flag,
    `--ego-perturb-with-dagger`, lifts that refusal for an argv that names it. The swap is drawn first; the perturbation is then drawn, at its
    own p and from its own seeded generator, only on samples the swap did not replace; no sample is both swapped and perturbed. With the flag
    off, every existing argv must build and train bit-identically (control K-BUILD).
  * **Tier and estimator.** The decisive rollouts are T1-kin (self-action open loop; plain arms, `--replan-steps 10 --limit causal`, dead-reckoned
    nav, logged scene, periodic, executor `plan`; the lead follows its logged future), run on the dev-box CPU; CTRL's, DAG's and P10's runs are
    banked. R-T0 and the T0 four families are specified on box 2's standard T0 eval (R-T0 was read; the T0 families were NOT RUN, see the families item below). Paired episode-cluster bootstrap, B = 2,000, seed 0, 141
    episodes / 2,399 windows, per seed. Three variances ride on the interval, and each is named: *another draw of EPISODES* (the bootstrap),
    *another TRAINING RUN* (the two seeds plus the replicate floors below) and *another INFERENCE RUN* (none: the planner is deterministic).
  * **Floors, bars and rows.** F = 0.1207 m for ADE6 (**[I]**, `RESULT_V8F_GRID.md`) and 0 for every other row, which carries RF0_r instead.
    RF0_r is the maximum over seeds of |DAG-0 − ORC-0| on Read-A row r (a banked same-seed, cross-build pair on M2b), computed and frozen
    before any DAG-ORC rollout was read (`raw/2026-10-27-dag-orc-read/P0_RF0.json`, sha256 `1608d70b…`). RF_r is the maximum over seeds of
    |REP − DAG-OACC|, read after REP's rollouts and before any A row. The bar of a row is bar_r = 2 × max(F_r, RF0_r, RF_r), inclusive. A row
    **clears** when X − CTRL is separated-better and the improvement is at least bar_r. It **fires** when X − P10 is separated-worse, DAG-OACC −
    P10 was not, and X − CTRL is separated-worse by at least bar_r.
  * **The decisive, guard and adoption rows** (Amendment 1 §2 reads "DAG" and "CTRL" as DAG-OACC). **R-CRAWL**: every row of C_s (yaw-rate MAE,
    curvature MAE, L2, M1) is not separated-worse for A − P10. **R-MOVE**: every row of C_s clears for A − DAG-OACC. Guards: R-TLAT and R-TLON
    (the T head's lateral and longitudinal macro-F1 at ticks ≥ 1, not separated-worse), R-ADE (T1-kin ADE6, not separated-worse), R-T0 (T0
    ADE6, point ≤ +F/2 = 0.0604 m) and R-NEW (no row of the domain D_s fires; D_s is every Read-A row not separated-worse for DAG-OACC − P10, with
    no exempt rows, so accel MAE and DK-VIOL are now guarded). Adoption: R-KEEP (T1-kin ADE6 A − DAG-OACC, point ≤ +0.0604 m) and R-NOBASE.
    R-ADE and R-KEEP share one contrast, A − DAG-OACC (disclosed).
  * **The outcome table** (the PREREG's §7 table with "DAG" read as DAG-OACC, and Amendment 1 §3's committed nexts):

| outcome | condition | committed next |
|---|---|---|
| **REFUSED** | any control fails | fix and re-read; no row is reported |
| **HARMS** | controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at ≥ 1 seed | A: not adopted; read B under the same table (B needs its own PI go). B: not adopted; the perturbation cannot be dosed under the guards, and NEXT-CRAWL follows |
| **FIXED / ADOPTED** | guards held; R-MOVE and R-CRAWL at both seeds; R-KEEP and R-NOBASE at both seeds | adopt A as the base and re-run Read A on it; DOMINANT there is the dominance claim |
| **FIXED / NOT ADOPTED** | guards held; R-MOVE and R-CRAWL at both seeds; R-KEEP or R-NOBASE fails at ≥ 1 seed | B (p = 0.1) under this table, a lower dose that may keep the base (own PI go) |
| **PARTIAL / ADOPTED** | guards held; R-MOVE at both seeds; R-CRAWL fails at ≥ 1 seed; R-KEEP and R-NOBASE at both seeds | adopt A; the crawl rows that still fail go to NEXT-CRAWL on A's states |
| **PARTIAL / NOT ADOPTED** | guards held; R-MOVE at both seeds; R-CRAWL fails at ≥ 1 seed; R-KEEP or R-NOBASE fails at ≥ 1 seed | as FIXED / NOT ADOPTED: B (p = 0.1) |
| **PARTIAL / ONE SEED** | guards held; R-MOVE at exactly one seed | not adopted, and never called an effect of the lever (`H-ESTIM-SEED-1`); NEXT-CRAWL |
| **NO EFFECT** | guards held; R-MOVE at neither seed | the state channel is eliminated as the crawl's lever (own-state levers would then have failed twice: DAgger iteration 2, then the perturbation); NEXT-CRAWL |

  The table is complete and exclusive: 524,288 cells, each matching exactly one row (REFUSED 262,144; HARMS 261,888; FIXED / ADOPTED 1; FIXED /
  NOT ADOPTED 15; PARTIAL / ADOPTED 3; PARTIAL / NOT ADOPTED 45; PARTIAL / ONE SEED 128; NO EFFECT 64), by `enumerate_cells.py`, with 13 / 13
  mutants of the enumerator exiting 1. Amendment 1 changed the base and no condition, so the enumeration stands.
  * **The four families** (per seed, per family, never pooled; **STRATEGIC is not paused for v8F**):

| family | T1-kin (primary), Read-A rows | T0 (box 2's eval), beside | not computable |
|---|---|---|---|
| **LONGITUDINAL** | speed MAE, along-track MAE, accel MAE; DK-VIOL / DK-HW / DK-TG (LEAD windows only) | speed, accel and along-track MAE; headway, time gap, TTC, LEAD-VIOL | none; the anti-echo block is reported beside every longitudinal number |
| **LATERAL** | heading, yaw-rate, curvature, cross-track, L1 lateral-accel, L2 | heading, curvature, yaw-rate, cross-track | none |
| **TACTICAL** | TAC-LAT / TAC-LON; R-TLAT / R-TLON | lateral and longitudinal head macro-F1 and accuracy; goal-head macro-F1 | T1-kin goal-head F1 and head accuracy (the paired tool reads eval `run` dumps, not tick dumps); anchor selection is UNAVAILABLE by design |
| **STRATEGIC** | S-POS@30 / 50 / 80 against CTRV, per run | S head checkpoint error and the strategic rows of the eval JSON | paired S-POS across arms is not computed |

  * **Known risk.** On M2b the lever cost T0 ADE6 +0.1103 / +0.0849 m, above the +F/2 bar at both seeds (§7.18; `PREREG_V8F_DAG_ORC` §0). That was the most likely way A would fail,
    and B at p = 0.1 is the registered response. R-T0 is one of the guards A failed (below).
  * **What this read cannot claim.** It is T1-kin, a self-action open loop (the lead follows its logged future and the cameras are logged), and
    not T2. A one-seed R-MOVE is not an effect. The lever has two components, longitudinal and yaw-rate, and neither half is credited. No bar,
    row, floor formula or routing changes after any DAG-ORC row is seen. One training run per arm and seed: the two seeds and the REP pair carry
    the training-run question, and no row is an effect of the lever on its own (`H-ESTIM-SEED-1`). B is not read, so nothing here says what p = 0.1 does.
  * **NEXT-CRAWL, registered before any row.** If the crawl does not move, the branch on DAG-OACC's states is LAUNCH-INTENT (**[I]**, Amendment 1
    §3): O's along-track loss over the executed horizon (knots 1–12 at m = 10), up-weighted on GT-launch TRAIN samples (v0 < 6 m/s and GT v(6 s)
    ≥ v0 + 1 m/s), a label-side subset. COMPOSE is done (DAG-OACC − DAG is not separated-better on the rows of C, §7.23), and N0, the nav probe,
    already read NOT NAVMOVE (**[I]**, Amendment 1 §3; the N0 RESULT is not re-read here).
* **Run history [M]** (mechanics of the launches, from the box-2 chain logs and the notes; the read follows):
  * **r1** (`chain_v8f_dagoaccorc.sh` at `e7ea20d1`, box 2, started 2026-10-28 01:16 under the PI's go of 2026-10-27, asked as about 3 h of box
    2). All four gates passed (A and REP, both seeds). A s0 then trained its 3,000 steps, was pulled and passed its run record, and
    **K-REACH failed**: `ego_perturb_n / ego_perturb_eligible` = 19,161 / 76,898 = **0.2492**, outside the band 0.3 ± 0.02, with a swap share of
    0.1013 and an overlap of exactly 0 (`raw/2026-10-28-dag-orc-a2/r1_refused/M2b_s0_dagoaccorc/summary.json`). The chain stopped as designed
    (`ZZDAGOACCORC-CONTROL-FAIL`); REP s0, A s1 and REP s1 never ran, and the refused run is never rolled out or read.
  * **The defect: two seeded generators with the same formula and the same seed are one generator.** `DaggerSwap._seed_of` and
    `EgoPerturb._seed_of` are the same formula, `(seed × 1,000,003 + step) mod 2^62`, and the registered argv sets both seeds to s. Both levers
    read their selection uniform from row 0 of `torch.rand(k, B)`, so the two rows are identical (`torch.equal` is True). A sample eligible for
    both is swapped iff u < 0.1 and perturbed iff 0.1 ≤ u < 0.3, which gives P(perturbed | not swapped) = 0.2 / 0.9 = 0.222 against 0.3 for a
    sample only the perturbation may touch; the measured 0.2492 lies between the two. The composition tests had missed it, because their fixture
    did not combine equal seeds with overlapping eligibility. K-REACH caught it on the first real run, which is the job the PREREG gave it.
  * **Amendment 2 (`A2-2026-10-28`): the perturbation gets an independent stream.** On the composed path only, the perturbation's generator takes a
    domain-separated seed: `EgoPerturb` takes a `stream`, 0 is the original formula (so flag off, either lever alone and plain M2b stay
    bit-identical), the composed stage sets `EGO_PERTURB_COMPOSED_STREAM = 1`, a sha256-mixed seed, and the run record carries it as
    `levers.ego_perturb_stream`. The new tests are RED on `e7ea20d1` (P(perturbed | not swapped) 0.0000 at 0.5 / 0.5 and 0.221 / 0.224 at 0.1 /
    0.3; a 200-step run at box 2's shape reads 1,561 / 6,345 = 0.2460) and GREEN on the fix (the same run reads 1,860 / 6,345 = 0.2931, swap 720 /
    7,380, overlap 0, `levers.ego_perturb_stream` 1); the mutation sweep kills 27 / 27; the seven v8F suites pass with 0 failures (two tests
    were skipped for a missing export in the fix's tree; the Master Mind's state note records the post-landing suites at 36 / 97 / 82 passed with 0
    skipped, **[I]**). All four runs
    were re-run from their gates at the fixed commit `8c24823f` (r2), and every r1 artifact was renamed `*_r1_kreach_stop` (nothing deleted). The
    registered argv, the K-REACH bands, the one-lever rule, the reader (`2e5f9f17…`), the crawl set, the domain, the floors and the outcome table
    did not change.
  * **r2's first launch (2026-10-28 03:27–03:40) was refused at its first run record** (`raw/2026-10-28-dag-orc-r2a/R2A_NOTE.md`). The box-2
    aside job printed `ASIDE-ALREADY-DONE` and moved nothing, because the marker it checked, `r2_aside_done.txt`, was DAG-OACC r2's (written
    2026-10-27 20:21): the DAG-ORC r2 library had copied DAG-OACC r2's aside mechanism, marker file name included, and both chains use the same
    runs root. The four gates then passed at `8c24823f`; the queue skipped A s0's training, because r1's `summary.json` was still in
    `M2b_s0_dagoaccorc`, and the eval stage reused r1's eval files. The pulled checkpoint was r1's byte for byte (`6ea4d6ba…`, the same sha256 as
    `v8f-dag-orc-oacc_r1_kreach_stop`), with r1's counts (19,161 / 76,898). The run-record check refused it: the record carries no
    `levers.ego_perturb_stream`, a field only the fixed trainer writes, so its absence is the signature of a stale run. The chain stopped
    (`ZZV8FMINI-STOP`), and nothing of the refused launch is rolled out or read.
  * **The second launch (r2b, 03:48).** The marker is now this chain's own name, `r2_aside_done_dagoaccorc.txt`. The self-test's r1 fixture carries DAG-OACC
    r2's marker as a decoy in the same runs root, and a new mutant restores the shared name and must be killed; the chain self-test passes 66 / 66
    tests with 224 / 224 mutants killed (`raw/2026-10-28-dag-orc-r2a/r2/logs/selftest_chain_r2.out`). The relaunch's aside moved 22 box-2 items
    to `*_r1_kreach_stop` (job end 03:48:22, `raw/2026-10-28-dag-orc-r2a/relaunch/box2_aside_10280348.log`). The dev-box T1-kin chain, with a two-step K-BUILD (the r1 golden
    writer, then the r2 fix re-run) against the base commit `aeb54468` and a self-test of 153 / 153 checks, was launched with it
    (**[I]**: the commit message of `01f3d34e` records the launch; the Master Mind's state note, `E:/Projects/TanitAD-artifacts/dagland/MM_STATE_2026-10-28.md`,
    not in the repo, gives 03:50; the package's `t1/NOTES.md` was written before launch and reads BUILT AND REHEARSED, NOT LAUNCHED). A s0 and REP s0
    trained and passed their run records (jobs ended 04:35 and 05:17); A s1 trained from 05:20 and was refused at its run record at 06:04 (next item).
  * **r2b's A s1 was refused at its run record (2026-10-28 06:04): a self-test run by another agent replaced the box-2 queue while A s1 was
    training** (`raw/2026-10-28-dag-orc-result/notes/R2C_NOTE.md`; the box-2 train log in the package shows a second `train M2b_s1_dagoaccorc` at 05:59:36).
    Per the note (**[I]**, not re-verified here): at 05:56:16 a dry-run self-test, run by the agent building the LAUNCH-INTENT box-2 chain, reached the
    real box 2, because Git bash put `/usr/bin` ahead of its stub `ssh` / `scp` on PATH and the stubs were never called. Its code-refresh step extracted a
    scratch tar over the box-2 repo. Of 65 paths, 62 were byte-identical to r2's; three were not (the unlanded launch-intent trainer, the model file, and
    the box-2 queue, which was r2's queue plus a launch-intent stage). A s1 had been training since 05:20 on modules imported from r2's code, and
    finished at about 06:01 (3,000 steps, 2,262 s). cmd.exe then resumed the queue `.cmd` by byte offset in the replaced file and started the trainer
    again; that second run was the launch-intent build, and it rewrote A s1's `summary.json`, `selfcheck.json` and checkpoint (its records carry
    `init_ckpt.applied = false, skipped = "resume"`). The A s1 T0 eval (06:01) imported the replaced model file. The run-record check refused A s1
    (*"summary.json init_ckpt not applied / read back: False / None"*), and the r2 chain, the T1-kin chain and the SPEEDSUB waiter all stopped as designed.
    A s0 and REP s0 had trained and been evaluated before 05:56 and are unaffected. A s1's refused artifacts are never rolled out or read.
  * **The repair, a disk-floor stop and the final launch (r2c, 06:09).** At 06:04:09 r2's own tar was re-extracted on box 2, and the three files read
    r2's sha256 again (trainer `1c888280…`, model `8ec5e190…`, queue `d8aa4f83…`). A s1's box-2 items were renamed `*_r2b_queue_edit_stop` (run dir, gate dir and
    log, three argv pins, four eval files), and nothing was deleted. The remote tip had moved to `94a47f4b` (SPEEDSUB; no trainer, model, data or queue
    change), the chain refreshes box 2 from the remote tip of its branch, and A s0 and REP s0 had been gated at `8c24823f` with the reader requiring one
    commit for all four runs. So the relaunch reads the branch `agent/dagorc-r2-pin`, created at `8c24823f`, through `OAF_BR`
    (`run_chain_v8f_dagoaccorc_r2c_pin.cmd`); the chain code is unchanged. The first relaunch stopped on the E: disk floor before any job (pulling 805,306,368 B
    would leave less than 1,073,741,824 B; free 1,858,076,672 B), and two aside dirs and two dead lander holds (414 MB) were moved to F:. r2c then ran from
    06:09 to 07:44 (95 min) and ended `ZZV8FMINI-DONE`: box 2 code `8c24823f`, its own aside marker recognised, A s0 and REP s0 skipped by the queue (already
    trained; checkpoint sha256 re-verified), A s1 and REP s1 trained 06:21–07:00 and 07:03–07:42. The launches, from the box-2 chain logs (RESULT §5; first
    to last timestamped line, minute resolution):

<<OR_LAUNCH>>

  * **Cost, disclosed.** Summed over the launches the box-2 chain logs span **296 min = 4.93 h** (53 + 12 + 136 + 95), against the **~3 h** the PI's go was
    asked as (Amendment 1 §4 planned 2.8 h) and against the **about 4.8 h** that `R2C_NOTE.md` states (an estimate that adds up the notes' own figures:
    `R2A_NOTE.md` 3.9 h after the first r2 launch, plus the 0.2 h and the 0.7 h that `R2C_NOTE.md` adds). The overrun beyond the ~3 h is 1.9 h by the logs. The
    four kept trainings are 2.52 h of the 4.93 h (A s0 2286 s, REP s0 2240 s, A s1 2293 s, REP s1 2237 s; `summary.json`); the rest is the refused r1 A s0
    training, the first r2 launch, the refused A s1 training, and the gates, evals and pulls of each launch.
  * **The class of the second stop:** *two chains that write the same marker into one directory share one marker.* It is the same class as the
    composed perturbation's stream: an identity (a seed, a file name) copied from a sibling is shared with that sibling. In both cases a positive
    content check caught it on the first artifact.
  * **The classes of the third stop (A s1):** *a batch file edited while cmd.exe runs it executes the new bytes from the old offset* (never rewrite a `.cmd`
    a live job may be executing; ship a new file name and switch only between jobs), and *a stub that relies on PATH order is not a stub on Git bash*
    (`/usr/bin` wins; a self-test that must not reach a host fails closed, by checking before any step that the stub resolves first and that the real
    host alias does not resolve) (`R2C_NOTE.md`).
* **Outcome [M]** (`RESULT_V8F_DAG_ORC.md`, landed `ea5e3d3b`; read 2026-10-28 07:58 by the pre-registered reader
  `raw/2026-10-28-dag-orc-build/rebase/read/dag_orc_read.py`, sha256 `2e5f9f17…` (started 07:54:18, rc 0 at 07:58:54, 275.4 s); all **162 reader controls PASS**;
  output `raw/2026-10-28-dag-orc-result/read/DAGORC_READ.json`, sha256 `70a7a3a4…`, and the replicate floors `RF_REP.json`, sha256 `64c9444d…`, written before
  any A contrast was computed). Tier: T1-kin (self-action open loop; m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, executor `plan`; plain arms);
  R-T0 is at T0 (teacher-forced own-code, eval-141, ages cycle: box 2's standard eval), a world-model diagnostic and not driving performance. Estimator:
  `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2,000, seed 0, 141 episodes / 2,399 windows, per seed, paired, on the same windows and matched cells. It
  answers *"would another draw of EPISODES say this?"*. The control is a different training run (DAG-OACC), so the training-run question is carried by the two
  seeds and the replicate floors RF0 and RF (`H-ESTIM-SEED-1`), and the planner is deterministic, so no inference-seed question arises. Arms (box 2, r2c chain
  pinned at `8c24823f`): A_s = DAG-OACC_s's training argv verbatim plus the six perturbation flags at the registered law (`--ego-perturb-p 0.3`) and
  `--ego-perturb-with-dagger`; REP_s = DAG-OACC_s's argv with `--seed 10` / `--seed 11` (same init M2b_s); CTRL = DAG-OACC_s, banked (no control was
  retrained). Checkpoints, sha256 prefixes: A `380b230d…` (s0) / `870dded7…` (s1); REP `b03fd6ef…` / `77853e56…`; CTRL `67265ec1…` / `90b01474…`.

<<OR_OUT>>

  R-ADE and R-KEEP read one contrast, A − DAG-OACC (Amendment 1 §2: DAG = CTRL = DAG-OACC), so the two ADE6 rows above are the same cells. The outcome is
  **HARMS**. The table's condition, verbatim: *"controls pass; a guard (R-TLAT, R-TLON, R-ADE, R-T0, R-NEW) fails at ≥ 1 seed"*. Three guard cells fail:
  R-ADE at s0, and R-T0 at s0 and at s1. REFUSED is decided first and HARMS second (PREREG §7, Precedence), so the R-MOVE flags do not change the word.
  Committed next, verbatim from the JSON: *"A: not adopted; read B (p 0.1) under this same table (B is NOT queued: it needs its own PI go, A1 section 4)."*
  ⚠ **Reported as written, beside the word.** R-MOVE is false at both seeds, with 0 of 4 crawl rows clearing the bar at either seed, and R-CRAWL is false at both
  seeds. The guard failures are the registered conditions; the table does not test whether the cost is the lever's (the replicate reading below).
* **Controls [M]** (each ran before its rows were read). The reader's **162 control groups all PASS** (`DAGORC_READ.json` `controls_PASS`: 162 keys, every value true).
  By the PREREG's names (§6; counts derived here from the key names): K-GATE 6, K-INIT 6, K-PAIR 4, K-REACH 4, K-ZERO 2, K-BUILD 1, K-P0 1 and K-RC 2 (the reader,
  with the floors at 0, reproduces the banked DAG2 and DAG-OACC rows), and 136 further groups that are Read A's and DAGGER2's reader controls, unchanged (K1 to K7,
  run summaries, argv, grid and window counts, T0 checkpoint and age checks, identity checks).
  * **K-GATE and the run records.** Each of the four new runs has a launch-gate PASS token binding the commit, the argv sha256 and the data sha256 (as in ORC and
    DAGGER2, the HMAC is not verified). The same tokens (same commit `8c24823f`, same argv sha256) were written in the second launch (r2b) and the final one (r2c). Four
    gates PASS, four run records PASS and four K-REACH PASS on the final four runs.
  * **K-REACH.** Bands: the perturbation share is 0.3 ± 0.02 for A and exactly 0 for REP, the swap share is 0.1 ± 0.02, and the overlap of swapped and perturbed samples
    on A is exactly 0. All four runs fall inside (table below): the perturbation share is 0.2986 (A s0) and 0.3005 (A s1) and exactly 0 on both REP runs, the swap
    share lies between 0.0999 and 0.1013, the overlap is 0 on A, and the composed stream is 1 on A and 0 on REP.
  * **K-PAIR.** argv(A_s) equals argv(DAG-OACC_s) except `--out`, the six perturbation flags and `--ego-perturb-with-dagger`; argv(REP_s) equals argv(DAG-OACC_s) except `--out` and
    `--seed` (10 vs 0, 11 vs 1). So REP − CTRL is a measured draw of **training variance** on this rig (`H-ESTIM-SEED-1`), the one thing the episode-cluster bootstrap does not see.
  * **K-BUILD** (launch commit `8c24823f` against `aeb54468`, the commit DAG-OACC trained at): 37 changed hunks in 2 files (`v8f_mini_train.py` 25, `v8f_mini_data.py` 12), every
    one behind `--ego-perturb-with-dagger`, default off, bit-identity test PASS; `v8f_mini.py` and `v8f_dagger.py` byte-identical. Record
    `kbuild/kbuild_r2_launch_8c24823f.json`, sha256 `b1c5c36b…`.
  * **K-P0 and K-ZERO.** K-P0: RF0's inputs match their `PINS.txt` sha256, and RF0's frozen sha256 (`1608d70b…`) predates every A / REP / B rollout's mtime (PREREG §6). K-ZERO:
    REP − REP reads exactly 0.0 with CI [0, 0] on every row, and feeding REP's dump as CTRL makes RF exactly 0 (PREREG §6; `controls_PASS` keys `K-ZERO:REP-REP` and
    `K-ZERO:RF-of-REP-as-CTRL`).
  * **The final four runs** (box-2 chain log `chain_v8f_dagoaccorc_r2.out`, the per-run lines and the `summary:` line `10-28 07:44 summary: M2b_s0_dagoaccorc=380b230d0603
    M2b_s0_dagoaccorcrep=b03fd6efebec M2b_s1_dagoaccorc=870dded74c71 M2b_s1_dagoaccorcrep=77853e564f52`; checkpoint sha256 from `SHA256SUMS.txt`, verified by `sha256sum -c`; the T1-kin
    rollout done-markers carry the same sha256):

<<OR_FINAL>>

* **Floors and bars [M]** (`floors`). `bar_r = 2 × max(F_r, RF0_r, RF_r)`, one value per row, applied at both seeds, inclusive. F is 0.1207 m on ADE6 and 0 elsewhere. RF0 is
  from the banked DAG-0 / ORC-0 pair (frozen `P0_RF0.json`, sha256 `1608d70b…`). RF is the maximum over seeds of |REP − CTRL| (`RF_REP.json`, written before any A contrast).

<<OR_FLOOR>>

  The REP pair sets the bar on 15 of 19 rows (every row of C among them); RF0 sets it on 4 rows (DK-VIOL, cross-track MAE, S-POS@50 and S-POS@80). **REP s0 sets RF on 17 of 19 rows**;
  REP s1 sets it on cross-track MAE and S-POS@80. On the four crawl rows the bar is the replicate's own draw doubled: 31.7106 on yaw-rate MAE (RF 15.8553), 0.2648 on curvature
  MAE (RF 0.1324), 0.1740 on L2 (RF 0.0870) and 0.0776 on M1 (RF 0.0388).
  **The replicate beside A.** REP differs from CTRL in the trainer seed alone (K-PAIR). A − CTRL and REP − CTRL, all 19 rows (T1-kin; point [95 % CI]; B / W = separated-better /
  separated-worse, – = not separated; row units are the reader's own):

<<OR_AC>>

  With nothing but the trainer seed changed, REP − CTRL reads separated on 15 of 19 rows at s0 and 2 of 19 at s1.
* **The four families [M]** (T1-kin, the primary tier; per seed, per family, never pooled; point [95 % CI]; B / W = separated-better / separated-worse, – = not separated; error rows
  are lower-better, TAC rows higher-better). Estimator: paired episode-cluster bootstrap, B = 2,000, seed 0, 141 episodes, per seed (`four_families.A.<seed>.<family>.rows`; each family block
  carries the same tier and estimator, and its leaves equal `read_a` cell for cell). A − CTRL is also A − DAG-OACC, cell for cell. P10 is the kinematic floor. For the three S-POS rows the P10
  contrast is each run's own S − CTRV, unpaired across runs, and the A − CTRL leaf is the difference of the two runs' (S − CTRV, ticks ≥ 1) contrasts on the same windows.

<<OR_FAM>>

  **STRATEGIC is reported, not paused, for v8F** (`four_families.A.s0.STRATEGIC.ruling`: *"NOT paused for v8F (PI ruling 2026-10-17)"*). The T-head macro-F1 values behind R-TLAT and R-TLON
  are not carried in the JSON, only the booleans (held), so they are UNVERIFIED as numbers. What was NOT RUN or NOT BUILT, per family (work items and never passes; the outcome does not use any
  of them):
  * **LONGITUDINAL.** The T0 rows and distance keeping at T0 (the T0 column of the families table in the design above) are NOT RUN. The anti-echo block (`longitudinal_claim_admissible`,
    hold-v0, copy detector) is NOT BUILT, so the longitudinal rows above are not checked against a hold-v0 copy. `M-LAUNCH` (A − CTRL of the mean (v_own − v_GT) at ticks 1-2 in the GT-launch windows,
    from the tick dumps) is NOT BUILT.
  * **LATERAL.** The T0 lateral rows are NOT RUN. `M2_and_L2_launch_subset` (M2 and L2 on the GT-launch subset) is NOT BUILT.
  * **TACTICAL.** The T0 tactical rows are NOT RUN. At T1-kin the goal-head F1 and head accuracy are not computable by design (the paired tool reads eval `run` dumps, not tick dumps), and
    anchor selection is not computable by design (one path per window).
  * **STRATEGIC.** The T0 strategic rows are NOT RUN. At T1-kin, paired S-POS across arms is not computed (the leaf above is the difference of two runs' contrasts).
  * **T0, all four families.** `t0_four_families.status` reads **NOT RUN**. The PREREG registered the T0 families as a build item of the reader (`orc_families.py` pointed at A / CTRL), and the
    reader did not carry it. The only T0 numbers in this read are the R-T0 guard points, T0 own-code ADE6 A − CTRL +0.1106 m (s0) and +0.0722 m (s1), with no interval in the JSON. The inputs
    exist (the pulled T0 evals of A and of CTRL, on E:). **Nothing is claimed for T0 LONGITUDINAL, LATERAL or TACTICAL here.**
* **What the numbers say [M]** (beside the word; item 1 is the table's; the shares, ranges and counts are computed from the tables above and from the read JSON):
  1. **HARMS rests on two ADE6 guards.** R-ADE fails at s0: T1-kin ADE6 A − DAG-OACC is +0.1434 [+0.0538, +0.2342] m (separated-worse; means 2.3855 against 2.2421), and +0.0565
     [−0.0115, +0.1255] m at s1 (not separated). R-T0 fails at both seeds: T0 own-code ADE6 A − CTRL is +0.1106 m (s0) and +0.0722 m (s1) against a bar of +0.0604 m, with no interval in the
     JSON. R-TLAT and R-TLON held at both seeds, and R-NEW held (0 of 15 rows at each seed). R-NEW fires on a row when A − P10 is separated-worse, DAG-OACC − P10 is not, and A − CTRL is
     separated-worse with the worsening at least the bar:

<<OR_NEW>>

  The closest approach is DK-VIOL at s0: A − CTRL +0.0185 is separated-worse and above its bar (0.0100), but R-NEW does not fire there because A − P10 on DK-VIOL is not separated-worse
  (+0.0202 [−0.0080, +0.0536] –).

  2. **The crawl rows move against DAG-OACC, and no row reaches its bar.** R-MOVE needs A − CTRL separated-better and an improvement (−delta) of at least the bar; R-CRAWL needs A − P10 not
     separated-worse. The crawl set C at both seeds:

<<OR_CRAWL>>

  Every one of the eight cells is separated-better against CTRL, and every one is below its bar: the largest is 33.8 % of the bar (M1 at s0) and the smallest 8.8 % (yaw-rate at s0). Against
  P10, seven of the eight stay separated-worse; only yaw-rate at s0 ends not separated-worse.

  3. **The replicate floor sets the bar, and A's crawl effects sit inside it.** REP s0 sets RF on 17 of 19 rows, and A's crawl effects are 8.8–33.8 % of the bar. The A − CTRL effects, set
     beside the floor they had to clear (size = |A − CTRL|; the bar is 2 × RF on all four crawl rows):

<<OR_REPS>>

  Two readings, arithmetic only. (i) On the four crawl rows A − CTRL is 17.5 % to 67.5 % of RF, so each is smaller than the REP s0 draw on the same row. (ii) REP s0 − CTRL on ADE6 is
  +0.4025 m, 3.33 × F (0.1207 m); the s0 cost A − CTRL on ADE6 that failed R-ADE, +0.1434 m, is 36 % of that replicate draw. REP s1 − CTRL on ADE6 is +0.0116 m. The bar is one value per row
  from the larger draw by registration (PREREG §3, *Two choices*), so a small s1 draw does not lower it. R-ADE is registered without a floor term (not separated-worse), so it fired on a cost
  smaller than the replicate's own draw; it is reported as written. §7.25 named the missing per-row replicate floor as an instrument gap: this is that floor, and on yaw-rate MAE, the row its
  newly-worse clause fired on, REP s0 − CTRL is +15.8553.

  4. **The share of the CTRL − P10 crawl gap that A closes** (beside; decides nothing; from the reader's deltas, (CTRL−P10 − (A−P10)) / (CTRL−P10)):

<<OR_SHARE>>

  A closes 27.6 % to 79.9 % of the gap that DAG-OACC leaves to P10 on the crawl rows (largest: yaw-rate at s0; smallest: M1 at s1). The two forms differ by at most 0.9 points, because the
  CTRL − P10 and A − P10 cells differ by one window on 6 of the 8 rows (the `n` fields). The bars were fixed before any A row was read.

  5. **By family, A − CTRL at T1-kin** (read off the A − CTRL table above; never pooled; the crawl share M1 is a Read-A row outside the four family tables, see item 2):
     * LONGITUDINAL: the separated-worse cells are along-track MAE at s0, DK-VIOL at s0 and accel MAE at s1, so none at both seeds; speed MAE, DK-HW and DK-TG are not separated at either seed.
     * LATERAL: heading MAE, yaw-rate MAE, curvature MAE and L2 are separated-better at both seeds; cross-track MAE is separated-worse at both seeds (+0.0371 / +0.0296 against a bar of 0.0640);
       L1 lateral-accel MAE is not separated at either seed.
     * TACTICAL: TAC-LAT is separated-better at both seeds (+0.0213 / +0.0242) and TAC-LON is not separated; R-TLAT and R-TLON held.
     * STRATEGIC: S-POS@30 is separated-worse at s0 (+0.2321 [+0.0609, +0.4341]) and no other S-POS cell is separated.
  6. **The longitudinal cost at s0 (A − CTRL).**

<<OR_LONG>>

  At s0 all three are separated-worse against CTRL; at s1 none is. Against the floors: ADE6 +0.1434 is below its RF (0.4025) and its bar (0.8050); along-track +0.1188 is below its RF
  (0.4596); DK-VIOL +0.0185 is above its RF (0.0017) and above its bar (0.0100, which RF0 sets), and R-NEW did not fire on it. At s1 the longitudinal family carries one separated-worse row,
  accel MAE (**+0.1064 [+0.0031, +0.2287]**).

  7. **The launch itself is unmeasured: M-LAUNCH and M2 are NOT BUILT.** `M-LAUNCH` (PREREG §4) is the cheapest discriminating row, and the PREREG's reading of it is *"If R-MOVE fails while M-LAUNCH
     moves, the speed deficit was repaired and the crawl rows did not follow. That outcome points away from the state channel."* M2 and L2 on the GT-launch subset are NOT BUILT as well. So this
     read cannot say whether the speed deficit at launch was repaired. What it does say is that the crawl rows improved against DAG-OACC (item 2), by 8.8–33.8 % of their bars.
  8. **The registered risk happened at T0.** On M2b the lever cost T0 ADE6 +0.1103 / +0.0849 m (the known risk above); on DAG-OACC the T0 own-code A − CTRL points are +0.1106 / +0.0722 m, above the
     +F/2 bar of 0.0604 m at both seeds. T0 is a world-model diagnostic, and the T0 four families that would show where the cost comes from were not run (families item).
  9. **Scope, as the RESULT states it.** One training run per arm and seed: the two seeds and the REP pair carry the training-run question, and no row here is an effect of the lever on its own
     (`H-ESTIM-SEED-1`). There is no lateral / longitudinal split of the perturbation's two components, and neither half is credited. B is not read: nothing here says what p = 0.1 does.
* **Next (Rule Zero).**
  * **A is not adopted, and the committed next is arm B, blocked on the PI's go.** The table's committed next, verbatim: *"A: not adopted; read B (p 0.1) under this same table (B is NOT queued: it needs its
    own PI go, A1 section 4)."* The JSON's `cascade.b_status` reads *"PENDING: train B (p = 0.1) through its gate and read it under this same table"*. What unblocks it: the PI's go, then box 2 for two runs
    through the launch gate, **[E]** about 1.4 h of box 2 (A's two trainings took 2286 s and 2293 s, `summary.json`, plus about 4 min per run for gate and eval, log minutes). The T1-kin rollouts and the read are
    dev-box CPU. DAG-OACC stays the adopted base, and the crawl set (yaw-rate, curvature, L2 and M1 against P10) stays open.
  * **What B must clear** (beside the table; **[I]**, the Master Mind's ranking in `RESULT_V8F_DAG_ORC.md` §6, arithmetic on the numbers above and not a forecast). R-MOVE needs, on every row of C at a seed, an
    improvement of at least the bar. The bars are defined from RF0 and the REP pair (PREREG §3), neither of which involves the dose, and the committed next reads B *"under this same table"* (PREREG §7
    also says B's claim carries B's own floors and seeds; if the reader recomputes floors for B, the factors change). A's crawl improvements stayed below RF and below the bar. The bar divided by A's
    improvement is the factor by which B's improvement would have to exceed A's:

<<OR_FACT>>

  B's dose is a third of A's (p = 0.1 against 0.3). **So B cannot reach R-MOVE unless its crawl effect exceeds A's by 3.0–11.4 ×,** at a third of the dose.

  * **Where B's outcomes lead.** Under PREREG §7 and Amendment 1 §3, HARMS of B, NO EFFECT and PARTIAL / ONE SEED all route to NEXT-CRAWL, which on DAG-OACC's states is LAUNCH-INTENT (N0 read NOT NAVMOVE;
    COMPOSE is not met, since DAG-OACC − DAG is not separated-better on the rows of C, §7.23). Only R-MOVE at both seeds routes elsewhere, so B changes the route only if it clears the factors above.
  * **The registered sequel is NEXT-CRAWL, LAUNCH-INTENT.** Its PREREG is a draft (`dagland/launch_intent/PREREG_V8F_LAUNCH_INTENT.md`, on E:, not in the repo). The draft reuses this REP pair, and therefore these
    floors: the bars above are the bars a LAUNCH-INTENT crawl effect has to clear unless its PREREG sets its own, and A's measured effects (item 2) are the reference size. Its REP-reuse licence is
    conditional; this read shows that the REP run records PASS (swap share 0.1013 / 0.0999, perturbation exactly 0, composition flag absent, stream 0). Its other conditions were not checked here (UNVERIFIED).
  * **Unblocked now, 0 GPU, each a registered item of this PREREG that did not get built** (work items, not done in this package): `M-LAUNCH` from the tick dumps on F:
    (`F:/TanitAD-scratch/dagorc_t1_out/ticks/`); the T0 four families (`orc_families.py` on the pulled `eval_*.npz` of A and CTRL); the anti-echo block; and M2 / L2 on the GT-launch subset.
* **Provenance.**
  * **Package.** The repo holds `raw/2026-10-28-dag-orc-result/`: the read JSONs (`read/DAGORC_READ.json`, sha256 `70a7a3a4…`; `read/RF_REP.json`, `64c9444d…`), the T1-kin chain log
    (`logs/chain.out`, `db5fa5a8…`) and reader log (`logs/dagorc_read.out`, `5a8679b7…`), the box-2 chain logs of every launch (the final one `box2/logs/chain_v8f_dagoaccorc_r2.out`, `d81f2d74…`), the pin
    launcher, the four run records, the K-BUILD record, `notes/R2C_NOTE.md` and the audit. The reader `dag_orc_read.py` (`2e5f9f17…`), the frozen `P0_RF0.json` (`1608d70b…`) and `R2A_NOTE.md` were already in the repo.
  * **Run times.** Final launch r2c 06:09–07:44 on box 2 (`ZZV8FMINI-DONE`); the T1-kin rollouts ran on the dev-box CPU; the reader ran 07:54:18–07:58:54 (rc 0, 275.4 s). The PREREG read for the
    RESULT is the file at `4c412181` (sha256 `d8b8885b…`, Amendments 1 and 2 included); the reader pins the PREREG it read (`38bc688b…`, the file at `3b7b404e`) and Amendment 1 (`f2f96157…`) in
    `provenance`. Amendment 2 changes the build and the re-run, not a row, a bar or a floor.
  * **Checkpoints and dumps (single copies, not in the repo).** The checkpoints (`ckpt_last.pt`: A `380b230d…` / `870dded7…`, REP `b03fd6ef…` / `77853e56…`) and the pulled T0 evals are on E:. The per-episode
    dumps and the tick arrays are on F: scratch only (`F:/TanitAD-scratch/dagorc_t1_out/`).
"""

SEC = SEC_SR + "\n" + SEC_D2 + "\n" + SEC_OR
for name, rows in TABLES.items():
    tok = "<<" + name + ">>"
    assert SEC.count(tok) == 1, tok
    SEC = SEC.replace(tok, "\n".join(rows))
assert "<<" not in SEC and ">>" not in SEC.replace("→", "")

# ---- edits --------------------------------------------------------------------------------------------------------------------
# status line
OLD_ST = "research report, v31 of 2026-10-27, results through repository tip `569a9d9b` (v30: `93aa1956`; branch"
NEW_ST = f"research report, v32 of 2026-10-28, results through repository tip `{TP}` (v31: `569a9d9b`; v30: `93aa1956`; branch"
OLD_ST2 = "The PI's bar of \"dominant driving results\" is **not met** (§8.3)."
NEW_ST2 = (OLD_ST2 + " v32 adds the SPEEDREAD read (§7.24), the DAgger iteration 2 read (§7.25) and the DAG-ORC read (§7.26, HARMS).")

# abstract
OLD_NEXT = ("* **What is next.** The open gap is the crawl set. DAG-ORC re-based on DAG-OACC (`PREREG_V8F_DAG_ORC` Amendment 1, `6cd741b6`) is aimed at it, and SPEEDREAD"
            " (`PREREG_V8F_SPEEDREAD`) asks whether the speed information O needs is present in T's own input. Both are pre-registered and not yet read, and no"
            " number for either is reported here.")
NEW_NEXT = (
    "* **SPEEDREAD reads NOT IN INPUT (§7.24).** The question was whether the per-window speed information O needs is present in T's own input. A ridge"
    " read-out of T's five speed cells (the PRIMARY) and a 2 × 256 MLP (the ALT), both fitted on 306,566 windows of 4,512 TRAIN clips and substituted in the"
    " code O reads, leave ADE6 where it was: −0.0097 [−0.0871, +0.0620] m at seed 0 and +0.0022 [−0.0534, +0.0564] m at seed 1, against a bar of −0.1207 m;"
    " the ALT costs +0.2122 / +0.2040 m (separated-worse). The ground-truth cells, a T0 ceiling and a control, would cut ADE6 by 0.7164 / 0.6676 m, so the"
    " read-outs capture 0.0135 / −0.0033 of what the cells are worth (CAPTURE). Tier: T1-class open loop; paired episode-cluster bootstrap, B = 2,000, 141"
    " episodes; one read-out fit serves both seeds. The negative is scoped to the pooled four-frame summary.\n"
    "* **DAgger iteration 2 reads HARMS (§7.25).** DAG's own states used a second time (T1-kin, two seeds): the crawl rows (curvature, L2, M1, and at seed 1"
    " yaw-rate) stay separated-worse against P10 and do not move against the control (R-MOVE fails at both seeds). The T-head, T1-kin ADE6 and T0 guards"
    " hold, and the guard that fires is a newly separated-worse yaw-rate MAE at seed 0. DAG stays the base.\n"
    "* **DAG-ORC (O-RECOVER's perturbation composed onto DAG-OACC) reads HARMS, and its crawl effects are smaller than the replicate floor (§7.26).** Arm A (p = 0.3; T1-kin,"
    " two seeds; paired episode-cluster bootstrap, B = 2,000, 141 episodes; all 162 reader controls PASS) fails the position guards: T1-kin ADE6 against DAG-OACC is +0.1434"
    " [+0.0538, +0.2342] m at seed 0 (separated-worse) and +0.0565 [−0.0115, +0.1255] m at seed 1, and T0 ADE6 is +0.1106 / +0.0722 m against a bar of +0.0604 m. All eight crawl"
    " cells (yaw-rate, curvature, L2 and M1, both seeds) are separated-better than DAG-OACC, but each reaches only 8.8–33.8 % of its bar, so R-MOVE fails at both seeds. The replicate"
    " arm REP (the same recipe, trainer seed 10 / 11) differs from DAG-OACC by +0.4025 m of ADE6 at seed 0 (3.33 × F) and sets the floor on 17 of 19 rows. A is not adopted and DAG-OACC"
    " stays the base. The T0 four families are NOT RUN, and M-LAUNCH, the anti-echo block and M2 / L2 on the launch subset are NOT BUILT. Box 2 was used for 4.93 h by the logs"
    " against the ~3 h asked, after four stopped launches (a shared RNG stream, a shared aside marker, a self-test that replaced the box-2 queue, and a disk floor).\n"
    "* **What is next.** The open gap is the crawl set (yaw-rate, curvature, L2 and M1 against P10 for DAG-OACC). DAG-ORC (§7.26) moved those rows against DAG-OACC but not past"
    " their bars. Its committed next is arm B (p = 0.1) under the same table, which needs the PI's go, and B would have to exceed A's crawl effect by 3.0–11.4 × at a third of the"
    " dose to clear the same bars; if it does not clear them, the registered branch is NEXT-CRAWL, LAUNCH-INTENT (its PREREG is a draft). The speed line that SPEEDREAD"
    " (`PREREG_V8F_SPEEDREAD`) asked about routes to perception (`PREREG_V8F_LORA` on the A40), which waits on the PI, and nothing on box 2 remains on it.")

# 8.1
OLD_811 = "The open gap is the crawl set (yaw-rate, curvature, L2 and M1 against P10); the arms aimed at it are pre-registered and not yet read."
NEW_811 = ("The open gap is the crawl set (yaw-rate, curvature, L2 and M1 against P10); the arm aimed at it, DAG-ORC re-based on DAG-OACC, reads HARMS and is not adopted (item 14, §7.26).\n"
           "12. **The speed information O needs is not recoverable from T's pooled four-frame input by a ridge or a 2 × 256 MLP** (§7.24; T1-class open loop,"
           " paired episode-cluster bootstrap). The PRIMARY read-out leaves ADE6 where it was (−0.0097 / +0.0022 m against a bar of −0.1207 m), while the T0"
           " ceiling of the true cells is −0.7164 / −0.6676 m. The scope is that summary on unseen clips: T's full 40-token view is not covered, and T fails"
           " on that view too (SPEEDCAL), so the two negatives together route the speed line to perception.\n"
           "13. **A second DAgger iteration on DAG's own states does not move the crawl rows** (§7.25; T1-kin). R-MOVE fails at both seeds, the crawl rows stay"
           " separated-worse against P10, and the swap costs acceleration (+0.5846 / +0.6034 m/s², separated-worse against DAG2-0). DAG stays the base.\n"
           "14. **O-RECOVER's perturbation composed onto DAG-OACC costs position and does not clear the crawl rows** (§7.26; T1-kin, with R-T0 at T0; paired episode-cluster"
           " bootstrap, B = 2,000, 141 episodes). Arm A reads HARMS: R-ADE fails at seed 0 (A − DAG-OACC ADE6 +0.1434 [+0.0538, +0.2342] m, separated-worse) and R-T0 fails at both"
           " seeds (+0.1106 / +0.0722 m against +0.0604 m). R-MOVE is false at both seeds: all eight crawl cells are separated-better against DAG-OACC, and each is 8.8–33.8 % of its bar"
           " (on the crawl rows the bar is twice the replicate arm's draw: REP s0 − DAG-OACC ADE6 is +0.4025 m, and REP − DAG-OACC is separated on 15 of 19 rows at seed 0). A closes"
           " 27.6–79.9 % of DAG-OACC's gap to P10 on the crawl rows, and seven of the eight cells stay separated-worse than P10. DAG-OACC stays the base; arm B (p = 0.1) needs the PI's go.")

# 8.3
OLD_83 = "**Update (v31).** On the model's own motion (T1-kin), the adopted base is now DAG-OACC, and its open rows are lateral and crawl rows:"
NEW_83 = ("**Update (v32).** Three reads are written up here, and the adopted base is unchanged (DAG-OACC):\n"
          "* **SPEEDREAD (§7.24) reads NOT IN INPUT** (T1-class open loop). A ridge or a 2 × 256 MLP read-out of T's five speed cells, fitted on 4,512 TRAIN"
          " clips from T's pooled input, does not move O's ADE6 (−0.0097 [−0.0871, +0.0620] / +0.0022 [−0.0534, +0.0564] m against −0.1207 m). Even the true"
          " cells (T0 ceiling, a control) leave the all-speed yaw-rate and curvature MAE not separated, so the speed line reaches L2 of the open crawl set and"
          " not yaw-rate or curvature (**[I]**, the Master Mind's ranking, scoped to M2b-mixo; M1 was not computed on this rig, UNVERIFIED). Its committed"
          " next, perception on the A40, waits on the PI.\n"
          "* **DAgger iteration 2 (§7.25) reads HARMS** (T1-kin): the crawl rows did not move, and the yaw-rate guard fired at seed 0.\n"
          "* **DAG-ORC (§7.26) reads HARMS** (T1-kin; R-T0 at T0). Arm A (O-RECOVER's perturbation, p = 0.3, composed onto DAG-OACC) fails R-ADE at seed 0 (A − DAG-OACC ADE6"
          " +0.1434 [+0.0538, +0.2342] m) and R-T0 at both seeds (+0.1106 / +0.0722 m against +0.0604 m), so it is not adopted. The crawl rows improve against DAG-OACC (all eight"
          " cells separated-better) but only to 8.8–33.8 % of their bars, which are twice the replicate arm's draw, so R-MOVE is false at both seeds. Against P10, curvature, L2 and M1"
          " stay separated-worse at both seeds, and yaw-rate at seed 1, so the crawl set is still open. The committed next, arm B (p = 0.1), needs the PI's go; the T0 four families are NOT RUN."
          " Box 2 was used for 4.93 h by the logs against the ~3 h asked (four launches stopped before the one that completed).\n\n" + OLD_83)

# 8.4 rows
OLD_1W = "| **pre-registered, not yet read**. Launch blocker: the composition flag and its reachability proofs |"
NEW_1W = ("| **HARMS** (§7.26; `RESULT_V8F_DAG_ORC.md`, 162 reader controls PASS, T1-kin; R-T0 at T0): R-ADE fails at s0 (A − DAG-OACC ADE6 +0.1434 [+0.0538, +0.2342] m) and R-T0 at both"
          " seeds (+0.1106 / +0.0722 m against +0.0604 m); R-MOVE is false at both seeds (0 of 4 crawl rows clear; the eight cells are separated-better against DAG-OACC at 8.8–33.8 % of"
          " their bars, set by the replicate arm: REP s0 − DAG-OACC ADE6 +0.4025 m). A is not adopted; DAG-OACC stays the base. The T0 four families are NOT RUN; M-LAUNCH, the anti-echo"
          " block and M2 / L2 on the launch subset are NOT BUILT. Next: arm B (p = 0.1) under the same table, which needs the PI's go; then NEXT-CRAWL, LAUNCH-INTENT |")
OLD_1X = ("**pre-registered, not yet read** (Amendment 1, `A1-2026-10-27`, before any SPEEDREAD number). Commissioned by FLAT-DAG's committed next"
          " (`RESULT_V8F_FLATDAG.md` §4 (a)); it runs first because it is the cheapest experiment that separates the speed being in T's input from its not being there |\n"
          "| 2 | **LoRA on DINOv3**")
NEW_1X = ("**NOT IN INPUT** (§7.24; `RESULT_V8F_SPEEDREAD.md`, 19 / 19 control flags PASS, T1-class open loop): ADE6 read − own −0.0097 [−0.0871, +0.0620] /"
          " +0.0022 [−0.0534, +0.0564] m against −F = −0.1207 m; ALT +0.2122 / +0.2040 m (separated-worse); T0 ceiling −0.7164 / −0.6676 m (a control). Amendment 1"
          " (`A1-2026-10-27`) landed before any SPEEDREAD number. Commissioned by FLAT-DAG's committed next (`RESULT_V8F_FLATDAG.md` §4 (a)); it ran first because it is the"
          " cheapest experiment that separates the speed being in T's input from its not being there. Next: perception on the A40 (waits on the PI); nothing on box 2 |\n"
          "| 1y | **DAGGER2**: DAG fine-tuned 3,000 steps at `--dagger-p 0.1` on the iteration-2 bank (DAG's own T1-kin states), against DAG2-0 (`--dagger-p 0`) |"
          " `PREREG_V8F_DAGGER2` outcome table (Amendment 1, `A1-2026-10-27`): FIXED / PARTIAL / NO EFFECT / HARMS / BLOCKED / REFUSED. R-CRAWL and R-MOVE on the crawl"
          " set; guards R-TLAT, R-TLON, R-ADE, R-T0 (point ≤ +0.0604 m) and no newly separated-worse row against P10 | **HARMS** (§7.25; `RESULT_V8F_DAGGER2.md`, 199 / 199"
          " reader controls PASS, T1-kin): R-CRAWL and R-MOVE fail at both seeds; the yaw-rate guard fires at s0. DAG stays the base. Next: DAG-ORC (row 1w) |\n"
          "| 2 | **LoRA on DINOv3**")

ANCHOR_8 = "single copy, not in the repo.\n\n---\n\n## 8. Discussion"
assert s.count(ANCHOR_8) == 1
E = [(OLD_ST, NEW_ST), (OLD_ST2, NEW_ST2), (OLD_NEXT, NEW_NEXT),
     (ANCHOR_8, "single copy, not in the repo.\n\n" + SEC.rstrip("\n") + "\n\n---\n\n## 8. Discussion"),
     (OLD_811, NEW_811), (OLD_83, NEW_83), (OLD_1W, NEW_1W), (OLD_1X, NEW_1X)]
for a, b in E:
    assert s.count(a) == 1, (s.count(a), a[:80])
    assert h.count(a) == 1, ("html", h.count(a), a[:80])
    s = s.replace(a, b)
    h = h.replace(a, b)

# ---- guards on the inserted text ----------------------------------------------------------------------------------------------
ins = "\n".join(b for _, b in E)
low = ins.lower()
assert "closed loop" not in low and "closed-loop" not in low and "drives" not in low
for w in ("void", "superseded", "retracted", "formerly", "no longer", "correction", "withdrawn"):
    assert re.search(r"\b" + w + r"\b", low) is None, w
assert re.search(r"retract|supersed|withdrawn|was wrong|not the floor|previously read|outdated|i said", low) is None
assert "$" not in SEC and "\r" not in ins
nocode = re.sub(r"`[^`\n]*`", "", SEC)
assert re.search(r"<[A-Za-z/!]", nocode) is None, re.search(r"<[A-Za-z/!]", nocode)
assert "DAGORC_RESULT" not in s and "DAGORC_RESULT" not in h and "@@" not in SEC, "a DAG-ORC placeholder is left"
for stale in ("is launched and not yet read", "launched, not yet read", "No DAG-ORC number is reported in this version", "No DAG-ORC row exists in this version", "marked slots"):
    assert stale not in s and stale not in h, stale
assert s.count("**HARMS: A is not adopted**") == 1 and s.count("reads HARMS, and its crawl effects are smaller than the replicate floor") == 2
assert all(f in SEC_OR for f in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")) and "STRATEGIC is reported, not paused, for v8F" in SEC_OR
assert "296 min = 4.93 h" in SEC_OR and "not yet read" not in SEC_OR and "T1-kin" in SEC_OR and "B = 2,000" in SEC_OR and "141 episodes" in SEC_OR
assert s.count("### 7.24 ") == 1 and s.count("### 7.25 ") == 1 and s.count("### 7.26 ") == 1
assert s.index("### 7.23 ") < s.index("### 7.24 ") < s.index("### 7.25 ") < s.index("### 7.26 ") < s.index("## 8. Discussion")
for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC"):
    for sec in (SEC_SR, SEC_D2):
        assert fam in sec or fam in "\n".join(TABLES["SR_RO"] + TABLES["D2_FAM"]), fam
assert "STRATEGIC is reported, not paused, for v8F" in SEC_SR and "STRATEGIC is reported, not paused, for v8F" in SEC_D2
assert "\r" not in s and "</script" not in s.lower()
assert s.count("**Status:** research report, v32 of 2026-10-28") == 1
# the html: the same edits on the v31 html must equal the template filled with the new md
assert tpl.replace("%%MD%%", s) == h, "html edited in place differs from template + md"

os.makedirs(OUT, exist_ok=True)
open(OUT + "V8F_SCIENTIFIC_REPORT.md", "w", encoding="utf-8", newline="\n").write(s)
open(OUT + "v8f_report.html", "w", encoding="utf-8", newline="\n").write(h)
print("report v32 built", len(s), len(h), "tip", TIP)
