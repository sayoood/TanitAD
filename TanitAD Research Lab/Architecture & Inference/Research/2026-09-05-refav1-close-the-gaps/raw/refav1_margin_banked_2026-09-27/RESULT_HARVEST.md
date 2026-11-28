# refav1 p4 arms — the read-back of the 17 stranded records (2026-09-27)

**What was run:** `taniteval/tools/refav1_paired_delta.py` (tip `b3f7ea6f`) on the ORIGINAL dev-box dumps
(`C:/Users/Admin/refav1_margin/p4out/`, untouched; the banked copies here carry sha12 ids), 0 GPU.
Commands: `harvest/run_pd.sh`; outputs `harvest/pd_p4.{md,json}` (40 windows / 8 episode clusters) and
`harvest/pd_ta.{md,json}` (the 75-window turn-asymmetry panel). **Estimator:** paired episode-cluster bootstrap,
n_boot 2,000. **Tier:** `cl` T1 (self-action open loop), floors T1, `ol` T0. **Known-value control: PASS.**
Checkpoint for every arm: `refav1-b1-v72-ep3-speed` step 21,109. ⚠️ Metrics computed with the tip's `taniteval`
(incl. the 2026-09-26 yaw-rate mask), so LAT yaw-rate cells can differ slightly from the 2026-09-05 tables.

## 1. The pre-registered verdict: `kammshift` = FAIL (outcome R)

Bar (`../../SPEC_CLOSE_THE_GAPS.md`, committed before the arm): PASS iff ADE ≤ **0.8892** AND `turn_left`
recall > 0. **MEASURED:** ADE **1.0652** (`rec_kammshift.json`, `legacy_epmean_row.ade_m` — the same field that
reproduces the spec's own references `wk15` 0.8934, `lonshift` 0.7868, `kamm07` 0.9927), `turn_left` recall
0.0909. Validity controls hold: `ha`/`ha0`/`ha0_ext`/`ol` ADE bit-identical to `kamm07`'s
(0.8888 / 0.9251 / 0.8772 / 0.8052). ⇒ **R — FAIL.** Paired: `kammshift − kamm07` ADE +0.0725 [−0.1360, +0.2699];
LON speed −0.0881 [−0.1059, −0.0721] (better); yaw-rate +0.0886 [+0.0452, +0.1283] (**worse**). The spec's own
reading of R stands: *the `a0_shift` gain does not transfer off the penalty base; `lonshift`'s gain was partly
`W_KAPPA`'s.*

## 2. The corrected jerk-seam arms (the follow-up to `D-REFAV1-LON-SEAM-INERT`) — never read until now

`D-REFAV1-LON-SEAM-INERT` recorded that Thor's `T_lonseam` measured nothing because it ran with `W_JERK = 0`.
The corrected arms were run on the dev box with `W_JERK = 0.02` and then stranded:

| pair (T1, 8 clusters) | ADE | LON speed | LON along | LAT |
|---|---|---|---|---|
| `seamon − seambase` (the seam, jerk on) | −0.0937 [−0.2044, +0.0233] | **−0.2360 [−0.3554, −0.1232]** | **−0.1383 [−0.2510, −0.0214]** | n.s. |
| `loncomb3 − lonshift` (jerk 0.02 + seam a0 on top of `lonshift`) | −0.0129 [−0.1187, +0.1226] | **−0.0992 [−0.1887, −0.0164]** | **−0.1073 [−0.1662, −0.0522]** | n.s. |
| inference-seed floor `lonshift_s1 − lonshift` | −0.0076 [−0.0299, +0.0188] | **−0.0279 [−0.0490, −0.0100]** | **−0.0198 [−0.0357, −0.0045]** | n.s. |

⛔ **The seed replicate itself reads "separated" on LON** — a separated CI is necessary, never sufficient
(`H-ESTIM-SEED-1`). Read against it: `loncomb3`'s LON-speed gain is **3.6×** the replicate difference, the seam's
**8.5×**. Other inference-seed floors on this panel: `best_seed1 − best` ADE +0.0015; `combined_seed1 − combined`
ADE **+0.1035** — the floor is strongly config-dependent, so each lever must carry its own replicate.

## 3. The best refav1 configuration measured so far: `loncomb3`, against the do-nothing floors

`loncomb3` = `--cost-metric ccos --cost-weights W_JERK=0.02,W_KAPPA=15.11245,W_VEND=64.297 --a-sustain-mode
a0_shift --jerk-seam a0 --plan-seed 0` (`2026-09-05-refav1-longitudinal/raw/queueLON3.sh:77`). Point estimates:
ADE 0.7739, LON speed MAE 0.4665 — the lowest of all 30 p4 arms; `turn_left` recall 0.

| `loncomb3 − floor` (T1, 8 clusters) | ADE | FDE | LON speed | LON along | LAT heading | LAT yaw-rate |
|---|---|---|---|---|---|---|
| `ha0_ext` (hold measured a0, κ0 — the strongest floor) | −0.1033 [−0.2676, +0.0851] | −0.4052 [−0.8923, +0.1086] | **+0.1608 [+0.0462, +0.2920] worse** | **−0.2660 [−0.4595, −0.0785] better** | **−3.36° [−5.70, −1.02] better** | **−0.1064 [−0.1668, −0.0488] better** |

⇒ Better than doing nothing on along-track position, heading and yaw-rate (separated); **still worse on speed**
(+0.16 m/s, down from `lonshift`'s +0.26); ADE/FDE better as point estimates but NOT separated on 8 clusters.
⚠️ `W_VEND = 64.297` is **inert** in every arm here: `target_speed` is never passed to the planner
(`test_refa_v1_a_sustain.py:148-150` claims otherwise; the arm tool's `.plan(` call passes none).

## 4. What this leaves (Rule Zero: the next arm, not just the verdict)
1. **Full-grid confirmation of `loncomb3` × 2 inference seeds** on the 141-episode / 282-window grid — pre-registered
   in `../../../2026-09-27-refav1-fullgrid-loncomb3/SPEC.md` before any data.
2. **The remaining LON-speed gap is the one the terminal-speed term exists for** — arming `W_VEND` with target
   max(0, v0 + a0·T) is reserved to the PI (`GOALS_AND_CLAIMS.md:7311`).
3. Turning: every `W_KAPPA = 15.11` arm loses left turns; `seamon` keeps 2/11, `wk7` 3/11 at `wk15`'s ADE
   (`wk7 − wk15` ADE +0.0000 [−0.0602, +0.0773]). An exploratory `wk7` + `loncomb3`-levers p4 arm is the cheap probe.
