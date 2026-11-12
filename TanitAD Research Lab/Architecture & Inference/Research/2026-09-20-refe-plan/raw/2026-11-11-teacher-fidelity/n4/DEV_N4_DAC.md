# TF-N4 DAC lever, developed on NAVTRAIN (EXPLORATORY, CPU): a map-certified re-aim of the P5-R target repairs 30 of 30 DAC-failing labels, 5 of 5 on RJ5 + P5-R

2026-11-12/13, drafting agent (Sonnet 5.5). Nothing staged, committed or pushed. No navtest token was scored or opened by me; navtest numbers below are quoted from `RESULT_N2_A1.md` only.
Code tip read: `924901b7` (ls-remote verified). Hold: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/teacher_fidelity/n4_dac/`.

## 0. Headline
- **Best navtrain DAC lever: REPAIR-D** = shift the TEACHER TARGET (not the label) along the scorer-computed return vector of the worst outside corner (+ 5 cm margin), re-run the unchanged label pipeline (Tc20s, then P5-R lambda 3.0), repeat up to 5 times, accept the first candidate that is certified (DAC = 1, NC / DDC / TTC / C each not worse than the base label's).
- On RJ5 + P5-R (form R): tracked DAC **98.93 -> 100.00** (5 failing tokens -> 0), PDMS **+1.07 [0.00, +2.39]** paired (B 10,000, seed 20260927, log clusters), **5 / 5** repaired, **0** NC / DDC / TTC / C worse. Pooled over the six banked arms' form-R labels: **30 / 30** repaired.
- The lower bound 0.00 is structural, not a near miss: all 5 failures sit in **2 of 13 logs**, so any resample that omits both reads exactly 0. The navtrain set cannot separate this; it can only show the mechanism works. Navtest (136 logs, 116 zero-DAC tokens) is where it is read, once.
- Wider CERT alone repairs 3 of the 5 (DAC 99.57, PDMS +0.61 [0.00, 1.32]); the 2 none-pass tokens are exactly the ones where the teacher's own plan is out of the area on every candidate. REPAIR-D covers them.
- **Draft card `PREREG_TF_N4_DRAFT.md`: the design, bars, controls, K (n = 45) and cost are complete; it is NOT yet freezable as a run**, because the navtest chain (`n4_nt_chain.py`), the bar analyzer, the pins file and the chain's reachability test are not written (card sec. 13). Section 6 below lists the MM items.

Evidence class for every number: MEASURED (ours), navtrain held-out, 469 tokens / 13 logs, tier T1 (NAVSIM v1.1 non-reactive, LQR-tracked), W-TRANSP DB-rebuilt scenery. **Levels are not harness-identical; contrasts are valid.** Estimator: paired log-cluster bootstrap, `n4_common.boot` (verbatim `tfn2_analyze.boot`), answers "would another draw of LOGS say this", nothing about training or inference variance (the rollouts and labels are deterministic). STRATEGIC: PAUSED (PI 2026-09-19).

## 1. Does DAC on the nuPlan-DB route equal the harness's DAC? (asked in the brief)
**Yes for DAC, by construction and by a measurement on navtest; INHERITED (not re-measured) for navtrain, where no harness cache exists.**
- Code: DAC = 0 iff at some of the 41 steps one of the 4 ego corners lies outside every drivable polygon: `pdm_scorer.py:318-356` (`_calculate_ego_area`) and `:422-429` (`_calculate_drivable_area_compliance`). Its only inputs are the ego corner coordinates of the LQR-simulated state and `drivable_area_map`, which `PDMDrivableMap.from_simulation(map_api, ego_state, 50)` builds from the maps and the ego state (`pdm_closed_planner.py:76`; ROADBLOCK, ROADBLOCK_CONNECTOR, INTERSECTION, CARPARK_AREA). It reads neither the objects nor the route. The W-TRANSP DB rebuild and the harness cache differ only in the objects (`WTRANSP_DIAG.md` sec. 1, 2c).
- Measured (navtest, N1-T rows against the banked harness CSV): "DAC, DDC and C agree on all 12,146 tokens" (`WTRANSP_DIAG.md` line 85); ego state, centerline length and drivable map identical (line 56).
- My own control, not independent: re-deriving every form-R/C/T label of the 12 failing tokens x 6 arms (216 label-scores) reproduces the banked DAC **and** PDMS, 216 / 216 (`Q1_anatomy.json` `control`; `Q3_repairs.json` `control` 30 / 30).
- Residual risk, named: the navtest builder will read the map from the harness metric cache (`--metric-cache-root`, the A1 route). The card's C-CERT control checks that every repaired token's builder DAC = 1 equals its harness DAC = 1.

## 2. Q1. Anatomy of the tracked-DAC failures (`Q1_anatomy.json`, `n4_q1.py`)
Failing form-R labels: RJ5 **5** of 469 (DAC 98.93, comparable to navtest N2-R's 99.04, 116 of 12,110); all six arms 30 instances on 7 tokens; any R or C label of any arm: 12 tokens. Human: 0.

| token (log) | RJ5.R: 1st outside | worst corner outside | return vector (lon, lat) m | teacher plan UNTRACKED outside by | teacher TRACKED | who else passes (DAC = 1) |
|---|---|---|---|---|---|---|
| 634f2012 (veh-40) | 3.0 s | 0.159 m at 3.8 s | (+0.01, +0.16) | 0.136 m | out | nobody (all 15 candidates fail) |
| 85772c23 (veh-40) | 2.4 s | 0.197 m at 3.4 s | (-0.02, +0.20) | 0.234 m | out | R0.R (PDMS 0.90), R0.T (PDMS 0: another factor) |
| a7b14cd0 (veh-50) | 1.7 s | 0.182 m at 1.8 s | (-0.03, +0.18) | 0.186 m | **in** | R0.T, RJ3.T, RJ5.T |
| d7587000 (veh-50) | 1.4 s | 0.030 m at 1.4 s (1 step) | (-0.02, +0.03) | 0.038 m | **in** | RJ3.R, RJ3.C, R5HZ.C, RJ5H5.C, R0.T, RJ3.T, RJ5.T |
| f2c6dba4 (veh-40) | 3.9 s | 0.061 m at 3.9 s | (+0.00, +0.06) | 0.045 m | out | nobody |

- **Cause, all 5 (and all 30 pooled instances): the teacher's own 20-pose plan, executed as given with no tracker, already leaves the drivable polygon**, by 4 to 24 cm (pooled label-tracked maximum: median 0.16 m, max 0.33 m). It is not a tracker artefact and not a P5-R artefact: P5-R faithfully inverts the tracker toward an intent that clips the boundary. In 2 of the 5 (a7b14, d7587) the registered tracker of the unmodified teacher plan happens to pull the corner back inside (T tracked passes), and the inversion removes that pull: that is a way P5-R costs DAC.
- **When / how far:** first outside at 1.4 to 3.9 s (pooled median 2.55 s, range 1.4 to 4.0); **late in the horizon and shallow**. The return vector is lateral-dominant in 29 of 30 pooled instances (one-sided, towards the lane); the teacher is shaving a boundary, not crossing it.
- **Alternatives:** the wider set rescues 3 of 5 (85772 via R0.R, a7b14 via R0.T, d7587 via RJ3.R). It does so by swapping in a different teacher setting, not by fixing the geometry; 634f and f2c6 have no passing candidate in 15, because the teacher's intent is outside on every arm.
- Navtest consequence (INFERRED from navtrain, not measured): the 79 navtest none-pass tokens (RESULT_N2_A1) are the same class, which is why no candidate swap reaches them.

## 3. Q2. Certified selection over every banked candidate (`Q2_cert_wider.json`, `n4_q2.py`)
**Order rule frozen before computing:** `CERT_ORDER_RULE.md` (written 22:58, before the first `n4_q2.py` run). 15 candidates, ORDER = RJ5.R, RJ5.C, R0.C, RJ3.R, RJ3.C, R0.R, R5HZ.R, RJ5H5.R, RG8.R, R5HZ.C, RJ5H5.C, RG8.C, R0.T, RJ3.T, RJ5.T (extends N2-A1's CERT2 order). CERT-A: first with DAC = 1. CERT-B: first with DAC = NC = DDC = 1. CERT-4: the N2-A1 CERT2 analogue [RJ5.R, RJ5.C, R0.C, R0.T].

| rule | none-pass | zero-DAC | PDMS | DAC | dPDMS vs RJ5.R [95 % CI] | dDAC | worse flips (NC/DAC/C/DDC) |
|---|---|---|---|---|---|---|---|
| RJ5.R alone | 5 | 5 | 95.00 | 98.93 | | | |
| CERT-4 | 2 | 2 | 95.41 | 99.57 | +0.41 [0.00, 1.12] | +0.64 [0.00, 1.37] | 1 / 0 / 0 / 0 |
| **CERT-A** | **2** | 2 | 95.61 | 99.57 | **+0.61 [0.00, 1.32]** | +0.64 [0.00, 1.37] | 0 / 0 / 0 / 0 |
| CERT-B | 6 | 2 | 95.72 | 99.57 | +0.72 [-0.01, 1.42] | +0.64 [0.00, 1.37] | 0 / 0 / 0 / 0 |
| ORACLE (max PDMS among DAC = 1; descriptive upper bound, never a bar) | 2 | 2 | 96.69 | 99.57 | +1.69 [0.90, 2.42] | +0.64 | |

Wider CERT beats the 4-candidate form by +0.20 PDMS and reaches the same DAC; it leaves **2 none-pass** (CERT-A), the tokens of Q1 where no candidate passes. CERT-A picks R0.R / R0.T / RJ3.R for 3 tokens and RJ5.R for 464: mixing teacher settings and the raw T seam on 3 tokens is a rollout-cost and provenance complication, not a cheap lever. (CERT-B's 6 none-pass is because it also demands NC and DDC: 4 tokens pass DAC but score PDMS 0 through another factor.)

## 4. Q3. Repairs (`n4_repair.py`, `Q3_repairs.json`, `Q3_read.json`, `Q3b_exec_shift.json`)
Candidates, cheapest first (each re-tracked by the same LQR tracker and certified by the same scorer call; the first certified in a fixed smallest-change order is taken): **B** blind lateral offset (ramped, +-0.1..0.5 m); **H** heading nudge (+-0.5..3 deg about the ego origin); **A** informed shift of the LABEL along the return vector; **D** the same informed shift on the TEACHER TARGET, then Tc20s + P5-R re-run (the "DAC margin in the P5-R solve": P5-R inverts the tracker toward a margin-bearing target). Run on all 30 failing form-R instances (6 arms, 7 tokens). Certificate: DAC = 1 and NC, DDC, TTC, C not worse than the base label's.

| repair | RJ5.R repaired | pooled form-R repaired | NC / DDC / TTC / C worse | executed-path shift vs the base label's execution (median, max) | PDMS of repaired tokens |
|---|---|---|---|---|---|
| B blind offset | 2 / 5 | 14 / 30 | 0 | 0.16 m, 0.34 m | 0.996 |
| H heading nudge | 3 / 5 | 21 / 30 | 0 | 0.20 m, 0.78 m | 0.994 |
| A informed, label | 2 / 5 | 11 / 30 | 0 | 0.16 m, 0.29 m | 0.995 |
| **D informed, target + P5-R** | **5 / 5** | **30 / 30** | **0** | **0.23 m, 0.58 m** | 0.992 |

Base PDMS of every failing token is 0 (DAC multiplier). Repaired PDMS is 1.00 on all 5 RJ5 tokens (EP 1.0, TTC 1.0, C 1.0). For scale, the unrepaired execution already differs from the teacher plan by a median 0.05 m, so D moves the executed path by about 5x that tracker-level difference on the failing tokens only (RJ5 D shifts: 0.28, 0.19, 0.11, 0.54, 0.27 m).
- **Why A fails where D works:** shifting the final label does not move the tracker's solution consistently (the tracker deviates 0.7 to 1.8 m from a P5-R/C label's own poses); re-solving the inversion for a shifted target does. Blind and heading repairs are one-parameter and miss the late, one-sided shallow clips (a7b14, d7587).
- Because every repair is gated by the certificate and touches only DAC-failing labels, it cannot change any other token (T4 test: a passing label is returned bit for bit at iteration 0).

### Pipelines on all 469 tokens (paired vs RJ5.R, B 10,000, seed 20260927, 13 log clusters)
| pipeline | repaired by the repair step | zero-DAC | PDMS | DAC | dPDMS [95 % CI] | dDAC | worse flips NC/DAC/C/DDC/TTC |
|---|---|---|---|---|---|---|---|
| RJ5.R | | 5 | 95.00 | 98.93 | | | |
| **P1 = REPAIR-D on RJ5.R** | 5 | **0** | **96.06** | **100.00** | **+1.07 [0.00, +2.39]** | +1.07 [0.00, 2.39] | 0 / 0 / 0 / 0 / 0 |
| P2 = CERT-A, then REPAIR-D on the 2 none-pass | 2 | 0 | 96.03 | 100.00 | +1.04 [0.00, 2.32] | +1.07 | 0 / 0 / 0 / 0 / 0 |
| REPAIR-H on RJ5.R | 3 | 2 | 95.64 | 99.57 | +0.64 [0.00, 2.03] | | 0 |
| REPAIR-A, REPAIR-B | 2 | 3 | 95.42 | 99.36 | +0.43 [0.00, 1.35] | | 0 |
P2 against CERT-A: +0.43 [0.00, 1.35]. **P1 and P2 do not differ (1.07 vs 1.04), and P1 needs no second rollout arm and no raw-T swap: P1 is the proposed candidate.**

### Four metric families (frame `executed_tracked`, form R; `fam_ci.py` paired, `families_run.py --per-window`, STRATEGIC PAUSED)
Baseline RJ5.R against P1 and P2: **61 paired metrics each, 0 separated.** LONGITUDINAL: speed bias +0.00001 m/s (P1), speed MAE +0.00002, accel MAE +0.00001 m/s2, progress ratio -4e-7 (all inside +-0.0004). LATERAL: cross-track MAE -0.0011 m (P1; -0.0035 P2), heading MAE -0.007 deg, yaw-rate MAE -0.0009 deg/s, curvature MAE -1.5e-5 1/m. TACTICAL: no separated change. The shifts are diluted by 464 untouched tokens; the per-token effect is the 0.11 to 0.54 m executed shift above. Tables: `Q3c_fam_summary.json`; inputs in `rows_spliced_and_fam.tar.xz`. Strategic and `goal_bearing_bias_deg` remain NO-PER-TOKEN-DATA / PAUSED.

## 5. Controls and tests (`test_n4_repair.py`, 5 tests OK, 16 s)
Zero shift is bit-identical; REPAIR-D certified and deterministic on a real failing token (two runs, same plan bit for bit); **regression arm: the sign-flipped informed shift never certifies, the correct sign does**; a passing label is returned unchanged by D; the certificate refuses a candidate that is inside the area but worse in NC, TTC, C or DDC. Reproduction control: 30 / 30 base labels re-derived equal the banked score (DAC and PDMS, 1e-6).

## 6. What the card needs from the MM, and what is NOT shown
1. **Power.** The primary arm has 5 failing tokens in 2 of 13 logs. The navtrain contrast cannot separate and the lower bound 0.00 is structural. The lever is a mechanism result (5/5, 30/30, no harm on 5 metrics), not an effect estimate; the card reads the effect once on navtest.
2. **Parameters were set before the data and not tuned:** margin 5 cm, 5 iterations, certificate as above. They were run on the 469 tokens that also served Q1 and Q2, so the navtest read is the first out-of-sample one.
3. **Label provenance.** D produces a label that differs from the teacher's execution by up to ~0.58 m on a few tokens, purely to keep the executed path inside the map. It is a map-derived label: admissible for the privileged teacher (PI 2026-08-03: labels may use anything), never an inference input.
4. **UNVERIFIED:** (a) the navtest builder will take the drivable map and ego state from the harness metric cache; my DAC-equivalence rests on the navtest measurement in WTRANSP_DIAG, not on a navtrain harness cache (none exists); (b) the cost of the full navtest build (ESTIMATED, section 4 of the card); (c) whether the repaired labels help or hurt a student (P5-C's question).
5. **K ledger:** current max n is 44 (TF-N2-A1, `ledger_K.json` at tip `924901b7`). The card proposes n = 45; the MM appends and re-checks the max at freeze.
6. Navtest has already been seen for N2-R and CERT2. This card's lever (D) was developed only on navtrain; its number on navtest is new, but the RJ5 + P5-R baseline it modifies is not.

## 7. Files
`n4_common.py` (loader, bootstrap), `n4_geom.py` (token context: scenery, drivable union, labels, scoring), `n4_repair.py` (B/H/A/D + certificate), `n4_q1.py` `n4_q2.py` `n4_q3.py` `n4_q3_read.py` `n4_q3b.py` `n4_fam.py`, `test_n4_repair.py`, `run.sh` (NAVSIM venv python, pinned devkit 3e8291b, CPU). Results: `Q1_anatomy.json`, `Q2_cert_wider.json`, `CERT_ORDER_RULE.md`, `Q3_repairs.json` (includes the repaired 20-pose labels), `Q3_read.json`, `Q3b_exec_shift.json`, `Q3b_pipeline_choices.json`, `Q3c_fam_summary.json`, `rows_spliced_and_fam.tar.xz` (spliced W-TRANSP rows P1, P2; family JSONs; `fam_ci` outputs). To rerun `n4_fam.py`: extract the tar here first. Inputs are the banked rows tars named in the brief (`n2/reroll/run/rolls_scores.tar.xz`, `n3_dose/run/rolls_scores_RJ3_RJ8.tar.xz`, `n3_combo/run/rolls_scores_RJ5H5.tar.xz`), extracted to scratch.
