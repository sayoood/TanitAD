# RL-1: where RL's TTC loss lives, and a zero-training head-wise composite (FINAL: steps 1-4 done, the one navtest open read: N-BOTH)

Research Lab agent (Sonnet), 2026-10-18/19; steps 4c-5 by the RL-1 continuation agent (Sonnet), 2026-10-19. Package target `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-rl-headwise/`. CPU only; no pod, no GPU lock, nothing committed, pushed or registered.
Pre-registrations (each hashed before the numbers it governs): A `PREREG_RL1_A_MECHANISM.md` sha256 `c7b1c1c59b...7335`; B `PREREG_RL1_B_COMPOSITES.md` sha256 `0593d79223...8f58` (full digests in the `.sha256` files; both re-verified unchanged with `sha256sum -c` on 2026-10-19, before the navtest read).

## 0. Headline

1. **Step 1 (MEASURED, RL-val selection tier): RL's flip to a TTC-failing pick lives mostly in the LANE output, not in the safety heads and not in EP/C.** R0 u1000 vs its own arm A, the 59 sets where B's pick fails TTC and A's passes, exact Shapley shares of the pick-pair margin change: lane **0.60** [0.52, 0.66], DAC 0.14, EP 0.13, TTC 0.07, NC 0.06, comfort 0.00; safety group {NC, DAC, TTC} **0.27** [0.22, 0.33], EP+C 0.13. Committed label **MECH-PROGRESS** (safety share CI upper bound < 0.5), same label at all five other states (R0 u250/u500/u750, R0b u250/u500). The earlier hybrids H1-H4 put the lane output on the progress side, which is why H1 recovered only 0.2 of the loss.
2. **The RL TTC head is slightly overconfident on its failing picks** against its own arm A (pass probability +0.018 [+0.010, +0.028] at R0 u1000, +0.039 at u250), but not resolvably against the deployed base0 (+0.015 [-0.002, +0.032]). Per-head drift is moderate (relative within-scene drift vs A: TTC 0.25, NC 0.34, EP 0.35, lane 0.32, DAC 0.53).
3. **Step 3 (RL-val, R0 u1000 vs the deployed base0 = the navtest bar's comparator): both composites read GAIN-LOST-VAL.** V1 = RL{EP, C} + base0{NC, DAC, TTC, DDC, lane}: PDMS **+0.17 [-0.12, +0.40]**, EP +0.64 [+0.40, +0.86] (0.36 of B's +1.76), TTC -0.07 [-0.38, +0.21], NC -0.08 [-0.19, +0.00], DAC -0.10 [-0.36, +0.10]. V2 (same RL heads, own-A safety source): PDMS -0.30 [-0.95, +0.34], EP -0.67. Selected for navtest: **V1** (deficit 0.115 vs 0.951). B itself reads PDMS +0.59 [+0.14, +1.02], TTC -0.17 [-0.71, +0.38] (reproduces the banked text).
4. **The gain and the cost share a carrier.** D3 (descriptive): B with only the lane output from base0 drops PDMS +0.59 -> +0.12 [-0.44, +0.58]; B with EP from base0 keeps PDMS +0.58 [+0.22, +0.94]; B with NC / TTC from base0 keep +0.55 / +0.60. So the lane gate is what carries both the PDMS gain and the TTC flip, and no cut between "progress heads" and "safety heads" separates them at zero training.
5. **Step 4 (navtest T1, MEASURED, the one open): OUTCOME = N-BOTH.** V1 - BASE, x100, 12,146 tokens / 136 logs, paired log-cluster bootstrap: **PDMS +0.10 [-0.07, +0.26]** (lower bound <= 0: gain not resolved), **NC -0.12 [-0.24, -0.02] (the one adverse bar component)**, DAC -0.12 [-0.26, +0.01], EP +0.53 [+0.37, +0.69], TTC -0.13 [-0.32, +0.06], C +0.00 [0.00, 0.00], DDC -0.14 [-0.20, -0.09] (reported, not in the bar). Level PDMS 88.3339 vs BASE 88.2388. All gates true. My committed navtest forecast, N-GAIN-LOST, was wrong in one component (it predicted no adverse one); the RL-val forecasts of PREREG_B section 6 were already wrong at step 3 (scoreboard in 6.7). Section 6.
6. **Committed next arm (PREREG_B section 5, row N-SAFETY-LOST / N-BOTH): R0e-pick** (`--epo-con-temp 0.01`), an A40 arm, **BLOCKED on compute** (P6 holds the A40; the PI declined extra compute). **RL-2's frozen rule for this outcome (`rl2/out/raw/MM_DECISION_NAVTEST2.md`): look #2 for L0.25 = NO.** The two documents name different next arms for this row (R0e-pick vs the anchored RL arm); both are A40-blocked. Section 4 and 6.9.
7. **R0d extension (MEASURED, run by the Master Mind, PREREG_A excluded R0d so this is an EXTENSION; code and rule unchanged): R0d replicates R0.** R0d u1000 vs own A, 55 TTC-flip sets: lane **0.609** [0.535, 0.665], safety 0.255 [0.211, 0.311], EP+C 0.136 [0.111, 0.170]; MECH-PROGRESS at all four states (lane 0.49-0.63, safety 0.23-0.30). Section 2b.

## 1. Evidence class, tier, estimator

* Tiers. **Steps 1-3: RL-val SELECTION tier** (the true NAVSIM v1.1 sub-scores of the argmax pick on 3,000 held-out RL-val sets / 75 logs), never an endpoint, not T1. **Step 4: T1 = NAVSIM v1.1 PDMS on navtest, an ego pseudo-simulation of an open-loop plan against logged agents** (self-action open loop, not closed loop), 12,146 tokens / 136 logs. T0 is not used anywhere. Evidence class MEASURED (banked live bf16 logits; labels `pdm_val` + `terms_val`; input sha256 in `raw/INPUTS_RLVAL.sha256`: c0 u1000 A and r0 u1000 B match `attrib_rls1b.EXPECT`; navtest rows from the W3 harness, section 6.1).
* Estimator: paired log-cluster bootstrap, 10,000 resamples, 95 % percentile, seed 20260927 (RL-val: `rlh.Boot`; navtest: `proxy_eval.boot` via `stage3_full.read_pair`, the FREEPOINT estimator). It answers **"another draw of LOGS" and nothing else**: one training seed per arm, no replicate arm (H-ESTIM-SEED-1), so a separated interval is necessary and not sufficient for any lever claim, and nothing here is a lever-effect claim. Inference is deterministic (argmax of banked logits), so the third variance (inference seed) does not apply. Ratio estimators (shares) drop a resample whose subset is empty. The families6 intervals are unpaired episode-cluster 95 % intervals (2,000 resamples) and are reported, not gating.
* Controls, all passed (`mechanism_r0.json` `controls`, `composites_rlval.json` `controls`): zero control exactly 0.0; Shapley efficiency max abs 3.3e-16 and endpoints 0.0 on every pair; planted EP-only change -> share_progress 1.0 (n 1,100), planted TTC-only -> share_safety 1.0 (n 3,000); banked pick counts reproduced (2,329 / 59 / 29 / 14); PDMS 90.05977 / 91.22299; NC AUC 0.9186 / 0.9271; H1 and H4 of `attribution.json` Q5 reproduced to 8.9e-16; R0-B vs base0 PDMS +0.59 [+0.14, +1.02], TTC -0.17 [-0.71, +0.38] reproduced; all-RL = B and no-RL = SRC bit-exact (both SRCs); deliberate-regression arm (V1 with a permuted SRC) reads dPDMS -4.29 [-5.17, -3.43], so the bar fails it; bootstrap identity with `proxy_eval.boot` max abs 0.0. **Navtest-side controls: section 6.1** (dry known-value test of the reader, same-pick row identity, harness C1/C2/C3).

## 2. Step 1: mechanism (R0 u1000 vs own A unless stated; `raw/mechanism_r0.json`, `raw/mechanism_r0b.json`)

| quantity | value |
|---|---|
| TTC pick sets (B fails, A passes / A fails, B passes) | 59 / 29 (the banked net -30 sets = -1.00) |
| NC pick sets | 15 / 4 |
| within-scene AUC delta B - A | TTC -0.0063 [-0.0131, +0.0005]; NC -0.0085 [-0.0173, -0.0001]; DAC -0.0036 [-0.0092, +0.0016]; EP>=0.99 -0.0013 [-0.0077, +0.0050] |
| relative within-scene drift vs A (rms drift / head's own rms) | NC 0.34, DAC 0.53, EP 0.35, TTC 0.25, C 0.36, DDC 0.30, lane 0.32 |
| TTC head at B's failing pick, pass probability | B 0.937, A 0.918, B - A +0.018 [+0.010, +0.028] OVERCONFIDENT; B - base0 +0.015 [-0.002, +0.032] UNRESOLVED |
| M4 shares, W_ttc (n 59) | safety 0.269 [0.219, 0.332]; EP+C 0.132 [0.109, 0.162]; lane+DDC 0.599 [0.524, 0.658]; per head lane 0.599, DAC 0.139, EP 0.132, TTC 0.066, NC 0.063, C 0.000 |
| M4 shares, Z_ttc (B fixes TTC, n 29) | safety 0.353 [0.269, 0.434]; lane 0.535: the lane gate moves the picks both ways |
| M4 vs base0 (n 31) | MECH-MIXED: safety 0.395 [0.232, 0.509], EP+C 0.318, lane 0.287 |
| consistency | vs own A the label is MECH-PROGRESS at R0 u250 / u500 / u750 and R0b u250 / u500 (lane 0.48-0.62, safety 0.24-0.31); vs base0 MIXED at five states, PROGRESS at one (R0 u750) |

Reading. The drift vs the deployed base0 is large for lane (mean logit +1.23, relative 1.35), C (+1.03) and DDC (+0.95); against own A it is small (-0.10 / -0.11 / -0.02), so that shift belongs to the BCE retrain both arms share, not to EPO. The EPO-specific drift is spread over all heads at 0.25-0.53 relative. Answer to the brief's question: the RL safety head is mildly overconfident on the failing picks, but "the progress head's increase dominates the rule" is also not the right description: the carrier is the multiplicative lane gate (and DAC), then EP.

### 2b. R0d extension (added 2026-10-19; `raw/mechanism_r0d.json`; MEASURED, RL-val selection tier)

PREREG_A excluded R0d because its dumps were not on the dev box. After the PI's go-ahead the dumps were downloaded (`r0d_dl/r0d/`, 43 files, 344.8 MB; sha256-verified against the hub, INHERITED from the Master Mind's note, not re-verified here; the two 1.29 GB SFT checkpoints were skipped) and the Master Mind ran `mechanism.py --arm r0d --dl C:/Users/Admin/par18/rl/r0d_dl --controls` (rc 0). **Code, quantities and label rules are PREREG_A's, unchanged; the arm is added after the fact, so this is an EXTENSION, not part of the registered read, and my step-1 prediction does not apply to it.** I read the JSON; I did not re-run it. Controls all pass (zero control; Shapley efficiency 2.2e-16, endpoints 0.0; planted EP -> share_progress 1.0 (n 1,098); planted TTC -> share_safety 1.0 (n 3,000); bootstrap identity 0.0).

| R0d vs own A, M4 on W_ttc | n W / n Z | safety {NC, DAC, TTC} | EP + C | lane | label | TTC head, B - A pass prob. at B's pick |
|---|---|---|---|---|---|---|
| u250 | 52 / 17 | 0.295 [0.229, 0.384] | 0.213 [0.178, 0.261] | 0.492 [0.375, 0.573] | MECH-PROGRESS | +0.037 [+0.026, +0.048] OVERCONFIDENT |
| u500 | 51 / 36 | 0.272 [0.219, 0.354] | 0.132 [0.108, 0.173] | 0.597 [0.484, 0.665] | MECH-PROGRESS | +0.015 [+0.003, +0.029] OVERCONFIDENT |
| u750 | 51 / 28 | 0.230 [0.183, 0.275] | 0.145 [0.119, 0.177] | 0.625 [0.572, 0.682] | MECH-PROGRESS | +0.018 [+0.009, +0.029] OVERCONFIDENT |
| **u1000** | **55 / 32** | **0.255 [0.211, 0.311]** | **0.136 [0.111, 0.170]** | **0.609 [0.535, 0.665]** | **MECH-PROGRESS** | **+0.019 [+0.011, +0.029] OVERCONFIDENT** |

* Per head at u1000: lane 0.609, EP 0.136, DAC 0.126, TTC 0.066, NC 0.063, C -0.000. Relative drift vs A at u1000: NC 0.32, DAC 0.52, EP 0.35, TTC 0.25, C 0.37, DDC 0.31, lane 0.33 (R0: 0.34 / 0.53 / 0.35 / 0.25 / 0.36 / 0.30 / 0.32). TTC AUC delta B - A -0.0064 [-0.0132, +0.0005].
* Vs base0 (R0d): MECH-MIXED at u250 (safety 0.476), u500 (0.443) and u750 (0.404); MECH-PROGRESS at u1000 (safety 0.387 [0.234, 0.495], lane 0.304). R0 vs base0 was MIXED at u1000 and PROGRESS at u750: the base0 label is state-dependent in both arms, the own-A label is not. TTC head B - base0 at u1000, UNRESOLVED: +0.018 [-0.000, +0.035] on the own-A flip sets, +0.018 [-0.008, +0.046] on the base0 flip sets.
* Reading. The carrier is the lane gate again, at the same magnitude (0.49-0.63 vs R0's 0.48-0.62), the safety group again 0.23-0.30, and the TTC head again mildly overconfident against own A. The same carrier shows up in a second RL arm, so the finding is not specific to R0's configuration. One training seed per arm, so this replicates a pattern across two arms of the same family and is not a lever-effect claim. The R0d composites and the lane-dose family are owned by RL-2 (`rl2/`); not duplicated here.

## 3. Step 3: the RL-val read of the composites (R0 u1000; x100; paired log-cluster CIs; vs base0, the deployed scorer)

| row | PDMS | NC | DAC | EP | TTC | DDC | label |
|---|---|---|---|---|---|---|---|
| B (R0, EPO) | +0.59 [+0.14, +1.02] | -0.17 [-0.39, +0.03] | -0.10 [-0.45, +0.21] | +1.76 [+1.31, +2.18] | -0.17 [-0.71, +0.38] | -0.13 [-0.32, +0.03] | (B itself clears this bar on RL-val; navtest does not: RL-val is blind here) |
| **V1** RL{EP,C}+base0 | **+0.17 [-0.12, +0.40]** | -0.08 [-0.19, +0.00] | -0.10 [-0.36, +0.10] | +0.64 [+0.40, +0.86] | -0.07 [-0.38, +0.21] | -0.07 [-0.21, +0.05] | **GAIN-LOST-VAL** (EP retention 0.36; 1,017 picks differ from base0) |
| **V2** RL{EP,C}+C0-A | -0.30 [-0.95, +0.34] | +0.10 [-0.14, +0.36] | -0.47 [-1.05, +0.14] | -0.67 [-1.42, +0.04] | +0.50 [-0.13, +1.17] | +0.63 [+0.32, +1.00] | GAIN-LOST-VAL |

vs own A (programme bar S, TTC and NC may not fall by more than 0.5 point): V1 TTC -0.90 [-1.71, -0.14] fails (most of it is the TTC gain A holds over base0 on RL-val, +0.83, which a base0 safety source does not have), V2 TTC -0.33 [-0.70, 0.00] passes. Consistency states: V1 reads GAIN-LOST-VAL at R0 u250 / u500 / u1000 and R0b u250 / u500 and BOTH-LOST-VAL at R0 u750 (NC -0.12 [-0.24, -0.02]); V2 GAIN-LOST-VAL at all six.
Descriptive rows at R0 u1000 vs base0 (PDMS / EP / TTC): D1 RL{EP,C,lane}+base0 +0.34 / +1.02 / -0.10; D2 lane from base0 +0.12 / +1.04 / -0.40 (NC -0.20 [-0.34, -0.07] adverse); lane from A +0.30 / +0.41 / +0.37; D4 RL{EP} only +0.17 / +0.64 / -0.07 (identical to V1: comfort is inert, its sub-score delta is 0 everywhere).
Families at this tier: LONGITUDINAL = EP and TTC above; LATERAL = DAC and DDC above; TACTICAL and lateral kinematics NOT COMPUTABLE (n = 0); STRATEGIC PAUSED (PI 2026-09-19, item 25).

## 4. Pre-registered next arms, applied

* RL-val label GAIN-LOST-VAL (V1) -> per PREREG_B section 4: the navtest confirm still runs (one open). Done.
* **Navtest outcome N-BOTH -> PREREG_B section 5 row "N-SAFETY-LOST, N-BOTH": "the TTC / NC cost is not removable at inference by anchoring; R0e-pick (registered, T = 0.01, `--epo-con-temp`) is the committed A40 arm. State in RESULT what it does and does not address."** Section 5 below does that. Cost ESTIMATED from R0d's wall clock: 3.7 A40-h per arm plus a control; needs the PI and a `launch_gate.py` PASS token (BINDING 2026-09-26) before any launch.
* **A reading of the row that the data complicates (stated, not applied):** the row's premise is "the TTC / NC cost is not removable by anchoring". On navtest, anchoring did remove the TTC cost to within its interval (V1 TTC -0.13 [-0.32, +0.06] against R0B -0.40 [-0.71, -0.09]) and about half of the NC cost (-0.12 against -0.25); what it did not remove is NC, and it removed the PDMS gain. So the row applies mechanically (label N-BOTH), and the committed arm stands; the NC component is the one a training-time constraint has to reach.
* **Conflict between two frozen documents, escalated to the Master Mind:** PREREG_B section 5 names R0e-pick for N-SAFETY-LOST / N-BOTH; `rl2/out/raw/MM_DECISION_NAVTEST2.md` says for the same outcomes "the next lever is the anchored RL arm (RL-1 section 7)". Both are A40-bound and both are blocked (6.9). I applied PREREG_B (it governs RL-1); which arm gets the first A40 grant is a PI / Master Mind choice. Evidence for that choice is in section 5.

## 5. What R0e-pick does and does not address (the failure branch)

R0e-pick (R0d's argv plus `--epo-con-temp 0.01`) constrains the pick-aligned E_pi[TTC] and E_pi[NC]; the dual acts on whichever parameters move those quantities, so it can in principle bind on the lane gate. It does NOT anchor the lane or DAC outputs, and step 1 + D2 show the PDMS gain and the flip share the lane gate, so a constraint that holds TTC and NC will also tax the gain through that carrier; R0d's O4 was already a gain-keeping failure at the pick (TTC -0.77 on RL-val). Nothing in this package contradicts running R0e-pick.
**What the navtest read adds (MEASURED, descriptive).** The adverse component on navtest is NC, and V1 shows it with the deployed NC head in charge of the safety channels: V1 changes NC on only 34 of 12,146 tokens (10 better / 24 worse) and TTC on 88 (36 / 52), yet the means move by -0.12 and -0.13. So (a) the NC channel R0e-pick constrains is the right one for the residual cost, and (b) an inference-time composite cannot remove an NC cost that comes from EP-led pick changes, because the NC head it keeps is the deployed head at its own resolution (HYPOTHESIS: the residual NC cost is a limit of the deployed NC head's resolution, not of RL's overconfidence; testing it needs a per-pick NC-head calibration read on the 24 worse tokens, which I did not run because the navtest open is spent and a second descriptive pass over those rows is a forking path). (c) The anchored RL arm of section 7 is the training-time analogue of V1 (only EP/C move, the rest fixed at base0); V1 is its zero-training approximation and fails the bar at NC, which weakens (does not refute) the premise that training EP/C against anchored heads removes the mismatch.

## 6. Step 4: the single navtest confirm (T1 NAVSIM v1.1, 12,146 tokens / 136 logs): READ, OUTCOME N-BOTH

### 6.1 Provenance and gates (all MEASURED)

* Variant V1 per the committed selection rule, built from the banked FREEPOINT chunk outputs (BASE and R0B logits and the u400 fan), no GPU, no pod, by `nt_build.py`. Build gates (`raw/navtest/build_V1.json`): BASE and R0B picks reproduced 12,146 / 12,146; no-RL composite = BASE picks and all-RL composite = R0B picks, bit-equal. V1 differs from BASE's pick on 3,895 tokens, from R0B's on 6,726; 2,397 tokens have a pose equal to none of BASE / C0A / R0B / M25.
* Scoring: W3's harness (`score_e.py` wrapper, sha256 `770a744f...2147`, byte-identical to the repo copy in `raw/2026-10-04-navtest-final/`; it loads a drive-repointed copy of `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/code/run_v1.py`, verified identical to the repo file after D:/ -> E:/ and CR removal), run through `nt_run2.py` + `score_e2.py` (section 10), **attempt 2 of the `nt_run2` relaunch** (23:55-00:23 Berlin, 1,682 s): 12,146 / 12,146 scenarios, 0 failed, harness status **PASS**, C1 max abs delta 0.0, devkit average-row vs mean of rows 2.1e-15, `DONE_V1` = OK. CSV `raw/navtest/harness/refe_rl1_V1.csv` sha256 `67905a3152c84d2160e19e82af36ea55fe22851e801399889fa6686fd1eb2a1e` (byte-identical copy, CRLF as the harness writes it).
* Reader: `nt_read.py` (sha256 `ea75b849...a1f2`), FREEPOINT's `fp_read` / `stage3_full` rows and estimator. **Dry known-value test first (before the real read; it is not a second navtest open: identical banked R0B rows, no scoring):** variant T = FREEPOINT's R0B seam, picks and CSV. Required: V_vs_BASE equals FREEPOINT R0B vs BASE, same-pick identity max diff 0.0. Observed: 73 numeric leaves, 67 bit-equal, max abs diff 1.4e-14, max relative 1.75e-16 (one ulp), all non-numeric leaves equal; PDMS +0.55 [+0.24, +0.84], NC -0.25, TTC -0.40; families adverse and levels equal; identity (a) max diff 0.0 on 4,532 / 2,663 / 12,146 same-pick tokens vs BASE / C0A / R0B. Verdict PASS, **no bug found, `nt_read.py` unchanged** between the dry and the real read (`raw/navtest/dry_known_value/`). Cosmetic only: a dead variable `seams = {"BASE": ...}` with an unused path sits in `nt_read.py`; left in place so the hashed file is the file that ran.
* Real-read gates, all true: `harness_PASS`, `rows_12146`, `identity_a_same_pick_rows_equal` (tokens whose V1 pick equals BASE's / C0A's / R0B's pick read that point's banked row exactly: 8,251 / 2,108 / 5,420 tokens, max abs diff 0.0). Gate (b) (the composite built from BASE only reproduces BASE) is `build_V1.json` above. The result is VALID.

### 6.2 Levels (T1, mean x100, all 12,146 tokens; `raw/navtest/result/navtest_V1_result.json`)

| point | PDMS | NC | DAC | EP | TTC | C | DDC | zero-PDMS picks |
|---|---|---|---|---|---|---|---|---|
| BASE (deployed) | 88.2388 | 98.94 | 96.57 | 81.19 | 95.63 | 99.97 | 97.66 | 530 |
| C0A | 87.3604 | 98.98 | 96.35 | 79.35 | 95.64 | 99.98 | 98.69 | 549 |
| R0B | 88.7855 | 98.69 | 96.53 | 83.00 | 95.23 | 99.98 | 97.55 | 566 |
| **V1** | **88.3339** | 98.82 | 96.45 | 81.72 | 95.50 | 99.97 | 97.52 | 560 |

### 6.3 V1 - BASE (paired, x100, 95 % interval; the bar's comparator)

| metric | V1 - BASE | better / worse / tied (tokens) | in the bar? |
|---|---|---|---|
| **PDMS** | **+0.0952 [-0.0738, +0.2573]** | 2,181 / 418 / 9,547 | lower bound > 0 required: **NOT MET** |
| NC | **-0.1235 [-0.2372, -0.0156]** | 10 / 24 / 12,112 | upper bound >= 0 required: **ADVERSE** (0.016 points below zero) |
| DAC | -0.1153 [-0.2641, +0.0081] | 13 / 27 / 12,106 | upper bound >= 0: met |
| EP | +0.5296 [+0.3698, +0.6862] | 2,187 / 384 / 9,575 | no |
| TTC | -0.1317 [-0.3197, +0.0594] | 36 / 52 / 12,058 | upper bound >= 0: met |
| C | +0.0000 [0.0000, 0.0000] | 0 / 0 / 12,146 | no |
| DDC | -0.1400 [-0.1981, -0.0866] ADV | 2 / 36 / 12,108 | no (reported) |

**OUTCOME: N-BOTH** (PDMS lower bound -0.0738 <= 0, and NC upper bound -0.0156 < 0; DAC and TTC clear). DEPLOYABLE-SAFE (vs BASE): False (`PDMS_lo_gt_0` False, `NC_hi_ge_0` False, `DAC_hi_ge_0` True, `TTC_hi_ge_0` True). The label is applied as written; note that it hinges on one bound 0.016 points from zero: had NC's upper bound been >= 0 the committed label would be N-GAIN-LOST (my forecast), and in either case the bar is not met and the PDMS gain is not resolved. No threshold was changed after the number. Reference rows in the same table family: R0B - BASE PDMS +0.5468 [+0.2426, +0.8352], NC -0.2511 [-0.4481, -0.0624] ADV, TTC -0.3952 [-0.7062, -0.0892] ADV (reproduces `D-REFE-FREEPOINT-1`); C0A - BASE PDMS -0.8784 [-1.4698, -0.1436].

Descriptive retention of R0B's effect by V1 (ratio of point estimates, NO interval, post-hoc, ESTIMATED from the table): PDMS gain 0.17 (RL-val: 0.29), EP gain 0.29 (RL-val: 0.36), NC loss 0.49, TTC loss 0.33, DDC loss 1.21 (larger than R0B's).

### 6.4 Lever view (paired, x100)

| contrast | PDMS | NC | DAC | EP | TTC | C | DDC |
|---|---|---|---|---|---|---|---|
| V1 - R0B | **-0.45 [-0.72, -0.17]** | +0.13 [-0.05, +0.32] | -0.07 [-0.26, +0.12] | -1.28 [-1.56, -1.00] ADV | +0.26 [+0.01, +0.53] | -0.02 [-0.05, 0.00] | -0.02 [-0.12, +0.08] |
| V1 - C0A | **+0.97 [+0.14, +1.62]** | -0.16 [-0.40, +0.08] | +0.10 [-0.71, +0.69] | +2.37 [+1.52, +3.16] | -0.15 [-0.60, +0.33] | -0.02 [-0.05, 0.00] | -1.17 [-1.54, -0.83] ADV |

Reading (descriptive, not a lever claim): relative to R0B the composite gives back 0.45 PDMS and 1.28 EP and recovers a resolved 0.26 TTC and an unresolved 0.13 NC; relative to the BCE-only retrain C0A it keeps a resolved +0.97 PDMS and +2.37 EP at an unresolved NC cost.

### 6.5 The four metric families (V1 vs BASE; STRATEGIC PAUSED)

| family | metric (T1) | BASE | V1 | V1 - BASE |
|---|---|---|---|---|
| **LONGITUDINAL** (paired, x100) | EP | 81.19 | 81.72 | +0.53 [+0.37, +0.69] |
| | TTC (the NAVSIM time-to-collision proxy; no headway / time-gap metric exists in this harness) | 95.63 | 95.50 | -0.13 [-0.32, +0.06] |
| | comfort C | 99.97 | 99.97 | +0.00 [0.00, 0.00] |
| (families6, unpaired episode-cluster 95 %) | speed MAE (m/s) | 0.7397 [0.7112, 0.7713] | 0.7265 [0.697, 0.757] | intervals overlap |
| | speed bias (m/s) | -0.2303 [-0.3258, -0.1369] | -0.1705 [-0.265, -0.0759] | overlap |
| | along-track MAE (m) | 1.034 [0.9879, 1.083] | 1.02 [0.9714, 1.068] | overlap |
| | along final bias (m) | -0.8263 [-1.19, -0.4778] | -0.6057 [-0.9737, -0.2535] | overlap |
| | accel MAE (m/s^2) | 0.5193 [0.5032, 0.5385] | 0.5112 [0.4952, 0.5295] | overlap |
| | target-speed acc. within 1.0 m/s | 0.7428 [0.7279, 0.7567] | 0.7492 [0.7352, 0.7634] | overlap |
| **LATERAL** (paired, x100) | DAC | 96.57 | 96.45 | -0.12 [-0.26, +0.01] |
| | DDC (not in the bar) | 97.66 | 97.52 | -0.14 [-0.20, -0.09] ADV |
| (families6, unpaired) | heading MAE (deg) | 2.532 [2.314, 2.729] | 2.533 [2.312, 2.737] | overlap |
| | yaw-rate MAE (deg/s) | 2.983 [2.8, 3.165] | 2.967 [2.784, 3.149] | overlap |
| | curvature MAE (1/m) | 0.01987 [0.01834, 0.02138] | 0.01955 [0.01806, 0.02101] | overlap |
| | cross-track MAE (m) | 0.3779 [0.3513, 0.4045] | 0.3777 [0.3504, 0.4036] | overlap |
| | cross-track final MAE (m) | 0.865 [0.7945, 0.9332] | 0.8676 [0.7964, 0.9361] | overlap |
| **TACTICAL** (families6; no interval, reported not tested) | lateral decision acc / kappa | 0.9252 / 0.8628 | 0.9253 / 0.8632 | +0.0001 / +0.0004 |
| | longitudinal decision acc / kappa | 0.6789 / 0.5157 | 0.6824 / 0.5214 | +0.0035 / +0.0057 |
| | 5-way maneuver (collapsed) acc / kappa | 0.7831 / 0.7182 | 0.7854 / 0.7218 | +0.0023 / +0.0036 |
| **STRATEGIC** | | **PAUSED** (PI 2026-09-19, `PI_DECISION_QUEUE` item 25) and UNAVAILABLE in NAVSIM | | never a pass, not omitted |

families6 gate (`families_adverse_vs_BASE`, 21 components, unpaired interval non-overlap, conservative, REPORTED NOT GATING): V1 adverse list **empty**. The families6 table is therefore silent on the one component the bar found adverse (NC), which is a per-pick collision outcome, not a kinematic error: the four families do not contradict the bar and they do not rescue it. Lateral kinematics and tactical decisions are unchanged within noise; the longitudinal kinematic improvement is directionally the same as R0B's (speed bias and final bias closer to zero) but not separated.

### 6.6 RL-val read vs navtest read, V1 vs the deployed scorer (descriptive)

| metric (x100) | RL-val (R0 u1000, 3,000 sets / 75 logs) | navtest T1 (12,146 tokens / 136 logs) |
|---|---|---|
| PDMS | +0.17 [-0.12, +0.40] | +0.10 [-0.07, +0.26] |
| NC | -0.08 [-0.19, +0.00] | -0.12 [-0.24, -0.02] |
| DAC | -0.10 [-0.36, +0.10] | -0.12 [-0.26, +0.01] |
| EP | +0.64 [+0.40, +0.86] | +0.53 [+0.37, +0.69] |
| TTC | -0.07 [-0.38, +0.21] | -0.13 [-0.32, +0.06] |
| DDC | -0.07 [-0.21, +0.05] | -0.14 [-0.20, -0.09] |
| label | GAIN-LOST-VAL | N-BOTH |

Every point estimate has the same sign on both tiers and every RL-val interval contains the navtest point estimate; the navtest interval is narrower for five of the six metrics (NC is the exception: 0.19 wide on RL-val, 0.22 on navtest). The labels differ only because NC's upper bound moves from +0.00 to -0.02. So for V1 the selection tier carried the direction of every metric. For B the RL-val point estimates had the right sign but understated the TTC loss (-0.17 against -0.40 on navtest), and neither RL-val interval separated. This qualifies the interim sentence that the RL-val bar "predicts nothing about navtest NC/TTC": it ranked V1 and V2 correctly in direction and could not resolve the loss.

### 6.7 Forecast scoreboard (committed before the numbers; checked)

| forecast | where committed | outcome |
|---|---|---|
| step 1 carrier is EP/C, label MECH-PROGRESS | PREREG_A section 0 | label RIGHT, carrier WRONG (lane gate 0.60) |
| V1 / V2 keep >= 0.5 of B's EP lift on RL-val | PREREG_B section 6 | WRONG (V1 0.36) |
| V1 / V2 recover part, not all, of B's TTC loss | PREREG_B section 6 | V1 RIGHT on RL-val (-0.07 vs B -0.17) and on navtest (-0.13 vs -0.40, 0.67 of it recovered); V2 WRONG on RL-val (+0.50, more than all) |
| RL-val label SAFETY-LOST-VAL or PASS-VAL, not GAIN-LOST-VAL | PREREG_B section 6 | WRONG (GAIN-LOST-VAL for both) |
| navtest label N-GAIN-LOST (no adverse component) | interim section 6 | WRONG: N-BOTH (NC adverse by 0.016) |
| navtest: the anchored heads must remove NC and TTC losses to within their CIs | PREREG_B section 6 | TTC yes, NC no |

### 6.8 Disclosures (three; none of them selected, tuned or re-ran anything)

1. **Two partial views of the average row were seen before the full read.** The reconstructed CSV of the first run (4,433 of 12,146 tokens, PDMS 88.44) was seen by the Master Mind while diagnosing the failure; the status JSON of `nt_run2` attempt 1 (8,756 tokens, PDMS 88.47, FAIL, missing 3,390 tokens) was seen by the Master Mind and, in `score_logs/refe_rl1_V1.attempt1.log`, by this continuation agent before the real read. V1, its bar and its selection rule were frozen in PREREG_B (sha256 `0593d79223...8f58`) before either view, the bar is a paired interval over all 12,146 tokens, and nothing was re-run, re-selected or re-scored on a partial. For scale: both partial averages (88.44, 88.47) sit 0.11-0.14 above the final V1 level 88.3339, and the partial prefixes are not random subsets, so they carry no information about the final interval.
2. **Quarantine.** Three RAM-guard aborts of the first run and one of the relaunch left partial CSVs; the first run's files are in `nav/quarantine_ram_abort_20261018/` (sha256 of the reconstructed partial `d2539c5f...f0a0`), were not read by this agent and are not staged; the relaunch's own 8,756-row partial was rejected by the harness's row checks and moved aside by the wrapper. Only attempt 2 of `nt_run2` is read. `raw/navtest/QUARANTINE_NOTE.md`.
3. **`rl2/out/raw/MM_DECISION_NAVTEST2.md` (RL-2, landed bd37576) was written after the first partial view (4,433 tokens, 88.44) and before the second (8,756 tokens, 88.47) and the full read.** Its rule is keyed on the outcome CATEGORY of PREREG_B section 5, selects nothing, and states itself "BEFORE V1's navtest outcome exists"; but its timing relative to the first partial view is stated here, as the Master Mind asked.

### 6.9 Which RL-2 rule the outcome triggers, the committed next arm, and what blocks it

* **RL-2 table row "N-SAFETY-LOST or N-BOTH" -> look #2 for L0.25: NO.** Its stated reason applies: L0.25 carries MORE RL lane than V1, and V1 already loses NC on navtest, so more lane is the wrong direction. No second navtest look is opened by this package. (Had the label been N-GAIN-LOST, the same table would have said YES; see the sensitivity in 6.3.) The RL-2 disclosure clause is not triggered because look #2 is not spent.
* **Committed next arm (PREREG_B section 5): R0e-pick.** What it does and does not address: section 5. **Named blockers (RULE ZERO): (a) compute:** the A40 runs P6 and the PI declined extra compute; unblocks on a PI grant of about 4-8 A40-h (ESTIMATED from R0d's wall clock, 3.7 A40-h per arm plus a control) or on P6 freeing the A40; **(b) the launch gate:** a `launch_gate.py` PASS token bound to commit, argv and manifests, with a G-DVB entry for `--epo-con-temp`; **(c) PI decision** on whether R0e-pick or the anchored arm (section 7) gets the first grant (the conflict in section 4).
* **Cheapest unblocked lever, not run here:** a CPU-only, RL-val-tier, PREREG'd read of a partial EP dose or a safety-conditioned dose (RL-2 named the latter as its unrun family (c)). It is outside steps 2-5, needs its own PREREG before any number, and any navtest read of it needs the caller / PI because this package's one open is spent. Per section 6.6 the selection tier does carry V1's direction, so it can rank candidates, with the intervals' width in mind.

## 7. Next lever if the composite fails or only holds safety (3 lines, cost)

The anchored RL arm: (1) in `scorer_finetune.py` give the NC, DAC, TTC, DDC and lane output slices of `score_head` lr 0 during EPO (only EP/C train); (2) R0's reward, `--epo-lam 1.0`, seed 0; (3) a logit anchor of those slices to the step-0 scorer behind a G-DVB-gated flag. ESTIMATED 3.7 A40-h per arm (R0d wall clock) plus a control; a launch-gate profile first; needs the PI. Caveat from D2/D3: if the gain lives in the lane gate, an arm that freezes lane may also lose the gain; the cheaper variant is to anchor only NC, DAC and TTC and constrain lane drift instead. **Status after the navtest read:** V1 is this arm's zero-training analogue and reads N-BOTH; the arm is therefore second in line to R0e-pick under PREREG_B and first under the RL-2 text (section 4), and the lane-freezing caveat is consistent in direction with the navtest read (V1 anchors lane together with the safety heads and keeps 0.17 of the PDMS gain; the RL-val D2 row isolates the lane).

## 8. Blockers and not-done

* R0d: READ as an extension (section 2b). Composites on R0d and the lane-dose family: owned by RL-2, not done here.
* Navtest: the one open is spent; no second variant is scored by this package.
* R0e-pick and the anchored arm: BLOCKED (6.9: A40 compute, PI grant, launch-gate token).
* Not done: the per-pick NC-head calibration read on V1's 24 NC-worse tokens (section 5), deliberately, to avoid a forking path on a spent look; a replicate arm for any lever claim (H-ESTIM-SEED-1), so nothing here is a lever effect.
* The RL-val bar vs base0 selected between V1 and V2 and carried V1's direction to navtest (6.6), but it did not separate B's navtest NC / TTC loss (B itself passes the RL-val bar).
* Provenance dependency to fix upstream: `score_e.py` loads W3's harness from `.../a085b71c-.../scratchpad/w3e/code` (a scratch copy of the repo's `run_v1.py` with D:/ -> E:/); the fix for the cross-drive `os.path.relpath` crash exists only as the process-local wrapper `score_e2.py`, not in `run_v1.py:459`.

## 9. Proposed register rows (NOT edited; for the Master Mind)

* `D-REFE-RL1-MECH` (2026-10-18): the flip of R0 to a TTC-failing pick is carried by the lane output (Shapley share 0.60 [0.52, 0.66] of the pick-pair margin change, 59 sets, R0 u1000 vs own A), safety group 0.27 [0.22, 0.33]; the RL TTC head is mildly overconfident vs its own arm A (+0.018 [+0.010, +0.028]). MEASURED, RL-val selection tier, one seed per arm. `raw/2026-10-18-rl-headwise/` (`mechanism_r0.json`).
* `D-REFE-RL1-MECH-R0D` (2026-10-19, EXTENSION, post-hoc to PREREG_A): R0d u1000 vs own A, lane 0.609 [0.535, 0.665], safety 0.255 [0.211, 0.311], EP+C 0.136 [0.111, 0.170], 55 TTC-flip sets; MECH-PROGRESS at u250-u1000; TTC head overconfident vs own A (+0.019 [+0.011, +0.029]). Replicates R0 in a second RL arm. MEASURED, RL-val selection tier, one seed per arm. `raw/mechanism_r0d.json`.
* `D-REFE-RL1-HEADWISE` (UPDATE of the 2026-10-18 row): zero-training head-wise composite V1 (RL{EP,C} + deployed {NC,DAC,TTC,DDC,lane}) reads GAIN-LOST-VAL on RL-val (PDMS +0.17 [-0.12, +0.40]) and **N-BOTH on navtest T1** (12,146 tokens / 136 logs; paired log-cluster bootstrap, "another draw of LOGS" only, one seed): PDMS +0.10 [-0.07, +0.26], NC -0.12 [-0.24, -0.02] ADV, DAC -0.12 [-0.26, +0.01], EP +0.53 [+0.37, +0.69], TTC -0.13 [-0.32, +0.06], DDC -0.14 [-0.20, -0.09]; level 88.3339 vs BASE 88.2388. MEASURED. `raw/navtest/result/navtest_V1_result.json`. The anchoring removed the TTC cost to within its interval and about half of the NC cost, and it removed the PDMS gain; not DEPLOYABLE-SAFE.
* `H-REFE-RL1-NCRES` (HYPOTHESIS, OPEN): the residual NC cost of V1 is a resolution limit of the deployed NC head under EP-led pick changes, not RL-head overconfidence. Discriminating read (needs a PREREG and, if it uses navtest rows beyond this package's look, the PI): the deployed NC head's pass probability at V1's 24 NC-worse picks vs at BASE's picks on the same tokens.
* BACKLOG: "REFE R0e-pick A40 arm: committed by PREREG_B N-BOTH row; BLOCKED on A40 (P6) + PI grant (~4-8 A40-h) + launch-gate PASS token with a G-DVB entry for `--epo-con-temp`"; "decide R0e-pick vs anchored arm for the first A40 grant (PREREG_B section 5 vs MM_DECISION_NAVTEST2 disagree)"; "land `nav/nt_run2.py` + `score_e2.py` pattern as a harness fix: relpath fallback in `run_v1.py:459` and a full-row-count requirement in the wrapper"; "stop `score_e2`/`nt_run2` from printing a summary row on a FAILED or partial run (two principals saw 88.44 / 88.47 before the full read)".
* TRAPS.md: entry in section 10.

## 10. TRAPS.md entry text (for the Master Mind to land; also referenced in `memory`: devbox-drive-letter-and-detached-jobs)

- ⛔ **A W3 NAVTEST RUN KILLED BY ITS RAM GUARD LEAVES A RECONSTRUCTED PARTIAL CSV THAT LOOKS
  FINISHED — AND THE HARNESS'S GLOB FALLBACK RE-COPIES IT ON EVERY LATER ATTEMPT; A CROSS-DRIVE
  `os.path.relpath` CRASH THEN SKIPPED THE ROW CHECKS THAT WOULD HAVE CAUGHT IT.** MEASURED
  2026-10-18 (RL-1 V1 navtest): the W3 RAM guard aborts the devkit when available memory falls
  under 3,000 MB (two other sessions' full stack test suites held 7.3 GB). The harness then
  **reconstructs a CSV from the streamed rows** (4,433 of 12,146), and with `--out` on C: and the
  package on E: it crashed at `run_v1.py:459` (`os.path.relpath` across drives) **before its
  C2/C3 row-count checks ran**; attempts 2 and 3 aborted at once on RAM and the glob fallback
  **re-copied the stale partial**, so the run dir held a file that read like a result. The
  relaunch lost a second attempt the same way (8,756 rows, a sibling's 3 GB stream scan) and
  there the checks RAN and rejected it (FAIL, missing 3,390 tokens) — the guards work. ⇒ **Fix
  pattern, `nav/nt_run2.py` + `nav/score_e2.py` (package `raw/code/`):** (1) a cross-drive
  `relpath` fallback; (2) a **memory gate before each attempt** (`psutil.virtual_memory().available`
  >= 6,500 MB for 3 consecutive minutes — the `\Memory\Available MBytes` perf counter does not
  exist on this German Windows and reads 0 forever); (3) **move every `*.csv` already in the
  devkit run dir aside before each attempt**; (4) **OK requires the full token-row count**
  (12,146), not just harness PASS. ⚠️ Corollary: **a failed or partial run must not print its
  average row** — two principals read 88.44 (4,433 rows) and 88.47 (8,756 rows) before the full
  read and had to disclose it. Also write status/trail files with LF (Windows text-mode `json.dump`
  and `open(..., "a")` write CRLF).

## 11. Deliverable manifest

Everything below lives under the package root `out/` on the dev box (`C:/Users/Admin/par18/rl/out/`) and is **STAGED FOR LANDING, not yet in git** (no `git add`/commit by agent rule; the Master Mind lands from `out/paths.json`). Repo prefix: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-18-rl-headwise/`. Every file is also listed with sha256 in `raw/MANIFEST_RL1.sha256`. Text files are LF with no BOM; the harness CSV is a byte-identical copy (CRLF as the harness writes it; the lander's LF check may need an exception, precedent `data/refe_navtest/score/*/*.csv`); `.npz` are binary.

| artifact | package path (repo = prefix + path) | exists in |
|---|---|---|
| this report | `RESULT.md` | dev box only until landed |
| landing list (repo path, source path, kind) and sha256 manifest of every staged file | `paths.json`, `raw/MANIFEST_RL1.sha256` | dev box only |
| pre-registrations A, B + hashes | `raw/PREREG_RL1_A_MECHANISM.md`, `raw/PREREG_RL1_B_COMPOSITES.md`, `raw/*.sha256` | dev box only |
| RL-val results | `raw/mechanism_r0.json`, `raw/mechanism_r0b.json`, `raw/composites_rlval.json`, `raw/INPUTS_RLVAL.sha256` | dev box only |
| R0d extension | `raw/mechanism_r0d.json` | dev box only (inputs `r0d_dl/` 344.8 MB: dev box, and HF `Sayood/tanitad-refe-rls1-r0d` private) |
| code | `raw/code/{rlh,mechanism,composites,nt_build,nt_run,nt_read,nt_run2,score_e2}.py` | dev box only |
| navtest seam, picks, build gates | `raw/navtest/{seam_V1.npz,points_V1.npz,build_V1.json}` | dev box only |
| harness rows | `raw/navtest/harness/{refe_rl1_V1.csv,refe_rl1_V1.counts.json,refe_rl1_V1_manifest.json}` | dev box + the devkit run dir `E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/runs/refe_rl1_V1/` (the 12 MB `_hooks.json`, 5.7 MB `_rows.jsonl`, 3 MB `.log`, 1 MB `.calls.jsonl` stay there, NOT staged) |
| navtest read | `raw/navtest/result/{navtest_V1_result.json,RESULT_TABLES_V1.md,read_V1.log}`, `raw/navtest/result/families/{fp_BASE,fp_R0B,rl1_V1}_families.{json,log}` | dev box only |
| run trail | `raw/navtest/score_logs/{refe_rl1_V1.status.json,refe_rl1_V1.attempt1.log,refe_rl1_V1.attempt2.log,refe_rl1_V1.nt_run2.log}` | dev box only (LF copies; originals on `nav/score_logs/` are CRLF) |
| quarantine note | `raw/navtest/QUARANTINE_NOTE.md` | dev box only |
| dry known-value test | `raw/navtest/dry_known_value/{navtest_T_result.json,RESULT_TABLES_T.md,read_dry.log,dry_vs_freepoint.json}` | dev box only |
| hand-off (completed) | `raw/HANDOFF_RL1.md` | dev box only |
| **NOT staged, by design** | the quarantined CSVs and first-run files (`nav/quarantine_ram_abort_20261018/`), the 8,756-row partial (`*.moved` in the devkit run dir), `families/*_seam.npz` duplicates, `nav_dry/` working copies of banked CSV and seam | dev box (quarantine, kept) |

Integration needed (headline): (1) land the package and the section 10 TRAPS entry; (2) register rows in section 9; (3) **decide R0e-pick vs the anchored arm for the first A40 grant** (PREREG_B and MM_DECISION_NAVTEST2 name different arms for this outcome); (4) fix `run_v1.py:459` and the wrapper's partial-summary printing upstream.
