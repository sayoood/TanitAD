# PREREG RL-1 / B: head-wise composites of the RL scorer and a safety source (zero training, CPU)

Written 2026-10-18 by the RL-1 research agent AFTER the mechanism read (`PREREG_RL1_A_MECHANISM.md`, sha256 `c7b1c1c5...`, verified unchanged; results in `mechanism_r0.json` / `mechanism_r0b.json`) and BEFORE any composite pick was scored against labels, on RL-val or on navtest. sha256 in `PREREG_RL1_B_COMPOSITES.sha256`.

## 0. Disclosures

* From step 1 (MEASURED, RL-val selection tier, R0 u1000, B vs own A, pick-pair margin change on the 59 sets where B's pick fails TTC and A's passes; Shapley shares, log-cluster CIs): lane gate **0.60** [0.52, 0.66], DAC 0.14, EP 0.13, TTC 0.07, NC 0.06, comfort 0.00; safety group {NC, DAC, TTC} 0.27 [0.22, 0.33]. The committed label MECH-PROGRESS fired, but the carrier is the **lane output**, which the earlier hybrids (`attribution.json` Q5 H1-H4) kept on the progress side. The same pattern holds at R0 u250/u500/u750 and R0b u250/u500 (lane 0.48-0.62). Against the deployed base0 the label is MECH-MIXED (safety 0.40 [0.23, 0.51], EP 0.32, lane 0.29). My step-1 prediction (EP/C carries it) was WRONG on the carrier; the label itself matched.
* INHERITED, not re-verified, read before writing: Q5 H1 (safety from R0-A + {EP, C, lane} from R0-B vs C0-A, R0 u1000): TTC -0.80 [-1.51, -0.14], PDMS +0.95; and `D-REFE-FREEPOINT-1` navtest T1 (12,146 tokens): R0B vs BASE PDMS +0.55 [+0.24, +0.84], NC -0.25 [-0.45, -0.06] ADV, TTC -0.40 [-0.71, -0.09] ADV; C0A vs BASE PDMS -0.88, TTC +0.02, NC +0.04. I have seen NO number of any composite below.
* The novelty relative to H1 is exactly one thing: **the lane output is anchored with the safety heads.** Everything else is the brief's head-wise composition.

## 1. The two variants (all that may go to navtest; fixed here)

Channels: 0 NC, 1 DAC, 2 EP, 3 TTC, 4 C, 5 DDC, 6 lane. A composite takes the logits of the listed RL channels from the RL arm's B scorer and every other channel from the source SRC, per proposal, then applies the unchanged rule `navsim_v1_lane`.

| variant | RL channels (from B) | SRC for the rest (NC, DAC, TTC, DDC, lane) | deployable meaning |
|---|---|---|---|
| **V1** | EP, C | **base0** = the deployed scorer (`scores_step0_base`) | safety and lane exactly as deployed; only EP and comfort are RL's |
| **V2** | EP, C | **own arm A** = the BCE-only arm (RL-val: C0-A at the same update; navtest: C0A, the stand-in for R0-A, PDMS +0.01 vs C0 on RL-val) | safety and lane from the BCE-only retrain |

"RL{EP} only" is registered as a DESCRIPTIVE row (D4), not a variant: the step-1 comfort share is 0.00, so it is predicted to pick the same as V1. Which checkpoint is not a choice: navtest has exactly one banked state, **R0 u1000 B** (`rls1-r0/scorer_states_u1000.pt` key B, the FREEPOINT R0B).

## 2. RL-val read (selection tier; 3,000 sets / 75 logs; MEASURED lookup of the true NAVSIM v1.1 sub-scores of the composite's pick)

* States: R0 u250 / u500 / u750 / u1000 and R0b u250 / u500 (SRC A = C0-A at the same update). **Selection uses R0 u1000 only**; the other states are consistency rows.
* Comparators, paired log-cluster bootstrap (10,000, seed 20260927, "another draw of LOGS" only, one training seed per arm, H-ESTIM-SEED-1): (i) **vs base0 = the navtest bar's comparator**; (ii) vs own A (the programme's bar S: TTC and NC may not fall by more than 0.5 point); (iii) B itself vs base0 for scale.
* Metrics per comparator: PDMS and the sub-scores NC, DAC, EP, TTC, C, DDC (x100); the four families as far as this tier computes them: LONGITUDINAL = EP and TTC, LATERAL = DAC (and DDC, which is not in the bar), TACTICAL and lateral kinematics NOT COMPUTABLE at RL-val (n = 0), STRATEGIC = PAUSED (PI 2026-09-19, item 25).
* **RL-val label per variant (the navtest bar's logic, vs base0):** PASS-VAL = dPDMS CI lower bound > 0 AND NC, DAC, TTC CI upper bounds >= 0. GAIN-LOST-VAL = dPDMS lower bound <= 0, no adverse. SAFETY-LOST-VAL = dPDMS lower bound > 0, at least one of NC / DAC / TTC with upper bound < 0. BOTH-LOST-VAL = neither.
* **Selection rule (one variant goes to navtest):** deficit(v) = max(0, -LB[dPDMS]) + sum over c in {NC, DAC, TTC} of max(0, -UB[dc]), all in x100 points, vs base0 at R0 u1000; the variant with the smaller deficit is selected; a tie within 1e-9 goes to the higher dPDMS point. A PASS-VAL variant has deficit 0. The navtest confirm runs for the selected variant **whatever its RL-val label**, because RL-val is a selection tier and R0B itself reads TTC -0.17 (not separated) on RL-val against -0.40 (separated) on navtest.
* **Descriptive rows (no selection, no navtest), same estimator, R0 u1000 and R0 u500:** D1 old-H1 shape with base0: RL{EP, C, lane} + base0{NC, DAC, TTC, DDC}; D2 B with ONLY lane from base0, and from A; D3 leave-one-head-anchored: B with head h from base0, for each of the 7 channels, and the reverse (base0 with only head h from B); D4 RL{EP} only + base0.

## 3. Controls (each must hold or the RL-val read is INVALID)

1. All 7 channels RL = B: picks identical to B's and every metric identical to B's; zero RL channels = SRC: picks identical to SRC's (bit-exact, both SRCs).
2. Machinery identity: reproduce `attribution.json` Q5 H1 and H4 (R0 u1000 vs C0-A: TTC -0.800 / -0.233, PDMS +0.949 / +0.626, with their CIs) to 1e-6, and R0-B vs base0 (banked text of `D-REFE-SAFEHEAD-1`: PDMS +0.59 [+0.14, +1.02], TTC -0.17 [-0.71, +0.38], ESTIMATED) to the printed precision.
3. Deliberate regression arm: V1 with SRC replaced by base0 logits of a DIFFERENT set (a fixed permutation of the 3,000 sets) must read dPDMS lower bound < 0 (collapsed), i.e. the bar must fail it.
4. Estimator identity: `rlh.Boot` equals `proxy_eval.boot` (0.0, already shown).

## 4. Committed outcomes of the RL-val read, and the next arm of each

| RL-val result (selected variant, R0 u1000) | reading | committed next arm |
|---|---|---|
| PASS-VAL | the composite clears the navtest bar's logic on the selection tier | run the navtest confirm (section 5); expect it |
| GAIN-LOST-VAL | anchoring removed the gain | navtest confirm still runs; the next lever is section 5's "anchored RL arm" |
| SAFETY-LOST-VAL or BOTH-LOST-VAL | anchored heads do not hold TTC / NC / DAC even at zero training | navtest confirm still runs (one open); the mechanism row to quote is the D3 table: which single anchored head recovers most of the loss |

## 5. The navtest confirm (opened ONCE, for the selected variant only)

* Tier: **T1 NAVSIM v1.1 navtest pseudo-sim**, 12,146 tokens / 136 logs, the FREEPOINT pipeline UNCHANGED (`fp_picks.py` composite logits from the banked chunk outputs `C:/Users/Admin/freept/work/out/chunk_00*.npz` = BASE / C0A / R0B logits and the u400 fan; the A7-repaired executed pose; W3's harness via `score_e.py`, full navtest filter, status PASS; read with `fp_read.py`'s rules). No GPU, no pod: all banked.
* Comparator **BASE** (the deployed scorer). **Bar:** the paired dPDMS (composite - BASE) CI lower bound > 0 AND NC, DAC, TTC CI upper bounds >= 0 (FREEPOINT's DEPLOYABLE-SAFE, mechanical). DDC is reported, not in the bar. Also reported: the lever view composite - R0B and composite - C0A, and the families6 table (STRATEGIC paused).
* Identity gates before the number is read: (a) on tokens where the composite's pick equals BASE's (resp. R0B's, C0A's) pick, the per-token PDMS and sub-scores must equal that point's banked CSV row exactly; (b) the composite built with all channels from BASE reproduces the BASE picks on 12,146 / 12,146 tokens; (c) the harness status PASS.
* Outcomes: **N-PASS** (bar met) / **N-GAIN-LOST** (dPDMS lower bound <= 0, no adverse) / **N-SAFETY-LOST** (dPDMS lower bound > 0, adverse) / **N-BOTH** (neither).

| navtest outcome | committed next |
|---|---|
| N-PASS | report FP-COMPOSITE: a zero-training deployable point (adoption is the PI's). Next lever, **the anchored RL arm**, 3-line design: (1) in `scorer_finetune.py` give the NC, DAC, TTC, DDC and lane output slices of `score_head` (and their last-layer rows) lr 0 during EPO, so only EP/C are updated; (2) keep R0's reward, `--epo-lam 1.0`, seed 0; (3) a logit anchor of those slices to the step-0 scorer as a G-DVB-gated flag. Cost ESTIMATED from R0d's wall clock: 3.7 A40-h per arm, +1 control; launch-gate profile work first. Needs the PI |
| N-GAIN-LOST | the composite is safe and does not keep the gain: same anchored arm, because training EP/C against the anchored heads removes the calibration mismatch that inference-time mixing leaves |
| N-SAFETY-LOST, N-BOTH | the TTC / NC cost is not removable at inference by anchoring; R0e-pick (registered, T = 0.01, `--epo-con-temp`) is the committed A40 arm. State in RESULT what it does and does not address: it constrains the pick-aligned E_pi[TTC] and E_pi[NC] wherever the flip lives, but nothing in it anchors the lane or DAC outputs that step 1 found carry the flip |

## 6. Forecast (a prediction, not a bar)

Committed before any number: V1 and V2 retain the majority of B's EP gain (>= 0.5 of B's EP lift vs base0 on RL-val) and recover part, not all, of B's TTC loss; the RL-val label is SAFETY-LOST-VAL or PASS-VAL, not GAIN-LOST-VAL. I put no weight on the navtest label beyond that: the navtest NC/TTC losses of R0B are 0.25 / 0.40 and the anchored heads must remove both to within their CIs.
