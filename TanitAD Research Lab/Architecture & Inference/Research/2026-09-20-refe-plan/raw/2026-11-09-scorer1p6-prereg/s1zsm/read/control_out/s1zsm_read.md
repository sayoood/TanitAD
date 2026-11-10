**CONTROL RUN: S1Z's own pass dir is fed as the arm. The outcome below is the control's, not an S1ZSM result.**

# S1ZSM read

**Status: READ**

- Card: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/PREREG_S1ZSM_P6.md` sha256 `ac75b2891e794de800102a6d5a0f0353e8092e5f976f31b543ac7700811ad6b8` (frozen `ac75b2891e794de800102a6d5a0f0353e8092e5f976f31b543ac7700811ad6b8`)
- Tier: E3 = T1-class open loop: NAVSIM v1.1 PDMS of the arm's own pick on the full navtest (self-action open loop, not closed loop); E2 = selection tier: PDM truth of P6's own 64 proposals on E2, the arm judged by which proposal it picks.
- Evidence class: MEASURED (ours; every input artifact and its sha256 is listed under 'inputs').
- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the evaluation set's own logs (sel_common.log_boot_idx draws via selc6_analyze.Boot). It answers 'another draw of logs' only. Training variance is read from the replicate floor F = |C0r - C0|; inference is deterministic (PREREG_P6 bit-identity check).

## Gates

- G0-PICK: **PASS** n_equal_to_banked_pick=12146, n=12146
- G0-P6CSV: **PASS** n_tokens_absent=0, n_invalid_rows=0
- G0-E2: **PASS** n_e2=2000, n_missing_in_table=0, n_missing_in_world=0, n_pdms_bit_equal=2000, n_sub_bit_equal=2000, max_abs_pdms_diff=0.0, formula_max_abs_diff=0.0, all_finite_valid=True, pick_is_argmax_of_p6_logits=2000, RED_shifted_pick_n_differ=1402
- G0-E3[C0]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[C0r]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1ZSM]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1Z reference]: **PASS** n=12146, n_tokens_not_bit_identical=0

## Verdict

- Outcome: **SM3 SWAMPED** (all matches: SM3, SM5)
  - SM3: S1ZSM seed 1, read under the R1/R2 rule
  - SM5: speed direction is not the EP leak. Next, zero GPU: the Shapley split on E2, and the per-head slow-pick share. Then TTC removed from L_twin under its own card
- Adopted: **False**; verdict: **NO ADOPTION**; adverse: EP
- Goal status: **BELOW GOAL**
- K ledger: **K = None** (no ledger written (--no-ledger): K NOT COUNTED)

## Bars

| id | value | bar | result |
|---|---|---|---|
| P1 PDMS x100 E3 | S1ZSM 91.354 [90.578, 92.142] | vs C0: CI-lo>0 and abs>2F (F=+0.3796); vs P6: CI-lo>0 | vs C0 +0.1247 [+0.0192, +0.2335] False; vs P6 +0.1571 [-0.0368, +0.3606] False; **P1 False** |
| M2 failure share E3 | S1ZSM 0.0617, P6 0.0686 | S1ZSM-P6 CI-hi<0 | -0.0069 [-0.0102, -0.0037]; **M2 True** |
| M3 EP x100 E3 | S1ZSM 86.5856, S1Z 86.5856, P6 87.0344 | S1ZSM-S1Z CI-lo>0 | +0.0000 [+0.0000, +0.0000]; **M3 False** |
| G1 TTC/NC/DAC pts E3 | TTC +0.412, NC +0.045, DAC +0.272 | each >= -0.5 | **True** |
| G2 delta regret E2 | -0.2231 [-0.7204, +0.2701] | CI-hi<+0.5 | **True** |
| G3 overspeed E2 | -0.0271 [-0.0536, +0.0036] | CI-hi<+0.10 vs P6 | **True** |
| G4 slow-pick E2 | +0.0426 [+0.0110, +0.0805] (S1ZSM +0.2209, P6 +0.1783) | CI-hi<+0.10 vs P6 | **True** |

## Family clause (arm - P6, paired CIs)

- adverse components: ['EP']
- STRATEGIC: PAUSED -- PI ruling Sayed 2026-09-19, PI_DECISION_QUEUE ITEM 25: "the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are driving with high quality. So the 4 layer rule is just temporarily paused and concerns only the strategic part." Not a pass; not omitted; lifting it is the PI's call.

## Reported, not deciding

- S1ZSM - S1Z PDMS x100 E3: +0.0000 [+0.0000, +0.0000]
- S1ZSM - S1Z failure share E3: +0.0000 [+0.0000, +0.0000]
- S1ZSM - S1Z slow-pick share E2 CRUISE: +0.0000 [+0.0000, +0.0000]
- EP sub-score vs P6 (E3): -0.4488 [-0.6424, -0.2488]
- Shapley split E2: {"v_empty": 92.45874724657341, "v_all": 98.12277024745981, "shapley_points": {"NC": 0.637281525943725, "DAC": 1.1252142274127526, "EP": 1.6354815231160817, "TTC": 2.2660457244138428}, "sum": 5.664023000886402}
- top-K oracle curve E2: {'1': 92.45874724657341, '2': 93.55409832343429, '4': 94.82753249307405, '8': 95.98680827779818, '16': 96.88477836321988, '32': 97.71899275347438, '64': 98.12279047169775}
- pair shares of the run: {"given": false}
- seed SD next to 2F: {"given": false}

## Control (S1Z as the arm against the banked s1z_read.json)

- leaves compared 89, mismatches 0: **EQUAL**
- M3 (S1Z - S1Z): {'point': 0.0, 'lo': 0.0, 'hi': 0.0, 'n': 12146} exactly zero with CI [0, 0]: **True**
- control PASS: **True**

## Ambiguities (S1ZSM)

- T1 P1, G1 to G4, goal status, family clause, Shapley split and top-K are read_s1z.s1z_read's own, with the S1ZSM arm in its S1Z slot (nothing re-implemented). The S1Z reader's M2 also computes a vs-C0 clause; that clause is NOT used here (card M2 row: S1ZSM - P6 only).
- T2 M2 = the failure-share difference S1ZSM - P6 on E3 (pick PDMS 0 or TTC 0; same paired draws as P1), CI-hi < 0 strict, no slack.
- T3 M3 = EP sub-score x100 of the pick on E3 (harness sub-score 'ego_progress' x100, as G1's sub-scores), S1ZSM - S1Z, paired log-cluster bootstrap on the same draws, CI-lo > 0 strict. S1Z's picks are read from its pass dir (logits + e3_changed_scores.csv) through the same score pool; S1Z must pass G0-E3 as an arm and must have every changed pick scored.
- T4 SM labels overlap as Batch 1's O-outcomes did: every match is listed; the primary label follows SM1, SM2, SM3, SM4, SM5, SM6, Unnamed (the card's order). SM1 = adopted and point >= 94.55; SM2 = adopted and point < 94.55; SM3 = S1ZSM vs C0 has CI-lo > 0 but |diff| <= 2F; SM4 = M2 and M3 hold, P1 fails, not SM3; SM5 = M2 holds, M3 fails; SM6 = M2 fails. Unnamed is reached when no label matches (for example P1 and M2 and M3 hold but a guard fails and the gain is not swamped) or when an input is undefined.
- T5 SM5 and SM6 are not conditioned on P1 (the card gives them only the M2/M3 conditions), so SM5 or SM6 can be listed next to SM1/SM2/SM3. The primary label still follows the card order.
- T6 The pair shares of the run (reported only) are read from --run-log: a pair_weight.json, or a JSON/JSONL log containing an object with 'sets_with_pair_share'. The card does not fix the file; if none is given or none is found the item reads 'not given'.
- T7 --rep-read: the S1Z-style rep_read.json (read_rep.py) key seed.{s_p, df, t_0.975}; t*s_p is printed next to 2F. Missing file or key: 'not given'.
- T8 The K ledger key is S1ZSM:<sha256 of the arm's e3_logits.npz>:E3 (family S1ZSM); the Bonferroni intervals are for best-of-K statements only, P1 is the arm's own test.
- T9 G0 for the reference S1Z: its proposals must be bit-identical to P6's (G0-E3, the same gate every arm passes). The 'logits bit-identical to the banked picks on unchanged tokens' check of the brief is read as that gate plus G0-PICK (P6 logits pick == banked pick); read_s1z has no further logit-level gate.
