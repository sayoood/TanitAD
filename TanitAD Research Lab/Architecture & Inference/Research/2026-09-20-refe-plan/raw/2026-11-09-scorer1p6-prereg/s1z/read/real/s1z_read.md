# S1Z read

**Status: READ**

- Card: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/PREREG_S1Z_P6.md` sha256 `8c7b6b697b94ccf1a4781a727ed28c10ed8b11878c33eeb2d016a507b36efa69` (frozen `8c7b6b697b94ccf1a4781a727ed28c10ed8b11878c33eeb2d016a507b36efa69`)
- Tier: E3 = T1-class open loop: NAVSIM v1.1 PDMS of the arm's own pick on the full navtest (self-action open loop, not closed loop); E2 = selection tier: PDM truth of P6's own 64 proposals on E2, the arm judged by which proposal it picks.
- Evidence class: MEASURED (ours; every input artifact and its sha256 is listed under 'inputs').
- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the evaluation set's own logs (sel_common.log_boot_idx draws via selc6_analyze.Boot). It answers 'another draw of logs' only. Training variance is read from the replicate floor F = |C0r - C0|; inference is deterministic (PREREG_P6 bit-identity check).

## Gates

- G0-PICK: **PASS** n_equal_to_banked_pick=12146, n=12146
- G0-P6CSV: **PASS** n_tokens_absent=0, n_invalid_rows=0
- G0-E2: **PASS** n_e2=2000, n_missing_in_table=0, n_missing_in_world=0, n_pdms_bit_equal=2000, n_sub_bit_equal=2000, max_abs_pdms_diff=0.0, formula_max_abs_diff=0.0, all_finite_valid=True, pick_is_argmax_of_p6_logits=2000, RED_shifted_pick_n_differ=1402
- G0-E3[C0]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[C0r]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1Z]: **PASS** n=12146, n_tokens_not_bit_identical=0

## Verdict

- Outcome: **Z5 SWAMPED** (all matches: Z5, Z6)
- Next lever: NOT PROVEN. Next: S1Z seed 1 and a second C0 replicate
- Adopted: **False**; verdict: **NO ADOPTION**; adverse: EP
- Goal status: **BELOW GOAL**
- K ledger: **K = 19** (K counts every navtest read in the programme's ledger (selectors/ledger_K.json); Batch 1 reads (C0, C0r, S1E) are in it only if they were written there (UNVERIFIED); best-of-K statements use these intervals, P1 is the arm's own test)

## Bars

| id | value | bar | result |
|---|---|---|---|
| P1 PDMS x100 E3 | S1Z 91.354 [90.578, 92.142] | vs C0: CI-lo>0 and abs>2F (F=+0.3796); vs P6: CI-lo>0 | vs C0 +0.1247 [+0.0192, +0.2335] False; vs P6 +0.1571 [-0.0368, +0.3606] False; **P1 False** |
| M2 failure share E3 | S1Z 0.0617, P6 0.0686, C0 0.0655 | CI-hi<0 vs P6 and vs C0 | vs P6 -0.0069 [-0.0102, -0.0037]; vs C0 -0.0039 [-0.0054, -0.0024]; **M2 True** |
| G1 TTC/NC/DAC pts E3 | TTC +0.412, NC +0.045, DAC +0.272 | each >= -0.5 | **True** |
| G2 delta regret E2 | -0.2231 [-0.7204, +0.2701] | CI-hi<+0.5 | **True** |
| G3 overspeed E2 | -0.0271 [-0.0536, +0.0036] | CI-hi<+0.10 vs P6 | **True** |
| G4 slow-pick E2 | +0.0426 [+0.0110, +0.0805] (S1Z +0.2209, P6 +0.1783) | CI-hi<+0.10 vs P6 | **True** |

## Family clause (arm - P6, paired CIs)

- adverse components: ['EP']
- STRATEGIC: PAUSED -- PI ruling Sayed 2026-09-19, PI_DECISION_QUEUE ITEM 25: "the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are driving with high quality. So the 4 layer rule is just temporarily paused and concerns only the strategic part." Not a pass; not omitted; lifting it is the PI's call.

## Reported, not deciding

- EP sub-score vs P6 (E3): -0.4488 [-0.6424, -0.2488]
- Shapley split E2: {"v_empty": 92.45874724657341, "v_all": 98.12277024745981, "shapley_points": {"NC": 0.637281525943725, "DAC": 1.1252142274127526, "EP": 1.6354815231160817, "TTC": 2.2660457244138428}, "sum": 5.664023000886402}
- top-K oracle curve E2: {'1': 92.45874724657341, '2': 93.55409832343429, '4': 94.82753249307405, '8': 95.98680827779818, '16': 96.88477836321988, '32': 97.71899275347438, '64': 98.12279047169775}
- head_within_token_AUC_E2: not computed: the card gives no pooling rule for a within-token AUC (per-token AUC then mean? tokens with one class?); computing one would invent the estimand
- near_twin_AUC_E2: not computed: needs the section 1 pair construction on raw poses (ADE <= 1.0 m among valid proposals, hard negatives of arm B's rule) from the E2 table's proposals; the table carries truth, not the pair list

## Ambiguities (S1Z)

- S1 M2 failure = the pick's PDMS == 0 (the harness score, an exact product with a zero factor) OR its TTC sub-score == 0, on all 12,146 E3 tokens, from the same score pool as P1. Share is a fraction (0 to 1); the diffs S1Z - P6 and S1Z - C0 use the same paired log-cluster bootstrap draws as P1; the bar CI-hi < 0 is strict with no slack (not a decimal bar).
- S2 G4 reuses the Batch 1 slow-pick definition (CRUISE tokens of E2 on the 0.5 s grid, pick's plan v4/v0 < 0.85; Batch 1 A1/A2), on E2's UNIFORM token set. P6_RESULT.md's 0.224 is a C3 stratum-weighted share on 20-step plans, so it is not the same number; the DEFINITION is shared, the value is read on E2. G4 uses the EPS slack of Batch 1 A12 (as G3).
- S3 G3 is read against P6 only (card line 64 has no 'and against C0' clause, unlike Batch 1's S1E). Adoption = P1 and G1 to G4.
- S4 P1 = both lines of the card row: S1Z vs C0 (CI-lo > 0 and |S1Z - C0| > 2F) AND S1Z vs P6 (CI-lo > 0). F = |C0r - C0| on PDMS x100 on E3 (Batch 1 A5); C0r is REQUIRED here.
- S5 Z-outcomes overlap as Batch 1's O-outcomes did (A7): every match is listed; the primary label follows Z1, Z2, Z3, Z5, Z6, Z4, Unnamed. Z3 = (P1 or M2 clears) and a guard fails. Z5 = S1Z vs C0 has CI-lo > 0 but |diff| <= 2F. Z6 = M2 holds, P1 fails, and NO guard failed ('no guard caused it'). Unnamed is reached only when an input is undefined (every defined combination of P1, M2, guards and the 2F test lands in Z1 to Z6).
- S6 Z3 lever: G4 failed -> speed-matched pairs; G1 failed -> remove the failing head(s) from L_twin; G2 or G3 failed alone -> no committed lever (stated, Shapley split on E2).
- S7 The Bonferroni interval is printed for the PDMS differences S1Z - P6 and S1Z - C0 at the ledger's running K (selcore.pct_pair). It is for best-of-K statements only; P1 is the arm's own test.
