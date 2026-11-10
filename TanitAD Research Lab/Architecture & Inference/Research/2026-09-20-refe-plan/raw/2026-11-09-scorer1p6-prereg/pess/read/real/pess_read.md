# PESS read

**Status: READ**

- Card: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/PREREG_PESS_P6.md` sha256 `185de4067ede99ff9aca0c0f559e5a62f0989872aef07b8b7c4bd8657d42f380` (frozen `185de4067ede99ff9aca0c0f559e5a62f0989872aef07b8b7c4bd8657d42f380`)
- Tier: T1-class open loop: NAVSIM v1.1 PDMS of the arm's own pick on the full navtest (self-action open loop, not closed loop)
- Evidence class: MEASURED (ours; every input artifact and its sha256 is listed under 'inputs').
- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the evaluation set's own logs (sel_common.log_boot_idx draws via selc6_analyze.Boot). It answers 'another draw of logs' only. Training variance is read from the replicate floor F = |C0r - C0|; inference is deterministic (PREREG_P6 bit-identity check). Training variance is not resampled: ensembles of fixed members are read as fixed scorers. Inference is deterministic.
- Members (8): C0, C0r, S1E, C0r2, S1E1, S1Z1, S1Z, S1ZSM
- Arms scored: ENS9, PESS, LCB1

## Gates

- G0-PICK: **PASS** n_equal_to_banked_pick=12146, n=12146
- G0-P6LOGITS: **PASS** P6 dir logits bit-identical to the table's logits=True
- G0-TABLE: **PASS** n_tokens_absent_in_p6_csv=0, n_pdms_equal_to_p6_csv=12146
- G0-E2: **PASS** n_e2=2000, n_missing_in_table=0, n_missing_in_world=0, n_pdms_bit_equal=2000, n_sub_bit_equal=2000, max_abs_pdms_diff=0.0, formula_max_abs_diff=0.0, all_finite_valid=True, pick_is_argmax_of_p6_logits=2000, RED_shifted_pick_n_differ=1402
- G0-E2-vs-TABLE: **PASS** E2 table pdms bit-identical to the full table on the E2 tokens=True
- G0-E3[C0]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[C0r]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1E]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[C0r2]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1E1]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1Z1]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1Z]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[S1ZSM]: **PASS** n=12146, n_tokens_not_bit_identical=0

## Verdict

- Outcome: **PE3 MOVES ZERO PICKS, NOT PDMS** (all matches: PE3)
  - PE3: read EP and TTC of the changed tokens; next: pessimism on NC and DAC only (LCB1's result, read here, decides between min and 1 SD)
- Inputs: {"P1": false, "P2": false, "M": true, "guards": {"G1": true, "G2": true, "G3": true, "G4": true}, "point_PESS": 91.47901718634304}
- Goal status of PESS: **BELOW GOAL**

- K ledger: K = 33 (K counts every navtest read in the programme's ledger; best-of-K statements use these intervals; P1 and P2 are PESS's own tests)

## Per arm vs P6 (E3, PDMS x100; paired log-cluster bootstrap)

| arm | PDMS | minus P6 | P1 clears | EP x100 minus P6 | zero-pick share | picks changed vs P6 | logits sha256 |
|---|---|---|---|---|---|---|---|
| ENS9 | 91.414 [90.609, 92.221] | +0.2174 [+0.0337, +0.4084] | False | -0.0821 [-0.2623, +0.1080] | 0.02956 | 5460 | `0c2a575f6ef884fb` |
| PESS | 91.479 [90.687, 92.283] | +0.2825 [+0.0756, +0.5042] | False | -0.5194 [-0.7055, -0.3157] | 0.02742 | 6669 | `8912f1d04ab3e225` |
| LCB1 | 91.493 [90.688, 92.294] | +0.2968 [+0.1140, +0.4918] | False | -0.3773 [-0.5572, -0.1797] | 0.02791 | 6232 | `ad04543143ac494e` |

## P2, M and guards

| arm | P2 (arm - ENS9) | M zero-pick share (arm - ENS9) | G1 | G2 | G3 | G4 |
|---|---|---|---|---|---|---|
| ENS9 | - | - | True | True | True | True |
| PESS | +0.0651 [-0.0699, +0.2177] False | -0.0021 [-0.0036, -0.0008] True | True | True | True | True |
| LCB1 | +0.0794 [-0.0212, +0.1958] False | -0.0016 [-0.0029, -0.0006] True | True | True | True | True |

## Four metric families vs P6 (per arm: adverse components)

- ENS9: adverse components: none
- PESS: adverse components: ['EP', 'curvature_mae_1pm']
- LCB1: adverse components: ['EP']
- STRATEGIC: PAUSED -- PI ruling Sayed 2026-09-19, PI_DECISION_QUEUE ITEM 25: "the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are driving with high quality. So the 4 layer rule is just temporarily paused and concerns only the strategic part." Not a pass; not omitted; lifting it is the PI's call.

## Ambiguities

- T1 P1 d is the POINT difference PESS - P6 in PDMS x100 (E3), compared strictly with the constant 0.4971 printed in the card (not recomputed as 2.776 x 0.1791 = 0.49718); CI-lo > 0 strict. The earlier readers' 2F rule is not used: the card fixes the floor.
- T2 M is the share of E3 tokens whose pick has PDMS exactly 0.0 (card: 'pick PDMS exactly 0'), NOT the S1ZSM reader's M2 (PDMS 0 or TTC 0). The failure share (PDMS 0 or TTC 0) is also printed, as reported-not-deciding. CI over the same paired log-cluster draws as P1.
- T3 G1 to G4 are the Batch 1 / S1Z definitions applied to PESS - P6: G1 sub-score points x100 from the table's sub-scores at the pick (each >= -0.5, EPS slack of read_scorer1p6.g1_ok); G2 regret on the 2,000 E2 tokens (best valid proposal PDMS x100 minus the pick's); G3 overspeed (plan v4/v0 > 1.15) share on E2 CRUISE, CI-hi < +0.10; G4 slow (v4/v0 < 0.85) share on E2 CRUISE, CI-hi < +0.10 (EPS slack of read_s1z.g4_ok). E2 quantities use the combined E3 logits restricted to the E2 tokens (the E2 logit files of members are not read; a member's e2 logits could differ from its E3 rows by float drift).
- T4 PE labels overlap and every match is listed; the primary label follows PE1, PE2, PE3, PE4, Unnamed. PE1 = P1 and P2 and M and G1..G4; PE2 = P2 holds and P1 fails (M and the guards are not conditions of PE2); PE3 = M holds and P2 fails; PE4 = neither P2 nor M holds. P1 holds, P2 holds, M fails (or a guard fails) matches none -> Unnamed.
- T5 Scoring = lookup in the full truth table at the pick; the table's valid flag must be True and the value finite at every pick (else the read stops). A gate checks that the table reproduces P6's csv scores on 100 % of tokens (G0-TABLE) and the E2 table on the E2 tokens; a check on S1ZSM's own picks vs its e3_changed_scores.csv runs in control c1.
- T6 Combine in float32: mean = np.mean(axis=0, dtype=float32) (the ENS3 convention of read_scorer1p6); min = np.min; SD = ddof-0 population SD computed in float64 from the float32 member logits, LCB1 = mean32 - SD cast to float32. With one member min = mean and SD = 0 exactly, so every arm is that member's own logits bit for bit (control c1 asserts it).
- T7 The ledger key is <ARM>:<sha256 of the arm's combined float32 [12146,64,6] C-contiguous logits>:E3 for each of PESS, LCB1, ENS9 (K + 3). The arm name stays ENS9 even if only eight members are present; the members used are listed in the output and the sha differs by member set.
- T8 Members are loaded as given by --member; the card's nine are C0, C0r, C0r2, S1E, S1E1, S1Z, S1Z1, S1ZSM, S1ZD. If S1ZD is not given the read states it uses n members. Each member must pass G0-E3 (proposals bit-identical to P6's) and its token set must equal the table's.
- T9 Family clause: the Batch 1 family tables of the arm vs P6 (paired CIs) incl. EP x100; STRATEGIC printed as PAUSED with the PI ruling. EP x100 vs P6 is reported for every arm.
- T10 The Bonferroni intervals at the ledger's running K are printed per arm for best-of-K statements; P1 and P2 are PESS's own tests.
