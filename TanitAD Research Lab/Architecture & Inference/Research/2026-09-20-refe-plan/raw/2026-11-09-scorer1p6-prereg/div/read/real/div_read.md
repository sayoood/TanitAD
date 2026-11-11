# DIV read

**Status: READ**

- Card: `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/PREREG_DIV_P6.md` sha256 `e5fb2f8e4123fcfe3a599b0cc71537bab187716c6a5d3e46735ce094bacf6fac` (frozen `e5fb2f8e4123fcfe3a599b0cc71537bab187716c6a5d3e46735ce094bacf6fac`)
- Tier: T1-class open loop: NAVSIM v1.1 PDMS of the arm's own pick on the full navtest (self-action open loop, not closed loop)
- Evidence class: MEASURED (ours; every input artifact and its sha256 is listed under 'inputs').
- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the evaluation set's own logs (sel_common.log_boot_idx draws via selc6_analyze.Boot). It answers 'another draw of logs' only. Training variance is read from the replicate floor F = |C0r - C0|; inference is deterministic (PREREG_P6 bit-identity check). Training variance is not resampled: ensembles of fixed members are read as fixed scorers. Inference is deterministic.
- Members (10): C0, C0r, S1E, C0r2, S1E1, S1Z1, S1Z, S1ZSM, DV0, DV1

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
- G0-E3[DV0]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-E3[DV1]: **PASS** n=12146, n_tokens_not_bit_identical=0
- G0-PESS8: **PASS** banked_json=E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/pess/read/real/pess_read.json, banked_json_sha256=4feb7f0a9c48348378fde92f39257a2887bc873fc339e92b6a958db037e15a72, banked_members=['C0', 'C0r', 'S1E', 'C0r2', 'S1E1', 'S1Z1', 'S1Z', 'S1ZSM'], members_equal_banked_order=True, banked_sha256=8912f1d04ab3e22571397d53580ddafebc3beb78e1e27d34fe21db25e5bf7ae5, rebuilt_sha256=8912f1d04ab3e22571397d53580ddafebc3beb78e1e27d34fe21db25e5bf7ae5, sha256_equal=True, sha256_has_card_prefix=True, banked_PDMS_x100=91.47901718634304, rebuilt_PDMS_x100=91.47901718634304, PDMS_equal_exactly=True

## Verdict

- Outcome: **DV3 NO DIVERSITY GAIN** (all matches: DV3)
  - DV3: the shared error sits in P6's frozen trunk, not in the fine-tune data. The ensemble line is closed; Decision A
- Inputs: {"Q1": false, "Q2a": false, "Q2b": false, "M_reported": false, "guards": {"G1": true, "G2": true, "G3": true, "G4": true}, "point_PESS-10": 91.42858201455567}
- Goal status of PESS-10: **BELOW GOAL**

## Bars

| bar | estimate | clears |
|---|---|---|
| Q1 PESS-DV - PESS-SEED | +0.0659 [-0.0903, +0.2251] | False |
| Q2a PESS-10 - P6 (floor 0.4971) | +0.2321 [+0.0377, +0.4417] | False |
| Q2b PESS-10 - PESS-8 | -0.0504 [-0.1341, +0.0313] | False |
| M zero-pick share PESS-DV - PESS-SEED | +0.0000 [-0.0018, +0.0018] (DV 0.02832, SEED 0.02832) | False |

- K ledger: K = 40 (before 35); K counts every navtest read in the programme's ledger; expected K_before + 5; best-of-K statements use these intervals; Q1, Q2, M are the card's own tests

## Arms (E3, PDMS x100; paired log-cluster bootstrap)

| arm | members | PDMS | minus P6 | EP x100 minus P6 | zero-pick share | picks changed vs P6 | G1 | G2 | G3 | G4 | logits sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DV0 | DV0 | 91.332 [90.530, 92.132] | +0.1356 [-0.0691, +0.3365] | -0.0961 [-0.2895, +0.1034] | 0.03096 | 5731 | True | True | True | True | `876ec4d8389ddd55` |
| DV1 | DV1 | 91.409 [90.608, 92.214] | +0.2122 [+0.0174, +0.4147] | -0.0377 [-0.2565, +0.1930] | 0.03055 | 5702 | True | True | True | True | `c82cc0ce421fa170` |
| PESS-DV | DV0,DV1 | 91.518 [90.772, 92.290] | +0.3219 [+0.1176, +0.5262] | -0.2321 [-0.4395, -0.0186] | 0.02832 | 5874 | True | True | True | True | `814fd44032fd0a07` |
| PESS-SEED | C0,C0r | 91.453 [90.642, 92.263] | +0.2561 [+0.0718, +0.4472] | -0.2723 [-0.4578, -0.0700] | 0.02832 | 6106 | True | True | True | True | `e43f13412203c950` |
| PESS-10 | C0,C0r,S1E,C0r2,S1E1,S1Z1,S1Z,S1ZSM,DV0,DV1 | 91.429 [90.657, 92.208] | +0.2321 [+0.0377, +0.4417] | -0.6585 [-0.8624, -0.4368] | 0.02758 | 6729 | True | True | True | True | `84b33d871310f6a0` |
| PESS-8 | C0,C0r,S1E,C0r2,S1E1,S1Z1,S1Z,S1ZSM | 91.479 [90.687, 92.283] | +0.2825 [+0.0756, +0.5042] | - | 0.02742 | 6669 | - | - | - | - | `8912f1d04ab3e225` |
| C0 | C0 | 91.229 [90.413, 92.043] | +0.0325 [-0.1584, +0.2307] | - | 0.03005 | 5998 | - | - | - | - | `fdf2f4e45ee93482` |
| C0r | C0r | 91.608 [90.827, 92.396] | +0.4120 [+0.1971, +0.6364] | - | 0.02865 | 5852 | - | - | - | - | `c7b30a0b852bc056` |

## Reported pairs

- DV0_minus_C0: +0.1031 [-0.1047, +0.3028]
- DV1_minus_C0r: -0.1998 [-0.3649, -0.0388]

## Four metric families vs P6 (adverse components per arm)

- DV0: none
- DV1: none
- PESS-DV: ['EP']
- PESS-SEED: ['EP']
- PESS-10: ['EP', 'curvature_mae_1pm']
- STRATEGIC: PAUSED -- PI ruling Sayed 2026-09-19, PI_DECISION_QUEUE ITEM 25: "the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are driving with high quality. So the 4 layer rule is just temporarily paused and concerns only the strategic part." Not a pass; not omitted; lifting it is the PI's call.

## Ambiguities

- D1 Every arm is PESS (read_pess.combine, rule PESS) over the named members; DV0 and DV1 as single arms are bit-for-bit the member's own logits (one member: min = mean). Member ORDER is part of the combined-logit sha256 (float32 mean); PESS-8 uses the banked order C0, C0r, S1E, C0r2, S1E1, S1Z1, S1Z, S1ZSM and PESS-10 appends DV0, DV1.
- D2 Q2 has two parts, both required: PESS-10 - P6 with CI-lo > 0 and the POINT d strictly above the printed constant 0.4971 (as P1 of the PESS card), and PESS-10 - PESS-8 with CI-lo > 0. 'Q2 fails' = not both parts.
- D3 Outcomes, card section 4: DV1 = Q2 (both parts) and G1 to G4; DV2 = Q1 holds and Q2 fails; DV3 = Q1 fails. M is a reported bar and is not a condition of any label. Labels may overlap (Q1 fails and Q2 and the guards hold matches DV1 and DV3): every match is listed, the primary follows DV1, DV2, DV3, Unnamed. Q1 holds, Q2 holds, a guard fails matches none -> Unnamed. Any undefined input -> Unnamed.
- D4 M is the share of E3 tokens whose pick has PDMS exactly 0.0, difference PESS-DV - PESS-SEED, over the same paired log-cluster draws as Q1; CI-hi < 0 strict.
- D5 G1 to G4 are read_pess.guards_vs_p6 (Batch 1 / S1Z definitions) for PESS-10 vs P6; E2 quantities use PESS-10's E3 combined logits restricted to the E2 tokens. Guards are also printed for the other arms, not deciding.
- D6 Scoring = lookup in the full truth table at the pick (valid and finite at every pick or the read stops). G0 gates of read_pess run for P6 and every member (G0-E3: proposals bit-identical to P6's). A further gate G0-PESS8 requires PESS-8 rebuilt here to equal the banked PESS read (combined-logit sha256 and PDMS x100) exactly; otherwise READ STOPPED and no verdict is written.
- D7 Ledger keys <ARM>:<sha256 of the combined float32 [12146,64,6] logits>:E3 for DV0, DV1, PESS-DV, PESS-SEED, PESS-10 (K + 5). PESS-8, C0 and C0r are already-read arms and are not keyed. A key that is already in the ledger is not added again (selcore.Ledger.add), so K + 5 holds only if all five are new. Controls never write the ledger.
- D8 Reported, not deciding (card section 3): DV0 and DV1 each vs P6 and vs C0 / C0r (paired); the failure share and the four metric families per arm vs P6 (STRATEGIC PAUSED). The trainer held-out monitor is read from the chain's training log, not by this tool.
