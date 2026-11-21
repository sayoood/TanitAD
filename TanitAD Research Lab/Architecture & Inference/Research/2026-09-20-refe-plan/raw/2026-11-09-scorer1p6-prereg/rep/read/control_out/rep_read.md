**PARTIAL_NOT_THE_CARD_READ**

# PREREG_REP_P6 read

**Status: READ**

- Card `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/PREREG_REP_P6.md` sha256 `4f59ae7cec6c812dea93991af349ded67aa04c6fb915b08acc8322c6805c4131` (frozen `4f59ae7cec6c812dea93991af349ded67aa04c6fb915b08acc8322c6805c4131`)
- Tier: E3 = T1-class open loop: NAVSIM v1.1 PDMS of the arm's own pick on the full navtest (self-action open loop, not closed loop)
- Evidence class: MEASURED (ours; every input artifact and its sha256 is listed under 'inputs')
- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the evaluation set's own logs (sel_common.log_boot_idx draws via selc6_analyze.Boot). It answers 'another draw of logs' only. Training variance is read from the replicate floor F = |C0r - C0|; inference is deterministic (PREREG_P6 bit-identity check).

## Verdict

- Outcome: **REP-B NOISE** (all matches: REP-B)
- Next lever: a 400-update scorer fine-tune is not distinguishable from training noise on E3 for the recipes read. Next: the dose arm (1,600 updates) of the recipe with the largest mean, with a PI check before pod time. The cards' 2*F bars are re-stated with F3 next to F (reported; no verdict is re-decided)
- F3 = +0.3796; s_p = 0.2684; df = 1; t = 12.706204736174694
- K ledger: K = None (no ledger path given: K NOT COUNTED)

## Recipes (mean of runs vs P6, R1; S1E and S1Z also R2 vs the C0 recipe)

| recipe | runs | mean E3 PDMS | d vs P6 [95 %] | R1 thr | R1 | R2 d vs C0 [95 %] | R2 thr | R2 | guards | adoptable | goal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | C0, C0r | 91.419 [90.625, 92.212] | +0.2222 [+0.0386, +0.4154] | 2.4113 | **NOT PROVEN** | n/a | n/a | n/a | FAIL C0 | False | BELOW GOAL |
| S1E | S1E | 91.271 [90.461, 92.098] | +0.0744 [-0.1273, +0.2720] | 3.4101 | **NOT PROVEN** | -0.1479 [-0.3002, -0.0077] | 4.1765 | **NOT PROVEN** | all pass | False | BELOW GOAL |

## Runs (E3, paired CI vs P6)

| run | recipe | PDMS x100 | vs P6 [95 %] | guards | changed picks |
|---|---|---|---|---|---|
| C0 | C0 | 91.229 [90.413, 92.043] | +0.0325 [-0.1584, +0.2307] | FAIL | 5998 |
| C0r | C0 | 91.608 [90.827, 92.396] | +0.4120 [+0.1971, +0.6364] | pass | 5852 |
| S1E | S1E | 91.271 [90.461, 92.098] | +0.0744 [-0.1273, +0.2720] | pass | 5517 |

## Seed ensembles ENS_R (reported, not deciding)

- C0: READ; 91.411 [90.600, 92.222]; vs P6 +0.2141 [+0.0259, +0.4091]; vs best single run C0r -0.1980 [-0.3370, -0.0655]
- S1E: NOT READ: one run only

## Families

- STRATEGIC: PAUSED -- PI ruling Sayed 2026-09-19, PI_DECISION_QUEUE ITEM 25: "the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are driving with high quality. So the 4 layer rule is just temporarily paused and concerns only the strategic part." Not a pass; not omitted; lifting it is the PI's call.
- longitudinal / lateral / tactical components per run are in the json under runs.<run>.four_families_vs_P6; adverse components:
  - C0: ['EP']
  - C0r: none
  - S1E: none

## Control (against the Batch 1 verdict)

- runs ['C0', 'C0r', 'S1E']: leaves compared 224, mismatches 0: **EQUAL**

## Ambiguities (this reader)

- P1 The recipe mean inside each draw is the mean of the runs' per-token E3 PDMS vectors; its bootstrap is the bootstrap of that mean vector on the same draws as every run (identical to taking the mean of per-run draw statistics because every run shares the token set and cluster weights).
- P2 t: the card's printed values for df 2, 3, 4 are used as written; any other df is computed with scipy.stats.t.ppf(0.975, df) if scipy is present and otherwise REFUSED. A partial set of runs (for example the control's C0, C0r, S1E: df 1) is therefore read with scipy and stamped PARTIAL_NOT_THE_CARD_READ.
- P3 R2 uses sqrt(1/n_R + 1/n_C0) with the C0 recipe's actual run count; the card writes 1/3, which equals this for n_C0 = 3 (flagged when n_C0 is not 3).
- P4 G3 of an S1E-recipe run is read against P6 and against the real C0 run (as Batch 1 does for S1E); G3 of a C0-recipe run against P6; G3 of an S1Z-recipe run against P6 only (the S1Z card). G1 to G3 are Batch 1's, G4 and M2 are read_s1z's.
- P5 The floor clause threshold is 2 * 0.3796 = 0.7592 exactly (the card writes it as '= 0.759').
- P6 Outcome overlap: every match is listed; the primary label follows REP-A, REP-C, REP-B, Unnamed. REP-B is 'no recipe PROVEN on R1' among the recipes read; a recipe PROVEN but not adoptable matches none of A/B (REP-C may still hold).
- P7 Goal status per recipe: GOAL MET iff the recipe mean >= 94.55 (Batch 1 EPS slack in the arm's disfavour), else BELOW GOAL.
- P8 K ledger: one key per run E3 read (REP:<run>:<logits sha>:E3) and one per recipe-mean read (REP:<recipe>:mean:<sha of the run shas>:E3). ENS_R reads are reported and are NOT keyed (the card keys runs and recipe means only). A real read re-reads C0, C0r and S1E, which Batch 1 already read; the REP keys are separate from any Batch 1 key.
- P9 A duplicate arm (two run names with the same e3_logits sha256) is refused unless --smoke-standin: a real seed replicate is never bit-identical.
