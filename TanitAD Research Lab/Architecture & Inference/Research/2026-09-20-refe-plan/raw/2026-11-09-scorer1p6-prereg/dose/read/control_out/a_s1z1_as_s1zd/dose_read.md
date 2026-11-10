**CONTROL_NOT_A_RESULT** -- a control run: S1ZD is fed another arm's logits. No number here is a result and none may be quoted as one.

# PREREG_DOSE_P6 read

**Status: READ**

- Card `E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p6base/PREREG_DOSE_P6.md` sha256 `037c127235b51c374780f31ad2e93855ac7c141a4051f89498f40f1beff52ebb` (frozen `037c127235b51c374780f31ad2e93855ac7c141a4051f89498f40f1beff52ebb`)
- Tier: E3 = T1-class open loop: NAVSIM v1.1 PDMS of the arm's own pick on the full navtest (self-action open loop, not closed loop)
- Evidence class: MEASURED (ours; every input artifact and its sha256 is listed under 'inputs')
- Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, over the evaluation set's own logs (sel_common.log_boot_idx draws via selc6_analyze.Boot). It answers 'another draw of logs' only. Training variance is read from the replicate floor F = |C0r - C0|; inference is deterministic (PREREG_P6 bit-identity check).

## Verdict

- Outcome: **DOSE-C SATURATED** (all matches: DOSE-C)
- Next lever: a longer fine-tune of the scorer's loss does not buy PDMS. Next: leave the fine-tune line and attack the largest regret term, zero-multiplier picks (2.81 of 6.70 points), with a lever outside the scorer-loss family, under its own card
- D1 (S1ZD - P6): d +0.3861 [+0.1705, +0.6148], bar CI-lo > 0 and d > 0.4972: **NOT PROVEN**
- D2 (S1ZD - S1Z recipe mean of S1Z, S1Z1): d +0.1145 [+0.0139, +0.2274], bar CI-lo > 0 and d > 0.6089: **NOT PROVEN**
- Guards G1 to G4: all pass; M2 clears (reported): True
- Goal 94.55: S1ZD 91.583 -> **BELOW GOAL**
- S1ZD E3 PDMS x100: 91.583 [90.809, 92.363]; S1Z recipe mean 91.468 [90.700, 92.253]
- K ledger: K = 30 (K counts every navtest read in the programme's ledger; D1 and D2 are this card's own pre-registered tests (card section 5))

## Runs (E3, paired CI vs P6)

| run | PDMS x100 | vs P6 [95 %] | G1 | G2 | G3 | G4 | changed picks |
|---|---|---|---|---|---|---|---|
| S1Z | 91.354 [90.578, 92.142] | +0.1571 [-0.0368, +0.3606] | ok | ok | ok | ok | 6454 |
| S1Z1 | 91.583 [90.809, 92.363] | +0.3861 [+0.1705, +0.6148] | ok | ok | ok | ok | 6234 |
| S1ZD | 91.583 [90.809, 92.363] | +0.3861 [+0.1705, +0.6148] | ok | ok | ok | ok | 6234 |

## Regret split (reported, not deciding; x100 PDMS points)

S1ZD: pick 91.583, best of 64 97.900, regret 6.318; P6: pick 91.196, best 97.900, regret 6.704

| class | P6 n | P6 contribution | S1ZD n | S1ZD contribution | S1ZD - P6 [95 %] |
|---|---|---|---|---|---|
| zero-multiplier picks (pick PDMS 0, a non-zero proposal existed) | 355 | 2.810 | 303 | 2.393 | -0.4175 [-0.6122, -0.2307] |
| zero pick, all 64 proposals score 0 | 29 | 0.000 | 29 | 0.000 | +0.0000 [+0.0000, +0.0000] |
| TTC-0 picks (non-zero pick, TTC sub-score 0) | 449 | 1.578 | 403 | 1.408 | -0.1692 [-0.3214, -0.0198] |
| rest (EP, comfort) | 11313 | 2.316 | 11411 | 2.517 | +0.2007 [+0.1368, +0.2658] |
| total regret | | 6.704 | | 6.318 | -0.3861 [-0.6148, -0.1705] |

## Trainer held-out curve (monitor, not deciding)

- not given

## Families (vs P6; STRATEGIC PAUSED)

- STRATEGIC: PAUSED -- PI ruling Sayed 2026-09-19, PI_DECISION_QUEUE ITEM 25: "the strategic layer will be only switched off temporarily until we proved that both tactical and operative layers are driving with high quality. So the 4 layer rule is just temporarily paused and concerns only the strategic part." Not a pass; not omitted; lifting it is the PI's call.
- longitudinal / lateral / tactical components are in the json under runs.S1ZD.four_families_vs_P6; adverse components:
  - S1Z: ['EP']
  - S1Z1: none
  - S1ZD: none

## G0-P6PASS: PASS (picks equal 12146/12146; proposals bit-identical True; logits bit-identical True)

## Ambiguities (this reader)

- D1 Runs: S1Z, S1Z1 (the D2 reference) and S1ZD each go through read_s1z.s1z_read in its S1Z slot, as read_rep does for S1Z-recipe runs; E3 scores of changed picks come from table_p6final_c.npz (sha 3bda2080...), never from an arm csv. C0 and C0r are required because s1z_read needs them (P1's zero point and the floor F = |C0r - C0|); the card's CLI lists only the S1Z recipe runs, so --c0 and --c0r are ADDED inputs. They decide nothing in this card.
- D2 'Guards' = G1, G2, G3 (Batch 1, vs P6) and G4 (CRUISE slow-pick share) -- all four must pass, exactly as read_rep.py's guards_ok for an S1Z-recipe run. M2 (failure share, S1Z card) is REPORTED next to them (as read_rep reports it) and does not decide. The S1Z/S1Z1 reference runs' guards are printed, not deciding.
- D3 Thresholds are computed from the card's printed s_p and t (2.776 * 0.1791 = 0.49718, times sqrt(1.5) = 0.60892) and compared strictly; the card's rounded 0.4972 / 0.6089 are printed next to them (a test pins that they round to the card's values). They differ by < 2e-5.
- D4 D2's reference is the mean of the S1Z and S1Z1 per-token E3 PDMS vectors inside each draw (card section 2), on the same draws as D1 and every other statistic. PROVEN iff CI-lo > 0 and d > 0.6089, both strict.
- D5 Outcome labels follow section 4 literally: A = D1 and D2 PROVEN and all guards pass; B = D1 PROVEN, D2 not; D = D1 PROVEN and a guard fails; C = D1 NOT PROVEN. Every match is listed; the primary label follows A, B, D, C. B carries no guard clause on the card, so B and D can both match (B first). Unnamed only when D1 or D2 is undefined (non-finite CI).
- D6 Regret split = split.py of the P6 navtest read (raw/2026-11-09-p6-navtest-read/selection/split.py) on the card's truth table table_p6final_c.npz, with the arm's pick (navsim_v1 rule on its logits, torch float32, first index on ties) in place of the stored pick; zero-multiplier = pick PDMS 0 while a non-zero proposal existed, TTC-0 = non-zero pick with TTC sub-score 0, rest = every other token; contribution = mean over ALL tokens of (best of 64 - pick) x 100 restricted to the class. Intervals: the shared 10,000 log-cluster draws of this read (split.py used its own multinomial draws, so its printed CIs differ in the later digits; the POINT values are the same).
- D7 --sft-log: a JSON-lines file (sft.jsonl or the stdout log); every row with event 'eval' is listed (tag, n, base and B held-out pick x100, B - base with its CI, B zero picks). The card says every 200 updates; the trainer's default --eval-every is 400, so the tags present are printed as found. A monitor only (PREREG_SCORER1P6 section 2): in-sample for P6, decides nothing. Not given: 'not given'.
- D8 --p6-pass: the dev-box arm pass's P6_control/e3_logits.npz must reproduce the banked P6 pick on 100 % of tokens and carry bit-identical proposals (gate G0-P6PASS); its logits' bit-equality with the banked P6 logits is reported. A failing gate is READ STOPPED (the pass pipeline would not be the banked P6's).
- D9 K ledger: one key per S1ZD E3 read, DOSE:S1ZD:<sha256 of e3_logits.npz>:E3 (family DOSE; style of REP:<run>:<sha>:E3 and S1ZSM:<sha>:E3); S1Z and S1Z1 are re-read as references and NOT keyed (their REP keys exist). D1 and D2 are this card's own pre-registered tests; the Bonferroni interval of d vs P6 at K is for best-of-K statements only.
- D10 A real read refuses an S1ZD whose e3_logits.npz is bit-identical to S1Z's, S1Z1's, C0's, C0r's or P6_control's (a 1,600-update arm is never identical to them); --control-standin allows it, stamps CONTROL_NOT_A_RESULT, writes only under F:/tanitad_scratch/dose_read_work and never touches the real ledger.
