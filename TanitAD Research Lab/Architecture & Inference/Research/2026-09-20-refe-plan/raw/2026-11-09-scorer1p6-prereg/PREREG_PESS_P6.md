# PREREG_PESS_P6 — a pessimistic ensemble over P6's fine-tuned scorers (no training, no GPU)

**Status.** Frozen by the Master Mind, by sha256, on 2026-11-10 at about 21:28Z. Written BEFORE any navtest truth of
PESS, LCB1 or ENS9 was read, and BEFORE S1ZD's E3 read (its logit pass is running). It does not change
`PREREG_DOSE_P6.md` or any earlier card. PI 2026-11-09: "optimizing the P6 scorer to achieve the possible 97.9 ... incl.
rule based, machine learning based, RL based". This arm needs no pod time and no GPU: it reuses banked E3 logits and the
full navtest truth table.

## 0. What is known before the freeze

- MEASURED, E1 (1,058 navtest tokens, already read, `why_miss/why_miss.json`): P6 regret 5.96, of which picks with
  PDMS 0 are 2.74 points (3.06 % of tokens; FIT, P6 in-sample, 1.01 %). On a failing pick P6 gives NC 0.991 and DAC 0.941
  (the nearest clean proposal 0.974 / 0.859): the scorer is over-confident exactly where it fails, and it fails out of
  sample. 81 % of failing tokens hold at least one clean proposal (median 11 of 64).
- MEASURED, E3: plain logit-mean ensembles do not clear the bar. ENS3 (C0, C0r, S1E) +0.1650 [-0.011, +0.346] vs P6;
  seed ensembles ENS_C0 +0.3286 [+0.1404, +0.5188], ENS_S1E +0.2720, ENS_S1Z +0.2969 (`rep/read`, not keyed).
- LABEL-FREE (logits only, no truth, no K key; `pess/ens_labelfree.json`), 8 members (S1ZD not yet in): each fine-tune
  changes P6's pick on 45-53 % of E3 tokens; the member-logit SD at P6's pick has median 0.30 (NC), 0.35 (DAC),
  0.29 (TTC). against the pick of the members' mean PROBABILITY (the label-free script's reference, not ENS9's logit
  mean), the minimum rule changes 26.9 % of tokens and the 1-SD logit rule 17.3 %. So the rules move picks; whether
  they move them to better proposals is the question.

## 1. Members and arms

**Members:** every scorer-only fine-tune of P6 whose E3 logits are banked and pass G0-E3 (proposals bit-identical to
P6's): C0, C0r, C0r2, S1E, S1E1, S1Z, S1Z1, S1ZSM, S1ZD (nine). If S1ZD's pass fails or is not done at read time, the
read uses the eight and says so. P6 is the comparator, not a member. No weight, threshold or member choice is fitted.

All arms combine member logits per token, per proposal, per head into one logit tensor, then pick with the unchanged
navsim_v1 rule (`read_scorer1p6.pick_logits`, torch float32):

| arm | NC, DAC, TTC heads | EP, C, DDC heads | role |
|---|---|---|---|
| **ENS9** | mean of member logits | mean | reference: plain ensemble |
| **PESS** | **minimum** of member logits | mean | **primary** |
| **LCB1** | mean minus 1 x SD (ddof 0) of member logits | mean | secondary |

## 2. Bars (E3 = full navtest, 12,146 tokens, T1-class open loop, PDMS x100; paired log-cluster bootstrap, 10,000 draws, seed 20260927)

| id | metric | bar |
|---|---|---|
| **P1** | PESS - P6 | CI-lo > 0 and d > t x s_p = 0.4971 (s_p 0.1791, df 4, t 2.776; the single-run floor of `PREREG_DOSE_P6.md` D1) |
| **P2** | PESS - ENS9 | CI-lo > 0 (pessimism adds over plain ensembling) |
| **M** | zero-pick share (pick PDMS exactly 0), PESS - ENS9 | CI-hi < 0 |
| G1 | TTC, NC, DAC sub-scores of the pick, PESS - P6, in points | each >= -0.5 |
| G2 | regret on E2, PESS - P6 | CI-hi < +0.5 |
| G3 | overspeed share on E2, PESS - P6 | CI-hi < +0.10 |
| G4 | slow-pick share on E2 CRUISE, PESS - P6 | CI-hi < +0.10 |

The estimator answers "another draw of logs". Training variance is not resampled; ensembles of fixed members are read as
fixed scorers, and that is stated with the result. Inference is deterministic. LCB1 is read with the same bars, reported,
and can only name the next lever (it is not adopted under this card). EP x100 of the pick vs P6 is reported for every
arm (pessimism is expected to cost EP; G4 is the guard).

## 3. Outcomes (committed; label order PE1, PE2, PE3, PE4)

- **PE1 ADOPT** (P1, P2, M and G1-G4 hold): PESS becomes P6's selector. Next: goal status vs 94.55; then the same rule
  with more diverse members (fine-tunes on disjoint navtrain halves; pod time, PI check).
- **PE2 MECHANISM, TOO SMALL** (P2 holds, P1 fails): pessimism over members is real but the members are too alike.
  Next: member diversity (data-split fine-tunes, a PI check before pod time) under its own card.
- **PE3 MOVES ZERO PICKS, NOT PDMS** (M holds, P2 fails): read EP and TTC of the changed tokens; next: pessimism on NC and
  DAC only (LCB1's result, read here, decides between min and 1 SD).
- **PE4 SHARED BLIND SPOT** (neither P2 nor M): the members share P6's out-of-sample error, so no selector over them can
  see it. Next: the perception lever: a check of each proposal against an estimate of drivable area and agents built from
  richer features than the scorer reads (architecture change, the PI's decision), with the zero-GPU E1 evidence above as
  its design input.
- **Unnamed** if none matches; reported as such.

## 4. Gates and controls (the reader fails closed)

- G0-E3 for every member and P6; G0-PICK (P6's logits reproduce its banked pick on all 12,146 tokens); G0-E2 as in
  `read_scorer1p6`.
- Control C1: an "ensemble" of one member (S1ZSM) must reproduce S1ZSM's banked E3 PDMS 91.297 to float equality.
- Control C2: ENS3 rebuilt by this reader (mean of C0, C0r, S1E) must reproduce +0.1650 vs P6 to 1e-4.
- Control C3 (deliberate regression): PESS built with MAX instead of MIN must differ from PESS on at least 10 % of
  tokens, and the reader must report it as a different arm.

## 5. K ledger

One key per arm read on E3: `PESS:<sha256 of the combined logits>:E3`, `LCB1:...`, `ENS9:...` (K + 3). E2 guard reads are
not keyed (as `PREREG_S1ZSM_P6.md`). Best-of-K statements use Bonferroni intervals; P1 and P2 are PESS's own tests.
The four metric families are reported per arm vs P6; STRATEGIC is PAUSED (PI 2026-09-19), not a pass.
