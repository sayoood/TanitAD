# P6 toward 94.55: what the scorer line has left, and the two decisions it needs (Master Mind, 2026-11-10 ~22:00Z)

All numbers are E3 = full navtest (12,146 tokens), T1-class open loop, PDMS x100, unless stated. Paths are relative to
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/`.

## 1. Where P6's gap to the human score sits (MEASURED, point decomposition, no interval)

`2026-11-09-p6-navtest-read/gap_to_human/p6_gap_to_human.json`: P6 91.196 vs human 94.551, gap 3.355 =
**zero-multiplier picks 3.040** (DAC 0 on 2.05 % of tokens, NC 0 on 1.12 %; 384 tokens score 0)
+ **TTC 0 picks 1.527** (4.92 % of tokens) + comfort -0.016 − **EP surplus 1.195** (P6's progress on gated tokens 89.88
vs the human's 86.92). ⇒ Against the human, P6 is not too slow; it picks unsafe or off-road plans too often.

## 2. What is exhausted (each row in `Project Steering/GOALS_AND_CLAIMS.md`)

| family | best result vs P6 | row |
|---|---|---|
| scorer fine-tunes, 400 updates (C0, S1E, S1Z, S1ZSM; 7 runs) | recipe means +0.19 to +0.27, none PROVEN | H-REP-P6-1, H-S1ZSM-P6-1 |
| selectors fitted on navtrain (rule shape, veto, Platt, GBDT, policy gradient; S2/A2) | no gain | H-P6-SELECT-1 (S2), H-P6-SELECT-ARM-1 (A2) |
| ensembles over the fine-tunes (ENS3, seed ensembles, ENS9, PESS, LCB1, LCB1ND) | +0.17 to +0.33; PESS cuts PDMS-0 picks only 2.96 % → 2.74 % | H-PESS-P6-1 (PE3, A1-2) |
| 1,600-update dose (S1ZD) | **read pending, ~01:50Z** | H-DOSE-P6-1 |

Why they stall (MEASURED): the failing picks are out of sample (PDMS-0 picks 1.01 % on P6's own training sets, 3.06 % on
E1), P6 is over-confident on them (NC 0.991, DAC 0.941 on the failing pick, `why_miss/why_miss.json`), and the
fine-tunes share that blind spot (pessimism over eight of them moves 26 of ~360 such picks).

## 3. Decision A (recommended): give the selection a vision-built drivable-area and agent check

- **Target:** the 3.04-point zero-multiplier term, DAC first (2.05 % of tokens vs NC 1.12 %).
- **Upper bound, MEASURED (C3, 1,058 tokens, u400 base):** the P3 reference check fed the TRUE drivable area ranks
  proposals at AUC 0.932 (`auc_head`) vs the scorer's own DAC head 0.787 (`auc_scorer`),
  `2026-10-13-aux-s1-readout/p3_reference.json`.
- **What failed before, MEASURED:** probes on the FROZEN trunk features reached drivable IoU 0.18 to 0.27 (0-40 m) against
  a bar of 0.8, kill switch K2 fired (`2026-10-13-aux-p3-floor/full_1058/floor_instrument.json`). The features do not carry the drivable area
  linearly, so this needs a TRAINED camera-to-BEV head (labels from maps are admissible; inference stays vision-only).
- **It is an architecture change**, hence yours. On a go the first step costs no GPU: a design card (head, labels from the
  banked store `E:/Projects/TanitAD-artifacts/aux_drivable`, the IoU bar, and how the check enters the pick, e.g. as a
  veto that keeps NC and DAC separate from the situation classifier's output, per the goal-input rule).

## 4. Decision B (smaller expected gain): diverse ensemble members

Two or three scorer fine-tunes on disjoint halves of navtrain, read under the PESS rule. Pod cost about 2 L40-h per
400-update member in a P7 pause (S1ZSM: 2.1 h). Expected gain is small: every ensemble so far saturates at +0.2 to +0.33,
and PESS shows the members share P6's errors; diversity from data splits may not change that.

## 5. Not a decision

S1ZD's verdict (~01:50Z). If it is DOSE-A or DOSE-B, its card's next step (a seed-1 replicate, 4.6 L40-h in one P7 pause)
needs your go for pod time. If DOSE-C, its card points to Decision A.
