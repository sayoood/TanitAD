# PREREG_V8F_PROOF_ADD142 — CONTACTRH10KB: on new rollouts, does RH close faster than UE on the same read (Q3 of record v0 - v_L)?

Frozen before any tool or run of this card and before any value of it exists. Open at this time: row 207 (card ADD141, landed at 3ddc0427). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions; logged camera frames; the lead does not react). Never driving performance.

## 1. Why this card

Row 207 read `CONTACTRH10K-UNRESOLVED`: its closing quantity of record (-c_src) is filled from the LOGGED own speed (`distrule/v8f_t1_eval_rule.py` lines 198 and 212) and is identical for both models by construction. The quantity the rule compares, v0 (the rolled own speed, which depends on the model) against v_L (the rule's lead speed), was read beside: RH minus UE +0.591 m/s [+0.198, +0.831] in the shared contact windows. That value has been seen, so it cannot be its own test. This card reads it as Q3 of record on rollouts no read has seen.

Stated guess before data: `CONTACTRH10KB-SAME-READ` (RH faster on the same read).

## 2. New rollouts (box 2, as card ADD140: readout K, rule every 5, heading bound 0.2, eval-141 grid, safe mode, one job at a time)

The reader form `RR` at map level 0.50 with the OTHER reader seed: model seed 0 with reader `TK16_s1` and map `QMAP_LR16_s1_q50.json`; model seed 1 with `TK16_s0` and `QMAP_LR16_s0_q50.json`; model seed 2 with `TK16_s1` and `QMAP_LR16_s1_q50.json` (in card ADD140 seed 2 used reader seed 0). Runs `H_RQ50Kx_s<m>` (RH_s<m>) and `U_RQ50Kx_s<m>` (UE_s<m>), m = 0, 1, 2: 6 runs, about 21 min. The (clip, t0) windows are those of eval-141, as in card ADD140; the rollouts, their rule logs and their contact windows are new. No value of card ADD140's or card ADD141's runs enters this read.

## 3. Quantities (the tool of card ADD141 with one change of its closing of record; its windows, ticks and estimator unchanged)

Sets X (contact for RH, not for UE) and S (contact for both) per seed from the landed headway reader's contact cells. At the last rule tick at or before each window's first contact step:
* Q1 misread: reader's signed gap error, X minus S (RH), as card ADD141.
* Q2 presence misses, X minus S (RH), as card ADD141.
* Q3 of record: v0 - v_L (the rolled own speed minus the rule's lead speed, both from the rule log), RH minus UE in S, paired by window. Beside: -c_src, which must read 0 exactly (a control, c5).
* Episode-cluster bootstrap, 2,000 draws, seed 142.

## 4. Controls (all must pass before any value is opened; a failed control stops the card, a successor is frozen)

* c0 runs: done files, 141 tick files, finite plans, rule logs; argv of each run equals its row (reader and map seeds as section 2); RH and UE dumps differ.
* c1 to c4 as card ADD141 and its amendment 1 (contact cells equal the landed reader's; ticks exist and the rule log's identity holds on 20 planted ticks; a planted +2.0 m gap error reads +2.0; the tool's tests).
* c5 -c_src, RH minus UE in S, reads 0 within 1e-9 (the structural identity row 207 found), and v0 - v_L differs between the two arms in at least one window of S (Q3 is not structurally zero).

Names opened after the controls pass.

## 5. Names

* `CONTACTRH10KB-SAME-READ` if Q3 is above zero (interval excludes zero) and Q1, Q2 are not separated.
* `CONTACTRH10KB-MISREAD` if Q1 or Q2 is above zero (interval excludes zero) and Q3 is not.
* `CONTACTRH10KB-BOTH` if Q3 and one of Q1, Q2 are above zero.
* `CONTACTRH10KB-UNRESOLVED` otherwise, or if X has fewer than 10 windows over three seeds, or S fewer than 10.
* Next lever committed: SAME-READ sends the contact cost to the rule's response (the row-204 rule change, a PI decision) or to the recipe (the other session); MISREAD to the vision-only reader.

## 6. Cost

6 rollouts on box 2 (about 21 min) and minutes of CPU. Box 2 per the PI's instruction of 2026-11-24.
