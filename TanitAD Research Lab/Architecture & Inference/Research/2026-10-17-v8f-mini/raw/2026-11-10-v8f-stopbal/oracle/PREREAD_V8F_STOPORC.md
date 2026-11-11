# PREREAD_V8F_STOPORC: does O plan a stop when T's code says stop? (Master Mind, 2026-11-11 ~18:22; frozen before any number of this read)

## Why (MEASURED, `theory/stopbal/read/VERDICT_STOPBAL.md`)
STOPBAL moved T's stop hit rate on the 117 stop-point windows from 0.060 to 0.291 (BAL-L s0) and from 0.145 to 0.393 (s1), while stop-point tactical ADE moved +0.22 / +0.20 m (worse in point estimate, not separated). The committed next arm, the second dose, can deliver the card's PRIMARY only if O turns a stop decision into a stopping plan. This read measures that ceiling with zero training, on the dumps that exist.

## Read (CPU, zero training; tier: T1-kin FIRST TICK, an oracle diagnostic, never driving performance)
* Arms: **BAL-NULL_s0 and BAL-NULL_s1 decide**; BAL-NULL_rep gives the floor; the four lever arms are read beside. Batches, windows and code path are STOPDEC's (`theory/stopdec/stopdec_read.py` through `theory/stopbal/read/stopdec_arm.py`, the STOPBAL tree, `topk_shim.py`). The plan of a code is `forward_diag(batch, code)["traj30"]`.
* Codes, per window: **OWN** = T's own code (`forward_infer`). **F-GT** = OWN with the STOP_POINT goal entry set to 1.0 and the lon part set to the one-hot of STOP when the window's GT allowed lon set contains STOP, else the one-hot of the allowed class with T's highest probability (the counts of each case are reported). **F-GOAL** (goal entry only) and **F-LON** (lon part only, the same rule) are beside. Lat, the other goal entries and the constraint cells stay T's own.
* Windows: the 117 STOPDEC stop-point windows (asserted equal across arms). Control set: the other has-and-present windows (2,234 at the banked dump).
* **Primary cell:** tactical ADE (knots 11-30) of F-GT minus OWN on the stop-point windows. Paired episode-cluster bootstrap (`taniteval.ci`, B 2,000, seed 0; "another draw of episodes" only). **Floor** = |delta(NULL_rep) - delta(NULL_s0)|, read at seed 0 and applied to both seeds.
* **Reaches-O control:** F-GT applied to the control-set windows (the stop forced where the human does not stop). Its tactical ADE delta is read with the same estimator: a separated change shows that O reads the forced parts.
* Families: ADE / FDE per band (operative 1-10, tactical 11-30); LONGITUDINAL speed and acceleration MAE; LATERAL heading and yaw-rate MAE; TACTICAL: n = 0, reason "T's decision is the forced input of this read"; STRATEGIC: PAUSED (PI item 25).

## Controls that must read known values before any outcome
1. IDENT: forcing with OWN itself reproduces the TOP1 plan exactly (max abs 0.0 m), on every arm.
2. The forced code differs from OWN exactly in the entries named above (counted per window); all other entries are equal bit for bit.
3. Red arm: a mutant that forces nothing (F-GT := OWN) must read a primary delta of exactly 0.0 and make the reaches-O control read "not separated".

## Outcomes (both NULL seeds must agree, else SEEDS-DISAGREE)
| outcome | rule | next step |
|---|---|---|
| **O-FOLLOWS** | primary delta separated, better side (lower ADE), above the floor, at both seeds | O plans the stop. Decision levers can deliver P, and the size of the delta is their ceiling. The second dose stays the next arm; the perception card follows it if the dose reads BAL-FP-COSTS. |
| **O-MISREADS** | reaches-O control separated at both seeds, and the primary is not a gain at either seed | O reads the stop code but its stopping plan does not match the human's. The second dose waits. Next: a card supervising O's plan under a stop code (O-side lever). |
| **O-IGNORES** | reaches-O control not separated at either seed, and the primary is not a gain at either seed | O does not read T's stop parts. The dose and perception levers cannot reach P. Next: the O-side card, with the conditioning path checked first (read-back and gradient of the code into O). |
| **SEEDS-DISAGREE** | any other combination | a third NULL seed's read (the seed-3 init, when banked) before any outcome is quoted |
