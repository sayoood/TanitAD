# PREREAD_V8F_STOPHARD: is the missing 1.5 m a confidence gap? T's OWN stop decision, hardened at inference (Master Mind, 2026-11-11 ~18:47; frozen before any number of this read)

## Why (MEASURED, `VERDICT_STOPORC.md`)
Forcing the GT stop decision into T's code (STOP_POINT goal 1.0, lon one-hot STOP) cuts first-tick tactical ADE on the 117 stop-point windows by about 1.5 m on every arm. The STOPBAL lever arms fire their own stop decision on two to four times as many of those windows, yet their OWN first-tick ADE equals the NULL arms'. O reads T's code as probabilities, so a decision that only just crosses 0.5 or wins the argmax may hand O little stop mass. This read writes T's OWN decision into the code as a hard value: **no GT and no new input**, a deployable inference-time transform of T's own output (the vision-only rule holds; no situation-classifier output enters any goal input).

## Read (CPU, zero training; tier: T1-kin FIRST TICK, never driving performance)
* Machinery, windows and arms as `PREREAD_V8F_STOPORC.md` (`stoporc_read.py`): seven STOPBAL dumps, the STOPBAL tree, `forward_diag(batch, code)`.
* Codes: **OWN** = T's own. **HARD** = OWN with (a) the STOP_POINT goal entry set to 1.0 where T's own p(STOP_POINT) >= 0.5, and (b) the lon part set to one-hot STOP where T's own lon argmax is STOP. Nothing else changes. HARD-GOAL ((a) only) and HARD-LON ((b) only) are beside. Per-window counts of (a) and (b) are reported, and must equal the STOPDEC decision counts of the same arm.
* Cells, HARD minus OWN, paired episode-cluster bootstrap (`taniteval.ci`, B 2,000, seed 0; "another draw of episodes" only): **S** = tactical ADE (knots 11-30) on the 117 stop-point windows; **A** = tactical ADE on all has-and-present windows. Each cell's floor = |delta(BAL-NULL_rep) - delta(BAL-NULL_s0)|, read at seed 0, applied to both seeds. Gain = separated, better side, above the floor; cost = separated, worse side, above the floor.
* Families beside, per window set and band: ADE / FDE; LONGITUDINAL speed and acceleration MAE; LATERAL heading and yaw-rate MAE; TACTICAL n = 0 (T's decision is the transformed input); STRATEGIC: PAUSED (PI item 25).

## Controls (known values before any outcome)
1. IDENT: HARD with an unreachable threshold (goal 2.0, lon never) equals OWN exactly (0.0 m) on every arm.
2. HARD differs from OWN only in the named entries, only on windows where the decision fires (counted).
3. The decision counts equal STOPDEC's for the same arm (STOPBAL M hit and false-positive counts).

## Outcomes, per lever (BAL-G with HARD-GOAL as its decision, BAL-L with HARD-LON), both seeds must agree, else SEEDS-DISAGREE
| outcome | rule | next step |
|---|---|---|
| **HARD-HELPS** | S a gain at both seeds, A not a cost at either | Confidence is the gap. Next: stage 2, the same transform inside the T1-kin rollout (the tier of STOPBAL's P), pre-registered before it runs. The second dose is read with and without it. |
| **HARD-TRADES** | S a gain at both seeds, A a cost at either | Hardening also hardens false stops. Next: a hardening threshold set on TRAINING windows only, pre-registered; the dose waits on it. |
| **HARD-NULL** | S not a gain at either seed | Confidence is not the gap; coverage is. The committed path stands: second dose, then perception. |
The NULL arms are read beside under the same rules (their decision rarely fires) and do not enter the outcome.
