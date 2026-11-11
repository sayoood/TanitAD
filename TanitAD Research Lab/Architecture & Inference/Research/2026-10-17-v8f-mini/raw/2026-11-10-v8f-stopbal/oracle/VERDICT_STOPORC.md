# VERDICT: PREREAD_V8F_STOPORC = O-FOLLOWS at both seeds (Master Mind, 2026-11-11 ~18:45)

Card frozen 18:22:20 (sha256 `49eb7d7b`, re-hashed equal after the run). Read by a Sonnet agent (`stoporc_read.py`, `RESULT_STOPORC.json`); re-checked by the Master Mind: `test_stoporc.py` 22 passed (CPU), outcome and floor read from the JSON. Tier: **T1-kin first tick, an oracle diagnostic, never driving performance**. Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 ("another draw of episodes" only). STRATEGIC: **PAUSED** (PI item 25). TACTICAL: n = 0 (T's decision is this read's forced input). All numbers MEASURED.

Controls (all seven arms, 0 problems): IDENT vs OWN 0.0 m and OWN vs the dumped plan 0.0 m; F-GT changes exactly the 8 named entries (goal 6, lon 27-33) on 117 of 117 stop-point windows and nothing else; the red-arm mutant (F-GT := OWN) reads every delta exactly 0.0 and a reaches-O control that is not separated. Windows equal STOPDEC's (117 stop-point, 25 episodes; 2,234 control). All 117 stop-point windows have STOP in the GT allowed lon set; 18 control windows have no allowed lon class, keep T's lon and are counted.

## Primary: tactical ADE (knots 11-30), F-GT minus OWN, 117 stop-point windows (m)
| arm | OWN -> F-GT | delta [CI] | reaches-O, 2,234 control windows |
|---|---|---|---|
| **BAL-NULL_s0** | 5.02 -> 3.48 | **-1.539** [-2.034, -1.036] | +0.258 [0.186, 0.329] |
| **BAL-NULL_s1** | 4.94 -> 3.47 | **-1.477** [-2.018, -0.952] | +0.117 [0.045, 0.187] |
| BAL-NULL_rep (floor) | 4.06 -> 2.92 | -1.134 [-1.588, -0.652] | +0.377 [0.287, 0.466] |
| BAL-G_s0 / s1 (beside) | 4.94 / 4.85 -> 3.50 / 3.51 | -1.439 / -1.347, separated | +0.199 / +0.082, separated |
| BAL-L_s0 / s1 (beside) | 5.11 / 4.84 -> 3.55 / 3.60 | -1.560 / -1.238, separated | +0.242 / +0.099, separated |
Floor 0.405 m. Beside at NULL s0 / s1: F-GOAL alone -1.04 / -0.98, F-LON alone -0.88 / -0.86 (all separated). At NULL_s0, F-GT also moves tactical FDE -4.03, speed MAE -0.94 and acceleration MAE -0.09 (all separated, better), and heading MAE +1.43 and yaw-rate MAE +2.93 (separated, worse; see below).

## Outcome, as committed: **O-FOLLOWS**
The primary is separated, on the better side and above the floor at both NULL seeds. O plans the stop when T's code says stop, and the ceiling of a correct stop decision on these windows is about **1.5 m tactical ADE** (31 % of OWN's 5.0 m). Committed next step: decision levers can deliver P, so the second dose stays the next arm, and the perception card follows it if the dose reads BAL-FP-COSTS.

## What it adds to STOPBAL (MEASURED, no bar)
* **The ceiling is large, and the levers reached none of it.** At first tick the lever arms' OWN tactical ADE on stop-point windows (BAL-G 4.94 / 4.85, BAL-L 5.11 / 4.84 m) is not below the NULL arms' (5.02 / 4.94), although their stop decision fires on two to four times as many of these windows (STOPBAL hit 0.22-0.39 against 0.06-0.15). Forcing the decision brings every arm, lever or not, to about 3.5 m.
* **A candidate cause (HYPOTHESIS, testable with zero training):** O reads T's code as probabilities. F-GT writes a hard 1.0 / one-hot; a "hit" in STOPBAL only means the STOP entry crossed 0.5 or won the argmax, possibly with little mass. If O needs a confident stop code, the levers changed the decision count without changing what O receives. `PREREAD_V8F_STOPHARD.md` tests this: T's OWN stop decision, hardened at inference (no GT), on the existing dumps.
* **Lateral beside cells.** Heading +1.43 and yaw-rate +2.93 MAE (worse) under F-GT are read on stop-point windows where the forced plan slows to a stop. Near standstill a heading tangent is ill-defined (`RETR-2026-09-26-YAWMASK`), so these two cells may score standstill geometry rather than steering. Not claimed either way until the masked form is read.
