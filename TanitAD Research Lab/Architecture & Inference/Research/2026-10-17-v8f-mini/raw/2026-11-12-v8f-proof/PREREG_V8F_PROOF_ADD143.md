# PREREG_V8F_PROOF_ADD143 — CONTACTRH10KC: after 6 s of own driving, does RH close faster than UE on the same read (Q3 of record v0 - v_L), reader form?

Frozen before any tool or run of this card and before any value of it exists. Open at this time: row 208 (card ADD142, landed at 32574f11). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions; logged camera frames; the lead does not react). Never driving performance.

## 1. Why this card, and what it does not answer

Rows 207 and 208 read the reader-form contact cost of RH (row 206, +0.0073) on the fresh windows of eval-141, at two reader seeds: Q3 (v0 - v_L, RH minus UE in shared contact windows) +0.591 beside and +0.594 of record, the latter with an interval including zero. The 1,271-window grid of card ADD70 lies inside eval-141 (0 windows outside), so with no pre-roll it gives no new windows. New start states on the same three model seeds come only from the pre-roll of 30 steps (6 s of own driving). This card reads the contact mechanism in those AGED states. A result here does not transfer by itself to row 206's fresh reader-form cost; new fresh start states would need new clips (an encode, a PI decision).

Stated guess before data: `CONTACTRH10KC-SAME-READ`.

## 2. New rollouts (box 2, as card ADD140 rows k and l: readout K, rule every 5, heading bound 0.2, pre-roll 30 steps on the grid of card ADD70, wrapper `v8f_t1_eval_reader_rule_rzgfhq.py`)

Reader form `RR` at map level 0.50 with the reader seeds of card ADD140 (0, 1, 0 for model seeds 0, 1, 2). Runs `H_RQ50K_P30_s<m>` (RH_s<m>) and `U_RQ50K_P30_s<m>` (UE_s<m>): 6 runs, about 3 min each.

## 3. Quantities (the tool of card ADD142 with the pre-roll handling of `integ10k/read_longroll10k_n4.py`; Q1 to Q3, sets X and S, estimator as card ADD142, bootstrap seed 143)

Contact windows and first contact steps from the landed headway reader's cells under the pre-roll handling of record (`read_longroll10k_n4.py`: step 1 of a pre-rolled track taken from the track's own start point, not the frame origin). The tick of a window is the last rule tick of the MAIN phase (after the pre-roll) at or before its first contact step.

## 4. Controls (all must pass before any value is opened; a failed control stops the card, a successor is frozen)

* c0 runs: done files, the 1,271-window grid count of tick files, finite plans, rule and pre-roll logs; argv of each run equals its row; RH and UE dumps differ; the two runs of a seed have the same (clip, t0) list.
* c1 the tool's contact windows equal the landed headway reader's cells under the pre-roll handling of record (`read_longroll10k_n4.py`, as card ADD140 read R3).
* c2 a main-phase rule tick at or before every first contact step; the rule log's identity g0 = g_src + a_l - a_m within 1e-3 m on 20 planted ticks; and for those ticks the logged v0 is the rolled speed at the aged tick: it equals the dump's own speed at the tick's step after the pre-roll (within 1e-6), not the logged speed at the window's t0.
* c3 a planted +2.0 m gap error in X reads Q1 = +2.0; and a planted start offset (the n3 check of `read_longroll10k_n4.py`) reads as card ADD140's n3.
* c4 the tool's tests (dev box). c5 -c_src, RH minus UE in S, reads 0 within 1e-9 while v0 - v_L differs between the arms.

Names opened after the controls pass.

## 5. Names (as card ADD142)

`CONTACTRH10KC-SAME-READ` if Q3 is above zero (interval excludes zero) and Q1, Q2 are not separated; `-MISREAD` if Q1 or Q2 is above zero and Q3 is not; `-BOTH` if Q3 and one of Q1, Q2 are; `-UNRESOLVED` otherwise, or if X or S has fewer than 10 windows over three seeds. The four families: LONGITUDINAL read at the contact windows; LATERAL, TACTICAL, STRATEGIC not read, n = 0.

## 6. Cost

6 rollouts on box 2 (about 20 min) and minutes of CPU; box 2 per the PI's instruction of 2026-11-24.
