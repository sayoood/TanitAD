# BASE_SOLVER_AMEND5 (written by the MM, 2026-11-17, mechanically from AMEND_HWMPLAN_4's result)

BASE_SOLVER: A0
BRANCH: B3

## The rule (AMEND_HWMPLAN_5 section 4, frozen sha256 de6f7477...)
* **B1:** an AMEND_4 lever arm reads CEILING. Then the base is that arm.
* **B2:** else an arm reads NARROWS. Then the base is that arm.
* **B3:** else the base is A0's upper levels: LON-PLAN's banked `score/A0_s{k}/plans_hwm_oracle_s{0,1}.npz`.

## The input (`amend4/readout/RESULT_AMEND4.json`, sha256 7f99ed76bf919ecb..., tier T0, paired episode-cluster bootstrap B 2,000 seed 0; controls_ok true, C-RS ok at both chassis)
| arm | R35_s0 | R35_s1 | label | effective |
|---|---|---|---|---|
| P | NULL | HARM | L2IMG-SEEDS-DISAGREE | L2IMG-HARM |
| P23 | HARM | HARM | L23IMG-HARM | L23IMG-HARM |

No arm reads CEILING and no arm reads NARROWS, so the branch is **B3** and the base is **A0**. The reader's own field agrees: `base_solver_for_RA_read: "A0"`, `multiplicity_guard_applies: false`. This was the expected branch (MM_RULINGS_AMEND5 M0, ESTIMATED from AMEND_4's TRAIN record).

## What follows (MM_RULINGS_AMEND5 M1)
Under B3, RA-L1 is the committed lever. RA-L2 on P's banked SCORE files is computed beside it, and no outcome is decided on it.
