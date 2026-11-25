# PREREG_V8F_PROOF_ADD147 — RULEE1A10K: does the lead rule acting every step also lower the unsafe and contact rates in AGED states, reader form, on the candidate (RH under readout K)?

Frozen before any tool or run of this card and before any value of it exists. Open at this time: rows 211 and 212 (cards ADD145, ADD146; read done, land pending the other session's read). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions; logged camera frames; the lead does not react). Never driving performance. PI, 2026-11-24, verbatim: "Try to Implement the levers without pod".

## 1. Why this card

Row 211: in the reader form on fresh windows, every step lowers U by -0.0142 [-0.0249, -0.0059] and contact by -0.0244 [-0.0404, -0.0099], no family WORSE. Row 206 found the candidate's cost after 6 s of own driving (aged states), and rows 207 to 209 could not settle the reader-form contact mechanism there. This card reads every step in the aged states of the 1,271-window grid of card ADD70 (pre-roll 30 steps), as card ADD143's runs `H_RQ50K_P30`.

Stated guess before data: `RULEE1A10K-U-LOWER`, C lower, families not separated.

## 2. Runs (box 2, RH_s0..s2, readout K, form RR at map level 0.50, reader seeds 0, 1, 0, heading bound 0.2, pre-roll 30 steps on the grid of card ADD70, wrapper rzgfhq 92a8034c unchanged)

| run | rule every |
|---|---|
| `H_RQ50K_P30` (card ADD143, reused) | 5 |
| `H_RQ50KE1_P30` (new, 3 runs) | 1 |

Argv of each new run equals that of `H_RQ50K_P30_s<m>` except `--rule-every 1` and the names. The rule acts every step in the pre-roll too (the policy of the arm).

## 3. Controls (all must pass before any value is opened; a failed control stops the card, a successor is frozen)

* r4 runs: done files, one dump file per grid clip with finite plans, rule logs, later-look logs and pre-roll logs; argv equal to the queue row and to the partner's apart from `--rule-every` and the names; the partner dump files equal the box-2 folders by sha256.
* the later looks happen: the later-look log of every new run has rows at q = 1 to 9 (the partner's at q = 5 only).
* the age read of card ADD140 (R3: `read_longroll10k_n4.py` grid engines with controls n1 to n3, n1 on card ADD139's `H_RLcA_P0` as card ADD140 amendment 2; route goal with s2 and s3 waived because both arms are pre-rolled), unchanged apart from the arms.

## 4. Reads and names (bar: spread of `H_RQ50K_P30` over the seeds)

`H_RQ50KE1_P30` minus `H_RQ50K_P30`: headway U (primary), C, F, and the four families with route goal. Name `RULEE1A10K-U-LOWER` if the pooled interval of U lies below zero, `-U-HIGHER` if above, `-U-NOT-SEPARATED` otherwise; families, C and F beside. Consequence committed: U not HIGHER, C not HIGHER and no family WORSE keeps every-step as the rule of the candidate's reader form at every age (row 211's consequence stands); U or C HIGHER, or a family WORSE, goes to the PI with its cost, together with row 211.

## 5. Cost

3 rollouts on box 2 (about 6 min each) and CPU. No training, no encode.
