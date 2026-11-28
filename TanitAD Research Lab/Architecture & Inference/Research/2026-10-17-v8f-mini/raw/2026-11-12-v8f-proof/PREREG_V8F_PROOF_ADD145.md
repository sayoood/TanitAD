# PREREG_V8F_PROOF_ADD145 — RULEE1R10K: does the lead rule acting every step lower the unsafe rate in the READER form (vision-only lead read) on the candidate (RH under readout K)?

Frozen before any tool or run of this card and before any value of it exists. Open at this time: row 210 (card ADD144, read done, land pending the other session's read). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions; logged camera frames; the lead does not react). Never driving performance. PI, 2026-11-24, verbatim: "Try to Implement the levers without pod".

## 1. Why this card

Row 210 (label form, a reference): the rule acting every step lowers U by -0.0148 [-0.0248, -0.0069], the forecast by -0.0136, both by -0.0243, no family WORSE; the card's committed consequence carries each to the reader form. The reader form `RR` (presence and gap from the token reader through the q50 map; closing speed zero) is the vision-only form of the rule. Its lead speed is the own logged speed (closing zero), so the forecast of card ADD144 has no lead-speed input there; the every-step part carries as it stands (`--rule-every 1` is an existing, tested option for form RR). This card reads that part.

Stated guess before data: `RULEE1R10K-U-LOWER`, smaller than in the label form; contact C not separated; families not separated.

## 2. Runs (box 2, RH_s0..s2, readout K, form RR at map level 0.50, reader seeds 0, 1, 0 as card ADD140 row j, heading bound 0.2, eval-141, 2,399 windows, wrapper rzgfh 373ffe72 unchanged)

| run | rule every |
|---|---|
| `H_RQ50K` (card ADD140 row j, reused) | 5 |
| `H_RQ50KE1` (new, 3 runs) | 1 |

Argv of each new run equals that of `H_RQ50K_s<m>` except `--rule-every 1` and the names.

## 3. Controls (all must pass before any value is opened; a failed control stops the card, a successor is frozen)

* r4 runs: done files, 141 dump files with finite plans, rule logs and the later-look logs (`rule_log_mid_*`, `*_rd.npz`); argv equal to the queue row and to the partner's argv apart from `--rule-every` and the names; the partner `H_RQ50K` dump files equal the box-2 folders by sha256.
* the later looks happen: the later-look log of every new run has rows at q = 1 to 9 (and the partner's at q = 5 only).
* the landed readers' controls (headway, families, route goal).

## 4. Reads and names (landed readers unchanged; paired episode-cluster bootstrap pooled over three seeds; bar: spread of `H_RQ50K` over the seeds)

`H_RQ50KE1` minus `H_RQ50K`: headway U (primary), C, F, and the four families with route goal. Name `RULEE1R10K-U-LOWER` if the pooled interval of U lies below zero, `-U-HIGHER` if above, `-U-NOT-SEPARATED` otherwise; families beside; C and F beside. Consequence committed: U LOWER with no family WORSE and C not HIGHER makes every-step the rule of the candidate's reader form (specification section 1, the rule line, via the other session); U LOWER with a family WORSE or C HIGHER goes to the PI with its cost; otherwise the candidate's reader form keeps rule-every 5, and the reader-form forecast (a lead-speed input from the reader) is the next arm.

## 5. Cost

3 rollouts on box 2 (about 3 to 5 min each; the every-step reader form reads the reader at 9 more looks per tick) and CPU. No training, no encode.
