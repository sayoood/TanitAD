# PREREG_V8F_PROOF_ADD144 — RULEFC10K: does the lead rule act better every step and with a forecast that carries the lead's measured change of speed, on the candidate (RH under readout K)?

Frozen before any tool or run of this card and before any value of it exists. Open at this time: row 209 (landed at 87a655da). Tier **T1-kin/hb0.2** (kinematic rollout on the model's own actions; logged camera frames; the lead does not react). Never driving performance. PI, 2026-11-24, verbatim: "Try to Implement the levers without pod" (in answer to the three levers of row 209, the first being this rule change).

## 1. Why this card

Rows 203 and 204 (label form, line models, readout K): the remaining unsafe windows are the rule's own response; at the last tick before the first unsafe step the rule is on and at its braking limit in every window, and the lead slowing below the rule's constant-speed forecast is the larger part in 0.8072 [0.6800, 0.8824]. The rule (`distrule/dist_rule.py`) acts every 5 steps (1.0 s) under readout K and forecasts the lead at constant speed: cap a_c(i) = g0 + v_L * dt * i - max(D, tau * v'(i-1)) (line 162). Row 206 made RH under readout K the candidate. This card tests the two parts of the lever, alone and together, on the candidate, in the label form (the rule's ceiling; label inputs are a reference, never a candidate).

Stated guess before data: every step alone lowers U a little (not separated); the forecast lowers U (separated); both together lower U the most; the families are not separated, LONGITUDINAL may read WORSE (more braking). Names guessed: `RULEFC10K-E1-U-NOT-SEPARATED`, `RULEFC10K-FC-U-LOWER`, `RULEFC10K-E1FC-U-LOWER`.

## 2. The change of the rule (wrapper variant; `dist_rule.py` and the landed wrappers are not edited)

* E1: the rule acts every step: `--rule-every 1` (an existing flag of the study wrapper).
* FC: the lead's displacement in the cap is the constant-acceleration forecast with the lead stopping at speed 0: S_L(i) = integral over 0..dt*i of max(0, v_L + a_L t) dt, replacing v_L * dt * i; a_L is the lead's measured change of speed over the last 1.0 s, from the same input that gives v_L: in the label form a_L = (v_L(j) - v_L(j - 5)) / (5 dt), v_L(j) = logged own speed at j plus the label closing at j (the label form's v_L), using only steps at or before j (causal). With a_L = 0 the cap is the rule of record bit for bit (control r1).
* dt = 0.2 s; D, tau and the braking limit unchanged (3.0 m, 1.2 s, 6.0 m/s2).

## 3. Runs (box 2, RH_s0..s2, readout K, label form `RLc`, heading bound 0.2, eval-141, 2,399 windows, wrapper of the fresh runs rzgfh plus the FC variant)

| run | rule every | forecast |
|---|---|---|
| `H_RLcK` (card ADD140, reused, controls passed) | 5 | constant speed (of record) |
| `H_RLcE1` | 1 | constant speed |
| `H_RLcFC` | 5 | measured change of speed |
| `H_RLcE1FC` | 1 | measured change of speed |

9 new runs, order of record m = 0, 1, 2, each E1, FC, E1FC.

## 4. Controls (all must pass before any value is opened; a failed control stops the card, a successor is frozen)

* r1 identity: the FC variant with a_L forced to 0 reproduces the rule of record's output bit for bit on a planted batch, and a run of the FC wrapper with a_L forced to 0 at seed 0 on 20 windows equals `H_RLcK`'s dumps on those windows.
* r2 planted forecast: for a planted lead with a_L = -2 m/s2 the cap equals the analytic S_L(i) within 1e-9, and with the lead stopping inside the horizon S_L stays constant after the stop.
* r3 causal: a_L at tick j uses no label at a step after j (checked by a planted future label that must not change a_L).
* r4 runs: done files, 141 tick files, finite plans, rule logs (with a_L logged for FC runs); argv of each run equals its row; the partner `H_RLcK` folders equal the files read in card ADD140.
* the landed readers' controls (headway, families, route goal).

Names opened after the controls pass.

## 5. Reads (landed readers unchanged; paired episode-cluster bootstrap pooled over three seeds; families with the two-level reader, bar the spread of `H_RLcK` over the seeds)

Each of E1, FC, E1FC minus `H_RLcK`: headway U, C, F (primary for this card), and the four families with route goal. Names per arm X in {E1, FC, E1FC}: `RULEFC10K-X-U-LOWER` if U is lower (interval excludes zero), `-U-HIGHER` if higher, `-U-NOT-SEPARATED` otherwise; with the family readings beside each name (a family WORSE is named in the row). Consequence committed: an arm with U LOWER and no family WORSE is the rule of the candidate's label form to be carried to the reader form (a successor card); an arm with U LOWER and a family WORSE goes to the PI with its cost.

## 6. Cost

9 rollouts on box 2 (about 3 to 4 min each, the every-step arms somewhat longer), and CPU. No training, no encode.
