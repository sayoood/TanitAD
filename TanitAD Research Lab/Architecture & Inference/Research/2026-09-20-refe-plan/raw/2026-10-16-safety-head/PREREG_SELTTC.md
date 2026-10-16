# PREREG_SELTTC — a TTC-heavier DEPLOYED selection on the real R0 states (zero training)

Written 2026-10-16 ~20:30Z on the MM's instruction, BEFORE any number below was computed. sha256 in `PREREG_SELTTC.sha256`.
Why: A2 (`realstates.json`) located 51-70 % of the real TTC loss in arm B's PROGRESS rows: the pick is pushed toward faster proposals. A selection that weighs TTC more heavily acts on exactly that push, at deploy time, with no training. It is the cheapest lever left that the validated proxy can judge on states that really lose TTC.
Evidence class of every number: ESTIMATED (real banked scorer states, cached visual context, CPU fp32, offline selection read; not T1; single training seed per state; deterministic selection, so no inference-seed variance).

## 1. States, pools, estimator

The 8 banked states of phead control (d): `rls1-{r0,r0b,c0}/scorer_states/scorer_states_u{u}.pt` (A and B), on the 364 cached sets (train 302 + val 62). Paired log-cluster bootstrap, 10,000, seed 20260927, pooled. Decision on R0 u250 / u500 / u1000; R0b reported; C0 is the floor.

## 2. Candidate selections (a fixed set of three; no sweep, no picking by result)

Each rule replaces `sel_score` (`scorer_finetune.py:750-752`: s = NC * DAC * (5 EP + 5 TTC + 2 C) / 12 * lane) in BOTH the policy pi and the pick, for BOTH arms, read on each arm's own logits (p = sigmoid):

| id | rule | parameter (fixed here) | identity value |
|---|---|---|---|
| **S1 lin** | s + NC * DAC * (beta * TTC) / 12 * lane (TTC's weight 5 -> 5 + beta) | beta = 5 (TTC weight doubled) | beta = 0 |
| **S2 gate** | s * TTC^alpha (TTC becomes a multiplicative gate, like NC and DAC) | alpha = 1 | alpha = 0 |
| **S3 veto** | s * 1[TTC >= tau]; a set whose 64 proposals are all vetoed keeps s | tau = 0.5 | tau = 0 |

## 3. Metrics

- **TTC (decision)**: P2 TTC (validated proxy) x100, **B_new - A_new** (the g5 form: both arms deploy the same rule, as any gate or A40 control would). Also reported: B_new - A_orig (vs today's control), A_new - A_orig (what the rule does to the control), P1 TTC, P2 NC, P2 DAC (unvalidated).
- **Retention (decision)**: rho = (J(B_new) - J(A_new)) / (J(B_orig) - J(A_orig)), J = sum_i pi_i R_i with the ORIGINAL reward R (pooled 364 sets). Also reported: pick PDMS B_new - A_new and B_new - A_orig, and rho_dep = (J(B_new) - J(A_orig)) / (J(B_orig) - J(A_orig)).
- **X = 0.5**, fixed here.

## 4. Identity control and mutation (must pass before any candidate number is read)

- **(i) identity**: S1 beta = 0, S2 alpha = 0 and S3 tau = 0 reproduce the original rule BIT FOR BIT on every state, both arms: per-set J, pick index, pi_comp, pick_comp `np.array_equal`.
- **(ii) reach**: on R0 u250 arm B, each candidate at its fixed parameter changes the pick on >= 1 set.
- **(iii) mutation** `SELTTC_MUTATE=1` (the emphasis line returns the original score): (ii) must go **RED** (0 sets changed for every candidate).

## 5. Committed outcome rows

- **SEL-GO**: a candidate with B_new - A_new P2 TTC >= -0.5 at >= 2 of 3 R0 states AND rho >= 0.5 at >= 2 of 3 R0 states AND the C0 floor holds (|B_new - A_new P2 TTC| < 0.5 at every C0 state). If several clear: the highest min-over-R0 rho; ties go to the simplest (S1 < S2 < S3). Next lever: its A40-free deployment read on navtest stage 2 (dev-box harness) with the rule applied to the RL arm and its control; command named in RESULT, not run here.
- **SEL-PARTIAL**: the TTC bar holds at >= 2 of 3 with 0.2 <= rho < 0.5, or rho >= 0.5 with the TTC bar at only 1 of 3. Next: report the trade to the PI (a safe deployment rule at a stated cost in gain); the objective lever stays open.
- **SEL-NO**: no candidate meets the TTC bar at >= 2 of 3 with rho >= 0.2. Then the selection alone cannot repair the trade, and constrained EPO or the TTC-penalised reward (both need an A40 arm; PI decision) are the only levers left.
- A floor failure (C0) makes the read INVALID, whatever the R0 numbers.
