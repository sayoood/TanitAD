# PREREG_SAFEHEAD addendum A2 — where does the REAL TTC loss live: arm B's safety rows or its progress rows?

Written 2026-10-16 ~20:05Z, after the validity row read **V-NO** (the sim's FULL arm shows no TTC loss at T K = 250: pooled P2 TTC +0.23 [-0.50, +1.04], P1 +1.65), and BEFORE any number below was computed. sha256 in `PREREG_SAFEHEAD_A2_REALSTATES.sha256`.
Why: under V-NO the sim cannot falsify a safety design on TTC (PREREG_SAFEHEAD.md s.8). The real arms DO lose TTC, and their banked scorer states read that loss on the same 364 sets (phead control (d): P2 TTC B - A -1.53 / -1.25 / -1.07 at R0 u250 / u500 / u1000). The cheapest experiment that can still discriminate is therefore a zero-training read of those REAL states through the safety-head selections.

- **States**: `rls1-{r0,r0b,c0}/scorer_states/scorer_states_u{u}.pt` (A and B), the 8 states of phead control (d). Pools: train 302 + val 62 = 364 sets. Evaluator arithmetic = `sh_common.eval_views` (identity (e) with phead's Evaluator passed).
- **Views**: FULL = B's own logits vs A's own (must reproduce phead `proxy_validity.json`; identity, |diff| <= 1e-5 x100). **SH_A-sel** = `where(SAFE, A, B)` vs A's own. **SH_F-sel** = `where(SAFE, base, B)` vs `where(SAFE, base, A)`. SAFE = NC, DAC, TTC.
- **Metrics**: P2 / P1 TTC, NC, DAC (x100, pooled, log-cluster bootstrap 10,000, seed 20260927); f_sel = (J(view) - J(ctrl)) / (J(B) - J(A)).
- **Rows** (decided on R0's three states; R0b reported; C0 is the floor and must read |P2 TTC| < 0.5 in every view, else the read is INVALID):
  - **A2-SAFETY-ROWS**: SH_A-sel P2 TTC >= -0.5 at >= 2 of the 3 R0 states. The real TTC loss lives in B's safety predictions; a separate safety head removes it at selection level. Next: the A40 SH_A arm (PI decision) with `SPEC_SAFETY_HEAD_FLAG.md`.
  - **A2-PROGRESS-ROWS**: SH_A-sel P2 TTC < -0.5 at >= 2 of the 3 R0 states. The loss lives in B's progress predictions (the pick is pushed toward faster proposals); a safety head cannot remove it at selection level. Next: the objective (A1's TTC-penalised reward, then Lagrangian EPO); no A40 safety-head arm recommended.
  - SH_F-sel read by the same rule, reported beside it.
- **Limit, stated in advance**: B's progress rows here were trained with the safety-row channel OPEN. A real safety-head run would train them differently (EPO could push the progress rows harder). This is a selection-level attribution on real states, not a simulation of a safety-head training run; it can refute the mechanism the safety head targets, it cannot certify the arm.
