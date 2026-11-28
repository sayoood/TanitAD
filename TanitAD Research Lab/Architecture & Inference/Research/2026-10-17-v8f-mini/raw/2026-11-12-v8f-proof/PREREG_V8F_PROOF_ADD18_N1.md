# PREREG_V8F_PROOF_ADD18_N1: one more arm, the rollout loss on the cell predictor, and the order of the arms (2026-11-14)

Written AFTER the reads of the arm P_s0 of ADD16 were seen (stated plainly: this note uses them) and BEFORE any arm of ADD18 exists (the levers are still being built; no gate has run).
It adds one arm and fixes the order. Outcomes, rules, controls and committed next steps of ADD18 are unchanged.

## 1. What was seen on P_s0 (MEASURED, T0, one seed, names nothing; all controls pass)

* One step ahead the cell predictor is BELOW the copy on held-out windows: error ratio 0.788 [0.770, 0.807] (context 1), 0.720 [0.705, 0.737] (context 3); the flat predictor of G reads 1.233 / 1.254 (`spatial/q2/ctx/STEP_CTX.md`).
* Lead read r-b at +2.0 s (441 windows, 139 episodes): rolled-out 0.501, copy 0.487, true future 0.532; rollout minus copy +0.013 [-0.037, +0.057] (G_s0 -0.054); rollout of P minus rollout of G, paired, +0.058 [-0.030, +0.163]. By the rule of ADD16 this seed reads `CELL-FLAT` (no interval excludes 0).
* Change of the gap: skill of the rollout +0.007 [-0.103, +0.122] against a ceiling of +0.190 [+0.062, +0.330] from the true future state. The history-seeded rollout adds nothing (+0.006 [-0.029, +0.043]).
* Position: eval-bank lead gap R2 0.529 (G_s0 0.552). Families at T1-kin, preliminary: no cell worse against G or the baseline.

Reading (HYPOTHESIS): the shape of the predictor removes the one-step defect that ADD18 addresses with RES on the flat predictor, yet its rollout still does not move the lead.
One step of 0.2 s holds very little of the motion of the lead; a loss on the 2.0 s rollout is the lever that puts that motion into the training signal.

## 2. The added arm

* PR = the argv of P (grid state, `--wm-l1-pred cell`) plus ROLL (`--wm-l1-roll-k 10 --w-wm-roll 1.0`) and `--wm-l1-diag on`; from scratch, 19,164 steps; seeds 0 and 1. No RES (the cell predictor is already below the copy one step ahead).
* Comparison arm: P of the same seed (banked reads). In the outcome I of ADD18, `MOTION-BEATS-BASE` for PR is read against the rollout of P, not of G. Everything else as ADD18 section 4.
* This is the lever ADD16 commits for `CELL-BEATS-FLAT` (a rollout loss over 2.0 s on the cell predictor), started before ADD16 is named with both seeds. Whatever ADD16 names, its own committed next step still runs.

## 3. Order (one slot of the pod; the two seeds of the most informative arm first, so that one outcome can be named with both seeds)

PR_s0, PR_s1, RR_s0, RES_s0, ROLL_s0, RR_s1, RES_s1, ROLL_s1. Budget (ESTIMATED): PR about 100 min per arm (P ran at 3.78 steps per second before the rollout term), the flat arms 75 to 90 min.
