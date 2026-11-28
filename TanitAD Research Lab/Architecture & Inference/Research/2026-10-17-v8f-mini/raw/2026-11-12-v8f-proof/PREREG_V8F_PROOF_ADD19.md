# PREREG_V8F_PROOF_ADD19: TARGET, what the level-1 predictor is GIVEN and what it is ASKED FOR: the measured ego speed as an input (EGOV) and the frozen token grid of the future frame as an explicit target (FUTTOK) (2026-11-14)

Written 2026-11-14 about 02:35Z: after the reads of P_s0 (row H-V8F-PROOF-32, per seed `CELL-FLAT`) and BEFORE P_s1 is read and before any arm of card ADD18 (MOTION) has a number.
It changes nothing in the cards ADD12, ADD16, ADD17, ADD18 or their notes. Tier of every read: T0 (a world-model diagnostic on logged frames, never driving performance) for the state and rollout reads,
T1-kin (kinematic rollout on the own actions of the model, logged frames) for the family read. STRATEGIC: PAUSED (PI item 25).

## 1. Why (MEASURED, with the artifact; one HYPOTHESIS, marked)

* Cards ADD16 (section 6, `CELL-FLAT`) and ADD18 (section 6, `MOTION-FLAT` with `STEP-BELOW-COPY`) both commit the same next lever: an explicit target of object motion, the frozen token grid of the future frame, cell by cell. P_s0 reads `CELL-FLAT` per seed, so this lever has to be ready; it is FUTTOK below.
* The change of the lead gap over 2.0 s is not held by the token or state history in a linear read: eval bank, frozen pooled tokens of two frames plus ego against the ego block alone -0.198 [-0.404, +0.041] (`spatial/s0c/VERDICT_S0C.md`, 1,586 pairs, 50 episodes); train probe clips, trained grid state with history minus the ego block -0.050 [-0.094, -0.004], history minus no history +0.010 [-0.001, +0.021] (`spatial/s1/VERDICT_S1_G_s0.md`, 24,206 pairs, 484 clips).
* On the same train probe pairs the ego block ALONE reads the change with skill +0.291 [+0.243, +0.334] (same file), while the rollout of the trained predictor reads it at about 0 on the eval windows (rows 26 and 32: G -0.063 / -0.147, P_s0 +0.007).
* The level-1 predictor is given the acceleration and the yaw rate of each transition and NOT the ego speed (`v8f_mini.py`, `WMAux.scale_actions`), and the state of a frame is a function of that frame alone (`WMAux.z1`: adapter tokens of one frame, projected).
* One step ahead the flat predictor is worse than a copy of the state (ratio 1.233 / 1.254) and the cell predictor is below it (0.788) (`spatial/q2/ctx/STEP_CTX.md`).

HYPOTHESIS (not measured): how far the scene moves between two frames scales with the ego speed. Neither a single-frame state nor (acceleration, yaw rate) holds the speed, so the predictor cannot know how far to move the scene; it can only learn the average displacement. If so, the measured ego speed as a third action channel lowers the one-step error and lets the rollout deliver at least the part of the change that the ego kinematics hold.

Scope note for the PI: EGOV is a lever that no earlier card named. The ego speed is measured ego motion, an admissible input (PI ruling: model inputs are the measured ego motion and its history). It enters the world-model auxiliary only; nothing in the driving path changes.

## 2. The two levers (both off by default; with the defaults the tree is bit-identical to the tree of ADD18)

* **EGOV** `--wm-l1-act speed` (default `acc_yaw` = today): the level-1 action gets a third channel, the measured ego speed at the start frame of the transition divided by the fixed constant `WM_SPEED_SCALE` = 10.0 m/s. One spelling (`scale_actions`) for training, the read tools and the planner. Levels 2 and 3 are unchanged in form (they read the embedded level-1 actions as today).
* **FUTTOK** `--wm-l1-futtok-k K --w-wm-futtok W` (default 0 and 0.0 = off; on needs the grid state, K from 2 to hmax - 1, W above 0): for the same rollout starts as the rollout loss of ADD18 (every second frame, one true start state, the shared rollout function), a linear head per cell (cell width to token width, the same weights for every cell) reads the rolled-out state of every step 1 to K and is trained with the squared error to the frozen cached token of the same cell at the same future frame, the target normalised over its feature axis without parameters and never carrying a gradient. The term is added to the level-1 loss with weight W. The latent one-step term stays as it is.
* Diagnostics with `--wm-l1-diag on` (no gradient, weights equal to the run without it): the FUTTOK term, and the same head on the START state against the same future targets (the copy through the same head).

## 3. Arms (grid state, cell predictor of ADD16, 19,164 steps, full data, the argv of P plus the diagnostics flag plus the lever)

| arm | lever | compared with (same seed) |
|---|---|---|
| PE | EGOV | P |
| PF | FUTTOK, K 10, W 1.0 | P, and PR of ADD18 (same rollouts, latent target) |
| PEF | EGOV + FUTTOK | P, PE, PF |

Order: PE_s0, PF_s0, PE_s1, PF_s1; PEF_s0 and PEF_s1 only if PE or PF reads at least `TARGET-BEATS-BASE` or `STEP-IMPROVES` at seed 0. Every arm passes the gate of the queue (selfcheck on real data, levers read back from the built model) before it trains. Never more than three training jobs on the pod.

## 4. Reads and outcome names (an outcome needs both seeds; a difference must also exceed the spread of the baseline seeds where the card of the read says so)

Read tools: those of ADD17 and ADD18 (`read_m_arm.sh`: position, lead read at +2.0 s with its change form, rollout variants, one-step ratio, family read). For an EGOV arm the tools pass the logged speed of each frame as the third action channel; for every other arm their output must be unchanged (control e1).

* **S, the step.** One-step error of the predictor over the error of the copy on the held-out windows (`ctx_step.py`), arm minus the comparison arm, episode-cluster bootstrap: `STEP-IMPROVES` if the interval is below 0 at both seeds, else `STEP-SAME`.
* **I, the level at +2.0 s.** Lead gap R2 from the rolled-out state minus from the present state: `TARGET-IMAGINES` if the interval excludes 0 above at both seeds; else `TARGET-BEATS-BASE` if the rollout of the arm minus the rollout of the comparison arm (paired, same windows) excludes 0 above at both seeds; else `TARGET-FLAT`.
* **V, the change.** Skill over persistence of the rollout on the change of the gap (note ADD16_N1): `TARGET-MOVES` if the interval excludes 0 above at both seeds, else `TARGET-STILL`. Beside it, as a reference that names nothing: the ego block alone on the same windows.
* **R, position.** As C2 of ADD16 against the comparison arm: `KEEPS-POSITION` / `LOSES-POSITION`.
* **F, families at T1-kin.** As Q3 of ADD12: `NO-COST` or `COST-<family>`; LONGITUDINAL, LATERAL, TACTICAL per family, STRATEGIC PAUSED.
* **K, the token read (PF and PEF only).** On the pixel read of the arm, learned rollout minus copy at 2.0 s: reported with its interval beside I; it names nothing.

## 5. Controls (all must pass; else the read names nothing)

* Build: defaults bit-identical to the tree of ADD18 (state dict keys, initial tensors, a short training run); each lever reaches the loss (non-zero gradient on the new parameters and on the predictor; a control group that must read exactly 0.0); a test fails when the adding line is removed; the selfcheck reads the levers back from the BUILT model and refuses a wrong expectation; the diagnostics flag leaves every weight equal.
* e1: the read tools on a non-EGOV arm give the same numbers as before (max abs difference 0.0 on P_s0).
* e2: for an EGOV arm the third channel shuffled across windows at read time must raise the one-step error (the predictor uses the speed), else S is not named for that arm.
* The controls of the reads themselves (x1 to x4 of ADD17, k1 to k3 of ADD16_N1, t1 to t6 of the lead read, q1 to q5 of the family read).

## 6. Committed next steps

* `TARGET-IMAGINES` or `TARGET-MOVES`, with `KEEPS-POSITION` and `NO-COST`: that predictor form is the proposed world-model default of the next flagship arm; next lever, own card: planning by imagination on it (candidate plans scored by the imagined lead gap; the go / stay cue at rest read from the imagined state).
* EGOV `STEP-IMPROVES` with `TARGET-FLAT` and `TARGET-STILL`: the speed fixes the step and the motion of the lead is still missing; next lever, own card: an object-level slot in the world-model state (lead gap and closing speed estimated from the history of the lead head) with a kinematic transition as a fixed rule and a learned residual.
* FUTTOK `TARGET-FLAT` and `TARGET-STILL` at both seeds: together with stage 0c the pooled token grid is not a target that carries the motion of the lead; the future-token target is not pursued further at this grid, and the next lever is the same object-level slot.
* EGOV `STEP-SAME`: the hypothesis of section 1 is not supported; reported as written; the object-level slot is the next lever.
* `LOSES-POSITION` or any `COST-<family>`: the arm is not promoted; it is reported as written, and the next lever of its outcome still runs.

## 7. Cost and limits

* About 100 min per arm without rollouts in the loss (PE), about 3 h with them (PF, PEF; to be corrected by the rate of the first PR arm): four arms about 9 L40-h, six about 15 L40-h (ESTIMATED). Pod queue of this fork, behind the arms of ADD18 and the seed-1 arms of ADD13.
* One training run per arm per seed; the eval bank has 139 episodes, 50 with a lead pair, so a small effect on the change read cannot be separated on it; the read on the train probe clips is in-sample for the model and is reported as such.
* The levers enter the world-model auxiliary only; no default and nothing in the driving path changes.
