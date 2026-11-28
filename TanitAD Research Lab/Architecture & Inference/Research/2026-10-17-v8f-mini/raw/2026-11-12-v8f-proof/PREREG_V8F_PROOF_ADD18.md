# PREREG_V8F_PROOF_ADD18: MOTION, a level-1 predictor that outputs the state plus a change, and a rollout loss over 2.0 s (2026-11-14)

Written after the reads of rows H-V8F-PROOF-26, -28, -30 and BEFORE any arm of this card exists (the levers are being built on the CPU; no gate has run).
Written before the lead read of the arm P_s0 of ADD16 was seen by the author. It changes nothing in ADD12, ADD13, ADD16, ADD17.

## 1. Why (MEASURED, with the artifact)

* The grid state G holds the position of the lead (LOCATION at both seeds, row 26), and its rollout names `IMAGINE-FLAT` at both seeds on the lead read and in pixels (rows 26, 28).
* The rollout context is not the limit (`CTX-FLAT`, row 30).
* On held-out windows the level-1 predictor of G is WORSE than the copy of the state at the step it is trained on: error ratio predictor over copy one step (0.2 s) ahead
  1.233 [1.180, 1.295] / 1.254 [1.197, 1.317] (seed 0 / seed 1; episode-cluster bootstrap, `spatial/q2/ctx/STEP_CTX.md`), with its full context 1.216 / 1.238.
* The true future state holds a part of the change of the lead gap that no rollout delivers (change skill +0.134 [+0.016, +0.254] / +0.180 [+0.035, +0.327]; rollouts at or below 0), and a ridge on the start state, the actions and the rollout is below the copy in pixels at 2.0 s (row 28).

Reading (HYPOTHESIS): the predictor outputs the next state directly, so most of its one-step target is the state it was given; what it must learn, the change, is small against its error in reproducing the state, and it is only ever trained one step ahead.

## 2. The two levers (built as flags on the tree of ADD16; defaults bit for bit the earlier arms)

* RES, `--wm-l1-residual on`: the level-1 predictor returns the state it was given plus a learned change; its last layer starts at zero, so at the start of training it IS the copy.
* ROLL, `--wm-l1-roll-k 10 --w-wm-roll 1.0`: besides the one-step loss, the predictor is rolled out 10 steps (2.0 s) from one state, with the context growing by its own predictions as at read time, from the start frames 0, 2, 4, ... of the training sequence; the squared error against the true states (targets without gradient, the start state with gradient) is added with weight 1.0.
* `--wm-l1-diag on` (all arms of this card): the train log gains the one-step error, the error of the copy and the rollout errors. It must not change the training (checked, section 5).

## 3. Arms

Grid state with the flat predictor, the argv of G plus the flags; from scratch, 19,164 steps; seeds 0 and 1. Comparison arm: G_s0, G_s1 (banked, every read exists).
* RR = RES + ROLL; RES alone; ROLL alone: 3 arms x 2 seeds = 6 arms. Order: RR_s0, RES_s0, ROLL_s0, then the same at seed 1.
* The cell predictor of ADD16 is not in this card; the levers on it are the committed follow-up if ADD16 names `CELL-BEATS-FLAT` or `CELL-IMAGINES`.
* Budget (ESTIMATED from G at 66 min beside two other queues, plus the rollout term): about 75 to 90 min per arm, 6 arms on one of three slots of the pod (L40).

## 4. Reads and outcomes (each name needs BOTH seeds of the arm; a mixed pair reads the weaker name)

All reads are T0 (world-model diagnostics on logged frames, never driving performance) except the family read (T1-kin).
* S, one step: `STEP-BELOW-COPY` if the error ratio predictor over copy one step ahead (context 1, held-out windows, `ctx_step.py`) has its interval below 1; else `STEP-NOT-BELOW`.
* I, imagination on the lead read r-b (frame 0, the banked rule: one state, context 1; `ctx_roll.py` frame-0 file, `rb_trained.py` unchanged):
  `MOTION-IMAGINES` if the rollout beats the copy AND the ego-only floor (both intervals excluding 0);
  `MOTION-BEATS-BASE` if not, but the rollout minus the rollout of G of the same seed, paired on the same windows, is above 0 with the interval excluding 0; else `MOTION-FLAT`.
* V, the change of the lead gap (the change form of note ADD16_N1, here a NAMED outcome): `MOTION-MOVES` if the change skill of the rollout is above 0 with the interval excluding 0; else `MOTION-STILL`.
* R, retention: `LOSES-POSITION` / `KEEPS-POSITION` by the rule of ADD16 C2 (lead gap R2 on the train probe set, arm minus G, paired over clips, against the spread of the baseline seeds).
* F, families at T1-kin: the rule of ADD12 Q3 (`NO-COST`, `COST-<family>`, `GAIN-<family>`), arm against G and against the baseline.
* Reported, names as their own cards define them: the pixel read (when the dev-box memory allows it), the three rollouts of ADD17, the train-side one-step ratio from the train log.
* Attribution: an outcome of RR is assigned to RES, to ROLL, or to both, by which single-lever arm carries the same name.

## 5. Controls (all must pass; else nothing is named)

* Build: defaults bit-identical to the tree of ADD16 for the argv of G, of P and the mean default (200 steps, max abs difference 0.0); RES at init equals the copy exactly (0.0);
  the rollout term reaches the predictor and, through the start state, the state projection with a non-zero gradient, and a frame that is only a target gets exactly 0.0;
  the rollout inside the loss equals the rollout of the stack at read time; the total moves by exactly the weight times the term; a wrong expectation is refused by the gate.
* `--wm-l1-diag on` against off: same weights after the same steps (max abs difference 0.0).
* Pod gate per arm on real data (selfcheck-only): the levers read back from the BUILT model, the rollout term finite, a deliberately wrong expectation refused; no training without the PASS token.
* Every read keeps its own controls (x1 to x4, t1 to t6, k1 to k3, s1, s2, q1 to q5, the train probe controls).

## 6. Committed next steps

* `MOTION-IMAGINES` or `MOTION-MOVES`, with `KEEPS-POSITION` and `NO-COST`: the predictor form of that arm is the proposed world-model default of the next flagship arm; next lever, own card: planning by imagination on it (candidate plans scored by the imagined lead gap; the go / stay cue at rest read from the imagined state), and the same levers on the cell predictor if ADD16 names `CELL-BEATS-FLAT` or `CELL-IMAGINES`.
* `MOTION-BEATS-BASE` only: one arm with the rollout over 3.0 s (K 15) at weight 2.0, and the explicit object-motion target below; own card.
* `MOTION-FLAT` and `MOTION-STILL` with `STEP-BELOW-COPY`: the predictor fits the step and the latent target does not carry the motion of objects; next lever, own card: an explicit target of object motion (the frozen token grid of the future frame, cell by cell).
* `MOTION-FLAT` with `STEP-NOT-BELOW`: read the train-side ratio. Below 1 on the training data = a gap between training and held-out data: next lever is the regulariser and the data side (own card). Not below 1 = the predictor does not fit: next lever is its capacity and learning rate (own card).
* `LOSES-POSITION` or any `COST-<family>`: the arm is not promoted; it is reported as written, and the next lever of its outcome still runs.

## 7. Limits

One training run per arm per seed (two seeds); a separated interval is necessary and not sufficient, so every difference between arms is also read against the spread of the baseline seeds;
probes show what is linearly readable; 139 held-out episodes; T1-kin is a kinematic rollout on logged frames. STRATEGIC: PAUSED (PI item 25).
