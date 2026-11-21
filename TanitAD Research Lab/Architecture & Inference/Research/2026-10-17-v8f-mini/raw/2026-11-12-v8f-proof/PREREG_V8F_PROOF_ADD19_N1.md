# PREREG_V8F_PROOF_ADD19_N1: the levers of card ADD19 as BUILT, written before any arm of the card is gated or trained (2026-11-14)

Card: `PREREG_V8F_PROOF_ADD19.md` (sha256 1255463b6f9e2f27e238257500c748c1ecd93f403adba5c43acbdda668800b32). This note changes no outcome name, no read and no rule of the card.
It records where the build (`spatial/target/`, `TARGET_DEFS.md`, pins in `PINS.txt`) fixes something the card left open, so that every arm is read against the lever as it was built.
No arm of the card has a gate token or a number at the time of writing.

## 1. EGOV as built is "the ego speed as a third action channel of level 1", in BOTH directions

* The level-1 inverse-dynamics head of the stack is sized by the action width. With EGOV on (and the world-model variant b, weight 100, which every arm of the card runs) that head is also trained to read the speed channel from two consecutive states, and its mean runs over three channels instead of two.
* So the arm PE tests two things together: the predictor is GIVEN the speed, and the pair of states is ASKED for it. The card (section 2) names only the first.
* Committed control, own arm, only if PE reads `STEP-IMPROVES` or better at seed 0: PEi = the speed as predictor input with the inverse-dynamics target on the two channels of today. Until that arm is read, a gain of PE is assigned to "the speed as an action channel", not to the predictor input alone.
* Building level 1 with three action channels re-draws the random initial weights of the world-model auxiliary; the rest of the model starts as P of the same seed. PE and P of one seed are therefore not the same draw of the auxiliary. This is one more reason why the card asks for both seeds.
* The speed is the logged ego speed of each cached frame in m/s (cache field `v`, batch key `wm_speed`, present in every batch and read only with the lever on), divided by 10.0. The transition from frame j to j+1 carries the speed of frame j. It is not read by the inference path (not in `INFERENCE_INPUT_KEYS`).

## 2. FUTTOK as built

* The head is one linear map from the 16 numbers of a cell to the 1,024 numbers of a cached token, the same for every cell (17,408 parameters), built only with the lever on.
* The rolled-out states inside the term equal the rollout of the stack at read time within 5.1e-7 (bar 1e-6), not 0.0: the loss embeds the 15 actions at once, the read embeds the slice; on identical embedded inputs the rollout core gives 0.0.
* Reduction: the plain mean over features, cells, steps, starts and batch.
* FUTTOK needs the cached token grid in the batch; with a trainable encoder it refuses.

## 3. Expectation strings and cost

* The gate of the queue compares whole floats as ints: the arms files write `levers.w_wm_futtok=1`. Every arm line passes `pod/chain2/dry_gate.sh` (selfcheck on real data plus the expectation strings, no training) before it is queued.
* Cost, corrected by the measured rate of the PR arms (1.09 s per step on the pod with three jobs): PF and PEF about 5.8 h each, PE about 1.6 h (ESTIMATED from the tiny-model ratios: PE 0.9, PF 4.5, PEF 4.7 times P).

## 4. Checks of the build (CPU; `spatial/target/checks/`, `checks_lead/`)

* Agent: defaults bit-identical to the tree of ADD18 for the argv of G, P and PR over 200 steps (0.0); each lever reaches the loss with a control group at exactly 0.0; a test fails when the adding line is removed; the selfcheck reads the levers back and refuses a wrong expectation; the diagnostics flag leaves every weight equal (0.0); 87 tests pass (32 new).
* Lead, independent: with a zero weight and a random bias in the head the term equals the value computed directly from the token grid (difference 0.0) and the formula shifted by one frame does not (0.0045); tokens of a frame that is never a target leave the term unchanged (0.0) and tokens of the first and last target frame change it; the third action channel equals speed / 10 exactly and the transition j carries the speed of frame j. The first run of the lead's own control had its sensitivity bar set too tight (i.i.d. tokens); the log of that run is kept (`lead_target_check_firstrun.log`).
