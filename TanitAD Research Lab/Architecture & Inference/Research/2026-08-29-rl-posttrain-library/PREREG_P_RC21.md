# PREREG — P-RC21: RL post-training of the EXISTING REF-C v2.1 base planner

`PRE-REGISTRATION, 2026-08-29, TanitAD_TrainingFlyWheel. Committed BEFORE any
training step. PI directive, verbatim intent: "take the existing refc model if
it is not the same and try to apply the RL training until refcv3 is finished."
This is the PILOT the refcv3 A1 campaign was blocked without — a real IL cold
start exists for v2.1 (refc-diffusion-base-v21-30k), so the library trains on a
real model TODAY and refcv3 swaps in when its cold start lands.`

---

## 0. What this pilot is, and is not

| | |
|---|---|
| **IS** | the first REAL RL post-training run of `stack/tanitad/rl/` — DDv2-style intra-anchor GRPO + truncated inter-anchor advantage + veto, decoder-only, on a real trained planner and real scene context |
| **IS** | the mechanism-transfer test: does the truncated advantage prune colliding fan candidates HERE, as DDv2 measured on NAVSIM (their raw-fan floor 75.3 → 84.4)? |
| **IS NOT** | an A1 result. v2.1 ≠ refcv3 (no PhiTac hierarchy, no v7 vocabulary, no goal graft). ⛔ Numbers do NOT transfer to refcv3; the METHOD verdict does |
| **IS NOT** | a driving claim. Everything here is **T0 training-side** (EVAL_DOCTRINE). The four-families T1 eval belongs to TanitEval if the pilot is ever promoted |

## 1. Assets (all verified by content this session)

| asset | fact |
|---|---|
| cold start | `Sayood/tanitad-refc-base` ckpt.pt, 1,250,838,325 B; registry md5 (eval copy) `8f10d6f934f4199e11ddc7352e074939` — **verify on arrival, refuse on mismatch** |
| model | REF-C v2.1 base: 104,191,577 params (encoder 90.5 M · decoder 8.63 M · rest heads); window 8 · horizons (5,10,15,20) ⇒ fan `[B,128,4,2]` at **dt 0.5 s** · truncated diffusion steps 2 |
| train data | dev-box epcache `physicalai-train-14231cd29c74` — ⚠️ **NON-PARITY, pilot-only** — **54/400 episodes joined** to `obstacle.offline` via the 30 locally-cached chunks (91.4 % of frames carry agents, 4,457 in-lane-ahead vehicle-frames) |
| val data | `physicalai-val-bb543bdf7836` — **15/100 episodes joined** (⚠️ thin: 15 bootstrap clusters ⇒ wide CIs, claims only if separated) |
| identity | `episode_id` = first-4-chars-of-clip-uuid int (`physicalai.py:740`), inverted against chunk members; 0 ambiguous |
| reward gate | **A0 PASS on this corpus** (`raw/a0_pilot.json`): 270 windows, every weighted component ranks where it applies (collision 50.7 % / spread 1.0; headway 22.2 % / spread 0.96) |

## 2. The arms

Common: decoder-only (`trainable_prefixes=("decoder",)`, **`decoder.conf_head`
EXCLUDED** — the v2.1 selector surface lives inside the decoder; training it
would be the TRAIN-C1 inversion again), encoder/heads frozen, `decoder_steps=2`
(the deploy path), reward = `DEFAULT_WEIGHTS` + v0-referenced progress, dt 0.5,
G=4, Dr.GRPO centre-only, multiplicative noise 0.1, w_imitation=1.0 (outside the
advantage), seed 0, 4060 only.

| arm | what | steps |
|---|---|---|
| **P0** | cold start, NO training — the before-readout | 0 |
| **P1** | `grpo`, full composed reward | 2,000 |
| **P2-reg** | ⛔ deliberate regression: `HACKABLE_WEIGHTS` (progress-only) | 300 |

## 3. Readout (fixed val window set, seed 0, identical before/after)

| id | metric | computed how |
|---|---|---|
| R1 | mean composed reward of the FULL fan (128 anchors) | reward spec on val windows |
| R2 | **fan collision rate** — fraction of fan candidates whose path intersects an obstacle | `collision` component < 0 |
| R3 | ADE of the CONFIDENCE-ARGMAX trajectory vs GT @ (0.5–2.0 s) | selector output used for READOUT ONLY — never in the reward |
| R4 | per-component means | coverage honesty |

## 4. ⛔ Committed outcomes — written before the run

| outcome | reading | consequence |
|---|---|---|
| **P1: R2 falls** (fan collision rate down vs P0, CI-separated over 15 episode clusters) **and R3 does not degrade > +10 %** | ⭐ the DDv2 mechanism transfers: the truncated advantage prunes colliding candidates without wrecking the selected behaviour | the library is validated ON A REAL MODEL; A1-on-refcv3 proceeds on the cold start with the same recipe |
| P1: R2 falls but **R3 degrades > +10 %** | the reward re-shapes the fan at the cost of the deployed trajectory — imitation anchor too weak | raise `w_imitation`, re-run once; if it persists, the recipe needs the true diffusion-step density (M-B9 family), not more steps |
| P1: R1 rises but **R2 flat** | the policy chases comfort/progress, not safety pruning — composition imbalance | rebalance toward `collision`/veto; do NOT ship |
| P1: nothing moves | surrogate Gaussian-on-offset carries too little signal at noise 0.1 | one retry at noise 0.3; then STOP — the lever is the estimator, not more steps |
| **P2-reg does NOT degrade R2/R3** | ⛔ our readout cannot see the failure the audit predicts ⇒ the METRICS are broken, and P1's result is void | fix the readout before any further RL claim |
| P2-reg degrades as predicted | the instrument chain sees what it must see | record; this is the pass condition for the AUDIT, not for the model |

## 5. Stated limits (each declared, none silent)

1. **NON-PARITY corpus** — nothing here enters cross-arm comparisons or the leaderboard.
2. **Time-base hypothesis**: stacked frame i ↦ t=(i+2)·0.1 s (`rl_pilot_join.py --t0-offset 0.2`). Wrong offset displaces agents by ego-motion×error; tol 60 ms.
3. **Static-lead approximation** over the 2 s horizon; affects headway magnitude, not whether it ranks. No-lead windows carry a far-lead sentinel (constant per window ⇒ cancels in both advantages).
4. **Surrogate Gaussian policy on the emitted offset** — not the true reverse-diffusion density; parity with DDv2's estimator UNVERIFIED (standing limit).
5. **15 val episodes** — episode-cluster bootstrap, claims only when the CI separates.
6. **4 waypoints** per trajectory ⇒ jerk from one sample; comfort is coarse.
7. Non-vehicle classes are kept in `obstacles` (a pedestrian is an obstacle too); `lead` is geometric (in-lane ahead nearest), class-blind — DECLARED difference from `taniteval.lead_source`'s vehicle-class rule.

## 6. Cost & safety

Step cost measured at launch (first 20 steps) before committing to 2,000; hard
stop if the P1 projection exceeds ~2 h. 4060 only, `nvidia-smi` clear first;
⛔ Thor untouched (B1 epcache build owns it overnight). Checkpoint + done-marker
per house rules; every knob in `config.json` via `PostTrainConfig.to_dict()`.

---

# ⭐ AMENDMENT 1 — 2026-08-29 — the ANCHOR SWEEP replaces the `w_imitation` remedy

`Landed BEFORE the re-run, with no readout_after for any anchored arm in
existence. Same precedence discipline as the original prereg (06c715d8b).`

## Why this amendment exists

P-RC21's first campaign produced a FAIL whose diagnosis was *"no policy anchor"*.
§4's remedy branch said **"raise `w_imitation`, re-run once"**. ⛔ **That branch is
WITHDRAWN**, for a reason of substance rather than convenience:

> `w_imitation` is an **imitation loss against LABELS**. GRPO's stabiliser is a
> **TRUST REGION against the REFERENCE POLICY**. They are different objects, and
> the original prereg named the wrong one.

Master Mind ruling, 2026-08-29. The two are used interchangeably in secondary
summaries of the RLHF/GRPO family, which is exactly how the substitution passed
unnoticed — logged as vocabulary in `VOCABULARY.md` and as TRAIN-C5's family.

## What replaces it

**The reference-policy anchor** (`stack/tanitad/rl/anchor.py`): a divergence
penalty between the training decoder's mean fan and the mean fan of a **frozen
deepcopy of the cold start**, applied to the LOSS and **never inside the
group-relative advantage** (same placement rule as the veto — it constrains the
policy rather than ranking candidates).

⚠️ It constrains the **policy's MEAN**, not its samples. Penalising the drawn
samples would penalise the exploration noise, whose scale is `|offset|·σ`, i.e.
it would SHRINK offsets rather than HOLD POSITION — a different objective.

## The sweep (replaces the single re-run)

| arm | `w_anchor` | purpose |
|---|---|---|
| **S0** | 0.0 | the unanchored null — reproduces the original P1 configuration exactly, now with a CORRECT readout |
| **S1** | low | is a light trust region enough? |
| **S2** | high | does a strong trust region kill the reward gain along with the drift? |
| **S-reg** | 0.0, `HACKABLE_WEIGHTS` | the deliberate-regression control, unchanged |

⭐ This also yields the **anchor-strength curve DDv2 never published**, which
makes the re-run a contribution rather than a repair.

## ⛔ Committed outcomes — written before any anchored arm runs

| result | reading | consequence |
|---|---|---|
| **R2 falls (paired-CI separated) at some `w_anchor` with R3 held within its re-derived guard** | the DDv2 mechanism transfers once the trust region is present — the original FAIL was a missing constraint, not a dead method | promote; A1-on-refcv3 proceeds with the anchored recipe |
| R3 held at every `w_anchor` but **R2 never falls** | the anchor fixes drift and the reward still cannot prune collisions ⇒ the REWARD is the limit, not the stabiliser | do NOT scale; the next lever is reward composition |
| **R2 and R3 both frozen as `w_anchor` rises** | the trust region is binding so hard the policy cannot move at all | the useful window is below the tested low value — re-sweep downward once |
| **monotone trade-off with no window** (any R2 gain costs R3 beyond guard) | the surrogate estimator, not the anchor, is the limit | STOP the surrogate line; escalate M-B9 (true diffusion-step density) from deferred |
| S-reg fails to degrade | ⛔ the readout is void again and NOTHING in the sweep may be read | fix the readout first |

## Readout changes since the original prereg (all forced by TRAIN-C5)

1. ⛔ **`.eval()` is mandatory and recorded** (`"eval_mode": true`). The original
   readouts ran in TRAINING mode with `ego_dropout=0.5` — every magnitude in
   `RESULT_P_RC21.md` §1 is superseded, not extended.
2. **The R3 guard is re-derived from the CORRECT instrument.** In eval mode the
   readout is EXACTLY deterministic (run-to-run spread **0.000 m** over 3 seeds),
   so the original "+10 % against a 17 % noise floor" is replaced by a guard that
   can actually fail: **R3 must not exceed the S0 baseline by more than 10 %**,
   and that 10 % is now well above zero measurement noise rather than below it.
3. **Per-episode values are stored**, so the PAIRED episode-cluster bootstrap the
   house rule requires is computable — the original readout kept aggregates only
   and could not answer whether R2's small move was separated.
4. **`ckpt_after.pt` is saved**, because P1's outcome could not be re-measured.

## Unchanged from the original prereg
NON-PARITY corpus · T0 only, no driving claim · surrogate Gaussian-on-offset is
not the true diffusion density · static-lead approximation · 15 val episodes ·
decoder-only with `conf_head` excluded · 4060 only, Thor untouched.
