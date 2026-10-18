# PREREG_V8F_BETA — rung β: the world model as an eval-time scorer of the 117-candidate fan, ZERO training

**Committed:** 2026-10-18 (Europe/Berlin), BEFORE any M2 checkpoint exists (H-MINI-K-1 is untrained). It is governed by `PLAN_V8F_REFAV1_HJEPA_2026-10-17.md` §4.3 (the ladder, the goal-admissibility rule), §6 (rung β) and §13, and it extends `PREREG_V8F_MINI.md`. RULE ZERO applies: a missed bar is reported FAILED and the named next lever runs. **Changing any number in §2, §3 or `taniteval/v8f_beta_eval.py::GATE` after an M2 number exists is a goalpost move and is refused.**

## 0. What this is, and how it differs from the plan's rung β

| | plan §4.3 / §6 rung β | this gate |
|---|---|---|
| scorer | a learned selector plus a scalar gate initialised at 0 | **none learned.** The world model of the finished M2 checkpoint ranks the fan at inference; **zero tuned parameters** |
| goal ĝ²(+6 s) | a trained head on `{trunk features, nav token, turn distance}` | **derived, not trained** (§1.4): the WM's own level-2 latent at the end of the route that the strategic layer S predicted, timed by the frozen P10 prior |
| baseline | α_best | **M2's own output** (the incumbent, §1.5) |
| training | yes | **none.** No GPU on Thor, the A40 or box 2. CPU is enough for the unit tests; a checkpoint run is on the dev box |

This is the **cheap discriminating experiment** that decides whether the trained ĝ² head of the plan is worth building: the gate separates "z² cannot rank candidates" from "S's route is not a good enough goal" (§3). A PASS here is a lever; a FAIL names its successor, and the successor is a pre-registration of its own.

## 1. The instrument (code anchors, all in the repo)

1. **The fan** is the plan's 117 candidates: a 13 × 9 product grid of **constant** `(a_lon, a_lat)` controls, `a_lat ∈ {−3.0 … 3.0 step 0.75}` m/s² and `a_lon ∈ {−4.0} ∪ {k · 7/12 : k = −6 … 5}` m/s². Each is rolled for 60 × 0.1 s from the window's **measured** speed `v0` with `κ = clamp(a_lat / max(v0, 4)², ±0.12)` through `tanitad.models.kinematic.rollout_unicycle`. Built in `stack/tanitad/models/v8f_beta.py::fan_controls` / `fan_paths`. It is `AnchorDecoder.roll_bank`'s `alat` branch (`stack/tanitad/refs/refc.py:2246` ff.), and its grid is pinned at `stack/tests/test_anchor_flyability.py:11-22`. `test_v8f_beta.py::test_beta_fan_paths_equal_a_real_decoder_roll_bank` checks `fan_paths` against a real `AnchoredDiffusionDecoder.roll_bank`.
   *The shipped artifact `anchors_117_alat_declared.pt` lives on a pod.* Without `--fan-artifact` the gate uses the pinned grid and records that it was **not** checked against the file; with it, `load_fan_controls` reads it through the declared-units reader and records the maximum distance to the pinned grid. **A read with a non-zero distance is reported with that distance and is not a PASS of this PREREG until the PI rules on it.**
2. **The action convention is training's.** `WMAux` pairs the transition `j → j+1` with the action channels measured at cache frame `j` (`a` over 0.2 s, the yaw rate over one 0.1 s step), scaled by `WMAux.scale_actions` (`a / 2.0`, `ω / ch1_scale`). `v8f_beta.wm_action_sequence` derives the same two channels from any 6 s path (a candidate, the goal path, M2's plan) with **one** function; transition 0 is the measured `(a, ω)` at NOW for every path. `WMAux.forward` now calls `scale_actions`, so training and β cannot drift (`test_beta_wmaux_refactor_is_bit_identical_to_the_inline_training_formula`). No steering angle appears anywhere (`D-PI-YAWRATE-1`).
3. **The score** is `HJEPAStack.score_candidates(z1_t, blocks, goal, level=2, return_details=True)`: roll level 1 under the candidate's 30 action pairs, lift through E², and take the squared distance of the level-2 latent at `+6 s` to the goal. Fixed: `level 2`, `last_k 1`, context 1, subgoal weight `λ_sub = 0`.
4. **The goal** is `route_goal_xy(ck_now, p30)`, a function of **exactly two arguments**: S's route checkpoints (30 / 50 / 80 m, re-expressed in the NOW frame with the measured odometry) and the frozen P10 prior's 6 s path (measured ego motion only). The route is the polyline origin → ck30 → ck50 → ck80 → a far point along ck80's heading; the pace is P10's arc length at each 0.1 s step. `goal_latents` then rolls the WM along that path and takes the same level-2 latent a candidate is scored against.
5. **The incumbent.** M2's own 6 s plan is a **118th candidate**, column 0. It is replaced by the best fan member only when that member's cost is **strictly lower**. Consequences, each pinned by a test: (i) a constant cost reproduces M2's output **bit for bit**; (ii) windows where β does not replace the plan are M2's output exactly; (iii) β can only differ from M2 where the WM says a fan member is closer to the goal.
6. **The entry point** is `stack/scripts/v8f_mini_eval.py run --plan beta` (default `--plan none`; the default runs the scorer **zero** times, test-pinned). `compare-beta` reads two runs and applies §3.

## 2. Arms, controls and what each is for

Result arm: **`beta`**. Everything else is a control or a T0 diagnostic and is never a result.

| arm | what | role |
|---|---|---|
| `m2` | M2's own plan (the incumbent), same windows | the comparator for P1 and P3 |
| `beta` | the incumbent + the fan, ranked by the WM against the S-route goal | **result** |
| `beta_shuf` | the same, with the goal taken from a window of **another episode** (`donor_indices`, seed 0) | P2 control — a gain that survives a wrong goal is not the goal |
| `beta_oracle` | the goal from the **ground-truth** 6 s path (T0) | tells "z² cannot rank" from "S's route is a poor goal"; reads the GT, never a result |
| `fan_random` | a uniformly random fan member (seed 0) | chance for the fan (T0) |
| `fan_oracle` | the fan member closest to the GT in ADE (T0) | the ceiling the vocabulary allows |

`signal.recovery = (ADE6(fan_random) − ADE6(beta_oracle, fan only)) / (ADE6(fan_random) − ADE6(fan_oracle))`: 1 = the oracle goal finds the fan's best member; 0 = no better than a random pick; < 0 = worse than random. "Fan only" ranks the 117 candidates **without** the incumbent so M2's plan cannot mask a dead ranking.

## 3. The decision — both outcomes committed in advance

Let `Δ = β − M2`, so a negative `Δ` of an error metric is β better. `F = |ADE6(M2_s0) − ADE6(M2_s1)|` on eval-141, the seed floor of the two M2 **training** seeds. Every interval is the **paired episode-cluster bootstrap** (`taniteval.ci`, B = 2,000, the same resampled episodes for both arms): it answers *another draw of EPISODES* only. Training-seed variance enters **only** through `F`; the two M2 checkpoints are the two seeds.

* **P1 (the headline):** `Δ ADE@6 s < 0`, separated, at **both** training seeds, and `|Δ| > 2F`.
* **P2 (attribution):** `ADE6(β_shuf) − ADE6(β) > 0`, separated, at both seeds — the shuffled-goal control loses.
* **P3 (non-regression):** at both seeds, **none** of the 13 components in §4 is separated-worse than M2.
* **Signal (T0 gate on the ranking space):** at both seeds `beta_oracle`'s fan-only ADE6 beats `fan_random`'s, separated, **and** `signal.recovery ≥ 0.5`.
* **Inert:** `replaced_frac < 0.05` at **both** seeds.

The verdict is the **first** row that holds, in this order:

| order | verdict | condition | the named next lever (runs in the same package) |
|---|---|---|---|
| 1 | **FAIL-INERT** | inert | β almost never replaces M2: score in z¹ only and re-run (plan §6; E2 says z¹-only should be worse, a known-direction check), then a learned scorer over WM-imagined features (paper §3.4). Nothing is trained by this gate |
| 2 | **FAIL-RANKING-DEAD** | the Signal gate fails at either seed | **z² is not a usable ranking space even with a perfect goal.** The world model's defect, not the goal's: apply the plan §5/§6 remedies to M2 (IDM at level 2 only, γ × 2, the time-repulsion floor) and re-measure `H-HJ-COST-1` **before** any goal head is built |
| 3 | **FAIL-GOAL** | P1 fails at either seed | z² ranks (the oracle works) but S's route is not a good enough goal: build the plan's trained ĝ² head (vision + nav + turn distance only, target = the level-2 latent `+6 s` ahead) under its own PREREG and launch gate |
| 4 | **FAIL-CONTROL** | P2 fails at either seed | β's gain is not attributable to the goal: treat it as a vocabulary / quantisation effect, not a world-model result; **do not promote** |
| 5 | **FAIL-NR** | P3 fails | β improves ADE6 but a family is separated-worse than M2: report the trade-off **per family**; do not promote on ADE |
| 6 | **PASS** | none of the above | β clears its bar on v8F-mini. Rung γ may be pre-registered; it still needs the AR-1 … AR-4 readiness gate before any A40 request |
| — | UNDECIDABLE | fewer than 2 seeds, or a required arm / block is missing | the gate cannot be read; no verdict is issued |

* `GATE = {"floor_mult": 2.0, "inert_replaced_frac": 0.05, "signal_recovery": 0.5, "n_seeds": 2}` in `taniteval/v8f_beta_eval.py` is this table; `test_beta_decision_table_reaches_every_verdict_in_the_committed_order` pins every row.
* **P3's multiplicity is stated, not hidden.** 13 components × 2 seeds is up to 26 one-sided looks, so under a pure-noise null a lone "separated-worse" flag is not rare. P3 is conservative by design; a FAIL-NR is read per family and is **not** a refutation of an ADE gain that cleared P1 and P2. It is also reached only after P1 and P2 hold.
* **The arm.** The gate is read **per M2 arm** (M2a and M2b), each at two seeds. A PASS on one arm is a PASS of that arm only; there is no OR across arms and no pooling.
* **Windows.** The grid of `PREREG_V8F_MINI.md` (eval-141, 2,399 windows, T1-equivalent open loop, the arm's own held outputs, the `--ages` of the K-1 read). The `run` command asserts that the incumbent β carries is **bit-identical** to the plan M2 emitted on the same windows; a mismatch aborts the run.

## 4. The metric families — ADE alone is not a result (PI binding rule, 2026-08-02)

ADE@6 s, ADE@2 s and FDE@6 s are reported **in addition** to the four families below, each per family, never pooled into one score, each as `β − M2` with the paired CI above. The families are `taniteval.four_families` blocks for the `beta` arm plus the per-window components behind a paired delta (`v8f_beta_eval.family_windows`, whose pooled values are pinned to the family blocks' printed numbers by test).

| family | components in the gate (P3) | what it reads |
|---|---|---|
| **LONGITUDINAL** | `lon_speed_mae`, `lon_accel_mae`, `lon_along_mae` | target-speed accuracy, acceleration and along-track error of the emitted plan against the human's 6 s |
| **LATERAL** | `lat_cross_mae`, `lat_heading_mae_deg`, `lat_yaw_rate_mae_degps`, `lat_curvature_mae_1pm` | cross-track, heading, **yaw-rate and curvature** error (a smooth-but-wrong path shows up in the last two, not in cross-track). Heading / yaw-rate / curvature are masked where a path does not move (no tangent): such a window reads `NaN` and is dropped from the pair, never scored 0 |
| **TACTICAL** | `tac_lat_acc`, `tac_lon_acc` (higher is better) | the **executed** lateral / longitudinal manoeuvre decision of the plan against the human's, from the plan's kinematics (`refc_tactical.factor_from_kinematics`). T's own goal / lateral / longitudinal heads are untouched by β and are carried with their own metrics |
| **STRATEGIC (plan)** | `strat_route_err_30m`, `_50m`, `_80m` | the distance from the plan's point at 30 / 50 / 80 m of arc to the GT route checkpoint (v9 RC-A), `NaN` where the plan is shorter than the arc or the checkpoint is unlabelled |
| CONSISTENCY | `consistency_viol_any` | the frozen violation rules of `PREREG_V8F_MINI.md`, on the plan |

* **STRATEGIC is reported only where v8F defines it.** S's own checkpoint error is carried per arm, but **β does not move it by construction**, so it cannot be a β result. The plan-adherence rows above are where β can act. A strategic **decision / route-choice** metric is **PAUSED**: v8F-mini defines the strategic layer as route checkpoints, not as a decision with a label. This is stated per family, not silently dropped.
* **Per-family availability** is reported with its `n`: STRATEGIC (the route labels), CONSISTENCY and T's own head metrics cover the 139 of 141 episodes that carry v9 labels; ADE, LONGITUDINAL, LATERAL and the executed-decision TACTICAL components (from the plan's kinematics against the human's) need no label and cover 141. A component with fewer than two finite paired windows or two episodes reads `UNAVAILABLE` with the reason, and is **not** counted as passing.
* **A missing instrument is a work item.** No component above lacks an instrument.

## 5. Selection discipline and the holdout — read this before reading any result

* β has **no tuned parameter** (§1): nothing is selected on any split, and eval-141 is read **once** for this configuration. Because the rule is deterministic, a re-run on eval-141 consumes no selection.
* Any variant β′ (a different level, `last_k`, subgoal weight, goal construction, fan) is a **new pre-registration**. It may be **selected on the 60-clip holdout only** — the last 60 of the first 600 train clips in sha12 order, the rule of `PREREG_HJ_STAGE0.md` Amendment 1 — and eval-141 then scores the chosen β′ **once**.
* ⛔ **The holdout is IN-SAMPLE for M2.** `v8f_mini_train.py` has no holdout option: it trains on `select_train_clips(clips_all, data_frac)` (`:290`). *MEASURED by source read; whether the S4 cache build itself excludes the 60 clips is UNVERIFIED.* A β′ selected there is therefore optimistic for every quantity the WM was trained to fit, and a selection made there has **not** been shown to transfer. Two honest options, both the PI's: (a) retrain M2 with those 60 clips withheld (`H-MINI-K-1`'s note already requires "the same 60-clip holdout rule" for the full-corpus run) or (b) accept the optimism and say so on the number. This gate does not need either, because it selects nothing.
* P10 in the K-1 comparison is the P10-600 prior (`D-RA1-P10-540`: v8F-mini keeps it; it is out of sample on eval-141).

## 6. The known risk, stated before the data

**RA-1 measured level-2 − level-1 goal-cost monotonicity NEGATIVE, and separated, on all six WM runs** (A0 −0.1425 / −0.1316, A1 −0.1909 / −0.1881, A2 −0.1826 / −0.1978; `…/2026-10-17-refav1-hj-mini/RESULT_RA1_YAW.md`). The plan's mechanism for β assumes that z² behaves like a subgoal space — that a level-2 distance to a future latent shrinks as a rollout nears it. It did not, on those WMs. Two caveats keep this honest, and they cut in opposite directions:
* **INHERITED, not re-verified for M2.** RA-1's WMs are the refav1-HJ mini arms, not v8F-mini's M2 auxiliary (a different trunk head, different action channels). The row is `OPEN: NOT READ` on the compliant run (`H-HJ-COST-1`, the A1x gate failed), so the number is a record, not a verdict.
* **β does not use the monotonicity statistic.** It scores the **endpoint** only (`last_k 1`, `λ_sub = 0`). What it needs is weaker and different: that nearer-in-trajectory candidates land nearer-in-z² at +6 s. The `beta_oracle` arm measures exactly that, so the risk is testable at once.

**What the risk predicts for β (HYPOTHESIS, committed now):**

| if z² is not a subgoal space for M2 | prediction |
|---|---|
| the ranking | `signal.recovery < 0.5`, or `beta_oracle` not better than `fan_random` ⇒ **FAIL-RANKING-DEAD** |
| the replacement rate | high (a noisy cost makes a fan member beat the incumbent on most windows), so `replaced_frac` ≫ 0.05 |
| the accuracy | β is **worse** than M2 where it replaces: the incumbent is a trained plan, a noise-ranked fan member is closer to `fan_random` |
| the distance spread | the fan's within-window endpoint spread over the between-window spread (`wm_diag.fan_endpoint_spread_ratio`) is near 0 |

| if z² ranks the fan | prediction |
|---|---|
| the ranking | `beta_oracle` recovers ≥ 0.5 of the random-to-ceiling range |
| the goal | the S-route goal is paced by P10, so it pulls the chosen member toward **P10's pace**; M2 is already a residual on P10, so the gain, if any, is **lateral**, and a longitudinal regression is the risk P3 watches (INHERITED, another model: 88.7 % of the refcv oracle gap is longitudinal, `CLAUDE.md`) ⇒ **FAIL-GOAL** is the most likely failure |

**Estimate (ESTIMATED, not measured):** PASS is the less likely outcome. The sharpest reading either way is `signal.recovery`, which is available before the headline.

## 7. Admissibility (PI binding rules 2026-08-03)

* **The goal path and the situation path are information-disjoint at inference.** `route_goal_xy(ck_now, p30)` takes S's checkpoints and the frozen P10 prior. It never reads `goal_prob`, `lat_logits`, `lon_logits`, `cons_pred`, an argmax, an embedding or a v9 label; T's heads are not on its path. *Shared trunk, stated:* S, T and O share the frozen-feature adapter and the goal path reads the same adapted scene through S, as the plan §4.3 allows ("trunk features"); the rule bans carrying the classifier's **output**, not the trunk. Tests: `test_beta_goal_is_a_function_of_ck_and_p10_only` (the signature and the identifiers in the function), `test_beta_perturbing_the_tactical_layer_moves_the_plan_but_not_the_goal` (perturb T's heads: M2's plan moves, the goal is **bit-identical**).
* **Inference is vision plus measured ego, no privileged channel and no steering angle.** `collect_beta_inputs` reads the batch through the key guard and asserts the keys are a subset of `M.INFERENCE_INPUT_KEYS`; the second action channel is the measured yaw rate. The ground truth enters only the scoring and the T0 arms (`beta_oracle`, `fan_oracle`), never the result arm.
* **A supplied route is optimistic by construction on PhysicalAI** (our only route supplier there is the ego's own future path). β uses S's **predicted** route, which sidesteps this; `beta_oracle` deliberately uses the GT and is labelled T0.

## 8. Evidence that the gate measures what it claims (all CPU, `CUDA_VISIBLE_DEVICES=-1`)

`stack/tests/test_v8f_beta.py`:
* **Known answer:** on a world model whose level-2 endpoint is an analytic function of the summed actions, β picks exactly the candidate the closed-form cost says, and `fan_idx`, `cost_min`, `cost_inc` equal it.
* **Identity control:** a constant cost reproduces M2's output bit for bit — at the planner and end to end through `run --plan beta`.
* **Deliberate-regression arms (source mutants, each must go RED):** delete the scoring line; pick the worst candidate; drop the goal; share one action sequence across all candidates; make the improvement non-strict.
* **Reachability from the actual caller:** `--plan beta` makes `HJEPAStack.score_candidates` run on `[B, 118, 30, 2]` blocks at level 2; the default runs it zero times; and a mutant of the caller that ignores the flag leaves the probe at zero calls. The output records how often the scorer ran (`score_calls`).
* **Admissibility, instrument and decision table** as listed in §3 and §7.

## 9. What would invalidate a read

* `IDENTITY BROKEN` (the incumbent is not M2's emitted plan) — the run aborts.
* `score_calls.candidates == 0` in a `--plan beta` output — the scorer never ran; the read is void.
* `fan.max_abs_diff_vs_pinned_grid > 0` — the fan is not the pinned grid (§1.1).
* An M2 checkpoint with `wm='off'` (M1) or `kin_mode != yaw` — refused by the run (no WM to score with / steering angle at inference).
* Two `--runs` naming the same checkpoint — refused: the seed floor needs two **training** seeds.

## 10. Evidence classes

| claim | class |
|---|---|
| the fan geometry, the action convention, the instrument's behaviour on the stub WMs | MEASURED (`test_v8f_beta.py`, CPU) |
| RA-1's negative L2 − L1 monotonicity | INHERITED (`RESULT_RA1_YAW.md`; not a verdict, not M2's WM) |
| the holdout is not withheld by the trainer | MEASURED by source read; cache exclusion UNVERIFIED |
| the P(PASS) statement | ESTIMATED |
| β's accuracy relative to M2 | **no number exists** — the gate has not been run (no trained M2 checkpoint) |

## 11. How to run it (when an M2 checkpoint exists)

See `TanitAD Research Lab/Architecture & Inference/Research/2026-10-18-v8f-beta/HANDOFF.md` for the exact commands. A run needs, per M2 arm, **two trained M2 checkpoints (training seeds 0 and 1)**; the scorer rolls 118 candidates through the WM for each of 2,399 windows (full-scale runtime NOT measured), so run it on the dev box or on box 2 **after** that box's GPU job; it adds no load to a training pod.
