# DIFF — our RL code vs DiffusionDriveV2's RL stage (the SPEC), component by component

**Package:** `…/2026-09-15-ddv2-rl-prep/` · **Date:** 2026-09-15 · **Reference:** `SPEC_DDV2_RL_PAPER.md` (§ and A-numbers below are its) · **Code:** worktree at `d4e8bc0` plus the files staged with this package.

**Two "ours" columns, because they are different objects:**
* **LIB** — the programme's RL stack as it stood before today: `stack/tanitad/rl/{advantage,posttrain,refcv3_adapter,rewards,config,control_space}.py` and the pilot `stack/scripts/rl_pilot_refc21.py`. It ran only on the July `refc-base-30k` and on refcv3, never on refcv4b, refcv5 or refcv5-v2.
* **PORT** — prepared today for refcv5-v2: `stack/tanitad/rl/ddv2_rl.py` (released arithmetic), `stack/tanitad/rl/ddv2_refc_chain.py` (binding), `stack/tanitad/rl/pdm_proxy.py` (reward) and `stack/scripts/ddv2_rl_refcv5.py` (driver). **LIB is unchanged**: no code path another arm uses was edited.

**Classes:**
* `FAITHFUL` — the same arithmetic, pinned by a test.
* `FAITHFUL-ARITH / DIFFERENT-EFFECT` — the same operation, but on a state that means something else here, so the effect is measured separately.
* `DIFFERENT` — a declared deviation, with the reason.
* `MISSING` — not in our code.
* `NOT-IMPLEMENTABLE-ON-OUR-DATA` — the input the release needs does not exist in our corpus.

**Evidence tags:** `[T]` pinned by a test in `stack/tests/test_{ddv2_rl,ddv2_refc_chain,pdm_proxy}.py`; `[D]` measured in `raw/ddv2_diagnose.json` (T0, 12 RL-train windows of the real checkpoint); `[S13]` read from the fork at the pinned commit (PUBLISHED-CODE-UNBANKED, SPEC §13).

## 1. Policy and chain

| # | component | DDv2 (SPEC) | LIB (before today) | PORT (today) | class (PORT) | why / measured consequence |
|---|---|---|---|---|---|---|
| 1 | object optimised | the truncated-diffusion trajectory head, cold-started from DD's IL weights; 1 decoder layer (A1) | a Gaussian **surrogate** over the emitted offset of one deterministic forward (`refcv3_adapter.py:24-30`, `:776-781`) | refcv5-v2's WP-4 sampler pass `_decode_ctrl` (`refc.py:1991-2023`) through `ddv2_refc_chain.make_x0_fn` (`ddv2_refc_chain.py:142-160`); 4 layers, d = 384 | FAITHFUL (structure) / DIFFERENT (model) | the model is ours. Binding parity: bitwise to `_sample` on a toy decoder [T] and on **12/12 real windows** [D, D1] |
| 2 | diffusion state | normalised waypoints x/50, y/20, 8 poses over 4 s | offsets in metres (or a 2 s control prefix in `rl_refcv3_min.py:570-585`) | normalised **control sequence** (a_lon/4, a_lat/3), 8 slots over 6 s, integrated by the programme's unicycle (`refc.py:2025-2040`, `refc_sampler.py:349-390`) | DIFFERENT (by design) | every sample is kinematically flyable; the price is rows 5, 8, 9 |
| 3 | truncated start | t = 8 anchored Gaussian on G tiled copies (`rl.py:813-820`) | none (no chain) | `ddv2_rl.truncated_start` (`ddv2_rl.py:385-401`), group-major tiling (`:372-382`) | FAITHFUL | bitwise vs diffusers `add_noise`; tiling order pinned by a regression [T] |
| 4 | rollout chain | 10 labels [18 … 0], one-unit t → t−1, ᾱ(−1) = 1 (§3.2) | none | `ddv2_rl.rollout_chain` (`:404-441`) | FAITHFUL (arithmetic) / DIFFERENT (support) | bitwise vs the released loop through the vendored scheduler [T]. ⛔ **refcv5-v2's sampler was trained only at labels {10, 0}** (`sampler_train_t_max` is read by nothing, `refc_sampler.py:397-404`), so labels 18 … 2 are off-support. With both clamps OFF the chain's fan endpoints sit **6.1 m** (mean, p90 13.9 m) from the deployed fan's [D, D3] |
| 5 | exploration | η = 1 ⇒ two multiplicative scalars per trajectory, floor 0.04 (§2, A3) | `two_scalar` mode on the offset, σ = 0.1·\|offset\|, drawn **once** on the final output (`refcv3_adapter.py:107-112`) | per step, inside `ddv2_rl.ddim_logprob_step` (`:154-229`) | FAITHFUL-ARITH / DIFFERENT-EFFECT | bitwise [T]. Multiplicative noise scales each axis, so a near-zero coordinate barely explores. In DDv2 that is lateral y ≈ 0 on straight driving; here it is **a_lon ≈ 0 and a_lat ≈ 0**, i.e. cruising straight gets almost no exploration on either axis |
| 6 | additive DDPM term | drawn, multiplied by 0 (A3) | — | the same draws, × 0 | FAITHFUL | keeps the RNG stream bit-equal [T] |
| 7 | likelihood | isotropic, σ = max(σ_t, 0.1), summed over 16 terms, at the detached sample (A4) | σ = \|offset\|·0.1, same scale as sampling (`refcv3_adapter.py:133-135`) | `ddim_logprob_step` | FAITHFUL | bitwise; analytic literal and gradient [T] |
| 8 | x̂0 clamp inside the step | diffusers `clip_sample=True` default, box ±1 = ±50 m / ±20 m (**A18**, found today) | — | `clip_sample` in `ddim_logprob_step` (`ddv2_rl.py:201-204`) | FAITHFUL-ARITH / DIFFERENT-EFFECT | here the box is **±4 m/s² a_lon, ±3 m/s² a_lat**. The deployed model's own x̂0 lies outside it on **6.6 % / 18.6 %** of coordinates; the release chain's final x̂0 on **9.2 % / 25.7 %** [D, D2] |
| 9 | decoder-input clamp | `clamp(x_t, −1, 1)` and offset added to the clamped state (`rl.py:826`, `:477`) | — | `input_clamp=True` in `make_x0_fn` | FAITHFUL-ARITH / DIFFERENT-EFFECT | binds on 1.3 % / 9.5 % of chain inputs. **Rows 8 + 9 together move the chain's fan endpoints from 6.1 m to 11.8 m (mean; p90 29.3 m) off the deployed fan** [D, D3]. Best-of-117 ADE and fan proxy reward barely change (0.88 / 0.86 / 0.89 m; 0.677 / 0.677 / 0.689) |
| 10 | dropout in the head (A2) | p = 0.1 live in rollout and grad pass | — | our decoder has **no dropout** (`refc.py:1316-1372`) | DIFFERENT (source absent) | the grad pass re-evaluates the exact rollout mean; A2's mask mismatch cannot occur |
| 11 | groups | G = 4, folded into the batch for the head (`rl.py:471-476`) | G = 4 over offset samples | G · N queries in one `_decode_ctrl` call | FAITHFUL | exact because no layer mixes queries: other queries changing leaves a query's output **bit-identical**; a mixing-layer mutant is caught [T]. `refc.py:2281-2291`'s `sampler_groups > 1` refusal is not crossed, because `forward` is never called with groups |
| 12 | anchors | 20 K-means, fixed | 128 (refc-base-30k) | 117 v0-conditioned control anchors (registry) | DIFFERENT (model) | 468 chains per window instead of 80 |
| 13 | heading of a candidate | Bézier derivative of the sampled xy (`rl.py:865`) | waypoint differences | the unicycle's own yaw, from the controls | DIFFERENT (better defined) | no dependence on waypoint spacing |

## 2. Reward

| # | component | DDv2 (SPEC) | LIB | PORT | class (PORT) | why / data |
|---|---|---|---|---|---|---|
| 14 | simulator | PDM: LQR-tracked kinematic bicycle, 40 × 0.1 s, replayed agents (§5.2) | waypoints differenced at dt 0.5 (`rewards.py:113-116`) | `pdm_proxy.ego_states_from_controls` (`pdm_proxy.py:116-144`): the candidate's own integration, 40 ticks + t0 | DIFFERENT | no tracker is needed for an integrated control sequence; horizon and tick match. Both are T1-class self-action OPEN loop |
| 15 | NC | at-fault polygon collision, {0 agent, 0.5 static, 1} [S13] | −1 if within 2 m of a t0 obstacle **snapshot** (`rewards.py:357-446`, `rl_pilot_refc21.py:120-141`) | `no_at_fault_collision` (`:258-285`): same values; first contact per track decides; at-fault = ego moving ∧ (track stopped ∨ front-edge contact); rear never; t0 overlaps ignored | DIFFERENT (lateral clause) | NAVSIM's lateral-at-fault clause needs lanes / drivable area. Data: obstacle.offline join, **139/141** eval and **4,427/4,572** train clips. Known value: human NC = 1 on **12/12** windows [D, D5] |
| 16 | DAC | footprint outside the drivable map at any tick [S13] | none (`control_space.py:24-26`: "a map-based drivable-area term we do not have") | `dac_from_drivable` (`:383-399`) on a SAM3 map GT grid; **≡ 1 in training** | **NOT-IMPLEMENTABLE-ON-OUR-DATA** (training) | SAM3 map GT (`stack/tanitad/data/semantic_map_gt.py`, landed at `4960c66`, not in this worktree) exists on the dev box for **3 clips, all in the held-out split**. The corpus-wide output is on Thor/HF, outside this task's rules. Mechanism tested; corner-only rule ⇒ an upper bound on compliance [T] |
| 17 | EP | centreline progress, normalised pairwise against **PDM-Closed** [S13 §13.1] | along-track ÷ max(v0·H, 5 m) (`rewards.py:191-259`) | `ego_progress` along the **human's own future path**, normalised pairwise against **the human** (`:374-381`, `:408-440`) | **NOT-IMPLEMENTABLE-ON-OUR-DATA** (faithfully) → DIFFERENT | no lane centreline and no PDM-Closed planner in PhysicalAI-AV. Consequence: the human's EP is always 1, so the ≥GT bar is harder to clear on EP than in DDv2, where PDM-Closed can out-progress the human |
| 18 | TTC | footprint at +0/0.3/0.6/0.9 s vs tracks ahead, or in intersection and not behind [S13] | a veto on gap-closing time < 1.5 s against the lead only (`rewards.py:545-570`) | `ttc_within_bound` (`:288-316`): same offsets and stopped threshold; ahead = ±30° cone | DIFFERENT (map clause) | no intersection map. Human TTC = 1 on 11/12 windows [D5] |
| 19 | comfort | six Savitzky-Golay checks, window = n_time, NAVSIM thresholds [S13] | exp(−(jerk/8 + lat_acc/4)) (`rewards.py:680-689`) | `comfort` (`:336-356`): same thresholds; one full-window polynomial (equals scipy's savgol) | FAITHFUL-UNBANKED | the derivative helper's transcription was partial (SPEC §13.5). A regression proves raw finite differences would fail every slot boundary [T] |
| 20 | aggregation | NC × DAC × (5 EP + 5 TTC + 2 C)/12 (P Eq. 14; exactly two multiplicative rows, A9 resolved) | Σ w_i · term_i (`rewards.py:777-783`) | `pdms` (`:402-405`) | FAITHFUL | literal [T] |
| 21 | GT scored in the same call | yes (`rl.py:867-873`) | separately, per window (`posttrain.py:277-285`) | human = proposal 0 in `score_candidates` | FAITHFUL | identity control: human-as-candidate scores exactly the human, **12/12** [D5] |
| 22 | ego footprint | nuPlan Pacifica 5.176 × 2.297 m, rear axle 1.461 m | point / 1 m radius | the same numbers | UNVERIFIED for the PhysicalAI vehicle | the PREREG's I4 known value (human NC = 1 on ≥ 98 % of training windows) is the check that would expose a wrong origin |
| 23 | selector-score inputs | reward never reads the model | `FORBIDDEN_REWARD_INPUTS` (`rewards.py:65-70`) | the proxy reads only the candidate controls, the human's poses and the join | FAITHFUL | — |

## 3. Advantage

| # | component | DDv2 (SPEC) | LIB | PORT | class (PORT) | why |
|---|---|---|---|---|---|---|
| 24 | intra-anchor normalisation | (r − mean_G)/(std_G + 1e-4), unbiased std (P Eq. 8, `rl.py:886-889`) | centre only by default (`advantage.py:85-87`, `config.py:120`) | `intra_anchor_advantage` (`ddv2_rl.py:231-281`) | FAITHFUL | hand literals [T] |
| 25 | truncation | clamp(min = 0) **per sample** (P Eq. 10, `rl.py:893`) | un-truncated intra term plus a clamped **across-anchor** term (`advantage.py:208-213`) | per sample | FAITHFUL | LIB's composite is a different object (regression) [T] |
| 26 | ≥GT mask | per sample, code only (A7) | on the per-anchor mean (`advantage.py:171-173`) | per sample | FAITHFUL | — |
| 27 | −1 branch | NC ≠ 1 **or** DAC ≠ 1, applied after the mask (A8) | collision or TTC veto on the anchor mean (`advantage.py:210-212`, `posttrain.py:201-209`) | `constraint_fail = (NC ≠ 1) ∨ (DAC ≠ 1)`, after the mask | FAITHFUL (DAC ≡ 1 ⇒ NC only) | a static-object 0.5 goes to −1, as in the release [S13 §13.3] |
| 28 | γ discount | 0.8^(T−i−1) per step (P Tab. 7) | none (no chain) | `discount_weights` (`:284-292`) | FAITHFUL | literal [T] |

## 4. Loss and regularisation

| # | component | DDv2 (SPEC) | LIB | PORT | class (PORT) | why |
|---|---|---|---|---|---|---|
| 29 | policy loss | −exp(logp − logp.detach())·A over [B, 80, 10] ⇒ REINFORCE (§6.1) | −(A·logp).mean() on ONE surrogate log-prob (`advantage.py:243-245`) | `rl_loss_per_row` (`:295-322`); training uses the exact per-step decomposition `step_loss_weights` / `per_step_loss` (`:464-501`) | FAITHFUL | per-step backward == one-graph released loss, value and gradient [T]; a wrong denominator is caught [T] |
| 30 | normalisation | mean over NON-ZERO-advantage samples (A11) | mean over all samples | the same | FAITHFUL | [T] |
| 31 | IL term | L1 of every chain at every step to the one GT, λ 0.1 / 1.0 per row, batch-global (A12–A14) | **not wired**: `imitation_loss` is never passed (`posttrain.py:448-450`), `w_imitation = 0.0` in both launchers | L1 of `state_to_path(x̂0)` in metres to the GT's 8 waypoints, λ per the release | FAITHFUL (form) / DIFFERENT (horizon) | our 8 waypoints span 6 s, the release's 4 s. MEASURED in the smoke run: at the cold start this all-modes L1 is **6.4 m** and dominates the step (see RESULT) |
| 32 | trust region | none; the IL term is the regulariser (§6.7) | optional L2 anchor to a frozen policy (`posttrain.py:359-376`, off in the pilot) | none | FAITHFUL | — |

## 5. What trains, how long, and evaluation

| # | component | DDv2 (SPEC) | LIB | PORT | class (PORT) | why |
|---|---|---|---|---|---|---|
| 33 | trainable set | `_trajectory_head.*` (§7.1) | `decoder.*` minus `conf_head` (`rl_pilot_refc21.py:882-884`) | `traj_proj`, `time_mlp`, `layers`, `control_head` = **9,474,832** params (`ddv2_rl_refcv5.py::trainable_params`) | FAITHFUL (spirit) / DIFFERENT (coupling) | ⛔ our deployed plan is ranked by `conf_head` on the **same** `layers`/`traj_proj` (`sampler_ranks_the_fan: true`), so RL moves the ranking's inputs with no ranking objective. DDv2 retrains a separate selector (stage II, not prepared here). Only upstream-of-capture parameters get no gradient [T] |
| 34 | optimiser | AdamW 2e-4, wd 1e-4 (P Tab. 7) | AdamW 1e-5, no wd (`posttrain.py:428`) | AdamW 2e-4, wd 1e-4 | FAITHFUL | — |
| 35 | schedule | "cosine with 10 % linear warmup" (P §7); the code steps per epoch, making warmup a no-op (A15) | none | warmup + cosine over **steps** (`lr_at`) | FAITHFUL to the paper's text | the per-epoch stepping has no meaning at a sub-epoch scale |
| 36 | scale | 10 epochs × navtrain, batch 512 on 8 L20s, 16-mixed, no clipping | 2,000 steps × batch 2, clip 1.0 | 600 steps × 4 windows, fp32, no clipping | DIFFERENT (scale) | ≈ 0.4 % of the release's samples: a mechanism validation, not a reproduction (PREREG §3) |
| 37 | frozen trunk | yes; per-module `.eval()` (A17 caveat) | frozen, eval | frozen; whole model in `eval()`, the decoder has no dropout | FAITHFUL | — |
| 38 | stage-II selector | coarse-to-fine scorer, BCE + margin rank, augmentations (§8) | a selector exists (`2026-09-10-ddv2-usable-now`) but is not on this path | not prepared | MISSING (out of scope) | the PI asked for the RL stage |
| 39 | evaluation | NAVSIM navtest PDMS (T1-class, "closed-loop" in the paper's words) | T1 four families (harness) | T0 held-out proxy read of the **deployed** fan with paired noise; T1 via `refcv3_arm.py` on the held-out clips | **NOT-IMPLEMENTABLE-ON-OUR-DATA** (PDMS) → DIFFERENT | no NAVSIM metric cache for PhysicalAI-AV. ⛔ Any "best of the fan" number is an ORACLE (A16) and is labelled `fan_pdms_best` |

## 6. PI DECISION QUEUE items 7 and 8, resolved for this stage

**Item 8 (the cold start's 487-vs-488 keys) — CLOSED on 2026-09-10 (option (c)). It does not apply to refcv5-v2.**
MEASURED today on the tip code: refcv5-v2 loads with **0 missing and 0 unexpected keys** (`tolerated_inert_buffers: []`). Its `decoder.anchor_controls` holds **212 non-zero entries**, the decoder is `anchor_v0_cond = True`, and D1 reproduces its sampler bitwise on 12/12 real windows. Verdict: **no action.** The declared allowance stays scoped to pre-2026-09-04 checkpoints; the DDv2 driver never needs it and refuses a non-strict load.

**Item 7 (the pilot's executed adapter is unguarded; should `v0` be REQUIRED) — answered for this stage.**
* **Q1, "guard `refcv3_adapter.py` in place, or repoint the pilot":** neither adapter is on the DDv2 path. The driver feeds `RefCV3Model` through the T1 harness's own conditioning (`refcv3_arm.load_model`, `build_corpus`, `refc_v3.ego_state_from_batch`, v0 = `pose_last[3]`, keep = 1). So the DDv2 stage **needs no decision on Q1**, and the pilot's default-if-silent (guard in place) stands. No LIB code was changed, because another arm's code path was out of bounds for this task.
* **Q2, "should `v0` be REQUIRED":** **for any v0-conditioned build (the refcv5-v2 family), yes.** The anchored Gaussian's bank, the sampler's integration speed and the X15 keep flag all read it. The DDv2 driver supplies it on every window. The binding's `wrong-speed` regression (`test_ddv2_refc_chain.py`) goes RED when the sampler integrates at anything but the window's own speed. **Still open for the PI:** whether to make v0 REQUIRED in the shared contract for July-style unconditioned builds. That changes LIB, and it is the owner's call, not this stage's.

## 7. What this diff means for the validation run

1. **The arithmetic is the release's** (rows 3–7, 11, 20–21, 24–30, 34), pinned bitwise or by literals, with **mutation proof** in `raw/MUTATION_SWEEP.json`.
2. **Three differences can move the outcome for reasons unrelated to RL. Each was measured before any arm ran:**
   * the off-support labels (row 4, a 6.1 m fan shift);
   * the release clamps acting on a control state (rows 8–9, +5.7 m more);
   * the all-modes IL pull (row 31, a 6.4 m L1 at the cold start).
3. **The reward is PDMS-shaped, not PDMS.** DAC is absent in training, EP has a human reference, and NC/TTC lack their map clauses (rows 15–18). No number from it may be quoted as PDMS.
4. **One coupling DDv2 does not have:** RL moves our ranking's inputs (row 33). So the T1 read of the **selected** plan tests RL together with an untrained ranker, and the T0 read of the **fan** is the cleaner test of the RL term. That is why the PREREG makes it primary.
