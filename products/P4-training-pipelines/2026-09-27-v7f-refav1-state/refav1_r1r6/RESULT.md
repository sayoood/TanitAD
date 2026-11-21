# refav1 — R1 / R3 / R5 / R6 built (default OFF, bit-identical when off)

**2026-09-27 · TrainingFlyWheel (P4 training pipelines) · base = origin tip `c36b6ddd` · STAGED, never committed.**
PI directive 2026-09-27 (BINDING), requirements R1–R6 for refav1; brief = "build the R1/R3/R5/R6 architecture
changes, default-OFF, bit-identical when off"; mid-task coordinator update = "make the R5 prior a selectable flag
(`kdx` / `damp50` / `cv`), keep the zero-init known-value control per prior, list it in G-DVB".

## 0. Headline
1. **R6 ✅ · R1 ✅ · R5 ✅ · R3 ✅ built, every lever DEFAULT OFF** (+ the coordinator's `--r5-prior {kdx,damp50,cv}`);
   nothing trained, no metric. R3's "ALL" needs the PI's 2026-09-16 cot-absence ruling applied to the v7.2 blob
   (sidecars built here, PROPOSED): default policy trains 17/22 goal classes, cot-absence 21/22 + SPEED_BAND args.
2. **Flags:** `--strategic-off` · `--vmax-input` · `--w-goal` (+ `--goal-negatives`, `--cot-negative-sidecar`) ·
   `--w-speed-band` · `--w-traj` (+ `--r5-prior`, `--r5-kappa-source`); refav1 G-DVB entries for all 48 trainer
   dests, WIRED into the trainer (refuses before step 1); launch-gate refav1 PROFILE = proposal only.
3. **Zero-init known value:** the R5 head's output == the chosen prior bit-for-bit for all 3 priors; the prior ==
   the banked `kd_x` reference on **2,399 real eval windows, max |Δ| = 0.0** (also `damp50`, `cv`).
4. **Tests:** brief's set **511 passed / 10 skipped / 0 failed** (baseline 424 / 10; +87 new); 27 other refav1-touching
   stack files 600 passed / 1 failed; 3 taniteval files 21 passed / 3 failed / 76 skipped — failures IDENTICAL by id on the unmodified
   tree (CRLF checkout hash; a `products/` file absent from the snapshot) ⇒ **0 new failures**.
5. **Bit-identical when off:** cross-version dump vs unmodified `c36b6ddd` — **6,088 tensors + 310 values
   BIT-IDENTICAL** (9 model configs, `plan()`, the loader on real episodes, 3 trainer smokes incl. checkpoints); the
   real-data OFF trainer smoke's rows + checkpoint + optimizer state identical to the unmodified trainer.
6. **Needs:** PI — D1 κ0 source (the steer channel at t0 is "unruled at inference" per SPEC_REFCV7 §10), D2
   cot-absence on v7.2, D3 SPEED_BAND target vs max-speed input (echo), D4 prior default; **Master Mind** — land
   the 11 files with the R1-VMAX batch, wire `refav1_arm.py` (R5 arm, `speed_max_ms` into `plan()`), build the gate
   profile; any real run is blocked on the deleted ~303 GB refav1 TRAIN fp8 cache.

## 1. What was built — per requirement, with its flag (every flag DEFAULT OFF)

| req | built | trainer flag(s) | model (`RefAV1Config`) | where |
|---|---|---|---|---|
| **R6** strategic OFF, tactical + operative + nav KEPT | ✅ | `--strategic-off` | `strategic_off` | `refa_v1.py` `_build_r1r6` / `_run_brains` / `forward` |
| **R1** max speed as a TACTICAL INPUT (train + inference) | ✅ | `--vmax-input` | `vmax_input` | `refa_v1.py` `_vmax_onehot` / `_run_brains`; loader `speed_max` |
| **R3** ALL tactical labels train the tactical layer | ✅ ("all" also needs the PI's cot-absence ruling on v7.2 — §8.2 D2) | `--w-goal`, `--goal-negatives`, `--cot-negative-sidecar`, `--w-speed-band` | `w_goal`, `w_speed_band` | `refa_v1.py` `_r1r6_losses`; loader `goal_targets` / `speed_band` |
| **R5** ONE combined 6 s trajectory = residual over a kinematic prior | ✅ | `--w-traj`, `--r5-prior {kdx,damp50,cv}`, `--r5-kappa-source {steer_t0,pose_past}` | `w_traj`, `traj_prior`, `traj_kappa_source` (+ fixed dims `traj_*`) | NEW `refs/refav1_traj.py`; `refa_v1.py` `traj_readout` / `trajectory()`; loader `traj_steps` |
| **R4** tactical decision conditions the operative plan | kept (goal token → cost, FiLM tactical → operative untouched) **+ the R5 head is conditioned on the tactical decision** | — | — | `refa_v1.py` `tac_decision` |
| **R2** nav → tactical + operative | untouched (already met) | — | — | — |
| G-DVB (item 4) | ✅ as a refav1 registry, **wired into the trainer**; launch-gate PROFILE = proposal (§7) | all 48 argparse dests (49 option strings) | — | NEW `train/declared_vs_built_refav1.py` |

**R6 — how.** The strategic policy and the strategic subspace predictor are BUILT and then DROPPED (so every shared
module's random init is bit-identical to the strategic-on arm at the same seed — measured, `test_b_every_shared_module…`);
`strategic_cfg` stays as the holder of the nav/ctx DIMENSIONS only; the tactical policy's FiLM cond becomes
`0 + nav_to_ctx(nav) (+ vmax_to_ctx(max speed))`; `loss = w_op·op + w_tac·tac (+ labels…)`; the route label is dropped
by the trainer BY NAME (and refused by the model); the loader runs with `str_ext_steps = 0`; the three strategic
weights are written as 0.0 into the config and `sanity()` refuses them non-zero. Deliberate-regression arm: the old
spelling (`--no-hierarchy`) — nav then reaches NO layer (`op_pred` identical under two nav commands), pinned.

**R1 — how.** `speed_max_ms` = the clip's RAW `g_tac.goals.SPEED_BAND.v_hi_ms` (every window of a labelled clip; a
set speed is a clip property) → the refcv6 4-way CONTAINING-WINDOW one-hot, applied ONCE on the model side
(`MaxSpeedOneHotEncoder`); an unknown clip is the ALL-ZERO row. It enters the tactical FiLM cond and the intent through
BIAS-FREE, down-scaled projections (the `nav_inj` pattern), so an all-zero row adds EXACTLY 0 (measured: vmax-on with
all rows unknown forwards bit-identically to vmax-off), and it enters the R5 head's condition. The limit
`limit_ms_of_bin(speed_max_bin(v_hi))` (`RefAV1.speed_limit_ms`) is the SAME function the R1-VMAX planner cap uses.
`plan()` / `imagined_goal()` / `trajectory()` accept `speed_max_ms`; `plan()` stamps `res.speed_max_input` only
when the input exists. `trajectory()` applies R1's HARD cap to the emitted 6 s path (`refav1_traj.cap_path_speed`:
re-timing along the path's own geometry, never ahead of it; the identity when it does not bind; a window whose
MEASURED v0 already exceeds the limit sheds speed at `a_max` and is counted apart). config.json carries the stamp
`speed_max_derivation_v6` = `SPEED_MAX_DERIVATION_REFAV1`, checked on the file as written by the shared guard
`assert_speed_max_stamp_v6` (both directions).

**R3 — how.** 22 independent sigmoids (multi-label BCE) on the tactical INTENT (the vector lat/lon read), through
the shared audited functions: `v7_labels.tactical_goal_targets` (per-window `(y, w)`, the band rule, the provenance /
entailment negatives, or the PI 2026-09-16 cot-absence ruling with its sidecar), `tac_goal_head.tac_goal_loss`
(no `if n>0` guard: an unsupervised batch is a real zero in the graph), `goal_pos_weight` + `mask_report` FITTED
FROM THE LOADED SPLIT into model buffers (`set_goal_supervision`; the loss REFUSES unfitted buffers). `SPEED_BAND`'s
presence bit is 1 on every record (a constant → masked); its CONTENT, `(v_lo_ms, v_hi_ms)`, is regressed from the
same intent (`--w-speed-band`, Smooth-L1 on v/30, in-band windows). **Why multi-label, not a factored CE:** the goal
set is a SET (2–7 goals/record, mean 2.751 — MEASURED here on the canonical train blob,
`raw/checks/goal_set_sizes_v72_train.txt`); a CoT-backed token's ABSENCE is "the caption did
not say so", so under the provenance policy that cell must be IGNORED — a factored CE with a "none" class would have
to supervise exactly those unknowable cells as "none"; mutual exclusion (the four traffic-light colours) enters as
ENTAILED negatives, which a per-cell BCE consumes directly.

**R5 — how.** `pred = prior(v0, a0, κ0) + head(regions, cond)`, 30 × 0.2 s = 6 s.
* **prior** (`--r5-prior`, `refav1_traj.TRAJ_PRIORS`): `kdx` (default) = `retime(damp50, ha0_ext)` — the banked
  `build_explore.py::retime`, batched in float64; `damp50` = `0.5·ha0 + 0.5·ha0_ext`; `cv` = `ha0`. All through the
  programme's ONE integrator (`unicycle_paths`). `ha0_ext` alone is not offered (6 s ADE 5.36 / FDE 15.39).
* **regions**: `_last_state(field)` (the ONE start state every rollout reads) pooled into 4 × 10 regions of the
  row-major 16 × 40 grid, each the mean of 4 × 4 tokens — exactly the trunk probe's
  `x.reshape(4,4,10,4,-1).mean(dim=(1,3))` (pinned: region (1, 7) of a ramp = 249.5; NOT the global mean).
* **cond** (54-wide at the real geometry): the probe's 8 kinematic features `(v, a, κ, v², va, vκ, v²κ, aκ)` on fixed
  scales, nav one-hot (4), max-speed one-hot (4, R1), and the TACTICAL DECISION (R4): lat (8) + lon (8) posteriors
  + the 22 goal posteriors when R3 is on — DETACHED (the decision is trained by its labels and conditions the
  trajectory; the imitation loss does not rewrite it).
* **head**: LN → Linear(1024 → 64) per region + learned region embedding → flatten → MLP(2614 → 512 → 512) →
  Linear(512 → 60) **zero-initialised (weight AND bias)**: 1,702,524 params. Loss: masked Smooth-L1 (β = 1 m) on the
  (x, y) waypoints against the recorded future ego path (frames 2(t+1)…2(t+30), ego frame at 2t, the arm's
  `gt_waypoints`), NaN-safe under masks; the loader masks steps whose frame does not exist or is non-finite.
* **κ0 source** (`--r5-kappa-source`): `steer_t0` (default, the brief's "exactly as `ha0_ext` measures it") or
  `pose_past` (refcv7's `ha0_ext_pose` convention on refav1's 0.2 s grid, through ITS function — see §2 and §8.2 D1).
* **inference**: `RefAV1.trajectory(feats, v0=, a0=, kappa0=, nav_cmd=, speed_max_ms=)` → the capped 6 s path + the
  uncapped one + the cap record.

## 2. ⛔ Admissibility — what every NEW input is computed from

| input (new) | consumed by | computed from | admissibility |
|---|---|---|---|
| `speed_max_ms` (R1) | tactical FiLM cond, intent, R5 head; the R1 caps | label `g_tac.goals.SPEED_BAND.v_hi_ms` → 4-way one-hot | ⛔ **ORACLE input** (ego-future: max realised speed over [t0+2, t0+6] s), PI-authorised as a declared INPUT (2026-09-16 refcv6; R1 2026-09-27); **stamped** (`speed_max_derivation_v6`); read only behind the manifest's oracle stamp; **not** a situation-classifier output (a geometry label field). A T1 read that feeds the label-derived value is an oracle-input arm and must be labelled so; deployment needs a set-speed / limit source. |
| `v0` (R5) | prior, head | `poses[2t, 3]`, measured speed at t0 | admissible (PI 2026-09-02) |
| `a0` (R5) | prior, head | `(v[2t] − v[2t−2]) / 0.2` | backward difference of past measurements — admissible |
| `κ0`, `steer_t0` (R5 default) | prior (`kdx`/`damp50`), head | `actions[2t, 0]` = the recorded egomotion curvature channel at t0 (`atan(2.9·κ)`), integrated as curvature exactly as the banked `kd_x` floor did | a measured ego state AT t0 — **but** SPEC_REFCV7 §10 (A5) records this channel as "**unruled at inference**" and refuses it for refcv7 ⇒ **PI decision (§8 D1)** |
| `κ0`, `pose_past` (R5 alt) | same | `wrap(yaw[2t] − yaw[2t−2]) / 0.2 / max(v0, 2)`, clamped ±0.3 (`kinematic_prior.prior_controls("ha0_ext_pose")`) | past poses only — the refcv7-ruled form |
| nav one-hot into R5 | head | the same `nav_cmd` as R2 | unchanged (oracle nav, stamped) |
| tactical decision (R4) into R5 | head | the model's OWN lat/lon/goal posteriors (vision + nav + max speed), detached | a model output, not a label; **not** the situation classifier (a different label family, `data/situations.py`); the goal SET enters only as a TARGET, never as an input |
| region-pooled field | head | adapter(std(DINOv3 fp8)) at t0 | vision |

Training TARGETS (never inputs): `goal_y/goal_w` (the goal set per window), `speed_band` (SPEED_BAND args, in band),
`traj_gt` (recorded future poses). Every index a new input reads is ≤ 2t — asserted by mutation on real episodes
(corrupting every frame after 2t leaves `a0`, `κ0` and the prior bit-identical; corrupting 2t−2 moves `a0`).

## 3. Proofs (each with the control that must read the other way)
### 3.1 Bit-identical when OFF — ACROSS VERSIONS (the unmodified `c36b6ddd` tree vs this one)
`raw/bitident/`: ONE dump script, run under each tree's `PYTHONPATH` (it asserts `tanitad` resolves into the tree it
names), using only pre-directive APIs; `bitident_compare.py` compares every tensor with `torch.equal` (dtype + shape
+ bits) and everything else with `==`:
* **model** — 9 configs (default, `--no-hierarchy`, speed channel, EMA targets, counterfactual term, motion
  injection, adapter target space + SigReg + var-floor, `proposal_k 2` + aux head, `tmix 1` + v6 vocabulary):
  init state dict, a forward on a fixed batch WITH labels / nav / v0 / str-ext targets (every output), every
  parameter gradient after backward, the state dict after one AdamW step, and an inference forward;
* **plan()** — 3 call variants (shipped; `ccosh` + seam + tactical grid; steer units + plan grid + κ ladder +
  goal-conditioned κ weight): controls, cost, source, baseline + fine costs, the RESULT's attribute set, goal action;
* **loader** — `RefAV1Windows` on 3 REAL eval episodes with the v7.2 labels + nav, 3 option sets × 5 batches;
* **trainer** — `refa_v1_train.py --smoke` × 3 (default; `--no-hierarchy`; `--speed-channel --ema-targets`),
  4 steps: every log row (minus wall-clock) and the final checkpoint's model AND optimizer state.

**VERDICT: BIT-IDENTICAL — 6,088 tensors (bitwise) + 310 other values, 0 differing entries, nothing only-in-one-side** (`raw/bitident/compare.out.txt`, re-run on the final code).**** The ONLY difference is that `config.json["cfg"]` carries the 14 new
`RefAV1Config` fields at their OFF defaults (printed by the compare script, never hidden).

### 3.2 Bit-identical when OFF — the MECHANISM, pinned in-tree (`test_refav1_r1r6_off_identity.py`)
With every flag off `_build_r1r6` builds nothing and draws nothing: a model built with it replaced by the
pre-directive stub has the identical state dict, the identical RNG state after the build and identical forward
outputs (4 variants); every new forward input is REFUSED while its flag is off; the trainer's new flags translate to
NO config kwarg and NO loader kwarg; the loader emits exactly the pre-directive keys. Controls: an ON flag DOES build
and DOES draw. Plus the pre-existing pin `test_refa_v1_precision::test_OFF_path_identity…` (step-1 row vs the
pre-edit trainer, rel 1e-6) and `test_tac_loss_logging::test_e…` (the OFF log row key-for-key) — both pass.

### 3.3 R5 (a) KNOWN VALUE — the zero-init head IS the chosen prior, and the prior IS the banked floor
* **In-model, every prior** (`test_a2…[kdx|damp50|cv]`): residual `== 0` exactly, `traj_pred == traj_prior ==
  kinematic_prior(name, v0, a0, κ0)` with `torch.equal`; `traj_ade_m == traj_prior_ade_m` in the row; control: one
  optimizer step and the prediction leaves the prior.
* **The prior vs the reference, synthetic edge cases** (`test_a1…`): v0 = 0 (degenerate path), decel to a stop,
  |κ0| up to 0.5, a0 > 0 (extrapolation branch) — the batched `retime_paths` == the VERBATIM
  `build_explore.py::retime` bit for bit at K = 10 and 30.
* **The prior vs the reference, REAL data** (`raw/checks/kdx_reference_check.out.json`): our priors rebuilt from the
  banked t0 kinematics `kin = (v0, a0, κ0)` ALONE vs the arm's own per-window 30-step paths banked by the trunk probe,
  **2,399 eval windows / 141 episodes: max |Δ| = 0.0 for `ha0`, `ha0_ext`, retime, `kdx`, `damp50`, `cv`**
  (`n_windows_kdx_not_bitexact` = 0).
* **Every window of the smokes** reads `traj_ade_m == traj_prior_ade_m` at step 1 (§5).

### 3.4 R5 (b) GRADIENT and (c) COUNTERFACTUAL (`test_refav1_r5_traj_head.py`)
* (b) back-propagating `loss_traj` ALONE: at step 0 `out` gets gradient and — the zero-init property as a KNOWN
  VALUE — the region projection, the MLP and the adapter get EXACTLY 0; after ONE optimizer step the region
  projection, the region embedding, the MLP AND the adapter (through the 4 × 10 pooling) all get non-zero gradient;
  the tactical policy gets exactly 0 (the decision is detached). Region pooling is pinned to the probe's layout at
  the real geometry and shown to carry spatial information the global mean does not.
* (c) swapping the tactical decision input (TURN_L vs LANE_KEEP, same lon) changes NOTHING at init (`torch.equal`) and
  changes the 6 s prediction after 3 training steps (max |Δ| > 1e-4 m).

### 3.5 R3 — every goal family trains the tactical LAYER; every regression arm goes RED (`test_refav1_r3_goal_head.py`)
A family is GREEN iff it has ≥ 1 supervised cell, a finite term, and back-propagating THAT term ALONE gives non-zero
gradient on its own head row AND on the tactical policy. On a batch supervising every token: **all 22 families
GREEN — the 21 BCE tokens (each traffic-light colour named and asserted individually) + SPEED_BAND's argument
regression.** Deliberate-regression arms, each must go RED on exactly the families it breaks: class-mask RED off →
{RED}; a loader that drops the YELLOW cells → {YELLOW}; the goal head fed a DETACHED intent → all 21 BCE families
(the layer never trains); the pre-directive model (no R3 head) → all 22; SPEED_BAND mask all-false → {SPEED_BAND}.
The pooled term IS `tac_goal_loss` (`torch.equal`) and the per-token split sums to it; unfitted split constants are
refused; an all-unsupervised batch is a real 0.0 with finite gradients.
**On the REAL blobs** (`raw/checks/goal_census_v72.*`, the trainer's own functions): v7.2 TRAIN under the default
`measured` policy → **17/22** BCE classes trainable (masked: YIELD, SPEED_BAND, CORRIDOR_OFFSET, GAP_TARGET,
REACT_ON_ONCOMING — positives, NO supervised negative); under `cot-absence-negative` with the v7.2 sidecar → **21/22**
(only SPEED_BAND, a constant, masked — its arguments train through the regression head, present on 4,572/4,572
records). All four traffic-light tokens are trainable under BOTH policies on train (RED 376 pos / 403 neg measured →
376 / 4,196 cot-absence). Pinned by `test_f_on_the_real_train_blob…` (builds the sidecar in `tmp` with the unchanged
builder). EVAL (147 clips): 13/22 and 17/22 — the small split lacks positives for some tokens.

### 3.6 R6 / R1 (`test_refav1_r6_strategic_off.py`, `test_refav1_r1_vmax_input.py`)
R6: strategic modules absent, tactical + nav kept, shared init bit-identical to the strategic-on arm, nav moves the
decision and both predictors, loss = the two remaining feature terms EXACTLY, route label / str-ext refused, EMA +
`plan()` run, `sanity()` refuses every half-off combination, trainer smoke row `loss_feat_str = None`; deliberate
regression = `--no-hierarchy` (nav reaches nothing). R1: the one-hot IS refcv6's encoder, unknown = all-zero, the limit
IS the R1-VMAX function, all-unknown forwards bit-identically to the input being off, a known max speed moves the
decision and both predictors, gradient reaches the input (only the active bin's column), refusals both ways, `plan()`
stamps it only when on, the stamp passes the shared guard (and the mirror refuses it), the loader emits the RAW
per-clip `v_hi` on every window (real data). The FiLM-cond inputs (`nav_to_ctx`, `vmax_to_ctx`) receive EXACTLY 0
gradient at step 0 — the tactical FiLM (`predictor.FiLM`) is zero-initialised — and non-zero after one step (pinned
as a known value; a gate G-LIVE must read them at step ≥ 2).

### 3.7 The in-band rules agree, and the eval side rebuilds the trained model
The goal targets' band rule (`window_in_band`) and the loader's lat/lon/SPEED_BAND rule agree on **9,443 / 9,443**
labelled eval windows (2,961 in band) — `raw/checks/band_rule_agreement.out.txt`. The eval tool's own
`refav1_arm.build_config` rebuilds an all-levers-on checkpoint from `ckpt["cfg"]` AND from config.json (JSON lists),
loads STRICT and emits bit-identical 6 s trajectories (`test_l…`).

## 4. Tests — counts, verbatim
CPU only (`CUDA_VISIBLE_DEVICES=-1`), `PYTHONPATH` = the tree under test (asserted: `tanitad.__file__` resolves into it).

| set | tree | result (verbatim last line) | log |
|---|---|---|---|
| **A** the brief's set `tests/test_refa_v1*.py tests/test_refav1*.py` — BASELINE (37 files) | unmodified `c36b6ddd` | `424 passed, 10 skipped in 442.09s (0:07:22)` | `raw/pytest_baseline_c36b6ddd.txt` |
| **A** same glob = the 37 + the 6 NEW files | R1–R6 | `511 passed, 10 skipped in 252.15s (0:04:12)` | `raw/pytest_A_refav1_set_NEW.txt` |
| **B** every OTHER stack test mentioning refav1 (27 files) | R1–R6 | `1 failed, 600 passed, 1 warning in 169.50s (0:02:49)` | `raw/pytest_B_other_refav1_mentions_NEW.txt` |
| **B** same 27 files | unmodified `c36b6ddd` | `1 failed, 600 passed, 1 warning in 275.79s (0:04:35)` | `raw/pytest_B_other_refav1_mentions_PRISTINE.txt` |
| **C** the 3 taniteval tests mentioning refav1 | R1–R6 stack | `3 failed, 21 passed, 76 skipped in 2.53s` | `raw/pytest_C_taniteval_refav1_NEW.txt` |
| **C** same | unmodified stack | `3 failed, 21 passed, 76 skipped in 1.80s` | `raw/pytest_C_taniteval_refav1_PRISTINE.txt` |

* **A: 0 failures; +87 passing test cases = the 6 new files** (`test_refav1_r1r6_off_identity` / `_r6_strategic_off` /
  `_r1_vmax_input` / `_r5_traj_head` / `_r3_goal_head` / `_dvb`, incl. 5 real-data tests on the dev-box eval kit and the
  real-train-blob R3 census test with `TANITAD_V72_TRAIN_LABELS` pointing at a byte-identical copy of the canonical
  blob, md5 `0ff90213…`). The 10 skips are the baseline's own, reasons verbatim in the log: 9 ×
  `test_refa_v1_dk_hook.py` "INCONCLUSIVE, never passing: RefAV1.plan() has no `dk_spec` parameter" (pre-existing,
  not mine) and 1 × `test_refa_v1_precision.py` "no CUDA device" (the CPU-only rule).
* **B / C: the SAME failures by test id in both trees ⇒ 0 new failures**, both environmental and pre-existing:
  B `test_launch_gate.py::test_G_SUITE_the_pinned_env_list_is_content_bound_by_the_profile` — the snapshot is a CRLF
  checkout, so `stack/ops/launch_gate_thor_env_failures.json` hashes `9b6aa814…` instead of the pinned `fc1214e5…`
  (its LF-normalised hash IS `fc1214e5…`; the file is identical in both trees); C 3 ×
  `test_the_checker_reads_the_refusal_as_a_WORK_ITEM…` — `products/P7-TanitEval/CRITERIA_REGISTRY.json` is not in
  the snapshot (only `stack/`, `taniteval/`, `tools/` were copied).
* Pre-existing pins that guard the OFF path and pass unchanged: `test_refa_v1_precision::test_OFF_path_identity…`
  (step-1 row vs the pre-edit trainer), `test_tac_loss_logging::test_e…` (the OFF log row key-for-key),
  `test_refa_v1_motion::test_plan_and_forward_share_the_injected_state` (it caught my first R5 draft reading
  `field[:, -1]` in `forward`; the head now reads the ONE `_last_state`).

## 5. CPU smokes of `refa_v1_train.py` on 3 REAL eval episodes (tiny architecture) — never a metric
`raw/smoke/smoke_real.py` drives the REAL trainer (`main(argv)`: refusals → G-DVB → build → the real loader on the fp8
eval cache + v7.2 labels/nav → forward/backward/step → log → checkpoint) with the architecture shrunk by patching the
trainer's `RefAV1` name (depth / heads / brain widths / R5 widths); the GEOMETRY is untouched (d_enc 1024, 640 tokens,
d_state 1024 — 42.3 M params). 3 eval episodes (a turn, a plain lane-keep, a red-light clip — `sha12` in
`smoke_data/picked_sha12.json`, not banked), `-X faulthandler`, all on the final code.

| run | new flags | steps × bs | wall | outcome |
|---|---|---|---|---|
| **OFF** | none | 3 × 2 | 73.5 s | rc 0, every loss finite; **bit-identical to the UNMODIFIED `c36b6ddd` trainer on the same real data: 3/3 log rows, all 167 checkpoint tensors AND the optimizer state** (`compare_off_rows.out.json`) |
| **ON, strategic OFF** | `--strategic-off --vmax-input --w-goal 0.1 --goal-negatives cot-absence-negative --cot-negative-sidecar <v7.2 EVAL sidecar> --w-speed-band 0.1 --w-traj 1.0` | 4 × 3 | 137.5 s | rc 0; route label DROPPED by name; row 1: `loss_feat_str` None, `loss_goal` 0.9468 (n_sup 5), `loss_speed_band` 1.1594 (n 1), `loss_traj` 2.2440 (90 valid steps), **`traj_ade_m` = `traj_prior_ade_m` = 2.3371 (the known value)**; stamp PASS; `trajectory()` from the checkpoint on 6 windows: limit 8.33 m/s (30 km/h) on all, **`n_rows_over_limit` 0**, 1 window whose MEASURED v0 8.87 m/s already exceeds it → 2 steps capped, residual 0.525 m/s on that window only (an input property, counted apart) |
| **ON, strategic ON, `damp50` + `pose_past`** | `--vmax-input --w-goal 0.1 --w-speed-band 0.1 --w-traj 1.0 --r5-prior damp50 --r5-kappa-source pose_past` | 2 × 2 | 73.4 s | rc 0; row 1 `loss_feat_op` 1.3551 = the OFF run's row 1 exactly (shared init unchanged, same windows); `loss_goal` 0.7915 (n 5, `measured`), `loss_traj` 1.2220, **`traj_ade_m` = `traj_prior_ade_m` = 1.3945**; `trajectory()`: `n_rows_over_limit` 0 |

On the 3-clip subset the goal head reports 5/22 classes trainable (most tokens have no positive in 3 clips) — a
smoke property, not a corpus one (§3.5). ⚠️ One EARLIER launch of the strategic-on `damp50` smoke died with exit
**139** (native SIGSEGV, no Python traceback) during step 1 while running concurrently with a 4-thread pytest; the
identical command re-run alone passed, and the final run (above) passed — **not reproduced, cause UNVERIFIED**
(`NOTE_damp50_first_attempt_exit139.md`). A pod launch should run under `-X faulthandler`.

## 6. EXPLORATORY numbers produced on the way (zero GPU, reported because they bear on defaults)
All on the trunk probe's banked 2,399-window / 141-episode stride-4 EVAL grid (`raw/checks/`), plain means (no
CI — the paired bootstrap is the coordinator's `floors_6s_exploratory.json`, whose four means these reproduce to 4 dp):

| prior (K = 30, 0.2 s) | ADE @ 2 s (first 10 steps) | ADE @ 6 s | FDE @ 6 s |
|---|---|---|---|
| banked 2 s `kd_x` (built on the 10-step paths) | 0.3085 | — | — |
| **`kdx`** (built on the 30-step paths — the R5 default) | **0.3063** | 3.5584 | 11.2158 |
| `damp50` | 0.3675 | **3.4540** | **10.1499** |
| `cv` (`ha0`) | 0.5650 | 4.1280 | 11.2109 |
| `ha0_ext` (not offered) | 0.5723 | — | — |

⚠️ **The 30-step `kdx` is not the 10-step `kd_x` truncated**: `retime` extrapolates past the end of a 10-step damped
path where the 30-step path continues, so the first 2 s differ on **1,274 / 2,399** windows — 1,157 of them with
a0 > 0 (the extrapolation branch: `ha0_ext` out-travels the damped path's own 10-step length); the 44 windows that
differ by > 0.5 m (max 9.59 m) have median |κ0| 0.22 vs 0.0025 overall and a0 > 0 on 91 %
(`raw/checks/kdx30_vs_kdx10_mechanism.txt`). On the mean the 30-step form is slightly BETTER at 2 s (0.3063 vs 0.3085). Any bar quoted for R5 must name WHICH `kd_x`. ⇒ D4: `kdx` wins at 2 s, `damp50` at 6 s on FDE.

## 7. G-DVB (item 4) and the launch gate
`stack/tanitad/train/declared_vs_built_refav1.py` = the refav1 G-DVB **entry set**, same contract and API names as the
binding `declared_vs_built.py` (`check` / `refuse_on_mismatch` / `coverage` / `register`; `Lever` / `Mismatch`
imported, not re-declared). **Every one of `refa_v1_train.py`'s 48 argparse dests has an entry** (a flag
without one is refused — `coverage()` is asserted empty on the real parser, and an added unregistered flag is caught
by name); the 9 new dests are `built` / `loss` levers read off the BUILT `RefAV1` (e.g. `--w-traj` ⇔ `traj_head is
not None` + `cfg.w_traj` + output width 60; `--strategic-off` ⇔ strategic policy AND subspace absent, tactical +
`nav_inj_emb` present, the three strategic weights 0.0; `--vmax-input` ⇔ both projections 4-wide and the R5 condition
carrying them; `--r5-prior` / `--r5-kappa-source` ⇔ the config fields the forward / loader read) or `elsewhere` /
`data` with a verifiable reason (`--goal-negatives`, `--cot-negative-sidecar`); the 39 pre-existing dests are
registered too (model levers with readers, the rest as data / runtime). Plus G-HYG on the model's config tree.
**It is WIRED into the trainer**: `main()` calls `refuse_on_mismatch(model, args, parser=build_parser())` after
`sanity()` and before the first step (`build_parser()` was extracted from `main()` for this — pure code motion).
Regression arms (each must be refused): a head argv declares but the model lacks (4 levers), a built strategic layer
under `--strategic-off`, a prior / κ0 source other than argv's, a head argv never asked for, an undeclared config
attribute (G-HYG), an unregistered flag, and a lossy `build_model` → the trainer exits BEFORE writing a log row.

⛔ **Why these entries are NOT in the shared registry:** `declared_vs_built.check` runs every registered `built` /
`loss` reader on the model it is handed; a refav1 reader there would run on a `RefCV3Model` at every refcv7 launch and
FAIL it. The refav1 registry is its own dict (pinned: the shared registry gains no refav1 key).

⚠️ **The launch-gate PROFILE is delivered as a PROPOSAL, not built** (the brief's fallback): `launch_gate.py`'s
G-LIVE / G-CKPT / G-EVAL / G-CLOCK drive `refc_v3_train.py` internals (`trainer_capture` patches that trainer's
functions; the eval loader is the refcv6 battery's). A `refav1` profile needs: (1) capture seams in
`refa_v1_train.py` (`build_model`, the step loop, `torch.save`); (2) G-DVB = `declared_vs_built_refav1.check(model,
args, build_parser())` (exists); (3) G-LIVE: every declared term finite (`loss_feat_op/tac[/str]`, labels,
`loss_goal`, `loss_speed_band`, `loss_traj`, `cf_loss`) and non-zero gradient on every trainable leaf **measured at
step ≥ 2** — at step 1 the R5 body and every FiLM-cond input (`nav_to_ctx`, `vmax_to_ctx`) read EXACTLY 0 by
design (zero-init output layer / zero-init FiLM; pinned as known values in the tests); (4) G-EVAL: `refav1_arm.
build_config` + strict load of the trainer's checkpoint — already shown bit-identical for the all-on model
(`test_l_the_eval_side_rebuilds…`); (5) G-CKPT save→resume (the goal buffers travel in the state dict); (6) G-CLOCK
on refav1's raw-timeline window clock `t·0.2 s` (loader docstring). Owner: the gate agent / Master Mind.

## 8. NOT done · decisions needed · integration
### 8.1 NOT done (stated so it is not read as done)
1. **No training run, no metric.** Every number about the R5 head's QUALITY is unmeasured; the smokes are 2–4 steps on
   3 episodes at random init. A first R5 arm needs its own pre-registration (bars vs the chosen prior AND the other two
   floors at 2 s and 6 s, all four families; a TRAINING-seed replicate — the head is deterministic, so there is no
   inference-seed floor to lean on).
2. **The eval harness is not wired for the new levers** (`taniteval/tools/refav1_arm.py` untouched): no R5 arm calling
   `RefAV1.trajectory()`, no `speed_max_ms` passed into `plan()` for a vmax-trained model, no pose-past κ0 derivation
   (its `hold_ext_controls` gives the `steer_t0` κ0 only).
3. **R1's hard cap on `plan()`** is the separate R1-VMAX batch (`PlanConfig.v_max`, staged earlier, NOT in
   `c36b6ddd`); this package does not touch `refa_v1_plan.py`. `trajectory()` carries its own cap.
4. **No refav1 launch-gate profile** (§7 — proposal; the G-DVB entries are built and wired).
5. **The refav1 TRAIN fp8 features no longer exist on Thor** (~303 GB to rebuild, full-grid RESULT §5) — any real run
   of these levers is blocked on that storage decision.
6. **The v7.2 cot-absence sidecars** (`raw/sidecars/`) were built by the UNCHANGED builder over the canonical blobs;
   the PI ruling was applied to v8.0 by the Data FlyWheel — landing these for v7.2 is PROPOSED, not done (§8.2 D2).
7. The R5 tactical decision is the model's own (detached) posterior; **no teacher-forced / scheduled-sampling
   variant** was built.
8. **A pre-existing defect found, NOT fixed (out of scope):** `v7_labels._MEASURED_GEOMETRY_TOKENS` is module state
   set by the LAST `load_v7_labels` call — loading the eval blob after the train blob in one process changes the
   negative policy (MEASURED: on the eval blob `LANE_CHANGE_L` is classified "geometry", so all its absent cells
   become supervised negatives, 147/147). A trainer that loads one blob is unaffected; an eval that reloads is not.

### 8.2 ⭐ Decisions needed (PI)
| # | decision | options | this build's default | evidence |
|---|---|---|---|---|
| **D1** | **κ0 at inference** (R5 prior + head input) | `steer_t0` = the recorded curvature channel at t0 (the brief's "exactly as `ha0_ext` measures it"; reproduces the banked floor) · `pose_past` = refcv7's `ha0_ext_pose` (past poses only) | `steer_t0` | SPEC_REFCV7 §10 (A5) records the steer channel at t0 as "unruled at inference" and REFUSES it for refcv7; refcv7's `ha0_ext_pose − echo` +0.0055 m ADE 0–2 s is INHERITED from `kinematic_prior.py`'s docstring (refcv6 battery surface), not measured on refav1 |
| **D2** | **R3 "ALL" needs the 2026-09-16 cot-absence ruling applied to the v7.2 blob** | `--goal-negatives measured` (17/22 BCE classes trainable; YIELD, CORRIDOR_OFFSET, GAP_TARGET, REACT_ON_ONCOMING carry NO negative) · `cot-absence-negative` + the v7.2 sidecar (21/22; every family incl. every traffic-light colour) | `measured` (flag default) | §3.5 census (MEASURED here, `raw/checks/goal_census_v72.*`); the ruling's cost — 692/867 = 79.8 % of visible-light clips labelled negative — is INHERITED from `v7_labels.py`'s docstring citing `…/2026-09-16-flywheel-negatives/RESULT.md`, not re-measured |
| **D3** | **SPEED_BAND regression target while max speed is an INPUT** | keep (v_lo, v_hi) · v_lo only · off | both (flag-gated, default off) | the input one-hot is the containing window of v_hi — a v_hi readout is partly echo; the config stamp says so |
| **D4** | **R5 prior default** | `kdx` (2 s floor) · `damp50` (6 s FDE, exploratory) · `cv` | `kdx` | §6: 6 s ADE kdx 3.558 / damp50 3.454 (n.s.), FDE 11.216 / 10.150 |
| **D5** | the max-speed input's inference source | label SPEED_BAND (oracle) · a map / nav set-speed service | label (oracle; stamped) | refcv6 ruling 2026-09-16 |

### 8.3 ⭐ Integration (Master Mind)
* **Land the 11 files together** (5 code + 6 tests; §9 table: base blob at `c36b6ddd` → new blob), **with the R1-VMAX
  batch** (`refa_v1_plan.py`, `refav1_arm.py`, `test_refav1_vmax_cap.py`) — NO file overlap between the two.
* Decide D1–D5 (PI), then: wire `refav1_arm.py` (R5 arm via `trajectory()`, `speed_max_ms` into `plan()`, the κ0
  source), and a refav1 launch-gate profile (§7).
* If D2 = yes: land `raw/sidecars/cot_absence_negative_v7.2_{train,eval}.json.gz` where the Data FlyWheel keeps the
  v8.0 one (digests are `sha12`, no clip id in the clear).

## 9. Deliverable manifest
Package root `P` = `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/refav1_r1r6/` (every `repo:` row
STAGED with `git add -- <exact path>`, blob-verified in §9.2; never committed).

| artifact | where | note |
|---|---|---|
| this report | `repo:P/RESULT.md` | |
| 5 code files, FULL (`stack/tanitad/refs/refa_v1.py`, `refs/refav1_traj.py` NEW, `data/refav1_loader.py`, `train/declared_vs_built_refav1.py` NEW, `scripts/refa_v1_train.py`) | `repo:P/code/fix/stack/...` | land at the same repo-relative path; table §9.1 |
| 6 test files, FULL (`stack/tests/test_refav1_{r1r6_off_identity,r6_strategic_off,r1_vmax_input,r5_traj_head,r3_goal_head,dvb}.py`) | `repo:P/code/fix/stack/tests/...` | all NEW |
| 11 unified diffs vs `c36b6ddd` (LF-normalised) + the blob table | `repo:P/code/diffs_vs_c36b6ddd/*.diff`, `repo:P/code/files_table.json` | |
| test logs (baseline A, final A, B ×2, C ×2) + the file lists + runners | `repo:P/raw/pytest_*.txt`, `raw/*_test_files.txt`, `raw/tests_*.txt`, `raw/run_suites.sh`, `raw/run_final.sh`, `raw/run_final.out.txt`, `raw/make_diffs.py` | |
| cross-version bit-identity (script, compare, logs) | `repo:P/raw/bitident/` | the two dumps (`base.pt`/`new.pt`) are NOT banked — regenerable in ~2 min by the banked script |
| real-data checks (prior vs reference; 2 s vs 6 s priors; R3 census; band-rule agreement; goal-set sizes; kdx30-vs-kdx10 mechanism) | `repo:P/raw/checks/` | |
| real-data trainer smokes (OFF, ON strategic-off, ON damp50 + pose-past strategic-on, OFF on the UNMODIFIED trainer + its row/ckpt comparison) + the driver + the exit-139 note | `repo:P/raw/smoke/` | `smoke_real_ON_damp50_strategic_on.log` = the exit-139 re-run, produced by an EARLIER code state (before `--r5-kappa-source`) — kept only as the crash evidence |
| v7.2 cot-absence sidecars (train 4,572 / eval 147 clips, `sha12` digests) + build logs | `repo:P/raw/sidecars/` | ⚠️ PROPOSED data artifacts (D2); built by the UNCHANGED builder; ONLY here + the dev-box working dir |
| the working tree (edited copy of `c36b6ddd` `stack/`) | `C:/Users/Admin/refav1_r1r6/stack` | dev box only — its 11 changed/new files ARE the staged `code/fix` copies (byte-compared) |
| the 3-episode smoke subset (copies of eval cache + v2ep files) | `C:/Users/Admin/refav1_r1r6/smoke_data` | dev box only, regenerable; NOT staged (clip-id-named data files) |
| local copies of the two canonical v7.2 blobs (md5 `0ff90213…` / `aa12c948…`) | `C:/Users/Admin/refav1_r1r6/data/` | dev box only; the originals are in the repo (`TanitAD Research Lab/Data Engineering/.../2026-09-04-v72-label-release/raw/`) |

### 9.1 Landing table — `P/code/fix/<repo path>` → `<repo path>`

| repo path | base blob at `c36b6ddd` | new blob (`git hash-object` of the LF content = the blob that lands) | package EOL | +/− lines |
|---|---|---|---|---|
| `stack/tanitad/refs/refa_v1.py` | `d504eb9b687899500662ad8a4d447e479bb3d049` | `6e0988aa2cec4634d101870995be11397aa8aa8e` | CRLF | +698 / −33 |
| `stack/tanitad/refs/refav1_traj.py` | `NEW` | `9359e6bcb8b016a6f828b34e6400779b67c43db3` | LF | +348 / −0 |
| `stack/tanitad/data/refav1_loader.py` | `2c5ac6eb18db02db0e08eccd19a4e4db1e3616d1` | `9047c0d31a042cc727abec00f206373ba0f845a3` | CRLF | +217 / −2 |
| `stack/tanitad/train/declared_vs_built_refav1.py` | `NEW` | `257e0938be6678a7f118ab16133890ba874e0501` | LF | +365 / −0 |
| `stack/scripts/refa_v1_train.py` | `9783142c31d44b6cb4a1fb39393e8206e69fc222` | `973dfd6063449f48afd84c806254a647a7bef8de` | CRLF | +356 / −12 |
| `stack/tests/test_refav1_r1r6_off_identity.py` | `NEW` | `082910ab042548f18b4b75d53ba5cf7482df35ed` | CRLF | +186 / −0 |
| `stack/tests/test_refav1_r6_strategic_off.py` | `NEW` | `77f0913bea76378ce75f70917ab0b9fe1b802404` | LF | +179 / −0 |
| `stack/tests/test_refav1_r1_vmax_input.py` | `NEW` | `1f2e05ea284e227ddc67623221d6293db4c759ce` | CRLF | +194 / −0 |
| `stack/tests/test_refav1_r5_traj_head.py` | `NEW` | `a1fb5ecc911a5d8abeaef236f64a167410c81cdb` | CRLF | +418 / −0 |
| `stack/tests/test_refav1_r3_goal_head.py` | `NEW` | `a49f30452ffb4d7ca35df196cd87e4de459ea007` | LF | +257 / −0 |
| `stack/tests/test_refav1_dvb.py` | `NEW` | `d1631f71764c892ffd0eb7e6888ead247f2a0665` | CRLF | +139 / −0 |

Base blobs read with `git rev-parse c36b6ddd:<path>`; the snapshot `C:/Users/Admin/tipsnap/c36b6ddd` was checked equal to them (CRLF-normalised) for these files before any edit. NO overlap with the R1-VMAX batch's files.

### 9.2 Staging verification
Staged with `git -C D:/Projects/TanitAD add -- <exact file path>` (68 paths, never a directory, never `-A`; nothing
committed; no other D: path touched). Verified per file: `git ls-files --stage` blob == `git hash-object` (with
the repo's filters), both 40-hex:
* **66 / 66 package files VERIFIED** (per-file record: `raw/staging_verification.json`); the 11 `code/fix` blobs are
  EQUAL to the §9.1 landing blobs; `raw/staging_verification.json` itself VERIFIED (`c176e43d…`); this `RESULT.md`
  is verified after it is staged (the blob is in the hand-off message — a file cannot carry its own hash).
* ⚠️ **One path needed `git add -f`:** `.gitignore:16` (`data/`) matches
  `P/code/fix/stack/tanitad/data/refav1_loader.py` (any directory named `data/`), so a plain `git add` SKIPPED it
  silently; it was force-added by its exact path and verified. The repo's own `stack/tanitad/data/refav1_loader.py`
  is tracked and untouched. **Landing it needs the same `-f` only if copied into another `data/` directory** — at its
  real path it is already tracked.
* Text files are stored as LF blobs (`core.autocrlf = true`); the package's working copies keep the EOL they were
  written with (CRLF for the edited shipped files, LF for most new files) — see §9.1.
