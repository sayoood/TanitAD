# v8F rung alpha, arm alpha-1 -- the refcv8 chassis with its residual prior swapped to the frozen POSE-ONLY P10

Author: Sonnet agent for the Master Mind, 2026-10-17 (package dated for the Thor slot, 2026-10-20). CPU only; nothing was launched,
nothing was committed, nothing was pushed. Evidence classes: **MEASURED** (ours, artifact path given) / **PUBLISHED** / **INHERITED** /
**ESTIMATED** / **HYPOTHESIS**. Tier: everything here is a build or an instrument, not a driving claim. The only ground-truth
comparisons (RESULT.md sec. 1 and 4) are **T1 (self-action OPEN loop) / T0, scored on the S1 eval139 grid**, labelled a pre-check /
diagnostic, and are not an eval result of a trained arm under the four-family rule.

**Amendment of 2026-10-17 (Master Mind D1-D5).** The first build of this hold swapped the prior to the *steer-reading* P10
(`--residual-prior p10`). The recorded steer at inference has **no PI ruling** (SPEC_REFCV7 amendment A5 excluded `ha0_ext` for exactly
this reason; NavSim has no steer channel), and alpha-0 is `ha0_ext_pose` (past poses only), so alpha-1 = `p10` would have moved TWO
things: the prior AND the input set. This build therefore adds **`p10_pose`**, computed from the past pose track only, and makes it
**alpha-1**. The steer build is kept, tested, and demoted to the **optional arm alpha-1s** (profile `v8f_steer`); the `v8f` profile
REFUSES `p10` by name ("steer at inference: no PI ruling (SPEC_REFCV7 A5)").

## 1. What alpha-1 is, and the one thing that changed

PLAN_V8F section 6, rung alpha: `alpha0` = the refcv8 recipe (rung baseline) and `alpha1` = `alpha0` with the **residual prior
swapped from `ha0_ext_pose` to a frozen learned kinematic floor** (P10) that reads **exactly the channels alpha-0's prior reads**:
the past pose track. Every other lever -- recipe, warm start, data, seed, geometry -- is the refcv8 launch's, verbatim. The pair
differs in `--residual-prior`, two new flags and `--out`, and in nothing else (`argv_alpha0_alpha1.md`, checked mechanically by
`tests/test_launch_gate_v8f.py::test_the_pair_differs_ONLY_in_the_prior_flags_and_the_run_dir`, for both variants).

The residual-prior architecture already made this a one-function swap in principle: the decoder composes **plan = P + Delta** in
control space through one function (`kinematic_prior.roll_plan`), `Delta == 0` rolls exactly to `P`, and the anchor vocabulary and
the sampler's state are residuals on `P`. What did **not** fit: every existing prior (`ha0_ext`, `ha0_ext_pose`, `cv_yawrate`) is a
**constant** `(a0, kappa0)` per window, while P10 is a **path** (30 knots at 0.2 s over 6 s). Hence the one structural change of this
build: the prior may be **per tick** `[B, 60]`.

### 1.1 The two P10 variants (same MLP, same fit recipe, different input channel)

| | `p10_pose` (**alpha-1**, profile `v8f`) | `p10` (optional **alpha-1s**, profile `v8f_steer`) |
|---|---|---|
| mode registry | `POSE_MODES` (reads the pose window only; supplied `actions` are REFUSED) | `ACTION_MODES` (needs `ego_actions`; a harness without them REFUSES) |
| `v0` | speed of the last frame from poses | same |
| `a0` | **one**-frame backward difference (`ha0_ext_pose`'s own) | two-frame backward difference |
| curvature `k` | `clip(omega0 / max(v0, POSE_KAPPA_V_FLOOR = 2.0), +-POSE_KAPPA_CAP = 0.3)`, `omega0` = wrapped one-frame yaw difference / 0.1 (`ha0_ext_pose`'s own) | recorded steer at t0, `tan(k) / 2.9` |
| `kd_x` base | `kd_x_pose`: `ha0_ext_pose` integrated as a curvature | `kd_x` on the steer unit |
| fit | **4,572 train clips**, stride 8, 41,139 rows, seed 0, 129.7 s CPU (`raw/p10_pose/p10_fit.json`) | v8F-mini: 600 clips, stride 8, 5,398 rows |
| artifact | `p10_pose_prior.pt`, 74,604 B, sha256 `e70a9ae3...693d`, `meta['units'] = 'pose_kappa'` | `p10_prior.pt`, 74,475 B, sha256 `af5a29dd...d7b3`, `meta['units'] = 'steer'` |
| open item | `V8F-RUNG-GEOMETRY` only | + `V8F-P10-ACTION-INPUT` (PI) |

The triple `(v0, a0, kappa0)` is computed in **float64** (`kinematic_prior.pose_kin3`): a float32 yaw-wrap rounds ~1e-7 rad, which at
6 s on a fast window exceeds the 1e-4 m known-value tolerance. The variant is part of the artifact (`meta['units']`) and is checked at
every seam: `load_frozen_p10(variant=)` refuses a file of the other variant **by name**, `prior_controls` refuses a module of the
other variant, and the gate's own independent load re-reads the unit from the file.

## 2. The build, file by file (all under `stack/`)

| file | change |
|---|---|
| `tanitad/models/p10_prior.py` (new) | torch port of `taniteval/floors_kin` P10: `kin3_from_window`, `kin_features`, `kd_x_base(steer_units=)`, `retime`, `FrozenP10(variant=)` (buffers only), `load_frozen_p10(path, expect_file_sha256, variant=)`, `P10_UNITS_TO_VARIANT`, `p10_fit_clip_ids`, `assert_fit_disjoint` |
| `tanitad/models/kinematic_prior.py` | modes `p10` (`ACTION_MODES`) and `p10_pose` (`POSE_MODES`), both in `RESIDUAL_PRIOR_MODES` and `TICK_MODES` (= `P10_MODES`); `pose_kin3` (float64), `knots_to_ticks`, `controls_from_path` (closed-loop inverse of `rollout_unicycle`), `p10_tick_prior` (picks the feature route by `p10.variant`), `slot_prior`; `withhold` / `prior_path` / `compose_ticks` / `absolute_controls` / `residual_controls` / `plan_check` accept a per-tick prior; **the constant-prior path is bit-for-bit unchanged** (pinned by `test_residual_prior.py`'s digest and `test_f_a_CONSTANT_prior_given_per_tick_is_the_constant_composition_bit_for_bit`); `prior_stamp(mode, p10=)` refuses a false-provenance variant |
| `tanitad/refs/refc.py` | `DecoderConfig.p10_prior_path/_sha256`; the decoder loads `FrozenP10` at construction for `mode in P10_MODES` with the matching variant (refusing a missing / wrong-sha / wrong-variant file, a path with another mode, a mode with no path); per-tick shape handling and a per-**row** lateral-prior dropout in `_residual_prior`; `_export_controls` / `_state_to_path` pass `horizons`; `_r8_extend` REFUSES the prior-free group with a per-tick prior; `_residual_prior_inputs` hands the decoder's P10 to `prior_controls` |
| `scripts/refc_v3_train.py` | `--p10-prior`, `--p10-prior-sha256`, `--residual-prior {p10, p10_pose}`; `_pin_trainer_cfg` pins them onto the decoder config (so `rebuild_config` and the eval loaders replay them); `_p10_identity` (file sha, live content sha, **variant**), `_p10_ckpt_extra`, `_resume_p10_guard`; `ckpt.pt` carries the P10 identity; a resume into another build REFUSES in both directions; the stamp is built from the file and `assert_seams_are_built` compares it to the live buffers |
| `tanitad/train/declared_vs_built.py` | `_c_p10_prior` (both modes; checks the built variant AND the artifact's declared unit), `V8F_RESIDUAL_PRIOR = "p10_pose"`, `check_refcv7_required(..., residual_prior=)` |
| `scripts/launch_gate.py` | `_v8f_profile(variant)` -> `PROFILES["v8f"]` (pose) and `PROFILES["v8f_steer"]`; per-value refusal reasons (`forbidden_value_each_why`); `p10_on`, `p10_variant`, `p10_declared`, `p10_independent_load`, `p10_live_observe` (the gate's own feature arithmetic per variant), `judge_p10_live/_ckpt/_eval`, `p10_clock_check`, `apply_p10_arm`; wired into `run_smoke`, `job_smoke` (G-LIVE, G-CKPT), `job_model` (G-DVB arms, build refusal), `eval_identity` (G-EVAL), `job_clock` (G-CLOCK) |
| `scripts/launch_gate_arms.py` | the `p10_*` arms (10 incl. `p10_pose_a0_stride2`, `p10_variant_flipped`); an arm tagged `only_residual_prior` runs only on the argv that names that prior |
| `tanitad/eval/refcv7_loader.py` | `PATH_REMAP_SPEC["--p10-prior"]` -> the tree under test's byte copy, by basename (`P10_ARTIFACTS` maps both artifacts); the `--p10-prior` default is `p10_pose_prior.pt` |
| `tanitad/configs/v8f_p10/` (new) | `p10_pose_prior.pt` + `p10_pose_fixture.npz` (565 real eval141 windows, **float64** poses, the fit script's own predictions, the three clip-id lists); `p10_prior.pt` + `p10_fixture.npz` (steer); `MANIFEST.json` |
| `tests/` | `test_p10_prior.py` (33, steer + shared), `test_p10_pose_prior.py` (23, new), `test_launch_gate_v8f.py` (56, parametrised over both variants); `test_residual_prior.py` and `test_declared_vs_built.py` updated (the `ACTION_MODES` pin; the registry size 276 -> 278) |

## 3. The conversion: 30 knots -> 60 per-tick controls (MEASURED, `raw/`)

`compose_ticks` holds each slot's residual over its span and adds the prior tick by tick, so the prior must be **controls**
`(a, kappa)` at 10 Hz, such that `rollout_unicycle` from `(0, 0, 0, v0)` reproduces P10's path. Three decisions, each measured on the
steer P10 (the conversion is variant-independent; the pose fidelity is re-measured in sec. 3.1):

1. **Closed-loop, not open-loop inversion.** `kinematic.unicycle_controls_from_path` takes no entry speed; rolled from the measured
   `v0` it leaves a *constant* speed offset for the whole horizon. `controls_from_path` tracks the rolled state, so an entry-speed
   mismatch costs one tick. MEASURED (eval139 S2, n 4,754, steer P10): `ADE6(P10u, P10)` **0.389 m open-loop vs 0.0238 m closed-loop**
   (`raw/interp_variants_final.json`).
2. **Knots -> 0.1 s ticks by 4-point interpolatory subdivision** (even ticks ARE the knots; C1; exact for a cubic; quadratic
   extrapolation at both ends), not the linear midpoints of `v8f_mini.upsample_prior`. MEASURED: the two score the same fidelity
   (0.0238 vs 0.0234 m, a 0.0004 m tie, `raw/interp_variants_final.json`), but the controls differ by **3.7x in total variation of
   accel (14.3 vs 53.2) and 1.7x in curvature (1.08 vs 1.88) per window** (`raw/smooth_variants_final.json`): a linear path has a
   heading kink at every knot, i.e. a control spike every other tick. The subdivision is kept on smoothness, not on ADE.
3. **A P10-specific curvature bound, `P10_KAPPA_CAP = 1.0` 1/m, not the pose prior's 0.3.** `POSE_KAPPA_CAP` keeps a *noisy
   pose-derived* curvature plausible; applied to a clean knot path it is the dominant conversion error. MEASURED
   (`raw/sens_cap_floor_final.json`, S2, steer P10): cap 0.3 -> **0.0469 m**; 0.5 -> 0.0346; **1.0 -> 0.0238**; 2.0 -> 0.0195;
   unbounded -> 0.0173. On the pose P10 (`raw/measure_alpha1_pose_floor.json`, S2): 0.0385 / 0.0290 / **0.0204** / 0.0163. 1.0 1/m is
   a 1 m radius: a numerical-safety bound no vehicle reaches.

`Delta == 0` rolls **exactly** to P10u (the path the controls re-roll to): `test_f_delta_zero_rolls_EXACTLY_to_the_per_tick_prior_path`
and, on the 565 real fixture windows, `max |roll_plan(0) - P10u| = 0.0 m` (inside G-CLOCK). The exported `u0_hat` /
`residual_controls` use the **slot mean** of the per-tick prior (`slot_prior`); the plan itself is always rolled with the per-tick prior.

### 3.1 Fidelity of the shipped pose P10 (MEASURED, `raw/measure_alpha1_pose_floor.json`)

`ADE6(P10u_pose, P10_pose)` -- the controls the decoder composes against the path P10 predicts, per-window mean, bar **0.05 m**:
**0.0204 m** on the S2 windows (n 4,754; episode-cluster bootstrap [0.0159, 0.0252]; median 0.0089, p95 0.0722, max 1.82; 7.6 % > 0.05)
and **0.0198 m** on the S6 windows (n 3,668; [0.0147, 0.0259]; 6.2 % > 0.05). Bar met on both; no escalation. On the 565-window
fixture the same quantity reads 0.0173 m.

## 4. Frozen by construction, identified by bytes

* **Frozen.** `FrozenP10` has no `nn.Parameter`; its seven tensors are **non-persistent buffers** (not in `state_dict`, so the
  `--init-from` warm start's strict load is untouched), the forward is a `torch.no_grad` island, and its output has no `grad_fn`.
  `grad` on a P10 tensor is therefore `None` -- exactly zero.
* **Identified by bytes, at five places.** `file_sha256` (the argv's `--p10-prior-sha256` is checked against the file's bytes
  *before* anything is built), `content_sha256` (a hash of the live tensors) and the **variant** are all: stamped into `config.json`
  (`residual_prior.p10`), saved in `ckpt.pt` (`p10_prior`), checked at resume (a mismatch in any of the three, in either direction,
  REFUSES before any state loads), re-derived by an independent second load in G-DVB, and compared trainer-vs-loader in G-EVAL.
* **Which artifact (D3, CLOSED).** `p10_pose_prior.pt` is a **refit with the `v8f_mini_fit_p10` recipe** (the same MLP 8 -> 256 -> GELU
  -> 60, inner CV over `(0, 1e-4, ...)` ridge, stride 8, seed 0) on **all 4,572 train clips** with pose-only features. Disjointness
  from eval139 and eval141 is **asserted by the fit script BEFORE it fits** and re-asserted by the tests and G-CLOCK: overlap 0 / 0 by
  clip id and 0 / 0 by stable episode id (`raw/p10_pose/p10_fit.json`, `fit_clip_ids.txt`). `manifest_md5 77dafb50...` is the
  4,572-clip manifest; `seed 0`; `best = [0, 0.0001, 59]`. The artifact and `p10_fit.json` are banked in this package
  (`raw/p10_pose/`) and in the repo at `stack/tanitad/configs/v8f_p10/`.

## 5. The clock, and why a one-tick shift fails

The pose P10 reads three numbers at the window's **last observed frame** `t0`, all from the pose track: `v[t0]`, `a0 = (v[t0] -
v[t0-1]) / 0.1`, `kappa0` (above). `pose_kin3` slices the window to `n_past` first, so a future read is impossible. The proof is a
**known value**: the in-decoder path (`pose_kin3 -> FrozenP10`) equals the fit script's own numpy predictions on **565 real eval141
windows to <= 1e-4 m** (MEASURED: max error **1.5e-5 m**, `raw/mutation_matrix_pose.json` control row; also equals an independent
numpy reference, `test_p10_pose_prior.py`), and a window that ends **one frame early** moves **98.05 %** of the windows beyond the
tolerance -- so the check *can* see a clock error. G-CLOCK re-runs both on every launch. The arms `p10_clock_shift` (features read one
frame early; the gate's own arithmetic is not patched, so the comparison sees it) and `p10_pose_a0_stride2` (a0 over two frames, the
steer P10's convention leaking into the pose build) turn it RED (up to 21.2 m / 10.9 m on 554 of 565 windows).

## 6. Reachability (the CLAUDE.md "built, tested, unreachable" rule) -- file:line per hop

Traced on the tree of this hold; each hop is observed, not inferred (`test_h_REACHABILITY_*`, `test_launch_gate_v8f.py::_live`; the
line numbers are those of the steer-only first pass and shift by the pose additions -- the hops and the tests are unchanged).

| hop | where | evidence |
|---|---|---|
| argparse | `scripts/refc_v3_train.py` `--residual-prior` (choices include `p10`, `p10_pose`), `--p10-prior`, `--p10-prior-sha256` | defaults are `None`/`off`: an arm that does not NAME them does not test the lever -- the gate's `required_flags` + `required_values` make the argv name all three |
| namespace -> config | `_pin_trainer_cfg` onto `cfg.core.decoder.{residual_prior, p10_prior_path, p10_prior_sha256}` | refuses P10 with a missing path/sha and a P10 flag with another mode |
| build | `refs/refc.py` decoder `__init__`: `self.p10 = load_frozen_p10(path, expect_file_sha256, variant=)` | sha- and variant-checked before build |
| inputs reach the model | `set_ego_window(ph, n, actions=batch["actions"])` only because `needs_actions("p10")`; for `p10_pose` NO actions are supplied (and a supplied `ego_actions` is REFUSED) | a build that does NOT read actions REFUSES a supplied `ego_actions` |
| prior | `refs/refc.py::_residual_prior_inputs` -> `kinematic_prior.prior_controls(mode in P10_MODES, p10=decoder.p10)` -> `p10_tick_prior` (-> `pose_kin3` or `kin3_from_window` by `p10.variant`) | the P10 module is CALLED on every training forward (G-LIVE counts the calls) |
| composition | `refs/refc.py::_residual_prior`, `_roll_residual_bank` / `_roll_state` -> `compose_ticks` / `roll_plan` | the plan that trains is `roll_plan(Delta, P10u)`; `Delta == 0` is P10u |
| loss | `refc_v3_train.py::compute_losses_v3` -> `losses["loss"].backward()` | **the P10 VALUE reaches the loss** (perturbing the live P10 buffers moves it by > 1e-4); **the residual heads take a non-zero gradient and P10 takes exactly none** |
| the zero control | `test_h_MUTATION_unfrozen_P10_takes_gradient_and_the_freeze_probe_goes_RED` | the same read on an *unfrozen* P10 is NON-zero, so the zero above is a reading and not a blind probe |

## 7. The gate

`PROFILES["v8f"]` = the refcv8 profile + exactly the P10 levers: `--residual-prior p10_pose`, `--p10-prior`, `--p10-prior-sha256` (a
literal, the pose artifact's), `--p10-prior` is a model-input flag (the model job builds the decoder, so it reads the file for real)
and a bound data input; `ha0_ext_pose / ha0_ext / cv_yawrate / off / p10` are refused **by name**, each with its own reason (`p10`:
"steer at inference: no PI ruling (SPEC_REFCV7 A5)"); the refcv8 *full-run geometry* literals are removed (open item
V8F-RUNG-GEOMETRY). `PROFILES["v8f_steer"]` is the same with the steer P10 (`p10`), its sha, and the extra open item
V8F-P10-ACTION-INPUT.

| check | the clause (reads a different fact than the others) | deliberate-regression arm (must go RED) |
|---|---|---|
| G-DVB | the BUILT decoder has the P10 the argv names: mode <-> module, **variant** (built and as the file declares it), config pins, file bytes now, live-buffer content vs an *independent* load, 0 Parameters, 0 requires_grad, 0 state_dict keys; `check_refcv7_required(residual_prior=)` | `p10_zero_weights`, `p10_unfrozen`, `p10_variant_flipped`, `p10_wrong_sha`, `p10_missing_file` |
| G-LIVE | the emitted per-tick prior == the gate's **independent recompute** (own 3-line feature arithmetic per variant + own read of the file) on every kept row; withheld rows carry **zero** prior and a non-zero one is a LEAK; P10 CALLED on every training forward; frozen; `config.json` names the declared P10 | `p10_fallback_ha0_ext` (falls back to `ha0_ext_pose` for the pose build), `p10_clock_shift`, `p10_unfrozen`, `p10_zero_weights`, `p10_pose_a0_stride2` |
| G-CKPT | `ckpt.pt` carries `p10_prior` equal to the declared file's and the independent read | `p10_ckpt_stamp_dropped` |
| G-EVAL | the eval loader's P10 == the trainer's == the declared file, including its variant | `p10_loader_other_p10`, `p10_variant_flipped` |
| G-CLOCK | known value <= 1e-4 m on 565 real windows; one-frame-early moves >= 95 %; `Delta = 0` == P10u; `ADE6(P10u, P10) <= 0.05 m`; fit clips disjoint | `p10_clock_shift`, `p10_pose_a0_stride2` |

The arm's regex in `launch_gate_arms.ARMS` is held to **match the guard's reason** (`test_G_*_DELIBERATE_REGRESSION_*`), otherwise
the real gate CLI would grade the arm MISCREDITED rather than CAUGHT. The matrices are raw artifacts:
`raw/mutation_matrix_pose.json` (alpha-1 as launched: **21 arm rows, all RED; 6 controls, all GREEN; 0 regex mismatches**) and
`raw/mutation_matrix_steer.json` (alpha-1s: 19 / 19 RED; 6 controls GREEN).

## 8. Decisions (D1-D5 of the Master Mind, 2026-10-17) and what is still open

* **D1 -- CLOSED BY CONSTRUCTION.** The recorded steer is not an input of alpha-1: `p10_pose` is in `POSE_MODES`, reads past poses only,
  and REFUSES supplied actions. `p10` (steer) is the optional arm **alpha-1s**, pending a PI ruling (`V8F-P10-ACTION-INPUT`, an open
  item of `v8f_steer` only). Not a blocker for alpha-1.
* **D2 -- OPEN (Master Mind, at launch).** The tiny rung's geometry stays a **placeholder**: the argv carries refcv8's full-run
  literals (`--steps 48852 --batch 16 --eval-batches 8 --eval-every 500`, B1 guard `0.21211`, which is defined on the 128-window eval
  = 8 x 16). The Master Mind sets it from refcv8's measured step time and the gate token is issued then (`V8F-RUNG-GEOMETRY`).
  Both arms must carry the same values.
* **D3 -- CLOSED.** The pose-only P10 is refit on the full 4,572-clip train view with the `v8f_mini_fit_p10` recipe, train clips only,
  disjointness from eval139/eval141 asserted before the fit (sec. 4).
* **D4 -- accepted as written; the continuation tree, not the alpha-1 argv (RULE ZERO).** `--init-from` (refcv7) stays in the pair,
  so alpha-1's residual heads and anchor vocabulary were trained as residuals on `ha0_ext_pose`; at step 0 the plan is
  `P10u + Delta_old`, shifted by `P10u - ha0_ext_pose` (RESULT sec. 4: 0.143 m ADE@2 s, 2.09 m @6 s between the two priors). And
  refcv8's `--r8-lat-prior-dropout 0.3` zeroes the prior's curvature on a row -- for P10 that drops the **whole lateral profile**,
  not a constant `kappa0`. Both are inherited by "one lever changed"; neither is attributable to the prior alone unless alpha-0
  shares them (it does, by construction). **If alpha-1 does not clear its bar, the continuation tree is, in this order:**
  (i) **cold-start the heads** (drop `--init-from`; the heads then learn residuals on P10u from the start), (ii)
  **`--r8-lat-prior-dropout 0`**, (iii) both. Each is an arm differing from alpha-1 in that one flag, and each is launched in
  the same turn that alpha-1 reads short -- a failed alpha-1 is a waypoint, not a deliverable.
* **D5 -- CLOSED.** The artifact lives in the repo package (banked, sha-pinned): `stack/tanitad/configs/v8f_p10/p10_pose_prior.pt`
  (copy: `raw/p10_pose/p10_pose_prior.pt`). The Thor checkout references it **repo-relative**
  (`.../launch_tree/stack/tanitad/configs/v8f_p10/p10_pose_prior.pt`), and the gate checks the sha256 (a literal in the profile's
  `required_values`) -- the path is the tree's, the identity is the bytes.

## 9. What this does not do

* No trained arm exists, so **no driving claim and no four-family eval of a trained model**. STRATEGIC is PAUSED (PI ITEM 25); the
  other three families need a model. RESULT.md sec. 1 reports the four families for the P10 *prior alone* (a floor, T1 open-loop).
* The arms are **wired and graded on the tiny CPU model** (through the gate's own judges and `apply_p10_arm`); they have **not** been
  run through the real `launch_gate.py run --arm` CLI, which needs the Thor data caches. The token arms
  (`launch_gate_arms.py --token-arms`) mint a synthetic PASS and therefore need the open item closed first.
* The full-suite verdict is in RESULT.md sec. 6.
