# RESULT -- HJ-UNITS (BACKLOG row HJ-UNITS; RETRACTION_LOG R46)

**2026-10-18. Agent: HJ-UNITS (Sonnet). Base: mirror tip `ac07b1a` (read via `git show`, never E:). Staged under `C:/Users/Admin/hjunits/out/`,
never committed, never pushed.** Evidence class of every number below: **MEASURED** from the log/JSON named beside it unless marked otherwise.

## 0. Headline (read this first)

1. ⭐ **The brief's premise was stale: Deliverable 1's core had ALREADY landed.** `taniteval/tools/refav1_arm.py:471` at `ac07b1a` reads
   `def paths_from_controls(controls, v0, dt, k, *, action_units: str)` -- keyword-only, no default -- from commit **`9939892`**
   ("HJ-UNITS: refav1_arm.paths_from_controls(action_units=) is now REQUIRED", 2026-10-17 16:27 +0200, a proven ancestor of `ac07b1a`, tests
   B3a-c in `stack/tests/test_steer_curvature_interface.py`). `Project Steering/BACKLOG.md:545` still lists the row as open. I did not trust the
   commit message (INHERITED): I re-read the signature, re-audited **every** live caller from source (s3), and re-ran its tests (24 passed).
2. **What was still open, and is now closed here (same class: "a control inherits the unit of the nearest default"):**
   * `floors_kin.const_control_paths` / `ha0_ext_paths` (`taniteval/taniteval/floors_kin.py:76,110`) still defaulted `action_units="kappa"` -- and
     their own main input `kin[:, 2]` IS the recorded steer angle (module docstring), so a defaulted call was wrong by construction for its main
     caller. `RESULT.md` of `2026-10-17-t2a-closedloop-prep` item 5 already named them as a standing trap. **Now required, keyword-only.**
   * `refav1_arm.run_dump` read `getattr(a, "action_units", "kappa")` (tip :707): a hand-built Namespace that forgot the unit silently integrated the
     recorded angle as a curvature (`ol` / `ha` / `ha0_ext`) -- R46 one level out. **Now refused by name** (`SystemExit ... kappa|steer, got None`).
     The CLI flag keeps its legacy default (B4-pinned, a PI/MM call -- see s7).
3. **New test file** `taniteval/tests/test_refav1_units_required.py` (49 tests): (a) omission raises, (b) bad value raises naming both legal values,
   (c) known answer vs the integrator's own closed form with a DERIVED tolerance, (d) mutation arms that go RED, (e) an AST census of every live call
   with a positive control and a mutation on a real file. **Mutation RED recorded** (s5).
4. **Frozen callers that now raise: 11 files / 19 call sites** (all `paths_from_controls`; they have raised since `9939892`). My `floors_kin`
   change adds **0** (no frozen script omits the `floors_kin` unit). Plus 3 files / 6 call sites define their OWN copy and are unaffected. List: s4.
5. **Deliverable 2 (the trunk-probe re-read) RAN, pre-registered, and the s1 leg of the claim STANDS on the correct-unit base (O-STANDS-WORSE).** The claim's numbers did
   use the wrong unit (the `kd_x` base and every readout target); the inputs are on this box (`D:/Archive/devbox-C/refav1_probe/feat_s4`). On the correct base
   `P4 trunk - P2 kin` ADE@2 s = **+0.0032 [+0.0015, +0.0050]** (banked, wrong unit, REPRODUCED to 0.0 by my kappa-mode control: +0.0042 [+0.0016, +0.0071]) -- the frozen
   trunk is still separated WORSE than kinematics alone. ⚠️ Scope: the global-mean ridge leg at one checkpoint only; **A2/A3 (the train-fitted MLP) is BLOCKED** (600-clip train
   features absent) and s2 (spatial) was not run. Three secondary banked rows DO move (s6d/s6e). Details s6.

## 1. Premise check (what the tip actually says)
| fact | where (MEASURED) |
|---|---|
| `paths_from_controls` has no default | `taniteval/tools/refav1_arm.py:471-472` at `ac07b1a`; `inspect` in `test_refav1_units_required.py::test_a2_*` |
| the landing commit | `9939892aef55...` "HJ-UNITS: ... now REQUIRED"; `git merge-base --is-ancestor 9939892 ac07b1a` -> yes |
| its own tests | `stack/tests/test_steer_curvature_interface.py` B3a/B3b/B3c: 24 passed on the pristine export (`raw/testlogs/base_stackA.log`) |
| the row is still open in the register | `Project Steering/BACKLOG.md:545` |
| what 9939892 deliberately did NOT change | its message: `refa_v1_plan.unicycle_paths`, `RefAV1.plan` and the `--action-units` CLI keep the legacy default |

## 2. What I changed (base blob at `ac07b1a` -> new blob; `git hash-object --no-filters`, LF bytes)
| repo path | base blob | new blob | change |
|---|---|---|---|
| `taniteval/taniteval/floors_kin.py` | `72b06cf0748c7b5b88cb3c6a90576de5b2ecb061` | `cc994b9f95f315f101216d36db7fae3dc66d587b` | `const_control_paths`, `ha0_ext_paths`: `action_units` REQUIRED keyword-only; `ha0_paths` states `"kappa"` explicitly (zero controls) |
| `taniteval/tools/refav1_arm.py` | `f383746492a9b2532946a60809fd5070b0ef088f` | `f0d04801c796ddd0945015f08f05f5632bbe173b` | `run_dump`: `getattr(a, "action_units", None)` (was `"kappa"`); message now says a programmatic caller must set it |
| `taniteval/tests/test_floors_kin.py` | `c68f6c231253dd039898190914a4aef912c900b9` | `1a7e80702c8daebc1ffd6af993aa6970ecd5c23c` | 3 zero-curvature calls now state `action_units="steer"` (results identical: k0 = 0) |
| `stack/tests/test_refav1_arm.py` | `933e0d1d6e43249a59193c6a3b887fef9a3eacd0` | `6d3300fdddf202bfe2d980e6b22ab0a43cad5739` | `_args` states `action_units="kappa"` (the legacy reading these tests pin); +1 test `test_run_dump_has_no_default_action_units_for_a_programmatic_caller` |
| `stack/tests/test_refav1_kin_contract.py` | `9dd09323f2b61243f124ff3fab945e240ed36466` | `30d330ba591d5879067774c8727d71dff9e9fdda` | `_args` states `action_units="kappa"` (the banked step-1,000 dumps' reading) |
| `taniteval/tests/test_refav1_units_required.py` | (new) | `bbf2aefd2faeb12da36057797cfa58e51945bced` | the 49-test file |
Diffs: `raw/diffs/*.diff`. ⚠️ Files are LF; the Windows `core.autocrlf` warning appears on `git diff`. Land with `hash-object --no-filters` / the coordinated lander.

## 3. Live-caller audit -- units chosen, with evidence (nothing assumed "kappa")
**Two producers exist, and no live caller integrates a model-head output of unknown unit.** (A) the RECORDED v2ep channel: `physicalai.py:621`
`steer = np.arctan(float(wheelbase) * curv)`; `refav1_loader.py:25-29` ("`actions[:, 0]` is a ROAD-WHEEL STEER ANGLE"); the loader serves it as
`kap = o["actions"][:, 0]` (`refav1_loader.py:273`) and `_kin_actions` returns `[a, kap[f]]` (:285); contract and `ACTION_UNITS = ("kappa", "steer")`
at `stack/tanitad/models/kinematic.py:20-55`. (B) a PLANNER candidate: curvature by construction -- `refa_v1_plan.py:352-358` ("the right unit for a
PLANNER candidate: `PlanConfig.kappa_max`, `_clip` and `GOAL_KAPPA_*` are all curvature"); the dump manifest records `action_units.planner_search = "kappa"` (`refav1_arm.py:1490-1493`).
| caller (tip file:line) | controls come from | unit stated | status |
|---|---|---|---|
| `refav1_arm.py:1018` `ol` | recorded `act[0]` (A) | `rec_units` (the run's declared reading of the recorded channel; correct run = `steer`) | explicit |
| `:1021` `ha` | `hold_action_controls` -> `kap[f]` (A) | `rec_units` | explicit |
| `:1027` `ha0` | `hold_v0_controls` = zeros | `"kappa"` (0 in either unit; test B5 / `test_b2_*`) | explicit |
| `:1032` `ha0_ext` | `hold_ext_controls` -> `kap[i0]` (A) | `rec_units` | explicit |
| `:1250` planner arms | `res.controls` of `plan()` (B) | `"kappa"` (the crossing to the model happens inside `plan(model_action_units=rec_units)`) | explicit |
| `refav1_add_floor.py:92` | `ext` built from `o["actions"][:,0]` (A) | `rec_units = manifest action_units.recorded or "kappa"` (:62) | explicit; ⚠️ legacy-manifest fallback, s7 |
| `paired_openloop.py:495` `derive_ha0` | zeros | `"kappa"` (moot) | explicit |
| `cost_surface_probe.py:505` | `plan()` result vs banked `cl` (B) | `"kappa"` | explicit |
| `refav1_kin_contract_probe.py:299` | recorded, DELIBERATELY unconverted | `"kappa"` (reproduces the legacy dumps; asserted equal) | explicit |
| `render_refav1_arms_video.py:859` | recorded (A), both readings drawn on purpose | loops `("steer","kappa")` | explicit |
| `refcv3_arm.py:1811` `integrate_select` (+ callers :2234/:2239/:2251) | recorded (A) / zeros | `units = a.action_units`, **CLI default `"steer"`** (:3740, :2035) | explicit |
| `floors_kin.py:107` `ha0_paths` | zeros | `"kappa"` -- was an implicit default, **fixed here** | explicit |
| `v8f_mini_data.py:333,607`, `v8f_mini_eval.py:191`, `v8f_mini_fit_p10.py:49` | `kin3` = recorded steer (`steer` mode) or yaw-derived curvature (`yaw` mode) | `KIN_UNITS = {"steer":"steer","yaw":"kappa"}` (`v8f_mini_data.py:259`, :586-588 refuses a mismatch) | explicit |
| `p10_prior.py:228-229` | torch port, bool `steer_units` | explicit at both calls; :274 `variant == "steer"` | separate API, s7 |
**Machine-checked:** `test_e_census_every_live_call_names_its_unit` (AST, 5 live roots) -- scanned calls are named in the failure message; positive control
`test_e2_*`; mutation on a real file `test_e3_*` (strip the keyword from `refav1_add_floor.py` -> census flags it).

## 4. Frozen callers that now raise (NOT edited -- they are the record of what ran)
AST scan of **every** `.py` at `ac07b1a` outside the live roots (`raw/frozen_scan.py` -> `raw/frozen_callers_scan.json`): 64 call sites of the four units-bearing
callees in 29 files; **25 omit the unit in 13 files.**
* **11 files / 19 call sites** call `paths_from_controls` without `action_units` and do not define it themselves -> **they raise `TypeError` when run against the
  live module** (all under `TanitAD Research Lab/`: `2026-09-04-refav1-cost-scale/tools/analyse_sweep.py` x2, `2026-09-05-refav1-ccos-eval/tools/refav1_source_ablation.py`,
  `2026-09-05-refav1-longitudinal/raw/{lon_designs,lon_designs2,lon_oracle}.py`, `2026-09-05-refav1-lonshift-t1/raw/lon_oracle.PORTABLE.py`,
  `2026-09-05-refav1-make-it-drive/tools/analyze_targeted.py`, `2026-09-05-turn-asymmetry/raw/floor_preview.py`,
  `2026-09-27-refav1-trunk-probe/code/extract.py` x6 (**the R46 culprit**), `Benchmarks & Evals/.../2026-09-03-refcv3-arm-UNVERIFIED/refcv3_arm.py:505`,
  `Data Engineering/Research/2026-09-03-steer-curvature-interface/raw/probe_interface.py` x2). Raising loudly is the correct behaviour.
* **3 files / 6 call sites** define their own copy of the callee (the `fix/` copies of `refav1_arm.py` in two packages and the UNVERIFIED `refcv3_arm.py`) and are unaffected.
* **My `floors_kin` change adds 0**: all 10 frozen `ha0_ext_paths` and 3 frozen `const_control_paths` calls already pass a unit.
* Live non-test code omitting a unit at `ac07b1a`: exactly one (`floors_kin.py:107`), fixed. Live TESTS omitting it: the 3 deliberate omission tests in
  `test_steer_curvature_interface.py` and 3 in `test_floors_kin.py` (fixed).

## 5. Tests
### 5a. The new file (`taniteval/tests/test_refav1_units_required.py`, 49 tests, 8-12 s)
(a) `test_a_*` / `test_a2_*`: omission -> `TypeError` naming `action_units`; keyword-only, no default -- for `paths_from_controls`, `const_control_paths`, `ha0_ext_paths`.
(b) `test_b_*`: `None, "", "Kappa", "STEER", "curvature", "radians", "kappa|steer", 0, ("kappa",)` -> `ValueError` whose message names **both** `kappa` and `steer`; `test_b2_*`: zero controls still must name a legal unit.
(c) `test_c_*`: constant steer integrated as `"steer"` == its equivalent curvature `tan(steer)/STEER_WHEELBASE_M` as `"kappa"` == the closed form of the integrator's own scheme
(`z_m = v dt sin(m theta/2)/sin(theta/2) e^{i(m-1)theta/2}`, theta = v kappa dt), the wheelbase imported from `kinematic.py`, never retyped.
**Tolerance DERIVED:** `tol = eps (R+1) (n + psi_max (n+2))` (running-sum roundings + curvature-conversion error carried through the heading + heading-sum roundings; 2x safety).
MEASURED error / tol = 0.010-0.020 (float32 torch) and 0.015-0.026 (float64 numpy) on four cases (`v0, steer` in {(10,.02),(10,.10),(12,-.13),(6,.05)}).
`test_c2_*` is the control: the SAME tolerance rejects the R46 error (steer fed as kappa: 5e4-8e4 x tol32) and a wrong wheelbase (3.085 m, the release's clip mean quoted in `kinematic.py`: 2.3e3-6.4e3 x tol32).
`test_c3_*`: float32 torch vs float64 numpy agree with acceleration in both units.
(d) `test_d_*` (in-process) and `test_d2_*` (the real test (a) run in a subprocess on a TEMP COPY with `= "kappa"` restored on all three signatures; the CLEAN copy is the control: 6 passed).
(e) `test_e*_`: the census above.
### 5b. Mutation RED (recorded)
* `raw/mutation_red_default_restored.log`: temp copy + restored default -> **`6 failed, 43 deselected`** (`test_a_omitted_units_raises_TypeError[...]` x3 and `test_a2_unit_parameter_is_keyword_only_with_no_default[...]` x3); the CLEAN copy -> return code 0, 6 passed.
* `raw/mutation_red_run_dump_getattr.log`: restoring `getattr(a, "action_units", "kappa")` -> the new `run_dump` test reads **`Failed: DID NOT RAISE SystemExit`** (the dump then runs to `REFAV1_DUMP_DONE` on the legacy reading); the fixed file -> `1 passed`.
* One of my own mistakes, banked: the first draft of `test_d2_*` selected itself through `-k "test_a_"` (its name ended `..._test_a_RED`), recursed and cost 92 s before failing; renamed and a `HJ_UNITS_NESTED` guard added. Class: *a name-substring selector selects the test that launches it*.
### 5c. Full-file runs, pristine export (base) vs edited (after) -- `raw/testlogs/`
Env: venv `C:/Users/Admin/venvs/tanitad` (py 3.13.5, torch 2.11 CPU via `CUDA_VISIBLE_DEVICES=-1`), `PYTHONUTF8=1`, LF tree exported from `ac07b1a`, max 2 concurrent runs. Every file run in full (no `-k`, no subset).
| group | base passed/failed/skipped | after passed/failed/skipped |
|---|---|---|
| stack A: `test_steer_curvature_interface` 24, `test_refav1_arm` 15->**16**, `test_refav1_kin_contract` 26, `test_refav1_lead_block` 9+2s, `test_refav1_narrow_console_guard` 4, `test_refav1_window_list` 17, `test_refcv3_ha0_ext_shared` 12, `test_refcv3_route_nav_alignment` 5, `test_render_refav1_arms` 24, `test_steer_conversion_complete` 14, `test_paired_openloop` 23 | 173 / 0 / 2 | **174 / 0 / 2** |
| stack B: `test_actdiv_anchored` 42, `test_argparse_help_percent` 3, `test_b1_agent_join` 19, `test_cost_ccos` 29, `test_p10_pose_prior` 23, `test_p10_prior` 33, `test_ra1_hjepa` 52, `test_refa_v1_lat_bias` 8, `test_refa_v1_oracle_seed` 7, `test_refa_v1_plan_goal` 7, `test_v8f_mini` 28, `test_refcv3_arm` 32, `test_refcv3_ablations` 28+1s | 311 / 0 / 1 (*) | **311 / 0 / 1** |
| taniteval: `test_floors_kin` 18, `test_refav1_components_yaw_mask` 11, `test_bench_navsim_reuse_floors` 6, `test_refcv3_arm_{astar_geometry 5, equalize_as_trained 3, lead_viol 5}` | 48 / 0 / 0 | 48 + **49 new** = **97 / 0 / 0** |
| extra (importers of the arm modules): `test_openloop_suite`, `test_t1_v2_adapter` 29, `test_residual_prior` 33+1s, `test_launch_closure_audit`, `test_kingate_contract` | 110 / 7 / 3 | 110 / 7 / 3 (**identical**) |
(*) The first pristine pass lacked three packages that three tests read (`2026-10-20-v8f-rung-alpha`, `2026-10-17-v8f-mini`, `2026-10-17-refav1-hj-mini`) and `CUDA_VISIBLE_DEVICES=-1` (`test_ra1_hjepa` skips otherwise): 256 / 3 / 53. I exported the packages and re-ran the three files on the PRISTINE tree (`raw/testlogs/base_pkg.log`: 23 + 28 + 52 passed) -> 311 / 0 / 1, the same as after.
The 7 failures in `extra` (4 `test_openloop_suite`, 2 `test_launch_closure_audit`, 1 `test_kingate_contract`) and the `tools/tests/test_criteria_check.py` 1 failed + 164 errors are **identical on the pristine tree**: they read repo documents / banked artifacts outside my partial export. Not caused by this change; not a clean-tree claim.

## 6. Deliverable 2 -- does "frozen DINOv3 adds nothing beyond kinematics" survive the correct-unit `kd_x` base?
### 6a. Did that claim's numbers use the wrong units? **Yes, partly -- derived from source, not assumed.**
* `2026-09-27-refav1-trunk-probe/code/extract.py:71-78` calls `ra.paths_from_controls(...)` with no unit (default `"kappa"` at the time) for `ol`, `ha`, `ha0_ext`, `ha0_6`, `ha0_ext_6`.
  `ext[1]` is `kap_ep[i0]` = the recorded STEER angle -> **`ha0_ext`, `ha0_ext_6`, `ol`, `ha` are over-rotated ~2.9x.** `ha0`/`ha0_6` are zero controls (unit moot).
* **Not touched:** the features (`model.encode(feats)`; no action enters), the GT (`g`, `g6` from poses), and `kin` (stored RAW as `(v0, a0, steer0)`) -- so the correct-unit base can be rebuilt from the stored `kin`.
* **Touched:** `P0 = kd_x` (built from `ha0_ext`: `probe.py::kdx`) and the readout TARGET `y = g - kd_x` that every learned arm P1..P5 is fitted to. R46 measured the base on this very set (2,399 windows / 141 episodes): 6 s ADE `kd_x` 3.5584 -> **2.9396 m**, `ha0_ext` 5.356 -> 2.838 m.
  Hence the quoted "P10 2.93 m vs `kd_x` 3.56 / `ha0_ext` 5.36" (RESULT s3) is the R46 number: against the correct-unit `kd_x` (2.94 m) the A3 P10 (2.9257 m, fitted on 600 TRAIN clips) leads by ~0.01 m at 6 s. ESTIMATED from two banked numbers on different fits; the like-for-like check is the re-read below.
### 6b. What can be re-run here
* **Present:** `D:/Archive/devbox-C/refav1_probe/feat_s4` (141 `ep*.npz`, 141 MB; = `C:/Users/Admin/refav1_probe` symlink), `feat_s4sp` (705 MB), banked `out/` with `probe_result.json` and `dump_P0_kdx`.
* **ABSENT -> BLOCKED:** the A2/A3 train features (`C:/Users/Admin/refav1_trainprobe/`; 5,398 windows / 600 TRAIN clips). Searched C:/Users/Admin, D:/Archive, D:/Projects, E:/ (`find -iname '*trainprobe*'` empty). Unblock: a PI-authorised Thor/pod re-encode (2,834 s GPU + a Thor read) or the folder from wherever it was last held. Without it the strongest leg of the claim (P9 vs P10, A3) cannot be re-read.
* Not run: s2 spatial ridge `P6..P8` (feat_s4sp present; Gram-PCA in 40,960 dims is beyond this budget; same `--units` switch makes it cheap to add next).
### 6c. The pre-registered s1 re-read -- what was run
`PREREG.md` (written 07:00 before any run; **Amendment A1** after a first launch that produced no result: the banked `np.linalg.svd` cost 10.4 s per 3,072-d fit, ~12 min/mode; replaced by the
equivalence-checked Gram-trick basis + a content-hashed basis cache, both modes in one process). One lever: the unit of the base. `code/probe_units_reread.py` = the banked `probe.py`
(`Readout`, `fit_select`, grids, outer 5-fold by `episode mod 5`, inner 4-fold, `default_rng(0)`, arms P0..P5, `pb` = `taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2,000, seed 0)
with the base from `floors_kin`. Inputs: 141 episodes / 2,399 windows (md5 of `ws`+`kin` `3c0eb8a200dc0dfd42ba26096ad49214`), checkpoint step 21,109. Runtime 227 s (kappa) + 105 s (steer), CPU.
**Controls, all MEASURED and all passed** (`raw/probe_units_{kappa,steer}.json -> checks, controls, C0_gram_vs_svd`):
| control | required | measured |
|---|---|---|
| C0 Gram basis == `np.linalg.svd` | min abs cos top-64 >= 1-1e-8; projector-128 diff <= 1e-6 | 0.999999999999998 / 2.9e-16 (1,428 x 3,072), 0.9999999999999984 / 6.0e-16 (1,429 x 3,072) |
| C1 kappa mode == banked `probe_result.json` | all means within 5e-5 m; P4-P2 delta/lo/hi within 5e-4 | max mean diff **2.3e-7 m**; P4-P2 delta/lo/hi diff **0.0** |
| C2 rebuild-from-kin | kappa rebuild == stored (2e-4 m); steer rebuild differs > 1 m | 9e-6 m (2 s), 9.6e-5 m (6 s); steer differs by **87.09 m** (= R46's 87.09) |
| C3 cross-package (R46 `refav1_units_check.json`) | +-1e-3 | `ha0_ext` 6 s ADE kappa **5.356479**, steer **2.838042**; `kd_x` steer **2.939635** (R46: 5.356479 / 2.838042 / 2.939635) |
| C4 base == banked `dump_P0_kdx` | 1e-5 m | 9e-6 m |
| V shuffled / constant arm not better than kin | `hi >= 0` | `P5-P2` +0.0015 [+0.0005, +0.0026]; `P1-P2` +0.0733 [+0.0543, +0.0959] -> valid |
### 6d. RESULT -- ADE (m), paired episode-cluster bootstrap over the 141 episodes (n = 2,399), banked-reproduced (wrong unit) vs correct unit
**PRIMARY `P4 trunk - P2 kin`, ADE@2 s: +0.0042 [+0.0016, +0.0071] -> +0.0032 [+0.0015, +0.0050]  => O-STANDS-WORSE** (committed outcome; `hi >= 0` so the claim stands at this leg).
| 2 s ADE pair | banked unit (reproduced) | correct unit |
|---|---|---|
| `P4 trunk - P2 kin` (primary) | +0.0042 [+0.0016, +0.0071] | **+0.0032 [+0.0015, +0.0050]** |
| `P3 raw - P2 kin` | +0.0045 [+0.0020, +0.0072] | +0.0031 [+0.0017, +0.0046] |
| `P4 trunk - P3 raw` | -0.0003 [-0.0022, +0.0017] | +0.0001 [-0.0015, +0.0017] |
| `P5 shuf - P2 kin` (validity) | +0.0022 [+0.0009, +0.0037] | +0.0015 [+0.0005, +0.0026] |
| `P2 kin - P0 kd_x` | -0.0558 [-0.0702, -0.0429] | **-0.0809 [-0.1042, -0.0608]** |
| `P4 trunk - P0 kd_x` | -0.0516 [-0.0669, -0.0376] | -0.0777 [-0.1011, -0.0578] |
Mean 2 s ADE P0..P5: banked 0.3085 / 0.2961 / 0.2527 / 0.2572 / 0.2569 / 0.2548; correct 0.2982 / 0.2906 / 0.2172 / 0.2204 / 0.2204 / 0.2188. The unit moves the 2 s `kd_x` by only 0.0103 m, but the learned arms by ~0.035 m.
**6 s -- EXPLORATORY, no decision weight (PREREG s5):**
| 6 s pair | banked unit | correct unit |
|---|---|---|
| `P4 - P2` ADE | +0.0516 [+0.0118, +0.0911] | +0.0361 [+0.0109, +0.0625] |
| `P4 - P2` FDE | +0.1275 [-0.0005, +0.2545] (n.s.) | **+0.1059 [+0.0201, +0.1951] (separated worse)** |
| `P2 kin - P0 kd_x` ADE | -0.0101 [-0.0761, +0.0447] (n.s.) | **-0.1974 [-0.3087, -0.0982] (separated better)** |
| `P4 trunk - P0 kd_x` ADE | +0.0415 [-0.0348, +0.1113] (n.s.) | -0.1613 [-0.2688, -0.0597] |
| `P4 trunk - P3 raw` ADE | **-0.0501 [-0.1027, -0.0010] (separated better)** | -0.0054 [-0.0231, +0.0135] (n.s.) |
| `P3 raw - P2 kin` ADE | +0.1017 [+0.0463, +0.1677] | +0.0414 [+0.0161, +0.0694] |
Mean 6 s ADE P0 / P2 / P4: banked 3.5584 / 3.5482 / 3.5999; correct 2.9396 / 2.7423 / 2.7783. Inner CV at 6 s: P4 lambda hit the grid ceiling (1e4) in 3 of 5 folds on the banked base; on the correct base it picked 100-316 in all five folds -- **the banked "6 s rows are underpowered by construction" caveat was an artefact of the wrong-unit base.**
### 6e. What this does and does NOT say
* **Stands (MEASURED):** at the s1 global-mean ridge leg, 2 s, one checkpoint (21,109), the frozen trunk adds nothing beyond kinematics -- it is separated slightly WORSE, with or without the unit error. The banked primary was not a unit artefact.
* **Moves (MEASURED, secondary):** (i) the 6 s `P4 - P3` "the adapter beats raw DINOv3" row (-0.0501, separated) is **not** reproduced on the correct base (-0.0054, n.s.) -- the banked RESULT s1 sentence quoting it should be corrected; (ii) the learned kinematic readout's lead over the fixed floor is **larger** on the correct base (2 s -0.0558 -> -0.0809; 6 s n.s. -> separated -0.1974): the "learned kinematic readout beats the strongest fixed floor" finding is real and unit-robust in ridge form; (iii) at 6 s the trunk arm is now separated worse than kin on FDE too.
* **NOT answered (named blockers):** **A2/A3** (`P9` vs `P10`, fitted on 5,398 windows / 600 TRAIN clips) -- the train features are absent from this box (searched C:/Users/Admin, D:/Archive, D:/Projects, E:/); unblock = a PI-authorised Thor/pod re-encode (2,834 s GPU + a Thor read) or the original `refav1_trainprobe/` folder. **s2** spatial ridge `P6..P8` -- `feat_s4sp` is present, not run (Gram PCA in 40,960 dims); cheap next step with the same `--units` switch.
  The headline claim "NO trajectory information beyond kinematics, linear or MLP, global or spatial, 2 s or 6 s" therefore has ONE leg (global ridge, 2 s) confirmed unit-robust and the MLP / spatial legs unexamined.
* **Which variance the interval answers:** another draw of EPISODES only (the ridge fit is closed-form; the shuffle is seeded). **Fold-assignment variance (episode mod 5) and a second checkpoint are NOT measured.** No lever claim is made.

## 7. Residual same-class hazards I did NOT change (each needs a decision; none is silent in the new tests)
| site | default | why I left it | what would close it |
|---|---|---|---|
| `refav1_arm.py:2671` CLI `--action-units` | `"kappa"` | deliberate legacy default, pinned by B4 ("a default that silently repaired a live eval would make every banked number incomparable"); the dump manifest records the unit and `paired_openloop` G5 flags `kappa` | **PI/MM call**: `required=True` + update B4. Blast radius: any launch line without the flag (live launchers: none found outside packages) |
| `refa_v1_plan.unicycle_paths(..., action_units="kappa")` (`:345`) | `"kappa"` | the shared integrator; correct for planner candidates; B2 pins the default; used by `refa_v1.py` internally | explicit `"kappa"` in the planner callers, then drop the default (stack/ change; needs a pod-parity check) |
| `RefAV1.plan(model_action_units="kappa")`, `imagined_goal(...)` (`refa_v1.py:2363,2397`) | `"kappa"` | the planner->model crossing; the legacy path; `test_steer_conversion_complete` pins | same family, same PI call as the CLI |
| `p10_prior.const_control_paths(..., steer_units=True)`, `kd_x_base(..., steer_units=True)` | `True` (safe direction for recorded actions) | a different API; only in-file callers, all explicit | make required (2 call sites) |
| `v8f_mini_data.kdx_base(kin3, units="steer")` (:325) | `"steer"` | safe direction; every call site passes its unit | make required |
| `refav1_add_floor.py:62`, `render_refav1_video.py:438` | `... or "kappa"` when the manifest lacks the field | a manifest without `action_units` IS a pre-2026-09-03 legacy dump, so "kappa" is its true reading | fail loudly on a post-2026-09-03 dump with no field |

## 8. Manifest (nothing is in only one place except `out/`; nothing committed or pushed)
Everything is under `C:/Users/Admin/hjunits/out/` mirroring repo paths; `paths.json` lists `[repo path, absolute source, "raw"]`.
Integration request (for the Master Mind, headline-level): **land the six repo files in s2 and the package `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-hj-units/`; update BACKLOG HJ-UNITS (drafted below).**

## 9. Drafts for the registers (I did not edit them)
**BACKLOG.md HJ-UNITS status line:**
`| HJ-UNITS | ✅ core DONE (9939892, tip ac07b1a: paths_from_controls action_units REQUIRED) + 2026-10-18 siblings closed: floors_kin.const_control_paths/ha0_ext_paths REQUIRED, run_dump programmatic default refused, taniteval/tests/test_refav1_units_required.py (49; mutation RED recorded), 11 frozen files/19 call sites now raise (listed). OPEN: CLI --action-units default (B4, PI call), unicycle_paths/plan() defaults. D2 trunk-probe re-read: see D2 row. | nothing / 0 GPU |`
**GOALS_AND_CLAIMS row (D2 ran; proposed id):**
`| D-REFAV1-TRUNK-UNITS | ✅ **THE s1 LEG OF "THE FROZEN refav1 TRUNK ADDS NOTHING BEYOND KINEMATICS" SURVIVES THE UNIT FIX; THREE SECONDARY ROWS MOVE.** MEASURED (pre-registered `PREREG.md` + A1, 2026-10-18; controls C0-C4+V all pass; kappa-mode reproduces the banked probe to 0.0): on the correct-unit `kd_x` base (`action_units="steer"`, R46), `P4 trunk - P2 kin` ADE@2 s = **+0.0032 [+0.0015, +0.0050]** (banked +0.0042 [+0.0016, +0.0071]) -> O-STANDS-WORSE; n = 2,399 / 141 episodes, `paired_episode_cluster_bootstrap` (answers "another draw of EPISODES" only; one checkpoint 21,109; fold assignment unmeasured). Moves: 6 s `P4-P3` -0.0501 (sep) -> -0.0054 (n.s.); `P2-P0` 6 s n.s. -> -0.1974 [-0.3087, -0.0982] (sep); the "6 s lambda ceiling" caveat was a wrong-unit artefact. ⛔ NOT re-read: A2/A3 (train-fitted `P9` vs `P10`, 600-clip features absent -> needs a PI-authorised Thor re-encode) and s2 spatial; the claim as a whole is therefore UNCONFIRMED beyond the global-ridge 2 s leg. T0/T1-equivalent open-loop readout (no action rollout). ARTIFACTS: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-hj-units/raw/probe_units_{kappa,steer}.json`, `PREREG.md`, `code/probe_units_reread.py` |`
