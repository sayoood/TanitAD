# RESULT -- HJ-UNITS-2 (BACKLOG row HJ-UNITS; RETRACTION_LOG R46; Master Mind decision 2026-10-18: no silent unit default anywhere in this path)

**2026-10-18. Agent: HJ-UNITS-2 (Sonnet). Base: mirror tip `5fb509a` (every repo file read via `git show`/`git archive`, never E:). Staged under
`C:/Users/Admin/hjunits2/out/`, never committed, never pushed, pod untouched.** Evidence class of every number: **MEASURED** from the log/JSON named beside it.

## 0. Headline
1. **Five signatures and one CLI flag lost their `"kappa"` default** (all keyword-only, all raise `TypeError` when omitted, all refuse an unknown value naming both `kappa` and `steer`):
   `refa_v1_plan.unicycle_paths(action_units)`, `RefAV1.plan(model_action_units)`, `RefAV1.imagined_goal(model_action_units)`, `RefAV1._imagine_tactical_goal(units)`,
   `cost_surface_probe.WindowContext.score(units)`, and `refav1_arm.py --action-units` (no default; required for every rollout).
2. **The brief's premise held for `unicycle_paths`, and for `plan()` only on the model side**: `plan()` integrates no caller-supplied control (its candidates are curvature by
   construction, `PlanConfig.kappa_max`, `refa_v1_plan.py:88`), but its `model_action_units` states the unit the *checkpoint's* action channel was trained on -- a property of the
   checkpoint that `plan` cannot see, and the default `"kappa"` hands a steer-trained predictor a curvature. That is R46 at the model boundary, so it was made required too
   (not "unit-free": proved below, s2).
3. **One decision I made that you should check:** `required=True` in argparse cannot be conditional on `--analyze-only`, and `--analyze-only` never reads the unit (it comes from the dump
   manifest). Demanding a unit there would invite a *new* mislabel (typing `kappa` for a `steer` dump). So the flag is `default=None` and the check runs after `parse_args`:
   **required for every rollout, not demanded for `--analyze-only`** (`refav1_arm.py:2881`). A rollout without it exits 2 before any heavy import.
4. **Tests:** 67 pre-existing test files that name or import an edited file: **1212 passed / 0 failed / 17 skipped before AND after, identical per file** (s5). The existing
   `test_refav1_units_required.py`: **49 -> 121 passed**, the 49 untouched. **Mutation RED recorded for every restored default** (s5b).
5. **Frozen callers that now fail: 42 files / 44 call sites** (37 CLI launch lines + 7 API calls in 5 files; none edited). List: `FROZEN_CALLERS.md`.

## 1. What became required, and the evidence for each unit
Units cited below are the values this agent put at LIVE call sites. Recorded actions are a STEER angle (`physicalai.py:621` `steer = np.arctan(float(wheelbase) * curv)`;
`refav1_loader.py:25-29`; contract `stack/tanitad/models/kinematic.py:20-55`, which says every trained refav1 / v6 / v7 checkpoint has consumed the steer channel; a planner candidate is curvature
(`refa_v1_plan.py:88` `kappa_max`; the `unicycle_paths` docstring, `refa_v1_plan.py:344-375` as edited, calls `kappa` "the right unit for a PLANNER candidate").
| surface (edited tree, line) | what the unit means there | evidence / who states it |
|---|---|---|
| `refav1_arm.py:2685` `--action-units` (`default=None`, gate `:2881`) | how a ROLLOUT reads the recorded channel | the corpus channel is a steer angle (above); `steer` is the repaired reading, `kappa` the legacy one the banked 21109 dump used (`taniteval/results/refav1-21109-openloop.json` `refav1.manifest.action_units.recorded == "kappa"`). Both legal; **the operator must pick**. `run_dump` already refused `None` from a hand-built Namespace (HJ-UNITS). |
| `refa_v1_plan.py:344` `unicycle_paths(..., *, action_units)` | unit of the controls it integrates | **NOT unit-free**: it converts with `as_curvature(controls, action_units)` then integrates. Its only live caller is `refav1_arm.py:502` (`paths_from_controls`), which forwards its own already-required argument; census scans 1 live call, 0 omit. `refa_v1.py:68` imports it but never calls it (0 calls, AST). |
| `refa_v1.py:2399/2403` `plan(..., *, model_action_units)` | unit the checkpoint's action channel 1 was TRAINED on; the model-boundary conversion `_model_actions` -> `as_command` | not unit-free on the model side (the default fed curvature to a steer-trained predictor, `kinematic.py` header); unit-free on the integration side (no control from the caller is integrated in `plan`). Live caller `refav1_arm.py:1127` passes `rec_units`; `cost_surface_probe.py:531` states `"kappa"` deliberately (convention A "as shipped" -- its control C1 compares that exact reading against the probe's own `ctx.score(units="kappa")`). |
| `refa_v1.py:2366` `imagined_goal(..., *, model_action_units)`, `:2218` `_imagine_tactical_goal(..., *, units)` | same crossing, for the goal | live internal callers pass `units=model_action_units` (`refa_v1.py` 3 sites); `cost_surface_probe.py:249` states `"kappa"` (convention A, same C1 reason). |
| `cost_surface_probe.py:258` `WindowContext.score(..., *, units)` | convention A (`kappa`, pass-through) vs B (`steer`) | all 5 live callers already passed it. ⚠️ **also fixed here:** any value other than `"kappa"` silently meant `steer` (`c if units == "kappa" else to_steer(...)`), so a typo changed the experiment; it now raises `ValueError` naming both. |

## 2. Live callers fixed (nothing assumed "kappa" without a reason)
* **Code:** `cost_surface_probe.py:249` (`_imagine_tactical_goal`), `:531` (`plan`) -> state `"kappa"` (byte-identical behaviour, reason above). No other live code called a changed signature without the unit (AST census, 5 live roots, 34 calls scanned, 0 omit).
* **Docs that invoke `refav1_arm.py` without the flag (the only live ones, `FROZEN`-scan s3):** `taniteval/tools/REFAV1_ARM.md` s6 and `taniteval/tools/RENDER_REFAV1_VIDEO.md` s7 -> `--action-units steer` (the corpus unit),
  each with a note that the banked 2026-09-04 run / delivered reel was rolled under `kappa` and that `--action-units kappa` reproduces it. No shell/ps1 launcher outside packages mentions it (126 `.sh/.ps1/.cmd/.bat` files outside packages; `git grep -l refav1_arm` over them returns 0).
* **Tests (104 call sites in 17 files, inserted mechanically, plus 3 hand-edited sites in `test_steer_curvature_interface.py`):** every `RefAV1.plan` / `imagined_goal` / `_imagine_tactical_goal` / `ctx.score` call now states `"kappa"` (the legacy reading these tests were written against); inserted by `code/inject_kw.py` (AST end-positions), then reviewed in the diffs.
  Six tests that PINNED a default were rewritten (or given a refusal check) to pin the refusal instead: `test_steer_curvature_interface.py` B2, B4, D1 (+ header), `test_steer_conversion_complete.py` A1, A2, A6 (+ header). Test counts per file are unchanged.

## 3. Frozen callers (not edited; they are the record of what ran) -- `FROZEN_CALLERS.md`
* **CLI: 37 launch lines in 37 files** (35 `.sh`, 2 `RESULT.md`) -- rollout invocations of `refav1_arm.py` with no `--action-units`; 1 of them (`2026-09-05-refav1-ccos-eval/tools/thor_chain_ccos.sh`) mentions the flag elsewhere in the file, so UNVERIFIED. Lower bound: a command built from an unresolvable variable is not counted. To reproduce a banked run add `--action-units kappa`.
* **API: 7 call sites in 5 files** (6 `plan`, 1 `unicycle_paths`; all under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-tacdec-L0-L1-E0/tools/` and `products/P4-training-pipelines/2026-09-27-v7f-refav1-state/refav1_r1r6/`); 0 define their own copy.
* Scope of the scan: mirror tip `5fb509a`, `code/frozen_scan2.py` (AST for `.py`, joined shell commands for the rest). Prior HJ-UNITS frozen list (11 files / 19 `paths_from_controls` sites) is not repeated.

## 4. Proof
### 4a. Pre-existing 49-test file stays green; extended (49 -> 121)
`taniteval/tests/test_refav1_units_required.py` (existing 49 untouched): + family (g) four more surfaces + the probe (omission -> `TypeError`; keyword-only no default; 9 bad values -> `ValueError` naming both; closed-form known answer for `unicycle_paths` under the SAME derived tolerance, with the R46 error and a wrong wheelbase rejected by it; model-boundary known answer `steer == arctan(2.9 kappa)` to 2 float32 eps with the pass-through and a wrong wheelbase rejected); (h) the CLI, behaviourally through the real `main` (exit 2 naming both values; bad value; stated value passes; `--analyze-only` passes; AST: `default=None`); (i) a CLI census over live docs/launchers.
**Census extension (e):** `_CALLEES` gains `unicycle_paths, imagined_goal, _imagine_tactical_goal, plan, score`; the generic names are claimed only where they mean the unit-bearing callee (`plan` with a `v0` keyword; `score` on `ctx` in `cost_surface_probe.py`). Controls: positive (`test_e5`: each new callee flagged when omitted, `**kw` flagged), **negative/selectivity** (`policy.plan(frames, intr, v, nav)` and another file's `ctx.score` are NOT claimed), per-callee minimum scanned counts so the census cannot go blind (`test_e4`; MEASURED live: `plan 2, _imagine_tactical_goal 4, score 5, unicycle_paths 1, paths_from_controls 11`), and an AST **mutation on six real call sites** (`test_e6`).
### 4b. Mutation: each default restored one at a time in a TEMP COPY (`code/mutation_each.py`, logs `raw/mutation_each_*.log`, `raw/mutation_each.json`)
Selection: units file `-k` primary checks (17) + `test_steer_curvature_interface.py` + `test_steer_conversion_complete.py` FULL (38). **CLEAN control: 17 + 38 passed, 0 red.**
| default restored | RED tests |
|---|---|
| CLI `--action-units default="kappa"` | `test_h_cli_rollout_without_units_is_refused_naming_both_legal_values`, `test_B4_...` |
| `unicycle_paths` | `test_g_omitted[...unicycle_paths]`, `test_g2_[...]`, `test_B2_...` |
| `plan` | `test_g_omitted[RefAV1.plan]`, `test_g2_[...]`, `test_D1_...`, `test_A1_legacy_OFF_...` |
| `imagined_goal` | `test_g_omitted[...imagined_goal]`, `test_g2_[...]`, `test_D1_...`, `test_A6_...` |
| `_imagine_tactical_goal` | `test_g_omitted[...]`, `test_g2_[...]`, `test_D1_...`, `test_A2_...` |
| `WindowContext.score` | `test_g_omitted[...score]`, `test_g2_[...]` (no pre-existing test pinned it) |
The file also carries its own arms (in-process: `test_g6`, `test_h6`; self-contained temp-copy subprocess: `test_g7` "10 passed" clean / "10 failed" mutant, `test_h7` "1 passed" / "1 failed", plus the existing `test_d2`).
### 4c. Base vs after, FULL files (no `-k`), py 3.13.5 / torch 2.11 CPU, `CUDA_VISIBLE_DEVICES=-1`, LF trees exported from `5fb509a` (`tree_base` vs `tree`)
67 files = the 51 that name an edited module (`refa_v1`, `refa_v1_plan`, `refav1_arm`, `cost_surface_probe`; the 52nd, the units file, is reported separately) + 16 importers/loaders (`refcv3_arm*`, `openloop_suite`, `v8f_mini`, `p10_*`, `ra1_hjepa`, `residual_prior`, `unicycle_action_space`, `refav1_lon_cost`, `refav1_openloop_report`, ...), list `raw/testlogs/run_list.txt`.
| | passed | failed | skipped |
|---|---|---|---|
| base, 67 files (excl. the units file) | 1212 | 0 | 17 |
| after, 67 files | **1212** | **0** | **17** (per-file identical, `raw/testlogs/*.tsv`) |
| units file | 49 | 0 | 0 |
| units file after (solo, nested runs in-process) | **121** | 0 | 0 |
`tools/tests/test_criteria_check.py` is meaningful here (182 passed, 4 skipped, before and after) because I also exported `products/P7-TanitEval/` (its registry); `test_launch_closure_audit` / `test_kingate_contract` (repo documents outside the export) were NOT run -- they are not importers of an edited file and failed identically on the pristine tree in the prior package.
Concurrency: <= 2 test processes except for ONE slip -- my first "after" launch put the nested-subprocess units file in a shard (3 processes for ~1 min); I killed that tree by explicit PID and relaunched without it.

## 5. What I did NOT change (named, each is outside the three residuals in the brief)
| site | default | note |
|---|---|---|
| `refcv3_arm.py:3740` `--action-units` (+ `integrate_select`) | `"steer"` | safe direction (recorded actions) but still silent |
| `openloop_suite.py:2271`, `render_refcv3_video.py:713` `--action-units` | `"steer"` | same |
| `actdiv_anchored.py:1384` `--action-units` | `"kappa"` | **the legacy direction**: swept level fed unchanged at the model boundary; same class as `plan()` |
| `p10_prior.py:148,222` `steer_units=True`; `v8f_mini_data.py:325` `units="steer"`; `v8f_mini_fit_p10.py:75` `kin_mode="steer"` | `True` / `"steer"` | safe direction; RESULT s7 of the prior package |
| `refc.py:1727`, `refc_sampler.py:350` `control_units="kappa"`; `refc_v3_train.py:8400` `_anchor_stamp(units="kappa")` | `"kappa"` | a different object (anchor bank controls), not audited |
| `refav1_add_floor.py:62`, `render_refav1_video.py:438` | `... or "kappa"` when the manifest lacks the field | a manifest without `action_units` IS a pre-2026-09-03 dump |
| `refav1_arm.py --analyze-only --action-units X` | flag accepted and ignored | unchanged; the dump manifest governs |
**Pod parity (not touched):** `refa_v1.py` / `refa_v1_plan.py` are `stack/` files. `refav1_arm.py` at the tip already passes `model_action_units=rec_units`, so land the `stack/` and `taniteval/` edits TOGETHER; an older pod copy of `refav1_arm.py` (before 2026-09-03) run against the new `refa_v1.py` fails loudly with `TypeError`, which is the intended behaviour.

## 6. Base blobs (`git rev-parse 5fb509a:<path>`) and new blobs (`git hash-object --no-filters`, LF)
| repo path | base blob (`5fb509a`) | new blob |
|---|---|---|
| `stack/tanitad/refs/refa_v1.py` | `d504eb9b687899500662ad8a4d447e479bb3d049` | `8763d85f757dcabdb2536fdb10a8c9a5792ca626` |
| `stack/tanitad/refs/refa_v1_plan.py` | `ba3a4a6722a1e89d8b3a4fb81917b6c38e898581` | `759395843ff68aa55cf78f798a4199397658b832` |
| `taniteval/tools/refav1_arm.py` | `f0d04801c796ddd0945015f08f05f5632bbe173b` | `43b38ef6fb743b1deaade43f49ca2e15a6b38bca` |
| `taniteval/tools/cost_surface_probe.py` | `9116b839d632c46edd7b88c29398e7624fdc87d0` | `3d6b3aafc0f22c35268308f7fa2dace6883dbe76` |
| `taniteval/tools/REFAV1_ARM.md` | `07975eca5a5f7824da5d1bcbb4f18828db661788` | `cc055dd2578dc5682b51458577044ce3413373bd` |
| `taniteval/tools/RENDER_REFAV1_VIDEO.md` | `a305fc06a53d970fe62b859475e05230be59d158` | `64c239c543cc10fab303f6b473c774c3e1e9d01f` |
| `taniteval/tests/test_refav1_units_required.py` | `bbf2aefd2faeb12da36057797cfa58e51945bced` | `210513a63e89639c33e632479fa8953b8749f40f` |
| `stack/tests/test_steer_curvature_interface.py` | `87775367ece948cd6b0cc244131f611c205f53df` | `2484f615492179d6df4b371db4c0f43e301d433b` |
| `stack/tests/test_steer_conversion_complete.py` | `f4d13f6a879f11033679176695000eb959329c72` | `02d94c76a86b9e2f53f483bd8b28a25047d5b326` |
| `stack/tests/test_cost_ccos.py` | `f038bd5913e4b478c1fdf336c17970f7ae2e631d` | `607c625f1bf795e798fa9ddd781782745f2f8f1e` |
| `stack/tests/test_cost_ccosh.py` | `992458bcf1c2192e0bcfb6b2040ebfa2a82d1b56` | `8635457bd0717bfa8181734adc6c24800aecac86` |
| `stack/tests/test_cost_chord.py` | `3ccdc0d1baf6ed0ca365e6094a6b7dc0738bd454` | `57d83e5e2eb3a7764f5186bbf62a623659707994` |
| `stack/tests/test_cost_surface_probe.py` | `b9890ad4bbef669747c66ef0953e88581342d949` | `c678b2272bdfabf307db08137319a19b6f7dcb37` |
| `stack/tests/test_refa_v1.py` | `653247fd0d35508b31e842606f523189ace1d3bc` | `2de823d76d66ffd37d97ebd8caf86e1edeee9e02` |
| `stack/tests/test_refa_v1_dk_hook.py` | `465a03f95fbcee9cede5eeffbb7fbd24aaefb09e` | `e7cf643c11674fe13b22281fd96f16d09840813b` |
| `stack/tests/test_refa_v1_goal_kappa.py` | `b9bde6c42a24ac7421833efd89fe36867d25a36d` | `1b2373141ae2495284fa991c48c440a146d0a1d7` |
| `stack/tests/test_refa_v1_goal_kappa_cost.py` | `a05925188947244184c85310c6c42e99a8e35d83` | `a5f73cf81a79e44409034bdc243e27be4ab25453` |
| `stack/tests/test_refa_v1_kamm_clip.py` | `c8ce8c37c122fba2206b792ff269aaefc1fb841c` | `737db9da138ac976b5bea8151e5c943c465c7a5e` |
| `stack/tests/test_refa_v1_lat_bias.py` | `a815d9e8e61e6383c2cfadfcb84fc888d5680d16` | `659db90ae0e7d8f3f84c5f89dd4c5c62e83acc8d` |
| `stack/tests/test_refa_v1_lon_end_to_end.py` | `bf433d8b8c360b1a136d2891a8e2a88302dce6ce` | `b5f903c3046f550fe27d641636c723a110a827db` |
| `stack/tests/test_refa_v1_plan_goal.py` | `a27ea1fa66576dfb0de86f11db3754b58fa73d65` | `85c83bb84fb968f357649897caa55a5764971bfb` |
| `stack/tests/test_refa_v1_proposal.py` | `40eb2413cb14b681d9fb2ec7ca14a50f634144a4` | `c6573b738515f54f71033a7198cac78deec15415` |
| `stack/tests/test_refa_v1_seed_ladder.py` | `352bc8ee8e46b7b8e580fb9f6ea7f2bc7438b89c` | `5117be0e9ae983304783f84614e7baad7fcdf53b` |
| `stack/tests/test_refa_v1_speed_channel.py` | `2c6fdef0201482c2ab22ea77d18acdd576c12c95` | `ccf810fc5ae46f48037632850cb3189286982a7d` |
| `stack/tests/test_seed_goal_agreement.py` | `1a013cd9bd2ebeecb167bb4f9eefeda123bd0a91` | `2e4052e5bd5534905084bca3d4ee5af608904b88` |

## 7. Manifest (all under `C:/Users/Admin/hjunits2/out/`; nothing is in only one place except this staging tree -- nothing committed, nothing pushed)
| artifact | where |
|---|---|
| 25 edited repo files (table s6) | `out/<repo path>` (`paths.json` lists `[repo path, abs src, "raw"]`) |
| package (this file, `FROZEN_CALLERS.md`, `raw/`, `code/`) | `out/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-18-hj-units-2/` |
| test logs base/after | `raw/testlogs/` |
| mutation logs | `raw/mutation_each*.{log,json}` |
| frozen scan | `raw/frozen_callers_scan2.json`, `code/frozen_scan2.py` |
| working trees | `C:/Users/Admin/hjunits2/{tree_base,tree}` (scratch; reproducible from `5fb509a`) |
**Integration request (headline for the Master Mind):** land the 25 files + the package; update BACKLOG HJ-UNITS (draft below). Land with `hash-object --no-filters` / the coordinated lander (Windows `core.autocrlf` otherwise rewrites the LF files).

## 8. Draft: BACKLOG HJ-UNITS status line (I did not edit it)
`| HJ-UNITS | ✅ DONE 2026-10-18 (9939892 + 5fb509a + HJ-UNITS-2): no silent unit default in the refav1 path -- REQUIRED, keyword-only: paths_from_controls, floors_kin.const_control_paths/ha0_ext_paths, refa_v1_plan.unicycle_paths, RefAV1.plan/imagined_goal (model_action_units), _imagine_tactical_goal (units), cost_surface_probe.WindowContext.score (+ now refuses an unknown unit); refav1_arm.py --action-units has NO default (required for every rollout, not for --analyze-only). test_refav1_units_required.py 49->121 (census over the new callees with positive/negative controls; mutation RED for each restored default); 67 test files base==after (1212/0/17). 42 frozen files / 44 sites now fail loudly (FROZEN_CALLERS.md). OPEN (named, not in scope): actdiv_anchored.py:1384 --action-units default "kappa" (legacy direction), refcv3_arm/openloop_suite/render_refcv3_video CLI default "steer", p10_prior/v8f_mini steer defaults, refc*/anchor control_units="kappa", the manifest-lacks-field "kappa" fallbacks. | nothing / 0 GPU |`
