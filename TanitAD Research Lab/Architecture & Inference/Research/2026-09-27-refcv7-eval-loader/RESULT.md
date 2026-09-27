# refcv7 EVAL LOADER for the launch gate's G-EVAL

Architecture & Inference, 2026-09-27 (Berlin). Builder agent -> Master Mind (single committer). Nothing staged,
committed or pushed. Evidence classes: MEASURED (a run here, artifact named) / STATIC (file:line at a named tip) /
INFERRED / UNVERIFIED.

## 0. Headline

1. **G-EVAL ACCEPTANCE PASSES on the Thor probe's own record** (sec. 4.2; `raw/g_eval_ACCEPTANCE_thor_record/
   G-EVAL.json`, dev box CPU, `--eval-stamps` = the Thor `config.json`, md5 `0a659ec830b0df19781cdfd1b21075b8`,
   `written_for_this_argv` **true**): status **PASS**, 0 reasons; strict **0 missing / 0 unexpected** over **1,131**
   keys; `state_dict.n_differ` **0** (1,131 vs 1,131); `forward.n_differ` **0** of **188**; `map_hires_state_keys`
   **54 / 54**; negative control **1** output moved; determinism control **0**; 15 / 15 model attributes equal;
   param_breakdown equal (**100,468,987**). The loader under test is the banked file (sha256 `9e0ea3ec...`).
2. **INTEGRATION NEEDED (sec. 7), two items, both acknowledged by the Master Mind.** (a) The refcv7 profile's
   default `eval_loader` still points at the refcv6 battery loader (`launch_gate.py:152-153,188`); it must name
   `stack/tanitad/eval/refcv7_loader.py` (4-line diff in `raw/`; routed into the A19 `launch_gate.py` change).
   Until then every G-EVAL run needs `--eval-loader <tree>/stack/tanitad/eval/refcv7_loader.py`. (b) The profile
   runs G-EVAL in its **thor** stage: pass `--eval-kit /home/nvidia` there (the loader's kit layout IS Thor's
   `data/`; without it the loader refuses by name), and not the dev box's `--eval-remap-overrides`.
3. **The loader and its unit test.** `stack/tanitad/eval/refcv7_loader.py` (NEW, 897 lines) replays refcv7's
   `train()` block by block; the vendored `refcv6_loader.py` is untouched (blob `14450c78...` = the tip's). On a
   tiny refcv7 rig built through the REAL `train()` (the gate's own capture): strict 0/0 over 559 keys, 559/559
   tensors identical, 54/54 `_map_hires.*` keys, 15/15 G-EVAL attributes, 180/180 forward outputs identical; 9 / 9
   tests pass (~10 s CPU); every red arm red; an 11-arm mutation check is 10 red + control green (sec. 3).
4. The refcv6 loader lacked **seven** blocks; they are named by a code-line diff of `train()`'s build region at
   287d72e (the refcv6 loader's reference) against the tip, not by memory (sec. 2).
5. Beyond G-EVAL (MEASURED, full size): the loader's own config.json stamp checks hold **34 / 34** against the
   Thor record verbatim (4.2) and against the stamps train() builds on the dev box (4.1b); the eval split WITH
   perception targets builds on the real kit (10 cm store frac_ok 0.9856 PASS, 95.3 % agent-labelled windows,
   VIS-1 on 139 clips; 4.1c); the trainer's own eval-time loss runs through the loader's model + dataset, finite,
   with the tactical terms live (4.1d).

## 1. What was built

| file | what |
|---|---|
| `stack/tanitad/eval/refcv7_loader.py` (NEW) | the loader: `trainer()`, `build_model(config, ckpt_path, device, remap, strict) -> (model, cfg, args, rec)`, `build_eval_dataset(model, cfg, args, config, with_perception_targets=)`, `inrun_eval_perm`, `load_config` -- the interface G-EVAL calls (`launch_gate.py:4233-4460`). Env: `REFCV6_REPO` / `REFCV6_KIT` / `REFCV6_REMAP_OVERRIDES` (the gate's names); `REFCV7_*` accepted; both set and DIFFERENT refuses. |
| `stack/tests/test_refcv7_eval_loader.py` (NEW) | 9 tests: the bit-identity GREEN (trainer-built reference), 5 red arms, departures/record, RNG neutrality. |

## 2. Block-by-block map (tip `train()` line -> loader line)

Tip = 37086c3 (== b711411 for every file the loader reads: 16 files blob-compared, 0 differ). **NEW** = the refcv6
loader lacks it (from the code-only diff of the build region 287d72e -> tip, `raw/` not needed: the diff is sec. 2.1).

| train() | block | loader (final file) | refcv6 loader |
|---|---|---|---|
| 7896-7902 | arg refusals (`_check_nav_from_v7_args`, `_check_max_speed_args`, `_check_goal_point_args`, `check_effective_weights`) + `main()`'s `_ew_parser`/`_ew_argv` (11364-11365) | 331-335, `parse_args` 269-282 | had them; `_ew_*` **NEW** |
| 7908 | anchor artifact first (units onto args) | 336-337 | had it |
| 7924-7927 | seed, cuDNN | 339-343 | had it |
| 7928-7930 | `_pin_trainer_cfg` (carries NEW-1 residual prior, tac8 prior, nav compliance, speed-ceiling filter, agent A9 fields; sec. 2.0) | 344-362 | had it |
| (loader) | `agent_queries_as_trained` (A9 R4; identity for a refcv7 record) | 347-359 | **NEW** |
| (loader) | FIX-3 trunk rows as trained | 363-364, `trunk_rows_as_trained` 286-300 | had it |
| 7932-7934 | **banked tau file** verified, `cfg.nav_compliance_tau_sha256` | 365-370 | **NEW** |
| 7939-7945 | registered hier/flat config delta (refusal only) | 371-378 | skipped there; **replayed** |
| 7946-7948 | `_check_anchor_artifact_against_cfg`, lan | 379-382 | had it |
| 7950 | `RefCV3Model(cfg)` | 383-385 | had it |
| 7955-7995 | `_w_agent`/`_w_bev_aux`/`_bev_shuffle`, cls weight, `_w_map`/`_w_box3d`, lift mask, visible filter | 386-406 | had it |
| 7998 | **`model._vis1`** | 407-408 | **NEW** |
| 8005-8069 | **NEW-2 10 cm branch**: s8 tap, class weights at the declared extent, `MapHiresConfig` (extent, grad ckpt, **near lift**, **near refine blocks**, weights sha256, decision rule), branch, `HiresLiftGeometryBank` (+ C26 rows), `_w_map_hires`, `_map_hires_class_weight` (+ stamp) | 409-412 -> `build_map_hires_block` 569-610, `map_hires_config` 553-568 | **NEW** |
| 8070-8130 | **perception branch as refcv7 builds it**: also under `map_hires_pool`; `bev_source`, `planner_crop_m`, A9 `_slot_refine_kwargs`, A14 `query_select` | 413-416 -> `build_perception_block` 611-648 | **DIFFERENT** (built only behind w_map/w_box3d, none of the four) |
| 8140-8149 | conflict-detector refusal | 417-421 | **NEW** (refusal) |
| 8156-8195 | tac-goal carriers, `_w_tac_v6`, refcv7 (DrivoR-T) weights + both built-vs-weight refusals | 422-437 | had it |
| 8196-8208 | `_w_u0`, `_w_goal_point`, `_rig_camera` | 438-442 | had it |
| 8211 | ground-prior probe | 443-446 (**replayed**) | a departure there |
| 8221-8290 | anchors + train()'s two refusals (controls without the flag; no {0,0} node) | 447-468 | load only; refusals **NEW** |
| 8342 | `refuse_eval_clips_in_train` = the gate's capture point | 469-470 (comment) | -- |
| 8483-8497 | tac-goal pos_weight / mask (TRAIN split) | 471-491 (from the stamp) | had it |
| 8962 | withheld bank | 492-498 | had it |
| 8979-8982 | strict checkpoint load | 499-521 | had it (here BEFORE G-DVB, as train()'s resume path) |
| 8995 | **G-DVB on the built model** | 522-529 | **NEW** |
| (loader) | config.json stamps vs the rebuild (`map_hires`, `refcv6_perception`, `seams.nav_compliance_tau_file`, `seams.agents.queries`, `seams.trunk_equalize_bottom_rows`) | 530-531, `stamp_checks` 649-735 | **NEW** |
| 8369-8376, 8724, 8732 | eval dataset class, kwargs, providers | 757-772 | had it |
| 8734-8748 | labels, A16 clock, **G3 with train()'s exact args** | 775-792 | G3 args **DIFFERENT** (no `synthetic`/`max_unverified_frac`) |
| 8751-8816 | nav, ego history, r7 future, tac-goal targets, max-speed v6 eval sidecar | 793-812 | had it |
| 8820-8863 | agent join (train pad from the stamp) | 813-828 | had it |
| 8868-8892 | **10 cm FINE store under `--map-hires on`** (else 0.5 m), join3d | 829-851 | **DIFFERENT** (always the 0.5 m `MapGTStore`) |
| 8894-8895 | **VIS-1 on the eval split** | 852-857 | **NEW** |
| 9000 (train split) | rig-camera coverage -- here measured on the EVAL split, plus both lift banks | 858-881 | **NEW** |
| 8905-8907 | the fixed eval permutation | 887-892 | had it |

### 2.0 Which refcv7 levers arrive through `_pin_trainer_cfg` (STATIC, tip trainer lines)

| lever | where it is built | reaches the loader by |
|---|---|---|
| `--residual-prior ha0_ext_pose` (NEW-1) | `cfg.core.decoder.residual_prior`, `_pin_trainer_cfg` :500 | the pin |
| `--graft-tac8-prior`, `--graft-nav-compliance`, `--speed-ceiling-filter` (FIX-4) | `cfg.core.*`, `_pin_refcv6_tactical` :1006 / :1036 / :1044 | the pin |
| `--nav-compliance-tau-rad` | the pin (FIX-4 block); the tau FILE is read by `train()` :7932 only | the pin + block (1) |
| AGENT head: 300 queries, focal presence, prior 0.01, deep supervision, VIS-1 | `AgentSeamConfig(queries=..., **_slot_refine_kwargs(args))`, `_pin_refcv5_seams` :1234-1245 | the pin |
| BOX head: the same four + `--slot-query-select learned_ref` | `PerceptionBranchConfig` replace, `train()` :8088-8089 -- NOT the pin | block (4), `build_perception_block` |
| `--map-hires*`, `--bev-source`, `--bev-planner-crop-m` | refusals in `_pin_map_hires` :1675; BUILT in `train()` :8005-8130 | blocks (3) + (4) |
| `model._vis1` | `train()` :7998 | block (2) |

### 2.1 How the list was derived (not trusted from the brief)

`git show 287d72e:stack/scripts/refc_v3_train.py` (the refcv6 loader's line reference) vs the tip; `train()` from
its def to `build_optimizer`, comments and blank lines stripped, `diff`: the model-affecting hunks are exactly
7932-7934, 7998, 8005-8069, the perception condition/config at 8080-8089, plus the data-side G3 (8740-8748), the
fine store (8871-8879) and VIS-1 (8894-8895). Everything else refcv7 adds reaches the model through
`_pin_trainer_cfg` (sec. 2.0), which the loader calls; `_pin_refcv7` (DrivoR-T) is OFF on this argv and
`_pin_map_hires` / `_pin_slot_refine` only refuse.

## 3. Unit-test evidence (MEASURED, dev box CPU, `raw/unit_test_pytest.log`)

Rig: resnet18 (no pretrained weights), 128 x 576 (the smallest DECLARED camera geometry), smoke decoder, and the
launch argv's refcv7 blocks (map-hires on, near lift 20 m + 1 refine block at 60 x 16 m, `map_hires_pool`, box A9 +
`learned_ref`, NEW-1, tac8, nav compliance + tau FILE, speed ceiling, ego history, v6 tactical decoder, max-speed
v6, trunk levers incl. `--trunk-compile` which the loader drops); every data path a non-existent sentinel.
The reference is `launch_gate.run_trainer_until(T, argv, "model")` -- the real `train()`, as G-EVAL runs it.

| test | expectation (literal) | result |
|---|---|---|
| GREEN bit-identity | missing `[]`, unexpected `[]`, 54 hires keys both sides, 0 tensors differ, 0 attributes differ, param_breakdown equal, loader G-DVB `[]`, forward: determinism 0, trainer-vs-loader `[]`, perturbed copy moves `map_hires_logits` | PASS |
| departures/record | only `--trunk-compile` dropped; probe `{checked: False}`; near 20.0 / 1; `map_hires_pool`; `learned_ref`; tau 0.18 | PASS |
| RNG neutrality | caller's RNG unchanged by `build_model` | PASS |
| RED: vendored refcv6 loader (no 10 cm block) | `RuntimeError` starting `Error(s) in loading state_dict for RefCV3Model:` containing `Unexpected key(s) in state_dict: "_map_hires.lift.unobserved"` (the key the Master Mind's dry run measured) | RED as required |
| RED: this loader, map-hires block REMOVED | `SystemExit` ... `bev_source 'map_hires_pool' but the 10 cm branch is not built` (cannot reach the load) | RED as required |
| RED: R2/R3 fields dropped from `MapHiresConfig` (the `g_box_overfit` replay drift) | strict load: no missing, unexpected == the 7 literal near keys | RED as required |
| RED: a record whose 10 cm stamp says near lift 0 | `config.json CONTRADICTS the rebuilt model` (GREEN on the trainer-derived stamp) | RED as required |
| RED: argv names a file the kit does not carry; kit absent | refused by name; override exempt | RED as required |
| RED: `REFCV7_KIT` != `REFCV6_KIT` | refused at import | RED as required |

**Same session, the neighbouring suites** (`raw/related_suites_pytest.log`): `test_refcv7_eval_loader.py` +
`test_g_box_overfit.py` (incl. the vendored refcv6 loader's provenance pin) + `test_map_hires_wiring.py` +
`test_refcv7_box_head_wiring.py` + `test_launch_gate.py`: **245 passed**, 0 failed / 0 skipped, pytest exit 0 --
no cross-test interference (the test loads its reference trainer under a PRIVATE module name and never hides the
GPU, so a Thor suite session is not blinded by `torch.cuda.is_available()` caching).

**Mutation check of the GREEN test** (`raw/mutate.py`, `raw/mutation_results.json`, 11 arms; mutated COPIES of the
loader, the real file never touched): control GREEN; M1 `query_select` dropped -> strict load (5 unexpected
`_perception.box_refpts/box_qpos` keys); M2 hires lift bank without the C26 rows, M3 `_vis1` unset, M5 decision
rule `raw`, M6 trunk rows 0, M7 uniform class weights -> the loader's own G-DVB refuses; M4 tau sha unstamped ->
the test's assertion. With G-DVB (and for M6 the stamp check) neutralised, **M2b and M6c are caught by the FORWARD
comparison alone** (the assertion `sorted(k ... if ot.get(k) != ol.get(k)) == []`) -- the forward check sees defects that change no state_dict tensor.

## 4. G-EVAL evidence (the gate's verdict is its JSON, never the exit code)

Run: `raw/run_geval_devbox.sh` (a copy of the Master Mind's `run_dry_devbox.sh`: TREE `C:/lgt/r7ldr`, `--checks
G-EVAL`, fresh `--out-dir`, `--eval-loader` = this loader, `--eval-stamps` = the record), CPU, OMP 4.

### 4.1 STAND-IN record -- NON-BINDING (`raw/g_eval_STANDIN_nonbinding/`)

The Thor probe's `config.json` was not yet available. The stand-in (`standin_config_refcv7_intended_argv.json`,
its `_STANDIN` field says so) carries ONLY: argv = the intended 154 tokens; `seams.trunk_equalize_bottom_rows` 43
(what the post-FIX-3 trainer stamps for this argv, `refc_v3_train.py:6004`); `seams.agents.queries` 300 (parser
default; no `--agent-queries`); `tac_goal_stats` + `withheld_bank` from refcv6-r101-s0 (same `--v7-labels` file,
same negatives mode; forward-irrelevant). It exercises every model-reaching config field; it does NOT exercise the
loader's stamp checks against the real trainer's `map_hires` / `refcv6_perception` / tau stamps.

`evidence/G-EVAL.json`: **status PASS, 0 reasons**; strict_load missing `[]` / unexpected `[]`, n_keys **1131**;
state_dict n_trainer **1131**, n_loader **1131**, **n_differ 0**; map_hires_state_keys trainer **54**, loader
**54**; forward n_outputs **188**, **n_differ 0**; negative_control_outputs_moved **1**;
determinism_control_outputs_differing **0**; 15/15 model attributes same; param_breakdown equal (total
100,468,987); written_for_this_argv **true**; map-hires output shapes: logits [1, 8, 1000, 600], lift valid
[1, 400, 240], fmap_s8 [1, 512, 52, 128]; forward 146.0 s each on CPU; job 15:30-15:36 UTC (second run; the
first, 15:19-15:24, read identically). Loader sha256 in the evidence `49ca3763...`; the banked file is `9e0ea3ec...`,
and reverting ONLY its module-docstring paragraph on the pinned levers reproduces `49ca37634102e0a6` exactly
(MEASURED) -- i.e. this run executed the banked code.

### 4.1b The stamps the stand-in lacks, taken from train() ITSELF at full size (MEASURED, `raw/fullsize_stamp_check/`)

`fullsize_stamps.py` runs the real `train()` on the intended argv (CPU; the four model inputs from the kit, every
other data path a sentinel) and reads `map_hires_stamp` (8043-8069), `perception_stamp` (8105-8127) and
`_navc_tau_stamp` (7932) out of train()'s own frame at its first data call (all three present). The loader then
rebuilds from the stand-in + those three stamps: strict **0 / 0** over **1,131** keys (**54** `_map_hires.*`),
**34 / 34** stamp comparisons EQUAL (`map_hires.config`, `.trunk_tap`, `.branch_params`, `.class_weights.sha256`,
`.bev_source`, `.planner_crop_m`, `.lift_valid_mask`, 22 `refcv6_perception.*` fields, the tau file's sha256 and
tau, `seams.agents.queries`, `seams.trunk_equalize_bottom_rows`), state **1,131 / 1,131, 0 differ**, loader G-DVB
**0** mismatches over 221 registry entries; RED on the same real stamps: a record claiming `near_lift_x_m` 0 is
REFUSED. ⚠️ `seams.*` is built after the capture point (9000+), so its two values here are the stand-in's
(STATIC: what the trainer stamps for this argv, 6004 and the agent seam config).

### 4.1c The eval split WITH perception targets (MEASURED, `raw/perception_targets_check/`)

Not part of G-EVAL (which passes `with_perception_targets=False`), but the path an in-run-style eval of a refcv7
checkpoint takes: `build_eval_dataset(None, cfg, args, config, with_perception_targets=True)` on the real kit,
136.6 s. **139** episodes / **23,772** windows; 10 cm FINE store at **100 m x +-30 m** (fine [1000, 600]):
frac_ok **0.9856** over 139 clips, verdict **PASS**; agent join **22,663 / 23,772** windows labelled (95.33 %), pad
397 (the STAND-IN pad: refcv6-r101-s0's train `agent_pad`; a refcv7 record carries its own); join3d 26,394 lines /
905,512 agents / 139 clips; VIS-1 on the eval split 139 clips (sidecar 4,508 clips, sha256 `278443b3356bca05...`,
= the box-head BUILD's). A collated batch of 3 real windows carries all 12 keys the loss reads (`map_fine` [3, 1000,
600], `map_fine_label`, `map_ep`, `agent_box` [3, 397, 4], `agent_valid`, `agent_vis_full/px/known`, `tac_goal_y/w`
[3, 22], `v_max_ms`, `pose_hist` [3, 8, 4]).

### 4.1d The eval-time LOSS through the loader (MEASURED, `raw/eval_loss_check/`)

The loader's model (strict-loaded from the real train()'s state at full size) + its eval dataset with perception
targets, one real window, the trainer's own `compute_losses_v3` in eval mode: both camera banks cover **139 / 139**
eval episodes; **loss finite** (214.50), **52** scalar terms, **0** non-finite; `map_hires` 2.19 (class weights
read: [0.141, 0.206, 0.9394, 1.0741, 2.9572, 1.5712, 2.153, 0.1649]), `box3d` 79.21 with its per-layer terms
(`box3d_layer0..2`), the agent head's per-layer terms and presence, VIS-1 on (`_vis1` True). ⚠️ On that window
(the in-run eval's first) every TACTICAL term reads exactly 0.0 -- it lies outside the v7 label band, a counted
zero by design -- so the `_tac_goal_pos_weight` (22) / class mask (17 of 22) carriers were NOT exercised by it.
**Second run** (`eval_loss_check2.*`): the first window whose NOW row is 80 (the labels' t0 8.0 s) and that carries
tactical supervision (`tac_goal_w` sum 14.0), a 10 cm label and a VIS-1-known agent (found at candidate 1): loss
finite (164.20), 52 terms, 0 non-finite; tactical terms LIVE -- `tac_v6` 0.1699 (`tacv6_goal_bce` 1.5762,
`tacv6_goal_conf_bce` 0.6963, `tacv6_lat_ce` 2.0579, `tacv6_lon_ce` 1.9392), `lat_tac` 2.0879, `lon_tac` 2.0361;
`map_hires` 1.9181, `box3d` 60.3992. The goal-set BCE reads the stamped pos_weight / mask on this window.

### 4.2 ACCEPTANCE: the Thor probe's own config.json (MEASURED, `raw/g_eval_ACCEPTANCE_thor_record/`)

The record: the Master Mind's local copy of Thor's `/home/nvidia/refcv7_probe/run/config.json` (INHERITED: equal to
Thor's), md5 `0a659ec830b0df19781cdfd1b21075b8` VERIFIED here before use, sha256 `df6204f1...`; written by the real
trainer for `probe_argv.json` = the intended 154 tokens with only `--steps 110` / `--out` changed (both in the
gate's `_NON_MODEL_FLAGS`; argv compared token by token, MEASURED). Run 15:56:18-16:01:16 UTC (17:56-18:01 Berlin),
`STAMPS=<that record> OUT=C:/lgt/r7ldr_gate/out_thor1 bash raw/run_geval_devbox.sh`.

`evidence/G-EVAL.json` (the gate's verdict):

| field | value |
|---|---|
| status / reasons | **PASS** / `[]` |
| strict_load | missing `[]`, unexpected `[]`, n_keys **1131** |
| state_dict | n_trainer **1131**, n_loader **1131**, **n_differ 0**, only_trainer `[]`, only_loader `[]` |
| map_hires_state_keys | trainer **54**, loader **54** |
| forward | n_outputs **188**, **n_differ 0** |
| negative_control_outputs_moved | **1** |
| determinism_control_outputs_differing | **0** |
| model_attributes | 15 / 15 same |
| param_breakdown | equal, total **100,468,987** |
| eval_stamps.written_for_this_argv | **true** |
| eval_loader.sha256 | `9e0ea3ec0bbf58eca6ee01f5ed465f236552cb5b84d515aefaaa2593cbfd5583` (= the banked file) |
| map-hires output shapes | logits [1, 8, 1000, 600], lift valid [1, 400, 240], fmap_s8 [1, 512, 52, 128] |
| probe | window 20658 of 23,772 (t 136), seed 1234, forward 104.8 s each on CPU |
| loader_departures | `--trunk-compile` removed; 12 flags remapped to the kit (basename-checked); tac-goal constants from the record |

**The loader's own stamp checks against the same record** (`raw/thor_record_stamp_check/`; the gate does not write
them to its evidence, so measured separately: the real train() builds the reference on the dev box, the loader
rebuilds from the Thor record VERBATIM): strict **0 / 0** over 1,131 keys (54 `_map_hires.*`), **34 / 34** stamp
comparisons EQUAL (0 absent: the record carries `map_hires`, `refcv6_perception`, `seams.nav_compliance_tau_file`,
`seams.agents.queries` 300, `seams.trunk_equalize_bottom_rows` 43), state 1,131 / 1,131 with 0 differing, loader
G-DVB 0 mismatches; RED on the same record: `map_hires.near_lift_x_m` set to 0 is REFUSED.

## 5. Departures (also in `rec["departures"]`, written into G-EVAL's evidence as `loader_departures`)

* `--trunk-compile` removed (torch.compile wraps the backbone CALL; module tree and state_dict unchanged).
* Thor paths remapped to the eval kit, per flag, with the argv's basename CHECKED against the kit file.
* TRAIN split never built: `agent_pad`, tac-goal `pos_weight`/mask, withheld-bank mode come from `config.json`.
* Training instruments after the capture point are not built (sampler, P0 calibration loader, optimiser, conflict
  detector, BN recalibration). The train/eval overlap refusal (8725-8731) needs the train cache: not replayed.
* Closed vs the refcv6 loader: the ground-prior probe and the config-delta refusal are now REPLAYED (both
  RNG- and state-neutral), and G-DVB runs on the loader's own build.

## 6. UNVERIFIED / known limits

* G-EVAL was run on the DEV BOX (CPU) against the Thor record; not on Thor (off limits for this agent). On Thor the
  profile runs it with `--eval-kit /home/nvidia` (sec. 7b) -- that host path layout is INFERRED from the kit, and
  `sam3_gt_v3` is chosen when `sam3_gt_v3_eval` is absent (recorded); G-EVAL itself never reads the map root.
* `build_eval_dataset(..., with_perception_targets=True)`: exercised on the real kit (4.1c) with a STAND-IN
  `agent_pad` (397, refcv6's); no LOSS was computed on those batches here.
* Thor: not run there (off limits). The kit design assumes `--eval-kit /home/nvidia` (INFERRED from the kit
  layout; `sam3_gt_v3` is chosen when `sam3_gt_v3_eval` is absent, recorded).

## 7. INTEGRATION NEEDED (Master Mind routes; `launch_gate.py` is another agent's file)

(a) `stack/scripts/launch_gate.py`, the refcv7 profile -- `raw/INTEGRATION_launch_gate_eval_loader.diff`,
generated with `git diff --no-index` against the b711411 blob `2361732e` (new blob `9cf46046`, LF);
`git apply --check` clean and the applied file equals the intended one (EOL-normalised, MEASURED):

```diff
--- a/stack/scripts/launch_gate.py
+++ b/stack/scripts/launch_gate.py
@@ -421,6 +421,10 @@ PROFILES: dict[str, dict] = {
     #: at 10 cm, one lift, 100 m x +-30 m) + A9/A10 (the refined box head)
     "refcv7": dict(
         _REFC, name="refcv7",
+        #: G-EVAL's loader: refcv7's OWN (`stack/tanitad/eval/refcv7_loader.py`). The refcv6 battery
+        #: loader cannot build NEW-2's 10 cm branch -- MEASURED 2026-09-27: its strict load fails on
+        #: `_map_hires.lift.unobserved`. On Thor pass `--eval-kit /home/nvidia` (the kit layout).
+        eval_loader="stack/tanitad/eval/refcv7_loader.py",
         required_checks=BASE_CHECKS + ("G-SUITE-PINNED", "G-MAP-OVERFIT", "G-BOX-OVERFIT"),
```

(`_resolve_repo_rel(tree, ...)` resolves it inside the launch tree, which ships `stack/`.) ROUTED by the Master
Mind (2026-09-27 ~17:55): the profile default goes into the A19 `launch_gate.py` change the map-head agent builds on
the MAPLIFT blob (not on 2361732e); this diff is the reference for that edit. This agent did not edit
`launch_gate.py`.

(b) The thor-stage G-EVAL needs `--eval-kit /home/nvidia` (or the refcv7 profile could default it); and NOT the
dev box's `--eval-remap-overrides` (a D: path). Noted by the Master Mind.

(c) Optional hardening of G-EVAL (not required for PASS): add `_vis1` and `_map_hires_class_weight` to the
attribute list at `launch_gate.py:4361-4363`; the forward comparison covers them today.

## 8. Flags for other owners (found in passing, not fixed)

* `stack/scripts/g_box_overfit.py::TrainerAdapter._build_model` (b711411 :764-767) builds `MapHiresConfig`
  WITHOUT `near_lift_x_m` / `near_refine_blocks`: under the intended 154-token argv its replay is not the launch
  model, and its own `dvb.check(m, a)` (:814-818) will refuse it. MEASURED on the rig with exactly that replay
  (`raw/gbo_dvb_check.py`): G-DVB names **5** mismatches (`--map-hires-near-lift-m` declared 20.0 / built 0.0 and
  'not built'; `--map-hires-near-refine-blocks` declared 1 / built 0, three readers); strict, the same drift fails
  the load on the 7 near keys (sec. 3). g_box_overfit itself was NOT run. Owner: the box-head builder / gate agent.
* `taniteval/tools/refcv3_arm.py::rebuild_perception_branch` (b711411 :1101-1102) does not rebuild `query_select`:
  a `learned_ref` checkpoint rebuilds `learned` and misses 5 keys (`_perception.box_refpts.xy`,
  `_perception.box_qpos.*`; the M1 mutation's message). STATIC.

## 9. Deliverable manifest (nothing staged, committed or pushed)

Package: `D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-refcv7-eval-loader/`

| artifact | where | copies |
|---|---|---|
| `stack/tanitad/eval/refcv7_loader.py` (NEW, LF) | package `code/stack/tanitad/eval/`; working tree `C:/lgt/r7ldr/stack/tanitad/eval/` | 2 (identical bytes) |
| `stack/tests/test_refcv7_eval_loader.py` (NEW, LF) | package `code/stack/tests/`; `C:/lgt/r7ldr/stack/tests/` | 2 |
| `LANDING_READY.txt`, `RESULT.md` | package root | 1 each (SINGLE COPY until landed) |
| unit-test + related-suite logs, mutation harness + results, G-DVB replay check | package `raw/` | 1 (scratch originals in `C:/lgt/r7ldr_scratch/`) |
| G-EVAL stand-in evidence (JSON, gate/job logs, the stand-in record), runner script | package `raw/g_eval_STANDIN_nonbinding/`, `raw/run_geval_devbox.sh`; originals `C:/lgt/r7ldr_gate/` | 2 |
| full-size stamp check (script, JSON, log) | package `raw/fullsize_stamp_check/`; `C:/lgt/r7ldr_scratch/` | 2 |
| **G-EVAL ACCEPTANCE evidence** (G-EVAL.json, gate + job logs, the Thor record md5 `0a659ec8...`) | package `raw/g_eval_ACCEPTANCE_thor_record/`; `C:/lgt/r7ldr_gate/out_thor1/` + `thor_probe_config.json` | 2 |
| the loader's stamp checks vs the Thor record (script, JSON, log) | package `raw/thor_record_stamp_check/`; `C:/lgt/r7ldr_scratch/` | 2 |
| eval split with perception targets; eval-time loss (2 runs) | package `raw/perception_targets_check/`, `raw/eval_loss_check/`; `C:/lgt/r7ldr_scratch/` | 2 |
| the INTEGRATION diff for `launch_gate.py` (not applied anywhere in the repo) | package `raw/INTEGRATION_launch_gate_eval_loader.diff`; `C:/lgt/r7ldr_scratch/intdiff/` | 2 |

The working tree `C:/lgt/r7ldr` is a copy of `C:/lgt/dry37086` (untouched: the Master Mind's tip-37086c3 archive +
the box A17 overlay, INHERITED) plus the two new files. MEASURED: the 16 files the loader and its test read
(trainer, gate, map-hires, perception, refc/refc_v3, trunk, G-DVB, slot/box heads, VIS-1, g_box_overfit, the
vendored refcv6 loader) are blob-identical to the landing tip b711411 (`hash-object --no-filters`, 0 differ).
