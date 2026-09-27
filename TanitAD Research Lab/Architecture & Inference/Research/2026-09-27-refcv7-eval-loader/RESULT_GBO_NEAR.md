# G-BOX-OVERFIT harness: the model replay builds the LAUNCH model under the A18 argv

Architecture & Inference, 2026-09-27 ~18:05-18:20 Berlin, on the Master Mind's instruction (launch-critical, box
binding HELD). Nothing staged, committed or pushed. Landing: `LANDING_READY_GBO_NEAR.txt`.

## 0. Headline

1. **Fixed (1 hunk, `stack/scripts/g_box_overfit.py` 3c051db7 -> ad88b61e, LF, md5 `41759cb3565ac437a34b3bac15343c91`).**
   `TrainerAdapter._build_model` built `MapHiresConfig` without `near_lift_x_m` / `near_refine_blocks`. It now
   passes `near_lift_x_m=float(_mhr.declared_near_lift_m(a))` and `near_refine_blocks=int(_mhr.declared_near_refine_blocks(a))`,
   exactly as `train()` (refc_v3_train.py:8023-8024). No other harness line changed: the A17 schedule and the arms
   are untouched. `raw/gbo_near/g_box_overfit_near_lift.diff` applies to 3c051db7 and yields the fixed file.
   Placed on Thor by the Master Mind (verified there; 3c051db7 kept as backup).
2. **Step 2: nothing else in the replay changes the built model** (MEASURED on the rig, `raw/gbo_near/gbo_step2.*`).
   The fixed replay vs the real `train()` at the same seed: state_dict **559 / 559** keys, symmetric difference
   `[]`, **0** tensors differ in bytes; param_breakdown equal; the forward on a fixed batch **180 / 180** outputs
   identical (the gate's `_forward_out`, eval mode, seed 1234); the harness's own G-DVB `[]`; 25 of 29 `_`
   attributes equal by value, the other 4 are objects (sec. 2).
3. **Test (NEW `stack/tests/test_g_box_overfit_near_lift.py`, blob 349ec6d8): 2 / 2 pass.** The GREEN test FAILS
   against the unfixed tip blob (MEASURED); the red arm re-installs the historical defect in a copy.

## 1. The test

Reference: `launch_gate.run_trainer_until(T, argv, "model")` -- the real `train()`, captured before any data read --
on the eval-loader package's tiny rig (resnet18, 128 x 576, every refcv7 block, `--map-hires-near-lift-m 20
--map-hires-near-refine-blocks 1`). The harness model: `TrainerAdapter._build_model` on an adapter carrying exactly
the three attributes it reads, after `torch.manual_seed(0)` as `setup()` does.

| test | expectation (literals) | result |
|---|---|---|
| GREEN | `dvb_mismatches == []`; key set symmetric difference `[]`; the near keys == the 7 literal names; 54 `_map_hires.*` keys; cfg near (20.0, 1); cfg == train()'s; 0 tensors differ | PASS |
| RED: the historical replay (the 2 fix lines removed from a COPY) | its own G-DVB refuses: message starts `[gbo] the replayed build is NOT the declared launch model:` and names `--map-hires-near-lift-m: declared 20.0 but BUILT 0.0` and `--map-hires-near-refine-blocks: declared 1 but BUILT 0`; with G-DVB silenced, the model lacks EXACTLY the 7 near keys and has none extra | RED as required |
| GREEN run against the UNFIXED tip blob 3c051db7 | must fail | FAILED (G-DVB refusal) -- `raw/gbo_near/green_on_tip_RED.txt` |

Same session with the neighbours (`raw/gbo_near/pytest_gbo_near.log` + the runs below): with `test_g_box_overfit.py`
(tip 2a582320), `test_refcv7_eval_loader.py`, `test_map_hires_wiring.py`: **90 passed**. With the box builder's
PENDING TOY file (cb6812e1, run from a temporary copy, deleted afterwards): **25 passed**. No collision: the new
test is its own file.

## 2. Step 2: the harness replay vs refcv7's `train()` build blocks

| train() block | harness `_build_model` | changes the built model? |
|---|---|---|
| 7896-7902 refusals | replayed (without `main()`'s `_ew_parser/_ew_argv`: only the effective-weight audit's explicitness) | no |
| 7908 anchor artifact; 7924 seed | replayed (seed via `setup`) | no -- init bit-identical, MEASURED |
| 7928 `_pin_trainer_cfg` (NEW-1 residual prior, tac8, nav compliance, speed ceiling, agent A9 + 300 queries) | replayed | no |
| **7932-7934 tau file -> `cfg.nav_compliance_tau_sha256`** | **NOT replayed** | no: a config field read only by the config.json stamp (refc_v3_train.py:6056); not in state_dict, `_w_*` or the forward; the harness's G-DVB does not read it. LISTED, not fixed |
| 7939-7945 config-delta refusal | not replayed | no (refusal only) |
| 7946-7948 anchor/cfg check, lan | replayed | no |
| 7950-7998 model, `_w_*`, cls weight, lift mask, visible filter, `_vis1` | replayed | no |
| **8005-8069 10 cm branch** | replayed; **near lift / near refine were MISSING** | **YES -- FIXED** |
| 8070-8130 perception under `map_hires_pool`, `bev_source`, `planner_crop_m`, box A9 fields, `query_select` | replayed (`query_select` / `dn_groups` behind a field-presence check; `query_select` present at the tip) | no |
| 8140-8149 conflict refusal; 8165-8195 built-vs-weight refusals; 8221-8290 anchor refusals | not replayed | no (refusals only; the launch argv trips none) |
| 8156-8162 tac-goal carriers | set to None (train() fills pos_weight/mask from the train split later) | no: the harness trains the `box3d` term only (`loss()` / `params()` read `["box3d"]`) |
| 8196-8211 `_w_u0`, `_w_goal_point`, `_rig_camera`, ground probe | replayed; the probe is not (no-op at `--agent-w-ground` 0) | no |
| 8962 withheld bank | replayed | no |
| 8995 G-DVB | replayed (`dvb.check(m, a)`, no parser: the flag-coverage half is skipped) | no |
| (harness-only) `m._dn_gen` Generator | extra attribute | no: read nowhere at the tip (no `dn_groups` in stack/tanitad or the trainer) |

The 4 attributes not compared by value (MEASURED names): `_dn_gen` (harness-only, inert); `_lift_bank_hires`,
`_rig_camera`, `_seam` -- same classes, built by the same calls (`HiresLiftGeometryBank(...)`,
`tr._build_rig_camera(cfg, a)`, the model constructor); the lift bank's geometry is covered by the identical forward.

## 3. Deliverables (package root = the eval-loader package)

| artifact | where |
|---|---|
| `stack/scripts/g_box_overfit.py` (3c051db7 -> ad88b61e, LF) | `code/stack/scripts/`; `C:/lgt/r7ldr/stack/scripts/`; on Thor (the Master Mind) |
| `stack/tests/test_g_box_overfit_near_lift.py` (NEW, 349ec6d8, LF) | `code/stack/tests/`; `C:/lgt/r7ldr/stack/tests/` |
| `LANDING_READY_GBO_NEAR.txt`, this file | package root (single copies until landed) |
| diff, test logs, the green-on-tip red run, the step-2 measurement | `raw/gbo_near/` |
