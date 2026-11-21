# Does refcv8 carry v7F's fixes? Audit of R1, R2, R3, R4, R6 (2026-10-18)

**Tip audited:** `f7f943e` on `agent/arch-inf-20260803` (read with `git show`/`git archive`, never the E:/D: trees).
**Live argv audited:** `stack/ops/runs.d/refcv8-r101-s0.argv.json` (210 tokens, assembled 2026-10-15, commit `3b04b89`).
**Method:** per class, trace flag -> namespace -> model -> loss/forward with `file:line` at the tip, check the DEFAULT
of every flag against whether the argv names it, then a CPU probe on the real `RefCV3Model` with a control that must
read exactly 0.0 and a source mutant that must go RED. Evidence class: MEASURED (this audit, CPU, artifacts below)
unless marked INHERITED / UNVERIFIED. v7F is the v6 line (`train_v6_staged.py`, `V6Stack`); refcv8 is the refc line
(`refc_v3_train.py`, `RefCV3Model`). No class was ported by diff; each was decided from refcv8's own code.

## 0. Headline and integration escalation

**4 of 5 classes are COVERED. 1 is MISSING: R1's hard cap.** refcv8 has R1's INPUT (the fed 4-way set speed) and an
ARGMAX FILTER, but nothing clamps the EMITTED plan, so a window whose fan has no compliant candidate is emitted
uncapped. MEASURED on the refcv8 rig: with a 30 km/h ceiling the filter-only emitted plan runs **14.6-17.4 m/s
(max over waypoint intervals) against an 8.33 m/s ceiling on 4 of 4 rows whose ego is already above it**. A fix is
staged below (`--plan-vmax-clamp`, default OFF, inference-only, bit-identical when off).

**ESCALATE TO THE MASTER MIND (integration, not a doc request):**
1. Land `paths.json`'s 5 edited files + 2 new test files + the audit dir in one commit (or `git apply` the one patch in the audit dir). The flag is OFF by default, so the running refcv8 is
   untouched; the next launch gate re-binds to the new commit and argv anyway.
2. The clamp is **inference-only**, so it can be switched on **for evaluation of the running refcv8's checkpoints without
   retraining** (set `cfg.plan_vmax_clamp=True` in the eval loader; the model needs the fed ceiling). Whether the NEXT
   launch argv carries `--plan-vmax-clamp` is the PI/MM's call (it changes the emitted plan on exempt rows).
3. Owed, not done here (needs the eval dump = GPU, forbidden this session): the exempt-row RATE on real windows. The
   instrument exists (`taniteval/ceil_obey.py`, `ladder_eval.py:fan_min_vmax`); the size of the R1 gap on real data is
   UNMEASURED.

## 1. Verdict table

| class | verdict | decisive `file:line` (tip) | flag named in the live argv? | proof |
|---|---|---|---|---|
| **R1** hard speed cap | input **COVERED**; cap **MISSING** (selection filter only) | `refcv6_selection.py:216` (`keep = keep \| empty[:, None]`), `refc.py:3941-3948`, `refc_v3.py:1025-1049` (`e9_rank`) | `--max-speed-input-v6` (idx 53), `--speed-max-sidecar-v6` (54), `--speed-ceiling-filter` (97) | rig A/B above; `test_refcv8_speed_input.py:406` asserts obedience only on `picked[~exempt]` |
| **R2** nav start defects | **COVERED** (defects 1-3 cannot exist; nav reaches) | `refc_v3_train.py:5346,5514`; `refcv8_train.py:567,589,594`; `timm_trunk.py:1337-1345`; `refc_v3.py:420,1184-1190,1652-1653` | `--nav-from-v7` (36), `--r8-nav-from-v9` (184); `nav_inject` defaults True and only `--goal-point-inject` (`refc_v3_train.py:608`, absent) turns it off | 6 probes, 2 mutants |
| **R3** every tactical label family -> loss, TL colours included | **COVERED** | `refc_v3_train.py:9547-9549`, `:4694-4696`, `:6252-6277`; `refcv6_tactical.py:819` | `--tac-decoder-v6` (47), `--w-tac-v6 1.0` (48); default 0.0 would be REFUSED (`refc_v3_train.py:1196-1207`) | 3 probes, 1 mutant |
| **R4** tactical behaviours -> operative plan | **COVERED** | `refc.py:3602`, `:1847-1855`, `:5126-5135`; `refc_v3.py:2003-2030`, `:2631`; `refcv8_conditioning.py` (`allocate`, `FactorizedSelection`) | `--graft-tac8-prior` (91), `--graft-behaviour-sel` (52), `--refcv8` (154), `--r8-n-alloc 32` (159), `--r8-alloc-emit` (161), `--r8-alloc-emit-start 1000` (162) | 3 probes, 2 mutants |
| **R6** strategic OFF | **COVERED** | `refc_v3_train.py:12682`->`:577`->`refc.py:838,5144,5047`; `refc_v3.py:1606`; `refc_v3_train.py:5685,5728` | `--no-strategic` (idx **13**) | 2 probes, 1 mutant |

## 2. Evidence per class

### R1 (MISSING cap; answers the Master Mind's question 1)
* The ceiling filter is a **selection** mask, not a clamp. `SpeedCeilingFilter.forward` (`refcv6_selection.py:178-227`)
  returns `keep`; a row with no compliant candidate gets `keep | empty` = the WHOLE fan (`:216`), which the docstring calls
  "the ONLY reason the obedience bar is 99 %". `e9_rank` (`refc_v3.py:1025-1049`) repeats the rule for the final
  re-rank. The emitted plan is `fan[arange(b), idx]` (`refc_v3.py:2634`): a fan member, never modified after selection.
  So it bounds the executed plan **only on rows where a compliant candidate exists**; on the other rows nothing does.
* It is **not equivalent to v7F's cap** (`plan_speed_cap.cap_fan`, per-step accel capped, re-integrated, every candidate,
  v0 over the limit brakes and is reported). The 5.5-9.3 % anchor change (INHERITED, MM) is the filter acting on
  NON-exempt windows; it says nothing about the exempt ones.
* The telemetry cannot see the gap: `e9_ceil_dead_frac` read **0.0** in the A/B below while 4 of 8 rows had an emitted plan
  over the ceiling, because the decoder has already turned an empty row into an all-True `ceil_keep`. Use the filter's own
  `speed_rows_empty` or CEIL-OBEY.
* Rig A/B (SAME windows, SAME sampler seed, `raw/probe_numbers.json` key `R1_AB_emitted_plan_max_speed_ms`): ceilings
  `[30 x4, 120 x4] km/h`, v0 `9-13.8 m/s`. Emitted plan max speed, filter only: `14.58 15.39 16.61 17.36 | 18.10 18.41
  18.97 18.94`; with the clamp: `8.375 9.375 10.375 11.375 | 18.10 18.41 18.97 18.94` (rows 4-7 BIT-identical, ceiling not
  binding). Rows 0-3 are bounded by the braking envelope `max(ceiling, v0 - 2.5*t_mid)` because the ego is already above
  the ceiling; this is v7F's rule verbatim ("brakes at -a_max until it is under"), reported as `plan_cap_v0_over_limit`,
  never hidden. On this rig an exempt row IS an ego above its ceiling (every candidate is rolled from v0).

### R2
* Defect 1 (synthetic batch has no nav keys): N/A. Every `V3Dataset` item carries `nav_cmd`/`nav_valid`
  (`refc_v3_train.py:4601-4615,4684`); the train step hard-indexes `batch["nav_cmd"]` (`:5346`), so a missing key is a
  `KeyError` (probe: key deleted -> raises), and the preflight batch sets nav for `--nav-from-v7` (`:8633-8636`).
* Defect 2 (token overwritten by `None`): N/A. ONE hop, `batch["nav_cmd"]` (`:5346`) -> `_fwd(..., nav_cmd=nav_cmd)`
  (`:5514`). The refcv8 wrapper merges `{**kwargs, **fwd}` (`:5513`) but `r8_before_forward` writes only `r8_nav`, `r8_rc`,
  `v_max_ms`, `v_max_valid` (`refcv8_train.py:567,589,594`). Probe: a spy on the model sees `[1, 2]` == the batch.
* Defect 3 (conditioner in no optimiser group): N/A. `build_optimizer` (`refc_v3_train.py:8308-8349`) uses
  `param_groups_dd`, which puts every `requires_grad` tensor in `encoder` or `head` and raises if the counts differ
  (`timm_trunk.py:1337-1345`). Probe: group sizes sum to the trainable count; the 4+ `nav*` tensors are in a group; a stray
  parameter is in none (control). The 50,176 `nav_inject` parameters are MEASURED in the Thor smoke build's stamp.
* Reach (the F7 question): operative nav rides the measurement vector, not a zero-init port, so the selection surface
  moves at init (`anchor_logits` 4.1e-3, `sel_score_v3` 1.1e-2 under a nav swap; same-nav control exactly 0.0). The tactical
  port `nav_to_tac` is the zero-init OUTPUT projection (`refc_v3.py:1184-1190`), so `z_tac` is unmoved at step 0 (0.0) but
  the port **receives non-zero gradient at step 0** (probe), and once opened nav moves `z_tac`.

### R3 (answers the Master Mind's question 3; `test_tactical_label_reach.py` pinned something else)
* **Which line the pin is on.** `test_tactical_label_reach.py` is a 10-test STATIC pin of the label layer: `V7Label.tac_goals`
  exists, `tactical_goal_targets` projects it, `tanitad.refs.tac_goal_head` (the refc line, D-TACGOAL-1, closed 2026-09-06) is
  its consumer. It imports no trainer and no model, so it proves "a label reaches a consumer", not "the consumer reaches a
  loss in refcv8's argv". v7F's twin is `test_tactical_label_reach_v6.py` (in the unlanded `v7f_merge`).
* **Which mechanism carries the goal set in refcv8.** NOT `--w-tac-goal`/`--tac-goal-tok-head` (neither is in the argv, on
  purpose: the trainer refuses both heads together, `refc_v3_train.py:1189-1192`, and refuses `--w-tac-v6 <= 0`, `:1196-1207`).
  It is `--tac-decoder-v6 --w-tac-v6 1.0`: 22 independent sigmoids + 8 lat + 8 lon queries (`refcv6_tactical.py`). Chain:
  `--w-tac-v6 > 0` sets `ds.tac_goal_targets = True` (`refc_v3_train.py:9547-9549`) -> the item carries `tac_goal_y/w` from the
  v9 bits (`:4694-4696`; the pos_weight and class mask are re-derived from the v9 train windows, `:9653-9670`) ->
  `tactical_behaviour_losses` (`:6252`) -> `loss = loss + _w_t6 * _t6_loss` (`:6277`). The v9 constraint heads
  (`refcv8_train.py:602-646`, `--w-r8-v9-cons 0.05`) carry the speed-goal family.
* **MEASURED in a banked Thor smoke `config.json`** (`...refcv8-tactical-conditioning/pregate_2026-10-12/thor_dir/work/stamps/
  config_refcv8_smoke_measure.json`, a smoke build, not the live run): `w_tac_v6` TRAINS; **21 of 22 goal classes trainable,
  all four `TRAFFIC_LIGHT_*` among them**; only `SPEED_BAND` masked ("no positives in this split"); G-DVB mismatches 0 of 127.
* **Probe** (`compute_losses_v3`, two windows, eval mode): same batch/seed control **0.0 exactly**; TL_RED cell 0->1 moves the
  total loss by 1.0e-3, TL_GREEN by 8.7e-4, and labelling RED vs GREEN differs by 1.4e-4 (the D-TLIGHT-1 mutation, read);
  a flip on the MASKED class (`SPEED_BAND`) and on a `w=0` cell read **0.0 exactly**; lat/lon label changes move it
  (8.1e-4, 1.8e-3) and all-ignored rows move nothing (0.0). The gradient of the TL term alone (loss(TL=1) - loss(TL=0))
  lands on **34 tactical-decoder tensors (|grad| 1.52)** and on the encoder (0.74); the strategic layer reads **0.0**.
  Mutant: the tactical term dropped from the total -> RED.
* Note, not a defect: `tac_SIT` is not supervised anywhere in the refcv8 trainer (0 hits); it is outside v7F's R3 term set
  and the PI's admissibility rule allows it as an auxiliary target only.

### R4
* Routes from the tactical decoder to the plan, all in the argv: (a) `tac8_{lat,lon}_to_anchor` (zero-init Linear,
  `refc.py:1847-1855`) added to the anchor confidence (`:3602`), the image-only prior REPLACED (`:5126-5135`); (b) the
  `--graft-behaviour-sel` gate; (c) refcv8's hypothesis posterior `p(h)` drives parameter-free **allocation** of 32 extra
  candidates, `PerCandidateModulation` and `FactorizedSelection` (`refc_v3.py:2003-2030`, `refc.py:3503-3509,3885`).
  Every feed is DETACHED (`planner_feeds`), which is v7F's recommended `--tac-op-cond detached`.
* Swap probe (the tactical lat/lon logits overwritten at the source by a forward hook, so every route is exercised):
  * **before `--r8-alloc-emit-start` (step < 1000):** all ports zero-init and no extra candidate may be emitted -> the swap
    moves the emitted plan by **exactly 0.0** (the warm-start identity; the same number v7F measured at its tip, where NO
    port existed). What differs: the plan loss lands **non-zero gradient on every port** (`tac8_lat`, `tac8_lon`, `beta_lat`,
    `beta_lon`, `mod_last`): |grad| sums 217.5 / 217.5 / 2.9 / 6.2 / 67.4 at zero-init.
  * **after emission starts, ports still zero:** a swap moves the FAN on each axis; the emitted plan moves **0.31 m** for
    TURN_L->TURN_R, **0.0** for BRAKE_TO->ACCELERATE alone, **0.94 m** for both (3 windows; stated, not hidden).
  * **ports opened:** the emitted plan moves 0.93 m on each axis; control (same swap twice) **0.0 exactly**.
  Mutants: the `r8` feed severed -> RED; the tac8 port severed -> RED.

### R6 (answers the Master Mind's question 2)
* `--no-strategic` is **token 13 of the live argv** (`refcv8-r101-s0.argv.json`). Trace: argparse `refc_v3_train.py:12682` ->
  `cfg.core.no_strategic` (`:577`, the one field both files read; `refc.py:838`) -> (i) the decoder gets `ctx=None`
  (`refc.py:5144`), (ii) the route graft is off (`refc.py:5047`), (iii) the strategic goal does not FiLM the tactical state
  and the E15 goal point is off (`refc_v3.py:1606` and the hook), (iv) the route CE and the strategic-goal loss are dropped
  from the total (`refc_v3_train.py:5685,5728`, so the strategic head cannot train the shared trunk). The 8 bypassed
  modules are declared in `BYPASS_DECLARATIONS` (`refc_v3.py:966-983`) and mirrored in G-DVB (`declared_vs_built.py:885`).
  The banked smoke stamp reads `no_strategic true, route_loss_applied false, goal_str_loss_applied false`.
* Probe: every parameter of those 8 modules perturbed by N(0, 0.5): with `no_strategic` ON, `traj`, `anchor_traj`,
  `sel_score_v3`, `g_tac`, `z_tac` are **bit-identical (0.0)** while the diagnostic `ctx` moves 2.2; with it OFF the same
  perturbation moves `sel_score_v3` 0.22, `g_tac` 1.79, `z_tac` 2.79. Mutant: `ctx_dec = ctx` restored -> RED.
  Existing: `test_route_loss_respects_strategic_bypass.py`, `test_grad_unreachable_declared.py` (green at the tip).

## 3. Fix package (R1 only), all default OFF, nothing the running argv relies on changes

| file | tip blob | EOL | change |
|---|---|---|---|
| `stack/tanitad/refs/refcv6_selection.py` | `7b2dd69c2befd7a3469fb4d5551fb789ce3eea8d` | LF | `+cap_plan_to_limit` (one `dt` derivation shared with `planned_max_speed`) |
| `stack/tanitad/refs/refc_v3.py` | `68ff839c6aa47ec18a65d5da97c03a1659663511` | LF | field `plan_vmax_clamp=False`; limit export in the scene hook; clamp in the E9 block (`not self.training`); 2 build guards |
| `stack/scripts/refc_v3_train.py` | `73d4d32122535e2257c19e90d9c77d70a67ef65a` | CRLF | `--plan-vmax-clamp`, pin (refused without `--max-speed-input-v6`), config.json stamp |
| `stack/tanitad/train/declared_vs_built.py` | `12f264455a2566ad3dbadaeef026455d7cc1b04d` | CRLF | G-DVB entry `plan_vmax_clamp` (argv == built `model.cfg.plan_vmax_clamp`) |
| `stack/tests/test_declared_vs_built.py` | `4dc1be8e0ec66ff1efd889497cd2fc1157f77f9a` | CRLF | registry-count pin 278 -> 279 |
| (all of the above) | | | as one patch: `Project Steering/audits/2026-10-18-refcv8-v7f-fixes/refcv8_v7f_R1_clamp.patch` |
| `stack/tests/test_refcv8_plan_vmax_clamp.py` | (new, LF) | | 22 tests |
| `stack/tests/test_refcv8_v7f_equivalence.py` | (new, LF) | | 16 tests (R2/R3/R4/R6 probes) |

The EOLs are MIXED in this repo (measured by byte count; `grep -c $'\r'` strips the CR and matches every line), and
`scripts/_edit.py` preserves each file's own. `apply_r1.py` regenerates all five edits from the pristine tip blobs.
* Semantics: a projection of the SELECTED plan: each waypoint interval scaled to `<= max(ceiling, v0 - brake*t_mid) * dt`
  (`brake` = the decoder's `sel.accel_max`, 2.5 m/s^2); headings kept; non-binding rows returned bit-identical; `+inf` = no
  cap; NaN or non-positive limit refused. It holds on the 0.5 s WAYPOINT grid, the grid CEIL-OBEY measures on.
* The training forward is uncapped by design (the label is the ego's own realised max, as in v7F).
* A launch that passes `--plan-vmax-clamp` and no fed ceiling is REFUSED at the pin and at model build.

## 4. Test counts (CPU, one process at a time; tip tree vs patched tree)

* Existing covering tests on the **tip**: 13 files, **227 passed** (`out/logs/baseline_existing.log`).
* New tests on the patched tree: **38 passed** (22 + 16).
* Regression A (14 existing files that import the changed modules, + the new R1 file): **506 passed, 2 skipped, 1 failed**;
  the failure was the pinned registry count 278, now 279 in the patch (`test_declared_vs_built.py` 40 passed after).
* Regression B (patched tree; `test_refc*`, `test_refcv*`, `test_launch_gate_v8f/safe_directory`, `test_no_unresolved_globals`,
  `test_built_heads_receive_gradient`, `test_map_hires_wiring`, `test_max_speed*`, `test_eval_window_dump*`, `test_grad_*`,
  `test_goal_point*`, `test_g_box_overfit*` and 7 more; 18 min): **2347 passed, 15 skipped, 6 failed**. The 6 are
  `test_refcv6_diffusion.py` x3 (needs git object `cbadba58`) and `test_refcv6_no_session_paths.py` x3 (needs repo documents);
  I re-ran exactly those 7 test ids on the PRISTINE tip tree: **the same 6 fail, 1 passes** - environmental for a bare
  `git archive`, none caused by the patch.
* Regression C (the other 67 test files that import a changed module; 6 min): patched tree **1211 passed, 99 skipped, 18
  failed**; the SAME batch on the PRISTINE tip tree reads **1211 passed, 99 skipped, 18 failed with the identical failure set**
  (`diff` of the two sorted FAILED lists is empty). The 18 are environmental or order-dependent (`test_openloop_suite` x4,
  `test_p10_pose_prior` x2, `test_p4_p13_p14_wiring` x3, `test_v8f_mini` x3, `test_yaw_rate_ruling` x6; e.g. a CUDA-saved
  object deserialised on a CPU box, a missing results file). ⚠️ Eight of them (`test_v8f_mini` x2, `test_yaw_rate_ruling` x6)
  pass when their files are run alone, on either tree: the failure is a property of the combined run, not of the patch.
  Total executed on the patched tree: 167 of 167 module-importing files plus the name-pattern matches; **zero failures that
  the tip does not also have**.
* Final re-run of the two new files on the patched tree: **38 passed**.
* Flag-OFF bit-identity: refcv8 rig, eval + train forward, speed channel fed, ceiling filter ON, tip tree vs patched tree:
  **124 tensors, 0 differing, no key-set change** (`scripts/bitid_dump.py`).
* Mutation (`raw/mutation_results.json`): 13 source mutants (7 R1, 1 R3, 2 R4, 1 R6, 2 R2); **13 KILLED, 0 survived**; each
  file restored byte-for-byte (sha256 asserted). The unmutated patched tree read 38 passed before the first mutant.

## 5. UNVERIFIED / not done

* The live run uses a Thor snapshot (`/home/nvidia/refcv8_run/launch_tree/`), which I did not read (no ssh, by instruction).
  Two commits touched the audited model files after the argv commit (`3766dcd` P10 prior, `780a025` ZG-WIRE); neither
  touches the five levers, but "launch tree == tip" is UNVERIFIED. The MEASURED smoke stamp shows the same levers on an
  earlier snapshot.
* The exempt-row rate on real windows (R1 gap size) is UNMEASURED; the probe shows the mechanism, not the prevalence.
* The probes use a 2-3 window synthetic rig: they prove REACH and detect severing, not effect size.
* The live `config.json` of the running refcv8 was not read.
* All 167 test files that import a changed module (of 598 in `stack/tests`) were run on the patched tree (A + B + C); the
  other 431 import none of the four changed modules and were not run.

## 6. The remainder, for whoever lands this
* The 431 test files that import none of the changed modules were not run (shared CPU). Command for the full suite:
  `cd <tree>/stack && python -m pytest -q -p no:cacheprovider` with `taniteval/` beside `stack/`; expect the same 6
  environmental failures on a bare archive.
* `git apply --check` of `refcv8_v7f_R1_clamp.patch` on the pristine tip tree passes, and applying it reproduces all 7 files
  byte-for-byte (sha equality, CRLF kept on the three CRLF files).
* Reproduce the numbers: `python scripts/probe_numbers.py <tree_with_stack>` writes `raw/probe_numbers.json`; the mutation run is
  `python scripts/mutate.py` (restores each file and asserts its sha256).

## 7. Manifest
Everything is under `C:/Users/Admin/r8v7f/out/<repo path>`; `out/paths.json` lists `[repo path, absolute source, "raw"]`.
Audit dir (new): `Project Steering/audits/2026-10-18-refcv8-v7f-fixes/` = `AUDIT.md` (copy of this file),
`refcv8_v7f_R1_clamp.patch`, `scripts/`
(`_edit.py`, `apply_r1.py`, `mutate.py`, `probe_numbers.py`, `bitid_dump.py`, `make_paths.py`), `raw/` (`probe_numbers.json`,
`mutation_results.json`, `r1_edited_files.json`). Nothing was staged, committed, pushed, or run on Thor or the GPU.
