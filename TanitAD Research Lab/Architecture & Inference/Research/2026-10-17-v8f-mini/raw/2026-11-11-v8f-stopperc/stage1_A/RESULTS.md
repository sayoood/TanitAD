# RESULTS: arm A (stop-location auxiliary), code + CPU tests, 2026-11-12

Base: `715ef8fcdf439fedfb42b5b495c28a6f1fc68ccf` (tip of `agent/v8f-stopbal-20261110`). Nothing committed, pushed, staged or run on a GPU. Python `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, `CUDA_VISIBLE_DEVICES=-1`, `tanitad.__file__` asserted to be the scratch export in every run. Card: `PREREG_V8F_STOPPERC.md` §3, §5 (arm A, NULL reuse rule, Launch), §8 (Q1, Q4).

## Headline for the Master Mind (integration and one data dependency)

1. **Integration:** `land_stopaux.sh` (NOT RUN) lands a new branch `agent/v8f-stopaux-20261112` as a child of 715ef8fc. It rebuilds the two patched files from the base blobs with the diff, and pins their sha256.
2. **Named data dependency, UNVERIFIED:** the trainer gets the camera pose from a per-clip `sensor_extrinsics` table (`--stop-aux-extr`). It never reached the trainer on the base (TRACE.md §4). The repo's table covers the **2,400 parity clips**; whether it covers the **4,512 train clips** of v8F-mini is not established. The launch gate **refuses below 99 % coverage**. If short, `build_rig_extrinsics_table.py` must extend the table first. This blocks arm A's launch, not its code.
3. **Probe risk:** the G-LIVE probe needs at least one in-view rest point in its one probe batch. The rate of such windows on the real corpus is unmeasured here. If the real probe batch has none, the gate refuses with a named message (`... cannot see the cell term`).

## What was built (priority order 1-5)

| step | outcome | evidence |
|---|---|---|
| 1 TRACE | done, `TRACE.md`. All hops exist except `sensor_extrinsics` in the trainer (built here, per-clip table, see above) | MEASURED, read from source |
| 2 implement | `--stop-aux {off,on}` (default off), `--w-stop-aux`, `--stop-aux-extr`; model head, label geometry, trainer stage, G-DVB clause, G-LIVE probe, W script | diff `stopaux_vs_715ef8fc.diff` (479 lines, 275 added, 7 removed) |
| 3 tests | `test_v8f_stopaux.py`: 38 tests, all pass on the patched tree | MEASURED, below |
| 4 gate list | below | |
| 5 package | this folder | |

Files: model `stack/tanitad/models/v8f_mini.py` (head built LAST and only when on; computed only in the TRAINING forward; one total line); trainer `stack/scripts/v8f_mini_train.py`; NEW `stack/tanitad/data/v8f_stopaux.py` (cylindrical projection to a 4x10 cell, per-clip camera arrays, label masks); NEW `stack/scripts/stop_aux_w_from_smoke.py`; NEW `stack/tests/test_v8f_stopaux.py`.
The patched `v8f_mini.py` (102,690 B) and `v8f_mini_train.py` (262,223 B) are not reproduced here: the diff plus the base blobs rebuild them, and `land_stopaux.sh` checks the result against `af428fb2...fde` and `066b1131...140b`.

## The OFF path is the base, bit for bit (card §5 NULL reuse rule)

* **MEASURED, 2026-11-12:** the same synthetic 4-step training (`--seed 5 --arm M2b --kin-mode yaw --d 32 --n-q 2 --steps 4`, identical argv text, separate cwd) run with the **base** trainer and the **patched** trainer with `--stop-aux off` gives the **same `ckpt_last.pt` sha256 `58c085e425b6f04033e12c9714de8d440fc94c8819763e91da3995597c009a2a`**. Their `train_log.jsonl` rows are equal except `sec`, and their `selfcheck.json` key sets are equal (top level and `levers`).
* Why the sha can match: the checkpoint's pickled `cfg` is written WITHOUT `stop_aux` / `w_stop_aux` when off (`save()`: one filtered dict). Without that filter the two new dataclass fields would have changed the file bytes with identical weights.
* Unit level (`test_off_equals_base_bit_for_bit`, base module read from the mirror and pinned by sha256 `41cadb40...a21f1`): same parameter list, state dict, every loss term, every gradient, `forward_infer`, `built_levers` keys, trainable groups.
* ON adds exactly the 8 head tensors (`stop_head.*`, `stop_pres_head.*`, 132,098 parameters at d = 256) and leaves every other parameter's init equal to the base's (the head is built last).

## Tests (card §5 Launch; priority 3 a-e)

New file, patched tree: `38 passed` (`pytest stack/tests/test_v8f_stopaux.py -q`, 23-46 s).
(a) off = base: `test_off_equals_base_bit_for_bit`, `test_off_checkpoint_cfg_keeps_the_base_key_set`, `test_end_to_end_off_is_the_default_run`.
(b) on: `test_b_term_gradients_and_zero_controls`: finite term, non-zero gradient on `stop_head.*`, `stop_pres_head.*` AND `adapter.*`, and the control **at w = 0 reads EXACTLY 0.0** on the heads (the off model has no head parameters at all). A hand computation of both BCE parts matches to 1e-5. `test_masks_select_the_rows`, `test_inference_never_reads_the_head`.
(c) mutation: **RED shown.** Deleting the one total line turned both checks red. MEASURED run in a scratch copy with that line removed (line 1105 of the patched model):
`FAILED test_b_term_gradients_and_zero_controls` and `FAILED test_end_to_end_on_trains_with_the_term_and_writes_the_record`, the latter through the trainer's own G-LIVE: `AssertionError: G-LIVE failed: ... "unexpected_dead": ["stop_head.0.bias", ..., "stop_pres_head.2.weight"] ... "pass": false`. In-tree the same mutation is exercised by `test_c_mutation_that_deletes_the_term_from_the_total_goes_red` (the term is still logged, the total does not move, `stop_aux_probe` raises) and a second mutation (head reads a detached token) by `test_c_mutation_that_cuts_the_head_from_the_adapter_goes_red` (adapter check red).
(d) vision-only guard, the form of `assert_lead_head_vision_only`: `M.assert_stop_head_vision_only` (static scan of `_stop`, runtime key guard allowing only `frame_ok`, bit-identity under a replacement of every non-vision key, sensitivity to the features). Red arms in `test_d_red_arms`: no head, a head naming `nav` (static), a head reading `p10f` through a computed key (evades the static scan, the runtime guard catches it), a head that ignores the features, a head that moves with non-vision state.
(e) projection, hand-computed (nominal camera at x 2.0 m, z 1.5 m): straight ahead 10 m = cell 24 (row 2, col 4); 30 deg left = 22; 30 deg right = 27; 4 m ahead = 34 (row 3); 55 deg left / right = 20 / 29; behind the camera, 70 deg (outside the 120 deg field), 1 m ahead (below the frame) and NaN = -1 (presence 0); a 1.2 m vs 1.67 m mount changes the row cell (24 vs 34 at 7 m); the pinhole column would be outside the frame at 55 deg, the module's is not; parity with `RigCamera.project` (25 points, a real clip's mount) and with `FrontWideExtrinsics.rotation_cam_to_vehicle` (12 random quaternions, max diff < 1e-12).
Also: gate refusals (`test_gate_refuses`, thin coverage, wrong file, 8x20 grid), every new flag has its `FLAG_CHECKS` entry, `--init-ckpt` leaves fresh only the declared keys (and an OFF model still loads strictly, verbatim), the W script, and end to end on the trainer's synthetic world (on trains with the term at every step and writes `stop_aux_probe`; thin coverage is refused; a probe with no in-view rest point is refused).

### Existing v8f suites that import the touched modules, base vs patched

**MEASURED:** all 44 files that import `v8f_mini`, `v8f_mini_train` or `launch_gate` (43 by import grep, plus `test_launch_gate_v8f.py`), one pytest per file, on the base export and on the patched export, same machine, CPU. **Totals: base 1220 passed / 178 failed / 1147 skipped; patched 1220 passed / 178 failed / 1147 skipped. The per-file summaries are identical on both sides (diffed, timing removed).** All 178 failures are therefore PRE-EXISTING and untouched: `test_ra1_hjepa` (4), `test_v8f_cz` (1), `test_v8f_flat` (1), `test_v8f_oacc` (1), `test_v8f_tac` (4), `test_v8f_kindx_s1` (149), `test_v8f_kscore` (4), `test_v8f_lora` (14). The stack-only export lacks Research Lab files; reasons were not read per test (UNVERIFIED). Directly relevant, equal on both sides: `test_launch_gate_v8f` 56 passed, `test_v8f_lead_head` 45, `test_v8f_level_head` 27, `test_v8f_launch_intent` 36, `test_v8f_stopbal` 8 (+1 skipped), `test_v8f_mini` 28, `test_v8f_init_ckpt_o_interface` 7, `test_v8f_mixo` 33, `test_v8f_dagger_train` 95, `test_v8f_ego_dagger_compose` 36. Logs: scratch `suite2_*.log`, `suite3_*.log`.

## Readings fixed here (the card is silent; MM to review)

1. **Head.** Two MLP(d, d, 1): `stop_head` on EACH of the 40 adapted NOW-frame tokens (one cell logit per token), `stop_pres_head` on the mean of the 40 tokens (presence). Built last, only when on; trained in the training forward only.
2. **Loss.** `stop_aux = stop_cell + stop_pres`, weights 1:1. `stop_cell` = mean over the 40 tokens of BCE(logit, one-hot cell), averaged over the windows with an in-view rest point. `stop_pres` = BCE(presence logit, 1 if in view else 0) averaged over the label-bearing windows. Both masked by `frame_ok[:, -1]` (the NOW frame must be real).
3. **Label frame.** The label is read from `lab` (the NOW row, `v8f_mini_data.py:595`), not `lab_t`. The head reads the NOW frame.
4. **Negatives.** A `has` window with NaN `stop_x_m` / `stop_y_m` (no stop within the horizon, OR already stationary: HOLD / CREEP rows are NaN too) is a presence-0 negative. A stop that exists but projects outside the frame is also presence 0, not excluded: behind the camera, past 60 deg azimuth, above row 0, or closer than the ground is visible (about 3 m ahead of the camera at a 1.5 m mount). It carries no cell, so no per-token term. ⚠️ So "presence 0" mixes three things; a stationary ego is a systematic negative.
5. **No offset.** The point projected is the ego pose origin (`to_now_frame` of the rest pose) at z = 0, the rig frame's road plane. It is not the front bumper and not a stop line (card §3: not a stop-line label). The camera sits 1.7-2.2 m ahead of the rear axle, so a stop closer than that is behind the camera.
6. **Cell.** `floor(col / 64)`, `floor(row / 64)` on the pixel index (centre (W-1)/2, as `rig_projection`), clamped; token = r * 10 + c (row-major, builder `pool_field`). Only the 4x10 grid is supported: the config refuses `n_tok != 40` with the head on, and the gate refuses `--grid pool8x20`.
7. **Camera pose.** Per-clip, from a JSON table (`--stop-aux-extr`, the format of `rig_extrinsics_train2400.json`). A clip with no row is out of the term (neither positive nor negative). Coverage of the TRAIN clips below 0.99 is a launch refusal (`MIN_EXTR_COVERAGE`).
8. **W.** `W = T["goal"] / T["stop_aux"]` at step 0, both unweighted, from the G-LIVE probe's forward (p_gt 0.5, first sampler batch) = `selfcheck.json stop_aux_probe.w_stop_aux_for_goal_equal`. `stop_aux_w_from_smoke.py` prints it rounded to 4 decimals and warns if the first `train_log.jsonl` row (p_gt at its start value) is more than 10 % away. `T["goal"]` is the goal BCE before `w_tac`; the arm keeps `--goal-pos-weight-stop` at 1.0.
9. **Checkpoint cfg.** `stop_aux` / `w_stop_aux` are written into the checkpoint's `cfg` only when on (see the OFF section). `V8FMiniConfig(**ck["cfg"])` in the eval loaders therefore builds the head for an ON checkpoint and does not for an OFF one.
10. **Probe batch.** `stop_aux_probe` runs on the one probe batch the other probes use and refuses if it has no in-view rest point (the cell head would be dead).

## Gate list for a STOPBAL-style checker (priority 4; NOT implemented: the checker is a 60 KB box-2 stage script)

**G-DVB (each new flag read back from the BUILT model):**
1. `argv --stop-aux` equals `selfcheck.json levers.stop_aux` (`on`; the key is ABSENT when off) and `--w-stop-aux` equals `levers.w_stop_aux` and `cfg.w_stop_aux` (the trainer's `bad_stop_aux` already refuses a mismatch).
2. The built model has exactly the 8 state-dict keys `stop_head.{0,2}.{weight,bias}` and `stop_pres_head.{0,2}.{weight,bias}` and `levers.params.STOPAUX` = 132,098 at d = 256; the off argv has none and no `STOPAUX` group.
3. `levers.stop_vision_only_source` is true; `stop_aux_probe.vision_only` has `source_problems` [] and `keys_read` [].
4. `selfcheck.json stop_aux_extrinsics`: `path` is the declared table, a 64-hex `sha256` equal to the shipped file's, `coverage_train` >= 0.99 (the trainer refuses below), `n_train_clips` equal to 4,512.
5. `--grid pool4x10`; the init is M2b with exactly the 8 stop keys fresh (`info.init_ckpt.fresh_keys` = `declared_new_keys` ∩ stop keys); `--stop-aux off` arms (NULL, NULL_rep, NULL_rep2) carry no `stop_*` key anywhere in `selfcheck.json`.
6. The frozen argv of arm A equals DAG-OACC s{k}'s argv verbatim plus exactly `--stop-aux on --w-stop-aux <W> --stop-aux-extr <path>`.
7. The checkpoint of a NULL run is byte-identical to BAL-NULL's (sha256) given the same argv: this is the card's reuse test, MEASURED here on a synthetic run (58c085e4...).

**G-LIVE (the card's three conditions plus the rest of `stop_aux_probe`):**
1. finite stop-aux term: `term` > 0 and finite, `step0_stop_cell` and `step0_stop_pres` finite.
2. non-zero gradient on the head AND the adapter: `head_grad_diff` > 0, `pres_head_grad_diff` > 0, `adapter_grad_diff` > 0.
3. exactly 0.0 with the head off, and at w = 0: `head_grad_at_w0_zero_control` == 0.0 (an `--stop-aux off` selfcheck has no head to read).
4. mutation: `contribution` equals `w_x_term` within 1e-3 (a deleted total line reads 0.0 and the trainer's `live` check also goes red).
5. the controls `control_repeat_*` == 0.0, `control_allfalse_*` == 0.0, `control_prior_grad_diff` == 0.0; `probe_n_pos` >= 1.
6. `w_stop_aux_for_goal_equal` equals the argv's W to the rounding the W script applies (4 decimals).
7. `live.pass` with `stop_*` in `terms`, and `unexpected_dead` empty.
8. G-EVAL: the eval loader builds the ON checkpoint (`V8FMiniConfig(**ck["cfg"])` + strict load) and its `forward_infer` is independent of the head.

## Commands (all run in the scratch export; export is `git -c core.autocrlf=false archive 715ef8fc stack taniteval` minus `taniteval/results`)

```
export CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH="$(cygpath -w $PWD/stack);$(cygpath -w $PWD/taniteval);$(cygpath -w $PWD/stack/scripts)"
python -c "import tanitad; print(tanitad.__file__)"                       # asserted: inside the scratch export
python -m pytest stack/tests/test_v8f_stopaux.py -q -p no:cacheprovider --basetemp <existing parent>/b   # 38 passed
python stack/scripts/v8f_mini_train.py --out outdir --synthetic --batch 16 --d 32 --n-q 2 --log-every 1 --warmup 2 --ckpt-every 100 --threads 1 --kin-mode yaw --seed 5 --arm M2b --steps 4   # base tree, then patched tree (OFF): sha256sum outdir/ckpt_last.pt equal
python -m pytest stack/tests/<file> -q --tb=line -p no:cacheprovider --basetemp <dir>                  # one file at a time, base tree and patched tree
git -c core.autocrlf=false apply --ignore-whitespace out/stopaux.diff                                  # on the base blobs: reproduces the two pinned sha256
```

Pre-existing failures are those present on BOTH sides in the table above; none was touched.

## Cleanup (priority 6)

**NOT DONE: no deletion was performed.** The shell permission layer denied every `rm -rf` of the scratch tree (twice, including a literal absolute path), so nothing under `F:/TanitAD-scratch/stopaux_build/` was deleted by this agent. It holds only this agent's own, re-creatable temporary data (a base export `xb`, the patched export `x`, a mutated copy `xm`, test `--basetemp` folders under `pt/`, two synthetic run dirs, logs, patch scripts); about 0.7 GB on F: (estimated from the 162 MB base export plus the patched export and test outputs; not measured by `du`, which was also denied at the end). Everything in it is re-creatable from the base commit plus this package; the only things worth keeping are in this folder. Under the PI rule (CLAUDE.md, 2026-11-07) the MM or the PI may delete that folder: `F:/TanitAD-scratch/stopaux_build`. The runners are finished; no process of this agent is left running.
