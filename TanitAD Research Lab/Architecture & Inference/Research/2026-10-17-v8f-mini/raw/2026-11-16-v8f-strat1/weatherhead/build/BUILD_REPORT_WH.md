# BUILD_REPORT_WH: form (a) of PREREG_V8F_WEATHERHEAD (built, tested, NOT run, NOT launched)

Builder: one Sonnet agent, 2026-11-16 evening. Base: `a4357e79` (`agent/v8f-strat1-20261116`). Card sha256 `f350b52c...` (frozen, not edited). Nothing here is a result; every smoke output is marked SMOKE.
Evidence class of every statement below: MEASURED (this build, artifact named) unless marked otherwise.

Work folder `F:/TanitAD-scratch/strat1/whead/`: `tree/` = edited code of record (export of `a4357e79` plus the diff below), `build/` = new study scripts, `smoke/` = SMOKE outputs, `logs/` = test and gate logs, `run/gate/` = gate records.

## 1. What changed in the code of record (3 files edited, 2 added; nothing else)

| file | change |
|---|---|
| `stack/tanitad/models/v8f_mini.py` | `V8FMiniConfig.ctx_pool` (default `off`, validated), `PoolHead`, `build_ctx_pool_heads`, the head built LAST under a forked RNG (`CTX_POOL_INIT_SEED` 20261117), `_ctx_pool_scene` / `_ctx_pool_apply` / `_ctx_pool` / `ctx_pool_fit_logits`, `out["ctx_pool_logits"]`, group `CTX_POOL`, built levers, `ctx_pool_terms` (fit loss), `ctx_pool_source_problems`, `assert_ctx_pool_vision_only`, `ctx_pool_isolation_problems`, `CTX_POOL_KEY_PREFIXES` in `NEW_LEVER_KEY_PREFIXES`. 141 changed lines, all additive. |
| `stack/scripts/v8f_mini_eval.py` | `--ctx-pool off|on` (default off), refusal when it differs from the checkpoint's cfg, `ctx_pool_logits` into the eval npz. 7 lines. |
| `stack/scripts/v8f_t1_eval.py` | the same flag and refusal, `ctx_pool_logits` per tick row with `--dump-heads`. 14 lines. |
| `stack/tests/test_v8f_weatherhead.py` (new) | 27 model tests incl. mutations m1 to m4 and the off snapshot. |
| `stack/tests/v8f_mini_tip_snapshot_whead.txt` (new) | the `a4357e79` model file (sha256 `776643b5...`), the off-equals-tip reference. |

NOT edited: `stack/scripts/launch_gate.py`, `stack/tanitad/models/__init__.py`, `stack/scripts/v8f_mini_train.py` (a test pins that the trainer has no `ctx_pool`: form (a) adds no trainer flag), anything under `E:/Projects/TanitAD/stack`.

## 2. Reachability, hop by hop (card 4.4 item 1), file:line in `tree/`

| hop | where |
|---|---|
| fit script flag `--ctx-pool` (required, only value `on`) | `build/wh_fit.py:51` |
| evaluator flags `--ctx-pool` (default off; named in every argv by `wh_run.py`) | `stack/scripts/v8f_mini_eval.py:658`, `stack/scripts/v8f_t1_eval.py:523`; refusal on mismatch `:238` / `:539` |
| config field + validation | `stack/tanitad/models/v8f_mini.py:268`, `:309` |
| the head is BUILT (`self.ctx_pool_head`), under a forked RNG | `v8f_mini.py:765` (`build_ctx_pool_heads`), `:935` to `:937` |
| `_forward` emits `out["ctx_pool_logits"]` | `v8f_mini.py:1139` to `:1140` |
| the fit path (adapter and scene under no_grad, heads with grad) | `v8f_mini.py:1019` to `:1026` (`ctx_pool_fit_logits`) |
| fit loss terms `T["ctx_pool/<var>"]` and their sum | `v8f_mini.py:1771` to `:1780` (`ctx_pool_terms`) |
| `loss = T["ctx_pool"]` and `loss.backward()` in the fit | `build/wh_lib.py:304` to `:309` (`fit_step`), `build/wh_lib.py:385` (`fit_head`) |
| evaluator writes the key | `v8f_mini_eval.py:298` to `:299`, `:362` to `:363`; T1 `v8f_t1_eval.py:294`, `:345` to `:346`, `:375` |
| reader takes `ctx_pool_logits` only (never `ctx_logits`) | `build/wh_read.py` `head_logits` |

Non-zero and exact-zero (item 2): on one real batch with `requires_grad` True on all 315 model parameters, backward gives non-zero gradient on each of the five sub-heads and EXACTLY 0.0 / None on every other parameter (S, T, O, adapter, `pos`, `time`, `e_*`, the shipped `ctx_head`): `tests/test_v8f_weatherhead.py::test_reachability_*`, and on the real S1_s0 checkpoint inside `wh_check.py g_live` (record in the token). The FEAT control reads NON-zero on the adapter, so the zero is a real zero.
Adam-state census (item 3): after a real 2-batch fit, 55 of 315 registered parameters have optimizer state, exactly the head's (`wh_lib.adam_census`; logic of `probe_zerograd3.py`). FEAT expects 61.
Defaults (item 4): `--ctx-pool` has no default in the fit parser (a missing flag does not parse); the evaluators default to off and REFUSE a head checkpoint run with off, and an off checkpoint run with on.

## 3. Mutation table (each must turn its named test RED; each reads RED, MEASURED)

| id | what is mutated (ONE edit) | check that must read RED | read |
|---|---|---|---|
| m1 | the fit-loss line deleted (`ctx_pool_terms` sum times 0; and `fit_step` loss line in `wh_lib.py`) | G-LIVE (non-zero on the five sub-heads) | RED: tree `test_mutation_m1_*`, study `test_live_check_reads_pass_and_each_mutation_reads_red`, and the regression arm inside `wh_check g_live` |
| m2 | the `no_grad` on the token path dropped (`torch.enable_grad()` in `ctx_pool_fit_logits`) | the exact-zero control (other grad 2,187.1 instead of 0.0) and the Adam census (extra parameters with state) | RED: tree `test_mutation_m2_*`, `wh_check g_live` regression arm (MEASURED `other_grad_abs_sum` 2187.10) |
| m3 | `ctx_pool_logits` fed into T's code (`code = ... + 0.01 * ctx_pool_logits`) | T-ISO (b) | RED: tree `test_mutation_m3_*` |
| m4 | an ego / nav key named inside `_ctx_pool` (`b["nav"]`) | T-VIS static name check; a runtime-only variant (key built from parts) is caught by the key guard | RED: tree `test_mutation_m4_*`, `test_mutation_an_ego_key_read_at_runtime_only_*` |
| m5 | the evaluator drops `--ctx-pool` (npz without `ctx_pool_logits`) | the reader reads "no output", never falls back; the evaluators REFUSE a flag that differs from the checkpoint cfg | RED: study `test_an_h_dump_without_ctx_pool_logits_*`; `wh_check geval` `flag_mismatch_refused` (rc 1) |
| m6 | the reader substitutes `ctx_logits` for an H arm | K-ID | RED: study `test_k_id_reads_pass_on_equal_logits_and_red_when_the_reader_substitutes_ctx_logits` |
| off | the head built unconditionally | off equals `a4357e79` (state dict, loss terms, gradients, `forward_infer`) | RED: tree `test_mutation_unconditional_pool_head_*` |
| rng | no forked RNG around the head init | the global RNG stream and every other init tensor equal the off model's | RED: tree `test_mutation_no_fork_rng_*` |
| hyg | the `ctx_pool` value check deleted | G-HYG | RED: tree `test_mutation_deleting_the_ctx_pool_check_*`; `wh_check g_hyg` regression arm (lenient config) |
| blind | pooled logits detached and zeroed | the T-ISO (b) probe itself (it must read "did not move") | RED: tree `test_mutation_the_pooled_head_never_reaches_*` |
| P-COMMIT | streamed source offset by one window | byte-equality of streamed and in-RAM features on two batches | RED: study `test_regression_arm_an_offset_of_one_window_reads_red` |
| clock | labels read at row + 1 | G-CLOCK against `label_rows` of `strat1_ctx_read.py` | RED: study `test_clock_matches_the_reader_and_a_shift_reads_red`; `wh_check g_clock` regression arm |
| K0 | one logit pair of S1's banked `ctx_logits` swapped | K0 anchor | RED: study `test_k0_reproduces_every_strat1_row_exactly_and_a_perturbation_reads_red` |
| ISO-c / ISO-d | S1 seed 0 compared with S1 seed 1 (different planner weights) | T-ISO (c) and (d) must read DIFFERENT | RED (as required, 342 batch-key differences in (c), 17 arrays in (d), SMOKE logs `SMOKE_iso_c.json`, `SMOKE_iso_d.json`) |
| G-EVAL | fit-side logits of a window shifted by one frame | G-EVAL | RED: max abs diff 0.1405 (text family) and 0.1379 (v9 family) against the 1e-5 bar |
| token | a check whose regression arm reads PASS / a hand-edited token / a changed argv / a wrong arm | the token writer and `verify_token` | RED: study `test_a_token_is_written_only_*`, `test_a_hand_edited_token_*` |

## 4. Test and gate results

* Model tests (tree): `logs/pytest_model_weatherhead.log`, 27 passed. Study tests (build): `logs/pytest_study.log`, 31 passed. Both on the dev box CPU, `CUDA_VISIBLE_DEVICES=-1`, tree-first PYTHONPATH (`tanitad.__file__` asserted inside the tree).
* G-SUITE (subset, `logs/gsuite_result.json`): new tree 238 tests (237 pass), `a4357e79` tree 211 tests (210 pass), **0 regressions**, 0 new-test failures, regression arm (a planted failure) reads FAIL as required. Scope: `test_v8f_weatherhead`, `test_v8f_strat1`, `test_v8f_mini`, `test_v8f_level_head`, `test_v8f_lead_head`, `test_strat1_token`, `test_launch_gate_v8f`. The full suite is the MM's (section 5 item 1).
* G-EVAL (`logs/geval_text.json`, `logs/geval_v9.json`): T0 on the full 2,399-window grid, evaluator vs fit script max abs diff 1.31e-06 (bar 1e-05); T1 tick 0 on 4 windows, diff 0.0; an evaluator run with `--ctx-pool off` on a head checkpoint is refused (rc 1); regression arm reads FAIL. Both nav families (S1_s0 text, C0_s0 v9).
* Gate mint, demonstration (`logs/mint_H_s0_checks.json`, `logs/mint_C0A_s0_checks.json`): tokens for H_s0, K4p_s0 and C0A_s0 were minted with all seven checks PASS and every regression arm FAIL, verified (tampered argv and wrong-arm refused), and then removed (section 7); the MM mints at launch.
* G-LIVE on the real S1_s0 checkpoint and real train windows (first 256 windows of 64 clips): sub-head gradient |sum| country 1067.7, weather 2474.3, surface 1677.7, illumination 1463.2, period of day 1273.7; every other parameter exactly 0.0; Adam state for 55 of 315 parameters.
* **SMOKE (never a result)**, `smoke/`: one H fit on 40 train clips (320 windows), epoch grid {1}, stream source, CPU: 14 s; its checkpoint through the T0 evaluator (`--ctx-pool on`, 2,399 windows, 102 s CPU): the planner arrays `traj30`, `ckpt`, `goal_prob`, `lat_logits`, `lon_logits`, `cons_pred` and the shipped `ctx_logits` are byte-equal to S1_s0's banked T0 npz (ADE6 1.9843, ADE2 0.2205 as S1_s0); `ctx_pool_logits` shape (2399, 41); K-ID max abs diff 1.07e-06; T-ISO (c) pass (with its regression arm), T-ISO (d) pass on 4 windows (32 arrays byte-equal, `ctx_pool_logits` the only extra) with its regression arm; the reader on the smoke dumps: K0 pass at B 2,000 (every STRAT1_CTX row reproduced), K2, K3, K5, K6, K-ID pass, and INSTRUMENT-INCOMPLETE for the checks not run (no outcome quoted).

## 5. Deviations and open items (stated, not hidden)

1. **G-SUITE scope.** The G-SUITE result file records `scope`. The run here covers a subset (the v8f test files that import the edited modules plus the launch-gate and STRAT1 token tests), new tree against the `a4357e79` tree. The card asks for a clean-tree full suite; the MM runs the full suite (pod or Thor environment, as for STRAT1) and re-mints with `--full`. Until then the token's G-SUITE record says "subset".
2. **Token validity.** A token binds the tree manifest and the sha256 of every `build/wh_*.py` file. Any later edit of either makes it read "differs"; the MM re-mints after landing (about 1 minute per arm plus one G-EVAL record per nav family).
3. **FEAT_s1 fit seed.** The card's row 8 reads "seed 0 / 1" for both the init seed and the fit seed; built as adapter init seed N and fit seed N for FEAT_sN. All other arms use fit seed 0 (H_s0_rep: 1).
4. **R1 label permutation** is built per card 5 row 5 (seed-0 clip permutation, rows by position modulo, five variables together) in `wh_lib.permute_labels_between_clips`; it is the probe's rule.
5. **The 60 unlabeled clips** (indices 540 to 599): their windows stay in the window set and enter no CE (all five valid bits False in the table; MEASURED 60 of 4,572 clips with zero valid labels, all inside 540:600). The arms need no `--exclude`.
6. **K4p / K-ID / T-ISO / reachability** produce records the reader consumes (`wh_chain.py instrument`); the reader prints INSTRUMENT-INCOMPLETE and quotes no outcome until each is read, and INSTRUMENT-FAIL (withholding every arm row) when one fails.
7. **Throughput (ESTIMATED, not measured on the GPU: the GPU lock was held by another job and the card reads 5.9 GB used during the build).** CPU smoke: 1 epoch of 256 windows about 2 s. The streamed source is the P-COMMIT fallback; its gathers run in a background thread ahead of the step (`wh_lib.prefetched`). The card's 3 to 6 h dev-box GPU estimate is unchanged and unverified here.
8. **Commit headroom** read 11.0 GB free at the first smoke and 3.4 GB while two other pytest runs were live: a feature tensor held in RAM needs 5.98 GB plus 5.5 GB, so `--source auto` will choose STREAM on this box in most states. Both paths are byte-equal by test.

9. **FEAT epochs.** The FEAT adapter is shared by the five sub-heads and trained with them, so sub-heads from fits at different E would sit on different adapters. FEAT therefore uses ONE fit at the weather E_v of its seed's H arm for all five variables (recorded as `E_v_used` in its `wh_meta.json`). FEAT decides the weather failure branch only; the other four variables' FEAT rows are beside, no bar. The card text says "E_v of H"; this is the one place the build narrows it, and the MM may rule otherwise (a per-E checkpoint set would need the reader to merge five passes).

## 6. Escalated to the MM

* Integration: commit the 5 tree files above (diff list and sha256 in `MANIFEST_WH.md`); copy `build/` (new study scripts, 7 files plus 2 test/launch helpers) into the repo where the STRAT1 study scripts live; nothing else of mine touches the repo.
* Tokens: mint per arm (`wh_check.py mint ARM --runs <R> --family text|v9`), after `wh_check.py suite` (and `--full` for the full suite) and `wh_check.py geval text` / `geval v9`.
* Launch: `launch_wh_wmi.ps1` starts `job_wh.cmd` -> `launch_wh.sh` (GPU fits take the shared lock, CPU stages are commit-gated). Written, not launched. Order of priority is the card's: P1 and P2 give the detection read at both seeds, P3 and P4 the wording control, P5 and P6 the floor.
* Decision still the PI's per the card: Q1 (conditional pod go for form (b)), Q2 (wording of a weather deliverable), Q3 (PI frames on minority classes). None is needed to run form (a).

## 7. Temporary data deleted by this build (literal paths, own data only)
All created by this build; none was the only copy of anything (every one is re-creatable by the command that made it, or kept as a record in `logs/`). Sizes from `du -sk` just before deletion (KB, allocated on exFAT F:).

| path | KB | why it is not needed |
|---|---|---|
| `F:/TanitAD-scratch/strat1/whead/run/gate/pt_base` | 1,171,327 | pytest basetemp of the base-tree G-SUITE run |
| `F:/TanitAD-scratch/strat1/whead/run/gate/pt_new` | 1,190,683 | pytest basetemp of the new-tree G-SUITE run |
| `F:/TanitAD-scratch/strat1/whead/base_tree` | 460,170 | the `a4357e79` export used for the G-SUITE base side; the manifest diff now reads the base from the git object store |
| `F:/TanitAD-scratch/strat1/whead/tmp` | 372,341 | TMP / TEMP and pytest `--basetemp` folders of the build |
| `F:/TanitAD-scratch/strat1/whead/smoke/iso` | 7,630 | SMOKE T-ISO dumps (the two result JSONs are kept in `logs/`) |
| `F:/TanitAD-scratch/strat1/whead/smoke/t0r`, `smoke/kidr` | 4,928, 888 | SMOKE copies made for the reader smoke |
| `F:/TanitAD-scratch/strat1/whead/run/gate/geval_text_work`, `geval_v9_work` | 2,788, 2,432 | G-EVAL work folders (the result JSONs are kept) |
| `F:/TanitAD-scratch/strat1/whead/smoke/H_s0/ckpt_wh.pt` | 33,865 | SMOKE head checkpoint (re-creatable: the smoke command is in section 4) |
| `F:/TanitAD-scratch/strat1/whead/run/gate/H_s0`, `K4p_s0`, `C0A_s0`, `x` | 12, 8, 12, 0 | demonstration tokens (their `checks.json` are kept in `logs/`); a token binds the tree and build sha256, so the MM mints its own |
| every `__pycache__` under `tree/` and `build/` | n/a | interpreter caches |

Kept: `tree/` (the code of record), `build/`, `logs/`, `smoke/H_s0/wh_meta.json`, `smoke/t0`, `smoke/kid`, `smoke/read` (SMOKE records), `run/gate/{gsuite_result,geval_text,geval_v9}.json` and the two junit files. No other session's data was touched. Free space on F: after the sweep: about 43 GB.
