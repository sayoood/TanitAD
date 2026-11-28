p = "F:/TanitAD-scratch/strat1/whead/BUILD_REPORT_WH.md"
s = open(p, encoding="utf-8").read()
mut = """| id | what is mutated (ONE edit) | check that must read RED | read |
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
| token | a check whose regression arm reads PASS / a hand-edited token / a changed argv / a wrong arm | the token writer and `verify_token` | RED: study `test_a_token_is_written_only_*`, `test_a_hand_edited_token_*` |"""
res = """* Model tests (tree): `logs/pytest_model_weatherhead.log`, 27 passed. Study tests (build): `logs/pytest_study.log`, 31 passed. Both on the dev box CPU, `CUDA_VISIBLE_DEVICES=-1`, tree-first PYTHONPATH (`tanitad.__file__` asserted inside the tree).
* G-SUITE (subset, `logs/gsuite_result.json`): new tree 238 tests (237 pass), `a4357e79` tree 211 tests (210 pass), **0 regressions**, 0 new-test failures, regression arm (a planted failure) reads FAIL as required. Scope: `test_v8f_weatherhead`, `test_v8f_strat1`, `test_v8f_mini`, `test_v8f_level_head`, `test_v8f_lead_head`, `test_strat1_token`, `test_launch_gate_v8f`. The full suite is the MM's (section 5 item 1).
* G-EVAL (`logs/geval_text.json`, `logs/geval_v9.json`): T0 on the full 2,399-window grid, evaluator vs fit script max abs diff 1.31e-06 (bar 1e-05); T1 tick 0 on 4 windows, diff 0.0; an evaluator run with `--ctx-pool off` on a head checkpoint is refused (rc 1); regression arm reads FAIL. Both nav families (S1_s0 text, C0_s0 v9).
* Gate mint, demonstration (`logs/mint_H_s0_checks.json`, `logs/mint_C0A_s0_checks.json`): tokens for H_s0, K4p_s0 and C0A_s0 were minted with all seven checks PASS and every regression arm FAIL, verified (tampered argv and wrong-arm refused), and then removed (section 7); the MM mints at launch.
* G-LIVE on the real S1_s0 checkpoint and real train windows (first 256 windows of 64 clips): sub-head gradient |sum| country 1067.7, weather 2474.3, surface 1677.7, illumination 1463.2, period of day 1273.7; every other parameter exactly 0.0; Adam state for 55 of 315 parameters.
* **SMOKE (never a result)**, `smoke/`: one H fit on 40 train clips (320 windows), epoch grid {1}, stream source, CPU: 14 s; its checkpoint through the T0 evaluator (`--ctx-pool on`, 2,399 windows, 102 s CPU): the planner arrays `traj30`, `ckpt`, `goal_prob`, `lat_logits`, `lon_logits`, `cons_pred` and the shipped `ctx_logits` are byte-equal to S1_s0's banked T0 npz (ADE6 1.9843, ADE2 0.2205 as S1_s0); `ctx_pool_logits` shape (2399, 41); K-ID max abs diff 1.07e-06; T-ISO (c) pass (with its regression arm), T-ISO (d) pass on 4 windows (32 arrays byte-equal, `ctx_pool_logits` the only extra) with its regression arm; the reader on the smoke dumps: K0 pass at B 2,000 (every STRAT1_CTX row reproduced), K2, K3, K5, K6, K-ID pass, and INSTRUMENT-INCOMPLETE for the checks not run (no outcome quoted)."""
assert "@@MUTATIONS@@" in s and "@@RESULTS@@" in s
s = s.replace("@@MUTATIONS@@", mut).replace("@@RESULTS@@", res)
s += """
## 7. Temporary data deleted by this build (literal paths, own data only)
@@DELETIONS@@
"""
open(p, "w", encoding="utf-8").write(s)
print("filled")
