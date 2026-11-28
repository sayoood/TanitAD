# HANDOFF -- lever (a) `--level-head {off,on}` + `--w-level` (tip b07982c0, 2026-11-05)

Nothing committed, pushed, landed or launched on a GPU. Lever (c) was DROPPED by the MM (motion probe WEAK / ABSENT); no motion-token code was ever written, nothing to set aside.
Hold (mods-rw + stage): `C:/Users/Admin/qland/hold_v8f_level_motion/`. Work tree: `F:/TanitAD-scratch/motion_c/build/{tip,new}/` (raw cat-file exports of stack/ + taniteval/). Patches that rebuild `new` from `tip`: `patches/pa_model.py`, `patches/pa_train.py`.

## What it is (file: change)
* `stack/tanitad/models/v8f_mini.py`: config `level_head` ("off") and `w_level` (0.0); head `lvl_head` = MLP(d,d,1) on the mean of the CURRENT tick frame's adapted tokens (vision only) and `e_level` = Linear(1,d), weight and bias ZERO; both built LAST (no RNG shift of any old module). `run_o` ADDS `e_level(v_hat.detach()/10)` to O's ego cond token; `run_o` keeps its 6-arg signature (the trainer probes, T1 sub-eval and tests wrap it). Loss: Huber(v_hat, logged speed) masked, ONE total line `+ w_level * level`. Guards `level_head_source_problems`, `assert_level_head_vision_only`.
* `stack/scripts/v8f_mini_train.py`: flags, FLAG_CHECKS entries, `bad_level` (head with no loss, weight with no head, built-vs-declared, non-zero e_level at fresh start, non-vision source), `load_init_ckpt` (lever on: EVERY old key loads, ONLY the declared new keys are fresh; off: the strict load verbatim), `bad_init_ckpt`, `level_stage` (label = LOGGED `kin3[:,0]`, taken BEFORE the ego stages, attached after), `level_head_probe` (G-LIVE), loop and record lines.
* `stack/tests/test_v8f_level_head.py` (27 tests) + `v8f_mini_tip_snapshot_level.txt` (frozen tip model, sha256 b337ccec...); `test_v8f_launch_intent.py` edited by ONE assertion (the golden's cfg now also gains `level_head`/`w_level`).

## DEVIATION from the brief, for the MM to rule on
The brief says "one extra conditioning token" AND "forward at step 0 equals the off model's". Both cannot hold: an appended zero token changes `cond.mean(1)` (the Reader's query) and adds a softmax key (test `test_an_appended_token_could_not_do_this...` shows it). I kept the binding test: the level embedding is ADDED to O's ego token, so with `e_level` = 0 the forward is bit-identical (MEASURED, also on the real C0 init checkpoint). Mutation "token dropped from O's cond" = drop that addition (RED).

## Evidence (logs in the hold package `logs/`)
New file 27 passed (`level_head_tests.log`); real C0 init (`M2b_s0_dagoacc/ckpt_last.pt` sha 67265ec1...): 276 tensors, mismatch 0, fresh = the 6 declared keys, lever = 66,561 params, step-0 forward equal EXACTLY (`real_c0_forward.log`). `test_v8f_launch_intent` 36 passed = golden K-BUILD (final weights, optimiser state, every log line bit for bit, 3 argv families). Source-level mutations: 15 of 15 RED, clean GREEN (`mutation_level.log`). Targeted 11 files: 375 passed, 375 skipped, 3 failed: oacc-hx and t1_speedsub fail on the tip too (BOTH); the launch_intent golden failed only on the cfg-key list, fixed (`launch_intent_after_edit.log`).

## Next steps (exact)
1. FULL SUITE is running detached since 13:00 (`runner/run_suite.cmd`, ~85 min + tip pass). Done when `logs/SUITE_DONE.marker` exists. Read: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe E:/Projects/TanitAD-artifacts/dagland/motion_c/build/runner/compare_suites.py` (exit 0 = NEW-ONLY empty; the quoted counts are in `logs/COMPARE.txt`). Expect BOTH ~318 like eccdabd.
2. Then `python -I mk_level_hold.py logs` refreshes the hold's `logs/` + MANIFEST. The MM lands the hold (land_hold.sh) after the tip pin check (`git ls-remote` first).
3. PREREG: W for "level loss ~10 % of the total at step 0" is NOT computed here (it needs the real data). On box 2 run the arm's argv with `--selfcheck-only --level-head on --w-level 1` and read `selfcheck.json -> level_head_probe.w_level_for_10pct_share`; commit that number in the CL PREREG. (Synthetic world gives 1.2: meaningless.)

## Unverified
Real-data smoke on the TRAIN corpus (box 2 only); `--encoder lora` with the head on; T1 evaluators on a level-on checkpoint (they rebuild `V8FMiniConfig(**ck["cfg"])` and load strictly: covered only by the test that rebuilds from the checkpoint's cfg). Single-seed claims: none made.
