# HANDOFF: LEADCOND, trainer/model lever `--lead-cond {off,on}`, 2026-11-15

State: DONE on the CPU. Nothing trained on real data; no git, no pod, no box 2. Folder `F:/TanitAD-scratch/v8proof/leadcond/`. Base = the RULEOWNL1 tree (`ruletrain3/work/tree_patch`, trainer `2ad71c9f...`, model `99bbaab6...`, copied unchanged to `work/tree_base`). Result `work/tree_patch`. Definitions with file:line: `LEADCOND_DEFS.md`.

## What changed (sha256 in `PINS.txt`)
* `stack/scripts/v8f_mini_train.py` `d45621201f4a5bc95df0d5ef2be7c07587d208a752e81f9abb1e935a84e22f9e`: flag, G-DVB entry and `bad_lead_cond`, `bad_init_ckpt` clause (fresh keys exactly the two), `lead_cond_probe` (G-LIVE) and its selfcheck record, cfg recorded only when on.
* `stack/tanitad/models/v8f_mini.py` `2e6622081a2cbbe50cd92367afb92f29beeda1f674c3144779953018a0713001`: config field, `_lead_u`, module built last with RNG restore, token added in `run_t` and `run_o`, built levers, trainable group.
* new `stack/tests/test_v8f_lead_cond.py` `9a402bd86eb7b91f978fbf82dfb30a957c6bb2896db20aebb6ae1a63c9003cef` (14 tests).
* All other pinned files unchanged (`v8f_t1_eval.py` `61e140ed...`).
* `LEADCOND.patch` is against `work/tree_base`; replay verified (the three files reproduce byte for byte). `patched/` holds the three files.

## Selfcheck strings
`levers.lead_cond=on` (absent when off), also `levers.lead_cond_zero_init`, `lead_cond_probe` (n_rows, p and g mean / std, grad norm of weight and bias, zero-u control 0.0, detach difference 0.0), `levers.init_ckpt.fresh_keys` = `lead_cond.bias`, `lead_cond.weight`.

## Controls (JSON + script in `checks/`, key `PASS`; all PASS)
* c1 `c1_identity.json`: off, tree_patch vs tree_base, 100 steps, shapes G, LS, LSFT, LSOWN: every logged field and final weights max abs diff 0.0, key lists and init tensors equal.
* c2 `c2_zero_init.json`: lever on from an LS-shape base checkpoint, forward and forward_infer outputs (12 tensors each) max abs diff 0.0 vs the base tree; from-scratch state dict equal apart from the two keys, next RNG draw equal; controls: an unrestored init moves the next draw, a non-zero weight moves traj60 / goal / lat outputs.
* c3 `c3_reach.json`: operative (O alone, code held fixed) and tactical outputs move with a non-zero weight; weight gradient 0.094 on a real loss step, exactly 0.0 with u zeros; plan-loss gradient on the lead head through u exactly 0.0 (control with the lead_sup term 1.88); trunk gradient through u exactly 0.0 (undetached control 4.7). Mutation: the adding line of run_o replaced by `pass` -> O reads 0.0 and the check is red; same for run_t -> T reads 0.0, red.
* c4 / c5 `c4_c5_time_vision_parity.json`: u exactly unchanged under perturbation of 22 batch keys (act, speed, p10f, nav, pose, limit, base ...), of frames before the window, of frames after t; controls (last frame, oldest window frame) change u by 0.81 / 0.73. Parity forward vs forward_infer 0.0; u vs the head's training-time output 4.8e-7 (bar 1e-5). `frame_ok` is the validity mask and is not an ego input (u is zero where a window frame is not real).
* c6 `c6_gate.json`: off selfcheck key set equals the base tree's (no `lead_cond` text); on carries the string, probe and fresh keys; gate_extra accepts the right expectation, refuses wrong ones; `--lead-cond on` without `--lead-sup on` rc 1; unknown value rc 2. Shapes LFT, LC, LO, LCO with the card argv (6,000 steps).
* c7 `c7_finetune_eval.json`: 20-step fine-tunes (lever; lever + `--w-rule-own 4.0 --rule-own-form l1`) finite; checkpoint cfg = base plus `lead_cond: on`, two extra keys; `v8f_t1_eval.py` loads and runs both; with zero lead_cond weights the eval dumps equal the base checkpoint's exactly (1,204 floats, diff 0.0); control: the fine-tuned checkpoint differs.
* Tests `tests.log`: 159 passed, 0 failed, rc 0 (145 of the RULEOWNL1 list + 14 new). Two earlier runs failed one existing test each (directory removed under the test) because two pytest runs shared one basetemp at the same time; the runs were overlapped by my own double launch, the failing tests pass alone and in the clean serial run. The first failed log is kept as `tests_run1_1failed_flaky.log`.

## Build the pod tree
`cp -r /workspace/v8proof/tree_ruleown2 /workspace/v8proof/tree_leadcond; cd tree_leadcond; patch -p1 --binary < LEADCOND.patch` (or copy `patched/` over `stack/`). Verify the `PIN|` lines of `ARMS_LEADCOND.txt`, run the LC_s0 selfcheck first and read `lead_cond_probe` (n_rows, p / g stats, gradient norms).

## Arms (`ARMS_LEADCOND.txt`, ten arms)
LFT, LFTb, LC, LO, LCO, each s0 / s1, from `/workspace/v8proof/runs/leadsup_a/LS_s{0,1}/ckpt_last.pt`, 6,000 steps, lr 1e-4, warm-up 100, LS flags. Fields the lead edits if the own-plan weight or form changes: on the LO and LCO lines the argv tokens `--w-rule-own X --rule-own-form F` and the expectation tokens `levers.w_rule_own=X` (whole floats as ints) and `levers.rule_own_form=F` (drop the form token if `sq`; drop both tokens for no own term). LFTb is the same argv as LFT with the same seed column (replicate).

## Open points for the lead
* LFTb with the same argv and seed may be bit-identical to LFT or differ only by non-determinism; if a seed-varied floor is wanted, change the seed column of the LFTb lines.
* T reads u at the current frame although its own scene tick is older; documented in the DEFS (allowed by the time rule).
* On the tiny world p is near 0.5 and g near 0; real-data u statistics are first seen in the pod selfcheck.
* Interaction with `--lead-head` / `--level-head` (older levers): untested here; not part of these arms.

## Temp paths and sizes (delete refused here; for you to delete)
`work/tmp` 1.1 GB (worlds, tiny checkpoints, selfcheck and eval outputs), `work/pt` 126 MB (pytest basetemp), `work/tree_mutO` and `work/tree_mutT` 39 MB each (mutation trees), `work/verify_patch` small. Kept: `work/tree_base`, `work/tree_patch` 39 MB each, `work/edits` (the edit scripts). Outside the folder, my edit scripts were first written to `C:/Users/Admin/v8proof_c/` (`lc_edit1.py` to `lc_edit4.py`, `fix.py`, a few KB; copies in `work/edits`).
