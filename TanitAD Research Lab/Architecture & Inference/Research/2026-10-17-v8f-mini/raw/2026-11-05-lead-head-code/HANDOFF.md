# HANDOFF -- lead head `--lead-head {off,on}` + `--w-lead` + `--aux-cond {add,token}` (arm LLR of PREREG_LEVELHEAD_A2; tip 219f6c76, 2026-11-05)

Nothing committed, pushed, landed or launched on a GPU. Hold (mods-rw + stage): `C:/Users/Admin/qland/hold_v8f_lead_head/`. Work tree: `F:/TanitAD-scratch/motion_c/lead_build/{tip,new}/` (raw cat-file exports of stack/ + taniteval/). Patches that rebuild `new` from `tip`: `patches/pb_model.py`, `pb_data.py`, `pb_train.py` (`python -I pb_x.py <tip file> <out file>`). Hashes: `SHA_CHANGES.txt`.

## What it is
* `v8f_mini.py`: cfg `lead_head`/`w_lead`/`aux_cond`; head `lead_head` = MLP(d,d,2) on the mean of the CURRENT tick frame's adapted tokens (presence logit, gap_hat = sigmoid = gap/40); `_lead()`, `_aux_in()`; `add`: `e_lead` = Linear(2,d) zero-init ADDED to O's ego token (like e_level); `token`: ONE extra cond token `e_aux` = Linear(3,d) default init of [v_hat/10 | 0, sigmoid(logit), gap_hat] (detached), REPLACING both additive paths (e_level/e_lead not built). Loss `w_lead x (BCE(presence) on labelled frames + Huber(gap_hat, min(gap,40)/40, delta 0.1) on frames with a lead <= 40 m)`, ONE total line. Guards `lead_head_source_problems`, `aux_source_problems`, `assert_lead_head_vision_only`, `assert_aux_vision_only`. Built after the level head (its init draws are the level-only model's).
* `v8f_mini_data.py`: `lead_label_arrays(J, lead_npz)` (row-aligned has/near/y from the `--lead-train` file, k = 2j; build_label_join untouched).
* `v8f_mini_train.py`: flags, FLAG_CHECKS, `bad_lead` (also needs labels: file, sha256, >0 labelled and near frames, >= 50 % of clips labelled), `bad_level` token-aware, init-ckpt fresh keys = `M.NEW_LEVER_KEY_PREFIXES`, `make_lead_labels`, `lead_stage(b, LL)` (before the ego stages), `lead_head_probe` (G-LIVE; `selfcheck.json -> lead_head_probe.w_lead_for_10pct_share`), level probe under token, loop/summary/selfcheck lines.
* Tests: `stack/tests/test_v8f_lead_head.py` (45) + `v8f_mini_tip_snapshot_lead.txt` (tip model, sha a3634276...); `test_v8f_launch_intent.py` edited by the two cfg-key lines (golden cfg gains `lead_head`/`w_lead`/`aux_cond`).

## Evidence (logs/)
lead tests 45 passed; level tests on the new tree 27 passed; launch_intent 36 passed (golden K-BUILD: flag-off = tip bit for bit); real C0 smoke (`real_c0_smoke.log`): 276 tensors, mismatch 0, fresh keys = the declared ones only, add: step-0 forward EXACT, token: differs by construction (max |d traj60| 3.1-3.6 m on a synthetic batch).
Mutations: `mutation_lead.log` = 35 source mutations, every one RED, clean GREEN (ALL-DONE 609 s). Run 1 (`mutation_lead_run1.log`) had 33 of 34 RED; its one GREEN (a forced gradient value) was replaced by two wrong-column reads, both RED.

## Next steps (exact)
1. FULL SUITE runs detached since 16:26 (`runner/run_suite.cmd`). Done when `logs/SUITE_DONE.marker` exists. Read: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe E:/Projects/TanitAD-artifacts/dagland/motion_c/lead_build/runner/compare_suites.py` (exit 0 = NEW-ONLY empty; counts in `logs/COMPARE.txt`). Expect BOTH ~318 like 6ece77f.
2. Mutation log: already final (see Evidence).
3. Then `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I mk_lead_hold.py logs` refreshes the hold package's logs/ + MANIFEST. The MM lands the hold after `git ls-remote origin agent/arch-inf-20260803` (the tip must still be 219f6c76).
4. PREREG: on box 2 run the LLR argv with `--selfcheck-only` and read `lead_head_probe.w_lead_for_10pct_share` (synthetic world gives ~9: meaningless); commit it in the card.

## Choices for the MM to rule on / Unverified
Huber delta 0.1 (4 m; the card names none); coverage guard 0.5; BCE on every labelled frame, Huber only on frames with a lead <= 40 m; token mode perturbs C0's step-0 plan by ~3 m (by construction). Unverified: real-data run on the 4572-clip train corpus (box 2), `--encoder lora` with the head on, T1 evaluators on a lead-on checkpoint (covered only by the cfg rebuild test), resume of a lead-on run, single-seed claims: none.
