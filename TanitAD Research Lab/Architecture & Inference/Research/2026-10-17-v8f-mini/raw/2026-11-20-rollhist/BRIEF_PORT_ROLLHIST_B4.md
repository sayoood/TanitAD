# Brief: port lever C (ROLLHIST N1-N4) onto cdbff971, UE stage A's code (dev box CPU; no pod, no GPU)

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage nothing, commit nothing, push nothing. Your deliverable is a PATCH against `cdbff971` plus records; the MM commits it.
* No edits to protected files (launch_gate.py, stack/tanitad/models/__init__.py, anything under E:/Projects/TanitAD/stack, the sha-pinned v8f_t1_eval.py).
* One pytest at a time. Gate each on dev-box free commit >= 5.5 GB (`F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`). Check `(Get-PSDrive F).Free` before each run and stop below 4 GB.
* Keep TMP/TEMP and every `--basetemp` under your folder.
* Never wait more than about 4 minutes in one call.

## Why (read it): `F:/TanitAD-scratch/rollhist/PREREG_V8F_ROLLHIST_AMENDMENT_1.md` (sha256 87928487)
Lever C was built at 2bda9e1a, a child of 26bdcd3d. The checkpoints it fine-tunes (UE_s0.A, UE_s1.A) were trained at **cdbff971**, which is not an ancestor of 2bda9e1a. The trainer and model code differ by about 1,400 lines. The lever must run on the code that trained its init.

## Do
1. **Export** cdbff971: `git -C C:/Users/Admin/tanitad-push -c core.autocrlf=false archive cdbff971 stack taniteval`. Verify it blob by blob.
2. **Port** `git -C C:/Users/Admin/tanitad-push diff 26bdcd3d 2bda9e1a` (8 files):
   * New files (`rollhist_pair_check.py`, `v8f_t1_readout.py`, and the three tests): copy them unchanged.
   * `stack/tanitad/data/v8f_dagger.py`, `stack/scripts/v8f_dagger_bank.py`: their base blobs are identical at 26bdcd3d and cdbff971 (check it), so the diff applies as is.
   * `stack/scripts/v8f_mini_train.py`: re-apply the lever's 24-line change by hand to cdbff971's trainer. That is the two FLAG_CHECKS entries, the two argparse flags with `_readout_token_arg`, and the `declared` / `bad_dagger` clause. Change nothing else.
   * Add `stack/scripts/v8f_t1_eval_hb.py` (blob at 26bdcd3d; its sha256 must be 74fd8445...) and its test `stack/tests/test_v8f_t1_eval_hb.py`. The bank imports it, and cdbff971 lacks it.
   * Check every import of the ported files resolves inside the new tree.
3. **Controls, each named, with the value it must read:**
   * **D0:** with the new flags at their defaults, the trainer's built model state dict and every loss term are bit-identical to cdbff971's on a CPU smoke. Use the trainer's own `--selfcheck-only` or the B1 records' method.
   * **D1:** the bank generator with default flags writes npz and `bank.json` byte-identical to cdbff971's generator on a tiny fixture (B1's `--readout pinned --kappa-max none` identity).
   * **B1 tests:** `test_v8f_rollhist.py`, `test_rollhist_pair_check.py`, `test_v8f_t1_readout.py` and `test_v8f_t1_eval_hb.py` pass on the new tree.
   * **Mutations:** re-run B1's mutation table on the new tree (`F:/TanitAD-scratch/rollhist/build/b1/run_mutants.py`, `mutation_table_rollhist.json`). Every mutation goes RED.
   * **Regression:** the v8f / dagger / strat1 / dscale test files on the new tree against cdbff971, same scope as B1's `run_regression.py`. Report 0 regressions, or list each one.
4. **Gate kit:**
   * Copy `F:/TanitAD-scratch/rollhist/build/b3gate/` into your folder.
   * Point `dscale10k_check_rh.py` and its tests at the new tree. Change any constant that binds 2bda9e1a to a placeholder the MM fills with the new commit; list each one.
   * Re-run its 37 tests and the mutation. The UE identity test must still reproduce the banked UE_s0.B token field for field.
5. **Pod scripts:** list every line in `F:/TanitAD-scratch/rollhist/build/b2/*.sh`, `*.py` that names 2bda9e1a, `/root/rollhist/tree`, or the G-SUITE baseline. Do not edit the pod.

## Deliverables in `F:/TanitAD-scratch/rollhist/build/b4port/`
* `rollhist_cdb.patch` against cdbff971, with a per-file sha256 list;
* the test logs, the mutation table, D0 and D1;
* the patched gate kit;
* `REPORT_B4.json`;
* `HANDOFF_B4.md` (at most 800 tokens);
* `MANIFEST_SHA256.txt`.

Delete your own temp, logged in `F:/TanitAD-scratch/rollhist/DELETIONS.log`.

Hand back in at most 250 words.
