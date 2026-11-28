# PREREG_V8F_ROLLHIST Amendment 1: the code of record moves to UE stage A's lineage (written before any bank, pilot, run or number)

Card: `PREREG_V8F_ROLLHIST.md`, sha256 781eea87 (landed c0a97279). This amendment is FROZEN at the sha256 beside it. Written by the MM on 2026-11-20 after 21:16Z. The chain stopped at its first gate, G-SUITE, at 21:14:28Z, before any GPU step, bank, pilot or number existed.

## What was found (MEASURED, git at the C: mirror)
* UE stage A trained at **cdbff971** (LOOP_STATE, the bank record of UE_s0/A and UE_s1/A). STAGEB's tree is the same commit. The G-SUITE baseline on the pod, `/root/strat1/junit/base`, is the cdbff971 run.
* The code of record 2bda9e1a is a child of 26bdcd3d, and **cdbff971 is not its ancestor**. The two share only 6a3fa0d7.
  * Between cdbff971 and 2bda9e1a, `v8f_mini_train.py` differs by 1,127 lines, `v8f_mini.py` by 265, `v8f_mini_data.py` by 99 and `v8f_dagger.py` by 39. 65 test files differ.
* So G-SUITE refused correctly ("base files missing in the new tree"). The freeze record's ruling also fails at 2bda9e1a: five gate checks are INHERITED from UE stage A because "lever C changes the bank only, not the model". That holds only on the code that trained UE stage A.

## The amendment
1. **Code of record:** a new commit = **cdbff971 + lever C's change** (the diff 26bdcd3d..2bda9e1a, 8 files) + `stack/scripts/v8f_t1_eval_hb.py` and its test.
   * The bank imports `v8f_t1_eval_hb.py`, the executor of record (blob sha256 74fd8445), and cdbff971 lacks it.
   * The lever is unchanged: the same flags, readout token `causal_g1x`, bound 0.2 and bank-generator semantics.
   * `v8f_dagger.py` and `v8f_dagger_bank.py` have byte-identical base blobs in both lineages. Only the trainer's 24-line declaration change is re-applied by hand.
2. **Unchanged:** arms, argv templates, readout, cells, bars, outcomes and cost. RH and B0 still run from one commit.
3. **Restored:** the freeze record's INHERITED-evidence ruling now holds as written, because the model, loss, label and clock code are UE stage A's.
4. **Before relaunch:**
   * the port's B1 tests and mutation table on the new tree;
   * a scoped regression against cdbff971;
   * the bit-identity of the defaults (trainer state dict and losses, and the bank generator with default flags) against cdbff971;
   * the gate kit's 37 tests against the new commit;
   * a fresh pod tree, verified blob by blob.
5. Class of the defect: a lever built on a different code lineage from the checkpoint it fine-tunes. The argv-string checks (render, pair) cannot see it. G-SUITE's baseline comparison caught it.
