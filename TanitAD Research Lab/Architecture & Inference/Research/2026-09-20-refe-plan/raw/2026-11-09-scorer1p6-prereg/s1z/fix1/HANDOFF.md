# HANDOFF fix1 (S1Z regressions). STATUS: PARTIAL. Two fixes staged, NOT yet verified by a test run. 2026-11-10

## Why unfinished
The 6 GB free-RAM gate never opened: MEASURED free (avail) RAM 3.2-3.8 GB of 31.8 GB for ~2 h (three other python jobs at 2.17 GB each). No test of mine ran. The 52-test pass, results_fix.json and the rls1_queue reruns are NOT done.

## Root causes (from source reading; UNVERIFIED by a rerun)
1. test_dump_scores "parser: exactly ONE new dest": the test enumerates later edits' default-OFF dests (aux, epo, hn, pair) and subtracts them; S1Z's four dests (scorer_finetune.py:1527-1530 --twin-w/-ade/-topk/-heads) were not in the list. Fix: `twin_dests` added to the subtraction, plus the `(dests & twin_dests) in (set(), twin_dests)` ALL-or-none check and twin_w==0.0 / twin_heads=="" default checks (same form as pair_dests). Nothing loosened.
2. test_rl_epo off_bit_identical (6 names) and p2/* : the OFF-path harness builds `a` with fixture `a_ns` (test_rl_epo.py:282), a SimpleNamespace with pair_w but NO twin_w. S1Z's train_losses (scorer_finetune.py:1880) evaluates `a.twin_w != 0.0` -> AttributeError -> the check counts as failed. The trainer's OFF path adds nothing (guard `twin_w is not None or a.twin_w != 0.0` skips the term; no zero tensor is added to lB), so I judge the trainer correct and the fixture stale, exactly as when pair_w was added. Fix: `twin_w=0.0` in a_ns. If the rerun still shows off_bit_identical RED, the cause is something else and the trainer must be diffed (peel(trainer) vs tip blob f5c8b769 is already pinned by test_s1z_layer).
3. test_rls1_queue "returned before S1 was over": timing (S1 stand-in sleeps 4 s, new run took 3 s, base run 751 s under load). Not touched, not rerun. Suspected load flake; rerun alone twice.
4. test_s1aux_chain and test_sft4r_chain: FileNotFoundError on `raw/...` files (S1_DATA_AUDIT.md, r2_smoke2_g3mut.log); the runner's export holds only refe/eval/code/splits, so base fails the same way (environmental, expect base rc 1; confirm from results.json base rows).

## Deliverables here
files/test_dump_scores.py, files/test_rl_epo.py (LF), s1z_fix1_vs_d253dd29.patch, CHANGED.txt.
## Exact next steps
cd tree copy of the S1Z export with these two files laid over; run, one at a time with >= 6 GB free, env as run_suite.py: test_dump_scores, test_rl_epo, test_rls1_queue (twice), then all 52 (selection in run_suite.py), compare to base rows in F:/tanitad_scratch/s1zsuite/results.json; write results_fix.json. Scratch F:/tanitad_scratch/s1zfix/ (tree copy, run1.py runner, tmp) is left for continuation: delete it after.
