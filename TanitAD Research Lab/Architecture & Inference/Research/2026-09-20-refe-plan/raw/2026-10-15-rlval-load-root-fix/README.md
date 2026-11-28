# `rl_val_eval.load_scores`: the memory defect fixed at its root (Master Mind, 2026-10-15)

**The defect.** `load_scores` indexed the `NpzFile` per row: `z["scores"][i]`. An `NpzFile` decompresses the WHOLE member on every `z[key]`, and the row
it returns is a view that keeps that fresh copy alive. So N sets retain N full copies. It had already cost two things:
- **the RL-S1 R0 kill monitor, SIGKILLed** (rc 137) before printing a line (`raw/2026-10-14-rls1-monitor-fix/`). It was worked around in
  `refe/rls1_checks.load_scores_light`;
- **the S1 read driver's numpy `MemoryError`**, about 33.5 GB per arm (`raw/2026-10-15-s1-read-driver/`). It was worked around with a shim that proves itself
  equal to the original on every run.

So the defect was being fixed at each caller while the source stayed broken. `rl_val_eval.py` ships with RL-S1's chain and runs there on every
arm (`rls1_pod.sh:559`).

**The fix.**
- **Code:** the six columns are read ONCE each (`lg, tk, st, rk, ck, sc = (z[c] for c in (...))`), then the loop indexes those arrays.
- **Result:** the returned dict is identical: same keys, ckpt, space and score bytes. A duplicate key and a bad space still raise.

**The tests.**
- `eval/test_rl_val_eval.py`:
  - **added check `npz_one_read_per_column`:** a counting `NpzFile` proxy shows every member is read exactly once, with the same dict;
  - **added RED mutation `npz_reread_per_row`:** it restores the old line, and the check goes RED.
  - Count: **31 → 32**.
- `refe/test_rls1_queue.py`:
  - **memory assertion (l.560).** It PINNED the defect ("the original retains N full copies", `p_old > 25 ×` the file). With the root fixed,
    it now asserts that **both** loaders stay under 4 × the file. The fixed original measures 1.47 MB against a 0.88 MB file.
  - **mutation "the original loader is back".** It lost its premise and is replaced by **"the per-row decode is back in the light loader"**,
    which must and does go RED.
  - Count: **101 → 101**.

**Unchanged counts on the fixed tree:**
- `test_aux_s1_readout` 63;
- `test_dump_scores` 16;
- `test_s1_read` 97 (its shim-equivalence check still holds against the fixed original);
- `test_rls1_chain` 392.

All suites were run on fresh `git archive`s of `94248ec`, with `TMP` on NTFS.

**What does NOT change.**
- No running chain is touched: S1, P2-s1 and P6 run from their own shipped trees.
- RL-S1 picks this up at its next ship, which is R0c's, if the PI says yes.
- `rls1_checks.load_scores_light` and the S1 driver's shim stay. They are now redundant but harmless, and both remain equivalence-checked.
