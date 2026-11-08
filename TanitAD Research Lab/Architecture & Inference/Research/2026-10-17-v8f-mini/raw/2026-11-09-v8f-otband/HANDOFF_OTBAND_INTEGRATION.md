# HANDOFF_OTBAND_INTEGRATION (Master Mind, 2026-11-09)
`otband.diff` applies to commit `1faa1a068f9f46dcb9d230c6b05efc20dce2fc1c` (the ladder's commit), checked with `git apply --cached --check` on a scratch index. It does not apply to the branch tip `f2fd990c`, whose `v8f_mini.py` and `v8f_mini_train.py` have moved. It is banked here as a diff and is not part of `stack/`. The O/T retrain runs from a separate branch cut at that commit, after the ladder frees box 2.

Open before any real-data run (no real-data run and no training has taken place):
1. Stream C's `stok` dump and `stok_meta.json` (interface unverified). Blocks OB-BAND-TOK only.
2. `v8f_t1_eval.py` key sets must carry `stok` and `stok_ok` when `stok_dim > 0`. Blocks the OB-BAND-TOK eval only.
3. The v10 train release md5 goes into `ACCEPTED_V10_TRAIN_MD5`, and `--v10-md5` in the argvs carries the same value.
4. Launch gate: `launch_gate.py` has no profile for the mini trainer (AUDIT_MODEL F12). Settled before any arm launches, against the rule that no training starts without a PASS token.
The step count `--steps-mult 2` is ESTIMATED; no run time was measured.
