# raw/2026-10-22-dk: v8F-mini LONGITUDINAL distance keeping (`RESULT_V8F_DK.md`)

* **The instrument and its read** (zero GPU, dev box; the reader is `code/v8f_dk_read.py`, its test `stack/tests/test_v8f_dk.py`):
  * `b1_eval_lead_block_k30.npz` (+ `.report.json`): the B1 lead block rebuilt at K = 30 (6 s) by `taniteval/tools/build_lead_block_b1.py --k 30`, on the same four inputs as the banked K = 10 block (sha256-identical). The reader's default picks it up from here.
  * `dk_read.json`, `dk_read.out`, `dk_perwindow.npz`: the read, re-run by the Master Mind from a clean export of the tip `abccc45` on the repo's banked inputs. The distance-keeping stream's first read (from local copies of the same evals) matches it in every numeric field and byte for byte on the per-window file.
  * `test_v8f_dk.out`: the stream's test run (25 passed); re-run before landing, 25 passed.
  * `mutate_dk.py`, `mutation_dk.json`, `mutation_dk.out`: 14 mutants of `lead_metrics.py` and the reader, **14 / 14 killed**, re-run against the tip export from a copy of the script that points there and deletes nothing.
  * `k30_control.py`, `.out`: the independent check of the K = 30 block against the banked K = 10 block.
* **What the 2 s closing comes from** (EXPLORATORY; each script states its reading rule, written before its run):
  * `ttc_decomp.py` (+ `.json`, `.out`): the min-TTC deficit split by closing class.
  * `ttc_tail_ci.py` (+ `.json`, `.out`): the < 5 s and < 10 s TTC shares, paired episode-cluster bootstrap.
  * `dk_oracle_read.py` (+ `.json`, `.out`): the GT-code oracle and the all-but-stoplead oracle. Verdict `UPPER` for M2b.
  * `dk_oracle_loo.py` (+ `.json`, `.out`): leave-one-out over the six code groups. Verdict `latcons` for M2b and M1.
  * `latcons_mech.py` (+ `.json`, `.out`): along-track progress, speed, lateral position and heading at 2 s, T's latcons against the GT latcons.
* The oracle evals they read are banked at `raw/2026-10-20-tac/eval/` and `raw/2026-10-21-tacgrp2/eval/`; the standard evals at `raw/full/eval/`.
* `mk_hold13.py`: the script that built this land.
