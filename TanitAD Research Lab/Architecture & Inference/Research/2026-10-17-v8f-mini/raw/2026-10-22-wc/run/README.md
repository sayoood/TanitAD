# raw/2026-10-22-wc/run: PREREG_V8F_WC seed 0 (STOP, FAIL-MECHANISM-OK); seed 1 not trained

* **Run:**
  * box 2, PI go 2026-10-22, SAFE MODE, the temperature guard (max USB 45 °C), boot unchanged;
  * chain `code/box2_ops/chain_v8f_wc.sh` at the remote tip `f135c49`; launched 07:59;
  * launch gate PASS 08:04; M2b-wc_s0 trained 08:04–12:04 (19,164 steps); evals 12:04–12:07.
* **`box2_logs/chain_v8f_wc.out`** ends `ZZV8FMINI-STOP wc read s0`: the original reader crashed at load (`KeyError: tac_code`) on M2b's banked 2026-10-20 GT oracles, which predate `tac_code`. No number was computed.
* **`wc_read_s0.json`:** the pre-registered seed-0 read, by Amendment 1's reader (the `code/v8f_wc_read.py` landed with this commit), run on the dev box over the chain's own eval dir.
  * Amendment 2's `eval_M2b_s*-tacorc-all-goal.npz` supplied M2b's GT constraints (`gt_source`).
  * Those files are banked at `raw/2026-10-21-tacgrp2/eval/` and are not re-banked here.
* **The M2b side** (`eval_M2b_s*.npz/.json`, `-tacabs`, `-tacorc`) is banked at `raw/full/eval/`, `raw/2026-10-22-cz/eval/` and `raw/2026-10-20-tac/eval/`.
* **`eval/`:** the three M2b-wc_s0 evals: the standard eval (`--project --families`), absmask, and the GT oracle.
* **`runs/M2b-wc_s0/`:** summary, selfcheck, argv, train_log, train.log. The checkpoint is at `D:/Projects/TanitAD-artifacts/v8f_wc/M2b-wc_s0/ckpt_last.pt` (md5 `86b1701075363dbaddf8c193e1330d86`), not in git.
* **`gate_M2b-wc/selfcheck.json`:** the in-process launch gate, asserted again by the chain.
* **`telemetry_v8fwc.csv`:** box-2 telemetry for the run.
