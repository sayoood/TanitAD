# raw/2026-10-22-cz: PREREG_V8F_CZ stage 1 (the absent-slot oracle); GATE FAIL, no training

* **Run:**
  * box 2, PI go 2026-10-22, SAFE MODE, the temperature guard (max USB 37 °C);
  * chain `code/box2_ops/chain_v8f_cz.sh` at the remote tip `10078b5e`;
  * 4 / 4 evals, 07:06–07:07.
* **`cz_gate.json`:** the gate read (`code/v8f_cz_gate_read.py`): rows per checkpoint, absmask and the full-oracle deltas, the four-family blocks, `DECISION`, `GATE`, `next`.
* **`eval/eval_*-tacabs.{npz,json}`:** the new evals (`v8f_mini_eval.py --tac absmask`). Each records `tac_manipulation` with `absent_slot_frac` 0.568.
* **The gate's other inputs are not re-banked here.** They are byte-identical to `raw/2026-10-20-tac/eval/eval_*-tac{pred,orc}.npz` (git blob ids checked 2026-10-22).
* **`telemetry_v8fcz.csv`, `box2_logs/`:** the chain log, the guard log, the refresh log, the tacabs job log.
* **`cons_curve.py` / `.out`:** a zero-GPU diagnostic for the next lever: T's constraint loss over M1 / M2b training, in 2k-step bins, read from the banked runs' `train_log.jsonl`.
* **`mk_hold10.py`:** the hold builder for this land.
