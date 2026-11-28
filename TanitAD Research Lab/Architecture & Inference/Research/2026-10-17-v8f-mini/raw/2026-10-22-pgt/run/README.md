# PREREG_V8F_PGT run, seed 0 (box 2, 2026-10-22)

The chain `code/box2_ops/chain_v8f_pgt.sh` ran at tip `14a27de` and stopped by its committed rule at seed 0 (FAIL-NO-MECHANISM). Seed 1 was not trained.

* `pgt_read_s0.json`: the committed read (`code/v8f_pgt_read.py --stage s0`). `check_inputs.json` ran on the real banked M2b files before training.
* `gate_M2b-pgt/selfcheck.json`: the in-process launch gate (PASS).
* `eval/`: M2b-pgt_s0's four evals (own code, absmask, GT-code oracle, all-latcons oracle), npz / json / log each.
* `runs/M2b-pgt_s0/`: summary, selfcheck, argv, train log. The checkpoint is not in git: `D:/Projects/TanitAD-artifacts/v8f_pgt/M2b-pgt_s0/ckpt_last.pt` (sha256 `982b3ff3…`).
* `box2_logs/`: the chain log, the guard log, every job log, and the wrapper.
  * `v8fpgt_after.out` holds the one hand-written `ZZV8FMINI-DONE` line the chain waited on. The WC predecessor's log ends `ZZV8FMINI-STOP wc read s0` because its original reader crashed at load; its committed seed-0 read was then run by hand with Amendment 1's reader and landed at `abccc45`. So the chain was pointed at this file with `AFTER_OUT` (see `run_chain_v8f_pgt.cmd`).
* `telemetry_v8fpgt.csv`: box-2 telemetry for the run.
