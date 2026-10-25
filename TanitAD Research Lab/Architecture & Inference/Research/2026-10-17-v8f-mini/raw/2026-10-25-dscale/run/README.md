# PREREG_V8F_DSCALE run, seed 0 (box 2, 2026-10-25)

`code/box2_ops/chain_v8f_dscale.sh` ran at tip `2d85add1` and wrote its committed read: NOT DATA-LIMITED. Seed 0 only, as pre-registered.

* `dscale_read.json`: the committed read. `check_inputs.json`: the B side, checked before training. `gate_M2b-df50/selfcheck.json`: the gate (PASS).
* `eval/`: Q's four evals. `runs/M2b_s0_df50/`: Q's run (no checkpoint in git).
* `manifest_train.jsonl`: box 2's production train-cache manifest (4,572 rows, `source=local256 / bf16 / cuda`), the evidence for R68.
* `box2_logs/`: the chain, guard and job logs, the wrapper, and the handoff.
* `dscale_facts.{py,out}`: the read's numbers. `accel_all_arms.{py,out}`: accel and speed MAE of every banked seed-0 arm against M2b_s0 (R69). EXPLORATORY.
* Checkpoint, not in git: the artifacts drive, `<drive>:/Projects/TanitAD-artifacts/v8f_dscale/M2b_s0_df50/ckpt_last.pt` (E: on 2026-10-25), sha256 `242c92cbb5c8a6c0f85a49d2193dc846359a8537661ff66a21fa250484683813`.
