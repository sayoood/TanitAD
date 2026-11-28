# PREREG_V8F_HIST run, seed 0 (box 2, 2026-10-23)

`code/box2_ops/chain_v8f_hist.sh` ran at tip `6b860b95` and stopped by its committed rule at seed 0 (FAIL-NO-MECHANISM); seed 1 was not trained.

* `hist_read_s0.json`: the committed read. `check_inputs.json`: the B side (banked M2b), checked before training. `gate_M2b-h10/selfcheck.json`: the gate (PASS).
* `eval/`: H s0's four evals. `runs/M2b_s0_h10/`: H's run.
* `box2_logs/`: the chain, guard and job logs, the wrapper, and the arming handoff.
* `hist_facts.{py,out}`: zero-GPU facts read after the run (H's families; GRID's and HIST's ADE6 gains window by window). EXPLORATORY.
* Checkpoint, not in git: the artifacts drive `<drive>:/Projects/TanitAD-artifacts/v8f_hist/M2b_s0_h10/ckpt_last.pt` (E: on 2026-10-25), sha256 `428a45847644b75e63bc5da4fe71a3e22bf381563fa79aa70ceeeb6738d5ed55`.
