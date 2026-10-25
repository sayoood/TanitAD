# PREREG_V8F_KIND run (box 2, 2026-10-25 07:17–07:32, eval only)

`code/box2_ops/chain_v8f_kind.sh` ran at tip `009d1cba` on the PI's go: "Go", the answer to *"run box 2's next jobs in order: AR-2 now, then KIND"*.

* `kind_read.json` is the committed read: INTERACTIVE. `kind_facts.{py,out}` prints its reads with their intervals.
* `check_inputs.json` holds the banked B side's checks. `ckpt_sha256.json` holds the two checkpoints' sha256s, asserted before and after the evals.
* `eval/` holds the 30 evals (15 per seed: own, all, 6 single-kind, 6 leave-one-out, the union), each with its json, per-window npz and log.
* `box2_logs/` holds the chain logs, the guard, the dry-run, the wrapper and the waker.
  * `chain_v8f_kind.wait1_stopped.out` is the first launch. It waited behind AR-2 run 2 and stopped, as designed, when that chain stopped.
  * The relaunch waited behind DSCALE's finished chain instead.
