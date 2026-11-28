# CAUSLIM10K_DEFS.md -- what each read, control and name of card PREREG_V8F_PROOF_ADD61 is computed from

Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; no lead rule in this card); never driving performance. Reader: `read_causlim10k.py` (numpy only), run on the pod by `podread_causlim.sh` after control e1. The card is `PREREG_V8F_PROOF_ADD61` and is not edited here.

## Tools

* Wrapper `v8f_t1_eval_reader_rule_rzc.py` = `v8f_t1_eval_reader_rule_rz.py` (readouts of card ADD55) plus `--alim-form path|chord` (default `path`; the function `accel_limit_chord` is the text of `make_feasc10k.py`, checked against `v8f_t1_eval_reader_rule_ac.py` at generation). Written by `make_causlim10k.py` with asserted edits. The pinned tool is imported by path and not edited. `--alim-form chord` without the two limits is refused.
* The limit acts on every committed plan of the model (class `AccelLimited` around the model planner) and of the kinematic prior alone (`limited_floor`); the readout acts through the replaced frames function on every committed plan. A run with both therefore has both on the model rollout and on the prior rollout.
* Run names (folder `t1/<name>_s<k>/`): `I10R0CE5`, `I10R0CE10` (causal readout, chord limit 3.0 / 8.0, 5 and 10 steps, both model seeds), `I10R0CEIDc` (control c1: causal, limits 1e9 / 1e9, chord, 5 steps, model seed 0), `I10R0CEIDe` (control c2: pinned readout, limits 3.0 / 8.0, chord, 5 steps, model seed 0). Reused, not rerun: `I10R0`, `I10R0M5`, `I10R0C5`, `I10R0E5`, `I10R0E10`.
* Queue `pod_causlim10k_queue.sh` (waits for `ACCWIN2S10K_DONE.txt` or `_STOP.txt`; `WAIT_MAX` seconds, default 86400): tool checks (marker `cl9check`), `I10R0CEIDc` then compare `ce1cmp`, `I10R0CEIDe` then compare `ce1ecmp`, then `I10R0CE5` and `I10R0CE10` at both seeds. The slot rule of the template is unchanged. Each run record `ticks/run_<arm>.txt` carries the checkpoint sha256 and the wrapper sha256.

## Reads (paired episode-cluster bootstrap of the headway and family readers, pooled over the two model seeds, per-seed values beside)

* SP = `I10R0CE5` minus `I10R0M5`; speed error = family cell `LONGITUDINAL:speed_mae_mps`. RS = share of the 2,399 windows whose last-step speed of the model rollout lies more than 5 m/s above the logged one (finite rollouts), per model seed; bars 0.1768 / 0.1726.
* PARTS = `I10R0CE5` minus `I10R0C5`, and `I10R0CE5` minus `I10R0E5`; GC = `I10R0CE5` minus `I10R0`; PR = `I10R0CE10` minus `I10R0`, beside it `I10R0CE10` minus `I10R0E10`. U, C, F from the headway reader, four families from the family reader for every pair.
* Prior alone: RS and ADE6 of the `fl` rollout of `I10R0CE5` (model seed 0; the maximum difference between the two model seeds is printed), ADE6 difference against the prior of `R0M5`, `R0C5`, `R0E5` with the episode-cluster interval (2,000 draws, fixed seed). Reported, names nothing.
* r1 = share of block-boundary steps (first step of a block, chord acceleration against the last step of the block before) beyond the limit by more than 0.25 m/s2, from the executed track `cl` of the dumps, for `R0CE5` and the banked `R0E5`, extra for `R0CE10` and `R0E10`. r2 = the speed at the next tick (limit log `v0`) against the chord speed of the last executed knot of the block before (median, p99). r3 = executed steps whose lower bound was not reached on the path (limit log). r4 = inside-block share of the prior alone. None of them gates a name.

## Names (as sections 4 and 5 of the card)

* Model: `CAUSLIM10K-MODEL-STABILISES-1S` if SP speed error is below 0 at both model seeds, its pooled interval below 0, and RS of `R0CE5` at most the bar at each seed; `-SLOWS-CLIMB` if the SP condition holds and RS does not; `-NO-EFFECT` otherwise.
* Parts: `CAUSLIM10K-BELOW-BOTH-PARTS` if U is below 0 at both model seeds with the pooled interval below 0 in both PARTS pairs; else `-NOT-BELOW-BOTH-PARTS`.
* GC on U: `-U-ABOVE-2S` / `-U-BELOW-2S` if the sign is the same at both seeds and the pooled interval lies on that side of 0; else `-U-NOT-SEPARATED-FROM-2S`.
* PR: `-COSTS-AT-2S` if a pooled interval of U, C or F lies above 0 or a family cell reads WORSE; `-HELPS-AT-2S` if none of that and an interval lies below 0 or a family cell reads BETTER; else `-NEUTRAL-AT-2S`.
* STRATEGIC: as note N1 of card ADD46 (T0 first tick; along the rollout NOT COMPUTED, n = 0, no instrument).

## Controls (all must pass; if one fails the reader prints the controls only: no name, no level, no difference)

* e1: `ENV_E1.json` of the pod read (leaf-for-leaf equality with the banked dev-box read of card ADD46) has PASS true.
* c1 / c2: `small_causlim/CMP_R0CEIDc.txt` / `CMP_R0CEIDe.txt` read `CMP files=141 differing=0` (cmp_dumps.py of the template, file for file against the banked `R0C5` / `R0E5` of model seed 0).
* c3: per run (six runs) the readout log and the limit log carry the readout, the interval, the limits and the form `chord`; the result file carries the interval; the plan rows of the limit log add up to 2,399 x (30 / interval) for the model and for the prior.
* c4: no chord speed in a limit log (model and prior) leaves the limit by more than 1e-6 m/s.
* c5: inside the executed blocks of `R0CE5` and `R0CE10` (both seeds) the share of steps beyond a + 0.25 or below -(8.0 + 0.25) m/s2 is at most 0.01.
* c6: causal runs: maximum speed difference and maximum acceleration difference of the readout log at most 1e-6 (the speed at the tick is the last executed step speed, the acceleration the executed one-step acceleration); the pinned control run is reported, not barred.
* c7: 141 dumps per run, 2,399 rows per tick, first-tick speed equal to the dump; counts of the headway reader (2,399 windows, 642 lead windows); the ten reused result files hash as recorded in `CAUSLIM10K_REUSED.json` (the reader reports how many differ); RS (model, prior), prior ADE6 where printed, and levels U C F of `R0`, `R0M5`, `R0C5`, `R0E5`, `R0E10` equal the earlier reads (tolerance 5e-5).
* c8: reader controls of the headway and family readers; checkpoint hash prefixes of the two models in each run record.
* c9: `TEST_ACCEL_LIMIT_CHORD_RZC.txt` ends `ZZ TEST-ACCEL-LIMIT-CHORD PASS 24/24`, `CHECK_TICK_READOUT_RZC.txt` ends `ZZ CHECK-TICK-READOUT-RZC all hold: True`; the wrapper file the runs used, the copy the checks ran with and the sha256 printed by both checks are identical and start with the prefix fixed in the reader; every run record carries that sha256.

## What was NOT exercised

* The reader `read_causlim10k.py` was compiled and its `--help` run on the pod; it was not run on any data. The queue was dry-run (DRY=1) on the dev box only; the chain, the podread and the waiter were not run.
* The two tool checks ran on the pod in the folder `causlim_pretest/` (CPU, no rollout, no data of the card); the queue repeats them in `causlim_test/` before the first rollout.
* The known values of the reused runs in the reader were copied from the dev-box copies of the earlier reads (`read_handover/pod`, `read_feasc/pod`), not recomputed here. The chord limit has no check on a bent path that is a model plan of this card; the 24 checks use straight plans and the bent executed tracks of the banked `R0M5` dump as path material.
* No rollout of this card has run; nothing about the results of the card is known to these tools.
