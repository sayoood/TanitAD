# ACCGAIN10K_DEFS.md -- what each read, control and name of card PREREG_V8F_PROOF_ADD63 is computed from

Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions); never driving performance. Reader: `read_accgain10k.py` (numpy only), run on the pod by `podread_accgain.sh` after control e1. The card is `PREREG_V8F_PROOF_ADD63` and is not edited here.

## Tools

* Wrapper `v8f_t1_eval_reader_rule_rzg.py` = `v8f_t1_eval_reader_rule_rz.py` (sha256 prefix fa7207d72311b5cf, asserted at generation) plus the readout `causal_g` with `--acc-gain G` and `--path-exec`. Written by `make_accgain10k.py` with asserted edits (each old text exactly once). The pinned tool is imported by path and not edited.
* Speed column of `causal_g`: at the knot of executed step i the step speed s[i], at the half step before it s[i] - 0.5 x G x (s[i] - s[i-1]); the one-frame acceleration read at every executed knot is G times the executed one.
* Formula routing (for the identity controls a1, a2): G = 1.0 goes through the parent's `causal` formula (`causal_frames(..., "step")`) and G = 0.0 through its `causal_z` formula (`"zero"`), unchanged, so the dumps equal those of the earlier readouts array for array; every other G uses the general formula. `check_acc_gain.py` reports how far the general formula at G = 1 and G = 0 lies from those two (expected at most 1e-12).
* `--path-exec`: positions and headings from `backward_frames(...)` (the pinned frames function given the executed knots only), speed column as above. Both flags are refused without `--readout causal_g`; `--readout causal_g` is refused without `--acc-gain`.
* Readout log (`save_rlog`) keeps the existing fields and adds `acc_gain`, `path_exec`, `max_abs_acceleration_minus_gain_executed` (largest absolute difference between the one-frame acceleration read at the last executed knot and G times the executed acceleration, over all committed plans; control a3). Wrapper line difference against the parent: see the hand-back (diff line count).
* Run names (folder `t1/<name>_s<k>/`): `I10R0G0`, `I10R0G50`, `I10R0Gm25`, `I10R0G0X` (10 steps, both model seeds), `I10R0GID1` (control a1: gain 1, 10 steps, model seed 0), `I10R0GID0` (control a2: gain 0, 5 steps, model seed 0). Reused, not rerun: `I10R0` (R0), `I10R0C10` (R0C10), `I10R0Bk10` (R0Bk10), `I10R0Z5` (control a2 only).
* Queue `pod_accgain10k_queue.sh` (waits for `RULEFAST10K_DONE.txt` or `RULEFAST10K_STOP.txt`; `WAIT_MAX` seconds, default 86400): tool check (marker `ga4check`, output `accgain_test/CHECK_ACC_GAIN.txt`), `I10R0GID1` then compare `ga1cmp`, `I10R0GID0` then compare `ga0cmp`, then per model seed `I10R0G0`, `I10R0G50`, `I10R0Gm25`, `I10R0G0X`. The slot rule (MAXGPU 3, MINGB 28) and the run-name prefix `I10` are those of the template. Each run record `ticks/run_<arm>.txt` carries the checkpoint sha256 and the wrapper sha256.

## Reads (paired episode-cluster bootstrap of the headway and family readers, pooled over the two model seeds, per-seed values beside)

* D1 = `R0G50` minus `R0G0`; D2 = `R0C10` minus `R0G50`; ZP = `R0G0` minus `R0`; PX = `R0G0X` minus `R0G0`. Beside, naming nothing: `R0Gm25` minus `R0G0`, `R0Gm25` minus `R0Bk10`, `R0G0X` minus `R0`, and `R0G0`, `R0G50`, `R0Gm25`, `R0G0X` each minus `R0C10` (the pair `R0G50` minus `R0C10` is the swap of D2).
* Named cells: LONGITUDINAL `speed_mae_mps`, `along_mae_m`; LATERAL `heading_mae_deg`, `yaw_rate_mae_degps`; TACTICAL `ade_tac_m`. All ten family cells are written per pair; U, C, F from the headway reader.
* A cell reads BETTER or WORSE by the family reader's verdict; a family reads BETTER if every named cell of it reads BETTER, WORSE if every named cell reads WORSE, NOT-SEPARATED if none reads BETTER or WORSE, else MIXED.
* Reported without a bar: mean and largest absolute acceleration read per run (readout logs), and RS (share of the 2,399 windows whose last-step speed lies more than 5 m/s above the logged one) for the model rollout and the prior alone, per run and model seed, banked runs beside.

## Names (as section 6 of the card)

* Dose per family: `ACCGAIN10K-<FAMILY>-RISES-WITH-OWN-ACCELERATION` if the family reads WORSE for D1 and D2, `-FALLS-WITH-OWN-ACCELERATION` if BETTER for both, else `-NOT-MONOTONE`. U: RISES if U of D1 and of D2 is above 0 at both seeds with both pooled intervals above 0, FALLS if below the same way, else `ACCGAIN10K-U-NOT-MONOTONE`.
* ZERO per family (ZP): `ACCGAIN10K-ZERO-<FAMILY>-BETTER-THAN-PINNED-AT-2S`, `-WORSE-THAN-PINNED-AT-2S`, `-MIXED-AT-2S`, `-NOT-SEPARATED-FROM-PINNED-AT-2S`; U: `ACCGAIN10K-ZERO-U-BELOW-PINNED-AT-2S`, `-ABOVE-PINNED-AT-2S`, `-NOT-SEPARATED-FROM-PINNED-AT-2S`.
* Path per family (PX): `ACCGAIN10K-PATH-<FAMILY>-BETTER`, `-WORSE`, `-MIXED`, `-NOT-SEPARATED`.
* STRATEGIC: as note N1 of card ADD46 (T0 first tick; along the rollout NOT COMPUTED, n = 0, no instrument).

## Controls (all must pass; if one fails the reader prints the controls only: no name, no level, no difference)

* e1: `ENV_E1.json` of the pod read has PASS true.
* a1 / a2: `small_accgain/CMP_R0GID1.txt` / `CMP_R0GID0.txt` read `CMP files=141 differing=0` (against the banked `R0C10` / `R0Z5`, model seed 0).
* a3: per new run (ten) the readout log carries readout `causal_g`, the gain, the path flag, the interval, calls above 0, and both maximum differences at most 1e-6.
* a4: `CHECK_ACC_GAIN.txt` ends `ZZ CHECK-ACC-GAIN all hold: True`.
* a5: for every new 10-step run (nine) the first 10 executed steps of the model and of the prior equal those of the banked `R0` at the same model seed (largest difference exactly 0.0).
* a6: 141 dump files and 2,399 windows in each new dump; 2,399 windows and 642 lead windows in the headway reader.
* a7: the seven reused result files hash as recorded in `ACCGAIN10K_REUSED.json`.
* a8: reader controls of the headway and family readers.
* a9: checkpoint prefixes of the two models in each run record; the wrapper file the runs used, the copy the check ran with and the sha256 printed by the check are identical and start with the prefix fixed in the reader (`93879db8c0233867`).

## What was NOT exercised

* The reader `read_accgain10k.py` was never run on data (syntax check only). The queue, the chain, the podread and the waiter were not run; only `bash -n`. The tool check ran on the pod CPU (output `read_accgain/CHECK_ACC_GAIN.txt`); no rollout of this card has run and nothing about its results is known to these tools.
