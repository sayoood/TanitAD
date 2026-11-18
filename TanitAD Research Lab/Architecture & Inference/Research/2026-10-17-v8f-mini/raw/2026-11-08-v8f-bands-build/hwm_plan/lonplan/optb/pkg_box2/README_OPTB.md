# OPTB: box-2 launch package, ACTRSP Option B (fixed-K readout, frozen `wm.proj`, fine-tune from R35)

Spec: `../AMEND_OPTB.md` (sha256 `7cb0245c...`, frozen). Rulings: `../MM_RULINGS_OPTB.md`. Code: `../optb.patch` on `a13f6f16` (branch `agent/v8f-actrsp-20261116`); the MM commits it by plumbing on `agent/v8f-optb-20261117`.
Fork of the ACTRSP package (names `actrsp_*` became `optb_*`). **Nothing here has touched box 2**: it ran only over stub transports and a fake box folder (`tests/DRYRUN_RESULT.log`).

## Chain order (`optb_chain.sh`)
1. wait for box 2 free (`BOX2_BUSY.txt`; optional predecessor chain), claim it.
2. `ship`: the commit's trainer closure into the OPTB-owned tree `D:\tanitad\optb\repo` (needles incl. `--wm-rsp-readout`, `def load_rsp_probe`, `def bad_wm_freeze`, `_response_loss_fixed_k`).
3. `inputs` (`stages/optb_inputs.sh`): `ladder_R35_s{0,1}.pt` (from `C:/Users/Admin/t2v8f_pulls/ladder_R35_s{k}/ckpt_last.pt`) and `readout_K10_R35_s{0,1}.pt` (from `frozen/inputs/`) to `D:\tanitad\optb\inputs\`. Each local file must hash to the checker's pinned constant before the put, and the copy on box 2 (certutil) must equal it. Any mismatch refuses.
4. `pilot`: `PILOT_FTRSP_s0` (FTRSP_s0 argv + `--steps 200`, own out dir); the same job gates the full-length FTRSP_s0, FTRSP_s2, FTNULL_s0 argvs; 72 h rule; stop if s/step > 1.0 or `max_memory_allocated` > 6.5 GB (MM reads `PILOT.verified.json`). The pilot also exercises the real gate on R35's trained WM (the float64 ladder reader).
5. the ten runs in `RUN_ORDER`: FTRSP_s0, FTNULL_s0, FTRSP_s2, FTNULL_s2, FTNULL_s4 (chassis 0), then FTRSP_s1, FTNULL_s1, FTRSP_s3, FTNULL_s3, FTNULL_s5 (chassis 1). Each: disk, GPU-idle, argv pin, `--selfcheck-only` gate (BUILT read-back of readout, freeze, init, C3), per-arm token minted on box 2, training under the token, post check (C5: 141 of 141 Adam tensors, `wm.proj` bit-equal to the R35 checkpoint, non-WM tensors bit-equal; C2 log identity), bank to the dev box (sha256 against box 2), cleanup of its own gate dir, then bank PRIVATE to HF (`stages/optb_hfbank.sh`, `HF_HUB_DISABLE_XET=1`, token read in place).
6. A gate refusal of one arm writes `ARM_REFUSED_<tag>.txt` and the next independent arm goes on; **a refused chassis-0 arm stops the chain before chassis 1** (`STOP.txt`, reason `chassis0-failed`). Any other failure stops the chain. A failed HF bank leaves `HF_BANK_PENDING_<tag>.txt` and ends the chain in STOP (never DONE) after all runs.

## Launch (MM; READY, NOT RUN)
1. Land `optb.patch` plus this package on `agent/v8f-optb-20261117`; copy this folder to `C:/Users/Admin/optb_pkg`, blob-check against `PACKAGE_MANIFEST.sha256`.
2. `bash optb_waiter.sh snap` (baseline BEFORE launch).
3. `powershell -ExecutionPolicy Bypass -File launch_optb_chain.ps1 -Commit <40 hex> -HfRepo <owner/name>` (WMI, CreateFlags 0x08000200, `Git\usr\bin\bash.exe`).
4. `bash optb_waiter.sh wait` in the background (wakes on a new `STOP.txt`, `DONE.txt`, `ARM_REFUSED_*`, `HF_BANK_PENDING_*`, or a dead driver process); check `chain.log` grows within 60 s.
State: `C:/Users/Admin/optb_state`. Banks: `C:/Users/Admin/t2v8f_pulls/optb_<tag>/`. Sentinels only in their own files (`ZZ OPTB-STOP`, `ZZ OPTB-DONE`, `ZZ OPTB-ARM-REFUSED`).

## Pins and UNVERIFIED items
* `ARGVS_SHA256` of `frozen/argvs_optb.json` is pinned in `optb_check.py`; regenerate with `make_argvs_optb.py` and re-pin if any token moves.
* The exact trainable-parameter count (2,011,970 minus `wm.proj`) is UNVERIFIED: the checker requires 0 < n_trainable < n_wm equal at three reads; pin the number from the pilot's `selfcheck.json`.
* The HF repo name is an input (`-HfRepo`), not chosen here.
