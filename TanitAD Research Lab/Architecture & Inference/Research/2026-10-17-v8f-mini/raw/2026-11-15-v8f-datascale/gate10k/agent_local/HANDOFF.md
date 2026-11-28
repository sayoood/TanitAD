# HANDOFF DSCALE10K block-1 gate (gate agent, 2026-11-17)

## State
Written and dry-run: the chain `/root/dscale10k/block1.sh` (NOT launched), `R_COLUMNS.md` (R not minted), the gate items. Detached on the pod (no agent waits):
1. `gate/bank_B_pilot.log`: a complete 4-shard bank of the B set from the pilot checkpoint (about 35 min, started 22:08Z pod time); `sentinel "bank4 done"`.
2. `gate/gpu_gate.sh` (waits for 1): reach probes B_s0, B_rep; the trainer's `--selfcheck-only` on the REAL stage-B argv of B_s0 and again with the swap mutated; U smoke bank + selfcheck; G-CKPT resume smoke. Ends `ZZ GPU-GATE DONE|FAIL ZZ` in `gate/gpu_gate.log`.
3. `gate/finish_gate.sh` (waits for 2): B_rep stage-A selfcheck, `stageb_items.py`, `assemble_evidence.py B_s0 B_rep U_s0`, **mints the stage-A tokens of B_s0, U_s0, B_rep** with `chk_arm.sh mint`, `verify_token.py` (the runner's refusal logic plus tampered variants), a dry stage-B mint on the pilot bank. Ends `ZZ FINISH-GATE DONE|FAIL ZZ` in `gate/finish_gate.log`.
4. `summary_watch.sh` writes `gate/GATE_RESULTS_BLOCK1.md`; `cleanup_watch.sh` then deletes the agent's scratch (log `gate/deletions_gate.log`).
Already MEASURED PASS with their RED regression arms: G-CLOCK, G-EVAL, G-CKPT in-process half, G-SUITE (identity with `/root/strat1/base`), G-HYG, C-INIT (step-0 parameters of B_s0 and U_s0 are equal), C-CACHE (14,512 of 14,512 md5), C-SESS (not computable). Dry run of the chain: 24 scenarios PASS. `mint_test_U.sh` (test evidence, scratch gate dir) was exercising the union mint (C-MERGE rebuild) when this was written: read `gate/mint_test_U.log`.

## Next step
`tail -n 3 /root/dscale10k/gate/{gpu_gate,finish_gate}.log`. If `FINISH-GATE DONE`: `bash F:/TanitAD-scratch/datascale/gate/pull_gate.sh`, read `gate/GATE_RESULTS_BLOCK1.md`, rule on `R_COLUMNS.md`, launch `block1.sh`. If a log says FAIL: the reason is on its last line; the failing step's log is next to it (`gate/probe2/*.log`, `gate/sb/*.log`, `gate/mint_<arm>.A.log`, `gate/resume.log`); fix, re-run only that step (`gate_items.py <item>`, `finish_gate.sh` after `gpu_gate.sh`).
Open risk: the in-chain stage-B mint and the real U bank (14,512 clips, about 2.3 h with 4 shards) are first exercised by the chain itself.

## Facts to keep
* Block 1 about 8.1 h ESTIMATED (stage A 3.97, stage B 0.47, banks 3.69 with the 111/358 factor), below 10 h.
* Checker patched on the pod (`study/dscale10k_check.py`, original `.pre_gate`, diff in `gate/new/PATCH_dscale10k_check.diff`); the MM lands it with the study folder (its sha256 is bound in every token).
* No banker exists: `bank_ok/<arm>.<stage>.sha256` is written by the MM's banker; until then the chain defers every deletion and the 60 GB guard decides.
* R_s0's argv `argv/R_s0.A.json` exists; its dry record (`dry/R_s0.A.dry.json`) does not.
