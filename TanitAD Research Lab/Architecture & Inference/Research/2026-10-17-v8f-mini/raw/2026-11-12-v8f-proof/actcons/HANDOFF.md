# HANDOFF: ACTCONS (`--w-act-cons W`, `--act-cons-max M`, card ADD37), 2026-11-15

State: DONE on the CPU, nothing trained on real data (no git, pod, box 2). Folder `F:/TanitAD-scratch/v8proof/actcons/`. Base = LEADCOND tree (trainer `85035ca4...`, 12 pins verified on both copies). Result `work/tree_patch`. File:line facts and every departure: `ACTCONS_DEFS.md`.

## Changed (sha256 in `PINS.txt`)
* new `stack/tanitad/models/act_cons.py` (the term); changed `stack/scripts/v8f_mini_train.py` (flags, G-DVB, `bad_act_cons`, `act_cons_probe`, loop block, log keys `act_cons` / `act_cons_n`, `levers.w_act_cons` / `act_cons_max`); new `stack/tests/test_v8f_act_cons.py` (14 tests).
* `v8f_mini.py` and all other pins unchanged: no module, no state-dict key, no cfg field. `ACTCONS.patch` vs `work/tree_base` (replay byte-exact); `patched/` has the three files.

## Checks (`checks/*.json`)
a1 identity (shapes G, LS, LSFT = LFT shape, LSOWN; 100 steps; diff 0.0), a2, a3 (planted 2.2e-7; mutations red), a4 smoke (fine-tune, cfg and keys equal base, strict load, t1_eval runs, deleted total line red). Pytest: `tests.log`.

## Pod
`cp -r tree_leadcond tree_actcons; patch -p1 --binary < ACTCONS.patch`; check the `PIN|` lines of `ARMS_ACTCONS.txt` (LA_s0/s1 = LFT argv + `--w-act-cons 1.0`; LA01 = 0.1); read `act_cons_probe` of LA_s0 first (`woken` must be false).

## Open points
* Real-data magnitude and valid share of the term are first seen in the pod selfcheck.
* Departures: `frame_ok` at s added to valid; gap range inclusive; one offset per sample shared by 3 starts; term in the trainer not `model.loss` (G-LIVE `live_check` omits it); float32 under amp.

## Temp (delete refused): `work/tmp` 1.2 GB, `work/pt` 0.5 GB, `work/mut`, `work/tmp/replay`. Kept: `work/tree_base`, `work/tree_patch`, `work/edits`.
