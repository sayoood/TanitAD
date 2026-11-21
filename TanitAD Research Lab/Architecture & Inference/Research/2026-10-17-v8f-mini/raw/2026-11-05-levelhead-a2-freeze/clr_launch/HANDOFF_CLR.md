# HANDOFF CLR: launch package for arm CLR (`motion_c/a2_prereg/PREREG_LEVELHEAD_A2.md`, section "Arms"), built 2026-11-05. NOTHING launched, no ssh, no commit, no contact with box 2.

Package: `E:/Projects/TanitAD-artifacts/dagland/motion_c/clr_launch/`. Cloned from the arm-CL launch package (`cl_launch/`, which ran cleanly today): same lock, telemetry (must be fresh), temperature guard, SAFE MODE (`%RUNSAFE%` + env.cmd),
one job at a time, sha256 re-hash on box 2, per-run launch-gate token, markers. Final state: there is no placeholder and no probe. W is CL's frozen 0.2027 (`cl_prereg/FROZEN_W.txt`), the card is frozen (`a2_prereg/FROZEN_SHA.txt`).

## What arm CLR is
* CLR_s{0,1} argv = RAPR-P01's `argv.json` (`v8f-recover/M2b_s{s}_recft_RAPRP01/argv.json`, copied to `rapr_argv/`, sha256 in `PINS.json`) with `--out` -> `...\M2b_s{s}_recft_CLR`, then `--level-head on --w-level 0.2027` appended. Nothing else changes.
  The recovery tokens are RAPR-P01's: `--recover-p 0.1 --recover-target rejoin --recover-rejoin-steps 10 --recover-seed {s} --recover-dev-law apriori` (seed 1 carries `--recover-seed 1`, as RAPR-P01's own argv does).
* Shipped stack: the same four files as CL, all four equal the blobs at remote tip `219f6c76` (`mk_clr_package.py verify-tip` re-checks against GitHub): trainer 9ffdf1b4, model a3634276, `v8f_recover.py` 0257ccb9, `v8f_mini_data.py` 209da7a6. Box 2's old files are moved aside as `<name>.pre_clr_<ts>`, re-hashed on box 2.

## What differs from CL (decisions for the MM to confirm)
* Own folders on box 2: `D:\tanitad\clr\` (checker `clr_check.py`, argv, .cmd files, `FROZEN_W.txt`), markers `D:\tanitad\runs\v8f_mini_full\clr\`, gate dir `gate_clr\`, outs `M2b_s{s}_recft_CLR`. The CL folder and the RPT folder are never touched.
* The chain-launcher's local check (`clr_local_check.py chain`, before the lock and any ssh) requires: `cl_prereg/FROZEN_W.txt` = the W in both argv files, the pins and the shipped copy; `cl_prereg/W_PROBE.json` present with the frozen W within 1 % of its `w_level_for_10pct_share`;
  the A2 card on disk = the card the pins were made from (sha256) = the hash `a2_prereg/FROZEN_SHA.txt` records, and it names `--w-level 0.2027`; RAPR-P01's argv copies = their pinned bytes.
* Box-2 gate (`clr_check.py gate`) reads selfcheck.json independently of the trainer, as CL's gate did (init ckpt carried whole with EXACTLY the six `lvl_head.` / `e_level.` keys fresh, level lever built on at W, token zero-init, vision-only source, the level probe's controls EXACTLY 0.0, head and token gradients > 0, term reaches the loss), and with these changes:
  the recovery lever must equal RAPR-P01's own gate values (the BUILT perturber: p 0.1, law apriori, target rejoin, 10 rejoin steps, recover_seed = the run's seed; `recover_probe.pass`; NO deviation table, NO bank, `levers.recover_table` None, `info.recover` absent);
  every other lever as C0's (dagger_p, launch_intent_w, ego_perturb_p all 0.0, no dagger bank, w_acc 49.6969, p_gt_start = p_gt_end = 0.2, kin_mode yaw; the declared-vs-built check covers the rest).
* Step-0 level share at the frozen W is gated in [0.08, 0.12] at seed 0 (CLR_0); at seed 1 (CLR_1) it is recorded in the token, not gated. The share formula is CL's: W x term / (step0_total_without_level + W x term).
* Post-run check: `recover_n > 0` (RAPR-P01's KR1: the recovery really perturbed samples) replaces CL's `recover_n == 0`; `summary.level_head == on`, `w_level == 0.2027`, steps 3000, argv.json = the pinned argv, KP1 data hashes.
* Same refusals as CL: the launcher refuses when the box-2 lock exists (default `CLR_WAIT_MAX=0`; `CLR_WAIT_MAX=<s>` opts into waiting); EVERY script that touches box 2 (launcher, waker, pull, wake_pull) refuses without `CLR_PI_GO`; the pull moves OUR lock aside on full success, only when the lock text is ours (`LEVELHEAD CLR two-run chain ...`).

## Exact MM sequence (git bash; `CLR_PI_GO` = the PI's go time; box-2 lock must be absent; CL and RPT jobs must be finished)
```
cd /e/Projects/TanitAD-artifacts/dagland/motion_c/clr_launch; PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
ls /c/Users/Admin/box2_ship/BOX2_BUSY.txt                       # must say: No such file
$PY -I mk_clr_package.py check                                   # OK, every generated file is what the generator writes
$PY -I mk_clr_package.py verify-tip                              # GitHub only: all four shipped files still equal their pins at the remote tip
$PY -I clr_local_check.py chain                                  # local check OK (chain)
CUDA_VISIBLE_DEVICES=-1 $PY -I tests/test_clr_launch.py          # every test PASS
export CLR_PI_GO="HH:MM"
# 1. CHAIN (CLR_0 then CLR_1, 3,000 steps each, gate per run)
bash launch_clr.sh                                               # last line: ZZCLR-LAUNCHED chain   (exit 6 = local refusal, 5 = lock held, 2 = aborted and lock released, 3/4 = lock kept, read it)
# 2. WAKER + PULL (background task; it re-arms itself after each run)
bash wake_pull_clr.sh chain                                      # exit 0 = both runs done, pulled, lock moved aside; 3 = a run FAILED (read the marker; re-arm with CLR_SEEN=<dones> CLR_SEEN_FAILED=<fails> bash wake_pull_clr.sh chain); 4 = guard or safety stop; 7 = pull failed
```
## Check commands
* Chain on box 2 (read-only): `ssh tanitad-box2 "powershell -NoProfile -Command \"Get-ChildItem D:/tanitad/runs/v8f_mini_full/clr -Name; Get-Content D:/tanitad/logs/clr_chain.log -Tail 12\""`.
* After the pull: `cat /e/Projects/TanitAD-artifacts/v8f-recover/clr_markers/RUN_CLR_{0,1}.done` (evidence: steps 3000, level_head on, w_level, recover_n, recover_share, ckpt sha256), `gate_token_CLR_{0,1}.json` (binding: commit, argv sha256, W, shipped sha256, level_share_at_step0),
  `sha256sum -c SHA256SUMS.txt` inside `M2b_s{0,1}_recft_CLR/`, and `ls /c/Users/Admin/box2_ship/` shows `BOX2_BUSY.clr_chain_done_*.txt` and no `BOX2_BUSY.txt`.
* A FAILED run: `cat .../clr_markers/RUN_CLR_<s>.FAILED` names every failed check (gate reasons list the selfcheck keys and values).

## Evidence
* MEASURED (tests, `logs/test_clr_launch.log`; mutation proof `logs/mutation_clr.log`; counts are in the hand-back): CLR's argv = RAPR-P01's real `argv.json` (both seeds) with `--out` replaced and exactly `--level-head on --w-level 0.2027` appended, and RAPR-P01's non-recovery tokens equal C0's real `argv.json` up to `--recover-p`;
  checker pin / preflight / gate (negative cases for each recovery lever, every other lever, each level-probe control and the share window) / post / finish / stop; local check refusals (card, W, probe, FROZEN_SHA record, argv copies, stack bytes);
  the waker scenarios; the pull and `wake_pull_clr.sh chain` end to end with local ssh and scp stand-ins; every box-2 script refuses without `CLR_PI_GO` (ssh/scp stubs record no call); the launcher end to end on a fake box (happy path on the REAL package, failure paths, lock released or kept as designed); the guard script under PowerShell; the .cmd files equal the RPT chain/run files after name normalisation.
* The gate stub is built from two REAL files per seed: CL's `gate_selfcheck.json` (the shipped trainer with the level head on at W 0.2027, real `level_head_probe`) and RAPR-P01's `gate_selfcheck.json` (the real recovery blocks).
* UNVERIFIED: (1) a real-data selfcheck with BOTH levers on (recovery p 0.1 and the level head) has never run (box 2 only). The shipped trainer applies the level label stage before the recovery stage by design (`level_stage` is documented to run before `rcv.apply`), and the gate reads the keys as the two real records write them; a different shape FAILS CLOSED (a FAILED marker naming the key), it does not pass.
  (2) the seed-0 share window [0.08, 0.12] assumes the probe batch is CL's; if not, CLR_0 is marked FAILED with the measured share and the MM decides. (3) box 2's other imported modules are not diffed (as for RPT and CL). (4) `launch.ps1`, `telemetry_long.ps1`, `env.cmd` on box 2 are used by path as RPT and CL used them.
  (5) the chain's duration (ETA text in the lock, about 2.5 h) is an estimate from CL's and RAPR-P01's runs, not measured for the combination.
* Test hooks exist only for the dev box: `CLR_SSH CLR_SCP CLR_PY CLR_SRC CLR_LOCK CLR_WAIT_MAX CLR_NAP CLR_DEST CLR_PREREG_DIR CLR_W_DIR CLR_SEEN CLR_SEEN_FAILED CLR_SLEEP CLR_WAKE_N`; production sets none of them (the launcher prints a WARNING line for any it sees).
