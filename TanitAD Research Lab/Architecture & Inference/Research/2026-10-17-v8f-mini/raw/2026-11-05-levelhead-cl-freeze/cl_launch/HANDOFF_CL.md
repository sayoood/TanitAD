# HANDOFF CL: launch package for arm CL (PREREG_LEVELHEAD_CL.md section 2), built 2026-11-05. NOTHING launched, no ssh, no commit, no contact with box 2.

Package: `E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_launch/`. Cloned from the RPT launch (`recover_next/`): same lock, telemetry (must be fresh), temperature guard, SAFE MODE (`%RUNSAFE%` + env.cmd), one job at a time,
sha256 re-hash on box 2, per-run launch-gate token, markers. State handed over: **W is the placeholder `__W__`**, so the chain launcher refuses locally until W is frozen. The W probe is launchable now.

## What differs from RPT (decisions for the MM to confirm)
* Own folder on box 2: `D:\tanitad\cl\` (checker `cl_check.py`, argv, .cmd files), own marker dir `D:\tanitad\runs\v8f_mini_full\cl\`, gate dir `gate_cl\`, outs `M2b_s{s}_recft_CL`, probe out `M2b_s0_wprobe_CL`. The RPT folder `D:\tanitad\rnext` is untouched.
* Four stack files over box 2's repo (old ones moved aside as `<name>.pre_cl_<ts>`, re-hashed on box 2): trainer **9ffdf1b4** and model **a3634276** (= `build/SHA_CHANGES.txt` `new`), plus `v8f_recover.py` 0257ccb9 and `v8f_mini_data.py` 209da7a6 (the RPT pins; both equal the blobs at remote tip c56689bb, checked 2026-11-05 with `verify-tip`).
  The trainer adds one stdlib import (`textwrap`) over the tip's; the model adds none.
* The launcher REFUSES when the box-2 lock exists (default `CL_WAIT_MAX=0`); `CL_WAIT_MAX=<s>` opts into the RPT-style wait. EVERY script that touches box 2 (both launchers, `wake_cl.sh`, both pulls, `wake_pull_cl.sh`) refuses without `CL_PI_GO`.
* The pulls move OUR lock aside on full success, only when the lock text is ours (`LEVELHEAD CL ...`); otherwise they leave it and say so.
* The chain launcher's local check (`cl_local_check.py chain`, before the lock and any ssh) also requires: `cl_prereg/W_PROBE.json` present, frozen W within 1 % of its `w_level_for_10pct_share`, the card on disk = the card the pins were made from (sha256), and the card names the W.
* Box-2 gate (`cl_check.py gate`) reads selfcheck.json independently of the trainer: init ckpt carried whole with EXACTLY the six `lvl_head.` / `e_level.` keys fresh, lever built on with the frozen W, token zero-init, vision-only source, every other lever off (C0's), and the level probe's controls (detach control, repeat, all-false, prior all EXACTLY 0.0; head and token gradients > 0; term reaches the loss).
  CL_0 also requires the step-0 level share at the frozen W in [0.09, 0.11] (the card's formula, same seed and probe batch as the W probe); CL_1's share is recorded in its token, not gated.
* The card writes `W = 0.1 x L_total / L_level`; the trainer's key is `(0.1/0.9) x L_without_level / L_level` (the level term is a tenth of the total that includes it). They agree when `L_total` means the total without the level term. The launcher uses the key; confirm that is the card's intent.

## Exact MM sequence (git bash; `CL_PI_GO` = the PI's go time; box 2 lock must be absent)
```
cd /e/Projects/TanitAD-artifacts/dagland/motion_c/cl_launch; PY=/c/Users/Admin/venvs/tanitad/Scripts/python.exe
ls /c/Users/Admin/box2_ship/BOX2_BUSY.txt                       # must say: No such file
$PY -I mk_cl_package.py check                                    # OK, every generated file is what the generator writes
$PY -I mk_cl_package.py verify-tip                               # GitHub only: the two unchanged files still equal their pins
export CL_PI_GO="HH:MM"
# 1. W PROBE (selfcheck only, about 20 min on box 2; the launcher returns after about 4 min)
bash launch_wprobe.sh                                            # last line: ZZCL-LAUNCHED wprobe   (exit 6 = local refusal, 5 = lock held, 2 = aborted and lock released, 3/4 = lock kept, read it)
bash wake_pull_cl.sh wprobe                                      # background task: waits for WPROBE.done, pulls, writes cl_prereg/W_PROBE.json, moves our lock aside (exit 0)
# 2. FILL W
$PY -I freeze_w.py                                               # prints W (4 significant figures) and the freeze line
$PY -I freeze_w.py --write                                       # writes cl_prereg/FROZEN_W.txt  (first line: W=<number>)
#    paste the printed freeze line into cl_prereg/PREREG_LEVELHEAD_CL.md (the pins bind the card's sha256, so edit the card BEFORE the next command)
# 3. FREEZE CARD
$PY -I mk_cl_package.py                                          # regenerates argv (W filled), cl_check.py, the FROZEN_W.txt copy, PINS.json
$PY -I mk_cl_package.py check; $PY -I cl_local_check.py chain    # OK, OK
sha256sum ../cl_prereg/PREREG_LEVELHEAD_CL.md                    # record in cl_prereg/FROZEN_SHA_CL.txt with the time, as the RPT FROZEN_SHA files do
CUDA_VISIBLE_DEVICES=-1 $PY -I tests/test_cl_launch.py           # all pass
# 4. CHAIN (CL_0 then CL_1, 3,000 steps each, about 38 min each plus gates)
bash launch_cl.sh                                                # last line: ZZCL-LAUNCHED chain
# 5. WAKER + PULL (background task; it re-arms itself after each run)
bash wake_pull_cl.sh chain                                       # exit 0 = both runs done, pulled, lock moved aside; 3 = a run FAILED (read the marker; re-arm with CL_SEEN=<dones> CL_SEEN_FAILED=<fails> bash wake_pull_cl.sh chain); 4 = guard or safety stop
```
## Check commands
* Probe: `cat ../cl_prereg/W_PROBE.json` (value, selfcheck sha256); `ls /e/Projects/TanitAD-artifacts/v8f-recover/cl_markers/` shows `WPROBE.done`.
* Chain on box 2 (read-only): `ssh tanitad-box2 "powershell -NoProfile -Command \"Get-ChildItem D:/tanitad/runs/v8f_mini_full/cl -Name; Get-Content D:/tanitad/logs/cl_chain.log -Tail 12\""`.
* After the pull: `cat /e/Projects/TanitAD-artifacts/v8f-recover/cl_markers/RUN_CL_{0,1}.done` (evidence: steps 3000, level_head on, w_level, ckpt sha256), `gate_token_CL_{0,1}.json` (binding: commit, argv sha256, W, shipped sha256, level_share_at_step0),
  `sha256sum -c SHA256SUMS.txt` inside `M2b_s{0,1}_recft_CL/`, and `ls /c/Users/Admin/box2_ship/` shows `BOX2_BUSY.cl_chain_done_*.txt` and no `BOX2_BUSY.txt`.
* A FAILED run: `cat .../cl_markers/RUN_CL_<s>.FAILED` names every failed check (gate reasons list the selfcheck keys).

## Evidence
* MEASURED (tests, `logs/test_cl_launch.log`; mutation proof `logs/mutation_cl.log`): counts are in the hand-back. The tests cover: CL's argv = C0's real `argv.json` (both seeds) with `--out` replaced and exactly `--level-head on --w-level W` appended; the probe argv = CL_0's with `--out`, `--w-level 1`, `--selfcheck-only`;
  checker pin / preflight / gate (25 deliberate-regression cases) / post / finish / stop; reader and freeze helper; wake scenarios; pulls with a local scp stand-in; every box-2 script refuses without `CL_PI_GO` (ssh/scp stubs record no call);
  both launchers end to end on a fake box (happy paths and failure paths, lock released or kept as designed); guard scripts under PowerShell; the .cmd files equal the RPT chain/run files after name normalisation.
* The gate stub is built from the REAL C0 `selfcheck.json` plus the level keys. The key `level_head_probe.w_level_for_10pct_share` exists in the new trainer (`v8f_mini_train.py` line 1370; record built at 1411; written to selfcheck.json at 2354-2355 when the head is on).
* UNVERIFIED: (1) a real-data selfcheck of CL's argv has never run (box 2 only): the gate reads `info.init_ckpt.{fresh_keys,declared_new_keys,n_mismatch_after_load}`, `levers.{level_head,w_level,level_token_zero_init,level_vision_only_source}` and the probe record as the build source writes them; a different shape FAILS CLOSED (a FAILED marker naming the key), it does not pass.
  (2) the seed-0 share window [0.09, 0.11] assumes the probe batch repeats between the probe and CL_0's gate; if not, CL_0 is marked FAILED with the measured share and the MM decides. (3) box 2's other imported modules are not diffed (as for RPT). (4) `launch.ps1`, `telemetry_long.ps1`, `env.cmd` on box 2 are used by path as RPT used them; the fake box models only the commands this launcher issues.
  (5) the probe's duration (about 20 min) is an estimate.
* Test hooks exist only for the dev box: `CL_SSH CL_SCP CL_PY CL_SRC CL_LOCK CL_WAIT_MAX CL_NAP CL_DEST CL_PREREG_DIR CL_SEEN CL_SEEN_FAILED CL_SLEEP CL_WAKE_N`; production sets none of them (the launcher prints a WARNING line for any it sees).
