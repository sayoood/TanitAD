# HANDOFF RPT (addendum 4): instruments built, NOTHING launched on box 2, no ssh, no commit (2026-11-04)

Arm = R's frozen argv (empirical TRAIN-bank law, p 0.3, rejoin target, the seed's own bank + sha256) with `--out ...\M2b_s{s}_recft_RPT` and `--recover-dev-shape pertick` appended. Runs `RPT_0`, `RPT_1`; argv `argv/RPT_{0,1}.train.txt` (64 tokens). MEASURED by test: it equals R's `R_{s}.train.txt` token for token except token 7 (`--out`) and the two appended tokens; it is RP01's file with p 0.3, run name RPT and the two tokens. Old copies of every changed file: `_pre_rpt/<name>.pre_rpt`; old -> new sha256: `_pre_rpt/SHA_CHANGES.txt`.

## Created
* `box2_files/rnext_chain_rpt.cmd` (CRLF): differs from `rnext_chain_raprp01.cmd` ONLY at lines 2, 3, 6 (comments / launch line) and 23 (`for %%T in (RPT_0 RPT_1)`), pinned by test.
* `launch_rpt.sh` + `mk_launch_rpt.py` (generated from `launch_raprp01.sh`, asserted replacements). Refuses at line 6 without `RPT_PI_GO`; lock text names RPT; old `CHAIN_RNEXT.done` moved aside as `CHAIN_RNEXT.raprp01_<ts>.done`; prints `ZZRPT-LAUNCHED`. New STACK-SHIP block (after lock, preconditions and the rnext ship; before telemetry): local sha256 of each source against its pin, then per file: box-2 file moved aside to `<name>.pre_rpt_<ts>` (Move-Item), scp, re-hash on box 2; any failure restores every file shipped so far (new bytes kept aside as `.rpt_failed_<ts>`), then `die` (releases the lock).
* `mk_rpt_pins.py`: one-command re-pin (recomputes the three sha256, rewrites the SHIPPED block, regenerates the launcher; `--check` writes nothing). `rpt_stack/stack/tanitad/data/v8f_mini_data.py` = git mirror tip 76a0d73c, byte for byte (209da7a6).
* `read/rnext_read_chain_RPT.cmd` (LF, like its siblings), `read/logs/dryrun_RPT.out`, `logs/mutation_rpt.log`, `logs/test_rnext_chain_rpt.log`.

## Changed
* `box2_files/rnext_check.py`: EXTRA += RPT tags; KINDS["RPT"] (p 0.3, empirical, rejoin, shape pertick, table); `expected_tokens`; SHIPPED = trainer bda468fe, v8f_recover 0257ccb9, v8f_mini_data 209da7a6 (from `pertick/stack_patch` x 2 and `rpt_stack/`); DEFAULTS gains `"dev_shape": "constant"`, DEFAULTS_LINE 23 -> 28. Gate for RPT also reads: built `recover_dev_shape`, `info.recover.dev_shape`, `levers.recover_table.per_tick` (ticks, three per-tick columns) and `.counts` (kept, `dropped_per_tick_row_unreal`), `recover_probe.pertick_v_max_abs_err <= 1e-4`. The eight older tags: argv spec, registered levers and audit verdicts identical (test against the pre-RPT checker). NOTE: SHIPPED now binds the NEW trainer and library for every tag.
* `pull_rnext.sh` (two tags appended), `read/code/recover_read.py` (ARM_SPEC["RPT"], shape flag as a lever flag of RPT only, KR3 shape reads; other arms identical), `read/tests/test_recover_read_arms.py`, `read/rnext_read_chain.sh` (accepts RPT), `read/PINS.txt`, `tests/test_rnext_chain.py`, three `read/logs/pytest_*.log`. Known answer recomputed with the new reader (PASS, 0 differences over 2,599 leaves, 15 of 15 arrays bit-identical; old files in `_pre_rpt/raw_R_ka/`). `wake_rnext.sh`, `precheck_pulled.py` (arm-generic through ARM_SPEC; compares no trainer sha), `recover_next_combine.py`, `ka_rarm_r.py`: unchanged.

## Launch sequence (after the PI go)
1. `RPT_PI_GO="<HH:MM>" bash E:/Projects/TanitAD-artifacts/dagland/recover_next/launch_rpt.sh`
2. Waker: `RNEXT_SEEN=8 bash .../recover_next/wake_rnext.sh` (box 2 holds 8 RUN_*.done, 0 FAILED: RAPR x2, RTL x2, RP01 x2, RAPRP01 x2). It exits on the first new done ("9 of 6", cosmetic): re-arm with `RNEXT_SEEN=9`; chain end exits 0.
3. `bash .../pull_rnext.sh` (re-pulls all 10 checkpoints, ~5 GB; expect `bad=0`), then move `BOX2_BUSY.txt` aside.
4. `python -I read/code/precheck_pulled.py --r-arm RPT` (rc 0).
5. `bash read/rnext_read_chain.sh RPT --dry-run`, then `powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='E:\Projects\TanitAD-artifacts\dagland\recover_next\read\rnext_read_chain_RPT.cmd'}"`; result `read/raw/RECOVER_READ_RPT.json`, marker `raw/RNEXT_RPT.done`.

## Unverified
* The per-tick table build on the REAL bank and clips (`ClipStore.get_kin`) and the gate's key names were read from the patched sources and a stand-in selfcheck, never from a real `selfcheck.json`: a different key name FAILS the gate closed (RUN_RPT_*.FAILED), it does not pass silently.
* Box 2's other imported modules were not diffed (only the five named ones are known equal to the tip). The STACK-SHIP block is proven on a local fake box, not on box 2.
* The combiner's `ARMS` has no RAPRP01 and no RPT; `MANIFEST.txt` files not regenerated; `t_package_consistency_and_hygiene` keeps its pre-existing failure (LF read-chain `.cmd`).
* A test-time run of `launch_rpt.sh` (made before its PI-go guard existed) sat in the lock-wait loop for about 6 minutes: no ssh, no lock file created, process killed by PID.
