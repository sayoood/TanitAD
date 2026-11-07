# HANDOFF: stream YAXD, inference-only ax transforms (P1) on u400 (written 2026-11-07 ~10:00; next agent = a FRESH Sonnet agent)

`$Y` = `E:/Projects/TanitAD-artifacts/yawrate_wp/YAXD/`; run root `$R` = `C:/Users/Admin/yaxd_run/` (smoke/, full/, tokens/). Python `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`. Nothing in the repo, YAX, YF3 or the pod was touched; nothing staged (MM lands).

## Frozen design
* `$Y/PREREG_AXD.md` sha256 **1d33a9ed9df630063b9d4cfbec47027b5e8102a2884ec4650f87e57cdc19b6ba** (`PREREG_AXD.sha256`), frozen before any arm was evaluated. Code pinned in `CODE_PIN_AXD.json` (5 files); the chain's `pins` stage REFUSES on a mismatch, so a code fix = re-pin + `AMENDMENT_n.md`.
* P1 arms: A0 (seam's own call) / IDT, IDTm (controls) / AX0, CLAMP0, CLAMPm05, SCALE05, SMOOTH4; gain arms +-0.5 on the 90 CRUISE tokens. Rule: PREREG section 5 (PDMS NI -0.5 and down-gain <= 0.55, CI-hi < 0.92 and < u400's CI-lo, <= half of u400's, D_lon <= 0.3).

## State: ONE detached chain (WMI, cmd PID 44948 -> `axd_run_all.py`), launched 09:55
1. SMOKE on CPU, no GPU lock (`--tag smoke --device cpu --no-lock`; mechanics only): the 8-token A5 log `2021.09.29.18.19.40_veh-28_00331_00426` (3 CRUISE), then 4 RED seams on the same log (`transform_not_applied`, `ax0_leak`, `idt_perturbed`, `memo_stale`; each must turn its gate red or the smoke FAILS), then harness/families6/perwindow for 6 arms and `axd_analyze.py`. At hand-back the main seam was loading u400 on CPU; **UNVERIFIED: the wrap of `REFePlanner.infer`/`_ego_vec`, the per-arm npz, harness compatibility** (CPU unit tests 9/9 PASS, `test_axd.out`). G-POS and R-A0BANK are not expected to pass on CPU/8 tokens (excluded from the smoke's mechanical check).
   * Attempt 1 (GPU smoke, 67 tokens) was killed by explicit PID at 09:53 (moved to `$R/smoke_attempt1_gpuqueued_killed`, `markers/*.attempt1`): it could never take the lock (below).
2. Only if `$Y/markers/axd_smoke.DONE`: FULL (`--tag full`, 923 tokens, GPU + lock only; the chain refuses CPU/no-lock for full) waits for `YAX/markers/yax_full.DONE|FAILED` (exact names; the renamed `*.attempt1_cublas` files do not count), exits SKIPPED if YAX `status.json` X2 = CANDIDATE, else seam (one lock hold) -> harness + families6 + perwindow for A0 and the 5 transforms -> `axd_analyze.py`.
* ⛔ GPU BLOCKER (MEASURED 09:40-09:53): `nvidia-smi` read 3,627-3,740 MiB used by desktop apps with NO compute job; `wait_idle.py` needs < 3,500 MiB on 4 polls in a row, so the lock is never granted. This blocks YAX X2's evals AND this full run (12 h give-up each). Unblock = the PI closes GPU-heavy desktop apps (Photoshop Express, Photos, VLC, Codex, browsers); the bar is not ours to change.
* Markers: `$Y/markers/axd_<smoke|full>.STARTED|DONE|FAILED|SKIPPED` (json, `why` on FAILED); logs `$Y/markers/axd_*.out`, `$R/<tag>/PIPELINE.log`, `$R/<tag>/logs/*.log`; result `$R/<tag>/analysis/P1.json|.md`.
* ETA (ESTIMATED): smoke seam ~5-10 min once the lock is free, harness 6 x ~3-5 min; full seam ~35-50 min GPU, harness 6 x ~25-35 min CPU (YAX's arm took 23 min seam + ~35 min scoring) -> ~4-5 h after YAX ends.
* ⚠ GPU risk (MEASURED 09:40): `nvidia-smi` read 3,740 MiB used by desktop apps with no compute job, above the lock's 3,500 MiB bar. If the lock keeps refusing, both YAX X2's evals and this chain queue (12 h give-up). Not ours to fix; tell MM.

## If FAILED
Read `why` in the marker and the stage log. Fix, re-pin `CODE_PIN_AXD.json` + write `AMENDMENT_1.md` (sha256), delete the FAILED marker, relaunch `axd_run_all.py` via CIM `Win32_Process Create` (CreateFlags 0x08000200, `cmd /c "... < NUL > log 2>&1"`, inline PowerShell; `.ps1` is blocked). Stages are idempotent (`seam.DONE`, `arm.json`).

## When `axd_full.DONE`
Quote `$R/full/analysis/P1.md` (tier-stamped: open loop shows COST and GAIN only). Each CANDIDATE transform goes to MM for the closed loop (PREREG section 7: scenario 720, ADOPT iff min speed > 1.0 m/s before the stop line on green AND EPDMS >= 0.748). The transform is a one-line change at `REFePlanner._ego_vec` (planner.py:448), applied to the honest ax.

## Next lever if no P1 CANDIDATE: P2 AXNOISE (pre-registered in PREREG section 6, NOT BUILT)
Build `axd_ft.py`: a wrapper around YAX `ax_ft.py` with policy real, adding `N(0, 1.0^2)` to ego[:,2] in TRAINING batches only (the `main` loop's `make_batch` call, ax_ft.py:204), from `torch.Generator(seed + 7919)`; gates: sigma 0 == AXON 5-step weights bit-for-bit, training-batch noise std within 10 % of sigma, gate/VAL batches exactly honest. Train s0, s1 under K1 (YAX `run_full/stages/select.json`), compose with YAX `ax_ckpt_tools.py`, eval with `axd_seam.py` (IDT + gain arms), control = YAX's banked `run_full/eval/X1_AXON_s{0,1}`. Checkpoints on C: (1.29 GB each, ~11.9 GiB free).
