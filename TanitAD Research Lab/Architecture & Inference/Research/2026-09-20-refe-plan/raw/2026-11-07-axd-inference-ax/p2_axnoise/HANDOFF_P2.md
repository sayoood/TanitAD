# HANDOFF_P2: stream YAXN, P2 AXNOISE (written 2026-11-07 ~18:00; next agent = a FRESH Sonnet agent; read, check markers, act)

`$Y` = `E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/`; python `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`. Nothing in the repo, YAX, YAXD, YF3 or the pod was touched; nothing staged (MM lands). Design = `YAXD/PREREG_AXD.md` section 6 (sha 1d33a9ed..., unedited) + `$Y/IMPL_P2.md` (operational reading, pinned in `CODE_PIN_AXN.json`).

## AMENDMENT_1 (MM ruling, sha256 a5dac854..., pinned): the relative gain clauses (d),(e) read AXON_s's IN-RUN gain per seed (AXON_s0/s1 re-composed from `YAX/run_full/arms/K1_AXON_s*/final.pt`, K1-K5 + identity gated, IDT/gain seam arms only, ckpt deleted after its seam via literal paths); u400's 1.0798 is descriptive only; no threshold moved. New stages `ref_axon_s0/s1` follow `eval_s*` in both smoke and full. 37 CPU tests PASS, 0 fail.

## State: ONE detached chain (WMI, cmd PID 34944 -> `axn_run_all.py`), RELAUNCHED 18:07 on AMENDMENT_1 code (first chain PID 68420 stopped by explicit PIDs at 17:59 during its first smoke seam)
1. SMOKE (`--tag smoke`, root `$Y/run/smoke`): pins (re-checked on every launch) -> gates_axn (sigma 0 == AXON bit for bit, sigma 1 != AXON, determinism, 4 RED arms) -> 4-step trainings s0,s1 -> compose + identity -> 8-token CPU seams (no GPU lock) -> harness/families6/perwindow -> 3 real-seam RED arms (`idt_perturbed`, `transform_not_applied`, `memo_stale`) -> `axn_analyze.py`. Mechanics only. ETA ~1 h.
2. Only if `$Y/markers/axn_smoke.DONE`: FULL (`--tag full`, root `$Y/run/full`): the same stages with K1 (200 steps, lr 5e-5) x seeds 0,1, 923 A5 tokens, seams on the GPU under YAX's shared lock (queues behind YF3, never forced; MemoryError => up to 5 retries, lock released while waiting; every subprocess stage also retries on a memory failure; logs `logs/retries.jsonl`). ETA (ESTIMATED): CPU ~1-1.5 h, then per seed seam ~30 min GPU + harness ~5 min, plus lock waits.
* Markers: `$Y/markers/axn_<smoke|full>.STARTED|DONE|FAILED` (json, `why` on FAILED). Logs: `markers/axn_*.out`, `run/<tag>/PIPELINE.log`, `run/<tag>/logs/*`. Result: `run/<tag>/analysis/P2.md|json` (tier-stamped; four families; TACTICAL n/a, STRATEGIC PAUSED).

## If FAILED
Read `why` and the stage log. Fix, `python make_pin.py` (re-pin) + write `AMENDMENT_1.md` with its sha256, delete the FAILED marker, relaunch `axn_run_all.py` by inline PowerShell `Invoke-CimMethod Win32_Process Create` (CreateFlags 0x08000200, `cmd /c "<python> <Y>/axn_run_all.py < NUL > <Y>/markers/axn_run_all.out 2>&1"`; `.ps1` is blocked). Stages are idempotent (`stages/*.json`); never touch the GPU lock file.

## When `axn_full.DONE`
Quote `run/full/analysis/P2.md`: overall verdict (CANDIDATE iff BOTH seeds; VOID = a gate red). Open loop shows COST (vs YAX X1_AXON_s) and GAIN only, never "stall fixed". CANDIDATE -> the composed `run/full/ckpt/AXN_s0.pt`/`_s1.pt` (1.29 GB, kept only then) go to MM for scenario 720 closed loop; otherwise re-compose from `run/full/arms/K1_AXNOISE_s*/final.pt` with `YAX/ax_ckpt_tools.py build`. Not CANDIDATE -> Rule Zero next lever (IMPL_P2 section 5): sigma 2.0 (predicted gain ratio 0.14) with the same gates, then ax-input dropout; CPU only, ~1 h per seed. Update `GOALS_AND_CLAIMS.md` in the same turn.

## Known gaps (named)
* UNVERIFIED until the smoke reports: the seam wrap on an AXNOISE checkpoint (CPU 8 tokens), harness/families6/perwindow on the AXN arms, `axn_analyze.py` end to end. Verified (CPU, `test_axn.out`, 0 fail): all training-side gates + RED arms, retry logic, pins refusal, compose + identity.
* The gain reference for clauses (d),(e) is u400's banked P1 gain (1.0798 [0.839, 1.313]); AXON_s's own gain is not measured (descriptive row, not in the rule).
* GPU: lock bar is memory.used <= 3,500 MiB; desktop apps at 3.6-3.7 GB have blocked it before (not ours to change).
