# HANDOFF: stream YAX, AXOFF vs AXON (written 2026-11-07 ~02:03 Berlin; next agent = a FRESH Sonnet agent: read the verdict, hand the checkpoint to MM)

Dir `$Y` = `E:/Projects/TanitAD-artifacts/yawrate_wp/YAX/`. Python `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`. Nothing in the repo, YF2, YF3 or PK was edited; nothing staged or committed (MM lands).

## Frozen design
* `$Y/PREREG_AXOFF.md` sha256 **bd7325f0bb0fdcddf4ac1a41d1310a534a3f97889427f81a796b17f792bb9ba7** (`PREREG_AXOFF.sha256`, frozen 2026-11-07T01:44:28 BEFORE any trained arm; no arm has been evaluated).
* `AMENDMENT_1.md` (sha in `AMENDMENT_1.sha256`): implementation of gate GB-STEP0 only (alignment-proof form); no threshold/rule/arm changed. `CODE_PIN.sha256` pins the 11 code files (the pipeline's `pins` stage REFUSES on a mismatch; a code edit means re-pin plus an AMENDMENT).

## State: ONE detached chain RUNNING
`ax_chain.py` = SMOKE pipeline (110 tokens, 4 steps) then, only if `markers/yax_smoke.DONE`, the FULL pipeline. cmd PID 58824 -> python 76676 -> pipeline 78244 (smoke) -> eval 82664 (currently queued behind the GPU lock held by YF3, owner `yf3_yaw_eval`; it never forces the lock, polls every 60 s, gives up after 12 h).
* Markers (last action): `$Y/markers/yax_smoke.DONE|FAILED`, `yax_full.DONE|FAILED`, `yax_chain.DONE|FAILED` (json). Logs: `markers/yax_smoke.out`, `yax_full.out`, per-stage `run_<smoke|full>/logs/*.log`, `run_*/PIPELINE.log`, `run_*/status.json`.
* Smoke CPU side is MEASURED green (2026-11-07 02:01): pins, gates_real (5 runs incl. the 3 RED arms), g8_mini, A0-banked per-window, selection, AXOFF s0 and AXON s0 train + compose (K1-K5) + identity (GA-GAIN: AXOFF decode bit-identical at ax in {0, +0.5, -0.5, real}; positive control u400 differs). **UNVERIFIED until the smoke eval returns: the GPU seam, the harness scoring, families6, per-window and `ax_analyze.py` on the new arms** (they are copies of YF3's proven path minus the yaw shim; a first-time failure is plausible).
* ETA (ESTIMATED, dominated by the lock shared with YF3, whose evals are 0.35-0.5 h each): smoke evals 2 x ~6 min once the lock is free; FULL: CPU gates+selection ~35 min, CPU arms 3 x 10-30 min, GPU evals 4 x ~0.4 h on 923 tokens -> X1 verdict ~3.5-6 h after the smoke finishes; X2 (only if X1 is NOT PROVEN/REFUTED) +2.5-4 h. Disk: E: had 25 GB free; AXON checkpoints are deleted after eval, the AXOFF ones (1.29 GB each) are kept for the sim.

## If it FAILED
Read `markers/yax_chain.FAILED` / `yax_smoke.FAILED` -> `why`; the stage logs say which command. The pipeline is idempotent: fix, re-pin (CODE_PIN + AMENDMENT_2 if code changed), delete the failed marker, relaunch with `ax_chain.py` via CIM `Win32_Process Create` (CreateFlags 0x08000200, stdin NUL; the `.ps1` launcher is blocked by the machine execution policy, so its body was run inline). A smoke dir that is half done is resumed, not restarted. Never delete another stream's lock.

## When `yax_full.DONE` exists
`run_full/FINAL.md`, `run_full/analysis/X1.json|.md` (and X2). Verdict per PREREG section 5: CANDIDATE / NOT PROVEN / REFUTED / VOID. On CANDIDATE: hand `run_full/ckpt/X1_AXOFF_s0.pt` (+ `_s1`) to MM for the closed-loop test (PREREG section 6: scenario 720, ax fed honestly, ADOPT iff min speed > 1.0 m/s before the stop line on green AND EPDMS >= 0.748). Selection record: `run_full/stages/select.json`. Quote only tier-stamped numbers (open-loop NAVSIM, not closed loop).

## Known gaps (named)
* Closed loop, not open loop, is the deciding read; open loop has no ax feedback. CRUISE has n = 90 tokens / 29 logs (MEASURED on the export), so the longitudinal CI is wide; the rule uses the point estimate (MM wording), the CI is reported.
* A0 reference = banked seam/CSV (INHERITED; YF3 owns the G6 bit-for-bit reproduction). G8 is mini (5-step reruns), not a full rerun. No planner-level (seam) ax perturbation test: GA-GAIN is on the composed file's decoder with cached scene context (same level as YF2's G-CKPT).
* Unit tests: `test_ax.py` -> `test_ax.out` (13/13 PASS, includes the RED arms), run under `CUDA_VISIBLE_DEVICES=-1`.
