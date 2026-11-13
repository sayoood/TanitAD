# HANDOFF: S0 read (PREREG_V8F_PROOF_ADD12 Stage 0). Folder `F:/TanitAD-scratch/v8proof/spatial/s0/`

STATE: built, tested, smoke-run on permuted labels. The full read on true labels has NOT been run (the lead runs it). Nothing staged or committed.
Tier T0, a world-model diagnostic, never driving performance. CPU only.

LAUNCH (detached, via WMI like the LATVIS job): `cmd /c F:\TanitAD-scratch\v8proof\spatial\s0\job_s0.cmd < NUL`
Waits for free commit >= 6 GB, runs `s0_read.py run`, writes `S0_DONE.txt` (rc 0) or `S0_STOP.txt` (rc 2 = a control missed, numbers still written, outcome names withheld; rc 1 = crash, see `s0_job.log`).
Direct: `set CUDA_VISIBLE_DEVICES=-1` then `C:\Users\Admin\venvs\tanitad\Scripts\python.exe s0_read.py run` (cwd = this folder).
Resumable: per-fold files `stages/fold<k>.npz`; a restart skips finished folds (signature checked). Delete `stages/` only after the result is read.

EXPECTED RUNTIME: about 35 to 45 minutes (ESTIMATED from the smoke: states about 80 s per fold at 7.5k rows, probes about 28 s each at 7.5k x 640 and 11 s at 1.6k, about 16 probes per fold, 5 folds, plus ego floors and the B 2,000 bootstrap). RAM about 2.5 GB. F: footprint under 30 MB.

OUTPUTS: `RESULT_S0.json`, `VERDICT_S0.md` (controls first; no outcome name if a control fails), `s0_job.log`.

FILES: `S0_DEFS.md` (rules R1..R17), `s0_read.py`, `test_s0_read.py` (10 tests, synthetic only, `python -m pytest -q -p no:cacheprovider test_s0_read.py`), `make_job.py` -> `job_s0.cmd`, `smoke/` (RESULT_S0_SMOKE.json, VERDICT_S0_SMOKE.md, stages), `smoke_run.log`.

TOKEN SOURCE: `E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips/<clip>.pt` key `pool4x10` (fp8 [T,40,1024], 141 clips present); windows from `wm/.../latents.pt` via `wmceil.LATENTS`; bank labels and ego from `res/cache/frames.npz`.

THINGS TO KNOW BEFORE READING THE RESULT
* Control s2 amplitude: the card's f5 wording (median token norm) is about 100 natural sigma along the planted direction and the mean state reads it too (AUROC 0.9996, 12-episode smoke, label free). Amplitude fixed at 12 sigma_v (DEFS R13), after a label-free calibration. If the lead disagrees, change `PLANT_SIGMAS` and re-run only s2 (delete `stages/` to redo all).
* Control s3 needs the ego-only floor of r-a gap R2 within 0.03 of 0.133 on bank rows (this build did not read it on true labels; FULLRES used bank + 2,365 windows, 11,686 rows). A miss there withholds all outcome names (rc 2); the numbers are still in RESULT_S0.json.
* r-c (pixels) is NOT RUN: LATVIS prep alone took 1,895 s CPU and `frames64.u8`, `LO_r128.i8` are deleted. Rebuild = `latvis.py prep` plus the encode.
* Windows with a lead at both times may be few (smoke: 36 of 156 on 12 episodes); full count is printed in RESULT_S0.json counts.

NAMED GAPS: r-c not run; true-label read not run; s3 ego-floor reproduction unverified until the full run.
