# HANDOFF_PROBE (2026-11-19) PROBE-N1D-FEAT, card PREREG_V8F_N1DW.md sec 10 (sha256 59f2b7f9... verified before work)

## State
Built, tested, smoked end to end (400 windows, both checkpoints). Full run LAUNCHED detached on the LoRA pod at 14:20Z (CPU only, cores 16-47, nice 10, 8 workers x 2 threads). No bar, outcome, feature set, probe setting or control was changed.
Nothing staged or committed: the E: tree lags the mirror tip and a stage would sweep a shared index. Proposed repo home: `raw/2026-11-19-v8f-n1dw/probe/` (MM lander).

## Pod paths (all under /root/strat1/n1dwprobe/)
code/ (7 files), lib/read_otband.py (sha 18afede9, pinned), pylib/ (sklearn 1.9.1, scipy, joblib, narwhals, cloudpickle: own --target dir, torch untouched), blobs_0c3b374f.txt,
out/feat1, out/feat2 (two independent extractions for K-DET), out/result/PROBE_RESULT.json and READ.txt, n1dwprobe.log, driver.pid.
Tree read from: /root/strat1/navrbd/tree = 2630 of 2630 non-test blobs equal to 0c3b374f (checked blob by blob).

## Sentinel and how to read
Last line of `n1dwprobe.log`: `ZZ N1DWPROBE DONE ZZ` or `ZZ N1DWPROBE FAIL ZZ <reason>`. On DONE: `cat out/result/READ.txt` (the reader's output; its last line is CONTROL STATUS). Raw: PROBE_RESULT.json (every CI), oof_correct.npz.
Expected: extraction ~1 min per checkpoint x4, then 440 fits (~30-60 min ESTIMATED), then bootstraps ~1-2 min.
Waiter (written, NOT started): `C:/Users/Admin/actsens_launch/wait_n1dwprobe.sh` (polls every 5 min, ends on DONE / FAIL / dead driver by driver.pid / 6 h cap; exit 0 DONE, 1 FAIL, 2 dead, 3 cap, 4 pod unreachable).
Relaunch if needed: `nohup setsid bash /root/strat1/n1dwprobe/code/run_n1dwprobe.sh` (rewrites the log; refuses nothing, reads only banked files).

## What was built (file:line citations are in n1dwprobe_extract.py's docstring)
F-TEXT = `_at(b["nav_rb"], last_now)` (v8f_mini.py:923-927), F-NAV = `_nav_tok` return in run_o (:926, :1045), F-PLAN = input of `head_o` = `read_o(...).flatten(1)` (:1048-1050; 1024 dims), F-VIS = mean of `adapt[:, NOW]` (:1061, :1091; 256 dims). Tick 0 of the T1-kin rollout (a_t = a_s = 0).
Probes: HGB (rbdist_zt.HGB settings, random_state = probe seed) and standardised logistic (C 1.0). Folds: unique episodes shuffled by RandomState(0), k-th -> fold k % 5. Bootstrap B 2000 seed 0.
Outcome logic: n1dwprobe_outcome.py (tests: 8 outcome + 3 pure, all green; one per label, multi-label, one-seed, bar edges, incomplete).

## Smoke findings the MM must know (not results)
* K-DET passed (two extractions bit-identical); tick-0 plan equals the banked N1D dump's `ol` EXACTLY (max abs diff 0.0 m) and the GT track / labels match 400/400: the extractor reproduces the read's windows.
* ⚠ The REGRESSION ARM as written (zero c0 and the exit columns c1..c12, keep c13 dist and c14 valid) did NOT separate below F-TEXT at 400 windows (-0.0225 [-0.0625, +0.0200]); zeroing ALL 15 columns does (-0.18). If it fails at full size too, the control reads FAILED as the card commits; the distance columns are the likely cause. Printed beside as an informational arm; I changed nothing.
* Tree probe seeds 0 and 1 give IDENTICAL predictions (HGB without early stopping has no random draw); the four-unit rule is still read over four units; the seed does drive SHUF.
* SHUF on 400 windows is out of the 0.02 band (small n); the full run decides.

## Deletions (CLAUDE.md rule, literal paths)
Pod: out/smoke 5.4 MB, out/smoke2 5.4 MB, out/smoke_res 72 KB, smoke.sh, smoke2.sh, smoke_probe.log 92 KB, smoke_probe2.log (all re-creatable). Dev box %TEMP%: n1dw_blobs.txt 284 KB, rbjk.py 8 KB, smoke.sh, smoke2.sh. Kept: the full-run features (needed by the probe), pylib.
