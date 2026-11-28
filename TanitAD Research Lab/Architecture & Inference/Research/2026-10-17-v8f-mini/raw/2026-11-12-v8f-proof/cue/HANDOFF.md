# HANDOFF CUEPROBE (PREREG_V8F_PROOF_ADD5), build agent, 2026-11-13. Zero training, CPU. Nothing staged, committed or pushed.

## State
Code complete: cueprobe.py (build, run), test_cueprobe.py (9 tests, all pass), job_cueprobe.cmd (CRLF), CUE_DEFS.md (definitions + deviations R1-R10). The full read was NOT run.

## Launch (orchestrator; do not pipe from an agent shell, use WMI or a normal cmd)
`cmd /c F:\TanitAD-scratch\v8proof\cue\job_cueprobe.cmd < NUL`   (waits for free commit >= 12 GB, polls every 2 min, up to 12 h)
Sentinels: cue\CUEPROBE_DONE.txt (`ZZ CUEPROBE-DONE`) or cue\CUEPROBE_STOP.txt (`ZZ CUEPROBE-STOP <reason>`; control-failed = rc 2, results are still written). Log: cue\cueprobe_job.log.
Outputs: cue\RESULT_CUEPROBE.json, cue\VERDICT_CUEPROBE.md, cue\cache\ (Gram matrices K40/K160/K40L/K160L 23 MB each float32, frames.npz).

## Expected runtime and memory (ESTIMATED from the smoke, not measured on the full set)
build ~6 min (20 clips took 42 s, 141 clips; temp memmaps ~1.8 GB on F:, deleted by the script), run ~15-25 min (5 fold seeds x 5 folds x 2 grids kernel PCA on 2,399 windows, ~800 bootstrap cells x 2 seeds). Peak memory ~3 GB (torch + tanitad import dominate; Gram matrices are small). 4 threads.

## Smoke (what was and was not tested)
* Unit tests (synthetic): conformal coverage, tau bound and minimality, leak guard on a later frame, k3 bit-identity, arm definitions, episode-disjoint folds, bootstrap sign, and the whole fold pipeline on planted-signal data (AUROC, selection, q, tau, coverage). 9 passed.
* Real-data smoke: build --max-clips 20 (340 windows, 14 stop) then run --smoke (1 fold seed, B 200), 10 s. Ran end to end and wrote both files; the probe cue chain was mostly degenerate there (14 stop windows), shield code, k3 and k4 guard ran. k1 and k4 read FAIL on that subset (AUROC below 0.5 from 14 positives, not informative); k2 cannot reproduce the banked ORACLE numbers on a subset (delta -2.58 vs -3.51 at s0) and is marked evaluated=False. Not smoke-tested on real data: the full-size DIST probe and the k2 equality at n = 131 (the ORACLE path uses the same shield and cell code as ruleshield_read).
* Gate note: the free-commit gate read 7 at the first smoke launch and 4 at the last smoke run (another process moved it); that last run was 10 s.
* Open point for the orchestrator: card k2 says "overrun 0"; the banked ORACLE has overrun share 0.0229 / 0.0153 (infeasible-at-start windows), so k2 checks equality with the banked values (CUE_DEFS R1).

## Deletions (own temp files, all re-creatable)
Smoke cache F:\TanitAD-scratch\v8proof\cue\cache\: K160.npy, K160L.npy, K40.npy, K40L.npy (462,528 B each), frames.npz (97,658 B), build_deleted_temp.json (317 B); smoke logs smoke_build.log (513 B), smoke_run.log (302 B); smoke RESULT_CUEPROBE.json (266,334 B) and VERDICT_CUEPROBE.md (76,309 B); cache dir removed. Script-deleted memmap temps in smoke builds: _tmp_X40/_X40L 27,852,928 B each, _tmp_X160/_X160L 111,411,328 B each (20-clip build); first 15-clip build: 20,889,728 B x2 and 83,558,528 B x2 (one removed by hand).
