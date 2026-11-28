# HANDOFF: stage 0e read tool (PREREG_V8F_PROOF_ADD12 note N7)

State: tool built, tested (13 synthetic tests pass), label-blind smoke done on real data. The full read on true labels has NOT been run.

Files (all in `F:/TanitAD-scratch/v8proof/spatial/s0e/`): `S0E_DEFS.md` (rules E1 to E18, departures D1 to D5, gaps G1 to G5), `s0e_read.py`, `test_s0e_read.py`, `make_job.py` -> `job_s0e.cmd`, `smoke/` (logs and `RESULT_S0E_SMOKE*.json`). `data/` untouched (PULL_DONE says mismatches 0; every shard is sha256-checked again on load).

Exact next step: launch `job_s0e.cmd` detached via WMI (`cmd /c ... < NUL`). It waits for 8 GB free commit, runs `s0e_read.py run`, writes `RESULT_S0E.json`, `VERDICT_S0E.md`, `S0E_DONE.txt` (exit 0) or `S0E_STOP.txt` (rc 2 = a control missed: numbers written, no outcome name anywhere; rc 1 = crash). Resumable per fold (`stages/fold0..4.npz`, out-of-fold predictions only); rerun the same command after an interruption. Check: `tail s0e_job.log`.

Before launching, decide G1: the note's e4 asks kinematics = pod cache values for 30 rows. No train cache is on this box, so that part prints "NOT CHECKED" and e4 can still pass. To check it, put `data/KIN_REF.npz` (arrays `clip` str, `t` int, `kin` float [n,3] = v, a, yaw_rate from the pod cache, exact comparison) before the run.

Estimate of the full run (measured, label-blind, 1,200 clips, one fold = 190 s): 5 folds about 16 min; dominant step = r-a probes (89 s per fold; 12 fits, two label sets), then the change read (62 s), e2 motion + position (34 s), states 5 s. Plus load (2 s), e3 eval-bank read (not timed on the full bank; the 12-clip smoke took 4 s, estimate a few minutes for `S0.load_data`), bootstrap (one 50-draw chunk of the presence AUROC = 0.23 s, so about 2 min in total). Total about 25 min, under the 60 min bar; no change proposed. Peak RAM measured 2.5 GB (one fold, 1,200 clips); the full run adds the e3 eval tokens (about 0.8 GB), expect under 5 GB.

Counts (smoke, label tracks swapped, so only counts that do not depend on label alignment are meaningful): frames 121,166; gap rows 33,123 (0 to 60 m); pairs 24,206 of 103,166 with three rows; 1,740 removed by the continuity rule.

Hand-back items for the lead: read `VERDICT_S0E.md` controls first; if rc 2, report which control and do not quote outcome names. Nothing is staged in git (scratch tool outside the repo).
