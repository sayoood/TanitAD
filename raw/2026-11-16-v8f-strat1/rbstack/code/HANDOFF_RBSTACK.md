# HANDOFF_RBSTACK (2026-11-19)

State: card PREREG_RBSTACK.md (sha256 bf4d5395...72901, verified, never edited; rbstack.py re-asserts it at start). rbstack.py + test_rbstack.py written; 10 tests pass (whole file); mutants recorded in out/MUTATIONS_RBSTACK.json (plain non-nested OOF probabilities: RED at the assertion and through the pipeline; test-probe-saw-test-fold: RED; 1e-6 reference drift flagged; shuffle keeps marginals). Real-data smoke (3 clips/fold, 1 epoch) ran end to end into smoke/run (numbers not quotable). Full run launched detached via WMI: job_rbstack.cmd, pid 46460 (cmd), started 01:33 local.

Run log (waiter matches its LAST line): F:/TanitAD-scratch/strat1/rbstack/out/rbstack_log.txt, final line `ZZ RBSTACK DONE ZZ` or `ZZ RBSTACK FAIL ZZ <reason>`. Same line in out/RBSTACK.done. Raw output out/stdout.log; FAIL traceback in out/traceback.txt. Expected runtime about 2.5 h CPU (per probe seed: 5 outer + 10 inner fits, about 3 and 2 min each; then the 5-way beside).

Banked as they finish (priority order): out/stage_s0.json (primary, probe seed 0), out/stage_s1.json, then out/RESULT_RBSTACK.json + .md (section-3 reading, controls K-REPRO/K-SHUF/K-NEST/K-V), then out/stage_lat5_s0.json (5-way beside, no bar) and the result files rewritten with the beside block. Per-fit probabilities are cached in out/cache/*.npz: a rerun resumes (relaunch job_rbstack.cmd; features are re-extracted in minutes if absent).

Deviations from the card (none changes the reading rules):
1. The pooled features cache rbjoint/out/feat/X8.npy had been deleted; rbstack re-extracts X8 only (same window rule, rbjoint.read_window) into out/feat/ from a copy of rbjoint's meta.npz. The 5-way beside checks the replay: outer-fit predictions vs rbjoint's RAW8_f*.npz (`feature_replay_check`).
2. Probe features exist only for RB-JOINT's strided windows, so EDV, EDV_SHUF and the added ED_STR control fit on strided labelled fold-train windows; ED and S stay dense (K-REPRO against RESULT_RBJK.json). Contrasts vs ED are the card's; vs ED_STR separate the sample-size effect.
3. The ordered inner pairs (j,k) and (k,j) are one fit set, so 10 inner fits per seed serve all 20; K-NEST reads the folds the probe was actually given.
Stage B: NOT-YET. D1-HO pooled 8x20 features and ED rows are not on this machine (strat1/navrbd/in absent), the card forbids pulling from the pod. Text is in RESULT_RBSTACK.md.

Self-cleanup: the job deletes out/feat/X8.npy (4 GB, re-creatable) after a successful run and logs it in out/deletions.txt. Already deleted by me: smoke/run/feat/X8.npy (126,156,928 bytes). Smoke dir smoke/run (small files) kept as the smoke record.

Next step when DONE: read out/RESULT_RBSTACK.json (reading, controls), write the MM reading, stage the files. If FAIL: read out/traceback.txt, fix, relaunch (resumes).
