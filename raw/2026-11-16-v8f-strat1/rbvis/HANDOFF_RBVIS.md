# HANDOFF_RBVIS (2026-11-18)
State: card PREREG_RBVIS.md + .sha256 written before any number; rbvis.py + test_rbvis.py (9 pass, whole file, CPU); a smoke run passed end to end (not quoted). The full run was launched detached (WMI) via job_rbvis.cmd (CPU, commit gate 5.5 GB first), est. 15-25 min.
Sentinel: F:/TanitAD-scratch/strat1/rbvis/out/RBVIS.done (first line DONE or FAIL; FAIL carries the traceback or the commit-gate text). Log: out/rbvis_log.txt, out/stdout.log.
Outputs: out/RESULT_RBVIS.json, out/rbvis_logits.npz.
Next step: when DONE, read RESULT_RBVIS.json (verdict, problems, controls K-CONST / K-DISJOINT / R-SHUF-TRAIN), write out/RESULT_RBVIS.md (all cells, CIs, controls, provenance: train cache has no meta.json, eval has pipeline ra1a-pooled-v1), then stage the files.
Do not rerun on a FAIL without reading the traceback; the card must not be edited (sha256 asserted at start).
