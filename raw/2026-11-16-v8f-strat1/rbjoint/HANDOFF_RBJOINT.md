# HANDOFF_RBJOINT (2026-11-18)
State: card PREREG_RBJOINT.md + PREREG_RBJOINT.sha256 (499860a2...) written and hashed before any number; rbjoint.py + test_rbjoint.py (13 pass, whole file, CPU); a smoke run passed end to end (not quoted). Full run launched detached (WMI) via job_rbjoint.cmd: commit gate 5.5 GB, then extract (fp8 memmaps ~6 GB in out/feat/) then 5-fold CV. Estimated 2-2.5 h.
Sentinel: F:/TanitAD-scratch/strat1/rbjoint/out/RBJOINT.done (first line DONE or FAIL; FAIL carries the traceback). Logs: out/rbjoint_log.txt, out/stdout.log. Per-fit results are cached in out/cv/*.npz (a rerun resumes; the memmap is rebuilt by the extract stage, so on resume run --stage cv only).
Outputs: out/RESULT_RBJOINT.json (arms, paired, validity, labels_held, next_lever_from, eval202_beside_no_label), out/windows_info.json.
Next step: when DONE, read RESULT_RBJOINT.json, write out/RESULT_RBJOINT.md (all cells, CIs, validity, labels), stage the files.
CLEANUP after the result is read: delete F:/TanitAD-scratch/strat1/rbjoint/out/feat/ (~6 GB memmap, re-creatable by rbjoint.py --stage extract; also out/cv/*.npz are small, keep). Keep F: >= 22 GB free.
Do not edit the card (sha256 asserted at start). Do not rerun on FAIL without reading the traceback.
