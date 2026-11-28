# HANDOFF -- P6 read tooling (stream BEHAV/P6READ), 2026-11-08. State: DONE, frozen, nothing running, nothing staged/committed/landed.
`$R` = E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/p6_read/. No P6 output was read; no pod access; GPU/lock untouched.
Next steps: (1) MM lands `$R` + rules on AMBIGUITIES.md items 1-3, 11 (hashes drifted vs the card; FP32 gate added). (2) After ZZP6_DONE follow RUNBOOK.md (pull ckpt/config/train_logs, P6 navtest read, smoke, p6_read.py).
Any edit to a pinned file needs an amendment BEFORE any P6 value, then `python make_pin.py`. Tests: see CONTROL_RESULT.md (22 pass).
NOT deleted (removal blocked by the safety check; PI/MM may delete): F:/TanitAD_p6read_scratch, 114 MB, my own temp (u400 re-run, pytest basetemps). Re-creatable.
