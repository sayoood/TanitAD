# HANDOFF: W-TRANSP / W-FAM instruments (2026-11-12)

State: code done and tested (6 unit tests pass, 1 skipped without banked rows). Controls pass on the first 252 tokens (`raw/controls_partial.json`). The FULL 1,018-token validation chain was launched detached (CPU, 2 processes, resumable):
`F:/tanitad_scratch/instr/val/go_full.sh` (2 shards of `transp_score.py`, arms T / Tc20s / Tw / human, `--controls`) then `go_finish.sh` (a 40-token rerun for C-RERUN, then `validate_controls.py`). Copies of both scripts are in `raw/`.

How to check: `F:/tanitad_scratch/instr/val/finish_done.txt` exists when finished; `controls.log` and `controls.json` there hold the verdict (`ALL_PASS`). Row count: `cat full_s*.jsonl | wc -l` (target 1,018, at most 3 s/token/process, expect about 1 h wall under contention). If the processes died, rerun `bash go_full.sh` then `bash go_finish.sh` (resumable; `rm full_done.txt` first). Memory was about 9 GB free at launch: check FreeVirtualMemory first.

Next steps (fresh agent, Sonnet is enough):
1. When finished, copy `controls.json`, `controls.log`, `rerun_r40.jsonl` and the 4 `full_s*.jsonl`/`rerun` rows (about 6 MB) into `instruments/raw/`; update the PARTIAL readings in `INSTRUMENTS.md` from `controls.json`.
2. If any control fails, fix before anything else; the validator names the worst token.
3. Refresh `hold/files/` from the working copies (`tr -d '\r'`).
4. Delete scratch (log it): `F:/tanitad_scratch/instr/` (`w/repo` 35 MB stack export, `w/refe`, `ref/`, `val/` rows and npz; all re-creatable from the tip or from `raw/`). Already deleted: old-format rows `T_s0.jsonl` 304,113 B, `T_s1.jsonl` 287,829 B, `T_s0.log`/`T_s1.log` 0 B, `s12.jsonl` 41,975 B, `fam_smoke.json` 236,051 B (all in `F:/tanitad_scratch/instr/val/`).

MM decisions: see the report (land to `eval/instruments/`; pinned-harness tracked-equality control on navtest; the lead-block definition; paired family CI; the metric-cache speed-up).
