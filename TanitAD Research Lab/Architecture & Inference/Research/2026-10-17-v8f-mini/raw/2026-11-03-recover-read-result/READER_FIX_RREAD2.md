# READER_FIX_RREAD2 — write root moved off F: (2026-11-03, rollouts only; no read had run)

**What failed.** Relaunch 2 (03:51) finished 9 of 15 rollouts (R_s0 C0_s0 KA1R_s0 R_s1 C0_s1 C0REP_s0 C0REP_s1 FZC0_s0 FZR_s1)
and lost 6 (`logs/chain.out`, `logs/run_<tag>.out`): FZC0_s1, RC_s0, C0C_s0, RC_s1, C0C_s1 with
`OSError: [Errno 28] No space left on device` while writing their ticks file, and FZR_s0 with numpy
`_ArrayMemoryError: Unable to allocate 5.00 MiB` (04:24). MEASURED 05:03: F: had 0 MB free of 1,907,727 MB; this package's
scratch on F: is 1.2 GB (out/ 246 MB). The space was taken outside `F:/TanitAD-scratch/recover_read/`, which this
package does not touch. No read ran (`ZZRREAD-FAIL missing done markers`); no number exists.

**The fix.** `recover_read_chain.sh` takes an optional write root `RREAD_W` (default: the old path, so a bare launch is
the old chain): `OUT=$W/out` and the pre-flight's `$W/smoke`. The eval tree (`TREE=$S/tree_dag`) stays on F: and is only
read. `out/` was COPIED (not moved) to `C:/Users/Admin/rr_scratch/out` with its 9 done markers, so finished rollouts are
skipped; the 6 lost ones re-run from scratch (the chain moves each half-written dir aside as `_incomplete_*`). Launcher
`recover_read_chain_c.cmd` sets `RREAD_W=/c/Users/Admin/rr_scratch`. Chain sha256 c6a9a699… -> 8c5f444d… (PINS.txt
records the earlier one; this note is the record of the change). The tool, reader, tests and both cards are unchanged.
The KA1 bit-identity control (KA1R_s0 vs R_s0) and every done marker's ckpt sha are re-checked by the reader as before.

Class: instrument environment (a shared scratch drive filled by other writers), not an analysis change.
