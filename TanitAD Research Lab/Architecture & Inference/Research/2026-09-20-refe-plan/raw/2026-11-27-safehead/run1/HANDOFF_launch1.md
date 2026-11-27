# SAFEHEAD run1 launch: BLOCKED at step 1 by a landed-code memory defect (2026-11-27)
State: LANDED 5eb0ae09 extracted at F:/TanitAD_safehead_tip; ship-manifest OK; test_safehead --fast ZZSAFEHEAD_PASS 12; prep OK (136 logs). C1 sha verified.
No gate token, no training, no chain launched.
Defect: safehead_data.load_cache (take()) returns a VIEW of a whole-file array read once per ROW, so each row pins its file's full array: ~40+ GB peak on navtrain (ArrayMemoryError at 23 MiB with 12 GB commit free); navtest 12,146 rows worse. Synthetic tests could not see it.
Fix: fix/safehead_data.py.PATCHED + fix/safehead_data.memfix.diff (read each array once per file, np.array copy of the row). Checked on a scratch copy: `check` on navtrain = ZZSH_CHECK_OK leak [] in 6 s.
Gate binds the landed commit/hashes, so the fix must be LANDED (new sha; CODE.sha256 + test pins move; rerun the 56-file regression) by the MM. Not hand-patched into the tip.
Adapter: index.csv 'file' is a bare name but npz sit in out/ (navtrain) and navtest/: scripts/adapt_index.py rewrites to absolute paths into F:/TanitAD_safehead_tmp/adapted*/ (not run1, seal refuses an index's own dir). npz keys match loader aliases.
Next: MM lands fix -> `bash scripts/step12.sh <NEWSHA>` (steps 1-2, untested end-to-end past seal) -> WMI-launch scripts/chain34.cmd (CreateFlags 0x08000200) -> fresh agent runs confirm on fam/fam.DONE.
Stop-check: s0 chosen_step 0 => NULL-HELDOUT (chain34 refuses and writes READ_BLOCKED.txt).
Deleted: F:/TanitAD_safehead_tmp/pt, /patched (scratch copies, <1 MB). Kept F:/TanitAD_safehead_tip (needed).
