# PREREG_V8F_N1DH, Amendment 5 (FROZEN 2026-11-21 11:21Z by the MM; after the D1-HO deciding cell was read, before any eval-141 guard or Z-POST value)

The card `PREREG_V8F_N1DH.md` (sha256 f1df05bc) and Amendments 1 to 4 stand. This amendment records two path repairs to read instruments. Neither changes a control's form, a cell, a bar or an outcome rule, and neither touches the deciding cell.

## State when it was written (MEASURED)
* The dev-box driver `F:/TanitAD-scratch/n1dc/read_n1dh/run_n1dh_reads.sh` had run the D1-HO reader (`out_real/RESULT_N1DH_D1HO.json`, status VALID, about 11:08Z) and K-LIM at both seeds (`KLIM_s0.json`, `KLIM_s1.json`, status VALID).
* The deciding cell had therefore been read. It is reported in the read note with this amendment's timing stated. The repairs below cannot move it: they concern the eval-141 guard's first control and the Z-POST run.

## Repair 1: K-HB0-141's ticks path (eval-141 chain, step `khb0`)
* The chain failed at its first control: `K-HB0-141 FAIL files 141 arrays 1269 unequal files 0 ticks missing`.
* The cause: `eval141/k_hb0_141.py` line 48 hard-coded the ticks file as `ticks_N1DC-N1D_s0_m10_causal_heads.npz`, the N1DC chain's tag. This chain's smoke dump is tagged `N1DH-N1D_s0` and wrote `ticks_N1DH-N1D_s0_m10_causal_heads.npz`, so the file the control looked for did not exist.
* The repair derives both ticks names from the dump folders' own tags (`dump_<TAG>` gives `ticks_<TAG>.npz`). The copy is kept as `k_hb0_141.py.btick`. The rule "141 episode files and the ticks file equal" is unchanged.
* Run by hand on the existing smoke dump, then by the chain: **K-HB0-141 PASS**, 141 files, 1,269 arrays, 0 unequal, ticks equal.
* The failed marker is kept as `eval141/out/N1DH141_khb0fail.done`. Nothing is deleted.

## Repair 2: Z-POST's reader modules on the pod
* Z-POST runs on the pod CPU after the RHp3 chain, so it never runs on a training pod. Its first start (11:18:48Z) failed at import: `ModuleNotFoundError: No module named 'n1dh_ho_read'`. The pod's `read_tools/` held `n1dh_zpost.py` and `n1dh_t0route_lib.py` only.
* The repair copies `n1dh_ho_read.py` (sha256 19babc0b) and `read_lib.py` (6c8facea) there, byte-equal to the read package's `MANIFEST_SHA256.txt`. The failed run log is kept as `zpost_run_importfail.log`.
* Z-POST restarted at 11:19:28Z, holding the GPU queue lock so that nothing trains while it runs.

## Lesson
A script adapted from an earlier card carries that card's names. Before a read chain starts, grep the adapted scripts for the previous card's arm tag, and run each step's tool once on the target machine (the pod read tools were never imported there).
