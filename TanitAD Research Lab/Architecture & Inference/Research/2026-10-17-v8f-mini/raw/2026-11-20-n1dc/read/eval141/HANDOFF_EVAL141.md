# HANDOFF_EVAL141 (N1DC eval-141 guard chain; STAGED, NOT STARTED)
Card PREREG_V8F_N1DC.md (sha256 2eae7964) sections 3-4. Folder: `F:/TanitAD-scratch/n1dc/read/eval141/`. No GPU. Nothing committed or pushed.

## State
* Tree `tree/` = `git archive bcf3ba3f stack taniteval` (autocrlf off) + `stack/scripts/v8f_t1_eval_hb.py` (blob bcf267ce, sha256 74fd8445). `tree_blobcheck.txt`: 3,433 of 3,433 blobs equal, 0 bad, 1 extra (the hb wrapper).
* `ckpts/`: N1DC_s0, N1DC_s1 (both ARM_DONE on the pod), N1D_s0, N1D_s1 (pod `navrbd/runs`; none was on the dev box). sha256 = the pod's own `sha256sum`, in `ckpts/SHA256_PODS.txt`, `sha256sum -c` OK.
* `banked_n1d/x/`: from HF `navrbd_read/navrbd_read_outputs.tar` (sha256 143f3b2a verified): old-executor N1D eval-141 dumps (+Z, ticks) and the six N1D T0 passes. The tar copy was deleted (logged).
* **K-HB0-141 PASS** (`K-HB0-141.json`): N1D_s0 through the wrapper with `--kappa-max none` equals the banked old-executor dump, 141 files, 1,269 arrays, 0 unequal; ticks equal (the `ckpt` and `tag` provenance strings are skipped, said in the file).
* `DRY_CHECK.json`: PASS (8 dump argvs per executor render, 6 T0 argvs, view plans, imports, `bash -n`, sha256).

## Chain `run_n1dc141.sh` (resumable, `out/steps/<name>.ok`)
ckpt sha, banked N1D check, 8 hb0.2 dumps (N1DC, N1D x s0, s1 x follow, Z), 2 old-executor N1DC dumps, 6 N1DC T0 passes, STRATEGIC (N1-slot view + frozen readers; S1-slot views for N1DC and the N1D reference via the unedited `n1dw_view_s1slot.py`; contrast N1DC-N1D deciding, N1DC-N1R beside), `n1dc_read141.py`. Gate 0 refuses unless `K-HB0-141.json` says pass.
Rules kept: CPU, 2 threads, one process at a time, each step gated on commit >= 5.5 GB. Log last line `ZZ N1DC141 DONE ZZ` or `ZZ N1DC141 FAIL ZZ <why>`; result `out/RESULT_N1DC_EVAL141.json`.

## Launch (MM says go; not before)
```
powershell -c "$s=([wmiclass]'Win32_ProcessStartup').CreateInstance(); $s.CreateFlags=0x08000200; $r=([wmiclass]'Win32_Process').Create('cmd.exe /c F:\TanitAD-scratch\n1dc\read\eval141\run_n1dc141.cmd < NUL','F:\TanitAD-scratch\n1dc\read\eval141',$s); 'pid='+$r.ProcessId+' rv='+$r.ReturnValue"
```
Expected run time about 55 min (10 dumps x ~5.5 min, 6 T0 passes ~0.5 min each, readers a few min; from the N1DW chain and the smoke's 344 s). Do not run it beside another CPU chain (the D1-HO read chain).
Optional wait: set `N1DC_WAIT_MARK` to a marker path in the .cmd.

## Caveats (said, not hidden)
* F2 (floor) is the banked old-executor C0 floor, used for hb0.2 pairs too; no hb0.2 replicate exists.
* K1 (self-pair 0) and Z-EVAL are in the eval-141 read; K-CONST 0.338215 belongs to the D1-HO read. The skip-zeroed Z-USE has no inference flag in bcf3ba3f, so Z-USE here is the D-channel zero twin only.
* N1D T0 passes are banked (tree 0c3b374f), not re-run under bcf3ba3f.
* `out/` is empty until the chain runs; `n1dc_read141.py` and the STRATEGIC scripts are dry-checked (imports, plans), not yet run on N1DC data.
