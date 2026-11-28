# KNOWN DEFECT: `run_fam_chain.sh` (MM 2026-11-28)

The `tactical` step passes the ARM labels (`SH0=`, `SH1=`, `C1pick=`) to `p5c_tactical.py`, which is hard-wired to the P5-C ROLE names `C1` / `C0` / `base` and refuses with *"seam for C1 missing"*. MEASURED 2026-11-28 on the EP-HEAD copy (attempt 3, `PK/raw/2026-11-27-l2next/run1/EPH_READ.FAILED.attempt3_tactical_roles`). This chain never ran end-to-end under SAFE-HEAD: its read was NULL-HELDOUT, so no families were read.

The file is left as landed because it is pinned in this card's `CODE.sha256`. Rewriting it would falsify the closed card's record.
**Use `PK/raw/2026-11-28-datascale/run_fam_chain_ds.sh`** instead. It is the proven EP-HEAD chain with the arm labels as arguments. Its tip export also needs `stack`, `taniteval` and `raw/2026-11-12-instruments/fam_ci`.
