# YAXN manifest (everything lives ONLY at `E:/Projects/TanitAD-artifacts/yawrate_wp/YAXN/` -- single copy, nothing staged in the repo; MM lands)
| artifact | what |
|---|---|
| axd_ft.py | AXNOISE trainer: thin wrapper over YAX ax_ft.py (policy real, N(0,sigma^2) on training ax from Generator(seed+7919)), added gates + RED arms |
| axn_common.py, axn_seam.py, axn_analyze.py, axn_chain.py, axn_run_all.py | constants/memory+disk guards; restricted-arms seam (axd_seam unchanged); P2 analysis (PREREG section 5/6); the chain (retry on MemoryError); detached entry |
| IMPL_P2.md | operational reading of PREREG section 6 (no threshold changed) |
| AMENDMENT_1.md, AMENDMENT_1.sha256 | MM ruling: AXON_s in-run gain is the reference of clauses (d),(e); added control arm |
| CODE_PIN_AXN.json, make_pin.py | sha256 pins of the above + IMPL_P2.md; chain REFUSES on mismatch |
| test_axn.py, test_axn.out, unit_gates_axn.json | CPU tests incl. RED arms (0 fail) and the integration gates_axn record |
| HANDOFF_P2.md, MANIFEST.md | state / next step; this table |
| run/smoke, run/full, markers/ | chain outputs (created by the detached chain) |
Dependencies, read-only: YAX/ (ax_ft, ax_ckpt_tools, ax_eval_arm, ax_stats, ax_analyze, gpu_lock, banked run_full/eval/X1_AXON_s0|1), YAXD/ (axd_common, axd_seam, axd_analyze), `C:/Users/Admin/yaxd_run/full/analysis/P1.json`.
