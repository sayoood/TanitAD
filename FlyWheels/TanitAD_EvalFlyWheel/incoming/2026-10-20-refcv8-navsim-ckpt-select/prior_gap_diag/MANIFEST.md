# MANIFEST (local package; the Master Mind lands it)

| artifact | role | sha256[:12] |
|---|---|---|
| `local:C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag/PRIOR_GAP_DIAG.json` | every number with n and CI (controls, Q1-Q4) | 4ba8636ac097 |
| `local:C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag/PRIOR_GAP_DIAG.md` | one-page ranked-lever write-up | 9c0c990b230d |
| `local:C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag/build_lever_seams.py` | builds the Thor seam arms + job file (CPU, numpy+scipy) | 1bec42e9f3b8 |
| `local:C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag/jobs_prior_gap_levers.jsonl` | 18 Thor scorer jobs (Thor dirs are placeholders) | 04d3c977e511 |
| `local:C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag/lever_seams_build_report.json` | builder controls + per-arm sizes + seam sha256 | a6e759dd6af8 |
| `local:C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag/prior_gap_diag.py` | analysis script (CPU, numpy) | 058cf891cd5c |
| `local:C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag/verify_independent.py` | independent pandas re-derivation (goes RED on a mutated JSON) | 305a72c70361 |
| `local:C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag/thor_seams/` (36 files: 18 seam_*.npz + 18 .manifest.json) | Thor seam arms; seam sha256 in lever_seams_build_report.json, per-file in MANIFEST.json | - |
