# DEVIATIONS_L2B (appended with the time; none allowed after the first real fan number exists)

* 2026-10-25 ~15:00 (BEFORE any roll): PREREG_L2B.md section 5 K_H tolerance changed 1e-6 -> 1e-4 (the battery computes ADE in float32; measured float-noise max 1.6e-6 on 3,668 real windows). The prereg was re-sealed (.sha256) after this edit and before the chain started; no fan / per-candidate number existed.
* The synthetic dry run (`smoke/l2b_dryrun.py`) patches the stage to identity at RUNTIME (test only); the production scorer has no such switch. K_F therefore reads False in that dry run by construction (264 vs 259) and the dry-run label is stamped INCONCLUSIVE, as it must be.
* The pre-seal sets: FB = 259 (seed 0), 255 (seed 1); MB = 300 / 301; DR = 339 / 338; CT pool = 1,273 / 1,271 (L1 BASE-os + the shipped stage; MEASURED by a CPU read before launch, used only for the K_F literal).
