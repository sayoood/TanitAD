# search_log — E-DEP-VGEO-4 (2026-09-17)

⛔ This package is a **re-run of an in-house instrument on a banked feature bank**, not a literature
pass. No web search was required to execute it. What follows is the provenance trail plus the two
in-repo lookups, and the empties are named per charter §5.4.

## In-repo lookups

| # | query / probe | where | result |
|---|---|---|---|
| R1 | the pre-committed next lever for I-1 | `Deployment & Optimization/Research/2026-09-15-i1-path-length-value-geometry/RESULT.md` §2 EXPERIMENT | HIT — `E-DEP-VGEO-4` / DP15-1, verbatim: *"replace the cross-clip speed null with a within-clip, position-matched null … that null must read \|point\| ≤ 0.03 on the latent arm before the P1 contrast is read"* |
| R2 | the rig to reuse | `…/2026-09-15-…/code/vgeo3b_control_repair.py` | HIT — fit/score split, P2 triplet construction, bootstrap and the speed join reused unchanged, so the two passes are comparable |
| R3 | is the 63 GB token bank still on disk? | `C:/Users/Admin/tanitad-caches/bevhead-20260913/` | HIT — `tokens_s32_fp16.npy` 6,232,146,048 B, `pix64_u8.npy` 2,549,514,368 B, `index.npz`, `bev_gt/` 136 files |
| R4 | is the 4060 free? (charter §5.6 precondition) | `nvidia-smi --query-compute-apps` + `tasklist` | ⛔ **NO** — four `python.exe` in the compute-apps list, GPU 100 % / 7,729 MiB. **GPU not used.** The precondition is recorded because a later reader would otherwise assume it was available |

## Empty searches (named, per charter §5.4)

| # | query | where | result |
|---|---|---|---|
| **E-DP1** | a published operating point for a **partial Spearman under near-collinear conditioning** in representation probing | not searched — ⚠️ **declared, not claimed**: F3's cause was established from our own per-cell data (ρ(oracle, \|r_lp\|) = −0.7116), so no literature was needed to settle it. A literature check on whether the standard fix is a screen or a different estimator is **owed** before `E-DEP-VGEO-5` fixes its threshold, and is listed as such | OWED |

⚠️ **E-DP1 is recorded as a debt, not as an absence.** No probe was run, so nothing here supports a
sentence of the form *"no published treatment exists"*.
