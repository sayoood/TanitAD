# PRE-REGISTRATION ADDENDUM — `E-DEP-VGEO-3b`: control repair after run 1 was VOID

**Date** 2026-09-15 · written AFTER run 1 (`raw/vgeo3_pathlen.json`) and BEFORE run 2 computed anything.
⛔ **Declared plainly: this addendum is POST-HOC.** It exists because two of SPEC §3's controls failed in run 1,
and it changes ONLY the controls and the P2 matching. **P1's statistic, fitter, split, primary contrast and
verdict table are unchanged**, so the primary cannot be tuned by it.

## 1. What failed in run 1, and why (diagnosed from the run's own numbers)

| control | bar | read | diagnosis |
|---|---|---|---|
| P2 raw-pixel arm | ∈ [0.45, 0.55] | **0.8327** | ⛔ matching defect: "nearest pixel distance in a 3,000-row pool" was never bounded — same-clip goal distances are smaller than almost any cross-clip distance, so the nearest match is still farther and the true goal wins by construction. **P2 in run 1 measures nothing.** |
| C-vshift | \|point\| ≤ 0.03 | **−0.1201 [−0.2603, +0.0166]** | ⛔ design defect: a circular shift of T/3–2T/3 *reverses* a monotone speed trend, so shifted path length is anti-correlated with the true one on accelerating/decelerating clips — not a null |
| C-mut | \|point\| ≤ 0.03 | −0.0315 [−0.0655, +0.0008] | ⚠️ missed by 0.0015 with CI containing 0; the 0.03 tolerance was arbitrary |
| C-const | 0 / 0.5 | 0.0000 / 0.5000 | ✅ |

## 2. Repairs (run 2)

1. **P2 matching tolerance:** the distractor pool is ALL rows of other scored clips; a triplet is kept only if `|d_match − d_true| / d_true ≤ 0.02`. Report the kept fraction per clip; clips with < 20 kept triplets are NaN. Bar for the raw-pixel arm stays **[0.45, 0.55]**.
2. **C-vshift → C-vswap:** path length computed from the speed series of a DIFFERENT scored clip (seeded derangement, truncated / edge-padded to length). Trends then pair with random sign across clips ⇒ null mean 0.
3. **Tolerance for C-vswap and C-mut:** **95 % CI contains 0 AND |point| ≤ 0.05.** (Loosened from 0.03 after seeing a 0.0015 miss — stated, not hidden; C-const stays exact.)
4. **Declared secondary (does not decide):** the same run at **STEP = 2** (twice the pairs per Δ) as a power sensitivity for P1.

## 3. Verdict rule

SPEC §4 unchanged. If any repaired control fails again ⇒ **VOID, no second repair this pass**, and the package reports the P1 contrast as *descriptive only*.
