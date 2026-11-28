# SEARCH-PESSIMISTIC Stage 0 — first sealed read: **INVALID (C-II-b)**. The gap is not read. (2026-10-22, MM)

**Pre-registration:** `PREREG_STAGE0.md`, sealed by the MM before any real read, sha256 in `PREREG_STAGE0.sha256` (`9cf8980d…`). Constants digest `c71c7dd7…d811c`.
**Run:** `run_stage0.sh` → pytest 56 passed, then the read: CPU only, a few minutes, `raw/run_stage0.out`.
**Evidence class:** MEASURED, SELECTION tier, label-only. This is not a T1 number.

## Outcome

**INVALID.** The one failed control is **C-II-b**, the planted-session positive control. Per the sealed rule (§7), the real-token gap is NOT read.
- The full `raw/RESULT_STAGE0.json` contains those gaps. It is **withheld** (dev box only, `C:/Users/Admin/par18/spess0/raw/`) so that the amended read stays blind.
- The bank holds `raw/RESULT_STAGE0_controls_only.json` instead.

| control | result |
|---|---|
| C-STRUCT | pass |
| C-I-eval (100 shuffles) | pass |
| C-I-trained (10 worlds) | pass |
| C-II-a (non-reversal) | pass: SEEN − OOF on fit tokens +0.022, token z 8.6 |
| C-III-real (exclusion deleted) | pass |
| **C-II-b (planted)** | **FAIL: no κ read** |

## Diagnosis (planted worlds only: real features, SYNTHETIC labels; no real label was read)

| κ | planted M1 gap | lo (Bonferroni) | swap-null ν | SEEN beats OOF | collapse when exclusion deleted |
|---|---|---|---|---|---|
| 0.5 | 0.014 | 0.002 | 0.005 | yes | 0.014 → 0.001 |
| 1 | 0.019 | 0.007 | 0.010 | yes | 0.019 → 0.002 |
| 2 | 0.025 | 0.007 | 0.009 | yes | 0.025 → 0.001 |
| 4 | 0.035 | 0.004 | 0.008 | yes | 0.035 → 0.004 |

**The instrument SEES the planted effect.**
- The gap rises with κ.
- SEEN beats OOF at every κ.
- Deleting the exclusion line collapses the gap.

**It fails the conjunct `lo > ν`.**
- On the real features, two IN models scoring the same tokens already differ by ν ≈ 0.005–0.010. That is training noise from fold composition and the seed.
- The Bonferroni lower bound over 21 session clusters stays at 0.002–0.007.
- The synthetic power simulation (§8) under-stated both.

So the prereg's listed cause, *"the recipe is too constrained to absorb a session"*, is **not** what failed: the recipe absorbs it. **The failure is the noise floor.**

## Next arm (RULE ZERO): amendment A1, the cheapest fix

- Average each arm over **several training seeds**. The same averaging applies to OOF, IN and every swap-null model, so the comparison stays symmetric.
- This reduces ν and the gap's own model noise.
- The seed count and any α change are chosen on PLANTED and κ = 0 worlds only: power and false-positive rate.
- Then re-seal and run the real read once.
- If A1 still cannot read κ ≤ 2, the 21-session table cannot power a head-level test, which makes the read **INSTRUMENT-LIMITED**. The route is then `D-SESSION-RESERVE` (PI) or the decoder-level variant (~1.5 GPU-h).

## Deliverable manifest

| artifact | where |
|---|---|
| sealed prereg, its sha256, the draft, `BUILD_NOTES.md`, `code/` (with `MANIFEST.sha256`), run scripts, `raw/` (pytest, mutation census, power sim, rehearsals, input checks, run log, the controls-only excerpt) | repo, this directory |
| `raw/RESULT_STAGE0.json` (full, with the real gaps) | **dev box only**, `C:/Users/Admin/par18/spess0/raw/`. Withheld until the amended read concludes; bank it then. |
