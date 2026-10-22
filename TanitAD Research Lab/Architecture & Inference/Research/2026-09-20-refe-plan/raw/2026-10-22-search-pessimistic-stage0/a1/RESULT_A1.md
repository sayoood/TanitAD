# SEARCH-PESSIMISTIC Stage 0, amended read A1 (S = 3): **NULL, UNDERPOWERED** (2026-10-22, MM)

**What was sealed:**
- `PREREG_STAGE0_A1.md` (sha256 `8fd4bc2a…`), sealed after the calibration and before the read.
- The amendment text is the committed draft `DRAFT_AMENDMENT_A1.md` (sha `7d6ca9f4…`, recorded before the calibration launched).
- Code: `stage0.py` `4eb4ca6b…`, constants digest `cb065815…`.

**Calibration:** `SELECT S=3`. At S = 3: C-II-b reads κ = 2, null-world false positives 0/24, null-world INVALID 0/24.

**Run:** `run_stage0.sh` → pytest 74 passed, then the read (`raw/RESULT_STAGE0_A1.json`).

**Evidence class:** MEASURED, SELECTION tier, label-only. These are head-level gaps on the 24-log table (21 sessions). It is not T1.

⚠️ **Disclosed post-seal change: TEST FILE ONLY.**
- The first run aborted on red. `test_constants_digest_changes_with_any_constant` used `N_SEEDS = 3` as its "different" value, which became a no-op once S = 3 was the real value.
- It now uses `S0.N_SEEDS + 1` (test sha `f45c4395…`).
- `stage0.py`, its constants and the prereg are untouched.

## Controls: ALL PASS

C-STRUCT, C-I-eval (×100), C-I-trained (×10), C-II-a, C-II-b (κ = 2), C-III-real.

## The gaps (IN − OOF; positive means the head scores UNSEEN sessions worse)

| measure | point | Bonferroni [lo, hi] | swap-null ν | Δ | seed replicate | LOSO stable | GAP conjuncts failing |
|---|---|---|---|---|---|---|---|
| M1 in-fan concordance | **+0.0122** | [+0.0005, +0.0281] | 0.0049 | 0.01 | +0.0174 | yes | `lo > ν` |
| M2 A3 concordance (Step A, 201 tokens) | **+0.0256** | [−0.0023, +0.0553] | 0.0219 | 0.02 | +0.0238 | yes | `lo > ν` |
| M3 argmax regret (PDMS points) | **+0.25** | [−0.91, +1.56] | 0.29 | 0.35 | +0.38 | no | uncertified (SE 0.51 > Δ/4); below Δ |

For context only (never a gap): P2's own head on the same search-half tokens reads M1 0.759, M2 0.765, M3 8.00 points.

## Reading, as the sealed rule licenses it

- **NULL-UNDERPOWERED.** No head-level session gap is detected, AND a gap of size Δ is **not excluded**: `hi_1` 0.028 > Δ_1, and M3 is uncertified.
  - The sentence "pessimism cannot be tested on P2" is NOT licensed.
  - **Stage 1 does not run.**
- **Descriptive, not a claim:**
  - All three point estimates have the same sign (unseen sessions scored worse), and both seed replicates agree.
  - M1 and M2 sit at or above their Δ, but inside the swap-null noise.
  - It is consistent with a small head-level effect that 21 sessions cannot resolve. It is equally consistent with none.

## The MM's decision on the routing (sealed §7: the decoder-level variant against D-SESSION-RESERVE)

**DEFER the decoder-level variant. The SEARCH-PESSIMISTIC line on P2 is PARKED behind P6 and D-SESSION-RESERVE.** Why:
1. **The binding limit is the number of sessions, not head capacity.** A1's calibration shows the instrument reads only the κ = 2 planted world. The bootstrap over 21 sessions and the swap-null floor do not shrink with a bigger model. A decoder variant (~1.5 GPU-h plus a build and a new calibration) would face the same floor.
2. **The dev-box GPU is held by another session's queue.**
3. **The lever with the largest measured effect is the scorer itself.** C1 against C2 is −0.71 points between seen and unseen sessions. That points at P6, whose rebuild ends tonight, and at a session-reserved trunk (D-SESSION-RESERVE, PI).
4. If the PI wants the decoder variant run anyway, it is unblocked apart from the GPU lock. Name it and it runs.

## Deliverable manifest

| artifact | where |
|---|---|
| the sealed A1 prereg and sha, the committed draft, `HANDOFF.md`, run scripts, `code/` (with `calib_a1.py` and the golden digests), `raw/` (calibration, pre-flights, census, pytest, `RESULT_STAGE0_A1.json`, run log, test-fix shas) | repo, this directory |
| the first read's full `RESULT_STAGE0.json` (S = 1, INVALID; withheld until now) | repo, `../raw/RESULT_STAGE0.json` |
