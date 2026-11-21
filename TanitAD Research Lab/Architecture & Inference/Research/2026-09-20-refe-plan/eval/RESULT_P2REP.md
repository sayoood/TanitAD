# RESULT: P2's training-seed replicate (P2-s1) — **NOT REPLICATED**: RR1 fails on an ADVERSE TTC, not on PDMS

**Master Mind, 2026-10-16.** Read mechanically by the dev-box auto-reader (`raw/2026-10-16-p2s1-read/`, `p2rep_read.py`) exactly as `eval/PREREG_P2_REPLICATE.md` §3 commits. All 25 gates PASS, and P2-s0 vs u400 reproduces the banked R2.

- **Tier:** T1 = NAVSIM v1.1 PDMS on the full navtest (12,146 tokens, 136 logs).
- **Estimator:** paired log-cluster bootstrap, 10,000 resamples, seed 20260927. It answers *another draw of LOGS* only.
- **Δ_seed:** ONE draw of the run-to-run difference between two training seeds, so F is a one-draw estimate, never a variance.

## The numbers (MEASURED, PDMS ×100, paired 95 % CI)
| sub-score | RR1: P2-s1 − u400 | RR2: P2-s1 − P2-s0 | P2-s0 − u400 (banked) |
|---|---|---|---|
| **PDMS** | **+0.90 [+0.36, +1.59]** | **+0.33 [+0.04, +0.64]** | +0.57 [+0.02, +1.18] |
| NC | −0.17 [−0.36, +0.03] | −0.04 [−0.16, +0.09] | −0.13 [−0.31, +0.08] |
| DAC | +0.65 [+0.11, +1.37] | +0.35 [+0.09, +0.64] | +0.30 [−0.23, +0.89] |
| EP | +1.80 [+1.33, +2.41] | +0.65 [+0.33, +1.01] | +1.14 [+0.61, +1.70] |
| **TTC** | **−0.49 [−0.97, −0.07] ADVERSE** | −0.18 [−0.47, +0.09] | −0.31 [−0.80, +0.14] |
| C | +0.02 [+0.00, +0.05] | +0.01 [+0.00, +0.03] | +0.01 [+0.00, +0.03] |
| DDC | −0.07 [−0.36, +0.23] | +0.08 [−0.07, +0.24] | −0.16 [−0.43, +0.13] |

PDMS means: u400 88.2469, P2-s0 88.8184, **P2-s1 89.1493**.

The seed floor is F = |Δ_seed| = **0.33**, which is below min(D_s0, D_s1) = 0.57.

## The committed outcome, as written
- **RR1** is amendment §4's rule: lo > 0 AND no PDMS sub-score adverse.
- P2-s1 clears the first clause (PDMS lo +0.36) and **fails the second**: TTC −0.49 is separated below 0.
- ⇒ **NOT REPLICATED**, with the committed meaning *"P2's lever claim is not supported"*.
- ⚠️ **The committed table's gloss does not describe THIS failure.** Its gloss reads "P2's +0.57 lies within run-to-run variance", but the measured F (0.33) is BELOW both seeds' Δ.
  - What did not replicate is **the safety clause**. The size of the PDMS gain did replicate.
  - The outcome stands as committed. The gloss is noted here; no goalpost moves.
- **The signature** is the same progress-for-TTC trade seen in every RL arm (R0 −1.00, R0b −0.57/−0.70, R0c −0.53/−1.10):
  - EP +1.80 [+1.33, +2.41];
  - TTC −0.49;
  - P2-s0's own TTC was −0.31 [−0.80, +0.14], same direction, not separated.
  - It is now on BOTH a supervised joint fine-tune and the RL arms. That is the strongest MEASURED support so far for the PI's separate-safety-head experiment (approved 2026-10-16, stage 1 running).

## Families (RR3; reported, not gating)
- **LONGITUDINAL / LATERAL:** 21 components compared per pair, none adverse (see `p2rep_result.json`). TTC and EP above are the NAVSIM longitudinal members.
- **TACTICAL:** rows without an interval, in the JSON.
- **STRATEGIC:** PAUSED by the PI (2026-09-19, item 25), and unavailable in NAVSIM.

## What follows (committed, never a stopping point)
1. **SFT-4R** (scorer-only on complete labels), per the NOT PROVEN row of `PREREG_P2_AMENDMENT_1.md` §4. Then a seed replicate of the better arm.
   - Tooling: `refe/sft4r_{relabel,chain,pod}.sh`, `refe/test_sft4r_chain.py`.
   - It has never had a dry run (`HANDOFF_2026-10-11_SFT4R_READY.md`).
2. **P6's training-seed floor:** F = 0.33 is a VALID read (`PREREG_P6.md` "Training variance" item 1), so the S1 no-head arms are not needed for it.

## Artifacts
- `raw/2026-10-16-p2s1-read/out/` (reader output, logs, state)
- `raw/p2_readout/p2s1_full/` (`r2_result.json`; `arm/` = the scored CSV md5 `8dce98b8…`, counts, manifest, seam npz, report; `MD5SUMS.txt`)
