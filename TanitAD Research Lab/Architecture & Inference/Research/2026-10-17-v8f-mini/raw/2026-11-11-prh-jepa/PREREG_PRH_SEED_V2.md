# PREREG_PRH_SEED_V2: instrument amendment after v1's deliberate-regression control failed (frozen 2026-11-11 ~11:53, before any v2 statistic)

**Why v2 exists.** v1 (`PREREG_PRH_SEED.md`, sha256 `411ddb72…`) ran to completion, and its deliberate regression **D-MIX did not separate** from the trained pair:
* `M(TR0,D-MIX)` 0.6796 against `M(TR0,TR1)` 0.6773, delta +0.0023, CI [−0.0044, +0.0092];
* source: `RESULT_PRH_SEED.json`.

By v1's own rule the read is **unusable**, and v1's outcome line (O2) is not a result.

A diagnostic with no bar shows why. `M(TR0, a pure random affine map of nm)` = 0.646: the mean canonical correlation over all 256 directions has a floor of about 0.63–0.65 for unrelated affine maps of the same frozen features, so it has almost no dynamic range here.

There is also a reason in theory. The paper's claim (Thm. 1, Prop. 2) is equality **up to one rotation** after whitening, i.e. orthogonal Procrustes. CCA aligns two representations up to any invertible linear map, which is the weaker, `GL(k)` notion the paper reserves for next-token prediction (Prop. 4). v1 measured the wrong group.

**Disclosure — already seen from v1, and NOT reused for any v2 decision:**
* the eight M values;
* Procrustes R² beside rows: 0.240 trained pair, 0.073 random pair;
* mutual 10-NN: 0.889 vs 0.873;
* P1 +0.0429, P2a +0.0139 / +0.0126, P2b −0.1807 / −0.1852.

Because those were seen, v2 picks its metric on **controls only (Step A)**, by a rule fixed here. Then, in Step B, it reads the arms once with that metric and with v1's predictions, outcomes and estimator unchanged.

## Step A — instrument calibration (controls only; NO statistic involving TR1, RI0, RI1, SLOW or PCA is computed or printed)
Same data, split, frames and bootstrap as v1.

**Candidate metrics** (each a similarity, 1 = identical):
* (a) `CC8` and (b) `CC32`: the mean of the top-8 / top-32 canonical correlations (fit half → score half, as v1);
* (c) `PROC`: whitened orthogonal Procrustes R² = `1 − E‖Ŵa − R·Ŵb‖² / E‖Ŵa‖²`. Both whitenings and the rotation are fit on the fit half and scored on the score half. This is the paper's own notion;
* (d) `CKA`: linear CKA on the score half (no fit).

**Calibration representations, built from TR0 and fresh random affine maps of `nm`:**
* `R_a`: weights drawn by the adapter's own init under torch seed 200;
* `R_b`: the same, seed 201;
* `R_c`: the same, seed 300, kept for Step B only.

**Calibration controls:**
* `SELF` = (TR0, TR0);
* `FLOOR` = (R_a, R_b);
* `MIX(β)` = (TR0, (1−β)·TR0 + β·R_a) for β ∈ {0.25, 0.5, 0.75};
* `SHUF` = (TR0, R_a with frames permuted across score clips).

**Admissibility rule** (all must hold):
1. `SELF` = 1 within 1e-6;
2. `MIX(0.25) > MIX(0.5) > MIX(0.75)` (point values);
3. `MIX(0.5) − FLOOR` separated (paired bootstrap over score clips);
4. `SHUF` below `FLOOR`.

**Choice:** among the admissible metrics, take the one with the largest `(MIX(0.5) − FLOOR) / half-width of its CI`. Ties within 10 % go to `PROC`. If none is admissible, v2 stops with **INSTRUMENT-NOT-FOUND** and names the blocker.

The choice and the Step A table are written to `CALIB_PRH_SEED_V2.json`, and its sha256 is printed **before** Step B runs.

## Step B — the read (once, with the chosen metric `S`)
* Predictions P1 and P2 exactly as v1 with M replaced by `S`.
* Outcomes O1 / O2 / O3 and their committed next steps exactly as v1.
* Controls: C-SELF, C-SHUF and C-ZC as v1 (C-ZC may reuse v1's identity check, since it tests extraction, not the metric).
* C-ROT, for rotation-invariant metrics: a random orthogonal map applied to TR1 leaves S unchanged within 1e-6.
* **A fresh deliberate regression `D-MIX2` = (TR0, 0.5·TR0 + 0.5·R_c), with R_c never used in Step A:** `S(TR0,D-MIX2)` must differ from `S(TR0,TR1)`, separated. Otherwise v2 is unusable as well, and the next step is a PI-visible note that no in-budget instrument answers the question.
* Beside, with no bar: the per-direction profile after Procrustes, ordered by SLOW's canonical correlations (Fig. 6 analogue).

## Scope (unchanged)
* A representation probe, not a driving eval: no tier, the four families do not apply.
* CPU only, zero training, eval-141 read and never re-selected.
