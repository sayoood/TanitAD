# AMENDMENT_1: a disclosed, control-only, post-data amendment to H-SMOOTH-T1-1

Written 2026-11-02 at 15:41 (local, +0100), **before any number of the 16-thread rollouts was read** (state of knowledge in section 1). Card: `PREREG_FROZEN.md`, sha256 `4030d22fcff7102d8849c9b07f51e4be9d7f264d7b653336b3d9b892a9886d67` (not edited). Tier T1-kin. Estimator of every interval: paired episode-cluster bootstrap, B = 2,000, seed 0 (answers *another draw of episodes* only). Evidence class: MEASURED (ours).

## 0. What changes and what does not

**Changes (the controls and the Y side of one contrast, nothing else):**
1. The reference for the decision contrast X - Y becomes **Y'**: fresh `--smooth off` rollouts of all four arms made by the same patched tool, at the same thread count (16), one at a time, in the same session as X. The decision contrast is **X - Y'**; the original **X - Y(banked)** is reported beside it.
2. The tolerance of the controls K1, K2 and the floor clause of K5 is set from a directly measured run-to-run nondeterminism (section 3), not from the card's flat 1e-4.
3. X - P10 uses the **banked P10 rows** (`SOUP_READ.json`, sha256 `dca10c07...`) as its reference, as the card does; the floor recomputed inside each rollout is never the reference.

**Does not change:** the operator S (SG(7, 2) residual, `P = out["p10"]`), the 19 rows and their pools, the outcome rule and its precedence (DOMINANT, HARMS, MOVED, NO EFFECT), the bars (`D1_POINT_M` -0.2415 m, `ADE_GUARD_M` +0.0604 m), the crawl set, the estimator and B / seed, the arms, the checkpoints, the rollout settings, the card text. The functions `better`, `worse`, `read_a`, `read_xy`, `arm_outcome` are imported unchanged from `code/smooth_t1_read.py`.

## 1. State of knowledge at the time of writing (disclosure)

* The first read (1-thread X rollouts against the banked 16-thread Y) printed descriptive outcome words (DAG-OACC NO EFFECT, SOUP MOVED) and refused them because K1 (6.3e-3 m), K2 (2.6e-4) and the K5 floor clause (3.8e-3 m) missed the 1e-4 tolerance (`RESULTS.md`, `READER_FIXES.md`). Those 1-thread tables are **already seen**; this amendment is post-data relative to them.
* The Master Mind then ran all five rollouts again at 16 threads (`smooth_t1_chain_m16.sh`, outputs `F:/TanitAD-scratch/smooth_t1/out_m16/`). Their reader run refused at the K1 extraction: the `--smooth off` rerun's P10 floor `fl` differs from the banked one by more than the extractor's 1e-2 m guard (`logs/read_m16.out`). **That is the only fact about the 16-thread runs known when this is written**: no 16-thread row, control magnitude or outcome word has been read, and the magnitude of that floor deviation has not been read.
* Known and cited: `raw/K1_DIAG.json` (the patched tool in `--smooth off` is bit-identical to the unpatched tip tool at equal thread count; thread count alone moves the rollout) and register `H-CRAWL-01` (the P10 floor path is thread-nondeterministic at the mm level; max 6.5 mm).

## 2. The new rollouts (`smooth_t1_chain_a1.sh`, CPU, 16 threads, one at a time, `CUDA_VISIBLE_DEVICES=-1`)

Same tool file, same source trees (`tree_dag` for DAG-OACC, `tree_li` for SOUP), same checkpoints, same arguments as `out_m16` except `--smooth off`. Tags, all under `F:/TanitAD-scratch/smooth_t1/out_m16/`:

| tag | role |
|---|---|
| `K1OFF_DAGOACC_s0` | Y' DAG-OACC s0 (already exists, run 2026-11-02 15:22) |
| `OFF_DAGOACC_s1`, `OFF_SOUP_s0`, `OFF_SOUP_s1` | Y' for the other three arm-seeds (new) |
| `OFFREP_DAGOACC_s0` | **Y''**: a second `--smooth off` rollout of DAG-OACC s0, same everything (new) |

X = the four existing `SMOOTH_{DAGOACC,SOUP}_s{0,1}` `--smooth sg72` rollouts of `out_m16` (not re-run).

## 3. The measured nondeterminism bound and the tolerances (computed by `code/a1_nondet_bound.py` BEFORE the reader computes any row; written to `raw_a1/NONDET_BOUND.json`)

Between Y' and Y'' (DAG-OACC s0; the same configuration run twice), measure the maxima of the absolute deviation:

* `N_dump` = max over all float fields of the 141 per-episode dump files (metres), `N_floor` = the same restricted to `fl`, `N_tick0` = max over the tick-0 rows of `plan`, `N_ticks` = max over all float fields of the ticks file,
* `N_row` = max over the 19 rows of |delta(Y' - Y'')| from `gapdecomp.decompose` (the same machinery as the rows).

Tolerance of each control class = **max(1e-4, the matching N)**: `tol_dump`, `tol_floor`, `tol_tick0`, `tol_ticks`, `tol_row`. These are bounds from **one** replicate pair, so they say how much the rig moved between two identical runs, not a proof that no larger movement exists; a control that passes at them cannot distinguish a deviation smaller than the measured one.

## 4. The controls as read under this amendment (all stated now)

| control | read as |
|---|---|
| **K1'** (DAG-OACC s0) | Y' against the banked dump and ticks: max abs deviation of the dump <= `tol_dump` and of the ticks <= `tol_ticks`; the deviations are reported whether or not they pass |
| **K2'** | (Y', Y banked) on all 19 rows: |delta| <= `tol_row`; also run for the other three arm-seeds (descriptive); and (Y', Y'') on all 19 rows (the bound's own consistency, passes by construction) |
| **K3**, **K4**, **K6**, **K7** | as in the card, unchanged (K6 = the pytest logs already in `tests/`, no code changed) |
| **K5'** (the four X runs) | tick-0 `plan_raw` equals the banked Y tick-0 `plan` <= `tol_tick0`; `plan` equals `P + SG(plan_raw - P)` recomputed offline <= 1e-4 m (algebra of float64, no thread noise; unchanged); X's floor `fl` equals the banked Y's <= `tol_floor`. X's floor against Y' (same session) is reported beside |

**Status ladder, committed now (one status is printed with the outcome words):**
* **QUOTABLE** : K1', K2', K5' (all three clauses), K3 and K4 pass at the measured tolerances.
* **QUOTABLE ON Y'** : at least one banked-Y comparison (K1', K2', the K5' tick-0 or floor clause) exceeds its measured tolerance while K3, K4, the K5' plan-recompute clause and K2' (Y', Y'') pass. The decision contrast X - Y' decides HARMS and MOVED, X - P10 (banked rows) decides D1-D3; the exceedance is reported as a finding about the banked reference, and the outcome is stated as conditional on Y' being the configuration of the banked Y, which the first read established at equal thread count (bit-identical to the unpatched tip tool, `raw/K1_DIAG.json`).
* **REFUSED** : K3, K4, the K5' plan-recompute clause or the window / grid identity fails (a defect, not noise).
If the outcome word computed on the original X - Y(banked) differs from the word computed on X - Y', both are printed and the word on Y' is the amendment's decision.

## 5. What the report contains (no hand-typed number)

Per arm and seed: the outcome word by the unchanged rule on (X - P10 banked, X - Y'), the number of separated-worse rows of 19 beside it, all four families, X - Y' and X - P10 for the 19 rows, the four crawl rows and ADE6, the original X - Y(banked) beside, Y' - Y (the rig noise on the rows), the descriptive curvature block (mean |kappa| and roughness in the v0 3-8 m/s band, executed path and issued plan before and after S), the controls with the measured bound, and the card's committed next arm for the arm's outcome (unchanged text: DOMINANT re-read on a fresh seed or window set; MOVED, S composed with the curvature-bounded decode; NO EFFECT or HARMS, the curvature-bounded decode as a flag of the same tool).

## 6. Limits, stated before the data

* Each arm is one training run per seed and the planner is deterministic: the interval answers *another draw of episodes* only. The training-variance question (`H-ESTIM-SEED-1`) is not addressed by this amendment and is not addressed by Y'.
* The amendment moves the reference of the controls and the Y side after seeing the 1-thread tables. It is disclosed, and the 1-thread tables stay in `RESULTS.md` as read.
* No operator, row, threshold or arm is touched. If the outcome word on Y' disagrees with the 1-thread descriptive word, the amendment reports it as it is.
