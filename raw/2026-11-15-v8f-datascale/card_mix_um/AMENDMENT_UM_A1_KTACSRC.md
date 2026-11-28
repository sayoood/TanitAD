# AMENDMENT UM-A1: K-TACSRC empty-batch bound (MM, 2026-11-18 ~16:30Z, written before any UM_s0 evaluation number exists)

## What happened (MEASURED)

* UM_s0 trained the full 61,568 steps (commit 648d4ca3, gate token with 20 of 20 checks True).
* The chain's `um_checks post` refused with `tac_src_empty_batches 10 > 5`.
* The card (`CARD_DSCALE_MIX.md` line 266) says: "`tac_src_empty_batches` ≤ 5 (E: under 1). Any miss is BLOCKED (reach)."
* **As committed, UM_s0 is BLOCKED (reach).** That is the outcome of record unless the PI accepts this amendment.
* Every other K-TACSRC field read its committed value:
  * `tac_src_w` 0.0;
  * list sha256 37f4f9ee;
  * `tac_src_weight_share_listed` exactly 1.0;
  * `rows_other` 2,708,437 > 0.
* Post gate: Adam state on 252 of 252 parameters. The checkpoint (sha256 39de0b05) is banked on private HF.

## Cause (MEASURED by a deterministic replay)

`/root/dscale10k/blockum/replay/um_empty_replay.py` runs the trainer's own `build_data`, `make_tac_src` and `ChunkedSampler` at seed 0 and chunk 48, on tree_um at 648d4ca3. It replays the 61,568 steps over indices only. Result (`um_empty_replay.json`):

| quantity | run (summary.json) | replay |
|---|---|---|
| batches with no listed (corpus) row | 10 | **10** |
| rows listed | 1,223,523 | **1,223,523** |
| rows other | 2,708,437 | **2,708,437** |

* The 10 batches have 1, 2, 1, 1, 1, 2, 2, 2, 1, 1 windows: 14 rows of 3,931,960 (0.0004 %).
* Each is the partial last batch of a 48-clip pool. The sampler emitted 1,192 such partial batches over the 4 epochs.
* The card's estimate ("under 1") assumed every batch is 64 iid rows. The sampler's pool structure makes tiny end-of-pool batches, and a 1-row batch from a pool with 69 % other rows is empty with probability about 0.69.
* So the miss is a deterministic property of the sampler. It is not a reach defect: the replay reproduces the run's row counts exactly, which is a stronger reach test than the bound.

## The amendment (proposed by the MM; it takes effect only with the PI's agreement)

* K-TACSRC's empty-batch item becomes: **`tac_src_empty_batches` equals the deterministic replay of the run's own sampler, AND the replayed listed / other row counts equal the run's exactly.**
  * That is an exact test, not a bound.
* UM_s0 passes it: 10 = 10, 1,223,523 = 1,223,523, 2,708,437 = 2,708,437.
* The read reports the K-TACSRC miss as written and this amendment beside every UM_s0 label.
* Every other card rule is unchanged. No UM_s0 evaluation, dump or trainer-validation number had been read when this was written.
* The training-loss effect of the 10 batches is that 10 of 61,568 steps had no tactical gradient (0.016 %). The same sampler structure applied to every arm of the study.
