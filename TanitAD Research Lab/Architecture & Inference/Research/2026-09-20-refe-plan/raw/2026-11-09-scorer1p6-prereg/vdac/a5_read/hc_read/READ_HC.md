# VDAC A5, HC pair read (frozen reader): HC-s0 is X-NULL. 1 m cells plus a boundary weight sharpen the boundary but lift neither bar.

Master Mind, 2026-11-14.

## Provenance

- **Cards:**
  - `PREREG_VDAC_P6_A5.md` (sha256 `a51b0349…`);
  - amendment A1 (`2b40a672…`) and **amendment A2 (`422541f5…`, frozen 2026-11-13 20:15:55Z, landed `45c5331f` at 20:45:17Z)**. A2 completes HC-s0 without retraining after its chain's post-training gate was told the 2 m head count;
  - the fix landed in `a469d4a5` (GAC `D-VDAC-A5-A2-1`, `D-VDAC-A6-HOLD-1`);
  - reader freeze: `a5_read/READER_FREEZE_A5.md`.
- **Reader:**
  - `eval/s1_read_v2.py`, sha256 `796aea21…`; frozen v1 `s1a_read/s1_read.py`, sha256 `58aa3a77…`;
  - `fill_manifest_sha.py` `0b6510a9…`; the S1-F wrapper `a5_hc_s1f_subset.py` `af12970e…`;
  - all re-hashed today and equal to the freeze table. This is the same code export the HB read used, byte-identical, so the reader tests were not re-run (HB read: `test_s1_read_v2.py` 36 OK, `test_a5_reader_hbonly.py` 7 OK).
- **Manifests:**
  - template `manifest_a5_hc.json`, sha256 `ae10e96f…`, equal to the freeze;
  - read manifest `inputs/manifest_a5_hc.filled.json`, sha256 `cb147151…`. It was filled from independent hashes (`inputs/shas_hc.json`: the HF LFS sha256 of each pulled array) **before any array was opened**.
- **Evidence class:** MEASURED.
- **Population:** 3,207 H48 frames over 45 log groups.
- **Estimator:** the reader's log-cluster bootstrap (n_boot 10,000, seed 20260927). It answers "would another draw of logs say this?" and is blind to training variance, so every number is **single head seed** (`H-ESTIM-SEED-1`).
- **Tier:** a perception read on navtrain H48 (S1), **not a driving number**.

## How the HC pair was completed

- **HC-s0** trained 10,000 updates in stage `A5HC` (17:59 to 21:10Z, 1.07 s/update).
  - Chain run check: `ZZVDAC_RUN_OK` (10,000 updates on 60,000 sets, 1,502 H48 frames forwarded).
  - The chain's post-training gate then FAILED, exactly as A2 section 1 predicted: *"the Adam state covers 3423490 head parameters, the card's built head has 2440450"*.
  - **A2 by hand** (`a5_a2/hc_s0_complete.sh` `3fb48a03`): `postgate_a2` `ZZVDAC_POSTGATE_OK`, 3,423,490 head parameters, every one with Adam state and a non-zero first moment, 10,000 steps.
  - Egress `ZZEGRESS_OK arm_vdac_hc-s0`, 13 files. Hub LFS sha256 equals the pod's for the head and for the export.
  - The initial-head file (`scores/head_pred_step0.npz`, key `hc_init_a`) is not on the chain's egress list, so it was banked separately under `vdac/hc-s0/step0` (`edc86109…`).
  - No chain marker was written by hand.
- **HC-SHUF** ran its own chain on the FIXED tree (`a469d4a5`, shipped tar `e5158817…`, stage `A5HCSHUF`, 00:19 to 01:42Z):
  - preflight, gate, run check (2,000 updates);
  - post-gate `ZZVDAC_POSTGATE_OK` at 3,423,490;
  - export; egress (LFS sha equal);
  - `ZZVDAC_CHAIN_DONE HC-SHUF`.
  - The re-run export is byte-identical to the banked one (`14560686…`).
- **P7** resumed after each pause: 21:10:09Z, then 01:42:50Z.
- **Pulls** (`HF_HUB_DISABLE_XET=1`, `inputs/pull_record_*.json`): pod `sha256sum` = HF LFS = local sha256 for every array.

| key | file | sha256 |
|---|---|---|
| `hc_s0_a` | `scores/head_pred_update10000.npz` | `57c0b79f…` |
| `hc_s0_b` | `head_pred_h48x.npz` | `1c9d97de…` |
| `hc_init_a` | `scores/head_pred_step0.npz` | `edc86109…` |
| `hc_shuf_a` | `scores/head_pred_update2000.npz` | `ae43c944…` |
| `hc_shuf_b` | `head_pred_h48x.npz` | `1e8bbeab…` |

## Result (`out/s1_read_v2.json`, `out/s1_read_v2_tables.md`, `out/hc_s1f_subset.json`)

| arm | S1-P1 IoU [0, 40) (bar ≥ 0.80, point decides) | S1-Q (bar ≥ 0.90) | S1-F lower bounds (P1, Q) | C-SHUF (< 0.50) | C-FLIP (< 0.80) |
|---|---|---|---|---|---|
| HA-s0 (reference) | 0.7802 [0.7587, 0.7974] | 0.8332 [0.7951, 0.8691] | 0.512, 0.089 | 0.4468 | 0.7056 |
| HB-s0 (read 2026-11-13) | 0.8113 [0.7924, 0.8260] holds | 0.8376 [0.7963, 0.8744] fails | 0.545, 0.090 | 0.4464 | 0.7031 |
| **HC-s0** | **0.7757 [0.7528, 0.7932] fails** | **0.8346 [0.7919, 0.8751] fails** | 0.485, 0.164 (holds) | HC-SHUF 0.4447 ok | 0.7064 ok |

**Paired HC-s0 − HA-s0** (same frames, same resamples):
- S1-P1: **−0.0045 [−0.0121, −0.0008]**. This is separated in the WRONG direction: 1 m cells plus the boundary weight cost a little IoU.
- S1-Q: **+0.0013 [−0.0086, +0.0114]**, not separated.
- X-LIFT: False.

**HC-s0 detail:**
- **Per-city IoU:** LV 0.889, BOS 0.733, PIT 0.726, SG 0.684.
- **Bands:** 0–20 m 0.811, 20–40 m 0.740, 40–72 m 0.649.
- **Boundary error:** the near-boundary SDF MAE is **1.241 m**, against HA's 1.630 and HB's 1.436, about 24 % sharper than HA. That is the boundary weight doing what it was built for, on a quantity that is **not a bar**.
- **Sensitivity over all 48 groups:** S1-P1 CI [0.7590, 0.7971], S1-Q CI [0.7946, 0.8690].
- **S1-F:** read on the SFT-held-out subset (1,705 frames, 24 groups, the PREREG 7.5 fallback). The P1 term had 3,787 of 10,000 resamples undefined, which the wrapper reports.

**Verdict (mechanical, card section 8): HC-s0 = X-NULL.**
- Both bars fail and neither delta is separated upward.
- The reader prints "INCOMPLETE" for the S1-FIT vs S1-WEAK distinction, because the in-sample IoU is not in the exports. That distinction does not enter the A5 row.

**Not read** (the reader reports it; none of it decides): `p6_dac_logit`, and S1-QT (`navtrain_metric_cache_t0_ego_states`).

## The pair, read together (card section 8, "Branch read")

| arm | row | committed next |
|---|---|---|
| HB-s0 | **X-BOUND** | **HBC**: 1 m cells plus the boundary weight on the LoRA copy, about HB's cost plus 1.1 L40-h per arm. New pod time, so a **PI decision**. |
| HC-s0 | **X-NULL** | The row stands. The "every arm X-NULL → close the DAC-only perception line" rule does **not** trigger, because HB is X-BOUND. |

## What the three arms say together (MEASURED, single seed; an observation, not a new claim)

- **S1-Q is flat across all three levers: HA 0.8332, HB 0.8376, HC 0.8346.** No paired S1-Q delta is separated.
  - The ceiling the same metric reaches with the TRUE field is 0.9408 (`true_field_auc_2m`).
  - So the 0.83 → 0.90 gap does not respond to a trainable trunk copy (HB) or to finer cells plus a boundary weight (HC).
  - **S1-Q is the clause that gates S2.**
- **HBC combines the one lever that moved IoU (HB) with the one that sharpened the boundary (HC).**
  - Neither component moved S1-Q, so the prior that HBC clears S1-Q is low. The card still commits HBC for X-BOUND, so HBC is put to the PI as written, with this disclosed.
- **Rule Zero, next lever in parallel (zero GPU):** the X-NULL row names the alternative, the **agent check (D-VDAC-6)**.
  - It builds labels from NAVSIM's agent annotations; NC 1.12 % and TTC 1.527 points (`FEASIBILITY.md` F5) make it the larger PDMS term.
  - The card orders "its zero-GPU ceiling read first", which needs no pod time and no PI decision.
