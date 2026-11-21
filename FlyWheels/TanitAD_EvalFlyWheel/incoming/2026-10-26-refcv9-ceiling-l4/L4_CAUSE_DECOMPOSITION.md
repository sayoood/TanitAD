# L4: cause decomposition of the 12.52 % fed-ceiling exceedance (refcv8)

2026-10-26 · Data Engineering FlyWheel · 0 GPU · MEASURED unless stamped · raw: `raw/l4_decomp.json`, `raw/l4_controls.json`, code `code/l4_decomp.py`, `code/l4_controls.py`

## What the fed ceiling actually is (source, not prose)

* **Built:** `stack/scripts/s2_geom_emit_v7.py:211` `v = p[lo:hi + 1, 3]` with `lo/hi` = anchor + 2 s / + 6 s (`:207-208`, `TACTICAL_S = (2.0, 6.0)` at `:51`); `:309-311` `SPEED_BAND.v_hi_ms = round(float(v.max()), 2)`. Anchor = `egomotion_source.RAW_T0_S = 8.0` s (`egomotion_source.py:57,145`). The v9 manifest stamps `GOAL_PROV.SPEED_BAND = "geometry"`.
* **Snapped:** `stack/scripts/build_refcv6_speed_max_window.py` → sidecar `{v_hi_ms, bin, limit_kmh, one_hot}`, CONTAINING-WINDOW snap-UP over `(30, 50, 100, 120)` km/h, >120 clamps to 120 (`refcv6_max_speed.py:91,166-172`).
* **Read by the trainer:** `refcv8_train.py:150-198` `read_road_type_sidecar` (feeds the step `R8_ROAD_TYPE_KMH[bin]/3.6`, never `v_hi_ms`).
* **Read by the eval mask:** `clamp_posthoc_r8.py:193-213` `fed_ceiling` (`MaxSpeedOneHotEncoder` → `limit_ms_of_bin`, all-zero → +inf), `compliant_reread_r8.py:96-104` `compliant_mask` (`gt_max_speed_6s <= lim`, strict).

⛔ **PREMISE CORRECTION.** The brief describes the fed ceiling as "the Alpamayo (VLM-CoT) speed-band upper edge". It is not. It is the **ego's own realised max speed over [anchor+2 s, anchor+6 s]**, measured once per clip at raw 8.0 s and broadcast to every window of the clip. Its provenance is ego-future (`SPEED_MAX_DERIVATION_V6` says so verbatim). The PI accepted that derivation on 2026-09-16 and 2026-10-11, conditional on road-type quantization.

## Controls (MEASURED, `raw/l4_controls.json`)

| control | result |
|---|---|
| fed value == snap-up(sidecar `v_hi_ms`) | **4,754 / 4,754** windows |
| exceedance reproduced | **595 / 4,754 = 12.516 %** (A-R8-CLAMP-2 quotes 12.52 %) |
| mask identical across trees/seeds (48852ema_pf s0/s1, 30000ema_unc, 48852ema_unc) | identical, all 4 |
| v9 `v_now` (separate artifact) reproduces `gt_max_speed_6s` | max abs 0.375 m/s, 85.6 % within 0.05 m/s; exceedances 601 vs 595 |
| anchor-band rule reproduces sidecar `v_hi_ms` from v9 `v_now` | 84.2 % of 139 clips within 0.2 m/s, median abs 0.036 m/s |
| 30 km/h class exceedance | **374 / 1,612 = 23.2 %** (amendment: 23.2 %) |

## Decomposition (strict `gt_max_speed_6s > fed`)

| cause | eval139 windows (n = 4,754, 139 clips) | train windows (n = 607,899, 4,572 clips) |
|---|---|---|
| **exceeding, total** | **595 (12.52 %)** | **75,082 (12.35 %)** |
| (i) SNAP: compliant against the raw `v_hi`, exceeds only the fed value. Snap-UP cannot lower a ceiling, so this is the >120 km/h CLAMP alone | **16** (2.7 %) | **4,611** (6.1 %) |
| (ii) TEMPORAL: the label was measured at another time in the clip (identity: every non-clamp exceedance has its max-speed instant outside the 4-s label band) | **579** (97.3 %) | **70,471** (93.9 %) |
| ↳ window future entirely BEFORE the label band / overlapping it / NOW AFTER it | 151 / 265 / 179 | 24,214 / 50,130 / 738 |
| ↳ human one road-type step above the fed class / two steps | 541 / 11 | 61,687 / 1,731 |
| (iii) human above a correct posted limit: independent reference = CoT unit-stated sign readings (ungrounded 0/57, ego-coupled; SL-SRC-1) | ref on 3 clips, 104 windows: 16 exceed, **16/16 above the CoT limit** | ref on 36 clips, 4,784 windows: 968 exceed, **799 above the CoT limit, 169 at or below** |
| (iv) missing / unknown band | **0** (all 139 valid) | **0** (4,572 / 4,572 valid) |

Excess above the fed ceiling: eval p50 **3.46 km/h**, p90 13.75, max 23.0 (bins (0,5] 371 · (5,10] 130 · (10,20] 83 · >20 11). Train p50 5.15, p90 13.82, max 50.1.

### Per fed class

| class | eval n win / clips | eval exceed | eval (i) / (ii) | train n win / clips | train exceed | train (i) / (ii) |
|---|---|---|---|---|---|---|
| 30 | 1,612 / 47 | **374 (23.2 %)**, 27 clips | 0 / 374 | 231,491 / 1,741 | 44,187 (19.1 %) | 0 / 44,187 |
| 50 | 1,673 / 49 | 145 (8.7 %), 12 clips | 0 / 145 | 211,867 / 1,593 | 17,514 (8.3 %) | 0 / 17,514 |
| 100 | 1,231 / 36 | 33 (2.7 %), 2 clips | 0 / 33 | 134,815 / 1,014 | 1,717 (1.3 %) | 0 / 1,717 |
| 120 | 238 / 7 | 43 (18.1 %), 2 clips | 16 / 27 | 29,726 / 224 | 11,664 (39.2 %) | 4,611 / 7,053 |

## Reading

1. **The dominant cause is (ii) TEMPORAL BROADCAST: 97.3 % on eval, 93.9 % on train.** One 4-s sample of the human's own speed, taken at raw 8–14 s, is fed as the ceiling for every window of a ~20-s clip. Wherever the human drives faster at another time, the window exceeds. This is a property of the label's *time support*, not of a mis-read scene.
2. **(i) SNAP is the 120 km/h clamp and nothing else**: 16 eval windows, all in 2 clips. Snap-UP is protective. The no-snap raw edge would exceed more (see candidates).
3. **(iii) cannot be measured at scale.** PhysicalAI-AV publishes no map, and egomotion carries no GNSS, so nuPlan posted limits **cannot be geo-joined** (no geographic overlap is establishable). The only independent reference is the CoT sign reading, on 3 eval clips and 36 train clips. Where it exists, 82.5 % (799/968) of train exceedances are a human above the sign as the VLM reads it. That is weak evidence: the reading is ungrounded and ego-coupled.
4. **(iv) is zero.** No missing band on either split.
5. The exceedance is **clip-concentrated**: 43 of 139 eval clips carry all 595 windows.
