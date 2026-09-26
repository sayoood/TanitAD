# PRE-REGISTRATION — what explains the UniAD GT-collision-floor gap? (written BEFORE computing anything)

**Registered** 2026-09-26 by the EvalFlyWheel orchestrator, sha256 in `PREREG_UNIAD_GT_FLOOR_GAP.sha256`,
staged before the discriminating number exists. **Asked by** the Master Mind: *"Pre-register what would
discriminate a harness convention from a genuine difference before chasing it."*

## The gap (MEASURED, banked)

UniAD-protocol GT-collision floor, run `20260926T131756Z-nuscenes_ol-none-21abb2`, n = 6,019, vs PARA-Drive
Table 8 (banked primary `paradrive-cvpr2024`, p.8), UniAD GT row:

| | 1.0 s | 2.0 s | 3.0 s | avg |
|---|---|---|---|---|
| ours | 0.3847 | 0.4244 | 0.4298 | 0.4130 |
| PARA-Drive | 0.35 | 0.38 | 0.35 | 0.36 |
| ratio | 1.10 | 1.12 | 1.23 | 1.15 |

⭐ Two facts frame it. The ratio **grows with horizon**. And the **VAD** protocol, after the name fix, matches
PARA-Drive to **1.5–3 %** — so the harness CAN reproduce this table closely.

## The code fact that suggests H1 (read BEFORE registering; the number it predicts is NOT yet computed)

`per_timestep_means` (`taniteval/adapters/nuscenes_planning.py`) uses TWO denominators:
* model-arm collision: `obj_box_col[sc].sum(0) / n` — over ALL scored samples, *"masked steps contribute 0"*
  (UniAD's reference `compute()`);
* **the GT floor**: `(gt_box_col[sc] * v).sum(0) / v.sum(0)` — over VALID timesteps only.

The numerators are identical (invalid steps contribute 0 either way). So the all-samples value is EXACTLY
`ours × n_valid(t) / n_all`, computable from the banked `gt_valid_n` / scores without any re-run.

## Hypotheses and what each PREDICTS — written before the discriminating number exists

| id | hypothesis | kind | predicts |
|---|---|---|---|
| **H1** | PARA-Drive divided the UniAD GT floor by ALL samples (UniAD's reference convention); we divide by VALID timesteps | **harness convention** | (a) `n_valid(t)/n_all` **decreases** with t; (b) `ours × n_valid(t)/n_all` lands within **5 %** of PARA-Drive at **every** horizon |
| **H2** | the UniAD occupancy genuinely differs (e.g. 0-point / invisible boxes counted — the builder documents that they COUNT) | **genuine difference** | the denominator correction does NOT close it: residual stays near the current 10–23 % |
| **H3** | PARA-Drive evaluated a different sample set | a scale factor | a CONSTANT ratio across horizons — already in tension with the observed 1.10 → 1.23 growth |

**Why 5 %:** if H1 is the whole story, the UniAD residual should look like VAD's after its fix (1.5–3.0 %
observed); 5 % allows margin over that, fixed now.

## Outcomes, committed in advance

| outcome | reading | action |
|---|---|---|
| **H1 CONFIRMED** — (a) and (b) both hold | the gap is OUR GT-floor denominator, not the world | report BOTH conventions, labelled; align the GT floor's default with UniAD's reference `compute()` (the convention the model arms already use) and re-pin the reproduction |
| **H1 PARTIAL** — residual 5–15 % | part convention, part genuine | report the corrected value AND name the remaining gap as open; test H2 next (filter 0-point boxes) |
| **H1 REFUTED** — residual ≥ ~15 %, or (a) fails | not the denominator | test H2 next; H3 if the ratio flattens |

⛔ No threshold is moved after the number is seen. ⚠️ Whatever the outcome, H1 is an inference about how
PARA-Drive computed its floor — the paper does not state the denominator. A pass means *"consistent with
UniAD's reference convention"*, not *"we have read PARA-Drive's code"*.
