# Where is the navtrain -> navtest gap of the aux drivable-area head? (2026-10-16)

Tier: T0-like (head perception on teacher-forced frames; never driving performance). Evidence class per block: MEASURED (ours, artifact named) unless marked.
Machine twin: `SHIFT_LOCATE.json` (assembled by `g_assemble.py`, every number read from a component JSON). CPU only, dev box, 1-2 processes, read-only on every shared artifact.

**Estimator.** Independent log-cluster bootstrap (the two domains have disjoint log sets), B = 4,000-5,000, population margins fixed. It answers *"would another draw of LOGS say this?"* ONLY. Training-run variance: the three H1 seeds are reported side by side, never pooled into an interval. Inference variance: the head is deterministic (ASSERTED, not measured here). The held-out side is **24 whole logs** (15 LV, 4 BOS, 4 PIT, **1 SG**), so every interval below is wide.

## HEADLINE (ranked)

**The gap is an evaluation-set composition effect, not a label or feature defect.** After direct standardisation, two composition factors account for the point estimate of the gap and the residual is inside the noise:

| rank | located contributor | size on P1 [0,40) (3 H1 seeds, mean; per-seed range) | interval (H1-s0, independent log bootstrap) | class |
|---|---|---|---|---|
| 1 | **CITY MIX.** LV is the easy city (P1 0.67-0.71 in BOTH domains) and is 59 % of held-out frames (65.5 % of the train store) but only **22 %** of the 1,058 navtest C3 tokens (33.5 % of the full navtest). BOS/PIT/SG read 0.31-0.51 in both domains. | **+0.077** (0.072-0.080), Shapley over both orders; 0.095 if applied first | [0.042, 0.120] | MEASURED, descriptive |
| 2 | **FAILURE ENRICHMENT of the navtest sample.** The C3 set is 529 tokens where u400 had NC x DAC = 0 (4.36 % of the 12,146) + 529 random. Zero stratum 0.434, random 0.564. Same direction in all four cities. | **+0.040** (0.036-0.043); 0.058 if applied first | [0.029, 0.054] | MEASURED, descriptive |
| 3 | place novelty (distance to the head's training poses), speed, drivable-share mix | +0.018 alone, **+0.000 after city mix**; speed 0.000 | - | MEASURED |
| - | **residual after BOTH are standardised** (target = held-out city mix) | **+0.013** (0.012-0.015); with the train-store city mix **-0.000** (-0.002..+0.002) | **[-0.071, +0.080]** | MEASURED |
| - | label construction, metric/region definition, precision path, image container | **0** (see below) | - | MEASURED |

* The gap re-derived: held-out 0.6307 / 0.6433 / 0.6211, navtest 0.4957 / 0.5211 / 0.4864 (H1-s0/s1/s2); gap **0.135 / 0.122 / 0.135**. Reproduced bit-for-bit (max abs diff 0.0) from the banked dumps and regenerated heads. The gap itself is only weakly established by the log-cluster bootstrap: **[0.041, 0.214]** (H1-s0), because the held-out set has 24 logs.
* **Honest limit.** The residual's interval is as wide as the gap. What is shown: *no residual shift is detectable at this resolution*, and the cleanest in-city test agrees (LV, natural frames only: held-out 0.706 vs navtest 0.696; gap +0.011, CI [-0.040, +0.069]). It is not shown that no residual shift exists below +-0.07.
* **A layout-prior baseline with NO image** (per-map per-cell drivable frequency from the train store) reads 0.571 on the held-out frames and 0.463 on the C3 tokens: a gap of **0.108 = 80 % of 0.135 before any feature is read**.
* **The K2 reading on navtest is not a rate.** Re-weighted to the natural navtest population (failure stratum 4.36 %, navtest's own city mix; ESS 575) H1 reads **0.561 / 0.577 / 0.554** instead of 0.496 / 0.521 / 0.486. K2 (bar 0.80) fires either way; the number that goes in a table should say which population.
* **The largest measured term for this head is therefore NOT the navtrain -> navtest transfer.** It is the in-distribution shortfall to the K2 bar (held-out 0.632 vs 0.80), carried by missed drivable area (recall 0.73 held-out / 0.59 navtest; false-positive share unchanged, 0.066 / 0.065 of cells) and by the minority cities.

**Next lever (Rule Zero: the refutation of "label/feature shift" is a waypoint).** See section 6: the one cheap run that separates "S1's 400-update head is budget-limited" from "its city prior is the limit" is the V probe at 400 updates on the same 5 folds (dev-box GPU, about 4 min; NOT run here: 0-GPU task).

## 0. Hypothesis table

| hypothesis | verdict | class | where |
|---|---|---|---|
| **H-a** label construction differs between the stores | **NOT FOUND**, four independent layers (L1 provenance, L2 code hash, L3 banked path-A/B gates, L4 data level incl. same-place cross-store agreement and head-as-instrument alignment) | MEASURED (negative) | section 1 |
| **H-b** scene composition | **LOCATED**: city mix + failure enrichment explain 90 % (held-out mix) to 100 % (train mix) of the point estimate | MEASURED, descriptive | section 2 |
| **H-c** P1 computed differently | **NO**: one function, one mask rule, one band; both reads reproduce exactly | MEASURED | section 3 |
| **H-d** feature shift in `visual_ctx` | **PARTIALLY TESTABLE**: precision path, image container and an in-city natural-frame comparison all read null; the trunk-ctx comparison itself is **NOT TESTABLE on this machine** | MEASURED (what ran) / NOT TESTABLE (what did not) | section 4 |

## 1. H-a: label construction (checked hardest)

A construction difference would be a BUG-class lever, so every layer that could have shown one was run; none did.

* **L1 provenance** (`a1_label_construction.json`): `code_sha256` identical in both builds (`1f7a69d8e1f33fc89e05f43c9edd645adeef7e6137ecc33809e8b9f74ce6bfa1`); raster spec equal; `maps_json_sha256` equal (`b9c0a623...`), same map root `nuplan-maps-v1.0`; schema `aux-drv-label-v1` both; 4 workers both. The two differences that exist: **origin source** (train 103,037 keys from the bank's `origin_world` + 102 from the DB; navtest 12,146 all from the DB `ego_pose`, `aux_label_store.py:325` / `:337`) and **log->map source** (train: 463 DB + 729 inferred; navtest: 136 DB). Bank vs DB origin, cross-checked on 37,932 rows: **0.0 m, 8.9e-16 rad** (`run_meta_all.json` `origin_xcheck`).
* **L2 code**: that sha is the tip file's bytes in CRLF form (the dev box's working-tree form; the LF form is `3078b7e9...`). Same code built both stores.
* **L3 banked gates**: train pre-gate AB (path A NuPlanScenario vs path B) **0 cells differing over 80 samples**, origin 0.0 m; navtest gate **0 cells differing over 136 samples** (one per log), origin 0.0 m; both PASS with all six deliberate-regression arms FAILING (`s0_gate_pre.json`, `s0_gate_navtest.json`).
* **L4 data level, per map** (all 103,139 + 12,146 keys; share of drivable cells in the head's 0-40 m window): LV 0.506 vs 0.507, BOS 0.345 vs 0.323, PIT 0.288 vs 0.298, SG 0.295 vs 0.335 (train vs navtest). Ego origin inside the label: 100 % in both stores.
* **L4 the sharpest test (`f1_cross_store_label_identity.py`)**: a label is a function of (map, pose), so navtest frames that sit on a navtrain frame's pose must carry the same fine mask. 5,980 navtest frames have a navtrain frame within 0.5 m (x, y, 20 cos h, 20 sin h). Fine-cell sign agreement at matched pose distance: **cross-store 0.98718, within-train 0.98721, within-navtest 0.98562**; per distance bin the three agree to <= 0.0004 (0-0.1 m: 0.99630 / 0.99648 / 0.99567). **Resolution, stated plainly:** the deliberate regressions on the same pairs read 0.9804 (x +0.5 m), 0.9725 (y +0.5 m), 0.9497 (x +2 m), 0.0130 (sign flip). My pre-written rule "every regression at least 0.02 lower" is therefore met only by the 2 m shift and the flip; a 0.5 m shift is 0.007-0.015 lower. The test excludes a systematic offset of about 0.5 m or more; it does not exclude a few centimetres (which the path-A/B gates cover).
* **L4 the head as an instrument** (`b1_decompose.json` S3): shifting the prediction relative to the label by integer 2 m cells, the lateral optimum is at 0 in both domains for all three seeds; y-flipped prediction reads 0.35-0.37 in both. Longitudinal IoU is flat (navtest best at +1/+2 cells, +0.001-0.002), so this probe cannot see a longitudinal lag; the 0.5 s-offset arm of the gates does.
* **Not done:** a rebuild of a sample of train labels through path A on the 729 inferred-map logs (no DB exists locally for them; their evidence is the 462/462 inference agreement and the per-key QC in the manifest, `qc_rank0_target_points_inside_label` mean 0.99999).
* **No minimal repro is delivered because there is no difference to reproduce. Nothing to escalate on the labels.**

## 2. H-b: scene composition (descriptive; strata overlap, so the shares do not add up)

Sources: `b1_decompose.json`, `b2_standardised_gap_ci.json`, `d1_place_novelty.json`. Re-weighting = frame weights in both the numerator and the denominator of the pooled IoU.

* **Per map, H1-s0** (n frames; logs): held-out LV 0.7065 (1,012; 15), BOS 0.4672 (257; 4), PIT 0.4464 (195; 4), SG 0.3379 (241; **1**). Navtest C3: LV 0.6716 (233; 44), BOS 0.4854 (310; 45), PIT 0.4070 (276; 21), SG 0.3310 (239; 13). Within each city the two domains differ by -0.035 .. +0.018; across cities by 0.37.
* **Standardisation, H1-s0** (3-seed mean in brackets): navtest sample 0.4957 -> failure rate only 0.5557 (0.5588) -> city mix only (held-out mix) 0.5933 (0.5955) -> both 0.6170 (0.6183) vs held-out 0.6307 (0.6317); to the train-store city mix 0.6308 (residual -0.0002; 3-seed mean residual -0.0001). Reverse: the held-out frames re-weighted to the C3 city mix read 0.5253 (0.5308).
* **Failure enrichment** (`b1` S6): zero 0.4264 vs random 0.5613 (H1-s0); LV 0.612 vs 0.696, BOS 0.477 vs 0.494, PIT 0.359 vs 0.467, SG 0.316 vs 0.364, so the direction holds in every city.
* **Speed** does nothing: navtest re-weighted to the held-out speed mix reads 0.4951. **Drivable-share** (IoU rises monotonically with it in both domains) re-weighted alone 0.512; with city 0.602 (ESS 457, 4 strata absent from the held-out).
* **Place** (`d1`): 92 % of held-out frames and 77.5 % of C3 tokens lie within 2 m (position + heading) of a training frame; median 0.34 m vs 0.70 m; 4.8 % of the C3 tokens are >= 25 m away (IoU 0.322). Within the <2 m bin the gap persists (0.640 vs 0.520), so coverage of streets is not the explanation. Re-weighting navtest to the held-out place mix: 0.5138 (+0.018); jointly with city 0.5935 (= city alone).
* **Error anatomy** (`b1` S2, S9): the extra navtest error is missed drivable area: false-negative share of cells 0.147 vs 0.112, false-positive 0.065 vs 0.066. By true-SDF depth the navtest excess sits in cells deep inside the road (SDF +3..+6 m: error 0.286 vs 0.166) as well as at the boundary (+0..+3: 0.519 vs 0.407), i.e. gross misses in the minority cities, not boundary jitter.
* The gap is **uniform in range**: 0.135 (0-20 m), 0.138 (20-40 m), 0.149 (40-72 m) for H1-s0, which is what a composition effect on a layout-limited head looks like.

## 3. H-c: region / metric

`eval/aux_s1_readout.py`: `band_cells` `:100`, `frame_stats` `:309` (true mask = `mask_frac >= 0.5` `:321`, predicted mask = `logit > 0` `:322`, known = all cells of a labelled frame), `perception_reads` `:345`, `k2_decision` `:388`, `K2_BAND = (0, 40)` `:66`. Both banked reads (`perception_heldout.json`, `perception_navtest.json`) are the same tool (`aux_s1_readout/1`), the same function, the same band; only the pinned label store and the frame set differ (`LABEL_PINS` `:75`). Re-derived through the programme's own readout AND through independent array code: held-out six arms **abs diff 0.0**, navtest three arms **abs diff 0.0** (`b0_repro_heldout.json`, `b1_decompose.json` `R0_navtest_repro`). The pooled ratio of sums weights frames by cells in the union, so a frame set with fewer drivable cells reads lower by construction (the drivable-share strata above).

## 4. H-d: feature shift

* **TESTED, null.** (i) *Precision path* (`e1_precision_sensitivity.py`): the held-out dump comes from the trainer's bf16 trunk, the navtest read from the planner's fp32 ctx (`aux_navtest_forward.py:19-21`, default `--head-ctx planner` `:593`). Applying the TRAINED H1-s0 head to the banked fp32 navtest ctx on 200 random tokens: bf16-rounded ctx (rel L2 0.17 %) **+0.0000**, bf16 ctx + head under autocast **+0.0001**, isotropic ctx noise at 1.5 / 3 / 6 % rel L2 (the measured upstream path difference is 1.05-1.46 %) **-0.0001 / -0.0001 / +0.00005**. Against 0.135 that is > 1,000x smaller. Caveat: isotropic noise; the structured bf16-trunk deviation is not emulated. (ii) *Input container* (`c1_image_provenance.py`): 240 navtest frames and 240 navtrain frames (the only train shard on this machine, leading members): all 1920x1080 RGB, identical JPEG luminance quantisation sum 1477 on every file, so no re-encode / resize difference. Cannot see photometric shift. (iii) *In-city natural frames*: LV, non-enriched navtest frames only, vs held-out LV: +0.011 [-0.040, +0.069] (3 seeds +0.005..+0.012).
* **The in-domain probe does not measure "transfer loss".** The scene-tap probe's V heads (S1 head config trained from scratch on ~861 navtest tokens, 5 log-disjoint folds) read 0.635 (2,000 updates) / 0.675 (10,000) on the SAME 1,058 frames where H1 reads 0.496 (both re-derived: V 2k reproduces the banked 0.634891 and V 10k the banked 0.675218 to 3e-7; H1's navtest reads reproduce exactly). Their gain over H1 is **largest where S1's data is thinnest**: LV +0.07 (2k) / +0.10 (10k), BOS +0.13 / +0.18, PIT +0.16 / +0.22, SG +0.21 / +0.25 (`d2_indomain_vs_transfer.json`); on LV natural frames only +0.03 / +0.06. The probe also trains on navtest's own city and failure mix. Its gain does **not** come from street memorisation: 68 % of the C3 tokens are >= 10 m from any other-fold frame and V (2k) still reads 0.611 there vs H1 0.485 (`d3_probe_place_leak.json`); the gain is larger where a neighbouring frame exists (<2 m: +0.23 at 2k) but the far-frame gain is +0.13. So: V - H1 mixes optimisation budget (2,000 vs 400 updates), city prior and failure-mix prior; it cannot be read as a domain-shift cost.
* **NOT TESTABLE here:** the trunk's `visual_ctx` on navtrain frames vs navtest frames (needs the u400 trunk and navtrain pixels; the dev box holds 6,044 of 412,148 train images, INHERITED from `raw/2026-10-13-aux-s1-data-audit/S1_DATA_AUDIT.md`, not re-counted). **UNVERIFIED:** whether the REFe base trunk (not u400's scorer stage) trained on the 24 held-out logs; `P2R_POD_RUNBOOK.md:99` says only that *u400* never trained on them, and the bank the trunk trained on contains them. If it did, the held-out read is optimistic; the LV comparison above shows no gap at this resolution.

## 5. What this does to the register

* "Transfer navtrain -> navtest loses ~0.13" should be restated: **0.13 is the difference between two differently composed frame sets; standardised for city mix and failure rate the point estimate of the gap is 0.00-0.015 (interval +-0.07).**
* S1_READ's `navtest_and_ho_agree: true` compares decisions, not values; the two P1 values are not comparable populations.
* Registry candidates (the Master Mind decides; nothing edited here): a P1 line on the natural navtest population and a per-city P1 table in `aux_s1_readout.py` output; a city-stratified held-out (>= 6 logs per city) for any future held-out read.

## 6. Next lever, with both outcomes committed in advance (a PROPOSAL, not run: 0-GPU task)

**Question left:** the 400-update H1 head reads 0.50 on the C3 tokens; an in-domain head reads 0.635. Is S1's shortfall (a) the 400-update budget or (b) the navtrain city prior (65.5 % LV)?
**Cheapest discriminating experiment:** `scene_tap_probe.py run --updates 400`, arm V, seed 0, the same 5 folds and cached ctx (about 46 s per fold on the 4060, about 4 min; shared GPU lock; identity check = V 2k reproduces 0.634891).
**Outcomes:** V400 <= 0.53 on the 1,058 => budget-limited (lever: updates; A2 already shows 10k > 2k by +0.040). V400 >= 0.605 => not budget-limited; the lever is the head's data prior (city-balanced head sampling or a fine-tune on minority-city frames). In between => both. Either way the lever is NOT a label fix or a feature-alignment fix, and a navtest-vs-navtrain "domain adaptation" run is not indicated by this evidence.
If a navtrain `visual_ctx` cache is ever produced on the pod (24 held-out logs through the planner path), H-d becomes testable directly by a ctx-statistics and head-swap comparison.

## 7. Caveats
* Descriptive standardisation, not a causal model; the factors overlap, hence the Shapley split. Cells with 1-4 logs (held-out SG/BOS/PIT) carry little weight individually.
* ESS after re-weighting: 370-575 of 1,058 frames.
* Seeds H1-s0/s1/s2 agree on every ranking; they are three head initialisations on one training set and say nothing about another training set.
* H2 arms were not regenerated for navtest: the banked H2-s0 minus H1-s0 head fit is about 2e-4 IoU on both sets (S1_READ `instrument/perception_*.json` `head_fit_pairs`; seeds 1-2 not inspected).
* d1/d3/c1/e1 use H1-s0 only.

## 8. Files and how to re-run

Run each with `C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I <script>` from this directory (`shf_common.py` puts the package `eval/` and `refe/` on `sys.path`, read-only). Order: `a2_attributes` -> `s1_regen_navtest_preds` (3 x 8-16 min CPU, writes `C:/Users/Admin/AppData/Local/Temp/shf_scratch/navtest_head_pred_H1-s*.npz`) -> `a1_label_construction` -> `b0_repro_heldout` -> `b1_decompose` -> `d1_place_novelty` -> `b2_standardised_gap_ci` -> `d2_indomain_vs_transfer` -> `d3_probe_place_leak` -> `c1_image_provenance` -> `e1_precision_sensitivity` -> `f1_cross_store_label_identity` -> `g_assemble`. Logs in `logs/`. The regenerated head predictions and the attribute/feature tables are scratch (sha256 in `logs/scratch_artifacts_sha256.txt`); they are bit-regenerable from the scripts and the banked heads/ctx.
