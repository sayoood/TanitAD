# PREREG_L1C — AMENDMENT 1 (operational; sealed BEFORE any fresh clip is selected, downloaded, built or rolled)

*Architecture & Data agent, 2026-10-25 ~14:30 Berlin. Parent: `PREREG_L1C.md` sha256 74b8ba88… (unedited). Nothing here moves a bar, an arm, a control
value or an outcome; it fixes what the parent's section 1 leaves ambiguous or states on a premise that MEASUREMENT contradicts, and it commits
the diagnostics that make the result readable. Evidence for every statement below: `FEASIBILITY.md` (MEASURED, artifacts named there).*

## A1.1 The clause "the SAME r0 selection filters the corpus used" is read as "the filters the corpus actually satisfied"

MEASURED 2026-10-25 (`F:/l1c/work/r0_gate_flat.py`, `corpus_r0_scores_flat.parquet`, all 4,719 corpus clips scored with `physicalai_r0._urban_score_from_egomotion`):
only **71.7 %** of the corpus passes the r0 hard gates (mean speed 2–14 m/s, stop fraction <= 0.8, distance >= 40 m); **26.3 %** have mean speed above
14 m/s (highway); the corpus spans all three default splits (train 2,786 / val 1,240 / test 693) while `physicalai_r0._load_catalog` keeps `split == train`
only; and the corpus intersects the r0 / r1 selection files in 38 / 144 clips. The corpus was selected by **availability of the Alpamayo2 augmentation**
(`s2_run_corpus.py:scope()` = "the 4,729 Alpamayo clips intersected with available egomotion"), which a fresh clip cannot have. Applying the r0 gates "unchanged"
would therefore build a calibration split that excludes every highway clip the eval set contains. **Operative rule:** the eligibility tests are E1–E5 below.
The r0 gates are NOT applied as filters. Each fresh clip's r0 status is computed from its egomotion and recorded, and the r0-passing subset is scored
as a BESIDE sensitivity (**S-r0**, no bar, never relabels the verdict).

## A1.2 Eligibility (public metadata and the clip's own calibration only; implemented by `code/l1c_select.py`)

* E1 `clip_is_valid`. E2 not in the programme's 4,719-clip universe (local mp4 mirror; sorted-id digest asserted `a48251e89c7a8603…`).
* E3 not in the programme digest lists (parity train 2,400 · deployed val40 · v7.2 eval 147) by sha256(clip_id).
* E4 `splits/alpamayo_aligned` != `alpasim` (reserved for AlpaSim closed-loop evaluation; the corpus holds 18 such clips, the fresh split holds none).
* E5 front-wide camera 1920x1080 with principal point cy in the corpus's observed rig support: rig A cy in [530, 560) or rig B cy in [740, 770)
  (corpus: A 1,996 / B 2,723; min 533.95, max 764.52; no corpus clip in (560, 740)).
* Order: sha256(clip_id) hex ascending. Candidates are processed in that order; the calibration split is the **first 300 whose whole input chain completes**
  (fetch, episode build, label emission, clock-sidecar row, v9 row). A clip failing any step is logged by sha12 with the reason and the next in order is taken.
  Candidates processed speculatively beyond the 300th are discarded unrolled. If fewer than 300 of the first 420 rig-eligible candidates complete, n < 300 is
  reported as n; no rule is relaxed.

## A1.3 How every input the eval139 roll reads is produced for the fresh clips (no substitute input; `FEASIBILITY.md` has file:line)

The roll builds its eval dataset with `with_perception_targets=False` (`r7_roll.py:401-402`): the agent join, the SAM3 map GT, join3d and VIS-1 are NOT read.
It reads: the v2 episode cache, the v8 label record per clip (nav token, `SPEED_BAND`, tactical class ids), the speed-max sidecar (**`v_lim`**), the clip-clock
sidecar, the v9 per-frame release (nav feed), the per-clip extrinsics table (the model's lift bank must cover every episode id), and the model-build files that
are clip-independent (anchors, tau file, class weights, join-defect masks, vis1 name only). Builders, all committed in the launch tree fb47a58:
`v2_compressed.py build` CLI unchanged (`--hfov 120 --height 416 --width 1024 --projection-mode cylindrical --codec png`, fed from a fetched root in the
layout the builder already supports) · `s2_geom_emit_v7.emit_one` · `build_v8_speed_max_labels.py` + `build_refcv6_speed_max_window.py` ·
`build_clip_clock_sidecar.py` · `build_v9_labels.py build` · `physicalai_front_wide_intrinsics.csv` rows from the dataset's own calibration · the extrinsics table
= the shipped 4,711-clip table plus the fresh clips' `sensor_extrinsics` rows (same format).
* **`v_lim` branch: PRIMARY** (the corpus's own pipeline can produce it: `SPEED_BAND.v_hi_ms` is geometry-only; reproduced 139/139 on eval139 with the Alpamayo
  augmentation absent). The fallback branch of the parent's section 1 is NOT taken; all three `v_lim` features stay in fit and in scoring.
* The Alpamayo augmentation (CoT, boxes, `meta_action`) does not exist for a fresh clip. It is absent for every fresh record and the loader opt-out is stamped.
  MEASURED effect on eval139: nav token / tactical actions differ in 2 of 139 clips (1.44 %), exactly the two whose shipped record carries a CoT-driven
  `turn_suppression`; `SPEED_BAND` and `g_str` differ in 0. This is a stated limitation of the fresh labels, not a control.
* The v9 release is built with the committed `build` subcommand without the lead pass and without the lane-change pass (both feed TARGETS only, never the model's
  inputs; the lead pass needs the obstacle join).
* The pose clock is the STANDALONE one (`R8_NO_POSE_SYNC=1`), as in both L1 rolls.

## A1.4 Window grid, populations, roll

* Fresh windows: every dataset-index window with `t % 5 == 0` (`ws = t + W - 1`), the stride and rule of the eval S2 grid and of the A6 train139 roll.
* Roll: `l1c_ztac_roll.py --mode eval` = `l1_ztac_roll.py --mode eval` (refcv8 EMA-48,852, md5 b4364029…, `roll_seed_r8_pf.py` per-frame nav feed, `--os-only`) with the
  kit root (`R8_EVAL_KIT`) pointed at a fresh kit directory of identical layout, the three remap entries (extrinsics, clock sidecar, v9 release) re-pointed, and
  `args.r8_v9_eval_md5` set to the fresh release's md5 (the loader's own check; the train shim does the same). Inference seed 0 (z_tac is upstream of the sampler:
  bit-identical across seeds, L1 K6).
* Fit rows (H2c, H2cs): the fresh S2 ceiling-compliant rows (compliance mask by `compliant_reread_r8.prepare_seed`, model-free). S6 is secondary with S6-fit heads.
* Estimators, tiers, the inference-seed floor (-0.0021), P1, P2, G, K1–K4, V-m and D-fresh are exactly the parent's.

## A1.5 Diagnostics committed now (beside, no bar; they decide how a result is READ)

* **D-platform.** The fresh episodes are decoded and resampled on the dev box (x86 PyAV); eval139's shipped episodes were built on Thor. MEASURED on 2 eval139 clips:
  poses and actions bit-identical, frames within 3/255 (`FEASIBILITY.md`). The 24 eval139 clips that are first in sha256 order are therefore rebuilt on the dev box with
  the same chain and rolled; **D-platform** = the domain AUC (PCA-32 logistic, episode CV) of rebuilt-vs-shipped z_tac on the same windows, plus the relative
  z_tac and plan differences. Reading rule: if D-platform > 0.70, D-fresh is reported "platform-confounded" and cannot be read as a distribution finding.
* **S-r0** (A1.1), **S-rig** (rig A and rig B fresh clips scored separately beside), and the fresh split's speed distribution against eval139's.
* Braking AUC (bar G) is `l2_roc.py` unchanged on the S6 compliant windows, paired against H0 as in L1 Amendment 2.

## A1.6 Hygiene

Raw clip ids exist only under `F:/l1c/`; everything under `refcv9_l1c/` carries sha12. The fresh split's z_tac bank, caches and mp4 live on F:. Nothing on E: (read-only), G:, Thor. No git.
