# SCOPE_DATASCALE (FROZEN before any pilot clip was touched; 2026-11-15)

Status: the PILOT (200 clips + 3 controls) is approved by the brief. The FULL encode is NOT approved.
Evidence classes: MEASURED (artifact path), PUBLISHED, INHERITED, ESTIMATED, HYPOTHESIS.

## 1. Source and what was measured before freezing (metadata only, no clip fetched)
* Source: `nvidia/PhysicalAI-Autonomous-Vehicles`, revision `14ed7a77897edafb113dc86fde8b8173ae7fb028` (gated; the pod's HF token reaches it, MEASURED
  `/root/datascale/meta/`). 3,146 chunks of about 97 clips; `clip_index.parquet` lists **306,151** clips, all `clip_is_valid`
  (the census file says 306,152: a 1-clip difference, UNVERIFIED which is right; this scope uses the index).
* Range reads of single zip members work (MEASURED: front-wide chunk 0100 is 1.53 GB, 100 mp4 of 15.3 MB; one member read 12.2 MB in 1.3 s;
  the zip directory opens in 2.3 s). So a clip costs about 15 MB of download, NOT a 1.5 GB chunk. Sampling is per clip, not per chunk.
* Splits: NVIDIA `default` = 153,625 train / 90,927 val / 61,599 test; `alpamayo_aligned` = 276,284 train / 28,317 test / 1,550 alpasim.
* MEASURED (`/root/datascale/selection/CENSUS_ELIGIBLE.json`): **our own v8f-mini corpus does not respect NVIDIA's splits** (train4572: 2,688 default-train,
  1,213 val, 671 test; eval141: 92 / 27 / 22). So "NVIDIA test never enters" is a STRICTER rule than the v8f-mini corpus follows. It is kept because the brief says
  every eval/test clip never enters.

## 2. Eligible pool (the sampling frame)
A clip is eligible iff ALL hold:
1. `clip_is_valid`; features egomotion, camera_front_wide_120fov, camera_intrinsics, sensor_extrinsics, vehicle_dimensions present (306,151 of 306,151).
2. NVIDIA `splits/default` == `train` AND `splits/alpamayo_aligned` == `train` (so no val, no test, no alpasim in either split definition).
3. Not in: our train4572 (`/root/ops/train4572.txt`), our eval141 (`/root/ops/clips141.txt`), the HF corpus `Sayood/tanitad-v7-training-corpus` camera id list (4,719),
   the pod v2ep ledger (4,713), NVIDIA `challenging_scenarios` (1,156), NVIDIA `ood_reasoning` val (290), the parity train 2,400 digests, the deployed val40 digests,
   the v72 eval 147 digests (`stack/tanitad/data/*_clip_digests.json`, sha256 of the clip id).
4. The final list passes `tanitad.data.parity.guard_corpus_build(role="train", mode="refuse")` (it must keep every clip).

MEASURED pool size after all steps: **133,929 clips** (1,566 chunks; 70,205 US, 19,949 DE, 25 countries; hyperion_8.1 95,761 / hyperion_8 38,168; 131,219 have `obstacle.offline`).
⇒ **The admissible ceiling under this scope is 133,929 clips, not 306,151.** The 306k row of the cost table is therefore reported as a ceiling that needs a PI ruling
(admit NVIDIA `val` clips, which our own corpus already did), not as a plan.

## 3. Sampling rule (seeded, stratified-permutation, nested)
* Seed `20261115` (numpy `default_rng`). Strata = country x platform_class. Inside each stratum draw a random permutation; each clip gets key = (rank + u)/stratum_size;
  the global order is the ascending key. Every prefix is proportionally stratified over (country, platform_class), and the prefixes are NESTED: pilot 200 inside 10k inside 45k inside the pool.
* Not stratified by hour, month or road type: those are reported for the pilot, not controlled (illumination is ESTIMATED in the census).
* Pilot = the first **200** clips of that order. Controls = **3** clips drawn with seed `20261116` from the 4,572 train clips ALREADY in our corpus; they are never part of the new corpus.
* The order file (`order_all_eligible.parquet`) and the lists are written by `ds_select.py` (sha256 recorded in the pilot report). No clip is hand-picked or dropped after drawing;
  a clip that fails to build stays in the ledger as FAIL with its reason.

## 4. New corpus id
`physicalai-datascale-v1-pilot200-s20261115` for the pilot; any later size is `physicalai-datascale-v1-n<N>-s20261115` and is a prefix of the same order.
The REF-C parity corpus `physicalai-train-e438721ae894` and the v8f-mini corpus are NOT touched, not re-selected, not written into.

## 5. Output format
Unchanged `cache_full` layout of `cache_pod.py` (pipeline `ra1a-pooled-v1`): `<out>/train/clips/<clip>.pt` with keys pool4x10 [T,40,1024], pool8x20 [T,160,1024] (fp8 e4m3fn),
mean (fp16), v, a, steer, yaw_rate, pose_grid, kin3, p10, meta; `manifest.jsonl`, `meta.json`, `runs.jsonl`, `pod_provenance.jsonl` (v2ep sha256 per clip). 256x640 cylindrical frames,
DINOv3 ViT-L/16 bf16, the builder code imported unedited (`pull_encode_pooled.encode_clip`). The 256-route is the existing one; the cache is the v2ep -> encoder step only.
Frames come from the production builder `pod_build_b1_epcache.build_one` (unedited) on a staging root laid out like the corpus one, fed from NVIDIA chunks by range reads.

## 6. Labels for new clips, per family (what can and cannot be built)
| family | new clips | reason |
|---|---|---|
| ego kinematics in the cache (v, a, steer, yaw_rate, pose_grid, kin3, p10) | YES | from NVIDIA `labels/egomotion`; control clips test that it equals our corpus values |
| obstacle join (`build_obstacle_join.py`, P4/P8) | YES for clips with `obstacle.offline` (97.44 % of the pool: 131,219 / 133,929) | NVIDIA `labels/obstacle.offline`; a pod-side join on the v2ep; pilot attempts it |
| geometry tactical/strategic labels (manoeuvre sequence, turn table, nav, route checkpoint; `s2_geom_emit_v7` + `build_v9_labels` geometry side) | PARTLY, UNVERIFIED | the egomotion part needs no VLM; the emitter imports Alpamayo records, so the pilot only PROBES whether a record can be emitted without them and reports it; not a gate |
| VLM-CoT tactical goals (traffic-light colour, stop sign, pedestrian, `vlm-cot` provenance), `meta_action` axes, chain_of_causation, VQA, grounding boxes | NO | the CoT exists only for the 4,729 clips of `Sayood/tanitad-alpamayo2-augmentation` (our corpus clips); NVIDIA publishes event CoC for 1,740 clips (`reasoning/ood_reasoning.parquet`, of which 1,364 are in the pool, MEASURED) and nothing for the rest. Generating CoT for new clips needs a VLM pass (compute and a PI decision), not built here |
| 2D boxes (agent_cuboid_gt path), BEV raster | via obstacle join only | same coverage as the obstacle join |
| map / lane / route / traffic-light state features | NO | not in the published corpus (layer: published corpus). NOTE: the repo `navigation/navigation.parquet` (16.3 M rows, `clip_id, timestamp, nav_text`) exists in the NVIDIA repo; whether it covers pool clips is not measured here and is flagged for the Master Mind (it may bear on the "no route signal" statements) |
Consequence: new clips are encode + kinematics + (obstacle) data. Hierarchy-label supervision on them would be IGNORE unless a VLM pass is approved.

## 7. Pilot size and gates
200 clips + 3 controls. Gates: (a) the control v2ep sha256 equals the ledger sha256 and its mp4 sha equals the ledger mp4 sha; (b) control cache tokens bitwise equal to
`/root/data/cache_pod/train/clips/<clip>.pt` (else the measured rel-L2, stated); (c) every pilot clip logged in the ledger as OK or FAIL with a reason.
Final marker in `/root/datascale/pilot.log`: `ZZ DATASCALE-PILOT DONE ZZ` or `ZZ DATASCALE-PILOT FAIL ZZ <why>`.
No training, no upload, no box 2, no Thor, no dev-box GPU. Pod work only under `/root/datascale/`.
