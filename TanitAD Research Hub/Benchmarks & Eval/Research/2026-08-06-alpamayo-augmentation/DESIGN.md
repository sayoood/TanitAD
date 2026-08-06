# Alpamayo-Super augmentation dataset for PhysicalAI-AV — design (PI task, 2026-08-06)

**Goal (Sayed, verbatim intent):** select **100 hours** of well-distributed PhysicalAI-AV data
(highway/urban/intersections, day/night, structured/unstructured, …), run it through Alpamayo 2
Super inference, log its **complete outputs — no exceptions**, link every output to the index of
the corresponding sensor + ego data (unique link, NO camera images in the dataset), publish as an
**HF public gated dataset**. VQA: a rich bank of **≥500 questions**, assigned randomly per clip.

## Measured facts the plan stands on

- **Corpus size: 306,152 clips ≈ 1,701 h** (MEASURED, `data_collection.parquet` row count,
  2026-08-06). 100 h = **18,000 clips** ⇒ a **5.9 % stratified sample** — feasible.
- Stratification attributes available NOW: `country, month, hour_of_day, platform_class,
  radar_config` (data_collection) + 36-feature presence flags (feature_presence) +
  `event_cluster` taxonomy for the 1,740 reasoning clips. ⚠️ **No road-type column found yet**
  (highway/urban/intersection): WORK ITEM — inventory the full HF `metadata/` tree; fallback is
  deriving road class from egomotion statistics (speed profile: highway ≥ 20 m/s sustained;
  intersection = stop + heading change; urban = low-speed varied) — labels from ego are
  admissible (labels-may-use-ego rule).
- **Throughput (MEASURED, our A40 4-bit pipeline):** meta-action = **40 s/clip** (1,561 s / 39).
  Trajectory / VQA / auto-label / grounding rates NOT yet measured — pilot measures them.
  ESTIMATE ~200 s/clip for the full 5-task battery ⇒ 18,000 clips ≈ **42 A40-days** (single
  GPU). ⇒ the 100 h target needs either an H100 rental (~5-8×), a pod fleet, or PI-approved
  schedule. **Pilot first: the 290 official-val clips** (~16 h single-A40) measures every rate
  and validates the schema end-to-end.

## Output record schema (one row per clip × task; parquet, no images)

```
clip_id (UUID, = PhysicalAI-AV index)     # THE unique link to sensor+ego data
task ∈ {trajectory, meta_action, vqa, auto_label, grounding}
t0_frame_idx, camera_ids, model_rev (HF commit), quant (4bit-a40|bf16-h100)
raw_output          # verbatim: full decoded text incl. CoC, special tokens
parsed              # task-specific: 64×SE(3) waypoints / 3-axis meta-action /
                    # VQA answer / auto-label JSON fields / grounding coords
vqa_qid             # which of the 506 bank questions was asked (vqa rows)
latency_s, sample_idx (trajectory supports multi-sample)
```

- Trajectories stored as float arrays (64×[x,y,z,R9]) — **complete**, per "no exceptions".
- The dataset ships: records parquet + `vqa_bank_500.json` (506 Qs, 21 categories) +
  selection manifest (clip_id + strata cell) + README with the join recipe
  (`clip_id` → PhysicalAI-AV shards via `clip2chunk`-style mapping).
- HF: `Sayood/tanitad-alpamayo2-augmentation` — public, **gated="auto"**.

## Stage plan

1. metadata inventory (road-type source) → 2. stratified 18,000-clip selection (equal-weight
cells over road-class × day/night × country, proportional fill, seed 0, manifest committed) →
3. pilot 290 val clips, all 5 tasks, rates + schema validation → 4. full run (provisioning per
pilot rates — PI decision) → 5. QC (parse-rate, per-task coverage, leak check: no image bytes)
→ 6. HF push + card.

*Everything to date banked under this folder; VQA bank: `tools/vqa_bank_500.json`.*
