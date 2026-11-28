# PREREG — the SAM3 map-extent coverage census (SPEC_REFCV7 §11.2, Amendment A6)

**Registered:** 2026-09-27 00:19 Berlin (file mtime), by the map-signal audit, **BEFORE any world map was read.**
- Up to this point only these were read:
  - the exporter SOURCE: `/home/nvidia/sam3map/eval/sam3map_export_gt.py` and `…/sam3map_prod.py` (`export_v2ep`);
  - file LISTINGS: 4,719 `<sha12>.worldmap.npz`, 4.4 GB, in `/home/nvidia/sam3map/corpus/out/`.
- No world-map byte and no coverage number existed.
- Changing any literal below after the run is a goalpost move. It needs a dated amendment; the first run stays on the record.

**The rule being measured (SPEC_REFCV7 §11.2, fixed by the Master Mind before this file):**
- *x_max (ahead) and y_half (to each side) are the largest 10 m steps at which at least 50 % of TRAIN frames have seen ground truth in that ring.*
- The extent is never smaller than 60 m × ±16 m.
- It must stay within the Thor memory/time budget. If the budget binds, the PI chooses.

## 1. Clips (TRAIN only)

- The 4,369 clip ids of `/home/nvidia/data/refcv6_train_clips.txt`: refcv6's train split on the parity corpus. No eval clip is read.
- Each clip is joined by `sha12 = sha256(clip_id)[:12]` to:
  - its world map `/home/nvidia/sam3map/corpus/out/<sha12>.worldmap.npz`;
  - its `/2` GT `/home/nvidia/sam3map/corpus/out/<sha12>.sam3mapgt.npz`.
- A clip missing either file is EXCLUDED and COUNTED. If more than 1 % of the 4,369 are missing, the census is **INCONCLUSIVE** and no extent is reported.

## 2. Pose source: how the rig is placed on the world map

- **The pose** is the `T_world_rig` array stored in the clip's own `/2` GT file. It is `[T,4,4]`, one row per raw v2ep frame.
  - It is the exporter's own pose: egomotion interpolated at that frame's exposure time `t_img` (`sam3map_prod.py` `export_v2ep`, lines 152–159).
- **The transform is the exporter's own**, character for character:
  - a rig point `p = (x, y)` maps to the world point `p @ R.T + t`, with `R = T[:2,:2]` and `t = T[:2,3]`;
  - the world-map lookup is `i = floor((x_w − x0)/res)`, `j = floor((y_w − y0)/res)`, returning 255 outside the map (`sam3map_export_gt.lookup`, lines 30–36; `wm["cls"]`, `wm["origin"]`, `wm["res"]`).
- The rig frame is +x forward and +y LEFT, with the origin at the rear axle on the road plane.

## 3. Frames

- **Census frames:** raw v2ep frames `r = 9 + 10k` (k = 0, 1, …) with `r ≤ T − 22`. These are the label frames of training windows (`t = r − 9`, over the trainer's index `t ∈ [0, T − 30)`: `_contract.py:120-121` with n_stack 3, window 8, max_horizon 20), one per second.
- **Pooling:** coverage is POOLED over all census frames of all clips (frame-weighted). The clip-weighted mean is reported beside it.

## 4. Cells, rings and "seen"

**Sampling grid.**
- The census reads the 10 cm grid at its cell centres, subsampled 1 in 5 per axis: centres at `x = (5a + 2.5)·0.1 m`, `y = −Y + (5b + 2.5)·0.1 m`.
- That is one sample per 0.5 m × 0.5 m block and 400 samples per 10 m × 10 m block. It is an unbiased estimate of a ring's not-255 fraction.
- **Census rectangle:** x ∈ [0, 200) m ahead and y ∈ [−60, +60) m. That is 400 × 240 samples per frame, which exceeds the brief's minimum of 150 m and ±50 m.

**The rings the RULE reads** (10 m steps):
- **x-ring k** (k = 0 … 19) is `x ∈ [10k, 10k + 10)` and `|y| < 16 m`, i.e. today's map width.
- **y-ring m** (m = 0 … 5) is `10m ≤ |y| < 10(m + 1)` and `x ∈ [0, 60)`, i.e. today's map depth. The left and right strips are pooled.

**Definition of "seen".**
- A frame **has seen GT in a ring** iff **≥ 20 %** of the ring's sampled cells have world-map code ≠ 255.
- Coverage(ring) = (census frames that have seen GT in the ring) / (all census frames).

**Also reported, NOT read by the rule:**
- the full block table `B(k, m) = {x ∈ [10k, 10k+10), 10m ≤ |y| < 10(m+1)}` over the whole rectangle;
- per-ring sensitivity at 5 % and 50 % seen;
- x-rings within |y| < 10 m;
- y-rings within x ∈ [0, 30) m.

## 5. The selection (mechanical)

- **x_max** = 10·(K + 1), where K is the largest k such that coverage(x-ring j) ≥ 0.50 for EVERY j ≤ k (a contiguous extent).
- **y_half** = 10·(M + 1), where M is the largest m such that coverage(y-ring j) ≥ 0.50 for every j ≤ m.
- **Floor:** x_max ≥ 60 m and y_half ≥ 16 m.
- **Census edge:** if the last census ring still passes, the result is reported as "≥ 200 m" (or "≥ 60 m" to the side). The census is then extended by amendment; the rule is never read past its census.

## 6. Validity control (gates the census; run BEFORE any coverage number is read)

- **C1, byte identity with `/2`.**
  - Take 24 frames: 2 per clip (the first and the middle census frame) from the first 12 census clips by sha12.
  - Recompute the full `[600, 320]` `fine_codes` with the exporter's expression, on the stored `T_world_rig`, from the world map.
  - Compare byte-for-byte with the stored `/2` `fine_codes`.
  - **Required: 24 / 24 identical (0 differing bytes).** Otherwise the census is **INVALID** and no extent is reported.
- **C2, the sampler is the exporter.** On those 24 frames, the census's own samples inside the old window (x < 60, |y| < 16) must equal the stored `fine_codes[2::5, 2::5]` byte-for-byte.
- **Interpreter.** Production ran under `/home/nvidia/venvs/tanitad-edge/bin/python` (`corpus_supervisor.sh:9`), so the census uses that interpreter, read-only, with no installs.

## 7. Where and how it runs

- **Machine:** Thor, CPU only, `nice 19`, at most 6 worker processes, in a fresh `/home/nvidia/msa_<HHMM>/`.
- **Shipping:** the script is shipped by scp with md5 verification.
- **Access:** world maps and GT are read-only. Nothing in `/home/nvidia/sam3map` or `/home/nvidia/data` is written. No other process is touched.
- **Outputs pulled back:**
  - `census_table.json`: the rule's rings, the block table, sensitivity, n frames and n clips per ring, and C1/C2;
  - the selected extent;
  - the log.
- **Afterwards, at the selected extent:**
  - a timed `/3` re-export dry run on 20 clips (the first 20 census clips by sha12), written to the scratch dir, never to the corpus, with its inside-old-window byte identity to `/2`;
  - an ANALYTIC memory/time estimate of the 10 cm decoder at b16 for 60 × 32 m and for the selected extent, from the NEW-2 builder's shapes (`…/2026-09-26-refcv7-map-hires/BUILD.md`, `raw/cost_analytic_cpu.json`).
