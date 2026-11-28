# refcv7 map head at 10 cm: why the map was 0.5 m, what 0.5 m cost, and what 10 cm needs

**Question (PI, 2026-09-26 ~21:10 Berlin, verbatim):**
> "Why did we choose 0.5 m cells? The original sam3 maps were very good and fine. Can we increase the resolution to 10 cm?"

**Earlier the same evening:**
> "the drivable area are little bit ok, but our model is seldomly detecting other sematic classes, no roadmarkings, no roadmarking infrstructure in intersections etc... We need to analyze it"

**Status:**
- §1–§3 are MEASURED, and are the census that answers the question.
- §4 registers the design change, NEW-2 in `Project Steering/SPEC_REFCV7.md` §6. It was registered BEFORE any NEW-2 code or number existed.

## 1. Why 0.5 m — the honest answer

The grid was **inherited, never chosen for road markings.** The chain has three links:

1. **The programme's BEV grid is `bev_raster.GRID_DEFAULT`: 120 × 64 cells at 0.5 m** (`stack/tanitad/data/bev_raster.py:111`). It was built for agent occupancy and lead-vehicle range. A car is about 4.5 × 1.8 m, so 0.5 m is adequate there.
2. **The SAM3 ground-truth exporter wrote its Cartesian training grid on that same 120 × 64 @ 0.5 m** (`cart_frac`), so the map head could share the BEV lift.
   - It ALSO wrote the map at 10 cm (`fine_codes`, 600 × 320).
   - The reader never opens it. Its own docstring says so: `semantic_map_gt.py:21`, *"(polar_frac, polar48_frac, fine_codes are not read here)"*.
3. **The refcv6 design put the prediction on exactly the label cells** (`Project Steering/REFCV6_DESIGN_GROUNDED.md:76`): *"The label and the prediction live on the same cells — no resampling between them."*
   - Its lift samples the **stride-16** trunk map (`:24`), whose native lateral footprint is already coarser than 0.5 m beyond ~15 m (see §3).

⛔ **Nobody measured what 0.5 m does to a 15–40 cm road marking before refcv6 trained on it.** This is a design miss, and it is the Master Mind's.

## 2. The census — MEASURED, CPU, read-only

**Instrument:** `code/map_res_census.py`.
- Output: `raw/map_res_census.json` (md5 `f06ccbd8…`). The script md5 is `e6f3f7b0…`.
- Sample: all **137** SAM3 GT files of the eval kit (`D:/refcv6_eval_kit/data/sam3_gt_eval_thor137/`), every 10th frame: **2,867 frames**.
- That is **500,909,554 seen 10 cm cells** and **20,036,498 seen 0.5 m cells**, under the trainer's rule: not-seen fraction < 0.5.

**Control (must read exactly zero):** `cart_frac` is recomputed from `fine_codes` as the 5 × 5 block fraction on each file's first sampled frame.
- **Bad cells: 0.**
- ⇒ The 0.5 m training target is EXACTLY an average of the 10 cm map; the file header confirms `source.world_map_res_m = 0.1`.
- ⇒ **The 10 cm map already exists for every clip. No SAM3 rebuild is needed.**

| class | share of seen area at 10 cm | 10 cm area kept at 0.5 m under the eval rule (cell fraction ≥ 0.5) | kept under argmax | median cell fraction where present |
|---|---:|---:|---:|---:|
| seen, no map class | 26.83 % | 96.8 % | 96.9 % | 1.00 |
| drivable | 33.75 % | 96.9 % | 97.4 % | 1.00 |
| sidewalk / verge | 36.56 % | 97.8 % | 98.1 % | 1.00 |
| **lane / road line** | **1.64 %** | **61.8 %** | 63.6 % | **0.40** |
| crosswalk | 0.52 % | 77.7 % | 78.3 % | 0.48 |
| arrow / text | 0.066 % | 64.7 % | 65.4 % | 0.32 |
| hatched area | 0.062 % | 78.7 % | 79.5 % | 0.48 |
| **non-drivable edge** | 0.58 % | **0.9 %** | **4.5 %** | **0.20** |

**Reading 1.** The 0.5 m grid **erases the non-drivable edge class completely**: 0.9 % of its area survives. It also erases **~38 % of lane lines** and **~35 % of arrows and text**.

**Reading 2.** The grid does **NOT** by itself explain why the model almost never predicts lane lines or crosswalks. At 0.5 m, **62–78 % of their area is still in the targets.**

## 3. What the model does at 0.5 m, and what the camera can resolve

**The model, MEASURED.** Source: the map-video agent's `per_frame.jsonl`.
- Setup: step 35,000; 513 windows over 3 clips; argmax IoU at 0.5 m.
- Package: `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-refcv6-map-video/raw/`.
- Scope: a perception diagnostic, open loop; one clip is one episode, so no interval is quoted.

| class | windows where GT or prediction has the class | mean IoU | windows with IoU > 0 |
|---|---:|---:|---:|
| drivable | 513 | 0.672 | 513 |
| seen, no map class | 513 | 0.530 | 513 |
| sidewalk / verge | 513 | 0.337 | 444 |
| **lane / road line** | 441 | **0.009** | 139 |
| **crosswalk** | 250 | **0.001** | 3 |
| arrow / text | 179 | 0.000 | 0 |
| non-drivable edge | 227 | 0.000 | 0 |

**Why lane lines vanish although their targets are present.** These are HYPOTHESES, ranked by what the arithmetic allows:

1. **The soft target out-votes them.**
   - The median lane-line cell is 0.40 lane / 0.60 drivable.
   - So a PERFECTLY calibrated model's argmax is "drivable" in most lane cells, and its lane probability rarely clears the eval's ≥ 0.5.
   - Add a quarter-metre of lateral uncertainty and the probability spreads across two cells, and never wins.
2. **Class imbalance.** Unweighted soft cross-entropy runs over a distribution where drivable, sidewalk and no-class are **97 %** of seen area; lane line is 1.6 %.
3. **The features are coarse.**
   - At 416 × 1024 the cylindrical camera has f = 488.9 px, so one image column spans 2.05 mrad: **4 cm at 20 m, 8 cm at 40 m, 12 cm at 60 m**. So 10 cm IS resolvable laterally to ~50 m.
   - But the lift samples the **stride-16** map: 16 columns per feature, **65 cm at 20 m, 1.3 m at 40 m**. Stride 8 halves that; stride 4 quarters it.
   - Longitudinally, one image row spans ≈ x² / (h · f). With h ≈ 1.5 m (ESTIMATED camera height) that is 0.14 m at 10 m, 0.55 m at 20 m and 2.2 m at 40 m.
   - Lane lines and continental crosswalk bars run ALONG x, so the lateral figure is the one that binds.

**The cheapest discriminating experiment.** Per-class **average precision** of the model's softmax (a ranking) against its argmax IoU (a decision), on the same windows at step 38,000.
- If lane-line AP is far above its 1.4 % base rate, the model SEES the lines and the target/decision rule hides them.
- If AP is near the base rate, the model never learned them, and the features plus the loss are the lever.
- **Queued with the map-video agent (GPU, after the agent-box video).** Result → §5 of this file.

## 4. What 10 cm needs (registered as NEW-2 in SPEC_REFCV7 §6)

1. **Target:** `fine_codes` @ 0.1 m (600 × 320, codes 0–7, 255 = not seen) through a reader with the same identity and time guards as `cart_frac`. It is a hard-label CE: a 10 cm cell is single-class, so the soft-target out-voting of §3.1 disappears by construction.
2. **Features:** a second, map-only lift at **0.25 m** (240 × 128) sampling the **stride-8** trunk map. It feeds a small BEV decoder, upsampled to **600 × 320** logits.
   - Every existing consumer is unchanged: box3d, the planner's BEV cross-attention, and the 30 × 16 BEV tokens.
   - The 0.5 m head and its loss are kept as in refcv6, so the planner-facing BEV features are shaped as before.
3. **Loss:** class weights by median-frequency balancing, computed ONCE from the **TRAIN** split's 10 cm GT, frozen in the config and recorded. Nothing is tuned on eval.
4. **Metrics:**
   - per-class IoU at 10 cm, banded by range: 0–20 / 20–40 / 40–60 m;
   - a 0.2 m-tolerance precision / recall / F1 for the thin classes (lane line, edge, crosswalk);
   - an episode-cluster bootstrap over the eval clips;
   - refcv6@38k's 0.5 m prediction, nearest-upsampled to 10 cm, as the baseline on the same frames.
5. **Guards:** `fmap_s8` must reach the branch; this is the F3 whitelist failure class. G-DVB checks the head is built at 600 × 320; G-LIVE checks the 10 cm loss is finite and the stride-8 path gets a gradient.
6. **Cost:** measured in the Thor G-LIVE smoke with `torch.cuda.max_memory_allocated()` and s/step. **If it adds more than 25 % to refcv6's 6.4 s/step, the PI decides before launch.**

## 5. AP probe result

⏳ Pending — the map-video agent, step 38,000.
