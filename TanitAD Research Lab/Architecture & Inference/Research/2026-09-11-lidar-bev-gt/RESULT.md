# LiDAR-supervised BEV: the ground truth exists, it is READABLE without a 32 GB pull, and our own 3D boxes land on it at 10.3× the no-information floor

**Date** 2026-09-11 (Europe/Berlin) · **Agent** Architecture & Inference FlyWheel ·
**Branch** `agent/arch-inf-20260803` · **Tree** `D:\Projects\TanitAD`
**Compute** dev-box **CPU only**. ⛔ **Zero GPU seconds**; `nvidia-smi` checked before starting
(RTX 4060, 2,095 MiB in use by the desktop, 13 %) and no torch job was run on it — another agent is
starting Qwen-Drive on Thor. `OMP_NUM_THREADS=6` throughout.
**Evidence class** MEASURED (ours) unless a row says otherwise. **Tier** N/A — this is a LABEL
artifact, not an eval result, so a T0/T1 stamp would be a category error.

⛔⛔ **BINDING, AND IT IS WHY THIS DIRECTION IS DOCTRINALLY CLEAN: labels may use ego, agents,
maps, future poses and LiDAR; INFERENCE IS VISION-ONLY** (PI, 2026-08-03). A LiDAR-built BEV target
training a camera-only head satisfies that rule **by construction rather than by argument** — there
is no admissibility question to litigate, because the LiDAR never appears at inference. Every
module written here is a target builder; **none is importable from an inference path**, and the
constraint is written INTO the artifact's own metadata so a later consumer cannot lose it.

---

## 1. The four questions, answered first

| question | answer |
|---|---|
| **Does the corpus really ship LiDAR?** | **YES.** `lidar_top_360fov`, verified by me from the dataset's own `features.csv` (36 rows: 7 camera + 6 calibration + 3 label + **1 lidar** + 19 radar). Pulled and decoded on **4 clips**, all present, all readable. |
| **What format, rate, extent?** | One **parquet per clip** inside a per-chunk zip; **199 spins on every clip probed**, `draco_encoded_pointcloud` (magic `DRACO\x02\x03\x00`), **10.0006 Hz** (inter-spin median **99,994 µs on all 4**), **298,559–333,688 points/spin**, xyz float32 + uint8 intensity + **per-point µs timestamp** + (laser row 0–127, azimuth column 1–3599). Extent ±208 m, z −5.29…+27.36 m in the sensor frame. |
| **Is it time-aligned with `camera_front_wide_120fov`?** | **YES — same clock, same units (µs).** Median \|camera frame − nearest spin midpoint\| = **30.2 ms**, p95 **36.5 ms**, max **169.7 ms** (the max is the clip ends, where the camera runs past the LiDAR and the frames must be **NO_LABEL**, never extrapolated). |
| **Do our own 3D boxes land on the returns?** | **YES — real beats the no-information floor by 10.30× and 4.99× on two independent clips**, and a 90° rotation of the same boxes reads **below** that floor on both (0.85×, 0.68×). ⚠️ The *mirror* control collapses on one clip (0.75×) but **not** the other (1.52×) — see §5.2, it is a scene-density effect and it is why two clips were run. |

⭐ **The unblocking engineering fact:** the LiDAR chunk zip is **31.9–33.5 GB for 94–99 clips**, and
the programme does not have to pull it. A zip's central directory is at the end of the file, so
`HfFileSystem` + `zipfile` streams **one member** — **240–403 MB per clip, mean 341.7 MB, i.e.
~1.07 % of the chunk**, with the central directory itself costing **2.4–12.7 s**. This is the
difference between "LiDAR is a 101.7 TB wall" and "LiDAR is a per-clip fetch", and it is why
nothing here needed a pod.

---

## 2. ⛔ Priced from the CONSUMER'S loader, and the consumer is named

**The rule this obeys:** *an artifact's cost is the cost of the file its CONSUMER opens, not the one
you found on disk* — the rule that exists because sizing an episode cache from the wrong file
produced a **1.38 TB "capacity wall"** that drove a format change, a geometry downgrade and a
sharding design, when the real figure was **161 GB**.

**The consumer here is:**

```
huggingface_hub.HfFileSystem.open(<chunk>.zip)          # ranged HTTP reads
  -> zipfile.ZipFile(fh).read("<clip>.lidar_top_360fov.parquet")
    -> pyarrow.parquet.ParquetFile(...).read_row_group(i, ["draco_encoded_pointcloud"])
      -> DracoPy.decode(blob)                            # -> [N,3] float32 + 3 attributes
```

named in `code/lidar_bev.py` and exercised in `code/p1_lidar_probe.py` / `p2_build_bev_gt.py`.

| quantity | MEASURED | scope |
|---|---:|---|
| chunk zip | **31,946,109,179 – 33,491,677,087 B** (94–99 members) | 4 chunks read live; `pai_sizes_and_revs.json` gives 32,340,273,976 for `chunk_0000` |
| ⭐ **what the consumer actually reads per clip** | **240,461,561 / 358,362,639 / 365,428,765 / 402,565,608 B — mean 341,704,643** | 4 clips, `raw/lidar_probe.json` |
| central-directory read (the whole 32 GB zip's index) | **2.36 – 12.73 s** | |
| single-stream member read | **1.59 – 11.57 MB/s** | ⚠️ 4 samples, 7.3× spread — see §8 |
| spins per clip | **199 on all four** (one **row group each** — the consumer never materialises the clip) | |
| mean points/spin | **298,559 – 333,688**; 63,943,046 points over one full clip | |
| Draco decode | **0.108 – 0.182 s/spin** | |
| **full-clip GT build, 199 spins** | **37.2 s** — decode **26.8 s (72 %)**, rasterise **9.7 s (26 %)**, parquet read **0.6 s (2 %)** | dev-box CPU, `OMP_NUM_THREADS=6` |
| **output GT** | **207,750 B compressed** for 4 frames × 16 channels ⇒ a lean `occ`+`observed` target is **~1–2 KB/frame** | |

⚠️ **`nonzero_frac` reads exactly 1.0 on all four clips** — the content assertion, not a file-size
check: a Draco decode that failed into a pre-allocated array would read 0 here and pass `ls`.

⛔ **DO NOT `ParquetFile.read()` THIS FILE.** A full read + `to_pydict()` held **>2 GB** and made
**no progress for 8 minutes at 5.9 s of CPU** — blocked, not computing, which reads exactly like a
hang and cost a debugging round here. ⚠️ The cause is **not** `io.BytesIO`: re-timed from the same
bytes, `read_table` from a path is **0.32 s** and from `BytesIO` is **0.01 s**. The row groups are
one spin each; read them one at a time. Pinned in the docstring of `p1_lidar_probe.py`.

---

## 3. The geometry, and the cross-check that makes it trustworthy

⛔ **The points are in the LIDAR SENSOR frame, not the rig frame**, and nothing in the file says so.

| fact | value |
|---|---|
| near-range (3–20 m) ground peak in the **raw decoded** z | **−1.925 m** |
| `sensor_extrinsics` `lidar_top_360fov` in the rig frame | x **1.0187**, y **−0.0011**, z **1.8897**, quaternion **(−0.00057, −0.00167, +0.00107, 0.99999)** ≈ identity |
| ⭐ **sum** | **−1.925 + 1.890 = −0.035 m** |

⭐ **That −0.035 m independently reproduces `D-V5A-GROUND1`** (rig z = 0 IS the road plane, to
0.05–0.13 m over 87,481 cuboids, with `protruding_object` at +1.676 m as the control that must
*not* touch the road). **Two facts derived by completely different routes agreeing to 3.5 cm is
what makes the frame determination a measurement rather than a guess** — and the per-spin
re-derivation over the sampled frames reads **−0.025 to +0.025 m**, i.e. stable.

**A second independent confirmation, from a different artifact:** the Draco attribute (row 0–127,
column 1–3599) matches `lidar_intrinsics.offline`'s published `row-offset-spinning` **128 rows ×
3600 columns** model exactly. Neither was consulted to build the other.

⚠️ **The quaternion→matrix conversion is NOT re-derived here.** `lidar_bev.quat_to_R` is
character-identical to `physicalai.FrontWideExtrinsics.rotation_cam_to_vehicle`, the programme's own
implementation — this is the `R-2026-09-08-wpa-mirror` family (a convention nobody asserts), and the
mitigation is to use the module that already agrees with the rest of the stack.

---

## 4. What the ground truth IS — units, extent, frame, dt, all written INTO the artifact

⭐ *An artifact is opened in isolation far more often than its run record* — the rule that exists
because a `controls` column read as curvature instead of lateral acceleration produced a **396 g**
table that was arithmetically perfect and completely wrong. So `lidar_bev.artifact_meta()` writes
the following into the `.npz` itself, as `meta_json`:

* **frame** `rig` — **+x forward, +y LEFT, +z UP**, origin rear axle on the road plane. Identical
  to `bev_raster` and `rig_projection`, so the LiDAR target and the `obstacle.offline` raster share
  one address space with **no transform between them**.
* **Cartesian grid** — x ∈ [0, 60] m, y ∈ [−16, +16] m, **cell 0.5 m ⇒ [120, 64]**: byte-for-byte
  the P8 extent `bev_raster` already rasterises.
* **Polar grid** — **[24 range × 20 azimuth]**, 2.5 m × 6.0°, 60 m, HFOV 120.000°, mirroring
  `bev_aux.PolarBEVSpec` so it is a drop-in replacement for the `obstacle.offline` target.
  ⛔ **column 0 = +60° (LEFT)**, the sense on which `bev_aux.azimuth_column` and
  `psg_targets.azimuth_column` — two independently authored modules — agree.
* **units** per channel (`occ` binary, `z_max` m above the rig road plane, `intensity_max` uint8
  0–255, `observed` binary, timestamps **µs on the clip clock**).
* **z-band** obstacle [0.30, 3.00] m **plus a per-cell vertical-spread test ≥ 0.30 m**, and the
  binding note that LiDAR is a **LABEL ONLY**.

### 4.1 ⛔ A flat z threshold is wrong at range, and it fails in the DANGEROUS direction

A pure `z ∈ [0.30, 3.00]` rule marks **24.9–35.0 %** of the 60 × 32 m grid occupied and drives the
marginal occupancy of an observed cell to **0.544** — a road with 0.5 % camber rises 0.30 m over
60 m, so **the road itself crosses the threshold**. An over-occupied target manufactures a high
base rate and then *every* arm looks registered.

Adding the per-cell vertical-spread test (a ground cell's returns are coplanar; a wall's are not)
moves occupancy to **2.43–6.25 %** and the marginal to **0.028–0.081**. ⭐ **The two rules are
BOTH computed and BOTH written into the artifact**, so the choice is auditable rather than baked
in — and the discrimination improves with it: on one frame the z-band rule gives real 1.000 vs
mirror 0.174 (lift 6.66×), the spread rule gives real 0.944 vs **mirror 0.000** (lift 15.2×).

### 4.2 ⭐ The third state is MEASURED here, not derived — and this is the real upgrade over WP-D

WP-D had to **derive** occlusion from `obstacle.offline` footprints and could only ever produce a
**lower bound**, for three stated reasons: the join drops `z`; the shadow is quantised to a range
bin; and — the binding one — **only labelled agents occlude**, so buildings, walls, parked rows and
vegetation cast no shadow at all. LiDAR replaces the derivation with the sensor's own statement:
the `observed` channel is built from the first obstacle return per bearing, and **it shadows
everything opaque**. MEASURED observed fraction **77.3–90.6 %** of the grid on the sampled frames.

⚠️ **v1 limit, stated rather than hidden:** where a bearing has *no* obstacle return at all, this
v1 marks the column observed-free out to 60 m. That is right for a road surface with ground returns
and wrong for a bearing pointing at empty sky; the `n_pts` channel is written so a consumer can
tighten it without a rebuild.

### 4.3 Content assertions, because a zero bank passes `ls`

⛔ *A decode that raises into a pre-allocated array leaves a full-size file of zeros and can still
exit 0* — and because that bank was the FLOOR arm, every trunk would have beaten it. Every builder
path here **refuses to write** a frame whose cloud decoded empty or all-zero, and reports content:
occupancy mean **0.06389** over the full clip, range **0.02031–0.14180**, and **0 of 199 spins with
zero occupancy**. Per-spin point counts and the occupancy mean are in the artifact.

---

## 5. ⭐ Does it register? — every spin of TWO clips, five arms

`code/p5_registration_sweep.py` → `raw/registration_sweep.json`, `raw/registration_sweep_clip2.json`.
**Every** LiDAR spin of each clip (199/199, none skipped; join \|dt\| median **32.5 ms**, max
**50.3 ms**; 7–24 agents per frame, median 12). Hit rates are **POOLED over cells**, never a mean of
per-frame ratios — a mean over frames holding 1–2 box cells is dominated by the sparsest frames.

| arm | clip `0dfe6321…` hit / cells → rate (**lift**) | clip `c64f0103…` hit / cells → rate (**lift**) |
|---|---|---|
| ⭐ **real** | 1,917 / 2,441 = **0.7853** (**10.30×**) | 14,334 / 22,171 = **0.6465** (**4.99×**) |
| ⛔ `rot90` ((x,y) → (−y,x)) | 458 / 7,108 = 0.0644 (**0.85×**) | 527 / 6,010 = 0.0877 (**0.68×**) |
| ⛔ `mirror_y` (y → −y) | 224 / 3,932 = 0.0570 (**0.75×**) | 4,203 / 21,365 = 0.1967 (**1.52×**) |
| `shift_y5` (5 m left) | 445 / 3,810 = 0.1168 (1.53×) | 4,090 / 19,397 = 0.2109 (1.63×) |
| `shift_x10` (10 m forward) | 778 / 4,454 = 0.1747 (2.29×) | 2,585 / 14,920 = 0.1733 (1.34×) |
| **marginal (no-information)** | **0.07625** over 1,284,886 observed cells | **0.12955** over 934,466 observed cells |

**What replicates on both clips:**
* **real ≫ marginal** — 10.30× and 4.99×.
* **real ≫ every wrong geometry** — real/mirror **13.8×** and **3.3×**; real/rot90 **12.2×** and **7.4×**.
* **`rot90` reads BELOW the no-information floor** — 0.85× and 0.68×.

### 5.2 ⛔ What does NOT replicate, and why the second clip was worth running

**On clip 1 `mirror_y` reads 0.75× — below chance. On clip 2 it reads 1.52× — above it.** Had I
stopped at one clip I would have published *"mirroring collapses below the floor"* as the headline
control, and it is **not a property of the pipeline**. The mechanism is scene density and symmetry:
clip 2 has **9× more box cells in the grid** (22,171 vs 2,441) and a **1.7× higher marginal**
(0.1296 vs 0.0763) — a two-way street with traffic and parked rows on *both* sides, where a
mirrored box lands on the mirror-image traffic.

⇒ **The robust statistic is the RATIO real/wrong, not the wrong arm's absolute lift**, and it is
3.3–13.8× across the two clips. ⛔ The absolute lift of a *wrong* arm is a statement about the
scene, not about the registration.

⚠️ **The translation controls never collapse** (1.34–2.29×), which is expected and worth saying: a
5–10 m shift lands a box on the *neighbouring* parked car or the same kerb line. They bound the
pipeline's **precision**; the rotation bounds its **correctness**.

⚠️ **54 of 199 frames on clip 1 have no `obstacle.offline` box cell inside the 60 × 32 m grid at
all** — agents exist but are behind the ego or beyond ±16 m. Those frames contribute to the marginal
and to no arm, which is why clip 1's arm denominators are small relative to 199 frames.

### 5.1 The pictures — `media/`

Two overlays, each showing the **canonical 256×640 cylindrical frame the model actually sees**
(LiDAR returns projected in, coloured by range; `obstacle.offline` cuboids as magenta wireframes)
**beside the metric BEV** (yellow = LiDAR occupancy, dark = shadow/unobserved, magenta = the same
cuboids as footprints), with the full text overlay the PI asks for: clip, frame, timestamps,
alignment deltas, point count, extrinsics, ego speed, skew, occupancy, **and the three hit rates**.

* `media/lidar_bev_overlay_0dfe63214bc2_f0300.png` — real **0.850** vs mirror **0.000**, marginal 0.028
* `media/lidar_bev_overlay_0dfe63214bc2_f0420.png` — real **0.944** vs mirror **0.000**, marginal 0.062

The camera path goes through `tanitad.data.calib.cylindrical_rectify` with the **per-clip** f-theta
principal point (it **refuses** the corpus median, because the front-wide has two rigs at cy ≈ 543 /
755 and a ray fan centred on the wrong one is ~215 px off) and `tanitad.data.rig_projection`, never
a re-derived formula.

⛔⛔ **THE FIRST RENDER OF THAT FIGURE WAS MIRRORED, AND EVERY NUMBER ON IT WAS CORRECT.** An extra
`[::-1]` on the BEV array flipped the panel while the hit rates — computed from the arrays, not the
picture — stayed right. **`R-2026-09-08-wpa-mirror` in a plotting costume**, found in this package,
inside the work whose whole subject is that defect. Two fixes shipped: the transpose is now asserted
against `disp.shape`, and **a labelled control marker is drawn ON the panel at a known rig
coordinate (x = 8 m, y = +10 m = 10 m LEFT)** so any reader can falsify the orientation without
reading code. ⚠️ A figure is an artifact like any other, and *"the numbers under it are right"* is
not a check on it.

---

## 6. ⛔ What the prior refutations DO and DO NOT say about this design

⭐ **This section exists so nobody later quotes a refutation against a design it does not cover.**

| prior finding | status | does it bind a LIDAR-LIFTED BEV? |
|---|---|---|
| `E-BEV-AUX-1` — a training-only, removable BEV auxiliary loss makes the trunk carry **transferable** agent localisation | **REFUTED as stated** (A2 fails: the lever does not separate from the raw-pixel floor; A4 not separated; A3 UNDERPOWERED) | ⚠️ **Partly.** It binds the *auxiliary-loss mechanism* and is the reason §7's ladder is about competition. It does **not** bind the target, which here is a different object: LiDAR carries static structure `obstacle.offline` does not contain, and a **measured** visibility state instead of a derived lower bound. |
| **The quantisation floor** — a dense BEV built on the image token grid: median **4** BEV cells (max 313) per token cell, only **242 of 640** token cells receiving any ground-plane cell; a **perfect** front-end reaches only **AP 0.4713** at 16×40 | **NOT RETRACTED — still load-bearing** | ⛔ **It binds the ADDRESS SPACE, not BEV.** It is a statement about **reusing the image token grid as a BEV address**. A **properly lifted** BEV — its own grid, its own resolution, its own splat index — **does not inherit it**, because the many-to-one map that creates the floor is exactly the step a lift replaces. ⭐ **Say this explicitly whenever the floor is quoted at this design.** |
| `E-READOUT-CEILING-1` — real-trunk AP 0.027–0.034 vs a 0.0325 marginal ⇒ *"the map does not contain agents"* | ⛔ **RETRACTED** (`R-2026-09-08-wpa-mirror`): the azimuth address was mirrored. `D0 − pos_only` is **separated on both geometries** (+0.02457 Cartesian, +0.07368 polar) ⇒ **the map already contained agents** | Do not quote the retracted AP column. The **oracle** ceiling (0.4713 / 0.3374) survives its re-read. |
| `D-P1-AGENTCOND-1` (+SEED1) — the sparse agent-detection auxiliary failed its bar at a **17 M** trunk; the deranged-join arm relocated the cost to **auxiliary-task capacity competition** | **CONFIRMED on two seeds** | ⭐ **This is the one that binds hardest**, and §7 is built around it. |

⚠️ **A documentation defect found in passing:** `stack/tanitad/data/bev_aux.py`'s module docstring
still asserts *"the map does not yet contain agents"* — a claim RETRACTED on 2026-09-08, living in
the module that the retraction was produced *by*. Flagged, not edited: it is another stream's owner
file and the fix belongs with whoever owns the WP-D package. **This is the stale-claim-in-prose
family that the feature-count pinning test exists for.**

---

## 7. The head — parameter cost COUNTED, and the brief's premise is half true

`code/p4_head_sizing.py` → `raw/head_sizing.json`. Modules **built and counted** with
`sum(p.numel() …)`; the lift-splat candidate was forward-passed on a real-shaped
`[2, 704, 8, 20]` tensor. Trunk baseline **108,257,502** (MEASURED: `E-BEV-AUX-1`'s
108,440,118 − its 182,616-param aux head). Ceiling **300 M**.

| candidate | params | % of trunk | × the WP-D aux head | total w/ trunk | headroom |
|---|---:|---:|---:|---:|---:|
| ⭐ **A — lift-splat, polar 24×20** | **190,258** | **0.18 %** | **1.0×** | 108,447,760 | 191.6 M |
| A — lift-splat, Cartesian 120×64 wide | 726,978 | 0.67 % | 4.0× | 108,984,480 | 191.0 M |
| B — cross-attn, polar 48×40, d192, L3 | 2,284,802 | 2.11 % | 12.5× | 110,542,304 | 189.5 M |
| B — cross-attn, polar 24×20, d256, L4 | 4,517,634 | 4.17 % | 24.7× | 112,775,136 | 187.2 M |
| B — cross-attn, Cartesian 120×64, d256, L4 | 6,360,834 | 5.88 % | 34.8× | 114,618,336 | 185.4 M |
| B — cross-attn, Cartesian 128×128, d256, L6 | 10,695,938 | 9.88 % | 58.6× | 118,953,440 | 181.0 M |

⭐⭐ **The brief's premise — *"a BEV encoder is a LARGER auxiliary task, so that risk rises"* — is
TRUE for cross-attention (12.5–58.6×) and FALSE for lift-splat at the polar geometry: 190,258
against 182,616 is a 4.2 % difference, i.e. 1.0×.** The lift is one 1×1 convolution; the entire
geometry — which BEV cell a (row, column, depth-bin) triple lands in — is a **precomputed index
with zero learned parameters**, built from the cylindrical ray model and the rig extrinsics.

⇒ **The cheapest admissible design is not a bigger bet than the one the programme has already
placed and paid for.** ⛔ And the parameter **ceiling is not the binding constraint at any rung** —
the largest candidate leaves 181 M of headroom. **The constraint is capacity competition**, which is
a different quantity, and confusing the two would be the same scope error as reading `df` on a pod.

### 7.1 ⭐ The cheapest experiment that catches competition early — **pre-registered**

Full document: **`PREREG_BEV_CAPACITY_COMPETITION.md`** (`E-BEVLIDAR-CAP-1`), written before any
outcome number exists, with both outcomes and four failure twins committed. In one paragraph:

**5 arms** — `C0` (aux off), **`C0b`** (aux off, *same seed*, `--out` the only differing token) and
**`C0c`** (aux off, seed 1) which exist solely to measure the replicate floor, `C1` (aux on) and
**`C2` (aux on, target SHUFFLED across the batch)**. `C2` is the instrument: same parameters, same
gradient path, **zero information** — the control that relocated `D-P1-AGENTCOND-1`'s cost from the
information to the task.

**The early detector costs one extra backward pass, no extra arm:** the cosine between the
trajectory gradient and the auxiliary gradient **on the shared trunk parameters**, logged every
step. It is readable from step 0, long before any planner metric separates. ⛔ It ships with **three
controls that have ANALYTIC targets and make the probe refuse if missed** — `cos(g,g) = +1.0`
exactly, `cos(g,−g) = −1.0` exactly (which fixes the sign convention nobody asserts), and a
**detached** head giving the zero vector (report `NaN`, never 0 by accident) — **plus a mutation arm
that must go RED**: scale the aux loss 30× to reproduce the MEASURED `E-DEC-18` mechanism (aux
~0.3–0.9 against an objective ~0.03), and the detector must fire. *A detector that cannot be made
to fire has not been shown to work.*

⛔ **Why three of five arms are controls:** on the REF-C planner rig a one-seed pair reads
`separated` on **5 of 9** metrics with **zero levers moved** — a **55.6 %** false-positive rate
(`H-ESTIM-SEED-1`, measured inside `E-BEV-AUX-1`). Without a replicate floor measured in the same
panel, no row in this ladder is readable.

---

## 8. What full-corpus GT generation would cost

Using the MEASURED mean **341.7 MB/clip** (4 clips, 240–403 MB) and **37.2 s/clip** of CPU:

| corpus | clips | LiDAR bytes to read | CPU to build the GT |
|---|---:|---:|---:|
| B1 (`D-CORPUS-B1`, the v7/refcv5 training corpus) | 4,713 | **~1.61 TB** | **~48.7 CPU-hours** (≈ **8 h** at 6 workers) |
| parity `physicalai-train-e438721ae894` | ~2,400 | ~0.82 TB | ~24.8 CPU-hours |
| B1 EVAL (the 139 joined clips — enough to validate the target end to end) | 139 | **~47 GB** | **~1.4 CPU-hours** |

**The GT itself is negligible** — a lean `occ` + `observed` target is ~1–2 KB/frame ⇒ **~1–2 GB for
all of B1**. The raw LiDAR is read and discarded.

⛔ **TRANSFER IS THE BINDING COST, AND IT IS THE ONE NUMBER I REFUSE TO EXTRAPOLATE.** Measured
single-stream rates over 4 clips: **1.59 · 2.63 · 5.41 · 11.57 MB/s** — a **7.3× spread** on
consecutive pulls of the same kind of object. A mean over four samples that disagree by 7× is not a
rate, so **no corpus-wide wall-clock from this package is admissible**. For scale only: at the
slowest sample 1.61 TB is ~12 days and at the fastest ~40 h, and parallel clip pulls, `hf_transfer`
or whole-chunk pulls on a bandwidth-rich pod each move that by an order of magnitude. ⇒ **A
bandwidth probe is the prerequisite for any full-corpus plan — 20 minutes of work, and not a GPU
decision.**
⭐ **Start with the 139-clip B1 EVAL slice (~50 GB, ~1.4 CPU-hours):** it is the set every v7 T1/val
number is computed over and it is already fully joined to `obstacle.offline`, so it validates the
target end to end before anything large is committed.

---

## 9. ⛔ Two more things the corpus has that nobody has looked at

1. **19 radar features**, verified by me in `features.csv` — `radar_front_center_imaging_lrr_1`
   (imaging long-range), 2 MRR, and 16 SRR around the vehicle. Coverage is **160,761 / 306,152
   clips** per the dataset card (**PUBLISHED**, not re-measured here), i.e. materially lower than
   LiDAR's 298,326. Radar would supply **range-rate**, which neither camera nor LiDAR gives directly
   and which is the LONGITUDINAL family's natural label. Unexamined.
2. **The read-set count moves from 6 to 7 if the LiDAR is ingested**, and
   `stack/tests/test_physicalai_feature_readset.py` pins that count against source in three layers.
   ⛔ **Do not hand-edit the prose count anywhere** — that number has gone stale four times, and it
   is pinned precisely so a change like this one fails the test and names the documents to update.
   This package **reads LiDAR in a research script, not in `physicalai.py`'s episode build**, so the
   pinned counts are **unchanged as of this turn**; they change the moment a builder lands.

---

## 10. Defects and traps this package hit (each cost real time, each is now pinned in code)

| # | what | where it is pinned |
|---|---|---|
| 1 | ⛔ **The join's `frame` is the ~10 Hz EPISODE index, not the 30 Hz camera frame.** Looking boxes up by camera-frame index returned agents ~3× too far back and made **every** registration arm read `n_box_cells_observed = 0` — a total-failure signature from an indexing bug. Fixed by keying on **`t_s`**, which both sides carry. | `p2_build_bev_gt.py`, comment + code |
| 2 | ⛔ **A flat z-band over-reports and drives the marginal to 0.544.** Fixed with a per-cell vertical-spread test; **both** rules written into the artifact. | `lidar_bev.ZBand` docstring |
| 3 | ⛔ **The BEV panel rendered mirrored while every number on it was right.** | `p3_overlay.py`, assertion + an on-panel labelled control marker |
| 4 | ⛔ **`ParquetFile.read()` on the clip parquet blocks for minutes at ~no CPU** (>2 GB held). Not a `BytesIO` problem — re-timed at 0.32 s / 0.01 s. Read row groups. | `p1_lidar_probe.py` docstring |
| 5 | ⚠️ The LiDAR span (0.0995–19.999 s) is **inside** the camera span (−0.0142–20.119 s), so the end frames have **no** spin within half a period — max \|dt\| **169.7 ms**. Those frames are **NO_LABEL**, never extrapolated. | reported in `align_dt_ms` with its note |
| 6 | ⛔ **A control that collapses on ONE clip is not a property of the pipeline.** `mirror_y` reads 0.75× on clip 1 and 1.52× on clip 2; the one-clip version of this report would have published the first as the headline control. Only the **ratio** real/wrong replicates. | §5.2; the sweep is written to run per clip |

---

## 11. Deliverable manifest

⛔ Everything is **STAGED, not committed, not pushed**, on `agent/arch-inf-20260803` in
`D:\Projects\TanitAD`. Nothing that took effort lives in only one place except where marked.

| artifact | where it lives |
|---|---|
| this result | `repo: TanitAD Research Lab/Architecture & Inference/Research/2026-09-11-lidar-bev-gt/RESULT.md` |
| pre-registration `E-BEVLIDAR-CAP-1` | `repo: …/2026-09-11-lidar-bev-gt/PREREG_BEV_CAPACITY_COMPETITION.md` |
| target builder (the reusable module) | `repo: …/code/lidar_bev.py` |
| ranged-zip LiDAR probe | `repo: …/code/p1_lidar_probe.py` |
| GT builder + per-frame registration | `repo: …/code/p2_build_bev_gt.py` |
| camera + BEV overlay renderer | `repo: …/code/p3_overlay.py` |
| head parameter sizing | `repo: …/code/p4_head_sizing.py` |
| full-clip registration sweep | `repo: …/code/p5_registration_sweep.py` |
| GT artifact (4 frames × 16 channels + `meta_json`) | `repo: …/raw/bev_gt_0dfe63214bc2.npz` (207,750 B) |
| per-frame report | `repo: …/raw/bev_gt_report_0dfe63214bc2.json` |
| **registration sweep, 199 spins, clip 1** | `repo: …/raw/registration_sweep.json` |
| **registration sweep, 199 spins, clip 2 (replicate)** | `repo: …/raw/registration_sweep_clip2.json` |
| 4-clip LiDAR probe (format, rate, sizes, attributes) | `repo: …/raw/lidar_probe.json` |
| head sizing | `repo: …/raw/head_sizing.json` |
| overlays (2 PNG) | `repo: …/media/lidar_bev_overlay_0dfe63214bc2_f{0300,0420}.png` |
| ⚠️ **raw LiDAR parquets (4 clips, ~1.37 GB)** | **`devbox: C:\Users\Admin\tanitad-caches\lidar-bev-20260911\` — ONE PLACE.** Deliberately not staged: they are gated PhysicalAI-AV bytes and are **re-fetchable by `p1_lidar_probe.py`** (35–230 s each, measured). |

⛔ **Clip ids are gated-confidential** (`parity_manifest.json`: *"the clip ids themselves are gated
PhysicalAI-AV content … only these digests are quotable in the repo"*). Every artifact here
identifies clips by **`sha256(clip_id)[:12]`**; the plaintext ids appear only in local cache
filenames, never in a staged document.

⛔ **Parity is untouched.** Nothing here re-selects episodes; the two clips are members of the
existing B1 EVAL join and were read, not chosen.

---

## 12. Escalations — these need a decision, and they will not surface on their own

1. ⭐ **The premise reordering.** The brief's *"a BEV encoder is a larger auxiliary task"* is
   **measured false for the lift-splat/polar rung (1.0× the WP-D head)**. If that rung is the one
   that runs, the capacity argument against it is **arithmetically void** and the ladder can start
   much cheaper than planned.
2. ⛔ **A bandwidth probe is the gate on any full-corpus plan**, not a GPU decision — ~20 minutes.
   Until it exists, no corpus-wide wall-clock figure from this package is admissible.
3. ⚠️ **`stack/tanitad/data/bev_aux.py` quotes a claim retracted on 2026-09-08.** It belongs to the
   WP-D stream; raised here rather than edited across an ownership boundary — and raised in a
   report's headline rather than written into a README, because that has been missed for 10 days
   before.
4. ⭐ **19 radar features are unexamined**, and radar is the only sensor in the corpus that measures
   **range-rate** directly — the natural label for the LONGITUDINAL family, which is **88.7 % of the
   measured oracle gap**.
5. ⚠️ **The next unblocked step is not a model.** In priority order, all 0-GPU: (a) the bandwidth
   probe (§8); (b) the **139-clip B1 EVAL** GT build (~47 GB, ~1.4 CPU-h), which is the set every v7
   T1/val number is computed over and is already fully joined to `obstacle.offline`; (c) a
   `taniteval`-side loader + a test that asserts the artifact's own `meta_json` against the builder's
   specs, so the units can never drift from the bytes. The **head** waits on (b), because a target
   that has not been built at corpus scale cannot train anything.

---

## ⛔ Corrections found 2026-09-13 — read before reusing this package's builder

Two defects in `code/lidar_bev.py` of THIS package were found and fixed in
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-13-bev-lidar-corpus-and-head/` (this file is not edited otherwise):
* **D-LIDAR-DESKEW-SIGN-1** — `deskew_rigid` rotated by −yaw_rate·dt; the correct inverse is +yaw_rate·dt (the corrected
  sign wins 24/24 high-yaw instants, spin-to-spin residual 0.0135 m vs 0.1545 m).
* **D-POLAR-OBSERVED-BIAS-1** — the polar `observed` rule (`~shadow ∨ occ`) kept occupied cells behind the first hit and
  dropped free ones (78–91 % of scored positives sat there); the new package writes all three variants into every artifact.
See `Project Steering/GOALS_AND_CLAIMS.md` (E-BEVHEAD-FROZEN-1 block) and `Project Steering/RETRACTION_LOG.md`.

<!-- CORRECTED-2026-09-13-BEVPKG -->
