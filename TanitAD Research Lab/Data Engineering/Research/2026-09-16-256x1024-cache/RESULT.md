# Wide-azimuth v2ep caches — 256x1024 AND 408x1024, built and validated on the 139 B1 EVAL clips

**2026-09-16 / 2026-09-17 · Data Engineering · worktree `agent/dataeng-256x1024-20260916`**
Pre-registration: [`PREREG.md`](PREREG.md) + its 408 ADDENDUM (both written
before the comparisons they govern; no tolerance was moved, and the one
allowance that was pre-registered turned out not to be needed).

---

## ⭐ ESCALATION 1 — the vertical field. BOTH caches are now built; choose with numbers.

`calib.cylindrical_rays` uses **the same `f_ref` on both axes**
(`phi = (u-(W-1)/2)/f_ref`, `y_n = (v-(H-1)/2)/f_ref`). Holding `H = 256` while
`f_ref` rises 305.577 -> 488.924 to keep the azimuth at 120 deg therefore
narrows elevation by the same 1.6x. Rather than ask blind, the alternative was
**built and put through the same gates**.

**Every cell below is read from a built artifact or a measurement receipt
(`raw/geometry_comparison.json`); nothing is retyped.**

| | **256x640** (today) | **256x1024** | **408x1024** |
|---|---|---|---|
| `f_ref` | 305.5774907 | 488.9239852 | 488.9239852 |
| HFOV | 120.000000 | 120.000000 | 120.000000 |
| **deg / column** | 0.187500 | **0.117188** | **0.117188** |
| **VFOV** | **45.4556°** | **29.3415°** | **45.2960°** |
| VFOV as % of today | 100.0 % | **64.5 %** | **99.6 %** |
| **rows of today's frame kept** | -0.5 … 255.5 | **47.5 … 207.5** | **0.0 … 255.0** |
| MB / clip (139 measured) | 37.77 | 58.05 | 82.00 |
| corpus GB (4,713) | 178.0 | 273.6 | 386.5 |
| stride 16: grid / tokens / deg-col | 16x40 / 640 / 3.000 | 16x64 / **1024** / **1.875** | 26x64 / **1664** / **1.875** |
| stride 32: grid / tokens / deg-col | 8x20 / 160 / 6.000 | 8x32 / **256** / **3.750** | 13x32 / **416** / **3.750** |
| resnet101 K=3 activations, MB | 645.9 | 1033.5 | 1662.9 |
| activation x vs 256x1024 | 0.625 | 1.000 | **1.609** |
| build wall at 6 workers | — | 6.62 s/clip (540/h) | 8.15 s/clip (442/h) |
| corpus build hours | — | 8.7 h | 10.6 h |

**What each geometry keeps of the source, plainly.** Expressed as rows of
today's 256x640 frame (`v = (H-1)/2 + 1.6*(v' - 127.5)`):

* **256x640** — today's view, rows -0.5 … 255.5.
* **256x1024** — rows **47.5 … 207.5**. It **discards the top 48 and bottom 48
  rows of what we have today** and keeps a 160-row central band. Azimuth is
  1.6x finer; elevation coverage is 64.5 % of today's.
* **408x1024** — rows **0.0 … 255.0**. It keeps **the whole band** (0.5 row short
  at each end, i.e. 99.6 % of today's VFOV) at the same 1.6x-finer azimuth.

⭐ **408x1024 is the only option that is strictly better than today on both
axes.** 256x1024 buys azimuth by spending elevation; 408x1024 buys azimuth and
spends disk and activations instead — **+41 % cache** (386.5 vs 273.6 GB),
**+61 % activations** (1662.9 vs 1033.5 MB), **+22 % build time** (10.6 vs 8.7 h),
and **1.63x the tokens** at every stride (1664 vs 1024 at stride 16).

⚠️ **One honest wrinkle, and it is a property of the CHECK, not the cache.** An
exact 1.6x supersample of 256x640 needs height **409.6**, which is not an
integer, so no integer height is both field-exact and integer-aligned to the 640
row grid. H=408 lands **half a pixel** off. Two integer-aligned heights exist if
the PI prefers one: **H=400** (VFOV 44.4952°, offset +3 rows) and **H=416**
(VFOV 46.0921°, offset -2 rows). H=416 is integer-aligned AND slightly *exceeds*
today's field, at ~2 % more disk than 408 — arguably the cleanest choice, but it
was **not built** (evidence class: derived from the frame, not measured).

A third option — giving `cylindrical_rays` a separate `f_ref_y` — is a change to
the shared resampler and is NOT recommended without its own validation.

Note DiffusionDrive/NAVSIM's 1024x256 is also a 4:1 frame, so 256x1024 is the
paper-matching shape; 408x1024 is the lose-nothing shape.

## ⚠️ ESCALATION 2 (RESOLVED, SCOPED) — the overlap is with the OLD parity corpus only

The ingest gate REFUSED the build (`role=eval`) with a MEASURED
**11 / 139 clips inside `physicalai-train-e438721ae894`** (and 0 of the 40
deployed-val episodes). That corpus is the **old parity corpus of the flagship /
refav1 line** — *not* the corpus refcv6 and the v7.2 arms train on. The gate can
only ask its one question, so its `decision_grade: false` is necessarily blanket.

**The stamp is therefore CONDITIONAL, and the cache manifests now carry that
scope explicitly** (`MANIFEST.decision_grade_scope`; the gate's own
`parity_ingest_gate` record is left verbatim, 11 sha12 included):

> `decision_grade: FALSE` for models trained on
> `physicalai-train-e438721ae894` (flagship / refav1).
> `decision_grade: TRUE` for models trained on the v7.2 / B1 corpus.

Disjointness from the v7.2 / B1 corpus was **re-measured here from primary
sources at the pinned revision**, not adopted:

| source | n train clips | overlap with the 139 |
|---|---|---|
| `index/clip_index_v7.2_train.json` -> `clips` | 4,572 | **0** |
| `splits/eval_split_v3.json` -> `train_clip_ids` (identical set) | 4,572 | **0** |
| Thor's built B1 train cache, by direct `ls` | 4,713 payloads | **0** |

And positively: **all 139 are inside** `clip_index_v7.2_eval` (147) and inside
`eval_split_v3.eval_clip_ids` (141 — the banked 139 is that split minus sha12
`843e4ee8d5d2` and `93489c4af0ab`). Split provenance: `v7.1-eval-split-3`, seed
7, `based_on_labels_md5 93b49273ba0bb038cd79dbac30820fd1`.

⚠️ I could not reproduce the label-blob md5 `fa89ea55` quoted to me; the split
file's own `based_on_labels_md5` is `93b49273…`. The **substance** (4,572 train
clips, 0 overlap) is independently confirmed by all three sources above, so the
conclusion stands on those, not on that hash.

**Still true:** a parity-trained model scored over all 139 is contaminated on
7.9 % of the set. The remedy is documented, not implied — score such a model on
the **clean 128-clip subset** (the built set minus
`parity_ingest_gate.overlap_sha12_parity_train`).

The scoping tool is `code/restamp_manifest_scope.py`. It re-measures the
disjointness itself on every run and **REFUSES** to write the exemption if the
built set overlaps the v7.2 train corpus — proven by mutation: fed the 139 plus
one real v7.2-train clip it exits **1**, prints
`REFUSING to write the exemption … (index 1, split 1)`, and leaves the manifest
untouched.

---

## The caches — TWO, both complete

| | clips | failures | size | MB/clip | build wall |
|---|---|---|---|---|---|
| `D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl/` | **139 / 139** | **0** | 8.069 GB | 58.05 | 15.3 min |
| `D:/Projects/TanitAD-artifacts/v2ep-eval139-408x1024cyl/` | **139 / 139** | **0** | 11.398 GB | 82.00 | 18.7 min |

Each carries a `MANIFEST.json` with, per clip: `clip_sha12`, `n_frames`,
`bytes`, `sha256` of the payload, `f_ref`, `hfov_deg`, `vfov_deg`, `height`,
`width`, `projection`, `deg_per_col` — plus the gate record and the
`decision_grade_scope` block (ESCALATION 2).

Geometry is **uniform across all 139 payloads in each cache** — one distinct
tuple per cache (`(1024, 256, 488.9239852, "cylindrical")` /
`(1024, 408, 488.9239852, "cylindrical")`), one distinct `hfov_deg` = 120.0 —
read back off each payload's own stored frame, not off the request.

⚠️ **A trap found on the way to 408, worth keeping.** `calib.as_frame(frame,
size, 266.0)` **raises** whenever `size != 256` and a `CanonicalFrame` is also
passed ("two sources of truth is the bug this object exists to remove"). The
builder had been passing `size=height`, which is correct-by-coincidence at
H=256 and would have failed **every** clip at H=408. `size` is the legacy
sentinel, not the frame height; the real geometry travels in `frame`, and the
payload's `image_size` stays 256 while `image_h`/`image_w` carry the truth.
Fixed and commented in `build_v2ep_wide.py`.

`f_ref` is **solved, never copied**: `HFOV = 2*(W/2)/f_ref` (cylindrical is
linear in azimuth) so `f_ref = 512 / 1.0471975512 = 488.9239851783`, asserted
against each payload to `1e-6` px and `|HFOV - 120| < 1e-3`.

Everything else is identical to the 640 cache: **PNG** (lossless), `n_stack` 3,
the same `_resampled` frame indices, the same per-clip `(cx, cy)` principal-point
crop (per-clip intrinsics CSV present — load-bearing, the rig-B fix), the same
pose/action payloads.

**One builder, one resampler.** `code/build_v2ep_wide.py` is a *driver*: every
pixel comes from `v2_compressed.build_compressed` ->
`calib.cylindrical_rectify`, the same call chain that produced
`physicalai-b1-w120-256x640cyl`. The only thing it changes is the
`CanonicalFrame`. No second resampler was written.

Source: `Sayood/tanitad-v7-training-corpus` @
`a0cf20dfb4eafa29b0ac1f3c05337f002bc33ca0` — the repo and revision read out of
Thor's own production feeder, not guessed. **139 / 139 mp4s sha256 + byte-length
verified against that revision's `camera/camera_sha256.json`; 0 mismatches.**

---

## The checks — every one reads a known value

| check | what it reads | 256x1024 | 408x1024 |
|---|---|---|---|
| **G0** builder parity | THIS box's 640 build vs Thor's **deployed** payload | **PASS** | (shared) |
| **G1** downscale geometry | x-corr peak vs the 640 band | **PASS** 30/30 | **PASS** 30/30 |
| **G2** downscale photometry | median / MAE / PSNR | **PASS** 6/6 | **PASS** 6/6 |
| **C1** half-pixel control | price of the 408 row offset, measured on the 256 case | — | **penalty measured** |
| **G3** mutations | deliberate defects | **3/3 DETECTED** | **2/2 DETECTED** |
| **G4** frame/label alignment | SAM3 `t_img_us`, bevhead token index | **PASS** exact | **PASS** exact |
| **G5** geometry via `rig_projection` | centre col/row, known azimuth, mirror | **PASS** | **PASS** (17/17 across all three frames) |
| **G6** token grid | real `timm resnet101` forward | **PASS** | **PASS** (6/6 levels) |
| **restamp guard** | refuses the exemption on a real leak | **exit 1, DETECTED** | **exit 1, DETECTED** |

### G0 — this box reproduces the DEPLOYED 640 cache
None of the 139 EVAL clips exist in any banked 640 cache (they are the held-out
split: **0 overlap** with Thor's 4,713-clip train cache, with `EVAL6`, and with
the local 603-clip val mirror). So the 640 side of the identity check had to be
built here — and that build is first proven against the deployed one on two
clips that ARE in it:

| clip | n frames | `poses` absmax | `actions` absmax | worst MAE | worst PSNR |
|---|---|---|---|---|---|
| `024138543dcc` | 201 / 201 | **0** | **0** | 0.7705 | 48.95 dB |
| `b38c4598ba22` | 201 / 201 | **0** | **0** | 0.7838 | 48.89 dB |

Pre-registered: MAE ≤ 1.0, PSNR ≥ 40 dB, pose/action absmax ≤ 1e-6. Poses and
actions came back **exactly zero** across architectures (Thor aarch64 vs this
x86-64 box); the sub-1-level pixel residual is h264/`grid_sample` last-ulp.

### G1 / G2 — the projection did NOT change
The correspondence is exact and derived, not fitted (PREREG §1): a 1/1.6 resize
of the 1024 frame maps output column `c` to 640 column `c` and output row `r` to
640 row `48 + r`. So the comparison is `resize(frame1024,(160,640))` vs
`frame640[48:208,:]`, integer-aligned, with **no interpolation of the reference**.

30 frames (6 clips x 5, `linspace`-chosen before looking at content):

| clip | worst median | worst MAE | worst PSNR | x-corr peaks |
|---|---|---|---|---|
| `0dfe63214bc2` | 0.000 | 1.348 | 37.34 dB | all (0,0) |
| `c64f0103aece` | 1.000 | 3.289 | 32.13 dB | all (0,0) |
| `54e3dc4b3ae4` | 0.000 | 1.262 | 39.06 dB | all (0,0) |
| `d0c6db065fa0` | 1.000 | 1.929 | 34.73 dB | all (0,0) |
| `c8831556989f` | 0.000 | 0.964 | 38.91 dB | all (0,0) |
| `bfc3a203bd06` | 0.000 | 0.824 | 41.00 dB | all (0,0) |

Pre-registered: median ≤ 2.0, MAE ≤ 4.0, PSNR ≥ 28 dB, peak at (0,0). All met.
The residual is the pre-registered kernel difference (a point-sampled 640 grid
vs an antialiased average of a 1024 grid), **not** a geometry change — G1 pins
that, and G3 shows the check would have caught one.

### G3 — the guards were proven by MUTATION, not inspection
Each defect was really built or really constructed, then run through the SAME
comparison:

| mutation | what it does | peak | median | MAE | PSNR | verdict |
|---|---|---|---|---|---|---|
| **M1** unscaled `f_ref` | W=1024 built with `f_ref` 305.5774907 — delivers **192.0000 deg**, not 120 | (-12,12) | 13.0 | 23.85 | 16.87 dB | **DETECTED** |
| **M2** wrong row crop | compare against `frame640[0:160]` | (12,12) | 16.0 | 29.85 | 15.20 dB | **DETECTED** |
| **M3** mirrored azimuth | horizontally flipped | (12,-12) | 19.0 | 43.92 | 11.87 dB | **DETECTED** |

M1 is the exact defect the brief warns about, and it was *built*, not imagined:
keeping the 640 focal at 1024 columns asks for a 192 deg field from a 120 deg
sensor.

### C1 + G1/G2/G3 at 408x1024 — and it passes the UNWIDENED tolerance

An exact 1.6x supersample needs H = **409.6**, so H=408 sits **half a pixel** off
the 640 row grid (PREREG ADDENDUM A1). The check therefore has to resample the
reference by half a pixel — so that half-pixel was **priced first**, on the
256x1024 comparison that had already passed:

| C1 control (worst over 30 frames) | median | MAE | PSNR |
|---|---|---|---|
| aligned (the passed 256x1024 case) | 1.0 | 3.2894 | 32.131 dB |
| same, reference shifted +0.5 px | 2.0 | 3.9222 | 31.186 dB |
| **half-pixel penalty** | **+1.0** | **+0.6328** | **-0.945 dB** |

The 408 tolerance was then widened by exactly that, and fixed before the 408
numbers were seen: median <= 3.0, MAE <= 4.6328, PSNR >= 27.055 dB.

| clip | worst median | worst MAE | worst PSNR | x-corr peaks |
|---|---|---|---|---|
| `0dfe63214bc2` | 0.000 | 0.940 | 40.67 dB | all (0,0) |
| `c64f0103aece` | 1.000 | 2.552 | 34.07 dB | all (0,0) |
| `54e3dc4b3ae4` | 0.000 | 0.783 | 42.98 dB | all (0,0) |
| `d0c6db065fa0` | 1.000 | 1.646 | 36.42 dB | all (0,0) |
| `c8831556989f` | 0.000 | 0.572 | 43.21 dB | all (0,0) |
| `bfc3a203bd06` | 0.000 | 0.559 | 44.78 dB | all (0,0) |

⭐ **The widening turned out to be unnecessary.** 408's worst numbers (median
1.000, MAE 2.552, PSNR 34.07 dB) clear the ORIGINAL unwidened tolerance
(2.0 / 4.0 / 28 dB) with room to spare — and are *better* than 256x1024's
(median 1.000, MAE 3.289, PSNR 32.13 dB), because resizing 408->255 is a gentler
resample than 256->160. The allowance was pre-registered and then not needed;
it is reported rather than quietly dropped.

Mutations re-run at 408: wrong row crop **DETECTED** (peak (12,-1), MAE 29.59,
PSNR 15.25 dB) and mirrored azimuth **DETECTED** (peak (12,-12), MAE 48.54,
PSNR 11.09 dB).

### Activation cost — measured here, and NOT the number I was given

⚠️ **I never measured 3,257 MB.** That figure appears nowhere in this package;
it did not come from my work and I cannot vouch for it. What I measured instead,
on CPU (the RTX 4060 belongs to another agent and was not touched):

**the summed bytes of every leaf module's output tensor in one `resnet101`
`features_only` forward, batch 1, `in_chans = 9` (K=3), fp32.**

| geometry | activations | returned features | x vs 256x1024 |
|---|---|---|---|
| 256x640 | **645.9 MB** | 28.75 MB | 0.625 |
| 256x1024 | **1033.5 MB** | 46.00 MB | 1.000 |
| 408x1024 | **1662.9 MB** | 73.50 MB | **1.609** |

⚠️ This is **not** a CUDA `max_memory_allocated` figure: a CUDA peak also carries
workspace, fragmentation, parameters, gradients and optimizer state, so it is a
larger number measured a different way. Do not set 645.9/1033.5/1662.9 MB beside
3,257 MB as if they were the same quantity. **The ratio is the part that
transfers**, and it is what the batch envelope turns on. If 3,257 MB really is
the 256x1024 peak, the implied 408x1024 peak is **~5,241 MB** — ESTIMATED,
evidence class derived, valid only insofar as activations dominate that peak.

### G4 — not a single row moved
The SAM3 map's own metadata states the grid it was built on verbatim
(`"v2ep EPISODE grid: t_query = linspace(t_cam[0], t_cam[-1], int(span_s*10)),
frame = first camera frame at or after t_query (v2_compressed._resampled);
axis0 = raw v2ep frame index"`). That grid was recomputed from the staged
`timestamps.parquet` and compared **exactly** (no tolerance):

* **135 / 135** clips that have a map file: `t_query` absmax **0**,
  `cam_frame_idx` absmax **0**, `t_img_us` absmax **0**, and
  `payload n_frames == len(t_img_us)`.
  *(4 of the 139 have no map file — the SAM3 eval run produced 135.)*
**Run separately on BOTH caches** (`raw/check_alignment.json`,
`raw/check_alignment_408.json`) — identical results, which is the point: the
time grid does not depend on the frame geometry, and that was verified rather
than assumed.

* Banked token index `bevhead-20260913/tokens/index.npz`: its 139 `clip_sha12`
  equal the 139 built **as a set AND in order**; **139 / 139** clips satisfy
  `max(raw_frame)+1 == n_frames` and `stacked_row == raw_frame - 2` (n_stack 3).
  27,664 rows total.

### G5 — geometry end to end, through `tanitad/data/rig_projection.py`

17 checks across all three frames, **all PASS**:

| | 256x640 | 256x1024 | 408x1024 |
|---|---|---|---|
| straight ahead (0,0,1) -> col | **319.5** | **511.5** | **511.5** |
| azimuth +30 deg -> col | **479.5** (= 160.0) | **767.5** (`f_ref*pi/6` = 256.0 exactly) | **767.5** |
| **mirrored azimuth -> col** | **159.5** — "same col" **FALSE** | **255.5** — **FALSE** | **255.5** — **FALSE** |
| col -> ray -> col round trip | max err **0.0 px** | **0.0 px** | **0.0 px** |
| **centre row is boresight** | row 127.5, elev **0.0°** | row 127.5, elev **0.0°** | row **203.5**, elev **0.0°** |
| edge-to-edge span | 119.812498° | 119.882814° | 119.882814° |

The centre-row check is new for this round — it reads the axis 408x1024 exists to
restore, and confirms the boresight lands on row **203.5** at H=408 (and that the
frame's own VFOV is 45.2960°).

The mirror test is deliberately run at a **non-zero** azimuth: a mirrored image
is invisible at azimuth 0 (both land on the centre column), which is how the
`R-2026-09-08-wpa-mirror` class escaped once. The mirrored column lands exactly
at `W-1-col`, i.e. the flip about the centre — proving the test is sensitive to
the defect it names.

### G6 — the token grid, MEASURED on a real frame
A real `timm resnet101 features_only` forward on a real cache frame (not
arithmetic):

| | stride 16 | stride 32 |
|---|---|---|
| 256 x 640 (today) | 16 x 40 = 640 tokens, 3.000 deg/col | 8 x 20 = 160 tokens, 6.000 deg/col |
| **256 x 1024** | **16 x 64 = 1024 tokens**, **1.875 deg/col** | **8 x 32 = 256 tokens**, **3.750 deg/col** |
| **408 x 1024** | **26 x 64 = 1664 tokens**, **1.875 deg/col** | **13 x 32 = 416 tokens**, **3.750 deg/col** |

Azimuth resolution improves **1.6x** at both strides. For reference the papers'
1024x256 over ~140 deg is 0.1367 deg/px; ours is now **0.1172 deg/px** — i.e.
**1.17x finer than DiffusionDrive/NAVSIM per pixel**, where the 640 cache was
1.37x coarser. ⚠️ Per-pixel only: their field is ~140 deg and ours is 120 deg, so
we are finer *within a 20-deg-narrower fan*, not strictly better. And on the
other axis we are now **worse** than before — see ESCALATION 1.

---

## Cost of the full corpus — MEASURED rates, projected

All rates measured on this box (CPU/IO only; the RTX 4060 was not used).

**Measured on the 139:**

| | 256x1024 | 408x1024 |
|---|---|---|
| download, serial (shared) | **2.74 s/clip**, 5.345 MB/s, 14.64 MB/clip, 2.035 GB | same sources, already local |
| build, 6 workers | **6.62 s/clip** wall (138 in 920 s = **540 clips/h**) | **8.15 s/clip** wall (138 in 1,125 s = **442 clips/h**) |
| cache size | **58.05 MB/clip** | **82.00 MB/clip** |
| growth vs the deployed 640 | **1.537x** | **2.171x** |

**Projected for the full corpus** (4,713 clips — 4,719 in `production_order`
minus the 6 the ingest gate drops as deployed-val; the source corpus is
**61.62 GB / 4,719 clips**, and the 139-clip sample averages 1.121x the corpus
mean, so these are slightly conservative):

| | 256x1024 | 408x1024 |
|---|---|---|
| download (shared) | **61.5 GB**, **3.2 h** serial (~0.8 h at 4x parallel) | same |
| build | **8.7 h** | **10.6 h** |
| **wall, pipelined** | **~8.7 h** | **~10.6 h** |
| wall, strictly serial | 11.9 h | 13.8 h |
| cache on disk | **273.6 GB** | **386.5 GB** |
| peak disk (source + cache) | **~335 GB** | **~448 GB** |

Reference point for both: the deployed 256x640 cache is **178.02 GB / 4,713
payloads** (`du -sb` on Thor, exact bytes — not the `du -sh` rounding).

Either fits comfortably on D: (**1.3 TB free**).

### Run it HERE, not on Thor — and the reason is disk, not speed

| | this dev box | Thor |
|---|---|---|
| cores | 24 | 14 |
| **free disk** | **1.3 TB on D:** | **146 GB** — *less than either cache needs (274 GB / 387 GB)* |
| availability | free now | SAM3 corpus production, **1,052 / 4,719** done at **26.3 clips/h** -> ETA ~22 Sep |
| source mp4s | already staged (2.0 GB of 61.6 GB) | evicted as produced; would re-download 61.6 GB |

Thor cannot hold either cache without first deleting the 178 GB 640 cache — the
reference this validation depends on. **Recommendation: run the corpus rebuild
on this box, starting now; it finishes in under a day (8.7 h at 256x1024, 10.6 h
at 408x1024) and never touches Thor.** Waiting for 22 Sep buys nothing and costs
the reference.

---

## Deliverable manifest

| artifact | where | copies |
|---|---|---|
| **256x1024 cache, 139 clips + `MANIFEST.json`** | `D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl/` | **ONE PLACE ONLY** |
| **408x1024 cache, 139 clips + `MANIFEST.json`** | `D:/Projects/TanitAD-artifacts/v2ep-eval139-408x1024cyl/` | **ONE PLACE ONLY** |
| 256x640 reference, 6 eval clips | `D:/Projects/TanitAD-artifacts/v2ep-eval6-256x640cyl-REF/` | **ONE PLACE ONLY** |
| G0 local 640 rebuild (2 clips) | `D:/Projects/TanitAD-artifacts/v2ep-g0-local640/` | ONE PLACE ONLY |
| G0 deployed 640 payloads (2 clips) | `D:/Projects/TanitAD-artifacts/v2ep-g0-thor640/` | also on Thor |
| M1 mutant payload (f_ref 305 at W=1024) | `D:/Projects/TanitAD-artifacts/v2ep-mutants/` | **ONE PLACE ONLY** |
| staged sources (mp4 + timestamps + egomotion zip + intrinsics) | `D:/Projects/TanitAD-artifacts/v2ep-eval139-build/` | mp4s re-fetchable from HF |
| builder, fetcher, 5 check/measure scripts, cost projector, re-stamper | `…/2026-09-16-256x1024-cache/code/` | in this worktree (staged) |
| check outputs, receipts, comparison, activation + cost projections | `…/2026-09-16-256x1024-cache/raw/` | in this worktree (staged) |
| build logs (both caches) | `…/raw/build1024.log`, `…/raw/build408.log` | in this worktree (staged) |

Clip ids appear as **sha12 only** in everything repo-bound (verified by regex
scan: no full UUID in any package file). Full ids exist only as local cache
**file names**, which the builder requires.

## Reproduce

```
# 1. fetch (sha256-checked against the pinned revision)
python code/fetch_corpus_clips.py --ids ids.txt --root <staging root>
#    then place calibration/physicalai_front_wide_intrinsics.csv under <root>

# 2. build at any (H, W); --width 640 --height 256 reproduces the deployed
#    cache (proven, G0). size is NEVER the height - see the as_frame trap.
python code/build_v2ep_wide.py --out <cache> --root <staging root>     --clips ids.txt --width 1024 --height 256 --workers 6 --role train
python code/build_v2ep_wide.py --out <cache408> --root <staging root>     --clips ids.txt --width 1024 --height 408 --workers 6 --role train

# 3. scope the decision_grade stamp (re-measures disjointness; refuses on a leak)
python code/restamp_manifest_scope.py --manifest <cache>/MANIFEST.json     --ids ids.txt --v72-train-index <index/clip_index_v7.2_train.json>     --eval-split <splits/eval_split_v3.json>

# 4. checks
python code/check_rig_geometry.py <raw dir> 256x1024=f1.png 256x640=f2.png     408x1024=f3.png
python code/check_downscale_identity.py --wide <1024> --ref <640> --ids ids6.txt     --root <staging root> --tmp <mutants> --out raw/check_downscale_identity.json     --g0-local <local640> --g0-thor <thor640>
python code/check_downscale_408.py --wide408 <408> --wide256 <1024>     --ref <640> --ids ids6.txt --out raw/check_downscale_408.json
python code/check_alignment.py --cache <cache> --root <staging root>     --ids ids.txt --maps <sam3 maps> --tokens <tokens/index.npz>     --out raw/check_alignment.json

# 5. cost + comparison
python code/measure_activation_cost.py --k 3 --out raw/activation_cost.json
python code/project_corpus_cost.py --fetch-receipt ... --manifest ...     --sha-table <camera_sha256.json> --ref-cache-bytes 178.024196794     --ref-cache-clips 4713 --out raw/corpus_cost_projection.json
python code/compare_geometries.py --manifest-256x1024 ... --manifest-408x1024 ...     --ref640-bytes 178024196794 --ref640-clips 4713     --activation raw/activation_cost.json     --geometry-check raw/check_rig_geometry.json     --out raw/geometry_comparison.json
```

⚠️ This box is **cp1252**: the ingest gate prints U+26A0 on the sanctioned-audit
branch and the build dies *at the guard* with `UnicodeEncodeError` unless stdout
is UTF-8. The drivers reconfigure their own stdout; run with
`PYTHONIOENCODING=utf-8` as well.

## Open items for the PI

1. ⭐ **256x1024 or 408x1024?** (ESCALATION 1). Both are built, validated by the
   same gates, and priced. **408x1024 is the only one that is not worse than
   today on either axis**; 256x1024 is the paper-matching shape and costs 35 % of
   the vertical field. Price of 408 over 256: +41 % disk (386.5 vs 273.6 GB),
   +61 % activations, +22 % build time, 1.63x tokens. **Blocking for the corpus
   rebuild.**
   * If the half-pixel row offset matters, **H=416** is integer-aligned to the
     640 grid and slightly exceeds today's field — derived, not built; ~20 min to
     add.
2. **Scoring parity-trained models** (ESCALATION 2, otherwise resolved). The
   `decision_grade` stamp is now conditional and the exemption is re-measured on
   every run. The only live question: when a **parity-trained** model is scored
   on this set, use the **clean 128-clip subset**? (For refcv6 and every v7.2 arm
   nothing changes — those are clean over all 139.)
3. `stack/scripts/pod_build_b1_epcache.py` is hard-wired to 256x640.
   `code/build_v2ep_wide.py` is a strict superset (G0 proves `--width 640`
   reproduces the deployed cache) and now also handles non-256 heights — the
   `as_frame` trap above is exactly the kind of thing a second driver hides.
   Recommend retiring the hard-wired driver rather than adding a third spelling.
   Not done here; it is a shared-stack change and wants its own review.
