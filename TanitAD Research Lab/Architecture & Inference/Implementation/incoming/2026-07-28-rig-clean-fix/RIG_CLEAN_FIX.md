# The rig-clean field — measured, and it is a SLICE, not a rebuild

*2026-07-27 (Europe/Berlin; pods UTC). Owner: the rig-clean-fix stream.
Retraction class **C26**. Repo HEAD at start `2b0f166`.*

---

## 0. Headline — the four things the PI asked for

| question | answer | class |
|---|---|---|
| **Slice or rebuild?** | ⭐ **SLICE.** A rebuild at the clean frame is **BIT-IDENTICAL** to a row/column slice of the frames already built — verified on **1,206 real decoded frames** of the real cache, 6 clips, both rigs, `max_abs_diff 0`, `n_pixels_differing 0`. **The ~3.5 h build is not wasted, and the fix can be applied in the LOADER at zero storage cost.** | MEASURED |
| **The band** | At 624 columns: rig A fully observes **rows [0, 255] — the whole frame**; rig B **rows [0, 218] at worst**. Both contiguous, every distinct geometry. Intersection ⇒ largest **centred** frame both rigs fully observe is **176 × 624** (rows `[40:216]`, cols `[8:632]`). n = 3,000, 0 failures. | MEASURED |
| **The field price** | **1.195 %** of parent-visible agent samples (n = 2,409,188 samples over 400 clips). ⭐ **98.6 % of that price is the 16 columns, not the 80 rows** — the height cut alone costs **0.0165 %**. Near-field ground: **+1.39 m** (rig A) / **+0.99 m** (rig B). | MEASURED |
| **The residual** | Pad/mask fraction on the fix: **rig A 0.0000, rig B 0.0000** (n = 240 real clips, 120/120, max as well as mean). ⛔ **But a rig-correlated residual survives that is NOT geometry**: real all-zero pixels **0.0001 (A) vs 0.0079 (B)**, ~95 % of it *scene* black (night/tunnel). No crop removes it. | MEASURED |

**Pre-registered verdict: CONFIRM**, with a named and quantified residual that belongs to a
different mechanism than C26. See §7.

🔴 **The escalation (§9):** the clean frame is a *loader* change, not a corpus change — nothing
needs rebuilding — but **the v5 trainer must be told to pass it**, and `_geometry.json`'s
one-clip `observed_frac: 1.0` declaration is wrong about the corpus and is fixed here.

---

## 1. What was measured, on what, and where

| instrument | host | n | raw |
|---|---|---|---|
| per-rig observed band, ray map, real per-clip intrinsics | pod3 (idle A40) | **3,000 clips**, 121 distinct sensor geometries, **0 errors** | `raw/band_full_3000.json` |
| the vertical band at 624 columns (unambiguous form) | pod3 | 121 distinct geometries | `raw/band624.json` |
| slice-vs-rebuild on the REAL built cache (rebuild from mp4) | pod2 (build finished) | 6 clips × 201 frames, both rigs | `raw/verify_val.json` |
| residual on real decoded pixels, inside the slice | pod2 | **240 clips** (120 A / 120 B) | `raw/residual_final.json` |
| residual mechanism split (persistent vs scene black) | pod2 | 200 + 240 clips | `raw/zero_diag_train200.json` |
| field cost, real `obstacle.offline` agents | dev box | **400 clips, 2,409,188 agent samples** | `raw/field_cost_2026-07-27.json`, `…_decomp_…json` |
| compute delta, real `ViTEncoder` fwd+bwd | pod3 (A40) | batch 8, 12 reps | `raw/token_cost.json` |

⛔ **pod1 was not touched** (training). **pod2's builds were finished before any work landed on it**
— the 2,400-clip train build (`DONE`, 80 GB) and the 600-clip val build (12/12 shards `DONE`, 20 GB,
`val_build_pids` all exited). Nothing was killed; the waiter had already fired at 11:35:27 UTC.

Everything below uses **real per-clip intrinsics**. Nothing uses the corpus-median fallback (whose
`cy` is a rig-B value).

---

## 2. ⭐ SLICE OR REBUILD — the answer, and why it is exact

### 2.1 The identity

`cylindrical_rays` puts the boresight at the output centre, `((W-1)/2, (H-1)/2)`. For a **centred**
sub-rectangle with even margins the child's ray coordinates are *identically equal* to the parent's:

```
u_parent(c0 + j) = (c0 + j - (W-1)/2)/f = (j - (w-1)/2)/f = u_child(j)     when c0 == (W-w)/2
```

The halves cancel against the integer margin, so this is an **exact float identity**, not an
approximation. Ray map ⇒ native `(u, v)` ⇒ `grid_sample` grid ⇒ observed mask ⇒ output pixels: every
one of them is the corresponding sub-block of the parent's.

### 2.2 Verified three ways, each able to fail

| leg | what | result |
|---|---|---|
| **L1** grid identity | `cylindrical_grid(sub) == cylindrical_grid(parent)[rows, cols]` | `torch.equal` **True**, 5 real intrinsics × 4 candidate frames |
| **L2** mask identity | same for the observed mask | **True** |
| **L3** pixel identity | `cylindrical_rectify` at the sub-frame vs the sliced parent output | **True**, `max_abs_diff 0` |
| **L4** ⭐ **on the REAL built cache** | rebuild the clip **from its own mp4** at 176×624, compare with the slice of the cache's own PNG frames | **6/6 clips bit-identical**, 201 frames each, `n_pixels_differing 0` |
| **NEG** off-centre slice / changed `f_ref` | must DIFFER; `subframe_slice` must REFUSE | differ; refused ✅ |

```
[rebuild] {"rig":"B","built_shape":[201,3,256,640],"rebuild_shape":[201,3,176,624],
           "bit_identical":true,"max_abs_diff":0,"n_pixels_differing":0}
```
(3 rig-A + 3 rig-B clips; `raw/verify_val.json`.)

### 2.3 ⚠️ The precondition that makes it work — and would break it

**The cache is `codec: "png"` — lossless.** A slice of a **JPEG** cache is *not* bit-exact: re-encoding
at a new crop offset moves the 8×8 blocks. `slice_v2_cache.py` **refuses** a lossy source unless
`--allow-lossy` is passed, and records the fact in the manifest either way. The v5 wide cache being
PNG is what turns this from "cheap" into "free".

### 2.4 The cost, three ways — and the cheapest is ZERO

| path | measured cost | storage |
|---|---|---|
| **rebuild from source** | 19.4 s/clip + **~374 GB HF egress**, ~3.5 h at 8 shards | +59 GB |
| **re-emit the cache** (`slice_v2_cache.py`) | **8.82 s/clip** single process (24 clips / 211.6 s), **no egress**; embarrassingly parallel on 96 idle cores | **25.2 MB/clip** measured ⇒ **59 GB** train + **15 GB** val (vs 80 + 20 GB) — a **26 GB saving** if it replaces, +74 GB if kept alongside |
| ⭐ **slice in the LOADER** (`load_compressed(..., frame=…)`) | **+0.113 s/clip on load** (1.214 s vs 1.101 s), and every downstream tensor 33 % smaller | **0 bytes** |

Loader-slice output verified **bit-identical to the re-emitted cache** on a real clip
(`max_abs_diff 0`, shapes `[199, 9, 176, 624]`). Loading the re-emitted cache is *faster* than the
parent (0.929 s vs 1.101 s), so a re-emit is worth it only if the 26 GB saving or the load time
matters — **it is not needed to apply the fix**.

⇒ **Scheduling fact for the PI: nothing has to be rebuilt, and nothing has to be re-emitted. The
clean fix is one argument at the data-loading seam.**

---

## 3. The bands, per rig, MEASURED on 3,000 clips

`stack/scripts/rig_band_scan.py`, run on pod3 over every clip in `r0_selection.parquet`.
**0 intrinsics failures, 0 fallbacks.** Rig assignment derived from the `cy` bimodality and
**checked**: largest gap **193.35 px**, boundary `cy` **650.872**, rig A `cy` 533.95–554.20,
rig B `cy` 747.55–764.52. No clip ambiguous.

| | n | share | masked @ 256×640 (mean) | masked @ 256×640 (max) |
|---|---:|---:|---:|---:|
| **rig A** | **812** | **27.07 %** | 0.0000172 | 0.00063479 |
| **rig B** | **2,188** | **72.93 %** | **0.0889661** | 0.10521239 |

Independently reproduces the wide-build census (8.897 % rig B, 812/2,188) on a different host with a
different instrument. ✅

### 3.1 What each rig actually observes

| | rig A | rig B |
|---|---|---|
| widest symmetric **HFOV** the sensor delivers | mean **121.174°**, min **119.247°** | mean **120.492°**, min **118.958°** |
| **downward** half-field | mean **33.518°**, min 32.733° | mean **19.999°**, min **19.279°** |
| **upward** half-field | mean 33.816° | mean 46.897° |

The frame asks for **120.000° × ±22.728°**. So:

- ⭐ **The rig split is a DOWNWARD-field split.** Rig B's principal point sits ~211 px lower, leaving
  it only **20.0°** of downward field against rig A's **33.5°**. That single number is the whole
  defect.
- ⛔ **AND THE 120° REQUEST OVER-RUNS THE SENSOR HORIZONTALLY — on BOTH rigs.** **260 of 3,000 clips
  (8.67 %)** have at least one masked pixel on the frame's vertical **centre** row (238 rig B,
  22 rig A; up to 5 px wide). **This was not in the prior record**, and it is why the clean frame
  needs 624 columns and not merely fewer rows. The horizontal excursion is largest exactly at the
  centre row (as `|v|` grows, `ρ` grows faster than `r(θ)`), so it appears where it is least expected.

### 3.2 ⭐ THE BAND ITSELF — stated at 624 columns, where it is unambiguous

⚠️ **At 640 columns the "fully observed rows" statement is not clean**, because the horizontal
deficit of §3.1 breaks the row runs *in the middle* of the frame: some clips' maximal run from row 0
ends at row 87. Quoting a vertical band off the 640-wide frame therefore mixes two different failures.
Measured again at **624 columns**, where the horizontal deficit is gone and the rig split is the only
thing left (`raw/band624.json`, `code/band624.py`, every distinct sensor geometry):

| | distinct geometries | run contiguous | first row | **last fully-observed row** | rows |
|---|---:|---|---:|---|---:|
| **rig A** | 57 | **all** | 0 | **255 (min = max)** — the entire frame | **256** |
| **rig B** | 64 | **all** | 0 | **218 worst, 223 best** | **219 worst** |

⇒ **The intersection is rows `[0, 218]`.** The largest *centred* band inside it is rows `[37, 218]`
= 182 rows; rounded down to the patch size, **176 rows = rows `[40, 215]`**. That is the frame.

### 3.3 Why it is expressed as a centred rectangle

Only a **centred** sub-frame is (a) expressible as a `CanonicalFrame` — which pins the boresight at
the output centre — and (b) a pure pixel slice. Rig B's shortfall is at the **bottom**, so a centred
band also gives up the same number of rows at the top (sky). That is the price of expressibility, and
§5 prices it: it is small.

**Worst-case masked fraction over every clip of the rig** (`raw/band_full_3000.json`):

| rows \ cols | 640 (120.000°) | 624 (117.000°) | 608 (114°) | 592 (111°) | 576 (108.000°) |
|---:|---|---|---|---|---|
| **256** (45.456°) | A 0.000635 / B 0.105212 | A 0 / B 0.103879 | A 0 / B 0.102822 | A 0 / B 0.101800 | A 0 / B 0.100816 |
| **192** (34.881°) | A 0.000846 / B 0.003581 | A 0 / B 0.002370 | A 0 / B 0.001765 | A 0 / B 0.001249 | A 0 / B 0.000832 |
| **176** (32.131°) | A 0.000923 / B 0.002184 | **A 0 / B 0** | **A 0 / B 0** | **A 0 / B 0** | **A 0 / B 0** |
| **160** (29.341°) | A 0.001016 / B 0.002402 | **0 / 0** | **0 / 0** | **0 / 0** | **0 / 0** |
| **128** (23.658°) | A 0.001270 / B 0.003003 | **0 / 0** | **0 / 0** | **0 / 0** | **0 / 0** |

⇒ **`176 × 624` is the largest centred sub-frame of the built frame that BOTH rigs fully observe.**
Rows `[40:216]`, cols `[8:632]` of the cache as built.

⚠️ **Width 640 never reaches zero at any height, on either rig.** Height alone is not the fix.

---

## 4. The two candidates, and the token-grid constraint

| | **256×640** (built) | ⭐ **176×624** | **128×576** |
|---|---|---|---|
| HFOV × VFOV | 120.000° × 45.456° | **117.000° × 32.131°** | 108.000° × 23.658° |
| slice of the built frame | — | rows `[40:216]`, cols `[8:632]` | rows `[64:192]`, cols `[32:608]` |
| tokens @ patch 16 | **640** (16 × 40) | **429** (11 × 39) | **288** (8 × 36) |
| readout tiles exactly (dims % 64) | **yes** | **no** (11 % 4 = 3, 39 % 4 = 3) | **yes** |
| `state_dim` | 2048 | **2048** | **2048** |
| **fwd+bwd, real `ViTEncoder`, A40, B = 8** | 0.27713 s | **0.17927 s (0.647×)** | 0.12160 s (0.439×) |
| peak activation memory | 4580.6 MiB | **3319.3 MiB (0.725×)** | 2493.4 MiB (0.544×) |
| masked on both rigs | A 0.0000172 / B 0.0889661 | **0.0000000 / 0.0000000** | **0.0000000 / 0.0000000** |

**On the "width must be a multiple of 64" rule.** It is not a constant — `geometry.tiling_report`
derives it as `patch_size × readout.grid = 16 × 4 = 64`, and the same rule binds the **height**.
`176 × 624` satisfies neither; measured consequence (not inferred): the readout falls back to
`AdaptiveAvgPool2d` with **uneven bins** (11 token rows → 3/3/3/2), `state_dim` **stays 2048**, and the
encoder runs — 0.17927 s, no shape failure. **`128 × 576` is the only zero-mask frame that tiles
exactly**, and it costs 148 more tokens' worth of field to get there.

⇒ **The token count is 429 (from 640, −33.0 %) and the measured compute delta is 0.647× per
forward+backward (−35.3 %).** Attention being quadratic is why the speedup slightly exceeds the token
ratio.

---

## 5. ⚠️ WHAT IT COSTS IN FIELD — and the result is not what the framing assumed

`code/field_cost.py`: every `obstacle.offline` cuboid, all 8 corners, projected into the **cylindrical
output frame** with each clip's **own** intrinsics *and* extrinsics. **400 clips (160 A / 240 B),
2,409,188 agent samples, 972,242 of them visible in the parent frame.** An agent counts as visible if
any corner lands inside **and** the sensor actually observed that pixel (so rig B is not billed for
content its mask already removed).

### 5.1 ⭐ The decomposition — the whole price is the WIDTH

| frame | what it cuts | agent samples lost |
|---|---|---:|
| **176 × 640** | **rows only** (256 → 176; VFOV 45.456° → 32.131°) | **0.0165 %** |
| **256 × 624** | **columns only** (640 → 624; HFOV 120° → 117°) | **1.1770 %** |
| **176 × 624** | both | **1.1948 %** |
| 160 × 592 | both, one step further | 3.4592 % |
| 128 × 576 | strict tiling | 4.6868 % |

⭐ **Cutting 31.25 % of the rows costs 0.017 % of decision-relevant content. Cutting 2.5 % of the
columns costs 1.18 %.** Agents cluster in a narrow band about the horizon and spread wide in azimuth,
so the vertical field is cheap and the lateral field is not. **The vertical band — the thing this
whole fix is about — is very nearly free.** The bill is for the 3° of width that the sensor could
never deliver in the first place.

### 5.2 Stratified — where the vertical loss actually lands

`176 × 624`, by range from the ego (n visible in parent / n lost):

| range | 0–5 m | 5–10 m | 10–20 m | 20–40 m | 40–80 m |
|---|---:|---:|---:|---:|---:|
| **fraction lost** | **5.443 %** | 2.244 % | 1.618 % | 1.430 % | 1.224 % |
| (samples in bin) | 5,548 | 24,375 | 82,656 | 210,444 | 365,196 |
| same at 128×576 | **20.963 %** | 7.877 % | 6.393 % | 5.850 % | 4.809 % |

The near field is where a vertical cut bites, exactly as expected — but the 0–5 m bin is **0.57 % of
all samples**, so it moves the total by 0.03 pp. By class: automobile 1.194 %, heavy truck 1.284 % —
no class is singled out.

Symmetry check: **rig A 1.145 % / rig B 1.243 %**. The fix does not bill one rig for the other's
defect.

### 5.3 The near-field ground — the honest vertical price

Cylindrical elevation is constant along a row, so a ground point at horizontal distance `d` from the
camera sits at `y_n = h_cam/d` exactly. Measured per clip from the **real extrinsics** (so per-clip
camera height **and** mount pitch are in it, not a constant — class C28; measured heights: rig A
1.2535–1.4749 m, rig B 1.2268–1.6622 m):

| frame | nearest visible ground, rig A | rig B |
|---|---:|---:|
| 256×640, frame bounds only | 5.081 m | 4.941 m |
| **256×640, what the sensor really shows** | **5.081 m** | **5.376 m** ⬅ 0.44 m of it was masked |
| **176×624** | **6.472 m** | **6.362 m** |
| 128×576 | 8.082 m | 8.047 m |

(median, measured ahead of the **rig origin**; the camera sits ~1.8 m forward of it.)

⇒ **The real ground price is +1.391 m (rig A) and +0.986 m (rig B)** — rig B pays less because part of
that near-field was already black for it. ⭐ **And note the parent frame is itself rig-inconsistent in
near-field ground (5.081 vs 5.376 m); the fix removes that too** (6.472 vs 6.362 m, and the residual
0.11 m is per-clip camera height, not rig geometry).

**Only 0.013 % of parent-visible agent samples were inside rig B's masked region** — the mask was
hiding *road surface*, not agents.

### 5.4 The trade, stated for the decision

- **176 × 624** — 1.195 % of agents, +1.4 m / +1.0 m of near-field ground, −33 % tokens, −35 % compute,
  **zero mask on both rigs**, uneven readout pooling.
- **128 × 576** — 4.687 % of agents (**20.96 %** inside 5 m), +3.0 m of ground, −55 % tokens,
  **zero mask on both rigs**, exact readout tiling.
- **176 × 640** — 0.017 % of agents, but **does not meet the criterion** (A 0.000923 / B 0.002184).
  It collapses rig B's mask by **~950×** and brings the two rigs from **1 : 5,172** to **1 : 3.7**.
  It is on the table only if the PI decides the last factor of 3.7 is not worth 1.18 pp of agents.

**Recommendation: `176 × 624`.** It is the only option that meets the pre-registered criterion at a
price the measurement shows to be small, and it is a pure slice.

---

## 6. The implementation

Default unchanged everywhere. `CANONICAL_256` still `is_canonical`; `geometry_params` still returns
`{}` for it, so `physicalai-train-e438721ae894` still hashes to exactly what it hashed.

### `stack/tanitad/data/calib.py`

| added | what it does |
|---|---|
| `centred_subframe(frame, h, w)` | the sub-frame; **refuses odd margins** (a half-pixel boresight shift) and refuses growth |
| `subframe_slice(parent, sub)` | the `(rows, cols)` slice; **refuses** any changed `f_ref`/projection — those are resamples |
| `observed_report(intr, frame)` | masked fraction + the widest field the sensor *can* deliver per axis, from the ray map (no decode) |
| `ftheta_crop_pad_report(...)` | the **deployed crop's** fabricated-pixel fractions — so the pre-cylindrical defect and the cylindrical one are on ONE instrument |
| `assert_fully_observed(intrs, frame)` → `RigAsymmetry` | ⭐ **the C26 guard**, zero-tolerance, over a **population** |
| `largest_fully_observed_subframe(...)` | the search; returns the winner **and the whole trade table** |
| `PHYSICALAI_WIDE120_256x640`, `PHYSICALAI_RIG_CLEAN_176x624`, `PHYSICALAI_RIG_CLEAN_128x576` | the frames, with their provenance in the docstring |

### `stack/scripts/`

- **`rig_band_scan.py`** — the 2-D census that produced §3. Runs anywhere the calibration lives; no
  decode; no clip UUID in the output.
- **`slice_v2_cache.py`** — re-emits a built cache at a centred sub-frame. Refuses a lossy source.
  Writes a `_geometry.json` carrying `sliced_from` (parent frame + exact row/col slice),
  `bit_exact_slice`, the **per-rig observability** declaration, and a **`pixel_digest_sha256`** map.
- **`v2_compressed.py`** —
  - `load_compressed(path, frame=…)` ⭐ **the zero-cost fix**: slices after the decode that already
    runs. `None` (default) is byte-identical to today.
  - `_rig_observability()` folded into the pre-decode geometry check.
  - `--require-fully-observed` (**off by default**, so no running build changes behaviour).

### The declaration — and the gap it closes

⚠️ **The v5 cache's own `_geometry.json` says `"observed_frac": 1.0`.** That is true of the one clip
it probed and **false of the corpus** (mean 0.911; rig B 0.910). The build check sampled a single
clip that happened to be rig A. It now samples a **population** and reports **both rigs separately**:

```json
"rig_observability": {"A": {"n": 8,  "masked_frac_max": 0.0,
                            "masked_frac_max_parent": 0.0},
                      "B": {"n": 16, "masked_frac_max": 0.0,
                            "masked_frac_max_parent": 0.10521239042282104},
                      "fully_observed_by_all_sampled": true}
```
(real output of `slice_v2_cache.py` on the val cache.)

⚠️ **Nothing hashes pixels**, as the brief says — so the declaration is what a consumer can check, and
it must be checkable. Two things now make it so: the per-rig `masked_frac_max` is **recomputable from
the clip ids and the calibration alone**, and `pixel_digest_sha256` gives a consumer actual bytes to
compare.

### Tests — `stack/tests/test_rig_clean_fix.py`, 30 tests

Fixtures are **real five-coefficient per-clip polynomials** for the worst clip of each rig, selected
by `code/pick_extremes.py` over the 121 distinct sensor geometries; their masked fractions at 256×640
**reproduce the census maxima exactly** (0.00063479 / 0.10521239), which is what makes them
admissible rather than plausible-looking.

**Every "it is zero" assertion is paired with a case where the same call is NOT zero** (class C13):

- `test_the_guard_REFUSES_the_frame_the_corpus_was_actually_built_at` — the guard raises on
  256×640, on the population **and on rig A alone**, and refuses an **empty** population.
- `test_height_alone_is_NOT_enough_at_640_columns` — 176×640 is non-zero on both rigs.
- `test_the_slice_claim_can_FAIL_off_centre_and_at_a_changed_focal` — off-centre and resampled
  frames differ; `subframe_slice` refuses them; odd margins refused.

---

## 7. ⛔ THE RESIDUAL — 0.0000 where it was asked for, NOT zero where it was not

**n = 240 real clips of the built train cache (120 rig A / 120 rig B), decoded, sliced to 176×624.**
Raw: `raw/residual_final.json`.

| quantity, inside the 176×624 slice | rig A | rig B |
|---|---:|---:|
| **ray-map masked / replicate-padded fraction — mean** | **0.0000000000** | **0.0000000000** |
| **— MAX over 120 clips** | **0.0000000000** | **0.0000000000** |
| persistent black beyond the ray map (mean / max) | **0.0000000000** / **0.0000000000** | 0.0002711156 / 0.0088869464 |
| transient (scene) black (mean / max) | 0.0000834375 / 0.0078876875 | 0.0076645026 / 0.0616805963 |
| **all-zero pixels, total (mean)** | **0.0000834** | **0.0079316** |

*(for reference, the same clips at the parent 256×640: ray-map masked **0.0000159 (A) / 0.0900333 (B)**.)*

**To 4 dp: the pad/mask fraction is 0.0000 on rig A and 0.0000 on rig B.** The pre-registered
criterion is met, at the maximum and not only the mean.

### ⛔ But a rig-correlated residual survives, and it is a different mechanism

Real all-zero pixels are **0.0001 (A) vs 0.0079 (B)** — a **95×** asymmetry. Decomposed
(`code/zero_residual_diag.py`, persistent = zero in ≥ 95 % of a clip's frames):

- **~97 % of it is TRANSIENT** — night, tunnels, dark vehicles. It moves between frames. **It is real
  image content, not a preprocessing artefact, and no choice of frame removes it.** It is
  rig-correlated because the two rigs recorded different scenes, which is a *dataset-composition*
  confound, not a *geometry* one.
- **~3 % is PERSISTENT and beyond the ray map** — 0.0000000 on rig A, 0.0002711 on rig B. Localised
  either at the extreme columns (θ 56–61°: the lens image circle, which the rectangle-based mask does
  not model) or just below the mask boundary (θ 17–27°).

⚠️ **SELF-CORRECTION.** At n = 24 this diagnostic suggested `160 × 592` would be "free of persistent
black". At **n = 240** the union of persistent-black regions covers **5.57 %** of the rig-B frame and
**no useful centred frame is free of it** (`largest_centred_free_of_persistent_black` degenerates to
16 × 176). **The n = 24 result was over-fitted to scene content. You cannot crop your way out of
night scenes** — and I would have shipped that recommendation on the smaller sample.

### What this means for C26

**C26 is removed**: the fabrication and its rig-correlated mask are gone, exactly, on both rigs. What
remains is a **new, separate finding** that deserves its own class: *the two rigs sampled
systematically different imaging conditions, so a rig-correlated luminance signal exists in the
content itself.* That is a **corpus-balance** question (which is checkable and possibly fixable by
sampling), not a preprocessing one, and it is **not** in scope here — but it must be on the record
before anyone declares the rig confound closed.

---

## 8. Pre-registration and scoring

Written before the numbers, in the brief; scored here.

| rule | criterion | value that would FAIL it | result |
|---|---|---|---|
| **CONFIRM** | a band both rigs fully observe, acceptable field cost, mask **0.0000** on both | any `masked_frac > 0` at the chosen frame on **any** clip of **either** rig | ✅ **176×624: max 0.0000000000 on 120 A + 120 B; agent cost 1.195 %** |
| **PARTIAL** | such a band exists but costs more than it is worth | agent loss large enough to dominate the defect — e.g. the >4.7 % (20.96 % inside 5 m) of `128×576` | not triggered at 176×624; **`128×576` would be a PARTIAL** and is reported as the alternative |
| **REFUTE** | no such band exists | the zero-mask set empty — which it **is** at width 640 at **every** height | not triggered; the search found 16 zero-mask frames |

**Can each rule return a failing value? Demonstrated, not asserted.**
`assert_fully_observed` raises `RigAsymmetry` on the frame the corpus is actually built at, on the
population *and on rig A alone*, and refuses an empty population (a guard with nothing to check
cannot fail, so it is treated as an error). The scan's zero-mask set **is empty for every 640-column
candidate** — the REFUTE branch is reachable and was reached for that column count. Pinned in
`stack/tests/test_rig_clean_fix.py`.

---

## 9. 🔴 ESCALATIONS — decisions, not documentation

1. **⭐ THE FIX IS A LOADER ARGUMENT. Someone must pass it.** `load_compressed(path, frame=…)` exists
   and is verified bit-exact, but **no trainer passes it today**. Until the v5 trainer does, v5 will
   train on the rig-asymmetric 256×640 frames. **This is the whole deliverable and it is one line in
   the v5 dataset path.** *(This is the failure mode of "please merge" in a README — hence it is
   here, in the headline, not there.)*
2. **`_geometry.json` in BOTH shipped v5 caches declares `"observed_frac": 1.0`.** It is a one-clip
   sample and it is wrong about the corpus (0.911). The builder is fixed; **the two manifests already
   on pod2 are not** — they should be re-stamped, or a consumer will read 1.0 and believe it.
3. **The PI must pick 176×624 or 128×576** (§5.4). Both meet the criterion. 176×624 keeps 3.5× more
   near-field content; 128×576 buys exact readout pooling for 3.5 pp more agent loss.
4. **The 120° request was never deliverable.** 8.67 % of clips cannot supply 120° horizontally
   (pooled min **118.958°**). Any future wide build should ask for **≤ 117°**, and
   `--require-fully-observed` now makes that abort instead of silently masking.
5. **New confound for the record (§7):** a rig-correlated *scene*-luminance asymmetry (0.0001 vs
   0.0079 black-pixel fraction) that geometry cannot remove. Owner needed; it is a corpus-balance
   question.

---

## 10. Test counts

| suite | before (brief) | after | new skips |
|---|---|---|---|
| `stack/` | 1379 passed, 12 skipped | **1415 passed, 12 skipped** | **0** |
| `taniteval/` | 565 | **606 passed** | **0** |

*(Both suites were run on the working tree as found, which already carried sibling stream work; the
delta above 30 new tests of mine is theirs. No default that any running arm reads was changed —
`CANONICAL_256`, `geometry_params`, `load_compressed(path)` with no `frame`, and the v2 builder's
behaviour without `--require-fully-observed` are all byte-identical.)*

---

## 11. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `RIG_CLEAN_FIX.md` (this) | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-28-rig-clean-fix/` | no |
| `raw/band_full_3000.json` — the n=3,000 census | repo (as above) + `pod3:/workspace/rigfix/` | no |
| `raw/verify_val.json` — slice-vs-rebuild on real frames | repo + `pod2:/workspace/rigfix/` | no |
| `raw/residual_final.json` — the n=240 residual | repo + `pod2:/workspace/rigfix/` | no |
| `raw/zero_diag_train200.json` — residual mechanism | repo + `pod2:/workspace/rigfix/` | no |
| `raw/field_cost_2026-07-27.json`, `raw/field_cost_decomp_2026-07-27.json` | repo | **repo only** |
| `raw/token_cost.json` — measured compute delta | repo + `pod3:/workspace/rigfix/` | no |
| `raw/extremes.json` — the real intrinsics behind the test fixtures | repo + `pod3:/workspace/rigfix/` | no |
| `raw/band624.json` — the vertical band at 624 columns | repo + `pod3:/workspace/rigfix/` | no |
| `code/field_cost.py`, `code/zero_residual_diag.py`, `code/verify_slice_on_built.py`, `code/slice_equiv.py`, `code/pick_extremes.py`, `code/token_cost.py`, `code/band624.py`, `code/rig_band_scan.py` | repo | **repo only** (copies on pods) |
| **`stack/tanitad/data/calib.py`** — the fix | `repo:stack/` (staged) | no |
| **`stack/scripts/rig_band_scan.py`** | `repo:stack/` (staged) | no |
| **`stack/scripts/slice_v2_cache.py`** | `repo:stack/` (staged) | no |
| **`stack/scripts/v2_compressed.py`** — loader slice + declaration | `repo:stack/` (staged) | no |
| **`stack/tests/test_rig_clean_fix.py`** — 30 tests | `repo:stack/` (staged) | no |
| `pod2:/workspace/data/rigclean-val-176x624-SMOKE/` — 24-clip proof output | pod2 | **pod only** (disposable; regenerable in 3.5 min) |
| `pod3:/workspace/rigfix/`, `pod2:/workspace/rigfix/` — shipped HEAD stack + tools | pods | no (all from repo) |

### ⚠️ SHARED-INDEX DISCLOSURE — my work was committed AND PUSHED by another stream

**I did not run `git commit` or `git push`, and I switched no branch.** I staged into the working
tree as the standard requires. While I was working, commit **`d5d5afb`** (`sayoood`,
2026-07-27 15:32:29 +0200) — whose message is *"the closed-loop metric was BLIND TO OVER-TRAVEL…"* —
**swept the entire index**, which contained my in-progress deliverables, and the branch was pushed to
`origin/agent/benchmarks-eval-20260721`.

**Verified byte-for-byte**: every file in §11 is in `HEAD` with a blob hash identical to my working
tree, and `git status --short` is clean apart from a pre-existing `.claude/settings.local.json` and a
stray `4}` that is not mine. **Nothing is stranded and nothing is lost** — but this stream's work now
lives under another stream's commit message, on the remote.

⇒ **This is the third occurrence of the CLAUDE.md §Git-hygiene hazard** (`git commit` commits the
ENTIRE INDEX, not what you just `git add`ed) — after `60265d3` and `3d41bd0`. I am **not** attempting
to repair it: rewriting a pushed commit is strictly more dangerous than the mislabelling.
**Recommended (a human decision):** leave the history and record the attribution here, which this
paragraph does.

**No branch switched. No push initiated by this stream.**

---

## 12. Limitations, stated plainly

- The residual and slice-equivalence measurements use the **val** and **train** caches on pod2; the
  agent field cost uses the **26 obstacle.offline chunks** present on the dev box (400 clips, 160 A /
  240 B) — a subset of the corpus, though it reproduces the corpus rig split to within 5 pp.
- `observed_frac` is a **rectangle** test on the native sensor. It does not model the lens image
  circle, which is why §7's persistent residual exists at all. A circle-aware mask would tighten the
  guarantee; it was not built here.
- The **agent** field cost counts `obstacle.offline` cuboids only. Lane markings, kerbs, and the road
  surface itself are not in it — the ground-distance number in §5.3 is the proxy for those.
- **No ADE was measured.** Whether 117° × 32.1° trains better or worse than 120° × 45.5° is an
  experiment, not a geometry fact. This work makes the clean frame free to try; it does not claim it
  wins.
