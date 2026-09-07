# WP-A — how much BEV localisation the 16×40 token grid carries that a pooled readout does not

**Date** 2026-09-07 · **Owner** Architecture & Inference · **Branch** `agent/arch-inf-20260803`
**Evidence class** MEASURED (ours), artifact named per number, unless a line says otherwise.
**Compute** dev-box **RTX 4060 only** (~50 GPU-minutes total across all panels). ⛔ Zero A40 — refcv5-v2 is training there.
⛔ Nothing pulled from HuggingFace: every input was already on local disk.

⛔ **NO EVAL TIER, AND THAT IS NOT AN OMISSION.** No model here produces a trajectory. This is a
**representation probe**, so a T0/T1 stamp and a four-family table would be category errors
(`EVAL_DOCTRINE.md` binds *capability* claims). Nothing below is a driving-performance number.

---

## 0. The answer, in three lines

1. ⭐ **Under a PERFECT front-end the readout costs a lot: the 16×40 grid carries ~3.0× the BEV
   agent localisation of a 4×4 readout and ~2.3× that of the deployed 4×8, and halves the lateral
   error (1.37 m vs 2.67 m at 15–30 m).** The azimuth axis costs ~1.65× what the elevation axis does.
2. ⛔ **On the REAL v7-tiny trunk the pool destroys addressable content the same way (fit AP
   0.2049 at 16×40 → 0.0581 at 1×1) — but NONE of it generalises across episodes** (test AP
   0.027–0.034 against a marginal control at 0.0325, raw pixels included). So today the readout is
   a **ceiling, not the binding constraint**.
3. ⇒ **Waypoint-indexed attention built now would index into a map that does not contain agents.
   The prerequisite is WP-D (give the map a reason to be spatially organised), not WP-A's readout
   exposure — and for a REF-C-shaped refcv5c the "4-column blocker" does not apply at all, because
   refcv5's feature map is 8×20.**

---

## 1. ⚠️ Two corrections to the premise — both MEASURED, both change the decision

### 1.1 The live v7-tiny readout is **4×8**, not 4×4

Read from the checkpoint's own module and weights (`C:\Users\Admin\k8-pull\ckpt.pt`, run
`v6-staged-S-W` = `k8clip05p30k`, **step 30000**), not from prose:

```
SpatialGridReadout  token_h/w (16, 40)   grid/grid_w (4, 8)   exact_pool True
                    pool AvgPool2d(kernel_size=(4, 5), stride=(4, 5))
                    readout.proj.weight (64, 128)      d_op 2048
config['args']:     --readout-grid 4 --readout-grid-w 8 --readout-dim 64
```

⇒ **8 azimuth columns = 15.0°/column**, not 4 columns / 30°. This is the `rdw8` geometry
**E-DEC-2 measured** (`grid 4 × grid_w 8 × d_readout 64 = 2048`).

The **4×4 / 30°** figure is the **v1 / v4 / v5f flagship** line: `dynamics_encoder.py:233-234`
(`grid 4`, `d_readout 128`) constructs `SpatialGridReadout` **with no `grid_w`**
(`dynamics_encoder.py:146-148`) ⇒ square 4×4, and `flagship_v15.py:97-98` pins
`state_dim 2048 # (4*4*128)`. Both readouts land on **2048** — the geometry firewall working, and
exactly why the two are easy to confuse.

⇒ **Name the line, never write a bare readout shape.** Same family as the `df` / `step_s` /
cylindrical-FOV traps: a true number quoted outside its scope.

### 1.2 ⛔ refcv5 has **no `SpatialGridReadout` at all** — the stated blocker does not apply to it

`REFCV5C_DESIGN.md` §3 says *"Ours is pooled to 4 readout columns"* and concludes *"deformable
attention indexed into a 4-column map is indexing into almost nothing."* That is a v6/v7 fact
quoted into a REF-C design. MEASURED:

| | value | source |
|---|---|---|
| refcv5-v2's argv | `--image-hw 256 640` | the run's own `config.json`, quoted in `…/2026-09-07-refcv5-v2-compose/LAUNCH_RECEIPT.md` §1 |
| how that reaches the encoder | `image_width = None if w == h else w` → **640** | `stack/scripts/refc_v3_train.py:216-221` |
| REF-C encoder geometry | ResNet **stride 32** ⇒ `grid_shape = (h//32, w//32)` = **(8, 20)** | `stack/tanitad/refs/refc.py:294-296, 333-337` |
| what the decoder already does with it | `kv = self.feat_proj(fmap.flatten(2).transpose(1, 2))` → **P = 160 tokens** cross-attended | `stack/tanitad/refs/refc.py:2112` |

⇒ **refcv5's map has 20 azimuth columns = 6.0°/column — 5× finer than the stated blocker — and the
anchor decoder already cross-attends it.** What refcv5c is missing is therefore **the indexing**,
not the resolution. WP-A is **not** a prerequisite for WP-B on a REF-C-shaped arm; it **is** one if
refcv5c were built on the v6/v7 trunk.

---

## 2. The geometry, derived twice by independent routes

⭐ A cross-check must be derived independently of the value it checks. Route 1 is
`bev_raster.fov_census`; route 2 is analytic from the projection. `code/s3_geometry.py` ·
`raw/geometry.json`.

* **Projection.** The cylindrical column is linear in azimuth: `2·deg((W/2)/f_ref)` = **120.000°**
  at `W 640`, `f_ref 305.5774907364391` — matching the rig's own name `camera_front_wide_120fov`.
  The pinhole formula gives **92.641°** and is wrong here. ⚠️ Scoped to **our** 256×640 cylindrical
  corpus; **not** a property of BEV lifting in general.
* **`fov_mask`** (half-angle 60°): **7 090 / 7 680 in field = 92.32 %** — reproduces the brief.
* **`fov_census` at 4 columns: 593 · 2 952 · 2 952 · 593 = 7.72 / 38.44 / 38.44 / 7.72 %** —
  reproduces the brief exactly; **two bins resolve 76.88 %** of the grid.

| readout columns | °/col | one column's lateral **full width** @30 m | @50 m | largest column's share of the grid |
|---|---|---|---|---|
| **40** — the 16×40 token grid | 3.00 | **1.571 m** | 2.619 m | ~1 % |
| **20** — refcv5's ResNet map | 6.00 | **3.144 m** | 5.241 m | 9.54 % |
| **8** — v7-tiny deployed (`rdw8`) | 15.00 | **7.899 m** | 13.165 m | 25.10 % |
| **4** — v1/v4/v5f flagship | 30.00 | **16.077 m** (±8.038 m) | 26.795 m | **38.44 %** |
| 2 | 60.00 | 34.641 m | 57.735 m | — |

The brief's *"±8 m lateral at 30 m"* is confirmed: it is the **half**-width of a 30° bin at 30 m.

---

## 3. The corpus, the trunk, the split

| | |
|---|---|
| pixels | `C:\Users\Admin\tanitad-data\refav1-eval141\eps\*.v2ep.pt` — **141** clips, `256×640` **cylindrical**, `f_ref 305.5775`, codec **png**, `n_stack 3` ⇒ 9 channels |
| GT agents | `C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz` — the B1 EVAL `obstacle.offline` join: **139 clips · 26 394 labelled frames · 905 512 boxes**, md5 `3ddb42ecbd3926066795a94587af2aed`. Its own alignment proof reads max \|Δspeed\| **1.47e-4 m/s** against a 1e-3 tolerance |
| target | `bev_raster.rasterize` on `GRID_DEFAULT` = **[120, 64]**, 60 m × ±16 m, 0.5 m cells |
| trunk | `k8-pull/ckpt.pt`, v6-staged `S-W` (`k8clip05p30k`), step **30000**; encoder **0.97 M params** (3-layer ViT, d 128, patch 16) — a *planning* trunk, never trained to detect |
| rows | **N = 13 223** (every 2nd labelled frame). Split **episode-disjoint**: fit 84 clips / 7 842 rows · inner-val 25 / 2 407 · test 30 / 2 974 |
| occupancy | **1.7402 %** of grid cells over the bank; **1.6198 %** on test ⇒ ⛔ **an all-zero predictor scores 98.38 % accuracy** — which is why not one headline number below is an accuracy |

⛔ Every λ and every operating threshold was fitted on **fit / inner-val only**; nothing was
selected on test. *(2026-08-22 precedent: λ chosen on test picks maximal regularisation, collapses
the ridge to the constant predictor, and every arm then reads exactly the no-information value.)*

---

## 4. Panel 1 — the REAL trunk carries no transferable localisation at **any** readout resolution

⭐ **This closes an open NEXT in the register.** `E-DEC-29` ends: *"run this same probe on the TOKEN
grid. If the fine per-column structure is present in the tokens and absent from the readout cells,
the deficit is the READOUT and the objective, not the encoder."* **It is present in the tokens as MEMORISABLE structure and absent from TRANSFER** — which points the deficit upstream of the readout, at the objective and the corpus, not at the pooling.

### 4.1 Linear (exact dual ridge, λ on inner-val) — `raw/bearing_k8.json`

Score = `R² vs constant` (constant = the fit-split mean) ⇒ **the constant control reads exactly
0.0000 by construction**; negative = worse than knowing nothing.

| arm | d | `y_cent_all` | `y_cent_15-30` | `y_near` | `x_near` | ⭐ `n_occ` (NON-spatial) |
|---|---|---|---|---|---|---|
| tok 16×40 | 81 920 | **−0.113** | −0.107 | −0.100 | +0.011 | +0.071 |
| tok 8×20 | 20 480 | −0.045 | −0.037 | −0.041 | +0.008 | +0.063 |
| tok 4×8 | 4 096 | −0.060 | −0.010 | −0.012 | −0.033 | −0.083 |
| tok 4×4 | 2 048 | −0.138 | −0.005 | −0.006 | −0.004 | −0.017 |
| tok 1×1 | 128 | −0.131 | −0.035 | −0.090 | −0.039 | +0.058 |
| **pix 16×40** (floor) | 5 760 | −0.073 | −0.122 | −0.028 | −0.115 | **+0.099** |
| **constant** (control) | — | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| **shuffled** (control) | 81 920 | −0.007 | — | — | — | −0.004 |

⇒ **Every lateral target is NEGATIVE on every arm, raw pixels included.** The only positive is the
**non-spatial** `n_occ`, and the **raw-pixel floor wins it** — the trunk adds nothing over pixels
there. ⭐ This **independently replicates E-DEC-2's environment half** (*"EVERY arm is NEGATIVE on
both environment targets and the CONSTANT control beats all of them"*) on a different corpus, a
different probe and different targets. ⚠️ A negative from a **linear** probe bounds *linear*
decodability only — which is why §4.2 exists.

### 4.2 Nonlinear, azimuth-indexed head — `raw/indexed_k8.json`, `code/s5_indexed.py`

Identical head (**82.6 k params**, 4 000 steps, same seed) for every arm; the BEV cell's own
geometry addresses the feature map — DiffusionDrive's coupling (1) in miniature. ⭐ **Report BOTH
columns: the fit column measures ADDRESSABLE CONTENT, the test column measures what GENERALISES.**

| arm | az cols | **AP (fit)** | **AP (test)** | IoU (test) |
|---|---|---|---|---|
| tok 16×40 | 40 | **0.2049** | 0.0295 | 0.0320 |
| tok 8×20 | 20 | 0.1743 | 0.0330 | 0.0354 |
| tok 4×8 | 8 | 0.1304 | 0.0318 | 0.0349 |
| tok 4×4 | 4 | 0.1445 | 0.0327 | 0.0343 |
| tok 4×2 | 2 | 0.1386 | 0.0332 | 0.0352 |
| tok 1×1 | 1 | **0.0581** | 0.0335 | 0.0348 |
| pix 16×40 (floor) | 40 | 0.1760 | 0.0270 | 0.0286 |
| pix 8×20 (floor) | 20 | 0.1565 | 0.0301 | 0.0346 |
| pix 4×4 (floor) | 4 | 0.1262 | 0.0266 | 0.0272 |
| **pos_only** — marginal control | — | **0.0359** | **0.0325** | 0.0335 |
| **shuffled** — control | 40 | **0.0361** | **0.0325** | 0.0341 |
| **all-zero** — control | — | — | **0.016256** | **0.000** |
| | | | *= base rate **0.016124*** ✓ | *F1 **0.000*** |

⭐⭐ **THE FIT COLUMN REPRODUCES THE POOLING LADDER ON THE REAL TRUNK:** 16×40 **0.2049**
→ 8×20 0.1743 → the 4-row rungs ~0.13–0.14 → 1×1 **0.0581**, which is essentially the
marginal control's 0.0359. ⇒ **the pool really does destroy addressable content here too** —
5.7× the marginal at 16×40, 1.6× at 1×1 — and the head's parameter count is identical in
every arm, so the only thing that changed is **how many distinct feature vectors exist to key on**.

⛔ **AND NONE OF IT GENERALISES.** On test every feature arm sits at **0.027–0.034**, i.e. at or
**below** the marginal control's **0.0325**; the pixel arms are strictly below it. ⇒ **the
addressable content the encoder carries is EPISODE-SPECIFIC, not agent-generic.**

⭐ **The two controls are what make that readable, and both land on their known values:**
`shuffled` (features taken from a random other frame) reads **fit 0.0361 / test 0.0325** —
indistinguishable from `pos_only`'s **0.0359 / 0.0325** to three decimals, so the fit ladder above is
genuinely feature-driven and not an artefact of the head; and `all-zero` reads **AP 0.016256 against
a test base rate of 0.016124** with **IoU and F1 exactly 0.000**.

⇒ **At this trunk, on 84 training episodes, the pooling ladder cannot be RANKED ON TRANSFER —
there is nothing at the top of it that survives an episode change.** §5 removes the encoder from
the question entirely.

⚠️ Read this as *"a 0.97 M planning-trained encoder shows no TRANSFERABLE agent localisation on
84 training episodes under this probe"* — not as "the encoder is bad", and not as a statement about
what a **trained** BEV head could do (that is WP-D). ⚠️ And note the fit ladder mixes information
with degrees of freedom: more distinct addresses also means more capacity to memorise. The
`shuffled` control bounds that — it has the same address count and cannot memorise — but the fit
column is an optimistic measure of addressable content, not a clean one.

---

## 5. Panel 2 — the ORACLE readout ceiling: the WP-A answer

⛔⛔ **ORACLE ARM. The input feature map is BUILT FROM THE TARGET.** It is not a perception result
and must never be quoted as one. It answers one question only: *if the encoder were perfect, how
much localisation does each readout geometry still deliver?* — which is what WP-A asked, and the
only form of the question a weak trunk cannot destroy.

**Construction** (`code/s6_oracle.py`). GT BEV occupancy re-projected into the encoder's own token
grid: `col = (f_ref·atan2(y,x) + W/2)/16` (cylindrical ⇒ linear in azimuth) and
`row = (v0 + f_ref·h_cam/range)/16` (ground-plane inverse depth; `v0 = 128 px`, `h_cam = 1.5 m` —
⚠️ a stated **model** of the vertical axis, not a calibration). ⭐ **Position lives ONLY in the
address**: the map's values are presence / count, never a coordinate. Painting azimuth as a *value*
would let even a 1×1 map recover position, and would measure nothing.

Every arm is `avg_pool(k) → nearest-upsample back to 16×40 → the SAME head` (identical
architecture, **82.6 k params**, 3 000 steps, same optimiser, same seed) ⇒ **the only difference
between arms is what the pool destroyed, at matched capacity.** ⭐ This is precisely the
dimensionality confound `E-DEC-25` flagged in its own row ladder (*"dimensionality grows with
[rows] … the decline may be an SNR artefact"*) — removed by construction here.

### 5.1 The ladder — mean of 3 training seeds (`raw/oracle_s0.json`, `_t1`, `_t2`)

| arm | rows × az | °/col | **AP** | IoU | lat. MAE 15–30 m | lat. MAE 30–45 m |
|---|---|---|---|---|---|---|
| **orc 16×40** — full token grid | 16 × 40 | 3.0 | **0.4713** | 0.313 | **1.372 m** | 1.717 m |
| orc 8×20 — **refcv5's map** | 8 × 20 | 6.0 | 0.3374 | 0.227 | 1.814 m | 2.571 m |
| orc 4×40 — rows pooled ONLY | 4 × 40 | 3.0 | 0.3083 | 0.225 | 1.975 m | 2.701 m |
| orc 4×8 — **v7-tiny deployed** | 4 × 8 | 15.0 | 0.2077 | 0.157 | 2.133 m | 2.955 m |
| orc 16×4 — cols pooled ONLY | 16 × 4 | 30.0 | 0.2028 | 0.161 | 2.246 m | 2.860 m |
| orc 4×4 — **v1/v4/v5f flagship** | 4 × 4 | 30.0 | 0.1583 | 0.130 | 2.669 m | 3.454 m |
| orc 4×2 | 4 × 2 | 60.0 | 0.1251 | 0.109 | 3.283 m | 3.545 m |
| orc 1×1 — global pool | 1 × 1 | 120.0 | 0.0909 | 0.086 | 4.550 m | 4.184 m |
| **pos_only** — marginal control | — | — | **0.0317** | 0.032 | 4.479 m | 4.175 m |
| **all-zero** — control | — | — | **0.01634** | **0.000** | — | — |
| | | | *vs base rate **0.016198** — ⚠️ **+0.881 %, NOT equal; the `=` and ✓ are RETRACTED, `R-2026-09-07-ap-ties`** (tie-blind AP scores a CONSTANT arm above its own base rate). The ladder and every conclusion below STAND.* | *F1 **0.000*** | | |

**Against the full 16×40 grid:**

| pooled to | AP retained | AP cost | lateral error 15–30 m |
|---|---|---|---|
| 8×20 (refcv5) | **71.6 %** | 1.40× | +0.44 m (1.32×) |
| 4×8 (v7-tiny) | **44.1 %** | 2.27× | +0.76 m (1.55×) |
| 4×4 (flagship) | **33.6 %** | **2.98×** | **+1.30 m (1.95×)** |

### 5.2 ⭐ Axis attribution — azimuth is the expensive axis, with the confound stated

Cutting **azimuth only** to 4 bins (`16×4`) costs **AP 0.4713 → 0.2028 = −57.0 %**.
Cutting **elevation only** to 4 bins (`4×40`) costs **AP 0.4713 → 0.3083 = −34.6 %**.
⇒ **cutting azimuth to a given bin count costs 1.65× what cutting elevation to the same bin count
costs**, and the combined 4×4 loss (−66.4 %) is close to multiplicative.

⚠️ **State the confound rather than the ratio alone: these are equal FINAL bin counts, not equal
pooling factors** — azimuth goes 40 → 4 (**10×**) while elevation goes 16 → 4 (**4×**), so the
azimuth cut also discards more bins. The comparison therefore supports *"azimuth is the more
expensive axis to cut **to a given resolution**"* — which is the form a designer choosing a readout
shape actually needs — and it is **not** a per-factor attribution. A matched-factor datum from the
same ladder: pooling **both** axes 2× (`8×20`) costs −28.4 %, against −34.6 % for pooling elevation
alone 4×.

⭐ `E-DEC-25` deliberately made **no** claim on the row axis because its ladder was confounded by
dimensionality; this ladder is not (matched capacity by construction), and it says rows matter
too — just less than columns.

### 5.3 The two variances, named separately

⭐ **Replicate arms — same split, TRAINING seed only (`--train-seed 0/1/2`)**, which is the variance
an episode bootstrap is structurally blind to (`H-ESTIM-SEED-1`):

| arm | seed 0 | seed 1 | seed 2 | **range** |
|---|---|---|---|---|
| orc 16×40 | 0.4773 | 0.4651 | 0.4715 | **0.0122** |
| orc 8×20 | 0.3367 | 0.3337 | 0.3419 | 0.0082 |
| orc 4×8 | 0.2106 | 0.2036 | 0.2090 | 0.0070 |
| orc 4×4 | 0.1594 | 0.1564 | 0.1592 | 0.0030 |
| pos_only | 0.0316 | 0.0320 | 0.0315 | 0.0005 |

⇒ **The run-to-run noise floor is ≤ 0.0122 AP**, and every headline gap clears it by a wide margin:

| comparison | ΔAP | × the 0.0122 floor |
|---|---|---|
| 16×40 → 8×20 | 0.1339 | **11.0×** |
| 16×40 → 4×8 | 0.2636 | **21.6×** |
| 16×40 → 4×4 | 0.3130 | **25.7×** |
| 8×20 → 4×8 | 0.1297 | **10.6×** |
| 4×8 → 4×4 | 0.0494 | **4.0×** |

⛔ **But two rungs are NOT distinguishable and must not be read as ordered:**
`orc_4x8` (0.2077) vs `orc_16x4` (0.2028) differ by **0.0049 — below the noise floor**, and
`orc_8x20` (0.3374) vs `orc_4x40` (0.3083) differ by 0.0291, only 2.4× it. ⭐ The first
non-separation is itself informative: **16×4 has twice the cells of 4×8 (64 vs 32) and buys
nothing** — consistent with §5.2's finding that azimuth bins are worth more per cell than
elevation bins.

**Paired episode-cluster bootstrap**, 1 000 resamples of the 30 test clips
(`raw/bootstrap.json`, `code/s7_bootstrap.py`). ⚠️ **This answers a DIFFERENT question from the
table above**: *would another DRAW OF EPISODES say this?* — not *would another TRAINING RUN say
this?* Both are reported because a separated interval from one seed is necessary and not sufficient.

| `orc_16x40` − arm | ΔAP | 95 % CI | sep | Δ lateral 15–30 m | 95 % CI | sep |
|---|---|---|---|---|---|---|
| − 8×20 | **+0.1413** | [+0.1192, +0.1629] | ✓ | **−0.454 m** | [−0.668, −0.267] | ✓ |
| − 4×40 | +0.1692 | [+0.1540, +0.1879] | ✓ | −0.638 m | [−0.900, −0.386] | ✓ |
| − 4×8 | **+0.2682** | [+0.2348, +0.3055] | ✓ | **−0.770 m** | [−1.015, −0.515] | ✓ |
| − 16×4 | +0.2759 | [+0.2413, +0.3147] | ✓ | −0.904 m | [−1.179, −0.672] | ✓ |
| − 4×4 | **+0.3199** | [+0.2824, +0.3633] | ✓ | **−1.336 m** | [−1.756, −0.960] | ✓ |
| − 4×2 | +0.3539 | [+0.3164, +0.3981] | ✓ | −1.930 m | [−2.597, −1.357] | ✓ |
| − 1×1 | +0.3884 | [+0.3449, +0.4357] | ✓ | −3.203 m | [−4.498, −2.153] | ✓ |
| − pos_only | +0.4461 | [+0.3999, +0.4960] | ✓ | −3.132 m | [−4.448, −2.072] | ✓ |

⇒ **every rung is separated from the full grid on BOTH metrics**, and each survives the replicate
spread above. ⚠️ The bootstrap does **not** rescue the two rungs §5.3 says are not distinguishable
from **each other** (`4×8` vs `16×4`): their intervals against 16×40 overlap heavily
([+0.2348, +0.3055] vs [+0.2413, +0.3147]), which is the same non-result read from the other side.

---

## 6. Dense BEV vs sparse queries — the implication, stated because it decides the build

Four measured facts bear on it, and they do not all point the same way.

1. ⭐ **The token grid is itself a lossy address space for a 0.5 m BEV.** Under a **perfect**
   front-end the full 16×40 grid reaches only **AP 0.471 / IoU 0.313**, not ~1.0. Measured cause:
   the address map puts **a median of 4 BEV cells (max 313) into one token cell**, and only
   **242 of 640** token cells receive any ground-plane BEV cell at all. ⇒ **a dense BEV built by
   indexing an image-token grid inherits a quantisation floor no training removes**, worst exactly
   where a planner cares — one column is 2.6 m wide at 50 m even at 40 columns, and 13.2 m at 8.
2. ⚠️ **But the pooled rungs are not "nothing".** 4×4 scores **5.0× its own marginal control** and
   **9.7× the all-zero predictor**. ⇒ *"indexing into a 4-column map is indexing into almost
   nothing"* is **too strong as written**; the supported form is *"loses ~2/3 of the available
   localisation"*.
3. ⭐ **The arm that would actually be built already sits at the good end of the ladder.** refcv5's
   8×20 map retains **71.6 %** of the full grid's AP for **+0.44 m** of lateral error at 15–30 m.
4. ⛔ **And today the map contains no agents anyway** (§4, and `E-DEC-29`'s *"knows roughly how
   crowded the scene is … and does not know WHERE any individual object is"*).

⇒ **Recommendation — a ranking, not a refusal:**

* ⭐⭐ **Reorder the design's work packages: WP-D before WP-B, and WP-A is not the gate.** WP-B
  indexes a feature map; §4 says the map has no transferable agent localisation in it at *any*
  resolution. Building the index first means indexing into an empty room. **WP-D (the BEV auxiliary
  loss) is the true prerequisite** — and the register already names the cheap winner for it:
  `E-DEC-8`, distillation into frozen DINOv3, took `n_agents` from −1.04 to **+0.33** at **no cost
  to ego**. That is the highest-measured-effect lever touching this question.
* **For a REF-C-shaped refcv5c, do NOT gate coupling (1) behind a readout change.** 8×20 → 16×40 is
  a 1.40× AP lever; the coupling not existing at all is the larger term.
* **On the v6/v7 trunk the readout IS a first-order lever once the map carries something** — 4×8
  gives up 56 % and 4×4 gives up 66 % of the ceiling. ⚠️ And it is not the only readout term:
  `E-DEC-25` measured the **128→64 projection** as the destroyer of `lead_range_m` (`tok 4x8`
  +0.0719 vs `readout 4x8` −0.1611, t 13.68, 24/24) at **identical grid and identical pooling**.
  ⇒ **the pool sets the ADDRESS resolution (this WP) and the projection sets what survives per
  address (E-DEC-25); indexing the TOKENS directly bypasses both.**
* ⭐ **SparseDrive's route is the better hedge for the far field, and the seam already exists.**
  `refc_agents.slot_features` (`stack/tanitad/refs/refc_agents.py:214`) passes **explicit metric
  range and bearing** per agent rather than a learned embedding — a *continuous* address that does
  **not** pay fact (1)'s token-cell quantisation floor at all. It is built and gated **off**
  (`--agents off` in refcv5-v2's argv). ⇒ the dense route should be justified **against** turning
  that seam on (PI decision-queue item 9), not assumed.

---

## 7. Which state my negatives are — and the third state is still missing

⛔ **A negative cell in this probe is NOT "seen-and-empty".** It is *"no `obstacle.offline` cuboid
covers this cell centre"*, which **merges** genuinely different states:

| state | in my negatives? | how it was handled |
|---|---|---|
| out of the camera's horizontal field | **NO — excluded** | `fov_mask` (half-angle 60°) removes **590 / 7 680**; a further **94** fall outside the addressable row/col range ⇒ **6 996 of 7 680 (91.09 %) scored** |
| seen-and-empty | **yes** | indistinguishable from the next row |
| occluded by another agent | **yes** | `obstacle.offline` carries no visibility flag; no object–object occlusion is modelled |
| beyond the ~20 s label span | **NO** | an absent `(clip, frame)` line is NO_LABEL and was never banked |

⇒ `fov_mask` is **two-state** (in-field / out-of-field): it says where the camera *could* look,
never whether a cell was *seen and empty*. **The third state is still missing programme-wide** —
its third appearance, after `NOT_APPLICABLE` vs `NOT_CHECKED` and `EXPLICITLY_ABSENT` vs `UNKNOWN`.
This WP does not fix it; it reports which side of it every number sits on.

---

## 8. A probe that FAILED, banked because the controls refused it

⭐ The first panel (`code/s2_probe.py`, `raw/panel_k8.json`) regressed the **dense 7 680-cell
raster** with exact dual ridge. Its controls refused it: the **CONSTANT control scored AP 0.03402
while the full 16×40 arm scored 0.0305** — every arm at or below "predict the per-cell marginal" —
and λ for the 81 920-dim arm **pinned at the grid maximum (1e5)**. That is the 2026-08-22
signature (n ≪ d ⇒ validation correctly picks maximal regularisation ⇒ every arm reads the
no-information value). ⇒ **a statement about the parameterisation, not about the readout.** It is
banked as a negative result about the probe, and it is the reason §4 and §5 use low-dimensional and
address-indexed targets instead.

---

## 9. What this does NOT claim

* ⛔ **§5 is an ORACLE.** It prices a *readout geometry*, not a perception system. No arm there
  detected anything from pixels.
* ⛔ **§4 is not "the encoder is bad"** — see the scoping sentence at the end of §4.2. More
  episodes exist (the B1 TRAIN join covers **4 427** clips) but their **256×640 pixels are not on
  this box**, and pulling them was out of scope. **That is the named blocker** for turning §4 into a
  real ceiling rather than a bound.
* ⚠️ The **row model** (`v = v0 + f_ref·h/range`, `h_cam 1.5 m`) is a stated approximation of the
  vertical axis. It affects the *row* rungs; the azimuth column mapping is exact for this corpus.
* ⚠️ **Column-linear-in-azimuth is scoped to our 256×640 `f_ref 305.577` corpus.** It does not
  travel, and a pinhole formula on the same data gives 92.6° and looks plausible.
* ⚠️ Occupancy here is **1.74 %** of grid cells on **B1 EVAL**. The DataFlyWheel's **1.16 % /
  median 0.45 %** is a **train-side** census. Both are MEASURED; state the corpus with the number.
* ⚠️ §4's trunk is `k8clip05p30k`. `E-DEC-2 / E-DEC-25 / E-DEC-29` probed `splitp30k`, `rdw8p30k`,
  `scale1`. The agreement across arms is corroboration, not identity.

---

## 10. Is the 16×40 readout sufficient for waypoint-indexed attention?

⭐ **Sufficient as an ADDRESS SPACE — it is far from the binding constraint today — but the design
would be indexing into a map that does not yet contain agents, so WP-B built now indexes into an
empty room, and the fix is WP-D, not WP-A.**

---

## 11. Deliverable manifest

| artifact | where |
|---|---|
| this report | repo `TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-wpa-readout-localisation/RESULT.md` |
| probe code (8 files, `s0`–`s7`) | repo `…/2026-09-07-wpa-readout-localisation/code/` |
| result JSONs + run logs | repo `…/2026-09-07-wpa-readout-localisation/raw/` |
| feature bank (2.4 GB, **not** committed) | dev box `C:\Users\Admin\wpa-readout\bank\k8r1\` (`tok.npy`, `pix.npy`, `y.npy`, `idx.json`) |
| stack snapshot the probes imported | dev box `C:\Users\Admin\wpa-readout\stack_snapshot\` (copy of repo `stack/` taken at run time) |
| trunk | dev box `C:\Users\Admin\k8-pull\ckpt.pt` + `config.json` (v6-staged `S-W`, step 30000) |
| GT join (unmodified, pre-existing) | dev box `C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz` |
