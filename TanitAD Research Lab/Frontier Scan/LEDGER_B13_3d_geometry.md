<title>LEDGER B13 — 3D, geometry, occupancy, rendering</title>

# LEDGER B13 — 3D · geometry · occupancy · rendering

⛔ **APPEND-ONLY.** Corrections are dated additions, never rewrites.

## Our position in this track

**2026-08-31.** ⛔ **This is the track nearest a MEASURED TanitAD ceiling.** v6 pools a 16×40 token
grid to **4×4** — **4 azimuth bins across a 120° cylindrical FOV, i.e. 30° per bin, 2.1–7.8× too
coarse for BEV localisation**. It is a **geometry** ceiling, not an encoder verdict. Our corpus is
**cylindrical** (`256×640`, `f_ref` 305.577), so the pinhole FOV formula is wrong here (it yields
92.6° against the rig's true 120°). We hold **no** map, lane graph or route signal; NuRec's
`map.xodr` is the only map prior we can reach.

---

## Entry 2026-08-31-01 — the geometry gap is a property of the field, not our bug

⭐⭐ **The reframing, and it is the most valuable thing in this track today.** Waymo's L7 states that
frontier VLMs *"lack sufficient spatial awareness"* — a limitation they hit with Gemini-class models
and route around via a fast/slow split. **We diagnosed the same class of defect in our own encoder
and treated it as a TanitAD failure.** It is a property of semantic-pretrained vision stacks.

⇒ **Strategic consequence:** fixing metric geometry in a small latent world model is a
**differentiator**, not a repair. This upgrades the v6 readout-geometry work from remediation to
contribution. *(Source: Waymo blog 2026-08-26, `PUBLISHED-BLOG`; adjudicated in
`Opponent Analysis/Research/2026-08-31-waymo-10-lessons-adjudication/`.)*

## Entry 2026-08-31-02 — candidate mechanisms banked

| paper | mechanism | why it is on this list |
|---|---|---|
| `2607.04732` **SparseOcc++** | *"geometry-aware sparse latent representation for semantic occupancy"* | A **sparse** latent is the obvious answer to a dense 4×4 pool that destroys azimuth resolution: keep resolution where there is structure, spend nothing where there is not. **Most direct candidate found for our ceiling.** |
| `2603.03482` **Beyond Pixel Histories** | world models with **persistent 3D state**; reported gains in *"spatial memory, 3D consistency, and long-horizon stability"* | Our predictor holds no explicit 3D state; persistence is what a 30 s strategic horizon would need. Bridges B13 and B12. |
| `2412.13772` (not banked) | occupancy world model via decoupled dynamic flow | separating static geometry from dynamic flow — relevant to our lat/lon decomposition |
| `2510.16729` (not banked) | 4D occupancy forecasting via implicit residual world models | vision-centric 4D forecasting + planning |

⚠️ **All four are abstract/title-level.** `2607.04732` and `2603.03482` are **banked but unread** —
neither may decide a GPU-day. **This is a scan, not a design review.**

⚠️ **The transfer risk, stated up front:** occupancy work assumes a **surround** multi-camera rig with
lidar-derived occupancy supervision. We have **one forward 120° cylindrical camera and no occupancy
labels**. A sparse-occupancy method may be unportable to our corpus for reasons that have nothing to
do with its merit. **Establish label availability before designing anything on it.**

**Proposed experiment (0 GPU first):** read `2607.04732`; determine what supervision its sparse
latent requires; then decide whether a monocular-cylindrical variant is even definable. **Pre-
committed read:** if it needs occupancy ground truth we cannot produce, the line is closed and we
say so rather than approximating it into meaninglessness.

`Next: read the two banked primaries; check NuRec map.xodr as a geometry-supervision source.`


---

## Entry 2026-09-01-01 - SparseOcc++ discharged at abstract level; its ONE load-bearing question is unanswered

**Source:** `arXiv 2607.04732`, "SparseOcc++: Geometry-Aware Sparse Latent Representation for Semantic
Occupancy Prediction" (Tang, Wang, Wang, Ren, Ma; 2026-07-06).
**Evidence class:** PUBLISHED lib 2607.04732 **abstract-only - DECLARED.**

Decouples **scene completion** (signed-distance regression on sparse anchor voxels via a scene
completion field) from **semantic segmentation**, restricting segmentation to geometrically verified
regions. Reported: **+2.3 IoU and 3.9x faster than SparseOcc on nuScenes**; **5.9x speedup over
OccFormer on SemanticKITTI**.

⛔ **The rotation asked this paper exactly one question - "establish supervision requirements before
any design" - and the abstract does not answer it.** Dense occupancy labels vs lidar vs camera-only
supervision is **not stated**.

**Why that is disqualifying for us specifically:** PhysicalAI-AV has **no map, lane graph, junction
annotation or occupancy label** (settled at five probes), and our only 3D agent source is
`obstacle.offline` (10 dynamic-agent classes, 87,481 cuboids). **The supervision requirement is the
only fact that decides whether this line is reachable for TanitAD at all.**

⇒ ⛔ **No design work on this until the full text answers it.** Carried as an open debt.

`Next in this track: full-text read of section 4 (training/supervision) ONLY - that is the whole question.`


---

## Entry 2026-09-02-01 — ⛔ SparseOcc++ debt D-5 DISCHARGED: the line is CLOSED on a pre-registered outcome

`PUBLISHED lib 2607.04732 · FULL TEXT READ 2026-09-02 · register debt D-5 · backlog P-11 / guideline S-3`

Backlog **P-11** pre-committed the read: *"if it requires occupancy GT we cannot produce, the line is
CLOSED and we say so."* **It requires exactly that, on two independent supervision paths.**

1. **Dense per-voxel semantic occupancy GT.**
   > *"SCF Ground Truth Generation. To supervise SCF learning, we efficiently generate ground truth (GT)
   > from **semantic occupancy labels**."*

   Training uses **SemanticKITTI** (*"LiDAR point clouds, LiDAR segmentation labels, and semantic scene
   completion labels"*; each voxel annotated empty or one of **19** semantic classes) and
   **nuScenes-Occupancy** (*"dense 3D semantic occupancy annotations for keyframes"*, **16** classes plus
   empty, built by the Augmenting-and-Purifying pipeline).

2. **LiDAR, separately, for depth.**
   > *"L_depth is calculated between the predicted depth map and the ground truth projected by **point
   > clouds** for the supervision of the LSS component."*

   Plus mask and classification losses via Hungarian bipartite matching on *"binary mask labels and
   corresponding semantic class labels"*, and a segmentation loss on the coarse binary classifier.

⛔ **PhysicalAI-AV has neither.** No map, no lane graph, no occupancy labels (settled at five probes); the
only 3D source is `obstacle.offline` — **10 dynamic-agent classes, 87,481 cuboids** — an agent-box corpus,
not voxel occupancy, and there is no LiDAR.

⇒ **THE SPARSEOCC++ LINE IS CLOSED FOR TANITAD.** Recorded because it was pre-registered, not because it
is convenient. Guideline **S-3** (claim the geometry gap as a differentiator) survives, but **must not be
built on this paper**.

### ⭐ The escape hatch, named by the paper itself — where B13 goes next

Its related work cites a live family needing **no** occupancy GT:

- **2D-rendering supervision:** RenderOcc (ICRA'24), OccNeRF, UniOcc — *"impose NeRF-style 2D supervision
  to implicitly learn 3D geometry"*.
- **Self-supervised:** SelfOcc (CVPR'24), GaussianOcc (ICCV'25, *"fully self-supervised"*), GaussTR
  (CVPR'25), GeoCC (IEEE TITS'25).

⭐ **We hold the assets this family runs on: NuRec scenes are open msgpack and gsplat does 492 FPS on
Thor.** ⚠️ But it does **not** inherit SparseOcc++'s numbers (+2.3 IoU, 3.9x over SparseOcc, 5.9x over
OccFormer on SemanticKITTI) — assuming it does would be the same scope error in a new place.

⚠️ The paper also notes these methods *"often entangle geometry with appearance or require complex
rendering pipelines"* — a real cost, and the reason this is a track direction rather than a plan.

### ⭐ One transferable ablation, independent of the supervision blocker

Their Table V (prediction heads):

> *"the linear head achieves the highest geometry IoU of **36.8**. We hypothesize that this occurs because
> the linear head is supervised by an **explicit geometry loss**, i.e., a cross-entropy loss between
> occupied and non-occupied voxels. Conversely, the transformer decoder formulates occupancy prediction as
> mask generation for semantic classes only, **imposing no explicit supervision on occupied voxels**."*

⇒ **Explicit geometry supervision beats an implicit semantic formulation at geometry.** Our MEASURED
readout-geometry ceiling (4 azimuth bins over 120 deg, 2.1-7.8x too coarse for BEV localisation) is trained
with **no explicit geometry term at all**.

⭐ **That is a materially cheaper hypothesis than re-architecting the readout, and it has never been
tested.** Proposed as a backlog row with a committed outcome: **if an explicit geometry loss does not move
azimuth-bin decodability, the ceiling is resolution-bound and the readout must be re-architected** — the
expensive branch, but then a known-necessary one rather than an assumed one.
