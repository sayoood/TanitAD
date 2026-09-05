# refcv5 BUILD — Architecture & Inference FlyWheel

**STATUS: IN PROGRESS** · opened 2026-09-05 · agent generation 10 (nine predecessors died mid-work)
**Branch:** `agent/arch-inf-20260803` · **Bank:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-05-refcv5-build/`

## The mandate (PI, 2026-09-05, verbatim)

> "I propose to prepare refcv5 for training, implement all missing pieces. Let's follow environment
> extraction from the front camera and train it based on GT data. If the whole refcv5 is driving and
> achieves good quality, we can extend it to multi cameras (first step 4) then to Lidar."

**This is a BUILD, not a refutation.** The deliverable is working code that makes refcv5 trainable.
Gates and controls apply — they stop us fooling ourselves — but a clean refutation is not a deliverable.

## Build order (each ships with a test that fails without it)

| # | component | status |
|---|---|---|
| 1 | Environment extraction from front camera, GT-supervised (`obstacle.offline` labels → mono-3D agent head → agent tokens) | **MODEL SIDE LANDED** (28/28 tests) |
| 2 | Score the fan we actually emit (four-family heads on the EMITTED fan) | PENDING |
| 3 | Diffusion in CONTROL space (anchored Gaussian, DDIM, x0-pred, per-layer AdaLN) | PENDING |
| 4 | Trainable end to end (`refc_v3_train.py` flags, seams stamp, tiny-rig smoke, launch argv) | PENDING |

---

## ⭐ THE FINDING THAT CHANGED THE BUILD — most of component 1 ALREADY EXISTED

Before writing a line I asked what was already in the tree. **Three complete
components were there**, and my first draft was re-implementing all three:

| already existed | file | what it gives |
|---|---|---|
| rig ⇄ canonical-**cylindrical** projection, both directions | `stack/tanitad/data/rig_projection.py` (323 lines) | `RigCamera`, `project_rig_to_frame`, `frame_to_cam_ray`, `ground_intersection` — line 1 says it was written *"for the refcv5 environment head"* |
| DETR slot decoder, exact Hungarian, DETR set loss, join→targets | `stack/tanitad/models/agent_slots.py` (741 lines) | `AgentSlotDecoder`, `hungarian`, `match_slots`, `slot_set_loss`, `targets_from_join` |
| the join reader | `stack/scripts/train_p8_occupancy.py::JoinFileReader` | `lookup` / `lookup_classes` / `raster`, NO_LABEL vs labelled-clear kept distinct |

⇒ **`refc_agents.py` was rewritten as a THIN ADAPTER.** Every one of those is
imported, none re-implemented. *(Two implementations of one geometry is how they
drift apart — the `advect` precedent, retired 2026-07-27.)*

⛔⛔ **AND THE REWRITE CAUGHT A FABRICATION IN MY OWN FIRST DRAFT.** I hand-wrote
a plausible-looking 10-class enum containing **`bicycle`, `motorcycle`,
`train_or_tram_car`** — ⛔ **none of which exist in the corpus.** The real
`obstacle.offline` enum (`bev_raster.py:93`, MEASURED over 87,481 cuboids) is
`automobile · heavy_truck · bus · other_vehicle · trailer · person · rider ·
stroller · animal · protruding_object`. The head would have trained against
three classes that can never appear and **silently dropped `other_vehicle`,
`stroller` and `animal`**. The enum now has exactly one spelling — the label
side's — and `test_class_enum_is_the_corpus_enum` pins it, naming all six words
so the error cannot recur. **Root-cause class: a plausible constant written from
memory instead of read from source** — the same family as the `df` / `step_s` /
cylindrical-FOV traps, with the object being a VOCABULARY.

## ⚠️ A DOC CORRECTION THIS RUNG FORCED — S2b's "bit-identical" is an overstatement

`refc.py`'s `anchor_prefilter` block claimed *"decoding a subset is bit-identical
for that subset"*. **MEASURED on the dev box, with the agent branch OFF and
nothing else changed:** a subset decode disagrees with the full decode by
**1.192e-07** (conf) and **1.490e-07** (offset) — a different candidate count
selects a different GEMM reduction order. The **structural** claim (no
candidate-axis mixing) stands and is what the flag actually needs; the claim the
measurement supports is the one the same block already makes — the **selection
index is identical on 881/881**. Bit equality was never measured. The comment is
corrected in place, and the test that found it is a **differential** control
(agent-branch-on vs agent-branch-off) rather than an absolute one, because the
absolute version fails for a reason that has nothing to do with agents.

## Component 1 — what landed

**Files:** `stack/tanitad/refs/refc_agents.py` (new) · `stack/tanitad/refs/refc.py`
(seam) · `stack/tests/test_refc_agents.py` (new, 28 tests).

**New surface (the four gaps the existing modules left open):**

1. `AgentTokenEmbed` — decoded slots → the `[B, N, d_dec]` tokens the planner
   cross-attends, + the padding mask. **This seam did not exist**: `AgentSlotDecoder`
   emitted a set of boxes and *nothing consumed them* (`V6Config.agent_slots` is in
   `LADDER_UNTRAINED_GROUPS` — it trains in no stage). Tokens carry **explicit
   range and bearing**, because every LONGITUDINAL metric we are bound to report
   is a function of exactly those two.
2. `monocular_projection_loss` — the **image-plane** term. `slot_set_loss` is
   entirely BEV metres; a monocular head supervised only in BEV is asked to
   regress the one axis it cannot directly see, with no term in the space it can.
3. `ground_range_prior` — the **free** metric anchor. Rig z = 0 **is** the road
   plane, so pixel + plane fixes range. Costs **no label**, so it runs on the
   NO_LABEL frames past ~20 s that are a large share of every clip.
4. `OracleAgentEmbed` + `degrade_boxes` — `E-AGT-ORACLE` and `E-AGT-BUDGET`,
   which must run **before** any detector GPU-day.

**Decoder seam (`CrossAttnLayer`):** a second cross-attention on a **zero-init
gate**. `cross_agent=False` constructs nothing (RNG order preserved ⇒ bit-identical
to refcv4b); `cross_agent=True` starts at exactly 0 with `dL/dgate ≠ 0` — gated,
not dead.

⛔ **A NaN trap fixed on the way, and it is a ROUTINE state not an edge case:** a
window with **no agents** (an empty road) makes `key_padding_mask` all-True on
that row; the attention softmax then normalises over all `-inf`, the output is
NaN, and **the backward is NaN even when the forward is patched** with
`nan_to_num`. Fully-padded rows are now un-masked and zeroed by a row factor —
NaN-free in both directions, and no device sync. Two tests pin it (all-padded and
the mixed row the corpus actually produces).

**Verification:** 28/28 new tests, and **123 passed** across
`test_refc · test_refc_select · test_refc_v3 · test_refc_agents · test_v6_agent_slots`
(the one red was my own `CUDA_VISIBLE_DEVICES=""`, which makes torch report CUDA
available with zero devices; green without it).

⚠️ **Two numbers the next rung must MEASURE, not inherit:**
* `agent_slots.N_QUERIES_DEFAULT = 16` is a **declared placeholder**. The val40
  join is **195,805 boxes / 7,400 frames = 26.5 agents/frame MEAN**, so 16 makes
  `match_slots` drop targets on most frames. This seam defaults to **32**; the
  **p99 must be measured** and recorded in the prereg before a full-scale run.
* `assert_fully_observed` **FAILS on 256×640** (~8.9 % masked on rig B, ~0.06 %
  on rig A — a **rig-correlated** black region). `176×624` is the measured
  rig-clean frame and is a bit-exact pixel slice of the existing PNG cache.
  refcv4b trains on 256×640, so the seam stays there for comparability — but a
  detector inherits that rig-correlated mask and it is an **escalation**, not a
  footnote.

⛔ **Do not launch the full training** — that is the Master Mind's call.
⛔ `refcv4b-b1-v72-40k` is LIVE on pod `tanitad-refcv3` — never touched.

## Binding constraints carried

- Frames are **256×640 CYLINDRICAL**, `f_ref` 305.577, **HFOV 120°** — the pinhole formula is WRONG
  here. Project only through `calib.py::CanonicalFrame`.
- Rig **z = 0 is the road plane** (MEASURED over 87,481 cuboids: ground-standing classes' bottom
  faces −0.05 to −0.13 m; `protruding_object` +1.68 m). Camera height 1.43–1.56 m.
- `obstacle.offline` is a **LABEL** (privileged); inference is **VISION-ONLY**.
- Canonical `ha0_ext` is the INTEGRATOR `refav1_arm.hold_ext_controls`, never the closed form
  (§M11 — they differ by 0.54 m at 2 s).
- Every seam **zero-init and gated**: a v5 build with all gates off must be **bit-identical to v4b**.

## Log

- `2026-09-05` STATUS header banked before first line of code.

## Deliverable manifest

(appended as components land)
