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
| 1 | Environment extraction from front camera, GT-supervised (`obstacle.offline` labels → mono-3D agent head → agent tokens) | **LANDED** (31 tests) |
| 2 | Score the fan we actually emit (four-family heads on the EMITTED fan) | IN FLIGHT (delegated) |
| 3 | Diffusion in CONTROL space (anchored Gaussian, DDIM, x0-pred, per-layer AdaLN) | **LANDED** (26 tests) |
| 4 | Trainable end to end (`refc_v3_train.py` flags, seams stamp, tiny-rig smoke, launch argv) | **WIRED** (11 tests) |

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

---

## Component 3 — the diffusion mechanism in CONTROL space (WP-4, `E-DDA-3`)

**Files:** `stack/tanitad/refs/refc_sampler.py` (new) · `refc.py` (decoder branch) ·
`stack/tests/test_refc_sampler.py` (26 tests, 1 skipped).

⭐ **This is the programme's advance over DiffusionDrive, and it is one decision:
the noise lives in `(a_lon, a_lat)`, not in metres.** Every sample re-rolls
through the programme's own integrator and is flyable **by construction**.

The arithmetic is now **pinned by tests** rather than asserted in prose:
`sqrt(1 - alpha_bar_8)` = **0.0316**. In DD's metre normalisation that is
**0.90 m / 0.73 m per waypoint, independently** — a 0.73 m lateral excursion over
a 0.5 s slot is **5.8 m/s² (0.6 g)**. In control units the same schedule gives
**0.126 / 0.095 m/s²**.

⛔⛔ **THE LOAD-BEARING IDENTITY, and the reason any of this is trustworthy:** a
**constant** control sequence rolled by `roll_controls` is **bit-identical** to
`roll_bank`, for **both** unit systems. Without it the sampler and the vocabulary
integrate different physics, every anchored-Gaussian claim is measured against a
fan the vocabulary never emitted, and **nothing else in the suite would notice.**
Getting there required expanding each slot's control over its own **tick span**,
because `roll_bank` integrates at the native 0.1 s tick and subsamples; one
variable-length step per slot is a *different integrator*. `alat → curvature` now
has exactly **one spelling** — `roll_bank` calls the sampler's function.

**The scheduler is re-implemented in ~40 lines rather than imported from
`diffusers`** — deliberate, not NIH: `uv pip install <anything>` has **twice**
silently replaced torch with a wheel the driver cannot run, and an analysis-time
import that fails *after* the rollout destroys a run whose compute is already
paid for. It is **pinned against the real `diffusers` object wherever that is
installed**, so it is checked rather than trusted.

**Two refusals instead of plausible-looking wrong experiments:** `ddim` on a
fixed-path vocabulary (where `anchor_controls` is all zeros, so the Gaussian
would be centred on "do nothing"); and `sampler_groups > 1`, which **names the
three consumers** that still assume an N-wide fan (`loss_cls`'s `a_star`, the
`[B, N]` priors, every `sel_idx` dump).

The **DD-literal metre-space arm is a runnable code path**, not a description — a
deliberate regression that cannot be run has never been tested.

## Component 2 — score the fan we actually emit (WP-7, `E-DDA-2b`)

**Files:** `stack/tanitad/refs/refc_selector.py` · `refc_selector_targets.py` ·
`stack/tests/test_refc_selector.py` (24 tests). *(Delegated stream; verified
here — files read from disk, tests re-run in this session.)*

Step 1 of the fix needed **no new code**: `SelectionConfig.refined` /
`score_emitted` already exist and are zero-parameter. Step 2 is the learned
four-family selector: candidate embed → coarse (1 layer, scene + optional agent
cross-attention, **candidate self-attention**, FFN) → top-32 → fine (3 layers) →
`σ(NC)·(5σ(TTC) + 5σ(progress) + 2σ(comfort) + w_c σ(compliance) + w_t σ(tactical))
/ (12 + w_c + w_t)`.

⭐ **Candidate self-attention lives ONLY here.** The generator keeps candidates
independent — that is what `anchor_prefilter` rests on — and the paired
`self_attn=False` control reads **exactly 0.000e+00** coupling.

**Its tests were mutation-tested** (self-attention removed, a tie-break bias
injected, the collision readout blinded, the `n_*` keys stripped) — **all four
caught**. The deliberate regression works: a progress-only selector picks the
fast **colliding** candidate, and switching the NC head on flips the pick.

⚠️ **The plan's WP-7 param estimate priced the ABLATION, not the arm.** §7.1 says
"≈ 3.5 M". MEASURED at `scene_dim=704, n_steps=8`: default **4,530,444**;
`self_attn=False` **3,475,724** — *that* is the 3.5 M. The 1,054,720 gap is
exactly the four candidate self-attention blocks, i.e. the module's whole reason
to exist. The launch preflight must assert **4,530,444**.

⚠️ **`rl/rewards.py::_collision` tests WAYPOINTS, not SEGMENTS** — a fan whose
waypoints straddle an obstacle passes through it undetected. Documented, not
changed: the predicate is shared with the RL stream, so a swept-segment version
is a cross-stream decision. ⇒ **ESCALATED.**

## Component 4 — trainable end to end

**Files:** `stack/scripts/refc_v3_train.py` · `stack/tests/test_refc_v3_refcv5_wiring.py`
(11 tests).

Flags (all default OFF, and **OFF means NOT CONSTRUCTED**): `--sampler`,
`--sampler-space`, `--sampler-{train-t-max,infer-t,steps,groups}`, `--w-u0`,
`--agents {off,head,oracle}`, `--w-agent`, `--agent-queries`,
`--agent-w-{project,ground}`, `--agent-{sigma-range,miss-rate}`,
`--agent-presence-hard`. Every lever is stamped into `config.json['seams']`,
**including the class vocabulary**, so a corpus-side rename cannot silently
re-map a trained head.

⛔⛔ **THE TWO GUARDS THAT MATTER MOST, because each refuses a run that would
otherwise MANUFACTURE A REFUTATION:**
* `--sampler ddim --w-u0 0` **refuses**. `control_head` is zero-init; with no
  loss on it, it stays at exactly zero forever. The arm would be the anchored
  Gaussian **with no denoiser at all**, every log row would say `sampler: ddim`,
  and the result would read as *"diffusion does not help"*.
* `--agents head --w-agent 0` **refuses**, for the same reason one level over: an
  unsupervised detector emits noise tokens, the zero-init gate never opens, and
  the arm reads as *"agent tokens do not help"*.

⛔ **The units trap, caught in the x0 loss and pinned by a test.**
`unicycle_controls_from_path_varstep` returns **curvature (1/m)**; the bank's
controls are **lateral acceleration (m/s²)** under `control_units="alat"`.
`a_lat = v²κ`, so at 36 m/s they differ by **~1300×** — and both tables look
plausible. This is the exact error `anchor_meta.py` exists to prevent (MEASURED
once as *"396 g at 36 m/s"*). The target is converted before the L1.

---

## ⛔⛔ THE FINDING THAT CHANGES THE TRAINING SET — 61.8 % of the supervision was unobservable

MEASURED 2026-09-05 on the val40 join (**195,805 boxes / 7,400 frames / 39
clips**; `raw/agent_density.json`, `measure_agent_density.py`):

`agent_slots.targets_from_join` sets `valid = True` for **every** agent in the
record — no azimuth filter, no range filter — and `match_slots` then keeps the
`n_queries` **NEAREST** of that set. At `n_queries = 16`, of the 82,247 targets
kept, **50,816 (61.8 %) are `occ == 1` — OUTSIDE the 120° field** — and **65,884
(80.1 %)** fall outside the decode box. **The "nearest" agents include cars
BEHIND the ego.**

⛔ **Raising `n_queries` makes it WORSE, not better** (61.8 % → 63.1 % at N = 32).
Without a filter a monocular head is trained to hallucinate on ~62 % of its
supervision, and its AP would measure how well it guesses the unobservable.

⇒ **`filter_targets_to_visible` / `visible_target_filter` implemented, ON BY
DEFAULT in `agent_losses`, and both the pre- and post-filter `n` are reported** —
a panel that reports only the post-filter count cannot say how much supervision
the filter removed, and that fraction is the whole finding. Turning it off is the
deliberate-regression arm and must be asked for by name.

### And `n_queries` is now MEASURED, not inherited

Over the only honestly-trainable target set (**in-field ∩ decode box**):
mean **3.16**, p99 **19**, **max 24**.

| N | frames dropped | boxes dropped | nearest sacrificed |
|---|---|---|---|
| **16** (`agent_slots`' placeholder) | 2.16 % | 2.22 % | **38.5 m** |
| **24** | 0 % | 0 % | — |
| **32** (this seam's default) | 0 % | 0 % | — |

⇒ **32**: zero drops with headroom, 2,048 params (0.07 % of the 2–4 M band), and
DD uses 30 so the audit gap stays on its own axis. **16 is REFUTED.**

⚠️ **The docstring's own recipe — "use the p99" — is itself wrong**, and is not
what was applied. p99 leaves 1 % of frames dropping, and that 1 % is **not
random**: it is exactly the crowded frames, which is the flattering-on-hard-frames
failure the recipe exists to prevent. The rule used is **a covering N with ZERO
drop on the FILTERED set**.

⚠️ **Scope, stated:** measured on **val40**, not on the 2,376-episode train
corpus, which has no join on this box. The train distribution is **UNMEASURED**.
Also: `occ` is flagged at the **sensor's 120°**, not at whatever sub-frame the
encoder is fed (the v5f run's centred sub-frame measured 117°), so the in-field
set is an **upper bound**. And the crowded tail is concentrated: **all 85 frames
still over-N at 32 come from a single clip** — the tail is one traffic-jam
episode, not a corpus property, so its effective *n* is 1–3 episodes, not 7,400
frames.

### Class support — which per-class claims would be underpowered

`automobile` is **77.9 %** of the in-field target set. Underpowered for any
per-class claim (**< 5 clips**, so < 5 bootstrap clusters): **animal (1 clip),
stroller (2), bus (3), other_vehicle (3)**; `rider` (8) and `protruding_object`
(7) are marginal.

---

## The tiny-rig smoke — four arms, each reaching a checkpoint

MEASURED 2026-09-05, CPU, off-Drive mirror. Every arm ran to `summary.json`.
The v0-conditioned bank is built by `make_smoke_anchors.py` (⛔ **CI-ONLY**, a
coarse grid, never a registered arm).

```
# the CI-only bank the sampler needs (a fixed-path bank has zero controls)
python make_smoke_anchors.py /tmp/anchors20.pt --n-lon 5 --n-lat 4

# (a) baseline, every seam off
python stack/scripts/refc_v3_train.py --arm hier --out OUT --smoke --steps 3 \
  --device cpu --synth-episodes 4

# (b) agent tokens, ORACLE (needs no detector loss — the rung that runs FIRST)
  ... --agents oracle --agent-queries 8

# (c) the CONTROL-SPACE SAMPLER
  ... --anchors /tmp/anchors20.pt --anchor-v0-conditioned \
      --anchor-control-units alat --sampler ddim --w-u0 0.5

# (d) both together
  ... (c) --agents head --w-agent 1.0 --agent-queries 8 \
          --agent-w-project 0.2 --agent-w-ground 0.1
```

Arm (c)'s log row carries the new term live: `loss 29.0148 traj 1.2182
cls 8.5347 **u0 0.5004**`, and its `config.json['seams']` reads
`sampler: ddim · sampler_space: control · sampler_infer_t: 8 · sampler_steps: 2 ·
sampler_groups: 1 · control_norm: [4.0, 3.0] · w_u0: 0.5`.

### ⛔⛔ TWO DEFECTS THE SMOKE CAUGHT — both of the "runs fine, means nothing" family

**1. The trainer's own guard refused MY anchor builder, and was right to.** My
first `make_smoke_anchors.py` used `torch.linspace` over the asymmetric a_lon
range (−4.0 … +2.9167), which **does not pass through 0.0** — so the vocabulary
could not express *"hold speed, go straight"*. The trainer's existing check
states the cost: such a set reads **1.2768 m** oracle-in-vocabulary against
**0.2610 m** — a **4.9× artifact that looks exactly like a resolution finding**.
Fixed with `grid_through_zero`, which asserts 0.0 is a node on both axes.
⭐ Recorded as a class: **a refusal at wiring time beat a warning**, and it is
the only point at which this was cheap to catch.

**2. `--agents head --w-agent 1.0` TRAINED NO DETECTOR AND SAID NOTHING.** The
detection-loss branch was guarded `... and "agent_box" in batch`, and the
synthetic corpus carries no `obstacle.offline` join — so the run trained,
converged, wrote a checkpoint, and **stamped `w_agent: 1.0` in `config.json`
while the detector was never supervised**. The head would have been shaped only
by the planner loss through its token gate, and the arm would have read as *"the
learned agent head does not help"*.
⭐ **This is the SAME failure my own `--w-agent 0` guard refuses, one level
down** — there the loss weight is missing, here the **labels** are, and a guard
on the flag alone cannot see it because the flag was set correctly.
⇒ Now **refuses loudly** and names the wiring it needs. Verified by run: arm (d)
above now exits with that message, while `--agents oracle` still reaches a
checkpoint.

### ⚠️ THE HONEST GAP THIS LEAVES — the dataset side is NOT wired

`V3Dataset` does **not** load `obstacle.offline` into the batch. So today:

| arm | state |
|---|---|
| `--agents oracle` (`E-AGT-ORACLE` / `E-AGT-BUDGET`) | **model side complete**; still needs boxes fed per window |
| `--agents head` (`E-AGT-HEAD`) | **REFUSES** until the join is wired — correctly |

⇒ **The next rung is dataset work, not model work**: a `JoinFileReader` per
episode + `agent_slots.targets_from_join` + the visibility filter, emitting
`agent_box / agent_yaw / agent_cls / agent_valid` per window. Every consumer of
those keys is written, tested and waiting. ⚠️ And there is **no join file for
the 2,376-episode train corpus** on this box — only val40 — so that build is a
DataFlyWheel prerequisite, not a same-turn fix.

## Verification

| suite | result |
|---|---|
| the four refcv5 modules together | **92 passed, 1 skipped** |
| the whole `stack/tests` slice (refc · agent · anchor · kinematic · calib · bev · v4) | **930 passed, 30 skipped, 1 failed** |

The single red is `test_bev_consumer_fov.py::test_the_banked_p8_frame_is_recoverable_from_the_launch_chains`,
which asserts a **shell script exists**; the file is present in the repo
(1,406 bytes) and simply absent from the off-Drive mirror. It references nothing
in this build (control: the module has 28 tests and 0 `refc` references).

## Housekeeping done in passing

⛔ The **shared index carried 54 staged deletions, and 54 of 54 were PHANTOM** —
every path present in `HEAD` *and* non-empty on disk (the `read-tree`-from-a-stale-tree
failure). They included `REFCV5_DESIGN_PLAN.md`, both `Decisions/2026-09-05-*`
files, 13 `stack/tests/` files. **A pathspec-free commit would have deleted all
54.** Repaired with `git reset --` in batches of 25; **54 → 0**. *(Every commit in
this stream used `mktree_commit.py`, which seeds a private index from HEAD and is
structurally immune — which is the argument for making it the default.)*

## Log

- `2026-09-05` STATUS header banked before first line of code.
- `2026-09-05` component 1 (agent-token seam) banked; fabricated class enum caught.
- `2026-09-05` component 3 (control-space sampler) banked; bit-identity to `roll_bank` proven.
- `2026-09-05` components 2 + 4 banked; visibility filter added on the density finding.

## Deliverable manifest

All paths in the repo, branch `agent/arch-inf-20260803`, **committed** (staged and
blob-verified via `mktree_commit.py`). Nothing is stranded on a pod or a worktree.

| path | what |
|---|---|
| `stack/tanitad/refs/refc_agents.py` | the agent-token seam, monocular losses, oracle/budget, the visibility filter |
| `stack/tanitad/refs/refc_sampler.py` | DDIM schedule, timestep embedding, `roll_controls` |
| `stack/tanitad/refs/refc_selector.py` | `SubMetricSelector`, coarse/fine, four-family heads, composition |
| `stack/tanitad/refs/refc_selector_targets.py` | rule-based environment targets |
| `stack/tanitad/refs/refc.py` | `cross_agent` seam, control-space sampler branch, `_bank_speed`, model wiring |
| `stack/scripts/refc_v3_train.py` | flags, guards, seam stamp, `loss_u0`, agent losses |
| `stack/tests/test_refc_agents.py` | 31 tests |
| `stack/tests/test_refc_sampler.py` | 26 tests |
| `stack/tests/test_refc_selector.py` | 24 tests |
| `stack/tests/test_refc_v3_refcv5_wiring.py` | 11 tests |
| `…/2026-09-05-refcv5-build/measure_agent_density.py` | the density instrument |
| `…/2026-09-05-refcv5-build/raw/agent_density.json` | its output (distributions, decision tables, class histograms) |
| `…/2026-09-05-refcv5-build/make_smoke_anchors.py` | ⛔ CI-ONLY v0-conditioned bank so the sampler can smoke end to end |

## ⚠️ ESCALATIONS — for the Master Mind, not for a reader to find later

1. **`rl/rewards.py::_collision` tests WAYPOINTS, not SEGMENTS.** A fan whose
   waypoints straddle an obstacle passes through undetected. Shared with the RL
   stream, so a swept-segment fix is a cross-stream decision.
2. **256×640 fails `assert_fully_observed`** (~8.9 % masked on rig B, ~0.06 % on
   rig A — a **rig-correlated** black region). `176×624` is the measured
   rig-clean frame and is a bit-exact pixel slice of the existing PNG cache.
   refcv4b trains on 256×640 so the seam stays there for comparability, but a
   **detector inherits that rig-correlated mask**.
3. **The train-corpus agent density is UNMEASURED** (no join file for the 2,376
   episodes). `--agent-queries 32` rests on val40.
4. **WP-7's "≈ 3.5 M" is the ablation's param count, not the arm's** (4,530,444).
   The plan should be corrected.
5. **`refc_selector_train.py` and `refcv3_arm.py --selector` are NOT written** —
   WP-7's frozen-generator trainer and its dump path remain open.
7. ⛔ **THE NEXT RUNG IS DATASET WORK: wire `obstacle.offline` into the batch.**
   `V3Dataset` emits no `agent_box`, so `--agents head` correctly REFUSES and
   `E-AGT-HEAD` cannot train. Everything downstream of those keys is written and
   tested. Needs a train-corpus join (only val40 exists on this box) — a
   **DataFlyWheel prerequisite**, and it gates the PI's named priority.
6. **The selector's compliance τ has no default** (must come from
   `nav_compliance.derive_tolerance`, else the head is omitted), and the tactical
   head's **2 s label vs 6 s candidate** horizon mismatch is recorded in
   provenance as `horizon_mismatch: True` and **needs a ruling**.
