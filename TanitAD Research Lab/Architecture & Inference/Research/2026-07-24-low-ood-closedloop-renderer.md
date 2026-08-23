# The cheapest path to a low-OOD closed-loop renderer — a ranked survey

**2026-07-24 (Berlin), Architecture & Inference research stream, pod-free.** Prep for the closed-loop
program's next phase. **This feeds a Sayed decision; it does NOT auto-commit GPU-days.**

**Evidence-class legend (CLAUDE.md operating standard):** `MEASURED` (ours, artifact-cited) ·
`PUBLISHED` (external, arXiv/URL-cited) · `INHERITED` (another of our docs, not re-verified) ·
`HYPOTHESIS` (my inference, not measured). A claim that would decide a GPU-day is MEASURED or PUBLISHED.

**Reads before writing:** `RETRACTION_LOG.md` (C1–C6 + the 07-22/07-23 closed-loop entries),
`AGENT_OPERATING_STANDARD`, `MODEL_REGISTRY §0/§1.2/§1.4`, the three wall artifacts
(`…/2026-07-24-refccl/`, `…/2026-07-23-refc-planner-closedloop/`,
`Research/2026-07-23-closed-loop-wm-training-verdict.md`), the low-OOD instrument source
(`…/2026-07-23-lower-ood-closedloop-source/lowood_probe.py` + `lowood_closedloop.py`), the AlpaSim
real-time/scenario notes (`…/2026-07-22-alpasim-closedloop-evalpod/`), the IDM pipeline
(`…/2026-07-24-idm-*`), and `LOOP_STATE.md` (LowOOD / LowOODhard / LaneKeep / G1clean streams).

---

## 0. TL;DR — the ranking, and the one reframe that changes the answer

**The reframe (the load-bearing idea of this doc, `HYPOTHESIS` grounded in our MEASURED artifacts):**
"low-OOD **AND** reactive-agent safety **AND** on-policy rollouts, together" is **two problems, not one**,
and every off-the-shelf sim (AlpaSim, HUGSIM) fuses them into one monolithic renderer and therefore pays
one monolithic OOD tax. Decoupled, they are:

- **(A) On-policy road-keeping / drift-recovery** — the ego drifts off its own path and must recover.
  This is the **D2 + RefcCL + Gate-1(b)** failure. It needs low-OOD + on-policy, **but NOT reactive
  agents** (a car drifting off a straight road departs with zero other actors). **We already own a
  faithful low-OOD on-policy instrument for this** (the P2 harness, MEASURED ≤1.5× OOD on 100 % of
  windows). It is currently an *eval* loop; promoting it to a *training* loop is the cheapest lever in
  the program and needs **no renderer at all**.
- **(B) Reactive-agent safety** — off-road/collision *under the pressure of agents that react to the
  ego's counterfactual position*. This is the **Gate-1(a)** failure and the *only* place the renderer
  gap is truly binding, because it needs (i) agents that re-pose off the log and (ii) large evasive ego
  excursions — both of which break the real-footage instrument's appearance fidelity.

**Ranked recommendation (cheapest first):**

| # | path | what it buys | build cost | verdict |
|---|---|---|---|---|
| **1** | **Promote our own low-OOD P2 harness eval→training** (Path 4, renderer-free) | closes sub-problem **(A)** at ~1.05× OOD | **~1 eval-pod-day, 0 new infra** | ⭐ **DO FIRST.** The D2/RefcCL "escape" the walls named *already exists as our instrument* |
| **2** | **Encoder-side OOD alignment to NuRec** (Path 5b) | shrinks the **3.2–3.75×** we already pay on AlpaSim, from the *model* side | ~1–2 eval-pod-days | ⭐ cheap complement; unlocks every reconstruction source incl. (B) |
| **3** | **Reactive-agent overlay on real frames** (Path 3, our `obstacle.offline` + IDM + GeoSim-style compositing) | adds sub-problem **(B)** while keeping real background | 1–3 eng-weeks | the right **(B)** path *inside the small envelope*; medium cost |
| **4** | **Adopt/port a closed-loop 3DGS sim — HUGSIM or AlpaSim's OmniDreams backend** (Path 1/2) | full **(B)** with large envelope + reactive agents | 2–4 eng-weeks + per-scene recon | bounded upside (EUVS wall); worth a **cheap OOD probe** before committing |
| **5** | **Off-the-shelf non-reconstruction sims** (CARLA/MetaDrive/Waymax/nuPlan) ± Cosmos-Transfer skin (Path 2/5a) | reactive agents + map + GT, at high appearance OOD | high (Cosmos-Transfer is a 7B model) | **fails our binding constraint** for a pixel encoder; useful only *decoupled* (borrow their behavior models) |

**The cheapest discriminating experiment** (fully pre-registered in §7): turn the existing P2 harness into a
**RoaD/CAT-K on-policy training loop** and test whether the closed-loop-*consistent* objective escapes the
departure↔ADE Pareto trade that the single-step *synthetic* objective (D2/RefcCL) could not. Cost **~1
eval-pod-day, 0 new renderer**. It either closes half the wall for free or proves the trade is deeper than
the objective — both outcomes decision-useful.

---

## 1. The wall, stated as measurement (why a renderer is the binding constraint)

Three MEASURED results converge (`MEASURED`, artifacts cited):

- **Gate-1** (`…/2026-07-23-gate1-clean-run/`, `…/2026-07-23-gate0-freefloor/`): all three free
  inference-time floors (cost-guided **selection**, per-denoise **gradient** synthesis, **WM-MPC**) leave
  junction off-road unchanged (Gate 0b junction 0.73→0.73 [0,0]); the plan is on-road yet the ego departs
  → only closed-loop-aware **training** shapes the executed path. But the on-policy states are NuRec
  reconstructions (**3.2×** OOD) → a fine-tune on them partly targets *reconstruction*-OOD.
- **D2** (`…/2026-07-23-refc-planner-closedloop/RESULTS.md`): in-envelope synthetic recovery augmentation
  halves held-out corridor departures (0.0174→0.0085, junctions +0.0333 SEP) but ADE goes
  0.587→0.875 (SEP-worse). The full decoder-only sweep (naive/g1/g2/g3 + g2s1/g2s2 speed-term) is
  **Pareto-bound** — no config holds departures AND recovers ADE.
- **RefcCL** (`…/2026-07-24-refccl/RESULTS.md`): unfreezing the encoder (material move, feat_cos 0.966,
  integrity canary holds) does **not** dissolve the trade — it slides along the same Pareto curve
  (pre-registered branch **c**). ⇒ the trade is **intrinsic to the single-step, in-envelope, synthetic
  objective**, not a parameter-subset artifact. The named escape: *"a closed-loop-CONSISTENT objective —
  train on the planner's OWN accumulated on-policy drift (RoaD/CAT-K), which needs a faithful low-OOD
  renderer to generate non-self-referential on-policy rollouts."*

**The instrument gap, as the clean Gate-1 run states it verbatim** (`MEASURED`, `LOOP_STATE` G1clean):
"the low-OOD source is MAP-FREE + AGENT-FREE → it structurally emits only drift/deviation, **NEVER
off-road/collision** … 'low-OOD' and 'junction off-road/collision' are **mutually exclusive with existing
instruments** — reactive-agent safety needs a SIM (→ reconstruction OOD; AlpaSim renders via NuRec = the
same 3.2×), low-OOD needs REAL footage (→ no reactive agents)."

⚠️ **C5/C6 discipline carried in:** the closed-loop win/loss ordering (REF-C > flagship) is confirmed on
three independent instruments (n=1→n=12 AlpaSim→n=40 real-footage) and is a **within-source RELATIVE**
read, not a real-world safety rate. Nothing below promotes a within-sim number to an absolute rate.

---

## 2. Our assets — what the survey must fit to (all `MEASURED` unless noted)

| asset | what it is | key numbers | artifact |
|---|---|---|---|
| **Real-footage low-OOD instrument** | arc-length re-index of the recorded frame manifold + a **ground-plane homography** for the residual (dlat, dψ); ego drives a bicycle, obs follows on-policy | Δ=0 open-loop ADE **0.4045** [0.313,0.515] (reproduces real 0.4271, recon-OOD=0); **on-policy** OOD peak longitudinal **1.017×**, junction **1.190×**, **100 % ≤1.5×**; envelope ≤1.16× across ±3 m/±12° | `lowood_probe.py`, `lowood_closedloop.py`, `LOOP_STATE` LowOOD/LowOODhard |
| **`corridor_departure_rate` metric + REF-C arm** | first absolute low-OOD closed-loop comparison | REF-C 0.564 vs flagship 1.488 ADE@2s; dep 0.0134 vs 0.0318; both at **1.02–1.20×** OOD | `…/2026-07-23-lowood-lanekeeping-refc/` |
| **`obstacle.offline`** | **real 3D agent tracks on 96.90 % of our corpus** — 36 features/track, our ingest currently reads 2 | (the mis-declared-nonexistent asset; RETRACTION C2 07-21) | `MODEL_REGISTRY`, RETRACTION_LOG |
| **IDM dynamics-labeler** | frozen-v1 encoder → multi-domain IDM head → per-frame speed/yaw pseudo-labels; **GO decision-grade** | pseudo captures **96–109 %** of the real-label ceiling (parity 4-seed) | `…/2026-07-24-idm-parity-validation/`, `…-idm-downstream-ablation/` |
| **AlpaSim on eval pod** | NVIDIA Alpamayo closed-loop microservice sim (Apache-2.0), **NuRec** + **OmniDreams** renderer backends, **reactive TrafficSim** | ~**0.8×** real-time @480×854, **0.29×** native (renderer-bound); reconstruction OOD **3.2–3.75×** | `…/2026-07-22-alpasim-closedloop-evalpod/`, `MetaDrive/AlpaSim verdict` |
| **v1 encoder** | trained ViT, in-dist ADE 0.452 (≫ frozen-DINO 2.13–2.92); **safely fine-tunable** (RefcCL canary holds at feat_cos 0.966) | — | `MODEL_REGISTRY §1.2`, RefcCL |
| **Parity corpus** | PhysicalAI-AV front-wide, 2,376 eps, key `e438721ae894`, **two rigs** (fisheye), + comma2k19 | — | `MODEL_REGISTRY §0.1` |

**The single most important asset fact:** the real-footage instrument's low OOD comes from **appearance
fidelity, not pose-exactness** — it stays ≪ NuRec because it shows a *real* frame re-indexed to the ego's
arc-length and only *warps* the small residual. That is also its ceiling: a **flat-road homography
under-models 3D parallax** (the source itself flags the lateral envelope as *optimistic*), so it degrades
once the ego leaves the road plane or deviates far (>~2–3 m / >12°), and it cannot **move other agents**
(they are baked into the real pixels at their logged positions). Every path below is really a proposal for
*how to extend appearance fidelity to off-log ego poses and to re-posed reactive agents*.

---

## 3. Path-by-path survey

Each path: **what it is · concrete tools/citations · feasibility · OOD vs our distribution · reactive-agent
support · build cost · honest failure.**

### Path 1 — Better neural reconstruction renderers (beyond NuRec)

**What it is.** Replace/augment NuRec's gsplat reconstruction with a renderer whose novel-view synthesis
degrades *less* as the ego leaves the recorded trajectory — the exact axis of the 3.2× tax.

**Concrete tools / citations (`PUBLISHED`):**
- **HUGSIM** (arXiv:2412.01718) — real-time (**>30 FPS**), photorealistic, **closed-loop** 3DGS AV
  simulator. Directly targets **viewpoint extrapolation** (multi-plane ground modeling + unicycle motion
  constraints) and **360° actor rendering**; reactive traffic via **IDM** *or* adversarial "attack"
  planning. Ingests KITTI-360/Waymo/nuScenes/PandaSet multicam. Open-source-intended. **The single most
  turnkey full-stack candidate.**
- **StreetCrafter** (arXiv:2412.13188) — **LiDAR-conditioned video diffusion**; uses LiDAR point renderings
  as pixel-level conditions to keep camera control while a generative prior fills off-trajectory content.
  Real-time, dynamic editing by moving LiDAR points. The leading *generative-hybrid* extrapolation method.
- **FreeSim / ReconDreamer / ReconDreamer++** — hybrid generative-reconstruction and progressive training
  that add generated off-trajectory views back into the reconstruction to support **multi-lane shifts**.
- **ViSE** (arXiv:2510.18341), **DriveExplorer** (arXiv:2512.23983), **ConFixGS** (arXiv:2605.09688),
  **AutoSplat** — 2025 vision-only / confidence-diffusion / constrained-gsplat refinements of the same
  extrapolation target.
- **OmniDreams** (arXiv:2606.03159) — NVIDIA's **generative** real-time world-model renderer, the *other*
  AlpaSim backend. Because it is generative (hallucinates plausible off-log content) rather than a fixed
  reconstruction, its on-policy-deviation OOD may differ from NuRec's — a **within-AlpaSim swap** we can
  test at near-zero integration cost.
- **Benchmarks:** **EUVS** (arXiv:2412.05256) and **XLD** (arXiv:2406.18360) measure exactly this
  extrapolation degradation.

**OOD vs our distribution.** This is the crux, and the published verdict is sobering. EUVS
(`PUBLISHED`, fetched): *"current NVS methods are prone to overfitting to training views,"* and — decisively
— *"incorporating diffusion priors and improving geometry **cannot fundamentally improve** NVS under large
view changes … the need is for more robust approaches and **large-scale training**."* So the frontier
methods **reduce** the extrapolation gap but the benchmark authors state incremental architecture does not
**close** it. Whether any hits **≪3.2×** on *our* fisheye corpus is **unmeasured** and, per EUVS,
unlikely to reach ~1× at large lateral offsets.

**Reactive agents.** HUGSIM: yes (IDM + attack planner + 360° actor rendering). NuRec/OmniDreams via
AlpaSim TrafficSim: yes.

**Build cost.** Medium-high: a **per-scene reconstruction pipeline** (multicam + ideally LiDAR), a renderer
service, and — for a *training* source — enough scenes to matter. HUGSIM/OmniDreams are the least-effort
because they are pre-built; a bespoke StreetCrafter-style renderer is a multi-week research build.

**Honest failure.** (i) EUVS says the extrapolation OOD is **not fundamentally closable** by better
reconstruction alone — so this path most likely takes 3.2× → maybe ~1.5–2×, not → ~1×; our *own*
real-footage instrument already sits at **1.05×** for on-log-adjacent poses, so a renderer only wins where
the ego deviates **beyond the homography envelope**. (ii) Our fisheye rig + 2-rig structure is off the
KITTI/Waymo/nuScenes distribution these renderers are tuned on. (iii) It does nothing about the
**model-side** gap that Path 5b attacks more cheaply. **Verdict: not the cheapest; a bounded-upside swap.
Worth a cheap OOD probe (OmniDreams-vs-NuRec, HUGSIM-on-our-scenes) before any commitment, never as the
lead.**

### Path 2 — Off-the-shelf closed-loop sims (CARLA, MetaDrive, Waymax, nuPlan, Nocturne, Bench2Drive)

**What it is.** Adopt a maintained closed-loop simulator with built-in reactive traffic.

**Concrete tools / citations (`PUBLISHED`):**
- **CARLA** (arXiv:1711.03938) — full UE4/UE5 sim, reactive **Traffic Manager**, map + off-road/collision
  GT. **Bench2Drive** and **Bench2Drive-R** (arXiv:2412.09647) are its closed-loop E2E harnesses. Our
  render blocker is **cleared** (RETRACTION C2 07-21: Vulkan ICD was in `/etc/vulkan/icd.d/`).
- **MetaDrive** (arXiv:2109.12674) — lightweight procedural sim, **reactive IDM** agents; **GO from source**
  on our stack (wrapper landed `stack/tanitad/data/metadrive_env.py`, `MetaDrive verdict` note).
- **Waymax** (arXiv:2310.08710) — JAX, hardware-accelerated, data-driven over Waymo Open Motion; reactive
  (IDM + log agents); **state/BEV only — does NOT render sensor data.**
- **nuPlan** (arXiv:2106.11810) — large real-world closed-loop planning benchmark, **reactive IDM**
  background agents, state/BEV, **no camera rendering** in the loop.
- **Nocturne** (arXiv:2206.09889) — 2D, Waymo-derived, partial-observability, **state-space, no pixels.**

**OOD vs our distribution — why we avoid them, made precise.** There is a clean split:
- The **reactive-agent-rich** sims (Waymax, nuPlan, Nocturne, MetaDrive-lite) are **state/BEV** → **zero
  appearance overlap** for our pixel ViT encoder = *infinite* appearance OOD. Perfect for a *state-space*
  planner; useless as an observation source for our WM.
- The one **photoreal-ish** sim (CARLA) carries **large synthetic-to-real appearance OOD** (game-engine
  textures/lighting vs real dashcam), and none of them is **fisheye** like our rig.

**Reactive agents.** All strong (that is their entire point). This is the asset to **borrow** even when the
sim itself is rejected: their IDM/behavioral controllers are exactly what Path 3/4 need — and we have
already **validated an IDM labeler of our own** (§2).

**Build cost.** Low to adopt any one; but the cost that matters (closing appearance OOD) is unbounded for a
pixel encoder.

**Honest failure.** These **fail the binding constraint directly** — none matches our real dashcam/fisheye
*pixels*. **Verdict: do not adopt as an observation source. Do harvest their reactive-behavior models
(IDM) for Paths 3/4, decoupled from their renderers.** (This decoupling — render separate from behavior —
is precisely what HUGSIM/Cam2Sim/AlpaSim do architecturally.)

### Path 3 — Hybrid: real footage + reactive-agent overlay ⭐ (the right (B) path inside the envelope)

**What it is.** Keep the **real** frame (appearance OOD ~0 for the background) and **composite reactive
agents** into it — re-posed off the log so they respond to the ego's counterfactual position. This is the
only way to get sub-problem **(B)** without paying the full reconstruction tax on the *background*.

**Concrete tools / citations (`PUBLISHED`):**
- **GeoSim** (arXiv:2101.06543) — the canonical recipe: geometry-aware **composition** of reconstructed
  vehicle assets into real images with **occlusion handling + neural in-painting** for seamless blending.
- **CADSim** (arXiv:2311.01447) — robust in-the-wild 3D asset reconstruction for controllable sensor
  insertion (Waabi).
- **OmniRe** (arXiv:2408.16760) — holistic gaussian **scene-graph** with per-actor nodes → re-pose/insert
  actors (incl. pedestrians via SMPL) in a reconstruction.
- **Cam2Sim** (arXiv:2607.04770) — neural scenario reconstruction that **explicitly decouples sensor
  rendering from behavior rollout** (a separate behavioral controller drives reactive agents) — the
  cleanest statement of the architecture we want.
- **Sim-on-Wheels** (arXiv:2306.08807) — inserts virtual reactive actors into a **real** camera stream
  (the physical-world-in-the-loop limit of this path).

**Our-asset fit — unusually strong.** We do **not** need to synthesize agents from scratch: `obstacle.offline`
gives **real 3D tracks on 96.90 %** of the corpus. We can (a) reconstruct real agent crops/gaussian assets
from their own frames (in-distribution appearance by construction), (b) **re-time/re-place** them with an
**IDM controller** (validated, §2) so they react to the ego, and (c) composite via a GeoSim-style
occlusion-aware blend. The background stays a real frame from the low-OOD instrument.

**OOD.** Background ~1.0× (real). The **new** OOD is the *inserted agent* (compositing/relighting/shadow
artifacts) — bounded and localized, far below a full-scene reconstruction's 3.2×. Ego-viewpoint OOD stays
whatever the homography instrument gives (≤1.16× inside the envelope).

**Reactive agents.** Yes — this is the path whose *entire purpose* is (B).

**Build cost.** Medium-high (1–3 eng-weeks): asset reconstruction from `obstacle.offline` + a compositor
with occlusion/inpainting + IDM reactivity + a scoring layer. **The hidden cost:** to insert a *reactive*
agent you must first **inpaint out the real logged agent** it replaces (else double-agents), and large ego
evasive maneuvers still leave the homography envelope → the background then needs a real reconstruction
(collapsing back toward Path 1).

**Honest failure.** Bounded to the **small ego-deviation envelope** (inherits the homography limit); the
inserted-agent appearance is a new (small) OOD; compositing consistency (shadows, inter-reflections) is
imperfect. But it is the **highest-fidelity, lowest-OOD way to get reactive-agent (B) metrics** we have,
and it reuses assets we already own. **Verdict: the recommended (B) path once (A) is banked — rank 3.**

### Path 4 — Log-replay + counterfactual reactive agents, NO full rendering ⭐⭐ (the cheapest; §7 lives here)

**What it is.** The RoaD/CAT-K on-policy training *signal* **without pixels for the agents** — the ego's
observation stays the low-OOD real frame; reactive agents enter as **relational state** (boxes/velocities
from `obstacle.offline`, re-posed by IDM); off-road and collision are scored **geometrically** (ego box vs
agent box; ego vs the recorded-path corridor). This is our existing P2 harness **promoted from eval to
training**, plus an agent-state channel.

**Concrete tools / citations (`PUBLISHED`):**
- **CAT-K** (arXiv:2412.05334, NVlabs/catk, **CVPR 2025 Oral**) — Closest-Among-Top-K rollouts: unroll the
  policy, at each step supervise toward the GT-mode-closest of the top-K, keeping the on-policy rollout
  on-manifold. **7 M CL-tuned beats 102 M open-loop** on the Waymo Sim Agents leaderboard. **Trajectory/
  token space — no rendering.**
- **RoaD** (arXiv:2512.01993) — "**Rollouts as Demonstrations**" closed-loop SFT. Evaluated on the
  **Physical AI AV NuRec dataset** — *our exact corpus family* — 10 Hz replan, 20 s, 574 scenes:
  **+41 % driving score, −54 % collisions**; Alpamayo-1.5 OmniDreams-FT collisions 6.9 %→4.2 %. The direct
  **published precedent** that closed-loop SFT via rollouts-as-demonstrations works on our data ecosystem.
  ⚠️ its own sim2sim degradation 0.75→0.58 is the reconstruction-OOD caveat — the reason we want to run the
  *same recipe on the ~1× real-footage source instead of NuRec's 3.2×.*
- Our **P2 harness** (`lowood_closedloop.py`) already implements the faithful low-OOD on-policy rollout
  (real frame + homography warp of the on-policy residual, `grid_sample` — differentiable). Our **IDM
  labeler** supplies reactive agent kinematics; `obstacle.offline` supplies the tracks; `corridor_departure`
  supplies the road-keeping score.

**OOD.** Ego observation **1.05×** (MEASURED, the instrument we own). **This is the single biggest OOD win
available** and it needs **no renderer**.

**Reactive-agent support — the honest boundary.** Two sub-modes:
- **(A) road-keeping / drift-recovery:** needs **no agents** → fully served *today*. This is the D2/RefcCL
  escape, and it is the §7 experiment.
- **(B) collision safety:** agents enter as **state** for *scoring* and for a planner that has an
  agent-relational **input**. ⚠️ **Our v1/REF-C encoder is pixel-only and barely uses agent state**
  (lead-state gate: **+1.16 % [CI∋0]**, RETRACTION 07-21) → a *state-only* reactive agent is invisible to
  the current encoder, so (B)-as-state trains a planner to react to something it cannot see. (B) therefore
  requires either an added agent-state input (architecture change) or the pixel compositing of **Path 3**.

**Build cost.** **Lowest in the survey for (A):** ~1 eval-pod-day to convert the existing eval harness to a
training loop (the rollout, warp, and CI machinery already exist and are hardened). Adding (B)-scoring
(box-overlap collision + IDM reactivity from `obstacle.offline`) is a further ~1 eng-week and is
**renderer-free**.

**Honest failure.** Cannot train the **encoder's agent-relational *perception*** (agents aren't in the
ego's pixels) — that is Path 3's job. Bounded to the homography envelope for large ego excursions. The
self-referential-imagination trap is **avoided** (unlike the MEASURED **DAGGER_HURTS** result on the
WM's-own-imagined states — the P2 source is *real footage*, not free imagination). **Verdict: rank 1 for
(A); the substrate for (B) once Path 3 adds agents to pixels.**

### Path 5 — Sim-to-real domain adaptation (reduce a source's OOD toward our distribution)

Two distinct flavors; they rank very differently.

**5a — Renderer-side: re-skin a synthetic sim into real appearance (`PUBLISHED`).**
**Cosmos-Transfer 2.5** (NVIDIA WFM, world2world transfer) takes CARLA's RGB/segmentation/depth/edge maps +
a text prompt (weather/location) → photoreal driving video; the **CARLA integration is official**
(carla.readthedocs.io `nvidia_cosmos_transfer`; Cosmos Cookbook CARLA SDG augmentation). **DriveCtrl**
(arXiv:2605.15116) and **LSD-3D** (arXiv:2508.19204) are peers. This gives reactive agents + map +
off-road/collision GT (from CARLA) *and* a photoreal skin. **But:** Cosmos-Transfer is a **7 B** model
(heavy), it adds **its own generative OOD**, "structural preservation" ≠ our **fisheye** rig, and it does
not close the *pose*-extrapolation problem (it re-skins whatever CARLA renders). **Verdict: a heavier
alternative to Path 1/3, not a cheaper one.**

**5b — Encoder-side: make v1 robust to the renderer's OOD instead of fixing the renderer ⭐ (`HYPOTHESIS`,
cheap, high-leverage).** Invert the problem: rather than drive NuRec's 3.2× → 1×, fine-tune **v1's encoder**
to **align real↔reconstructed features** on **paired** frames (AlpaSim renders every scene from a real log,
so real/reconstructed pairs are free). A feature-consistency loss (cosine/L2 on pooled + fmap features,
frozen-real teacher) pulls the encoder's response to a NuRec frame toward its response to the source real
frame. **RefcCL MEASURED that v1's encoder is safely fine-tunable** (material last-stage move, canary
holds), so this is low-hazard. It **directly shrinks the 3.2× we already pay** and thereby de-risks Gate-1
*on NuRec* and unlocks every reconstruction source (Path 1) and AlpaSim (B)-metrics **without a new
renderer.** **Build cost ~1–2 eval-pod-days.** Honest failure: it narrows but will not zero the gap (the
reconstruction destroys information the encoder cannot re-hallucinate); it treats the symptom (encoder
sensitivity) not the cause (reconstruction fidelity). **Verdict: rank 2 — the cheapest complement, run
alongside Path 1 (as a probe) and before any Gate-1-on-AlpaSim.**

---

## 4. Cross-cutting verdict — the three requirements are separable, so buy them separately

| requirement | cheapest source we have / can build | OOD | cost |
|---|---|---|---|
| low-OOD observation, on-policy | **P2 real-footage instrument** (own) | **1.05×** MEASURED | owned |
| + road-keeping / drift training signal | **P2 promoted eval→training** (Path 4-A / RoaD/CAT-K) | 1.05× | ~1 pod-day |
| + reactive-agent collision **scoring** | `obstacle.offline` + IDM + box-overlap (Path 4-B, state) | 1.05× obs, geometric score | ~1 wk |
| + reactive-agent collision **perception** | GeoSim-style overlay on real frames (Path 3) | ~1× bg + small insert-OOD | 1–3 wk |
| + **large-envelope** ego excursion + agents | closed-loop 3DGS sim (HUGSIM / OmniDreams) ± encoder-align (5b) | ~1.5–2× (EUVS-bounded) | 2–4 wk |
| shrink the AlpaSim 3.2× we already pay | **encoder-side alignment** (Path 5b) | 3.2×→? (narrows) | ~1–2 pod-days |

**The staircase:** we already stand on step 1 (MEASURED). Steps 2 and the 5b complement are cheap and
renderer-free. Only the top step (large-envelope reactive safety) needs a real renderer, and EUVS says even
the best one is bounded — so we should reach it **last** and **only if** steps 2–4 leave a residual that
provably needs it.

---

## 5. Where each path fails (single honest paragraph each)

- **Path 1** fails the *magnitude* test: EUVS (`PUBLISHED`) says better reconstruction **cannot
  fundamentally** close extrapolation OOD, and our fisheye corpus is off-distribution for these renderers —
  expect 3.2×→~1.5–2×, not ~1×.
- **Path 2** fails the *binding constraint*: reactive-agent sims are pixel-free (∞ appearance OOD for our
  encoder); CARLA is photoreal-synthetic, not real, and not fisheye.
- **Path 3** fails on *envelope*: it inherits the homography's small ego-deviation limit and adds a (small)
  inserted-agent OOD, and it needs inpainting of the replaced real agent — real engineering, not free.
- **Path 4** fails on *encoder perception of agents*: state-only reactive agents are invisible to a
  pixel-only encoder that barely uses agent state — so (B) needs Path 3's pixels or an architecture change.
- **Path 5a** fails on *cost/appearance*: a 7 B re-skinner adds its own OOD and is not fisheye; **5b** fails
  on *ceiling*: it narrows but cannot zero the reconstruction gap (lost information can't be re-encoded).

---

## 6. Recommendation to Sayed (ranked, with the reasoning compressed)

1. **Promote the low-OOD P2 harness eval→training and run the §7 experiment (Path 4-A).** It is the
   D2/RefcCL "escape" the walls explicitly named, it needs **no renderer**, it runs on assets and code we
   already hardened, and it either closes the road-keeping half of the wall or proves the trade is deeper —
   both decision-useful. **~1 eval-pod-day.**
2. **In parallel, an encoder-side NuRec-alignment probe (Path 5b).** Cheap, low-hazard (RefcCL-proven),
   shrinks the 3.2× we *already* pay and de-risks every reconstruction-based (B) metric. **~1–2 pod-days.**
3. **If (A) succeeds and a reactive-agent (B) residual remains → build Path 3 on `obstacle.offline` + IDM.**
   The highest-fidelity, lowest-OOD reactive-safety instrument available, inside the envelope. **1–3 wk.**
4. **Only for large-envelope reactive safety → probe HUGSIM / AlpaSim-OmniDreams (Path 1), gated on a cheap
   OOD measurement first** (does OmniDreams or HUGSIM beat NuRec's 3.2× on *our* scenes, open-loop-vs-known
   control, the RETRACTION-C6 protocol?). Adopt only if the probe clears a pre-set bar. **2–4 wk.**
5. **Do not** adopt an off-the-shelf sim as an observation source; **do** harvest its IDM/behavior models
   (Path 2, decoupled).

**Standing-constraint note:** the AlpaSim (B) work and the Path-1 renderer probes both touch the eval pod's
NuRec stack; none of them needs the training pods, and none is blocked by the live `Sayood/` HF-storage 403
(that blocks *pushes*, not eval reads). The §7 experiment does not touch any training pod.

---

## 7. The cheapest discriminating experiment (pre-registered, both outcomes committed)

**Name:** `LOWOOD-CL-TRAIN` — closed-loop-consistent recovery on the real-footage low-OOD source.

**Question it decides.** D2 + RefcCL proved the departure↔ADE trade is intrinsic to the **single-step,
synthetic-perturbation** objective. They named the escape as a **closed-loop-consistent (on-policy
accumulated-drift) objective**, and asserted it *"needs a faithful low-OOD renderer."* **We already have
that renderer — it is the P2 harness (MEASURED 1.05× OOD, renderer-free).** So the decisive, cheap question
is: **does a RoaD/CAT-K on-policy recovery objective, trained on our real-footage low-OOD rollouts, escape
the Pareto trade that the synthetic single-step objective could not?**

**Setup (reuses hardened code; no new renderer, no training pod).**
- Source: `lowood_closedloop.py` promoted eval→training — roll the deployed REF-C planner (the anchored-
  diffusion head, per Stream D "do not change it") on-policy on the real-footage arc-length-re-indexed +
  homography-warped observation; at each visited (deviated) state, supervise with the **CAT-K target** (the
  top-K-closest-to-GT plan, per arXiv:2412.05334) + a **RoaD rollouts-as-demonstrations** imitation term
  toward the GT path 0.5–2 s ahead (labels already collected, G1prep `ab3ecfce`).
- Guards: gate every ckpt on the **operative-rollout WM canary** (RefcCL protocol; v1 baseline 0.452) so
  the WM cannot be silently degraded; **decoder-only first** (frozen encoder = WM-safe), then the
  RefcCL-cleared **encoder-in-loop** variant only if decoder-only is promising.
- Held-out: episode-disjoint 28:40 (12 eps / 264 windows), **paired episode-cluster bootstrap**
  (`taniteval/ci.py`) — identical to D2/RefcCL so the comparison is clean (C6).
- Envelope honesty: report the fraction of on-policy training windows inside the MEASURED ≤1.16× envelope;
  windows that leave it are flagged (the homography-optimism bound), not silently trained on.

**Pre-registered outcomes (both committed in advance):**
- **WIN** — overall held-out `dCDR` ≥ +0.005 & CI∌0 (departures still cut) **AND** `dADE` CI∋0 (ADE back
  within noise of base 0.587) **AND** canary holds. ⇒ the closed-loop-*consistent* objective **escapes** the
  single-step Pareto trade → the D2/RefcCL wall is a property of the *objective*, and the **real-footage
  low-OOD source is a sufficient training renderer for road-keeping (A)**. Greenlight scaling (A);
  reactive-agent (B) becomes the *only* remaining renderer question → proceed to Path 3.
- **BOUND** — `dADE` stays CI-separated-worse (as in every D2/RefcCL config) even with the on-policy
  objective. ⇒ the trade is **deeper than the objective** — it is intrinsic to real-footage lane-keeping
  without a map / large-excursion rendering → the cheap instrument is exhausted for (A), and the renderer
  paths (1/3) become **necessary**, not merely nicer. This *justifies* the Path-1 spend that would
  otherwise be premature.
- **CANARY-DEGRADE** (encoder-in-loop variant only) — back off per RefcCL branch (b); decoder-only result
  stands.

**Cost:** ~1 eval-pod-day (rollout + FT + paired eval + canary), **zero new renderer, zero training-pod
touch.** **This is the cheapest experiment in the program that can move the closed-loop question**, and it
is a strict extension of code already MEASURED-validated this session.

**Why this is the right discriminator (not a Path-1 renderer bake-off):** the walls did not fail for lack
of a *large-envelope* renderer — they failed on the **objective** (single-step vs on-policy-consistent),
and we can vary *only that* on a source whose OOD is already MEASURED at 1.05×. Running a renderer bake-off
first would confound "objective" with "renderer OOD" (C6) and cost 10–100× more. Decouple, cheapest lever
first.

---

## 8. Deliverable manifest

| artifact | where it lives | status |
|---|---|---|
| **This survey** | `TanitAD Research Hub/Architecture & Inference/Research/2026-07-24-low-ood-closedloop-renderer.md` (repo) | **staged** (`git add`), not committed, not pushed |
| Ranked recommendation (§0/§6) + pre-registered experiment (§7) | this file | complete |
| Source artifacts read (walls, instrument, assets) | paths cited inline §0-§2 (all pre-existing in repo) | unchanged, read-only |

**No pods touched. No GPU spent. No commit, no push** (STAGE-NEVER-PUSH). No sub-agent fan-out. The §7
experiment is a **proposal** gated on Sayed's go; it does not auto-launch.

---

## 9. Sources

**Ours (`MEASURED`):** `…/2026-07-24-refccl/RESULTS.md` · `…/2026-07-23-refc-planner-closedloop/RESULTS.md` ·
`Research/2026-07-23-closed-loop-wm-training-verdict.md` ·
`…/2026-07-23-lower-ood-closedloop-source/{lowood_probe.py,lowood_closedloop.py,P1_DECISION_GRADE_FINDINGS.md}` ·
`…/2026-07-23-lowood-lanekeeping-refc/` · `…/2026-07-22-alpasim-closedloop-evalpod/scenario_and_realtime_NOTE.md` ·
`…/2026-07-24-idm-{parity-validation,downstream-ablation,pipeline-derisk}/` · `MODEL_REGISTRY §0/§1.2/§1.4` ·
`RETRACTION_LOG` (C1–C6, 07-21 lead-state / 07-22 REF-C-OOD / 07-23 flagship-vs-REF-C) · `LOOP_STATE.md`.

**External (`PUBLISHED`):**
- Reconstruction / NVS: EUVS benchmark arXiv:2412.05256 · XLD arXiv:2406.18360 · HUGSIM arXiv:2412.01718 ·
  StreetCrafter arXiv:2412.13188 · OmniDreams arXiv:2606.03159 · OmniRe arXiv:2408.16760 ·
  ViSE arXiv:2510.18341 · DriveExplorer arXiv:2512.23983 · ConFixGS arXiv:2605.09688 · FreeSim / ReconDreamer
  (named, 2024–25).
- Closed-loop training: CAT-K arXiv:2412.05334 (CVPR'25 Oral, github.com/NVlabs/catk) ·
  RoaD arXiv:2512.01993 (Physical AI AV NuRec, +41 %/−54 %).
- Hybrid / composition: GeoSim arXiv:2101.06543 · CADSim arXiv:2311.01447 · Cam2Sim arXiv:2607.04770 ·
  Sim-on-Wheels arXiv:2306.08807 · Bench2Drive-R arXiv:2412.09647.
- Sims: CARLA arXiv:1711.03938 · MetaDrive arXiv:2109.12674 · Waymax arXiv:2310.08710 · nuPlan
  arXiv:2106.11810 · Nocturne arXiv:2206.09889.
- Sim-to-real: NVIDIA Cosmos-Transfer 2.5 (carla.readthedocs.io/en/latest/nvidia_cosmos_transfer) ·
  DriveCtrl arXiv:2605.15116 · LSD-3D arXiv:2508.19204 · Challenger arXiv:2505.15880.
- AlpaSim/Alpamayo ecosystem: github.com/NVlabs/alpasim · developer.nvidia.com/blog (closed-loop post-train).

⚠️ **Citation hygiene:** arXiv ids for EUVS, HUGSIM, StreetCrafter, CAT-K, RoaD, OmniDreams were confirmed
by direct fetch/search this session; the remainder are from search-result URLs (reliable ids, abstracts not
individually re-fetched — treat their *magnitudes* as directional single-source, per the closed-loop-verdict
doc's standard). Several ids (OmniDreams 2606.*, RoaD 2512.*, Cam2Sim 2607.*, DriveCtrl 2605.*) postdate this
agent's training cutoff and were verified via live search only.
