# TanitDataSet v1 — Semantic / Language / Reasoning-Label Survey

**Author:** Data Engineering (research/planning session). **Date:** 2026-07-19. **Status:** research + design
only; no code, no pod, nothing committed. **Scope:** survey the driving datasets/benchmarks that carry
**semantic descriptions, scene tags, driving QA, captions, and reasoning / chain-of-thought traces**, to inform a
**camera-first** "TanitDataSet v1" that should ship scenario descriptions, scene tags, and reasoning-trace GT.

> ⚠️ **Two honesty flags up front, because they decide the whole recommendation.**
> 1. **Almost every rich language/reasoning driving dataset is built *on top of* a non-commercial base corpus**
>    (nuScenes, Waymo, ONCE, BDD) — so it *inherits that base's NC/research license*. It is a **text layer over
>    someone else's camera data**, not an independent camera dataset. The few independent-camera ones (LingoQA,
>    CoVLA, DRAMA, Rank2Tell, MAPLM, Alpamayo) are themselves NC / research-gated / proprietary / unreleased.
>    **None of the strong semantic layers are permissive (MIT/CC-BY).** This is the opposite of the owned-core
>    doctrine in `OWN_DATASET_PLAN.md` §4.
> 2. **A large fraction of reasoning-trace GT is MODEL-generated** (GPT-4/4o, ChatGPT, VLM auto-labelers,
>    template engines), sometimes with a human-verification pass, sometimes not. Model-generated reasoning is
>    **weaker ground truth** — it can launder a VLM's hallucinations into "labels." Human-authored explanation GT
>    (BDD-X, DRAMA, Rank2Tell, Talk2Car, nuScenes native) is scarcer and smaller.
>
> The practical consequence (developed in §5): the nuScenes/Waymo semantic stacks are best used **firewalled &
> internal** exactly like PhysicalAI-AV (D-002) — for eval, distillation targets, and format templates — while the
> **shippable** TanitDataSet v1 semantic layer is most cleanly obtained by **generating our own reasoning traces
> over our owned camera corpus** (comma2k19 / ZOD / PandaSet / Cosmos-DD), using the Alpamayo **Chain-of-Causation**
> schema as the format template and the CoVLA / WOMD-Reasoning auto-label+verify pipelines as the method template.

---

## 0. How this extends the existing plan

`OWN_DATASET_PLAN.md` and `DATASET_LANDSCAPE.md` cover the **action / ego-motion / sensor / license** axis (can we
ship the `[T,9,256,256]` episode bytes?). This survey adds the **orthogonal semantic axis**: which corpora carry
*language* — descriptions, tags, QA, and reasoning traces — and how those attach to camera pixels. The episode
contract has **no text field today**; a v1 semantic layer means adding a per-clip / per-frame sidecar
(`captions`, `scene_tags`, `qa`, `reasoning_trace`) alongside `frames/actions/poses`. Nothing here changes the
running contract.

**Already-owned semantic signal we forget we have:** our owned synthetic corpora already ship model-generated
semantics — **PhysicalAI-WorldModel-Synthetic-Scenarios** carries a **per-camera Qwen2.5-7B caption** plus
`{weather, time_of_day, surface_type, region}` scene tags and a **safety-critical scenario family label**
(emergency / lanechange / nudging / pedestrian / weather_degradation), under **OpenMDW-1.1 (permissive)**; and
**Cosmos-Drive-Dreams** clips carry weather/scenario structure under **CC-BY-4.0**. These are the *only*
license-clean semantic labels in the current inventory (both model/pipeline-generated, not human).

---

## 1. Template study — Alpamayo (NVIDIA PhysicalAI) reasoning-trace format

Our real corpus is PhysicalAI-AV (NVIDIA "Alpamayo" ecosystem), so its reasoning-trace format is the natural
schema template. NVIDIA's public **Alpamayo-R1** release (arXiv 2511.00088, Nov 2025; `nvidia/Alpamayo-R1-10B`)
documents it:

- **Model I/O:** multi-camera video (4 RGB cams: front-wide, front-tele, cross-left, cross-right) + recent ego
  history (position + rotation @10 Hz over the last 1.6 s) → **a text reasoning trace *and* a physical
  trajectory**. The model writes *why* before it emits motion.
- **The reasoning object = "Chain-of-Causation" (CoC):** a **decision-grounded, causally-linked** trace in a
  **structured English format that links each driving decision to its causal factors** (the salient agents /
  scene elements that cause the decision), aligned to the actual behavior the ego took. Conceptually a chain:
  *observation → the agent/element that matters → why it matters → resulting decision → action*, rather than free
  narration.
- **Scale & provenance:** **~700K CoC reasoning traces** in training, built by a **hybrid auto-labeling +
  human-in-the-loop pipeline** (auto-draft, human refine) over ~80,000 h of internal multi-camera driving. So
  even NVIDIA's flagship reasoning GT is **semi-synthetic** (auto-drafted, human-verified) — a realistic quality
  bar, not pure human authorship.
- **Availability / license:** **model weights = non-commercial license; inference code = Apache-2.0. The CoC
  reasoning-trace *dataset* is NOT publicly released.** So Alpamayo is a **format & method template, not an
  ingestible source.** (The underlying PhysicalAI-AV real data stays gated/confidential per D-002.)

**Takeaway for TanitDataSet v1:** adopt a CoC-shaped reasoning schema — `{observation, critical_agents,
justification, decision, action}` per decision point, causally linked, in the ego frame — and produce it the same
way NVIDIA did: **auto-label then human-verify**, over our **owned** camera clips. That is both the strongest
format and the only license-clean route (§5).

---

## 2. Master comparison table

Columns: **Labels** (D=description/caption, T=scene tags/attributes, QA=question-answer, R=reasoning/CoT,
Risk=risk/importance/intention, CC=corner-case tag) · **Gran.** (frame/clip/object) · **Base camera corpus**
(*own* = ships its own camera; *on X* = text layer over X's images) · **License** · **Reasoning-GT provenance**
(H=human, M=model/template, H+M=model-draft+human-verify, EXP=privileged-expert/sim).

| Dataset | Labels | Gran. | Size | Base camera corpus | License | GT provenance |
|---|---|---|---|---|---|---|
| **DriveLM-nuScenes** | QA,R,D | frame/obj (graph) | ~4.8k keyframes, ~377k–450k QA | **on nuScenes** (+CARLA split) | code Apache-2.0; **lang data CC-BY-NC-SA-4.0**; + nuScenes terms | **H** (annotators), graph-structured |
| **NuScenes-QA** | QA | frame/obj | 34k scenes, **460k QA** | **on nuScenes** | code MIT; inherits nuScenes **CC-BY-NC-SA** | **M** (manual templates → programmatic) |
| **Talk2Car** | D (referral command) | frame/obj | 9.2k imgs, ~12k commands | **on nuScenes** | code MIT; inherits nuScenes **NC** | **H** (AMT-written commands) |
| **NuPrompt** | D (object prompts),QA | frame/obj | ~35k object-centric prompts | **on nuScenes** | inherits nuScenes **NC** | **H+M** (GPT + human) |
| **OmniDrive** | QA,R (counterfactual) | frame/scene | nuScenes-scale QA | **on nuScenes** | inherits nuScenes **NC** | **M** (GPT counterfactual synth) |
| **DriveLMM-o1** | R (step-by-step),QA | frame/scene | 18k train / 4k test | **on nuScenes** (multiview+LiDAR) | research (inherits nuScenes **NC**) | **H+M** (GPT-4o draft → human verify) |
| **NuScenes-MQA** | QA (markup) | frame/obj | nuScenes-based | **on nuScenes** | inherits nuScenes **NC** | **M** (markup templates) |
| **OpenLane-V2** | T,topology-R (structured) | frame/obj | 2,000 segments | **on Argoverse2 + nuScenes** | **CC-BY-NC-SA-4.0** + both base terms | **H / semi-auto** (structured, not free text) |
| **Reason2Drive** | R (perception→pred→reason),QA | obj/scene | **600k** video-text pairs | **on nuScenes + Waymo + ONCE** | inherits 3× **NC** bases | **M** (auto-annotation schema + templates) |
| **WOMD-Reasoning** | D,R (interaction/intention),QA | scene/agent | **3M QA** | **on Waymo (WOMD)** | **Waymo NC** license | **M** (ChatGPT + rule translator, ~90%) |
| **BDD-X** | D (action)+R (justification) | clip | 6,984 videos, ~26k pairs, ~77h | **on BDD** (dashcam front) | annotations BSD-3; BDD base = research | **H** (AMT action+justification) |
| **CODA-LM** | D,R (suggestion),CC | frame | 9,768 corner-case scenes | **on CODA** (nuScenes/ONCE-derived) | code Apache-2.0; annotations research | **M** (GPT-4V) + human-verified test |
| **Impromptu-VLA** | R (CoT),QA,action | clip | ~80k clips from 2M source | **distilled from 8 sets** (nuScenes/Waymo/ONCE/Argo2/KITTI/Mapillary/NAVSIM/IDD) | open data, but **inherits 8 base licenses (mostly NC)** | **H+M** (VLM CoT → human verify) |
| **LingoQA** | QA,D,R (commentary) | clip (video) | **28k** scenarios, **419k** QA | **own** (Wayve, front cam) | **custom Wayve non-commercial, revocable** | **H** (+ some model, human-scored) |
| **CoVLA** | D (frame captions)+action/traj | frame | 10k clips, 80h, **6M frames** | **own** (Turing, Tokyo, front cam) | **proprietary CoVLA terms + VideoLLaMA-2 (NC research)** | **M** (auto-label + MLLM auto-caption) |
| **DRAMA** | D (risk caption),Risk,QA | clip/obj | **17,785** clips (Tokyo) | **own** (Honda dashcam front) | **research / NC, request-gated** | **H** (human risk captions + boxes) |
| **Rank2Tell** | Risk (ranking)+R (why) | clip/obj | 116 clips ×~20s @10fps | **own** (Honda, SF Bay, 3-cam) | **research / NC, university-gated** | **H** (human importance + free-form reasons) |
| **MAPLM / MAPLM-QA** | T,D,QA (map+traffic) | frame | 2M frames; QA 61k/14k frames | **own** (Tencent, panoramic front + BEV/LiDAR) | **CC-BY-NC-3.0** | **semi-auto** (HD-map pipeline) + template QA |
| **DriveCoT** | R (CoT)+control | frame (2 Hz) | 1,058 scenarios, 36k samples | **CARLA sim** (multi-cam) | research | **EXP** (rule-based expert GT + ChatGPT rewrite) |
| **DriveLM-CARLA** | QA,R | frame | CARLA split of DriveLM | **CARLA sim** | Apache/CC-BY-NC-SA | **M/EXP** (sim + templates) |
| **nuScenes (native)** | T (8 attrs),D (per-scene text),23 classes | scene/obj | 1,000 scenes | **nuScenes itself** (6-cam incl. front) | **CC-BY-NC-SA-4.0** | **H** (annotators) |
| **Waymo (native + ROAD-Waymo)** | T (action/agent/location),scenario-mining | frame/agent | ROAD-Waymo 12.4M labels/198k frames | **Waymo itself** (5-cam) | **Waymo NC** | **H + rule-based** |
| **Alpamayo CoC** | R (Chain-of-Causation)+traj | decision point | ~700k traces | **NVIDIA internal** (4-cam) — *not released* | model wts non-commercial; **dataset unreleased** | **H+M** (auto-label + human-in-loop) |

*Sizes marked "~" are approximate / best-effort from abstracts and cards; verify against the source before any
build. Several licenses (DRAMA, Rank2Tell exact terms, Reason2Drive code) are "research/NC, request-gated" by the
host page but lack a crisp SPDX tag — treat as NC until a license file is read.*

---

## 3. Per-dataset notes (what each actually gives us)

### Built on nuScenes (one base corpus → a whole semantic stack)
- **DriveLM** (ECCV'24 Oral, OpenDriveLab): the reference **graph-VQA** dataset — perception→prediction→planning
  QA with logical dependencies over ~4.8k nuScenes keyframes, plus a CARLA split. **Language data is
  CC-BY-NC-SA-4.0**, code Apache-2.0. Human-annotated graph QA. The de-facto benchmark; strong *format*
  reference for scene-graph reasoning.
- **NuScenes-QA** (AAAI'24): the biggest template-QA set (460k QA, 5 question types, 0/1-hop). Programmatic from
  3D scene graphs → cheap but shallow; **not reasoning**, more perception QA. Inherits nuScenes NC.
- **Talk2Car** (EMNLP'19): natural-language **referral commands** grounded to a 3D box — the "which object does
  this instruction mean" task. Human-written. Good for command-conditioning (our H12), NC-inherited.
- **OmniDrive** (CVPR'25): adds **counterfactual reasoning** ("what if the ego did X") QA on nuScenes — the
  closest public analog to our NuRec counterfactual idea, but as *language* not rendered pixels. GPT-generated.
- **DriveLMM-o1** (2025): explicit **step-by-step reasoning** annotations, GPT-4o-drafted then **human-verified**
  — one of the better-provenance reasoning sets, but small (18k/4k) and nuScenes-NC.
- **NuPrompt, NuScenes-MQA**: additional object-prompt / markup-QA layers on nuScenes.
- **Structural point:** all of the above **co-attach to a single nuScenes camera ingest**. Pull nuScenes 6-cam
  (front included) once and you unlock *all* of them at the semantic layer — but every one is **non-commercial**,
  because nuScenes is CC-BY-NC-SA (D-012 `research/NC`; the plan keeps nuScenes probe-only, never trained).

### Built on Waymo / ONCE / multi-source
- **WOMD-Reasoning** (ICML'25): **3M QA** on interaction & intention reasoning over Waymo Open Motion — the
  largest reasoning-oriented set, but **ChatGPT-generated** (rule translator + LLM, ~90% self-reported accuracy)
  and **Waymo-NC**. Strong for interaction/intention *labels*, weak as trustworthy GT.
- **Reason2Drive** (ECCV'24): 600k video-text pairs with **chain-based reasoning** (perception→prediction→
  reasoning) auto-generated from **nuScenes+Waymo+ONCE** object metadata via templates. Broad but **model/template
  GT** and triple-NC-inherited.
- **Impromptu-VLA** (NeurIPS'25): **80k clips distilled from 8 open datasets**, focused on **unstructured
  long-tail** (unclear boundaries, temporary rules, odd obstacles, bad road conditions) with planning-oriented CoT
  QA + action trajectories, **VLM-drafted → human-verified**. "Open data," but the clips inherit **8 different
  base licenses** (nuScenes/Waymo/ONCE/Argoverse2/KITTI/Mapillary/NAVSIM/IDD — mostly NC). Excellent *taxonomy*
  reference for corner-case tags.

### Built on BDD
- **BDD-X** (ECCV'18): the pioneer — **human** action *descriptions* + *justifications* ("moves left **because**
  the bus ahead is stopping") on ~7k dashcam-front videos with synced speed/course/GPS. Annotations are
  permissively coded (BSD-3) but the **BDD base itself is research-licensed / members-only-commercial** (plan
  Tier-C). The cleanest *human justification* schema in the field; the base license is the blocker.

### Independent own-camera corpora (bring their own pixels)
- **LingoQA** (Wayve, ECCV'24): 28k video scenarios / 419k QA incl. free-form commentary/justification. **Own
  front camera.** License is a **custom Wayve non-commercial, revocable-at-will** grant with a feedback-grab
  clause — usable for research, **not** for an owned/redistributable or commercial product.
- **CoVLA** (Turing, WACV'25): the largest own-camera **VLA** set (6M frames, frame-level captions + future
  trajectory), Tokyo. **Model-generated** (auto-label + MLLM auto-caption). License is **proprietary CoVLA terms +
  VideoLLaMA-2 (NC research)** — already in our landscape as an H12 candidate; NC.
- **DRAMA** (Honda, WACV'23): 17,785 clips with **human** risk captions + important-object boxes + multi-level
  risk QA. Own dashcam. **Research/NC, request-gated.** Best public **human risk-explanation** GT.
- **Rank2Tell** (Honda, WACV'24): **importance ranking + free-form "why"** reasons, human-authored, on highly
  interactive SF-Bay scenes. Own 3-cam. **Research/NC, university-gated**, small (116 clips). Best public
  **importance-justification** schema.
- **MAPLM** (Tencent, CVPR'24): map + traffic-scene understanding — scene type, lane count/description, scene
  caption, unusual-object, lane-change/speed QA over panoramic front + BEV/LiDAR. 2M frames; MAPLM-QA 61k QA.
  **CC-BY-NC-3.0.** Own camera; strongest **map/HD-semantic tag** set.

### Simulator (perfect but synthetic GT)
- **DriveCoT** (CARLA): 36k samples with CoT labels from a **privileged rule-based expert** (then ChatGPT-
  rewritten to diversify language). GT is *causally correct by construction* (expert sees ground truth) but
  synthetic-domain. Pairs naturally with our owned **CARLA off-expert** arm (D-014) — the one place we can mint
  **license-clean, causally-grounded** reasoning GT ourselves.

### Native tags (no separate language dataset needed)
- **nuScenes native:** 8 object **attributes** (moving/stopped/parked, rider/no-rider…), 23 classes, and a short
  **human text description per 20-s scene** — CC-BY-NC-SA.
- **Waymo native / ROAD-Waymo / scenario mining:** per-agent **action/agent/location tags** (12.4M labels), mined
  scenario categories (unprotected turns, merges, cut-ins) — Waymo NC.

---

## 4. The structural finding — what stacks onto what

The single most useful lens for TanitDataSet: **semantic datasets cluster by base camera corpus**, and ingesting
one base unlocks its whole language stack.

| Base camera corpus | Semantic layers that ride on it | Combined license floor |
|---|---|---|
| **nuScenes** (6-cam incl. front) | DriveLM · NuScenes-QA · Talk2Car · NuPrompt · OmniDrive · DriveLMM-o1 · NuScenes-MQA · (Reason2Drive/OpenLane-V2/Impromptu subsets) | **CC-BY-NC-SA-4.0 (NC)** |
| **Waymo (WOMD/WOD)** | WOMD-Reasoning · (Reason2Drive/Impromptu subsets) · ROAD-Waymo tags | **Waymo NC** |
| **BDD** | BDD-X | **research / members-commercial** |
| **ONCE / Argoverse2 / KITTI / Mapillary / NAVSIM / IDD** | Reason2Drive / OpenLane-V2 / Impromptu subsets | **NC (mixed)** |
| **Independent own camera** | LingoQA (Wayve) · CoVLA (Turing) · DRAMA + Rank2Tell (Honda) · MAPLM (Tencent) | each NC / research / proprietary |
| **CARLA sim** | DriveCoT · DriveLM-CARLA | permissive-ish (sim self-gen is owned) |
| **NVIDIA internal (unreleased)** | Alpamayo Chain-of-Causation | dataset not released |

**Densest stack = nuScenes** (7+ semantic layers on one ingest). **But the entire nuScenes/Waymo/BDD semantic
economy is NC** — it cannot enter an owned/redistributable/commercial TanitDataSet, only an internal,
firewalled one (same status as PhysicalAI-AV). **No permissively-licensed rich reasoning layer exists** in the
public field as of mid-2026.

---

## 5. Shortlist — what to fuse into TanitDataSet v1, and how it attaches

Because heterogeneity is *desired* and sources need not share content, the fusion is by **role**, not by base
corpus. Three tiers, split by license posture (mirroring the D-002 firewall doctrine):

### Tier 1 — OWN-GENERATED semantic layer (the shippable core; license-clean)
The only way to get a **redistributable** reasoning/description layer is to **mint it ourselves over our owned
camera corpus**, exactly as Alpamayo/CoVLA/WOMD-Reasoning minted theirs:

1. **Reasoning traces (Chain-of-Causation schema) over owned clips.** Run a VLM auto-labeler → **human-verify**,
   producing CoC-shaped `{observation, critical_agents, justification, decision, action}` per decision point on
   **comma2k19 (MIT), ZOD (CC-BY-SA), PandaSet (CC-BY), Cosmos-DD (CC-BY)**. Provenance is *ours*; license is the
   clip's own permissive license. This is the license-clean analog of the whole nuScenes reasoning stack.
2. **Fold in the semantics we already own for free:** WorldModel-Synthetic-Scenarios **Qwen2.5-7B captions +
   {weather,time_of_day,surface_type,region} tags + scenario-family labels** (OpenMDW, permissive) and
   Cosmos-DD weather/scenario structure (CC-BY). These become the **scene-tag + caption** columns with zero new
   labeling and clean licenses.
3. **CARLA off-expert CoT (DriveCoT-style).** Our owned CARLA arm (D-014) can emit **privileged-expert,
   causally-correct** reasoning GT — the one source of *human-trustworthy-grade* reasoning we can generate at
   scale and ship. Best reasoning-GT quality of any owned option.

*Attach:* all three sit on **owned camera pixels already in (or targeted by) the plan** (comma2k19/ZOD/PandaSet/
Cosmos-DD/CARLA). They stack directly onto the existing episode cache as a text sidecar. **No NC inheritance.**

### Tier 2 — INTERNAL-only fusion (firewalled; eval + distillation, never shipped)
Treat these like PhysicalAI-AV: powerful, use internally, **never** in a public/redistributable artifact, tagged
`data:nc` for one-grep exposure control.
- **DriveLM** + **DriveLMM-o1** — best *human/verified* reasoning-graph and step-reasoning **eval benchmarks**
  and distillation targets (on nuScenes).
- **BDD-X** — best **human action-justification** schema to imitate (on BDD).
- **DRAMA** + **Rank2Tell** (Honda) — best **human risk / importance-justification** GT; small enough to use as
  gold eval sets for our risk/intention labels.
- **WOMD-Reasoning** / **Reason2Drive** — large interaction/intention reasoning for pretraining signal, accepting
  model-generated weakness + NC.
- **CODA-LM** / **Impromptu-VLA taxonomy** — corner-case tag schemas to *copy* for our own long-tail tagging.

*Attach:* each rides its NC base (nuScenes/Waymo/BDD). If we ever ingest nuScenes camera **internally** (probe-
only per D-012), the whole nuScenes stack lights up at once — the cheapest way to get a large internal semantic
set, at the cost of NC-firewall.

### Tier 3 — schema/format templates only (copy the design, ingest nothing)
- **Alpamayo Chain-of-Causation** — the reasoning-trace *format* (§1); dataset unreleased, weights NC.
- **DriveLM graph-QA** structure; **DriveCoT** aspect decomposition; **MAPLM** map-tag taxonomy; **Rank2Tell**
  4W+1H importance schema. Adopt the schemas; generate the data ourselves (Tier 1).

### Recommended v1 label schema (superset to design toward)
`caption` (frame/clip) · `scene_tags{weather,time_of_day,road_type,maneuver,scenario_family}` ·
`risk{important_agents[], risk_level, justification}` · `reasoning_trace{CoC steps}` · `qa[]` (optional) —
each field carries a `provenance ∈ {human, model, model+verified, expert-sim}` and a `source_license` stamp, so
the firewall and the human-vs-model quality of every label is one column away.

---

## 6. License / provenance gotchas (do not fabricate cleanliness)

- **NC is the default, not the exception.** nuScenes, Waymo, ONCE, Argoverse2, KITTI, Mapillary, OpenLane-V2,
  MAPLM (CC-BY-NC-3.0), and every text layer built on them are **non-commercial**. A "private" HF repo shared with
  a third party is still distribution and does not rescue NC (OWN_DATASET_PLAN §4.2).
- **DriveLM specifically:** code Apache-2.0 but **language data CC-BY-NC-SA-4.0** *and* nuScenes terms — two NC
  layers.
- **LingoQA:** custom Wayve license is **revocable at any time** and grabs rights to your feedback — fine for a
  paper, unsafe to build a product on.
- **CoVLA:** the *text* is under the **VideoLLaMA-2** model license (NC research), separate from the video/CAN
  terms — a compound NC.
- **BDD-X:** the annotations are permissively coded, but they are **useless without BDD's videos**, and BDD is
  research/members-commercial — the base gates the whole thing.
- **Model-generated reasoning ⇒ weaker GT.** NuScenes-QA (templates), Reason2Drive (templates), WOMD-Reasoning
  (ChatGPT ~90%), OmniDrive (GPT), CoVLA (MLLM), CODA-LM (GPT-4V) are **auto-generated**; treat their reasoning as
  *bootstrap/pretraining signal*, not gold. Prefer **human** (BDD-X, DRAMA, Rank2Tell, Talk2Car, nuScenes native)
  or **model+verified** (DriveLMM-o1, Impromptu, Alpamayo CoC) for anything used as an **eval** target.
- **Alpamayo CoC dataset is not released** — we get the *format*, not the data. Don't plan to ingest it.
- **Verify before build:** DRAMA / Rank2Tell exact license text, Reason2Drive code license, and CoVLA's PDF terms
  were host-page-level only in this pass — read the actual license files before any ingest, per the §2 caveat.

---

## 7. Bottom line

- **Format template:** copy **Alpamayo Chain-of-Causation** (`observation → critical agents → justification →
  decision → action`, causally linked, ego-frame; ~700k-trace, auto-label + human-verify method).
- **Ship the semantic layer by generating it ourselves** over the **owned** camera corpus (comma2k19/ZOD/
  PandaSet/Cosmos-DD/CARLA) — plus the free, already-owned **WorldModel-Synth captions/tags** (OpenMDW) and
  **Cosmos-DD** scene structure (CC-BY). This is the only route that stays permissive and redistributable.
- **Use the public reasoning datasets internally & firewalled** (nuScenes stack: DriveLM/DriveLMM-o1/OmniDrive;
  Waymo: WOMD-Reasoning; BDD: BDD-X; Honda: DRAMA/Rank2Tell) as **eval benchmarks, distillation targets, and
  schema templates** — never in the shippable core, because they are all NC/research/proprietary and many are
  model-generated.
- **The densest single unlock is nuScenes** (7+ semantic layers on one ingest) — but it is NC, so it belongs in
  the internal-firewalled tier alongside PhysicalAI-AV, not the owned core.

---

### Sources (fetched/searched 2026-07-19; verify licenses against the actual license file before ingest)
- DriveLM — arXiv [2312.14150](https://arxiv.org/abs/2312.14150); [github.com/OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM)
- LingoQA — arXiv [2312.14115](https://arxiv.org/abs/2312.14115); [github.com/wayveai/LingoQA](https://github.com/wayveai/LingoQA/blob/main/LICENCE)
- Reason2Drive — arXiv [2312.03661](https://arxiv.org/abs/2312.03661); [github.com/fudan-zvg/Reason2Drive](https://github.com/fudan-zvg/Reason2Drive)
- BDD-X — arXiv [1807.11546](https://arxiv.org/pdf/1807.11546)
- NuScenes-QA — arXiv [2305.14836](https://arxiv.org/abs/2305.14836); [github.com/qiantianwen/NuScenes-QA](https://github.com/qiantianwen/NuScenes-QA)
- DRAMA — arXiv [2209.10767](https://arxiv.org/abs/2209.10767); [usa.honda-ri.com/drama](https://usa.honda-ri.com/drama)
- Talk2Car — arXiv [1909.10838](https://arxiv.org/pdf/1909.10838); [talk2car.github.io](https://talk2car.github.io/)
- Rank2Tell — arXiv [2309.06597](https://arxiv.org/abs/2309.06597); [usa.honda-ri.com/rank2tell](https://usa.honda-ri.com/rank2tell)
- CODA-LM — arXiv [2404.10595](https://arxiv.org/html/2404.10595v2); [coda-lm.github.io](https://coda-lm.github.io/)
- MAPLM — [github.com/LLVM-AD/MAPLM](https://github.com/LLVM-AD/MAPLM); [huggingface.co/datasets/LLVM-AD/maplm_v2](https://huggingface.co/datasets/LLVM-AD/maplm_v2) (CC-BY-NC-3.0)
- OpenLane-V2 — arXiv [2304.10440](https://ar5iv.labs.arxiv.org/html/2304.10440); [github.com/OpenDriveLab/OpenLane-V2](https://github.com/OpenDriveLab/OpenLane-V2) (CC-BY-NC-SA-4.0)
- nuScenes — arXiv [1903.11027](https://ar5iv.labs.arxiv.org/html/1903.11027); [nuscenes.org](https://www.nuscenes.org/nuscenes)
- Waymo Open / ROAD-Waymo — [waymo.com/open](https://waymo.com/open/about/); [waymo.com/blog scenario data](https://waymo.com/blog/2021/03/expanding-waymo-open-dataset-with-interactive-scenario-data-and-new-challenges/)
- Alpamayo-R1 — arXiv [2511.00088](https://arxiv.org/abs/2511.00088); [huggingface.co/nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B); [NVIDIA blog](https://developer.nvidia.com/blog/building-autonomous-vehicles-that-reason-with-nvidia-alpamayo/)
- WOMD-Reasoning — arXiv [2407.04281](https://arxiv.org/abs/2407.04281); [github.com/yhli123/WOMD-Reasoning](https://github.com/yhli123/WOMD-Reasoning)
- CoVLA — arXiv [2408.10845](https://arxiv.org/abs/2408.10845); [huggingface.co/datasets/turing-motors/CoVLA-Dataset](https://huggingface.co/datasets/turing-motors/CoVLA-Dataset)
- DriveLMM-o1 — arXiv [2503.10621](https://arxiv.org/abs/2503.10621); [github.com/ayesha-ishaq/DriveLMM-o1](https://github.com/ayesha-ishaq/DriveLMM-o1)
- Impromptu-VLA — arXiv [2505.23757](https://arxiv.org/abs/2505.23757); [impromptu-vla.c7w.tech](https://impromptu-vla.c7w.tech/)
- OmniDrive — arXiv [2405.01533](https://arxiv.org/abs/2405.01533) / [2504.04348](https://arxiv.org/abs/2504.04348)
- DriveCoT — arXiv [2403.16996](https://arxiv.org/abs/2403.16996); [drivecot.github.io](https://drivecot.github.io/)
- NuPrompt — arXiv [2309.04379](https://arxiv.org/html/2309.04379)
