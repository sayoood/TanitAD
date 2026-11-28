# Should we fork AlpaSim as TanitSim? — a decision-grade recommendation

**Date:** 2026-07-26 · **For:** Sayed (PI) · **Author:** consolidation agent
**Companions:** `ALPASIM_STATE.md` · `BUILD_AND_USE.md`
**Evidence legend:** `MEASURED` (ours + path) · `PUBLISHED` (cited) · `INHERITED` (our docs, not re-verified) ·
`ESTIMATED` · `HYPOTHESIS`

---

## 1. RECOMMENDATION — **DO NOT FORK.** Build the renderer instead.

> **The thing worth owning is not the thing a fork would give us.**
>
> AlpaSim's orchestration layer is Apache-2.0, works, and we have made **exactly one** modification to
> it in four months of use — pruning three heavy dependencies out of `pyproject.toml`
> (MEASURED 2026-07-26: `git status --short` on the pod shows `M pyproject.toml` and nothing else).
> A fork of a codebase you have not needed to change is pure maintenance cost.
>
> The **3.21× OOD tax** — the one thing that degrades every number this asset produces — lives entirely
> in the **NRE renderer**, which is a **closed NVIDIA binary under a licence that forbids us from
> modifying it** (`NGC-DL-CONTAINER-LICENSE` §4b/§4c, read on the pod this session). **Forking AlpaSim
> confers no right whatsoever to touch it.**
>
> And we do not need a fork to replace it: AlpaSim **already ships an external-renderer contract**
> (`src/wizard/configs/deploy/external_video_model.yaml` + `docs/VIDEO_MODEL.md`, MEASURED on the pod) —
> any gRPC service on the `renderer` endpoint. The chief scientist's "single unowned dependency" is
> **renderer-shaped, not simulator-shaped.**

**The posture I recommend, in three parts:**

| | Action | Cost |
|---|---|---|
| **A** | **Keep AlpaSim unforked and pinned.** Vendor nothing. Pin commit `55814289d8047bf239206712d31a745f2ad8f5ea`; keep our work as the adapter package it already is (`refc_driver.py`, `flagship_v1_driver.py`, the launch scripts). Move `pyproject_pared.toml` from a file-swap to a documented `uv` override so even that one edit disappears. | ~0.5 eng-day |
| **B** | ⭐ **Give the RENDERER an owner** — build `TanitRender` as an external gRPC service speaking AlpaSim's documented renderer contract. This is the chief scientist's Tier-4 #15, correctly scoped. **Gate it behind a cheap pre-registered OOD probe (§7.2) before committing weeks.** | probe ~1 pod-day; build 2–6 eng-weeks *if the probe passes* |
| **C** | **Turn on the reactive agents we already own.** `src/trafficsim/alpasim_trafficsim/catk/smart/` — SMART + CAT-K, **Apache-2.0, in-tree, freely modifiable, and never once enabled by us** (MEASURED: present on the pod; disabled by default in every run we have made). Apache-2.0 means we can patch it without forking anything. | ~1–3 eng-days to enable and validate |

**A fork buys us (C) and nothing else — and (C) is already free.**

---

## 2. LICENSING — the load-bearing analysis

This is where the answer is actually decided, so it is stated with the exact evidence.

### 2.1 The three-layer licence stack

AlpaSim is **not one licence**. It is three, and they get *stricter* as you move toward the thing we
want to change.

| Layer | Licence | Evidence | Fork implication |
|---|---|---|---|
| **Orchestration source** (controller, driver, eval, grpc, physics, plugins, runtime, tools, **trafficsim**, utils, utils_rs, wizard) | **Apache-2.0** | 🟢 **MEASURED 2026-07-26** — `LICENSE` on the pod reads "Apache License / Version 2.0". Corroborated by four committed docs incl. `Tools&DevEnv/Research/2026-07-18-…:94` (*"AlpaSim is public & clonable — Apache-2.0"*) | **Fork permitted.** Permissive, no copyleft, notice+attribution only. |
| **NRE renderer container** (`nvcr.io/nvidia/nre/nre-ga:26.04` → `pycena_nrm_full`) | **NVIDIA Deep Learning Container Licence** | 🟢 **MEASURED 2026-07-26** — `NGC-DL-CONTAINER-LICENSE` at `/workspace/nre/rootfs/`, read directly this session | **Fork forbidden and pointless.** See §2.2. |
| **NuRec scene data** (every USDZ we render) | **NVIDIA AV Dataset Licence** | 🟢 **MEASURED** (`Data Engineering/Research/2026-07-07-physicalai-av-license-review.md`) — internal AV development only · **NVIDIA Confidential** · **12-month term, destroy all copies on expiry** | **Expires.** See §2.3. |

⚠️ **Correcting a live assumption in our own docs.** Five committed documents say "AlpaSim, Apache-2.0"
without qualification, including the operational `RUN_RECIPE.md:15`. **That is true of the repo and
false of the system.** The Explore sweep this session found **no licence recorded anywhere in the repo
for the NRE renderer** — the docs record the *credential* gate (NGC key) but never the *terms*. This
document is the first place those terms are written down. **Anyone assuming rendered frames are
"Apache-2.0-clean" is wrong.**

### 2.2 The NRE container licence — verbatim clauses that decide this

Read directly from `/workspace/nre/rootfs/NGC-DL-CONTAINER-LICENSE` (MEASURED 2026-07-26):

| § | Clause (quoted in part) | Consequence for a fork |
|---|---|---|
| **§1a** | grants a *"non-exclusive, non-transferable licence, without the right to sublicense"* to install/use, and to modify **only** *"samples or example source code delivered in the CONTAINER"* | We may **run** it. We may not modify the renderer itself. |
| **§1b** | may *"deploy the CONTAINER on infrastructure you own or lease to offer a service to third parties, without distributing the CONTAINER"* | An internal or hosted eval service is fine. |
| **§4b** | *"You may not reverse engineer, decompile or disassemble…"* | ⛔ The obfuscated Bazel binary stays a black box. |
| **§4c** | *"…you may not copy, sell, rent, sublicense, transfer, distribute, **modify, or create derivative works** of any portion of the CONTAINER… you may not distribute or sublicense the CONTAINER as a stand-alone product."* | ⛔ **This is the clause that kills the fork rationale.** Fixing the OOD-fidelity gap *means* modifying the renderer. We may not. |
| **§4e** | no bypassing *"any technical limitation… or authentication mechanism"* | ⛔ No working around the NGC gate. |
| **§4f** | *"You may not replace any NVIDIA software components in the CONTAINER… with other software that implements NVIDIA APIs."* | ⚠️ **Read carefully.** This restricts substitution *inside* the container. It does **not** restrict AlpaSim's own Apache-2.0 `renderer` gRPC endpoint, which is NVIDIA's *documented, supported* extension point (`docs/VIDEO_MODEL.md`). ⚠️ `HYPOTHESIS` — my reading, not counsel's. If TanitRender ever ships externally, get this confirmed. |
| **§4g** | may not use the CONTAINER *"in any manner that would cause it to become subject to an open source software licence"* | ⛔ **We may never GPL/AGPL anything that links the container.** A copyleft TanitSim is off the table while NRE is in the loop. |
| **§4h** | *"the CONTAINER as delivered is not tested or certified by NVIDIA for use in… Critical Applications… Examples… include use in avionics, navigation, **autonomous vehicle applications**, ai solutions for automotive products…"* | 🔴 **Directly contradicts "safety-grade".** See §2.4. |
| **§4i** | full indemnification of NVIDIA for Critical-Application use | 🔴 Liability sits with us. |
| **§2a** | distribution requires *"material additional functionality, beyond the included portions of the CONTAINER"* | ✅ TanitAD easily qualifies — but §4c still forbids shipping the container itself. |

**⇒ The single most important sentence in this document:** *the one capability a TanitSim fork is
supposed to buy — the ability to fix reconstruction fidelity — is the one capability the licence
specifically denies, and no amount of forking Apache-2.0 code changes that.*

### 2.3 The expiry nobody has diaried

The NuRec scenes carry the NVIDIA AV Dataset Licence: **§1** grants use *"solely for your internal
development of autonomous vehicles… using NVIDIA technology"*; **§3** treats the data as **NVIDIA
Confidential Information**; **§8** **expires 12 months after download, with an obligation to destroy all
copies.** (MEASURED, `2026-07-07-physicalai-av-license-review.md`.)

Our scenes were downloaded **2026-07-22** (MEASURED, `DL_EXIT=0`). ⇒ **`ESTIMATED` expiry ≈ 2027-07-22.**

Two consequences that a fork makes *worse*, not better:

1. **Everything we render is `gated-confidential` by our own firewall.** The sibling source
   `physicalai_av` is registered `SourceLicense("gated-confidential", "NVIDIA-AV-internal", …)` at
   `stack/tanitad/lake/schema.py:111`, and `stack/tanitad/lake/license_guard.py` **refuses any export
   scope containing it**. Under our own augmentation rule (*"synthetic/derived processing does NOT
   launder a licence"*), AlpaSim renders inherit it. **No AlpaSim-derived pixel may appear in a paper,
   an HF push, or a benchmark submission.**
2. **A "TanitSim" whose only scenes are time-limited, confidential, third-party reconstructions is a
   simulator with a shelf life.** Building a *fork* on that foundation compounds the exposure; building
   an *adapter* on it does not.

### 2.4 "Safety-grade closed-loop" — the phrase needs retiring for this stack

The chief-scientist review (`Reviews/2026-07-25-…/R5_strategy_research_management.md:33-37`) names the
reactive-agent renderer as gating *"safety-grade closed-loop, D5/D6, the renderer-half of the
beyond-ADE suite, and any NAVSIM/Bench2Drive entry."* That framing is right about the *need* and
optimistic about *this stack*: NRE's own licence disclaims AV use as an uncertified Critical Application
and indemnifies NVIDIA (§4h/§4i). AlpaSim+NRE can produce **research evidence**; it cannot produce a
**safety argument**. That is another reason the investment belongs in a renderer we own, not in a fork
of a wrapper around one we do not.

### 2.5 Are we in a `refuse` class? — **No, but there is no registry entry at all**

Our `SOURCE_REGISTRY` vocabulary is `("owned-safe", "nc-research", "gated-confidential", "refuse")`
(`stack/tanitad/lake/schema.py:44`). Two findings:

- **Nothing here is `refuse`-class, and nothing is copyleft.** Apache-2.0 is permissive; the NGC licence
  is restrictive-proprietary but not viral (and §4g actively *prevents* virality). ✅ No blocker of that kind.
- ⚠️ **`SOURCE_REGISTRY` is a *dataset* registry only.** AlpaSim, NRE, CARLA, HUGSIM, MetaDrive,
  Bench2Drive — **none has a registry entry**, so none has mechanical enforcement. The one dataset entry
  that *does* bite (`physicalai_av` → `gated-confidential`) is the one that matters, and it bites
  correctly. **Recommend:** add a tool/simulator axis (`TOOL_REGISTRY`) with AlpaSim (Apache-2.0),
  NRE (NGC-DL-CONTAINER, non-redistributable, expiring data), CARLA (MIT + CC-BY), HUGSIM, and
  Bench2Drive (whose licence our own docs record as **self-contradictory** — repo `LICENSE` says
  CC-BY-**NC-ND**, HF card + paper say Apache-2.0). *~0.5 eng-day, and it prevents the next version of
  this exact ambiguity.*

### 2.6 What is NOT established, and what would settle it

Per the brief's instruction not to assume permissive where evidence is absent:

| Open question | Status | What would settle it |
|---|---|---|
| Does §4f block an external renderer on AlpaSim's own gRPC endpoint? | `HYPOTHESIS` — I read it as *no* (it restricts substitution *inside* the container, and NVIDIA documents the external-renderer path themselves) | Counsel review, **before** any external distribution. Internal research use is not at risk either way. |
| May we redistribute **renders** of *non-PhysicalAI* geometry (comma2k19 / CARLA) made with the NuRec **tool**? | 🔴 **UNRESOLVED and already flagged to you** — `Data Engineering/OWN_DATASET_PLAN.md:345` asks exactly this and has no answer | NVIDIA clarification, or adopt an **open 3DGS** engine for a fully-owned pipeline (which is precisely what §7 recommends) |
| Exact AV-dataset expiry date | `ESTIMATED` 2027-07-22 from a 2026-07-22 download | Read the acceptance record on the HF account |

---

## 3. What forking would buy — item by item, honestly

The brief names four hoped-for gains. Three are already available without a fork; one is unavailable
*with* one.

| Hoped-for gain | Verdict | Detail |
|---|---|---|
| **Control over the reactive-agent model** | ✅ **ALREADY OURS — no fork needed** | `src/trafficsim/alpasim_trafficsim/catk/smart/` (MEASURED on the pod): SMART trajectory-tokenization + **CAT-K** closed-loop fine-tuning, **Apache-2.0**. Apache-2.0 permits unlimited modification and vendoring. ⚠️ **We have never enabled it** — every run to date has trafficsim off. *The correct first move is to turn on the asset we own, not to fork the repo that contains it.* |
| **Ability to fix the OOD-fidelity gap** | ❌ **A FORK CANNOT DELIVER THIS** | The 3.21× is the **renderer's** reconstruction fidelity. NRE is closed (§2.2 §4b/c). The only legal routes are (i) an **external renderer** on AlpaSim's documented endpoint, or (ii) **encoder-side alignment** on our model. Neither needs a fork. |
| **Map / lane awareness** | ✅ **ALREADY OURS — no fork needed**, but *not shippable* | 🟢 MEASURED (`gate0_prereq_probe.json`): every scene USDZ embeds a **`trajdata.VectorMap`** — 130–472 lane polygons, 130–393 road edges, 27–180 wait-lines per junction scene — loadable at inference via `ArtifactSceneProvider.from_path(...).get_data_source(scene_id).map`, in the **same world frame** as the ego pose the driver already receives. The brief is right that **PhysicalAI-AV has no map/lane/traffic-light feature**; AlpaSim's scenes supply exactly that gap **today**. ⚠️ But those maps are NVIDIA-AV-licensed and `gated-confidential` — usable for research, **never shippable**, and expiring (§2.3). |
| **Scenario authoring for the weak-spot corpus** | ✅ **MOSTLY ALREADY SOLVED — and a fork would not have helped** | NuRec scenes are *reconstructions of recorded drives*, not authored worlds, so whole-cloth junction authoring is out (actor perturbation via `--enable-editing-actors` is in). But the operative lever is **scene *selection*, and that pipeline is built, run, and committed**: `kf_download.sh` → `kf_batch.py` → `select_suite.py` → `scaled_wizard_gen.sh` → `scaled_master.sh`. It screened **356 candidate keyframes** from the 1606-scene `public_2604` pool and produced a **balanced n=37 suite covering all five categories** — including the two that were previously at zero (roundabout 8, traffic-light 7). Weak-spot corpora are a **download-and-label** problem, entirely orthogonal to source control. |

**One further thing a fork would not buy:** real-time. 0.29× at native resolution is **renderer-bound**
(MEASURED, `alpasim_realtime_a40.json`). Only a different renderer changes it.

---

## 4. What forking would cost

| Cost | Magnitude | Note |
|---|---|---|
| **Initial** | ~1–2 eng-weeks | Vendoring 12 packages, CI, protobuf/Rust build, our own release process |
| **Maintenance** | **~0.5–1 eng-day per upstream release, forever** `ESTIMATED` | The real cost. AlpaSim is an active NVIDIA project; NRE images are versioned (`26.04`) and **§5 of the container licence says the CONTAINER "may change without prior notice" and may introduce incompatibilities**. A fork must chase renderer-image changes it cannot see inside. |
| **Divergence risk** | high, asymmetric | The moment our fork drifts, we lose upstream's NuRec/OmniDreams backend work — the *only* part we cannot write ourselves |
| **Opportunity cost** | ⭐ **the decisive one** | Every week on a fork is a week not spent on the renderer, which is the actual unowned dependency. The chief scientist's finding was *"no pod is building it… the loop keeps running cheap experiments around it"* — a fork is another experiment *around* it. |
| **Compliance surface** | non-trivial | A repo named "TanitSim" invites the assumption that the whole thing is ours to publish. It is not (§2.2/§2.3). Every contributor becomes a licence-boundary risk. |
| **Naming/attribution** | small but real | §4d: may not imply NVIDIA sponsorship or endorsement |

---

## 5. The alternatives, scored

| # | Option | Buys | Costs | Verdict |
|---|---|---|---|---|
| **1** | ⭐ **Thin adapter, no fork** *(status quo, formalised)* | Everything we have now; upstream improvements for free; zero licence exposure beyond running it | ~0.5 eng-day to formalise | ✅ **ADOPT — this is (A)** |
| **2** | **Fork → TanitSim** | Control over code we have modified exactly once | 1–2 wk + perpetual maintenance + divergence + compliance | ❌ **REJECT.** Cannot fix the OOD gap (§2.2); everything else is already free |
| **3** | **Contribute upstream** | Goodwill; someone else maintains our fixes | PR overhead; NVIDIA's cadence; our valuable fixes are *bare-run* patches they may not want (they ship Docker) | 🟡 **OPPORTUNISTIC.** Worth offering the bare-run recipe as a doc PR. Do not block on it. |
| **4** | ⭐ **Own renderer on AlpaSim's external contract** ("TanitRender") | **The actual fix.** Low-OOD frames, real-time control, no NGC dependency, fully shippable, **owned** | 2–6 eng-weeks after a positive probe | ✅ **ADOPT, GATED — this is (B)**. §7.2 |
| **5** | **Different simulator — CARLA** | MIT code + CC-BY assets; fully shippable; reactive agents; maps; authoring | **Fails our binding constraint**: synthetic appearance is *far* more OOD than NuRec for a pixel encoder trained on real dashcam. `2026-07-24-low-ood-closedloop-renderer.md` ranks it #5 of 5 | ❌ for pixels; 🟡 **borrow its behaviour models**, decoupled |
| **6** | **nuPlan** | Large planning benchmark | **CC-BY-NC-SA-4.0** — NC *and* **share-alike**; would be `nc-research` class and is **explicitly not in `SOURCE_REGISTRY`**; 16 TB camera subset | ❌ **REJECT.** NC blocks commercial; SA is the one copyleft-shaped hazard on this list |
| **7** | **Bench2Drive** | A public leaderboard | ⚠️ **Licence self-contradictory in our own records**: repo `LICENSE` CC-BY-**NC-ND** vs HF card/paper Apache-2.0. Our standing note: *"do not build on it until the licence is in writing — ND would block publishing any derived label"* | ❌ **BLOCKED on licence**, not on merit |
| **8** | **HUGSIM** (arXiv:2412.01718) | 3DGS closed-loop, **>30 FPS**, explicitly targets **viewpoint extrapolation** + 360° actors, IDM/adversarial reactive traffic, open-source-intended | Licence **never verified by us**; per-scene reconstruction cost; 2–4 eng-weeks | 🟡 ⭐ **THE STRONGEST EXTERNAL CANDIDATE for (B).** Verify the licence, then run the §7.2 OOD probe against it *before* building anything ourselves |
| **9** | **Split instruments**: real-footage harness for road-keeping, AlpaSim for collision only | Uses each where it is strong; renderer-free for the bigger half | Two instruments to maintain | ✅ **ADOPT as the interim operating rule — see §6** |

---

## 6. The split-instrument rule — what to use AlpaSim *for*, starting now

This is the part that changes behaviour tomorrow regardless of the fork decision.

| Question | Instrument | Why |
|---|---|---|
| **Road-keeping / drift / lane departure / recovery** | 🟢 **Real-footage low-OOD harness** — **NOT AlpaSim** | On-policy OOD **1.02–1.20×**, 100 % of windows ≤1.5× (MEASURED) vs AlpaSim's 3.21×. A 3× cleaner instrument for the same question. |
| **Off-road / collision under reactive agents** | 🟡 **AlpaSim — the only option** | The real-footage instrument is **map-free and agent-free by construction**: agents are baked into real pixels at their logged positions, so it *structurally* cannot emit a collision (`LOOP_STATE.md` G1clean). |
| **Absolute safety rates** | ❌ **Neither** | Within-source relative only, both. |
| **Map/lane-aware evaluation** | 🟡 **AlpaSim (research only)** | Its USDZ VectorMaps are the only maps we have; they are gated-confidential and expiring. |

⚠️ **And when you do use AlpaSim, use the BALANCED suite.** The widely-quoted "REF-C beats flagship,
8/12 vs 2/12, Δ −0.43" comes from a suite that is **8/12 straight-or-urban — REF-C's best category.**
The balanced **n=37** suite gives **Δ −0.1228 [−0.2079, −0.0412]**: still a REF-C win with the CI
excluding zero, but **~3.5× smaller**, with **roundabout and highway TIED** and **both models collapsing
at uncontrolled intersections** (flagship 0/7). ✅ MEASURED, `scenario_stratified_scaled_results.json`.
This matters for the fork question only insofar as it shows the asset's *scene-selection* machinery —
not its source tree — is what produces better science. See `ALPASIM_STATE.md` §4.1.

⚠️ **Do not re-run the 07-24 mistake.** `RETRACTION_LOG.md:57` (class **C3**) records the over-broad
claim *"the closed-loop program is gated on ONE thing — a renderer"*, and its same-day update. Then
`RETRACTION_LOG.md:65` (class **C6**) records the *next* correction: the "renderer-free is enough"
reframe was itself **horizon-confounded** — measured at a 2 s window the base model "rarely failed", but
at the **18.5 s horizon that matches a real junction crossing** it fails on **59–84 %** of windows
(paired common-start, B=2000, OOD ≤1.30, **separated**). The honest state: **the renderer-free
instrument has ample road-keeping signal once measured at the event's timescale**, and E2a localizes the
remaining loss as **91 % downstream — a training-objective problem, renderer-free.** ⇒ the split above is
*better* supported today than it was two days ago, and the renderer is needed for **(B) collision only** —
which is a smaller, better-defined build than "a simulator".

---

## 7. THE DECISION RULES

### 7.1 Fork rule — **fork AlpaSim only when all three fire**

> **Fork only if:**
> 1. **≥3 substantive changes** to the Apache-2.0 orchestration layer are required that **cannot** be
>    expressed as a plugin, a config, or an external gRPC service; **AND**
> 2. an upstream PR for those changes has been **opened and rejected or stalled >30 days**; **AND**
> 3. **TanitRender is running**, so AlpaSim's orchestration sits on the critical path of a dated
>    deliverable rather than being a convenience.

**Current state: 0 of 3.** (1) is at **one** trivial change, and it is a dependency prune expressible as
a `uv` override. (2) no PR has ever been opened. (3) no renderer work has started.
**⇒ DO NOT FORK. Re-evaluate when (3) fires.**

**Independent re-opening triggers** (any one → revisit immediately, do not wait for the three):
upstream **archives or destructively rewrites** the repo · we must **redistribute** a closed-loop
harness externally (benchmark entry) · the NVIDIA AV dataset licence **lapses** and we substitute
owned scenes · NVIDIA changes AlpaSim's licence.

### 7.2 ⭐ Renderer rule — the pre-registered gate, both outcomes committed in advance

The mistake to avoid is committing eng-weeks to a renderer that lands at 2× instead of 1.2×. So gate it
on the **cheapest discriminating experiment**, which is the one this asset already invented:

> **The probe (~1 pod-day, no new infra):** for each candidate renderer, render our existing 4-scene
> diagnostic set and run **the §7.1 open-loop force-GT control from `BUILD_AND_USE.md`** — REF-C base,
> in-distribution poses, ADE scored exactly as `taniteval`. Reference: **1.5157 (NuRec) vs 0.4728 (real)
> = 3.21×**. Candidates, in order: **(a) HUGSIM** · **(b) AlpaSim's own OmniDreams/FlashDreams video
> backend** (already wired, `external_video_model.yaml`) · **(c) a StreetCrafter-style
> LiDAR-conditioned build of our own**.
>
> **Pre-committed outcomes:**
> - **≤1.5× OOD** → ⭐ **BUILD/ADOPT IT.** That matches our real-footage instrument's envelope and makes
>   collision metrics trustworthy. Commit the pod and the weeks.
> - **1.5×–2.5×** → **ADOPT only if it also brings reactive agents + real-time**, i.e. it must pay for
>   itself on a second axis. Otherwise hold.
> - **>2.5×** → 🔴 **DO NOT BUILD.** No better than NuRec. Fall back to §6's split rule and put the
>   effort into **encoder-side OOD alignment** instead (ranked #2 in
>   `2026-07-24-low-ood-closedloop-renderer.md`, ~1–2 pod-days, attacks the same 3.21× from the model
>   side and unlocks *every* reconstruction source at once).
>
> **Order of operations is load-bearing:** probe **before** build. The program's own meta-pattern —
> *five* over-claims of closure reopened by a cheap follow-up in a single session
> (`RETRACTION_LOG.md:58`) — argues for the $0 test first, every time.

### 7.3 Sequencing — what to do in what order

| Order | Action | Cost | Gate |
|---|---|---|---|
| 1 | **Commit the NRE pull script** + `git add -f` the four videos + fix `RUN_RECIPE.md` §13 | ~1 h | none — pure hygiene, closes the only stranding |
| 2 | **Turn on `trafficsim` (SMART/CAT-K)** and re-run one suite scene | ~1–3 d | none — Apache-2.0, already on the pod, never used |
| 3 | **Add a `TOOL_REGISTRY`** with the licence facts from §2 | ~0.5 d | none |
| 4 | ⭐ **Run the §7.2 OOD probe** on HUGSIM and on AlpaSim's OmniDreams backend | ~1 pod-day | eval pod free |
| 5 | **Renderer build** or **encoder-side alignment** | 2–6 wk / 1–2 d | **whichever branch §7.2 selects** |
| 6 | **Re-run the OOD control at native 1080×1920** to tighten the 3.21× | ~0.06 pod-day | opportunistic |
| — | ~~Fork AlpaSim~~ | — | **not scheduled; §7.1 is 0/3** |

---

## 8. Summary for the PI

**No fork.** AlpaSim's code is Apache-2.0 and we have changed one line of it in four months — there is
nothing to own. The renderer, which is where the entire 3.21× problem lives, is a closed NVIDIA binary
whose licence explicitly forbids modification, reverse engineering and redistribution, and which
disclaims autonomous-vehicle use as an uncertified Critical Application. **A fork would give us
custody of the part that works and no claim at all on the part that does not.**

Two of the four things a fork was meant to buy — **the reactive-agent model** (Apache-2.0, in-tree,
**and we have never switched it on**) and **map/lane awareness** (a full `trajdata.VectorMap` in every
scene USDZ, the exact feature our PhysicalAI corpus lacks) — are **already ours today, unforked**.
The third, scenario authoring, is limited by scene supply, not source control. The fourth, fixing
fidelity, is the one a fork cannot deliver.

The chief scientist is right that a reactive-agent renderer is the unowned dependency. The correct
response is to **build a renderer**, not to fork a wrapper around one — and AlpaSim already publishes
the plug for it. Gate that build behind a one-pod-day OOD probe with both outcomes pre-committed, so
we learn whether the candidate actually beats 3.21× before spending weeks. Meanwhile use the
real-footage harness for road-keeping and AlpaSim for collision, which is what each is honestly good at.

**Decision rule, in one line:** *fork only when we have three plugin-inexpressible changes, a rejected
upstream PR, and our own renderer already running — currently zero of three.*
