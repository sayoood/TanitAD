<title>Frontier Scan 2026-09-20 — the hierarchy edge finally has a published number and it is small, a BEV target is the worst of four, and a hard geometric target loses to no target at all</title>

# Frontier Scan · 2026-09-20 (LAB-RUN-017)

`22/22 tracks scanned · Band D ×5 claims, TWO standing debts CLOSED (D-7 after five passes, D-14) · FULL TEXT ×3 via local pypdf (Drive-HWM, 1604.06915, 2606.08860) + 3 dedicated Band-B deep-reads (B5 / B6 / B13) · 9 empties or blocks named · Library 521 → 529 · 1 correction to this register's own 09-19 framing`

⭐ **Scope of this run:** the four domain packages **were** produced today (three by this pass, one pre-existing) — gate half (a) and half (b) are both closed for 2026-09-20.

---

## 0 · Findings first — the six that change something

| # | finding | class | moves |
|---|---|---|---|
| **1** | ⭐⭐⭐ **H1b's missing datapoint exists, and it is +0.8 PDMS with two of three settings showing no edge at all.** Drive-HWM `2609.03572` (2026-09-03, **full text, local pypdf**) Table IV, NAVSIM v1: **`Fast only` (flat) 93.0 · `Slow only` 90.2 · hierarchical K=4 93.0 · K=8 93.8 · K=12 93.2.** ⇒ the **first published matched hierarchy-vs-flat ablation** we hold. But the edge is **+0.8 at one horizon**, **+0.0 at K=4** and **+0.2 at K=12**, one run per row, **no CIs**. The authors' own reading of the `Slow only` row: *"long-horizon predictions alone cannot promptly adapt to newly observed changes."* | PUBLISHED, local full text | ⭐⭐⭐ **backlog row 18 (H1b, never measured)**: the thesis now has an external number, and it is **modest and horizon-sensitive** — this should raise the priority of running H1b *and* lower the expected effect. Also the **three-planner hierarchy directive** |
| **2** | ⭐⭐⭐ **A BEV raster is the WORST of four world-model targets for motion information.** Drive-HWM Table VI, linear probes on frozen slow-world-model latents (FEM = future ego motion, MC = motion consistency): **BEV 71.2 / 63.8 · Depth 74.5 / 66.9 · RGB 76.8 / 69.4 · Optical Flow 83.7 / 76.1.** | PUBLISHED, local full text | ⛔⭐⭐ **a direct, actively-sought caution on FS19-3 and on the PI's SAM3-map-as-BEV-GT direction (2026-09-13)**: the BEV-raster-target arm is the one this evidence predicts will lose. ⇒ FS19-3 should carry a **flow** target arm and must not be run BEV-only |
| **3** | ⭐⭐⭐ **A hard geometric target UNDERPERFORMS having no such target.** DualPathOcc `2609.06370` (2026-09-06) Table 4, Occ3D-nuScenes: **one-hot depth supervision 35.96 mIoU < no depth supervision 36.92**; Gaussian depth (σ=1.5) **36.94** ≈ none. Authors: hard surface-centered targets create *"a mismatch between surface-based depth signals and volumetric occupancy learning."* | PUBLISHED (HTML ablation) | ⭐ with finding 2 and 09-19's DeepSight/World Tokens, that is now **four** independent 2026 measurements saying the *target* decides, and that the **intuitive** target is repeatedly the wrong one. ⇒ any new supervision we add needs a **no-supervision control arm**, which our prereg template does not currently require |
| **4** | ⭐⭐⭐ **I-1's untouched QUALITY half has its first external answer — and the value target is PROGRESS, not distance.** RISE `2602.11075`: a Compositional World Model pairs a controllable dynamics model with a **progress value model**, produces advantages from imagined rollouts, and updates the policy **without physical interaction**: **+35 % / +45 % / +35 %** absolute on three real-world manipulation tasks. | PUBLISHED (abstract-deep); ⚠️ **no matched no-self-improvement baseline in the abstract** — `H-ESTIM-SEED-1` class | ⭐⭐ **converges with today's own `E-DEP-VGEO-6`**: our frozen trunk beats pixels at **place** and ties/loses at **metric distance**, so a *progress* value head is what it can support. Two independent lines, one conclusion ⇒ strengthens **LR15-2** |
| **5** | ⭐⭐ **I-3's label-free half now has a route — in robotics, not driving.** RISE's loop is label-free at the point of improvement (advantages come from the imagined rollout and the progress model, not from stored expert labels). 09-19 established that the *driving* literature's TTA (DriveVLA-M0) is **label-supervised**. ⇒ the label-free form exists, one discipline over. | PUBLISHED (abstract-deep) | **I-3 served (partial, 2nd time)**. ⚠️ the progress value model is itself trained on something, and the transfer from tabletop manipulation to a 6 s driving horizon is a **HYPOTHESIS**, not a result |
| **6** | ⛔⭐ **The actively-sought counter-argument to our semantic-target thread, recorded.** WMPO `2511.09515` chooses **pixel space on purpose**: *"In contrast to widely used latent world models, WMPO focuses on pixel-based predictions that align the 'imagined' trajectories with the VLA features pretrained with web-scale images."* ⚠️ The abstract gives **no numbers and no matched latent baseline**, so the contradiction is **unquantified**. Note it is a *different* argument from Jim Fan's (N-19-2, `UNSUPPORTED-AS-STATED`): alignment-with-pretraining, not physics-from-pixels — and that version is **not yet refuted by anything we hold**. | PUBLISHED (abstract-deep) | ⭐ **T-12's falsifier should be widened**: FS19-3 currently pits semantic vs pixel targets on a from-scratch trunk; WMPO's argument only bites when the downstream consumer is **web-pretrained**, which our trunk is not. Recording the boundary keeps T-12 honest |

---

## 1 · Band D — two standing debts closed

⭐ Full seven-step adjudication in **`TanitAD Research Lab/Opponent Analysis/Research/2026-09-20-red-and-the-modularity-proof/RESULT.md`**; five rows (**W-20-1…4, M-20-1**) appended to `OPPONENT_CLAIMS_REGISTER.md`. Headlines:

- ⛔ **D-14 / Waymo `ReD` — `NOT-ADJUDICABLE-AS-PUBLISHED`.** The doctrine is a blog with **zero numbers**; the experiment is *Nature Comms* `s41467-026-73345-0`, which returns **303 → an IdP login**. ⇒ **no ReD number may enter our registry.** New PI-queue debt **D-15**.
- ⭐⭐⭐ **W-20-3, the concession of the pass (Waymo's own, on NIEON):** the model *"inherits the pre-conflict behaviors … of the ADS being evaluated"* and represents a driver *"that does not exist in the human population."* **A benchmark partly built out of the system under test** — structurally our own **T1 self-action open loop**. An opponent has documented our failure mode inside its safety case.
- ⭐⭐⭐ **D-7 CLOSED — and it changes sides.** `1604.06915` read in full: the *"exponentially larger"* result is a **construction**, not a theorem, driven by `c`-approximate independence over a **conjunctive** target. ⇒ it does not say *"modularity beats end-to-end"*; it says *"a task that factors is cheaper to learn factored."* **Our hierarchy is a factorisation claim, so this is an argument FOR our design** — and it hands `H1b` a precise burden: show **our** factorisation is approximately independent.
- ⛔ **Process finding:** D-7 was **banked in our own Library the entire time**. Five consecutive passes recorded it as "not reached". The cost of reaching it was one `find`. ⇒ a debt re-declared without re-probing its cheapest route is a habit, not a blocker.

## 2 · Band A

| track | read |
|---|---|
| **A1** world models | ⭐⭐⭐ **DEEP, full text** — Drive-HWM (findings 1, 2). Also Table VII: slow→fast conditioning **FiLM 93.8 > AdaLN 93.3 > GCA 93.1 > cross-attn 93.0 > concat 92.5** ⇒ published support for the FiLM conditioning our v7 predictor already uses |
| **A2** JEPA | ⭐⭐ **DEEP via A1's Table V** — as the *slow world model backbone*, **V-JEPA 93.8 beats CogVideo 93.0 and WAN 93.2**, i.e. a JEPA encoder beats two **video-generation** backbones inside a driving world model. ⚠️ single run, no CI. V-1 re-find: `2601.00844` (our I-1 founding citation) re-read locally — *"we **learn representations such that** the Euclidean distance … approximates the negative goal-conditioned value"*; the encoder is **shaped**, and the string *"frozen"* appears **0 times** |
| **A3** vision encoders | ⛔ **E-A3, 5th consecutive driving-term empty for a dedicated probe.** As on 09-19, the A3 evidence arrives only as **encoder-as-backbone** (A2 above). Named, not hardened |
| **A4** VLA | scan — DriveVLA-W0 is Drive-HWM's strongest baseline (93.0 PDMS, 1×Cam); AutoVLA 92.1 |
| **A5** benchmarks | ⛔⭐ **a stamp catch.** Drive-HWM's **EPDMS 86.4** is **NAVSIM v2 navtest**, not navhard — the same table carries an **`Ego Status` blind row at 64.0** and TransFuser at **76.7**, which are navtest magnitudes. ⇒ **86.4 does NOT enter the navhard table**, where **57.1** (DriveZero) still stands. This is FS18 row 32's two-stamp rule catching its second paper |

## 3 · Band B — three dedicated deep-reads (bar met)

| track | read |
|---|---|
| **B5** post-training / RL | ⭐ **DEEP (abstract)** — WMPO `2511.09515`, finding 6. Scanned: WAM-RL, TACO, RehearseVLA, WorldRFT |
| **B6** self-improving | ⭐⭐ **DEEP (abstract)** — RISE `2602.11075`, findings 4 and 5 |
| **B13** 3D / occupancy | ⭐⭐ **DEEP (HTML ablation)** — DualPathOcc `2609.06370`, finding 3. Component ladder from FlashOcc 32.08: Spatial Enhancer +3.23, height-aware loss +3.63, dual-path +4.12, +SENet +4.84, all +5.29 → **37.37 mIoU**. BEV compression runs **800×800 → 200×200** |
| **B12** memory / horizon | finding via A1's K-sweep: the planning horizon optimum is **interior** (K=8 beats both 4 and 12). ⚠️ counted as a **finding**, not a dedicated deep-read — it comes from A1's paper |
| **B1 / B2 / B3 / B4 / B7 / B8 / B9 / B10 / B11** | scanned; see §5 for the four named empties |

## 4 · Band C sweep

| track | read |
|---|---|
| **C1** releases | Waymo **14 US metros, ~4,000 vehicles, ~500 k paid rides/week** (Denver / San Diego / Tampa launched 2026-09-01). **Wayve** launched London's first self-driving taxi on Uber **2026-09-03** and closed **$1.2 B at an $8.6 B valuation** (NVIDIA, Stellantis). NVIDIA: physical AI ≈ **$6 B** annual revenue. `PUBLISHED-BLOG` / `RELAYED` |
| **C2** release notes | ⭐ **JetPack 7.2.1 ships Jetson Linux 39.2.1, CUDA 13.2.1, TensorRT 10.16.2.** ⇒ **Thor remains on the 10.16.x line**, which is exactly where `D-B1-GATE`'s silent-FP32-fallback hazard lives. **Second independent probe confirming LR15-10's premise**: TRT 11 (weak typing removed) is not reachable from JetPack, so the recipe must be made strongly-typed on 10.16.x. `PUBLISHED-RELEASE-NOTE` |
| **C3** regulatory | The **UN global technical regulation on ADS** was adopted at the **June 2026 WP.29 session** — the same session already identified for **UN R185** in debt **D-4**. Consistent with, and **not a substitute for**, the pinned document request. `PUBLISHED-BLOG` (UNECE press) |
| **C4** community | no navhard entry above **57.1** located; Drive-HWM's 86.4 is navtest (§2, A5) |

## 5 · ⛔ Coverage honesty

**22/22 tracks carry a named query.** **DEEP:** D1 (×2 opponents), A1 (full text), A2 (via A1 + a local re-read), A5, **B5, B6, B13** — three dedicated Band-B deep-reads, the bar met **without** counting B12's A1-derived finding. **A3 is not deep-read (E-A3, 5th).** **Full texts: 3 via local pypdf** (Drive-HWM, `1604.06915`, `2606.08860`) + **4 via the fetch summariser** (RISE, WMPO, DualPathOcc, the Waymo blog) — ⚠️ per **FS19-9** the summariser-sourced numbers in findings 3–6 are stamped and **may not enter the registry before a local re-read**; **finding 1 and 2's numbers were re-read locally and are registry-admissible.** **9 empties/blocks named** (§6). **Library 521 → 529.**

⚠️ **One clause under-served, stated rather than smoothed:** **B12** carries a finding but no dedicated probe of its own beyond the scan.

## 6 · Named empties and blocks

| id | probe | result |
|---|---|---|
| **E-A3** (5th) | dedicated vision-encoder probe with driving terms | ⛔ empty again; A3 evidence keeps arriving as encoder-as-backbone |
| **E-B2** (4th) | post-transformer → video/WM transfer | ⛔ all hits are LLM/image SSM (VideoSEMA, Spatial-Mamba, RS-SSM); no robotics/WM transfer |
| **E-B3** (2nd) | efficient decoding → rollout depth inside 100 ms | ⛔ all KV-cache/LLM-serving (CaliDrop, ShadowKV, OxyGen, UltraQuant); none prices a **world-model rollout** |
| **E-TSR** *(new)* | a standardised acceptance threshold for a deployed speed-limit reader | ⛔ empty — GTSRB-family accuracy only. ⇒ our 0.80 bar has **no published anchor** and must be described as a programme convention |
| **E-ABSTAIN** *(new)* | an abstention/no-read scoring convention for sign value accuracy (`2606.08860`) | ⛔ empty — precision and recall reported separately, never combined. The same hole LR15-5's gate has |
| **E-VPR-GEOM** *(new)* | frozen features: place recognition **vs** metric distance, contrasted across encoder stages | ⛔ empty at 1 probe — VPR work optimises retrieval; no located analogue for today's s32-vs-s16 × place-vs-geometry contrast |
| **E-D15** *(new)* | the `ReD` experiment (*Nature Comms*) | ⛔ **BLOCKED** — 303 to IdP login. PI-queue |
| **E-WV20** *(new)* | Wayve doctrine (Axios 2026-09-16) | ⛔ **BLOCKED** — HTTP 403. Next route: `wayve.ai` technical blog |
| **D-4** | UNECE GRVA / UN R185 primary | ⛔ **not re-probed** by standing instruction — blocked on a human action, not on a search |

## 7 · Next rotation — pre-committed

1. **Local pypdf re-read** of RISE, WMPO, DualPathOcc before any of their numbers enters the registry (FS19-9).
2. **B12 dedicated deep-read** — the one under-served clause today.
3. **D-11 (GAIA-4)** and **D-13 (Elluswamy transcript)** — untouched for three passes.
4. **E-WV20** via `wayve.ai`'s own technical blog, not a news aggregator.
5. **A3 with a changed term again** — five empties is a signal about the term, not about the field.
