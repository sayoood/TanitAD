<title>Frontier Scan 2026-09-17 — the imagination-necessity cluster, and the programme's oldest Band-D debt discharged</title>

# Frontier Scan · 2026-09-17 · LAB-RUN-014

**Band D ×2 adjudicated (D-1 discharged after 17 days) · Band A DEEP: A1 A2 A3 A5 · 3 FULL TEXTS · Library 503 → 507**
`Charter: DAILY_RESEARCH_CHARTER.md + v2 AMENDMENT. Search log with every empty: raw/search_log.md`

---

## 0 · The day in five findings

| # | finding | class |
|---|---|---|
| **FS17-1** | ⭐⭐⭐ **A controlled ablation says test-time imagination is unnecessary — while training-time future modelling is essential.** Fast-WAM `2603.16666`: skipping future prediction at inference matches imagine-then-execute (**RoboTwin 91.8** vs Joint 90.6 / IDM 91.3; **LIBERO 97.6** vs 98.5 / 98.0), but **removing video co-training collapses it to 83.8 / 93.5** — and to **10 %** on the real-robot task. **190 ms vs 810 ms, >4×.** ⇒ *"modelling the future is vital for learning representations; generating those frames at test time contributes very little."* | PUBLISHED lib `2603.16666` **FULL TEXT** |
| **FS17-2** | ⛔⭐⭐⭐ **WA-JEPA names our predictor's objective as the defect and prices the fix.** *"V-JEPA is built around random-mask completion and **deterministic regression**, making it **fundamentally ill-suited** for autonomous driving planning."* Ablation: masking **+0.4**, **flow matching over regression +1.0 EPDMS** — 2.5× the larger lever. ⚠️ **91.7 is `navtest`, not navhard** — excluded from our stamp table. | PUBLISHED lib `2608.20974` **FULL TEXT** |
| **FS17-3** | ⭐⭐⭐ **The A3 debt is discharged, and it lands on a LIVE decision.** FROST-Drive `2601.03460`, same backbone: **frozen InternVL3-14B RFS 8.17 / ADE@5s 1.88** beats **fine-tuned 8.13 / 2.19** — *"the fine-tuned 14B VLM performs worse than its frozen counterpart."* ⚠️ The frozen encoder is **300 M**, not 14 B. Open-loop only. | PUBLISHED lib `2601.03460` **FULL TEXT** |
| **FS17-4** | ⭐⭐ **D-1 DISCHARGED — "The Waymo World Model", unread since 2026-08-31.** Verdict: the world model is **for simulation only**, built on **Genie 3**, generating **4D point clouds** and multi-sensor camera+lidar output. ⛔ **No controlled experiment of any kind** — demonstration videos. ⭐ **Two concessions**, and they are ours: long-horizon stability, and the failure mode of reconstruction-based sim. | PUBLISHED-BLOG, adjudicated §1 |
| **FS17-5** | ⭐⭐ **A published admissibility ladder for world-model simulators, with a measured reversal.** `2607.07196`: a WM used as a test oracle must be **accredited** (L0 fidelity → L4 sim-to-real correlation). Applied to two driving WMs, *"the model that ranks higher on visual generation quality (L0) ranks **lower** on action-following (L1–L2), so visual fidelity does not predict the action-robustness a closed-loop verdict depends on."* | PUBLISHED lib `2607.07196` |

⭐⭐ **The day's thesis, and it is a position the programme does not currently hold.** FS17-1 and
FS17-2 are usually read as opposites — *"don't imagine"* vs *"imagine better"*. They are not. Both
are statements about **where** future modelling belongs: WA-JEPA improves the **training**
objective; Fast-WAM removes the **inference-time** rollout. Together: ⭐ **model the future
stochastically while training; do not roll it at inference.** v7 does the opposite on both counts —
a deterministic regression objective, rolled at planning time.

---

## 1 · ⭐⭐ BAND D — opponent doctrine (amendment §7.1, all seven steps)

### 1.1 Waymo — "The Waymo World Model" (2026-02) · register debt **D-1**, open since 2026-08-31

| step | |
|---|---|
| **1 · CLASSIFY** | Waymo corporate blog, 2026-02, published while Waymo was scaling 6th-gen operations and NVIDIA/Wayve were marketing end-to-end learned stacks. Evidence class **PUBLISHED-BLOG**, never `PUBLISHED`. |
| **2 · ⛔ IS IT AN EXPERIMENT?** | ⛔ **NO — and not even weakly.** There is **no ablation, no comparative evaluation, no statistical test, no metric**. The only quantities are exposure counts (*"nearly 200 million fully autonomous miles"*, *"billions of miles in virtual worlds"*). Support is **demonstration videos**. This is the purest instance of attribution-from-exposure the register has logged: the 10-lessons post at least described design choices; this one shows outputs. |
| **3 · CONFIRMING (independent)** | The *simulation-is-valuable* premise is independently supported: `2607.07196` treats WM-as-test-oracle as the live assurance question, and NVIDIA ships **AlpaGym** to do it. Waymo is not alone in building one. |
| **4 · ⭐ CONTRADICTING (actively sought — D-Q7)** | ⭐⭐ **Found, and it is a whole literature.** The claim *"a driving world model's role is simulation, decoupled from the policy"* is contradicted by **PWM `2510.19654`** (*"unify world modeling and trajectory planning within a unified architecture"*), **GraphWorld `2606.16274`** (long-horizon planning *with* world models), **UniDrive-WM `2601.04453`**, **DriveWAM `2605.28544`**, **CoWorld-VLA `2605.10426`**, **DAWN `2605.11550`**, and **WA-JEPA** (FS17-2). The field's own framing: *"world models are mostly learned for world simulation and decoupled from trajectory planning"* — stated as **the gap being closed**, not the settled answer. |
| **5 · VERDICT** | **`UNSUPPORTED-AS-STATED`** for any reading of the post as *"world models belong in simulation rather than onboard"*. ⚠️ **Scope it honestly: the post never asserts that.** It describes what *Waymo's* WM is for and is silent on the onboard question. The unsupported claim is the one a reader would take away, and the post's framing invites it. |
| **6 · ⭐⭐ DOES IT BIND ON US?** | ⛔ **NO — and conflating the two would be a category error.** Waymo's object is a **generative multi-sensor scene simulator** (Genie 3, 4D point clouds, camera + lidar) whose product is *training and test data*. Ours is a **latent predictive model** whose product is *a plan*. Different artifact, different output, different success criterion. A high-fidelity video simulator saying nothing about latent planning is not evidence against latent planning. ⚠️ **Where it DOES bind:** on **backlog row 14**, our NuRec/gsplat pseudo-simulation pilot — that *is* the same artifact class. |
| **7 · GUIDELINE** | **STRATEGIC S-5:** the programme argues that *"world model"* names **two different artifacts** — a **generative simulator** (Waymo, Genie 3, AlpaGym) and a **latent predictive planner** (ours, JEPA-family) — and that evidence about one is not evidence about the other. State this in the paper; it is the cheapest available defence against our largest opponent's framing. **TACTICAL T-6:** before row 14's pilot is designed, adopt `2607.07196`'s ladder and state which rung the pilot targets (§2.3). |

⭐ **The concessions — the sentences that cost the author something (amendment §7.2):**

| claim | why it matters to us |
|---|---|
| ⭐⭐ *"the longer the simulation, the tougher it is to compute and maintain stable quality"* | **Long-horizon rollout stability is conceded as unsolved by the best-resourced opponent in the field.** Our own drift and the k=60 BPTT divergence are the same failure family. ⇒ this is not a TanitAD weakness; it is a **field-wide open problem**, and our measurement of it is an asset rather than an embarrassment. |
| ⭐⭐ *"purely reconstructive simulation methods … suffer from visual breakdowns due to missing observations"* | ⛔ **This is a direct, named hazard for backlog row 14.** Our pilot is exactly a *reconstructive* method (NuRec / 3DGS replay of recorded scenes). Waymo names its failure mode — missing observations under counterfactual viewpoints — which is precisely what a perturbation grid creates. **Row 14 must carry this as a pre-registered risk.** |

⚠️ **Opponent strengths, recorded (a Band-D package listing none is INCOMPLETE):** multi-sensor
generative simulation producing **camera and lidar together** is a capability we do not have and
have no path to; **Genie 3** gives them a frontier generative backbone through DeepMind that no
independent programme can match; and the exposure record (200 M autonomous miles) is real evidence
about *deployment*, whatever its weakness as evidence about *architecture*.

### 1.2 NVIDIA — Alpamayo 2 Super + AlpaGym (2026-08-04, read 2026-09-17)

| step | |
|---|---|
| **1 · CLASSIFY** | NVIDIA blog + HF model card, released 2026-08-04 into a market where NVIDIA sells the compute its own models require. **PUBLISHED-BLOG** / **PUBLISHED-CARD**. |
| **2 · ⛔ IS IT AN EXPERIMENT?** | **Split.** The **closed-loop-training rationale** (*"AlpaGym exposes compounding errors and edge-case failures that static datasets miss"*) is an **argument, not an ablation** — no controlled comparison of closed-loop-trained vs static-trained is offered. The **SOTA claim** (*"state-of-the-art performance in … reasoning quality, trajectory accuracy, alignment"*) carries **no baseline table** in the blog ⇒ **UNSUPPORTED-AS-STATED**. ⭐ The **model card**, by contrast, does publish three numbers. |
| **3 · CONFIRMING** | The compounding-error argument is the classical DAgger/covariate-shift result and is independently supported by **our own** measurement — open-loop S-curve reproduction **97.9 %** vs **0.0 %** hold-action vs ~5 % closed-loop. Independent of NVIDIA. |
| **4 · ⭐ CONTRADICTING (actively sought)** | ⚠️ **On the rationale: none found, and the counter-search is recorded as returning nothing.** On the **scale** premise, contradicting evidence is strong: FROST-Drive (FS17-3) shows a **frozen 300 M** encoder beating fine-tuning at 14 B, and Fast-WAM (FS17-1) shows a **6 B** model matching imagine-then-execute while skipping the expensive half. Bigger is not where these gains came from. |
| **5 · VERDICT** | closed-loop-training rationale → **`SUPPORTED`** (argument + independent confirmation, no ablation of its own). SOTA claim → **`UNSUPPORTED-AS-STATED`** in the blog. |
| **6 · ⭐⭐ DOES IT BIND ON US?** | ✅ **The rationale binds and we already comply** — it is our own open/closed-loop ruling, arrived at independently. ⛔ **The scale does not bind**: 34 B is 100× our ceiling. ⭐ **What binds hardest is the SUPPLY change** — see the licence row below. |
| **7 · GUIDELINE** | **STRATEGIC S-6:** NVIDIA moving the **entire Alpamayo lineup to OpenMDW-1.1** (permissive, commercial redistribution, no field-of-use restriction) changes the opponent landscape from *"closed frontier models"* to *"a commercially usable 34 B baseline anyone can fine-tune"*. Our differentiation can no longer be *access*; it must be **efficiency and hierarchy**, measured. **TACTICAL T-7:** re-check register row **N-3** (the Alpamayo licence split) against OpenMDW-1.1 — the restriction it records may be **superseded**. |

⚠️ **The numbers, stamped (V-5 discipline):** **34 B total = 32 B Cosmos 3 Super Reasoner backbone
+ 2.3 B action expert.** Lingo-Judge **79.2**; **AlpaSim closed-loop 1.50 ± 0.13**; open-loop
**minADE₆ @ 6.4 s = 0.911 m**. ⛔ **`minADE₆` is best-of-6 and is NOT our `fwd_ade`** — it may not
enter any TanitAD table. ⚠️ The blog says *"32-billion parameter VLM backbone"* while three
secondaries say *"34B model"*; **both are right about different objects**, and the card settles it.
That is the same params-token trap as 09-09, caught before it could be quoted.

⚠️ **Opponent strengths, recorded:** a **published closed-loop score** (AlpaSim 1.50 ± 0.13 **with an
interval**) is more than most of the field offers; shipping **AlpaGym** as closed-loop *training*
infrastructure, not just evaluation; and the permissive licence, which is a genuine strategic
concession of moat in exchange for ecosystem.

---

## 2 · Band A deep-reads — the five dimensions

### 2.1 ⭐⭐⭐ Fast-WAM `2603.16666` (FULL TEXT) — A1 / B3

**Design:** three variants share training and differ at inference — Fast-WAM (no test-time future),
Fast-WAM-Joint (shared attention over video+action during denoising), Fast-WAM-IDM (generate video,
then condition). Plus a **direct control**: Fast-WAM **without video co-training**. 6 B total
(5 B video DiT from Wan2.2 + 1 B action expert). LIBERO 2,000 trials / 40 tasks; RoboTwin 100 trials
per task. ⚠️ **No std, no seed count** — a real weakness for a 91.8-vs-91.3 comparison.

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | Strikes the programme's central inference-time mechanism. Touches row 5 (h≥2 heads), P-15, I-1's rollout budget, and the whole *imagination-in-the-loop* thesis. |
| **CONSEQUENCE** | ⭐⭐ **It splits our thesis into two claims that must now be evidenced separately:** (a) *a world model is the right thing to TRAIN* — Fast-WAM **supports** this, strongly and with the sharpest number in the paper (83.8 → 91.8 when co-training is added back, and 10 % → high on the real robot); (b) *rolling the world model at inference is what makes the planner good* — Fast-WAM finds **no measurable benefit** on its benchmarks. ⛔ Our planner rolls at inference. |
| **COMBINATION** | ⭐⭐⭐ It **combines with our own most stubborn measurement rather than contradicting it**: our h=1 action/scene ratio is **0.004** — the predictor barely conditions on action — and our closed-loop degradation (0.45 → 1.69 m). A rollout whose predictor is action-insensitive is nearly a **constant** unrolled; Fast-WAM's result says you can drop such a rollout with no loss, which is exactly what our numbers would predict. ⭐ It also **combines with WA-JEPA** into the day's thesis: *stochastic future in TRAINING, no rollout at INFERENCE.* ⚠️ And it **cuts against I-1's framing**: if imagination is not needed at test time, the value-vs-fan-scoring question (breadth 5.94× cheaper than depth) is answered from a different direction — there may be nothing to be deep about. |
| **CHANCES / RISKS** | **Upside:** a 4× latency cut inside our 100 ms budget, and a principled reason to stop paying for rollout depth. **Risks:** ⛔ **no driving experiment at all** — LIBERO, RoboTwin and towel-folding are tabletop manipulation; **no seeds/std**; the authors *"omit the outer auto-regressive loop"*, which is precisely the long-horizon regime driving needs; and their own real-world result concedes *"pretrained π₀.₅ remains the strongest method"*. ⇒ **the transfer to driving is a HYPOTHESIS.** |
| **EXPERIMENT** | **`E-AI-NOIMAG-1` (0 GPU on banked dumps):** on an existing v7 arm, score planning with the **composed h=1 rollout** against the **same planner with the rollout replaced by a single-step readout**, on identical windows. **Committed in advance:** if the no-rollout arm is within the rig's replicate noise floor, our inference-time imagination is **not paying for itself** and the latency belongs elsewhere; if it degrades beyond that floor, imagination is load-bearing for *us* and Fast-WAM does not transfer — **either answer is publishable and neither is currently known.** ⛔ Per `H-ESTIM-SEED-1` the comparison needs a **replicate arm**, and per the refav1 rule an **inference-seed** replicate too if the planner samples. |

### 2.2 ⭐⭐⭐ WA-JEPA `2608.20974` (FULL TEXT) — A2 / B7

*Full five-dimension treatment in today's A&I package (`…/2026-09-17-frozen-trunk-external-evidence/RESULT.md` §2(ii)), with the pre-registered `E-AI-FLOW-1`.* Headline for the ledger: the criticism is
of **deterministic regression**, the fix is **conditional flow matching over latent futures**, the
ablation prices it at **+1.0 EPDMS** against **+0.4** for the masking change, and the headline number
is **navtest**.

### 2.3 ⭐⭐ `2607.07196` admissibility ladder — A5 / C3

**L0** visual fidelity → **L1** action-responsiveness → **L2** declared operating envelope with OOD
detection → **L3** failure attribution separating simulator from policy error → **L4** measured
sim-to-real correlation. Authors Oefinger, Schäfer, Moller, Piccinini, Betz; 2026-07-08.

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | **Backlog row 14** (pseudo-simulation pilot) has no admissibility criterion; this supplies a published one. Also G3 and the Phase-1 safety case (row 30). |
| **CONSEQUENCE** | ⭐ **Row 14's design changes before it is run.** Its stated target — *"does it hold R² ≥ 0.7 against our T1?"* — is an **L4** claim (measured sim-to-real correlation), which is the **top** rung. The ladder says L1–L3 must be cleared first, and our pilot currently has no L1 (action-responsiveness) check at all. ⇒ **row 14 should be re-scoped to target L1–L2 first**, which is also far cheaper. |
| **COMBINATION** | ⭐⭐ The measured **reversal** (higher L0 ranks *lower* on L1–L2) is **our own probe-panel doctrine arriving from outside**: a metric that rewards surface quality (FVD, visual realism) can be **anti-correlated** with the property the verdict depends on. That is the `CLAUDE.md` ridge-probe lesson — *a control must read a known value, and the impressive number is the one to distrust* — in an assurance costume. It also sharpens Waymo's W-17-4 concession: reconstruction quality is exactly an **L0** virtue. |
| **CHANCES / RISKS** | **Upside:** a vocabulary for what our pseudo-sim pilot would be *allowed to conclude*, and a cheap re-scope. **Risks:** the rung definitions were not recoverable from the abstract page (see limits below), so the ladder is adopted by **name and ordering** and not yet by operational definition; n = 2 driving WMs for the reversal. |
| **EXPERIMENT** | **`E-BE-LADDER-1` (0 GPU):** read the full text, extract the operational definition of **L1 action-responsiveness**, and apply it to the NuRec/gsplat assets we already hold. **Committed:** if our reconstructive assets cannot be made action-responsive (they replay recorded observations, so a counterfactual action has no response — Waymo's W-17-4 concession names exactly this), then **row 14 is capped at L0 by construction** and must be re-scoped or closed, and we say so. |

⚠️ **Limit on 2.3:** the L0–L4 definitions above come from the **abstract and a secondary summary**;
the fetch of the abs page did not render the per-rung definitions. The PDF **is banked**
(`2607.07196`, 293 KB) and is a **full-text debt** carried to the next pass. No rung threshold is
quoted as a number anywhere in this pass.

---

## 3 · Band C sweep

| track | finding | class |
|---|---|---|
| **C1** | **Alpamayo 2 Super**, 2026-08-04: 34 B VLA, **OpenMDW-1.1 across the whole lineup**, Apache-2.0 code, commercial redistribution and derivative models permitted, no field-of-use or geographic restriction; 400 k cumulative reasoning-model downloads claimed. | PUBLISHED-CARD / PUBLISHED-BLOG |
| **C2** | **JetPack 7.2.1 = Jetson Linux 39.2.1 · CUDA 13.2.1 · TensorRT 10.16.2**; DRIVE OS 7.0.3 on Thor. ⭐ **Consistent with 09-13's finding that TensorRT 11.2.1 excludes JetPack** — the Jetson line is on the 10.x branch. Relevant to **FS-4** (re-run the FP8/FP4 census on 7.2.1). ⛔ **No September-2026 note at a fifth probe.** | PUBLISHED-RELEASE-NOTE |
| **C3** | UN **GTR on ADS** adopted at WP.29 **23–26 June 2026** (GRVA text agreed 19–23 Jan 2026); NHTSA's GTR comment period **closed 2026-02-23**; NHTSA **foot-brake NPRM 2026-06-26**, comments closed **07-27**, **no final rule**; ⭐ **today, 2026-09-17: NHTSA places the Tesla Cybercab under sworn oath regarding the foot-brake requirement.** | RELAYED / PUBLISHED-BLOG |
| **C4** | Waymo E2E Driving Challenge leaderboard (FROST-Drive **3rd**); HUGSIM closed-loop, **436 scenarios**; NAVSIM **navtest** board (WA-JEPA 91.7 / Discrete-WAM 90.4 / SparseDriveV2 90.1). ⚠️ **scan-through only, no dedicated leaderboard probe.** | PUBLISHED |

---

## 4 · ⛔ Completeness statement (charter §6, stated rather than smoothed)

| requirement | status |
|---|---|
| Band D — every open item adjudicated | ✅ **2 adjudicated; the standing debt D-1 is DISCHARGED after 17 days.** ⚠️ D-2 (*Demonstrably Safe AI*) and D-3 (Foundation Model post) remain open |
| ≥ 2 FULL-TEXT reads (V-2) | ✅ **3** — Fast-WAM, WA-JEPA, FROST-Drive |
| Band A rotation | ✅ A1, A2, **A3 (debt discharged)**, A5 DEEP. ⚠️ **A4 scan-only** |
| ≥ 3 Band-B deep-reads, staleness-ordered | ⚠️ **PARTIAL, and this is the pass's weakest column.** B3 and B7 are served by *transfer numbers inside A-band full texts*, not by dedicated Band-B primaries; B5, B8, B9 were scanned only. **Counted as ~2, not 3.** |
| Band C sweep | ✅ C1, C2, C3 with findings; C4 scan-through |
| All 22 tracks scanned | ⛔ **NO — 21 of 22. B11 (physics-informed operators) carries no query today** |
| Ledger appends | ✅ §5 |
| Every empty named | ✅ 9 empties in `raw/search_log.md`, each with the scope it does and does not support |

⛔ **Per charter §6 this pass is INCOMPLETE on two counts — one unscanned track (B11) and a thin
Band-B deep column — and says so here rather than narrowing the claim.**

## 5 · Ledger appends and register motions (all in this turn)

`LEDGER_A1_world_models.md` · `LEDGER_A2_jepa.md` · `LEDGER_A3_vision_encoders.md` ·
`LEDGER_A5_benchmarks.md` · `LEDGER_B7_diffusion_flow.md` · `LEDGER_C1_releases.md` ·
`LEDGER_C3_regulatory.md` · `Opponent Analysis/OPPONENT_CLAIMS_REGISTER.md` (**W-17-1…4**,
**N-17-1…3**) · `Frontier Scan/TRACKS.md` (ninth pass).

## 6 · Banked today

| key | title | how read |
|---|---|---|
| `2603.16666` | Fast-WAM: Do World Action Models Need Test-time Future Imagination? | **FULL TEXT** |
| `2608.20974` | WA-JEPA: Rethinking the Video JEPA Paradigm for World-Action Modeling | **FULL TEXT** |
| `2607.07196` | Validate the Dream Before You Trust Its Verdict | abstract + secondary; **PDF banked, full-text debt** |
| `2608.07409` | UniJEPA | abstract-only |
| `2601.03460` | FROST-Drive | **FULL TEXT** — already banked, cited-by updated (V-1 re-find) |
| `2606.31232` | Delta-JEPA | ⛔ **banked, UNREAD** — PDF extraction failed (E-AI4) |

**Library 503 → 507 entries, 3,438.2 MB.**
