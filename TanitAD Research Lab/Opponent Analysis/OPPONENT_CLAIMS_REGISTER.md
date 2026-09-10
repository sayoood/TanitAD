<title>OPPONENT CLAIMS REGISTER</title>

# OPPONENT CLAIMS REGISTER

⛔ **APPEND-ONLY.** One row per adjudicated opponent claim. A verdict is **re-opened** when new
evidence arrives — that is why this file exists and a one-off report does not.
`Protocol: DAILY_RESEARCH_CHARTER_v2_AMENDMENT.md §7.1 (seven steps, counter-search mandatory).`

**Verdicts:** `CONFIRMS-US` · `SUPPORTED` · `CONTESTED` · `UNSUPPORTED-AS-STATED` · `REFUTED`.
**Binds:** does the claim constrain TanitAD, given that our constraints differ from theirs?

---

## Waymo — *"10 AI Lessons from Driving 200+ Million Fully Autonomous Miles"* (2026-08-26)

`Adjudicated 2026-08-31 → Opponent Analysis/Research/2026-08-31-waymo-10-lessons-adjudication/RESULT.md`
`⛔ Document-level finding: ZERO controlled ablations across all ten lessons. Evidence class PUBLISHED-BLOG throughout.`

| # | claim | verdict | binds on us? | what would flip it |
|---|---|---|---|---|
| **W-1** | Multimodal sensors are indispensable; camera-only is insufficient for L4 | ⚖️ **CONTESTED** | ⛔ **NO — category mismatch.** Their claim is about *sensor capability*; our vision-only rule is *leak avoidance* (labels may use ego, inference may not). They do not meet. | A controlled camera-only-vs-multimodal ablation at matched compute and data. **None found on either side.** |
| **W-2** | HD maps are a powerful prior | ✅ **SUPPORTED** as *helpful*; **NOT** as *necessary* | ⛔ **YES — names a real structural weakness.** PhysicalAI-AV has no map/lane-graph/route and no GNSS (5 probes). Our only route is NuRec `map.xodr`. | A mapless system matching a mapped one on the same route distribution. Online lane-graph work is the live counter-line. |
| **W-3** | Fewer, larger models are better ("scaling laws") | ⚠️ **UNSUPPORTED-AS-STATED** — and **self-undercutting** | ⭐ **YES, in our favour.** Their own *"teacher-student to optimize onboard compute"* concedes they **deploy small**. Alpamayo: 10.5 B→34.3 B moves open-loop minADE₆ 0.916→0.911 m (0.5 %), closed-loop intervals overlap. | A controlled scaling study on a *trajectory* metric showing monotone gains. Vendor self-report is not one. |
| **W-4** | Pure end-to-end is a black box; an independent validation layer is required | ✅ **SUPPORTED** | ✅ YES — argues **for** an inspectable hierarchy, i.e. our thesis. Also implies their own E2E model is not trusted alone. | — (standard safety-architecture position) |
| **W-5** | Closed-loop simulation reveals more edge cases; open-loop traffic is *"completely indifferent to your actions"* | ⭐⭐⭐ **CONFIRMS-US** | ✅ **YES — strongest item in the article.** Third independent line agreeing with `EVAL_DOCTRINE`'s T0/T1 split. | Would require showing open-loop rank-orders arms the same as closed-loop. Our own data (97.9 % vs 0.0 % hold-action) says it does not. |
| **W-6** | A learned "Critic" stops the system *"grading its own homework"* | ⚠️ **CONTESTED** — goal right, **independence asserted not demonstrated** | ✅ YES — adopt the pattern, **with controls**. `2606.03238` / `2604.13602`: learned critics fail *correlated* with the policy. | An independence argument: the critic must be able to fail where the policy cannot. Waymo neither defines nor evidences this. |
| **W-7** | VLMs help reasoning but are *"too slow for real-time control"* and *"lack sufficient spatial awareness"* | ⭐⭐⭐ **CONFIRMS-US ×3** | ✅ YES — corroborates (a) the fast/slow hierarchy, (b) our measured latency wall (203 ms @ 37.8 M, K=60), (c) ⭐ our v6 geometry ceiling as a **property of semantic vision stacks, not our bug**. | A VLM demonstrating in-loop latency **and** metric spatial precision. |
| **W-8** | AI effectiveness depends on safety governance | ✅ SUPPORTED, low information | organisational | — |
| **W-9** | A data flywheel enables continuous improvement | ✅ SUPPORTED, low information | already our org design | — |
| **W-10** | L2→L4 progression is *"a false summit"* | ✅ SUPPORTED as competitive thesis | ⛔ does not engage TanitAD (we are neither) | Transferable content = W-5 restated: supervised capability overstates unsupervised capability. |

---

## NVIDIA — Alpamayo model-card claims (adjudicated 2026-08-31)

| # | claim | verdict | binds on us? | what would flip it |
|---|---|---|---|---|
| **N-1** | Scaling the VLA backbone improves AV capability | ⚠️ **UNSUPPORTED on trajectory** — 0.5 % open-loop over 3.3× params; closed-loop intervals overlap; **language scales (74.2→79.2)** | ⭐ YES, in our favour — external support for sub-300M | A controlled scaling study with intervals on a trajectory metric. |
| **N-2** | Alpamayo is trained on `PhysicalAI-Autonomous-Vehicles` (+ `-NuRec`) | ✅ **SUPPORTED** (their own card) | ✅ YES — first possible **same-corpus** external opponent | Split-overlap check vs parity key `e438721ae894` (**OPEN — backlog P-3**) |
| **N-3** | OpenMDW-1.1 licence terms | ⚠️ **CONTESTED** — the 1.5 card says *"non-commercial use... commercial licensing on request"*; 2-Super says it *"permits commercial deployment globally"* | immaterial for research use; **material if anything ships** | reading the licence text itself (**not done**) |

---

## ⛔ Open verification debts carried by this register

| # | debt | owner |
|---|---|---|
| **D-1** | `2505.06113`'s **85.1 % / 92.1 %** camera-vs-lidar parity figures are **RELAYED** and are the most decision-relevant numbers in the W-1 debate | next Band-D pass (backlog T-3) |
| **D-2** | The architectural-convergence claim (*Waymo's model is end-to-end "just like Tesla and Wayve"*) is **RELAYED**, and it is the single most load-bearing counter to W-1's framing | next Band-D pass |
| **D-3** | Waymo's three linked technical sources — *Demonstrably Safe AI* (2025-12), Waymo Foundation Model (2025-12), **The Waymo World Model (2026-02)** — are **unread**. Any claim about *how* they do these things is unverified. | next Band-D pass — the World Model post is the highest-value unread document currently known |


---

## Waymo — *"The Waymo World Model"* (2026-02-06) + *"Demonstrably Safe AI"* (2025-12)

`Adjudicated 2026-09-01 → Frontier Scan/Daily/2026-09-01/RESULT.md`
`⛔ Document-level finding: ZERO ablations in EITHER document. Evidence class PUBLISHED-BLOG throughout.`
`⭐ These two primaries DISCHARGE register debts D-2 and D-3, and CORRECT the register's own D-2 framing.`

| # | claim | verdict | binds on us? | what would flip it |
|---|---|---|---|---|
| **W-11** | The Waymo World Model is a frontier generative model for **large-scale AV simulation**, built on Genie 3 via "specialized post-training" | ✅ **SUPPORTED** as a description of *purpose*; **UNEVALUATED** as a capability — no metric or ablation of any kind is published | ⛔ **NO — category mismatch, and this MATTERS.** Their WM **renders futures for training/testing**; ours **predicts futures inside the driving loop**. Different products on different axes. ⇒ Their announcement is **not** competitive pressure on our thesis and must stop being read as such. | Any published simulation-fidelity or downstream-policy-transfer metric. **None exists in the document.** |
| **W-12** | ⭐ **CONCESSION** — *"the longer the simulation, the tougher it is to compute and maintain stable quality"* | ⭐⭐⭐ **CONFIRMS-US** | ✅ **YES, strongly in our favour.** Long-horizon rollout stability is **our** measured problem (drift 0.4531 → 0.36–0.40 under EMA, MM-E1). **Waymo, with Genie 3 and Google-scale compute, concedes it is unsolved.** ⇒ We are not behind on a solved problem. Same shape as the B13 geometry reframing: *a field property, not our bug.* | A world model demonstrating stable quality at extended horizon. **Independently counter-searched:** WorldRoamBench (`2606.31672`) evaluates 10+ IWMs and reports *"None reliably satisfies all dimensions"* — the concession is **corroborated, not merely admitted**. |
| **W-13** | The Waymo Foundation Model's hybrid design — *"learned embeddings as a rich interface between model components"* with *"full end-to-end signal backpropagation"* — beats **both** pure-E2E and modular | ⚠️ **UNSUPPORTED-AS-STATED** (advocacy, no ablation) — but ⭐⭐ **the ARCHITECTURE ITSELF CONFIRMS-US** | ✅ **YES, in our favour.** Sensor Fusion Encoder (fast) + Driving VLM (slow semantic) → World Decoder **is a fast/slow inspectable hierarchy with differentiable interfaces** — TanitAD's thesis in an opponent's production stack. ⛔ **But adopting it *because Waymo does it* would be the exact `INHERITED` failure the operating standard bans.** It raises our prior; it decides no GPU-day. | A controlled three-way (embedding-interface vs modular vs pure-E2E) at matched params. **Neither Waymo nor anyone else has published one.** Ours is proposed on the v7-tiny ladder. |
| **W-14** | ⭐ **CONCESSION** — *"these Teacher models are too big to run on vehicles for real-time decision making"*; teachers are distilled into students for onboard use | ⭐⭐⭐ **CONFIRMS-US** — and it is now **PRIMARY**, no longer relayed | ✅ **YES.** This is **W-3's self-undercut promoted to a primary source**: Waymo **trains large and deploys small**. External support for the sub-300M deployment thesis from the largest operator in the field. | A production AV stack running its largest model on-vehicle. Waymo's own text says they do not. |
| **W-15** | *">ten-fold reduction in crashes with serious injuries compared to human drivers"* over *">100 million fully autonomous miles"* | ✅ **SUPPORTED as an exposure record**; ⛔ **attribution-from-exposure, NOT an experiment** — no counterfactual separates *what they built* from *what is necessary* | ⛔ **NO.** It is a fleet-safety outcome, not an architectural argument. It cannot adjudicate any design question. ⚠️ **But it is a genuine opponent strength and is recorded as one** (§7.2). | A matched comparison isolating any single architectural choice. Not offered, and arguably not obtainable from fleet data. |

### ⛔ Corrections to this register, issued 2026-09-01

| row | correction |
|---|---|
| **D-1** *(debt)* | ⛔ **DISCHARGED AND THE NUMBERS DO NOT SUPPORT THEIR USE.** `2505.06113`'s abstract reads *"up to 85% road segmentation accuracy and 85-90% vehicle detection rates when compared against LiDAR ground truth, with average positional errors limited to 1.2 meters"*. **(a)** There is **no camera-vs-lidar comparison** — lidar is the **ground truth**, so this measures *agreement with a reference*, which cannot establish **parity of capability**. **(b)** The relayed version **dropped the 1.2 m positional error**, the one decision-relevant number (our deployed v1 is 0.452 m `fwd_ade`). **(c)** *"92.1 %"* **does not appear**. **(d)** Single author, no venue, not peer-reviewed, no ablation. ⇒ **W-1 stays CONTESTED, but we no longer hold the counter-evidence we believed we held.** *(V-5: the scope error was the corruption path.)* |
| **D-2** *(debt)* | ⛔ **DISCHARGED AND REFUTED AS FRAMED.** The convergence claim *"Waymo's model is end-to-end just like Tesla and Wayve"* is **UNSUPPORTED-AS-STATED** by the primary, which describes an explicitly **hybrid** modular/E2E design (W-13). ⇒ W-1's framing weakens with it. |
| **D-3** *(debt)* | ✅ **DISCHARGED.** *The Waymo World Model* (2026-02) and *Demonstrably Safe AI* (2025-12) both read → W-11…W-15. ⚠️ *Waymo Foundation Model* (2025-12) remains covered only via the Demonstrably-Safe-AI post; a dedicated read is not owed unless a claim depends on it. |

### Opponent strengths recorded (§7.2 — a Band-D package listing none is INCOMPLETE)

1. **A Genie-3-derived simulator + Critic + RL-in-sim is a closed-loop training loop we do not have and cannot cheaply build.**
2. **Their closed-loop evaluation culture is ahead of ours**, and is stated as architecture (Driver/Simulator/Critic *"fueled by the same underlying AI"*), not aspiration.
3. **They have solved the compression problem in production** — teacher→student distillation running on-vehicle in a commercial fleet. We hold the thesis; they hold the deployment.
4. **>100 M autonomous miles.** Unablated, but an exposure record no research programme can match.

### ⛔ Open verification debts carried forward

| # | debt | owner |
|---|---|---|
| **D-4** *(new)* | The **UNECE GRVA primary text** (draft ADS regulation) is **403-blocked and UNREAD**. The "online-learning ban" is *unsupported at two probes*, **not refuted**. Blocks injected row **I-3**'s design. | next Band-D/C3 pass |
| **D-5** *(new)* | **SparseOcc++ `2607.04732` supervision requirements** — the one question that decides reachability for us — are **not in the abstract**. ⛔ No design work until a full-text read answers it. | next B13 pass |
| **D-6** *(new)* | **A2 (JEPA) and A4 (VLA) had no DEEP this pass.** Band A is `PARTIAL` and says so. | next pass, top of Band A rotation |


---

## Mobileye — *"Compound AI: The framework powering scalable autonomy"* (2025-07-31) + *"On the Sample Complexity of End-to-end Training vs. Semantic Abstraction Training"* (arXiv 1604.06915, 2016)

`Adjudicated 2026-09-02 under DAILY_RESEARCH_CHARTER_v2_AMENDMENT.md §7.1, all seven steps.`
`⭐ FIRST Band-D item backed by a PROOF rather than by advocacy or exposure — a case the protocol had not met.`
`Primary banked this pass: lib 1604.06915.`

### Document classification (step 1)

| | |
|---|---|
| **Doctrine post** | Mobileye blog, **2025-07-31**. `PUBLISHED-BLOG`. |
| **Academic primary** | Shalev-Shwartz & Shashua, arXiv **1604.06915**, 2016-04-23. `PUBLISHED` (preprint, no venue). ⚠️ **Abstract + framing read; FULL TEXT NOT READ — the construction behind *"cases in which"* is unexamined (new debt D-7).** |
| **Competitive context** | Mobileye is a Tier-1 supplier that **sells modularity as a product**; the post is advocacy for the architecture its business model depends on, published while competitors marketed end-to-end. |

### Is it an experiment? (step 2)

⛔ **The blog: NO.** Zero ablations, zero A/B, zero matched comparison. Its only external citation is
Berkeley AI Research's general remark that *"state-of-the-art AI results are increasingly obtained by
compound systems"*. Its only numbers are hardware specs (EyeQ6 High **34 TOPS INT8**, *"over 1,000 frames
per second on pixel-labeling neural networks"*) — **specifications, not evidence for the architecture claim.**

⭐⭐ **The 2016 paper: NEITHER an experiment NOR attribution-from-exposure — a PROOF.** *"We demonstrate
cases in which the number of training examples required by the end-to-end approach is exponentially larger
than the number of examples required by the semantic abstraction approach."*

⚠️ **The qualifier is load-bearing.** *"cases in which"* = a **constructed existence result** in a declared
regime (*"where an extremely high accuracy is necessary"*). It proves the separation **can** occur, not
that it **does** occur in any real driving stack. ⛔ **A proof of possibility read as a proof of necessity
is the scope error our retraction log is built around.**

### Adjudicated claims

| id | claim | evidence class | verdict | binds on us? | what would FLIP it |
|---|---|---|---|---|---|
| **M-1** | Compound/modular AI is the right architecture for scalable autonomy | `PUBLISHED-BLOG`, no ablation | ⚠️ **CONTESTED** — confirmed by two independent stacks (Waymo W-13; Dual-System VLA `2512.16760`), contradicted by a documented field-wide shift to E2E (`2603.16050`) and two commercial E2E deployments (Tesla, Wayve). **Neither side has run a controlled comparison.** | ⭐ **PARTIALLY** — it removes "pure E2E is the industry consensus" from our positioning, but does not support our specific hierarchy | A matched-params, matched-data modular-vs-E2E ablation on a shared benchmark — **which is exactly backlog row 18 (H1b)** |
| **M-2** | Semantic-abstraction training needs exponentially fewer samples than end-to-end | `PUBLISHED` (proof, existence result) | ⚠️ **SUPPORTED-AS-A-POSSIBILITY / UNSUPPORTED-AS-A-GENERAL-CLAIM** — the construction is not shown to instantiate in driving | ⛔ **NO, not as stated** — see the binding analysis below | A demonstration that the paper's construction (or an analogue) holds for a realistic driving decomposition; or an empirical sample-complexity curve favouring decomposition at matched accuracy |
| **M-3** | Modularity + redundancy + abstraction yield resilience across edge cases | `PUBLISHED-BLOG` | ⛔ **UNSUPPORTED-AS-STATED** — asserted with no ablation, no metric, and a single acknowledged trade-off (*"a careful balance between flexibility and efficiency"*) | NO | Any published edge-case ablation isolating modularity from data scale |

### ⭐⭐ Does M-2 bind on us? (step 6 — separate from whether it is true)

⛔ **NO, and getting this wrong would flatter us.**

| | Mobileye | TanitAD |
|---|---|---|
| components | **hand-specified**, semantically named, formal interfaces (RSS, REM) | **learned latent** strategic / tactical / operative levels |
| decomposition | **given by design** | **must itself be learned** |
| interfaces | engineered, formally verifiable | differentiable embeddings |

**The 2016 separation assumes the semantic decomposition is GIVEN** — its advantage comes from supervising
components directly against semantic targets. **A hierarchy whose decomposition must be discovered does not
automatically inherit that advantage.** ⇒ **M-2 raises our prior for H1b. It is not evidence for it, and it
decides no GPU-day.**

⭐ **What DOES bind:** Mobileye is a **third** independent stack whose deployed architecture carries a
modular or fast/slow seam (with Waymo W-13 and the Dual-System VLA family). *"Pure end-to-end is the
industry consensus"* is not supportable and should not appear in our positioning.

### Guidelines derived (step 7)

- ⭐⭐ **S-5 (STRATEGIC, proposed):** State in the paper that **the modular-vs-E2E question is unresolved
  and unablated on BOTH sides**, and that TanitAD's contribution is **to run the matched-params comparison
  nobody has run** (backlog row 18, H1b). **This converts the programme's biggest positioning weakness — a
  thesis hypothesis with no measured datapoint — into its differentiator.**
- ⛔ **T-3 (TACTICAL, proposed):** **Never cite `1604.06915` as support for TanitAD's hierarchy without the
  given-vs-learned decomposition caveat.** Recorded here so a future pass cannot quote it bare.

### ⚠️ Opponent strengths recorded (amendment §7.2 — a Band-D package listing none is INCOMPLETE)

1. ⭐ **Mobileye has a genuine theoretical argument, which is rare in this literature.** Waymo's doctrine
   post had zero ablations and zero theory; Mobileye has a proof, however narrow. **On evidence quality for
   an architecture claim, Mobileye is ahead of Waymo — and ahead of us.**
2. **Shipping silicon with published efficiency numbers** (EyeQ6 High, 34 TOPS INT8, >1,000 FPS on
   pixel-labeling nets). Our efficiency claim is a thesis; theirs is a part number.
3. **Formal safety interfaces (RSS) give them something we lack entirely** — a verifiable envelope
   independent of the learned components. We have a safety *case*, not a formal guarantee.
4. **Doctrinal consistency since 2016**, with a product line that reflects it. Not evidence, but not
   marketing drift either.

---

## ⛔ Open verification debts — updated 2026-09-02

| # | debt | status |
|---|---|---|
| **D-4** | UNECE GRVA **primary text** unread | ⛔ **STANDS — THIRD ROUTE FAILED (HTTP 403).** Online-learning ban now **unsupported at three probes**, still **NOT refuted** — all three probes were secondaries, and a secondary's silence is not the primary's. ⭐ **Real finding instead: the SMS spans *"post-deployment"*, so an adapting model is a SAFETY-CASE obligation, not an illegality.** ⇒ **ESCALATED TO THE PI QUEUE** — the Lab has exhausted its routes. |
| **D-5** | SparseOcc++ `2607.04732` supervision requirements | ✅ **DISCHARGED 2026-09-02 — and the PRE-REGISTERED OUTCOME CLOSES THE LINE.** It needs **dense per-voxel semantic occupancy GT** *and* **LiDAR-projected depth**. PhysicalAI-AV has neither. Backlog **P-11 retires as served-and-closed**; guideline S-3 redirects to the self-supervised / 2D-rendering occupancy family (SelfOcc, GaussianOcc, GaussTR, RenderOcc), where our NuRec + gsplat assets apply. |
| **D-6** | A2 (JEPA) and A4 (VLA) had no DEEP | ✅ **DISCHARGED 2026-09-02.** A2 = Delta-JEPA `2606.31232` (full text); A4 = driving-VLA survey `2512.16760` (full text). **Both ledgers CREATED**, fixing the FS-6 broken links. |
| **D-7** *(new)* | `1604.06915` **full text unread** — the construction behind *"cases in which"* decides how far M-2 generalises, and M-2 is the strongest theoretical claim any opponent has made against our positioning | next Band-D pass |
| **D-8** *(new)* | **B9 primary unread.** The Open-Sora *70M→10M* curation figure is `RELAYED` from a survey summary and is **barred from deciding anything**; MiniWorld `2608.01127` banked but unread | next B9 pass |
| **D-9** *(new)* | **LeWorldModel `2603.19312` and Sub-JEPA `2605.09241` are UNBANKED and unread.** ⛔ LeWM is the model whose measured action-insensitivity is this pass's headline — **we are citing a failure we have only read second-hand, through its critic.** | next A2 pass, high priority |

### ⚠️ Corrections entered this pass

- ⛔ **The "sub-50 ms driving-VLA latency requirement" is UNATTRIBUTED and is struck.** It came from a
  search-engine summary; the cited primary `2512.16760` contains **no latency number at all**.
  *(Third consecutive day on which the failing load-bearing number was one nobody had opened the primary
  for — cf. D-1's 85.1/92.1 and B2's 25 %.)*


---

# Band-D pass 2026-09-05 — WAYVE (GAIA-4). Claims V-1 … V-4

`⭐ FIRST WAYVE ADJUDICATION. The register held Waymo (W-*), NVIDIA (N-*) and Mobileye (M-*) only —`
`the architecturally closest opponent on the WORLD-MODEL axis had never been adjudicated at all.`
`Source: wayve.ai/thinking/gaia-4/, 2026-08-03, 13 named authors. PUBLISHED-BLOG.`
`Full seven-step working: Frontier Scan/Daily/2026-09-05/RESULT.md F1.`

### Document classification (step 1)

| field | value |
|---|---|
| **Document** | *"GAIA-4: Multimodal World Models Powering Closed-Loop Simulation for Safe and Scalable Autonomy"*, `wayve.ai/thinking/gaia-4/`, **2026-08-03** |
| **Evidence class** | `PUBLISHED-BLOG` — never bare `PUBLISHED` |
| **Competitive context** | Published **one month before** Wayve's London robotaxi launch on Uber (2026-09-03, `RELAYED`). No operator has yet completed VCA registration for unsupervised service. **A safety-credibility document ahead of a commercial launch.** Framing, not disqualification |
| **Companion** | GAIA-3 press page, **dated 2 December 2025**: 15 B params, *"reduced synthetic-test rejection rates fivefold"* |

### Is it an experiment? (step 2)

⭐ **One genuine controlled comparison exists** — *"Training GAIA for this task improves how faithfully it
preserves the recorded world by 2.5x"* (task-trained vs not). ⛔ **Metric unnamed, no absolute values.**
⛔ **The load-bearing doctrinal claim has NO experiment**: *"Early studies show that GAIA-3 simulated
testing closely mirrors real-world driving results"* — no study named, no metric, no coefficient. **This
is attribution-from-exposure.** The post's own fidelity question 1 — *"Do the simulated runs reproduce the
outcomes we observed in the real world?"* — **is never answered numerically** (empty **E6**, found in the
primary itself).

### Adjudicated claims

| id | claim (Wayve) | step 2: experiment? | step 3: confirming (independent) | step 4: contradicting (actively sought) | step 5: VERDICT | step 6: BINDS ON US? | what would FLIP it |
|---|---|---|---|---|---|---|---|
| **V-1** | Generative world-model simulation **measures end-to-end safety at scale** — *"transforming world modeling from scene generation into a way to measure end-to-end safety"* | ⛔ **NO.** *"Early studies"*, unnamed, unnumbered | ✅ **NAVSIM v2 navhard Stage 2 uses 3DGS counterfactual views to simulate closed-loop evaluation from logs**; a method reports **50.9 EPDMS** there. NeuroNCAP `2404.07762` is a second independent framework | ✅ **`2403.16092`** *"Are NeRFs ready for autonomous driving?"* — *"to trust the results achieved in simulation, one needs to ensure that AD systems perceive real and rendered data in the same way"*; identifies **FID/LPIPS** as real-to-sim gap indicators. ⚠️ Abstract-level read; a **class** argument (2024, NeRFs), not like-for-like | ⚠️ **CONTESTED** — the *method* is where the field's evaluation has gone; the *validity* claim is unnumbered while the independent literature says the transfer must itself be measured | ⭐ **NO, and it does not challenge our ruling.** Our binding rule bars *a planner feeding its own predictor*. GAIA-4 has a **separate** world model rendering sensors into the driver's **real** perception stack — architecturally **AlpaSim's role for us**. ⇒ GAIA-4 **satisfies** our ruling | A published measurement of the real-to-sim gap for GAIA-class generative simulators — or Wayve/DriveSafeSim publishing a correlation coefficient against road outcomes |
| **V-2** | ⛔ **THE CONCESSION.** *"every other agent in the scene keeps the exact behavior it showed in the real-world log"* · *"the evaluation stays conservative: no vehicle, pedestrian, or cyclist changes its behavior in response to the AI Driver"* | ✅ It is a **stated design constraint**, not a claim needing one | ✅ NAVSIM navhard Stage 2 is likewise non-reactive; Waymo's and the benchmark standard's evaluations share the constraint | — (no contradiction sought: it is a concession against interest) | ✅ **SUPPORTED** (they state it against their own interest) | ⭐⭐ **YES, AND IN OUR FAVOUR.** **GAIA-4's "closed loop" is EGO-ONLY against a frozen world.** So is a NuRec/gsplat replay of ours. ⇒ **We are not behind on the loop; we are behind on having BUILT it.** Backlog row 14 is our GAIA-4 | A GAIA-4 successor demonstrating reactive agents **with a measured transfer result** |
| **V-3** | *"the addition of radar measurably shifts the closed-loop trajectory, providing evidence that the generated radar carries decision-relevant information"* | ⚠️ **An intervention, yes — but the wrong one for the conclusion** | — | — (internal defect, no external search needed) | ⛔ **UNSUPPORTED-AS-STATED** — a **SENSITIVITY** result presented as a **VALIDITY** result. That the trajectory *moves* proves the input is not inert; it says nothing about whether it moves **correctly** | ⭐ **YES — as a warning about OUR OWN reporting.** Our doctrine already names this class: `W_JERK` was measured **inert**, and the ccos repair *"works mechanically and is refuted as an improvement"*. **A lever that engages is not a lever that helps** | A comparison of radar-on vs radar-off closed-loop outcomes against **logged real** outcomes, not against each other |
| **V-4** | *"Reactive agents is a capability we can switch on, not a limitation of the world model"* | ⛔ **NO** — asserted, not demonstrated | — | — | ⛔ **UNSUPPORTED-AS-STATED** (capability claimed without demonstration) | ⚠️ **Symmetrically binding.** We may not price this into their lead — **and we may not make the same move ourselves.** An unbuilt capability is not a capability | Any published GAIA-4 result **with** reactive agents enabled |

### ⭐ Guidelines derived (step 7)

- **S-6 (strategic).** *Ego-closed-loop-against-a-frozen-world is the state of the art, not a compromise.*
  Waymo, Wayve and the NAVSIM benchmark standard all evaluate this way. **TanitAD should say so and stop
  treating the absence of a reactive simulator as our gap** — it is the field's.
  ⛔ **Falsifier (V-3 rule):** a published AV evaluation with fully reactive agents shown to transfer to
  road outcomes better than a rails-based one. **None found at this probe.**
- **T-4 (tactical).** **Re-price backlog row 14 upward — it is our GAIA-4.** Every ingredient is already
  MEASURED (NuRec msgpack open, gsplat 492 FPS on Thor, the T1 harness), the target formulation is now
  published (NAVSIM navhard Stage 2), and the opponent's own version concedes the same rails constraint.
  ⛔ **Falsifier:** our 3DGS reconstructions failing the published R² ≥ 0.7 against T1.
- **T-5 (evidence hygiene).** ⛔ **No GAIA-3/GAIA-4 number may enter a TanitAD comparability table.**
  2.5× has no named metric; fivefold measures test **throughput**, not validity; **no parameter count
  exists**. Update `Benchmarks & Eval/LEADERBOARD.md:1370` — it still calls GAIA-3 *"offline"*, which
  GAIA-4 supersedes — and mark both rows **capability-only**.

### ⚠️ Opponent strengths recorded (§7.2 — a Band-D package listing none is INCOMPLETE)

1. **Radar generated by the SAME world model as the video**, not bolted on afterwards — *"rather than being
   added separately afterward"*. A genuine architectural achievement. **We have no multimodal generative
   capability at all.**
2. ⭐ **A component-level validity control we do not have**: they perturb ego pose and verify that other
   vehicles and lane markings stay stable. **That is a good control and we should copy it.**
3. **A three-level fidelity framework** (outcome / system-output / component). Even unanswered, asking the
   right three questions in public is ahead of where our own eval doctrine states them.
4. **External, government-funded validation** — DriveSafeSim with Warwick Manufacturing Group, scrutinising
   exactly the claim we mark unsupported. **They are trying to close the gap we flagged.**
5. **Shipped.** London robotaxi service on Uber, 2026-09-03. Doctrine backed by deployment.

### ⛔ Corrections entered this pass

- ⛔ **Debt D-9 is WRONG ON ITS FIRST HALF.** It reads *"LeWorldModel `2603.19312` and Sub-JEPA
  `2605.09241` are **UNBANKED** and unread."* **Both are present in `library.json`** — verified
  2026-09-05 by direct substring check **before** any web search (V-1). **The unbanked half is FALSE; the
  unread half STANDS.** ⭐ *Root-cause class: the same one this register keeps recording — a property
  asserted in the same turn as the intention, never verified. Here it cost a mis-stated debt; V-1 exists
  precisely to catch it, and did.*
- ⚠️ **`LEADERBOARD.md:1370` is superseded** — GAIA-3 is recorded as an *"offline generative world model"*;
  GAIA-4 (2026-08-03) is not offline. Category change, not just a version bump.

### ⛔ Open verification debts — updated 2026-09-05

| # | debt | status |
|---|---|---|
| **D-4** | UNECE GRVA primary text unread | ⛔ **STANDS — with the PI.** Not re-probed this pass (three routes already 403) |
| **D-7** | `1604.06915` full text unread | ⛔ **STANDS.** Not reached this pass |
| **D-8** | B9 primary unread (Open-Sora 70M→10M `RELAYED`; MiniWorld `2608.01127` banked-unread) | ⛔ **STANDS.** B9 not scanned this pass |
| **D-9** | LeWM `2603.19312` + Sub-JEPA `2605.09241` | ⚠️ **HALF-CORRECTED (see above): banked, still UNREAD.** ⛔ Still high priority — we cite LeWM's action-insensitivity **only through its critic**, and ATM `2606.09028` (read in full this pass) is a **second** paper we are now citing about LeWM without having read LeWM |
| **D-10** *(new)* | ⛔ **The navhard EPDMS SCORING BASIS is unreconciled.** Three independent 2026 tables cap navhard at **≤ 45.0**; our records carry **55.5 / 56.3**. Until resolved, **row 32's efficiency wedge is BARRED** | ⛔ **OPEN — blocks any external comparability claim** |
| **D-11** *(new)* | **GAIA-4 has no published parameter count and no simulated-vs-real correlation** (empty E6). Both are load-bearing for V-1's verdict | ⛔ **OPEN — watch DriveSafeSim/WMG outputs** |

---

# Band-D pass 2026-09-09 - NVIDIA (Alpamayo 1.5). Claims A15-1 ... A15-4

`Adjudicated under DAILY_RESEARCH_CHARTER_v2_AMENDMENT.md section 7.1. All seven steps, no exceptions.`
`Retrieval date 2026-09-09. Source: github.com/NVlabs/alpamayo1.5 model card and README.`

### Document classification (step 1)

`NVlabs/alpamayo1.5` - an **open-weights release with an accompanying model card**, published by NVIDIA
into a competitive context in which NVIDIA sells the compute its own doctrine implies. Evidence class
**`PUBLISHED-RELEASE-NOTE`**, never bare `PUBLISHED`.

**Why it is Band D and not merely Band C:** the card does not only say what shipped, it argues a
*shape* - a 10B reasoning VLA whose chain-of-causation trace is the interface between perception and
action, with a teacher-student deployment story attached. That is a claim about what is true.

**It is genuinely new to us.** Register rows N-1 to N-3 adjudicate **Alpamayo 1** (2026-08-31). N-3
touched the 1.5 card for its licence text only. **The 1.5 doctrine has never been adjudicated.**

### Is it an experiment? (step 2) - the question that does the most work

⛔ **No controlled ablation is offered for any of the four claims.** The card reports capabilities and
improvements; it does not report a matched comparison isolating the reasoning channel, and it does not
publish a reasoning-suppressed arm. **A1.5 improvements over A1 are attribution-from-release, which has
the same defect as attribution-from-exposure: several things changed at once (RL post-training,
navigation inputs, VQA head, multi-camera flexibility) and no counterfactual separates them.**

⚠️ One genuine measurement-shaped statement exists and it is a *limitation*, not a claim:
*"model accuracy may degrade with fewer cameras"* - directional, unquantified.

### Adjudicated claims

| # | claim (verbatim where quoted) | evidence class | confirming (independent) | contradicting (independent, ACTIVELY SOUGHT) | verdict | binds on us? |
|---|---|---|---|---|---|---|
| **A15-1** | RL post-training improves driving: *"Alpamayo 1.5 has undergone RL post-training, achieving improvements in reasoning quality and **reasoning-trajectory alignment**"* | `PUBLISHED-RELEASE-NOTE`, no ablation | LCDrive `2512.10226` reports *"larger improvements under interactive RL"* for reasoning-aligned tokens - an independent group finding RL helps this family | ⛔ **`2608.29583` "Drive the Thoughts: Runtime Monitoring of VLA Reasoning-Trajectory Consistency"** exists precisely because reasoning and trajectory **come apart at runtime**; a third party built a monitor for the failure NVIDIA claims to have fixed. **Alignment improved is not alignment achieved.** | ⚠️ **CONTESTED** - the direction is independently supported, the sufficiency is not | ⚠️ **PARTIALLY.** We have no language channel, so "reasoning-trajectory alignment" has no referent for us. What DOES bind is the shape: **a model whose two heads can disagree needs a runtime consistency check.** Our analogue is the planner-vs-predictor disagreement, and we have no monitor for it. |
| **A15-2** | ⭐ **THE CONCESSION.** *"Alpamayo 1 is the Teacher: a high-capability reasoning model that **runs in cloud or on-premise infrastructure**"*, generating traces for smaller consumers | `PUBLISHED-RELEASE-NOTE` | ⭐⭐ **Waymo W-14, independent of NVIDIA**: *"these Teacher models are too big to run on vehicles for real-time decision making"*; and W-3's *"teacher-student models to optimize onboard compute"* | None found. The counter-search returned no source arguing that frontier-scale reasoning models are deployable on-vehicle today. | ⭐⭐⭐ **CONFIRMS-US** | ⭐⭐ **YES, in our favour.** **Two of the three largest players now state, unprompted, that their big model does not go in the car.** Combined with the hardware line - *"minimum 24 GB VRAM for single-sample inference, up to 60 GB with CFG"* - this is a **quantified** admission: 24-60 GB against Thor's budget. Backlog P-10 (*"argue small-at-deployment as a position, not a constraint"*) gains its second independent opponent citation and its first hard number. |
| **A15-3** | Verbose natural-language chain-of-causation is the right action-facing interface (implied by the shipped architecture: *"the VLM generates chain-of-causation reasoning, then a diffusion expert produces trajectory predictions"*) | `PUBLISHED-RELEASE-NOTE`, architectural argument, no ablation | The design is widely copied, which is adoption, not evidence. **No independent controlled support found.** | ⛔⛔ **TWO independent sources, both against.** **XCoT-VLA `2608.10976`**: *"verbose natural-language Chain-of-Thought is poorly suited to real-time control because it is **open-ended, costly to decode, and difficult to optimize as an action-facing representation**"* - and reports lower longitudinal and lateral error with **2-6 executable tokens**. **LCDrive `2512.10226`**: latent CoT gives *"faster inference, improved driving quality"* versus **both** non-reasoning **and text-reasoning** baselines. | ⛔ **UNSUPPORTED-AS-STATED** - and contradicted on both cost and quality by two independent groups | ⛔ **NO - and that is the useful part.** We have no language channel to reform. **It binds as CORROBORATION of a choice we already made**: our compact latent interface is the direction two independent groups moved *toward* from the verbose end. ⭐ It also independently re-confirms **Waymo W-7** (*VLMs "too slow for real-time control"*) from a second stack. |
| **A15-4** | Scale plus reasoning is the path for AV capability (10B, 8.2B backbone + 2.3B action expert) | `PUBLISHED-RELEASE-NOTE` | None independent found this pass. | ⛔ **Our own register row N-1 already stands against it**: Alpamayo's own card showed **0.5 % open-loop trajectory gain across 3.3x parameters**, with language scaling (74.2 to 79.2) while trajectory did not. ⭐ **`2603.24581` Latent-WAM is the harder contradiction: 104M inference-time parameters, in the same year, claiming to beat the 84-89 EPDMS cluster.** A ~100x smaller model is competitive on the metric that matters to us. | ⛔ **UNSUPPORTED-AS-STATED on trajectory** (SUPPORTED on language capability, which is not our target) | ⭐⭐ **YES, in our favour** - but ⚠️ **the Latent-WAM number is under D-10's bar and may NOT be quoted until its split is stamped.** The argument stands on N-1 alone; the Latent-WAM half is held back deliberately. |

### ⭐⭐ Does it bind on us? (step 6, separate from whether it is true)

**A15-1 and A15-3 are true-or-false about a language channel TanitAD does not have.** Conflating "their
CoT interface is the wrong shape" with "our latent interface is therefore right" would be exactly the
category error the amendment warns about in the Waymo L1 case. **What transfers is the mechanism (two
heads can disagree, so measure it), not the verdict.**

**A15-2 binds hardest, and in our favour**, because it is a claim about deployment economics, which is
the one axis where our constraints and theirs are genuinely commensurable.

### ⚠️ Opponent strengths recorded (section 7.2 - a Band-D package listing none is INCOMPLETE)

1. ⭐ **They ship open weights with a runnable card.** Alpamayo is the only big-league opponent we can
   actually inspect and measure rather than only read. That is a real and unusual advantage, and it is
   why H-OPP-1 (backlog row 16) is worth more than any amount of further reading about Waymo.
2. ⭐ **Their limitations section is honest and specific**: *"not a fully fledged driving stack...
   lacks access to critical real-world sensor inputs, does not incorporate required diverse and
   redundant safety mechanisms"*, plus the camera-count degradation note. **We should copy this
   practice**; several of our own artifacts assert capability without an equivalent boundary statement.
3. ⭐ **A published, reproducible RL post-training recipe** for a driving VLA. Whatever the alignment
   claim is worth, the *recipe* being public is an asset the field did not have, and it is more than we
   publish.
4. ⚠️ **Ecosystem lock-in as a genuine moat**: teacher in the cloud, student on DRIVE Thor, tooling and
   corpus (`PhysicalAI-AV`) all from one vendor - the same corpus we train on. **Our parity corpus is
   their product**, which is a strategic dependency worth naming even though it is not a technical
   weakness of ours.

### Guidelines derived (step 7)

| id | kind | guideline | the measurement that would OVERTURN it (rule V-3) |
|---|---|---|---|
| **S-6** | strategic | **Argue the deployment-economics case with opponent numbers, not ours.** Two independent opponents now concede the teacher does not ride in the car, and Alpamayo attaches **24-60 GB VRAM** to its inference. State small-at-deployment as the industry's own position. | Any opponent shipping a frontier-scale reasoning model running on-vehicle inside an automotive power and latency budget. |
| **S-7** | strategic | **Claim the compact-latent interface as convergent, not contrarian.** Two independent groups moved from verbose text CoT toward compact executable or latent reasoning **and reported both faster inference and better driving**. | A controlled ablation showing verbose natural-language CoT beating a compact latent interface at matched latency on a driving metric. |
| **T-4** | tactical | ⛔ **Never quote the Latent-WAM 89.3, the DrivoR 56.3, or the DriveFuture 55.5 until each carries a SPLIT STAMP.** D-10 is resolved in mechanism, not in provenance. | A leaderboard submission, or an author statement naming the split and NAVSIM version for each. |
| **T-5** | tactical | **Every parameter count in a comparability table carries INFERENCE-TIME or TRAINING-TIME**, the way every metric carries its eval tier. Latent-WAM is 104M at inference and 191M in training; DrivoR's ~40M is unverified (backlog L-15). | Not falsifiable - a notation rule, not a claim. Adopted for hygiene, and marked as such. |
| **T-6** | tactical | **Build the disagreement monitor we lack.** A15-1's contradiction is that a third party had to monitor reasoning-trajectory consistency at runtime. Our analogue: **log planner-selected versus predictor-implied trajectory divergence per window** on existing arms. 0 GPU on banked dumps. | If planner and predictor never measurably disagree on banked windows, the monitor has no signal and the line closes. |

### Corrections entered this pass

⚠️ **Backlog row 4 / MM-E1's premise is half wrong.** It states *"our fixed tau=0.996 has no published
operating point - every EMA-teacher line ramps tau."* **`2606.28383` uses a FIXED alpha = 0.996, does
not ramp, and measures a 44-fold collapse in discriminative power without EMA.** The sentence *"no
published operating point exists"* is **retracted**. ⛔ **The design question is NOT settled** - that
result is a 1.29M-parameter model on structured agent-state vectors, not pixels, at 1,322 scenarios.
See `Frontier Scan/Daily/2026-09-09/RESULT.md` F4.

### Open verification debts - updated 2026-09-09

| # | debt | state |
|---|---|---|
| **D-4** | UNECE GRVA primary text | ⛔ **STANDS - FIVE ROUTES NOW.** Route 4 (direct unece.org PDF) **HTTP 403**; route 5 (jasic.org) returned a font stream with no text layer. ⭐ **Character changed: this is a RETRIEVAL-CHANNEL block on a document we have located, not a discovery failure.** With the PI; a human browser settles it in minutes. |
| **D-7** | `1604.06915` full text unread | ⛔ **STANDS.** Not reached this pass. |
| **D-8** | B9 primary unread | ⛔ **STANDS and is now WORSE.** Kairos `2606.16533`'s curation section failed at **three routes** today (PDF too large, landing page only, HTML truncated mid-section 4.2). Its two-level-curation claim is `RELAYED` and decides nothing. |
| **D-9** | LeWM `2603.19312` + Sub-JEPA `2605.09241` banked, unread | ⛔ **STANDS and HARDENS. `2608.12939` is the THIRD independent critic of LeWM we have now read without reading LeWM itself.** This is the register's own named failure class, three deep. |
| **D-10** | navhard EPDMS scoring basis | ⭐⭐ **MECHANISM RESOLVED 2026-09-09** - two disjoint populations (navhard 23.1-45.0; "NAVSIM v2" 12k-scenario 84.8-89.3), and the maintainers discourage self-reported splits. ⛔ **PROVENANCE STILL OPEN** for our own 55.5/56.3, which match neither cluster. Row 32 stays BARRED; guideline T-4. |
| **D-11** | GAIA-4 parameter count / sim-real correlation | ⛔ **OPEN.** Not re-probed this pass. |
| **D-12** *(opened and CLOSED the same pass)* | Today's primaries cited but not banked during the mount outage | ✅ **DISCHARGED 2026-09-09.** The mount recovered late in the pass; **8 primaries banked or citation-updated, all rc=0** (`2608.12939`, `2606.28383`, `2603.24581`, `2509.23958`, `2608.10976`, `2512.10226`, `2505.24808`, `2608.29583`). Library **487 -> 491 entries, 3,198.3 MB**. ⛔ **V-1 measured at 4 of 8 already banked (50 %) - and `2603.24581`, which carried this pass's biggest finding, was one of them, unread.** |
