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

---

# 2026-09-10 · BAND-D ADJUDICATION — NVIDIA, closed-loop post-training doctrine

`Appended by LAB-RUN-011. Amendment §7.1, all seven steps. Append-only.`

**Document.** *How to Post-Train Autonomous Vehicle Models in Closed-Loop with NVIDIA Alpamayo* — Boris Ivanovic & Marco Pavone, NVIDIA Autonomous Vehicle Research Group, **2026-05-31**, NVIDIA Technical Blog.

### Step 1 — CLASSIFY

Evidence class **`PUBLISHED-BLOG`**, never bare `PUBLISHED`. Competitive context: NVIDIA is positioning Alpamayo as the open model stack for L4 robotaxis (Alpamayo 2 Super released for commercial use in the same window), and this post is developer-adoption material for **AlpaSim**, its simulator. It is a **how-to**, and its doctrine is carried in the framing rather than in results.

### Step 2 — ⛔ IS IT AN EXPERIMENT?

**NO. Emphatically not, and this is the finding.** The post reports **zero** performance metrics — no accuracy, no safety rate, no planning score, no before/after. Every number in it is an operational specification: 10 B parameters, ≥40 GB VRAM, ~100–150 GB disk, ~21 GB weights, ~1.5 GB/scene, ~1.5 TB for the `public_2601` suite. **There is no ablation, no A/B, and no matched comparison anywhere in the document.** The central claim — that closed-loop post-training improves the policy — is asserted, not measured.

### Step 3 — CONFIRMING EVIDENCE (≥1 independent)

The **diagnosis** is well confirmed independently. `2607.08072` (*Post-Training in End-to-End Autonomous Driving*): *"Closed-loop evaluation is most aligned with post-training because the policy's actions affect future states."* AD-R1 (CVPR 2026) and RAD-2 `2604.15308` both build closed-loop post-training pipelines on the same premise. ⭐ Our own record agrees three ways (action echo 97.9 % vs 0.0 % hold-action; Alpamayo's open-loop metric flat across 3.3× params; Waymo L5).

### Step 4 — ⭐ CONTRADICTING EVIDENCE (actively sought — query Q10)

The **prescription** is contested at ≥2 independent sources:

- **Sim2Real-AD `2604.03497`**: *"Most simulator-trained policies are tightly coupled to their training environment: their observations rely on simulator-native representations unavailable on real vehicles, and their action semantics are calibrated to simulator dynamics. Without mechanisms to bridge both couplings, **direct deployment fails even when the policy performs well in simulation**."*
- **AD-R1 (CVPR 2026)**: *"Conventional RL Post-training relies on external simulators, suffering from a sim-to-real gap and heuristic rewards"* — and it replaces the external simulator with an internal world model **for that reason**. An independent group arguing against the exact mechanism this post recommends.
- **`2607.08072`** (already in our B5 ledger): a scalar rollout reward **hides cross-dimension degradation** — so closed-loop post-training can improve an aggregate while regressing a family, which is what we measured on refcv3's RL stage.

⛔ **The post concedes no fidelity limitation of AlpaSim at all** — no sim-to-real discussion, no domain-randomisation need, no transferability constraint. For a document whose entire proposal is "train on simulator rollouts", that silence is the load-bearing omission.

### Step 5 — VERDICT

| claim | verdict | reason (one sentence) |
|---|---|---|
| **A16-1** — *open-loop training/eval is insufficient because errors compound when actions affect the environment* | ✅ **CONFIRMS-US** | independently confirmed at ≥3 sources and by our own three measurements; it is our binding open-vs-closed-loop ruling stated by an opponent |
| **A16-2** — *closed-loop post-training in AlpaSim improves the policy* | ⚠️ **CONTESTED** | asserted with **no metric and no ablation** in the post, while ≥2 independent sources report that simulator-trained policies fail to transfer without explicit bridging |
| **A16-3** — *simulation should be training experience, not only a final evaluation stage* | ⚠️ **CONTESTED, and narrower than stated** | the direction is sound, but with **no fidelity limits disclosed** and `2607.08072`'s scalar-reward masking, "turn rollouts into experience" is under-specified as an engineering instruction |

### Step 6 — ⭐⭐ DOES IT BIND ON US? (separate from whether it is true)

**A16-1 BINDS, and favourably.** It is the same statement as our own ruling. ⭐ **And its recommended mechanism is one we already possess**: AlpaSim runs bare on an A40 in our fleet, and closed-loop videos already exist. The opponent's prescribed closed-loop path is *open to us today*, which is unusual and should be said out loud.

**A16-2 and A16-3 DO NOT BIND as prescriptions.** Our constraint is different in kind: we are gated on a **fixed parity corpus** and a tiny fleet, not on simulator access, and our binding ruling already requires AlpaSim or a real vehicle for any closed-loop claim. Adopting NVIDIA's *workflow* on the strength of a blog with no numbers would import doctrine and call it evidence — the `INHERITED` failure the operating standard bans (rule V-3).

⚠️ **Also non-binding on scale:** their 10 B model *"does not fit on one GPU"* and a distillation script to a single-GPU checkpoint is *"planned"*. Our sub-300 M positioning is untouched by this document.

### Step 7 — GUIDELINES

- **S-6 (strategic).** *Cite A16-1 as an opponent-side confirmation of the open-vs-closed-loop ruling, and cite it as diagnosis only.* NVIDIA's own doctrine post supports **why** open-loop is insufficient and supplies **no evidence** that their fix works. **Falsifier: any NVIDIA publication reporting a controlled before/after of AlpaSim closed-loop post-training on a fixed eval; that would move A16-2 to SUPPORTED.**
- **T-7 (tactical).** *Any TanitAD closed-loop post-training arm reports per-family reward decomposition from the first run, not after a regression appears* (FS9-5's mechanism, now with an opponent-side instance of the omission). **Falsifier: a run where the aggregate and every family move together, at which point the granularity requirement is over-engineering for our setting.**

### ⚠️ OPPONENT STRENGTHS, recorded (a package listing none is INCOMPLETE)

1. **They have the simulator and we have it too — but they built it.** AlpaSim plus the NuRec scene corpus is a working reconstruction-based closed-loop rig, and the post makes it reproducible by a third party in an afternoon.
2. **Open weights and a documented post-training path.** Alpamayo 1.5 at 10 B with published VRAM and disk requirements is a genuinely inspectable opponent — still the only one at this scale.
3. **They named the right problem before we finished measuring it.** Error compounding under closed loop is the correct diagnosis, and they published it as developer guidance rather than as a paper claim.

### New register debts

| id | debt | when |
|---|---|---|
| **D-12** *(new)* | **No AlpaSim fidelity characterisation exists in any NVIDIA source we have read.** A16-2 cannot move off `CONTESTED` without one, and we run AlpaSim ourselves — so this is answerable by measurement, not only by reading. | next Deploy/Band-D pass |

### Standing debt status this pass

| id | status |
|---|---|
| **D-4** (UNECE GRVA) | ⛔ **STANDS — not re-probed.** Five routes already failed; with the PI. Declared, not concealed |
| **D-7** (`1604.06915` full text) | ⛔ **STANDS.** Not reached |
| **D-8** (Kairos B9 curation) | ✅ **DISCHARGED 2026-09-10 at the FIFTH route** → `Data Engineering/Research/2026-09-10-kairos-curation-fourth-route/RESULT.md`. ⛔ **And the claim is `UNSUPPORTED-AS-STATED` as evidence: the paper says the pipeline *"does not yet compute CID directly"*.** ⚠️ The claim exists only in **v3**; v1 does not contain it |
| **D-9** (LeWM + Sub-JEPA unread) | ✅ **HALF-DISCHARGED — LeWM `2603.19312` read in full.** ⭐ It conditions by **AdaLN at each layer** (our FiLM family) and **reports no action-sensitivity control at all** ⇒ the insensitivity we cite three times is a third-party measurement, never a self-report. **Sub-JEPA `2605.09241` STANDS unread** |
| **D-10** (navhard scoring basis) | ⭐⭐ **COUNTS RESOLVED 2026-09-10** — navhard **450 S1 / 5,462 S2** from the maintainers' primary; navtest ≈12,000. The 84.8–89.3 cluster is **navtest**. ⛔ **BASIS STILL OPEN: PDM-Closed reads 51.3 AND 56.6 on "navhard".** Row 32 stays BARRED |
| **D-11** (GAIA-4 params / sim-real correlation) | ⛔ **OPEN.** Not re-probed this pass |


---

# 2026-09-13 · BAND-D ADJUDICATION — Waymo compute doctrine (+ Tesla Cybercab, RELAYED)

`Appended by LAB-RUN-012. Amendment §7.1, all seven steps. Append-only. Full reasoning: Frontier Scan/Daily/2026-09-13/RESULT.md F1 / F1b.`

**Documents.** (a) *A look under our trunk: what's in our compute* — Satish Jeyachandran (VP Engineering) & Daniel Rosenband, Waymo blog, **2026-08-20**, `PUBLISHED-BLOG` (read via fetch summariser; short quotes only). (b) Tesla Cybercab launch, Austin, **2026-09-03** — **no Tesla primary read**; trade press only ⇒ `RELAYED`.
**Context.** (a) was published two weeks before (b) and paired in the press with Waymo's *"cameras … aren't enough"* argument (TechCrunch 2026-09-01). ⚠️ Waymo's 10-lessons post is **already adjudicated** (W-1…W-13); not re-opened.

| id | claim | opponent | date | experiment? | confirming (independent) | contradicting (sought) | verdict | binds on us? | evidence that would FLIP it |
|---|---|---|---|---|---|---|---|---|---|
| **A17-1** | custom 5 nm ASIC, >1,000 TOPS dedicated to front-end sensor processing, is what L4 compute needs | Waymo | 2026-08-20 | ⛔ no — a specification | none independent (press relays of the post) | TOPS is a synthetic-workload proxy — an accelerator with 7.5× a GPU's TOPS delivered ~4× network throughput (RELAYED, aiMotive / Hailo engineering blogs) | **UNSUPPORTED-AS-STATED** | no (hardware); **yes as a rule: never quote TOPS** | a published Waymo latency/accuracy comparison of the ASIC stack against a commodity stack on the same models |
| **A17-2** | pixels-to-actuation latency is optimised across **every percentile** | Waymo | 2026-08-20 | ⛔ no numbers | COLA `2305.07147` (tail latency is safety-critical in a real L4 stack; lib, abstract-only); `2209.05487` (99th-percentile outliers, scan) | none found arguing mean latency suffices (EMPTY, D12) | **CONFIRMS-US** | ✅ yes — identical to D10-2 (`Tpeak` gate) | a study showing planning safety insensitive to tail latency at fixed mean |
| **A17-3** | L4 compute must be two independent engines sharing load with seamless failover | Waymo | 2026-08-20 | ⛔ no | fail-operational requirement for L4 in the ISO 26262 literature (`2011.00892`, scan; PatSnap blog RELAYED) | asymmetric failover (primary + weaker safety-only unit) is a published alternative (patent literature as summarised by search — RELAYED) | **SUPPORTED** as practice, not as necessity | ⚠️ not now (research reference on one Thor); a P6 requirement the day deployability is claimed | a type-approved L4 system shipping without compute redundancy |
| **A17-4** | camera-based end-to-end networks are the core of L4; 380 k unsupervised miles *"no notable incidents"* | Tesla | 2026-09-03 | ⛔ attribution-from-exposure | NHTSA SGO mid-June→mid-July 2026: 2 incidents, robotaxi stationary, other party at fault (RELAYED) | 17 Austin incidents Jul 2025→Mar 2026 at 5.9 mph average; remote-operator crash Houston (Electrek 2026-07-20, RELAYED) | **CONTESTED** (weakest-sourced row in the register) | ⛔ no — our vision-only rule is admissibility, not sufficiency | NHTSA SGO primaries with exposure-normalised rates vs a matched human baseline |

### ⭐ Concessions

* **Waymo:** *"…especially in the low-batch regimes we often operate"* — the real operating point is batch ≈ 1, where TOPS says least. Matches our Thor measurement (throughput flat across a 6× batch range, saturation at batch 8).
* **Tesla (RELAYED, needs a primary):** press describes Cybercab's FSD 14.3.3 as a **"vision-plus-radar stack"**.

### Guidelines

* **S-6b (strategic)** — *our latency claims are worst-case claims.* Falsifier: a TanitAD deployment table quoting a mean without a percentile.
* **S-7 (strategic)** — *"small at deployment" gains a redundancy argument: a sub-300 M model can run on the degraded half of a fail-over pair* — only after measuring on half a Thor. Falsifier: the model does not meet its budget at half compute.
* **T-8 (tactical)** — *never quote TOPS; measured ms at a named percentile on the target only.*
* **T-9 (tactical)** — *read NHTSA SGO primaries before A17-4 is quoted anywhere.*
⚠️ **Naming collision noted:** LAB-RUN-011 already used **S-6** (cite A16-1 as diagnosis only); today's strategic latency guideline is therefore **S-6b**. The Master Mind should renumber on the next register review.

### ⚠️ OPPONENT STRENGTHS, recorded

1. **Waymo ships a custom ASIC, redundant compute and vehicle-integrated liquid cooling** — production engineering we do not have and do not claim.
2. **200 M fully autonomous miles; riders in 14 cities** (2026-09-01) — the exposure record is real even where it is not an experiment.
3. **Tesla put a no-wheel, no-pedal L4 vehicle into public launch** on a camera-centric stack — whatever its evidence, it is a deployment fact.

### Standing debt status this pass

| id | status |
|---|---|
| **D-4** (UNECE GRVA) | ⛔ **STANDS — three more routes failed (6/7/8, all HTTP 403). Eight total.** The UN ADS Regulation was adopted 2026-06-24 (RELAYED); with the PI |
| **D-7** (`1604.06915`) | ⛔ **STANDS.** Not reached |
| **D-9** (Sub-JEPA `2605.09241`) | ⛔ **STANDS (half).** Not reached |
| **D-10** (navhard scoring basis) | ✅ **CLOSED 2026-09-13** → `Benchmarks & Evals/Research/2026-09-13-pdmc-stage2-provenance/RESULT.md`: 51.3 = `2506.04218` v2 (08/2025 snapshot), 56.6 = **v3** (03/2026 snapshot); Stage 1 identical 9/9, Stage 2 re-scored 7/9. ⛔ Our 09-10 reading of v2 while holding v3 caused the contradiction (LI10-1 class) |
| **D-11** (GAIA-4) | ⛔ **OPEN.** Not re-probed |
| **D-12** (AlpaSim fidelity) | ⛔ **OPEN** — and sharpened: NuRec's `map.xodr` speed field is unit-mislabelled and value-inconsistent on the one scene we hold (DE package F2/F3) — one measurable fidelity defect already on file |


---

## Pass 2026-09-17 (LAB-RUN-014) — ⭐⭐ debt **D-1 DISCHARGED**: "The Waymo World Model" adjudicated after 17 days open

`Seven-step adjudication per DAILY_RESEARCH_CHARTER_v2_AMENDMENT.md §7.1. Full working: Frontier Scan/Daily/2026-09-17/RESULT.md §1.`

| id | opponent · date | claim | experiment? | verdict | binds on us? | what would FLIP it |
|---|---|---|---|---|---|---|
| **W-17-1** | Waymo · 2026-02 · PUBLISHED-BLOG | The world model's role is **high-fidelity multi-sensor SIMULATION** (camera + lidar, 4D point clouds, Genie 3) for preparing the Driver — presented with no onboard-planning role | ⛔ **NO — demonstration videos only.** No ablation, no comparison, no metric. The only quantities are exposure counts (200 M autonomous miles, "billions" of virtual miles) | **UNSUPPORTED-AS-STATED** for the reading *"world models belong in simulation rather than onboard"* — ⚠️ which the post **never asserts**; it is the takeaway its framing invites | ⛔ **NO.** Different artifact: a **generative multi-sensor simulator** producing training data vs our **latent predictive planner** producing a plan. ⚠️ It **does** bind on backlog **row 14**, which is the same artifact class | Waymo publishing an onboard-vs-simulation ablation, or a controlled result that a generative simulator outperforms latent planning at matched compute |
| **W-17-2** | Waymo · 2026-02 | *"By simulating the 'impossible', we proactively prepare the Waymo Driver for some of the most rare and complex scenarios"* (tornado, elephant) | ⛔ **NO** — no downstream driving metric is attached to the synthetic scenarios | **UNSUPPORTED-AS-STATED** — the inference from *"we can generate it"* to *"the Driver is prepared"* is the missing step | ⚠️ **Partially.** Our long-tail augmentation ambitions rest on the same unproven inference; we should not repeat it | any published measurement that training on generated rare events improves real long-tail performance |
| **W-17-3** ⭐ | Waymo · 2026-02 · **CONCESSION** | *"the longer the simulation, the tougher it is to compute and maintain stable quality"* | it is an admission, not a claim | **SUPPORTED** (and independently by our own k=60 BPTT divergence and drift) | ⭐⭐ **YES, favourably.** Long-horizon rollout stability is **conceded unsolved by the best-resourced opponent** ⇒ our measurement of it is an asset, not an embarrassment | a Waymo (or other) result demonstrating stable long-horizon generative rollout with a metric |
| **W-17-4** ⭐ | Waymo · 2026-02 · **CONCESSION** | *"purely reconstructive simulation methods … suffer from visual breakdowns due to missing observations"* | an admission | **SUPPORTED** | ⛔⭐ **YES, and it is a direct hazard to backlog row 14** — our NuRec/gsplat pilot IS a reconstructive method, and a perturbation grid is exactly what creates missing observations | a reconstructive pipeline shown to hold quality under counterfactual viewpoints |
| **N-17-1** | NVIDIA · 2026-08-04 · PUBLISHED-BLOG | *"AlpaGym exposes compounding errors and edge-case failures that static datasets miss"*; closed-loop training *"exposes AV models to the consequences of their actions at training time"* | ⚠️ **NO ablation offered** — an argument, not a controlled comparison of closed-loop-trained vs static-trained | **SUPPORTED** — classical covariate-shift result, and independently confirmed by **our own** 97.9 % open-loop vs 0.0 % hold-action vs ~5 % closed-loop | ✅ **YES, and we already comply** — it is our own open/closed-loop ruling, reached independently | an ablation showing closed-loop training gives no advantage at matched data |
| **N-17-2** | NVIDIA · 2026-08-04 | *"State-of-the-art performance in multiple aspects including reasoning quality, trajectory accuracy, alignment"* | ⛔ **NO baseline table in the blog** | **UNSUPPORTED-AS-STATED** in the blog. ⚠️ The **model card** does publish three numbers (Lingo-Judge 79.2 · AlpaSim 1.50 ± 0.13 · minADE_6 @6.4s 0.911 m) — cite the card, never the blog | ⛔ scale does not bind (34 B vs our sub-300 M ceiling) | a comparison table with named baselines on a stamped benchmark |
| **N-17-3** ⭐⭐ | NVIDIA · 2026-08-04 · PUBLISHED-CARD | **OpenMDW-1.1 across the ENTIRE Alpamayo lineup**: fine-tuning, derivatives, **commercial redistribution**, no field-of-use restriction; code Apache-2.0. Model = **34 B = 32 B Cosmos 3 Super Reasoner + 2.3 B action expert** | a licence fact, not a claim | **CONFIRMED** at the card plus three secondaries | ⭐⭐ **YES — strategically.** ⛔ **Register row N-3 (the Alpamayo licence split) may be SUPERSEDED and must be re-checked.** Our differentiation can no longer be *access*; it must be **efficiency and hierarchy, measured** | NVIDIA restricting the licence, or the card's terms differing from the blog's description |

### Guidelines derived this pass

* **S-5 (strategic)** — ⭐⭐ *"world model" names TWO different artifacts — a **generative simulator** (Waymo/Genie 3, AlpaGym) and a **latent predictive planner** (ours, JEPA-family) — and evidence about one is not evidence about the other.* State it in the paper. **Falsifier:** a controlled result showing a generative simulator outperforms latent planning at matched compute on the same task.
* **S-6c (strategic)** — *our differentiation is efficiency and hierarchy, not access to a frontier driving model* (N-17-3 removed the access argument). **Falsifier:** an open 34 B baseline matching our efficiency claims at our parameter budget. ⚠️ **Numbering:** S-6 and S-6b are both taken (LAB-RUN-011, -012); this is **S-6c**. The Master Mind should renumber the S-6 family in one pass.
* **T-6 (tactical)** — *before backlog row 14 is designed, state which rung of the `2607.07196` ladder it targets.* Its current R² ≥ 0.7 goal is **L4** while it has no **L1** check. **Falsifier:** row 14 clearing L1-L3 and still failing L4.
* **T-7 (tactical)** — *re-check register row **N-3** against OpenMDW-1.1 this week.* A licence restriction we are carrying may no longer exist.

### ⚠️ OPPONENT STRENGTHS, recorded (a Band-D package listing none is INCOMPLETE)

4. **Waymo generates camera AND lidar together** in one simulator — a multi-sensor generative capability we do not have and have no path to.
5. **Waymo has Genie 3** through DeepMind: a frontier generative backbone no independent programme can match.
6. **NVIDIA publishes a closed-loop score WITH an interval** (AlpaSim 1.50 ± 0.13) — more than most of the field offers, and more than several of our own rows.
7. **NVIDIA ships AlpaGym as closed-loop TRAINING infrastructure**, not merely evaluation.
8. ⭐ **NVIDIA conceded its moat deliberately** (OpenMDW-1.1 across the lineup) in exchange for ecosystem — a strategic choice, and a confident one.

### Standing debt status this pass

| id | status |
|---|---|
| **D-1** (The Waymo World Model, 2026-02) | ✅ **DISCHARGED 2026-09-17** — adjudicated as W-17-1…W-17-4 above. Open since 2026-08-31 |
| **D-2** (*Demonstrably Safe AI*, 2025-12) | ⛔ **STANDS.** Not reached |
| **D-3** (Waymo Foundation Model post, 2025-12) | ⛔ **STANDS.** Not reached |
| **D-4** (UNECE GRVA primary) | ⛔ **STANDS** — not re-probed today; eight routes failed previously; with the PI |
| **D-7** (`1604.06915`) | ⛔ **STANDS.** Not reached |
| **D-9** (Sub-JEPA) | ✅ **CLOSED 2026-09-15** → `Architecture & Inference/Research/2026-09-15-subjepa-rank-direction/RESULT.md` |
| **D-11** (GAIA-4) | ⛔ **OPEN.** Not re-probed |
| **D-12** (AlpaSim fidelity) | ⛔ **OPEN** — ⭐ and now equipped with a published ladder (`2607.07196`) to state fidelity claims against, rather than arguing them informally |

## Pass 2026-09-18 (LAB-RUN-015) — TESLA doctrine adjudicated (T-18-1 … T-18-3) + a bookkeeping correction

`Seven-step adjudication per DAILY_RESEARCH_CHARTER_v2_AMENDMENT.md §7.1. Full working: Frontier Scan/Daily/2026-09-18/RESULT.md §1.`
`Documents: Tesla AI FSD v14 Lite release notes 2026.20.5.1 (2026-06-29, PUBLISHED-RELEASE-NOTE via verbatim reproduction); Elluswamy ICCV-25 / ScaledML-26 talks (RELAYED — no transcript primary).`

| id | opponent · date | claim | experiment? | verdict | binds on us? | what would FLIP it |
|---|---|---|---|---|---|---|
| **T-18-1** | Tesla · 2026-06-29 · PUBLISHED-RELEASE-NOTE | *"Distilled the intelligence from HW4 V14 into HW3 … unlocks the improvements … including Reinforcement Learning (RL)"* | ⛔ NO — no metric in the notes | **SUPPORTED** as a practice (Waymo teacher-student; DriveZero `2609.06055` measured) · ⚠️ **UNSUPPORTED-AS-STATED** that RL gains transfer · counter: **TAKD `1902.03393`** (large capacity gap degrades distillation) | ⭐ **YES, favourably** — train-large/deploy-small, now Waymo **and** Tesla doctrine | a published HW3 v14-Lite vs v13 comparison showing no gain |
| **T-18-1c** ⭐ | Tesla · 2026-06-29 · **CONCESSION** | *"This feature does not make your vehicle autonomous"* — the distilled student ships **supervised** | admission | **SUPPORTED** | ⭐ YES — distillation is shown to reach *supervised* quality only | an unsupervised deployment on a distilled student |
| **T-18-2** | Tesla · ICCV-25 · RELAYED | *"loss on open-loop predictions might not correlate to great performance in the real-world"* ⇒ closed-loop neural world simulator | ⛔ NO | **SUPPORTED** for open-loop *loss* (confirm: `2605.00066`, `2605.31041` +7.1 % vs −14.6 %, our 97.9 %/0.0 %); counter: pseudo-sim R² 0.8 `2506.04218` — some offline metrics do correlate | ⭐⭐ **YES — CONFIRMS-US** (`EVAL_DOCTRINE.md`); also a concession about Tesla's own imitation objective | an open-loop loss ranking closed-loop outcomes at ρ ≥ 0.8 across architectures |
| **T-18-3** | Tesla · ICCV-25 · RELAYED | end-to-end is *"the only scalable solution"* | ⛔ NO — anecdotes | **CONTESTED** (counter: Waymo W-13 hybrid, Mobileye M-1, DriveZero decomposes perception/action) | ⚠️ PARTIALLY — our hierarchy is neither pole; row 18 (H1b) is the test | matched-params E2E vs hybrid ablation on a shared benchmark |

### Guidelines derived this pass
* **S-7 (strategic)** — train-large/deploy-small is Waymo + Tesla doctrine; **our** claim is that a sub-300 M model trained as a world model needs no giant teacher. **Falsifier:** our distilled student beating our directly-trained model at matched deploy params.
* **T-8 (tactical)** — budget a teacher-assistant step when teacher/student params exceed ~10× (TAKD). **Falsifier:** direct large→small matching the TA route at our scale.
* **T-9 (tactical)** — quote T-18-2 in the paper as opponent corroboration of the tier doctrine, labelled **RELAYED** until a transcript primary is banked.

### ⚠️ OPPONENT STRENGTHS, recorded
9. Tesla ships a distilled student onto hardware with ~15 % of the teacher platform's memory bandwidth (RELAYED figure), at fleet scale.
10. Tesla runs a closed-loop neural world simulator for evaluation **and** RL at fleet-data scale.
11. (academic, context) DriveZero `2609.06055` reaches **57.1 EPDMS navhard two-stage** from frozen VFMs + a 5.70 M privileged PPO teacher — the bar G3 must now clear.

### ⛔ CORRECTION (append-only; nothing above is rewritten)
The 2026-09-17 debt table lists **D-2** and **D-3** as *"STANDS. Not reached"*. This register's own lines under *"Waymo — The Waymo World Model + Demonstrably Safe AI"* already record both as **DISCHARGED** (W-11…W-15). Today's primary fetch of *Demonstrably Safe AI* (2025-12-09) reproduces W-13's quotes exactly. ⇒ **D-2 and D-3 are CLOSED**; the 09-17 listing re-opened already-discharged ids. D-3's residual (a dedicated Foundation-Model read is owed only if a claim depends on it) stands.

### Standing debt status this pass
| id | status |
|---|---|
| D-2, D-3 | ✅ **CLOSED** (see correction) |
| D-4 (UNECE GRVA primary) | ⛔ STANDS — not re-probed |
| D-7 (`1604.06915`) | ⛔ STANDS — not reached |
| D-11 (GAIA-4 params / sim-real correlation) | ⛔ OPEN — **re-probed today, still EMPTY** (2nd probe) |
| D-12 (AlpaSim fidelity) | ⛔ OPEN — now has the ladder's L2 envelope as its instrument (see `Benchmarks & Evals/Research/2026-09-18-ladder-l1-row14/RESULT.md`) |
| **D-13** *(new)* | Elluswamy talk **transcript primary** — T-18-2/-3 are RELAYED until it is banked |

## Pass 2026-09-19 (LAB-RUN-016) — NVIDIA (Jim Fan WAM thesis), WAABI (simulator realism), MOBILEYE (bias-variance): seven claims, all new

`Seven-step adjudication per DAILY_RESEARCH_CHARTER_v2_AMENDMENT.md §7.1. Full working: Frontier Scan/Daily/2026-09-19/RESULT.md §1.`
`Documents: Jim Fan, Sequoia AI Ascent 2026 talk (RELAYED via verbatim-quoting summary, eventual.ai 2026-05-08; video primary linked, not transcribed) + NVIDIA Technical Blog "Pretrained to Imagine, Fine-Tuned to Act" (2026-06-15, PUBLISHED-BLOG, read in full); Waabi "Simulator realism: the new safety standard" (Urtasun, 2025-03-11, PUBLISHED-BLOG); Mobileye "Autonomous decisions: the bias-variance tradeoff" (Shashua & Shalev-Shwartz, 2024-05-15, PUBLISHED-BLOG).`

| id | opponent · date | claim | experiment? | verdict | binds on us? | what would FLIP it |
|---|---|---|---|---|---|---|
| **N-19-1** | NVIDIA · AI Ascent 2026 · RELAYED | *"Rest in peace [VLAs]. Long live world action models."* | ⛔ NO — the NVIDIA blog concedes *"This is not proof that WAMs are the better default"*; DreamZero 1750 vs π0.5 1622 Elo is not matched | **CONTESTED**: *WM objectives help action* SUPPORTED (World Tokens `2608.09730`, matched baseline 59.4 → 76.0 %); *"VLAs are dead"* UNSUPPORTED-AS-STATED. Counter: NVIDIA's **own** AV product Alpamayo 2 Super is a 34 B VLA | ⭐⭐ YES favourably (we are a WM); the **pixel-video-backbone** half does not bind | matched-data/params real-world VLA vs WAM with the VLA winning |
| **N-19-2** | NVIDIA · AI Ascent 2026 · RELAYED | *"If the video prediction works, the action works. If the video hallucinates, the action fails."* / physics emerges from next-pixel prediction at scale | ⛔ NO | **UNSUPPORTED-AS-STATED** for pixels: DeepSight T3 (VAE target 27.75 vs DINOv3 74.79 DS) and World Tokens T2 (RGB anchor 91.5 < no-WM 92.8), both independent and controlled; Physics-IQ `2501.09038` (understanding unrelated to realism) | ⭐⭐ YES — **CONFIRMS-US** (semantic latent targets) | a matched ablation where a pixel-reconstruction target beats a semantic-feature target in closed loop |
| **N-19-3** ⭐ | NVIDIA blog · 2026-06-15 · **CONCESSION** | *"3–4x slowdown at inference time, which matters a lot for real-time control"*; Fast-WAM 590–800 ms vs π0.5 ~190 ms | admission | **SUPPORTED** | ⭐⭐ YES — train-time WM, sub-300 M deploy (S-8) | a pixel WAM inside 100 ms on an edge SoC at matched quality |
| **WB-19-1** | Waabi · 2025-03-11 · PUBLISHED-BLOG | Waabi World **99.7 % realism** = `(1 − avg relative trajectory distance) × 100`, full stack sim vs real ⇒ sim can replace much real-world testing | ⭐ PARTLY — a **paired** sim/real measurement (a real counterfactual), but no n, no module breakdown, no CI, no third party | **UNSUPPORTED-AS-STATED** as a safety standard: an average distance is dominated by nominal driving and hides the rare divergent event; the reality gap is per-facet (`2509.22379`). Confirm: pseudo-sim R² 0.8 (`2506.04218`), ladder `2607.07196` | ⭐ YES as INSTRUMENT DESIGN — D-12 must not adopt an average realism score (T-11) | per-event realism on the safety-critical subset, with n and CI, at the claimed level |
| **WB-19-1c** ⭐ | Waabi · **CONCESSION** | *"variations in actor behavior, differences in simulated sensor data, differences in the latency of the system"* compound trajectory differences | admission | **SUPPORTED** | ⭐ YES — the three error sources our T1 tier cannot see (EVAL_DOCTRINE T1 ≠ T2) | — |
| **M-19-1** | Mobileye · 2024-05-15 · PUBLISHED-BLOG | "Zero-bias" E2E needs exponentially more data than an engineered system whose abstractions trade bias for variance | ⛔ NO — theory + analogy; formal source is their own `1604.06915` (D-7, not independent) | **SUPPORTED** as sample complexity (DriveZero decomposes and wins; ours MEASURED: speed-as-action-channel 3.73 → 0.83 m); **CONTESTED** as a system claim | ⭐ YES favourably — E2E-trained with structure-shaped heads | a pure E2E net with no structured heads matching the structured one at small data |
| **M-19-1c** ⭐ | Mobileye · **CONCESSION ×2** | *"Waymo is clear evidence that the bias of an engineered system is sufficiently small"*; E2E nets *"can include sensing-state branches … without sacrificing the E2E label"* | admission | **SUPPORTED** | ⭐⭐ YES — the E2E/modular axis is not the real one; **the intermediate supervision is** | — |
| **M-19-2** | Mobileye · 2024-05-15 | Tesla v12.3.6 ~300 mi per critical intervention ≈ 10 h MTBF, *"6 orders of magnitude"* from 10⁷ h | ⛔ NO — crowd-sourced (RELAYED), 2024 vintage | **UNSUPPORTED-AS-STATED** today (stale by ≥ 2 FSD majors) | NO | a current audited intervention rate |

### Guidelines derived this pass
* **S-8 (strategic)** — world model at **training** time, not in the deploy loop (World Tokens; NVIDIA's latency concession; our sub-300 M thesis). **Falsifier:** planning-by-rollout beating train-time-only WM aux at matched deploy latency on v7-tiny.
* **T-10 (tactical)** — exclusive routing of the action head through the WM-shaped representation, plus a bypass control in every arm (94.1 < 95.0 < 97.0). **Falsifier:** FS19-1 bypass ≥ routed.
* **T-11 (tactical)** — D-12 fidelity is stated per event on the safety-critical subset with n and CI, never as an average trajectory distance. **Falsifier:** average and tail realism ranking simulators identically (ρ ≥ 0.9, ≥ 5 simulators).
* **T-12 (tactical)** — WM prediction targets are semantic features (DINOv3-class, incl. DINOv3 of a BEV raster), never pixels. **Falsifier:** FS19-3 pixel ≥ semantic.

### ⚠️ OPPONENT STRENGTHS, recorded
12. Waabi has paired sim/real replay infrastructure and publishes a quantitative sim-real number; we have neither (AlpaSim unprovisioned, D-12 unmeasured).
13. NVIDIA runs both paradigms at scale (DreamZero/GR00T WAMs and the 34 B Alpamayo VLA) and can hedge the question with compute we lack.
14. Mobileye has absorbed the E2E critique (its sensing-state concession) on a decade of production fleet, which makes it harder to caricature than its headline.

### V-1 register re-find
TechCrunch 2026-09-01 ("Waymo goes on offense"): its primary is the *10 AI Lessons* post (adjudicated 2026-08-31) plus an Axios interview. **No new Waymo claim.**

### Standing debt status this pass
| id | status |
|---|---|
| D-4 (UNECE GRVA primary) | ⛔ STANDS — routes 9–13 failed (UNECE PDF ×2 methods, regulations.gov, federalregister.gov, transportation.gov, globalpolicywatch: 403 / bot challenge). Not bypassed |
| D-7 (`1604.06915`) | ⛔ STANDS — now carries M-1 **and** M-19-1; next rotation #3 |
| D-11 (GAIA-4) | ⛔ OPEN — not re-probed |
| D-12 (AlpaSim fidelity) | ⛔ OPEN — T-11 now fixes the FORM its measurement must take |
| D-13 (Elluswamy transcript) | ⛔ STANDS — 2nd probe EMPTY; speaker's own X post exists but is unfetchable here |
| **D-14** *(new)* | Waymo *Reference Driver* (2026-06 blog + Nature Comms, TU Delft) — unadjudicated A5/Band-D claim about benchmarking AVs against humans |

## Twelfth pass — 2026-09-20 (LAB-RUN-017): the two oldest debts, closed

`Source: TanitAD Research Lab/Opponent Analysis/Research/2026-09-20-red-and-the-modularity-proof/RESULT.md. Seven-step adjudication per amendment 7.1. APPEND-ONLY.`

| id | claim | experiment? | confirming (independent) | contradicting (actively sought) | verdict | binds on us? | flips it |
|---|---|---|---|---|---|---|---|
| **W-20-1** | Waymo `ReD` establishes a reference model of competent human collision-avoidance response (2026-06 blog, `PUBLISHED-BLOG`) | NOT INSPECTABLE - the Nature Comms primary `s41467-026-73345-0` returns **303 to an IdP login**; the blog reports **zero numbers** | none independent of Waymo | the predecessor's own scope concession (W-20-3); the three-expert context objection | **NOT-ADJUDICABLE-AS-PUBLISHED** | YES as a RULE: **no ReD number may enter our registry or any comparison table** | the Nature Comms full text, read (new debt **D-15**) |
| **W-20-2** | `ReD` models **proactive** avoidance, not only last-second reaction | NO - capability assertion, no number | - | the claim's value depends on conceding NIEON's defect | SUPPORTED-AS-INTENT, UNSUPPORTED-AS-MEASURED | YES, favourably: the largest operator concedes a reactive-only benchmark is insufficient - our strategic-tier argument | a proactive-avoidance score with n and CI |
| **W-20-3** | **CONCESSION** (Waymo's own, NIEON): it *"is a tool for evaluating collision avoidance only, as it **inherits the pre-conflict behaviors** ... of the ADS being evaluated"*; models a driver *"that does not exist in the human population"* | admission | - | - | **SUPPORTED** | **YES, strongest of the pass** - a benchmark partly built from the system under test is structurally our **T1 self-action open loop**; an opponent documented our failure mode in its own safety case | - |
| **W-20-4** | **CONCESSION:** counterfactual/disengagement simulation *"cannot definitively predict exactly what would have occurred"* | admission | pseudo-sim R2 0.8 vs closed loop (`2506.04218`) | - | **SUPPORTED** | YES - bounds **D-12**; a counterfactual evaluator is an estimator and must be reported as one (**T-11**) | - |
| **M-20-1** ⭐ | Mobileye `1604.06915`: end-to-end needs *"exponentially larger"* sample complexity than semantic abstraction | **NO - a CONSTRUCTION**, 4 pages, PAC-style, no dataset. Engine: Definition 1 (c-approximate independence) on a **conjunctive** target `g = z1^...^zT` ⇒ `P[g] <= c^T * prod P[zt]`. Its own abstract: *"cases in which"* | the arithmetic is sound; the worked figure (P[zt]~1e-6, c=1.1, T=3 ⇒ P[g] <= 1.34e-18) is internally consistent | DeepSight (monolithic E2E VLA tops Bench2Drive, but with structure-shaped aux losses); **ours, MEASURED**: speed-as-3rd-action-channel took REF-A 3.73 -> 0.83 m, which **confirms** | **SUPPORTED-UNDER-ITS-ASSUMPTION.** Not "modularity wins" but "**a task that factors into approximately-independent sub-events is cheaper to learn factored**" | **YES, favourably - and it retires a mis-framing.** Our hierarchy IS a factorisation claim, so this is an argument FOR a structured design. What it binds is the **burden**: show OUR factorisation is approximately independent ⇒ **H1b** (backlog row 18) | a task provably factoring this way where E2E matches modular at equal data |

⛔ **Correction to this register's own 2026-09-19 framing.** Row **M-19-1** recorded the argument as *"SUPPORTED as a sample-complexity argument; CONTESTED as a system claim"* while noting the source was **unread**. With the source read, the sharper statement is M-20-1: the sample-complexity half is **conditional on a task assumption**, and naming that condition is exactly what the unread text was hiding. M-19-1 is **not retracted** - it is superseded in precision.

### ⚠️ OPPONENT STRENGTHS, recorded
15. Waymo publishes its own benchmark's limitations, in the open, **years before** outside critics reach them. Our eval docs match that discipline only in places.
16. `ReD` is *"fully automated, thus potentially allowing for application on large test sets with thousands of scenarios"* - a scaled counterfactual evaluator we do not have (D-12 is unprovisioned).
17. `ReD` attempts the **right repair** to the inheritance defect (proactive avoidance), rather than defending the predecessor.
18. Mobileye's argument has stood a decade and has absorbed its own counter-examples; it is stronger read than caricatured.

### Standing debt status this pass
| id | status |
|---|---|
| **D-7** (`1604.06915` full text) | ✅ **CLOSED 2026-09-20** - read in full from the bank. ⛔ **Process note: it was banked all along.** Five passes recorded "not reached"; the cost of reaching it was one `find`. A debt that is re-declared without re-probing its cheapest route is a habit, not a blocker |
| **D-14** (Waymo *Reference Driver*) | ✅ **CLOSED as adjudicated-unverifiable** - doctrine recorded, experiment unreadable ⇒ split into **D-15** |
| **D-15** *(new)* | *Nature Communications* `s41467-026-73345-0` (ReD, with TU Delft) - paywalled/IdP. ⛔ **PI-queue**, same class as D-4: a human saves the PDF, the Lab banks and reads it |
| **D-4** (UNECE GRVA / UN R185) | ⛔ STANDS - not re-probed (blocked on a human action, not a search). Noted from the C3 sweep: the UN ADS **global technical regulation** was adopted at the **same June 2026 WP.29 session**, consistent with the pinned request |
| **D-11** (GAIA-4), **D-13** (Elluswamy transcript) | ⛔ STAND - not reached |
| **E-WV20** *(new empty)* | Wayve doctrine via Axios 2026-09-16: **HTTP 403**. A Band-D Wayve item was attempted and not obtained; next route is `wayve.ai`'s own technical blog |
