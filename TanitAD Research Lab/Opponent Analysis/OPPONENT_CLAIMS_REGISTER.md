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
