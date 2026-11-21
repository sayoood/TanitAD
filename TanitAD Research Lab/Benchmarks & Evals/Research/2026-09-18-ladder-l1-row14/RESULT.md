<title>Admissibility ladder read in full — row 14's committed branch does NOT fire, but row 14's target is anchored on the wrong tier</title>

# `E-BE-LADDER-1`: `2607.07196` read in full. Its L1 is an **action-following** test the paper deliberately scopes **to generative models** — so a 3DGS replay is not "capped at L0". ⛔ **But row 14's R² target is correlated against T1, which our own doctrine rules is not closed loop, so it can never be an L4 anchor**

**2026-09-18 · Research Lab (LAB-RUN-015) · Benchmarks & Evals · serves LR14-8 + LR14-9 (backlog row 14) and Band A5**
0 GPU. Primary read in full (pypdf, 10 pp); ⭐ debt `2607.07196` full-text **discharged**.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐ **The operational L1 is published, and it is concrete.** Feed a commanded action template, generate the rollout, recover the executed manoeuvre and trajectory with the **ACT-Estimator**, and score **IEC** (manoeuvre match over 8 commandable categories, chance 1/8). A model *"clears L1 if its IEC lies significantly above chance."* Supporting metrics: ADE, DTW, success rate (FDE < 3 m). The instrument is validated first: real-clip ADE **0.77 m**, classifier reproduction **93.4 %** vs 94.0 % published, Vista IEC reproduced at **30.7 %**. | PUBLISHED lib `2607.07196` §B.2 (full text) |
| **F2** | ⭐⭐⭐ **LR14-8's committed branch does NOT fire, and the reason is in the primary.** The paper explicitly **excludes reconstruction-based simulators**: *"This paper targets that class"* (fully generative), because NeRF/3DGS *"reduce the sim-to-real gap largely to appearance fidelity, which is partially measurable against the source recording."* For an ego-pose-driven 3DGS replay, **L1 is satisfied by construction** for the ego: the camera *is* moved by the commanded action. ⇒ *"row 14 is capped at L0"* is **false**. The ladder's L1 is the **wrong instrument** for a reconstruction sim, and it would pass trivially. | PUBLISHED (§III-A) + analysis |
| **F3** | ⭐⭐⭐ **What does bind row 14 is L2, the envelope.** L2 requires *"a declared training envelope and rollout horizon, with out-of-distribution detection and refusal outside."* For 3DGS the envelope is **spatial**: novel-view quality decays with **lateral/heading distance from the recorded trajectory**. That is exactly Waymo's **W-17-4** concession (*"visual breakdowns due to missing observations"*). Row 14's perturbation grid lives **outside** the recorded path by design. ⇒ **row 14's first rung is an envelope measurement, not an L1 check.** | PUBLISHED + W-17-4 |
| **F4** | ⛔⭐⭐ **Row 14's target is anchored on the wrong tier.** *"R² ≥ 0.7 against our T1"*: our `EVAL_DOCTRINE.md` defines T1 as **self-action OPEN loop**, *"⛔ NEVER 'closed-loop'"*. L4 requires *"measured in-sim ↔ real correlation"*. **A correlation between two open-loop-class instruments is not an L4 anchor**, however high. The published R² = 0.8 (pseudo-simulation, `2506.04218`) was measured against **closed-loop** nuPlan. ⇒ at ≥ 0.7 row 14 would have validated agreement with T1, not validity. | MEASURED (doctrine text) + PUBLISHED |
| **F5** | ⭐ **The paper's own L0 ↔ L1 reversal is n = 2 models**, and it says so: *"A single model pair does not establish how often the two properties diverge."* Good as a **possibility** proof, not a rate. Epona's L2 horizon is **h\* = 3.2 s** at a 1.8 m band vs Vista's **1.6 s**. ⚠️ Our 6 s tactical horizon exceeds **both**. | PUBLISHED |

**Verdict against LR14-8 (committed on 09-17):** the *"cannot be made action-responsive ⇒ capped at L0"* branch **does NOT fire** (F2). This is a **WORSE-BRANCH**: the ladder's L1 does not discriminate for reconstruction sims. The rung that binds is L2 (F3), and the L4 anchor is mis-specified (F4). **LR14-9's re-scope is therefore upheld, with a sharper target.**

## 1 · Row 14, re-scoped (proposed, not self-ranked)

| rung | what row 14 must show first | instrument we already hold |
|---|---|---|
| **L2-envelope** | NuRec/gsplat render quality vs lateral offset **d** and heading offset from the recorded path. Declare **d\*** where a frozen-encoder feature distance to the nearest *real* frame exceeds the real-frame noise floor | frozen refcv5-v2 trunk (VGEO-5 measured today that it carries **place identity**, P2 +0.2088, so it can score "is this still the same place") |
| **L4-anchor** | correlate the pseudo-sim score with **AlpaSim (T2, closed loop)** on the same scenes, **never with T1** | AlpaSim (bare on an A40; provisioning is the PI's) |

## 2 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | Backlog row 14, LR14-8/-9, register W-17-4, T-6; `EVAL_DOCTRINE.md` tiers. |
| **CONSEQUENCE** | Row 14 must be re-worded before it is designed: target **L2-envelope first**, anchor L4 on **T2**. Its current R² ≥ 0.7-vs-T1 bar would have produced an uninterpretable pass. |
| **COMBINATION** | ⭐ Joins today's VGEO-5 result: the same frozen trunk that **fails** on path-length geometry **passes** on place identity. That is exactly the property an envelope detector needs ("is this rendered frame still recognisably this place?"). One instrument, two uses. |
| **CHANCES / RISKS** | **Upside:** an envelope measurement is cheap (render + encode, no training) and tells us where our own sim can be trusted. **Risks:** (a) a feature-distance envelope can be fooled by blur that preserves place identity; pair it with a pixel/LPIPS floor; (b) T2 anchoring needs AlpaSim provisioning, a PI item. |
| **EXPERIMENT** | **`E-BE-ENVELOPE-1` (proposed BE18-1):** on the NuRec scenes already on disk, render at lateral offsets {0, 0.5, 1, 2, 3} m. Score each render by frozen-trunk distance to the nearest real frame, normalised by the real-frame same-place floor. **Committed:** d\* = the largest offset whose median normalised distance stays ≤ **1.5×** the floor. d\* ≥ 2 m ⇒ row 14's perturbation grid is inside the envelope and the pilot may proceed. d\* < 1 m ⇒ the grid is out of envelope and row 14 is **closed** as specified. Controls: the offset-0 render must read ≤ 1.2× the floor, or the renderer itself fails L0. |

## 3 · Stopping condition (Rule Zero)

**(3b), named.** The committed branch was answered as written (it does not fire). The next lever, BE18-1, needs **NuRec rendering**, and gsplat runs on **Thor**. ⛔ The Lab may not touch Thor, so BE18-1 is blocked on **a FlyWheel slot on Thor** (or a dev-box gsplat install), and is proposed with its bar committed.

`Deliverables: RESULT.md · raw/search_log.md`
