<title>LEDGER B10 - semantic search, retrieval, embeddings</title>

# LEDGER_B10 - semantic search / retrieval / embeddings

`APPEND-ONLY. The running report for track B10, with our position in it.`
`Opened 2026-09-09. Prior B10 coverage lived as a section inside LEDGER_C3_regulatory.md (2026-08-31); this file is the proper home and future appends go here.`

---

## 2026-09-09-01 - The clip-value signal is the model's own surprise, and 0.314 of it is constant velocity

`lib 2606.28383 "Zero-Label Driving Scenario Complexity Detection via Joint Embedding Predictive Architecture". FULL TEXT. Class PUBLISHED. Retrieved 2026-09-09.`

**Score:** `s = || z_hat_tgt - z_tgt ||_2`, the JEPA's own temporal prediction error. Verbatim: *"no
labels are used during training or in defining the surprise score itself."*

**Ranking behaviour:** highest surprise on *"stopping at traffic light without lead"* (2.008) and
*"starting unprotected cross turn"* (1.978); lowest on lane-following and stationary traffic.
Downstream anomaly detection: **AP 0.512 against a 0.436 chance baseline.**

**The four ablations - and they are the control panel CLAUDE.md mandates:**

| ablation | Spearman rho | reading |
|---|---|---|
| shuffled scores | -0.126 | no-information value, read correctly |
| random encoder | 0.024 | untrained floor reads approximately zero |
| **constant-velocity baseline** | **0.314** (30 % lower score spread) | **a large share of "complexity" is kinematic** |
| no-EMA training | **44-fold collapse** | EMA is the mechanism, not a refinement |

**Our position.** This lands on backlog row 22 / H-DATA-1. The signal costs one forward pass and we
already compute it every training step and discard it.

**Two things we must add that the paper does not:**
1. **A constant-velocity floor beside every clip-value number.** 0.314 is our CTRV problem in a new
   costume, and the same family as the ridge-probe failures where latent, `[z, dz]` and raw pixels all
   read +0.54 by reproducing the mean.
2. **Separation as well as surprise.** MM-1 MEASURED that our h=1 ratio moved mostly through its
   denominator; a clip-value score built on prediction error alone inherits that confound - a clip
   could score "valuable" because the scene drifted. Pair it with SR from `LEDGER_A2_jepa.md`
   2026-09-09-01. **Neither paper states this; the composition is ours.**

**Correction entered against our own record.** Backlog row 4 / MM-E1 says *"our fixed tau=0.996 has no
published operating point - every EMA-teacher line ramps tau."* This paper uses **fixed alpha = 0.996**,
does not ramp, and measures a 44-fold collapse without EMA. **That sentence is retracted.** Scope:
1,289,130 parameters, structured agent-state vectors, nuPlan mini, 1,322 scenarios - **not pixels, not
our scale**, so the ramp-or-not design question at v7 scale is untouched.

**Independent second line, same direction:** RealDrive `2505.24808` (abstract-level) uses embeddings
*"derived from a planning model"* because prior work *"focused on interaction between agents rather than
decision-relevant information."* **Select on decision-relevance, not scene statistics.**

**Not admissible:** Kairos `2606.16533`'s two-level-curation claim - unread at three routes today,
`RELAYED`, decides nothing (debt D-8).

## 2026-09-10-01 (B9 section) — Kairos reached at the FIFTH route; the claim is a proposal (debt D-8 DISCHARGED)

`FULL TEXT, BOTH VERSIONS` · arXiv **2606.16533** — v1 (90 pp.) and **v3 (119 pp.)**, both retrieved 2026-09-10.

⛔⭐⭐ **AN ARXIV ID IS NOT A DOCUMENT.** Measured greps: **v1** `control-relevant` ×1 (an unrelated VideoDiT ablation), `two-level` ×3 (**all "two-level batching"**, an I/O technique), `control-informative` ×0, `regret` ×**0** — while the arXiv title reads *"A **Regret-Aware** Native World-Action Model Stack"*. **v3** `control-relevant` ×42, `regret` ×96, the claim present verbatim. The paper **changed its own title** and grew 29 pages between versions. ⇒ **reading v1 would have filed a confident, false `REFUTED` on a true claim.**

⭐ **The retrieval block was SIZE, not access.** Routes 1–4 (fetch-tool PDF, landing page, HTML, HTML v1) all truncated at the same sentence; **route 5 — byte download via `urllib` + local PyMuPDF extraction — succeeded** (42,560,348 B for v1; 43,862,955 B for v3; both `%PDF-` header, valid `%%EOF`). Three "unreachable" verdicts across two passes were one tool's size limit on a fully public document.

⛔⛔ **THE CLAIM IS A PROPOSAL. Level 2 is unbuilt.** Verbatim v3: *"Therefore, Kairos interprets data curation as a two-level process: basic quality filtering followed by control-relevant event filtering."* — then: *"The second level **would** prioritize clips according to control information density. **The current pipeline does not yet compute CID directly**; future iterations should explicitly target: …"* and *"these control-relevant filters **should be presented as a guiding extension**."*
⇒ **No CID computation, no level-2 retention rate, no ablation of curation against downstream model quality.** `PUBLISHED` **as design intent**, `UNSUPPORTED-AS-STATED` **as evidence that control-relevance filtering works.** ⛔ **It may not decide a GPU-day, and the 2026-09-09 `control-relevant-clip-value` package needs a citation correction.**

⭐⭐ **What transfers — the six-family taxonomy:** near-boundary failures · recovery events · near-boundary successes · contact transitions · safety/anomaly events · long-horizon dependencies. Their argument, verbatim: *"A model that never sees near-boundary failures may generate clean success futures but remain unable to predict where deployment will break. A model that never sees recovery may fail to plan after mistakes."* ⇒ **FS9-3 and row 22 lacked an operational definition of "control-relevant clip"; this supplies one, and TanitAD can run the measurement Kairos deferred.**

**What IS implemented (level 1):** PySceneDetect, *"over 95 % segmentation precision and 80 % recall"* (77.44 % after throughput optimisation); segments 5–40 s, >40 s split to 20 s, <5 s discarded; aesthetic score (CLIP backbone + MLP head, thresholded), motion score, temporal-coherence / safety / redundancy filters. ⚠️ **All thresholds unstated; no filter ablated.**

⭐ **Kairos's own concession:** *"Direct validation of real-robot closed-loop regret reduction … remains an important [direction]"*, with results offered as *"proxy evidence"* — our open-vs-closed-loop binding ruling, stated by an industrial full-stack programme.

→ `Data Engineering/Research/2026-09-10-kairos-curation-fourth-route/RESULT.md`
