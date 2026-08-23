# PRE-REGISTRATION — written and staged BEFORE any source was read

**Date:** 2026-07-27 (Europe/Berlin) · **Stream:** Benchmarks & Eval · **Mode:** CPU / web only, NO pod contacted.
**Task:** how should TanitAD evaluate closed-loop driving, given that its own closed-loop instrument
has been shown incapable of producing a MEASUREMENT at any admissible horizon.

This file is frozen. The findings live in `CLOSEDLOOP_EVAL_RESEARCH.md`; this file is never rewritten
to match them. If a conclusion here is falsified, that is recorded in the research doc as a
self-refutation, not edited away here.

---

## 0. What I knew before starting (disclosure — this is NOT a blind study)

`INHERITED` from the brief and from `…/2026-07-26-p1-envelope-revalidation/P1_REVALIDATION.md`
(read §0–§1 only, before writing this):

- Last horizon at which our closed loop is a pure MEASUREMENT: **0.4 s (k = 4)**. `GATE_PROTOCOL` §0.3
  refuses K ≤ 20. ⇒ every admissible gate horizon is an EXTRAPOLATION, including the K = 20 in use.
- Out-of-envelope fractions: K=20 **12.3 %**, K=60 **50.7 %**, K=185 **90.2 %**.
- Widening cannot rescue it: at yaw = ∞ the *lateral* clause alone leaves **3.75 %** of K=20 windows
  outside; `MEASUREMENT` requires zero.
- The 12° yaw edge was never measured — it was the last entry of a `--yaw-grid` default string
  (retraction class C14). Usable edge 15.47°, destroyed at 26.41°.
- ⭐ The envelope is **NOT a renderer-fidelity limit**: the yaw warp is geometrically exact for
  arbitrary depth (`max|ΔH| = 0.000e+00`, 30 conditions). **~Half of what it measures is our own arm's
  OOD sensitivity.**
- AlpaSim's NuRec/gsplat renderer: NGC-DL-CONTAINER-LICENSE, forbids derivatives; reconstruction 3.21× OOD.
- PhysicalAI-AV has **no map, no lane graph, no route/goal signal** (five probes).

I have **not** read any of the five benchmark papers before writing this section. I have run **one**
web search (NAVSIM ↔ closed-loop correlation) whose result list surfaced a paper title —
*"Do Open-Loop Metrics Predict Closed-Loop Driving? A Cross-Benchmark Correlation Study of NAVSIM and
Bench2Drive"* — but not its content. Disclosed so that its later use cannot look like a lucky find.

---

## 1. ⭐ THE PRE-REGISTERED QUESTION (the brief's explicit demand)

> **What evidence would make me recommend REPAIRING our own instrument instead of adopting an external one?**

Committed in advance, before any evidence was gathered. Each is a *falsifiable* condition, not a
preference. If **any** of R1–R4 holds, "repair ours" outranks "adopt external" and I say so even if
the external option is more publishable.

| # | Condition that would make me recommend REPAIR | Why it is decisive | How I would know |
|:--:|---|---|---|
| **R1** | **No external benchmark publishes evidence that its score predicts real or full closed-loop driving** — i.e. they all *assert* validity rather than *demonstrate* it. | Then adoption buys comparability but **not** validity, and we would be trading a known-broken instrument for an unknown-validity one. Our own instrument at least has a measured envelope. | Read each benchmark's own paper for a correlation/agreement study against an external criterion. Absence at one location is not absence (rule 2) — I probe paper + docs + a follow-up study. |
| **R2** | **Every external benchmark requires an asset our corpus structurally cannot supply** — specifically a **map / lane graph / drivable-area polygon**, which PhysicalAI-AV has been shown (5 probes) not to have — *and* the benchmark cannot be run on its own bundled data as a substitute. | Then adoption is not an option at all, only a corpus migration, and the honest recommendation becomes repair-plus-relabel. | Check each benchmark's required inputs. **Key discriminator:** can we run it on *their* data (nuPlan / Waymo Open Motion / OpenScene) rather than porting ours? If yes, R2 does **not** fire. |
| **R3** | **The out-of-envelope problem is shown to be a property of OUR ARM, not our substrate, and external benchmarks share it.** P1 already found the warp is geometrically exact and "roughly half" the degradation is our arm's OOD sensitivity. If the literature shows every log-replay benchmark has the same unbounded-deviation problem and simply *does not report it*, then our instrument is not worse — it is merely **honest**, and the fix is to publish the envelope label, not to switch tools. | This would reframe the wound as a disclosure advantage. | Look for explicit treatment of ego-deviation validity limits in nuPlan/Waymax/NAVSIM. If they are silent, our envelope is a contribution. |
| **R4** | **A cheap, bounded repair exists that converts our instrument from EXTRAPOLATION to MEASUREMENT** — e.g. a *short-horizon* one-shot protocol whose windows are provably 0 % out-of-envelope, with a metric that still discriminates policies. | The 0.4 s measurement horizon is not useless if the metric is redesigned around it rather than around 2 s rollout. NAVSIM's very existence is evidence this class of design is respectable. | If NAVSIM's argument holds, it is *simultaneously* an adoption target and a template for repair. I must not conflate the two. |

### 1.1 The symmetric condition — what would make me recommend ADOPT

Stated so that R1–R4 are not a stacked deck:

- **A1** At least one external benchmark **demonstrates** (not asserts) that its score ranks policies
  consistently with a fuller closed-loop or real-world criterion, with a reported statistic (ρ, τ, or
  rank agreement) and a sample size.
- **A2** It is runnable on **CPU or a single non-training GPU**, on **its own** bundled data, under a
  licence that permits our use (research at minimum; **code licence and data licence checked separately**).
- **A3** It yields a number that is **comparable to a public leaderboard** — something no internal
  instrument can ever give us.

**If A1–A3 all hold AND none of R1–R4 fires, I recommend ADOPT and say plainly that our instrument is
not worth repairing.** If both sets partially fire, I rank and give both sides rather than forcing one.

---

## 2. ⚠️ The C13 gate applied to MY OWN method, before I cite anything

*"A guard that cannot fail is not a guard."* This is a literature study; its failure mode is
**confirmation by citation volume**. Pre-committed controls:

| risk | control committed in advance |
|---|---|
| **Asserted validity read as demonstrated** | Every benchmark row carries a **separate** column for *what it DEMONSTRATED* vs *what it ASSERTED*. A claim with no statistic, no n, and no external criterion is logged as ASSERTED even if the paper's abstract states it confidently. |
| **Licence-from-short-name** (this program has made this error **twice**: ZOD, then nuScenes = `CC-BY-NC-SA`, copyleft) | No licence is recorded from a badge, a GitHub sidebar, or a short name. **Fetch the terms document.** Record **code licence and data licence as separate fields** — they differ for nuPlan, Waymo, and NAVSIM. If I cannot fetch the text, the field is `UNVERIFIED`, not a guess. |
| **Correlation cherry-picking** | If a correlation study reports both a strong and a weak result (e.g. strong on one metric, near-zero on another), **both** go in the table. I pre-commit to reporting any *negative* correlation finding at the same prominence as a positive one. |
| **Recency bias / narrative clock** | Repo notes are dated ahead of wall-clock in this program. Every citation carries its **year and venue**; a preprint is labelled a preprint. |
| **Recommending the thing I researched most** | The ranking must be justified against the R/A conditions above, not against how much material I found. |

## 3. Priority order (a killed run still yields value)

1. Per-benchmark validity table (simulates / does not / evidence class / licence).
2. ⭐ NAVSIM's correlation evidence — potentially decisive.
3. Replay-validity bounds (the literature on our exact envelope question).
4. Reactive-agent options.
5. Metric discrimination vs saturation.

Banked incrementally: each section is written to disk as it completes, not held for a final synthesis.

## 4. What I predict (so I can be wrong on the record)

`HYPOTHESIS`, committed before reading:

- **P-a** NAVSIM will turn out to publish a correlation against **PDM-Closed / nuPlan closed-loop**, not
  against real driving, and the "closed-loop correlation" claim will be **weaker than its reputation**.
- **P-b** No benchmark will publish an explicit ego-deviation validity bound. **Our envelope will turn
  out to be more rigorous than the field's practice** — which is a publishable contribution, not a wound.
- **P-c** The strongest adoption target will be a **map-carrying** benchmark, so R2 will *partially*
  fire and the real recommendation will be "run on THEIR data", not "port our corpus".

If P-a is false and NAVSIM has real-driving correlation, adoption wins outright and I will say so.
