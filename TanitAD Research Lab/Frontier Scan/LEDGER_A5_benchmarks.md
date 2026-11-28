<title>LEDGER A5 - benchmarks and evaluation (driving, embodied)</title>

# LEDGER — A5 · Benchmarks + evaluation

`APPEND-ONLY. Charter DAILY_RESEARCH_CHARTER.md §3. Opened 2026-09-01.`
`Opened because TRACKS.md recorded A5 as "partial - no dedicated DEEP" since the charter began.`

---

## Entry 2026-09-01-01 - WorldRoamBench: trajectory-level metrics HIDE action-following failure

**Source:** `arXiv 2606.31672`, "WorldRoamBench: An Open-World Benchmark for Long-Horizon Stability of
Interactive World Models" (Xu, Sui, Gao, Shi, Yang, Liu, Sun, Sun, Pan, Jiang, Xu, Fan, Gao, Li,
Chen; submitted 2026-06-30, revised 2026-07-06).
**Evidence class:** PUBLISHED lib 2606.31672 **abstract-only - DECLARED.** Banked this pass (the only
new bank of the day).

> *"existing benchmarks evaluate action following only at trajectory level and ignore memory and
> interaction physics"*

Four dimensions: **(i) Action** - a per-frame action metric "exposing failures hidden by trajectory";
**(ii) Vision** - a segment-based drift metric "capturing non-monotonic mid-sequence collapse missed
by start-vs-end comparisons"; **(iii) Physics** - controllability-gated scoring over mechanics, optics
and 3D consistency; **(iv) Memory** - scene memory via transition-localized 3D point-cloud
reconstruction, subject memory via tracking + VLM reasoning.

Over **10+ open/closed-source models: "None reliably satisfies all dimensions."**

⭐⭐ **What it changes for us - an independent line arrives at our action-echo finding.**
"Failures hidden by trajectory[-level metrics]" is precisely `EVAL_DOCTRINE` §1.12 MEASURED:
open-loop S-curve reproduction **97.9 %** vs hold-action **0.0 %** vs closed-loop **~5 %**. A further
independent line now holds that trajectory-level open-loop scoring conceals action-following failure.

⚠️ **Two instruments we do not have:** (a) a per-frame action-fidelity metric; (b) a **segment-based**
drift metric. ⛔ **Our drift is currently read start-vs-end, which by their construction cannot see
mid-sequence collapse** - so every drift number we hold may be a lower bound.

**Risk:** built for open-world *interactive* world models, not driving; the physics/memory dimensions
may not port. **Abstract-only: no degradation curves or per-model numbers were readable, so nothing
here may decide a GPU-day yet.**

**Pre-registerable experiment (0 GPU, proposed unranked):** port ONLY the segment-based drift metric
into `taniteval` and re-score the banked v7 rollout dumps. **Committed outcome:** monotone drift =>
start-vs-end is adequate and the instrument stands; **non-monotone mid-sequence collapse => every
drift number we hold is a lower bound and the instrument must change.**

`Next in this track: full text for the segment-metric definition; then the per-frame action metric.`


---

## Entry 2026-09-02-01 — WorldRoamBench full text: the drift metric is now portable, and non-monotonicity is MEASURED

`PUBLISHED lib 2606.31672 · FULL TEXT READ 2026-09-02 (was abstract-only 2026-09-01) · unblocks backlog FS-1`

### The segment-based drift formulation, exactly

Partition the rollout into **N = 10** windows; take the **best-quality** and **worst-quality** windows;
report the **relative percentage change** between them.

> *"D = 0.05 means the worst 10% window of the video is 5% below the best 10% window in average score;
> unlike a fixed start/end comparison, this localizes the strongest dip wherever it occurs, so transient
> mid-rollout collapses that are later masked by recovery still contribute to D."*

Relative rather than absolute *"keeps drift comparable across models with different score magnitudes"*.
Instantiated on aesthetic and imaging score sequences to give two lower-is-better metrics.

### ⭐ The non-monotonicity FS-1 was written to test is MEASURED, not hypothesised

- *"A subset of models additionally show transient mid-rollout collapse, dipping sharply mid-sequence
  before partially recovering; **a start-vs-end comparison would treat them as stable**, but our
  segment-based best-vs-worst drift still localizes and penalizes the dip."*
- `minWM`: *"selective collapse, with its aesthetic score crashing after frame 150 even as imaging stays flat"*.
- `Matrix-Game 3.0`: *"progressive, accelerating decline driven by compounding autoregressive error accumulation"*.
- `HY-World 1.5` best: aesthetic drift **13.85**, imaging drift **15.91**. `Yume 1.5` *"starts strong yet
  accumulates 23.01 imaging drift"*.
- *"low drift and high average quality are correlated but not identical"* — frame-averaged quality cannot
  substitute for a stability measure.

### ⭐⭐ Second independent confirmation of our action-echo finding, now with numbers

> *"Models with high trajectory alignment (trajectory score above 85) can exhibit **below 65% per-frame
> strict action accuracy**, revealing a hidden failure mode in which a model eventually drifts."*

⇒ Three independent lines now agree that trajectory-level open-loop scoring conceals action-following
failure: ours (S-curve 97.9 % open-loop / 0.0 % hold-action / ~5 % closed-loop), Alpamayo, and this.
**Guideline S-1 (T1 is the burden of proof) strengthens.**

### ⚠️⛔ SCOPE LIMIT — this changes what FS-1 may port

Their drift is computed over **image-quality** scores (aesthetic / imaging). **Ours is a trajectory/latent
quantity.** ⇒ **Port the segment-based best-vs-worst FORMULATION, not the metric.** Quoting their
13.85 / 23.01 against our drift numbers would be the `df` / `step_s` scope error in a new costume.

⇒ FS-1 is **unblocked and executable at 0 GPU**: N=10 segments, best-vs-worst, relative change, over the
banked v7 rollout dumps. **Committed outcome unchanged: monotone ⇒ the instrument stands; non-monotone
⇒ every drift number we hold is a LOWER BOUND and the instrument must change** — including the MM-E1 EMA
read (0.4531 -> 0.36-0.40) currently deciding a recipe.

`Next in this track: C4 leaderboard numbers landed 2026-09-02 (see LEDGER_C1_releases.md C4 entry) but`
`are NOT admissible into a comparability table until backlog row 3 (portfolio) and D-EPDMS-FAM land.`


---

## Entry 2026-09-05-01 — ⛔ THE navhard EPDMS LEVEL DOES NOT TRIANGULATE WITH OUR OWN RECORDS

`Three independent 2026 sources read this pass (Q12 full text, Q16 search). Cited by:`
`Frontier Scan/Daily/2026-09-05/RESULT.md F4.`

### What the field reports on navhard

| source | evidence class | navhard EPDMS |
|---|---|---|
| GuideFlow `2511.18729` Table 1 (**FULL TEXT**) | PUBLISHED | LTF **23.1** · GTRS-DP\* **23.8** · DiffusionDrive\* **24.2** · DriveSuprim **42.1** · GuideFlow+Scorer **43.0** · DiffVLA **45.0** |
| IDOL `2605.31476` | PUBLISHED (abstract-level) | IDOL **38.0** — *"highest final EPDMS among all comparable methods"*, +10.1 over WoTE |
| RAP `2510.04333` | PUBLISHED (abstract-level) | RAP-DINO **36.9**, claimed SOTA on NAVSIM v2 |

### ⛔ What WE hold

`LAB_BACKLOG.md` row 32 and the 2026-09-02 frontier pass carry **DrivoR 56.3**, **privileged PDM-Closed
56.6** and **DriveFuture 55.5** on navhard.

**Every one sits ~11–13 points above the highest number in any of the three tables above, and two of the
three claim SOTA at 36.9–38.0.** ⇒ **Three independent papers do not miss a 56.3 that outranks them by 18
points.** These are not one leaderboard: different split, different NAVSIM version, or a different
scoring-basis era. **This is V-5's named corruption path, arriving exactly as V-5 predicted.**

### Consequence — binding

⛔ **Row 32's efficiency wedge is BARRED from quoting any of these levels until the scoring basis is
reconciled.** This is *stronger* than the 2026-09-02 split-mixing caution: we now hold three independent
contradicting tables, not a general rule.
⚠️ **It does NOT retract CW-1's resolution** (`2026-09-02-cw1-resolution-and-evidence-integrity`), which
settled the *internal consistency* of DriveFuture's own rows. This is a **new** question one level up, on
the *level*. The two findings are compatible and must not be merged.

### ⭐ The other half of the same table — it confirms Band D

A 3DGS-based framework reports **50.9 EPDMS on a pseudo-closed-loop navhard**, and NAVSIM v2's navhard
Stage 2 *"uses 3D Gaussian Splatting to synthesise counterfactual camera views after policy deviations,
thereby simulating closed-loop evaluation from logged data"*. ⇒ **The benchmark our portfolio decision
(backlog row 3) already ranks GO is itself a counterfactual-rendering pseudo-closed-loop evaluation** —
independent confirmation of the Wayve GAIA-4 doctrine (register V-1), and direct support for backlog
row 14 (pseudo-simulation on our NuRec/gsplat assets).

`Position: we cannot enter ANY navhard comparability table until the era question is settled. Proposed`
`row FS5-3 makes that a work item rather than a standing caveat.`

---

## 2026-09-09-01 - Debt D-10: navhard and "NAVSIM v2 12k" are TWO DISJOINT POPULATIONS

`lib 2603.24581 Latent-WAM, FULL TEXT results. Plus the NAVSIM maintainers' guidance (PUBLISHED-BLOG). Retrieved 2026-09-09.`

| cluster | values (EPDMS) | basis | class |
|---|---|---|---|
| navhard | 23.1-45.0 (GuideFlow) - 38.0 (IDOL) - 36.9 (RAP-DINO) | 450 Stage-1 + 5,462 Stage-2 observations | PUBLISHED, read 2026-09-05 |
| **"NAVSIM v2", 12k scenarios** | **89.3** Latent-WAM - **86.1** DriveVLA-W0 - **85.1** Epona - **84.8** World4Drive | *"12k evaluation scenarios"*, verbatim; **split never named** | PUBLISHED, read today |
| ours | 55.5 DriveFuture - 56.3 DrivoR | **unknown** | INHERITED, provenance unrecorded |

**Mechanism, from the benchmark's own maintainers:** they *"discourage the use of **self-reported and
unofficial 'NAVSIM v2' benchmark splits**"* and direct submitters to the leaderboard *"to ensure both
consistency and visibility of submissions."*

**Verdict: the roughly 40-point spread is not a capability ranking, it is two measurement bases wearing
one metric name.** Rule V-5's named corruption path, caught before it entered a table.

**Our own two numbers match NEITHER cluster** - above navhard's 45.0 ceiling and well below the 12k
cluster's 84.8 floor. That makes their provenance unknown, not merely disputed.

**D-10 state: MECHANISM RESOLVED, PROVENANCE OPEN.** Row 32 and FS5-3 stay BARRED.
**Guideline T-4: no external EPDMS number enters a TanitAD table without a SPLIT STAMP.**

**Caveat recorded against ourselves:** the assignment of "12k evaluation scenarios" to a navtest-class
split is **our inference from the scenario count**, not a quotation. Two probes (abstract, results) found
no split name; an appendix probe was not possible.

**Cross-link:** 2026-09-05 established that **navhard Stage 2 IS a 3DGS counterfactual pseudo-closed-loop
evaluation**. That gives a physical reason for the gap rather than a bookkeeping one, and it makes
navhard the tier our T1 doctrine says is comparable to us.

## 2026-09-10-01 — D-10: the counts partition, the scoring basis does not

`PUBLISHED` · arXiv **2506.04218v2** (retrieved 2026-09-10) — *Pseudo-Simulation for Autonomous Driving*, CoRL '25, **the NAVSIM v2 maintainers' own paper.**

**Verbatim:** *"It uses a subset of nuPlan that we refer to as navhard, involving **450 Stage 1 and 5462 Stage 2** observations."*
**navhard leaderboard anchors, same primary:** PDM-Closed **51.3** · Latent TransFuser **23.1** · MLP **12.7** · Constant Velocity **10.9**. EPDMS range **[0, 1]**.
**A separate correlation-study subset** — *"244 initial observations (Stage 1) and 4164 synthetic observations (Stage 2)"* — ⚠️ **is NOT navhard and must never be reported as such.**
**navtest** ≈ **12,000 samples** (`PUBLISHED-SECONDARY`, consistent across multiple 2026 method papers; ~136 test logs).

⭐⭐⭐ **FS9-2's first branch fires: the two populations separate by count.** navhard ~5.5 k Stage-2 · navtest ~12 k. ⇒ **the 84.8 / 85.1 / 86.1 / 86.4 / 89.3 cluster is navtest-EPDMS and may NOT be called navhard.** Drive-HWM's 86.4 (read today) is stamped **navtest**; its 93.8 PDMS is **NAVSIM v1**, a different metric.

⛔ **RESIDUE — the part that matters.** Our records carry **PDM-Closed = 56.6 on navhard** (LAB-RUN-008, the basis for row 32's *"within 0.3"*); this primary reports **51.3 on navhard**. **One privileged planner, one split name, a 5.3-point spread.** ⇒ **the split name does not pin the scoring basis.** Candidate causes, unverified: a NAVSIM version bump between CoRL '25 and the 2026 tables; a two-stage weighting change; EPDMS sub-metric additions.

⇒ **D-10: MECHANISM RESOLVED (09-09) · COUNTS RESOLVED (09-10) · BASIS OPEN. Row 32 stays BARRED. Guideline T-4 stands.**

⚠️ **DATED CORRECTION to our own record** (append-only discipline — the 2026-09-05 entry stands as written). It reads *"three independent tables cap navhard at ≤ 45.0"*. The maintainers' primary puts PDM-Closed at **51.3** on navhard. **The ≤ 45.0 cap is SUPERSEDED and must not be re-quoted.**

**Metric definitions banked for D-EPDMS-FAM (backlog row 12):** `PDMS = NC · DAC · (5·EP + 5·TTC + 2·Comf)/12`. EPDMS adds **DDC** (driving-direction compliance), **TLC** (traffic-light compliance), **LK** (lane keeping), and replaces Comf with **HC** (history comfort) and **EC** (extended comfort). ⭐ **None of the eleven sub-metrics maps to `tac.manoeuvre_decision` or `strat.route_goal`** — the four-families gap is confirmed from the metric's own definition, not merely suspected.

→ `Benchmarks & Evals/Research/2026-09-10-navhard-split-stamp/RESULT.md`


## 2026-09-13-01 — ⛔ DATED CORRECTION to 2026-09-10's residue: the PDM-Closed 51.3-vs-56.6 spread was TWO VERSIONS of one paper — D-10 CLOSES

`FULL TEXT` · arXiv **2506.04218 v3** (20 Mar 2026, banked 2026-08-23) vs **v2** (27 Aug 2025, HTML) · `2606.07170` TOAD Tables 2/6.
v2 Table 2: PDM-C navhard two-stage **51.3**. v3 Table 2 (*"navhard leaderboard. Snapshot from 03/2026"*, 11 entries, source `huggingface.co/spaces/AGC2025/e2e-driving-navhard`): PDM-C **56.6**, DrivoR 54.5, SimScale 53.2, GuideFlow 51.5, ZTRS 48.1, RAP 39.6, NavFormer 34.1, LTFv6 31.9, LTF 25.1, Ego MLP 14.1, CV 11.4. Both versions: 450 S1 / 5,462 S2.
**Stage 1 identical 9/9** (94.4/98.8/100/99.5/100/93.5/99.3/87.7/36.0); **Stage 2 moved 7/9** (NC 88.1→90.5, TTC 83.1→86.6, EC 25.4→29.7, LK 73.7→74.2, HC 91.5→91.9, DDC 96.3→95.4, TLC 98.5→98.4) — v3 = TOAD exactly.
⇒ the **Stage-2 substrate** was re-scored between the 08/2025 and 03/2026 snapshots. **BE10-1 branch 1 fires; D-10 CLOSED.** The 2026-09-10-01 entry stands as the record of what we believed; its cause was reading v2 while holding v3 (LI10-1 class).
**Admission test from now on:** a shared deterministic baseline's Stage 1 must match this fingerprint; the number carries its snapshot date.
→ `Benchmarks & Evals/Research/2026-09-13-pdmc-stage2-provenance/RESULT.md`


---

## 2026-09-17 (LAB-RUN-014) — the #151 primary read, and a published admissibility ladder for WM simulators

**(a) `E-BE-FIX151-1` executed — the mechanism confirms, the committed partition does NOT.**
NAVSIM issue #151 (opened **2025-09-03**) is titled *"Question Regarding the human_penalty_filter
Mechanism: Discrepancy Between Reported Sub-score and Final Score Calculation"*: when the human
driver fails a metric, the agent's score is exempted to **1.0 in the reported sub-scores** while the
final total still used **0.0**, because the exemption did not modify the **`weighted_metrics_array`**.

⭐ **EPDMS partition, recorded once so it is never re-derived:**

| kind | members |
|---|---|
| multiplicative penalty gates | **NC · DAC · DDC · TLC** |
| weighted metrics (weights) | **EP (5) · TTC (5) · LK (2) · HC (2) · EC (2)** |

⇒ the bug lives in the **weighted** array, which **contains EP**. `E-BE-FIX151-1` committed: *"if the
fix touches EP or DAC, the 7/9 pattern has an unexplained component and the stamp table carries a
caveat."* **That branch fires.** And its complement is sharper: **#151 cannot directly move NC / DAC /
DDC / TLC**, so any Stage-2 movement on the four *safety* terms has a second, still-unnamed cause.
⚠️ The fix commit itself is **unread** — the issue is closed with none linked on its page (one probe).

⛔ **CORRECTION to this ledger's 09-15 stamp table (dated entry, history not rewritten): an external
EPDMS now needs TWO stamps** — the **split** (`navtest` / `navhard`) *and* the **fix**
(`pre-` / `post-#151`). **WA-JEPA 91.7 is `navtest` and is excluded from every navhard table.**

**(b) `2607.07196` — "Validate the Dream Before You Trust Its Verdict"** (Oefinger, Schäfer, Moller,
Piccinini, Betz; 2026-07-08). A WM used as a **test oracle** must be *accredited* before its verdicts
are evidence: ladder **L0** visual fidelity -> **L1** action-responsiveness -> **L2** declared operating
envelope + OOD detection -> **L3** failure attribution separating simulator from policy error -> **L4**
measured sim-to-real correlation. Built on VV&A, SOTIF and scenario-based testing practice.

⭐⭐ **The measured result, and it is our own doctrine from outside:** applied to two driving WMs,
*"the model that ranks higher on visual generation quality (L0) ranks **lower** on action-following
(L1-L2), so visual fidelity does not predict the action-robustness a closed-loop verdict depends
on."* That is the ridge-probe lesson in assurance costume: **the impressive surface metric is the one
to distrust.**

⭐ **Our position.** **Backlog row 14** (NuRec/gsplat pseudo-simulation pilot) targets *"R2 >= 0.7 vs
our T1"*, which is an **L4** claim — the top rung — while having **no L1 check at all**. Re-scope it
to L1-L2 first; that is also far cheaper. ⛔ And Waymo's own concession the same day (*"purely
reconstructive simulation methods suffer from visual breakdowns due to missing observations"*) names
the hazard: a replay-based asset may be **capped at L0 by construction**, because a counterfactual
action has nothing to respond with. `E-BE-LADDER-1` registered.

⚠️ **Limit:** the rung definitions here come from the abstract + a secondary. **The PDF is banked and
its full text is a debt.** No rung threshold is quoted as a number.

→ `Benchmarks & Evals/Research/2026-09-17-navsim-fix151-mechanism/RESULT.md` · `Frontier Scan/Daily/2026-09-17/RESULT.md` FS17-5

## 2026-09-18-01 — Ladder read in full: L1 is trivial for a 3DGS replay; L2 (envelope) is what binds row 14; T1 can never anchor L4

`FULL TEXT (local pypdf)` · arXiv **2607.07196**. Operational L1 = IEC via ACT-Estimator, significantly above 1/8 chance; instrument validated (0.77 m real ADE; 93.4 % vs 94.0 %; Vista 30.7 % reproduced). The paper **scopes OUT reconstruction simulators** (NeRF/3DGS). Epona L2 h* 3.2 s vs Vista 1.6 s at a 1.8 m band — both below our 6 s horizon. n = 2 models (conceded).
**Our position:** LR14-8's *"capped at L0"* branch does NOT fire; row 14 must target **L2-envelope** first and anchor L4 on **T2**, never T1 (`EVAL_DOCTRINE.md:29`). Experiment BE18-1 (`E-BE-ENVELOPE-1`). Leaderboard (C4): DriveZero-Scale 57.1 navhard two-stage; DriveVLA-M0 `2608.10413` 47.0 navhard.

## 2026-09-19-01 — Stamp-table additions; no new navhard entry above DriveZero-Scale

`FULL TEXT (local pypdf)` DeepSight **2605.10564**: Bench2Drive base, Think2Drive expert, 220 routes: **86.23 DS / 71.36 % SR** (w/o CoT 84.52 / 65.91); nuScenes open-loop avg L2 **0.33 m** (⛔ open loop, and a different L2 convention from our `fwd_ade`, per V-5). Inputs at inference include **ego state + target point** ⇒ **benchmark-arm only**.
`HTML via summariser` DriveVLA-M0 **2608.10413**: navhard **47.0 EPDMS, two-stage**; navtest **94.1 PDMS**; ego at inference.
**Verified today from the local PDF:** DriveZero **57.1 (Scale) / 51.5 (base)** are both navhard **two-stage** EPDMS; SimScale data: Stage 1 **−1.8**, Stage 2 **+8.3**.
⚠️ **E-A5:** no new SOTA after DriveZero-Scale. **D-14 (new):** Waymo's *Reference Driver* (2026-06) is an A5 claim about benchmarking AVs against humans and is unadjudicated.

### 2026-09-20-01 — a two-stamp catch: Drive-HWM's EPDMS 86.4 is **navtest**, not navhard ⛔⭐

The NAVSIM v2 table reports Drive-HWM **EPDMS 86.4**, but the same table carries an **`Ego Status`
blind row at 64.0** and **TransFuser at 76.7** — navtest magnitudes. In-PDF search for `navtest` and
`navhard` returns **0 and 0**: the split is **not stamped in the paper**.
⇒ **86.4 does NOT enter the navhard stamp table**, where **57.1** (DriveZero, two-stage) still stands.
This is FS18 row 32's two-stamp rule (split **and** fix) catching its **second** paper.
**Class:** MEASURED (our in-PDF probe) on a PUBLISHED table.
**Source:** Daily/2026-09-20/RESULT.md
