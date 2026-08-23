# F-18 SLOT PROBE — THE POWERED RE-RUN · lead-selected corpus, five-point trajectory

**Date:** 2026-08-17 · **Branch:** `agent/arch-inf-20260803` · **Agent:** slot-probe-parity
**Pre-registration:** `…/incoming/2026-08-16-agent-slot-decoder/AGENT_SLOT_DECODER.md` §1.4 / §1.4b —
followed, not redesigned.
**Cites, and does not touch:** `…/incoming/2026-08-16-slot-probe-run/SLOT_PROBE_RUN.md`.
**Eval tier:** ⛔ **T0-DIAGNOSTIC.** A frozen-latent readout is a world-model diagnostic and is
**never** driving performance.

---

## ⭐ THE ANSWER, IN ONE PARAGRAPH

**The negative HOLDS on data that can carry it, and it is now a sharper negative than 2026-08-16's.**
On **70 lead-carrying eval episodes** (vs 13) and **2 721 GT-lead windows** (vs 454), the agent-slot
readout **fails K1 at all five checkpoints and at all three seeds** — it never beats a constant — and
it fails the new attribution control **K5 at every point**, sitting **+2.1 to +4.0 m worse than
simply knowing which episode it is looking at**. The pre-registered `tokens` control fires **D1** at
this higher power: 640 raw patch tokens × 768 dims, **240× the `cells` surface**, still does not
rescue it, so the loss is at the **ENCODER, not the readout**. ⭐ **The sharpest form:** on a
window-matched random-latent null, **every checkpoint's error interval overlaps the noise arm's**
(5.949 [5.397, 6.543]) — the trained v6 latent serves this readout no better than random vectors of
the same shape. ⛔ **AND I HAD TO WITHDRAW MY OWN MOST QUOTABLE FINDING.** The five points read
5.98 → 5.44 → 5.58 → 6.15 → 7.17 m, which looked like *"agent structure is progressively
discarded"* — §1.4b's third outcome. Re-fitting **one** checkpoint at three seeds spans **1.83 m of
K1**, *more* than the 1.73 m spanned by the whole trajectory: the trend was fit noise. **What the
trajectory actually shows is no resolvable change at all.** ⚠️ Every number here is **EARLY-READ** at
**37.5 %** of training; 30 k remains the primary read and **the D1 DROP is NOT executed**.

---

## 0. ⛔ THE STAMPS THAT BIND EVERY NUMBER BELOW

1. **`v6F-SW-30k@<step>` — the checkpoint is part of the arm (§1.4b).** Every number carries it.
2. ⚠️ **EVERY POINT HERE IS AN EARLY-READ.** The primary read is **30 000**; the latest point here
   is **11 250 = 37.5 %** of training. §1.4b forbids quoting an earlier checkpoint as the headline.
   **Nothing here is the verdict, and the pre-registered D1 DROP is NOT executed** — that is the
   PI's call at 30 k.
3. **fp16 trunk at every point** (`fp16 -> fp32`, lossy) — the trainer's own snapshot layout. All
   five points go through the *identical* quantisation, so the trajectory is unaffected by it.
4. ⚠️ **LEAD-ENRICHED, NOT PARITY — §2.3.** The episodes are drawn *from* the canonical parity
   corpus by a declared content criterion, but the 130-clip subset **is not the parity episode
   set**, and `parity: False` is recorded in every cache meta.

---

## 1. What this run changes, and what it does not

The 2026-08-16 run returned the pre-registered **D1** (*the encoder does not carry agent geometry*)
and stated two limits. Both are now removed:

| limit, as stated 2026-08-16 | this run |
|---|---|
| ran on a **61-clip HF prefix**, clips chosen by **download convenience** | clips chosen by **lead-bearing content over the FULL 2 308-episode train join** (§2) |
| **13 lead-carrying eval episodes** — and the bootstrap CLUSTERS ON EPISODES, so 13 was the real n behind every interval | **70 lead-carrying eval episodes** (§2.2) |

⭐ **And two limits it did not state, both found here:** its own `n_slot_queries` fit was taken on a
sample **2.4× sparser** than a lead-probe actually works in (§2.4), and its **K2 could not have
separated whatever the head had learned** (§5.3). A third defect — **K3 cannot fail at this
operating point** — is found by the null control (§5.1), and a fourth — **the registered controls do
not cover episode identity** — is filed as a pre-registration amendment (§5.4).

---

## 2. The corpus — selected by CONTENT, declared before any result

### 2.1 The census

The train-corpus obstacle join was pulled from HF and **md5-verified against the published digest**
(`24cbdca8c3b23aafc2fb17e6bf99cf76`, matched — `raw/p1_pull_join.json`). Every joined frame was then
scored with **`sp2_probe.gt_lead_gap` itself** — the same function that later scores the metric, so
the selection predicate and the scoring predicate cannot drift apart:

| MEASURED over the full join (`raw/p2_lead_census.json`) | |
|---|---|
| records / clips | **433 040** / **2 308** |
| agent boxes | **12 122 129** |
| frames carrying a GT in-corridor lead (`cx>0`, \|`cy`\|≤1.75 m, `cx`≤30 m) | **84 607** |
| clips with **any** lead frame | **1 133** of 2 308 (49.1 %) |
| clips with **≥120** lead frames | **302** |

### 2.2 The declared selection (`code/p3_select.py`, `raw/p3_selection.json`)

**Fixed before any result existed and not revisited afterwards:**

* **stratum** — the **302** clips with `n_lead_frames ≥ 120`. A *stratum*, not the top-130: taking
  the extreme tail would define the sample by an order statistic, and its lead density would then
  be a sample of nothing.
* **sample** — **130** clips drawn uniformly without replacement, **seed 0**.
* **split** — **70 EVAL / 60 TRAIN** by the *same* seeded permutation, so both halves are draws
  from one distribution. Assigning the lead-richest clips to EVAL would have been a covariate shift
  dressed as a split.

| | 2026-08-16 | **this run** | factor |
|---|---|---|---|
| eval episodes | 20 | **70** | 3.5× |
| **eval episodes CARRYING A LEAD** ⇐ *the bootstrap's real n* | **13** | **70** | **5.4×** |
| eval windows carrying a GT lead | 454 (27.2 % of 1 669) | **2 721** (90.0 % of 3 023) | 6.0× |
| probe-train leads calibrating C-CONST | 140 | **2 231** | 15.9× |

*(the window counts are from the real stride-4 caches; the synthetic null cache enumerates the join
directly and lands on 3 089 of 3 468 at the same 89–90 % lead rate.)*

### 2.3 ⚠️ WHAT "PARITY" HONESTLY MEANS HERE — stated loudly, not buried

The episodes come from the canonical `physicalai-train-e438721ae894` split on HF, **but 130 of 2 376
is not the parity episode set.** The cache directory is deliberately given an **unregistered name**
(`slotprobe-lead130-w120-256x640cyl`) so the parity guard fires and `parity: False` is written into
every cache meta. Naming it the canonical key to silence the guard would be exactly the lie the
guard exists to catch.

⇒ **What improved over 2026-08-16 is not parity — it is POWER and PRINCIPLED SELECTION.** The
earlier subset was "whatever downloaded first"; this one is a seeded sample of a declared,
content-defined stratum of the whole corpus. **A reader who wants corpus-level prevalence cannot
get it from either.** This eval set is deliberately lead-**enriched**; the primary metric is only
*defined* on lead-bearing windows, so enrichment buys power for exactly the claim being made and
buys nothing else.

⚠️ **Unchanged and still binding:** probe-train and probe-eval are episode-disjoint, but **both are
WM-TRAIN clips.** For a *negative* that is conservative — it is the probe's best case. A *positive*
would need the val split, and the val-corpus join still does not exist.

### 2.4 ⭐ `n_slot_queries` RE-FITTED — and the stratum is much denser than the corpus

Prereg §2 requires this be fitted, never inherited. MEASURED over the 130 clips' own 25 790
labelled frames (`raw/p6_nq_fit.json`):

| population | mean | median | p90 | p95 | **p99** | max |
|---|---|---|---|---|---|---|
| all agents in the join | 52.96 | 40 | 108 | 132 | 218 | 256 |
| **in-grid** (0<`cx`≤60 m, \|`cy`\|≤16 m) | 11.61 | 8 | 24 | 35 | **73** | 94 |
| in-grid AND visible (`occ` 0) | 10.87 | 8 | 22 | 32 | 68 | 85 |

⇒ **`n_slot_queries = 74`** (in-grid p99 73, rounded up). Overflow **0.95 %** of frames, against
**5.97 %** at the inherited 32 and **19.29 %** at the prereg placeholder 16.

⭐ **THE FINDING INSIDE THE FIT: lead-bearing clips are DENSE-TRAFFIC clips.** The in-grid p99 is
**73** here, **33** corpus-wide, and **31** on the 2026-08-16 subset. So the earlier run's `32` was
fitted on a sample **2.4× sparser** than the frames a lead-probe actually has to work in. A
sensitivity arm at the inherited 32 is run anyway (§6), because a re-fit that changes the head
between runs is itself something to check rather than assert — and because it is the one change
that makes this run's numbers non-paired with 2026-08-16's.

---

## 3. What was run

| item | value | evidence |
|---|---|---|
| trunk | `v6F-SW-30k` @ **2000 · 9000 · 9250 · 10000 · 11250**, FROZEN | `_meta["step"]` of each artifact |
| memory surfaces | `cells` [16, 128] · `tokens` [640, 768] | `V6Stack.cells(z_op)` / `encode_window(..., return_tokens=True)` |
| head | `AgentSlotDecoder`, 74 queries — **3 222 293** params (cells) / **3 545 877** (tokens) | inside the prereg §6 2–4 M band |
| corpus | 130 clips, 25 790 labelled frames, 1 365 857 boxes | `raw/lead130_join_meta.json` |
| split | EVAL 70 clips / TRAIN 60 clips, episode-DISJOINT | `raw/p3_selection.json` |
| labels | the SAME join, byte-copied to the 130 clips — **line and box counts reproduce the census exactly** | `code/p5_filter_join.py` refuses on any mismatch |
| estimator | `taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2000 | ⛔ `overlapping_holdout_se` never imported |
| GPU | RTX 4060 — **Thor was never used for compute** | trainer PID 25477 alive throughout |

### 3.1 ⛔ Thor: read, never run — and every checkpoint verified twice

The two banked snapshots were **pulled**; the live point required a snapshot that did not exist, so
`code/thor_snap_live.py` ran **CPU-only, `mmap=True`**, one tensor at a time — the pattern measured
safe on 2026-08-16. It produced `v6F_sw_step011250.fp16.pt`. Thor's trainer **PID 25477 was alive
before and after** (`1-13:57` → `1-13:59` elapsed) and still training at the end of the run.

⭐ **Both verifications, because either alone is weaker than the pair:** the bytes are md5-checked
against the source, **and** the step is read from `_meta["step"]` inside the artifact rather than
from the filename — the §1.4b discipline that turned a single point into a trajectory last time,
when a file whose name and neighbouring logs said 8 800 turned out to be step **2 000**.

| snapshot | md5 (both sides) | `_meta["step"]` | filename agrees | tensors |
|---|---|---|---|---|
| `v6F_sw_step009250.fp16.pt` | `9574e1bc2dc38fb457d23088cfaaa2de` | **9250** | ✅ | 573 |
| `v6F_sw_step010000.fp16.pt` | `a4e2c0e1eb0ca455448472853ccc46d7` | **10000** | ✅ | 573 |
| `v6F_sw_step011250.fp16.pt` (made here) | `baa1545a399ae8e6c4948221783c0b93` | **11250** | ✅ | 573 |
| `weights_fp16_s9000.pt` (banked 2026-08-16) | `bd7762f768f3430eb1b3b6d37852b337` | **9000** | — | 573 |
| `weights_fp16.pt` (HF, the run's pod phase) | `fa1c34bc659a4b6c6526cf155d7f1775` | **2000** | ⚠️ **filename carries no step at all** | 573 |

⚠️ The last two md5s are **re-MEASURED here**, not inherited from the 2026-08-16 report — the
`bd7762…` digest happens to agree with what that report published, which is a check, not a citation.

### 3.1a ⚠️ TWO TRANSPORT FAILURES, NAMED — neither touched the result, both cost time

**(a) The HuggingFace client died silently at 124 of 130.** No traceback, no non-zero exit, the
process simply gone, leaving **six 0-byte `.incomplete` stubs** and six held lock files. It
reproduced on a clean resume: the same six clips, the same silent death, `.incomplete` never leaving
0 bytes. ⛔ **This is the shape the programme keeps getting caught by — a failure that looks like
"still running" rather than like an error.** The pull loop's own 4-attempt retry could not help,
because the process was not raising; it was dying. **Detection came from noticing the file count had
not moved, not from any error channel.**

**(b) A Thor-sourced helper was tried first and failed on 121 of 130 clips.** My error capture was
wrong (`exit=$?` read after an intervening `rm`, so every failure logged `exit=0`), so its cause was
never established while it ran.

⭐ **Resolution, and it corrects (b):** the six clips HF could not deliver were pulled **from Thor,
md5-verified per file, first attempt, no failures** — once the checkpoint transfers had finished and
the link was free (`code/p4c_last6.sh`, `raw/p4c_last6.log`). ⇒ **The Thor path is sound; the
earlier 121 failures were almost certainly link saturation, not a broken route.** I am stating that
as the likeliest reading with its evidence, not as a proven root cause.

MEASURED link facts for whoever needs them next: the dev box's egress caps at **2.44 MB/s aggregate
across all streams**; Thor wifi single-stream **2.27 MB/s**, Thor ethernet **1.79 MB/s**, HF
single-stream **2.08 MB/s**, HF 8-worker **~1.25 MB/s sustained**. ⇒ **Thor is the faster source for
episode data and it is a plain disk read — no GPU, no compute on the training box.**

### 3.2 The frozen-trunk proof — at the PRODUCTION geometry, and it can fail

`assert_isolation` with the head built at **74 queries**, both arms (`raw/sp0_isolation_nq74.json`):

| edge | violations | **n_probed** |
|---|---|---|
| `planner_to_encoder` | 0 | 151 |
| `tactical_to_below` | 0 | 342 |
| `strategic_to_below` | 0 | 427 |
| **`perception_to_trunk`** | **0** | **538** |

⭐ **The mis-wired control FIRES:** with `isolate_interp_from_encoder=False`, `perception_to_trunk`
reports **151 live forbidden edges** and raises `IsolationViolation`. A guard that cannot fail is
not a guard.
⚠️ And the probe is isolated more strongly than that edge: it trains on a **banked, detached**
latent tensor with no autograd graph, so no gradient path to the trunk exists even in principle.

---

## 4. ⭐ THE INSTRUMENT WAS PROVED EQUIVALENT BEFORE IT WAS BELIEVED

`sp2_probe.py` was edited here (a declared-split option, two new controls, two new diagnostics). An
edited instrument producing a different answer would be uninterpretable, so it was **re-run on the
2026-08-16 banked cache at that run's own settings — twice, once before the amendment of §5.4 and
once after** (`raw/results_REGRESSION_vs_20260816.json`, `raw/results_REGRESSION_post_amendment.json`):

| | 2026-08-16 published | pre-amendment re-run | **post-amendment re-run** |
|---|---|---|---|
| Δ `cells` − C-CONST | **+9.8401 [5.6855, 13.7524]** sep | +9.8401 [5.6855, 13.7524] sep | **+9.8401 [5.6855, 13.7524]** sep |
| Δ `cells` − C-SHUF | **−0.0569 [−2.0867, 1.7501]** not sep | −0.0569 [−2.0867, 1.7501] not sep | **−0.0569 [−2.0867, 1.7501]** not sep |
| K3 τ\*-gated recall | 0.3238 | 0.3238 | **0.3238** |
| median abs err | 17.22278 m | 17.22278 m | **17.22278 m** |

The training loss trace is **identical line for line** (step 0 `80.3456`, 200 `10.7594`, 400
`12.8752`, 600 `10.8874`). ⇒ **The edits are inert on the scoring path**, and any difference below
is a difference in the DATA, not in the code.

⭐ **And it surfaced the number the earlier report never printed:** that run's
`n_bootstrap_clusters` was **13**.

---

## 5. ⛔ THE PIPELINE NULL CONTROL — run FIRST, and it calibrates every criterion

Before any real latent, the identical scoring path was run on a **random-latent cache with the REAL
targets and the REAL declared split** (`raw/results_NULLCONTROL.json`). A probe on noise **must**
fail K1. It does — and because it is scored on exactly the windows the real arms are scored on, its
output is not merely a leak check: **it is the empirical FLOOR of every diagnostic in this report.**

| quantity | **random-latent floor** | what that establishes |
|---|---|---|
| **K1** Δ arm − C-CONST | **+1.1646 [0.6242, 1.6636]** separated ⛔ | ✅ no target→prediction leak: noise loses to the constant, so a passing K1 below cannot be an artefact of my scoring code |
| **K5** Δ arm − C-EPMEAN | **+3.0651 [2.3982, 3.7554]** separated ⛔ | ✅ the new attribution control also fails correctly on noise |
| **K2** Δ arm − C-SHUF | +0.0852 [−0.0009, 0.1787] not sep | ✅ the anti-echo control behaves on noise |
| **XEP** Δ arm − C-SHUF-XEP | +0.0087 [−0.1384, 0.1454] not sep | ✅ and the cross-episode permutation is **clean**: `c_shuf_xep_same_episode_windows = 0` — not one window kept its own episode |
| **K3** τ\*-gated recall | **0.5002** — ⛔ **PASSES the ≥0.50 criterion** | ⛔ **K3 IS VACUOUS HERE — see §5.1** |
| oracle-slot median | **0.790 m** | ⛔ **the oracle diagnostic is DEAD at 74 queries — see §5.2** |
| **C-CONST** own error | **5.173 [4.666, 5.779] m** | the constant is much stronger on this lead-rich set than on the 2026-08-16 sample (7.28 m) |
| ⭐ **C-EPMEAN** own error | **3.273 [2.905, 3.677] m** | ⛔ **THE REAL BAR.** Knowing only *which episode this is* gets you to 3.27 m. An arm that lands between 3.27 and 5.17 m has beaten the constant **without necessarily seeing an agent.** |
| clusters / windows | 70 / 3 089 | the null runs at the real power |

⭐ **THE SINGLE MOST USEFUL NUMBER IN THIS TABLE IS `C-EPMEAN = 3.273 m`.** It re-scales the whole
report: **beating C-CONST (5.17 m) is not the bar for an agent-perception claim on this data — 3.27 m
is**, because everything between the two is available from episode recognition alone. §5.4 explains
why that gap exists and why C-SHUF cannot police it.

### 5.1 ⛔ K3 CANNOT FAIL AT THIS OPERATING POINT — a C13 defect in the pre-registered criterion

τ\* is defined as *the median lead-slot presence of the `cells` arm over GT-lead windows*. **A
median puts ~50 % of windows above it by construction.** When the geometric emission rate is 1.0 —
which it is here, 74 slots make some slot in-corridor on every frame — and most predicted gaps fall
inside the 30 m stratum, the τ\*-gated recall is **pinned at ≈0.50 whatever the head has learned.**

MEASURED: a head trained on **pure noise** scores **0.5002**, i.e. it **PASSES `K3 ≥ 0.50`**.
Its C-SHUF twin scores 0.5028 and also "passes".

⇒ **K3 is reported below for completeness and is NOT used as evidence in either direction.** This
is the same family as the defect the 2026-08-16 run itself caught and fixed (recall over *any*
in-corridor slot was true 99.8 % of the time); the τ\*-gating fixed the 0.998 version and left a
0.500 version behind. It went unseen there because that run's sparser sample gave a geometric
emission rate below 1 and larger predicted gaps, so K3 landed at 0.29–0.32 **by luck, not by
construction.** ⛔ **A criterion whose threshold coincides with the definition of its own operating
point is not a criterion. K3 needs re-specification before it is quoted again** (§7).

### 5.2 ⛔ THE ORACLE DIAGNOSTIC IS DEAD AT 74 QUERIES — do not read it

The 2026-08-16 report warned that a head scattering slots widely earns a small oracle for free, and
estimated ~0.94 m from 32 slots by geometry alone. At the re-fitted **74** queries the effect is
now total: **random latents give an oracle median of 0.790 m.** Any real arm's oracle at or above
that is **at or below chance**. The oracle numbers are printed below with this floor attached and
are used for **nothing**.

### 5.3 ⭐ AND HOW MUCH CAN C-SHUF POSSIBLY SEE? — a control ON the control

New here, measured from the **labels only**, before any head exists: C-SHUF swaps memory between
windows of the *same* episode, so the most it can perturb is the within-episode variation of the GT
lead gap.

| | 2026-08-16 sample | this sample (real cache) |
|---|---|---|
| within-episode GT gap SD | 4.249 m | **3.688 m** |
| between-episode GT gap SD | 6.972 m | 6.200 m |
| **realised swap MAE** (the perturbation actually applied) | **4.768 m** | **3.670 m** |
| episodes contributing | 13 | **70** |

⇒ ⚠️ **The 2026-08-16 K2 reading was uninformative and should not have been weighed.** That run's
arm error was **17.2 m** against a realised swap of **4.8 m**: the control moved the question by
about a quarter of the arm's own error, over **13** clusters. "Δ ≈ 0, not separated" was the only
thing it could have said. **K2 is only readable once an arm's error is comparable to ~4 m.**

### 5.4 ⛔ PREREG AMENDMENT — the registered controls do not cover EPISODE IDENTITY

**Full amendment, with its date, reason and ordering evidence:
`PREREG_AMENDMENT_EPISODE_IDENTITY.md` (this package).** Summarised here because it changes how
every number in §6 must be read.

The numbers in §5.3 do more than weaken K2 — they expose a confound **§1.4's control list
(C-CONST / C-SHUF / C-TOK / C-V5F) does not cover.** The GT lead gap varies **3.855 m within** an
episode against **6.239 m between**. A head that does nothing but **RECOGNISE WHICH EPISODE it is
looking at** — trivial from appearance — can emit that episode's typical gap and beat the global
constant while never locating an agent. ⛔ **C-SHUF scores such a head IDENTICALLY**, because every
window it swaps in comes from the *same* episode and implies the *same* answer.

**MEASURED, so this is not a hypothetical:** on the 2026-08-16 windows, knowing only the episode's
own leave-one-out mean gap scores **3.899 m** against **7.283 m** for the global constant.

**Two controls were added — they rule out different things and neither substitutes for the other:**

| control | destroys | preserves | a null on it means |
|---|---|---|---|
| **C-SHUF** *(registered)* | window identity | **episode identity** | nothing varying *within* the episode is used. ⚠️ blind to episode identity |
| **C-SHUF-XEP** *(new)* | window **and** episode identity | — | the head does not read its input **at all** — pure prior |
| **C-EPMEAN** *(new)* | — (label-side **ORACLE**) | — | the **ceiling** on the episode-identity strategy: *how much* of a score it explains |

⭐ **The decomposition:** C-SHUF Δ≈0 **and** XEP Δ≈0 ⇒ pure prior. C-SHUF Δ≈0 **but** XEP Δ<0 ⇒ the
head reads its input only at **episode granularity** — scene recognition, not agent perception, and
C-EPMEAN then quantifies it. C-SHUF Δ<0 ⇒ window-to-window information is in use, the only regime in
which an agent claim is available at all.

> **K5 — an arm that beats C-CONST but NOT C-EPMEAN has shown nothing about agents.**

⚠️ **K5 is NOT pre-registered and NEVER gates KEEP** (that stays K1 ∧ K2 ∧ K3 as registered); it is
an **attribution** test, read only when K1 passes.

⚠️ **THIS IS NOT A RETRACTION OF 2026-08-16, and the ordering is checkable.** The amendment was
written **while the corpus was still downloading, with no arm yet fitted on it** — the only fits in
existence were the random-latent null and two re-runs of the 2026-08-16 cache. **The gap bites only
on a POSITIVE result:** episode-identity leakage can make a head look better than it is, never
worse, so it cannot manufacture a head that loses to a constant by 9.84 m. Re-scoring that run's own
cache with the new control confirms it directly — the 2026-08-16 head was **+13.224 [8.846, 17.314]
WORSE than the episode-identity ceiling**, separated. It was not exploiting episode identity; it was
not exploiting anything. **D1 as reported on 2026-08-16 stands.**

---

## 6. THE RESULT

**Primary: `lead_gap_abs_err_m`.** Paired episode-cluster bootstrap, n_boot 2000, identical windows.
**Positive Δ = the arm is WORSE than the control.** Every point: **70 bootstrap clusters, 2 721
paired windows, zero abstentions on every arm and every control.**

### 6.1 The trajectory, `cells`

Constants are identical at every point because they are label-side on the same windows:
**C-CONST 5.133 m · C-EPMEAN 3.122 m** (@2000 sees 2 702 windows and reads 5.142 / 3.125).

| stamp | arm err (m) | **K1** Δ vs C-CONST | **K5** Δ vs C-EPMEAN | **K2** Δ vs C-SHUF | Δ vs C-SHUF-XEP | median | K3 |
|---|---|---|---|---|---|---|---|
| `v6F-SW-30k@2000` | **5.980** [5.327, 6.727] | **+0.838** [+0.146, +1.523] sep ⛔ | **+2.855** [+2.043, +3.721] sep ⛔ | −0.056 [−0.133, +0.016] ns | −0.532 [−1.268, +0.236] ns | 5.440 | 0.4972 |
| `v6F-SW-30k@9000` | **5.442** [4.902, 6.036] | **+0.309** [+0.030, +0.579] sep ⛔ | **+2.320** [+1.648, +3.035] sep ⛔ | +0.007 [−0.057, +0.067] ns | +0.107 [−0.044, +0.257] ns | 5.262 | 0.5002 |
| `v6F-SW-30k@9250` | **5.580** [4.960, 6.274] | **+0.447** [+0.143, +0.759] sep ⛔ | **+2.459** [+1.706, +3.258] sep ⛔ | +0.028 [−0.056, +0.107] ns | +0.194 [−0.089, +0.498] ns | 4.510 | 0.5002 |
| `v6F-SW-30k@10000` | **6.153** [5.430, 6.961] | **+1.020** [+0.473, +1.637] sep ⛔ | **+3.031** [+2.151, +3.962] sep ⛔ | +0.041 [−0.082, +0.156] ns | −0.091 [−0.860, +0.610] ns | 5.298 | 0.5002 |
| `v6F-SW-30k@11250` | **7.169** [6.514, 7.892] | **+2.036** [+1.242, +2.848] sep ⛔ | **+4.047** [+3.268, +4.887] sep ⛔ | +0.105 [+0.051, +0.166] sep ⛔ | +0.122 [−0.201, +0.475] ns | 6.801 | 0.4969 |
| `@11250` **seed 1** | **6.026** [5.416, 6.647] | **+0.894** [+0.327, +1.467] sep ⛔ | — | — | — | 5.491 | — |
| **RANDOM-LATENT NULL, matched windows** | **5.949** [5.397, 6.543] | **+0.816** [+0.367, +1.264] sep ⛔ | **+2.827** [+2.145, +3.539] sep ⛔ | −0.046 [−0.134, +0.037] ns | +0.020 [−0.134, +0.171] ns | 5.534 | 0.5002 |

**VERDICT at every point: `DROP/RE-SCOPE`. K1 fails everywhere, K5 fails everywhere.**
⇒ **The pre-registered negative SURVIVES the 5.4× power increase and the content-based selection.**

⭐ **AND THE TRAJECTORY IS FLAT, NOT MONOTONE.** @2000's K1 (**+0.838**) sits *between* @9250's
(+0.447) and @10000's (+1.020). Across 2 000 → 11 250 the K1 range is **+0.31 … +2.04 m = 1.73 m**,
while re-fitting **one** checkpoint at a different seed moves it **1.14 m** (§6.3). There is **no
resolvable emergence and no resolvable decline** — of §1.4b's three outcomes, the measured one is
**"neither"**.

### 6.2 ⭐ BUT THE FAILURE HAS CHANGED CHARACTER — and the honest reading is not "same answer"

| | 2026-08-16 @9000 | **this run @9000** |
|---|---|---|
| arm error | 17.12 m | **5.44 m** |
| C-CONST | 7.28 m | **5.13 m** |
| K1 Δ | **+9.84 m** (2.4× the constant's own error) | **+0.31 m** (6 % of it) |
| bootstrap clusters | 13 | **70** |

⛔ **These two are NOT a paired comparison and I am not presenting them as one** — different corpus,
different window set, different `n_slot_queries` (32 → 74). What is admissible is the *shape*: on
2026-08-16 the head was **hopeless**; here it lands **within a metre of the constant** at @9000 and
is beaten by only 0.31 m. That is a different qualitative state, and it is the reason the extra
controls in §5 had to exist before this point was read.

### 6.3 ⛔ I DREW A TRAJECTORY CLAIM AND THE FIT-NOISE CONTROL REFUTED IT — read this before §6.1

The three points above ran **5.442 → 6.153 → 7.169 m**, K1 widening **+0.31 → +1.02 → +2.04 m**,
@9000 and @11250 not overlapping. That reads exactly like §1.4b's third outcome — *agent structure
is **progressively discarded** as the world model trains* — and it is the most quotable sentence
this run could have produced. **I wrote it, then measured whether a probe fit is even repeatable.**

⛔ **IT IS NOT. Re-fitting the SAME @11250 cache at a different seed moves the headline by more than
the whole trajectory does:**

| @11250, identical cache, identical windows, only the seed differs | arm err (m) | **K1 Δ vs C-CONST** | median |
|---|---|---|---|
| seed 0 | 7.169 [6.514, 7.892] | **+2.036** [+1.242, +2.848] sep | 6.801 |
| seed 1 | 6.026 [5.416, 6.647] | **+0.894** [+0.327, +1.467] sep | 5.491 |
| seed 2 | 5.343 [4.821, 5.924] | **+0.210** [+0.044, +0.413] sep | 4.915 |

| | K1 range | K1 sd |
|---|---|---|
| **SEED-only**, one checkpoint (n=3) | **1.826 m** | 0.922 |
| **CHECKPOINT-only**, five points at seed 0 | **1.727 m** | 0.682 |
| **ratio** | **1.06×** | — |

⛔ **THE SEED SPREAD IS LARGER THAN THE CHECKPOINT SPREAD.** Re-fitting one frozen cache with a
different optimiser seed moves the headline slightly *more* than walking 9 250 training steps of the
world model does. ⇒ **The "progressive degradation" reading is WITHDRAWN.** Of §1.4b's three
outcomes — emerges, plateaus, is progressively discarded — what this run supports over
2 000 → 11 250 is **none of them resolvably**: the trajectory is flat within fit noise.

⚠️ **The bootstrap CI cannot catch this, and it is important to see why.** The episode-cluster
bootstrap resamples EVAL EPISODES; it estimates the uncertainty of *this fitted head's* score. It
says nothing about the variability of **which head the fit produces**. Three seeds here give three
tight, mutually non-overlapping intervals (+2.036 [1.242,2.848], +0.894 [0.327,1.467], +0.210
[0.044,0.413]) that differ **only** by the seed. **A tight interval is not a reproducible number**,
and every per-point interval in §6.1 must be read as an interval on *that head*, never on the
checkpoint.

⭐ **THE CORE NEGATIVE IS UNAFFECTED — which is exactly why the two had to be separated.** K1 **fails
at all three seeds and at all five checkpoints**, always positive, always separated. **"The head does
not beat a constant" is robust. "It gets worse with training" was an artefact of a single fit.**

⚠️ **The bootstrap CI cannot catch this and it is important to see why.** The episode-cluster
bootstrap resamples EVAL EPISODES; it estimates the uncertainty of *this fitted head's* score. It
says nothing about the variability of **which head you get** from the fit. Two intervals can be
tight and non-overlapping while the difference between them is entirely the optimiser's seed. Every
per-point interval in §6.1 is correct as an estimate of its own head's error and **must not be read
as an interval on the checkpoint.**

⭐ **THE CORE NEGATIVE IS UNAFFECTED, AND THAT IS THE POINT OF SEPARATING THEM.** K1 **fails at both
seeds** (+2.036 and +0.894, both separated, both positive) and at every checkpoint. **"The head does
not beat a constant" is robust to the seed. "It gets worse with training" was not.**

### 6.4 ⛔ WHAT I DID NOT LET MYSELF CONCLUDE UNTIL IT WAS CONTROLLED

1. ✅ **Fit noise** — measured, and it cost me the trajectory claim (above).
2. ✅ **The window-matched null** — and it produces the cleanest statement in this report (§6.4a).
3. ✅ **All five checkpoints** — §6.1. ✅ **The `tokens` arm** — §6.4b.
4. ✅ **`n_slot_queries` sensitivity.** The fit changed the head between this run and 2026-08-16
   (32 → 74), so the inherited value was re-run on the **same @11250 cache**:

   | @11250, seed 0 | head params | err (m) | **K1 Δ vs C-CONST** | **K5 Δ vs C-EPMEAN** |
   |---|---|---|---|---|
   | `n_slot_queries` = **32** (inherited) | 3 211 541 | 6.625 [5.858, 7.463] | **+1.492** [+0.714, +2.239] sep ⛔ | **+3.503** sep ⛔ |
   | `n_slot_queries` = **74** (fitted) | 3 222 293 | 7.169 [6.514, 7.892] | **+2.036** [+1.242, +2.848] sep ⛔ | **+4.047** sep ⛔ |

   ⇒ **The verdict is insensitive to the choice.** Both fail K1 and K5; the 0.544 m difference
   between them is well inside the **1.826 m** seed spread, so it is not attributable to the head
   geometry either.

### 6.4a ⭐ THE SHARPEST FORM OF THE NEGATIVE — the latent is indistinguishable from NOISE

§6.1's unmatched null compared 2 721 windows against 3 089 — not a paired set. So a null was built
by replacing `cells` with random vectors **inside the real cache**, keeping every window, target,
clip and the declared split (`code/pA_null_matched.py`). On the **identical 2 721 windows / 70
clusters**:

| arm | `lead_gap_abs_err_m` | overlaps the null? |
|---|---|---|
| **RANDOM-LATENT NULL (matched)** | **5.949 [5.397, 6.543]** | — |
| `v6F-SW-30k@9000` | 5.442 [4.902, 6.036] | ✅ yes |
| `v6F-SW-30k@10000` | 6.153 [5.430, 6.961] | ✅ yes |
| `v6F-SW-30k@11250` seed 1 | 6.026 [5.416, 6.647] | ✅ yes |
| `v6F-SW-30k@11250` seed 0 | 7.169 [6.514, 7.892] | ⚠️ touches at the edge (6.514 vs 6.543) |

⇒ ⛔ **AT EVERY CHECKPOINT THE TRAINED v6 LATENT SERVES THIS READOUT NO BETTER THAN RANDOM VECTORS
OF THE SAME SHAPE.** That is a stronger and more useful statement than "worse than a constant",
because it removes the obvious rejoinder *"the constant is just a strong baseline on lead-rich
data"* — the noise arm faces the identical baseline on the identical windows and does the same.

⚠️ **Stated precisely: these are OVERLAPPING MARGINAL intervals, not a paired test.** The arm and the
null come from separate `sp2` runs, and `sp2` does not bank per-window error arrays, so a paired
bootstrap between them is not computable from what exists. The admissible claim is
**"indistinguishable at this power"**, not "identical". ⭐ **Banking the per-window arrays would make
it a paired test and costs nothing — that is a work item, §8.**

⚠️ And the matched null reproduces every §5 pathology on the real windows: **K3 = 0.5002 (passes,
vacuously)**, oracle median **0.916** (chance), K2 and XEP both unseparated. It fails K1 by
**+0.816** and K5 by **+2.827** — i.e. *noise* sits inside the same band the real arms occupy.

### 6.4b ⛔ THE `tokens` CONTROL — **D1 FIRES**, and it is not D2

Both arms from **one cache**, so `cells` and `tokens` are on **identical windows** (stride 8 —
1 362 paired windows, 70 clusters, `raw/results_tok11250.json`). This is the pre-registered arm
PAIR, not a choice: `cells` failing while `tokens` passed would have been a **positive** finding
about aggregation.

| | `cells` [16, 128] | `tokens` [640, 768] | C-CONST | C-EPMEAN |
|---|---|---|---|---|
| head params | 3 222 293 | 3 545 877 | — | — |
| `lead_gap_abs_err_m` | 5.751 [5.145, 6.388] | **5.327** [4.726, 5.986] | **5.119** | **3.184** |
| median | 5.397 | 4.604 | — | — |
| **K1** Δ vs C-CONST | **+0.632** [+0.110, +1.144] sep ⛔ | **+0.208** [−0.038, +0.491] **not sep** ⛔ | — | — |
| **K5** Δ vs C-EPMEAN | **+2.567** [+1.893, +3.313] sep ⛔ | **+2.143** [+1.386, +2.967] sep ⛔ | — | — |
| K2 Δ vs C-SHUF | +0.042 [−0.041, +0.127] ns | −0.042 [−0.111, +0.018] ns | — | — |
| Δ vs C-SHUF-XEP | −0.199 [−0.700, +0.272] ns | +0.138 [−0.154, +0.469] ns | — | — |
| oracle median | 0.792 (chance) | 0.759 (chance) | — | — |

⇒ ⛔ **`d_rule` = "D1 — K1 fails on BOTH arms: the ENCODER does not carry agent geometry."**
Handing the head the encoder's full **640 raw patch tokens at 768 dims — 240× the `cells` surface —
does not rescue it.** The readout grid is **not** the bottleneck. This is the same call the
2026-08-16 run made, now at **70 clusters instead of 13** and on content-selected windows.

⭐ **The one honest nuance, and it is in `tokens`' favour:** `tokens` is the only arm anywhere in
this run whose K1 is **not separated** — it *ties* the constant rather than losing to it (+0.208
[−0.038, +0.491]). ⚠️ **Tying a constant is not evidence of agent geometry**, and the attribution
control settles it: `tokens` is **+2.143 m worse than simply knowing which episode it is**,
separated. A surface that cannot beat episode identity has not demonstrated perception.

### 6.5 The diagnostics — and the oracle is at chance, as §5.2 predicted

| stamp | mean predicted gap | mean GT gap | oracle median | vs null floor **0.790** |
|---|---|---|---|---|
| @9000 | 16.754 | 15.529 | 0.765 | **at chance** |
| @10000 | 16.906 | 15.529 | 0.937 | **at chance** |
| @11250 | 20.851 | 15.529 | 0.809 | **at chance** |

⭐ The mean predicted gap tracks the truth closely at @9000/@10000 (16.8 / 16.9 vs 15.5) and then
**drifts away at @11250 (20.9)** — the same degradation the headline shows, visible in the head's
central tendency rather than only in its error.

---

## 7. The FOUR METRIC FAMILIES

⛔ ADE is not reported: this is a perception readout, not a trajectory eval. Per family, with the
reason where it does not apply:

| family | served? | how / why not |
|---|---|---|
| **LONGITUDINAL** | ✅ **directly — this is what the head is for** | **headway** = the lead slot's `cx`, which IS the primary of §6, with its estimator and CI. **Time-gap**: GT mean **19.252 s** over 2 721 windows. **TTC** under a **0.5 m/s physical closing floor** (§5 of the 2026-08-16 run's defect list) — see below. ⚠️ `taniteval.lead_metrics.distance_keeping` is **NOT** attached: it consumes a predicted ego **path** `(W,K,2)`, and this is a single-frame readout that produces no path. That is the T1 integration prereg §4.6 defers — stated, not silently skipped. |
| **LATERAL** | ❌ not served | the head emits **agent** geometry, not ego path; heading / curvature / yaw-rate / cross-track are ego quantities this readout never produces. |
| **TACTICAL** | ⚠️ **enabling condition only — NOT COMPUTED** | slot-referent agreement needs a **trained goal head** with the categorical `agent_slot` arg on. S-T has not run, so `GAP_TARGET` / `YIELD_AT` / `WAIT_FOR_ONCOMING` / `EVADE_IN_CORRIDOR` still index an empty set. A follow-on, per prereg §3. |
| **STRATEGIC** | ❌ not served | `obstacle.offline` has **no map, lane graph, junction, traffic-light or route feature** — 10 classes, all dynamic agents. Nothing strategic is derivable from this label at all. |

### 7.1 LONGITUDINAL — `lead_ttc_abs_err_s`, and it points the same way as the headline

Reported only where **both** the GT lead and the predicted lead are closing faster than 0.5 m/s, with
its own n (a receding lead has no finite TTC):

| arm | @9000 | @11250 | **matched random-latent null** |
|---|---|---|---|
| `cells` | 13.945 s [11.676, 16.216] · n=272 | **15.198 s** [12.963, 17.631] · n=415 | **9.462 s** [8.216, 10.953] · n=475 |
| its C-SHUF twin | 13.825 s · n=276 | 14.857 s · n=374 | 9.171 s · n=461 |

⇒ The vision-only lead's time-to-collision is wrong by **~14–15 s** — unusable — and the **noise arm
is BETTER (9.5 s)** than the trained latent, the same ordering §6.4a found on headway.
⚠️ **C-CONST and C-EPMEAN report `UNAVAILABLE`, n = 0, with the reason:** their closing rate is the
probe-train median, which does not clear the 0.5 m/s floor, so TTC is undefined for them rather than
silently large. Per family, with the reason and the n — which is the rule.

---

## 8. ⛔ ESCALATIONS — integration decisions this run needs

1. ⛔⭐ **A SINGLE PROBE FIT IS NOT A MEASUREMENT — AND NOTHING IN THIS INSTRUMENT SAID SO (§6.3).**
   Three seeds on one frozen cache span **1.826 m of K1**, *larger* than the **1.727 m** spanned by
   five checkpoints across 9 250 training steps. Every F-18 number ever published — 2026-08-16's
   included — is a **single-seed** fit whose reproducibility was never measured, and the
   episode-cluster bootstrap **cannot detect this** because it resamples eval episodes, not fits.
   ⇒ **Any future slot-probe comparison across checkpoints or arms must report ≥3 seeds and compare
   the between-condition spread against the between-seed spread.** It costs one extra fit per point
   (~13 min, no trunk compute). **This is the highest-value fix in this report** — it is a class of
   error, not a number: it would silently manufacture a trend in any frozen-latent probe study.
   ⚠️ It does **not** touch the core negative, which is robust at every seed and every checkpoint.
2. ⛔⭐ **K3 IS VACUOUS AS SPECIFIED AND MUST BE RE-SPECIFIED BEFORE IT IS QUOTED AGAIN (§5.1).**
   τ\* is *the median presence over GT-lead windows*, so the τ\*-gated recall is pinned at ≈0.50 by
   construction; **a head trained on pure noise scores 0.5002 and PASSES `K3 ≥ 0.50`.** This is the
   third instance of the same family in this instrument's short life (the 0.998 any-slot version
   the 2026-08-16 run caught and fixed; the ε-based TTC it also caught; now the τ\*-median). A fix
   needs a threshold that does not coincide with the operating point — e.g. τ\* set on the
   **random-latent null** rather than on the arm, or recall reported as a **PR curve with the null
   floor drawn on it**. **Owner: whoever holds `AGENT_SLOT_DECODER.md` §4.4.**
3. ⛔ **THE PRE-REGISTRATION'S CONTROL LIST IS INCOMPLETE — amendment filed, not silently applied.**
   §1.4 / §4.3 do not cover **episode identity**, and C-SHUF is structurally blind to it (§5.4).
   `PREREG_AMENDMENT_EPISODE_IDENTITY.md` in this package records **C-EPMEAN** and **C-SHUF-XEP**
   with the date, the reason, and the evidence that it was written before any arm on this corpus was
   fitted. ⛔ **I have NOT edited `AGENT_SLOT_DECODER.md`** — it is another stream's document, and
   folding the amendment into it is an integrator decision.
4. ⚠️ **THE ORACLE DIAGNOSTIC IS DEAD AT 74 QUERIES (§5.2)** — random latents give an oracle median
   of **0.790 m**. The 2026-08-16 report's own warning has become total. Anyone re-running F-18 must
   print the null floor beside the oracle or drop the column.
5. ⭐ **THE STEP-STAMPED CHECKPOINT REQUEST FROM 2026-08-16 WAS ACTED ON, AND IT IS WHAT MADE THIS
   TRAJECTORY POSSIBLE.** `tanitad-thor-wifi:~/ckpt_snaps/` now holds periodic 0.67 GB fp16
   snapshots. **Keep doing it** — this run consumed exactly those, plus one live snapshot it made
   itself, at zero GPU cost to the training run.
6. ⚠️ **30 000 IS STILL THE PRIMARY READ AND THIS IS NOT IT.** Every point here is EARLY-READ; the
   latest is 37.5 % trained. **The pre-registered D1 DROP is NOT executed** — that is the PI's call
   at 30 k, and the re-run costs ~30 min per point once that checkpoint exists.
7. ⚠️ **C-V5F STILL NOT RUN** (named on 2026-08-16, still open). It is the control that says whether
   O2/O3/O4 bought any agent content relative to v5f. Needs one 3.5 GB pull plus one cache pass.
8. ⚠️ **A POSITIVE RESULT WOULD NEED THE VAL SPLIT, WHICH STILL HAS NO JOIN.** Probe-train and
   probe-eval are episode-disjoint but **both are WM-TRAIN clips**. That is conservative for a
   negative and inadmissible for a positive. The val-corpus `obstacle.offline` join has still not
   been built.
9. ⚠️ **`n_slot_queries`: 32 → 74 on lead-bearing data** (§2.4). The lead-bearing stratum is
   **2.4× denser** than the 2026-08-16 sample (in-grid p99 73 vs 31). Anyone re-running F-18 on
   lead-selected windows must re-fit rather than inherit either number.

---

## 9. Test state — my delta is ZERO

⛔ **This run changed NO repository code.** Every script it added lives under
`…/2026-08-17-slot-probe-parity/code/`; nothing under `stack/` or `taniteval/` was edited.

MEASURED with the named interpreter, both suites:

| suite | result | baseline in the brief |
|---|---|---|
| `stack` | **3782 passed · 0 failed · 7 skipped · 2 xfailed** (416.09 s) | 3782 / 0 / 7 / 2 ✅ |
| `taniteval` | **1092 passed · 0 failed** (121.72 s) | 1092 / 0 ✅ |

Both match exactly. Artifacts: `raw/suite_stack.txt`, `raw/suite_taniteval.txt`.

---

## 10. Deliverable manifest

| artifact | where | only one place? |
|---|---|---|
| this report | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-17-slot-probe-parity/SLOT_PROBE_PARITY.md` | staged |
| ⭐ **prereg amendment** (episode identity) | `repo:…/2026-08-17-slot-probe-parity/PREREG_AMENDMENT_EPISODE_IDENTITY.md` | staged |
| full-join lead census (2 308 clips) | `repo:…/raw/p2_lead_census.json` | staged |
| **declared selection** (stratum, seed, 70/60 split) | `repo:…/raw/p3_selection.json` | staged |
| `n_slot_queries` re-fit | `repo:…/raw/p6_nq_fit.json` | staged |
| join pull + md5 verification | `repo:…/raw/p1_pull_join.json` | staged |
| filtered-join integrity check | `repo:…/raw/lead130_join_meta.json` | staged |
| isolation proof @74 queries + mis-wired control | `repo:…/raw/sp0_isolation_nq74.json` | staged |
| instrument equivalence, pre-amendment | `repo:…/raw/results_REGRESSION_vs_20260816.json` | staged |
| instrument equivalence, post-amendment | `repo:…/raw/results_REGRESSION_post_amendment.json` | staged |
| pipeline null control (the empirical floors) | `repo:…/raw/results_NULLCONTROL.json` | staged |
| ⭐ **window-matched null** (§6.4a) | `repo:…/raw/results_NULLMATCHED.json` | staged |
| ⭐ **fit-noise control, 3 seeds** (§6.3) | `repo:…/raw/results_s11250{,_seed1,_seed2}.json` | staged |
| `n_slot_queries` sensitivity | `repo:…/raw/results_s11250_nq32.json` | staged |
| the `tokens` arm, matched windows | `repo:…/raw/results_tok11250.json` | staged |
| the five trajectory points + cache metas | `repo:…/raw/results_s0{2000,9000,9250}.json`, `results_s1{0000,1250}.json`, `raw/cache_meta_*.json` | staged |
| rendered tables (generated from the JSON, not transcribed) | `repo:…/raw/RENDER_TABLES.md`, `raw/SUMMARY.json` | staged |
| the last-6-clip transport recovery log | `repo:…/raw/p4c_last6.log` | staged |
| roll-up | `repo:…/raw/SUMMARY.json` | staged |
| suites | `repo:…/raw/suite_stack.txt`, `raw/suite_taniteval.txt` | staged |
| code (pullers, census, selection, probe, chains) | `repo:…/code/*.py`, `code/*.sh` | staged |
| live-checkpoint snapshotter | `repo:…/code/thor_snap_live.py` **and** `tanitad-thor-wifi:~/thor_snap_live.py` | staged |
| **fp16 snapshot @11250** (0.67 GB) | `tanitad-thor-wifi:~/ckpt_snaps/v6F_sw_step011250.fp16.pt` **and** `<scratch>/sp2/ck/` | ⚠️ **NOT in the repo — too large.** Reproducible from Thor's `ckpt.pt` via `code/thor_snap_live.py`; md5 verified both sides |
| fp16 snapshots @9250 / @10000 (0.67 GB each) | `tanitad-thor-wifi:~/ckpt_snaps/` **and** `<scratch>/sp2/ck/` | ⚠️ pre-existing on Thor; md5 verified on pull |
| the 130-clip episode cache (4.9 GB) | `<scratch>/sp2/cache/slotprobe-lead130-w120-256x640cyl/` | ⚠️ **scratch only** — regenerable from HF + `raw/p3_selection.json` via `code/p4_pull_eps.py` |
| filtered join (130 clips, 25 790 lines) | `<scratch>/sp2/lead130_agents.jsonl` | ⚠️ **scratch only** — regenerable via `code/p5_filter_join.py`; its integrity meta IS staged |
| latent caches (5 × cells, 1 × tokens) | `<scratch>/sp2/cache_*/` | ⚠️ **scratch only** — regenerable, ~15 min each |
| trained probe heads | `<scratch>/sp2/out_*/head_*.pt` | ⚠️ **scratch only** — the probe is the disposable part; the cache and the result are what matter |
