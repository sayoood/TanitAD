# R5 — Strategy & Research-Management Review (independent)

**Reviewer role:** independent research-strategy consultant (external chief-scientist lens).
**Scope:** strategy, prioritization, resource allocation, the ≥5-works goal, dataset bet, agent-loop
management. **Not** an operational/ops-correctness review.
**Date:** 2026-07-25 · Phase 0 ~day 21/42 · **Read-only** (no pods, no compute, no git-add).
**Evidence class** on every load-bearing number: `MEASURED` (ours+artifact) · `PUBLISHED` · `INHERITED`
· `ESTIMATED` · `HYPOTHESIS`. Sources: `Mission Plan.md`, `Master Plan.md`, `PROGRAM_OVERVIEW.md`,
`LOOP_STATE.md`, `MODEL_REGISTRY.md`, `RETRACTION_LOG.md`, `RESOURCE_LEDGER.md`, the Research Hub tree,
`TANITDATASET_TIER_INTEGRATION_2026-07-21.md`.

---

## 1. Executive verdict (blunt)

**The bet is coherent and the execution discipline is genuinely world-class — but the program is
mis-prioritized against its own goal, and the hero is currently being beaten by its own understudy.**

Three facts frame everything:

1. **The flagship 4-brain WM loses the one contest that matters.** Open-loop it *ties* two reference
   diffusion arms (`MEASURED`: flagship 0.4522, REF-C-base **104 M** 0.4523, REF-C-XL 0.458 — a
   three-way tie no paired test orders). Closed-loop, measured properly for the first time at n=40 on a
   low-OOD instrument, the deployed flagship head scores **1.488 vs REF-C-base 0.564** (`MEASURED`,
   paired, CI-separated). A 104 M "reference baseline" is the best driver in the program. That is not a
   footnote — it is a challenge to the entire "4-brain is the core innovation" thesis.

2. **The one headline the program actually needs is unmeasured.** The C2 data-efficiency *slope* — "we
   need magnitudes less data" — is, in the program's own words, "the one headline claim still entirely
   unmeasured" (`INHERITED`, PROGRAM_OVERVIEW §8). Everything shipped so far is either a trivial-floor
   open-loop win, a clean *negative*, an instrument, or a *directional* pilot.

3. **The single build that would unblock half the thesis is owned by nobody.** A lower-OOD reactive-agent
   renderer gates: safety-grade closed-loop, D5/D6, the renderer-half of the beyond-ADE suite, and any
   NAVSIM/Bench2Drive entry (i.e. *any opponent-facing number*). It is named repeatedly as "the largest
   remaining build on the board" — and no pod is building it. The loop keeps running cheap experiments
   *around* it.

The program is not failing. It is producing an unusually honest, well-instrumented body of *solid and
negative* results. But it is optimizing for **rigor and breadth** when its stated goal demands
**one converging, external-facing, world-beating positive**. Left uncorrected, Phase 0 exits with a
beautiful measurement apparatus, five publishable-*incremental* results, and no defensible "we beat
anyone" claim.

**Strategic-coherence grade: C+.** **Progress toward the literal vision (beat Waymo/Wayve/Pony/Momenta/
Autobrains): ~5%. Toward the re-scoped research thesis (data-efficient, honestly-measured hierarchical
WM): ~30–35%, with the decisive headline still at zero.**

---

## 2. Findings (evidence · path · severity · confidence)

### F1 — The hero-vs-baseline inversion is the central strategic problem. **[SEV: HIGH · CONF: HIGH]**
`MEASURED` (`MODEL_REGISTRY §6`, PROGRAM_OVERVIEW §5.1/§5.2d). The flagship's *supervised head* is a lossy
readout (3.38 m open-loop, worse than constant-velocity's 0.825) of a good world model whose *rollout* is
0.452. The program has correctly diagnosed this ("heads are a lossy readout") and pivoted to planners —
but the best planner it has (REF-C anchored diffusion) is nominally a *reference arm*, and it wins
closed-loop. **Consequence:** the "4-brain hierarchy is the core innovation" claim (H1, the Mission
Plan's headline edge) is not yet supported by the flagship out-driving anything non-trivial; the actual
positive results cluster around (a) the WM as a *differentiable simulator* and (b) anchored diffusion as a
*planner*. The narrative and the evidence point at different heroes.

### F2 — The program's output is dominated by negatives, instruments, and retractions. **[SEV: HIGH · CONF: MED-HIGH]**
`MEASURED` (RETRACTION_LOG; PROGRAM_OVERVIEW §7). This round's settled results: H4 closed *negative*
(frozen encoder); Branch-B camera-conditioning *refuted*; INT8 *rejected*; frozen-WM planner is a
*fallback not a contender*; recovery-augmentation *not promotable*. These are honest and valuable — two
(frozen-encoder ceiling, camera-conditioning refutation) are publishable in their own right. But they are
*subtractive*. The additive, external-facing wins (C2 slope; a benchmark number; closed-loop competence)
are all still pending. A research program is judged on its positives; this one is currently long negatives
and short positives.

### F3 — Premature-certainty churn is the dominant *management* failure mode, and it is expensive in the binding resource. **[SEV: HIGH · CONF: HIGH]**
`MEASURED` (RETRACTION_LOG, 07-24/07-25). **Five "this direction is closed/bound/resolved" claims were
reopened by a ~$0 follow-up in a single session** ("canary descent confirmed" C5 · "planner is the
headroom" C6 · "closed-loop needs a renderer" C3 · "closed-loop closed" C3 · "closed-loop BOUND" C6). The
countermeasure is now doctrine ("run the cheapest reopening check *before* declaring closure"). **But the
self-correction machinery working is not the same as the churn being cheap.** The binding resource here is
**not GPU-dollars** (see F5) — it is the *single agent loop's attention and the PI's decision bandwidth*.
Every assert→retract cycle, every stale headline that survives in a header for four days (the "v1.6 best
in program" C4 propagation), every 472-line drumbeat re-derivation consumes exactly that resource. The
loop generates a great deal of motion that reads as progress.

### F4 — The critical unblocking build is perpetually deferred. **[SEV: HIGH · CONF: HIGH]**
`INHERITED`→`MEASURED-absence` (PROGRAM_OVERVIEW §8 item 3, §7 constraint 4; LOOP_STATE Auth-2 "the
low-OOD-vs-safety gap is ~fundamental"). The reactive-agent renderer is the acknowledged single
dependency for the entire safety/closed-loop half of the thesis, yet the fleet shows 0 pods on it and the
standing plan defers it ("this is a build; the cheap experiments around it are exhausted"). The program
has *proven the cheap experiments are exhausted* and *still not started the expensive one*. This is the
clearest opportunity-cost failure on the board: months of Phase-0 runway are being spent on corpus
balancing and geometry calibration while the item that gates every opponent-facing number waits.

### F5 — Compute is not the bottleneck; attention and PI decisions are. **[SEV: MED · CONF: HIGH]**
`MEASURED` (RESOURCE_LEDGER: **$0 settled / $120 planned**, ≤$50/wk guardrail; 4 pods, 0 idle). The
resource ledger is almost comically underspent for the ambition. The real scarce resources are (a) the
orchestrator loop's synthesis bandwidth and (b) the queue of "**Sayed's click**" decisions (HF cleanup
~13 GB, YouTube licensing, kill-v4.1-or-bank-WM, the renderer commit). High-value moves repeatedly stall
on a PI decision while the loop fills idle time with *more* cheap experiments (and more retractions).
**The program is compute-rich and decision-throughput-poor** — the opposite of what the "limited
resources = our strength" constitution assumes.

### F6 — Single points of failure around state and checkpoints. **[SEV: MED · CONF: HIGH]**
`MEASURED` (LOOP_STATE, MEMORY). The live 3.2 GB flagship checkpoint exists on **one pod disk** with HF
backup **403-blocked** (mid-checkpoint-loss risk, explicitly flagged). A connection loss (ENOTFOUND)
killed 3 agents mid-report this session. Cron re-arms "silently break"; monitor/notification delivery has
gaps. `LOOP_STATE.md` has grown to **472 lines / ~58 k tokens** of dense self-referential drumbeat — it is
now itself a liability that *breeds* the stale-headline propagation of F3 (a header out-lived its own
retraction by 4 days). The state artifact has outgrown the reader.

### F7 — The publishable record and the measured record sit on different data. **[SEV: MED-HIGH · CONF: HIGH]**
`MEASURED` (MODEL_REGISTRY §0.1; Master Plan §5 risk row). Every headline model number is on NVIDIA
**PhysicalAI-AV**, which is license-firewalled and "never in public claims." The *publishable* commercial
tier (`C` = comma2k19/L2D, MIT) has **no model trained at flagship scale on it**. So the flagship's 0.452
cannot be shown externally as-is; a clean-licensed re-derivation has not been done and is not scheduled
until Phase 2. The program's strongest evidence is on a corpus it cannot cite.

### F8 — "≥5 world-class works" has no definition, criteria, or acceptance test. **[SEV: MED · CONF: MED]**
`MEASURED-absence` (grep across `Project Steering` + hub: the phrase appears only as a LOOP_STATE
paraphrase — "the ≥5 works correction" — never as a criteria-defined target). A goal with no definition of
"world-class" and no per-work acceptance test cannot be steered toward or measured against. This is a
governance gap: the program cannot tell itself whether it is on track because it never said what the
finish line is. (The task brief supplies the goal; the *program* has not operationalized it.)

### F9 — Where discipline is genuinely excellent (counter-weight, so the above is calibrated). **[CONF: HIGH]**
`MEASURED`. The evidence-class doctrine, the `RETRACTION_LOG` root-cause taxonomy, TanitEval's
episode-cluster bootstrap default, the **hard-refusal-in-code of a 78%-leak val split**, "never quote an
interval without its estimator / an exponent without R²+n+window" — this is a level of measurement honesty
that most funded AV labs do not reach. It is a real moat and it is why the negatives above are *trustable*.
The critique in F1–F8 is about *what the discipline is pointed at*, not the discipline itself.

---

## 3. The ≥5-works scorecard

Rating = honest external read of whether the artifact is on a trajectory to be **world-class /
beat-published-SOTA** (WC), **solid & publishable but incremental** (SI), or **enabling infrastructure /
not a standalone work** (INFRA). "On track?" answers *to world-class specifically*.

| # | Candidate work | On track to **world-class**? | Est. level today | Basis (evidence class) |
|---|---|:--:|---|---|
| 1 | **Flagship 4-brain latent WM** (the hero driver) | **No** | **SI** — WM-as-differentiable-simulator is a genuine positive (3.77 M planner through frozen WM 0.599, within noise of the 0.4045 oracle ceiling); as a *driver* it beats only trivial floors open-loop and **loses closed-loop to a 104 M baseline** | `MEASURED` (registry §6, §5.2b) |
| 2 | **Planner / closed-loop coupling** (co-evolution crux, H27) | **Maybe** | **PROMISING-UNRESOLVED** — "coupling failure was a warm-start artifact; co-evolve from random init" is a real find, but in-loop only, formal gate deferred, and *the level it reaches is unknown*. WC only if the co-evolved planner beats REF-C closed-loop | `MEASURED`(in-loop)/`HYPOTHESIS`(level) |
| 3 | **YouTube-IDM / VPT-for-driving** (data efficiency, H7) | **Maybe — highest ceiling** | **DIRECTIONAL** — pseudo-label WM pretraining ≈92% of real-label ceiling (80 clips/3 seeds); the **decision-grade scale-up was NOT delivered** (bot-blocked, cooldown to 07-26) and the C2 *slope* is unmeasured | `MEASURED`-directional (pilot) |
| 4 | **The dataset** (TanitDataSet-C/R + v2 50 h corpus) | **No** (as a dataset paper) | **SI/INFRA** — thoughtful 2-tier licensing; C published (14 shards). But R==C (8 NC sources pending), and the v2 corpus is a PhysicalAI *subset*. WC potential lives in the *YouTube-IDM* corpus (novel, action-from-video, privacy-safe), not the balancing work | `MEASURED` (QA, tier doc) |
| 5 | **TanitEval + the measurement/rigor methodology** | **Borderline** | **SI→WC-as-methodology** — episode-cluster bootstrap default, leak-split refused in code, retraction taxonomy, evidence-class doctrine. A real "how not to fool yourself in AV research" contribution — but methodology rarely lands as "world-class" in the beat-SOTA sense; it is a **moat, not a headline** | `MEASURED` (153 tests, ci.py) |
| 6 | **Inference-efficiency / deployment** (CNCE, two-tick, FP16, INT8-reject) | **No** | **SI** — honest real latency (18.75 ms p50 composed tick), INT8 rejected on evidence, ONNX→TRT-FP16 proven. But CNCE is a *self-defined* metric (Mission Plan warns against self-defined claims), and it's efficiency engineering, not a SOTA-beating headline | `MEASURED` |
| 7 | **Clean negatives** (frozen-encoder ceiling; camera-conditioning refutation) | **Partly** | **SI** — two rigorous, causal, publishable negative results. Genuinely useful to the field; not "world-class" but real works | `MEASURED` |
| 8 | **GeoCalib per-video intrinsics front-end** | **No** | **INFRA** — useful enabling utility (median 66.6° vs assumed 100°, found real bugs), honestly bounded (r=0.41 abs-focal). Supporting component of #3, not standalone | `MEASURED` |
| 9 | **Safety / self-monitoring + rule compliance** (H9/H11, SC-14/TLC) | **No (this phase)** | **EARLY** — instruments built, D8 separation preview-only (p≈0.047), closed-loop half renderer-gated. Least-mature edge | `MEASURED`(instruments)/gated |

**Honest count.** World-class *today*: **0**. Credible path to world-class *if delivered*: **1–2** (#3 the
data-efficiency slope; #2 the co-evolved planner *iff* it beats REF-C). Solid/publishable-incremental
already in hand: **~4–5** (#1 WM-as-simulator, #5 methodology, #6 efficiency, #7 negatives, arguably #4
infra). **So the program is plausibly on track for ≥5 _works_ — but likely 0–2 _world-class_ works.** The
entire gap between "5 works" and "5 world-class works" rides on the two Maybes, both currently blocked or
in-flight.

---

## 4. Strategic-coherence grade & progress-toward-vision

**Coherence grade: C+.**
- *What earns the "C" and not lower:* the underlying bet is legitimate and the Master Plan/PROGRAM_OVERVIEW
  have correctly **re-scoped** the constitution's unicorn rhetoric into a defensible research thesis
  ("drives better per unit compute than 15–120× larger stacks, on OOM-less data, zero perception labels").
  The discipline (F9) is A-grade. Compute is spent frugally and local-first.
- *What holds it down from B/A:* (1) mis-prioritization — instruments and negatives over the one headline
  (F2); (2) the hero-vs-baseline inversion left unresolved (F1); (3) the critical unblocking build owned by
  nobody (F4); (4) an undefined finish line (F8); (5) churn consuming the true bottleneck (F3, F5).

**Progress toward the *literal* vision (beat the 5 named opponents, unicorn in 3 months): ~5%.** `ESTIMATED`.
No opponent comparison, no recognized-benchmark entry, no closed-loop competence, OOD generalization
failing (comma2k19 17.5% win-rate). The literal vision is not being pursued and could not be hit on this
timeline; the program has — sensibly — not actually tried to build a Waymo-beating production stack.

**Progress toward the *re-scoped research thesis*: ~30–35%.** `ESTIMATED`. Real: a WM that beats honest
floors open-loop and behaves as a good differentiable simulator; two clean negatives; a productionized,
trustworthy measurement stack; a de-risked (not proven) data-efficiency mechanism. Missing and decisive:
the C2 slope (0%), a closed-loop win (currently negative vs own baseline), a clean-licensed publishable
record (F7), and any opponent-facing number (0%).

**The 2026-10-05 (P7) risk.** On the current vector, the first hard evaluation arrives with an excellent
apparatus and no falsifiable "we beat X" claim. The vision demands *recognizable* metrics vs named
opponents; the program has not scheduled a single one.

---

## 5. Concrete proposals (prioritized by payoff)

> Ordered by strategic payoff per unit of the *actual* bottleneck (loop-attention + PI decisions), not per
> GPU-dollar. P1–P3 are the ones that change whether Phase 0 produces a world-class positive.

### P1 — Commit one pod to the reactive-agent renderer **now**; stop deferring the one build that unblocks half the thesis. **[PAYOFF: HIGHEST]**
It gates safety-grade closed-loop, D5/D6, the renderer-half of beyond-ADE, and — decisively — *any*
NAVSIM/Bench2Drive entry, i.e. the only route to an opponent-facing number before P7. The program has
already proven the cheap experiments around it are exhausted (F4); the honest next move is the expensive
build, pre-registered with a falsifier (target on-policy OOD < ~1.3× with reactive agents). This is the
single highest-leverage reallocation on the board.

### P2 — Elevate the C2 data-efficiency slope to *the* headline objective and resource it as such. **[PAYOFF: HIGH]**
It is the most defensible world-class result and "the one headline still entirely unmeasured." Concretely:
(a) after the 07-26 cooldown, run the staged gentle YouTube-IDM scale-up to *decision-grade* (~300+ clips,
4+ seeds, GeoCalib intrinsics); (b) **design and pre-register the actual slope experiment now** —
matched-param supervised baseline vs WM across ≥4 data decades — rather than treating slope as a
someday-Phase-1 item. If a pod must move, take it from v2-corpus balancing (P4-priority work), not from
the flagship or the renderer.

### P3 — Resolve the flagship-vs-REF-C identity question at the v4 30 k gate. **[PAYOFF: HIGH]**
Force the decision the evidence is demanding (F1): *either* the co-evolved v4 planner beats REF-C
closed-loop (justifying the 4-brain-as-hero narrative), *or* elevate REF-C anchored-diffusion to co-hero /
product candidate and reframe the flagship contribution around **WM-as-simulator + data-efficiency**
(where the positives actually are). Pre-register both branches with committed outcomes. Do not let a
"reference baseline out-drives the hero" state persist unexamined into Phase 1.

### P4 — Cut/park low-value threads to reclaim orchestrator bandwidth (the real bottleneck). **[PAYOFF: MED-HIGH]**
- **Retire REF-A and REF-B from active compute.** H4 is answered (REF-A's job is done; keep as archived
  control). REF-B (no-WM BC, 0.592) is now redundant given REF-C. `INHERITED`/`MEASURED`.
- **Compress the 7-agent weekly cadence to the 3 on the critical path** (Architecture & Inference, Data
  Engineering, Benchmarks & Eval) at weekly, and move Opponent Analyzer / Tools&DevEnv / Production &
  Optimization to *fortnightly*. The weekly-7 structure is inherited from the org design in the Mission
  Plan; the program now has one narrow critical path and the synthesis load on the single orchestrator is
  the scarce resource (F3/F5).

### P5 — Fix the two SPOFs and de-bloat the state. **[PAYOFF: MED]**
(a) Escalate the ~13 GB HF cleanup as a *blocking* PI decision (it simultaneously unblocks the checkpoint
backup, the formal v4 gate, and a benchmark arm — F5/F6), not a background note. (b) Get the live
checkpoint off single-pod disk immediately after (F6). (c) **Split `LOOP_STATE.md`** into a ≤1-screen live
header (fleet, active streams, open decisions) + an archived append-only log; the 58 k-token drumbeat is
now causing the stale-headline propagation it is meant to prevent (F3/F6).

### P6 — Define the finish line: operationalize "≥5 world-class works." **[PAYOFF: MED]**
Write a one-page `WORKS_LEDGER.md`: for each candidate work (§3), state the *acceptance test* that would
make it "world-class" (e.g. "#3 = C2 slope with a matched-param supervised baseline, CI-separated, on a
clean-licensed corpus"), the current status, and the single next gate. A goal you cannot measure against
cannot be steered toward (F8). This also forces the F7 licensing migration onto the schedule, because most
acceptance tests should require a *citable* corpus.

### P7 — Pre-commit the licensing migration of the headline record. **[PAYOFF: MED, rising]**
Schedule the re-derivation of the flagship (or REF-C) headline on comma2k19/L2D **before** Phase 2, not
inside it (F7). The strongest current evidence is on a corpus that cannot be cited; the publishable claim
must be built on clean data with enough runway to survive its own retraction cycle.

---

### One-paragraph bottom line
TanitAD is a rigorously honest, frugally-run research program built on a legitimate architectural bet,
and it has already banked several solid and publishable results plus a measurement moat most labs lack.
But it is spending its scarce resource — a single agent loop's attention and the PI's decisions — on
instruments, negatives, and self-correction, while its hero model loses closed-loop to its own baseline,
its one decisive headline (the data-efficiency slope) sits unmeasured, and the one build that would unlock
every opponent-facing number is owned by nobody. Point the same discipline at **one** converging positive
— the renderer that unblocks a benchmark number, and the C2 slope that proves the data thesis — and the
≥5-*works* count can include ≥1–2 genuinely *world-class* works. Leave the prioritization as-is and
Phase 0 exits with an A-grade apparatus and no A-grade claim.
