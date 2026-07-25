# R2 — Independent Chief-Scientist Review: Scientific Approach & Architecture

**Reviewer role:** external, skeptical peer reviewer of the *scientific approach and architecture* of
TanitAD (sub-300 M hierarchical "4-brain" latent world model for AD). **Date:** 2026-07-25.
**Discipline:** primary sources only (`MODEL_REGISTRY.md`, raw eval JSON paths, `Paper/TANITAD_PAPER.md`,
architecture hub); evidence class on every load-bearing claim
(`MEASURED`/`PUBLISHED`/`INHERITED`/`ESTIMATED`/`HYPOTHESIS`). Read-only; nothing staged.

This review is deliberately hard. Where the program is strong I say so plainly (§Findings F0), but my
value here is exposing where the *architecture is complexity-without-payoff*, where the *evidence does not
support the design*, and where it *lags or duplicates SOTA*.

---

## 1. Executive verdict

The world model is real and the program's measurement discipline is world-class, but **the architecture's
distinctive bets are its least-supported ones, and its own budget-matched control is beating it.** A flat
104 M anchored-diffusion reference arm (REF-C-base — a re-implementation of published DiffusionDrive) is
statistically *tied* with the 263 M hierarchical flagship open-loop and *beats* the flagship's deployed
head closed-loop; the 3-level hierarchy has **one of three seams load-bearing** and one actively harmful,
so on every number that exists the strategic/tactical layers (~22 % of params) are unpaid-for. The two
headline "edges" that would make TanitAD more than a small competent WM — **imagination and
self-monitoring — are measurably broken past one step** and below their own gates. The corpus (13 h,
74 % straight, 0 % semantic, ~473 K frames — 1–2 orders below the published multimodality threshold)
**cannot test the scale-vs-structure thesis**, so "sub-300 M is enough" is currently unfalsifiable rather
than proven. The frozen→from-scratch pivot is *principled in method* but arrives, after six failed
flagship variants, back at "train v1's recipe from scratch with planners instead of heads" — a sound bet,
evaluated in-loop only, not yet beating v1. **Net: excellent science, honestly reported, around a thesis
whose differentiators remain unproven and whose best current result re-implements the field.**

---

## 2. Findings

Each finding: evidence (class + path) · severity · confidence.

### F0 — What genuinely works, credited up front (so the criticism below is calibrated)

- **The causal vision-anticipation panel is a real, well-controlled result.** On high-CTRV-divergence
  windows the flagship beats the CTRV *oracle* by +0.796 m on 94 %, and the *entire* advantage is vision:
  mean-replacing the scene inverts it to −0.529 m → **vision effect +1.325 m, CI [+1.04, +1.64]**,
  CI-separated; monotone in divergence; upcoming-curvature decodes at R² 0.254 vs 0.031 ego-only;
  occluding the road-ahead perturbs prediction 1.6× more than periphery. `MEASURED`
  (`Paper/TANITAD_PAPER.md` §7.3; `MODEL_REGISTRY.md` §1.2 "genuine-prediction panel";
  `taniteval/generalization.py`). This is the strongest scientific claim in the program and is genuinely
  publishable — *in-distribution*.
- **The instrument doctrine + `RETRACTION_LOG.md` is exceptional and rare.** Estimator correction
  (overlapping-holdout SE → episode-cluster bootstrap, measured 1.28–2.06× too narrow), the leak-split
  refused in code, the two-tick latency doctrine, I8/I9/I10. Most academic AD work would not survive its
  own scrutiny; this program applies it to itself. This *is* a contribution.
- **Clean, publishable negatives:** frozen-encoder ceiling (H4), Branch-B camera-conditioning refutation,
  INT8 rejection on evidence. `MEASURED`.
- **Efficiency is solved for the deployed arm:** composed planning tick 100.29 → **18.75 ms p50** (5.35×),
  10 Hz at p99 with 5.3× headroom; the CEM-imagine-and-select feasibility (20.82 ms K=8) *refuted* the
  723 ms projection that nearly retired the planning thesis. `MEASURED` (`MODEL_REGISTRY.md` §1.2;
  `taniteval/results/eff_levers_flagship-30k.json`).

Severity n/a · Confidence HIGH. **These are real. They are also, with one exception (the vision panel),
about execution quality rather than the architectural thesis.**

---

### F1 — The 4-brain hierarchy is not justified by measurement, and the one head-to-head test points against it — **[architecture-astronomy risk: HIGH]**

**The hierarchy panel is a mixed-to-negative verdict, self-reported.** At 30 k, **one of three top-down
seams is load-bearing** (ctx→tactical, vs-mean maneuver Δ +0.044 CI-separated); **intent→operative is
*harmful* when ungated** (magnitude-swamped, ‖31.4‖ vs ‖28.3‖, deployed intent-free by design, needs a
ReZero gate to be merely inert); **nav→strategic is a pure command-echo** (route-from-vision skill 0.0).
`MEASURED` (`Paper/TANITAD_PAPER.md` §7.5; `MODEL_REGISTRY.md` §1.2). The program's own architecture doc
is blunter still: *"H26 has not been demonstrated on any arm. It has been demonstrated on one seam of one
arm, and that seam feeds a head we do not deploy,"* and *"Our best arm's headline number is measured on a
path that **structurally excludes its hierarchy**"* — the deployed 0.452 rollout is **intent-free and
bypasses all three seams** (`…/Architecture & Inference/ARCHITECTURE_WIRING_COMPARISON.md` §2.6, §5.1;
`Paper/TANITAD_PAPER.md` §9.1: "the hierarchy exists as structure, not as functioning planning").

**The program's own positive reads disagree and are marginal where positive.** `PROGRAM_OVERVIEW.md` (H26)
and the wiring doc say **1 of 3** seams load-bearing; the 2026-07-25 gate dry-run says **2 of 3** — but its
second "load-bearing" seam is nav→strategic counted *"by construction"* (route acc 1.0 *with* the command,
"vision route NONE" without — the very command-echo the wiring doc calls **inert**), and its ctx→tactical
"load-bearing" rests on goal-cos Δ **0.0084**, *below the panel's own MIN_COS floor of 0.01*, with maneuver
and waypoint deltas **not CI-separated**
(`…/Benchmarks & Eval/…/2026-07-25-v4-gate-dryrun/raw/hierarchy_flagship-30k.json`). So the hierarchy's
positive evidence is inconsistent across the program's own emitters and marginal-to-below-threshold where
asserted.

**The flagship's supervised tactical head (3.38 m) is worse than constant velocity**, while the same
model's operative rollout is 0.452 m (`MODEL_REGISTRY.md` §6, row "tactical head"). The upper brains are a
lossy readout, not a decision-maker.

**The killer comparison — the control beats the treatment.** REF-C-base, a **flat 104.2 M**
anchored-diffusion arm with **no hierarchy**, is **paired-tied** with the 263 M hierarchical flagship
open-loop (Δ +0.0013 [−0.0281, +0.0316], not separated; `MODEL_REGISTRY.md` §4.3) and **beats the
flagship's deployed head closed-loop** (0.564 vs 1.488, triple-confirmed; §F4). The upper hierarchy costs
**~57.7 M / 263.4 M ≈ 22 %** of the flagship (strategic_policy 8.39 M + tactical_policy 22.74 M +
tactical_pred 26.53 M; `MODEL_REGISTRY.md` §1.1) for **zero measured benefit** on any shipping metric.
*(In fairness the cost falls in v4: the three-planner restructure is ~25 M / ~9 % — the strongest defense
against the "astronomy" charge is that the hierarchy is genuinely cheap; the wiring doc's own rule is to
"judge them on Gate H and per-window content, **never** on whether they move ADE alone," because "they
cannot buy proposals." The point stands regardless: cheap ≠ demonstrated, and the benefit is unmeasured.)*

**The evidence that IS positive supports a different claim than the one being made.** P2 (a CEM planner
over the frozen v1 WM) beats the tactical head +2.257 m open-loop (72 %) and drifts 38 % less closed-loop
(`MODEL_REGISTRY.md` §5). But **P2 is a *flat* operative planner, not a 3-level hierarchy.** So the
program's real, measured result is **"planning-over-the-WM > supervised heads"** — which is a strong claim
— *not* **"3 levels > 1 level,"** which is nowhere isolated. No arm compares a flat planner-over-WM to the
full strategic/tactical/operative stack. The hierarchy's justification is currently **theoretical**
(Cui et al. T-linear regret → per-level short-horizon bounds, `PUBLISHED` arXiv 2606.27014; Michon/Donges
driver models) and **not empirical**.

**In fairness:** the program correctly *refuses to kill* the hierarchy, because the one finding that
looked like a refutation ("the fan is a speed fan ⇒ strategic choice is a 2 % lever") was confounded
(`RETRACTION_LOG.md` 07-21: REF-C evaluates `nav_cmd=None`, so a route-blind decoder learned the
marginal). That restraint is correct. But "not refuted" is not "justified." The honest status is: **an
unproven bet, currently unpaid-for, whose only head-to-head evidence points the other way.**

Severity **HIGH** · Confidence **HIGH** (`MEASURED`).

---

### F2 — The program's best arm is a re-implementation of published work (DiffusionDrive), and it is beating the flagship — **[SOTA-positioning: HIGH]**

REF-C is explicitly **DiffusionDrive** (`PUBLISHED` arXiv 2411.15139) / DiffusionDriveV2 (91.2 PDMS /
85.5 EPDMS on NAVSIM, arXiv 2512.07745); the program's own research note says *"This is the field's answer
… Do not change the decoder family"*
(`…/Research/2026-07-25-closed-loop-diffusion-planner/CLOSED_LOOP_PLANNER_RESEARCH.md`). On every MEASURED
axis this re-implementation matches or beats the differentiated hierarchical WM:

- Open-loop: REF-C-base 104 M **ties** flagship 263 M and REF-C-XL 252 M — a genuine three-way tie no
  paired test can order (`MODEL_REGISTRY.md` §6, §4.3). **Scale bought nothing above 104 M on this corpus**
  (small 54.7 M < base ≈ XL; §4.2).
- Closed-loop: REF-C-base **out-drives** the flagship's head (§F4).
- Latency: REF-C-base is the *cheapest* tick in the table (21.8 ms fp32 p50 vs flagship's 103 ms
  un-optimized; `MODEL_REGISTRY.md` §6 reading 3).

**Implication:** the program's genuine novelty (from-scratch, label-free SSL driving WM) has **not
translated into a driving-competence lead over a compact published baseline.** The defensible counter is
that the flagship's *operative rollout* (0.452), not its head, is the WM's real output — and the v4 line
exists precisely to replace the lossy head with a planner over that rollout (§F7). That is coherent, but
**v4 is unproven and in-loop-only**, so as of today the strongest cost-adjusted arm in the program is a
re-implementation, not the thesis.

Severity **HIGH** · Confidence **HIGH** (`MEASURED` + `PUBLISHED`).

---

### F3 — The differentiating "edges" (imagination, self-monitoring) are the least-validated, and imagination is measurably broken past one step — **[edge-validity: HIGH]**

The program's four edges (`PROGRAM_OVERVIEW.md` §2) place **hierarchy + imagination + self-monitoring** as
the north-star differentiators ("win on hierarchy + imagination + self-monitoring + per-scenario
excellence, not on scale," §0). Those are exactly the unvalidated ones:

- **Imagination (H15) is temporally anti-calibrated.** Under blind autoregressive rollout, fidelity falls
  0.357 → 0.011 (chance) by k=4 **while predicted log-variance shrinks** (−7.79 → −8.55) — the field
  becomes *more confident as it decays*. The §3.5 calibration property holds **at one step only**, and it
  **reproduces on the shipping operative model** (falsifier "the speed recipe fixed it" NOT met).
  `MEASURED` (`Paper/TANITAD_PAPER.md` §7.2; `…/Architecture & Inference/Research/STATE.md`;
  `GOALS.md` G2). A self-monitor built on a signal that is confidently wrong when blind is a **safety
  anti-pattern**, and every deployed use is capped at 1-step as a result.
- **Self-monitoring has not reached its gate.** D8 AUROC > 0.85 not achieved; the only real number is a
  paired weather-degradation shift at **p ≈ 0.047**, "weak and confounded this early"
  (`Paper/TANITAD_PAPER.md` §7 self-knowledge; `PROGRAM_OVERVIEW.md` H11).
- **`vision_use` is flat at ~12 %** (`PROGRAM_OVERVIEW.md` H15) — the encoder redundantly re-encodes fed
  proprioception (in-latent yaw R² 0.89), so the "world model" is closer to a dynamics integrator than the
  scene-reasoning system the vision implies (the §F0 causal panel is the honest counter-evidence, but
  12 % is the aggregate).
- **The theoretical foundation is not yet met by the model.** SIGReg/LeJEPA isotropy is
  **NOT-YET-ADMISSIBLE** (rms_offdiag 0.32 > 0.1), so *"the LeJEPA optimal-planning corollary is still
  withheld"* (`…/Research/STATE.md` E2). The generalization theory that motivates the whole approach
  (2606.27014) uses a spectral-contrastive term, not SIGReg — "the constants do not transfer"
  (`…/Research/2026-07-06-jepa-generalization-theory-and-hit-jepa.md`). The theory is a *motivation*, not
  a *guarantee the model satisfies*.

Severity **HIGH** (these are the claimed differentiators) · Confidence **HIGH** (`MEASURED`).

---

### F4 — Open-loop is the entire evidence base; the one clean closed-loop signal says the WM head loses; C1 ("it drives") is unproven — **[core-claim risk: HIGH]**

Every leaderboard number is **open-loop** (`MODEL_REGISTRY.md` §6), and the program itself repeats that
open-loop ⊥ closed-loop (arXiv 2605.00066). The one decision-grade closed-loop instrument (n=40
real-footage log-replay, on-policy OOD 1.02–1.20× — a genuine methodological advance over the 3.2× NuRec
reconstruction and the self-referential imagination-in-the-loop harness) returns:

| | flagship v1 head | REF-C base | paired Δ | sep |
|---|---|---|---|---|
| closed-loop ADE@2s | **1.488** [1.329,1.647] | **0.564** [0.452,0.676] | +0.924 | ✅ |
| corridor departure @1.75 m | 0.0318 | 0.0134 | +0.0184 | ✅ |

`MEASURED` (`Paper/TANITAD_PAPER.md` §7.8). Triple-confirmed (n=1 retracted → n=12 AlpaSim 8/12 vs 2/12 →
n=40). The deficit is **longitudinal, not lane-keeping** (both keep lanes; flagship ADE 4× worse in the
longitudinal stratum) — consistent with the 89 %-along-track open-loop signature. And the *safety* half
(off-road, collision, TLC/LAL/OKRI/LOPS) is **renderer-gated** — unmeasurable with current assets
(`PROGRAM_OVERVIEW.md` §2④, §7).

**The field has a directly relevant datum the program is only now catching up to:** CAT-K shows a **7 M
closed-loop-tuned model beats a 102 M open-loop one** (`PUBLISHED` arXiv 2412.05334, via
`CLOSED_LOOP_PLANNER_RESEARCH.md`) — i.e. **closed-loop *training*, not scale or hierarchy, is the lever**,
and TanitAD's entire program optimizes and gates on open-loop. No NAVSIM/Bench2Drive/nuPlan entry exists.

Severity **HIGH** · Confidence **HIGH** (`MEASURED` + `PUBLISHED`).

---

### F5 — The corpus cannot test the central thesis; "sub-300 M is enough" is confounded by an easy corpus — **[thesis-testability: HIGH]**

`MEASURED` (`Paper/TANITAD_PAPER.md` §7.9): 13.13 h, **74 % straight**, a 2-parameter kinematic oracle
(CTRV, 0.523) **tops the open-loop table above every learned arm** (§7.2), **42.6 % of clips never turn**,
**0 % semantic scenarios** (lights/roundabouts/merges). Corpus size ~473 K frames — and the closed-loop
research note flags this is **"1–2 orders of magnitude below the published multimodality threshold"**
(mode collapse ≤ 100 K frames, multimodality emerges at 20–70 M frames; `PUBLISHED` arXiv 2602.22801) —
so the anchored-diffusion multimodality the whole planner design leans on is **structurally under-fed**.

Two consequences: (i) **"sub-300 M is enough" is currently unfalsifiable, not proven.** REF-C-base 104 M
ties the flagship *because the corpus is kinematically trivial* — a regime where a 2-param oracle wins. The
thesis ("structure beats scale on *hard* driving") requires hard scenarios the corpus does not contain.
(ii) The one place generalization is tested, it fails: comma2k19 OOD 0.849 vs floor 0.372 (17.5 % win),
anticipation advantage collapses 0.80 → 0.15 m, path feasibility 97.8 % → 62.8 % (§7.4). Notably,
**"sub-300 M" is nowhere argued head-on as an advantage** in the design corpus — the real efficiency claim
is *latent-vs-pixel* ("GAIA-class pixel models need 1000× our data," theoretical/`PUBLISHED`), and the
concrete measured *handicap* is data scale, not params.

Severity **HIGH** · Confidence **HIGH** (`MEASURED` + `PUBLISHED`).

---

### F6 — Version churn driven by premature certainty: six failed flagship variants — **[process → science: MEDIUM]**

Flagship lineage: v1 (good) → **v2** (killed step 7.8 k, 9× worse — "all ten levers at once") → **v3enc**
(RESTART at 10 k) → **v4/v4.1/v4.2/v4.2b** (all warm-start failures) → v4-fromscratch (in-loop only).
`MEASURED` (`MODEL_REGISTRY.md` §1.3, §1.4, §1.4b; `Paper` §7.6). `RETRACTION_LOG.md` records **five
"this direction is closed" claims reopened by zero-cost checks in a single session** (07-24), and
`PROGRAM_OVERVIEW.md` §6 names *"premature certainty … the recurring failure mode, not sloppiness."*

The mitigating truth is real: each failure was diagnosed cheaply and redirected (the cosine pre-probe
saved ~1.3 A40-days; §F7), and the retraction discipline converts errors into method. But the **cadence**
— ten levers at once, then a staged restart, then four warm-start arms — indicates the program repeatedly
mistook a hard coupling problem for a series of tractable tweaks. The cost is not just GPU-days; it is that
the flagship's *architectural* verdict has been re-litigated five times while REF-C quietly answered the
same corpus once.

**The more precise diagnosis** (visible in F1 and F7) is subtler than "premature certainty": the program
reliably concedes what is unproven, but repeatedly **decides on the cheaper of two available
measurements** when a more decisive one is affordable — v1's existence-proof instead of the ~0.66 A40-day
λ=0 control (F7); the in-loop val instead of the held-out gate; the 2 s ADE window instead of a
horizon-honest closed-loop instrument (which, when finally run, overturned the "BOUND" verdict, corridor
departure 0.0035→0.5877 at K=185 vs K=20). Not fabrication — under-powered decisiveness, self-flagged each
time. It is the one habit most likely to let a wrong architectural verdict survive to a GPU-day.

Severity **MEDIUM** · Confidence **HIGH** (`MEASURED`).

---

### F7 — The v4 from-scratch co-evolution bet: sound and well-reasoned, but it is v1's recipe with planners for heads, evaluated in-loop only, and not yet beating v1 — **[current critical path: MEDIUM]**

The redirect was genuinely good science: rather than buy PCGrad gradient surgery, they measured the seam
geometry — **cos(g_wm, g_plan) = +0.0043** over 512 windows, 47.9 % negative, PCGrad removes 2.2 % — and
correctly concluded surgery is a no-op, for ~0 GPU (`Paper` §7.6; `MODEL_REGISTRY.md` §5.2a). The
resulting thesis ("planner–WM interference was a *warm-start* artifact") is plausible and maps cleanly to
the LP-FT literature (Kumar 2202.10054: full-fine-tuning distorts features).

Three cautions the reviewer must weight:

1. **v4-fromscratch is "the same architecture and command with the trunk randomly initialised — v1's own
   recipe"** (`Paper` §7.6). After six variants, the fix is *v1, with three planners replacing three
   heads*. That is defensible convergence, but it means the novelty reduces to "planners instead of heads"
   — the same swap P2 already validated flat.
2. **The evidence is in-loop, not eval-harness** (C1 risk; v1.6's in-loop read 10 % optimistic). At
   ~step 19.4 k (~64 %) the arm has *survived the coupling ramp* — the "v4.x death zone" where every
   warm-start arm blew up — with the WM canary descending under λ_plan = 1.0 (a genuine, meaningful
   positive signal; `…/LOOP_STATE.md`), but in-loop ADE ≈ 0.48 is still **not beating v1's 0.427**, and
   the formal gate **cannot render a verdict — 3 of 8 kill secondaries have no emitter anywhere in the
   codebase** (`Paper` §7.6, §10). A gate that cannot complete is an instrument failure carried as a
   pending model result.
3. **The commit was made on the cheaper of two available measurements.** The surgery *refutation* (cos
   +0.0043) is well-measured and robust. But the *positive* mechanism the from-scratch pivot assumes —
   warm-trunk re-optimisation degrading the WM — was left as `HYPOTHESIS`: a ~0.66 A40-day λ_plan = 0
   control that would have *measured* the failure cause was pre-registered and **skipped**, the bet placed
   instead on v1's existence proof ("cheaper than a 4th floor-roulette";
   `…/2026-07-23-v4.2b-fork/V4.2B_FORK_NOTE.md`). This is not thrashing — it is the program's recurring
   softer pattern (see F6): choosing a cheaper existence-proof over an affordable discriminating
   measurement. The success condition (beat 0.427 *and* fix longitudinal closed-loop) is not in hand, and
   the descent trend was itself the subject of a same-day n=1 retraction (`RETRACTION_LOG.md` 07-24).

Severity **MEDIUM** · Confidence **MEDIUM-HIGH** (`MEASURED` in-loop; the level is `HYPOTHESIS`).

---

### F8 — Decision-grade artifacts remain single-disk / uncommitted — **[reproducibility: MEDIUM-HIGH]**

`TanitEval` (produces every headline ADE), **REF-B v2's architecture** (the 0.592 arm — "cannot be rebuilt
from this repo today"), **P2** (the single strongest evidence for the v3/v4 pivot), the **v1 speedjerk
trainer** (not byte-reconstructible: `--jerk-weight`/`--aux-accel` absent from committed args), and the
**live v4 checkpoint** (single pod disk, HF backup 403-blocked) are each one disk-loss from gone.
`MEASURED` (`MODEL_REGISTRY.md` §0.3 risk block, §3.5, §5, §1.2 risk block; `PROGRAM_OVERVIEW.md` §5.3).
This is process, not science — but a program whose thesis rests on cheap reproducibility cannot have its
eval harness and its pivot's evidence living on one pod.

Severity **MEDIUM-HIGH** · Confidence **HIGH**.

---

## 3. Achievement estimate

I score two distances separately because they diverge sharply.

### 3a. Distance to the stated *vision* — **~15–20 %**

The vision (`PROGRAM_OVERVIEW.md` §0): a sub-300 M hierarchical, self-monitoring, imagining WM that
*drives* (L3/L4), needs orders less data, and beats 15–120× larger incumbents, winning on
**hierarchy + imagination + self-monitoring + per-scenario excellence**. Scoring its load-bearing claims:

| Vision component | Status | Demonstrated? |
|---|---|---|
| Running 4-brain WM | Built, trained to 30 k | ✅ |
| Beats trivial floors open-loop | 0.452 m < CV/CTRV/floor | ✅ |
| **Drives (closed-loop)** | Head loses to 104 M baseline | ❌ |
| **Hierarchy functions as planning** | 1/3 seams; "structure not function" | ❌ |
| **Imagination edge** | Broken past 1 step (false confidence) | ❌ |
| **Self-monitoring w/ guarantees** | D8 AUROC unmet; p≈0.047 | ❌ |
| **Data-efficiency slope (C2)** | Unmeasured (mechanism promising) | ❌ (◐) |
| **Beats incumbents on recognized bench** | No NAVSIM/B2D/nuPlan entry | ❌ |
| **Per-scenario excellence** | Corpus 0 % semantic | ❌ |
| Efficiency / Orin envelope | 18.75 ms tick; silicon blocked | ~✅ |

~2.5 of ~10 demonstrated, and they are the **table-stakes** components (the WM runs; it clears an easy
floor; the tick fits). **Every one of the seven vision *differentiators* is un-demonstrated or negative.**
The program is early (day ~21/42 of Phase 0), which is the fair mitigation — but the achievement is
**front-loaded onto the easy half**, and the hard half is where the current evidence is adverse.

### 3b. Distance to a recognized *SOTA* result — **large and, critically, unmeasured**

- **Rigor / honesty: A− / B+.** The measurement culture exceeds most published AD work.
- **Demonstrated SOTA-competitive driving: D / F.** No entry on any recognized benchmark. The internal
  proxy for "are we competitive" is REF-C, which *is* DiffusionDrive — so the honest read is the program
  is at **"competent re-implementation that ties/loses to itself,"** not "ahead." The one genuinely
  distinctive positive (causal vision anticipation) is real but is an *in-distribution interpretability*
  result, not a driving-competence or benchmark result, and it does not survive OOD.
- Where it is *legitimately* novel (positioning, not performance): the **strict-parity frozen-vs-trained
  encoder head-to-head for a driving WM** (the field mostly asserts one side), and **from-scratch,
  label-free SSL at ~$40 compute**. These are real contributions *if* the C2 slope materializes.

**Overall grade: process A−, thesis-validation D+, distance-to-vision ~15–20 %, distance-to-recognized-SOTA
unmeasured-and-probably-large.** The gap between the two column-scores *is* the finding: this is a superbly
run program pointed at a thesis it has not yet begun to validate on the axes that would make it matter.

---

## 4. Concrete proposals (prioritized by payoff / effort)

**P1 — [HIGH payoff / LOW effort] Run the one experiment that actually tests the hierarchy, or stop
claiming it.** No arm isolates "3 levels > flat planner-over-WM." Take P2/v4-operative (flat) vs the full
strategic/tactical/operative stack on the identical 881 windows, paired episode-cluster bootstrap. Until a
CI-separated win exists, the **57.7 M upper-hierarchy is unpaid-for** and every "hierarchy" claim should be
downgraded to "planning-over-WM" (which the evidence *does* support). Pre-register G1 (counterfactual
plan-ranking, `Paper` §9.4) at 30 k as the falsifier. *Payoff: resolves the program's central architectural
question. Effort: one eval-pod pass + one gate.*

**P2 — [HIGH / MEDIUM] Get ONE recognized-benchmark number.** Run **REF-C-base on NAVSIM (EPDMS) or
Bench2Drive (DS)** — it is a DiffusionDrive-class arm and the harnesses exist in the field
(`CLOSED_LOOP_PLANNER_RESEARCH.md` cites both). This is the only way to convert "we tie our own
re-implementation" into a real SOTA coordinate and the only way to test "sub-300 M is enough" against a
recognized bar the 74 %-straight corpus cannot provide. *Payoff: the program's first external ground
truth. Effort: harness integration, no new training.*

**P3 — [HIGH / MEDIUM] Measure the C2 data-efficiency slope — the #2 goal, entirely unmeasured, and the
single most defensible potential world-class result.** The mechanism evidence is already strong (pseudo-
label WM pretraining ≈ 96 % of real-label value; YouTube pilot ≈ 92 % of ceiling; `Paper` §7.9). Turn it
into an actual *slope* vs a supervised baseline at matched params. A label-free WM matching supervised
data-efficiency at ~$40 is the one headline that would be novel *and* defensible. *Payoff: the thesis's
best shot at a world-class claim. Effort: a controlled sweep, mechanism already de-risked.*

**P4 — [HIGH / HIGH] Adopt a published closed-loop-consistent *training* recipe; stop gating on open-loop.**
The field converged on failure-gated CL-SFT (R2LPL 2606.30537, CAT-K 2412.05334) and TanitAD is catching
up, not leading. Closed-loop competence is the binding constraint on C1, and CAT-K's "7 M CL-tuned beats
102 M open-loop" says the lever is training regime, not architecture. Pair with the lower-OOD
reactive-agent renderer (the largest remaining build; `PROGRAM_OVERVIEW.md` §8.3). *Payoff: attacks the
one claim ("it drives") that is currently failing. Effort: high (renderer build).*

**P5 — [MEDIUM / LOW] Stop shipping "imagination" and "self-monitoring" as edges until they pass their own
gates.** The multi-step σ is a false-confidence failure (§F3); externally this would read as overclaiming.
Either fix it (horizon-aware σ / parallel-horizon decode, already scoped in `STATE.md`) or scope every
claim to "1-step familiarity signal." *Payoff: protects credibility of the safety case. Effort: low
(honest scoping) to medium (the fix).*

**P6 — [CUT] Freeze the REF-A frozen-encoder zoo and end within-family flagship churn.** H4 is closed
negative and re-localized; the ijepa/dino320/speedyaw arms were diagnostic-only (one val-leaked). The v4
line must **not** spawn a v5 warm-start — pre-register one co-evolution gate and hold it. Redirect that
compute to P2/P3. *Payoff: stops the version-churn tax (§F6). Effort: a decision.*

**P7 — [DECIDE / LOW] Resolve the hierarchy commitment.** Given F1, either (a) commit with P1's
falsification gate, or (b) collapse to a flat planner-over-WM — which the current evidence favors — and
reallocate the 57.7 M to encoder/data. The status quo (carry 22 % of params on a theoretical argument) is
the one option the evidence does not support. *Payoff: architectural clarity. Effort: gated on P1.*

**P8 — [MEDIUM / LOW] Commit the stranded decision-grade artifacts** (TanitEval, REF-B v2 arch, P2, v1
trainer flags) and unblock the v4 checkpoint backup. §F8. *A program built on cheap reproducibility cannot
keep its eval harness on one pod.*

---

## 5. One-line bottom line

TanitAD is a **rigorously-run world-model program whose distinctive architectural bets — hierarchy,
imagination, self-monitoring — are its least-validated, whose budget-matched re-implementation of published
work is currently outperforming it, and whose central thesis (structure > scale, sub-300 M, drives) remains
untested on the closed-loop and recognized-benchmark axes that would make it matter.** The science is
excellent; the thesis is unproven; the fastest path to a defensible claim is the *data-efficiency slope*
and *one external benchmark number*, not another flagship variant.
