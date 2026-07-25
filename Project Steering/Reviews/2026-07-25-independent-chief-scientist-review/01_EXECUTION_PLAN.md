# TanitAD — Execution Plan (all tiers) + the Hierarchy Proof Program + full hypothesis actions

**Date:** 2026-07-25 · **Author:** chief-scientist synthesis · **Supersedes** §10 of
`00_CHIEF_SCIENTIST_REVIEW.md` **on proposal #6 only** (see Part A — PI direction).
**Companion documents:** `00_CHIEF_SCIENTIST_REVIEW.md`, `R1`–`R6`.

---

## PART A — The thesis is not in question. The proof is.

### A.0 PI direction (2026-07-25, binding)

> *"It's not about changing the thesis of the hierarchy. I'm deeply convinced based on my experience
> of 20 years in the field of autonomy — the question is **how to prove it consequently**."*

Proposal #6 as originally written ("test the hierarchy or drop the claim") is **withdrawn**. It is
replaced by the **Hierarchy Proof Program (HPP)** below. This is not deference to the PI — it is the
scientifically correct reading, and the review's own evidence forces it.

### A.1 Why the reframe is *scientifically* right, not a concession

**⚠️ SELF-CATCH — this is the SECOND time the program has nearly designed the hierarchy away on
confounded evidence.** `RETRACTION_LOG.md` 07-21 (C6) already records:

> *"the fan is a SPEED fan ⇒ strategic choice is a ~2 % lever"* — **CONFOUNDED**: REF-C evaluates with
> `nav_cmd=None` → a decoder that never had a route input learns the marginal. **Nearly designed the
> hierarchy away.**

Finding F1 of `00_CHIEF_SCIENTIST_REVIEW.md` ("the control beats the treatment") was walking into the
identical trap through a different door. **Every adverse datapoint is confounded**, and each confound is
independently `MEASURED`:

| "Adverse" evidence | What it actually establishes | Confound (MEASURED) |
|---|---|---|
| REF-C-base (104 M) ties flagship open-loop, beats it closed-loop | A flat planner beats *our supervised heads* | REF-C evaluates with **`nav_cmd=None`** (CLAUDE.md §Operating-standard-5) — the route input is **never exercised** in the comparison |
| Deployed flagship number 0.452 m | The **operative** brain works | The number **structurally excludes its hierarchy** (`V4_FLAGSHIP_DESIGN`, R2) — the tactical/strategic levels are not in the loop being scored |
| nav→strategic seam not load-bearing | The current strategic **implementation** is a command echo | **`route_skill = 0.0`** — zero vision-derived route inference; and v4.2b `nonav_route_beats_majority` FAILS straight 240/240. A broken implementation, not a refuted concept |
| Hierarchy shows no net driving advantage | Nothing — **the gates were never run** | **D5/D6 topology gates have NEVER RUN** (`Phase 0 Plan.md §4`, R3) |
| Flat model competitive on the leaderboard | The corpus is easy | **74 % straight, 0 % semantic events**; a **2-parameter CTRV oracle tops the table** (R2). There are almost no route decisions to make |
| No hierarchy benefit at ADE@2s | Nothing about hierarchy | **Strategic value is a 10–20 s quantity.** The 2 s metric was proven blind to an 18 s failure on this very corpus (E1a). Same instrument, same blindness |

**The logical position:** *absence of evidence is not evidence of absence when the instrument is
provably incapable of detecting the effect.* Six independent confounds, each measured, all pointing the
same way. The honest status of H1's hierarchy edge is **UNTESTED**, and the correct response to
"untested" is **to build the test** — not to drop the claim.

### A.2 What "proving it consequently" requires

A structural hypothesis is not proven by one ADE comparison. It is proven by a **battery of
discriminating predictions that only the structure explains**, each pre-registered with a falsifier.
That is the standard the HPP is built to.

**Four pre-conditions must hold before any hierarchy-vs-flat number is admissible.** Running the
ablation before these are met produces another confounded null — the exact failure we are correcting.

| Pre-condition | Why | Gate to pass |
|---|---|---|
| **PC1 — the route input must actually work** | You cannot measure the value of a strategic level whose input is a command echo | `route_skill > 0` (vision-derived route inference, CI-separated from the majority-class baseline); `nonav_route_beats_majority` PASSES |
| **PC2 — the hierarchy must be in the loop for the scored number** | The deployed number bypasses all three brains | The evaluated policy path provably traverses strategic→tactical→operative (assert in the eval harness, not by inspection) |
| **PC3 — the instrument must be able to see it** | Strategic value is long-horizon and junction-local | Horizon ≥ 18 s (Tier-1 #1), junction-stratified, plus route-compliance metrics that ADE cannot express |
| **PC4 — the corpus must contain decisions** | 74 % straight / 0 % semantic has no strategy to choose | A route-decision-rich eval set: multi-option junctions, ≥N per stratum, powered (n ≥ 40 episode-clusters) |

### A.3 The discriminating-prediction battery (the heart of the proof)

Pre-register **all six**, with both outcomes committed in advance. A flat model of equal capacity
should fail these *by construction*; a working hierarchy should pass them. **This is the deliverable
that converts a conviction into a proof.**

| # | Prediction (what a hierarchy implies that a flat model does not) | Measurement | Falsifier |
|---|---|---|---|
| **HP-1** | **Advantage grows with horizon.** Hierarchy ≈ flat at 2 s, separates by 10–20 s | Paired Δ(departure, ADE) vs K ∈ {20, 60, 120, 185}; interaction test | Δ flat across horizon ⇒ no hierarchical benefit |
| **HP-2** | **Advantage concentrates at decision points**, not cruise | Junction/multi-option stratum vs straight-cruise stratum, paired | Uniform Δ across strata ⇒ the gain isn't strategic |
| **HP-3** | **Route-conditionality: same scene + different `nav_cmd` ⇒ different, correct trajectory.** A flat marginal model *cannot* do this | Counterfactual route swap on identical observation windows; measure trajectory divergence + correctness vs the commanded route | Trajectories identical under different commands ⇒ still a command echo (PC1 regression) |
| **HP-4** | **Compositional generalization to unseen route topologies** | Held-out junction *types* (not just episodes); hierarchy should degrade less | Equal degradation ⇒ no compositional benefit |
| **HP-5** | **Data-efficiency: structure substitutes for data.** The hierarchy should reach a given competence with fewer episodes | Matched-param learning curves, hierarchy vs flat, ≥4 seeds (ties directly to C2) | Identical/worse slope ⇒ structure buys no sample efficiency |
| **HP-6** | **Recovery/re-planning after perturbation** — the strategic level should re-acquire the route where a flat model drifts | Lateral-offset perturbation → measure route re-acquisition rate (reuses the E2a perturbation machinery) | No difference in re-acquisition ⇒ no hierarchical control benefit |

**Admissibility rules** (per `CLAUDE.md`): every Δ carries the **paired episode-cluster bootstrap**
(`taniteval/ci.py`, B=2000) with its estimator named; matched parameters (± 5 %) and matched training
steps; ≥ 4 seeds where a curve is claimed; no exponent without window + R² + n.

**Publication value:** HP-1…HP-6 passing is a *far stronger* result than "our model has lower ADE" —
it is a mechanistic demonstration that hierarchy causes specific, predicted behavioral advantages. This
is the shape of a world-class paper, and it is only available because the hierarchy is the thesis.

### A.4 HPP execution phases

| Phase | Work | Compute | Depends on | Output |
|---|---|---|---|---|
| **HPP-0** *(now, 0 GPU)* | Confound audit: trace `nav_cmd` end-to-end through train + eval; verify what the scored path actually traverses; quantify current `route_skill`; inventory route-decision episodes in the corpus | none | — | `HPP0_CONFOUND_AUDIT.md` — the honest baseline of what is broken |
| **HPP-1** | **Fix the route input** (PC1): real vision-derived route inference; kill the command echo. Likely a label + loss + conditioning fix, not an architecture change | small train | HPP-0 | `route_skill > 0`, `nonav_route_beats_majority` PASS |
| **HPP-2** | **Build the decision-rich eval set** (PC4) + route-compliance metrics (PC3): multi-option junction strata, counterfactual-route pairs for HP-3 | 0 GPU (data) | Tier-1 #1 | `route_eval_v1` + metrics in `taniteval` |
| **HPP-3** | **Pre-register HP-1…HP-6** with both outcomes; wire the assertion for PC2 into the eval harness | 0 GPU | HPP-1, HPP-2 | `HPP_PRE_REGISTRATION.md` |
| **HPP-4** | **The ablation ladder at matched params**: flat → +tactical → +strategic, paired, ≥4 seeds, on the fixed instrument | 1 pod × ~several days | HPP-1..3 | The proof (or a *fair* negative, which is then real information) |

**Honest note kept in the record:** if HP-1…HP-6 run under PC1–PC4 and the hierarchy still shows no
separated advantage, *that* is a real result and must be reported as such. The reframe buys the
hypothesis a **fair test**, not immunity. The difference between this and the withdrawn #6 is that we
now measure the thing under conditions where it *could* show up.

---

## PART B — Execution plan, all tiers

### B.0 Resourcing reality (as of 2026-07-25 ~15:15 UTC, MEASURED)

| Resource | State | Frees |
|---|---|---|
| **pod2** | flagship-v4-fromscratch, step ~19.5k/30k, RAM 53.7/55 GB — **do not touch** | ~21 h → then ckpt backup + formal 8-metric gate |
| **pod1** | flagship-v2corpus, step ~3.1k/30k, ~11 s/step | ~90 h |
| **pod3** | E1b failure-gated CL-SFT (mining→SFT, PID 2161706) | ~1 day → E1b paired eval, then HPP |
| **eval pod** | flagship-v4 mid-train eval (transfer leg-2 in flight) | hours |
| **dev box** | all Tier-1/Tier-3 tooling — **no GPU needed** | now |

**Key scheduling insight:** every Tier-1 and most Tier-3 items need **zero GPU**. They run *now*, in
parallel with the trainings, and they are precisely the items that make later results trustworthy.

### B.1 Wave 1 — NOW, zero GPU, unblocks everything (target: 24–48 h)

| # | Item | Effort | Why first | Done-when |
|---|---|:--:|---|---|
| T1-2 | Route `closedloop.py` headline/compounding/divergence CIs through `ci.episode_cluster_bootstrap` | S (~1 h) | The **most decision-relevant** stats leak; `driving.py` was migrated, this wasn't | Deprecated `_agg`/`_jack` gone from headline paths; test pins the change |
| T1-4 | **`safe_commit.py`** — pathspec-free `-F`, foreign-index abort, `Keys.txt` refuse, auto-clear `index.lock`, retry | S | Every future commit rides it; kills the segfault + sibling-sweep class | Used for all commits from here |
| T1-5 | **`registry_lint.py`** — JSON-pointer re-read + fail-CI-on-drift + **multiline header sweep** vs RETRACTION_LOG phrases | S/M | Kills the 4-day-stale-headline class at the root | CI red on an injected stale headline |
| T1-3 | **Parity by content-hash** — episode-id manifest; every trainer asserts `count==2376` + `sha256(sorted uids)`; test that a truncated cache is refused | M | Silent-truncation hole under the *sacred* invariant | Truncated-cache test goes red→green |
| T3-11 | Land the 1-line dense-path persistence (`taniteval/rollout.py:94`, keeps 4/20) | S | Unmerged since 07-09; unlocks the entire comfort/behavioral axis for ~1 MB/arm | Jerk/curvature/plan-stability computable |
| T3-12 | **`repo_janitor.py`** — worktree prune, dated `incoming/` ledger, future-date flag | S | Restores Glob reliability (a false "stranded" claim already cost a session) | ~90 stale bundles triaged |
| P1/P2 | **Merge the hypothesis ledgers** into ONE living ledger; split H4→H4a/H4b, H1→H1-operative/H1-hierarchy/H1-coupling; add PARKED state | S/M | Two ledgers have diverged; a reader trusting the wrong one gets a stale answer | Single ledger, mandatory columns (see Part C) |

### B.2 Wave 2 — the instrument, before any new verdict (target: 2–4 days)

| # | Item | Effort | Note |
|---|---|:--:|---|
| T1-1 | **Gate primary change**: `corridor_departure_rate @ K=max` as co-primary in `run_gate.py`; ADE@2s demoted to diagnostic | M | **The single highest-leverage correction in the review** |
| T1-1b | **Re-gate v3enc and v4.1** on the corrected primary | M | Both verdicts were rendered on a metric proven horizon-blind; they may be wrong. ⚠️ v3enc's re-gate must also fix the **decorr-never-on** instrument bug (H25) or it re-runs void |
| T3-13 | Doc-layer provenance CI: reject any ADE/departure/comparative number entering LOOP_STATE/reports/registry without `estimator=` + `n=` (n ≥ 12 closed-loop) | S/M | Targets C1/C5/C4 where they actually escape — in prose |
| T3-14 | Split LOOP_STATE + size guard; consolidate the 3×/day cadence with auto-compile | S/M | 122 KB run-on file has shipped stale instructions twice in a day |
| HPP-0 | **Hierarchy confound audit** (Part A) | S/M | 0 GPU; produces the honest baseline for the proof program |

### B.3 Wave 3 — the science (as pods free)

| When | Pod | Item |
|---|---|---|
| E1b done (~1 d) | pod3 | **E1b paired eval** — junction corridor-departure@K185, base vs FT. First real closed-loop verdict on the corrected instrument |
| pod2 at 30k (~21 h) | pod2→eval | Flagship **ckpt backup to HF** + the **formal 8-metric gate** (HF space is free; both unblocked). ⚠️ 3 of 8 kill-secondaries still lack emitters — wire them or the gate cannot render a verdict |
| after HPP-0..3 | pod3 or freed pod | **HPP-4 ablation ladder** (the hierarchy proof) |
| after cooldown 07-26 12:00 UTC | pod1/pod3 | **C2 data-efficiency slope** (H7) — the unmeasured headline. Gentle config, one run |
| opportunistic | eval | **One recognized-benchmark number** (NAVSIM/EPDMS or Bench2Drive) — first external ground truth |

### B.4 Wave 4 — strategic (PI decisions, not agent-autonomous)

| # | Item | Decision needed |
|---|---|---|
| T4-15 | **Reactive-agent renderer: give it an owner + a pod.** The single dependency for safety-grade closed-loop, D5/D6 topology gates, HP-2/HP-6, and any benchmark entry | Sayed: assign + schedule. This is the critical path to every external number |
| T4-16 | **Flagship-vs-REF-C at the v4 30k gate** — with the *corrected* framing: this is no longer "which arm wins" but "does the co-evolved hierarchy beat a flat planner **once PC1–PC4 hold**" | Sayed: confirm the gate reads under HPP conditions |
| — | **REF-A / REF-B retirement from compute** | Sayed: both have delivered their decision-grade signal (frozen ceiling; camera-conditioning refuted). Retiring frees a pod for HPP-4 |
| — | **Dataset tier-R value-add** (8 NC sources) | Sayed: scope. R == C today; suggest nuScenes + Argoverse2 as a 2-source proof before all 8 |

---

## PART C — Full hypothesis portfolio: review + improvement action for EVERY hypothesis

**This is the part missing from the first review.** R3 delivered status + degree-of-achievement; here
each row gets its **concrete improvement action**, owner-type, and cost. Status/DoA from
`R3_hypothesis_portfolio.md` (MEASURED where an artifact exists).

**Legend — action classes:** `PROVE` (build the decisive test) · `FIX` (broken instrument/impl before
retesting) · `MEASURE` (instrument exists, just run it) · `SPLIT` (a settled negative is masking a live
question) · `PARK` (explicit, with revisit-trigger) · `RETIRE` (settled; stop spending attention).

### C.1 Constitutional hypotheses (H0–H15)

| ID | Claim | Status | DoA | **Improvement action** | Cost |
|---|---|---|---:|---|:--:|
| **H0** | E2E > rule-based | Confirmed (premise) | 100% | `RETIRE` — accepted starting point, not our test | — |
| **H1** | **4-brain hierarchy** | Partially; **hierarchy-edge UNTESTED** | 40% | **`SPLIT` + `PROVE` → the entire HPP (Part A).** H1-operative = validated (0.452). H1-hierarchy-edge = the D5/D6 gates that never ran → HPP-4. H1-coupling = H27, in-flight | **L** |
| **H2** | Attention modality steering | Open | 15% | `PARK` {revisit: after HPP-4; Phase-1} — nothing measured on our stack | S |
| **H3** | Latent WM core (LeJEPA/SigReg) | Confirmed as representation | 75% | `MEASURE` — the *raison-d'être* (data-efficiency) is unmeasured → folds into HP-5/C2 slope. Also: SIGReg still "NOT-YET-ADMISSIBLE" → either make it admissible or stop leaning on the LeJEPA corollary | M |
| **H4** | Frozen vs trained encoder | Confirmed (neg) ⚠️ | 85% | **`SPLIT` — the flat "CLOSED NEGATIVE" is UNSAFE.** H4a (frozen + supervised regression) = refuted, RETIRE. **H4b (frozen + feature-prediction + planning) = OPEN and reads POSITIVE (0.599 m)** → this is the real v3 question, keep alive | S (doc) |
| **H5** | Efficient inference / CNCE moat | Confirmed | 80% | `RETIRE` from active work; keep the deployed numbers. Note CNCE is self-defined → if used externally, needs a standard baseline | S |
| **H6** | Opponent weak-spot corpus | Partially | 45% | `FIX` — scenarios are **design-oracle only**; model-side is renderer-gated → depends on T4-15. Until then, cannot yield a model number | M (gated) |
| **H7** | **1000× data (C2 slope)** | Partially; **slope UNMEASURED** | 35% | **`MEASURE` — top priority after cooldown.** ≥300 clips, ≥4 seeds, GeoCalib intrinsics. This is the program's best shot at a novel result | **M** |
| **H8** | MoE beyond sensors | Open (parked) | 5% | `PARK` {revisit: Phase-1} | — |
| **H9** | Rule compliance / hard barriers | Partially | 40% | `FIX`+`MEASURE` — TLC oracle shows rule_barrier 1.0 vs soft 0.0 but that's the **design oracle, not our model**. Needs the renderer (T4-15) for a model-side number. **Traffic-light handling — a specific PI ask — lives here** | M (gated) |
| **H10** | Latent-RAG continual learning | Open (toy) | 20% | `PARK` {revisit: post-HPP; D7 unrun} — known −24 % interference → surprise-gating designed but untested | S |
| **H11** | Self-monitoring w/ guarantees | Partially | 35% | `MEASURE` — D8 AUROC **below gate** (p≈0.047, target >0.85). σ dissipates to chance by k=4 → couple to H22 (shortcut-trained imagination) which targets exactly this measured failure | M |
| **H12** | Text as part, not core | Open | 25% | `PARK` {revisit: Phase-1} | — |
| **H13** | Extraction heads / probes | Confirmed (pattern) | 70% | `RETIRE` from active work — probes ship and work (curvature R² 0.254 vs 0.031 ego-only) | — |
| **H14** | Physical grounding | Partially | 35% | `SPLIT` — narrow Track-1 done (95.9 % physically-shaped). **Broad vision (physical law/ethics/culture injection) untouched** → either scope it as a real work item or PARK explicitly | M |
| **H15** | Imagination of unobserved areas | Partially | 30% | **`MEASURE` + `FIX`** — module fires but `vision_use` flat ~12 %, **D9 hidden-sector driving-gain NEVER ablated**, σ→chance by k=4. Run D9; if the gain is real, fix the k>1 decay via H22 | M |

### C.2 Sayed-added (H16–H18)

| ID | Claim | Status | DoA | **Improvement action** | Cost |
|---|---|---|---:|---|:--:|
| **H16** | Active depth interrogation | Open (Phase-1) | 8% | `PARK` {revisit: Phase-1 ~Sep} — F1–F3 pre-registered, legitimate window | — |
| **H17** | Unified-FOV masked periphery | **Stale-Orphaned** | 5% | `PARK` {reason: no owner 13 d; revisit-trigger: post-HPP} — or schedule with an owner. Leaving it "open" is dead weight | S |
| **H18** | Hierarchical action grounding | Partially (operative Confirmed) | 55% | **`PROVE` — folds into HPP.** Grounding dominance *grew* Δ2.70 m at 30k (MEASURED) — this is **supporting evidence for the hierarchy** and should be an explicit HPP input; extend to tactical/strategic | M |

### C.3 Survey-derived (H19–H24)

| ID | Claim | Status | DoA | **Improvement action** | Cost |
|---|---|---|---:|---|:--:|
| **H19** | Discrete tactical vocabulary → anchor prior | **Confirmed** | 70% | `RETIRE` as a question — realized as REF-C's anchor prior and now the v4 proposal mechanism. **Note: this is a hierarchy-supporting result** (discrete tactical structure helps) → cite in HPP | — |
| **H20** | Plan-persistence bridging (BridgeAD) | Stale-Orphaned | 5% | `PARK` {revisit-trigger: after E1b} — plan-stability is newly measurable once T3-11 lands | S |
| **H21** | Latent RFT (GRPO/WorldRFT) | Open (blocked) | 5% | `PARK` {blocked on renderer T4-15} | — |
| **H22** | Shortcut-trained imagination (DreamerAD) | Stale-Orphaned | 5% | **`UN-PARK` — the most worth reviving.** It targets a **MEASURED** failure (σ-dissipation to chance by k=4) that caps H15 *and* H11. Schedule after HPP-0 | M |
| **H23** | Interpretable cost-map decode head | Open (blocked) | 5% | `PARK` {blocked on BEV pseudo-labels} | — |
| **H24** | Oracle-gap curriculum (ACID + CTRV floor) | Stale-Orphaned | 10% | `PARK` or run — was ranked **#1 do-next on 07-17** and never ran; the v2 corpus used *selection-balancing*, a **different mechanism**, so H24 as specified is still untested. Decide explicitly | M |

### C.4 Flagship-line (H25–H28)

| ID | Claim | Status | DoA | **Improvement action** | Cost |
|---|---|---|---:|---|:--:|
| **H25** | Vision-decoupling (ego re-encoding) | **Confounded** | 25% | **`FIX` then re-run, or RETIRE.** The entire gate window had **decorr measured NEVER-ON** — a void instrument, so v3enc was never actually tested. ⚠️ **This also means the v3enc "RESTART" verdict is doubly suspect** (void instrument *and* horizon-blind primary) | M |
| **H26** | Hierarchical cross-alignment | Partially (confounded history) | 30% | **`FIX`+`PROVE` → HPP.** 1/3 seams load-bearing; intent→operative *harmful*; nav→strategic pure echo (`route_skill 0.0`). **The broken seams are HPP-1's work list.** Downgrade the "core-goal proof" headline to "1/3 seams, mechanism open" until HPP-4 | L |
| **H27** | Planner–WM coupling = warm-start artifact | Partially (supported, in-flight) | 55% | **`MEASURE`** — canary descends under full coupling (λ=1.0, WM loss 2.10 and falling, MEASURED today). **Finish to 30k + run the FORMAL gate**; 3 of 8 kill-secondaries have no emitter → wire them or no verdict is renderable | M |
| **H28** | Frozen-WM residual is aleatoric | Confirmed (neg) | 85% | `RETIRE` as a contender; keep frozen-WM as the honest ~0.60 m fallback. CEM/learned-value retired as a product path | — |

### C.5 Implicit hypotheses (IMP-1…IMP-8)

| ID | Claim | Status | DoA | **Improvement action** | Cost |
|---|---|---|---:|---|:--:|
| **IMP-1** | Speed-input fix | **Confirmed** | 95% | `RETIRE` — the program's strongest single positive (+2.21 m [2.04,2.39] causal). **Write it up**; it is publishable as a clean mechanism result | S |
| **IMP-2** | Supervised heads are a lossy WM readout | **Confirmed** | 90% | **Feed into HPP** — this is *why* the hierarchy must be evaluated through planning, not heads (+2.257 m open-loop; 38 % less closed-loop drift). Directly motivates PC2 | — |
| **IMP-3** | Open-loop ⊥ closed-loop | **Confirmed** | 90% | `RETIRE` as a question; **enforce as policy** — no open-loop-only claim may be called "drives" (0.452 open → 1.488 closed, triple-confirmed n=1→12→40) | S |
| **IMP-4** | Closed-loop is improvable | **Open (REOPENED today)** | 25% | **`MEASURE` — E1b running now.** E2a localizes the fix to the training objective (offset perceivable R²=0.72; 91 % downstream). Paired eval on the next drumbeat | M |
| **IMP-5** | Branch-B camera-conditioning | **Refuted** | 100% | `RETIRE` — clean negative at power. **Publishable negative** | — |
| **IMP-6** | INT8 deployment precision | **Refuted** | 100% | `RETIRE` — clean negative | — |
| **IMP-7** | Recovery-augmentation lever | **Refuted** | 100% | `RETIRE` — reversed at n=40. Keep the machinery + the 2 measurement lessons | — |
| **IMP-8** | Kinematic dataset-balancing (v2 50 h) | **Open** | 20% | `MEASURE` — corpus built, **QA pending, UNMEASURED on driving**; pod1 arm running now → gate at 30k. Note "kinematic selection cannot buy semantic scenarios" is a real limit | M |

### C.6 Portfolio actions summary

| Action | Count | IDs |
|---|---:|---|
| **PROVE** (HPP) | 4 | H1, H18, H26, (+H19/IMP-2 as supporting inputs) |
| **MEASURE** (instrument exists — just run) | 7 | H3, H7, H11, H15, H27, IMP-4, IMP-8 |
| **FIX** (broken instrument before retest) | 4 | H6, H9, H25, H26 |
| **SPLIT** (negative masking a live question) | 3 | H1, H4, H14 |
| **PARK** (explicit + revisit-trigger) | 9 | H2, H8, H10, H12, H16, H17, H20, H21, H23, H24 |
| **UN-PARK** (targets a measured failure) | 1 | **H22** |
| **RETIRE** | 9 | H0, H5, H13, H19, H28, IMP-1, IMP-3, IMP-5, IMP-6, IMP-7 |

**What this reveals:** the portfolio's live surface is far smaller than it looks — **~11 hypotheses
genuinely need work**, and 9 are retire-able today. Nine more are parked but presented as "open,"
which is what makes the program feel spread thin. Cleaning this is a half-day of doc work that
materially sharpens focus.

---

## PART D — Sequenced summary (what happens, in order)

1. **Now, 0 GPU:** Wave 1 (T1-2, T1-4, T1-5, T1-3, T3-11, T3-12, ledger merge) + **HPP-0 confound audit**.
2. **Next 2–4 days:** Wave 2 — **gate primary change**, then **re-gate v3enc + v4.1**; provenance CI; LOOP_STATE split.
3. **As pods free:** E1b paired eval → flagship 30k backup + formal gate → **HPP-1..3** → **C2 slope** after cooldown.
4. **Then:** **HPP-4 — the hierarchy ablation ladder under PC1–PC4**, i.e. the proof.
5. **PI decisions:** renderer owner (T4-15), REF-A/B retirement, dataset-R scope.

**The through-line:** fix the instruments *first* (they are cheap and they are why the current evidence
is ambiguous), then run the experiments that can actually see the effects we care about. The hierarchy
is not on trial — **the measurement apparatus is**, and it is what we are fixing.
