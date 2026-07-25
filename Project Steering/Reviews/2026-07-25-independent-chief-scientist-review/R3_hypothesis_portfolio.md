# R3 — Independent Hypothesis-Portfolio Audit (TanitAD)

**Auditor role:** independent research auditor (read-only, no compute).
**Date:** 2026-07-25 · **Scope:** the entire hypothesis portfolio H0–H28 + the implicit hypotheses
the program actually tested.
**Primary sources used:** `Project Steering/MODEL_REGISTRY.md`, `Project Steering/PROGRAM_OVERVIEW.md`,
`Project Steering/Mission Plan.md`, `Project Steering/RETRACTION_LOG.md`,
`TanitAD Research Hub/HYPOTHESIS_LEDGER.md`,
`TanitAD Research Hub/Project Steering/Research/2026-07-17-external-survey-derivation.md`,
raw eval JSON under `taniteval/results/`, and the in-flight evidence under
`.../Implementation/incoming/2026-07-25-closedloop-horizon-and-shift/E1a_E2a_RESULTS.md`.
**Evidence class on every status:** MEASURED (+path) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.

---

## 1. Executive verdict

The portfolio is **rich, honestly tracked, and lopsided**: the program has *settled a large number
of NEGATIVES cleanly and expensively* (frozen-encoder ceiling H4, camera-conditioning Branch B,
INT8, the recovery-augmentation lever, the CEM/learned-value "contender") — this is real epistemic
progress and roughly half the portfolio's delivered value. But **every POSITIVE core edge that the
bet depends on is either untested at the deciding instrument or currently counter-indicated**:

- **C1 "it drives" is open-loop only.** The 4-brain clears every trivial floor open-loop (0.452 m,
  MEASURED) but **loses closed-loop to a 104 M non-hierarchical diffusion arm** (1.488 vs 0.564,
  n=40, MEASURED). The closed-loop "BOUND" verdict was **overturned today** (2026-07-25) as
  horizon-confounded — at a realistic 18.5 s horizon the deployed arm departs the corridor on
  59–84 % of windows (E1a, MEASURED). Closed-loop competence is **reopened, not delivered**.
- **C2 "1000× less data" — the headline slope is entirely unmeasured.** H7's enabling *mechanism*
  has first end-to-end evidence (pilot ≈92 % of ceiling, DIRECTIONAL), but the actual
  data-efficiency slope — the PRIO-2 claim — has never been produced, and the scale-up is
  cooldown-blocked.
- **The hierarchy's net driving advantage (H1) is untested.** The constitutional D5/D6 topology
  gates have **never run**; the only head-to-head evidence (the open-loop tie + closed-loop loss to
  REF-C) currently argues the 4-brain buys nothing over a much smaller flat arm on this corpus.

Two systemic risks stand out from the record itself. (1) **Premature certainty is the dominant
failure mode:** `RETRACTION_LOG.md` shows **five** "this direction is closed/bound" verdicts reopened
by ~$0 follow-ups in a single session. (2) **The ledgers have diverged:** `HYPOTHESIS_LEDGER.md`'s
status table is frozen at 2026-07-05 while `PROGRAM_OVERVIEW.md` carries the live statuses — a
reader who trusts the wrong one gets a stale answer.

**Overall weighted hypothesis-completion estimate: ≈40 %** (claim-delivery basis) / **≈60 %**
(decision-relevance basis). Weighting stated in §3.1.

---

## 2. Master hypothesis table

Status vocabulary: **Confirmed** / **Refuted** / **Confounded** / **Partially** / **Open** /
**Stale-Orphaned**. "Confirmed (neg)" = the hypothesis was tested and the answer is a clean negative.
Degree-of-achievement (DoA) = how far the question is *decided or the claim delivered*, 0–100 %.

### 2.1 Constitutional hypotheses (Mission Plan H0–H15, verbatim source: `Mission Plan.md`)

| ID | Hypothesis (short; source `Mission Plan.md` lines) | Status | DoA | Conf | Deciding evidence (artifact) |
|---|---|---|---:|:--:|---|
| **H0** | E2E > rule-based; "won't attack this fact" (L36) | Confirmed | 100% | H | PUBLISHED premise; ledger `confirmed`. Not a TanitAD test — an accepted starting point |
| **H1** | 4-brain hierarchy: strategic/tactical/operative + fallback-MRC (L38–44) | **Partially** (driving-edge Open) | **40%** | M | Operative MEASURED 0.452 (`driving_flagship-30k.json`); heads-as-decision-makers **falsified** (head 3.38 > CV); **D5/D6 topology gates NEVER RUN** (`Phase 0 Plan.md §4`); 104 M REF-C ties/beats it (`MODEL_REGISTRY §6`) |
| **H2** | Attention-based modality steering (L56) | Open | 15% | M | PUBLISHED support (DriveMoE/GEMINUS); Phase-0 exit demo unbuilt; nothing MEASURED on our stack |
| **H3** | Latent world model core, LeJEPA/SigReg (L58) | **Confirmed** (as representation) | **75%** | H | Vision effect **+1.325 m [+1.04,+1.64] CI-sep** (`driving_flagship-30k.json`); frozen WM as differentiable simulator 0.599 within noise of oracle 0.4045. *Caveat: data-efficiency raison-d'être unmeasured; loses closed-loop* |
| **H4** | Frozen vs trained encoder (L60) | **Confirmed (neg)** ⚠️ | **85%** | H | REF-A dyn-in **2.9196**, monotone 5k→30k (`eff_refa-dynin-30k.json`, MEASURED). **⚠️ UNSAFE as flat "closed": the same frozen latent → 0.599 through its dynamics — see §4** |
| **H5** | Efficient inference transfer / CNCE moat (L62) | **Confirmed** (deployed arm) | **80%** | H | CNCE median **210,551**, tick **18.75 ms p50** (`eff_flagship-30k.json`). Note retracted "11.16 ms" (RETRACTION_LOG C1/C6) |
| **H6** | Opponent weak-spot corpus (L64) | Partially | 45% | M | Scenarios shipped (Stop-Arm, Work-Zone, Stationary-Lead, SC-14) — **design-oracle only**, model-side renderer-gated (`SCENARIO_DATABASE.md`) |
| **H7** | 1000× data via IDM + focal canonicalization (L66) | **Partially** | **35%** | M | Pilot **≈92 % of ceiling**, pseudo-label ≈96 % (`PROGRAM_OVERVIEW §5.2e`, MEASURED but **DIRECTIONAL**: 80 clips/3 seeds/unknown intrinsics, yaw≈0). **The C2 slope is UNMEASURED**; scale-up cooldown-blocked (RETRACTION_LOG 07-25) |
| **H8** | MoE beyond sensors (L68) | Open (parked) | 5% | L | Prio-2, interface ready, untested |
| **H9** | Inherent rule compliance / hard barriers (L70) | Partially | 40% | M | SC-14 TLC design oracle **rule_barrier 1.0 vs soft_prior 0.0** — DESIGN ORACLE, not our model; model-side renderer-gated |
| **H10** | Latent-RAG continual learning (L72) | Open (toy) | 20% | L | Validated-toy w/ known −24 % interference → surprise-gating; **D7 not run** |
| **H11** | Self-monitoring w/ guarantees (L74) | Partially | 35% | M | D8 **preview only, p≈0.047** (AUROC>0.85 **not reached**); σ-dissipation caps at 1-step (MEASURED, ledger 07-17/18); new WM-integrity canary in daily use |
| **H12** | Text as part, not core (L76) | Open (supported) | 25% | L-M | PUBLISHED 1 B LLM bridge; command-conditioning only; not integrated-measured |
| **H13** | Extraction heads / probes (L78) | Confirmed (pattern) | 70% | M | Trajectory/BEV/curvature probes ship; curvature R² 0.254 vs 0.031 ego-only (`driving_flagship-30k.json`) |
| **H14** | Physical grounding (L80) | Partially | 35% | M | **Narrow** Track-1 done (kinematic bicycle + Kamm; 95.9 % physically-shaped paths MEASURED). **Broad vision** (physical laws/ethics/culture injection) untouched |
| **H15** | Imagination of unobserved areas (L82) | **Partially** | **30%** | M | Module live+firing (22 M, MEASURED); **`vision_use` flat ~12 %**; **D9 hidden-sector driving-gain NEVER ablated**; σ dissipates to chance by k4 |

### 2.2 Sayed-added hypotheses (ledger H16–H18)

| ID | Hypothesis (source `HYPOTHESIS_LEDGER.md`) | Status | DoA | Conf | Deciding evidence |
|---|---|---|---:|:--:|---|
| **H16** | Active depth interrogation, σ-triggered ROI depth (07-11) | Open (Phase-1 deferred) | 8% | L | Dossier `H16_ACTIVE_DEPTH_INTERROGATION.md`; F1–F3 pre-registered; **no experiment** — legit Phase-1 window (~Sep) |
| **H17** | Unified-FOV masked-periphery training (07-12) | **Stale-Orphaned** | 5% | L | Dossier only; **no experiment in 13 days**, no scheduled owner |
| **H18** | Hierarchical action grounding (07-12) | Partially (operative Confirmed) | 55% | M-H | Operative grounding shipped; **grounding dominance grew Δ 2.70 m at 30k** (hierarchy panel, MEASURED); tactical/strategic extension not done |

### 2.3 External-survey-derived proposals (H19–H24, `2026-07-17-external-survey-derivation.md §2`)

All six were PROPOSED with pre-registered falsifiers+gates. Only H19 has been realized/validated.

| ID | Hypothesis (verbatim short) | Status | DoA | Conf | Deciding evidence |
|---|---|---|---:|:--:|---|
| **H19** | Discrete tactical vocabulary → **anchor prior** (VQ/LAMP → anchored decoders) | **Confirmed** | 70% | M-H | Realized as REF-C anchor-prior graft; **REF-C base 0.4523 ties flagship** (`MODEL_REGISTRY §6`); now the flagship v4 proposal mechanism (256 anchors). *Note: PROGRAM_OVERVIEW relabels H19 "maneuver→anchor prior" — evolved from the original VQ-codebook wording* |
| **H20** | Plan-persistence bridging (BridgeAD) | **Stale-Orphaned** | 5% | L | Proposed, ranked #3 do-next; **no experiment** |
| **H21** | Latent RFT (GRPO/WorldRFT) | Open (blocked) | 5% | L | Gated on CARLA/renderer; prep not started; **untested** |
| **H22** | Shortcut-trained imagination (DreamerAD) | **Stale-Orphaned** | 5% | L | Fixes a **MEASURED** problem (σ-dissipation); ranked #4 do-next; **no experiment** |
| **H23** | Interpretable cost-map decode head (PLAN-S) | Open (blocked) | 5% | L | Blocked on BEV pseudo-labels (Cosmos-DD/PandaSet); untested |
| **H24** | Oracle-gap curriculum (ACID + CTRV floor) | **Stale-Orphaned** | 10% | L | Ranked **#1 do-next 07-17**; **not run** — the v2 corpus used *selection-balancing*, a different mechanism, so H24 as specified is still untested |

### 2.4 Flagship-line hypotheses (H25–H28)

| ID | Hypothesis (source) | Status | DoA | Conf | Deciding evidence |
|---|---|---|---:|:--:|---|
| **H25** | Vision-decoupling — encoder redundantly re-encodes ego dynamics (ledger 07-18) | **Confounded** | 25% | L-M | v3enc family did not recover speed probe (0.393 vs v1 0.861) **BUT decorr was measured NEVER-ON** during the gate window (RETRACTION_LOG C3, `postmortem_b_egodropout_v3enc10k.json`) → **under-tested, not refuted**. `vision_use` still ~12 % |
| **H26** | Hierarchical cross-alignment = core-goal proof (Sayed 07-18) | **Partially** (confounded history) | 30% | M | **1/3 seams load-bearing** at 30k (ctx→tactical +0.044 CI-sep, MEASURED hierarchy panel); intent→operative harmful; **nav→strategic pure command-echo (route_skill 0.0)**. The "strategic=2 % lever" read was itself **confounded** (RETRACTION_LOG 07-21) |
| **H27** | Planner–WM coupling failure is a **warm-start artifact** (new 07-23) | **Partially** (supported, in-flight) | 55% | M | Seam **cos +0.0043** orthogonal (MEASURED, n=512); 4 warm-start arms degrade WM (`flagship-v4.1-10k.json` etc.); random-init canary **descends** under full coupling. **⚠️ in-loop, formal gate deferred, ~40 % of run** |
| **H28** | Frozen-WM planner residual is **aleatoric**, not capacity/search (new 07-24) | **Confirmed (neg)** | 85% | H | 11× scaling flat 0.599→0.601→0.599 (none sep); learned-value search **worse** 1.016 (`MODEL_REGISTRY` D1; RETRACTION_LOG C6). Frozen-WM = ~0.60 m **fallback**, not contender |

### 2.5 Implicit hypotheses the program actually tested (not formally numbered)

| ID | Implicit hypothesis | Status | DoA | Conf | Deciding evidence |
|---|---|---|---:|:--:|---|
| **IMP-1** | **Speed-input fix** (v0 as a 3rd action channel recovers ego-dynamics) | **Confirmed** | 95% | H | REF-A fwd-ADE 3.73→0.83; no-speed 2.918 vs speed 0.452 **causally +2.21 m [2.04,2.39]** (`eff_flagship-nospeed.json` vs `eff_flagship-speed.json`). The program's strongest single positive |
| **IMP-2** | **Supervised heads are a lossy readout of a good WM** | **Confirmed** | 90% | H | P2 CEM planner beats head **+2.257 m open-loop**, drifts 38 % less closed-loop (`MODEL_REGISTRY §6`, `PROGRAM_OVERVIEW §7`) |
| **IMP-3** | **Open-loop ⊥ closed-loop** (open-loop ADE does not predict closed-loop) | **Confirmed** | 90% | H | 0.452 open → **1.488 closed** n=40 (`PROGRAM_OVERVIEW §5.2d`); triple-confirmed n=1→n=12→n=40 |
| **IMP-4** | **Closed-loop is improvable** (was "BOUND") | **Open (REOPENED today)** | 25% | M | **BOUND overturned 2026-07-25**: corridor-departure 0.35 %→59 % (18.5 s horizon), junction 84 %; **fix is a training-objective one** (E2a: offset perceivable R²=0.72, 91 % downstream loss). Renderer-free E1b pre-registered (`E1a_E2a_RESULTS.md`, MEASURED) |
| **IMP-5** | **Branch-B from-scratch camera-conditioning** (GAIA-2 style) | **Refuted** | 100% | H | Cross-rig speed R² **−0.667 vs frozen v1 +0.657**, paired CI excludes 0 on 3/4 arms (`PROGRAM_OVERVIEW §5.2e`) |
| **IMP-6** | **INT8 is a viable deployment precision** | **Refuted** | 100% | H | W+A INT8 collapses readout head to cos 0.566, +0.0215 m over 20 steps, no latency win (`PROGRAM_OVERVIEW §2③`) |
| **IMP-7** | **Recovery-augmentation halves closed-loop departures + generalizes** | **Refuted** | 100% | H | n=12 +0.0089 **reverses** to n=40 **−0.0302 [−0.0595,−0.0088]** (departs 3.3× more) (RETRACTION_LOG 07-24) |
| **IMP-8** | **Kinematic dataset-balancing (v2 50 h) lifts driving** | **Open** | 20% | L | Corpus built (key `physicalai-v2bal-4b7eeeac222d`, turns 14→28 %) but **QA pending, UNMEASURED on driving**; "kinematic selection cannot buy semantic scenarios" (`PROGRAM_OVERVIEW §5.2f`) |

---

## 3. Portfolio synthesis

### 3.1 Overall completion estimate + weighting

**Weighting rule (stated):** each hypothesis is weighted by its contribution to the three undeniable
claims plus embedded efficiency, with the two **PRIO goals** (data-efficiency C2, safety-by-design
C3) and **C1-drives-closed-loop** carrying the most weight; Phase-1/prio-2 items (H8, H10, H12,
H16, H20–H24) carry little.

| Claim bucket | Constituent hypotheses | Delivered |
|---|---|---|
| **C1 — it drives (closed-loop)** | H1, H3, H14, IMP-3/IMP-4 | **~30 %** — open-loop cleared; closed-loop competence reopened; hierarchy-edge untested |
| **C2 — magnitudes less data (PRIO)** | H3, H4, H7 | **~25 %** — mechanism evidenced; **slope unmeasured** |
| **C3 — inherently safe/compliant (PRIO)** | H9, H11, H15, H14 | **~30 %** — instruments built; model-side numbers renderer-gated; D8 preview |
| **Efficiency (embedded)** | H5, H2, H8 | **~80 %** — CNCE, tick budget, INT8 all decided |

**Weighted overall ≈ 40 % (claim-delivery).** But the program's *epistemic* completion — "have we
learned what we need to steer correctly?" — is markedly higher (**≈60 %**): the negatives are clean,
the failure axis (longitudinal) is triangulated, and the crux (planner-WM coupling is a warm-start
artifact, not an objective conflict) is answered directionally. The gap between 40 % and 60 % is the
distance between *"we know what to do"* and *"we have proven the edge."*

### 3.2 The 3–5 highest-value hypotheses still OPEN or UNDER-TESTED (best next GPU-day)

1. **C2 data-efficiency slope (H7).** The single headline claim of the entire program that has
   **never been measured**, and it is a PRIO goal. The mechanism is de-risked; the missing piece is a
   decision-grade slope (≥300 clips, ≥4 seeds, GeoCalib intrinsics). Currently cooldown-blocked on
   the live source — highest value, real dependency.
2. **Closed-loop competence via failure-gated CL-SFT (IMP-4 / E1b).** Today's E1a/E2a result makes
   this the **cheapest high-impact experiment on the board**: the fix is a *training objective*
   (offset is perceivable at R²=0.72, planner ignores it — 91 % downstream loss), renderer-free, and
   pre-registered. Directly attacks C1 and the closed-loop loss to REF-C.
3. **Planner–WM co-evolution to 30k + a *formal* gate (H27 / H1).** The crux is answered in-loop but
   **not gated** — and 3 of 8 kill-secondaries still have no emitter, so the gate literally cannot
   render a verdict (`v1_g1_dryrun_gate.json` found this). Finishing this converts the program's
   central claim from "in-flight" to "measured."
4. **Hierarchy net-driving advantage (H1 D5/D6 + H26).** The constitutional core claim — that the
   4-brain hierarchy *drives better* than a flat arm at matched params — is **untested**, and the one
   available datapoint (a 104 M non-hierarchical arm ties open-loop and **beats** closed-loop) argues
   against it. This must be measured on the right instrument or the central thesis is unsupported.
5. **Longitudinal control (cross-cutting; underlies H1/H3/C1).** The most-triangulated weakness in
   the program (open-loop high-speed stratum, every v4 arm's failure axis, and the *entirety* of the
   n=40 closed-loop deficit). Instruments already exist (`pathspeed.py`, corridor+band-ADE).

*(Runners-up: H15 D9 driving-gain ablation and H11 D8 AUROC — under-tested but lower marginal value.)*

### 3.3 Hypotheses to RETIRE or formally PARK (stop consuming attention)

- **H4a (frozen encoder + supervised regression)** — settled negative (2.92 m, monotone). Stop
  relitigating; **keep H4b (frozen + feature-prediction + planning) OPEN** — it is the actual v3
  question and it reads *positive* (0.599 m).
- **CEM / learned-value search as a contender (part of H28)** — settled negative (aleatoric wall +
  the deployable search is worse). Retire as a *product path*; keep the frozen-WM planner only as the
  ~0.60 m fallback it is.
- **Branch-B camera-conditioning (IMP-5), INT8 (IMP-6), recovery-augmentation lever (IMP-7)** — all
  cleanly refuted at power. Retire.
- **H25 v3enc lever family** — either re-run with **decorr actually ON** (the whole family's gate
  window had it off — a broken instrument) or retire it; as-is it occupies "open" with a known-void
  test.
- **Orphaned survey proposals H20, H22, H24 (and H17)** — either schedule with an owner+date or move
  to an explicit **PARKED** state with a revisit-trigger. Leaving them "open" indefinitely is how the
  ledger accretes dead weight. (H22 is notable: it targets a *measured* failure, so it is the one
  most worth un-parking.)

### 3.4 Verdicts I believe are UNSAFE vs the actual evidence

1. **"H4 CLOSED NEGATIVE" (unqualified).** UNSAFE as a flat statement: the same frozen latent
   supports **0.599 m through its dynamics** (D1 planner). The safe form is the split the 360-review
   already proposed (W-A5): **H4a refuted / H4b open-and-positive**. `PROGRAM_OVERVIEW` has begun this
   ("re-localized"); `HYPOTHESIS_LEDGER.md` still carries the flat "CLOSES NEGATIVE."
2. **"Closed-loop improvement is BOUND / closed."** Was UNSAFE — **overturned 2026-07-25** as
   horizon-confounded (2 s metric on an 18 s event). Correctly retracted; any doc still carrying
   "BOUND" is now stale and should be swept.
3. **H26 "hierarchical cross-alignment = the core-goal proof."** The *headline over-reaches the
   evidence*: only **1/3 seams** are load-bearing, nav→strategic is pure command-echo (route_skill
   0.0 — zero vision route inference), and the "strategic is a 2 % lever" reading was itself
   **confounded** (RETRACTION_LOG 07-21). The honest status is "1/3 seams, mechanism open," not a
   proof of the core goal.
4. **Any framing that the 4-brain hierarchy's *driving advantage* is demonstrated (H1).** "Operative
   validated" is fair; a *hierarchy-beats-flat* claim is not — D5/D6 are unrun and REF-C
   (non-hierarchical) ties/beats the flagship on the measured axes.
5. **Data-integrity bug feeding hypothesis evidence:** `MODEL_REGISTRY.md:1737` states
   `physicalai-val-f1b378f295ae` is "episode-disjoint" from the parity train set; a byte-level check
   measures **78.5 % overlap** (`E1a_E2a_RESULTS.md §1.1`, MEASURED). The split is already refused in
   code, but the registry line is wrong and any historical number computed on it inherits the leak — a
   correction is owed.

---

## 4. Concrete proposals for hypothesis MANAGEMENT going forward

**P1 — One living ledger, mandatory columns, auto-stamped.** `HYPOTHESIS_LEDGER.md`'s status table is
frozen at 2026-07-05 while `PROGRAM_OVERVIEW §3` carries the live statuses — they have visibly
diverged (e.g. H4, H25–H28 exist only in the overview). Merge to a **single** ledger where every row
carries: `status · evidence-class · deciding-artifact-path (or "untested") · gate · last-retested-date
· owner`. A status with no MEASURED artifact path or explicit "untested" is inadmissible. This kills
the "which doc do I trust?" failure directly.

**P2 — Split multi-recipe hypotheses; give orphans an explicit PARKED state.** Several hypotheses hide
an open sub-claim behind a settled one: **H4 → H4a (refuted) / H4b (open)**; **H1 → H1-operative
(validated) / H1-hierarchy-edge (untested) / H1-planner-coupling (in-flight)**. Splitting prevents a
clean negative from masking the live question. Symmetrically, replace the permanent "open" of
un-owned proposals (H17, H20, H22, H24) with **PARKED {reason, revisit-trigger}** so the portfolio's
true active surface is visible.

**P3 — Make "closed/bound/proven" an inadmissible same-session verdict without a horizon+power+reopen
check.** The record is unambiguous: **five** closure claims were reopened by ~$0 follow-ups in one
session (RETRACTION_LOG 07-24/25), the latest because a verdict inherited a 2 s metric's horizon on an
18 s event. Require any "closed" status to name (a) the metric's **horizon and n**, (b) the
**estimator** (episode-cluster bootstrap, per CLAUDE.md), and (c) a **pre-registered cheapest-reopening
check that was run and did not reopen it**. Until that check exists, the admissible status is
"directional," not "closed."

---

### Appendix — status counts (37 hypotheses audited)

| Status | Count | IDs |
|---|---:|---|
| **Confirmed** (incl. 2 confirmed-negative, 1 published premise) | **10** | H0, H3, H4(neg), H5, H13, H19, H28(neg), IMP-1, IMP-2, IMP-3 |
| **Refuted** | **3** | IMP-5, IMP-6, IMP-7 |
| **Confounded** | **1** | H25 |
| **Partially** | **10** | H1, H6, H7, H9, H11, H14, H15, H18, H26, H27 |
| **Open** | **9** | H2, H8, H10, H12, H16, H21, H23, IMP-4 (reopened), IMP-8 |
| **Stale-Orphaned** | **4** | H17, H20, H22, H24 |

*Auditor's note on method:* every DoA and status is triangulated to a MEASURED artifact where one
exists; where the deciding instrument has not been run (D4/D5/D6/D7/D9, the C2 slope, the formal v4
gate) the row says so explicitly rather than inferring a status from prose — which is the same
discipline the program's own operating standard (CLAUDE.md §Operating-standard) demands.
