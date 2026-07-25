# TanitAD — Independent Chief-Scientist Review

**Date:** 2026-07-25 · **Commissioned by:** Sayed (PI) · **Scope:** whole program — code, approach, strategy,
achievements, hypothesis portfolio, workflows/automation/documentation.

**Method.** Six independent reviewers, each on a distinct lens, each mandated to be a skeptic and to ground
every claim in **primary sources** (the code, raw eval JSON, `MODEL_REGISTRY.md`, `RETRACTION_LOG.md`) —
never the program's own prose, because *quoting a faster-moving source than the eval harness is the program's
#1 documented error class*. Reports: `R1_code_engineering.md`, `R2_scientific_approach_architecture.md`,
`R3_hypothesis_portfolio.md`, `R4_measurement_results_rigor.md`, `R5_strategy_research_management.md`,
`R6_workflow_automation_ops_docs.md`. This file synthesizes them; the numbered findings live there with
`file:line` / JSON paths. Reviewers were read-only and added zero load to the two live training runs.

---

## 1. Executive verdict

**TanitAD is world-class *process* wrapped around an *unproven thesis*, and it is currently measuring the
wrong thing on a corpus that cannot test the thesis.** The measurement infrastructure, the evidence-class
culture, the root-cause retraction discipline, and several clean negative results are genuinely top-decile —
better than most funded labs. But the program's **distinctive scientific bets are its least-supported ones**,
and the single instrument that decides every go/no-call (open-loop ADE@2s) was proven four days ago to be
blind to the dominant failure mode.

The one sentence a reviewer cannot un-see: **the control is beating the treatment.** A flat, 104 M-parameter,
no-world-model, no-hierarchy reference arm (REF-C-base, essentially a re-implementation of published
DiffusionDrive) **statistically ties the 263 M hierarchical flagship open-loop and beats its deployed planner
closed-loop (0.564 vs 1.488, n=40, MEASURED).** Everything the program claims as novel — the 4-brain
hierarchy, imagination-in-the-loop, self-monitoring, "structure beats scale at sub-300 M" — is either
untested at the deciding instrument or currently reads neutral-to-negative.

This is **not** a failing program. It is a program that has spent its first phase building an *exceptionally
honest measurement apparatus and clearing the field of dead ends* — and now has to point that apparatus at
the three questions that actually decide whether it produces a world-class result. None of the three has been
answered yet, and two have never been measured.

**The single most important action:** stop deciding on ADE@2s, and force the three thesis-defining
measurements — *(a)* hierarchy vs a flat planner-over-WM, *(b)* the data-efficiency slope, *(c)* one
recognized external benchmark number. Details in §10–§11.

---

## 2. Achievement scorecard

| Dimension | Grade | One-line justification |
|---|:--:|---|
| Measurement infrastructure (the `ci.py` / gate / retraction machinery) | **A−** | Correct, provenance-stamped, reproduces every published mean digit-for-digit |
| Scientific process & rigor (evidence classes, pre-registration, clean negatives) | **A−** | Rare discipline; ~half the delivered value is expensively-settled negatives |
| Code — library core (`ci.py`, data contract, epcache, leakage guards) | **A−/B+** | Strong, unusually well-tested |
| Code — trainer / orchestration layer | **C** | 13 trainers / ~7.9 kLOC, window class copy-pasted 4×, no training determinism, parity by path-substring |
| Measurement **discipline-in-use** | **C+** | The gate primary is a metric proven horizon-blind; prose over-runs verification |
| Core-thesis validation (hierarchy / imagination / sub-300 M) | **D+** | Control ties/beats treatment; differentiators unvalidated or below gate |
| Strategy & prioritization | **C+** | Coherent bet, but the headline claim is unmeasured and the critical-path build is unowned |
| Workflow / automation / documentation | **B−** | Guardrails-rich, automation-poor; recurring tax + re-derivation + rescue load |
| **Overall program maturity** | **B−** | *Bimodal:* A− process/tooling around a D+ thesis-validation. The average hides the story. |

**Progress toward the stated vision:** ~15–20 % (per the architecture review) toward the literal "sub-300 M
hierarchical WM that drives"; ~30–35 % toward a re-scoped, defensible research thesis. **World-class works
delivered today: 0. Credible paths to one, if executed: 1–2** (data-efficiency; planner/closed-loop).
**Hypothesis portfolio:** ~40 % by claim-delivery, ~60 % by decision-relevance.

---

## 3. The central finding — world-class process, unproven thesis

Three sub-findings, all `MEASURED`, all independently surfaced by ≥3 of the 6 reviewers.

### 3.1 The control beats the treatment, and the hierarchy is unpaid-for
REF-C-base (104 M, DiffusionDrive re-impl) is **paired-tied** with the 263 M flagship open-loop and **beats
its head closed-loop** (0.564 vs 1.488). The 3-level hierarchy has **1 of 3 seams load-bearing** (one
actively *harmful*), and the deployed 0.452 m number **bypasses all three brains** — i.e. ~22–57 M of
parameters carry no measured benefit. The evidence supports *"planning-over-a-world-model beats supervised
heads"* (via the flat P2 planner) — it does **not** support *"three levels beat one,"* which no arm isolates.
The program's own architecture doc concedes the deployed number "structurally excludes its hierarchy."
→ *This is the crux of the whole review.* Either the hierarchy earns a CI-separated win in a paired test, or
the headline contribution should be reframed as *planning-over-WM + data-efficiency*, and the parameters
reallocated.

### 3.2 The differentiating edges are the least validated
- **Imagination (H15)** is temporally *anti-calibrated*: confidence *rises* as blind-rollout fidelity decays
  to chance by k=4. Reproduces on the shipping model. Usable at 1 step only.
- **Self-monitoring (D8)** AUROC is below its gate (p≈0.047).
- **SIGReg / LeJEPA** is still "NOT-YET-ADMISSIBLE," so the optimal-planning corollary the theory leans on is
  withheld, not established.

### 3.3 The corpus cannot test the thesis
The training/eval corpus is **74 % straight-driving, 0 % semantic events, ~473 K frames** — 1–2 orders of
magnitude below the multimodality threshold DiffusionDrive-class planners need — and a **2-parameter CTRV
constant-turn oracle tops the leaderboard.** On a corpus this easy, "sub-300 M is enough" is
**unfalsifiable, not proven**, and a no-perception baseline can score well (the program's own AD-MLP warning).
OOD already fails. Every leaderboard number is **open-loop**; the one clean closed-loop signal is adverse.

---

## 4. The measurement problem — right tools, wrong primary

The infrastructure is A− (the paired episode-cluster bootstrap is correct and consistently used for standing
decisions; the gate refuses train-log slopes; the leaderboard is a model of honest win/tie/LOST reporting).
**The risk is the discipline in *using* it:**

- **The gate primary is ADE@2s — proven horizon-blind on 2026-07-25** (E1a: corridor-departure 0.35 %→59 %
  as horizon goes 2 s→18.5 s, while ADE barely moves — the instrument hid the failure ~170×). Yet ADE@2s
  **still fires every kill/continue call.** Two standing verdicts — **v3enc "RESTART" and v4.1 "FAIL"** — were
  rendered on this primary and *may be measuring the wrong thing.* This is the highest-leverage correction in
  the review.
- **Two decision-relevant CI leaks remain in code:** `taniteval/closedloop.py` still reports its headline
  ADE/FDE, compounding-error, and divergence through the **deprecated overlapping-holdout estimator**
  (1.28–2.06× too narrow) even though the open-loop path was migrated; and **parity is enforced by a
  path-substring, not a content hash**, so a truncated corpus (the known MooseFS-quota failure mode) lands in
  a correctly-named directory and silently passes.
- **Behavioral coverage is structurally absent:** collision, off-road, TTC, lane-keeping, intersection
  compliance, comfort/jerk — none are measured. "Does it drive?" has no instrument beyond geometry.
- **True-vs-claimed:** the numbers are real; the *rank / absolute / thesis* wrappers are over-stated. The
  REF-C > flagship closed-loop *ordering* is solid, but every absolute closed-loop number is a 2 s number,
  the instrument is map/agent-free (no collision at any horizon), and part of the ordering rests on n=12
  (one n=12 "win" already reversed at n=40).

---

## 5. The real headline is unmeasured — data-efficiency (C2 / H7)

The program's most *novel and defensible* possible result is not the hierarchy — it is **a label-free world
model that matches supervised data-efficiency.** The mechanism is already de-risked (`MEASURED`): pseudo-label
WM-pretraining captures ~96 % of real-label value; the YouTube-IDM pilot hit ~92 % of ceiling. **But the
actual data-efficiency *slope* — the matched-parameter, matched-step curve that would prove "structure/label-
free beats scale" — has never been measured.** It is a PRIO goal that keeps being deferred to "Phase 1." At
~$40 of compute this is the cheapest shot the program has at a genuinely world-class, publishable claim, and
it is the one most worth pulling forward.

---

## 6. Hypothesis portfolio (37 audited)

| Status | Count | Examples |
|---|:--:|---|
| Confirmed (incl. 2 confirmed-*negative*, 1 published premise) | 10 | H0, H3, H4a(frozen ceiling), H5, H13, speed-fix, heads-lossy, open-loop⊥closed-loop |
| Refuted | 3 | Branch-B camera-conditioning, INT8, recovery-augmentation lever |
| Confounded | 1 | H25 (v3enc; decorrelation measured never-on) |
| Partially | 10 | H1, H6, H7, H9, H11, H14, H15, H18, H26, H27 |
| Open | 9 | H2, H8, H10, H12, H16, H21, closed-loop-improvability *(reopened today)*, dataset-balancing |
| Stale / Orphaned | 4 | H17, H20, H22, H24 |

**The lopsided shape:** the program has *cleanly settled many negatives* (≈ half its delivered value) but
**every positive core edge is untested at the deciding instrument or counter-indicated.** Two doc headlines
are **unsafe vs the evidence**: "H4 CLOSED NEGATIVE" (the H4b positive reads 0.599 m, open) and H26 as a
"core-goal proof" (only 1/3 seams load-bearing).

**Highest-value still-open (the next GPU-days):** ① C2 data-efficiency slope (H7) · ② failure-gated
closed-loop SFT (E1b — *launched today*) · ③ planner–WM co-evolution to 30 k + a *formal* gate (H27; 3 of 8
kill-secondaries still have no emitter, so the gate cannot render a verdict) · ④ hierarchy net-driving
advantage (H1 D5/D6 + H26 — the constitutional claim, untested) · ⑤ longitudinal control (the entire n=40
closed-loop deficit).

---

## 7. Engineering & reproducibility

Library core is strong and well-tested; the trainer/orchestration layer is the liability. Concrete, actionable:

- **[HIGH] Parity by content-hash, not path-substring** (`train_flagship_v4.py:574` is a substring check; no
  trainer asserts `episode_count==2376` or recomputes the skip-hash; the keys are not reproducible from the
  repo — the real gate needs uncommitted pod-side files).
- **[HIGH] `closedloop.py` headline CIs use the deprecated estimator** (`_agg`/`_jack`), while `driving.py`
  was migrated. The *more* decision-relevant axis kept the wrong stats.
- **[HIGH-debt] Trainer sprawl** — 13 trainers, the window/label class copy-pasted 4×, the flagship importing
  it *from a REF-B script*, a 1,396-LOC trainer carrying a `_training_loop` its own docstring says is unused.
- **No training determinism** beyond `manual_seed` — restart/continue decisions ride RNG noise.

Both test suites collect clean (839 + 153, no errors). Nothing here is fraud; it is hygiene well below the
rest of the program's bar.

---

## 8. Process & the premature-certainty tax

The `RETRACTION_LOG` read as data (38 entries) is *both* a genuine self-correction strength *and* a symptom:

| Class | Meaning | Share |
|---|---|:--:|
| C3 | Mechanism asserted instead of measured | 19 % |
| C4 | Inherited / propagation (the *expensive* class: 12-day, 4-day survivals) | 19 % |
| C6 | Confounded comparison | 19 % |
| C5 | Scalar off a noisy curve / n=1 (dominates the *recent* window) | 17 % |
| C2/C1/C8/other | absence-from-one-probe / faster-source / premature-root-cause / live-source churn | 26 % |

≈ 27 caught same-session (cheap), ≈ 11 propagated (expensive); the expensive ones cluster on/before 07-21 and
detection latency has since collapsed — **but "the headline reached chat/reports/registry" recurs on nearly
every recent entry.** The lesson is precise: *the discipline `ci.py` enforces in code is not enforced in
prose.* Five "closed/bound/proven" verdicts were reopened by ~$0 follow-up checks **in one session**. Taxonomy
gaps to close: C7 is used but never defined; C8 is not in the legend.

---

## 9. Operations, automation & documentation

Guardrails-rich, automation-poor. The throughput cost is not the experiments (well-run) — it is the recurring
**tax + re-derivation + rescue** load. Ranked time-sinks (with evidence in R6):

1. **Stranding** — work outside git (REF-B v2, TanitEval, the orthogonality instrument idle 10 days, LAL-v2,
   `v2_compressed.py`; *today, the E1a/E2a driver scripts* — now rescued). Tens of agent-days latent.
2. **Stale-doc propagation in the source-of-truth registry** (the "best-ADE" header survived 4 days after its
   own body retracted it; ≥1 near-miss "designed the hierarchy away").
3. **Measurement re-derivation churn** (exponent re-fit across 5 windows; s/step "unresolved a day"; the CI
   recompute after the SE was found too narrow).
4. **Commit-path corruption + tax** (~50 % `git commit -- <pathspec>` segfault → 4 doc rewrites; stale
   `index.lock`; the whole-index default swept sibling work into the wrong commit *twice*).
5. **~1 MB/s dev-box relay + HF-403** for multi-GB moves (blocked the gate + a REF-C arm + the ckpt-backup at
   once; *this review's own flagship-eval is paying this tax right now*).

---

## 10. Unified proposal build-order  *(the deliverable)*

De-duplicated across all six reviews, ranked. Effort S/M/L; each names the reviewer(s) who raised it.

### Tier 1 — Correctness & integrity (cheap, high-payoff, do first)
| # | Proposal | Effort | Payoff | Src |
|---|---|:--:|---|:--:|
| 1 | **Change the gate primary.** Promote a horizon-matched `corridor_departure_rate @ K=max` to gate **co-primary** in `run_gate.py`; demote ADE@2s to a diagnostic. **Re-gate v3enc and v4.1** at ~18 s before either verdict stands. | M | Stops decisions on a metric proven blind to the dominant failure | R4,R3 |
| 2 | **Route `closedloop.py` headline/compounding/divergence CIs through `ci.episode_cluster_bootstrap`** (already imported 40 lines away). | S (~1 h) | Fixes the most decision-relevant stats leak | R1,R4 |
| 3 | **Parity by content-hash.** Commit an episode-id manifest; make every trainer assert `count==2376` + `sha256(sorted uids)` in-process; test that a truncated cache is *refused*. | M | Closes the silent-truncation hole under the sacred invariant | R1 |
| 4 | **`safe_commit.py`** — pathspec-free `-F` commit, foreign-index abort, `Keys.txt` refuse, auto-clear `index.lock`, retry on segfault. | S | Kills time-sink #4 + the sibling-sweep structurally | R6 |
| 5 | **`registry_lint.py`** — re-read JSON via `<!-- src: …#field -->` pointers, **fail CI on drift**; multiline-scan registry **headers** for any retracted phrase. | S/M | Kills the 4-day-stale-header class at the root | R6,R4 |

### Tier 2 — Thesis-validation experiments (the science that decides the program)
| # | Proposal | Effort | Payoff | Src |
|---|---|:--:|---|:--:|
| 6 | **Test the hierarchy or drop the claim.** Pre-register G1: flat planner-over-WM vs the full 3-level stack, paired, CI-separated. Until a separated win exists, downgrade "hierarchy" → "planning-over-WM" and reallocate ~57 M. | L | Resolves the control-beats-treatment crux; protects the headline from collapsing under review | R2,R3,R4 |
| 7 | **Measure the C2 data-efficiency slope** — matched-param/step, label-free-WM vs supervised. Pull forward from "Phase 1"; run the YouTube-IDM scale-up to decision-grade after the 07-26 cooldown. | M | The program's best shot at a *novel, defensible* result, ~$40 | R2,R3,R5 |
| 8 | **Get one recognized-benchmark number** — REF-C-base on **NAVSIM (EPDMS)** or **Bench2Drive**. | M | First external ground truth; the only bar the 74 %-straight corpus cannot provide | R2,R5 |

### Tier 3 — Automation & results-documentation (throughput)
| # | Proposal | Effort | Payoff | Src |
|---|---|:--:|---|:--:|
| 9 | **`results_ledger.py`** — auto-generate `RESULTS.md` + `results.jsonl` + `GATES.md` from raw JSON; the registry **cites by pointer, never copies**. | M | The results-documentation centerpiece; retires stale-doc drift at the root | R6 |
| 10 | **`ckpt_relay.py` + provenance stamper** — pod-side HF push/pull at ~118 MB/s, md5-verify, terminal `UPLOAD_COMPLETE` sentinel, per-ckpt provenance sidecar. | M | Kills the 1 MB/s relay tax + the async-completion overclaim class | R6 |
| 11 | **Land the 1-line dense-path persistence** (`taniteval/rollout.py:94`, currently keeps 4/20 steps) — unmerged since 2026-07-09. | S | Unlocks the whole comfort/behavioral metric axis for ~1 MB/arm | R4 |
| 12 | **`repo_janitor.py`** — `git worktree prune` + delete unique-commit-free worktrees + a dated `incoming/` ledger + future-date flag. | S | Restores Glob reliability; clears ~90 stale bundles | R6 |
| 13 | **Enforce `ci.py` provenance at the doc layer** — CI rejects any ADE/departure/comparative number entering LOOP_STATE/reports/registry without `estimator=` + `n=` (n≥12 for closed-loop). | S/M | Targets C1/C5/C4 at the prose layer where they actually escape | R4,R6 |
| 14 | **Split LOOP_STATE + add a size-guard; consolidate the 3×/day report cadence** with auto-compile. | S/M | Stops the 122 KB run-on file shipping stale instructions | R6,R5 |

### Tier 4 — Strategic
| # | Proposal | Effort | Payoff | Src |
|---|---|:--:|---|:--:|
| 15 | **Give the reactive-agent renderer an owner and a pod.** It is the single dependency for safety-grade closed-loop, D5/D6, and *any* benchmark entry — the only route to an opponent-facing number. Stop running cheap experiments *around* it. | L | Unblocks the entire safety-grade closed-loop program | R5 |
| 16 | **Force the flagship-vs-REF-C decision at the v4 30 k gate.** Either the co-evolved planner beats REF-C closed-loop, or elevate REF-C to co-hero and reframe the flagship as *WM-as-simulator + data-efficiency*. Retire REF-A/REF-B from compute; compress the weekly agent cadence to the critical path. | S (decision) | Reclaims orchestrator bandwidth; ends the hero/baseline ambiguity | R5,R2 |

---

## 11. The five things to do this week

1. **Re-gate on horizon, not ADE@2s** (Tier-1 #1). Nothing else is trustworthy until the primary is right —
   including the v3enc and v4.1 verdicts already on the books.
2. **Ship `safe_commit.py` + `registry_lint.py`** (Tier-1 #4/#5) — the two cheapest fixes that each kill a
   recurring, sometimes decision-grade, loss.
3. **Pre-register G1: hierarchy vs flat planner-over-WM** (Tier-2 #6). The program cannot claim its headline
   until this is CI-separated; the current evidence is neutral-to-adverse.
4. **Commit to the C2 data-efficiency slope as *the* headline** (Tier-2 #7) and resource the YouTube-IDM
   scale-up to decision-grade after the 07-26 cooldown.
5. **Give the renderer an owner** (Tier-4 #15) — it is the unowned critical path to every external number.

**The one thing, if only one:** *measure the two questions that decide the thesis — hierarchy-vs-flat and the
data-efficiency slope — and change the metric you measure them with.* The program has already built the most
expensive thing (a trustworthy apparatus). It has not yet pointed it at the questions that make it matter.

---

*Balance owed to the record:* the strengths cited here are real and rare — the causal vision-anticipation
panel, the correct paired-bootstrap CI, the root-cause retraction doctrine, the honest leaderboard with an
explicit un-measured contract, two genuinely publishable clean negatives, and solved inference latency. The
critique is not that the science is sloppy — it is unusually careful. The critique is that this careful
apparatus is, today, validating the *wrong* claims on the *wrong* corpus with the *wrong* primary metric, and
that the program's best result is the one it has not yet measured.
