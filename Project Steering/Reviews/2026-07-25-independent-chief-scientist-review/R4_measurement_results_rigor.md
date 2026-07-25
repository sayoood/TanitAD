# R4 — Independent Metrology & Results-Integrity Audit

**Reviewer role:** independent results-integrity auditor (adversarial). **Date:** 2026-07-25 (Europe/Berlin).
**Mandate:** are the headline results CREDIBLE, what is the true-vs-claimed delta, and where is the program
still fooling itself? **Discipline:** primary sources only (raw eval JSON · `MODEL_REGISTRY.md` ·
`taniteval/ci.py` · `GATE_PROTOCOL.md` · `LEADERBOARD.md` · `RETRACTION_LOG.md`); read-only; no pods, no
compute; no git-add. **Evidence classes** on every claim: `MEASURED (+path)` · `PUBLISHED` · `INHERITED` ·
`ESTIMATED` · `HYPOTHESIS`.

---

## 1. Executive verdict

**The program's measurement *infrastructure* is genuinely excellent — arguably A-grade and ahead of most
published AD work at this scale. The *discipline in using it* is where the risk lives, and it is real.**

Three things are true at once:

1. **The instruments are sound.** `taniteval/ci.py` (episode-cluster bootstrap, paired form, fails-loud on
   empty eid, provenance stamped into every result dict) is a correct, well-reasoned replacement for the
   anti-conservative `overlapping_holdout_se`, and it is used consistently for every *standing* decision
   (`GATE_PROTOCOL.md` mandates it; `taniteval/driving.py` *refuses* to emit the deprecated estimator;
   `CI_RECOMPUTE_2026-07-20.json` reproduces every published mean digit-for-digit). `LEADERBOARD.md` is a
   model of honest reporting: three-way win/tie/LOST, "ADE is one column, not the verdict", an explicit
   "what we deliberately do NOT measure" contract, and a standing open-loop⊥closed-loop footnote.

2. **The primary metric was structurally blind to the #1 failure mode, and this was discovered *four days
   ago*.** ADE@2s — the gate **primary** for every arm (`GATE_PROTOCOL.md` §2) — cannot see the dominant
   closed-loop failure. E1a (`…/2026-07-25-closedloop-horizon-and-shift/`) measured corridor-departure
   **0.35 % → 59 % (84 % at junctions)** as the horizon goes 2 s → 18.5 s on the *same* in-distribution
   frames; ADE@2s barely moved (0.485→0.496) because "it CANNOT see 18 s drift". Nearly every closed-loop
   conclusion written before 2026-07-25 inherited the wrong horizon.

3. **The communication layer systematically outruns the verification layer.** The `RETRACTION_LOG.md` holds
   **38 retractions**; the recurring meta-pattern — flagged by the program itself five times on 2026-07-24
   alone — is a firm claim reaching chat / `LOOP_STATE` / reports / the registry *before* the $0 check that
   reopens it. Detection latency has dropped sharply (post-07-21 almost everything is caught same-session),
   but the *generation* of premature claims has not stopped.

**Net grade: B− (§3).** The A-grade toolchain is pulled down by (a) a primary metric proven horizon-blind
to the behavior that matters, still driving model-selection gates; (b) a map-free suite that by the
program's own admission cannot discriminate perception/world-model quality — and a flat, no-world-model
baseline (REF-C-base) ties the hierarchical flagship, exactly the signature you would expect if the metric
rewards kinematics over driving; and (c) narrow behavioral coverage: collision, off-road, TTC,
lane-keeping, and intersection capability are all structurally unmeasured, so "does it drive safely" has no
number at all.

**Headline credibility, one line each (full bands in §3):** flagship-v1 ADE is a *real number* but a
*mis-ranked* claim (a statistical tie REF-C-base matches with 40 % of the params and 4.5× less latency);
REF-C-closed-loop is a solid *relative ordering* wrapped around an *absolute* claim that collapses at
realistic horizon; the *hierarchy's value is unmeasured-to-negative*; the *YouTube-IDM transfer is
refuted by measurement*, not merely unproven.

---

## 2. Findings

Severity: 🟥 high (could mislead a GPU-day or a deployment call) · 🟡 medium · 🟢 low.
Confidence: HIGH / MED / LOW that the finding itself is correct.

### F1 🟥 (HIGH) — The gate PRIMARY is ADE@2s, a metric just proven horizon-blind to the dominant failure
`MEASURED` (`GATE_PROTOCOL.md` §2 "Primary: Held-out val ADE@2s … Never a train-log slope";
`Project Steering/Gates/flagship-v3enc.card.json` primary `ade_0_2s ≤ 2.5`; v4 card primary `ade_0_2s ≤
0.60`, registry §1.5.2) **and** `MEASURED` (E1a, `…/2026-07-25-closedloop-horizon-and-shift/E1a_E2a_RESULTS.md`
§3: corridor-departure 0.0035→0.5877 as K goes 20→185 while `closed_ade2s` moves 0.485→0.496).
E1a fixed the *closed-loop* instrument's horizon, but **the open-loop ADE@2s that gates every kill/continue
decision was not re-examined**. The program continues to select and kill arms on a metric it has now
demonstrated is blind to the failure mode that separates a drivable model from an undrivable one. v3enc was
`RESTART`ed and v4.1 `FAIL`ed on `ade_0_2s`; both verdicts may be measuring the wrong thing. *This is the
single most consequential open exposure.*

### F2 🟥 (HIGH) — A flat, no-world-model baseline ties the hierarchical flagship on the primary metric
`MEASURED` (`CI_RECOMPUTE_2026-07-20.json` pairs; `LEADERBOARD.md` §1). Flagship v1 (0.4271, 263 M,
hierarchical WM) vs REF-C-base (0.4728, **104 M**, flat anchored-diffusion, **no world model, no
hierarchy**): the three rank-1 arms are a **statistical tie** no paired test can order (flagship−XL Δ
+0.0443 [−0.0544, +0.1465], p=0.81). Combined with §7.2's AD-MLP warning (a map-free ego-log suite is
gameable by a no-perception MLP; the in-corpus ego-status ceiling is 0.5735 and flagship clears it by only
~0.15 m), this is the classic signature of a metric that **rewards kinematic competence, not
world-modeling**. The program's two antidotes (ego-status ceiling + one vision-ablation number +1.325 m
[+1.04, +1.64]) are real but thin. **Implication:** the flagship's headline does not, on this evidence,
demonstrate that the world model or the hierarchy earns its 2.5× parameter and 4.5× latency premium.

### F3 🟥 (HIGH) — The hierarchy's value is UNMEASURED-to-NEGATIVE, yet it is the architecture's thesis
`MEASURED` + `HYPOTHESIS`. Every direct read of the strategic/tactical hierarchy is null or negative:
the v1 strategic head was a **command echo** (`route_skill_vs_chance = 0.0`, registry D-A6/D-033); the
intent→operative seam was **net-harmful** (cos −0.238, D-033); the one pessimistic reading ("fan is a speed
fan ⇒ strategic is a ~2 % lever") was **retracted as confounded** because REF-C evaluates with
`nav_cmd=None` (`RETRACTION_LOG.md` 07-21, C6) — so the hierarchy value is unmeasured *in either
direction*. Every attempt to build a proper joint hierarchy has failed or is pending: v4 KILLED, v4.1
primary FAIL (0.8522 vs 0.60), v4.2 worse (0.9869; canary breach), v4.2b LIVE/pending (registry §1.5.1–4).
And `nav_cmd=None` propagates into the closed-loop instrument too (E1a `e1a_horizon.py`:
`model(fw, nav_cmd=None, …)`), so route-conditioned behavior is **never actually exercised** in any
standing eval. **Any claim that "the hierarchy works" is HYPOTHESIS**, and the strongest datapoint (F2) runs
against it.

### F4 🟥 (HIGH) — Absolute closed-loop safety is entirely unmeasured; the ordering rests partly on n=12
`MEASURED` (`LEADERBOARD.md` §5.5; `RETRACTION_LOG.md` 07-22/07-23/07-25). The REF-C-base > flagship-v1
closed-loop *ordering* is well-established (triple-confirmed n=1→n=12→n=40, survived resolution and
reconstruction-OOD controls). But: (a) every absolute number is a **2 s** number (mandatory 07-25
qualifier), and E1a shows 2 s hides the failure by ~170×; (b) the n=40 real-footage instrument is
**map/agent-free — it emits lane-drift only, never off-road or collision**; (c) the n=12 NuRec suite is
reconstruction-OOD-confounded (open-loop 3.21× off-distribution) and one scene = 8.3 pp — an n=12 "win"
already **reversed at n=40** once (07-24 D2/RefcCL, C5). So "REF-C base drives closed-loop" is false at
realistic horizon, and "…drives *safely*" has no measurement anywhere.

### F5 🟥 (HIGH) — The closed-loop evidence is a confound-chain in which every "closed" verdict reopened
`MEASURED` (`RETRACTION_LOG.md`, six entries 07-22→07-25). The chain: reconstruction-OOD → resolution →
horizon → training-stratum, each confound fixed by the next cheap experiment, and **each fix reopened a
verdict previously stated as closed/bound** ("needs a renderer" → "doesn't need a renderer" →
"BOUND" → "BOUND is horizon-confounded"). The log's own words (07-25): *"the pattern is now so reliable it
is itself the strongest evidence that 'closed' should never be a same-session verdict."* The risk is not
any single number — it is that the program keeps *believing it has closed closed-loop* when only a relative
lane-drift ordering at sub-realistic horizon is actually established.

### F6 🟡 (HIGH) — A registry disjointness claim is FALSE at the byte level (val-leak inverted)
`MEASURED` (E1a §1.1, `probe_env.json`: `physicalai-val-f1b378f295ae` = 79 distinct ids, **62 also in the
parity train corpus = 78.5 % overlap**; chance ≈ 2.8). `MODEL_REGISTRY.md:1737` states this val is
"episode-disjoint from …-train-e438721ae894". **It is not.** *Precision (so this is not over-claimed):* the
**canonical 881-window val `0c5f7dac3b11`** used for every standing leaderboard number is a *different* set,
stated disjoint in §0.3, and is **not** implicated here. But the leaked `f1b378` val is exactly the set R8
prescribes for "re-evaluate the leak-contaminated REF-A I-JEPA number on the clean val" (registry §7 R8) —
**so the prescribed fix for one leak would run on a 78.5 %-leaked set.** A disjointness assertion in the
harness (F-prop-3) would have caught this.

### F7 🟡 (MED) — The deprecated estimator still sits beside headline numbers in the source-of-truth doc
`MEASURED` (`MODEL_REGISTRY.md` §4.1 table "0.458 ± 0.057"; §1.4b table "0.4886 ± 0.0800"; §2.1
"2.1355 ± 0.1963"; REF-A §2.3 "± 0.1821"). These `± ci95` are `overlapping_holdout_se`, **1.28–2.06× too
narrow** (`CI_RECOMPUTE_2026-07-20.json` `widen_x`). They are labeled deprecated inline, but a casual reader
sees a tight ± next to the headline. The registry itself carried a retracted headline ("v1.6 best in
program") in a **section header for four days** after its body retracted it (07-25, C4) — and
`LEADERBOARD.md` was *already correct*, i.e. **the derived document out-disciplined the source of truth.**
The rule "never quote an interval without its estimator" is mostly honored, but the deprecated estimator is
*contained, not eradicated*, from the one document CLAUDE.md says is the only quotable source.

### F8 🟡 (MED) — Two live metric-definition conflicts remain unreconciled while their numbers circulate
`MEASURED`. (a) **Junction-departure**: `LEADERBOARD.md` §5.5 flags 0.368 (window-level) vs 0.0134/0.064
(episode-level) as "a metric-definition mismatch that must be reconciled BEFORE either is quoted as the
junction rate." (b) **Latency**: registry §6 says flagship tick 103.42 ms, the committed
`eff_flagship-30k.json` says 97.32 ms, its own repeatability file says 99.03–100.05 — three values, none
matching (registry §7 R14, "reported not resolved"). Neither changes a headline conclusion, but a
prose figure disagreeing with its own cited artifact is precisely the class the registry exists to prevent.

### F9 🟡 (MED) — The deployed model is not byte-rebuildable from HEAD
`MEASURED` (registry §1.2 R4: committed `train_flagship4b.py` has no `--jerk-weight`/`--aux-accel`; the
deployed v1's `config.json` records both; the only trainer that produced it shows `M` on pod2). The
program's stated acceptance test — "a reader with this repo can rebuild that exact model" — **fails for the
one model in production.** Reconstruction risk, not a results-integrity error, but it caps how much any v1
number can be independently re-verified.

### F10 🟢 (MED) — The RETRACTION_LOG's own class legend is stale (a recursive C4)
`MEASURED` (`RETRACTION_LOG.md` header table defines **C1–C6 only**, "ranked by how often they have bitten
us"; entries use **C8** three times — 07-24/07-25 — and C8 is explicitly called "the recurring failure mode
this session"; **C7 is never defined as a class** and appears only as an *ordering label* "the C7 ordering"
in `LEADERBOARD.md`). The log that exists to catch propagation-of-stale-headers has a stale header: its
most-recent recurring class (C8, premature root-cause from too few points) is absent from its own ranked
legend. Cosmetic, but it is the same C4 mechanism the log is meant to teach.

### F11 🟢 (HIGH) — Regime/behavioral thresholds are PROPOSED, not validated, and two panels disagree
`MEASURED` (`LEADERBOARD.md` §10: cruise/transient split at ±0.5 m/s², curvature at |κ|<1e-3 and 5°/15° are
"PROPOSED", sensitivity sweep "owed"; `driving.py` buckets curvature 5°/15° while `bench.by_curvature` uses
5°/20° — "not interchangeable until reconciled"). The effects are large enough to be unlikely threshold
artifacts, but several §2–§4 behavioral readings rest on un-swept thresholds.

**What is genuinely strong (stated for balance):** `ci.py` is correct and provenance-stamped; the paired
bootstrap is used for real decisions; `GATE_PROTOCOL.md` refuses train-log slopes and caps exponent
extrapolation at 2× with an R²≥0.80 floor; pre-registration with both outcomes committed in advance is real
(E1a, the from-scratch fallback); the program caught its *own* ADE-horizon blindness and its *own*
val-leak inversion. This is a program that finds its own errors — the question §3 grades is whether it finds
them *before* they decide something.

---

## 3. Measurement-maturity grade + true-vs-claimed deltas

### Grade: **B−** (infrastructure A− · discipline-in-use C+)

| Axis | Grade | Basis |
|---|---|---|
| Interval estimator & tooling | **A−** | `ci.py` correct, paired, provenance-stamped; deprecated estimator refused by `driving.py`; recompute reproduces every mean |
| Gate protocol & pre-registration | **A−** | held-out only, no train-log slopes, exponent floor + 2× cap, restart caps, both-outcomes-committed experiments |
| Reporting honesty (leaderboard) | **A−** | win/tie/LOST three-way, "ADE is one column", explicit un-measured contract, open-loop⊥closed-loop footnote |
| Primary-metric validity | **D** | ADE@2s gates every decision and is proven blind to the dominant (18 s) failure; still unaddressed for the *gate* |
| Behavioral / scenario coverage | **D** | no collision/off-road/TTC/lane-keeping/intersection; "drives safely" has no number |
| Claim→communication discipline | **C** | 38 retractions; the firm-claim-before-cheap-check pattern recurs and reaches chat/reports/registry |
| Source-of-truth hygiene | **C+** | deprecated ± beside headlines; a retracted headline stood 4 days; derived doc out-disciplined the source |

The toolchain deserves an A−. It is dragged to **B−** because the tools are aimed at the wrong horizon for
the decision that matters, and because the discipline of *using them before speaking* is not yet reliable.

### True-vs-claimed deltas (most skeptical *fair* reading)

| Headline | Claimed | True-achievement band (skeptical-fair) | Delta |
|---|---|---|---|
| **Flagship v1 ADE / "best in program"** | 0.45 m ADE@2s, program's best sub-300 M model | Number is solid & reproduces (`CI_RECOMPUTE`, repro_ok). But it is a **3-way statistical tie** with two REF-C arms; REF-C-**base** matches it with **104 M vs 263 M and 21.8 ms vs 97.3 ms**, and *meets* 10 Hz where flagship misses it in all 3 precisions. The ADE edge is **lateral only** (along-track win NOT separated) and it **loses to doing-nothing on cruise across 72.5 % of the corpus**. Legit CV-beater with a separated vision effect — that part holds. | "Best model" → **"tied-for-first on a scalar its rivals match with 40 % of the params; Pareto-dominated on deployment; lateral-only."** Rank claim overstated by ~one tier. |
| **REF-C closed-loop** ("REF-C base beats flagship, drives closed-loop") | REF-C base > flagship, triple-confirmed | **Relative ordering: solid** (survived resolution + reconstruction-OOD controls). **Absolute: collapses** — every number is 2 s; at 18.5 s REF-C base departs the corridor on **59 %** of windows (84 % junctions). Instrument is map/agent-free → **no off-road/collision measured at any horizon.** | Ordering survives; **"drives closed-loop" (absolute) → false at realistic horizon**; "drives safely" → **unmeasured.** |
| **Hierarchy value** (strategic/tactical planners) | The architecture's thesis; 3-planner hierarchy | **No measured positive value.** Strategic head = command echo (skill 0.0); intent→operative seam net-harmful; the pessimistic reading was retracted as confounded (`nav_cmd=None`). Every v4 hierarchy build FAILED/pending. A flat no-WM baseline **ties** the hierarchical flagship (F2). | Claimed thesis → **HYPOTHESIS, currently unmeasured-to-negative.** Largest gap between narrative and evidence. |
| **YouTube-IDM transfer** | A YouTube-scale IDM pipeline yields a decision-grade lift | Supervised IDM **FAILS the cross-domain gate** (comma speed R² 0.657 / yaw 0.000; same-corpus rig-B speed R² **−2.465**, worse than the mean). The rig-robust-encoder fix was **built and REFUTED** (cross-rig R² −0.667 vs frozen +0.657). Scale harvest **not delivered** (YouTube-blocked, 07-25). Only evidence is one 80-clip directional pilot (+0.563), explicitly **not** upgraded. | Anticipated lift → **refuted by measurement**, not merely unproven. Claim shrinks to a single directional pilot contradicted by two failed transfer gates. |

---

## 4. RETRACTION_LOG error-class frequency table

**38 entries, 2026-07-21 → 2026-07-25.** Compound entries (e.g. "C3/C4") count toward each named class, so
tokens (42) exceed entries. Same-session = corrected within the iteration it was made (cheap; often "cost 0
but the headline reached chat"). Propagated = stood for ≥~1 day / entered a durable doc (expensive).

| Class | Definition | Tokens | Share | Notable / cost |
|---|---|---:|---:|---|
| **C3** | Mechanism instead of measurement (a causal story asserted as fact) | **8** | 19 % | Tied-most-frequent. "CEM infeasible 723 ms" nearly killed v3's thesis; "closed-loop needs a renderer" (over-broad) |
| **C4** | Inherited without re-verification / **propagation** (can't name the file the number lives in) | **8** | 19 % | The expensive class: "pods cannot render" 12 d; "v1.6 best" header stood **4 d**; LAL-v2 "unmerged 12 d" (was merged) |
| **C6** | Confounded comparison (contrast varies >1 thing) | **8** | 19 % | "fan is a speed fan" (nearly designed the hierarchy away); the whole closed-loop confound-chain (F5) |
| **C5** | Scalar off a noisy curve / **n=1 over-read** (no window, R², or n) | **7** | 17 % | **Dominates the recent window** (07-22→24): every retracted n=1 closed-loop headline |
| **C2** | Absence from a single probe (one path/process/dir) | **4** | 10 % | "no EGL devices" (probed `/usr/share` not `/etc`, 12 d); "job died" (was complete) ×2 |
| **C1** | Faster-moving source than the harness (trainer log / HUD / progress print quoted as eval) | **3** | 7 % | "v1.6 0.4420 best" (~10 % optimistic trainer val, entered registry) |
| **C8** | Premature root-cause from too few points / surface-read / tool artifact (**not in the legend**) | **3** | 7 % | All 07-24/07-25: git-segfault theory ×2 wrong; "stranded on worktree" (Glob mtime-cap artifact) |
| **new** | Operational churn against a rate-limited live third-party source (**uncategorized**) | **1** | 2 % | YouTube-IDM harvest hard-blocked; not delivered |
| **C7** | *never defined as an error class* (appears only as an ordering label) | 0 | — | Taxonomy gap (F10) |

**Same-session vs propagated:** ~**27 same-session** (cheap) / ~**11 propagated** (expensive). **The expensive
ones cluster on/before 2026-07-21** (the operating-standard audit day, 22 entries). From 07-22 the
detection latency collapses — almost everything is caught within the iteration — **but the caveat "the
headline reached chat/LOOP_STATE/reports/registry" recurs on nearly every recent entry.**

**Reading — strength or symptom? BOTH, and the split is the point.** A high retraction rate with root-cause
classes IS a real self-correction mechanism (the log genuinely teaches; C1/C2 recur less over time). But the
*generating* process — **firm claims published before the $0 check** — is not fixed: C5/C6/C8 (n=1 reads,
confounds, premature root-causes) dominate the recent window, and the program flagged its **own 5th
over-claim-of-closure** in a single day. **The discipline that `ci.py` enforces in code is not enforced in
prose.** That asymmetry is the durable risk, and it motivates the proposals below.

---

## 5. Concrete proposals

### 5.1 Metrics beyond ADE
1. **Promote a horizon-matched closed-loop departure metric to gate co-primary.** Add
   `corridor_departure_rate @ K=max` (E1a's instrument, `…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon.py`)
   to `run_gate.py`'s primary set; keep ADE@2s as a *proposal-quality diagnostic*, not the kill criterion.
   **Re-gate the standing arms** (flagship v1, REF-C base/XL) at ~18 s before any is called "deployed" —
   directly retires F1/F4.
2. **Land the Tier-1 dense-path persistence — the ONE line at `taniteval/rollout.py:94`** (keeps 4 of 20
   steps). It unlocks jerk/comfort bounds, curvature *profile*, decel-onset lead time, and plan-stability —
   the entire comfort/behavioral axis that is currently dark — for ~1 MB/arm, and the downstream metric
   (LAL-v2 anticipation) is **already implemented and unmerged since 2026-07-09** (`metrics.py:202-251`).
3. **Reconcile the two open metric conflicts before either number is quoted again** (F8): junction-departure
   window-vs-episode definition, and the flagship latency (registry §6 vs committed artifact, R14).

### 5.2 Scenario-driven evaluation
4. **State the safety gap as a first-class UNMEASURED, not a footnote.** Collision, off-road, TTC/headway,
   lane-centre deviation, and intersection/roundabout capability are all structurally absent
   (`LEADERBOARD.md` §6). Put a standing "SAFETY: UNMEASURED — no agent boxes / no map / 2 s horizon" banner
   on any deployment-readiness claim, so "0.45 m ADE" is never read as "drives safely".
5. **Strengthen the perception-discrimination antidote (F2).** The map-free suite cannot, alone, tell a
   world-model from an ego-status MLP; the current defense is one vision-ablation number. Add a **routine
   perception-necessity probe per arm** (vision-ablation on high-divergence windows + the ego-status ceiling
   as a *gate*, not a footnote), and treat "a flat baseline ties the hierarchical arm" as an alarm requiring
   an explicit hierarchy-value experiment, not a curiosity.
6. **Exercise route conditioning at least once.** Every standing eval runs `nav_cmd=None`, so the strategic
   layer is never tested (F3). A single route-conditioned vs route-free paired eval would move the hierarchy
   value from HYPOTHESIS to MEASURED in one direction or the other.

### 5.3 Automated guardrails against the recurring classes
7. **"No headline before an estimator" — enforce the `ci.py` provenance discipline at the *doc* layer**
   (targets C1/C5, the most frequent recent classes). A pre-commit / CI check that any ADE / departure /
   comparative number entering `LOOP_STATE.md`, a program report, or the registry carries an `estimator=`
   and `n=` tag — the same fields `ci.py` already stamps into every dict. Closed-loop claims additionally
   gated on **n ≥ 12** (the log's own rule, violated ≥4×).
8. **Multiline header-sweep for retracted numbers** (targets C4, the expensive class). A CI grep over
   registry/leaderboard **section headers** (multiline, so a claim wrapped across a newline can't hide — the
   exact 07-25 failure) for any headline number lacking a raw-JSON path. Would have caught the "v1.6 best"
   header that stood 4 days.
9. **Automated val-disjointness assertion in the harness** (targets F6 and the C2 class). Three lines:
   assert `len(set(val_episode_ids) ∩ set(train_episode_ids)) == 0` at eval start, fail loud. Would have
   caught the `f1b378` leak-inversion at the byte level instead of a wrong disjointness line surviving in the
   registry to `:1737`.
10. **Fold C7/C8 into the RETRACTION_LOG legend** (F10) and add C8's rule — *before asserting a mechanism
    for an intermittent/low-n failure, re-run the identical command / add a second point* — to the standing
    consequences list. Small, but it closes the log's own propagation gap.

---

*Provenance: every number above is `MEASURED` from the cited primary artifact or `PUBLISHED`; where a claim
is the program's own hypothesis it is marked `HYPOTHESIS`. No pods were touched; no compute run; nothing was
git-added. Reviewer: independent results-integrity audit, 2026-07-25.*
