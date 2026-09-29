# Stream R4 — Evaluation, Metrics, Statistics & Closed-Loop

**Scope:** whole-programme review, static (repo + web only; no pod/torch access this session).
**Repo:** `/home/user/TanitAD`, branch `claude/optimistic-shannon-vixpo6`, HEAD `467ce8a` (2026-08-04).
**Reviewer discipline:** primary sources only — raw eval JSON, source `file:line`, `MODEL_REGISTRY.md`,
`GATE_PROTOCOL.md`. Every number below carries an evidence class: **MEASURED** (this session, from a
named artifact) · **PUBLISHED** (external, cited, dated) · **INHERITED** (another doc/stream, not
re-verified by me) · **ESTIMATED** · **HYPOTHESIS**. ADE numbers are **full-set mean** unless marked
`heldout` (the deprecated `overlapping_holdout_se` split-mean — CLAUDE.md: differs from full-set by
−6.67% to +11.69%, bidirectional, 27 arms).

Banked incrementally; sections are independent and safe to read out of order.

---

## 1. Headline findings (ranked)

**F1. The 2026-08-02 binding four-family rule has a real, well-engineered instrument
(`taniteval/four_families.py`) that is NOT wired into the canonical eval pipeline.** MEASURED: `grep -rn
"all_families\|four_families"` across the repo shows zero callers in `runner.py`, `bench.py`, or
`driving.py` — the three modules that actually run for every arm (`runner.py:52` `run_one`, wired axes at
`runner.py:106` `efficiency`, `runner.py:117` `driving`). Zero of the 27 committed
`taniteval/results/driving_*.json` / `eff_*.json` result files contain the module's own sentinel key
`_binding_rule` (checked all 103 files under `taniteval/results/`). Every four-family output that exists
lives under `TanitAD Research Hub/**/incoming/` — ad hoc, per-stream, one-off — never merged. This is the
single most important fact for Q1 and is detailed as a matrix in §2.

**F2. TACTICAL and STRATEGIC have *never* been measured on the full 40-episode canonical val for a
production arm.** MEASURED: the most complete four-family artifact on the canonical corpus
(`.../incoming/2026-08-04-distance-keeping-arms/raw/four_family_panel_val40.json`, `n_windows=881,
n_episodes=40`, arms `flagship-30k`/`refc-base-30k`/`cv`/`gt_oracle`) still reports `TACTICAL` and
`STRATEGIC` as `UNAVAILABLE` for **every** arm in it (`families_unavailable` block, reason: "a world-model
FIDELITY pass does not traverse the hierarchy"). The only runs that ever populated TACTICAL/STRATEGIC via
`hierarchy.run` (`taniteval/taniteval/hierarchy.py:428`) used a **19-episode** subset
(`fourfam_v1-lf19.json` / `fourfam_v2corpus-lf19.json`, `n_episodes: 19`, 2026-08-02) — not the 40-episode
/ 881-window corpus every other headline in the registry uses.

**F3. `hierarchy.run` — the only path that produces TACTICAL/STRATEGIC decisions — is architecturally
restricted to flagship-family arms.** MEASURED, `taniteval/taniteval/runner.py:341-342`: "`hierarchy.py`
currently supports only (tactical_policy + strategic_policy) arms." REF-B and REF-C are `direct_head`
architectures (`runner.py:60-61`, own trajectory surface, no grounded operative rollout) and are excluded
by construction. So even a full re-run on 40 episodes could not, today, give REF-C or REF-B a TACTICAL/
STRATEGIC number — the two arms statistically tied with flagship on ADE (§6 of the registry) are
structurally unmeasurable on 3 of the 4 binding families.

**F4. `MODEL_REGISTRY.md` §0.3's standing "TanitEval is uncommitted / reconstruction risk" claim is
stale — and has been since the review's own baseline snapshot.** MEASURED (this session):
`git log --diff-filter=A -- taniteval/taniteval/runner.py` → first added at `abe3dd3`; `git log -1 --
taniteval/` → last touched `3b51214` (2026-08-04, the day of HEAD); `git ls-files taniteval/ | wc -l` →
**263 tracked files**, not gitignored. `MODEL_REGISTRY.md:104-109` still reads "🟥 RECONSTRUCTION RISK —
TanitEval is uncommitted... exists only on `tanitad-eval:/root/taniteval`." The risk this passage warns
about is resolved; the registry — the program's single source of truth — has not been updated to say so.
Low severity (the news is good), but it is exactly the class of stale-doc-vs-reality gap CLAUDE.md exists
to catch, sitting in the one document exempted from that failure mode.

**F5. Statistical power at n=40 episodes is real but narrow, and the noise band is comparison-dependent,
not a single number.** MEASURED from `MODEL_REGISTRY.md` §6's own paired episode-cluster-bootstrap
results: a paired ADE@2s delta of **+0.0443 m** (rank 1 vs rank 2, flagship v1 vs REF-C-XL) is **not**
separated (CI95 half-width ≈0.10 m), while **+0.1199 m** (REF-C vs REF-B v2) **is** separated (half-width
≈0.056 m) and **+0.1642 m** (flagship vs REF-B v2) is separated at half-width ≈0.121 m. The half-width
depends on how correlated the two arms' errors are (same-family pairs are tighter), so there is no single
MDE — only a bracket, detailed in §3.

**F6. The AlpaSim NuRec closed-loop evidence base contains a genuine trap for anyone who grabs the first
matching file instead of the paired rerun.** MEASURED: `REFC_suite_base_results.json` (2026-07-22, solo)
gives REF-C-base pass **6/12**; the correct decision-grade artifact,
`flagship_vs_refc_suite_results.json` (2026-07-23, **paired**, identical 12 scenes, both models on
bit-identical NuRec renders) gives REF-C-base **8/12** vs flagship v1 **2/12**
(`paired_flagship_minus_refc.score_sign_test.two_sided_p = 0.0078`, `pass_mcnemar.two_sided_p = 0.03125`).
Both files are real, both say "REF-C suite," and they disagree on REF-C's own pass count (6 vs 8) because
one is unpaired-solo and one is the paired rerun. `MODEL_REGISTRY.md` §4.4/§6 cites the correct (8/12 vs
2/12, paired) number, and the orchestrator's fact sheet matches it — but nothing in the file layout stops
a future reader from citing the superseded 6/12 file. Recommend a `SUPERSEDED_BY:` pointer be added to the
older file's neighbours (§7).

**F7. There is a live, well-matched, time-boxed external benchmark opportunity the programme is not yet
tracking.** PUBLISHED (web, 2026-09): NVIDIA's **AlpaSim E2E Closed Loop Challenge 2026** runs its "PAI"
track on the **PhysicalAI-AV NuRec** dataset — the same data family TanitAD already trains and evaluates
on, with an AlpaSim/NuRec harness already built (`stack/experiments/alpasim-gsplat/`,
`.../incoming/2026-07-22-alpasim-closedloop-evalpod/`). Opened 2026-06-15; rules/format froze 2026-09-15
(10 days before this review); **public leaderboard closes 2026-10-31** — about 5 weeks from today
(2026-09-25). `grep` across `LOOP_STATE.md`, `PROGRAM_OVERVIEW.md`, `BACKLOG.md` finds **zero** mentions —
this is new information to the programme. Caveat (UNVERIFIED): the PAI track's default sensor interface
is 4 cameras (front-wide, front-tele, cross-left, cross-right) at 1080×1920→320×576, 0.4 s history —
wider than TanitAD's front-wide-only, 3-frame-@100ms input; whether a policy may legally ignore 3 of the 4
streams is not established from public sources reachable this session. Detailed in §5.

**F8. `driving.py`'s own docstring records a refusal that the later binding rule overrides, and the
refusal was never revisited.** MEASURED, `taniteval/taniteval/driving.py:63-68` ("REFUSALS HONOURED"):
"headway / distance-keeping / TTC (no lead-agent state exists — `lead_state` is a `None` stub)." That
refusal predates the 2026-08-02 binding rule that makes distance-keeping mandatory, and predates the
2026-08-03 `obstacle.offline` lead-track build that makes it computable. `driving.py` — the module that
*is* wired into every default run — still does not carry it; the fix lives only in `four_families.py`
(unwired) and one `incoming/` script.

**F9. The same "instrument built, never wired" pattern recurs at least four more times, independently
diagnosed by different streams — this is systemic, not a one-off gap.** MEASURED: (a) `stack/tanitad/eval/
idm_families.py:8` states outright "no IDM script imports `four_families`" for the inverse-dynamics
model; (b) `stack/tanitad/eval/sitclf_deploy.py:7-16` states `sitclf.late_fuse_scores` "was written to fix
a MEASURED defect and then **had no caller**" because the situation classifier had no deployed scoring
path in `stack/` at all; (c) `taniteval/taniteval/control.py` (the dedicated longitudinal/lateral CONTROL
suite — dynamic-range validated, bidirectional, same-row-masked, more rigorous on several axes than
`four_families`) has no caller outside its own tests; (d) `taniteval/taniteval/lateral.py` (the M1 lat/lon
decomposition module) is consumed by `control.py` and one-off `incoming/` scripts, never by `runner.py`.
Each was independently discovered and documented by the agent who built it — the programme is aware of
each instance locally, but no one has closed the general pattern.

**F10. The prior R4 review's (2026-07-25) process-improvement recommendations show mixed uptake two
months later — some fixed same-day, one still open.** MEASURED: recommendation "promote a horizon-matched
closed-loop departure metric to gate co-primary" → **DONE** (`GATE_PROTOCOL.md` §0, amended 2026-07-26,
the day after). "Land the dense-path persistence at `rollout.py:94`" → **DONE**
(`four_families.py:527-533`, `prefer_dense`, since 2026-07-25). "Automated val-disjointness assertion" →
**DONE** (`taniteval/taniteval/data.py:111-124`, `list_val_episodes` hard-refuses the leaky
`f1b378` split unless `allow_leaky=True`). "Exercise route conditioning at least once" → **DONE**
(`hierarchy.py`'s `seam_nav_to_strategic`, `route_acc_follow` vs `route_acc_nav`, 2026-08-02/03). But
"fold C7/C8 into the RETRACTION_LOG legend" → **NOT DONE**: `RETRACTION_LOG.md:12-19` still lists only
C1–C6 and (new) C15; C7, C8, C13, C45, C56 are used as informal labels throughout the corpus (e.g.
`control.py:34` cites "class C13") but remain outside the log's own ranked table, the exact defect F10 of
the 07-25 review flagged.

---

## 2. Q1 — Four-family compliance matrix

**Instruments that exist, by family, with file:line, wiring status, and which arms/corpus they have
actually been run on.**

### LONGITUDINAL

| sub-metric | instrument | wired into `runner.run_one`? | arms actually run, corpus |
|---|---|---|---|
| speed MAE/bias/RMSE, along-track MAE/bias, accel MAE | `four_families.longitudinal` (`taniteval/taniteval/four_families.py:157-206`) | ⛔ No caller in `runner.py`/`bench.py` | ad hoc: `fourfam_v1-lf19`/`v2corpus-lf19` (19 eps), `fourfam_rrctl`/`rr20` (881 win / 40 eps, internal rollout-recovery arms), `four_family_panel_val40.json` (881win/40eps: `flagship-30k`, `refc-base-30k`, `cv`, `gt_oracle`) |
| speed-decoupled cross-track, L1/L2 cruise-vs-transient, along/cross split, progress (`progress_abs_err_m`) | `taniteval.driving` tier-0 (`driving.py`, ✅ wired, `runner.py:117`) + `taniteval.lateral` (M1, unwired) | ✅ driving.py tier-0 only | **25 of 27** `taniteval/results/driving_*.json` files, canonical 881win/40eps (2 are an unrelated 88-window/4-episode smoke corpus, MEASURED this session) |
| ego progress (arXiv-2605.00066-anchored, dt-invariant) | `taniteval.progress` (new, reused by `four_families._ego_progress`) | ⛔ not in `driving.py`, only via `four_families` | same ad hoc set above |
| **distance-keeping** (headway / time-gap / min-TTC vs lead agent) | `four_families._distance_keeping` + `taniteval.lead_metrics` (`four_families.py:222-270`) | ⛔ No — `driving.py:63-68` explicitly REFUSES it ("no lead-agent state exists") and that refusal was never revisited after the fix landed | admitted by control **D-LEAD-1** (2026-08-03, GT-vs-CV, 14,027 win/1,431 clusters — **not** the canonical val); computed for canonical 40-ep on **3 arms only** (`flagship-30k`, `refc-base-30k`, `cv`) in `four_family_panel_val40.json` (2026-08-04) |

### LATERAL

| sub-metric | instrument | wired? | arms / corpus |
|---|---|---|---|
| heading MAE, curvature MAE/bias, yaw-rate MAE, cross-track MAE/bias/final | `four_families.lateral` (`four_families.py:273-314`) | ⛔ | same ad hoc set as above |
| along/cross decomposition (ego + Frenet), M1 finding (ADE is 98.6% longitudinal by squared-error energy) | `taniteval.lateral` (`lateral.py`, full module) | ⛔ not in `runner.py`; consumed by `control.py` and `strategic_probes.py` and ~9 `incoming/` scripts | never on the full canonical set inside a committed result; always via one-off scripts |
| L3/T1–T4 (cross-track, curvature-stratified heading, curvature-sign agreement) | `taniteval.driving` tier-0 | ✅ wired | 25 of 27 `driving_*.json` arms on the canonical 881win/40eps corpus; 2 (`refc-v12-smoke-*`) are an 88-window/4-episode smoke corpus, MEASURED this session — do not mix them into a leaderboard comparison |

### TACTICAL

| sub-metric | instrument | wired? | arms / corpus |
|---|---|---|---|
| manoeuvre-decision accuracy + confusion + `never_predicted` | `four_families.tactical` → `_decision_family` (`four_families.py:317-354, 357-402`) | ⛔ | needs `maneuver_pred`/`maneuver_gt` in the window dict, which only a hierarchy-traversing pass emits |
| manoeuvre-vs-trajectory κ, tactical seam beneficial-of-3 | `hierarchy.run` (`hierarchy.py:428`) → consumed by `four_families.tactical` | ⛔ separate CLI subcommand (`runner.py` `hierarchy`/`hier-all`, **not** in `run_one`/`run-all`) | **flagship-family only** (`runner.py:341-342`: arch must carry `tactical_policy`+`strategic_policy`); measured values exist only for `v1-lf19`/`v2corpus-lf19` (**19 episodes**) |
| IDM-derived tactical (factored lat/lon manoeuvre, avoids the mixed-5-way-softmax defect) | `stack/tanitad/eval/idm_families.py` (new, dt-corrected) | ⛔ its own header (line 8) states no IDM script imports `four_families`; this module is the fix and is itself not yet wired into any IDM driver | tests only (`stack/tests/test_idm_families.py`) |
| situation classifier (lane_change/roundabout/intersection) four-family read | `stack/tanitad/eval/sitclf_deploy.py::four_family_report` | ⛔ | UNVERIFIED whether run on committed data — module is new (post-`sitclf.py`) and had no caller before it |

### STRATEGIC

| sub-metric | instrument | wired? | arms / corpus |
|---|---|---|---|
| route/goal accuracy under nav/follow/zeronav conditioning | `hierarchy.run`'s `seam_nav_to_strategic` → `four_families.strategic` | ⛔ same as TACTICAL above | `v1-lf19`/`v2corpus-lf19`, 19 episodes; **PRIVILEGED path** (`route_acc_nav`) explicitly barred from being read as skill — must use `route_acc_follow` vs `majority_straight_rate` (`four_families.py:474-490`) |
| map-derived option-set strategic scoring (supersedes the ego-yaw route label; the only path that can tell "no branch existed" from "didn't take it") | `taniteval.strategic_optionset.strategic_family` (`strategic_optionset.py:558`), + `conditioning_echo_control` (line 492) for the input-echo guard | ⛔ opt-in parameter to `four_families.strategic`, no default source of `optionset` in `runner.py` | requires `stack/experiments/nurec-gsplat/strategic_gt.py`-derived labels; UNVERIFIED how many arms have these labels built |
| goal-input admissibility (situation-classifier leak into the goal path) | `stack/tanitad/eval/goal_admissibility.py` (`echo_score`, `horizon_disjoint`, `incremental_information`) | Standalone instrument, invoked ad hoc | Found the v1 nav-echo (369/369 and 81/81 bijection) that made `route_acc_nav=1.0000` meaningless — a real catch, but not a standing gate check |

**Net compliance read.** LONGITUDINAL and LATERAL have a *de facto* substitute (`driving.py` tier-0,
genuinely wired, 25 arms on the canonical corpus) that covers most of what the binding rule asks — except
distance-keeping, which is refused in the wired path and only computed for 3 arms in an unmerged 2026-08-04
artifact. TACTICAL and STRATEGIC have **no wired path at all**, are architecturally restricted to the
flagship family, and have only ever been measured on a **19-episode** subset — never the 40-episode
corpus every ADE headline in the registry uses. **No arm in the registry's §6 leaderboard (the document
CLAUDE.md names as the only quotable source) carries a four-family block next to its ADE number.** Per the
binding rule's own clause 3 ("a missing metric is a work item, not an excuse"), this is an open work item
on every one of the 27 leaderboard rows, not a rounding gap.

---

## 3. Q2 — Statistical power at n=40 / 881 windows

**Estimator.** `taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap` (`ci.py:261-317`): resamples
the **40 episodes** with replacement (B=2000), recomputes `reduce(a)-reduce(b)` per draw, reports the
percentile interval; `separated = lo>0 or hi<0` on the **unrounded** bounds (`ci.py:285-289`). The unit of
resampling is the episode, not the window — 881 windows collapse to an effective n of 40 clusters.

**Empirical MDE bracket for ADE@2s (MEASURED, `MODEL_REGISTRY.md` §6, all n=40 episodes / 881 windows,
`taniteval/ci.py` paired form):**

| pairing | Δ ADE@2s (m) | CI95 half-width | separated? |
|---|---:|---:|:--:|
| flagship v1 vs REF-C-XL (rank 1 vs 1=) | 0.0443 | ≈0.100 | **no** |
| REF-C vs REF-B v2 (different family) | 0.1199 | ≈0.056 | **yes** |
| flagship v1 vs REF-B v2 (different family) | 0.1642 | ≈0.121 | **yes** |
| flagship v1-30k vs v1-19k (same ckpt lineage) | 0.1881 | ≈0.038 | **yes** |
| flagship v1 vs REF-A (different architecture) | 2.6200 | ≈0.581 | **yes** |

**Reading.** There is no single MDE — the half-width depends on how correlated the two arms' per-window
errors are (a mid-training checkpoint of the *same* run is tightly paired; two different architectures are
not). As a practical rule: **a paired ADE@2s delta needs to be roughly ≥0.10–0.15 m to be reliably
separated at n=40/881 between architecturally different arms**, and finer than that between closely
related checkpoints of the same lineage. This directly explains why the programme's own headline —
"ranks 1–3 are a statistical tie" (Δ 0.0443, 0.0013–0.0054 pairwise) — sits below this floor while
"flagship beats REF-B v2" (Δ 0.164) sits above it.

**Family metrics beyond ADE — power is unknown, not merely wide, for TACTICAL/STRATEGIC**, because they
have never been run on more than 19 episodes (F2) — there is no in-repo empirical CI to anchor an MDE the
way §6 anchors ADE's. The only adjacent number is `hierarchy.py`'s own bootstrap on the 19-episode set:
e.g. `maneuver_vs_trajectory_agreement` CI95 half-width ≈0.045 at n=418 windows/19 episodes
(`fourfam_v1-lf19.json`) — narrower than ADE's because accuracy has lower per-window variance, but this is
**INHERITED, not re-derived**, and it is on fewer than half the canonical episodes.

**Would the 600-episode deployment fix it?** MEASURED, `MODEL_REGISTRY.md:226-266` (§1.2a): flagship v1's
600-episode read has CI half-width **0.0159** vs the 40-episode **0.0299** — a **×2.8–3.9 (mean ≈3.4)**
shrink across 8 open-loop metrics, MEASURED against a √15≈3.87 theoretical prediction (`v1_40_vs_600.json`)
— so yes, more episodes buys real, close-to-theoretical power. **But it does not fix comparability**: the
600-episode corpus is explicitly a **different, easier** deployment (CV floor 0.8377→0.6917, §1.2a), every
other arm in the registry is evaluated only on the 40-episode canonical set, and the registry states in
terms that leave no ambiguity: *"do not substitute a 600-episode number into this table... no arm's
40-episode number may be compared to another arm's 600-episode number."* Switching wholesale would orphan
the entire existing leaderboard; a verdict already flipped once on power alone at n=600
(`along_track_vs_cv`: "tie" at 40 → "separated" at 600, point estimate moving only 0.7%).

**Recommended val design (extends a pattern the programme has already used once, for the pseudosim
composite — `.../2026-07-27-small-validation/SMALL_VALIDATION.md:135-172`, INHERITED, not re-verified by
me beyond reading the doc):**
1. **Freeze the 40-episode / 881-window corpus as the permanent primary comparison set** — every historical
   number is on it; do not retire it.
2. **Pre-register a nested escalation ladder — 40 ⊂ 120 ⊂ 600 episodes** (order-preserving prefix, as
   `MODEL_REGISTRY.md:229-230` already verified for the 600-episode build: `published40[i]==val600[i]` for
   `i∈[0,39]`) for any *specific* comparison whose n=40 MDE (≈0.10–0.15 m per the bracket above) exceeds the
   effect being tested — computed **before** running, not after seeing a "not separated."
3. **Apply the same ladder to the four-family metrics** once TACTICAL/STRATEGIC are wired (§6 below) — they
   currently have no MDE anchor at all because they have never run past 19 episodes.
4. **Never mix corpora inside one comparison table**; state n (both windows and episode clusters) on every
   quoted interval, per `MODEL_REGISTRY.md:250`'s own rule.

---

## 4. Q3 — Closed-loop: which instrument should decide, and what open-loop↔closed-loop evidence exists

**Three closed-loop instruments exist, answering three different questions, and none is a full
closed-loop safety test:**

| instrument | what it measures | fidelity | status |
|---|---|---|---|
| `taniteval.closedloop` (imagination-in-the-loop, `closedloop.py:1-90`) | drift when the flagship WM consumes its **own** predictions (distribution shift) | **self-referential** — "the loop cannot surprise itself with an event outside its own imagination" (own docstring); no collision/drivable-area (no map/boxes) | wired as `runner.py` subcommand `closedloop`/`closedloop-all`, not part of `run_one` |
| AlpaSim NuRec (`stack/experiments/alpasim-gsplat/`, `.../2026-07-22-alpasim-closedloop-evalpod/`) | external-simulator pass/fail + progress, on **photoreal reconstructions** | ⚠️ reconstruction-OOD confounded: REF-C's open-loop ADE **on the reconstructions** is 1.52 m vs 0.4728 real-footage — **3.21×** off-distribution (`REFC_openloop_diagnostic.json`, `MODEL_REGISTRY.md:1641-1646`, RETRACTION C6) | n=12 scenes (subset of a 916-scene public suite; 1 scene = 8.3 pp); the correct paired comparison is `flagship_vs_refc_suite_results.json` (REF-C-base **8/12** vs flagship v1 **2/12** pass, sign-test p=0.0078, McNemar p=0.031) — see F6 for the superseded-file trap |
| `taniteval.pseudosim` (NAVSIM-v2-style pseudo-simulation, `pseudosim.py:1-70`) | bounded-grid perturbation (heading + longitudinal only; **lateral REFUSED** on measured geometry — flat-road warp error = 100% at camera height) — deviation is *chosen*, not accumulated, so it cannot leave the validated envelope | the only closed-loop-shaped instrument that is a **measurement**, not an extrapolation, at any horizon (sequential closed loop leaves the envelope on 12.3%→90.2% of windows as K goes 20→185) | composite is explicitly **"NOT a Driving Score"** (`pseudosim.py:1527-1532`) — no collision gate exists; `filter_m`/EPDMS contract recorded but **NOT applied** (needs a human-reference rollout — `pseudosim.py:1589-1614`; `BACKLOG.md:17` "Still NOT applied — needs a human-reference rollout + the chunk download") |
| `taniteval.clhorizon` (long-horizon closed loop, K up to 185/18.5s) | the E1a finding: corridor-departure rate **0.0035→0.5877** (0.8414 at junctions) as horizon goes 2s→18.5s, on the **same 43 windows** where paired ADE@2s barely moves (Δ0.0109, not separated) | ✅ this IS in-repo open-loop-vs-long-horizon-closed-loop divergence evidence — the strongest such evidence in the repo, though it is a horizon effect, not a correlation coefficient | promoted to gate **co-primary** in `GATE_PROTOCOL.md` §0 (2026-07-26); `⚠️ RETRACTED 2026-07-26`: the "in-distribution" certificate on this finding was withdrawn (the OOD ratio estimator saturates), leaving the verdict as **EXTRAPOLATION at K=185 for every arm measured** — the horizon finding itself stands, the in-distribution claim does not |

**Which should decide?** Per `GATE_PROTOCOL.md` §0.2/§0.5, the programme has already made this call:
**`corridor_departure_rate` at a pre-registered horizon (20<K≤190) on the closed-loop surface is
co-primary; `ade_0_2s` is demoted to a proposal-quality diagnostic.** This review concurs and adds one
qualifier: at K=185 the finding is EXTRAPOLATION, not in-distribution measurement (the OOD ratio estimator
that certified it saturates at |dlat|=3m/|dψ|=12°, per `taniteval/ood.py`'s fix) — so the co-primary
should be read as "the failure exists and is this large," not "this is what happens in-distribution."
`pseudosim`'s bounded-grid protocol is the more defensible **measurement**-class instrument, but it is not
a Driving Score (no collision gate) and its EPDMS-shaped extension is contract-only, unapplied.

**Open-loop↔closed-loop correlation evidence in-repo:**
- **No in-repo regression exists** with enough arms to compute a correlation coefficient (AlpaSim has 2
  models × 12 scenes; that is an ordering, not a correlation sample).
- The programme's own rationale for building `progress.py`/`four_families.longitudinal.ego_progress` is an
  **external PUBLISHED** correlation: arXiv 2605.00066 (Wang et al., 2026-04-30), n=8 methods,
  NAVSIM×Bench2Drive — traditional L2 (ADE/FDE) vs closed-loop Driving Score **ρ=−0.36, p=0.43 (not
  significant)**; PDMS aggregate ρ=0.90; **Ego Progress alone ρ=0.83** (`taniteval/taniteval/progress.py:
  1-10`, `tanitad_metrics.py:1-12`). Scepticism stated in-repo and repeated here: n=8, no CI, two differing
  benchmarks — direction only, not a baseline to claim beating.
- The E1a horizon finding (`clhorizon.py` row above) is the closest **in-repo** analogue: it demonstrates
  that a 2s open-loop metric is blind to an 18.5s closed-loop-relevant failure on the *same* windows —
  evidence of divergence, not a correlation number.
- **External, newly surfaced (this session, PUBLISHED, web 2026-09):** Waymo's WOD-E2E (arXiv 2510.26125,
  CVPR 2026) reports its Rater Feedback Score (RFS) "vastly outperforms ADE (82% vs 51%) top-1 accuracy"
  at picking the closed-loop-optimal trajectory, p<0.001 — the most direct published evidence that
  open-loop displacement metrics under-predict closed-loop quality, independent of the arXiv 2605.00066
  citation already in-repo. Worth adding to `TANITEVAL_V2_METRIC_SUITE.md`'s citation list.

---

## 5. Q4 — External benchmarks: cheapest route to one recognised number

Given: front-camera-only input, no maps, vision-only at inference (binding, CLAUDE.md 2026-08-03).

| benchmark | structurally computable for TanitAD? | why / why not | current PUBLISHED standings (web, 2026-09; dates as cited) |
|---|:--:|---|---|
| **NAVSIM v2 EPDMS (navhard)** | ⛔ **No** — already established in-repo | Needs agent boxes, drivable-area polygons, route centreline, traffic-light state — none exist in PhysicalAI-AV (`LEADERBOARD.md:663`: "structurally not computable... a partial PDMS is never published") | DrivoR 56.3, DriveFuture 55.5, PDM-Closed baseline 51.3 — all navhard EPDMS, all PUBLISHED and dated 2026-03/04/06 (`LEADERBOARD.md:660-662`, in-repo); my own 2026-09 web search returned a same-order-of-magnitude but imprecisely-dated "50.9" for "more recent approaches" — **not independently confirmed this session; treat the in-repo dated citations as current until someone reads the live HF leaderboard** |
| **Bench2Drive / CARLA closed-loop** | ⚠️ Partial — CARLA town rendering, not our data; would need a from-scratch sim policy, not a repurposed eval | `LEADERBOARD.md:671`: "Phase 1; MetaDrive closed-loop first, then CARLA/Bench2Drive" — not started | in-repo cites TF++ (VLAAD-MIL) DS 86.97 / SR 71.97% (arXiv 2603.25946, 2026); my web search independently returned AutoVLA DS 78.84 as "top" from a different 2026-09 survey (arXiv 2609.01659) — **these two PUBLISHED figures disagree in ordering and I could not reach the live CARLA leaderboard (egress-blocked) to adjudicate; flag as UNVERIFIED which is current** |
| **nuPlan (closed-loop CLS / OLS)** | ⛔ No — needs the nuPlan simulator + HD map + background-agent IDM (`TANITEVAL_V2_METRIC_SUITE.md:334`: "REFUSE. Not approximable.") | — | original nuPlan planning challenge is widely understood to have concluded years ago; my web search's claim of an active 2026 "soft-open" nuPlan challenge is a low-confidence, possibly-conflated search synthesis (no primary source reached) — **do not act on it without a direct source check** |
| **Waymo WOD-E2E (RFS)** | ⚠️ Partial — open-loop-style, rater-preference-based, camera-only by design (no HD map needed) — closer to a computable fit than NAVSIM/nuPlan | Uses **8 surrounding cameras** + routing info (PUBLISHED, arXiv 2510.26125, CVPR 2026); TanitAD is front-wide-only — would need to check if partial-camera submissions are scored, or accept scoring a front-wide crop against a full-surround expectation | New (2025 challenge, CVPR-2026 paper); UNVERIFIED current 2026 leaderboard standings — not reachable this session (egress-blocked to relevant hosts) |
| **⭐ NVIDIA AlpaSim E2E Closed Loop Challenge 2026 (PAI track)** | ⚠️ Best structural fit of all four — same **PhysicalAI-AV NuRec** data family TanitAD already trains/evaluates on, and TanitAD already has a working (if reconstruction-OOD-confounded for open-loop) AlpaSim harness | Opened 2026-06-15; rules/format froze **2026-09-15**; **public leaderboard closes 2026-10-31** (~5 weeks from today). Default PAI sensor config is **4 cameras** (front-wide, front-tele, cross-left, cross-right) @ 1080×1920→320×576, 0.4s history @10Hz — wider than TanitAD's front-wide-only 3-frame input; **UNVERIFIED whether a policy may legally use only 1 of the 4 provided streams** — I could not reach the devkit/rules text this session (egress-blocked to huggingface.co, arxiv.org, the challenge's own `.hf.space` host, kesai.eu) | New; not yet in any TanitAD doc (`grep` across `LOOP_STATE.md`/`PROGRAM_OVERVIEW.md`/`BACKLOG.md` = 0 hits) |

**Recommendation — cheapest route, ranked:**
1. **AlpaSim E2E Closed Loop Challenge 2026, PAI track (time-critical).** Reuses the existing
   `stack/experiments/alpasim-gsplat/` + `2026-07-22-alpasim-closedloop-evalpod/` harness almost directly.
   First step (0 GPU, <1 day): pull the challenge's rules/devkit from a machine with open egress and
   answer (a) is front-wide-only eligible or must the model at least accept 4 camera streams, (b) is the
   PhysicalAI-AV split used by the challenge disjoint from TanitAD's own train corpus
   (`physicalai-train-e438721ae894`) — if it overlaps, any submitted number needs the same leakage guard
   `taniteval/taniteval/data.py` already enforces. **This is the single highest-leverage Q4 action and it
   is on a real clock** (~5 weeks to close).
2. **WOD-E2E RFS**, as a second, slower-moving target — camera-only in spirit, no map dependency, and its
   own published validation (82% vs 51% top-1 vs closed-loop) makes it a stronger proxy than raw ADE even
   before any submission. Needs confirming whether a front-only crop is scoreable.
3. **NAVSIM v2 EPDMS / Bench2Drive / nuPlan**: correctly refused in-repo already; nothing to add except
   "re-check the live leaderboard before quoting a number" given the ordering conflict this session
   surfaced between two PUBLISHED sources for Bench2Drive.

---

## 6. Q5 — Gate protocol & metric pitfalls

**Soundness — what `GATE_PROTOCOL.md` gets right (MEASURED from the doc):**
- Refuses to decide from a train-log slope (`check` → `BLOCKED` without `--eval-json`, §1–2).
- Refuses a bare exponent below R²<0.80, caps extrapolation at 2× the fitted window (§3) — with a MEASURED
  demonstration on the same log giving −0.421 to −1.021 depending on the fit window (§3 table).
- Refuses `overlapping_holdout_se` at gate time (§0.5) and flags a live tripwire against renaming the
  `heldout` block key that the refusal keys on (§0.5's "do not clean up" note).
- **§0.7 VOID SECONDARIES**: a secondary whose value is fixed by a label/harness defect (e.g.
  `nonav_route_beats_majority`, forced to 0.0 by construction because the route target is literally a
  lookup of the route input) is adjudicated INSTRUMENT-FAIL, never MODEL-FAIL, and excluded from the kill
  conjunction — this is a genuinely good pattern, directly reusable for the four-family STRATEGIC nav-echo
  problem (§2 above).
- **§0.8 PRIVILEGED-INPUT PRIMARIES**: a gate metric computed from ego-future-derived inputs must be
  stamped `goal_provenance: ORACLE` and may not be worded as a deployed-capability claim — this is the
  in-repo precedent for the newer binding rule (2026-08-03) that a goal signal must not carry the
  situation-classifier's output; the two rules should be cross-referenced explicitly (they currently are
  not — `GATE_PROTOCOL.md` predates the classifier-leak rule by a week and does not cite it).

**Horizon-blindness — the single largest correction, already made:** `GATE_PROTOCOL.md` §0 (added
2026-07-26, the day after the prior R4 review flagged it as "the single most consequential open exposure,"
F1) demotes `ade_0_2s` to a diagnostic and installs `corridor_departure_rate` at a pre-registered horizon
as co-primary. **Residual gap this review adds:** the horizon fix applies to the *gate* path
(`run_gate.py`); it has not propagated to the *four-family* LATERAL family, which still reports only
2.0s-horizon cross-track/heading/curvature (`four_families.lateral`, `driving.py` tier-0) — the exact axis
E1a showed to be 168× more informative at 18.5s than at 2s. A LATERAL family computed only at K=20 is
subject to the same blindness the co-primary was built to fix.

**Estimator issues (heldout vs full_set) — status: fixed in tooling, not fully purged from prose.**
`ci.py` is correct and consistently used for every *standing* decision (prior review's own A− grade,
independently reproduced here). Residual: `MODEL_REGISTRY.md` itself still prints
`legacy_split_mean ± overlapping_holdout_se` columns beside every headline (labelled deprecated, but
present) — per the prior review's F7, a reader who stops at the bold number, not the caption, still sees a
tight `±` next to the headline. **This review's own re-check confirms it is still true at HEAD**: §6's
table (read this session) prints both columns on all 15 rows.

**Trainer-log vs eval discrepancy (~10% optimistic, historically):** CLAUDE.md's own worked example —
"v1.6 is best-in-program" was a trainer log, ~10% optimistic vs `eval_*.py` — is logged as `RETRACTION_LOG.md`
class **C1** ("faster-moving source than the harness"), 2026-07-21 (`RETRACTION_LOG.md:29`). This class is
the **3rd-most infrequent** of the six ranked classes in the prior review's frequency table (3/42 tokens,
7%) — i.e., already among the rarer failure modes by 2026-07-25, consistent with `driving.py`/`efficiency.py`
being wired as **default axes** on every `run_one` call (`runner.py:106,117`) so a held-out number is
produced automatically rather than requiring a separate step someone might skip.

**A naming collision worth flagging (minor):** there are two unrelated things called "gates" in this
codebase — `stack/tanitad/eval/gates.py` (the Phase-0 **D1–D3 decode-gate** runner, instrument-doctrine
gated, `gates.py:1-30`) and `stack/scripts/run_gate.py` (the **restart/continue** GATE_PROTOCOL CLI, 2196
lines). They serve different programme eras (Phase 0 decode gates vs the post-2026-07-20 restart/continue
protocol) and are not cross-referenced from either file's docstring. Low risk, but worth a one-line
disambiguation comment in each.

---

## 7. Q6 — Recommended decision-grade eval stack v3 + external-benchmark plan

**Ranked recommendations** (fix · defect · expected effect · cost · risk · cheapest experiment):

1. **Wire `four_families.all_families` into `runner.run_one` as a default axis, exactly as `efficiency`
   and `driving` already are (`runner.py:106,117`).**
   Defect: F1 — the binding rule has zero committed compliance. Effect: every future
   `taniteval/results/driving_*.json` carries LONGITUDINAL/LATERAL/TACTICAL/STRATEGIC blocks (TACTICAL/
   STRATEGIC will legitimately read UNAVAILABLE for non-flagship arms — that is the honest state, and the
   module already renders it correctly). Cost: **~0.5 eng-day** (the module is written and tested; this is
   a wiring change plus re-running `driving-all`, CPU-only, no GPU, per `driving.py`'s own header). Risk:
   low — `all_families` is exception-safe and already unit-tested (`test_four_families_dt.py`,
   `test_strategic_optionset.py`). Cheapest experiment: run it once on `flagship-30k` and `refc-base-30k`
   and diff against the already-banked `four_family_panel_val40.json` — should reproduce bit-for-bit.

2. **Fold `four_families._distance_keeping` into `driving.py` and retire the "REFUSALS HONOURED"
   docstring line (`driving.py:63-68`) now that `obstacle.offline`-derived lead tracks exist.**
   Defect: F8. Effect: closes the LONGITUDINAL gap on the wired path instead of only in an unwired one.
   Cost: **~1 eng-day** (mostly plumbing the lead block through `rollout.collect`). Risk: low — the D-LEAD-1
   control already validated the metric (GT vs CV, separated on all three sub-metrics).

3. **Build a hierarchy-traversing pass that runs on the full 40-episode canonical val, for every
   flagship-family arm, and commit it to `taniteval/results/`.**
   Defect: F2/F3 — TACTICAL/STRATEGIC have never been measured past 19 episodes, and never for REF-B/REF-C.
   Effect: closes the biggest compliance gap in §2; gives TACTICAL/STRATEGIC their first real MDE anchor.
   Cost: **~1–2 eng-days** for the flagship-family run (mostly re-running `hierarchy.run` at `episodes=40`
   instead of 19 — CPU/GPU cost is one more `hierarchy` pass, cheap relative to a training run); REF-B/REF-C
   support would require a genuinely new decision surface (their architectures have no `tactical_policy`/
   `strategic_policy` heads) — **flag this half as a design question for the PI**, not an eng estimate.
   Risk: medium — the flagship-only restriction is architectural, not a bug, so "TACTICAL/STRATEGIC:
   flagship only" may simply be the honest ceiling until REF-B/REF-C grow the equivalent decision heads.

4. **Add a `SUPERSEDED_BY:` marker (or delete) the six-scene solo AlpaSim result files
   (`REFC_suite_base_results.json`, `REFC_suite_xl_results.json`) now that the paired rerun exists.**
   Defect: F6. Effect: removes a live citation trap. Cost: **~1 hour** (a one-line header edit; these are
   `incoming/` artifacts, not code — do not delete without checking nothing downstream still reads them).
   Risk: none.

5. **Cross-reference `GATE_PROTOCOL.md` §0.8 (privileged-input primaries) with the 2026-08-03
   goal-admissibility binding rule; consider promoting `goal_admissibility.py`'s three checks
   (`echo_score`, `horizon_disjoint`, `incremental_information`) to a standing pre-gate check, the same way
   §0.7's void-secondary pattern is standing.**
   Defect: two independently-discovered instances of the same failure (nav-echo, §0.8; the classifier-leak
   rule) with no shared standing check. Effect: the next echo-shaped defect gets caught mechanically instead
   of by a fresh audit. Cost: **~1 eng-day**. Risk: low.

6. **Escalate the AlpaSim E2E Closed Loop Challenge 2026 to the PI immediately — it is on a real clock.**
   Not a code fix; a scheduling decision. Cost of the *investigation* (confirm sensor-interface eligibility,
   confirm train/eval split disjointness against the challenge's own PhysicalAI-AV split): **~1 eng-day, 0
   GPU**. Cost of an actual submission, if eligible: **UNKNOWN pending the devkit** — flag as ESTIMATED
   3–5 eng-days plus whatever GPU the AlpaSim harness needs per rollout (the existing 12-scene harness took
   the programme roughly one pod-day in July, per the incoming bundle's scope — ESTIMATED, not re-timed
   this session).

7. **Apply the nested 40⊂120⊂600-episode escalation ladder (§3) the next time a specific paired comparison
   needs power beyond the n=40 bracket**, rather than ad hoc per-stream re-derivations of an MDE (as
   `SMALL_VALIDATION.md` already did once, well, for the pseudosim composite).
   Cost: **design is free** (the prefix-nesting is already verified); per-arm eval cost scales as already
   measured (~13 min/arm at 40, ~40–50 min/arm at 120, ~3.2h/arm at 600 — INHERITED from
   `SMALL_VALIDATION.md:156-161`, not re-timed this session).

8. **Fold C7–C15's informal usages into `RETRACTION_LOG.md`'s ranked legend** (F10's still-open item).
   Cost: **~1 hour**. Risk: none. Purely a hygiene fix, but it is the one item from the prior review that
   is both cheapest and still outstanding two months later — a useful canary for whether *this* review's
   own recommendations get actioned.

**What a "minimal decision-grade eval stack v3" looks like, concretely:** `runner.run_one` emits, per arm,
in one committed JSON: (a) `driving.py` tier-0 [existing, wired] + distance-keeping [item 2] for
LONGITUDINAL/LATERAL; (b) `four_families.tactical`/`.strategic` populated where `hierarchy.run` support
exists, honestly UNAVAILABLE elsewhere [item 3]; (c) the co-primary `corridor_departure_rate` at a
pre-registered K [already standing, `GATE_PROTOCOL.md` §0] computed on the same arm; (d) every interval
via `taniteval/ci.py` paired episode-cluster bootstrap, n stated as both windows and episode clusters. That
is the full binding contract (CLAUDE.md's four-family rule + the horizon rule + the estimator rule) in one
pipeline instead of three unwired ones plus a gate script.

---

## 8. Open questions / UNVERIFIED

- **AlpaSim E2E Closed Loop Challenge 2026 sensor eligibility** (§5, F7): whether a front-wide-only policy
  can legally compete on the PAI track, or must accept (even if internally ignoring) the other 3 camera
  streams. Could not resolve this session — huggingface.co, arxiv.org, the challenge's own `.hf.space`
  host, and kesai.eu were all egress-blocked to WebFetch from this container.
- **Current Bench2Drive/CARLA leaderboard ordering**: two PUBLISHED 2026 sources disagree (TF++ 86.97 DS
  in-repo vs AutoVLA 78.84 DS from an independent 2026-09 web search) — UNVERIFIED which is current; the
  live leaderboard was not reachable this session.
- **Current NAVSIM v2 navhard EPDMS top score**: in-repo citations (dated 2026-03/04/06) are internally
  consistent; my own web search's "50.9" figure for "more recent approaches" could not be pinned to a
  dated, named system — treat as unconfirmed noise, not a correction.
- **nuPlan's actual 2026 status**: a web search returned a description of an active "soft-open" 2026
  challenge that reads as plausibly conflated with the AlpaSim challenge or another unrelated benchmark; no
  primary source was reached. Do not act on this without a direct check.
- **Whether `sitclf_deploy.py::four_family_report` has been run on any committed data** — the module is new
  and I found no output artifact for it in the time available; UNVERIFIED, not confirmed absent (per
  CLAUDE.md's own "absence at one location is not absence" rule, this needs a second probe I did not have
  budget for).
- **`pytest -q` status**: CLAUDE.md states it "must stay green before any commit." This review's container
  has no torch and could not execute the suite (most of `taniteval/tests/` and `stack/tests/` import
  `torch`); I did not attempt a partial run. **This is a real gap in this review, not a finding about the
  code** — flagging honestly per the operating rules rather than guessing a pass/fail.
- **Exact GPU-hour cost of an AlpaSim challenge submission**: ESTIMATED only, from the scope of the existing
  July harness; not re-timed.

---

## 9. Deliverable manifest

| artifact | location |
|---|---|
| This report | `repo:Project Steering/Reviews/2026-09-25-programme-review/streams/R4_eval_metrics_closedloop.md` |

No code, data, or config files were modified or created by this stream — this is a read-only, static
analysis. No other artifacts exist to strand: everything produced is in the one file above, staged into
the working tree per the operating rules (this stream does not commit or push).
