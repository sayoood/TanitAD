# TanitAD Programme Review

**Date:** 2026-09-25 · **Commissioned by:** Sayed (PI) · **Repo state reviewed:** `main` at `467ce8a` (last commit 2026-08-04) · **Scope:** the model and its hierarchy, training and anti-collapse, data and encoding, evaluation, the agent harness, frontier research that can be adapted, and a plan.

**How to read this.** Section 1 is the verdict and the five decisions it asks for. Sections 3–9 are the review, one area each, with every claim tagged by evidence class. Section 10 maps frontier research onto the programme's defects. Sections 11–14 are the strategy: proven levers, disruptive bets, the fix list and the plan. Stream reports with full file:line evidence are in `streams/`. Model facts follow `Project Steering/MODEL_REGISTRY.md` and raw result JSONs; ADE numbers are full-set means with episode-cluster bootstrap CI95 over the canonical 881 val windows / 40 episodes unless stated.

**Evidence classes:** MEASURED (programme artifact, path given) · PUBLISHED (cited paper) · INHERITED (another document, not re-verified here) · ESTIMATED · HYPOTHESIS · UNVERIFIED. "CODE" marks a static fact read from source at the cited file:line.

---

## 1. Verdict

TanitAD has built three things of lasting value:
- **An unusually rigorous measurement culture.** Evidence classes, pre-registration, a paired episode-cluster bootstrap, and a retraction log organised by root cause.
- **A camera world model whose latent integrates given driving controls about as well as a kinematic bicycle model:** 0.4271 [0.3675, 0.4871] vs 0.4518 [0.3097, 0.6174]. The comparison is unpaired and not separated; the paired delta has not been computed.
- **A credible embedded path.** A v1 tick (encoder, heads, a 9-manoeuvre imagined fan, scoring) runs in 60.3 ms p50 / 63.1 ms p95 on Jetson Thor against a 100 ms budget (MEASURED, `TanitAD Research Hub/Production & Optimization/THOR_DEPLOYMENT_RUNBOOK.md:286-304`, ±13 % run-to-run). That selector path has no open-loop score yet. NuRec scenes render on Thor; where the closed-loop driver's inference ran is not recorded.

It has not yet built a *hierarchical* driver that beats its own flat reference. REF-C clears all three trivial floors (CV separated; CTRV and best-of-3 on the point estimate, `MODEL_REGISTRY.md` §6), while the hierarchy's two decision paths (1.9028 plan-tracked, 3.3839 direct) do not clear CV (0.8377). The review now explains why, and the explanation changes the plan.

**1. The headline "tie" is not a driving result.** The flagship's 0.4271 m is scored by feeding the world model the expert's recorded future steering and acceleration (CODE `taniteval/taniteval/rollout.py:146-147`; the harness itself labels it `actions_source="expert_future"`, `honest_metric_name="wm_fidelity_ade_2s"`, `:181-185`). A zero-parameter bicycle model given the same controls scores 0.4518 [0.3097, 0.6174] (MEASURED, `…/incoming/2026-07-26-closedloop-artifact-rerun/closedloop_flagship-30k.CORRECTED.json`). REF-C, which must predict the future, scores 0.4728. The registry has recorded this since 2026-07-27 (`MODEL_REGISTRY.md:192`), yet its §6 leaderboard still ranks the two as "1=". On the two paths where the hierarchy itself decides, the flat 104 M REF-C-base beats it by 4.0× to 7.2× (0.4728 vs plan-tracked 1.9028 and tactical head 3.3839, all full-set, `closedloop_flagship-30k.CORRECTED.json`). The P2 CEM planner's 0.893 is a deprecated split-mean of a path handed a future speed target, and is not comparable (registry §6 row 10).

**2. The hierarchy is not running as designed.**
- The tactical and strategic cadences are configured but read by no code (`config.py:118,140`).
- All three brains see the same 0.8 s window.
- The strategic brain is trained on an echo of its own label and receives a constant command at inference.
- **0 of 3 inter-brain seams are load-bearing.** The July review's "1 of 3" was an artefact of the deprecated estimator.
- About 85 M of the 263 M parameters sit off every scored path.
- In the v4/v5f line a further 22 M (the H15 imagination field) never receive a gradient (CODE, verified; §4 and §6).

**3. The decision defect is a missing cost model and a wrong training target, not a missing layer.**
- **The wrong target.** Selectors are trained to guess which candidate happened to be closest to one realised future, not to minimise expected cost.
- **The missing feature.** No arm receives the single most informative longitudinal observable, a 0.1 s speed difference.
- **What the right target plus that feature achieved.** A per-candidate expected-cost regressor over REF-C's fan, given that feature, reduced ADE from 0.5015 to 0.3917 with no goal input, and to 0.3040 with a `(v, ax_fd)` goal head. These are MEASURED-by-path (`…/incoming/2026-07-28-egoal-4-joint/raw/e4_select_sel.json`) but out-of-fold *within* val-600; the selector was never trained on the train corpus. They are also 600-episode numbers, a different and easier deployment (CV floor 0.692 vs 0.838; REF-C-XL's own pick there is 0.5015), so they are not comparable with the 40-episode figures in claim 1. This is still the programme's largest measured decision-quality lever, and it came from a better target and a better feature, not from more hierarchy. It must replicate on train-corpus fans before it decides any GPU-days (decision D2a).

**4. The longitudinal gap is largely a data and interface fact.**
- **Noisy acceleration channel.** The dataset's logged acceleration correlates only r = 0.434 with the pose-derived one (INHERITED, `IDM_DIAGNOSIS.md`).
- **No speed history.** Ego speed is broadcast as a constant across the whole window, so the world model never sees how speed has been changing (CODE `flagship_losses.py:227-238`).
- **The lead car is barely visible.** At the parity crop's 4.98 px/deg, a car 50 m ahead is about 10 px wide (ESTIMATED, R3).
- **The speed bias is a compounding artefact.** The world model is trained at 4 rollout steps and scored at 20. A 20-step fine-tune removed the bias (driving harness +0.19 → −0.002 m/s; four-family instrument +0.94 → −0.009 m/s) and cut WM-fidelity ADE from 0.424 to 0.348 (MEASURED, raw JSON in `…/incoming/2026-08-02-rollout-recovery-verdict/`, not yet registered; primary `CR_k` pending).

**5. The strategy is running on a sliver of its assets and has skipped its deciding experiments.**
- **Data usage is a sliver.** The programme trains on 0.78 % of PhysicalAI-AV: 13.2 h of about 1,700 h, 1 of 7 camera streams, no image augmentation (R3; dataset size MEASURED by the in-repo feature probe `…/2026-07-26-physicalai-feature-probe/PHYSICALAI_FEATURE_PROBE.md` and PUBLISHED on the Hugging Face card).
- **The deciding experiments were skipped.** The three experiments that decide the thesis were not run: hierarchy vs flat, the data-efficiency slope, and one recognised external number. The July review asked for all three.
- **Where things stand.** The repository has been silent for 52 days. The mission plan's first final evaluation is on **05.10.2026**, ten days from now.

### What to present on 05.10.2026, and what not to

Do not present 0.4271 as a planning result or the hierarchy as validated. Present:
- **A camera-only world model** whose latent integrates given controls as well as a kinematic model (not separated), and a v1 selector tick that runs inside budget on Thor. State that the accuracy of that tick's path is not yet measured.
- **A flat planner that drives,** plus a measured, pre-registered route to making it better through cost-aware selection.
- **A safety envelope** designed around it.
- **An evaluation apparatus that caught its own headline error.** That last point is a strength when stated plainly.

### The five decisions this review asks of the PI

| # | decision | why | cost |
|---|---|---|---|
| D1 | **Split the leaderboard into two tables.** Planners (the model chooses) and world-model fidelity (given the expert's controls). No rank may cross them. | Row 1= compares unlike surfaces; everything downstream inherited it | 0 GPU, 1 eng-day |
| D2 | **D2a, now:** retrain the expected-cost selector on parity-train fan dumps (Δ2) and score it paired on val-40 and val-600; E-GOAL-4 graduates only if its gain survives. **D2b, on graduation:** make REF-C the driving baseline-of-record and build v6 around the decision layer. That means REF-C's fan (or a factorised lat×lon fan), the expected-cost selector with `ax_fd`, multi-target costs distilled from `obstacle.offline` tracks, a K = 20 world model as a consequence feature, a kinematic decoder, and a safety envelope | the only lever with a large measured effect is selection; the hierarchy has 0/3 load-bearing seams | D2a ≈ 0.5 A40-day; the whole ladder ≈ 5.8; one full v6 run after it ≈ 5–7 |
| D3 | **Re-scope the strategic brain to what the mission defines and the data can supervise.** A supervisor (engage/degrade/ODD, calibrated uncertainty, runtime assurance) plus a predicted goal point. Pre-register the tactical-level test now (Δ4, 0.3 A40-day, frozen dumps); pre-register the strategic-level test once a corpus with a route or goal source exists (AlpaSim routes, L2D) | on this corpus the strategic level has no information the operative lacks (R1 §2.4) | 0 GPU to re-scope; the test later |
| D4 | **Run the data-efficiency slope now.** Label-free world-model pretraining on the unused part of PhysicalAI-AV's *train* split (never its val/test splits or any clip whose NuRec scene is in the closed-loop suite or the challenge), and later LFG-style front-camera video, then supervised heads at 1/3/10/30/100 % of labelled hours, against REF-C at the same fractions, on the four families | "1000× less data" is the mission's first priority goal, and it has never been measured | ≈ 10–25 A40-days for a first slope |
| D5 | **Enter one external closed-loop benchmark.** The NVIDIA AlpaSim E2E Closed Loop Challenge 2026 has a track on PhysicalAI-AV NuRec (the programme's own data and harness); the closing date (reported 2026-10-31, UNVERIFIED) and front-camera-only eligibility (the default configuration is 4 cameras) must both be confirmed this week | no recognised external number exists | ≈ 1 eng-day to confirm eligibility |

---

## 2. Scorecard

Grades compare against the independent review of 2026-07-25 (`Reviews/2026-07-25-independent-chief-scientist-review/00_CHIEF_SCIENTIST_REVIEW.md` §2).

| dimension | 07-25 | now | why |
|---|:--:|:--:|---|
| Measurement infrastructure (instruments, estimator, gate) | A− | A− | four-family, distance-keeping and corridor instruments now exist and pass their own controls |
| Measurement in use (what decisions actually rest on) | C+ | **C** | the four-family instrument has zero callers in the standard runner; tactical/strategic never measured on the canonical val; the leaderboard still mixes WM fidelity with planners |
| Core-thesis validation (hierarchy, imagination, sub-300 M structure) | D+ | **D** | graded as implemented, not as a refuted idea (the strategic level is untestable on this corpus); seams corrected 1/3 → 0/3; the hierarchy's own decisions lose to a flat planner 4.0–7.2×; cadence never executed; v5f's imagination tokens have no candidate axis |
| World model as a dynamics model | — | B+ | latent integration of given controls matches a bicycle model; K = 20 fine-tune fixes the speed bias |
| Architecture and trainer code | C | C | 22 M dead parameters in the v4/v5f line; no `torch.compile`, fused optimiser or flash attention in 8 trainers; seeds only via `torch.manual_seed` |
| Data strategy | — | **D+** | 0.78 % of the corpus, 1 of 7 camera streams, no augmentation, the IDM route refuted as built, the corpus contrast confounded |
| Embedded deployment | — | A− | 60.3 ms p50 on Thor for v1's 9-manoeuvre selector path (whose accuracy is not yet measured); NuRec rendering on Thor |
| Workflow, automation, documentation | B− | **C+** | no hooks, CI or pre-commit enforce any rule; no LLM spend ledger; `LOOP_STATE.md` grew 122 → 203 KB; 52 days of silence |

---

## 3. What happened after the 2026-07-25 review

The previous review ranked 16 proposals. Follow-through, checked in the repository today (MEASURED by file presence and grep):

| proposal | status | evidence |
|---|---|---|
| Gate co-primary on corridor departure at a registered horizon | done | `stack/scripts/run_gate.py:65` |
| Closed-loop CIs through the episode-cluster bootstrap | done | `taniteval/taniteval/closedloop.py:782` |
| `safe_commit.py`, `registry_lint.py`, `repo_janitor.py` | done | `tools/`, with tests |
| `results_ledger.py`, `ckpt_relay.py`, `LOOP_STATE` split, fan-out governor | not done | no files; `LOOP_STATE.md` grew instead |
| **Hierarchy vs flat planner-over-WM, pre-registered** | **not done** | still "PROVE — build the decisive test" (`PROGRAM_OVERVIEW.md:81`); its precondition (a live route input) is the target of `PREREG_lan_refc.md`, whose falsifier fired on 2026-08-03 |
| **Data-efficiency slope** | **not done** | the phrase appears only inside the 07-25 review files |
| **One recognised external benchmark number** | **not done** | AlpaSim-first was adopted; no external number exists |
| Give the reactive renderer an owner | largely done | NuRec scenes render on Thor; closed-loop videos of v1 and REF-C were rendered there; where the driver's inference ran is not recorded |

The pattern the July review named has repeated. The apparatus improved on every axis, while the three experiments that decide the thesis were not run. Then the repository went quiet: the last commit is 2026-08-04 (MEASURED, `git log`). 149 commits fall on 9 calendar days, 49 of them on 2026-08-03 alone (MEASURED).

---

## 4. The world model and its hierarchy

*Stream R1 (Opus), `streams/R1_architecture_causal.md`. The load-bearing code facts in this section were re-verified by the orchestrator.*

### 4.1 What each headline number measures

| path | what it receives at inference | brains on the path | ADE@2s | what it measures |
|---|---|---|---|---|
| flagship v1 `flagship-30k` | 8 encoded frames, past controls, v0, **the expert's future controls for all 20 steps** | encoder → operative predictor (intent-free) → step readout | **0.4271** [0.3675, 0.4871] | how well the latent integrates known controls; nothing is chosen |
| kinematic bicycle, same expert controls | v0 + expert future controls | none | 0.4518 [0.3097, 0.6174] | the information content of the controls themselves |
| REF-C-base (104 M) | the same frames + v0; no future information | ResNet trunk → anchored decoder → argmax anchor | **0.4728** [0.3835, 0.5699] | a planner: it must predict |
| v1.6 (REF-C's decoder on v1's latent) | frames + v0 | encoder → REF-C-style decoder | 0.4375 [0.3423, 0.5501] | the one flagship-line planner that ties REF-C, at a 144 % cost to the world model |
| v1 hierarchy plan → pure pursuit → bicycle | frames, constant `follow` command | strategic → tactical | 1.9028 [1.732, 2.0745] | the hierarchy driving its own plan |
| v1 tactical head, direct | frames, constant `follow`, **no v0** | strategic → tactical | 3.3839 [2.8336, 3.9722] | the hierarchy's own decision |
| CV · CTRV · best-of-3 kinematic · ego-status MLP | v0 (+ yaw rate) | none | 0.8377 · 0.523 · 0.5005 · 0.5735 | trivial forecasters |

Sources: MODEL_REGISTRY §6 and §1.4b; `taniteval/results/driving_*.json`; `…/2026-07-26-closedloop-artifact-rerun/closedloop_flagship-30k.CORRECTED.json`. All MEASURED. The bicycle comparison is unpaired, and the paired delta has not been computed.

[[CHART]]

**Reading.** To first order, the whole leaderboard is ordered by how much speed and control information each scored path is handed:
- expert future controls + v0: 0.43, the same as the bicycle given them (0.45);
- v0 + vision: REF-C 0.47;
- v0 + yaw rate: CTRV 0.52;
- v0 alone: CV 0.84;
- no metric speed at all: no-speed control 3.02 and v1 tactical head 3.38.

Every number is MEASURED. The causal reading is an analysis (HYPOTHESIS-grade), and no number in the table contradicts it.

**The fair counterpoint, and the measurement that decides it.** The like-for-like test of the world model as a *forecaster* is to roll it under the held last action, with no future information, on the canonical 881 windows, paired against REF-C-base. It has not been run. The only such number found scores 0.3351 against hold-speed's 0.5933 (INHERITED; R1 §8). It comes from the 599-window blind-imagination subset (windows with ≥ 18.5 s of future), so it is not comparable to the 881. It is still the strongest hint that the world model predicts well on its own. It costs about 0.1 A40-day (R1 R-2), and it belongs in the 05.10 dossier.

**Collateral.**
- **The "strongest scientific claim" is not identified.** The vision-anticipation panel (`generalization.py:371-399`) also keeps the true future actions and only replaces the latent. It shows that the metric decode needs the scene latent. It does not show that the model anticipates a manoeuvre. That is not a refutation, but the claim cannot be credited to vision yet. The fix costs about one A40-hour: re-run it with zero-order-hold actions.
- **Some "hierarchy-supporting" evidence should be withdrawn.** The HPP0 audit's "grounded rollout beats the direct head by 7.9×" (`HPP0_CONFOUND_AUDIT.md:457`) compares a path handed the expert's controls and v0 with one that has neither.

### 4.2 The hierarchy as built

```
INTENDED                                            REAL (v1, as trained and scored)
frames → ViT → z[8] ─┬─ STRATEGIC(z, nav) /20 ticks   train:  STRATEGIC(z, nav = echo of its own route label)
                     ├─ TACTICAL(z, ctx)  /5 ticks            → TACTICAL → intent → OPERATIVE (JEPA k=1,2,4 only)
                     └─ OPERATIVE(z, a, intent) /tick          grounding: z + EXPERT controls → OPERATIVE(intent=None) ×20
                                                       score:  z + EXPERT future controls + v0 → OPERATIVE ×20 → metres
                                                               [strategic, tactical, intent, cadence: all absent]
```

Three code facts (CODE, R1 §2.1):
- **Cadence does not exist.** `TacticalPolicyConfig.cadence = 5` and `StrategicPolicyConfig.cadence = 20` are read by no code. Training runs every brain every step, and the closed-loop harness re-plans every tick (`closedloop.py:265-271`).
- **One window for all levels.** Every brain reads the same 8-frame, 0.8 s window. The three grounding levels decode the same intent-free rollout at 0.4, 1.6 and 2.0 s: levels of horizon, not of abstraction.
- **Train/eval horizon mismatch.** The evaluation decodes a 20-step rollout with the readout trained on 4 steps (`train_flagship4b.py:794`).

**Seams: 0 of 3 load-bearing.**
- **Nav → strategic: an echo.** Route accuracy is 1.0 with the command and skill-vs-chance is 0.0 without it.
- **Strategic → tactical: not load-bearing.** The manoeuvre-accuracy gain is +0.0148 on the full set, below the 0.02 floor; the July "+0.0439" was the deprecated split-mean.
- **Tactical intent → operative: harmful, then removed.** Its cosine vs no intent is −0.238, and it is excluded from every scored path.

The 2026-08-02 deep review independently found 0/3 for v1 and for v2corpus (`V5_FLAGSHIP_DEEP_REVIEW.md` §1 P2).

**Parameters.** The scored path uses about 180.6 M of 276.9 M (model + grounding heads). About 96 M sit only on training-time or unscored paths: 85 M of the model (tactical predictor 26.5 M, tactical policy 22.7 M, H15 22.1 M, strategic 8.4 M, inverse dynamics 5.2 M) and 11.3 M of the 13.4 M grounding heads. MEASURED from `taniteval/results/eff_flagship-30k.json`.

**A cheap fix that is already measured.** The headline decodes all 20 steps with the `op` readout, which was trained on 4 steps. The programme's own C8 readout rule (`op` up to 0.5 s, `str` beyond) measured −0.3414 m at 2 s on the deployable regime (INHERITED, `readout_selection.py`). No production caller applies it yet (0.05 A40-day).

**The strategic brain has nothing to learn from on this corpus.** PhysicalAI-AV carries no map, route, lane graph or traffic-light signal, and its coordinates are clip-local. The strategic brain sees the same 0.8 s of pixels as the operative, plus a constant. No architecture change can make it load-bearing on inputs that carry no strategic information. The mission plan also defines the strategic layer differently: as the engage/disengage/diagnosis/degradation authority, plus navigation. That supervisory role *is* supervisable on this data (§12, Bet B).

### 4.3 The decision defect

| arm | how it decides | symptom | cause |
|---|---|---|---|
| v1 tactical head | one transformer → 5-way manoeuvre softmax + 4 unimodal L2 waypoints | 3.3839 direct; 1.9028 tracked | no v0; unimodal L2 regresses to the conditional mean; waypoint loss ÷100; the softmax priority-collapses two orthogonal axes |
| REF-C base/XL | 128/256 FPS anchors, argmax of the unrefined confidence | oracle-in-fan 0.1640 vs pick 0.4714 (XL); pick > 2× oracle on 45.4 % of windows; never emits `accelerate` | confidence trained by CE against the nearest anchor to one realised future; v0 dropped on 50 % of training samples; no acceleration input |
| v4/v5 head | REF-C decoder on the WM's 16 cells × 8 frames, refined logits | v4.1 gate FAIL 0.8522 | hard-argmin CE against the hindsight-best candidate (`flagship_v15.py:861-866`) |
| v5f "conditional imagination" | the same head + 32 imagination tokens | trainer-log `plan_ade` 1.03 vs `oracle_ade` 0.53 (not quotable) | **the imagination tokens are identical for all 256 candidates** (`IMAGINATION_HAS_CANDIDATE_AXIS = False`, `flagship_v15.py:638-656`), so they add context but cannot rank. The v5 restart adopted this lever six days after the code recorded that fact |

**Oracle − pick measures fan dispersion, not selector quality.** It grows with fan size by construction: REF-C-XL's oracle is 12.0 m at K = 1, 0.81 m at K = 32, 0.26 m at K = 128 and 0.16 m at K = 256 (MEASURED, `scaleab_refc-base-30k_vs_refc-xl-30k.json`). The registry's "v5f would be ~2× better if it chose correctly" overstates the recoverable share.

**Root causes** (R1 §3.2):
- **(a) No cost model anywhere.** The world model "obediently simulates" a 181 km/h candidate and rates it self-consistent (E-V5-1). Consistency is not plausibility.
- **(b) The wrong decision-theoretic target.** Selectors are trained on P(hindsight-best) instead of argmin E[cost].
- **(c) Starved of the key longitudinal feature.** No arm receives `ax_fd`, the 0.1 s speed difference.
- **(d) Candidate consequences never reach the ranking.**
- **(e) The manoeuvre label destroys information.** It overwrites a live longitudinal manoeuvre as a turn on 9.68 % of windows (INHERITED, `…/incoming/2026-08-03-dtac1-tactical-head/`).

**Already settled, so v6 must not re-buy it:**
- Discriminative re-scoring of the v4 fan is ceilinged at 0.4907, even in-sample.
- Imagination-consistency scoring is refuted (0.5645).
- Conditioning the fan on v0 is refuted: the fan already contains speed-matched candidates.
- REF-C v1.2's 47 learned re-scorers recovered at most 8.4 % of the gap.
- The expected-cost regressor with `ax_fd` (E-GOAL-4) recovered roughly a third of the gap with no goal: 32–36 %, depending on whether the gap is measured to the oracle-in-fan or to the `R_goal2s` floor.
- The registry's standing claim that the oracle gap is about 92 % irreducible (`MODEL_REGISTRY.md:1355-1357`) is contradicted by E-GOAL-4 and must be revisited.

The last two disagree by roughly 4× and must be reconciled before v6 is funded. The leading hypotheses are the `ax_fd` feature, v1.2's top-8 restriction, and the deployment (40 vs 600 episodes).

**Ranked fixes** (R1 §3.4):
1. A per-candidate expected-cost selector (regress each candidate's cost, pick the argmin). Feed it candidate kinematics, v0, `ax_fd` and scene tokens, and train it on the parity train corpus rather than val folds.
2. Hydra-MDP-style multi-target distillation into that selector. Collision, TTC, progress and comfort sub-scores are computed offline from `obstacle.offline` tracks (97.44 % coverage). There are no drivable-area terms, because there is no map.
3. Factorised lateral × longitudinal heads as features.
4. Predicted goal-point conditioning as an inductive bias (Δ4). Do not fund a goal *supplier* yet. The only goal tested so far (E-GOAL-4) was itself a fitted function of `(v, ax_fd)` (`EGOAL_4.md:50`, audit G-6), so its gain (0.3917 → 0.3040, 600-episode, out-of-fold within val) is a re-parameterisation of ego kinematics, not new information. The oracle route command also failed at the fusion (§7). A vision-predicted geometric goal point is untested here; Δ4 is where it earns or loses its place.
5. World-model rollout scoring, only after a cost exists.

### 4.4 The longitudinal deficit

| fact | number | class |
|---|---|---|
| v1's separated win over CV is entirely lateral | cross-track +0.7720 [+0.4166, +1.1914]; along-track +0.2543 [−0.0278, +0.5304] not separated | MEASURED `driving_flagship-30k.json` |
| even given the true controls, along-track error barely beats CV | mean \|along\| at 2 s: WM 0.826 · bicycle 0.829 · CV 0.915 m | MEASURED `latlon_decomposition.json` |
| v1 is 2.0× worse than holding current speed on the 639 steady windows (72 %) | speed MAE 0.4231 vs 0.2109 m/s | MEASURED; every arm with a dump is separated-worse there |
| the logged longitudinal channel is a poor measurement | native `ax` ↔ pose dv/dt r = 0.434; ↔ `ax_fd` 0.759 | INHERITED |
| 87 % of v1's WM-fidelity squared error is longitudinal | 0.8733 | MEASURED `…/2026-07-26-closedloop-artifact-rerun/latlon_decomposition.json` |
| the flagship's WM-fidelity path runs fast vs the human | +0.1911 m/s [+0.0922, +0.2846]; REF-C indistinguishable from the human | MEASURED `…/2026-08-04-distance-keeping-arms/DISTANCE_KEEPING_ARMS.md:160,370` |
| a perfect lead-vehicle state buys little at 2 s on this corpus | 41.65 % of windows have no vehicle within 50 m; +2.3 recovery points, not separated | INHERITED (E-GOAL-1) |

**Four stacked causes:**
1. **Curvature is handed in; speed change is not.** Lateral wins because future curvature is supplied as steer on the headline surface. Longitudinal change depends on things invisible at this resolution or absent from the data.
2. **The best longitudinal observable is withheld.** Speed is broadcast as a constant over the window, and `ax_fd` is fed to no arm. This is a one-column, parity-neutral input fix.
3. **There is no "keep speed" default.** Both families emit absolute trajectories, so holding speed, which is correct on 72 % of windows, has to be reproduced rather than being the zero-residual default. A residual-over-kinematic-prior parameterisation is the cheap test.
4. **The world model is trained at K = 4 and scored at K = 20.** The K = 20 fine-tune (RR-20) cut WM-fidelity ADE 0.4244 → 0.3485 (paired, separated; MEASURED `…/2026-08-02-rollout-recovery-verdict/rr20.json`, `rrctl.json`, `ab_rrctl_vs_rr20.json`, 881/40). It removed the speed bias on both instruments that measure it:
   - driving harness +0.1879 → −0.0018 m/s (`rr20.json`);
   - four-family instrument +0.9397 → −0.0092 m/s (`fourfam_rr20.json`).

   The two instruments derive speed differently and disagree 5× on the pre-fix bias, which must be reconciled before either is quoted alone. The costs: curvature MAE 2.2× worse, miss@2m 0.043 → 0.056, and a loss on sharp-curvature windows (win-rate 0.43, n = 122). RR-20 is not in `MODEL_REGISTRY.md`, and its pre-registered primary `CR_k` was never computed (§6).

The 2026-08-04 analysis concluded that the longitudinal gap is a speed-setting bias rather than a headway failure. That holds, but it was measured on the WM-fidelity path, so the bias belongs to the world model's integration and not to a planning decision.

**What the camera can see** (ESTIMATED, pinhole geometry; R1 §4.3, R3 §2):
- **Ego speed, partly.** A linear probe on the frozen latent gets R² 0.772, with 17 % shrinkage toward the mean. That leaves errors of order metres per second.
- **Closing rate on a lead car, essentially no.** A car 40 m ahead is about 12 px wide, and closing at 5 m/s changes its width by about 0.3 px across the 200 ms frame stack.

Closing rate therefore cannot come from this input. It needs a higher-resolution or tele view, or explicit tracks at label time.

### 4.5 The latent state, resolution and temporal context

| | v1 (256×256 pinhole) | v5f (120° cylindrical, 176×624 valid) | REF-C decoder input |
|---|---|---|---|
| angular sampling | ≈ 4.6–5.0 px/deg, 51.4° HFOV | 5.333 px/deg | same frames as v1 |
| what the planner sees per frame | **4×4 cells × 128 = 2,048 floats**; each cell averages 16 tokens = 64×64 px ≈ 13.8° × 13.8° | **the same 2,048 floats**; each cell ≈ 48×160 px, **2.2× coarser horizontally** than v1 | **45,056 (base) / 63,488 (XL) floats**, 22–31× more than one WM state |
| effective dimensionality | covariance effective rank ≈ **30 of 2,048**, mostly ego-motion | not measured | — |

CODE + INHERITED (R1 §5.1). At 25 m/s a 2–4 s headway puts the lead car 50–100 m ahead, where it is 5–10 px wide: under half a patch, averaged into a 64×64 px cell with 15 other tokens (ESTIMATED).

**Verdict.** The 4×4×128 readout suits a world model whose job is ego-motion, and for that job it is adequate. It is the wrong substrate for a decision layer, and it makes a cost model over agents impossible in principle:
- **Richer map, better proposals.** On the *same* decoder algorithm, REF-C on its 8×8×F map proposes about 2× better than v1.5 on the WM state (oracle-in-fan 0.164 vs 0.338; INHERITED, registry §4.1).
- **v5f bought periphery at the cost of detail.** Because the state stayed 2,048-d, its wider field of view is spent at coarser resolution per cell.
- **No memory beyond 0.8 s.** No arm remembers anything older, which is too short for closing-rate estimation at range and, by construction, for anything strategic.
- **The resolution null is narrower than it sounds.** "A 1.5× resolution change is a null" was measured on two *global* probes (scene class, ego speed). It says nothing about object-level observability.

### 4.6 Representation, and why the later lines failed

- **REF-A (frozen DINOv2 / I-JEPA) fails at metric ego-motion decoding, not at scene understanding.**
  - It scores 2.1675 on the WM-fidelity surface, while *given* the expert's controls; a bicycle given the same controls scores 0.4518.
  - Its error is 94.2 % longitudinal, with a 4.5× train-to-held-out gap.
  - It encodes only the latest RGB frame of each stack (CODE `stack/scripts/dino_precompute.py:44`), while the flagship encodes the 9-channel three-frame stack, which makes optical flow linearly available.
  - H4 is therefore established for "frozen single-frame semantic features plus a temporal adapter as a metric-odometry substrate", not for "pretrained encoders are worse for driving". PUBLISHED counter-evidence exists (DINO-WM, arXiv 2411.04983; not V1-checked). The untested case that matters is a frozen or fine-tuned **video** encoder (BACKLOG B5).
- **v2 and v3enc were architecture-caused, by lever design.** The "anti-shortcut" pack attacked every metric-speed channel at once:
  - v0 zero-filled on 25 % of samples, where 0 m/s is an in-distribution "stationary";
  - an encoder penalised for encoding speed;
  - the metric inverse-dynamics gradient scaled down.

  Encoder speed R² fell 0.861 → 0.30 (INHERITED). R1 calls this the programme's clearest self-inflicted architectural wound.
- **v4.x and v5 degraded the world model whenever a planner's gradient entered the trunk.**
  - v1.6: the canary went 0.452 → 1.1022.
  - v4 from scratch: 1.1409 at 30 k.

  The common factor is architectural: the planner and the world model compete for one ~30-effective-dimension, 16-cell state. Their gradients are orthogonal on average (cos +0.0043), the signature of two objectives that want different representations. v6 separates them (Δ6).
- **v5f is not what its description says.** The production trainer builds v1's whole `WorldModel`, so all three v1 brains train as auxiliaries. On top of that sit one REF-C-class operative planner and a `GoalScalarHead` that feeds nothing, plus the 22 M untrained H15 field (CODE `train_flagship_v4.py:99-103,1282-1354`). It is not a three-planner hierarchy. The goal MLP feeds nothing during training and is used only at evaluation, which is a train/serve skew (R1). Every published v4 MODE-B number also used future-derived ("oracle") goal tokens by default (CODE `eval_flagship_v4.py:91-107`).

### 4.7 Imagination

Three different mechanisms share the word:
- **(i) H15:** a 22 M sector-masked belief field trained with a one-step NLL, on no inference path.
- **(ii) Blind predictor rollouts:** every "grounded rollout", including the headline, is already one.
- **(iii) v5f's probe tokens,** which have no candidate axis.

The anti-calibration finding (confidence rises as fidelity decays) is expected by construction. The σ-head was only ever trained one step ahead on encodings of real frames, and nothing regularises rolled-out states (R2: inter-sample belief cosine 0.219 → 0.805 by k = 8). The H15 target branch is also not stop-gradient (CODE `train_flagship4b.py:229`).

**What imagination has measurably bought:**
- Re-planning every tick on the imagined latent beats committing to one plan: −0.1711 [−0.2615, −0.0838] closed loop (MEASURED).
- One reference roll of a good world model, used as a selection reference, gave −0.2918 [−0.4233, −0.1598]. It was *harmful* (+0.2090) when v4's own world model scored its own fan (INHERITED).
- Probe tokens used as shared decoder context gave −0.1355 (INHERITED, legacy estimator).
- Per-candidate consistency scoring is refuted.

**The imagination role with measured value is the cheapest one:** a single reference roll as a selector feature. Imagination as a consequence model for choosing needs three things it lacks: a cost, horizon-aware uncertainty, and candidate-conditioned rolls restricted to a small top-K.

### 4.8 Deployability

- **Current tick.** The measured Thor tick with a batched 9-candidate fan through a bf16 dynamic TensorRT engine is 60.3 / 63.1 ms p50/p95 (MEASURED, runbook §3). On an A40, all levers take the v1 "plan tick" from 97.3 ms to 18.75 ms (MEASURED `eff_levers_flagship-30k.json`).
- **The timed tick is v1's `TacticalSelector`.** It runs a 9-manoeuvre (3 steer × 3 accel) imagine-and-select over the world model with a heuristic displacement-plus-comfort score (`fourbrain.py:566-571`, runbook `:288`). It is a decision, but on a path that has never been scored open- or closed-loop (`eval_behavior.py:23-25`; no results JSON), so the 60.3 ms and the 0.4271 belong to different paths. A world-model-scored decision over a REF-C-class 256-candidate fan would be about 32 saturated Thor batches × ~70 ms ≈ 2.2 s, roughly 22× over budget (ESTIMATED, R1 §8).
- **The sequential 20-step roll is launch-bound, not FLOP-bound.**
- **What fits 10 Hz on Thor with margin:** a REF-C-class decoder, a per-candidate scorer over 256 candidates, and one reference roll at k = 10 with a cached encoder. At most a top-8 imagined-consequence pass can be added.

### 4.9 Verdict on the model, and v6

The programme has demonstrated two things:
- **a good ego-motion world model:** a from-scratch visual-odometry encoder whose latent integrates given controls about as well as a bicycle does;
- **a good flat decision layer borrowed from DiffusionDrive,** whose remaining weakness is longitudinal selection.

It has not demonstrated, on any number, that the hierarchy or the world model improves decisions. The two small exceptions point at the right role for the world model: shared context, and one reference roll. **The thesis is not refuted. It is unpaid-for, and on PhysicalAI-AV its strategic level cannot be paid for, because the corpus contains no strategic information.**

```
frames ─► ViT (from scratch) ─┬─► z[8] ─► operative predictor = THE WORLD MODEL (JEPA + grounding + SIGReg, K = 20)
                              │             └─► ONE reference roll, held last action ─► r            (feature)
                              └─► last-frame PATCH TOKENS ───────────────┐
ego: v0, ax_fd, 8-step speed history, yaw rate (learned null row) ───────┤
                                                                         ▼
          anchored decoder (REF-C algorithm; anchors = residuals over a CTRA prior) ─► 256 candidates
          TACTICAL  = factorised lateral(3) × longitudinal(3) posterior, kinematic labels ─┐
          STRATEGIC = predicted goal point from vision + ego (never from the situation classifier) ─┤
                                                                         ▼
          per-candidate EXPECTED-COST head: along/cross error, collision, TTC, progress, jerk
                                                                         ▼
          argmin cost ─► 20-step plan ─► safety envelope (§10.6) ─► controls
removed: v1 tactical/strategic policies, tactical predictor, H15, probe tokens, nav-echo CE, hard-argmin CE
```

**Inference inputs, stated per the 2026-08-03 rulings:**
- The tactical posterior reads patch tokens and z only. It is vision-only at inference, and its kinematic labels are used at label time only.
- The goal head reads patch tokens, z and ego kinematics, and never the tactical posterior or any situation-classifier output.
- Both share the encoder trunk. That is admissible because neither head's output is an input to the other; Δ4's R² test against the flat inputs checks that the shared trunk has not re-created the leak.

The v6 ladder (R1 §9.2) is nine discriminating experiments with both outcomes pre-stated. Together they cost about **5.8 A40-days**, mostly in parallel, and Δ2/Δ3/Δ4/Δ8 run on frozen fan dumps:

| Δ | change | fixes | adopt if … / otherwise … | A40-days |
|---|---|---|---|---:|
| Δ1 | feed `ax_fd` and the true 8-step speed history; learned null row instead of zero-fill | withheld longitudinal observable | ≥ 20 % separated reduction in steady-window speed error → adopt in the WM; null → keep `ax_fd` in the selector only | 1.0 |
| Δ2 | per-candidate expected-cost regression replaces the hard-argmin CE | wrong decision target | separated gain on both deployments when trained on *train* episodes → adopt; gain only out-of-fold within val → E-GOAL-4 was an artefact | 0.5 |
| Δ3 | Hydra-style cost targets (collision, TTC, progress, comfort) from `obstacle.offline` | no cost model | TTC violations separated-lower with ADE non-inferior (Δ < +0.02 m) → adopt; null → keep for closed loop only | 0.3 |
| Δ4 | honest hierarchy: factorised tactical posterior + predicted goal, both into the selector; cadence implemented; nav-echo CE deleted | seams 0/3 | hierarchical separated-better with R² < 0.99 against the flat inputs → the level carries information; ≥ 0.99 → inductive bias only; null → tactical level falsified on this corpus | 0.3 |
| Δ5 | remove about 80 M of off-path auxiliaries and H15 | unpaid parameters | no separated harm → remove, and spend the budget and step time on Δ6 | 1.0 |
| Δ6 | the decoder also cross-attends last-frame patch tokens; the WM keeps its compact state | readout bottleneck, planner/WM competition | oracle-in-fan and pick both separated-better → adopt; null → the WM state is a sufficient substrate | 0.7 |
| Δ7 | train rollout and readout at K = 20 (curriculum 4 → 20) with the curvature term active | compounding speed bias | bias fix kept with curvature cost < 1.5× → K = 20 from the start; otherwise a late fine-tune | 0.9 |
| Δ8 | one reference roll `r` as a selector feature, with the arm's **own** WM | WM not on the decision path | separated gain with the arm's own WM → the WM earns its place in the loop; otherwise it stays a representation learner | 0.1 |
| Δ9 | anchors as residuals over a CTRA prior | no "keep speed" default | first arm ever to beat hold-v0 on steady windows → adopt | 1.0 |

**Falsifying the hierarchy, level by level:**
- **The tactical level is falsifiable now.** It fails if the hierarchical selector (Δ4) is not separated-better than the flat one at matched capacity, and its inputs are regressable from the flat inputs at R² ≥ 0.99.
- **The strategic level is not testable on PhysicalAI-AV.** It needs a corpus with a route or goal source (AlpaSim scenes, L2D, or an external mapped corpus).
- **The world model as a decision component is falsifiable now (Δ8).**
- **The whole thesis** is falsified at this scale if a flat REF-C-class planner with the same inputs and budget matches v6 on all four families and in closed loop. The one existing hierarchical-vs-flat pair (REF-B v2 0.5913 vs REF-C-base 0.4728, separated) leans this way. It is confounded by decoder family and label version, so it is a prior, not evidence.

---

## 5. Why the models have the results they have

| arm | result | first-order cause | evidence |
|---|---|---|---|
| flagship v1, headline | 0.4271 | a world model given the expert's future controls and v0; integrates them as well as a bicycle | CODE `rollout.py:146-147`; MEASURED bicycle 0.4518 |
| flagship v1, tactical head | 3.3839 | speed-blind by construction (no v0 port); unimodal L2 regression; loss weight ÷100; priority-collapsed manoeuvre label | CODE `fourbrain.py:328-361`; MEASURED |
| flagship no-speed control | 3.0175 | no metric-speed channel; the latent's speed content (R² ≈ 0.77) leaves m/s errors that compound to metres | MEASURED; ESTIMATED arithmetic |
| REF-C base/XL | 0.4728 / 0.4714 | a multimodal anchored planner on a rich 8×8×F map; weakness is selection (pick > 2× oracle on 45 % of windows) and longitudinal (v0 dropped 50 % in training, no acceleration) | MEASURED `scaleab_…json`; CODE `refc.py:1888-1891` |
| REF-C-small | 0.5261 | the same algorithm with fewer anchors; the ladder's knee is anchor count, not encoder scale | registry §4.2 verdict |
| REF-B v2 | 0.5913 | hierarchical behaviour cloning without a world model; unimodal heads | MEASURED |
| REF-A (frozen DINOv2) | 2.1675 | single-frame semantic features cannot be decoded into metric ego-motion | MEASURED; CODE `dino_precompute.py:44` |
| v2 (killed) / v3enc (RESTART) | 5.94 / 1.9654 | anti-shortcut levers removed the metric-speed channels | INHERITED registry §1.3/1.4 |
| v4.1 (gate fail) | 0.8522 | planner gradient into the shared trunk degraded the world model; hindsight-argmin selector | MEASURED; CODE `flagship_v15.py:861-866` |
| v1.6 | 0.4375 | REF-C's decoder on v1's latent: ties REF-C, but degrades the world model 144 % | INHERITED §1.4b |
| v5f (never evaluated) | trainer log only | imagination tokens cannot rank; 22 M untrained parameters; 19.58 s/step | CODE; registry §1.8 |
| every arm on steady windows | worse than holding speed | absolute-trajectory outputs with no zero-residual "keep speed" default | MEASURED registry §6 reading 4 |

---

## 6. Training recipes and non-collapse

*Stream R2, `streams/R2_training_anticollapse.md`.*

**What is right.** SIGReg is implemented faithfully to LeJEPA (Balestriero & LeCun, arXiv:2511.08544): a sliced Epps–Pulley test with 512 fresh directions per call, forced to fp32 (`stack/tanitad/models/sigreg.py`). There is no EMA teacher and no stop-gradient anywhere in the training stack, which is the LeJEPA recipe and removes a whole class of collapse heuristics. MEASURED (code read).

**What is wrong, in order of consequence.**

1. **The regulariser guards the wrong subspace.** `free_dims = 64` exempts 64 dimensions from the isotropy test, while an orthogonality report on the trained checkpoint puts the energy-carrying subspace at only about 19–30 of 2,048 dimensions. Isotropy improves during training (0.254 → 0.546), but mostly on the low-energy complement. This agrees with the flat `vision_use ≈ 12.9 %` and the route head's zero skill: the state is low-rank, and SIGReg is not what keeps it informative. INHERITED from R2's reading of the in-repo orthogonality report; re-check on the next checkpoint.
2. **Nothing regularises the rollout.** SIGReg applies to encoder states, never to recursively predicted or imagined states. The 2026-07-18 diagnostic measured the predicted failure: inter-sample belief cosine rises from 0.219 to 0.805 by k = 8 (states collapse toward an attractor) while epistemic variance shrinks (false confidence). This is the mechanism behind the "anti-calibrated imagination" finding. MEASURED (programme artifact, cited in R2).
3. **The v4/v5f line carries about 22 M parameters that never train.** Verified by the orchestrator in the source: `train_flagship_v4.py:390,395` builds `WorldModel(flagship4b_config())`, which instantiates the H15 `ImaginationField` (`fourbrain.py:456`); `train_flagship_v4.py:999` puts every world parameter in the trunk optimiser group; the file never computes an H15 loss (v1's trainer does, `train_flagship4b.py:655-658`); and `fourbrain.py` never calls the module in `forward`. In the from-scratch v5f run those weights stay at random initialisation. MEASURED (static). Cost is memory and a misleading parameter count, and the H15 capability is simply absent from the line that was meant to test imagination.
4. **Temporal collapse is ungated.** `latent_screen.py`'s pre-flight gates fail on v1's latent (frame-to-frame jitter 51.0× against a ≤ 2× gate; derivative correlation +0.089 against > 0.50). Isotropy converging and the latent being temporally smooth are different properties, and only the first is watched.
5. **The loss is dominated by supervised metric terms.** For v1's active terms the SSL core weighs about 2.6 against about 11.5 for supervised metric-motion terms (≈ 4.4 : 1 supervised). The fix the 2026-07-18 research recommended (`invdyn_gradscale = 0.25`) exists in code and ships at the no-op default 1.0.
6. **A pre-registered result has no admissible verdict.** `PREREG_rollout_recovery.md` names `CR_k` as the primary and forbids trading it for ADE; the finished run reports only ADE (0.3485 vs 0.4244 m, a large separated win) and no `CR_k`. Computing `CR_k` from the existing checkpoints costs about 0.05 GPU-days.
7. **Compute is left on the table.** None of the eight trainers uses `torch.compile`, fused AdamW or explicit flash attention. v5f's `--cond-imagination` runs a sequential 20-step frozen-predictor rollout at batch × 32, 16 times per optimiser step, the same shape that CUDA graphs already sped up 1.75–3.46× at inference. v5f runs at 19.58 s per optimiser step (MODEL_REGISTRY §1.8), about 3.3 samples/s. No per-stage profile exists, so the split is ESTIMATED from code.
8. **Reproducibility is unchanged since July.** Only `torch.manual_seed` is set in all eight trainers; data order and CUDA kernels are not seeded or made deterministic.

**Why the failed lines failed** (R2 §3): v2 changed ten levers at once and is unattributable; v3enc's RESTART verdict came from a gate window in which the decorrelation lever it was testing was held at 0.0 throughout, so the lever was never engaged and is not refuted; v4.1/v4.2 hit a real Pareto tension between keeping the world model good and training a planner on the same trunk; v1.6 is a clean statistical tie. None of the four implicates the encoder, SIGReg or the grounding core.

## 7. Data, encoding and resolution

*Stream R3, `streams/R3_data_encoding_labels.md`.*

1. **The programme trains on under 1 % of the corpus it has.** PhysicalAI-AV has 306,152 clips, about 1,700 h, 7 camera views (front-wide 120°, front-tele 30°, cross left/right 120°, rear left/right 70°, rear-tele 30°), LiDAR and radar (MEASURED, in-repo feature probe `PHYSICALAI_FEATURE_PROBE.md:70-76`; PUBLISHED, Hugging Face card). Its own split is train 153,625 / val 90,928 / test 61,599 clips. The parity corpus is 2,376 clips = 13.2 h = 0.776 %; the balanced v2 corpus is 2.94 %; v5f is 0.78 %. Every trained arm reads only `camera_front_wide`. MEASURED (code and manifests).
2. **No image augmentation of any kind exists on real-camera pixels**: no mirror flip (with steer-sign and lateral-coordinate flip, the cheapest data doubling in driving), no random crop, no colour jitter. MEASURED (grep over the data and training tree; R3 §1.4).
3. **The input cannot see what the longitudinal family needs.** Three RGB frames at 100 ms spacing, channel-stacked into 9 channels at 256 × 256, give the encoder 200 ms of motion baseline. The ego-speed scalar `v0` carries speed. That explains the no-speed control collapsing to 3.0 m, and it bounds how well closing speed to a lead vehicle can be inferred from pixels. The parity crop is 256 px over 51.4° (≈ 4.98 px/deg): a 1.8 m-wide car is ≈ 10 px at 50 m and ≈ 5 px at 100 m, less than one 16-px patch; a traffic-light face at 50 m is ≈ 1.7 px and a lane stripe at 30 m ≈ 1.2 px. ESTIMATED (R3 §2, simple trigonometry from `calib.py:38` and `situations.py:67`). A distant lead vehicle is barely present in the input, and traffic-light state is not legible at all.
4. **Labels.** Steer = atan(2.9·κ) against true wheelbases of 2.73–3.22 m (+8.6 % cross-track error, MODEL_REGISTRY §0.1.1). Situation labels are derived from ego dynamics, which makes any ego-fed classifier partly read its own label source (the PI's 2026-08-03 ruling). The intersection label's cross-traffic half (`sc_cross.py`, needs `obstacle.offline`) is stranded in `incoming/` and never promoted, so `situations.py` computes only the turn half.
5. **The distance-keeping instrument works; the standard harness does not call it.** The 2026-08-04 val-40 panel (`…/incoming/2026-08-04-distance-keeping-arms/DISTANCE_KEEPING_ARMS.md`) computed lead-vehicle metrics for 25 arms (LEAD 270 / NO_LEAD 551 / NO_LABEL 60 windows), but it was built by scripts in that incoming folder. R3 is right that `taniteval`'s runner still reports distance-keeping as unavailable.
6. **The v2 corpus comparison is doubly confounded.** 21 of 40 canonical val episodes are in v2corpus's training selection (the headline now uses the 19 leak-free ones), and the launch also changed the whole `--v2` lever pack and `rollout_k` (12 vs 4). "More data helped" is not yet admissible. The lever-matched control is designed and was never launched.
7. **The IDM pseudo-labelling thesis is refuted as built.** `dynenc-branchB` failed its pre-registered held-out-rig gate (rig-B speed R² −0.667 against a > 0.9 gate) and underperformed v1's own encoder in-domain. The YouTube-IDM route needs a different design (below).
8. **The goal fix has an adverse prior.** REF-C given the oracle route did not separate on ADE and got worse on cross-track, losing about 91 % of the oracle's benefit downstream. The bottleneck is the fusion architecture, not the label, so a better goal label alone will not rescue a decoder that cannot use it.

## 8. Evaluation and statistics

*Stream R4, `streams/R4_eval_metrics_closedloop.md`.*

1. **The four-family rule is binding on paper only.** `taniteval/four_families.py` implements all four families as specified, but `runner.py`, `bench.py` and `driving.py` never call it, and none of the 103 files in `taniteval/results/` carries its `_binding_rule` sentinel. Every four-family number lives in an unmerged `incoming/` one-off. MEASURED.
2. **Tactical and strategic have never been measured on the canonical val for any arm.** The only populated runs use a 19-episode subset. The most complete canonical panel (2026-08-04, 881 windows / 40 episodes) reports both families UNAVAILABLE for flagship-30k and refc-base-30k. `hierarchy.run`, the only producer of those families, accepts flagship-family arms only (`runner.py:341-342`), so REF-B and REF-C can never get a tactical or strategic number as the code stands.
3. **"Instrument built, never wired" is systemic.** The same pattern appears in `idm_families.py`, the late-fusion fix in `sitclf_deploy.py`, the control suite in `control.py`, and `lateral.py`: each is tested and has zero callers outside its tests.
4. **Power.** At 40 episodes the minimum detectable paired effect on ADE@2s is roughly 0.06–0.16 m, depending on how correlated the two arms are (Δ 0.044 is not separated; Δ 0.12–0.16 is). The 600-episode deployment shrinks intervals 2.8–3.9× (close to the √15 theory) but is an easier corpus (CV floor 0.838 → 0.692) and cannot be mixed with the 40-episode leaderboard.
5. **Closed loop.** `closedloop.py` lets the world model simulate itself; AlpaSim NuRec is reconstruction-OOD-confounded (REF-C's ADE on reconstructions is 3.21× its real-footage value); `pseudosim.py` is the only non-extrapolating instrument and is explicitly not a driving score (no collision gate, EPDMS `filter_m` contract recorded but not applied). There is also a citation trap: two AlpaSim result files both called "REF-C suite" disagree (unpaired solo run: base 6/12; paired rerun `flagship_vs_refc_suite_results.json`: base 8/12 vs flagship 2/12, p = 0.0078). The registry cites the right one.
6. **A matching external benchmark exists and has a deadline.** NVIDIA's AlpaSim E2E Closed Loop Challenge 2026 has a track on PhysicalAI-AV NuRec, the data and harness TanitAD already uses. The challenge and its PhysicalAI-AV NuRec track are confirmed by an independent search (streams/V1_citation_check.md); the 2026-10-31 closing date and front-camera-only eligibility are UNVERIFIED (the default configuration is 4 cameras). Confirm both this week.
7. **The gate protocol is sound** (it refuses train-log slopes, bare exponents and the deprecated estimator), but the 2026-07-26 horizon fix has not reached the four-family lateral metric, which is still computed at 2.0 s only.
8. **Stale registry statement.** MODEL_REGISTRY §0.3 still says TanitEval is uncommitted; `taniteval/` is 263 tracked files.

## 9. The agentic harness

*Stream R5, `streams/R5_agentic_harness.md`.*

1. **No rule is enforced by a machine.** `.claude/settings.json` and `settings.local.json` define no hooks; there is no pre-commit configuration and no CI workflow. Claude Code's `Stop` hook can block a turn from ending and `PreToolUse` can deny or redirect a specific command. Those are exact mechanical matches for "never end a turn having only reported" (flagged four times) and "`safe_commit.py` is the only sanctioned commit path". MEASURED + PUBLISHED.
2. **The resource that blew the budget is the one nothing measures.** `RESOURCE_LEDGER.md` tracks GPU dollars only. There is no LLM token or API spend ledger, although CLAUDE.md records a fan-out that exhausted the weekly API budget and `LOOP_STATE.md:46` records a 215-agent, 12.9 M-token burst on 07-29.
3. **The largest error source is the orchestrator's relay layer, by the programme's own scorecard.** `BOOST_PROGRAM.md` (07-26) marks every mechanised measure as working and marks "no unverified premise in briefs" as failing "at my layer". Retraction R-2026-08-04-briefs is the same class.
4. **State documents grow instead of shrinking.** `LOOP_STATE.md` went from 122 KB to 203 KB (472 → 1,493 lines) in 9 days after two reviews called it a liability; its "read this first" pointer is wrong by about 1,300 lines.
5. **Boom, bust, silence.** 149 commits fall on 9 calendar days (49 on 2026-08-03 alone), then nothing for 52 days, with no handoff note found.
6. **The prior automation proposals were 3/10 built** (`safe_commit.py`, `registry_lint.py`, `repo_janitor.py`).
7. **No model-routing policy exists** for the production loop.
8. **"≥ 5 streams, always" should be conditional.** Anthropic's published multi-agent guidance advises against fan-out for highly interdependent work. Parallelism pays on independent questions (this review) and costs on coupled ones (one trainer, one registry).

---

## 10. Frontier research, mapped onto TanitAD

*Streams W1, W2, W3 (Sonnet; about 130 web searches) and the verification pass V1 (Haiku).*

This container's egress policy blocked arxiv.org and huggingface.co pages, so the research streams worked from search results. V1 then independently re-searched the 31 citations this section leans on (`streams/V1_citation_check.md`):
- **24 confirmed:** the id, the title and the claim match.
- **6 exist with the specific number unconfirmed.** Those numbers are marked UNVERIFIED below.
- **1 ID mismatch:** arXiv:2608.26533 is a barrier-function conformal safety-clearance paper, not the imagination-calibration work it was cited for.

Treat every PUBLISHED number here as search-verified, not full-text-verified, until someone reads the paper before spending GPU-days on it.

**Verdict scale:**
- **ADOPT:** do it now; the evidence is strong and the cost is low.
- **TRIAL:** run a pre-registered, discriminating experiment first.
- **WATCH:** track it, and don't spend on it yet.
- **AVOID:** it doesn't fit this programme, or it is hype at this scale.

### 10.1 Planning and trajectory selection (the decision defect)

The field has converged on generating many candidates and scoring them well.
- **Hydra-MDP** (arXiv:2406.06978) distils rule-based metric teachers into per-candidate scorers.
- **GTRS** (arXiv:2506.06664) won the NAVSIM v2 challenge with sensor-robust generalized scoring.
- **DriveSuprim** (arXiv:2506.06659) reaches 87.1 EPDMS with coarse-to-fine selection.
- **WoTE** (arXiv:2504.01941) scores candidates with a BEV world model, reaching PDMS 88.3.
- **ZTRS** (arXiv:2510.24108) drops imitation and learns the scorer from reward alone.

All five are confirmed by V1. They are the published form of the programme's own E-GOAL-4 result and of v6 steps Δ2/Δ3.

- **GoalFlow** (arXiv:2503.05689, PDMS 90.3, confirmed) shows goal points help. No goal tested on this corpus has added information beyond `(v, ax_fd)`, but the only one tested was constructed from `(v, ax_fd)` (§4.3). Treat the geometric goal point as an untested inductive bias, not a refuted lever.
- **SimLingo** (arXiv:2503.09594, confirmed) is a vision-only closed-loop leader with separate path and speed outputs, which is the lateral/longitudinal factorisation D-TAC1 found necessary.
- W1 flags one counter-finding (not V1-checked): re-ranking a fixed fan cannot find trajectories the fan lacks, so search can beat scoring.

**Verdicts:**
- **ADOPT** expected-cost, multi-target scoring (Δ2 + Δ3) and a path + speed-profile output factorisation.
- **TRIAL** a ZTRS-style reward-only scorer as a competing arm.
- **TRIAL** world-model scoring only after a cost exists (Δ8).

### 10.2 World models and imagination-based training

- **Dreamer 4** (arXiv:2509.24527, confirmed) trains agents entirely inside a scalable world model learned from offline data.
- **DriveVLA-W0** (arXiv:2510.12796, confirmed) finds that world modelling as dense supervision amplifies data scaling in driving models.
- **LeJEPA** (arXiv:2511.08544, confirmed) is the programme's anti-collapse basis, and R2 found the implementation faithful.

What this means here: TanitAD's world model is a good dynamics model, but it has no cost and anti-calibrated uncertainty. RL inside it today would optimise against an exploitable model. The literature's own failure mode is reward rising while imagination error at visited states climbs.

**Verdicts:**
- **ADOPT** world modelling as a trunk-shaping auxiliary loss (keep JEPA and grounding).
- **TRIAL** imagination fine-tuning of the decision layer (Dreamer-style) *after* Δ3 (a cost) and horizon-aware uncertainty exist, with imagination error monitored at visited states as the kill signal.
- **TRIAL** RL in NuRec re-simulation on Thor in Phase C.

### 10.3 Foundation models, VLAs and robotics transfer

- **Alpamayo-R1** (arXiv:2511.00088, confirmed) is NVIDIA's reasoning VLA with chain-of-causation traces. Its link to PhysicalAI-AV training data is UNVERIFIED.
- Robotics foundation lines (π0.x with RECAP-style learning from experience, GR00T N1.x, Gemini Robotics 1.5) show the same pattern: a large pretrained backbone, a small action head, and RL or advantage-conditioning from deployment experience. These were reported by W1 and not V1-checked.

A multi-billion-parameter VLA cannot be the 10 Hz planner inside a sub-300 M, Thor-resident budget. It can be a label-time **teacher**, which the PI's rules allow: privileged and knowledge signals are fine at label time.

**Verdicts:**
- **TRIAL** a VLM/VLA as an offline labeler for hazards, manoeuvre intent and rule-relevant events (BACKLOG B8 already compares Cosmos Reason1 and Reason2).
- **TRIAL** RECAP-style advantage conditioning from logged selection outcomes (log-only, cheap).
- **AVOID** a VLA as the onboard planner.

### 10.4 Data efficiency: needing far less labelled driving (Goal 2's first priority)

In driving, action labels come free from odometry. The scarce resource is *hours of paired video and egomotion*, and the multiplier is *label-free video*. The claim worth making is "closed-loop quality Q reached with N hours of paired data, where competitors need M". It needs a measured slope, which the programme has never had.

- **LFG, "Learning to Drive is a Free Gift"** (arXiv:2602.22091, CVPR 2026): label-free pretraining from front-camera video reaches 85.2 PDMS (confirmed). The "81.4 PDMS with 10 % of labels" figure is UNVERIFIED. It is the closest sensor match found.
- **DriveVLA-W0** (confirmed) finds that world modelling amplifies data scaling.
- **SE(2) and mirror symmetry** (arXiv:2403.11304, confirmed): +20.6 % L2@3s on small data. TanitAD has no mirror augmentation at all (R3).
- **Cosmos-Drive-Dreams** (arXiv:2506.09042, confirmed): synthetic driving data from world foundation models, already licence-cleared internally.
- **MOSAIC** (arXiv:2604.08366; exists, "80 % fewer examples" UNVERIFIED): scaling-aware data selection for end-to-end driving.
- **Privileged teachers** (learning by cheating, Roach, PlanT, Hydra-MDP): dense per-sample supervision from signals the model cannot see at inference.
- **Rig transfer** for inverse-dynamics pseudo-labels: Rig3R (arXiv:2506.02265, confirmed) and focal-length canonicalisation. The programme's own IDM line failed held-out-rig transfer. Audit whether that test was truly held-out before redesigning.

**The cheapest label-free source is already on disk:** the unused part of PhysicalAI-AV's *train* split, about 153 k clips / 850 h, same rigs, in-domain. Use it before YouTube. Never use the HF val/test splits or any clip whose NuRec scene is in the closed-loop suite or the challenge. Register that exclusion list before the run, or the one external number is contaminated (the REF-A I-JEPA leak class).

**Verdicts:**
- **ADOPT** mirror augmentation (with steer and lateral sign flips).
- **ADOPT** the data-efficiency slope experiment (decision D4).
- **TRIAL** the LFG weights probe, then reproduction.
- **TRIAL** `obstacle.offline` auxiliary heads and a privileged cost teacher.
- **TRIAL** a Cosmos-Drive-Dreams mixture once the slope exists.
- **TRIAL** a rig-canonicalised IDM after the protocol audit.
- W2's sequenced programme (`streams/W2_…` §"10x less labelled data") is the data track: Phase 0 about 0–5 A40-days of diagnostics, Phase 1 about 20–35, Phase 2 about 25–45 (ESTIMATED), with a gate between phases.

### 10.5 Structured physical world understanding

The clean way to inject structure without violating vision-only inference is **train-time-only structure**. GraphPilot (arXiv:2511.11266, confirmed) finds that relational supervision at training time keeps its gains with the graph removed at test time. TanitAD's structure source is `obstacle.offline`: 3D agent tracks on 97.44 % of clips, 10 dynamic classes, all unused. Two routes use it: auxiliary heads that predict agent boxes, occupancy and TTC from the latent, and the per-candidate cost targets (Δ3).

The readout bottleneck (§4.5) is the other half. Structure the planner cannot see is structure it cannot use, hence Δ6 (the planner reads patch tokens).

**Verdicts:**
- **ADOPT** train-time-only structure (auxiliary heads and cost targets).
- **WATCH** object-centric slot world models.
- **AVOID** full occupancy world models (OccWorld-class) until monocular metric depth is solved; there is one camera and no GNSS.

### 10.6 Mathematical guarantees and the safety envelope (Goal 2: "no compromise in safety")

Nothing can guarantee the learned planner itself. Neural-network verification scales to local robustness, not to ViT-scale semantic correctness. Safety must be an *envelope* property: small, verifiable layers around the learned core. W3's envelope, corrected by V1:

```
L0 learned planner (v6)  ─► candidates + one reference roll
L1 rule-tiered rerank     Safety ≻ Legal ≻ Comfort over agent/kinematic predicates  (RECTOR, arXiv:2605.25095; its violation-rate numbers UNVERIFIED)
L2 trust gate             conformally calibrated uncertainty + refusal signal        (conformal planning: Lindemann et al., RA-L 2023; DreamLedger arXiv:2608.23863)
L3 hard filter            CBF-QP on road-edge and lead-vehicle barriers + RSS check  (BarrierNet arXiv:2203.02401; RSS, Shalev-Shwartz et al. 2017)
L4 runtime-assurance switch  Simplex: low trust or infeasible QP ─► fallback       (Synergistic Simplex arXiv:2605.08190)
L5 verified fallback      small controller (e.g. Koopman-LQR or kinematic MPC), stop-in-lane
L6 post-hoc guard         neuro-symbolic check of the final command                  (arXiv:2608.11451)
```

**What is guaranteed, and under which assumptions:**
- CBF forward invariance holds only if the dynamics model and the state estimate are within their assumed error bounds.
- The Simplex switching logic is small enough to test exhaustively.
- Conformal coverage is marginal and assumes exchangeability; it must be recalibrated when the deployment distribution moves.

**The design constraint the review adds.** L3 needs lead-vehicle range and closing rate at inference, and the 256-px front crop cannot provide closing rate at range (a 40 m car changes width by about 0.3 px across the frame stack; R1 §4.3). So the envelope's perception channel must come from a high-resolution centre crop or the **front-tele camera**, which PhysicalAI-AV provides and the programme does not use, plus a monocular range head supervised by `obstacle.offline`. All of that is vision-only, so it is admissible.

**The safety case.** Use the safety-case patterns for VLA-based driving (arXiv:2603.16013, confirmed) and the 2026 standards set: ISO 26262, ISO 21448 SOTIF, UL 4600 and ISO/PAS 8800.

**Verdicts:**
- **ADOPT** RSS longitudinal check + CBF-QP filter + Simplex switch + calibrated trust gate as v6's safety layer.
- **TRIAL** rule-tiered reranking.
- **AVOID** "provably safe" language without its assumptions.

### 10.7 Symbolic and neuro-symbolic reasoning; knowledge injection

- **Formalised traffic rules** exist: Rulebooks (Censi et al., ICRA 2019), STL/LTL monitors, and CommonRoad's rule formalisation. Most rules, though, need semantics this corpus lacks: lanes, signals, speed limits, right of way. A rule layer here is limited to agent-relative and kinematic predicates (headway, TTC, comfort, speed consistency) until a mapped corpus or simulator supplies more.
- **Knowledge injection** that respects the vision-only rule happens at label time: a VLM or VLA teacher labels hazards and manoeuvres, and a rule-based teacher supplies cost targets.
- **Chain-of-thought is not a safety mechanism on its own.** W3 cites reasoning-trace attacks (not V1-checked).

**Verdicts:**
- **TRIAL** kinematic and agent rule monitors inside L1/L6.
- **TRIAL** VLM-teacher labels.
- **WATCH** program synthesis of policies (LangProp) as a route to an interpretable fallback controller.
- **AVOID** reasoning traces as the safety argument.

### 10.8 Physics-grounded neural operators and dynamics

The programme's own evidence says the world model behaves like a bicycle model when given controls (§4.1), so make the physics explicit where it helps:
- a differentiable kinematic decoder (controls → trajectory, feasible by construction; the specific paper arXiv:2603.12421 exists, its decoder claim is UNVERIFIED, and the technique is standard);
- anchors as residuals over a CTRA prior (Δ9);
- a physics-consistent regulariser on imagined rollouts (kinematic-consistency error).

The neural-operator families mostly miss this problem:
- **FNO, DeepONet and PINO** solve PDE-scale fields. A single front camera with no map and no sensor network offers no such field, so they are a scale mismatch here.
- **Hamiltonian and Lagrangian networks** encode energy conservation, the wrong invariant for a dissipative, externally actuated vehicle.
- **Koopman operators** are useful in one place: a linear-in-lifted-space controller is a good candidate for the verified fallback (L5).

**Verdicts:**
- **ADOPT** a kinematic decoder and residual-over-prior outputs.
- **TRIAL** a Koopman or kinematic-MPC fallback.
- **AVOID** FNO/DeepONet/PINO and HNN/LNN for vehicle dynamics at this scale.

### 10.9 Tool use for autonomous driving

Tool-calling LLM agents are orders of magnitude too slow for a 10 Hz loop. Tools belong in the *offline* loop:
- **Scenario generation** to expand the closed-loop suite, which at 12 AlpaSim scenes is the statistical bottleneck. Chat2Scenic (arXiv:2607.14387, confirmed) compiles 76.42 % of LLM-written Scenic scenarios.
- **Reward and cost design** (Eureka-style).
- **Labelers and data mining.**

**Verdicts:**
- **ADOPT** LLM-generated scenarios, with human review, to reach at least 50–100 closed-loop scenes.
- **TRIAL** LLM-proposed cost terms judged by the harness.
- **AVOID** tool calls in the control loop.

### 10.10 Continual and continuous training

With no fleet, TanitAD's data engine is:
1. mine hard windows from the unused train split of PhysicalAI-AV and from closed-loop failures in NuRec/AlpaSim;
2. fine-tune on them with replay;
3. re-gate on the four families.

Region and rig adaptation, which is a mission goal, fits parameter-efficient adapters per rig or region with replay against forgetting.

**Verdicts:**
- **TRIAL** a failure-mining → replay fine-tune loop in Phase B.
- **TRIAL** per-rig adapters, since the two camera rigs are a known confound.
- **WATCH** test-time adaptation on Thor.

### 10.11 Brain- and cortex-inspired approaches

The useful brain analogy is not four brains that each predict a trajectory. It is the division of labour the neuroscience actually describes, and each part has a measurable job in v6:

| brain function | v6 counterpart | measurable job |
|---|---|---|
| habitual fast policy (System 1) | the anchored decoder | proposes candidates every tick |
| basal-ganglia action selection with lateral inhibition | expected-cost argmin + duplicate suppression | picks one; its failure is the measured selection gap |
| cerebellar forward model | the operative world model (one reference roll) | predicts consequences; value measured by Δ8 |
| deliberation on demand (System 2) | top-K consequence rolls only when the trust gate is low | the cadence done right: thinking slow when needed, not on a fixed timer |
| prefrontal supervision | the strategic supervisor (Bet B) | engage/degrade decisions, calibrated against closed-loop outcomes |

Three published findings bear on this:
- **TRM** (arXiv:2510.04871, confirmed): a tiny recursive network beats the hierarchical HRM with no hierarchy at all. Any hierarchy claim must beat a flat baseline with matched compute per decision, not just matched parameters.
- **Energy-Based Transformers** (arXiv:2507.02092, confirmed) recast selection as energy minimisation.
- **Neural Circuit Policies and liquid networks** show compact interpretable controllers, relevant to the fallback.

**Verdicts:**
- **ADOPT** the dual-process control flow.
- **TRIAL** basal-ganglia-style inhibition in the selector (cheap).
- **WATCH** EBT selection.
- **AVOID** citing neural analogy as evidence.

### 10.12 AI that engineers AI; harness engineering

- **Where agents are reliable today.** Agent systems are strongest on bounded tasks of a day or less, with a clear check. That is exactly the shape of the v6 ladder: nine pre-registered, frozen-dump experiments.
- **Reward hacking is measured.** A 2026 study of autonomous research agents reports a 30.5 % hack rate on open-ended tasks (arXiv:2609.28614; that figure confirmed, a second figure UNVERIFIED). TanitAD's gate rules (void secondaries, privileged-input primaries, pre-registration) are genuine countermeasures to that class and should be kept.
- **The gap is mechanical enforcement** (§9).

**Verdicts:**
- **ADOPT** hooks, CI, a token ledger and model routing (§13).
- **TRIAL** an autonomous experiment loop on the v6 ladder, with humans at spend, publish and thesis decisions.
- **AVOID** unbounded research agents without pre-registration.

---

## 11. Catalogue of proven strategies for this programme

Ranked by evidence × fit × cost. "In-house" means TanitAD's own measurement (class as cited in §4–§8). PUBLISHED items carry the V1 status from §10.

| # | strategy | maturity | key evidence | fixes | cost to test | first experiment |
|---|---|---|---|---|---|---|
| 1 | Per-candidate **expected-cost** selection instead of hindsight-best CE | proven in-house + published | E-GOAL-4 0.5015 → 0.3917 (INHERITED); Hydra-MDP, GTRS, DriveSuprim | selection | 0.5 A40-day | Δ2 on parity-train fan dumps |
| 2 | **Multi-target rule-based teacher** (collision, TTC, progress, comfort) distilled into the scorer | proven (NAVSIM winners) | Hydra-MDP arXiv:2406.06978 | no cost model | 0.3 | Δ3 from `obstacle.offline` |
| 3 | **Speed history + measured acceleration** (`ax_fd`) as inputs | proven in-house | v0 as an action channel: +2.21 m [2.04, 2.39]; E-GOAL-3/4 | longitudinal | 1.0 | Δ1 |
| 4 | **Train at the evaluation horizon** (K = 20) | measured in-house on a secondary endpoint (ADE); primary `CR_k` pending; lateral family worsened | RR-20: 0.4244 → 0.3485, speed bias removed | compounding | 0.9 | Δ7 |
| 5 | **Planner reads rich spatial features**, not the 2,048-float WM state | proven in-house | same decoder: oracle-in-fan 0.164 on REF-C's map vs 0.338 on the WM state | readout bottleneck | 0.7 | Δ6 |
| 6 | **Factorised path × speed outputs** | proven (published + in-house) | SimLingo arXiv:2503.09594; D-TAC1 | tactical | 0.3 | Δ4 |
| 7 | **Mirror / SE(2) symmetry** | proven (published) | +20.6 % L2@3s on small data, arXiv:2403.11304 | small data | ≈ 1 | mirror-flip control arm |
| 8 | **Kinematic decoder and residual-over-prior outputs** | proven technique; untested here | standard; R1 Δ9 | steady-window loss, lateral consistency | 1.0 | Δ9 |
| 9 | **Train-time-only structure** from privileged tracks | proven (LbC, Roach, PlanT; GraphPilot arXiv:2511.11266) | gains persist with structure removed at test time | structure, longitudinal | 1–3 | `obstacle.offline` auxiliary heads |
| 10 | **Label-free video pretraining** of the world model | promising → proven | LFG arXiv:2602.22091 (85.2 PDMS); DriveVLA-W0 arXiv:2510.12796 | data efficiency | 10–25 | the slope (D4) |
| 11 | **Safety envelope:** RSS + CBF-QP + Simplex + calibrated trust gate | components proven; integration promising | RSS 2017; BarrierNet arXiv:2203.02401; Simplex arXiv:2605.08190; conformal planning 2023 | safety by design | 2–4 eng-weeks | closed-loop A/B with and without the envelope |
| 12 | **Conformal calibration** of uncertainty from two-head disagreement | proven (published) | Lindemann et al., RA-L 2023 | anti-calibrated imagination | 0.2 | calibrate on the 600-episode deployment |
| 13 | **LLM-generated closed-loop scenarios** | promising | Chat2Scenic arXiv:2607.14387 | 12-scene power bottleneck | eng-days | grow the suite to ≥ 50 scenes |
| 14 | **RL fine-tuning in re-simulation or imagination** | promising | Dreamer 4 arXiv:2509.24527; RAD; CaRL | closed-loop compounding | 5–20 | only after #2 and #12 exist |
| 15 | **Training-throughput engineering** (profile, bf16, `torch.compile`, CUDA graphs, fused AdamW) | proven engineering | CUDA graphs already 1.75–3.46× at inference (R2) | 19.58 s/step | eng-days | 200-step profile, then one lever at a time |
| 16 | **Mechanical guardrails** (hooks, CI, token ledger, model routing) | proven engineering | Claude Code hooks; R5 | process tax, idle turns, budget blow-ups | eng-days | wire four hooks to existing tools |

---

## 12. Disruptive bets

Each bet is sized to leapfrog rather than catch up, and each carries a pre-committed kill criterion.

### Bet A — The world model becomes the critic, not the headline

**Thesis.** Stop asking the world model to be the planner. Make it the consequence model inside a cost-aware selector: REF-C-class proposals, an expected-cost regressor with Hydra-style multi-target costs, and one reference roll from the arm's *own* world model as a feature. A ZTRS-style reward-only scorer runs as the competing arm.

**Why disruptive.** It is the only path in which the programme's distinctive asset (a vision world model) earns a measured place in decisions. It keeps the latency inside Thor's budget (§4.8).

**First gate:** Δ2 + Δ3 + Δ8 on frozen fans, about 1 A40-day.

**Kill:** Δ8 with the arm's own world model never beats the same selector without it, on either deployment. Then the world model is a representation learner, and the thesis is stated that way.

### Bet B — The strategic brain becomes the safety supervisor the mission describes

**Thesis.** The mission's strategic layer engages, degrades and blocks autonomy and diagnoses the system. Build exactly that:
- a supervisor fed calibrated uncertainty (two-head disagreement, conformal thresholds), imagination-vs-observation error, and envelope signals (QP feasibility, RSS margin);
- it outputs engage / cautious / degrade / minimal-risk manoeuvre;
- it is trained and validated on closed-loop failure outcomes in NuRec/AlpaSim.

**Why disruptive.** It turns the hierarchy from an unmeasurable claim into the "safety by design, no compromise between modular and E2E" architecture of Goal 2. It is also the natural carrier for UN ADS / WP.29 evidence, and it answers W1's modular-certifiability counter-argument.

**First gate:** the supervisor's risk score predicts closed-loop failures on held-out scenes at a pre-registered AUROC (proposed ≥ 0.75).

**Kill:** at matched progress, supervisor-driven degradation does not reduce failures (paired, n ≥ 50 scenes).

### Bet C — "1000× less labelled data" becomes a measured curve

**Thesis.**
- Pretrain the world model label-free on the unused part of PhysicalAI-AV's *train* split, about 153 k clips / 850 h, front-wide + front-tele. Never use the HF val/test splits or any clip whose NuRec scene is in the closed-loop suite or the challenge; register the exclusion list before the run. Later, add front-camera web video (LFG-style).
- Then train v6's heads at 1, 3, 10, 30 and 100 % of the paired hours.
- Train REF-C at the same fractions.
- Report all four families and closed loop.

The deliverable is the slope, fitted over at least 4 fractions with R² ≥ 0.8 as CLAUDE.md requires, plus the fraction at which pretrained v6 matches REF-C trained on 100 %.

**Why disruptive.** It is the mission's first priority goal, and nobody has published it for front-camera-only world-model planners at this scale.

**First gate:** at 10 % of paired hours, the pretrained arm is separated-better than the non-pretrained arm on at least two families.

**Kill:** no separated shift at two or more fractions. Then move the data-efficiency claim to structure priors and privileged teachers (#7–#9 in §11).

### Bet D — Train in imagination and in re-simulation

**Thesis.** Once Bet A supplies a cost and §11 #12 supplies calibrated uncertainty, fine-tune the decision layer in two ways:
- by RL inside the world model (Dreamer-4-style);
- by RL in NuRec re-simulation on Thor (RAD-style), where the programme already renders at usable rates.

**Why disruptive.** Closed-loop training attacks compounding error directly, which open-loop imitation cannot.

**Kill:** imagined reward rises while imagination error at visited states rises (reward hacking), or closed-loop success does not improve (paired, n ≥ 50).

### Bet E — Let agents engineer the stack, under mechanical guardrails

**Thesis.** The v6 ladder, the data slope and the envelope are sequences of bounded, pre-registered tasks, which is the shape at which current agents are reliable. Run them as an autonomous loop:
- **Models:** Sonnet implements, Opus adjudicates, Haiku lints and extracts.
- **Guardrails:** hooks enforce the rules, a token ledger prices every stream, and every result lands in the registry by pointer.
- **Humans decide** on spend, publication and thesis questions.

**Kill:** the retraction rate per validated finding, or the cost per validated finding, exceeds the programme's pre-loop baseline over two weeks.

---

## 13. Fix list

Found by the streams and checked where marked. P0 = this week at near-zero GPU; P1 = next two weeks; P2 = month two.

| pri | defect | where | fix | cost |
|---|---|---|---|---|
| P0 | Leaderboard §6 ranks WM fidelity (given the expert's controls) as 1= with planners | `MODEL_REGISTRY.md` §6 vs `:192`; CODE `rollout.py:146-147` | two tables (decision D1); replace the "legacy only" tactical-head row with 3.3839 [2.8336, 3.9722] | 1 eng-day |
| P0 | Evidence built on the unlike comparison | HPP0 "7.9×" (`HPP0_CONFOUND_AUDIT.md:457`); registry §6 reading 2; the vision-anticipation panel (`generalization.py:371-399`) | withdraw as hierarchy evidence; re-run anticipation with zero-order-hold actions | 1 A40-hour |
| P0 | Four-family instrument has no callers; tactical/strategic never measured on the canonical val | `taniteval/four_families.py`; `runner.py:341-342` | wire `all_families` into `runner.run_one`; fold distance-keeping into `driving.py`; extend the horizon fix to the lateral family | 1.5–2 eng-days |
| P0 | RR-20 has no admissible verdict under its own pre-registration | `PREREG_rollout_recovery.md` (primary `CR_k` missing) | compute `CR_k` on the existing checkpoints | 0.05 A40-day |
| P0 | RR-20 / RR-CTL are unregistered; two speed-bias instruments disagree 5× under one key | `…/incoming/2026-08-02-rollout-recovery-verdict/`; `four_families.py:185` vs `driving.py:342` | register both arms in `MODEL_REGISTRY.md` with all five JSONs; state which `speed_bias_mps` is canonical | 1 hour |
| P0 | The Thor-timed path (`TacticalSelector`, 9 primitives) has no accuracy measurement | `eval_behavior.py:23-25`; runbook `:134-137` | score it once on val-40 (four families), so the dossier's latency and accuracy refer to one path; otherwise label the tick "latency of the v1 selector path, accuracy not measured" | ≤ 0.2 A40-day |
| P0 | 22 M untrained H15 parameters in every v4-line run incl. v5f | CODE `train_flagship_v4.py:390,395,999`; `fourbrain.py:456` | confirm with a CPU state_dict diff; remove or train (Δ5) | CPU |
| P0 | v5f's defining lever cannot rank | CODE `flagship_v15.py:638-656` | evaluate v5f's latest checkpoint once on the four families with `--goal-mode produced` or `neutral`, never `oracle`; stop the line at 10 k unless it beats REF-C-XL on the decision surface (R1 R-8) | ≤ 0.5 A40-day |
| P0 | The world model has never been measured as a like-for-like forecaster on the canonical windows | R1 R-2 | roll v1 under the held last action on the 881, paired vs REF-C-base, CTRV and the bicycle | 0.1 A40-day |
| P0 | The headline decodes 20 steps with a readout trained on 4 | R1 R-11; `readout_selection.py` | apply the committed C8 rule (`op` ≤ 0.5 s, `str` beyond) to every production caller | 0.05 A40-day |
| P0 | Registry-integrity suite red at HEAD | 2 MISSING citations (§ Appendix C, O1) | commit `BACKUP_LOG.txt` from the dev box; cite the HF copy of the no-speed checkpoint or record it as `unresolved` deliberately | dev box, minutes |
| P0 | Registry §0.3 says TanitEval is uncommitted | it is 263 tracked files | correct the statement | minutes |
| P0 | Two AlpaSim result files both named "REF-C suite" disagree (6/12 vs 8/12) | R4 §3 | mark the unpaired one superseded | 1 hour |
| P0 | AlpaSim E2E Challenge date and sensor eligibility unknown | R4, V1 | confirm both | 1 eng-day |
| P0 | Cloud sessions cannot run the test suite | `download.pytorch.org` denied by the environment's network policy | allow the host, or provide a CPU-torch wheel cache | settings |
| P1 | No image augmentation of any kind | R3 §1.4 | mirror flip with steer and lateral sign flip, then crop and colour jitter | 2 eng-days |
| P1 | `v0` broadcast as a constant; `ax_fd` fed to no arm | CODE `flagship_losses.py:227-238` | Δ1 | 1 A40-day |
| P1 | Nothing regularises rolled-out states; the σ-head is trained one-step only | R2 §2; R1 §7 | SIGReg/isotropy on rollout states; two-head disagreement for uncertainty | 2–3 A40-days |
| P1 | `invdyn_gradscale` fix ships at the no-op default | R2 §6 | ablate 0.25 vs 1.0 at a 5 k gate | 1 A40-day |
| P1 | Seeds only; no deterministic data order | 8 trainers | `seed_everything` + seeded DataLoader | 0.5 eng-day |
| P1 | No profile, no compile / fused optimiser / flash attention | 8 trainers; 19.58 s/step | profile 200 steps, then add levers one at a time | 1–2 eng-days |
| P1 | Cadence configured, never executed | CODE `config.py:118,140` | implement in the deployed loop, or delete the claim | 1 eng-day |
| P1 | Intersection label half-built | `sc_cross.py` stranded in `incoming/` | promote into `stack/` | 1 eng-day |
| P1 | No mechanical enforcement of any CLAUDE.md rule | `.claude/settings*.json` have no hooks; no CI | `Stop` idle guard; `PreToolUse` on commit → `safe_commit.py`; `SubagentStop` manifest check; `SessionEnd` → `session_guard.py`; CI for `pytest` and `registry_lint` | 2–3 eng-days |
| P1 | No LLM spend ledger; no fan-out governor | R5 | per-stream token ledger; semaphore on sub-agent spawns | 1–2 eng-days |
| P1 | `LOOP_STATE.md` 203 KB and growing | R5 | split into a ≤ 1,500-token header + dated shards; size-guard hook | 1 eng-day |
| P2 | `hierarchy.run` accepts flagship arms only | `runner.py:341-342` | tactical/strategic families for REF-B/REF-C (an architecture question) | PI + 2–3 eng-days |
| P2 | The v2corpus contrast is confounded | R3 §4 | launch the lever-matched control before any "more data helped" claim | ≈ 6 A40-days |
| P2 | Registry copies numbers rather than pointing at them | 07-25 proposal #9 | `results_ledger.py` | 3 eng-days |

---

## 14. The plan

### 14.1 The ten days to the 05.10.2026 evaluation

The goal is a dossier the PI can defend in front of anyone: every number with its estimator, every family reported, and the hierarchy stated as it stands.

**Priority order, so a slipped item still leaves a defensible deliverable:**
1. registry split (D1);
2. four-family panel on val-40 for v1 and REF-C;
3. Δ2 on val-40;
4. the Thor tick stated honestly;
5. the existing 12-scene closed-loop table as it is;
6. envelope v0;
7. new closed-loop runs.

**Dependency.** The val-40 fan dumps are in the repo (`taniteval/results/fan_refc-{base,xl}-30k.pt`, `windows_*.pt`). The 600-episode fan dump is not: its poses lived on the terminated pod2 (`EGOAL_4.md:81`). Either cost its regeneration (the REF-C-XL checkpoint, the val-600 corpus and one GPU) or run days 2–5 on val-40 only and move the 600 arm to Phase A.

| days | work | GPU | output |
|---|---|---|---|
| 1 | Fleet check after 52 days (what is alive, where v5f stopped, what it costs); allow `download.pytorch.org` for cloud sessions; registry split (D1) | 0 | a truthful fleet and leaderboard |
| 1–2 | Wire the four families into the runner; compute `CR_k` for RR-20; measure the world model as a held-last-action forecaster on the 881; apply the readout rule; confirm the dead-H15 diff; confirm AlpaSim Challenge date and eligibility | ≈ 0.3 A40-day | four-family panels become routine; the like-for-like forecaster number |
| 2–5 | Frozen-fan ladder on REF-C-XL: Δ2 (expected-cost selector trained on parity-train fans, reconciling E-GOAL-4 with v1.2), Δ3 (`obstacle.offline` cost targets), Δ8 (reference roll with v1's world model), each on val-40 and the 600-episode deployment, paired | ≈ 1–1.5 A40-days | the first decision-layer result on the correct target |
| 3–7 | Safety envelope v0 in simulation: RSS longitudinal check + kinematic CBF-QP + Simplex fallback on the selected trajectory. Labelled as an oracle-perception envelope test, because the simulator supplies lead state; the deployable version needs the vision range head of Phase A | Thor | envelope mechanism measured: collisions, off-road, progress |
| 5–9 | Closed loop on Thor: REF-C-base vs REF-C + selector (± envelope) vs flagship v1's tactical path, paired, on every available NuRec scene. Report n; 12 scenes are underpowered | Thor | the closed-loop table |
| 9–10 | The dossier: four families per arm with paired CIs; closed loop; Thor latency; what the hierarchy has and has not shown; the plan below | 0 | the 05.10 deliverable |

### 14.2 Days 11–90

| phase | days | aim | contents | gate to pass |
|---|---|---|---|---|
| A | 11–30 | make decisions measurable and fix the decision layer | the rest of the v6 ladder (Δ1, Δ4–Δ7, Δ9); mirror augmentation; `obstacle.offline` auxiliary heads; strategic supervisor v0 with conformal thresholds; the 600-episode deployment as the paired decision set and the closed-loop suite grown to ≥ 50 scenes; hooks, CI, token ledger | on the 600-episode deployment (val-40's MDE is 0.06–0.16 m): v6 separated-better than REF-C-base on at least one family and non-inferior on the rest, with pre-registered margins (e.g. paired ADE@2s upper bound < +0.02 m); non-inferior in closed loop; ≥ 4 of 9 ladder steps graduate |
| B | 31–60 | scale what graduated | label-free pretraining on 10–30 % of PhysicalAI-AV (front-wide + front-tele) → the data-efficiency slope (Bet C); envelope v1 with a vision range head; RL fine-tuning in imagination or NuRec (Bet D); failure-mining with replay; the strategic test on a corpus with route information (AlpaSim routes or L2D) | slope with R² ≥ 0.8 over ≥ 4 fractions; supervisor AUROC gate (Bet B) |
| C | 61–90 | prove it externally | AlpaSim Challenge entry or its successor; a second external benchmark if feasible; the safety case (ISO/PAS 8800 / UL 4600 patterns); the data-efficiency paper | an external number with a rank; a safety case with explicit assumptions |

**Compute** (ESTIMATED): Phase A 10–20 A40-days, Phase B 30–60, Phase C 10–20 plus Thor time. That is within a few A40s over 90 days if training throughput improves as §13 intends. At today's 19.58 s/step, Phase B does not fit.

### 14.3 Decisions only the PI can make

- **D1–D5** (§1).
- **Whether to stop the v5f line** once its last checkpoint is evaluated.
- **Pods to keep, release or provision** for Phases A–B.
- **Cloud-environment network access** (`download.pytorch.org`).
- **Whether the 05.10 dossier states the hierarchy result as it stands.** This review recommends it does.
- **Which reading of "4B architecture" the 05.10 evaluation scores.** Phase 0 (`Mission Plan.md:181`) asks for "a running architecture with the most important hypotheses like the 4B architecture". v1 satisfies the four brains as built, and it loses to a flat planner. v6 keeps four measurable *roles*: proposer, selector, forecaster, supervisor. This review recommends scoring the roles, and saying so in the dossier.

### 14.4 How to run it with agents, efficiently

| role | model | effort | why |
|---|---|---|---|
| orchestrator: briefs, relay, verification of load-bearing claims | Opus 5.5 (Opus 5 where 5.5 is not yet available) | high | the programme's own scorecard locates its dominant error source at this layer (R5) |
| architecture judgment, gate adjudication, causal analysis | Opus 5.5 | high to xhigh | low error tolerance; this review's architecture stream is the example |
| implementation, experiments, literature screens | Sonnet 5 | medium to high | bounded tasks with a check; about half the Opus price |
| extraction, lint triage, citation and number checks | Haiku 4.5 | low | the judgment lives in the tool; this review's fact base and citation check cost little |
| red-team of anything going to the PI or outside the programme | Fable 5.1 | high, sparingly | the most capable model where one wrong claim costs GPU-weeks |

Prices, per million input/output tokens: Haiku 4.5 $1/$5, Sonnet 5 $2/$10, Opus 5.5 $4/$20, Fable 5.1 $10/$50 (Claude API reference, cached 2026-06-24).

**Operating rules that save tokens without losing rigour:**
1. Five to seven streams on *independent* questions. For coupled work (one trainer, one registry), use a single stream with depth.
2. Every brief carries the preamble, a priority order, a file budget (≤ 1,000 lines) and a summary budget (≤ 1,200 words).
3. The orchestrator reads summaries, then targeted ranges, never whole state files.
4. Sub-agents never spawn sub-agents.
5. Every stream logs its tokens.
6. Load-bearing claims are re-checked by a second, cheaper model before they reach the PI. This review's Haiku pass caught a wrong arXiv id and six unconfirmed numbers for a few cents.

---

## 15. What would change this assessment

- **A hierarchy decision path wins.** If the tactical or strategic path beats REF-C-base, paired on the same windows, the hierarchy verdict in §4.9 reverses.
- **Δ2 fails to replicate.** If the selector trained on parity-train episodes does not reproduce E-GOAL-4, the selection lever is smaller than §1 claims. v6's centre of gravity then moves to Δ6 (planner substrate) and Δ9 (residual anchors).
- **Label-free pretraining does not shift the curve.** The data-efficiency claim then rests on structure priors and privileged teachers, and "1000×" should not be stated.
- **The AlpaSim Challenge cannot take a front-camera-only entry.** Then choose a benchmark that can (for example Bench2Drive with a front camera), not a different sensor set.
- **v5f's last checkpoint evaluates well on all four families.** Revisit §4.3's reading of the probe-token lever. This is unlikely, since the lever cannot rank by construction, but the checkpoint should be measured, not assumed.

---

## Appendix A — Streams and deliverables

| stream | scope | model | file |
|---|---|---|---|
| R1 | architecture and causal diagnosis | Opus 5.5 | `streams/R1_architecture_causal.md` |
| R2 | training, anti-collapse, compute | Sonnet 5 | `streams/R2_training_anticollapse.md` |
| R3 | data, encoding, resolution, labels | Sonnet 5 | `streams/R3_data_encoding_labels.md` |
| R4 | evaluation, statistics, closed loop, external benchmarks | Sonnet 5 | `streams/R4_eval_metrics_closedloop.md` |
| R5 | agent harness, token efficiency, AI-engineering-AI | Sonnet 5 | `streams/R5_agentic_harness.md` |
| W1 | frontier: world models, planning/selection, VLAs, brain-inspired | Sonnet 5 | `streams/W1_frontier_worldmodels_planning_brain.md` |
| W2 | frontier: data efficiency, pretraining, distillation, continual learning | Sonnet 5 | `streams/W2_frontier_data_efficiency.md` |
| W3 | frontier: structure, reasoning, tools, neural operators, guarantees | Sonnet 5 | `streams/W3_frontier_structure_reasoning_guarantees.md` |
| H1 | fact base: registry arms, retractions, hypotheses, timeline | Haiku 4.5 | `streams/H1_fact_base.md` |
| V1 | independent citation check of 31 load-bearing references | Haiku 4.5 | `streams/V1_citation_check.md` |
| X1 | red-team of this synthesis: 7 must-fix and 11 should-fix findings, all applied | Fable 5.1 | `streams/X1_redteam.md` |

The orchestrator (Opus 5.5) wrote this synthesis and re-verified these load-bearing facts in the source:
- the expert-future-controls headline (`rollout.py:146-147,181-185`);
- the untrained H15 module in the v4 line;
- the 2026-08-04 distance-keeping numbers (resolving a conflict between R3 and the repository);
- the Thor tick artifact;
- the follow-through audit.

All deliverables are in this folder and nothing lives only in the container.

## Appendix B — Reconciliations between streams

- **Distance-keeping.** R3 says distance-keeping is unavailable for every arm. The 2026-08-04 artifact (`…/incoming/2026-08-04-distance-keeping-arms/DISTANCE_KEEPING_ARMS.md`) computed it for 25 arms on val-40. Both are true: the numbers exist, but they come from incoming-folder scripts that the standard runner does not call.
- **The flagship's speed bias.** The 2026-08-04 "speed-setting bias" (+0.1911 m/s) was measured on the WM-fidelity path. It is a world-model integration bias, fixed by K = 20 training, not a planning decision.
- **Retraction classes.** H1's retraction-class counts (C3 19, C4 15, C5 12, C1 8, C2 7, C6 7 of 68) are a Haiku grouping of free text and disagree with the log's own ranked order. Treat them as indicative only.
- **V1's summary line.** V1 reports 23 confirmed / 7 unconfirmed; its table rows show 24 confirmed, 6 with the number unconfirmed, and 1 ID mismatch. The table is used here.
- **Applied from the red-team (X1).** The corrections, and the source each rests on:
  - PhysicalAI-AV has 7 camera streams, not 6 (in-repo feature probe).
  - The Thor tick is v1's 9-manoeuvre selector, a decision on an unscored path, not a single rollout of given actions.
  - The hierarchy's losses are 4.0–7.2× on full-set numbers; the deprecated P2 split-mean is no longer used.
  - The goal-information claim was true by construction and is restated.
  - E-GOAL-4 is out-of-fold within val-600 and must replicate on train-corpus fans before it decides GPU-days.
  - Label-free pretraining is restricted to the HF train split, with a registered exclusion list.
  - Two RR-20 speed-bias instruments disagree 5× and are both quoted.

## Appendix C — Method, limits and orchestrator findings

### How this review was produced

Ten streams (nine in parallel, then a citation check), each on the cheapest Claude model that could do its job, each writing a banked stream report into `streams/` with a deliverable manifest. The orchestrator (Opus 5.5) wrote the synthesis and checked the load-bearing numbers against `MODEL_REGISTRY.md` and the repo.

| stream | scope | model | why this tier |
|---|---|---|---|
| R1 | architecture & causal diagnosis of the 4-brain world model | Opus 5.5 | highest-judgment question in the programme |
| R2 | training recipes, objectives, anti-collapse, compute | Sonnet 5 | structured code reading with good judgment at half the Opus price |
| R3 | data, encoding, resolution, labels | Sonnet 5 | same |
| R4 | evaluation, statistics, closed loop, external benchmarks | Sonnet 5 | same |
| R5 | agentic harness, token efficiency, AI-engineering-AI | Sonnet 5 | doc-heavy reading plus web research |
| W1 | frontier: world models, imagination, planning/selection, VLAs, brain-inspired | Sonnet 5 | web research and screening |
| W2 | frontier: data efficiency, pretraining, pseudo-labels, distillation, continual learning | Sonnet 5 | web research and screening |
| W3 | frontier: structure, symbolic reasoning, knowledge, tools, neural operators, guarantees | Sonnet 5 | web research and screening |
| H1 | fact base: registry arms, retraction log, hypotheses, decisions, git timeline | Haiku 4.5 | mechanical extraction from 200–330 KB files |
| V1 | independent citation check of 31 load-bearing references | Haiku 4.5 | mechanical search-and-compare |

Prices used for routing (Claude API reference, cached 2026-06-24, per million input/output tokens): Haiku 4.5 $1/$5 · Sonnet 5 $2/$10 · Opus 5.5 $4/$20 · Fable 5.1 $10/$50. Fable 5.1 was used once, for a bounded red-team of this synthesis (`streams/X1_redteam.md`).

Limits of this review: it is static. This container has no pod access, so nothing here re-runs a model; every model number is quoted from `MODEL_REGISTRY.md` or a committed results JSON, and every research claim cites its paper. `torch` could not be installed (the network policy denies `download.pytorch.org`), and every `stack/tests` and `taniteval/tests` module needs it at collection (directly, or through `tanitad` imports), so those suites were not run. The torch-free `tools/tests` suite was run: 237 pass, 2 fail, 1 skip (the 2 failures are real and pre-existing at HEAD; see "Fix now").

### Alignment with the mission plan

`Project Steering/Mission Plan.md` sets the bar: beat Wayve, Waymo, Pony, Momenta and Autobrains "with clear theoretic and practical proofs"; **Goal 1** a running closed-loop system on embedded (Orin, Thor); **Goal 2** generalisation with far less data ("magnitude less e.g. 1000x less"), safety by design with no safety/E2E compromise, inference efficiency, UN ADS regulation compliance, adaptability; **P7: first final evaluation on 05.10.2026**, ten days from this review.

The mission's own definition of the layers matters for the architecture verdict. The strategic layer is "engaging, deactivating, ending or blocking the usage of autonomous driving, diagnosis of the overall state of the system", plus navigation and explanations. The tactical layer plans 3–5 s and owns manoeuvre selection and interactions. The implementation narrowed the strategic brain to four nav commands and a route head on a corpus with no routes, and trains and evaluates the tactical brain at a 2 s horizon. Both brains were built for jobs the data cannot supervise and the metric cannot see.

### Fix-now items found by the orchestrator (MEASURED 2026-09-25)

| # | defect | evidence | fix | owner |
|---|---|---|---|---|
| O1 | The registry-integrity suite is red at HEAD: 2 MISSING citations | `python tools/registry_paths.py --only-bad` → `_pod_backup/pod2-2026-08-03/ckpts/BACKUP_LOG.txt`, `…/flagship4b-phase0-30k_ckpt.pt`; `tools/tests/test_registry_paths_allow.py` 2 failures | Commit `BACKUP_LOG.txt` from the dev box (it is small and not git-ignored; only `*.pt` is, `.gitignore:48`). For the 3.3 GB no-speed checkpoint, cite the HF copy `Sayood/tanitad-flagship-4b-phase0` as the primary location, or record it as `unresolved` and raise `max_unresolved` deliberately. The test forbids allowlisting to make it pass. | PI dev box |
| O2 | The no-speed ablation control's `config.json`, `train_log.jsonl` and gate JSONs have no reachable copy | `MODEL_REGISTRY.md:153` | Accept the loss formally in the registry (the control's ADE survives in `taniteval/results/driving_flagship-nospeed.json`), or re-run the control if any future claim leans on its training curve | registry maintainer |
| O3 | The stack and TanitEval test suites cannot run in a cloud session | `download.pytorch.org` denied by the environment network policy; all `stack/tests` and `taniteval/tests` modules need `torch` at collection | Allow `download.pytorch.org` in the cloud environment's network settings, or add a CPU-torch wheel cache, so cloud agents can honour "pytest -q green before any commit" | PI (environment settings) |
