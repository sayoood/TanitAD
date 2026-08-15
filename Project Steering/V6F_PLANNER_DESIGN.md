# V6F PLANNER — design and staged implementation plan

**Written 2026-08-15, branch `agent/arch-inf-20260803`, HEAD `30d6d60`.**
Answers the PI's request: *"What is our concrete plan for the planner of v6f, how is the design and
the implementation of all learnings from 1.7 v1.8 and v5.8 regarding Diffusion planner, mpc and
selector. Don't forget that we aim to train the strategic and tactical layers using their own
predictors and each layer conditioning the lower layer.. the resulting trajectory must be 6s long as
a result of operative and tactical planning."*

**Reading rules in force.** `MODEL_REGISTRY.md` + raw JSON are the only quotable sources for model
facts; every number carries its evidence class and its T-tier; intervals are the **paired
episode-cluster bootstrap** (`taniteval/ci.py`) and `overlapping_holdout_se` appears nowhere.
⛔ Nothing here was trained. All code landed by this document is **gated and default-off**, and the
all-off build is proved **byte-identical** to HEAD (§7).

---

## THE MECHANISM, IN ONE PARAGRAPH

v6f produces **one** 6 s trajectory, not a stitch: the operative planner emits a single 60-step
`(a, κ)` control sequence at 10 Hz from `feat = [plan_proj(z_op) + cand_query_i ‖ e_g_tac]` and
integrates it through **one** unicycle rollout from the true `v0`, so steps 0–19 (the operative band,
0–2 s) and steps 20–59 (the tactical band, 2–6 s) are slices of the same integrated path and
`V6Config` **refuses** any band gap or overlap at construction (`v6.py:754-764`) — the seam is
absent by construction and X2 verifies rather than repairs it. Each layer contributes through **its
own predictor**, which is already built: `predictor_str` (`FTac`, `d_str` 256, 0.5 Hz, `stride_str`
20) rolls the strategic latent under the strategic layer's own action embedding `e_a_str`;
`predictor_tac` (`FTac`, `d_tac` 512, 2 Hz, `stride_tac` 5) rolls the tactical latent under the
**factored** `e_a_tac = [LAT ‖ LON]`; `predictor_op` (`OperativePredictor`, FiLM/intent-conditioned,
189.96 M in config E) rolls the operative latent under the emitted controls and is the rollout engine
for any per-candidate consequence check (`v6.py:1394-1395, 1402-1403`). Conditioning flows **down
only, through exactly four named ports**: `g_str → cond_tac → goal_head_tac(z_tac_p, cond=e_g_str)`;
`g_tac → cond_op → predictor_op(intent=…)`; `g_tac → the emission feature`; and — added by this
design — `g_tac → the selector's goal point`. Every one of those ports crosses a **declared detach**:
the planner-side views `z_plan / z_tac_p / z_str_p` are cut by `isolate_planner_from_encoder`
(`v6.py:1374-1377`), `e_g_tac` enters `predictor_op` detached, and the goal-seam term `zhat_op_seam`
runs on `z_op_win.detach(), actions.detach()` so its gradient can reach `intent_proj` and nothing in
any encoder (`v6.py:1417-1418`) — which is why X3 isolation still measures
`{planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0}` with the new head attached
(`tests/test_v6_selector.py::test_scorer_is_in_planner_surface_and_isolation_holds`). The **tactical
layer's authority over the 2–6 s band is exercised through the goal, never through waypoints**: its
`ANCHOR_GOAL(anchor_id, t_reach_s)` supplies the geometric goal point the selector scores against and
its `SPEED_BAND(v_lo, v_hi)` supplies the longitudinal envelope — the family that owns **~99 % of the
measured T1 divergence** — with the strategic `REDUCE_TO` acting only as an upper bound on that band.

## THE SELECTION ANSWER, IN ONE PARAGRAPH

The winner's curse is **a property of the cost, not of the fan size**, and that is now MEASURED
rather than argued: re-analysing the banked in-repo REF-C-XL fan today (881 windows / 40 episodes /
256 candidates, `stack/scripts/sel_winners_curse_law.py`), a **world-model roll-consistency score** —
the W7 quantity, present in that dump as `cons_score` — has a normalised argmax error-rank that
**RISES** with the candidate count (0.241 at N=4 → **0.286** at N=256, where 0.5 is a coin flip)
while its lower-tail hit rate **COLLAPSES** (0.57 → **0.28**) and its ADE stays pinned at ~6.2–6.45 m
while the oracle falls 4.606 → **0.164**; whereas REF-C's **supervised** selector on the *same fan*
moves the opposite way — rank **0.099 → 0.014**, lower-tail **0.77 → 0.99**, ADE **5.365 → 0.471** —
and the pattern **replicates on an independently trained model** (REF-C-base: roll-consistency
0.243 → 0.283 / p10 0.41 → 0.25; supervised 0.082 → 0.021 / p10 0.62 → 0.98). ⇒ **Shrinking the fan
does not fix a bad cost and costs 15× of oracle headroom, and the registry's suggested remedy —
top-m aggregation — is refuted here as a standalone fix** (top-8 medoid is **+0.1294 m
[+0.0645, +0.2029] WORSE** than argmax, paired-separated; the top-m *mean* is flat at the fan's own
mean, reproducing `w7_selection_rules.json`). What *does* work is changing the **estimand** to a
**candidate-independent reference**, which has no degenerate minimiser because inaction cannot
minimise a distance to a goal: selection by distance from the candidate's endpoint to a goal point of
1σ accuracy **0.5 m** beats the trained selector by **−0.1591 m [−0.2300, −0.0894], separated**,
while at **1.0 m** it loses by **+0.0943 [+0.0241, +0.1650]** — so the design carries a **measured
admission threshold (σ\* ≈ 0.8 m at this fan's 2 s horizon ≈ 1.7× the incumbent's selected ADE)
instead of a hope**, and the zero-information control (goal replaced by its corpus marginal) sits far
away at **7.8237 m**, so a win cannot be an artefact of "any goal point works". v6f therefore
**trains** the selector (S-T, on the trunk it consumes), scores it **at the ranked object**, gives it
the **goal as its reference**, and judges it on **normalised error-rank and lower-tail hit rate**, not
on ρ — because ρ is a bulk statistic and argmax is an extreme one.

## WHAT I DID NOT RESOLVE — stated up front

1. ⛔ **Nothing here is measured on a v6 fan.** v6 has never emitted one; S-W is at step ~6,300 of
   30 k. Every number in §3 is from banked REF-C fans at a **2 s / 4-waypoint** resolution. The
   *directions* (the N-law, the goal crossover as a ratio) are structural; the **absolute σ\* at 6 s
   must be re-measured, never extrapolated** (the ≤2× extrapolation rule).
2. ⛔ **The W7 per-window arrays are stranded.** `w7_eval_windows.pt` / `w7-full-roll/` lived on
   pod4/pod5, which are gone, so W7's own top-m medoid can never be computed. §3 substitutes an
   independent surface; it is not the same fan.
3. ⚠️ **A consumer-invalidation risk INSIDE the ladder, newly identified and unmeasured** — see §4.3.
   S-S retrains `goal_head_str`, which moves `e_g_str` → `g_tac` → the goal the S-T-frozen selector
   consumes. That is registry §1.14's mechanism one level up and it has no gate today.
4. ⚠️ **Where the goal's supervision comes from is not solved by this document.** `PH0_COVERAGE_AUDIT`
   measures that **PH0 emits none of the nine `g_tac` tokens** and that five need agent slots we do
   not extract. Without `ANCHOR_GOAL` supervision the goal head is trained only by the plan loss
   flowing back through the scorer, which is a much weaker signal than the design assumes.
5. ⚠️ The `"mlp"` **capacity control is specified but not implemented** (§5.3). Until it exists a
   `"goal"` win is not attributable between mechanism and capacity — `V6Config` refuses `"mlp"` today
   with exactly that message rather than silently allowing an unattributable arm.

---

## 1. THE MECHANISM, END TO END

### 1.1 One control sequence, one rollout, two bands of authority

| | value | source |
|---|---|---|
| `PLAN_STEPS` | 60 | `v6.py:117` |
| `DT` | 0.1 s | `v6.py:118` |
| `HORIZON_S` | 6.0 s | `v6.py:119` |
| `OP_BAND_S` | (0.0, 2.0) | `v6.py:120` |
| `TAC_BAND_S` | (2.0, 6.0) | `v6.py:121` |
| band-gap / overlap | **refused at construction** | `v6.py:754-764` |
| tactical band must end at the horizon | **refused otherwise** | `v6.py:760-764` |

`V6Stack.emit` (`v6.py`) builds `feat = [plan_proj(z_op) + cand_queries[i] ‖ e_g_tac]` and calls the
W4-gated `UnicycleEmission` (imported from `scripts/train_v58f_unicycle_head.py`), which returns
`(a, κ)` per step and integrates **one** unicycle rollout to waypoints. `cfg.split_bands` is the only
place in the programme that turns the band boundaries into indices — one convention, no second
opinion.

⭐ **Why this is not a stitch, restated as the property that matters:** a stitched design has two
generators and needs a continuity repair at the seam. Here there is **one generator, one integrator
and one state**; the 2 s boundary is a *slice index*, so continuity in position, heading, speed and
curvature is arithmetic. The X2 seam metrics therefore *verify*; they cannot be asked to *fix*.

### 1.2 What each of the three predictors contributes

| layer | predictor | clock / stride | conditioned by | contributes to the 6 s trajectory |
|---|---|---|---|---|
| **operative** | `OperativePredictor` (FiLM, `intent_dim=d_goal_embed`) | 10 Hz / 1 | the emitted controls **+ `intent = e_g_tac` (detached)** | the **physics**: action-responsiveness of the latent, and the rollout engine for per-candidate consequence checks. Stage-A proved this interface is repairable (lateral gain 0.27 → 0.971/0.966, longitudinal sign → 1.0/1.0, P6 exactly 3-dim) |
| **tactical** | `FTac(d_tac=512, d_goal=2·d_goal_embed)` | 2 Hz / 5 | `e_a_tac = [LAT ‖ LON]`, its **own** factored actions | the **2–6 s shape**, via `g_tac`: `ANCHOR_GOAL` → the selector's reference point; `SPEED_BAND` → the longitudinal envelope; the goal embedding → the emission feature and the operative intent port |
| **strategic** | `FTac(d_str=256, d_goal=d_goal_embed)` | 0.5 Hz / 20 | `e_a_str`, its **own** actions | the **envelope on the tactical goal**: `cond=e_g_str` into `goal_head_tac`, and `REDUCE_TO` as an upper bound on `SPEED_BAND` |

⚠️ The tactical target sits one **tactical** tick ahead (`stride_tac = 5`) and the strategic one
strategic tick ahead (`stride_str = 20`) — *"predicting one operative tick ahead and calling it a
tactical prediction is an identity map wearing a hierarchy's name."* This is already enforced in
`V6Stack.layer_targets`.

### 1.3 The four downward ports, and their detach points

```
 z_str ──►[goal_head_str]──► g_str ──►[cond_tac]──► e_g_str ──┐   (port 1)
                                                              ▼
 z_tac ──────────────────────────────────►[goal_head_tac(z_tac_p, cond=e_g_str)]──► g_tac
                                                              │
                          ┌───────────────────────────────────┼──────────────────┐
                          ▼ (port 2)                          ▼ (port 3)         ▼ (port 4, NEW)
              predictor_op(intent = _cut(e_g_tac))    emit(feat ‖ e_g_tac)   cand_score(wp, e_g_tac)
```

| # | port | detach | why it is X3-safe |
|---|---|---|---|
| 1 | `g_str → goal_head_tac` | `z_str_p = _cut(z_str, isolate_planner_from_encoder)` | the goal heads read a **cut** latent, never the raw one |
| 2 | `g_tac → predictor_op.intent` | `intent = _cut(e_g_tac, cut)`; the seam term additionally runs on `z_op_win.detach(), actions.detach()` | gradient reaches `intent_proj` + the goal embeddings and **nothing in the encoder, by construction** |
| 3 | `g_tac → emission` | `z_plan = _cut(z_op, cut)` | the emission's only trunk input is already cut |
| 4 | `g_tac → cand_score` | consumes `e_g_tac` and `plan["waypoints"]`, both already downstream of the cuts | added to the **declared** `planner_side` list, so `assert_isolation` probes it |

⛔ **The `intent_proj` grouping is deliberate and load-bearing** (`v6.py:1153-1155`): `predictor_op.intent_proj.*`
and `intent_gate` belong to the **planner** group, not to `predictor_op`. MEASURED 2026-08-13: with
them under `predictor_op`, S-W trains them while `intent=None` (zero gradient, dead weight at random
init) and S-T **freezes them exactly when `g_tac` first flows** — the hierarchy's downlink would stay
random until S-J. The dynamics stay trunk-frozen in S-T; only the goal-injection port moves.

### 1.4 The admissibility audit, applied to every planner input (PI 2026-08-03)

| signal at inference | computed from | situation-classifier path? | verdict |
|---|---|---|---|
| `g_str`, `a_str` | `z_str_p` ← `adapter_str(z_tac)` ← vision | none exists in `GoalHead` | ✅ |
| `g_tac`, `a_lat`, `a_lon` | `z_tac_p` + `e_g_str` | none | ✅ |
| the emitted fan | `z_op` (vision) + `e_g_tac` + `cand_queries` + true `v0` | none | ✅ (`v0` is ego **speed**, an input the programme has always fed; it is not a classifier output) |
| **`ĝ` (the goal point, NEW)** | `e_g_tac` only | none | ✅ |
| `SPEED_BAND` envelope | `g_tac` args | none | ✅ |

⚠️ **The honesty hazard the goal creates, stated rather than buried.** A goal head trained to regress
the future endpoint *is* a trajectory predictor; selecting on it moves the planning into the goal
head and calls the result a hierarchy. Two structural bounds, both enforced in code:
1. **The goal is 2 numbers** (`nn.Linear(d_goal_embed, 2)`) against a 60×2 plan. A "goal" wide enough
   to carry the path is not a goal, and `GoalDistanceScorer` cannot emit one.
2. **The goal-echo control** must be reported beside every number. MEASURED today: the echo (goal
   replaced by its corpus marginal) selects at **7.8237 m** against the live goal's **0.7862 m** —
   the same shape as the nav-echo test that caught flagship v1's route head scoring 1.0000.

---

## 2. WHAT EACH PRIOR LEARNING CONTRIBUTES — AND WHAT IT FORBIDS

### 2.1 REFUTED — must not return in any form

| refuted | evidence | what it forbids in v6f |
|---|---|---|
| ⛔ **L1 "MPC as re-ranker" as posed** (score the fan by roll-consistency, emit the argmin) | **W7-FULL, MEASURED 2026-08-12 [T0, 881]:** all 256 candidates, every stale component removed, **selected 3.3348 against a 0.4505 gate over a fan whose oracle is 0.1273**; within-window ρ 0.445/0.497, across-window ρ 0.3185 [0.2064, 0.4086], **argmin error-rank 132 of 256 — the median** | no design may propose "score the fan better" without a mechanism against the winner's curse |
| ⛔ **A larger anti-degeneracy weight as the fix** | **W7-PROG, MEASURED 2026-08-12 [EXPLORATORY]:** rank falls monotonically 132.3 → 130.31 → 126.69 — **5.6 positions of 256 (2.2 %)** — while gate ADE gets monotonically **worse**, and across-window calibration **flips sign** (+0.3185 → −0.4244) | ⇒ binding: *"the cost needs a **goal-conditioned** component, not a larger anti-degeneracy weight, and W7-style self-consistency selection is retired as a headline route"* |
| ⛔ **Top-m aggregation as the standalone remedy** | `w7_selection_rules.json`: top-m **mean is flat at ~5.32 for every m — the fan's own mean**. Independently reproduced today: top-8 medoid on the shipped score is **+0.1294 [+0.0645, +0.2029] WORSE**, separated | aggregation is a **variance** fix; the defect is **information**. It may be a tie-breaker, never the answer |
| ⛔ **Keeping a planner across a trunk repair** | registry §1.14: repairing the trunk moved the frozen selector **0.7933 → 4.4159**; fan oracle 0.1077 → 0.289; winner-in-shortlist 19.6 % | the selector is a `planner`-group module and is (re)trained in **S-T, on the S-W trunk it consumes** |
| ⛔ **A learned value model as the search cost** | frozen-WM quartet, MEASURED 2026-07-23/24: V-search **1.0162 [0.809, 1.273]** vs W **0.5989**, **+0.417 [+0.237, +0.605] separated WORSE**; *"the entire gap to the search ceiling is V's imperfection"* | no `V(z)` term |
| ⛔ **Feed-forward planner capacity as the lever** | same quartet: W 0.599 → mlpbig (30.8 M) 0.601 → mlpwide (42.6 M) 0.599, **flat**; bigger query-decoders **overfit** to 0.82–0.86 | the selector is +267 params, not +40 M. Capacity is the **control**, never the lever |
| ⛔ **Free-XY candidates** | H1, `DIFFUSION_PLANNER_COMPARISON`: a per-waypoint offset head amplifies the same ε by **25×** in acceleration at dt 0.1 vs 0.5; the v5f dense fan measured **97.6 %** infeasible steps / **100 %** infeasible candidates | candidates are **controls**, always. `UnicycleEmission` is feasible by construction (census violations 0.0) |

### 2.2 SURVIVES — and how each is used

| survives | evidence | role in v6f |
|---|---|---|
| ⭐ **The roll-cost as a calibrated signal, *if* goal-conditioned** | ρ **0.399** frozen trunk → **0.7164 [0.5847, 0.7696]** stage-A repaired (P7 PASS, the strongest calibration in the programme) | **NOT the selector.** It is admissible only as a *secondary* term inside a goal-admissible shortlist, and only after the primary endpoint is met. MEASURED today why: inside a 2-candidate goal-admissible set, roll-consistency scores **0.7481** against the goal-alone **0.7862** — it adds **≈0.04 m**, not 5.7 m |
| **Kinematic cost as a tie-breaker on a feasible fan** | §1.14: top8+kincost **0.4815** beats the trained rescorer's argmax **0.560**; W1 refuted it on the *rough* fan (−16.7 %) | a low-weight tie-break term inside the shortlist, with a pre-registered zero-weight ablation |
| ⭐ **The unicycle emission, 60 steps** | W4: **109 k new params, 4 k steps**, violations **0.0**, selected accel MAE 0.774 vs 9.297, oracle **0.1077** (nearly halved). Trunk md5-identical | the v6 output contract, already imported |
| ⭐ **Supervision at the ranked object** | E-S1-0: supervised t=0 conf selects **0.4728**; **the SAME weights'** unsupervised refined readout selects **1.3100** — a **2.8×** penalty purely for scoring off-distribution. **Reproduced today on the XL fan: shipped 0.4714 vs refined 1.3901 (2.95×)** | the selector reads the **emitted trajectory**, and is trained on it |
| ⭐ **Metric-aware, hard-target ranking objective** | E-OBJ-1, MEASURED, LOEO, paired cluster bootstrap: swapping a fitted ranker from one-hot CE to **`softade`** recovers **−0.0974 m (base) / −0.1670 m (XL)**, separated, and the recovery is **LONGITUDINAL** (`speed_abs` −0.1102 / −0.1816); **softening the CE target is separated WORSE (+0.0909 m) at every τ** | ⇒ metric-awareness helps, target-softness hurts. `w_select` uses **`softade`**. 0 params |
| **Reachability prefilter** | REF-C S2/S2b deletes **72.08 %** of the fan for a paired ΔADE of exactly **0.0000** and a **3.5×** per-candidate compute saving | the admission ticket for any per-candidate WM roll; 0 params |
| **`sel_ce_reach` — normalise the ranking objective over the survivor set** | `refc.py:486-499`: the CE is a full-fan softmax while the selector solves a ~26–28 % sized problem; *"a statistic over the whole candidate axis is DOMINATED by candidates no selector ever picks"* | the S-T selection loss normalises over the **admissible** set. 0 params |
| **`cl − ol` as the S-W gate** | registry §1.14 / paper §6.3 [T1, 6,844 windows / 40 eps]: repaired arm `ha` **0.4246** vs `cl` **9.3697** (22×), `cl − ol` **+9.0039 [6.3659, 11.8487]** separated, and the divergence is **~99 % LONGITUDINAL** (LON 9.2655 vs LAT 0.7446) | the planner is **not** the fix for `cl − ol`; S-W must clear it first, and the planner's target family is **longitudinal** |
| **`ha` as the floor** | same block | ⛔ **no planner claim is admissible against `ol`.** The bar is `ha`, at T1 |
| **The four integration levels' *topology*** | `DIFFUSION_MPC_SYNTHESIS`: learned goal-conditioned proposal + explicit per-candidate cost | kept. **L2 (cost-guided denoising), L3 (receding-horizon warm start), L4 (amortisation) were never run** and stay the post-S-T ladder |
| **v1.6 / v1.7 (registry §§1.10–1.11)** | v1.6: accel MAE 1.824 → **0.5499**, jerk RMS 36.17 → **1.1334**, net-yaw −65 %, **WM-reliance 0.6233 PASS**, and **statistically indistinguishable from GT on all three distance-keeping metrics**. v1.7: ADE −16 % separated, but **P1 decel-response 0.1547 vs a 0.40 gate ❌ and P2 accel lag +0.173 s vs +0.15 ❌** | v1.6's loss set (pos-L1 + heading + net-yaw + p99 accel/jerk **barriers**) is the comfort/imitation prior. ⛔ **v1.7's refuted lever forbids "tune the speed-loss weight" as the longitudinal answer** — the pre-registered next lever is the **event-weighted near-term accel-matching** term, and it belongs in the tactical `SPEED_BAND`, not in a global weight |
| **v5.8f** | selected **0.4815 [0.3928, 0.5771]**, oracle **0.1077**, sel accel MAE **0.515 vs 8.10 (16×)**, violations ~0 — *"the entire deficit is SELECTION"*, over a **feasible** fan with 0.37 m of recoverable headroom | the quantitative target v6f's S-T must beat, and the proof that the remaining problem is selection |

### 2.3 The one open divergence in our own documents — now closed

`ALPAMAYO2_SUPER_ANALYSIS` §13 measured that `tanh` is not a safe saturating squash while
`V6_TRAINER_DESIGN` §1.1 imported the `tanh` form. **Resolved at HEAD `30d6d60`:**
`V6Config.emission_squash = "squash"` (`kinematic._squash` — identity in range, C¹ rational tail,
gradient alive at 100× the limit), after MEASURING that float32 `d/draw tanh(raw)` is **exactly 0.0
from raw ≥ 10** and that this run's own S-W history logged a **gnorm 354,076** spike. Free to set:
`emission.` is in the `planner` group, S-W does not train it, and an activation holds no parameters —
**no state_dict key or shape moves and a strict resume is unaffected.** Escalation #3 of the campaign
addendum is closed; this document records it so the addendum's open-items list can be updated.

---

## 3. ⭐ THE SELECTION DESIGN — the crux

### 3.1 The measurement that settles the mechanism (MEASURED today, 0 GPU)

**Instrument:** `stack/scripts/sel_winners_curse_law.py`.
**Surface:** `…/incoming/2026-08-03-esel-verdict/raw/fan_refined_refc-xl-30k.pt` — **881 windows,
40 episodes, 256 candidates, 4 waypoints at 0.5/1.0/1.5/2.0 s**, with GT, CV, `v0`, episode ids and
three scores, one of which (`cons_score`) is by its own provenance string a **world-model
roll-consistency** score: `-mean_sq(law_head([pooled, fan_i]) - encode_pooled(frame_{t+5}))`.
**Evidence class: MEASURED (ours) — no model, no GPU, no re-inference.**
**Class: EXPLORATORY** — a banked fan from a different model at a 2 s horizon. The *structural*
claims are the quotable ones; the absolute ADEs are this fan's.
**Instrument-parity proof:** this run reproduces the banked E-S1-0 dose-response independently —
shipped **0.4714** (E-S1-0: 0.4728) and refined **1.3901** (E-S1-0: 1.3100).

**Fan reference points:** oracle **0.1639** · fan **mean 13.9564** · shipped supervised selector
**0.4714 [0.3896, 0.5556]** (episode-cluster bootstrap) · CV **0.8377**.

#### A — the N-law. Selection ADE / **normalised argmax error-rank** / lower-tail p10

| N | oracle | fan mean | **supervised selector** | refined (unsup.) | **WM roll-consistency** | random |
|---|---|---|---|---|---|---|
| 4 | 4.606 | 14.02 | 5.365 / **0.099** / 0.77 | 5.146 / 0.086 / 0.79 | 7.640 / **0.241** / 0.57 | 14.082 / 0.503 / 0.24 |
| 8 | 2.516 | 13.85 | 3.397 / 0.080 / 0.65 | 3.202 / 0.077 / 0.65 | 6.447 / 0.256 / 0.39 | 13.818 / 0.500 / 0.12 |
| 16 | 1.472 | 13.94 | 2.281 / 0.062 / 0.78 | 2.289 / 0.070 / 0.73 | 6.205 / 0.268 / 0.37 | 14.024 / 0.499 / 0.13 |
| 32 | 0.811 | 13.95 | 1.414 / 0.042 / 0.83 | 1.723 / 0.064 / 0.70 | 6.160 / 0.276 / 0.29 | 13.767 / 0.497 / 0.10 |
| 64 | 0.470 | 13.94 | 0.940 / 0.029 / 0.92 | 1.473 / 0.060 / 0.74 | 6.252 / 0.280 / 0.27 | 13.979 / 0.501 / 0.09 |
| 128 | 0.273 | 13.96 | 0.645 / 0.020 / 0.97 | 1.371 / 0.060 / 0.80 | 6.382 / 0.283 / 0.28 | 13.980 / 0.501 / 0.10 |
| **256** | **0.164** | 13.96 | **0.471 / 0.014 / 0.99** | 1.390 / 0.063 / 0.82 | **6.450 / 0.286 / 0.28** | 14.172 / 0.505 / 0.10 |

*(0.0 rank = always the true best · 0.5 = a coin flip. p10 = P(chosen is in the window's true-best
decile); a random pick gives exactly 0.10, so the null needs no simulation. 8 seeded repeats per row.)*

⭐ **THE FINDING: the two scores move in OPPOSITE directions in N.** The supervised selector's rank
falls **0.099 → 0.014** and its lower-tail rises **0.77 → 0.99** — it has lower-tail dependence and it
**improves monotonically with a deeper fan**. The roll-consistency score's rank **rises 0.241 → 0.286**
and its lower-tail **collapses 0.57 → 0.28**, and its ADE is **flat at ~6.2–6.45 while the oracle
falls 28×** — it extracts a fixed, small amount of information no matter how much the fan offers.
**Replicated on an independently trained model** (REF-C-base, 104.2 M, 881 windows, 128 candidates):
supervised 0.082 → **0.021** rank / 0.62 → **0.98** p10; roll-consistency 0.243 → **0.283** / 0.41 →
**0.25**. ⚠️ `fan_deploy_refc-xl-30k.pt` reproduces the XL numbers exactly and is therefore counted as
the same surface, **not** a third replication.

⛔ **Two design instincts die here.** (1) *"Shrink the fan to 8 to avoid the winner's curse"* — at
N=8 the oracle is **2.516** against **0.164** at 256, i.e. shrinking pays **15× of headroom** for a
problem shrinking does not solve (roll-consistency's rank is still 0.256 at N=8). (2) *"ρ tells us
whether a cost can select"* — the roll-consistency score is separated **better than random**
(−8.2832 [−9.9050, −6.6000]) and still **+5.9787 [+5.3217, +6.7625] worse than the trained selector**,
both separated. Bulk information is real and useless at the extreme.

#### B — the goal-requirement curve, with both negative controls

Rule: `argmin_c ‖endpoint_c − ĝ‖`, with `ĝ = GT_endpoint + N(0, σ²I)`.
⛔ **EVIDENCE CLASS: a REQUIREMENT CURVE, not a capability number.** σ=0 is GT-derived and not
deployable; the deployable point on the same axis is the CV-goal row.

| goal 1σ (m) | N=8 | N=32 | **N=256** (ADE / rank / p10) |
|---|---|---|---|
| 0 | 2.660 | 0.840 | **0.171 / 0.001 / 1.00** |
| 0.25 | 2.662 | 0.846 | 0.212 / 0.003 / 1.00 |
| **0.5** | 2.670 | 0.872 | **0.309 / 0.008 / 1.00** |
| 1.0 | 2.715 | 0.961 | 0.556 / 0.019 / 1.00 |
| 2.0 | 2.820 | 1.312 | 1.073 / 0.045 / 0.91 |
| 4.0 | 3.189 | 2.106 | 2.162 / 0.102 / 0.55 |
| 8.0 | 4.289 | 3.493 | 3.841 / 0.187 / 0.29 |
| 16.0 | 6.139 | 6.115 | 6.598 / 0.302 / 0.16 |
| **CV goal** (deployable, 0 params) | 2.783 | 1.163 | **0.786 / 0.029 / 0.91** |
| ⛔ **goal-echo** (zero information) | 8.146 | 7.807 | **7.824 / 0.310 / 0.14** |
| *(reference)* oracle | 2.516 | 0.811 | 0.164 |
| *(reference)* supervised selector | 3.397 | 1.414 | 0.471 / 0.014 / 0.99 |

**Paired episode-cluster bootstrap, Δ = arm − supervised selector; positive = worse:**

| arm | Δ (m) | CI95 | separated |
|---|---|---|---|
| **goal σ = 0.5 m** | **−0.1591** | [−0.2300, −0.0894] | ✅ **BETTER** |
| goal σ = 1.0 m | +0.0943 | [+0.0241, +0.1650] | ✅ worse |
| goal σ = 2.0 m | +0.5943 | [+0.5172, +0.6701] | ✅ worse |
| CV goal | +0.3143 | [+0.1599, +0.4927] | ✅ worse |
| top-8 medoid (shipped score) | +0.1294 | [+0.0645, +0.2029] | ✅ worse |
| CV-goal prefilter(8) → score | +0.1532 | [+0.0632, +0.2632] | ✅ worse |
| **WM roll-consistency argmax** | **+5.9787** | [+5.3217, +6.7625] | ✅ worse |
| WM roll-consistency − random | −8.2832 | [−9.9050, −6.6000] | ✅ better than chance |

⭐ **THE ADMISSION THRESHOLD, MEASURED:** the crossover sits between σ 0.5 (better, separated) and
σ 1.0 (worse, separated) ⇒ **σ\* ≈ 0.8 m** on this fan. Expressed as ratios so it transfers:
**σ\* ≈ 1.7 × (incumbent selected ADE)** and **≈ 4.9 × (fan oracle)**. ⚠️ Both ratios must be
re-measured on the v6 fan at 6 s; the ≤2× extrapolation rule forbids carrying 0.8 m across a 3×
horizon change.
⭐ **The goal rule's rank IMPROVES with N at every σ ≤ 2** — it is in the supervised selector's family,
not the roll-cost's, because **a candidate-independent reference has no degenerate minimiser**.
⚠️ **At σ = 16 m the goal rule becomes indistinguishable from the roll-cost** (6.598 / rank 0.302 vs
6.450 / 0.286). That is the sharpest available characterisation of W7's cost: *it carries about as
much selection information as a goal point with ~16 m of error at 2 s.*

#### C — aggregators, on the full 256 fan

| rule | supervised selector | WM roll-consistency |
|---|---|---|
| argbest | **0.4714** | **6.4501** |
| top-2 / 3 / 4 medoid | 0.5498 / 0.4940 / 0.5245 | 6.4788 / 6.3020 / 6.2379 |
| top-8 / 16 / 32 medoid | 0.6008 / 0.7612 / 1.2659 | 6.1115 / 5.7285 / **5.4774** |
| top-8 / 32 centroid | 0.5763 / 1.2670 | 6.0978 / 5.4900 |
| top-8 / 32 **mean error** | 1.0340 / 2.5033 | 6.2660 / 6.0489 |
| top-8 / 32 **ceiling** | 0.2026 / **0.1713** | 4.2103 / 2.4617 |

⇒ **Aggregation helps a bad score slightly (−15 % on roll-consistency at m=32) and hurts a good one
(+0.13 m at m=8, separated).** It is a variance fix for an information problem. ⭐ **But the
`ceiling` row is the actionable one:** the supervised selector's **top-32 shortlist retains
0.1713 m against a 0.1639 m oracle — 96 %**. A two-stage design has real headroom; it is the *rule*
inside the shortlist that must change, not the shortlist.

#### D — composition (CV-goal prefilter → score, and the reverse)

| m | 2 | 3 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| goal → supervised | 0.7651 | 0.7538 | 0.7247 | 0.6247 | 0.5731 | 0.4963 |
| goal → **roll-consistency** | **0.7481** | **0.7468** | 0.7773 | 0.8827 | 1.2245 | 2.0081 |
| supervised → goal | 0.4920 | 0.5097 | 0.5260 | 0.5524 | 0.6120 | 0.7193 |
| roll-consistency → goal | 5.7978 | 5.4325 | 5.1087 | 4.3008 | 3.5496 | 2.7037 |

⇒ **The roll-cost's 6.4501 → 0.7481 inside a 2-candidate goal-admissible set is almost entirely the
goal's doing** (the goal alone scores 0.7862), and the honest statement is that the roll-cost adds
**≈0.04 m** there — real, small, and *only visible* once the ranking is restricted to admissible
candidates. That is the brief's *"restrict rank statistics to reachable candidates"* rule, measured.
⇒ **Neither ordering beats a good trained selector.** A weak goal cannot rescue a good scorer, and a
good scorer cannot be improved by a weak goal.

### 3.2 THE DESIGN, and why it is not "re-selecting on a noisier estimate of the same quantity"

| # | lever | param cost | what it changes |
|---|---|---|---|
| **SEL-1** | `GoalDistanceScorer` — `score_i = −‖endpoint_i − ĝ‖/τ + b_i`, `ĝ = W·e_g_tac + c` | **+267** (MEASURED, test-pinned) | **the ESTIMAND**: a candidate-**independent** reference replaces a candidate-conditional self-consistency |
| **SEL-2** | `w_select` with the **`softade`** objective (expected fan error under the score's own softmax), normalised over the admissible set | **0** | **the ESTIMATOR**: trained at the extreme, metric-aware, hard optimum — not fitted to the bulk |
| **SEL-3** | `plan_wta_eps` — ε-relaxed winner-take-all | **0** | **the CANDIDATE DISTRIBUTION**: bounds the fan mean, which pure WTA does not |
| **SEL-4** | admissibility prefilter (reachability + feasibility), scoring restricted to survivors | **0** | **the SET**: rank statistics over reachable candidates only |
| **SEL-5** | `n_candidates` as a declared ARM (8 vs 32) | **+6,144** (8→32) **+24** on `cand_bias` | **the SIZE**: measured, not assumed |
| **SEL-6** | `selector="mlp"` — the **capacity control** | **+41,089** (arithmetic; **not implemented**) | makes a SEL-1 win attributable to mechanism rather than capacity |

**Why this is a different quantity, not a noisier estimate of the same one — four independent reasons:**

1. **The estimand has a different failure mode.** A self-consistency cost is minimised by a
   near-stationary candidate; *inaction cannot minimise a distance to a goal.* The paper states the
   same thing about the loops we copied: *"V-JEPA 2-AC, DINO-WM minimise distance to a **goal**, which
   inaction cannot minimise, and we had dropped that term."* MEASURED consequence: the goal rule's
   normalised rank **falls** with N at every σ ≤ 2; the roll-cost's **rises**.
2. **The estimator optimises the statistic that governs selection.** `softade`'s gradient concentrates
   on the low-error candidate under the score's own softmax — a **lower-tail** objective. A
   regress-the-error scorer (W4b/W4c, learned scorers ρ 0.05–0.26) optimises the **bulk**, which is
   exactly the statistic the paper says does not govern argmax. E-OBJ-1 measured both halves of the
   decomposition: metric-awareness **−0.0974 / −0.1670 separated better**, target-softness **+0.0909
   separated worse**.
3. **The endpoint is different, and it is pre-committed.** The primary endpoint is the **normalised
   error-rank of the selected candidate** and the **lower-tail hit rate**, not ρ and not ADE alone —
   W7-PROG's precedent (*"the mechanism under test is a ranking claim"*). The trainer logs
   `sel_norm_err_rank` every step.
4. **It has a measured admission threshold that can refuse it before a GPU-hour is spent.** σ\* ≈ 1.7×
   the incumbent selected ADE. If the goal head cannot reach that, SEL-1 is refused — *before* S-T,
   on banked data.

⛔ **And what the design explicitly does NOT do:** it does not re-rank by roll-consistency; it does
not aggregate top-m as its answer; it does not enlarge the anti-degeneracy weight; it does not shrink
the fan to hide the problem; and it does not replace a trained selector with a hand-written rule —
today's measurement says every one of those loses, separated.

### 3.3 Where the roll-cost and the kinematic cost still live

Sequenced **behind** SEL-1/2, never in front of it, and each with a pre-registered zero-weight
ablation (`MPC_WM_DESIGN`'s attribution discipline — *"a planner that improved with ten live terms is
unattributable"*):

```
fan (N candidates, feasible by construction)
  └─ SEL-4  admissibility: reachability ∧ kinematic feasibility        [0 params]
       └─ SEL-1/2  goal-conditioned trained score → top-m shortlist    [+267]
            └─ (post-S-T) per-candidate WM roll, cost = w_r·roll + w_k·kin   [0 new params]
                 └─ emit                                                 argmax, NOT argmin-over-256
```

The roll term is admitted only if it improves the shortlist decision **inside** the admissible set —
the regime where today's measurement shows it contributes ≈0.04 m — and it is scored by the same
rank endpoint. Its calibration on the repaired trunk (**ρ 0.7164 [0.5847, 0.7696]**) is why it is
kept at all.

---

## 4. THE STAGED IMPLEMENTATION PLAN — S-T / S-S / S-J

⛔ **S-W is untouched.** The planner group is frozen there, `STAGE_LAMBDA_PLAN["S-W"] = 0.0`, the
trainer refuses a non-zero `--lambda-plan` in S-W, and **preflight now also refuses `--selector` in
S-W** — because building a scorer there would create untrainable dead weight *and* change the
state_dict, breaking the strict resume of the live run. The S-W gate stays **`cl − ol` stability under
the model's own actions**, with `ha` as the floor (registry §1.14).

### 4.1 S-T — the planner stage (10,000 steps, `--init-from` the S-W ckpt, `--max-horizon 60`)

**Trains:** `layer_tac` (adapter, `P_T`, `goal_head_tac`, factored LAT/LON heads, `vocab_tac`,
`vocab_a_lat/lon`) **+ `planner`** (`cond_op`, `plan_proj`, `cand_queries`, `emission`,
`predictor_op.intent_proj/intent_gate`, **and `cand_score`**). Everything below frozen — Drive-JEPA's
shape, independently re-derived by our own consumer-invalidation result.

**Launch (the deltas against `V6_TRAINER_DESIGN` §3.2):**

```
--stage S-T --init-from <S-W ckpt> --prev-gate <S-W stage_gate.json> --max-horizon 60 \
--selector goal --selector-tau-m 1.0 --w-select <declared> --plan-wta-eps 0.05 \
--n-candidates 8        # ARM A ; ARM B is --n-candidates 32
```

**Parameter cost, MEASURED on this box (`V6Config()` base 87,893,449):**

| lever | delta | note |
|---|---|---|
| `--selector goal` | **+267** | `goal_point` Linear(128→2) 258 · `cand_bias`[8] 8 · `log_tau` 1 |
| `--plan-wta-eps` | **0** | a loss term |
| `--w-select` | **0** | a loss term |
| `--n-candidates 32` | **+6,144** | `cand_queries` Embedding(N, 256) |
| both, at N=32 | **+6,435** | |
| *(control)* `selector="mlp"` | **+41,089** | **154× the goal head** — the point of the control |

⇒ within the programme's accepted band (**+897 / +385 / +128 / 0**), and **1/1,019** of the
**+272,001** an earlier two-MLP tactical head cost before its own capacity control caught it.

**S-T GATE — pre-registered, three-sided, all outcomes committed here.**
Primary endpoint is a **ranking** claim; ADE is secondary and reported per family.

| | PRIMARY (ranking) | SECONDARY (ADE) | four families |
|---|---|---|---|
| **BETTER** | `sel_norm_err_rank` ≤ **0.10** at the run's N **and** lower-tail p10 ≥ **0.5**, both on val at **T1** | `sel_gap ≤ 0.5 ×` fan oracle at T1 (the standing S-T bar) | LON speed MAE and accel MAE **not worse** than v5.8f's, paired-separated |
| **WORSE** | rank ≥ **0.25** *or* p10 ≤ **0.15** (i.e. in the roll-cost's family) | `sel_gap > 1.0 ×` fan oracle | any family CI-worse |
| **NOT SEPARATED** | the interval spans the bar | `sel_gap` CI spans 0.5× oracle | — |

**Committed consequences.** ⭐ **BETTER** ⇒ SEL-1/2 are the v6f selector; the post-S-T ladder (L2
cost-guided denoising → L3 warm start → L4 amortisation) opens, in that order.
⛔ **WORSE** ⇒ SEL-1 is **refuted as posed**; the fallback is *not* another cost — it is the
**supervision** branch: `ANCHOR_GOAL` labels from the PH0/`obstacle.offline` line, because a
goal-distance selector whose goal is unsupervised is being asked to invent its own reference.
⚠️ **NOT SEPARATED** ⇒ **inconclusive is not a pass** (`pass: null`); S-S is refused without
`--allow-inconclusive-gate` **and** a stamped `--gate-off-reason`, and the first re-run is the
**capacity control** (`selector="mlp"`), not a weight sweep.

**Pre-registered controls, none of them defaults:** `--w-select 0` with the scorer built (inert-head
control, requires `--i-know-this-is-the-control-arm`) · `--plan-wta-eps 0` (pure WTA) ·
`--n-candidates 8|32` · goal-echo at eval (goal ← corpus marginal) · `selector="mlp"` (capacity).

### 4.2 S-S — the strategic stage (8,000 steps)

**Trains `layer_str` only**; λ_plan 0, `w_select` 0, the planner **frozen**.
⚠️ **S2 (`g_str` supervision) is not wired and must not be faked.** Until PH0→PH1→PH2 lands, S-S
trains S1 (strategic latent prediction) only and the **STRATEGIC family is reported `n/a` with its
reason and its n** — never silently dropped.
**Parameter cost of this design in S-S: 0.**

### 4.3 ⚠️ THE CONSUMER-INVALIDATION RISK INSIDE THE LADDER — new, unmeasured, and it needs a gate

S-S retrains `goal_head_str` ⇒ `e_g_str` moves ⇒ `goal_head_tac(z_tac_p, cond=e_g_str)` moves ⇒
`g_tac` moves ⇒ **`ĝ` moves under a selector that was frozen at the end of S-T**. That is exactly
registry §1.14's mechanism — *"you cannot repair a trunk and keep its planner"* — one level up, and
the ladder has **no gate for it today**.

Partial structural protection already exists: **one vocabulary, two views** (the goal table is the
same `nn.Module` object in the emitter and the consumer, pinned by `is` identity), so a change in
*which token* is emitted is a change in a meaning the consumer already shares. It does **not** protect
the continuous args, and `ĝ` is decoded from the arg-bearing embedding.

**Required, and cheap:** add `sel_gap`, `sel_norm_err_rank` and `sel_ade` to S-S's **reported-only**
probes with a pre-registered **no-harm bar** (`sel_gap` not CI-worse than at the end of S-T, paired on
the same windows). If it degrades, the committed response is a **planner refit micro-stage (S-S′)**,
not a silent carry-forward. **0 parameters.**

### 4.4 S-J — optional joint polish (3,000 steps, lr 3e-5, isolation ON)

Run **only if S-T/S-S plateau**. Gate: the frozen battery **FLAT** across the joint phase (H-COTRAIN),
zero live forbidden X3 edges. **Parameter cost: 0** — S-J trains what already exists.

### 4.5 What the tactical layer must actually own, per family

| family | tactical instrument | why it is the tactical layer's job |
|---|---|---|
| **LONGITUDINAL** | `SPEED_BAND(v_lo, v_hi)` as a **selection term** and a reported envelope; `GAP_TARGET(agent_slot, time_gap_s)` once agent slots exist | PI decision 2026-08-11: target speed is a tactical responsibility, bounded above by strategic `REDUCE_TO`. **~99 % of the T1 divergence is longitudinal**, so this is the load-bearing port |
| **LATERAL** | `CORRIDOR_OFFSET`, `NUDGE_L/R` — severity in continuous args | the A2 comparison measured that our declarations lose exactly the fine lateral information a severity axis carries |
| **TACTICAL** | factored LAT × LON `a_tac` (6 × 6), already built | retires the 5-way mixed softmax **as representable**, not merely as a defect |
| **STRATEGIC** | `n/a` with reason + n until S2 lands | no map, no lane graph, no route label — settled at five probes |

⛔ **`GAP_TARGET`, `YIELD_AT`, `STOP_POINT`, `WAIT_FOR_ONCOMING`, `EVADE_IN_CORRIDOR` need agent
slots PH0 does not extract**, and that is also why the binding LONGITUDINAL family's distance-keeping
cannot be computed. **Wiring `obstacle.offline` (87,481 cuboids, 10 dynamic classes, 97.44 % of the
corpus; we read 4 of 36 features) is the highest-value 0-GPU prerequisite of this design.**

---

## 5. THE CHEAPEST DISCRIMINATING EXPERIMENT

### 5.1 E-WC — **already run, this turn, 0 GPU** (§3.1)

It discriminated the mechanism before any v6 GPU-hour: the winner's curse is a property of the score,
top-m aggregation is refuted as the remedy, and the goal rule's admission threshold is measured.
Cost: ~1 minute of CPU on a fan already in the repo. Reproduce with:

```
python stack/scripts/sel_winners_curse_law.py \
  --fan "TanitAD Research Hub/.../2026-08-03-esel-verdict/raw/fan_refined_refc-xl-30k.pt" \
  --out .../ewc_law_refc-xl.json
```

### 5.2 E-WC2 — the next 0-GPU step, **before S-T is launched**

**Question:** can the *tactical goal head* reach σ\* on our corpus at all?
**Method:** fit the smallest admissible predictor of the 6 s endpoint from **frozen S-W latents only**
(a ridge, the P1/P2 battery's method) on the 40 val episodes, LOEO; report its 1σ endpoint error at
2 s **and** 6 s beside the fan oracle and the incumbent selected ADE, i.e. as the two **ratios** σ/ADE
and σ/oracle so it composes with §3.1's curve.
**Cost:** 0 GPU, banked latents.
**Committed outcomes:** σ/ADE ≤ 1.7 ⇒ SEL-1 is funded and S-T launches with it. σ/ADE ≥ 3.0 ⇒ **SEL-1
is refused before launch**, and the work moves to `ANCHOR_GOAL` supervision (PH0 + `obstacle.offline`).
In between ⇒ inconclusive; run the capacity control first.

### 5.3 What would REFUTE this design

| observation | refutes |
|---|---|
| S-T's `sel_norm_err_rank` ≥ 0.25 or p10 ≤ 0.15 at T1 | SEL-1/2 — the goal-conditioned score is in the roll-cost's family after all |
| the **goal-echo** arm scores within CI of the live goal | the goal carries no per-window information; the "hierarchy" is a per-candidate bias |
| `selector="mlp"` (+41,089) matches or beats `"goal"` (+267) | the win is **capacity**, not mechanism — SEL-1's story is wrong |
| ε-relaxed WTA does not reduce the fan mean, or reduces the oracle | the fan-distribution account (SEL-3) is wrong; the diversity guard fires |
| N=32 does not beat N=8 on selected ADE at equal rank | the N-law does not transfer from REF-C's fan to a learned 8-query fan — a real possibility, which is why N is an arm |
| S-T improves selected ADE while `cl − ol` does not improve | the planner is papering over an unstable world model; **stop and fix S-W** |
| a σ\* re-measured at 6 s exceeds 3× the 2 s value | the ratio form does not transfer; the threshold must be re-derived, not scaled |

---

## 6. THE FOUR FAMILIES — how every claim in this design is reported

⛔ **Per family, never pooled. ADE alone is an incomplete result.** Every number carries its
**T-tier** and the **paired episode-cluster bootstrap**; `overlapping_holdout_se` is used nowhere.
Every family is reported at **both 0–2 s and 0–6 s** (§4b's eval consequence).

| family | what the planner reports | instrument | status |
|---|---|---|---|
| **LONGITUDINAL** | target-speed accuracy, speed MAE/bias, **accel MAE**, and **distance-keeping** (headway / time-gap / TTC) | `taniteval.four_families`, `lead_metrics` | ⛔ distance-keeping **blocked on agent slots** — a WORK ITEM, not a pass. Report `n/a` **with reason and n** |
| **LATERAL** | heading, **curvature**, **yaw-rate**, cross-track | `taniteval.lateral` | available |
| **TACTICAL** | selected vs executed manoeuvre, confusion over LAT × LON, **goal/anchor selection** (`sel_norm_err_rank`, p10, `sel_gap`) | `taniteval.hierarchy`, `selgap` | available. ⚠️ re-read at `DIR_YAW_RAD` **0.10**, not 0.15 — the A2 correction showed 0.15 is ~6.5× the typical 2 s turn and it touches every published manoeuvre-coherence κ |
| **STRATEGIC** | strategic decision + route/goal quality | — | **`n/a` with reason and n** until S2 lands. Settled: PhysicalAI-AV ships no map, lane graph, junction annotation or route signal |

⚠️ **The tier rule bites hardest here.** T0 and T1 differed by **25×** on the same checkpoint and the
same windows (0.3659 vs 9.3697). **No planner claim is admissible at T0**, and `ha` (0.4246) — not
`ol` — is the floor any closed-loop claim must clear.

---

## 7. WHAT IS IMPLEMENTED TODAY — gated, default-off, tested, NOT trained

| file | change | default |
|---|---|---|
| `stack/tanitad/models/v6.py` | `GoalDistanceScorer` (+267); `V6Config.selector` / `selector_tau_m` / `plan_wta_eps`; `cand_score.` → the **`planner`** group; `sel_*` added to `emit()`'s output and to the **declared `planner_side`** surface | `selector="none"`, `plan_wta_eps=0.0` — **nothing is built** |
| `stack/scripts/train_v6_staged.py` | `V6LossWeights.w_select` (the `softade` selection loss); ε-relaxed WTA in the plan term; `fan_mean_ade` / `fan_oracle_ade` / `sel_ade` / **`sel_norm_err_rank`** / `sel_gap` logging; CLI `--selector`, `--selector-tau-m`, `--plan-wta-eps`, `--w-select`; four new preflight refusals | `w_select=0.0` — **the term is not constructed** |
| `stack/scripts/sel_winners_curse_law.py` | **NEW** — the E-WC instrument (§3.1), 0 GPU | n/a |
| `stack/tests/test_v6_selector.py` | **NEW** — 10 tests | all pass |

**⭐ BYTE-IDENTITY PROOF (MEASURED, this box, 2026-08-15).** `torch.manual_seed(0); V6Stack(V6Config())`:

```
HEAD 30d6d60  state_dict md5 a012aad286309d3283f8e055b75bfb32 · 87,893,449 params · 405 keys
after change  state_dict md5 a012aad286309d3283f8e055b75bfb32 · 87,893,449 params · 405 keys
keys identical: True | every tensor torch.equal: True
```

The `"none"` path constructs no module and therefore **draws no random numbers** — `cand_score` is
built at the very end of `__init__` and only when asked, so no earlier module's initialisation moves.
⇒ **the live S-W resume is unaffected**, which is the property that actually matters tonight.

**Param delta, MEASURED:** `selector="goal"` **+267** · `n_candidates 32` **+6,144** · both **+6,435**
· `plan_wta_eps` / `w_select` **0**. `test_param_delta_is_exactly_267` additionally asserts that
flipping the flag perturbs **no pre-existing tensor** — that is what makes it a capacity control and
not merely a count.

**Test evidence:** `tests/test_v6_selector.py` **10 passed**; the v6 set
(`test_v6_staged` + `test_v6_selector` + `test_v6_ckpt_layout_compat` + `test_v6_probe_trunk`)
**113 passed**; **full suite `PYTHONUTF8=1 python -m pytest -q` → 2830 passed, 17 skipped, 2 xfailed
(311 s)**.

⛔ **Nothing was trained, nothing was launched, and Thor was not touched.**

---

## Deliverable manifest

| artifact | where it lives | state |
|---|---|---|
| `Project Steering/V6F_PLANNER_DESIGN.md` (this document) | repo working tree + index | **staged** |
| `stack/tanitad/models/v6.py` (gated selector) | repo | **staged** |
| `stack/scripts/train_v6_staged.py` (gated losses + CLI + preflight) | repo | **staged** |
| `stack/scripts/sel_winners_curse_law.py` (the E-WC instrument) | repo | **staged** |
| `stack/tests/test_v6_selector.py` (10 tests) | repo | **staged** |
| `…/incoming/2026-08-15-v6f-planner-design/raw/ewc_law_refc-xl.json` | repo | **staged** |
| `…/incoming/2026-08-15-v6f-planner-design/raw/ewc_law_fan_refined_refc-base-30k.json` | repo | **staged** |
| `…/incoming/2026-08-15-v6f-planner-design/raw/ewc_law_fan_deploy_refc-xl-30k.json` | repo | **staged** |
| HEAD reference `state_dict` (byte-identity proof input) | scratchpad only | ⚠️ **one place** — the md5 and counts are banked in §7 and the build is reproducible in ~20 s |

**Nothing was committed and nothing was pushed** (`AGENT_OPERATING_STANDARD` rule 1).

## Escalations — requests, not notes in a README

1. ⭐ **The S-S consumer-invalidation gate (§4.3) does not exist and needs to be added before S-S
   runs.** It is 0 parameters and 0 GPU, and without it a strategic-stage regression in selection
   would be discovered as a mystery three stages later. **This is the highest-priority item in this
   document.**
2. ⭐ **`obstacle.offline` → agent slots is the prerequisite for half of the binding LONGITUDINAL
   family and five of the nine `g_tac` tokens.** 0 GPU, no PI decision. Until it lands, the family
   that owns ~99 % of the T1 divergence cannot be scored, and the tactical layer cannot emit the goals
   this design gives it authority through.
3. ⚠️ **E-WC2 (§5.2) should run before S-T is launched** — it can refuse SEL-1 for 0 GPU, and refusing
   before a launch is 10 k steps cheaper than refusing after one.
4. ⚠️ **`selector="mlp"` (+41,089) must be implemented before a `"goal"` arm is judged.** `V6Config`
   refuses it today with that message rather than allowing an unattributable arm.
5. ⚠️ **The W7 per-window arrays are lost with pod4/pod5.** Any future fan dump should be banked in
   the repo at emission time, not after the verdict — the same rule that made today's re-analysis
   possible at all.
6. ⚠️ **`DIR_YAW_RAD` 0.15 → 0.10 re-read is still owed** and it gates the TACTICAL family's history.
   0 GPU.
7. ✅ **Campaign-addendum escalation #3 (`tanh` vs softsign) is CLOSED** at HEAD `30d6d60` —
   `emission_squash="squash"`. The addendum's open-items list should be updated.
