# `ANCHOR_GOAL` supervision — what it would take, and the 0-GPU test that splits it in two

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **HEAD at start `6c27b38`**
**Cost: 0 GPU. Thor was not contacted — no ssh, no disk read, no job.**
**Pre-registration:** `stack/scripts/e_ag1_anchor_floor.py` `PREREG` — **staged before the first run**
(index blob `dc31534`), in the shape `e_wc2_sigma_star.py` used to refuse SEL-1 this morning.
**Tier:** ⛔ **T0-DIAGNOSTIC** — a geometry/representation-capacity probe on banked latents.
**No T1 capability claim may cite any number here.**

---

## 0. The one-line answer

> ⭐ **The `ANCHOR_GOAL` FORMULATION clears the admission bar. The SURFACE does not.**
> The branch is therefore **not one decision but two**, and they are now separately measured:
>
> | | MEASURED | verdict |
> |---|---|---|
> | **the formulation** — the quantisation floor of the **vocabulary the programme already ships** (256 FPS anchors, 200,000-window pool, parity corpus `physicalai-train-e438721ae894`) | **σ(2 s) = 0.5637 m** per-axis, CI **[0.5085, 0.6185]**, **σ/ADE 1.196** | ⭐ **inside §5.2's FUNDED band (≤ 1.7)** — a *perfect* anchor classifier would be funded |
> | **the surface** — a K-way anchor classifier on the **same** frozen REF-C latents, **same** LOEO folds | top-1 **0.1101** at K=256; **σ = 9.4868 m**, **+4.7502 [+3.0514, +6.3981] separated WORSE** than the free ridge E-WC2 already refused | ⛔ **WORSE_OR_FLAT — the pre-registered refusal** |
>
> ⇒ **`ANCHOR_GOAL` supervision is NECESSARY AND NOT SUFFICIENT.** Building the labels does not
> fund the goal head. On the only surface we can measure today, **discretising into anchors is
> 2.0× WORSE than the free regression that was already refused**, separated at every K (8→256)
> and under both vocabulary constructions (FPS and k-means), and it **replicates on REF-C-base**
> (Δ **+5.4570 [+3.8345, +7.1073]**). ⭐ **Therefore: do not spend the label pipeline before the
> ~10–25 GPU-min S-W latent dump**, which is the one input that can move the half that is failing.

### ⭐⭐ Three further results, each of which changes an action

**(a) THE FAILURE IS THE ESTIMATOR, NOT THE ESTIMAND — and the fix is already measured elsewhere.**
The instrument's `snap` arm holds the *estimator* fixed (the same ridge) and changes only the
*estimand* (free point → nearest anchor). It is **NOT separated from the ridge**: Δ **−0.0002
[−0.1031, +0.0703]** at K=256, and **+0.0383 [−0.2125, +0.2338]** even in the good regime of (c).
⇒ **discretisation per se costs nothing; direct K-way classification costs +4.75 m.** The one-hot
target is exactly the metric-BLIND objective **E-OBJ-1 already measured as inferior** (one-hot CE →
`softade` recovers −0.0974 / −0.1670 m separated, and the recovery is **LONGITUDINAL**). ⇒
**`ANCHOR_GOAL` must be supervised by a metric-aware, distance-weighted target over anchors — or
implemented as regress-then-snap — never by a one-hot `anchor_id` CE.**

**(b) ONE SCALAR — `v0` — IS WORTH 2.85× ON THE WHOLE PROBLEM, AND IT IS THE CONTESTED ONE.**
Appending ego speed to the same 1088-dim vision block (1089 dims) moves the ridge
**4.7367 → 1.6626 m** per-axis at 2 s. The decomposition is exact and one-sided:
**σ_long 6.6132 → 2.0917 (3.16×)** while **σ_lat 1.0667 → 1.0739 (unchanged, +0.7 %)**.
⛔ `v0` is classed **ECHO / inadmissible** by `e_wc2_sigma_star.py:188` and **✅ admissible** by
`V6F_PLANNER_DESIGN.md:169`. **Both are HEAD.** This is now the single highest-value open
decision in the branch — §8.

**(c) AND THE SOBERING HALF: even vision + `v0` loses to a 0-parameter kinematic rule.**
1.6626 (σ/ADE **3.527**, still REFUSED) vs constant-yaw-rate's **1.1888** (σ/ADE 2.52), which
uses `v0` **and** yaw-rate and no learning at all. ⇒ **on the frozen REF-C surface, vision adds
essentially nothing to the 2 s goal beyond what ego kinematics already supply.**

⭐ **AND THE STRUCTURAL FINDING THAT SHOULD DRIVE THE DESIGN:** the 2 s goal point's corpus
variance is **98.8 % LONGITUDINAL** (goal-echo null: σ_long **19.0578** vs σ_lat **2.0723**), and
every arm's residual is longitudinal too — the anchor classifier reaches σ_lat **1.3310** against
a floor of 0.6802 while its σ_long is **13.3502** against a floor of 0.8954.
**A single K-way `anchor_id` mixes the lateral and longitudinal axes into one categorical
decision — which is the 5-way manoeuvre softmax defect, surviving one level up in the goal
vocabulary after `a_tac` was factored to retire it.** §6.4.

---

## 1. What an `ANCHOR_GOAL(anchor_id, t_reach_s)` label must contain

**Established from source, not from a summary.**

| # | field | type | where the definition lives | notes |
|---|---|---|---|---|
| 1 | `anchor_id` | **categorical**, `∈ [0, K)` | `HIERARCHY_VOCABULARY.md:84` — *"anchor_id ∈ fan vocab, t_reach_s"* | "fan vocab" = the **FPS anchor vocabulary** built by `stack/scripts/build_refc_anchors.py` (`DIFFUSION_PLANNER_COMPARISON.md:64`), **not** the learned `cand_queries` embedding, which carries no geometry |
| 2 | `t_reach_s` | continuous, s | same | the tactical band is **2–6 s** (`v6.py:121` `TAC_BAND_S`), so a well-formed label has `t_reach_s ∈ (2.0, 6.0]` |
| 3 | the **vocabulary itself** | `[K, S, 2]` frozen buffer + `horizons` + provenance | `build_refc_anchors.py:75-78` (`method`, `horizons`, `n_anchors`, `pool_size`, `source`, `seed`) | it must ship **with the checkpoint**: `anchor_id` is meaningless without the exact table, and a rebuilt table silently re-labels the whole corpus |
| 4 | `arg_mask` | 8 bools | `v6.py:163-171`, `GoalVocabulary.encode` | *"Unset = unconstrained"*; a slot no label can fill must not be regressed against a fabricated 0 |
| 5 | validity + **reason** | bool + str | — | windows whose `t_reach` runs past the episode end are **excluded with n**, never imputed (E-WC2's 681-of-881 discipline) |
| 6 | frame convention | ego frame at the window origin, **x forward, y left**, metres | `lead_source.py:38-42`, `driving_diagnostic._ego` | one convention per programme; two is a retraction waiting to happen |
| 7 | label provenance | corpus + parity key + skip-hash | `CLAUDE.md` invariants | anything that re-selects episodes breaks cross-arm comparability and must be refused |

⛔ **AND THE DEFECT THAT BLOCKS FIELD 1 IN CODE TODAY — new, MEASURED from source.**
`GOAL_ARG_NAMES` (`v6.py:168-171`) is eight slots, and `v6.py:166-167` states the type
verbatim: *"Args are **PHYSICAL UNITS** (m, s, m/s)"*. Both ends are continuous —
`GoalHead.arg_head = nn.Linear(hidden, vocab.n_args)` (`v6.py:574`) emits 8 floats, and
`GoalVocabulary.arg_proj = nn.Linear(self.n_args, self.d_embed)` (`v6.py:508`) consumes 8 floats.
**`anchor_id` is a categorical index into a table, not a physical quantity**, and regressing it
is a type error: with an FPS-ordered vocabulary, anchor 5 is not "between" anchors 4 and 6 in any
geometry. ⇒ **the implemented `GoalHead` cannot emit a well-formed `ANCHOR_GOAL`, even given
perfect labels.** §2.3 shows this is not specific to `ANCHOR_GOAL` — it hits **seven of the nine
tokens**.

---

## 2. THE SUPERVISION GAP — established by measurement, two probes each

### 2.1 ✅ CLAIM A VERIFIED — **PH0 emits none of the nine `g_tac` tokens**

| probe | what was read | result |
|---|---|---|
| **1 — the schemas** | `stack/scripts/ph0_v2.py:39-43` and the schema block `S_B1`/`S_B2`/`S_B3`/`S_B4` (`:55, :72, :95, :111`) | `GOAL_KINDS` is an **11-token, term-for-term lowercase of `STRATEGIC_GOAL_TOKENS`**; `ACTION_VERBS` is `STRATEGIC_ACTION_TOKENS`. There is **no tactical goal enum and no B5 call** — the schema list ends at B4 |
| **2 — an unbounded token grep** | every one of the nine names, repo-wide, tracked **and** untracked, no `-maxdepth` | non-prose occurrences exist **only** in `stack/tanitad/models/v6.py` (the vocabulary itself), its tests, and `stack/scripts/ph0_sam3.py` — where all four hits are inside **docstrings** (`:14-19`, `:198-199`) naming the gap, not code emitting a token |

⚠️ **And a C70b-class false positive, named so it is not re-discovered:** `SPEED_BAND` appears to
have three implementations in `taniteval`. It does not. Those are
`four_families.TARGET_SPEED_BANDS_MPS` (eval **tolerance** bands, `:56`) and
`lead_metrics.SPEED_BANDS` (speed **strata** for lead metrics, `:58`) — a substring collision with
the `g_tac` token `SPEED_BAND(v_lo, v_hi)`, which is not emitted anywhere. *(This is exactly the
class that mis-commissioned an agent yesterday: a name that makes an unbuilt thing look built.)*

### 2.2 Per-token: what exists **TODAY**

| `g_tac` token | args (`HIERARCHY_VOCABULARY.md:84-92`) | emitted by PH0? | the pieces that DO exist today | what is missing |
|---|---|---|---|---|
| **`ANCHOR_GOAL`** | `anchor_id ∈ fan vocab`, `t_reach_s` | ⛔ no | ⭐ **the vocabulary is BUILT and banked** — `refc_anchors_full_REBUILD.pt` (256 FPS anchors, `pool_size 200000`, `source .../physicalai-train-e438721ae894`), plus `refc_anchor_vocab.pt` (128/256) and `flagship_v4_anchors_dense.pt` (256×20). ⭐ **the assignment is one `argmin`** over ego-frame endpoints, which `ph0_pilot.engine_a_summary` already computes (`polyline_xy`, `:363`) | (a) a **6 s** vocabulary — every shipped one stops at **20 steps = 2.0 s**; (b) the categorical arg slot (§1); (c) the label emitter itself |
| `CORRIDOR_OFFSET` | `lat_offset_m`, `arc_m` | ⛔ no | Engine A's `lane_change_events[].lat_m` + `arc_from_t0_m` (`ph0_pilot.py:345-349`) — **both continuous, both already computed** | assembly only. ⭐ **the cheapest of the nine** |
| `SPEED_BAND` | `v_lo`, `v_hi` | ⛔ no | B2 sign `kind=speed` + verbatim `text` (OCR); Engine A `speed_profile.{v_t0,v_min_future,v_max_future,net_dv}` (`ph0_pilot.py:375-381`) | the **combination** — never built. Both halves are continuous, so no type gap |
| `GAP_TARGET` | **`agent_slot_id`**, `time_gap_s` | ⛔ no | ⏹ `obstacle.offline` **is wired**: `select_lead_causal` returns a **`track_id`** (`lead_source.py:239`) and `gap0_m`, measured `n_ok 40/40` on val40 | a **stable slot vocabulary** (a `track_id` is clip-local) + the categorical arg slot |
| `YIELD_AT` | `position_arc_m`, **`gap_slot`** | ⛔ no | B2 has a `yield` sign kind; `obstacle.offline` gives the agent | **no arc position for any perceived object** (§2.4) |
| `STOP_POINT` | `position_arc_m`, **`reason ∈ {sign,light,queue,hazard}`** | ⛔ no | Engine A `lonmode` gives `stop_dist_m` (`ph0_pilot.py:354`) | the **reason** is a 4-way categorical that nothing extracts, and the arg slot cannot hold it |
| `WAIT_FOR_ONCOMING` | `narrow_arc_m`, **`oncoming_slot`** | ⛔ no | `obstacle.offline` has the agents; oncoming-ness is derivable from their tracks | corridor-narrowing geometry: **nothing at all** |
| `EVADE_IN_CORRIDOR` | `lat_offset_m`, **`obstacle_slot`**, `past_arc_m` | ⛔ no | `obstacle.offline` includes `person, rider, stroller, animal, protruding_object` — exactly the evade classes | the corridor frame + the slot vocabulary |
| `TRAFFIC_LIGHT_REACT` | **`light_slot_id`**, `state`, **`stopline_arc_m`** | ⛔ no | B2 gives `state ∈ {red,amber,green,none}` and a B3 box | ⛔ **`obstacle.offline` CANNOT supply the light slot — ever** (§3). No stopline range |

### 2.3 ⭐ NEW — the **arg-TYPE** gap, which no document has stated

Counting the arg types in the table above: **seven of the nine tokens carry at least one
CATEGORICAL arg** (`anchor_id`, `agent_slot_id`, `gap_slot`, `reason`, `oncoming_slot`,
`obstacle_slot`, `light_slot_id`, `state`). Only `CORRIDOR_OFFSET` and `SPEED_BAND` are purely
continuous. The implemented head and conditioner are **continuous-only on both sides** (§1).

⇒ **Even with perfect labels and perfect agent slots, `goal_head_tac` can express exactly 2 of
the 9 tokens.** This is a code gap, not a data gap, and it is **0 GPU** to close.

⚠️ **A second representational limit, from the same read.** `GoalHead.forward` returns one
`logits.softmax(dim=-1)` over the nine tokens (`v6.py:590-592`) and `GoalConditioner.forward`
takes **one** id-or-prob vector (`v6.py:612-614`). The layer therefore emits **one goal token per
step**. §6.4's measurement says the goal must be **`ANCHOR_GOAL` (lateral) ∧ `SPEED_BAND`
(longitudinal)** simultaneously — a *pair*, which a single 9-way softmax cannot represent.

### 2.4 ✅ CLAIM B VERIFIED — and **refined**: five need slots, `obstacle.offline` supplies four

Read from `HIERARCHY_VOCABULARY.md:86, 88, 90, 91, 92`, the five tokens whose args name a slot are
`GAP_TARGET(agent_slot_id)`, `YIELD_AT(gap_slot)`, `WAIT_FOR_ONCOMING(oncoming_slot)`,
`EVADE_IN_CORRIDOR(obstacle_slot)`, `TRAFFIC_LIGHT_REACT(light_slot_id)` — **five**, as recorded.

⛔ **But the record's implied remedy is wrong for one of them.** `obstacle.offline`'s enum is
**10 classes, all dynamic agents**, measured over 87,481 cuboids
(`…/2026-07-26-physicalai-feature-probe/PHYSICALAI_FEATURE_PROBE.md:23`, artifact
`pai_label_schemas.json`): `automobile, person, heavy_truck, trailer, bus, rider, other_vehicle,
protruding_object, stroller, animal`. **No traffic light, no sign, no static infrastructure** —
the same probe records G5 as *"NO-GO in-house, permanently"* (`:328`). ⇒ **`obstacle.offline`
unblocks 4 of the 5 slots and can never unblock `TRAFFIC_LIGHT_REACT`'s.** That token's only
supply is the VLM's B2 `state` plus a range estimate that does not exist yet.

---

## 3. What `obstacle.offline` gives toward `ANCHOR_GOAL`, and what it does not

| | |
|---|---|
| ✅ **gives** | a **stable per-window lead identity** — `select_lead_causal → (track_id, gap_m, size_x)` (`lead_source.py:239`), **strictly causal** (cuboids timestamped ≤ t0 only); a three-state window label `LEAD / NO_LEAD / **NO_LABEL**` that refuses to manufacture free flow; the measured registration `n_ok 40 / n_failed 0` on val40 with `canonical_881 true`; a measured frame convention (**x fwd / y left**, 1,756 of 2,778 long-lived tracks world-static under it vs 236 mirrored) |
| ✅ **therefore unblocks** | the **LONGITUDINAL** family's distance-keeping (`status "OK"`, n 2846, `_families_unavailable []`), and the **agent slot** for `GAP_TARGET` / `YIELD_AT` / `WAIT_FOR_ONCOMING` / `EVADE_IN_CORRIDOR` |
| ⛔ **does NOT give — and this is the part that matters for `ANCHOR_GOAL`** | **nothing at all.** `ANCHOR_GOAL(anchor_id, t_reach_s)` is a statement about the **ego's own future geometry**. Not one of its two args, and not one of its four constraint slots, references another agent. `obstacle.offline` is the prerequisite for **four other tokens** and for the LONGITUDINAL family; it is **not** a prerequisite for `ANCHOR_GOAL` |
| ⛔ also not given | any map, lane graph, junction annotation, roundabout label, traffic-light feature or route/goal signal — settled at five probes; the card says verbatim *"we do not include open maps data"* |

⚠️ **The consequence for sequencing, stated plainly:** the committed fallback is named *"`ANCHOR_GOAL`
labels from the **PH0/`obstacle.offline`** line"* (§4.1). **`obstacle.offline` is not on
`ANCHOR_GOAL`'s critical path.** `ANCHOR_GOAL` needs **ego poses and an anchor table**, both of
which are already in the repo. That makes it *cheaper* than the record implies — and it also means
the `obstacle.offline` work already landed does **not** count as progress toward it.

---

## 4. ⛔ ADMISSIBILITY AND ECHO — audited on **my own** design, not on the incumbent's

### 4.1 The signal table, in the shape §1.4 requires

| signal at inference | computed from | could it have been computed from the situation classifier's output? | verdict |
|---|---|---|---|
| `anchor_id` (the emitted token/arg) | `goal_head_tac(z_tac_p, cond=e_g_str)` — vision latents + the strategic goal | **No.** No path from any classifier exists into `GoalHead` (`v6.py:548-554`), and the label deriver proposed here never reads one (§4.2) | ✅ |
| the **anchor table** `[K, S, 2]` | a **frozen buffer** built offline by FPS over TRAIN-corpus futures | **No** — it is the same tensor for every window, so it carries **zero per-window information** by construction | ✅ |
| `ĝ = anchor_table[anchor_id]` | the two rows above | **No** | ✅ |
| `t_reach_s` | the head's continuous arg | **No** | ✅ |
| ⛔ **`v0` (ego speed)** | the vehicle's own speedometer | No — but **it is classified two ways in two live HEAD documents** | ⛔ **PENDING PI ADJUDICATION — §8** |

### 4.2 The disjointness rule, enforced rather than documented

**Binding (PI 2026-08-03):** the goal path and the situation path stay information-disjoint at
inference; a goal must never carry the classifier's output *in any form*.

The hazard here is **not** the inference path — it is the **label** path, and it is subtle enough
to be worth naming. `ph0_pilot.engine_a_summary` now computes a `situations` block (`:384`) from
the very pose track an `ANCHOR_GOAL` label would be derived from. If the label deriver consumed
that summary wholesale, the goal head would be trained to reproduce a function of the classifier's
own detectors, and **attribution would die exactly as it did in the `--v2` conflation** — a planner
gain would be unassignable between the goal and the situation.

⇒ **Adopted, and stricter than the rule requires: the `ANCHOR_GOAL` label is
`argmin_k ‖endpoint − anchor_k‖` over ego-frame FUTURE displacement, and the deriver never reads
`tanitad.data.situations` in any form.** Pinned in code by
`tests/test_e_ag1_anchor_floor.py::test_no_situation_classifier_path`, which fails on the mere
presence of `detect_lane_change` / `detect_intersection` / `detect_roundabout` /
`situations_from_poses` in the instrument's source **and** on the module appearing in
`sys.modules`. *(Precedent: `ph0_pilot._fmt_engine_a` computes `situations` and deliberately does
not forward it into the B4 goal prompt — an omission its own comment calls **load-bearing**.)*

### 4.3 ⭐ THE ECHO TEST, run on this design

flagship v1's route head scored **1.0000** because it was an exact bijection of the nav it was fed.
The test that catches that is: *does any input at inference contain something the label was derived
from?*

| | |
|---|---|
| **the label** is derived from | ego poses over `[t, t+h]` — the **future** |
| **the inputs at inference** are | `z_tac_p` (vision at ≤ t), `e_g_str` (vision at ≤ t), and a frozen table |
| **overlap** | **none.** No inference input is a function of `poses[t:t+h]` |
| **the residual hazard, stated** | the *anchor table* is built from TRAIN futures. It is shared by every window and by every episode, so it cannot carry per-window label information — but it **can** carry corpus-level information, and that is exactly what the **`marginal` arm** (goal ← the vocabulary's own centroid) measures. **MEASURED: σ = 13.5553 m** against the live arms' 0.79–9.49, i.e. the zero-information null is 1.4–17× away. A win here cannot be an artefact of "any anchor works" |
| ⚠️ **the honest half** | a goal head trained to select a future-derived anchor **is** a coarse trajectory predictor. The bounds are structural and unchanged: the goal is **2 numbers** against a 60×2 plan, and `GoalDistanceScorer` cannot emit a wider one (`v6.py:667`) |

⚠️ **One echo this design does NOT have, and the incumbent's kinematic floor does.** E-WC2's
0-parameter baselines reach 1.1888 m by reading the **ego pose history**; `cv_goal_floor.py:17-21`
labels them ⛔ inadmissible for exactly that reason. The design here reads no pose at inference.
That is why its floor is 0.5637 m *from geometry* and its achieved σ is 9.4868 m *from vision* —
the gap between those two numbers is the whole problem, and it is honest.

---

## 5. ⭐ THE PRE-REGISTERED EXPERIMENT — both outcomes committed **before** the run

The pre-registration is **code, not prose**: `stack/scripts/e_ag1_anchor_floor.py` `PREREG` /
`VERDICTS`, staged at blob `dc31534` **before the instrument was first executed**. Thresholds are
§5.2's own, converted from ratios at §3.1's published incumbent ADE **0.4714**:
**σ ≤ 1.7 × 0.4714 = 0.80138 ⇒ FUNDED** · **σ ≥ 3.0 × 0.4714 = 1.41420 ⇒ REFUSED**.

| arm | question | ⭐ committed if it goes one way | ⛔ committed if it goes the other |
|---|---|---|---|
| **E-AG1** the **quantisation floor** | can `ANCHOR_GOAL` reach the bar with a **perfect** classifier? A bound from **geometry alone** — no model, no surface, no GPU | `σ_quant ≤ 0.80 m` at some K ≤ 256 ⇒ **FLOOR_CLEARS**: the formulation has headroom, proceed to E-AG2. **This does not fund it — it only fails to refuse it** | `σ_quant ≥ 1.41 m` at **every** K ≤ 256 ⇒ ⛔ **REFUSED on geometry**, with a perfect classifier. The only surviving form is anchor **+ continuous residual**, which is SEL-1's regression again and inherits its refusal |
| **E-AG2** discretise **vs** regress, same surface, same folds | is the anchor formulation itself a lever, or is the surface the whole problem? | σ_anchor **<** σ_ridge, paired-separated ⇒ discretisation is a real lever; build the labels now | σ_anchor **≥** σ_ridge or the paired interval spans 0 ⇒ ⛔ **the SURFACE is the whole problem**; `ANCHOR_GOAL` must be justified by something other than σ |
| **E-AG3** the `v0` contradiction | how large is the disagreement between two live HEAD documents? | — | ⛔ **PENDING_PI_ADJUDICATION** by construction. Reported as *the magnitude of a contradiction*, **never** as a funded arm |

**Method, and why each choice is the load-bearing one.**
*LOEO, and the vocabulary is REBUILT inside each fold* — a vocabulary built on all 40 episodes
would place an anchor on the held-out episode's own endpoints and the "floor" would be a leak.
E-WC2 measured that exact leak at **2.06×** for the ridge; the anchor-set analogue is pinned by
`test_loeo_vocabulary_never_contains_a_held_out_endpoint`.
*Two vocabulary constructions* — **FPS** (what the programme ships; spreads over the support so
the rare turns survive) and **k-means** (descends `mean‖x−a(x)‖²`, which **is** the σ statistic).
Assuming they are equivalent would have been an assumption; both are measured.
*⭐ And the headline arm uses the REAL vocabulary* — `refc_anchors_full_REBUILD.pt`, FPS over a
**200,000-window pool of the canonical TRAIN corpus**, so no val endpoint was ever in its pool and
no fold scheme is needed. `shipped_vocab_arm` **refuses** unless the vocabulary's last horizon
equals the requested endpoint, because scoring a 6 s ground truth against a 2 s anchor would look
exactly like a result (`test_shipped_vocab_refuses_a_horizon_mismatch`).
*Estimator* — point estimates **full-set**; intervals **episode-cluster bootstrap**
(`taniteval/ci.py`, 2000 draws); arm-vs-arm **paired** on the same windows.
⛔ `overlapping_holdout_se` is used **nowhere**.

⛔ **THE 6 s RULE, enforced in code.** §5.3's refutation check **fired this morning**
(σ(6 s) = 3.75 × σ(2 s) > 3 ⇒ **REDERIVE**), so **no 6 s threshold exists and none may be scaled
from the 2 s one**. This instrument emits `verdict: None` at 6 s under **every** branch —
`test_no_branch_ever_emits_a_6s_verdict`. ⚠️ **The task brief's bars (σ ≤ 0.80 / ≤ 1.41) are 2 s
bars.** Every verdict below is a 2 s verdict; the 6 s rows are **reported and not judged**.

---

## 6. THE NUMBERS

Surface: the **backfilled REF-C latent dumps** (`…/2026-08-04-lambda-findability/raw/latents_refc-{xl,base}-30k-ep.pt`)
— the identical 881-window / 40-episode canonical grid E-WC2 ran on, with the identical
`gt_endpoint` at steps 20 and 60. Features **`pooled` (992) + `ctx` (96) = 1088, VISION_ONLY**,
filtered through E-WC2's own `build_features`.

**Instrument parity control.** This instrument's free-endpoint ridge reproduces E-WC2's:
**4.7367 vs 4.7104** at 2 s (**+0.56 %**) and **18.3159 vs 18.3519** at 6 s (**−0.20 %**). The
residual difference is declared, not mysterious: E-WC2 picks λ **per axis**, this one picks a
**shared** λ by GCV summed over targets so a 256-way one-hot classifier costs one SVD per fold
instead of 256. ⇒ every number below sits on a surface that reproduces the refusing instrument.

### 6.1 ⭐ E-AG1 — the quantisation floor of the **shipped** vocabulary (2 s, 881 windows / 40 eps)

| | value |
|---|---|
| vocabulary | `refc_anchors_full_REBUILD.pt` — **K = 256**, `method fps`, `pool_size 200000`, `source …/physicalai-train-e438721ae894` |
| **σ per-axis** | **0.5637 m** |
| **95 % CI** (episode-cluster bootstrap) | **[0.5085, 0.6185]** |
| **σ/ADE** | **1.196** — ⭐ inside the FUNDED band (≤ 1.7); the CI's **upper** bound (0.6185) is still **1.30× below** the 0.80138 line |
| σ_long / σ_lat | 0.5674 / 0.5599 — **isotropic** |
| radial p50 / p90 | 0.658 / 1.223 m |
| runner-up anchor σ | 0.9165 m |
| **required top-1 to reach σ = 0.80 m** | **0.3788** (miss ⇒ runner-up) |
| **required top-1 to leave REFUSED (1.41 m)** | **0.0** — even a classifier that *always* lands on the runner-up clears it |
| **VERDICT** | ⭐ **FLOOR_CLEARS** |

⚠️ **The optimism in "required top-1 = 0.3788" is declared:** it assumes a miss lands on the
**second-nearest** anchor. Under the opposite assumption — a miss lands on a *uniformly random*
anchor — the required top-1 is **0.998**. *(The far-miss cost is read off the LOEO `marginal` row,
σ 13.5553; the marginal is a corpus statistic about the vocabulary's centroid and is near
vocabulary-independent, as the two files' identical floors in §6.5 illustrate.)* The truth depends
entirely on the classifier's error structure, which is what E-AG2 measures. **This is why E-AG1
alone cannot fund the branch, and the pre-registration says so.**

### 6.2 ⛔ E-AG2 — discretise vs regress, on the same surface (2 s, FPS vocabulary, LOEO)

| K | floor `oracle` | `clf` (K-way, vision) | **Δ vs ridge** (paired, +ve = worse) | separated | top-1 acc | chance |
|---|---|---|---|---|---|---|
| 8 | 4.0985 | 8.3161 | **+3.5795 [+2.3718, +4.8424]** | ✅ | 0.5528 | 0.125 |
| 16 | 2.1774 | 6.9181 | **+2.1815 [+1.0835, +3.2687]** | ✅ | 0.4404 | 0.0625 |
| 32 | 1.4369 | 7.1303 | **+2.3937 [+1.4099, +3.4384]** | ✅ | 0.2690 | 0.03125 |
| 64 | 1.0680 | 7.5179 | **+2.7813 [+1.5094, +4.1630]** | ✅ | 0.1907 | 0.0156 |
| 128 | 0.8703 | 7.7322 | **+2.9956 [+1.4465, +4.6017]** | ✅ | 0.1453 | 0.0078 |
| **256** | **0.7951** | **9.4868** | **+4.7502 [+3.0514, +6.3981]** | ✅ | **0.1101** | 0.0039 |

*(reference: free ridge **4.7367**; goal-echo null `marginal` **13.5553**; `snap` — the ridge's own
prediction snapped to the nearest anchor — **4.7364**, Δ **−0.0002 [−0.1031, +0.0703]**, **not**
separated.)*

**k-means replicates it**: Δ **+2.2554 … +4.8920**, separated at every K; its own floor at K=256 is
0.8111. ⇒ the result is not an artefact of FPS's tail-covering objective.

⛔ ⇒ **E-AG2's committed `WORSE_OR_FLAT` outcome is MET, at every K and under both constructions.**
On the frozen REF-C surface, **discretising the goal into anchors is 2.0× worse than regressing it
freely**, and the free regression was already refused at σ/ADE 9.99.

⚠️ **And it settles §6.1's optimism, which is why both miss-cost references were pre-declared.**
Inverting the two-point model on the measured pair (top-1 **0.1101**, σ **9.4868**, hit cost
`2 × 0.7951²` — this arm's own LOEO floor) gives a **miss** cost of σ ≈ **10.05 m** against the
uniformly-random anchor's
**13.56 m**. ⇒ **when this classifier is wrong — 89 % of the time — its pick is only 1.35× better
than a random anchor.** It does not fail by picking the neighbour; it fails almost uninformatively.
So the operative required-accuracy figure is §6.1's pessimistic branch (**0.998**), not its
optimistic one (0.3788).

⭐ **The three facts that make this readable rather than merely negative:**
1. **The classifier is not broken — it is coarse.** Top-1 is **28.2× chance** at K=256 and 4.4×
   at K=8; vision does carry goal information. It simply resolves the goal to **roughly one part
   in eight** (K=8 top-1 0.5528) when the bar needs **one part in 256**.
2. ⭐⭐ **`snap` isolates the estimand from the estimator, and exonerates the estimand.** `snap`
   is the *same ridge prediction* rounded to the nearest anchor — the estimator is held fixed and
   only the estimand changes. It is **NOT separated** from the ridge for K ≥ 64
   (K=256: Δ **−0.0002 [−0.1031, +0.0703]**), and still not separated in §6.5's much stronger
   `v0` regime (**+0.0383 [−0.2125, +0.2338]**). ⇒ **quantising the goal onto the shipped
   vocabulary is FREE. What costs +4.75 m is training a K-way one-hot classifier to pick the
   anchor.**
3. ⭐ **And that failure mode has already been measured on an independent surface.** A one-hot
   `anchor_id` target is metric-BLIND: it treats "picked the adjacent anchor" and "picked one
   40 m away" as the same error. **E-OBJ-1** measured exactly this axis — swapping a fitted ranker
   from one-hot CE to **`softade`** recovers **−0.0974 m (base) / −0.1670 m (XL), separated**, and
   the recovery is **LONGITUDINAL** — while *softening the target* is separated **worse**. ⇒ the
   prescription is a **metric-aware, distance-weighted** anchor target (or regress-then-snap),
   **not** a one-hot CE, and it is not a new hypothesis: it is E-OBJ-1's finding applied one level
   up. §7 item 1b.

⚠️ **The honest caveat on E-AG2, stated rather than buried.** The classifier is a **linear
one-hot least-squares** classifier — matched to the ridge in feature access and in linearity, but
it is *not* the best linear classifier obtainable (a softmax with a proper loss would beat it).
§6.5 quantifies the size of that caveat directly: given `v0`, the ridge improves **2.85×** and the
same-featured classifier improves **1.03×**. ⇒ **part of E-AG2's margin is estimator weakness, and
fact 2 is the reason that does not rescue the arm as posed** — the estimand was separately
exonerated, so the pre-registered conclusion ("the surface/estimator is the problem, not the
discretisation") stands and is now *localised*.

### 6.3 The 6 s rows — **reported, not judged** (681 of 881 windows; 200 excluded with reason)

| arm | σ(6 s) per-axis | note |
|---|---|---|
| floor `oracle`, K=256 LOEO FPS | **4.3387 [3.2036, 5.3724]** | no **6 s** vocabulary has ever been built; the shipped one is 2 s and the instrument **refused** to score it (`ValueError: refusing rather than scoring a 6s ground truth against a 2s anchor`) |
| `clf`, K=256 | 30.6560 | top-1 **0.0690** |
| free ridge | 18.3159 | reproduces E-WC2's 18.3519 |
| goal-echo `marginal` | 42.7912 | |
| **verdict** | ⛔ **`None`, every branch** | §5.3 fired REDERIVE this morning; no 6 s threshold exists |

⚠️ **This is the gap that matters for v6f and it is not closed by anything here:** the v6f
selector scores the **6 s** endpoint (`waypoints[:, :, -1]`, 60 steps), and **every anchor
vocabulary the programme owns stops at 2.0 s**. Building a 6 s vocabulary is CPU-only and cheap —
§7 item 2 — but no admissible **bar** exists to judge it against until §5.3's threshold is
re-derived on a 6 s fan, and v6 has never emitted one.

### 6.4 ⭐ THE FOUR FAMILIES — and the decomposition that should change the design

⛔ Per family, never pooled. Each number is on the same windows as the ADE-like quantity beside it.

| family | reported here | value |
|---|---|---|
| **LONGITUDINAL** | `sigma_long_m`, every arm, both horizons | floor **0.8954** · ridge **6.6132** · **clf 13.3502** · echo-null **19.0578** (2 s, K=256). ⚠️ target-speed and distance-keeping are **n/a with reason**: this probe rolls no trajectory |
| **LATERAL** | `sigma_lat_m`, every arm | floor **0.6802** · ridge **1.0667** · **clf 1.3310** · echo-null **2.0723**. ⚠️ heading / curvature / yaw-rate **n/a with reason**: no trajectory is rolled |
| **TACTICAL** | ⭐ **the headline** — goal/anchor **selection** admissibility *is* this family | §6.1 / §6.2 |
| **STRATEGIC** | **n/a with reason**, n = 0 | PhysicalAI-AV ships no map, lane graph, junction annotation or route signal — settled at five probes |

⭐⭐ **THE FINDING.** Read the LONGITUDINAL and LATERAL rows against each other:

* the **problem** is 98.8 % longitudinal — the zero-information null's variance splits
  `19.0578²` vs `2.0723²`, a **9.2× ratio in σ** and **84× in variance**;
* the **anchor classifier is nearly adequate laterally and hopeless longitudinally** — σ_lat
  **1.3310** against a floor of 0.6802 (1.96×), σ_long **13.3502** against a floor of 0.8954
  (**14.9×**);
* the **quantisation, by contrast, is isotropic** — 0.5674 vs 0.5599. ⇒ **the shipped FPS
  vocabulary spends half its resolution on the axis that carries 1.2 % of the variance.**

⇒ **A single K-way `anchor_id` forces one categorical decision to carry both axes at once. That is
the 5-way manoeuvre softmax defect — the programme's "single largest known defect" — surviving one
level up in the goal vocabulary, after `a_tac` was explicitly factored into LAT × LON to retire
it.** The design consequence is §7 items 1a/1b.

### 6.5 Replication on REF-C-base, and the E-AG3 `v0` arm

**REPLICATION — an independently trained model, same grid, same folds** (`…/raw/e_ag1_anchor_floor_refc-base-30k.json`):

| | XL | base |
|---|---|---|
| free ridge σ(2 s) | 4.7367 *(E-WC2: 4.7104)* | **4.5469** *(E-WC2: 4.5545, −0.17 %)* |
| `clf` σ(2 s), K=256 FPS | 9.4868 | **10.0039** |
| **Δ `clf` − ridge**, paired | **+4.7502 [+3.0514, +6.3981]** ✅ | **+5.4570 [+3.8345, +7.1073]** ✅ |
| top-1, K=256 | 0.1101 | 0.1260 |
| **shipped-vocabulary floor σ(2 s)** | **0.5637 [0.5085, 0.6185]** | **0.5637 [0.5085, 0.6185]** |

⭐ **The floor row is bit-identical across the two files, and that is a self-check rather than a
coincidence:** E-AG1 is a statement about *geometry and a frozen table*, so it **must** be
model-independent. A floor that moved between two model dumps would have meant a latent had leaked
into it.

**E-AG3 — the `v0` arm** (`…/raw/e_ag3_v0_contradiction_refc-xl-30k.json`; features
`pooled` 992 + `ctx` 96 + **`v0` 1** = 1089; `any_echo: true`;
stamped ⛔ `PENDING_PI_ADJUDICATION`):

| horizon | arm | σ per-axis | 95 % CI | **σ_long** | σ_lat | σ/ADE |
|---|---|---|---|---|---|---|
| **2 s** | ridge, VISION_ONLY *(§6.2)* | 4.7367 | [3.8055, 5.7262] | **6.6132** | 1.0667 | 10.048 |
| **2 s** | ⭐ **ridge + `v0`** | **1.6626** | **[1.4522, 1.8789]** | **2.0917** | **1.0739** | **3.527** |
| 2 s | `snap` + `v0`, K=256 | 1.7009 | [1.4952, 1.9128] | 2.0636 | 1.2361 | 3.608 |
| 2 s | `clf` + `v0`, K=256 | 9.2255 | [7.7886, 10.5555] | 12.9786 | 1.3331 | 19.571 |
| 6 s | ridge, VISION_ONLY | 18.3159 | [15.8514, 20.9488] | 23.2124 | 11.4949 | 38.854 |
| 6 s | ⭐ **ridge + `v0`** | **13.0448** | **[11.3564, 14.5944]** | **14.3868** | 11.5478 | 27.672 |

⭐⭐ **The decomposition is one-sided and exact: `v0` collapses the LONGITUDINAL axis by 3.16×
(6.6132 → 2.0917) and leaves the LATERAL axis untouched (1.0667 → 1.0739, +0.7 %).** One scalar
buys 2.85× on the headline — **more than any representation change measured in this programme** —
and it buys it entirely on the axis that carries 97.4 % of the residual's squared error.

⚠️ **Three things this arm does NOT say.**
1. It is **not funded**: σ/ADE **3.527 ≥ 3.0** is still inside §5.2's REFUSED band.
2. It **loses to a 0-parameter rule**: constant-yaw-rate reaches **1.1888** (σ_long 1.4932,
   σ_lat 0.7725) using `v0` **and** yaw-rate, no learning. So vision + `v0` is beaten on **both**
   axes by kinematics ⇒ **vision is adding essentially nothing to the 2 s goal on this surface**.
3. The **classifier barely moves** with `v0` (9.4868 → 9.2255, **1.03×**) while the ridge moves
   **2.85×** — the estimator-weakness caveat of §6.2, quantified.

---

## 7. THE COSTED PLAN — what to build, from what, and in what order

⛔ **Read the order literally: item 0 is a PI decision, items 1a–2 are 0 GPU and unblocked, and
item 3 is the gate that decides whether items 4–6 are worth building at all.** §6.2 says a full label pipeline on today's surface
would buy nothing.

| # | work item | source | cost | unblocked? |
|---|---|---|---|---|
| **0** | ⛔ **ADJUDICATE `v0`** (§8). It is one scalar and it is worth **2.85× MEASURED**. Nothing about the goal head's input set should be fixed before it is settled, and everything else is measured and waiting on it | PI decision | **0** | ⛔ **PI** |
| **1a** | ⭐ **FACTOR THE GOAL, on the axis §6.4 measured.** Split `ANCHOR_GOAL`'s single mixed decision into a **lateral/shape** anchor and a **longitudinal/progress** arg, and let `SPEED_BAND(v_lo, v_hi)` — which the vocabulary **already has** and which §4.5 already assigns the LONGITUDINAL family — own the progress half. Requires the **multi-label emission** §2.3 shows the head lacks | `v6.py` (owned by another agent — **escalation, not an edit**) | design + ~200 lines; **0 GPU** | ✅ |
| **1b** | ⭐ **SUPERVISE THE ANCHOR WITH A METRIC-AWARE TARGET, NOT A ONE-HOT `anchor_id`** — a distance-weighted soft target over anchors, or regress-then-snap. §6.2 fact 2 exonerates the estimand (`snap` is free) and facts 1/3 localise the failure in the one-hot objective, which **E-OBJ-1 already measured as the inferior half of exactly this axis, with a LONGITUDINAL recovery** | the S-T loss (`train_v6_staged.py`, another agent's file — **escalation**) | **0 params, 0 GPU** | ✅ |
| **2** | **A 6 s anchor vocabulary.** `build_refc_anchors.py --horizons 5,10,…,60` over the canonical train corpus. ⚠️ needs the **train epcache**, which is not on this box; the val40 poses are reachable in **81 s / 18 MB** by HF range-read (`hf_poses_pull.py`) but a val-built vocabulary is only admissible under LOEO | `build_refc_anchors.py` (built, unmodified) | ~1 CPU-hour on a box with the epcache; **0 GPU** | ⚠️ needs the train cache |
| **3** | ⭐⭐ **THE GATE: re-run THIS instrument, unchanged, on frozen S-W latents.** The dump is the **only** missing input; the instrument, the endpoint backfill and the val40 poses are all in place | `refc_dump_latents.py --endpoint-steps 20,60` on the S-W ckpt | **~10–25 GPU-min**, a deliberate training pause | ⛔ **Thor is training** (27.18 s/step, ~7.4 d left) |
| **4** | The **categorical arg channel** — a typed slot that can hold `anchor_id`, `agent_slot_id`, `reason`, `state` (§2.3) | `v6.py` — **escalation** | ~150 lines; 0 GPU | ✅ (blocked only on ownership) |
| **5** | The **label emitter**. ⏹ **The `ANCHOR_GOAL` half is BUILT this turn** — `stack/tanitad/data/anchor_goal.py`, 16 tests, information-disjoint from `situations` **by enforcement** (§4.2, §10). What remains is `CORRIDOR_OFFSET` from Engine A's polyline and the corpus-wide emit | `ph0_pilot.engine_a_summary` | ~½ day; 0 GPU | ✅ — corpus-wide emit **gated on item 3** by §6.2 |
| **6** | **Slot vocabulary** for the four agent tokens: a stable, clip-local→window-local id from `lead_source`'s `track_id` | `lead_source.py` | ~1 day; 0 GPU | ✅ — but not on `ANCHOR_GOAL`'s path (§3) |

### 7.1 The next pre-registered refusal — **E-AG4**, committed here

**Question:** does an **anisotropic** vocabulary — resolution allocated along-track in proportion
to variance — beat the isotropic FPS floor at equal K?
**Method:** rebuild the vocabulary in a whitened frame (divide each axis by the corpus σ of §6.4's
`marginal` row) before FPS/k-means; re-run E-AG1 unchanged. 0 GPU, minutes of CPU.
**Committed outcomes:** ⭐ σ_quant falls **and** σ_long falls by more than σ_lat rises ⇒ the
vocabulary construction is a free lever and the shipped one is leaving margin on the table.
⛔ σ_quant does not fall ⇒ the isotropy is not costing anything and item 1's factorisation must be
justified by the **classifier's** anisotropy (§6.4) alone, which is the stronger of the two
arguments anyway.

---

## 8. ⛔ ESCALATION — the `v0` contradiction is worth **2.85× MEASURED**, on one scalar

Two documents at HEAD classify the same signal in opposite ways:

| | says | file:line |
|---|---|---|
| the E-WC2 instrument | `FEATURE_ADMISSIBILITY["v0"] = "ECHO"` ⇒ **inadmissible**, refused without `--allow-echo-features` | `stack/scripts/e_wc2_sigma_star.py:188` |
| the v6f design's own admissibility audit, §1.4 row 3 | the emitted fan is computed from `z_op` (vision) + `e_g_tac` + `cand_queries` + **true `v0`**, situation-classifier path **none**, verdict **✅** — *"`v0` is ego **speed**, an input the programme has always fed; it is not a classifier output"* | `Project Steering/V6F_PLANNER_DESIGN.md:169` |
| the 0-parameter floor | ⛔ *"these read the EGO POSE HISTORY, the privileged channel the vision-only rule forbids at inference"* | `…/2026-08-16-ewc2-result/code/cv_goal_floor.py:17-21` |

**Why it is not a bookkeeping question — MEASURED HERE, not inferred.** Appending `v0` to the same
1088-dim vision block and changing nothing else (§6.5):

| | σ per-axis (2 s) | σ_long | σ_lat | σ/ADE |
|---|---|---|---|---|
| ridge, VISION_ONLY | 4.7367 | 6.6132 | 1.0667 | 10.048 |
| ⭐ **ridge + `v0` (one scalar)** | **1.6626** | **2.0917** | 1.0739 | **3.527** |
| **improvement** | **2.85×** | ⭐ **3.16×** | **1.00× (none)** | 2.85× |

⭐ **The effect is entirely longitudinal, to within 0.7 %.** `v0` is not "a useful extra feature";
it is *the* missing longitudinal state, and its absence is 97.4 % of the goal head's squared error.

**And the corroborating structure from E-WC2's own artifacts** (`cv_goal_floor.json`,
`ewc2_sigma_star_refc-xl-30k.json`), at 2 s:

| | σ_long | σ_lat | σ per-axis |
|---|---|---|---|
| ridge on frozen **vision** latents | **6.5752** | 1.0688 | 4.7104 |
| `constant_velocity` (0 params, reads v0) | **1.4506** | 2.2133 | 1.8712 |
| `go_straight` (0 params, reads v0) | **1.4509** | 2.0091 | 1.7524 |
| `constant_yaw_rate` (0 params, reads v0 + ω0) | **1.4932** | 0.7725 | **1.1888** |

⭐ **All three kinematic baselines pin σ_long at ≈1.45–1.49 m; the vision ridge is 6.5752 —
4.40× worse on the axis that carries 97.4 % of its squared residual.** Laterally the ordering
reverses in part: vision (1.0688) beats constant-velocity (2.2133) and loses to constant-yaw-rate
(0.7725). ⇒ **`v0` is precisely the missing longitudinal state, and `ω0` is precisely the missing
lateral one** — and the two tables agree: given `v0`, the ridge's σ_long lands at **2.0917**,
between the kinematic 1.45–1.49 and the vision-only 6.61.

**The adjudication needed from the PI, stated as a question with its consequence:**
> Is **instantaneous ego kinematic state at t** (`v0`, and optionally yaw-rate `ω0`) admissible as
> an input to the **goal head** at inference — as `V6F_PLANNER_DESIGN.md` §1.4 already declares it
> for the emission — or is it excluded as *ego at inference*?

* If **admissible**: the goal head should read `[z_tac_p ‖ e_g_str ‖ v0(, ω0)]` and starts from a
  ~1.19 m floor rather than a 4.71 m ceiling. ⚠️ Even then **1.1888 m is σ/ADE 2.52 — still not
  FUNDED**; vision would have to add 1.49× on top.
* If **inadmissible**: `V6F_PLANNER_DESIGN.md` §1.4 row 3 must be corrected, because the emission
  currently integrates from the **true `v0`** and the design calls that ✅.

⚠️ **The echo test on `v0`, for the record:** the label is `poses[t : t+h]`; `v0` is speed **at t**.
Disjoint in time — `v0` is strongly *predictive* of the label by physics, which is legitimate
prediction, not an echo. It is **not** the same defect as flagship v1's route head, which was fed
the very quantity it was scored on. The contested point is the *vision-only* rule's **scope**, not
leakage.

---

## 9. What this changes in the record

1. ⭐ **The committed fallback is now split.** *"the work moves to `ANCHOR_GOAL` supervision"*
   (§5.2 / E-WC2 §6) is **necessary and not sufficient**: MEASURED, the formulation clears the bar
   (σ/ADE 1.196) and the surface refuses it (Δ +4.75 separated worse than an already-refused
   ridge). A doc that reads "`ANCHOR_GOAL` supervision is the branch" without that split
   over-promises the label work.
2. ⭐ **`obstacle.offline` is NOT on `ANCHOR_GOAL`'s critical path** (§3). The fallback's own
   phrasing — *"labels from the PH0/`obstacle.offline` line"* — attaches it to four **other**
   tokens and to the LONGITUDINAL family. `ANCHOR_GOAL` needs ego poses and an anchor table, both
   already in the repo.
3. ⭐ **The record's "five need agent slots" is right; its implied remedy is wrong for one of
   them.** `obstacle.offline` can never supply `TRAFFIC_LIGHT_REACT`'s `light_slot_id` — 10 dynamic
   classes, zero infrastructure, G5 *"NO-GO in-house, permanently"* (§2.4).
4. ⭐ **A new gap class, not previously stated: the arg TYPE gap.** Seven of the nine tokens need a
   categorical arg; the head and the conditioner are continuous-only on both sides. **2 of 9 tokens
   are expressible today**, with perfect labels (§2.3).
5. ⚠️ **No anchor vocabulary reaches 6 s**, and the v6f selector scores a 6 s endpoint (§6.3).
6. ⚠️ **The task brief's σ ≤ 0.80 / ≤ 1.41 bars are 2 s bars.** §5.3's REDERIVE fired this
   morning; applying them at 6 s would violate the ≤2× extrapolation rule *and* the instrument's
   own committed 6 s rule. Every verdict here is a 2 s verdict, by construction.

---

## 10. What was implemented, CPU-only

| file | what | tests |
|---|---|---|
| `stack/scripts/e_ag1_anchor_floor.py` | ⭐ the E-AG1/2/3 instrument **and its pre-registration as data** | 22 |
| `stack/tanitad/data/anchor_goal.py` | ⭐ the **label deriver** — `ANCHOR_GOAL(anchor_id, t_reach_s)` from ego-frame future endpoints + a frozen anchor table | 16 |

⭐ **The label deriver refuses more than it emits today, on purpose, and both refusals are §6/§2's
findings turned into code:**
1. it **refuses a 2 s vocabulary as a tactical label** — `t_reach` must lie in `(2.0, 6.0]`, and
   *no vocabulary the programme owns reaches the band* (§6.3). An override exists and **stamps
   itself** (`off_band_stamp`), so a 2 s diagnostic can never be re-read as a tactical label;
2. it **never writes `anchor_id` into a physical-units arg slot** (§2.3) — the id travels in its
   own integer field and the continuous slot stays **NaN with mask 0**, the `IGNORE` discipline,
   so the representational gap stays visible instead of being papered over with a plausible float.

Its input contract ("ego-frame displacement at the horizon, x forward / y left") is **pinned
against the programme's own producer**, `driving_diagnostic.gt_ego_waypoints`, so the phrase means
one thing in both places.

**Test properties pinned** (the ones whose failure would produce a *number*, or a *label*, instead
of an error): the multi-target ridge pinned against the incumbent `RidgeSVD`; the FPS **prefix**
property that makes the K sweep nested; k-means beating FPS on a density-skewed pool (the trade-off
measured, not asserted); `required_top1` returning **`None`** rather than 1.0 when the floor already
exceeds the target — *a refusal rendered as difficulty reads as hard when it is impossible*; the
2 s-vocabulary-vs-6 s-endpoint refusal in **both** modules; **no branch emitting a 6 s verdict**;
invalid rows excluded with n, `anchor_id = −1` and NaN residuals, never imputed; `σ_perax` proved
to be the per-axis form (not the √2-larger radial RMS); `v0` refused without the echo flag and
stamped `PENDING_PI_ADJUDICATION` with it; the LOEO vocabulary proven episode-disjoint by a
far-away hold-out episode; and **`test_no_situation_classifier_path` / `test_module_has_no_situation_classifier_path`**,
the enforced form of the binding disjointness rule.

### 10.1 ⛔ The disjointness guard was itself defective — three times, and the fixes are the finding

The guard on the **binding** goal/situation disjointness rule went through three failures. Each is
a distinct root-cause class and each was caught by *making the guard fail on purpose*, which is the
part worth keeping.

| # | the defect | how it presented | the class |
|---|---|---|---|
| **1** | `assert "tanitad.data.situations" not in sys.modules`, asserted **in-process** | ⛔ **false FAIL** in the full suite (`sys.modules` there holds *every other test's* legitimate imports) and — worse — a ⚠️ **false PASS in isolation**, where nothing else had imported it either, so the check had never actually run | **an isolation check whose scope is the SESSION, not the SUBJECT.** Same family as `df` / cgroup `usage_in_bytes` / Thor `free`: a probe that answers about the wrong scope |
| **2** | the obvious repair — add the bare token `"tanitad.data.situations"` to the source scan | would have fired on a **clean** module: both modules' docstrings contain that literal string *in the sentence saying they never read it*, and `run()` emits it in a provenance string | **a guard that matches its own documentation** |
| **3** | the source scan's *qualified* spellings (`"import situations"`, `"from tanitad.data.situations"`) | ⚠️ **MEASURED: neither matches `import tanitad.data.situations`.** The negative control's very first form slipped straight past the token loop; only the subprocess probe caught it | **a guard whose pattern set is narrower than the thing it guards** |

⭐ **The shape that holds is TWO guards, and the negative control proves they are complementary,
not redundant** — MEASURED, by adding a forbidden import and watching what fires:

| forbidden import | AST check | subprocess probe |
|---|---|---|
| `import tanitad.data.situations` (module level) | ✅ fires | ✅ fires |
| `from tanitad.data.situations import kinematics as _k` | ✅ fires | ✅ fires |
| ⭐ **lazy, function-local** (`def _lazy(): import tanitad.data.situations`) | ✅ **fires** | ⛔ **says `CLEAN`** — it is never executed at import time |
| a *transitive* import (a helper module that imports it) | ⛔ blind — it reads one file | ✅ fires |

⇒ the **AST** walk reads only real `import`/`from` nodes, so it catches every written spelling
including deferred ones **and cannot match prose**; the **subprocess** catches what the AST of a
single file cannot see. Restored to clean, both go green (`2 passed`), and the modules were never
left mutated (index blob == disk hash after every round).

---

## 11. Test evidence

**Baseline, briefed at HEAD `6c27b38`: 3036 passed / 0 failed / 17 skipped / 2 xfailed.**

`PYTHONUTF8=1 python -m pytest -q -p no:cacheprovider` from `stack/` →
⭐ **3154 passed, 17 skipped, 2 xfailed, 0 FAILED**, exit 0, 378 s. Log: `raw/stack_pytest.txt`.

⚠️ **The count is well above the briefed baseline and MOST OF THE EXCESS IS NOT MINE** — the same
caveat E-WC2 recorded this morning. **My contribution is +38 tests**: `test_e_ag1_anchor_floor.py`
(**22**) and `test_anchor_goal_labels.py` (**16**), both new files. 3036 + 38 = 3074, so **~80 are
concurrent sibling work** landing in the same tree. **0 failures, 0 errors** ⇒ nothing here
regressed anything.

⚠️ **An intermediate run of this suite read `1 failed, 3137 passed`** — that failure was §10.1's
defect **1**, in my own test, and it is fixed and negative-controlled. It is recorded rather than
quietly overwritten because the suite catching it is the point.

---

## 12. Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| **This report** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-anchor-goal-supervision/ANCHOR_GOAL_SUPERVISION.md` | staged |
| ⭐ **The instrument + its pre-registration** | `stack/scripts/e_ag1_anchor_floor.py` | staged **before the first run** (blob `dc31534`) |
| **Its 22 tests** | `stack/tests/test_e_ag1_anchor_floor.py` | staged |
| ⭐ **The `ANCHOR_GOAL` label deriver** | `stack/tanitad/data/anchor_goal.py` | staged |
| **Its 16 tests** | `stack/tests/test_anchor_goal_labels.py` | staged |
| ⭐ **E-AG1/E-AG2 result, REF-C-XL** | `…/2026-08-16-anchor-goal-supervision/raw/e_ag1_anchor_floor_refc-xl-30k.json` | staged |
| **Replication, REF-C-base** | `…/raw/e_ag1_anchor_floor_refc-base-30k.json` | staged |
| ⛔ **E-AG3 `v0` contradiction arm** (PENDING_PI_ADJUDICATION) | `…/raw/e_ag3_v0_contradiction_refc-xl-30k.json` | staged |
| Test log | `…/raw/stack_pytest.txt` | staged |

**Nothing was committed and nothing was pushed** (`AGENT_OPERATING_STANDARD` rule 1).
⛔ **Thor was not contacted.** No file owned by another live agent was edited
(`v6.py`, `train_v6_staged.py`, `planner_p2.py`, `ci.py`, `build_obstacle_join.py`,
`bev_raster.py`, `CLAUDE.md`, `RETRACTION_LOG.md`, `V6F_PLANNER_DESIGN.md` — all untouched).

---

## 13. Escalations — requests, not notes in a README

1. ⭐⭐ **The `v0` admissibility contradiction (§8) must be adjudicated before any goal-head input
   set is fixed.** Two live HEAD documents disagree, and it is worth **4.40×** on the axis that
   carries **97.4 %** of the goal head's squared error. This is a PI decision; everything around it
   is measured and waiting.
2. ⭐ **`v6.py` needs a categorical arg channel and a multi-label goal emission** (§2.3, §6.4).
   Today the head can express **2 of the 9** tokens and cannot emit the `ANCHOR_GOAL ∧ SPEED_BAND`
   pair the measurement calls for. `v6.py` is another agent's file this turn — **this is a request
   to schedule it, not a note.** 0 GPU.
2b. ⭐ **When `ANCHOR_GOAL` supervision is written, its loss must be metric-aware.** §6.2 fact 2
   proves the quantisation is free and facts 1/3 localise the failure in the **one-hot target** —
   the same axis **E-OBJ-1** measured, with a **longitudinal** recovery, on an independent surface.
   A `--anchor-ce` arm shipped with a one-hot target would repeat a refuted objective. **0 params,
   0 GPU**; the file (`train_v6_staged.py`) belongs to another live agent.
3. ⭐ **Add the S-W re-run of THIS instrument to the same ~10–25 GPU-min pause already owed to
   E-WC2.** Both read the same dump (`refc_dump_latents.py --endpoint-steps 20,60`); running them
   together costs nothing extra and turns one pause into two verdicts. **Pre-register E-AG2's bar
   before the pause is spent**: on S-W latents, `clf` must be **paired-separated better** than the
   free ridge, or the anchor formulation is refused on the new surface too.
4. ⚠️ **`V6F_PLANNER_DESIGN.md` §5.2/§4.1 and `MODEL_REGISTRY.md` should record §9's split** —
   the fallback is necessary-and-not-sufficient, and `obstacle.offline` is not on its critical
   path. Both files belong to the orchestrator/PI; the text is in §9 ready to lift.
5. ⚠️ **`RETRACTION_LOG.md` gets TWO root-cause classes from this turn.** Append-only and the
   orchestrator's file; both texts are ready to lift.
   * **§2.1** — *a name collision that makes an unbuilt thing look built*
     (`TARGET_SPEED_BANDS_MPS` / `SPEED_BANDS` vs the `SPEED_BAND` token). Same family as C70b,
     which mis-commissioned an agent yesterday.
   * **§10.1** — **THREE classes from one guard**, all caught by a deliberate negative control:
     (a) *an isolation check whose scope is the SESSION rather than the SUBJECT* — an in-process
     `sys.modules` assertion that passed standalone and failed in the suite, and whose standalone
     green had established nothing; (b) *a guard that matches its own documentation* — the obvious
     repair would have fired on a clean module, because the module's docstring contains the very
     string it forbids; (c) *a guard whose pattern set is narrower than the thing it guards* —
     MEASURED, `"import situations"` and `"from tanitad.data.situations"` **both miss**
     `import tanitad.data.situations`. The durable shape is an **AST import walk** (catches every
     spelling incl. lazy/function-local; cannot match prose) **plus** a **subprocess probe**
     (catches transitive imports the AST of one file cannot see) — proven complementary, not
     redundant, in §10.1's table.
6. ⚠️ **No 6 s anchor vocabulary exists** and the v6f selector scores a 6 s endpoint (§6.3).
   Building one is CPU-only but needs the **train** epcache, which is not on this box.
