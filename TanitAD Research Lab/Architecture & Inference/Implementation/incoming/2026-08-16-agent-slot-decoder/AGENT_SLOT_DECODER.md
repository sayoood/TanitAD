# F-18 — THE PERCEPTION AGENT-SLOT DECODER · pre-registration

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **Author:** F-18 subagent
**Status:** ⬜ IMPLEMENTED · ⛔ **NOT TRAINED, NOT MEASURED.** Every number below that is not
stamped MEASURED is a threshold committed **before** any result exists.
**Eval tier:** ⛔ **T0-DIAGNOSTIC** for everything in §4. See §6 for why this may never be
quoted as driving performance and what a T1 claim would require.

---

## 0. What this closes

`DIAGRAM_CONFORMANCE.md` (2026-08-16) audited the binding v6 diagram element by element.
§4.2's first interpretation-head row —

> perception agent slots (bbox cx,cy,yaw,l,w · state v,yaw-rate,occluded · class & size)
> — ⬜ **NOT BUILT** — §6: *"NEW — design here; DETR-style slot decoder ~2–4 M params on
> spatial tokens"*. No slot-decoder module exists in `tanitad/` (2 probes). The obstacle
> **join** (label side) exists (`build_obstacle_join.py`). Needs: the prereg'd head +
> ~1 GPU-day. — Fix **F-18**

was the **last unbuilt PERCEPTION cell**. The head now exists, default-OFF, with its tests
and this pre-registration. ⛔ It is **not trained**: Thor is the only GPU and it is running the
30k S-W (that constraint is why this document exists instead of a result).

---

## 1. What was built (MEASURED, this box, 2026-08-16)

| item | where | fact |
|---|---|---|
| the module | `stack/tanitad/models/agent_slots.py` | DETR-style slot decoder + exact Hungarian matcher + set loss + join→target adapter |
| wiring | `stack/tanitad/models/v6.py` | `V6Config.agent_slots` (default **False**), built LAST in `__init__`; new `interp` module group; new `ISOLATION_MATRIX` row; new probed edge |
| introduction | `stack/scripts/train_v6_staged.py` | `STAGE_MAY_INTRODUCE["S-T"] += ("agent_slots.",)`; CLI `--agent-slots` + 5 knobs; 3 preflight refusals |
| tests | `stack/tests/test_v6_agent_slots.py` | **41 tests, all passing** |

**MEASURED numbers** (recompute on drift, never inherit):

* **Production geometry: `3,207,445` params / `62` state_dict keys** — 16 queries ×
  `d_model` 256 × depth 3 × 8 heads over the 16 readout cells of width 128. Inside §6's
  pre-registered **2–4 M** band, which is what the band is for. Built through the real CLI
  (`--agent-slots` at the default geometry) the stack reads **91,100,894 params**
  (`= 87,893,449 + 3,207,445`), **467 keys**, `per_group["interp"] = 3,207,445`, and
  `assert_isolation` returns
  `{planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0, perception_to_trunk: 0}`.
* **Inertness: the default build is `87,893,449` params / `405` state_dict keys — UNCHANGED.**
  Verified by direct instantiation before *and* after the edit. Config E (the live run) is
  `336,542,025` / `573`, also unchanged. Byte identity is proved **per tensor** with
  `torch.equal` against a **CONTENT-anchored** pre-change revision of `v6.py` (never HEAD —
  C75), together with the **RNG stream** and the forward's **output key set**.
* Slot channel width **21** (`SLOT_FIELDS`), class vocabulary **10** (imported from
  `bev_raster.ALL_CLASSES`, never re-listed).

⚠️ **One inertness break was CAUGHT BY AN EXISTING GUARD during this build and fixed.** The
first implementation returned `interp_side`/`agent_slots` unconditionally (as `None` when the
head is absent) and
`test_v6_gstr_port.py::test_default_forward_is_bit_identical_and_emits_no_new_key` **FAILED** —
the default forward's *key set* is part of the resume contract, not only its tensors. The keys
are now added only when the head exists. Recording it because the guard working is the finding.

### 1.1 What it reads at inference — ⛔ VISION ONLY

**Inputs at inference: the spatial memory computed from `frames`, and NOTHING else.**

`AgentSlotDecoder.forward(self, memory)` takes **one positional tensor and has no keyword and
no `**kwargs`** — the signature is the audit, the same discipline as `GoalHead`'s refusal of an
undeclared conditioning path. No `v0`, no recorded actions, no `e_g_tac`/`e_g_str`, no pose, no
situation-classifier output; there is no door for one to arrive through later either.

Pinned two ways in `test_v6_agent_slots.py`:
1. `inspect.signature` — exactly one parameter, no var-args, no var-kwargs;
2. the detector that can actually fail — move `v0` (×3 + 7) and the actions (+5) with the frames
   held: **every slot field is bit-identical**; move the FRAMES: the slots change. The
   comparison is proved non-vacuous by asserting the same privileged move DOES change the plan.

**Labels may and do use privilege** — the admissible half of the PI's 2026-08-03 rule.
`obstacle.offline` cuboids reach the ego frame through **egomotion** (`build_obstacle_join.py`
transform 1), and the per-slot rates are a finite difference of those label positions.

### 1.2 The gradient decision — ⛔ the head does NOT train the encoder

**Decision: the slot decoder's gradient reaches NOTHING but its own `interp` group.** The
memory is `_cut` (detached) at the seam in `V6Stack.forward`.

**Why this is not a formality.** Its supervision is a **PERCEPTION LABEL**, and the diagram's
header row — audited ✅ CONFORMS in §2.1 — is *"no perception label, map or reward in any trunk
loss"*. Two independent reasons, either sufficient:
1. **It would break the binding rule.** A live edge here makes `obstacle.offline` cuboids a
   trunk supervisor, i.e. the exact thing the label-free O1–O6 programme is constructed around.
2. **It would destroy the head's own meaning.** A readout that TRAINED its input can no longer
   answer *"what does the latent already carry"* — the §1.10 latents-only discipline P8 states
   in its own docstring. The measurement and the isolation are the same requirement here.

**Consequences, all implemented:**

* A **new module group `interp`** — deliberately **not `aux`**. `aux` **may** backprop into the
  encoder by the X3 matrix (O3/O6 are label-free trunk losses and that is their job). Filed
  under `aux`, the matrix would have **permitted** the edge and `assert_isolation` would have
  reported a **pass over a real violation**. That is the whole argument for the new group.
* A new matrix row: `ISOLATION_MATRIX["interp"] = ("interp",)`.
* A **fourth probed edge, `perception_to_trunk`**, whose forbidden set is *every parameter that
  is not `interp`* — the matrix row written out. It is added **only when the head is built**:
  a probe over an absent module reports zero violations and has established nothing, and it
  keeps the existing exact-three-key assertions in `test_v6_ladder_edges.py` true.
* ⛔ **The edge can FAIL, which is what makes it a check.** `isolate_interp_from_encoder=False`
  is the deliberately mis-wired control arm (CLI `--no-isolate-interp`), on the same list as
  `--no-isolate-planner`/`--no-isolate-uplink`. **MEASURED:** it raises `IsolationViolation`
  with 19 live encoder/readout parameters at the toy geometry, while the other three edges stay
  at zero — one mis-wire, one lever. Without it, `perception_to_trunk` would be a guard that
  cannot fail (the C13 family).
* `assert_isolation` **passes** with the head on, at both `slot_src` arms, non-vacuously
  (`n_probed["perception_to_trunk"] = 188` at the toy geometry).
* The declared `interp_side` surface is proved **TOTAL** — every `interp` parameter is reachable
  from it — so a field added to the head without declaring it fails the test rather than
  escaping the probe (the `intent_proj` defect, moved into the audit).
* And the claim is proved **through the real loss, not only the probe's synthetic reduction**:
  backprop `slot_set_loss` on a full stack and assert the set of parameters that received
  gradient contains **nothing outside `interp`**.

### 1.3 Which stage trains it — ⛔ none of them, and that is deliberate

⛔ **CORRECTED 2026-08-17 — the sentence that stood here was FALSE, and it is what made a real
defect look deliberate.** It read: *"`interp` appears in no `STAGE_GROUPS` entry except S-J's
(which is `MODULE_GROUPS` by definition)"* — treating S-J's inclusion of `interp` as a harmless
consequence of an alias. It was not harmless. MEASURED at production geometry with
`agent_slots=True`: `apply_stage_freeze(·, "S-J")` marked **all 62 tensors / 3,207,445 parameters
trainable while the S-J loss reached exactly 0** — i.e. precisely the lie this very paragraph goes
on to say the design avoids, sitting inside the paragraph disclaiming it.

⚠️ **"By definition" was a RESTATEMENT OF THE ALIAS, NOT A REASON**, and two tests had hardened it
into an assertion (`test_v6_staged.py` pinned `set(STAGE_GROUPS["S-J"]) == set(MODULE_GROUPS)`;
`test_v6_agent_slots.py` asserted `requires_grad is True` at S-J and excused it as *"the sole
exception BY DEFINITION"*). A defect defended by its own tests is the hardest kind to see.

⚠️ **HOW IT ARRIVED — a ONE-WAY INVISIBLE COUPLING, worth generalising.** `STAGE_GROUPS["S-J"] is
MODULE_GROUPS` (verified, identical `id`), and because `MODULE_GROUPS` is a **tuple** no edit could
ever *mutate* one through the other — so the alias looked safe. But commit `06b8782` appended
`interp` to `MODULE_GROUPS` and thereby **changed what S-J trains without touching the line that
declares S-J**. ⇒ *An alias makes one edit silently rewrite a declaration somewhere else; tuple
immutability protects against mutation, not against meaning.*

⇒ **NOW TRUE, and derived rather than asserted:** `LADDER_UNTRAINED_GROUPS = {"interp"}` and
`"S-J": tuple(g for g in MODULE_GROUPS if g not in LADDER_UNTRAINED_GROUPS)`, with
`stage_trainable_groups()` raising on violation. Derived, so a future group flows into S-J
automatically — hand-spelling the seven names would have fixed today's bug and installed
tomorrow's. MEASURED after the fix: `interp` trainable = **0 in all four stages**, every other
group bit-identical, default build unchanged at 87,893,449 params / 405 keys.

The reasoning below stands and is why the fix takes the shape it does:

`interp` must appear in **no stage's trainable set**. The v6 training batch carries
`frames/actions/poses/future_*` and **no agent
labels** — the episode contract has no agent tracks (`tanitad/data/_contract.py`; `grep obstacle
tanitad/data/physicalai.py` → zero). Listing it as trainable in a stage whose loss never reaches
it would report a module as "training" while it receives exactly zero gradient — the same lie
`V6LossWeights.for_stage` zeroes its planner terms to avoid. **No new `V6LossWeights` term was
added, on purpose.**

The head is trained by a **frozen-trunk probe in the P8 idiom**
(`scripts/train_p8_occupancy.py` is the template: frozen trunk via
`tanitad.eval.v6_probe_trunk.load_trunk_auto`, labels via `JoinFileReader`). It lives in the
`state_dict` so the checkpoint ships the interpretation head with the model, and
`STAGE_MAY_INTRODUCE["S-T"]` is what lets a later stage **carry** it over an S-W checkpoint that
never had it.

⚠️ **This sharpens what an allowlist entry means**, and the test docstring now says so: an
entry is about **KEYS ARRIVING**, not about the stage optimising the module. `fallback.`
(0 trainable params) was already this shape; F-18 is the second case, which is when a pattern
should be written down.

### 1.4 The declared arm and its control

⛔ **AMENDMENT ADOPTED 2026-08-17 — THE CONTROL LIST BELOW WAS INCOMPLETE, AND ONE SEED IS NOT A
MEASUREMENT.** Both were filed by the parity re-run **before any arm on that corpus was fitted**
(ordering evidence in `…/incoming/2026-08-17-slot-probe-parity/PREREG_AMENDMENT_EPISODE_IDENTITY.md`);
they are adopted here rather than left in a sibling package.

**1. EPISODE IDENTITY was uncontrolled, and C-SHUF is structurally blind to it.** Within-episode
gap SD is only **3.9 m**, so a head that merely recognises *which episode it is in* and emits that
episode's mean could clear C-CONST while perceiving no agent. **C-SHUF permutes WITHIN episode, so
the episode mean survives it intact** and the control reports "no echo" while an echo is exactly
what is happening. ⇒ **C-EPMEAN** (beat the episode's own mean) and **C-SHUF-XEP** (permute episode
identities) are now required.
⚠️ **This does NOT retroactively threaten the 2026-08-16 D1**, and the re-run checked rather than
assumed: re-scored against the new control, that head was **+13.224 [8.846, 17.314] WORSE than the
episode-identity ceiling** — it exploited nothing. The gap only ever bites a POSITIVE result.

**2. ⛔ A SINGLE PROBE FIT IS NOT A MEASUREMENT.** MEASURED: **three seeds on one frozen latent
cache span 1.826 m of K1 — LARGER than the 1.727 m spanned by five checkpoints across 9,250
training steps.** ⇒ Every F-18 number published to date, 2026-08-16's included, is single-seed with
**unmeasured reproducibility**, and the apparent "progressively discarded" trajectory
(5.98 → 5.44 → 5.58 → 6.15 → 7.17 m) is **flat within fit noise**.
⭐ **And the estimator cannot catch this**: the episode-cluster bootstrap resamples **eval
episodes**, not **fits** — so three tight, non-overlapping intervals differed only by seed. *An
interval quantifies the uncertainty it was built to quantify, and is silent about every other
one.* ⇒ **≥3 seeds required**, and between-condition differences must be compared against
**between-seed spread**, not against the bootstrap alone (~13 min/point, no trunk compute).

`slot_src` is a **pre-registered arm with its control**, because a null result is otherwise
unattributable:

| arm | memory | what a failure means |
|---|---|---|
| `"cells"` (default) | the readout's 16 spatial cell tokens — the surface every other v6 probe reads (`V6Stack.cells`; O2/O3 act on exactly these) | the LATENT does not carry agents |
| `"tokens"` | the encoder's raw patch tokens | the ENCODER does not carry agents |

If `cells` fails and `tokens` succeeds, the agents **are** visible to the encoder and the
**readout** destroys them — a finding about the geometry firewall (RC1's *"lead geometry lives
in these cells and dies in aggregation"*), not about the world model. Without the control a
failure is unattributable between "cannot see agents" and "the 4×4 grid cannot carry them" —
the C6 confound one floor down.

#### ⛔ 1.4b THE CHECKPOINT IS PART OF THE ARM, AND THIS PRE-REGISTRATION OMITTED IT (PI, 2026-08-16)

The PI's question — *"are you aiming to train the two arms on the latent of one of the
checkpoints?"* — exposes a gap: §1.4 pins WHAT each arm reads and says nothing about **WHICH
TRAINED STATE it reads from**. A frozen-trunk probe measures *that checkpoint's* latent, not "the
architecture", so an unstamped result is unattributable in a second dimension the control above
does not cover.

⛔ **THE BINDING ADDITION: no slot-probe number is admissible without its trunk checkpoint and
step.** Quote it as *"cells R = x at `<run>`@`<step>`"* or do not quote it.

⚠️ **AND THE OBVIOUS CHECKPOINT IS THE WRONG ONE.** The only v6 trunk in existence is the LIVE
`v6F-SW-30k` run, currently ~step 8,900 of 30,000 — a **world model that is 30 % trained**. A null
`cells` result there would confound the pre-registered claim *"the LATENT does not carry agents"*
with *"the latent has not finished learning"*. That is the same shape as the C6 confound this
section already guards against, displaced from the READ SURFACE onto the TRAINING STATE.

⇒ **The primary read is the FINAL S-W checkpoint (30 k)**, which is also the natural point
because `STAGE_MAY_INTRODUCE["S-T"]` exists precisely to carry this head over an S-W checkpoint at
the S-W→S-T boundary.

⭐ **AND THE CHEAP UPGRADE: run it at SEVERAL S-W checkpoints, not one.** The trunk is frozen and
the probe is small, so each extra point costs a probe fit and no trunk compute. A single point can
only say *"agents are/are not decodable at 30 k"*; a trajectory says whether agent structure
**emerges, plateaus, or is progressively discarded** as the world model trains — which is the
actually interesting question, and it converts a possible null into a measurement either way. Any
checkpoint before 30 k is stamped **EARLY-READ** and may not be quoted as the headline.

### 1.5 The label path — ⛔ the existing one, not a second one

Every target comes from the join `build_obstacle_join.py` writes and
`train_p8_occupancy.JoinFileReader` reads: `agents_to_array`'s
`[A, 6] = (cx, cy, yaw, l, w, occ)` rows plus the reader's per-agent `cls` column.
`targets_from_join` is a pure re-shaping — it opens no file and re-derives no geometry — and the
test proves it by writing a real join file, reading it back **with `JoinFileReader`**, and
building targets from what *that* returns.

Two label facts that had to be handled rather than assumed:

* ⛔ **`obstacle.offline` carries no velocity column** (MEASURED, join doc §1). So the diagram
  cell's *"state v, yaw-rate"* is **derived**, and the derivation is a decision:
  **the rates are EGO-FRAME RELATIVE** — `v_rel_x/v_rel_y` are d/dt of the agent's own
  `(cx, cy)` and `yaw_rate_rel` is d/dt of its `yaw`, by central difference over the join's own
  `t_s`, matched by `track_id`. Three reasons, in order of weight:
  1. it needs **only** the join, whereas an absolute ground-speed target additionally needs the
     egomotion poses composed per frame — a **second derivation**, i.e. the parallel label path
     the discipline forbids;
  2. it is what the LONGITUDINAL family consumes: closing speed is `-v_rel_x` and
     `TTC = cx / max(-v_rel_x, ε)`, with no ego-speed term to supply;
  3. it is what a monocular sequence **shows** (looming). An absolute target asks the head to
     infer ego speed and add it, and its failure would be unattributable between "cannot see the
     agent" and "cannot see its own speed".
  ⚠️ An unobserved rate is **MASKED, never zero-filled** — zero is a legitimate value (a
  stationary car) and filling it would teach the head that unseen means still. Yaw differences
  are wrapped before dividing, or a ±π crossing manufactures a ~63 rad/s spike from a car
  driving straight (tested).
* ⛔ **`occ` IS the field-of-view mask.** MEASURED 2026-08-16 (registry P4 predicate stamp):
  `visibility_occ` and `bev_raster.fov_mask` are the SAME PREDICATE — 0 of 7,680 cells disagree
  at every half-angle tested, defaults bit-identical IEEE doubles. So the `occluded` channel
  means **out of the front camera's field while the track continues** — the sharpest available
  form of the P4 question — and it must never be reported as generic object-object occlusion.

Two states that must not be conflated (join doc §4): a frame **absent** from the join is
`NO_LABEL` and `targets_from_join(None)` **raises**; an **empty** agents list IS a label
(labelled clear) and is a valid, scored target. Tested both ways.

### 1.6 Design details that are decisions, not defaults

* **Yaw is a `(sin, cos)` pair, not a scalar.** A scalar regression is discontinuous at ±π and
  its L1 is not a metric on the circle. The loss is `1 − cos Δ` on the unit pair; tested that a
  slot at −π against a target at +π scores **exactly 0**.
* **No `tanh` anywhere.** MEASURED 2026-08-15 in fp32: `d/draw tanh(raw)` is EXACTLY 0.0 from
  `raw ≥ 10`, and this programme has a gnorm-354,076 spike on the record. A saturating
  coordinate head cannot recover a far agent. Coordinates are emitted normalised and scaled by
  `bev_raster.GRID_DEFAULT`'s extents (60 m fwd, ±16 m — the **P8 field**, shared so the
  perception heads do not drift apart); sizes go through `softplus`.
* **The matcher is exact and dependency-free.** `hungarian` is the classic O(n²m) rectangular
  assignment in numpy, because `scipy` is **not** a core dependency (`pyproject.toml`: torch +
  numpy) and every stack use of it is a lazy import inside a *script*. The duplication is
  admissible **only with the equivalence proof**: MEASURED **0 mismatches over 200 random and
  tied matrices** against `scipy.optimize.linear_sum_assignment`, compared on the **optimal
  cost** (ties have several optimal assignments; pinning one would pin an implementation detail
  rather than the answer). Same contract `bev_raster.yaw_from_quaternion` carries against
  `physicalai.quaternion_yaw`.
* **Over-full frames drop their FARTHEST targets and COUNT the drop** (`n["dropped"]`), never
  silently — a head that quietly stops being scored on crowded frames would report its best
  numbers exactly where driving is hardest.
* **Every loss term is returned separately with its own unit and its own count** (metres · nats ·
  m/s · rad/s · dimensionless). A term computed over zero items reports `0.0` **with** `n = 0`
  rather than vanishing. Pooling them into one score would be the ADE-only failure in a
  detector's costume.
* **`n_agent_slots` and `n_slot_queries` must agree when `goal_cat_args` is on** — refused at
  build time. The categorical `agent_slot` arg that four `g_tac` tokens index (`GAP_TARGET`,
  `YIELD_AT`, `WAIT_FOR_ONCOMING`, `EVADE_IN_CORRIDOR`) **indexes the slots this decoder emits**;
  two cardinalities make an emitted index that refers to nothing — the type error that channel
  exists to remove.

---

## 2. ⚠️ UNVERIFIED / UNMEASURED — what this document does not know

| item | state |
|---|---|
| `n_slot_queries = 16` | ⚠️ **A DECLARED PLACEHOLDER, NOT A FITTED VALUE.** The right number is the join's measured per-frame agent-count distribution (its 99th percentile). **No join file exists in the repo** (the artifacts are pod-side), so it is UNMEASURED here. Measure with `JoinFileReader` before the run and record it. |
| a **TRAIN-corpus** join | ⛔ **BLOCKER, not yet built.** `build_obstacle_join.py`'s documented invocation builds a **val40** join. Training needs the join over the train corpus (`physicalai-train-e438721ae894`, 2376 episodes, skip-hash `f09e44db`). This is a CPU-only pod-side step and must NOT run on a training pod. |
| training cost | §7's *"~1 GPU-day"* is **INHERITED** from `DIAGRAM_CONFORMANCE.md`, not re-derived. My own **ESTIMATED** read: the frozen-encoder forward dominates and is **cacheable once** (the P8 idiom already does this), which should make it hours rather than a day at 3.2 M trainable params. ⛔ Neither figure is MEASURED — take a 50-step timing before committing the run. |
| absolute error scale | No prior exists for lead-gap error from a latent readout. This is precisely why §4's criteria are **relative**, not absolute (see the P8 lesson in §4.0). |

---

## 3. Which of the FOUR METRIC FAMILIES this head serves

⛔ **Detection quality (mAP, IoU, recall) is not one of the four families and will NOT be
reported as the headline.** It is an internal diagnostic. The four families are the yardstick.

| family | served? | how, concretely |
|---|---|---|
| **LONGITUDINAL** | ✅ **directly, and this is the point of the head** | 88.7 % of the oracle gap is longitudinal, and `four_families.longitudinal` reports its **distance-keeping half UNAVAILABLE** without a `lead` dict — which today can only be built from the **privileged** `obstacle.offline` label. This head is the **vision-only source of that dict**: the lead slot's `cx` gives headway, `cx/v_ego` the time-gap, and `cx / max(-v_rel_x, ε)` the min-TTC, all through the existing `taniteval.lead_metrics.distance_keeping` (keys `headway_min_m`, `time_gap_min_s`, `min_ttc_s`). Scored as §4. |
| **TACTICAL** | ✅ **indirectly, and it is the enabling condition** | Four `g_tac` tokens carry a categorical `agent_slot` arg that today **indexes an empty set** — there is no module that populates it, so `GAP_TARGET`/`YIELD_AT`/`WAIT_FOR_ONCOMING`/`EVADE_IN_CORRIDOR` are emittable but not *referential*. This head is the referent. Scored as **slot-referent agreement**: does the slot the goal head selects correspond to the GT lead / yielded agent from the join, with the confusion over the four agent-referencing tokens reported per class. ⚠️ That score is only computable once S-T has trained a goal head **and** the arg channel is on; it is a follow-on, not part of §4. |
| **LATERAL** | ❌ **not served** | The head emits agent geometry, not ego path. Heading / curvature / yaw-rate / cross-track are ego quantities. Saying so is the rule ("per family with the reason"), not an omission. |
| **STRATEGIC** | ❌ **not served** | Route/goal setting is `layer_str`'s. `obstacle.offline` has **no map, lane graph, junction, roundabout, traffic-light or route feature** — 10 classes, all dynamic agents — so nothing strategic can come from this label at all. |

---

## 4. ⛔ THE PRE-REGISTRATION — both outcomes committed IN ADVANCE

### 4.0 The form of the criterion, and why it is relative

**MEASURED precedent that dictates this (registry, P8 attempt 2, 2026-08-12):** a ~1 M-param
readout on frozen latents against sparse targets scored **absolute IoU 0.01869**, and the
registry's own stamp is that *"the admissible claim is the RETENTION RATIO (one instrument, two
inputs), not the absolute occupancy quality"*. The same shape binds here: a lead-gap MAE in
metres from a 3.2 M head on frozen latents has **no interpretable floor**, so an absolute
threshold committed today would be a number invented, not a criterion.

⇒ **Every criterion below is a PAIRED DELTA against a control run through the SAME instrument.**

### 4.1 The primary metric

**Lead-gap error.** From the slot set, the predicted lead is the highest-`presence` slot with
`cx > 0` and `|cy| ≤ 1.75 m` (an in-corridor rule fixed **now**, and applied identically to the
GT agents from the join so the two sides cannot drift). Then:

* **PRIMARY: `lead_gap_abs_err_m`** = `|ĉx_lead − cx_lead|`, metres, per window.
* **SECONDARY: `lead_ttc_abs_err_s`** = `|T̂ − T|` with `T = cx / max(−v_rel_x, ε)`, seconds.
* **SECONDARY: `lead_presence_recall`** — fraction of windows with a GT in-corridor lead inside
  30 m for which the head emits one. The operating point (presence threshold) is chosen on the
  **ENCODED** arm — the conservative side, the P8 τ\* discipline — and then **frozen** for every
  other arm.

Windows with **no GT lead** are reported as their own stratum with their `n`, never folded into
the mean.

### 4.2 The estimator — binding

* **`taniteval.ci.paired_episode_cluster_bootstrap`**, `n_boot = 2000`, clustered on the **40
  val episodes**, paired on the **same windows** for every arm-vs-control comparison
  (`taniteval/taniteval/ci.py`).
* Unpaired single-arm intervals: `episode_cluster_bootstrap` from the same module.
* ⛔ **`overlapping_holdout_se` is FORBIDDEN.** It is neither a jackknife nor a valid SE, and it
  **biases the point estimate**: MEASURED over 27 arms, headline shifts **−6.67 % to +11.69 %,
  bidirectional**, up to **×−4.15 on paired deltas including a SIGN FLIP**. Never combine two
  intervals in quadrature; use the paired form.
* "Separated" means the 95 % CI of the paired delta **excludes 0**.

### 4.3 The controls, all run on the same windows

| id | control | what it isolates |
|---|---|---|
| **C-CONST** | predict the corpus-median lead gap for every window | a head that cannot beat a constant has measured nothing |
| **C-SHUF** | the trained head, same weights, latents **permuted across windows within the episode** | the anti-echo: proves the number comes from **this** window's latent and not from a corpus prior absorbed into the head |
| **C-TOK** | the same head geometry on `slot_src="tokens"` | separates "the latent does not carry it" from "the readout destroyed it" |
| **C-V5F** | the same head geometry on the **v5f** trunk's latents | §4.2 records lead state as *absent in v5f (measured)*; this asks whether O2/O3/O4 bought it |

### 4.4 ⭐ KEEP — the head is admitted and F-18's diagram cell flips to ✅

**ALL THREE must hold, on the `cells` arm:**

* **K1** `lead_gap_abs_err_m` is **separated-better than C-CONST** (paired CI excludes 0,
  favourable sign).
* **K2** `lead_gap_abs_err_m` is **separated-better than C-SHUF**.
* ⛔ **K3 — WITHDRAWN AS SPECIFIED 2026-08-17. IT CANNOT FAIL.** It read
  *"`lead_presence_recall ≥ 0.50` at the ENCODED-arm operating point"* — but τ\* is the **median**
  presence score, so recall is pinned at **≈0.50 by construction** and the threshold sits exactly
  on the value the definition guarantees. **MEASURED: a head trained on PURE NOISE scores 0.5002
  and PASSES.** ⇒ A criterion that a random head satisfies is not a criterion; it is decoration
  that makes a gate panel look stricter than it is (the C13 family — a guard structurally unable
  to report failure). **It must be re-specified before it is quoted again**, and no past K3 pass
  may be cited as evidence of anything.
  ⚠️ Note this is the SECOND vacuity found in this same criterion: the 2026-08-16 run already
  caught it returning **0.998** because the emission ignored τ, fixed it to be τ-gated — and the
  τ-gated version is the one now shown to be pinned at 0.5. *A metric repaired once is not
  thereby correct; it was repaired to a different broken state.*

⚠️ **And a fourth clause that decides what "keep" MEANS**, committed now so the head cannot be
promoted by momentum:

* **K4** if K1–K3 pass **and** the median `lead_gap_abs_err_m` is **smaller than 0.9769 m** —
  the MEASURED Δ headway of the D-LEAD-1 GT-vs-CV admission control, i.e. the size of the
  effect the LONGITUDINAL metric is known to be able to see — the head is admissible as an
  **inference-time lead source** and a T1 pre-registration follows.
* If K1–K3 pass but K4 does **not**, the head is admitted as a **T0 DIAGNOSTIC ONLY**. It goes
  in the registry as an instrument, it does **not** feed a planner, and the claim is *"the
  latent carries lead geometry, at an accuracy below the planner's requirement"* — which is a
  real finding and not a disappointment.

### 4.5 ⛔ DROP / RE-SCOPE — committed in advance

* **D1 — K1 fails on BOTH `cells` and `tokens`.** The encoder does not carry agent geometry.
  ⇒ The head is **DROPPED** as a readout. F-18's cell stays ⬜, now with a MEASURED reason
  instead of an absence, and the work item becomes an **encoder-objective** question (a
  supervised detection branch, or an O-measure that makes agents predictable) — which needs its
  own pre-registration and must not be smuggled in as "tuning the head".
* **D2 — K1 fails on `cells`, PASSES on `tokens`.** The **readout grid is the bottleneck**.
  ⇒ Do **not** keep the head on cells and do **not** quietly switch the default to `tokens` —
  the readout is the geometry firewall and `d_op` is derived from it, so widening the grid moves
  the whole state width, the param budget, and every banked comparison. The work item is a
  readout-geometry pre-registration with a matched-param control.
* **D3 — K1 passes, K2 fails.** The head is reading a corpus prior, not the window. ⇒ **DROP
  the number entirely.** This is the echo family (nav-echo 1.0000; T1 action echo 97.9 %
  open-loop / 0.0 % hold-action; P1 speed echo R² 0.995 → −0.72) and it has cost this programme
  a published claim more than once.
* **D4 — the mis-wired arm is needed to make it work.** If the head only reaches K1 with
  `isolate_interp_from_encoder=False`, that is **not a result** — it is a perception label
  training the trunk, which the binding diagram forbids and which also invalidates the
  measurement (a readout that trained its input cannot say what the latent carried). ⇒ Report
  the fact and DROP.

### 4.6 The tier stamp

⛔ **Everything in §4 is T0-DIAGNOSTIC.** It is a frozen-latent readout: a WM diagnostic, never
"driving performance". A claim that this head **improves driving** requires it wired as a
planner input and evaluated at **T1** (`taniteval/tools/t1_eval.py`), as a separate later
pre-registration.

⚠️ **And when that day comes, the goal-input rule bites** (PI 2026-08-03): a slot-derived lead
state may feed a LONGITUDINAL cost, but the admissibility check must be run — *"could this have
been computed from the situation classifier's output?"* The slot decoder reads only pixels, so
the answer is no by construction **as long as nothing else is concatenated into it**. That is
the property the one-tensor signature protects, and it must be re-checked at wiring time.

---

## 5. What would run, and what it would cost

**Not run here** (Thor is the only GPU and it carries the 30k S-W). The recipe, for whoever
runs it:

1. **Pod-side, CPU only, never on a training pod:** build the join for the TRAIN corpus with
   `build_obstacle_join.py` (the val40 invocation in its header is the template). Record the
   per-frame agent-count distribution and set `n_slot_queries` to its 99th percentile.
2. **Cache the frozen trunk's latents once** for the train and val windows (the P8 idiom). This
   is what turns the estimate from a GPU-day into hours; it is also what makes C-SHUF and C-V5F
   cheap, since they are re-reads of a cached tensor.
3. Train the head on the cached latents (`interp` group only; everything else frozen — enforced
   by construction, not by a flag).
4. Score §4 on the **40 val episodes** with `taniteval.ci.paired_episode_cluster_bootstrap`.
5. Attach the LONGITUDINAL family through `taniteval.lead_metrics.distance_keeping` with the
   head's own lead in place of the label-built one, and report **both** so the gap between the
   vision-only lead and the privileged lead is the headline number.

**Cost: ESTIMATED, not measured.** §7's *"~1 GPU-day"* is INHERITED. Take a 50-step timing first.

---

## 6. Escalations

1. ⛔ **`STAGE_MAY_INTRODUCE["S-T"]` and `MODULE_GROUPS` both grew.** The allowlist test pins the
   tuple EXACTLY (by design) and its docstring was extended with the reasoning. Any agent
   holding a stale copy of either will conflict — **integrator, please sequence.**
2. ⚠️ **The train-corpus join is a real blocker** (§2) and it belongs to whoever owns the
   pod-side data steps, not to this stream. It is a CPU-only job.
3. ⚠️ **`DIAGRAM_CONFORMANCE.md` line 183 and its F-18 row are now stale** — the module exists.
   I did not edit that document (it is another stream's deliverable); the row should move
   ⬜ NOT BUILT → 🟨 PARTIAL (built, untrained, pre-registered).
4. ⚠️ **The stated test baseline is stale.** See §7.
5. ⚠️ **The chain does not enforce the carry rule for this flag, deliberately.** If an S-T run
   is ever launched with `--agent-slots`, its checkpoint carries `agent_slots.*` and S-S/S-J
   must carry the flag too, or `load_stage_init` is fatal on unexpected keys.
   `v6_chain.assert_geometry_carry` catches exactly this for `--selector` and `--tac-goal-cond`
   **from a JSON read before the corpus mounts**, but it enumerates its levers first-class and
   there is no `Step.agent_slots` — because no chain step sets the flag (no ladder stage trains
   the head). Plumbing it is the follow-on the moment a chain step wants the head; the mechanism
   is recorded in `STAGE_MAY_INTRODUCE`'s comment so it is not rediscovered at 3 a.m.
6. ⚠️ **Another agent holds `stack/tanitad/models/v6.py` and `stack/scripts/train_v6_staged.py`
   in the worktree `.claude/worktrees/beautiful-wu-ce02f3/`.** Both files are edited here.
   **Integrator: expect a conflict and sequence the merge.**

---

## 7. ⚠️ THE TEST BASELINE IS NOT WHAT THE BRIEF SAYS — measured, not assumed

The brief gave the baseline as **3572 passed / 0 failed / 7 skipped / 2 xfailed**. MEASURED on
this working tree **before any edit of mine** (`pytest -q`, 422.68 s):

> **9 failed, 3563 passed, 7 skipped, 2 xfailed** — 3572 **collected**, not 3572 passed.

The nine are other streams' in-flight work, not regressions:

```
tests/test_bev_consumer_fov.py::test_figure_main_never_writes_outside_its_output_dir
tests/test_bev_consumer_fov.py::test_figure_caveats_the_missing_frame_when_none_is_recorded
tests/test_e_wc2_sigma_star.py::test_cli_print_contract
tests/test_ff_v58f.py::test_tool_REFUSES_the_biased_estimator_by_name
tests/test_ph0_sam3.py::test_liveness_probe_calls_a_dead_engine_dead
tests/test_ph0_sam3.py::test_run_clip_frames_banks_the_alarm_a_structural_check_would_miss
tests/test_ph0_sam3.py::test_run_clip_frames_distinguishes_an_empty_scene_from_a_dead_engine
tests/test_ph0_sam3.py::test_live_is_ANY_control_not_ALL_because_sky_can_be_occluded
tests/test_ph0_sam3.py::test_a_dead_engine_is_still_dead_under_the_any_rule
```

Five are `ph0_sam3`, which the brief names as a third agent's live file and which I did not
touch. ⇒ **The admissible statement about my change is a DELTA against that measured baseline,
not "the suite is green"** — quoting an inherited baseline as if it were current is exactly the
class this programme keeps retracting.

### 7.1 AFTER — and every difference accounted for

MEASURED after the edits (`pytest -q`, 422.55 s):

> **4 failed, 3643 passed, 7 skipped, 2 xfailed**

| difference | account |
|---|---|
| **failures 9 → 4** | The four that remain are a strict SUBSET of the baseline nine, in files I did not touch (`test_bev_consumer_fov` ×2, `test_e_wc2_sigma_star`, `test_ff_v58f`). |
| **five failures DISAPPEARED** | All five were in `tests/test_ph0_sam3.py`. ⚠️ Per C80 a failure that becomes a pass is a finding and must be attributed, not enjoyed: `stack/scripts/ph0_sam3.py` (mtime **19:50:38**) and `stack/tests/test_ph0_sam3.py` (**19:51:19**) were modified DURING this session by the agent that owns them (the brief names it), and both were already staged `M ` in git. Nothing I changed can reach that module. |
| **passed 3563 → 3643 (+80)** | **+41** are this stream's new `tests/test_v6_agent_slots.py` (verified by running that file alone: 41 passed). **+5** are the ph0_sam3 failures now passing. **+34** are new tests the same agent added to `test_ph0_sam3.py`, which now collects **33** (MEASURED by `--collect-only`) against a much smaller baseline. |
| **skipped 7, xfailed 2** | UNCHANGED. In particular the byte-identity proofs did **not** skip — git produced a pre-change revision and the per-tensor comparison really ran. |

⇒ **My change adds 41 passing tests and zero failures.**

---

## Deliverable manifest

| artifact | where |
|---|---|
| the module | `repo:stack/tanitad/models/agent_slots.py` (new) |
| v6 wiring | `repo:stack/tanitad/models/v6.py` (modified) |
| trainer wiring | `repo:stack/scripts/train_v6_staged.py` (modified) |
| tests (41) | `repo:stack/tests/test_v6_agent_slots.py` (new) |
| allowlist pin + reasoning | `repo:stack/tests/test_v6_stage_init_introduction.py` (modified) |
| this pre-registration | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-agent-slot-decoder/AGENT_SLOT_DECODER.md` (new) |

*Nothing lives on a pod or in a worktree. Every artifact is in the working tree and staged on
`agent/arch-inf-20260803`.*
