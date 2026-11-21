# The goal head repeated the programme's oldest defect one level up — FIXED IN STRUCTURE

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **HEAD at start `37220d2`**
**Cost: 0 GPU.** ⛔ **Thor was not contacted** — no ssh, no disk read, no job. v6F S-W was training
throughout and nothing here touched it.
**Tier:** this file is **MECHANISM, not measurement**. Every parameter count below is
**MEASURED (ours, on this box, torch 2.11.0+cu128, seed 0)**; every σ / Δ / CI is **INHERITED**
from `…/incoming/2026-08-16-anchor-goal-supervision/` and is re-quoted with its own stamp.
**No number produced by calling this code is quotable as a capability claim** — those come from
T1 (`taniteval/tools/t1_eval.py`) with its estimator and its four metric families.

---

## 0. The one-line answer

> ⭐ **`g_tac` is now FACTORED LAT × LON, the anchor objective is REGRESS-THEN-SNAP by default with
> the refuted one-hot kept reachable as its control, the arg channel is TYPED, and the head can emit
> a PAIR — and the default build's `state_dict` is BYTE-IDENTICAL to the pre-change architecture,
> proved per tensor with `torch.equal` against the last revision of `v6.py` that predates the change,
> with a NEGATIVE CONTROL showing the guard fires.**
>
> ⛔ **And the blocker is now enforced rather than documented: NO 6 s anchor vocabulary exists, so
> the factored head is NOT EXPRESSIBLE AT 6 s TODAY.** The head **refuses** a 2 s table against the
> 6 s plan horizon instead of scoring one against the other. §5.

| # | brief item | state |
|---|---|---|
| **1** | byte-identity of the default build | ⭐ **DONE + negative-controlled** (§1) |
| **2** | lat × lon factoring, MEASURED param delta | ⭐ **DONE — +470,939 MEASURED** (§2) |
| **3** | regress-then-snap default, one-hot control | ⭐ **DONE** (§3) |
| **4** | categorical arg channel + multi-label emission | ⭐ **DONE — 2 of 9 → 9 of 9 expressible** (§4) |
| **5** | X3 isolation + admissibility unchanged | ✅ **MEASURED on 8 arms** (§6) |
| **6** | the 6 s expressibility question | ⛔ **BLOCKER, stated plainly and enforced in code** (§5) |

---

## 1. ⛔ THE BYTE-IDENTITY PROOF — and the reference that is NOT `HEAD`

**All five levers default OFF.** `goal_factored=False` · `goal_multilabel=False` ·
`goal_cat_args=False` · `anchor_goal="none"`. Every new module is constructed at the **very end of
`V6Stack.__init__`**, after `cand_score`, and **only when its flag is set** — so the default path
draws **no random numbers**, creates **no `state_dict` key**, and every earlier module's
initialisation is bit-for-bit what it was. That construction order is the whole mechanism; it is
also why `GoalVocabulary`'s categorical channel and `GoalHead`'s categorical output are
**lazily attached** (`attach_cat_channel` / `attach_cat_head`) rather than built in `__init__` —
building them in place would have drawn RNG **mid-stream** and moved every module after them.

**MEASURED, default build, unchanged:**

| config | params | state_dict keys |
|---|---|---|
| the small structural config | **611,293** | **223** |
| ⭐ the real `V6Config()` — what the live run uses | **87,893,449** | **405** |

Both were measured **on the unmodified file before the first edit** and re-measured after; they are
also pinned as literals in the test file.

### 1.1 ⚠️ THE REFERENCE HAD TO CHANGE, AND THAT IS A FINDING

The brief said to prove it "the way `test_all_off_is_byte_identical_to_head` does". I wrote exactly
that — `git show HEAD:stack/tanitad/models/v6.py`, import side-by-side, compare per tensor — and it
**SKIPPED with "v6.py matches HEAD byte-for-byte"**.

**MEASURED cause: HEAD moved from `37220d2` to `a558b79` while this file was being written, and the
commit swept in the in-progress `v6.py`** (`git show HEAD:…v6.py | grep -c AnchorGoalHead` → **6**).
That is the `CLAUDE.md` "git commit commits the ENTIRE INDEX" hazard, and its consequence here is
sharper than a messy commit: **a HEAD-relative identity test becomes a module compared with itself —
a test that passes by construction and establishes nothing.** It would have gone green forever.

⇒ **The guard now walks `v6.py`'s OWN history for the newest revision that does not yet carry the
new flag** (`goal_factored`), and compares against that. Found: **`b12c190`**. This reference is
stable no matter how many commits land afterwards, and it is the semantically correct one — it *is*
the architecture the live S-W checkpoint was built from.

**Root-cause class (for `RETRACTION_LOG.md`, §9):** *a guard whose reference is a MOVING pointer
silently degenerates into a self-comparison.* Same family as C72 (a hash over a non-canonical
container) and as the sibling agent's *"isolation check whose scope is the SESSION, not the
SUBJECT"*: the check runs, reports green, and has measured nothing.

### 1.2 ⭐ THE NEGATIVE CONTROL — the guard was made to FAIL, twice

Because a byte-identity test that has never failed is a test whose sensitivity is unknown.

| injected into the DEFAULT path | `params`/`keys` check | flag-flip check | ⭐ per-tensor vs `b12c190` |
|---|---|---|---|
| a registered `nn.Linear(3,3)` (adds a key) | ⛔ **fires** | ✅ passes | ⛔ **fires** |
| ⭐ a **discarded** `nn.Linear(3,3)` — consumes ONE RNG draw, registers **nothing** | ✅ **PASSES** | ✅ **PASSES** | ⛔ **fires** (`masked_cells.pos MOVED`) |

⇒ **the count checks are not a substitute for the per-tensor comparison**, and the second row is
exactly the failure mode that would break a strict resume while every cheaper check stayed green.
Both injections were reverted and the file verified clean.

⚠️ Deliberately **not** used: a digest of a `torch.save` file. That container's bytes are not
canonical (RETRACTION_LOG **C72**) — it can differ for two identical `state_dict`s and agree for two
different ones.

### 1.3 The flag-flip invariant

For every arm below, **no pre-existing tensor moves** — the ON build is the OFF build **plus**
tensors, never the OFF build re-initialised. That is what makes a delta attributable, and it is the
same discipline `test_param_delta_is_exactly_267` enforces for the selector.

---

## 2. ⭐ THE LAT × LON FACTORING — and the MEASURED param delta

**`TACTICAL_GOAL_TOKENS` is partitioned by WHICH AXIS the token constrains**, and the partition is
total and disjoint (pinned by a test, not asserted here):

| axis | tokens |
|---|---|
| **LAT** | `ANCHOR_GOAL`, `CORRIDOR_OFFSET`, `EVADE_IN_CORRIDOR` **+ `LAT_UNCONSTRAINED`** |
| **LON** | `SPEED_BAND`, `GAP_TARGET`, `YIELD_AT`, `STOP_POINT`, `WAIT_FOR_ONCOMING`, `TRAFFIC_LIGHT_REACT` **+ `LON_UNCONSTRAINED`** |

3 + 6 = the nine, exactly. Each axis carries its **own abstain**, because a factored head that
cannot say *"this axis is unconstrained"* must invent a constraint on every window — §2's
*"Unset = unconstrained"*.

**It follows the `a_tac` idiom rather than inventing a second one** (pinned): same class
(`GoalHead`), same input (`z_tac_p`), same `d_in`, and the §5 *one vocabulary, two views* rule holds
by `id()` identity — `goal_head_tac_lat.vocab is cond_op_lat.vocab`. The one deliberate difference
from `act_head_lat/lon`: the **goal** pair is conditioned on `e_g_str` (goals flow **down**, §5)
exactly as the mixed goal head is, while the action heads are not.

⭐ **The mixed `goal_head_tac` SURVIVES and is still emitted.** It is this arm's **control** — a
comparison with no control is unattributable (C6) — and deleting it would break the live strict
resume. Only the **downlink** moves to the pair:
`e_g_tac = ½·(enc(g_lat) + enc(g_lon))`. The **mean**, not the sum: a factored embedding at twice
the mixed one's scale would change the conditioning magnitude the operative FiLM sees, and a scale
change between an arm and its control is a confound wearing an architecture's name.

### 2.1 ⚠️ MEASURED, NOT ARITHMETIC — the full `V6Config()`

*(`selector="mlp"`'s design-time estimate of +41,089 was never realised; the implementation cost
+33,801. A delta quoted from a shape calculation is an ESTIMATE wearing a measurement's stamp.)*

| arm | **Δ params** | Δ keys | total | pre-existing tensors moved |
|---|---|---|---|---|
| default | 0 | 0 | 87,893,449 | — |
| ⭐ **`goal_factored`** | **+470,939** | +46 | 88,364,388 | **0** |
| `goal_multilabel` | **+0** | +0 | 87,893,449 | **0** |
| `goal_cat_args` | **+110,880** | +5 | 88,004,329 | **0** |
| `anchor_goal="snap_lat"` (+cat) | +111,138 | +11 | 88,004,587 | **0** |
| `anchor_goal="snap_xy"` (+cat) | +111,138 | +11 | 88,004,587 | **0** |
| ⛔ `anchor_goal="onehot"` (+cat) | +143,904 | +11 | 88,037,353 | **0** |
| **everything on + `selector="goal"`** | **+804,104** | +71 | **88,697,553** | **0** |

**The anchor head alone: +258** (snap modes — `Linear(128→2)`) / **+33,024** (the one-hot control —
`Linear(128→256)`), on top of the categorical channel.
**Sub-300M invariant: 211,302,447 params of headroom remain** even with every lever on.

The factoring's cost is **the two heads and the two tables and nothing hidden in the seam**: both
conditioners project `d_embed → d_embed`, so `proj` is `nn.Identity` and holds **0** parameters
(asserted). `goal_multilabel` is **exactly 0 parameters and 0 keys** — the same `type_head` logits
read a second way; a second output layer would have made it a capacity change wearing an
expressivity change's name.

---

## 3. ⭐ THE ANCHOR OBJECTIVE — regress-then-snap by DEFAULT, one-hot as the CONTROL

**INHERITED, MEASURED 2026-08-16** (881 windows / 40 episodes, LOEO, paired episode-cluster
bootstrap): `snap` — the same ridge, rounded to the nearest anchor — is **NOT separated** from the
free ridge (**−0.0002 [−0.1031, +0.0703]** at K=256; **+0.0383 [−0.2125, +0.2338]** in the `v0`
regime), while the one-hot K-way classifier costs **+4.7502 [+3.0514, +6.3981]**, separated at every
K from 8 to 256, under both FPS and k-means, and replicating on REF-C-base (**+5.4570 [+3.8345,
+7.1073]**). ⇒ **quantisation is FREE; the one-hot TARGET is what costs.** Independently corroborated
on another surface by **E-OBJ-1** (one-hot CE → `softade` recovers **−0.0974 / −0.1670 m separated**,
and the recovery is **LONGITUDINAL**).

`AnchorGoalHead` implements that literally:

| mode | what it is | why it exists |
|---|---|---|
| ⭐ **`snap_lat`** (the default when on) | regress ĝ = (x, y); **quantise ONLY y** onto a lateral sub-vocabulary; **x stays a continuous progress arg** | §6.4's prescription: the variance is 98.8 % longitudinal while FPS quantisation is isotropic (0.5674 / 0.5599), so a joint K-way index spends half its resolution on the axis carrying **1.2 %** of the variance |
| **`snap_xy`** | regress then snap to the nearest 2-D anchor | the arm that was MEASURED FREE — the reference |
| ⛔ **`onehot`** | a K-way classifier, hard table lookup, **no differentiable point at all** | the pre-registered CONTROL. Metric-blind by construction, which is the property being controlled for |

**The snap is STRAIGHT-THROUGH** (`raw + (snapped − raw).detach()`), which is what makes
regress-then-snap a *trainable object* rather than a post-hoc rounding: the emitted point is
quantised while the gradient reaches the **continuous regression**, so the loss can stay
metric-aware instead of becoming the CE the measurement refuses. Pinned two ways — forward equals a
table row exactly; backward equals the free regression's gradient exactly.

⭐ **`snap_lat` is verified to quantise ONE axis:** the longitudinal coordinate is **bit-identical**
to the free regression (`torch.equal`), the lateral one is on the bin grid and demonstrably moved,
and it snaps to the **nearest** bin, not merely to some bin.

**The two estimators do not share parameters.** `snap_*` builds only the regression; `onehot` builds
only the classifier. Giving each the other's parameters would leave dead weight at random init in
both — the `intent_proj` defect. They are matched where it matters: **identical input (`e_g_tac`),
and nothing else.**

⚠️ **The lateral bins are an ESTIMATED construction, and are labelled as one**: equal-mass quantiles
of the anchor table's own lateral marginal. §7.1's pre-registered **E-AG4** is the experiment that
would settle whether an anisotropic construction beats the isotropic FPS one; until it runs this is
a construction, not an optimum.

### 3.1 The seam into selection — and a defect it exposed

The structured ĝ overrides `GoalDistanceScorer`'s free decode (a new optional `goal_point` argument;
default `None` keeps the incumbent path bit-for-bit). ⚠️ **The first implementation made
`cand_score.goal_point` UNREACHABLE from the declared planner surface** — a planner parameter that
no declared output reaches is invisible to the X3 probe, which is the `intent_proj` defect verbatim
(*a path present in the diagram, absent from the optimisation*). Caught by running
`test_planner_surface_is_total`'s logic against every new arm **before** writing the tests.

⇒ the scorer now **also returns `goal_point_free`** when overridden. Two payoffs: the parameter stays
declared and probed, **and** it hands E-AG2's paired **free-vs-structured** comparison on the same
window, in one forward, at zero extra parameters. It is zero-init, so an unfitted free decode reads
as exactly that rather than as noise.

⚠️ **The `"mlp"` capacity control is deliberately NOT handed the ready-made goal point** — that would
make it an *information* control and its result would stop speaking to capacity (§5.3). It keeps
reading `e_g_tac`.

---

## 4. ⭐ THE ARG-TYPE GAP AND THE PAIR — 2 of 9 expressible → 9 of 9

**The gap, MEASURED from the code rather than asserted from a doc** (pinned by a test that computes
expressibility from `GOAL_CAT_ARG_TOKENS` and the vocabulary's own slots): `arg_head` emits **8
floats**, `arg_proj` consumes **8 floats**, and seven of the nine tokens need a **categorical** arg.
Before the channel the expressible set was **exactly `{CORRIDOR_OFFSET, SPEED_BAND}`**.

**The channel:** `GOAL_CAT_ARG_NAMES = ("anchor_id", "lat_bin", "agent_slot", "reason", "state")`,
cardinalities `(n_anchors, n_lat_bins, n_agent_slots, 4, 4)`.

* ⚠️ **The five slot-valued args are ONE kind, not five.** `agent_slot_id` / `gap_slot` /
  `oncoming_slot` / `obstacle_slot` / `light_slot_id` are all *an id into the window's agent-slot
  vocabulary*, and — checked token by token against `HIERARCHY_VOCABULARY.md:84-92` — **no token needs
  two at once**, so one channel serves all five. Asserted, not assumed.
* ⭐ **`anchor_id` (joint) and `lat_bin` (factored) are DIFFERENT vocabularies** and a token uses one
  or the other, never both. Conflating them would be the very type error the channel removes.
* **One `nn.Linear(sum(cards), d_embed, bias=False)`** — the concatenated per-slot tables as a
  matmul, the same trick `embed_tokens` already uses, so a **hard one-hot and a soft posterior travel
  the SAME code path**. A separate "soft-only" path is how two conventions get invented.
* **Per-slot softmax, never one global one** — one softmax over the concatenation would make picking
  an `anchor_id` *compete with* picking a `reason`. Pinned.
* **Unset slots contribute EXACTLY zero** via a `cat_usage` mask derived from the token actually
  emitted — §2's *"Unset = unconstrained"*, the same IGNORE discipline the continuous slots follow.
  The soft mask (`probs @ cat_usage`) **coincides with the hard one at a one-hot** (pinned: a
  differentiable generalisation must not be a second convention), and is **clamped to [0,1]** because
  multi-label gates are not a simplex — two emitted tokens that both use `agent_slot` would otherwise
  give that slot a mask of 2 and silently amplify its embedding. *A mask says whether a slot is set;
  it is not a weight.*
* `cat_usage` is **non-persistent**: it is derived from module constants, so shipping it in the
  checkpoint would be shipping a copy of the source.
* ⛔ `anchor_goal != "none"` **requires** `goal_cat_args=True`, and the refusal says why: an emitted
  id that reaches nothing downstream is a head wearing an emission's name.

**Multi-label emission** closes §2.3's second limit: `gates = logits.sigmoid()`, independent per
token, consumed through the same `probs @ table.weight` path so a **set** of goals is a sum of token
embeddings. Pinned by construction — with the two logits forced high, `gates` for
`ANCHOR_GOAL` **and** `SPEED_BAND` both exceed 0.99 while the simplex's `probs` cannot exceed ~0.5
each.

⭐ **But the FACTORED pair is the structured form of the same fix and is strictly better
attributable**: one token per **axis**, so the pair is emitted *by construction* and **each half is
separately scoreable** — which the unstructured multi-label form cannot give you. Multi-label
remains the fallback for the un-factored head.

⚠️ **This closes a CODE gap, not a data gap.** PH0 still emits none of the nine tokens; §2.1 of the
sibling report stands unchanged.

---

## 5. ⛔ THE 6 s QUESTION — answered plainly: NOT EXPRESSIBLE TODAY, and the code now refuses

**MEASURED this turn — all five banked anchor vocabularies probed directly:**

| file | shape | horizons (steps) | max horizon |
|---|---|---|---|
| `…/2026-08-04-instrument-durability/refc_anchors_full_REBUILD.pt` | `[256, 4, 2]` | `[5, 10, 15, 20]` | **20 = 2.0 s** |
| `…/2026-07-22-refc-small-30k/refc_anchors_small64.pt` | `[64, 4, 2]` | `[5, 10, 15, 20]` | **20 = 2.0 s** |
| `…/2026-07-27-percandidate-labels/raw/anchors_dev256.pt` | `[256, 20, 2]` | `[1 … 20]` | **20 = 2.0 s** |
| `…/2026-07-28-pod-migration-rescue/flagship_v4_anchors_dense.pt` | `[256, 20, 2]` | `[1 … 20]` | **20 = 2.0 s** |
| `…/2026-08-04-fan-width/raw/refc_anchor_vocab.pt` | `[128,4,2]` + `[256,4,2]` | *not stored inline*; **4 points**, consistent with the `[5,10,15,20]` build | **2.0 s** |

Against `PLAN_STEPS = 60` / `HORIZON_S = 6.0`, and a v6f selector that scores `waypoints[:, :, -1]`
— **the 6 s endpoint**.

> ⛔ **ANSWER: the factored head is NOT expressible at 6 s today. This is a BLOCKER, not a caveat,
> and it is not closed by anything in this turn.**

What was done instead of building on the 2 s one:

1. `AnchorGoalHead.load_anchor_table` **REFUSES** any table whose requested step is not the plan
   horizon. Verified against **both** real shipped shapes: `[5,10,15,20]` and `[1…20]` are refused at
   a 6 s plan horizon, `table_ready` stays `False`, and nothing is emitted.
2. **The refusal has ONE definition in the programme** — it *imports*
   `tanitad.data.anchor_goal.anchor_endpoints`, the label side's own refusal, rather than
   re-implementing it. Two copies of a refusal is one copy that will drift. Pinned by a source check.
3. **A negative control for the refusal**: a **2 s** head accepts the **same** `[5,10,15,20]` table.
   *A guard that refuses everything guards nothing* — the refusal fires on the **horizon**, not on
   the table.
4. `forward` **refuses entirely** until a table is loaded: *a zero table would snap every goal to the
   origin and still return a number*, which is the failure class the refusal exists to make
   impossible.

**What would unblock it:** `build_refc_anchors.py --horizons 5,10,…,60` over the canonical train
corpus — CPU-only, ~1 CPU-hour — but it needs the **train epcache, which is not on this box**.
⚠️ And even then there is **no admissible BAR at 6 s**: §5.3's refutation check fired `REDERIVE`
(σ(6 s) = 3.75 × σ(2 s) > 3), so the σ ≤ 0.80 / ≤ 1.41 thresholds are **2 s bars** and scaling them
would violate both the ≤2× extrapolation rule and the instrument's own committed 6 s rule.

---

## 6. ⛔ X3 ISOLATION AND ADMISSIBILITY — unchanged, MEASURED on eight arms

**X3 holds on every arm**, on a real autograd graph:
`{planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0}` — and the test also asserts
`n_probed > 0` for each edge, because *a probe over an empty parameter set reports zero violations
and has established nothing*.

**The declared planner surface stays TOTAL on every arm.** New tensors appended to `planner_side`:
the factored pair's `logits`/`args`, every head's `cat_logits`, and
`anchor_goal_point` / `anchor_goal_point_raw` / `anchor_cls_logits` / `sel_goal_point_free`.
`anchor_cls_logits` is what makes the **one-hot control** reachable — its emitted point is a hard
table lookup and carries no gradient at all, by design.

**Admissibility, audited on MY OWN design:**

| | |
|---|---|
| inputs to every new head | `z_tac_p` (vision-derived, already cut) and `e_g_str` (the goal handed down) — **exactly what the mixed head reads**. The factoring changes the DECISION's shape, not its information, so a difference cannot be an information effect |
| `AnchorGoalHead`'s only input | `e_g_tac`. Its signature is **`forward(self, g_embed)` and nothing else** — pinned by an `inspect.signature` test |
| situation-classifier path | **none.** No new signature accepts `situation`/`ego`/`v0`/`**kwargs`; `tanitad.data.situations` appears nowhere in `v6.py` (source-scanned) |
| the anchor table | a **frozen buffer**, identical for every window ⇒ **zero per-window information by construction** — the quantity the goal-echo null (13.5553 m against live arms at 0.79–9.49) bounds |
| 🟥 **`v0`** | **nothing built here depends on `v0` being admissible.** Pinned by a behavioural test: **tripling and offsetting `v0` leaves `anchor_goal_point` bit-identical.** The contradiction (`V6F_PLANNER_DESIGN.md` §1.4 ✅ vs `e_wc2_sigma_star.py:188` ⛔, worth a MEASURED **2.85×**) remains an **OPEN PI DECISION** and this turn does not pre-empt it |

---

## 7. ⚠️ A PRE-EXISTING GAP FOUND IN PASSING — not introduced here, not fixed here

With `selector="mlp"`, `cand_score.fc1.{weight,bias}` are **invisible to the X3 planner-surface
probe**: `fc2` is zero-init by design, so `dL/dfc1 ∝ W_fc2 = 0` exactly and the probe reports them
unreachable. `test_planner_surface_is_total` never sees this because it runs only on the default
config. This is **pre-existing** (`MLPCandidateScorer` was not touched) and is the same class as the
documented emission/FiLM zero-init caveats — reachability is an ARCHITECTURE property, so the probe
should run off the zero init. My own test perturbs `fc2` before probing, which is why the `"mlp"`
arms pass here. **Escalated, §9 item 3.**

---

## 8. Test evidence

**Baseline briefed at HEAD `37220d2`: 3154 passed / 0 failed / 17 skipped / 2 xfailed.**

`PYTHONUTF8=1 python -m pytest -q -p no:cacheprovider` from `stack/` →
⭐ **3235 passed, 17 skipped, 2 xfailed, 0 FAILED**, exit 0, 321 s.

⚠️ **My contribution is +65 tests**, all in one new file (`tests/test_v6_factored_goal.py`).
3154 + 65 = 3219, so **~16 are concurrent sibling work** landing in the same tree — the same caveat
the last two reports recorded. **0 failures, 0 errors** ⇒ nothing here regressed anything, including
the 257 pre-existing v6 tests, which were run in isolation first and again in the full suite.

Properties pinned that would otherwise produce a **number** or a **label** instead of an error:
the byte-identity against the pre-change revision (**negative-controlled twice**); the RNG-draw
count of the default path; the partition's totality and disjointness; the straight-through
gradient identity; `snap_lat` touching one axis only; the one-hot control having **no**
differentiable point; the 6 s refusal in **both** real shipped shapes **plus** its negative control;
`_expressible` computed from source (2 of 9 → 9 of 9); the soft/hard mask coincidence; the
per-slot-not-global softmax; X3 on eight arms with `n_probed > 0`; and `v0`-invariance of the
emitted goal.

---

## 9. Escalations — requests, not notes in a README

1. ⭐⭐ **The loss is the other half and it belongs to `train_v6_staged.py` (another agent's file).**
   The head now *emits* a metric-aware object; nothing yet *trains* it that way. The supervision
   must be a distance-weighted / endpoint-distance target on `anchor_goal_point`, **never** a
   one-hot CE on `anchor_id` — E-AG2 fact 2 exonerates the estimand and facts 1/3 localise the
   failure in the objective, which **E-OBJ-1 already measured as the inferior half of exactly this
   axis with a LONGITUDINAL recovery**. A `--anchor-ce` arm shipped as the default would repeat a
   refuted objective. The `"onehot"` mode exists so that arm can be run **as the control**, and
   `cls_logits` is the only differentiable thing it emits, deliberately. **0 GPU.**
2. ⛔ **No 6 s anchor vocabulary exists (§5) and the head now refuses without one.** Building it is
   `build_refc_anchors.py --horizons 5,10,…,60`, CPU-only, but it needs the **train epcache, which is
   not on this box**. ⚠️ And no admissible 6 s **bar** exists either — §5.3 fired `REDERIVE` this
   morning. **Both halves are needed before any 6 s anchor number can be judged.**
3. ⚠️ **`test_planner_surface_is_total` should run off the zero init for the `"mlp"` arm** (§7). It
   is a real hole in a *control* arm's isolation coverage: a mis-wire in `MLPCandidateScorer.fc1`
   would not be caught today. One-line fix in `tests/test_v6_staged.py` — not mine this turn.
4. ⚠️ **`RETRACTION_LOG.md` gets one root-cause class from this turn** (orchestrator's file; text
   ready to lift, §1.1): ***a guard whose reference is a MOVING POINTER degenerates into a
   self-comparison.*** MEASURED — a `HEAD`-relative byte-identity test skipped itself with *"matches
   HEAD byte-for-byte"* after HEAD moved `37220d2 → a558b79` and swept the file under test into the
   commit. The durable shape is to resolve the reference **by content** (walk the file's own history
   for the newest revision lacking the change marker), which is also the semantically right
   reference: the architecture the live checkpoint was built from. Same family as C72 (a hash over a
   non-canonical container) and as *"an isolation check whose scope is the SESSION, not the
   SUBJECT"*.
   ⚠️ **Second, smaller:** the `CLAUDE.md` whole-index-commit hazard fired again during this turn —
   `a558b79` swept an in-progress `v6.py` from a live agent.
5. ⚠️ **`V6F_PLANNER_DESIGN.md` should record the new levers and their MEASURED deltas** (§2.1) so
   the next design read does not re-estimate them. PI/orchestrator's file; the table is ready to lift.
6. 🟥 **The `v0` adjudication (§6) still gates the goal head's input set.** Nothing here depends on
   it, by construction — which is precisely why it can be decided cleanly.

---

## 10. Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| **This report** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-factored-goal-head/FACTORED_GOAL_HEAD.md` | **staged** (index blob `367c281`, verified `git ls-files --stage` == `git hash-object`) |
| ⭐ **The factored goal head + anchor head + categorical channel + multi-label** | `stack/tanitad/models/v6.py` | ⚠️ **already IN HEAD** — see the note below (index blob `0273378`, identical to HEAD; `git diff HEAD` is empty) |
| ⭐ **Its 65 tests** | `stack/tests/test_v6_factored_goal.py` | **staged** (index blob `4d8f1e0`, verified) |

⚠️ **`v6.py` was NOT committed by me — it was SWEPT INTO `a558b79` by another agent's whole-index
commit while I was working** (`git log -S goal_factored -- stack/tanitad/models/v6.py` → `a558b79`,
whose message is *"The operator runbook was stale in ELEVEN ways…"* and has nothing to do with this
work). The content is banked and correct — it is byte-identical to what the tests here were run
against — but **its provenance now lives in an unrelated commit message**, which is exactly the
`CLAUDE.md` hazard *"`git commit` commits the ENTIRE INDEX"*, firing for the third recorded time.
It is also what forced §1.1's change of reference. **I committed nothing and pushed nothing**
(`AGENT_OPERATING_STANDARD` rule 1).
⛔ **Thor was not contacted; 0 GPU.** No file owned by another live agent was edited —
`MODEL_REGISTRY.md`, `RETRACTION_LOG.md`, `V6F_PLANNER_DESIGN.md`, `CLAUDE.md`,
`scripts/v6_chain.py`, `scripts/train_p8_occupancy.py`, `scripts/build_obstacle_join.py`,
`scripts/train_v6_staged.py`, `tanitad/data/anchor_goal.py` — all untouched.
