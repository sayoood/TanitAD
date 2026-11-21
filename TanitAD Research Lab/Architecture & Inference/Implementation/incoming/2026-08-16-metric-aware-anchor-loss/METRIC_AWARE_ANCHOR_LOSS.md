# The head could emit a metric-aware goal and NO COMMAND COULD BUILD IT — the loss, and the launch surface it needed

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **HEAD at start `70588b8`**
**Cost: 0 GPU.** ⛔ **Thor was not contacted** — no ssh, no disk read, no job. v6F S-W was training
throughout and nothing here touched it.
**Tier:** this file is **MECHANISM, not measurement.** Every param count, identity proof and
loss trace below is **MEASURED (ours, this box, torch 2.11.0+cu128, CPU, seed 0)**; every σ / Δ / CI
is **INHERITED** from `…/incoming/2026-08-16-anchor-goal-supervision/` and E-OBJ-1 and is re-quoted
with its own stamp. ⛔ **No number produced by running this code is a capability claim** — those are
T1 only (`taniteval/tools/t1_eval.py`), with the four metric families and the episode-cluster
bootstrap.

---

## 0. The one-line answer

> ⭐ **`ANCHOR_GOAL` is now TRAINABLE, and the objective the measurement refuses cannot arrive by
> default: `metric` (regress-then-snap, straight-through) is the default, `softanchor` (the
> distance-weighted target over anchors) is the K-way head's metric-aware objective, and the one-hot
> `ce` is reachable ONLY behind `--i-know-this-is-the-control-arm` — the same acknowledgement
> `--no-isolate-planner` uses. MEASURED BY EXECUTION, all three.**
>
> ⭐ **The axis weighting is RAW METRES, and that is the evidence-weighted choice rather than the
> symmetric one** — §3.
>
> ⛔ **AND THE THING THE BRIEF DID NOT KNOW: the goal-head config family had NO CLI PATH AT ALL.**
> `goal_factored` / `goal_multilabel` / `goal_cat_args` / `anchor_goal` / `n_anchors` / `n_lat_bins`
> / `n_agent_slots` existed in `V6Config` and `build_stack_from_args` set **none** of them
> (MEASURED from source). A loss for a head no launch command can build is a loss nobody can run,
> so the launch surface is part of this turn. §4.
>
> ⛔ **The default build is BYTE-IDENTICAL, and so is the DEFAULT LOSS — two content-anchored proofs,
> both negative-controlled.** §1.

| # | brief item | state |
|---|---|---|
| **1** | byte-identity, CONTENT-anchored (C75) | ⭐ **DONE ×2 + negative-controlled ×2** (§1) |
| **2** | the metric-aware loss, CE as a named control only | ⭐ **DONE, MEASURED BY EXECUTION** (§2) |
| **3** | axis weighting justified by the 98.8 % measurement | ⭐ **DONE — raw metres, arithmetic in a test** (§3) |
| **4** | the `"mlp"` `fc1` probe hole | ⭐ **DONE — and the hole was BIGGER than stated** (§5) |
| — | the launch surface the loss needs | ⭐ **found and closed** (§4) |
| — | two findings in passing | ⚠️ §6 |

---

## 1. ⛔ THE TWO IDENTITY PROOFS — and why one of them is new

The live v6F S-W run resumes from a checkpoint of this exact architecture. This turn edits the
function that **builds** it from the CLI and the function that **computes its loss**, so there are
two things to protect, not one.

### 1.1 The reference is resolved BY CONTENT, never by `HEAD` — C75, logged hours ago

A `HEAD`-relative byte-identity test **SKIPPED itself** with *"matches HEAD byte-for-byte"* after a
sibling's whole-index commit swept the file under test into `HEAD`. **A guard whose reference is a
MOVING POINTER degenerates into a self-comparison**: it runs, reports green, and has measured
nothing. Both guards here walk the **file's own history** for the newest revision that lacks the
change marker.

| guard | file | marker | ⭐ resolved reference (MEASURED this turn) |
|---|---|---|---|
| architecture | `stack/tanitad/models/v6.py` | `goal_factored` | **`b12c190`** — *"Two streams: the capacity control…"* |
| ⭐ **loss** | `stack/scripts/train_v6_staged.py` | `w_anchor` | **`efd49f5`** — *"The banned estimator was deciding G1…"* |

Both are real, distinct, pre-change commits; neither is `HEAD` (`70588b8`). If git cannot answer the
test **skips** — a skipped test is honest, a self-comparison dressed as a real one is not.

### 1.2 ⭐ ARCHITECTURE — through `build_stack_from_args`, not around it

MEASURED, the **default CLI build** at the production geometry:

| | value |
|---|---|
| params / `state_dict` keys | **87,893,449 / 405** |
| per-tensor `torch.equal` vs `b12c190` | ⭐ **every tensor identical, key list identical** |
| RNG stream after the build | ⭐ **`torch.equal(rng_before, rng_after)` — no extra draw** |

⭐ **Independent corroboration:** 87,893,449 / 405 is **bit-for-bit the sibling's**
`HEAD_FULL_PARAMS, HEAD_FULL_KEYS`, measured through a completely different entry point (their
`V6Config()`, my `build_parser() → build_stack_from_args`). Two paths, one number.

The test goes **through the edited function**. A byte-identity test that builds a hand-made
`V6Config` would route around exactly the code this turn changed and prove nothing about it.

### 1.3 ⭐ THE LOSS IDENTITY — the guard this turn actually needed

Same stack, same batch, same seed, both trainers, **all four stages**: the total **and every
per-term tensor** must be bit-identical, plus the log's key set (an operator reads the log; a
silently changed key is a run row that stops being comparable with the ones before it).

**MEASURED: `torch.equal` on the total and on every term, S-W · S-T · S-S · S-J, against `efd49f5`.**

⚠️ The old module deliberately imports the **current** `tanitad.models.v6` — this holds the MODEL
fixed and varies only the LOSS code, so a difference can only be mine.

### 1.4 ⭐ BOTH GUARDS WERE MADE TO FAIL

*A guard that has never failed is a guard whose sensitivity is unknown.*

| control | cheap checks | ⭐ per-tensor / per-term |
|---|---|---|
| a **DISCARDED** `nn.Linear(3,3)` — registers nothing, consumes ONE RNG draw | ✅ key list **passes**, param count **passes** | ⛔ **FIRES** |
| `w_anchor=1.0` on an otherwise identical trainer | — | ⛔ **FIRES** (new term, different total) |

The first row is the negative control the brief named, reproduced: **the count checks are not a
substitute for the per-tensor comparison.** ⚠️ And deliberately **not** used anywhere: a digest of a
`torch.save` file — C72, those bytes are not canonical.

### 1.5 The default term is SKIPPED, not multiplied by zero

`w_anchor` defaults to `0.0` and is **zeroed by `for_stage` in S-W and S-S** (the planner is absent
in one and frozen in the other). MEASURED: at the default no `anchor` term appears in `log["terms"]`
and **no `anchor_*` key appears in the log at all** — a term that is skipped costs no compute and,
the part that bites, cannot appear in the log looking like it trained something.

---

## 2. ⭐ THE OBJECTIVE — metric-aware by default, the refuted arm behind an acknowledgement

### 2.1 The three objectives, and the measurement that put each where it is

**INHERITED, MEASURED 2026-08-16** (881 windows / 40 episodes, LOEO, paired episode-cluster
bootstrap): `snap` — the same ridge rounded to the nearest anchor — is **NOT separated** from the
free ridge (**−0.0002 [−0.1031, +0.0703]** at K=256; **+0.0383 [−0.2125, +0.2338]** in the `v0`
regime), while a K-way one-hot classifier costs **+4.7502 [+3.0514, +6.3981] WORSE**, separated at
every K from 8 to 256, under both vocabulary constructions, replicating on REF-C-base (**+5.4570
[+3.8345, +7.1073]**). ⇒ **quantisation is FREE; the one-hot TARGET is what costs.**

| `--anchor-objective` | what it is | where it comes from |
|---|---|---|
| ⭐ **`metric`** (DEFAULT) | `‖ĝ − g*‖` in **metres** on the **emitted** goal point. The snap is straight-through, so the estimand is quantised while the gradient reaches the **continuous regression** — the half E-AG2 **exonerated** | E-AG2 fact 2 |
| **`softanchor`** | `Σ_k p_k·d_k` with `d_k = ‖a_k − g*‖` and `p` the head's own softmax — the **EXPECTED anchor error under the model's own distribution**, whose optimum is still all mass on the nearest anchor. E-OBJ-1's `softade`, one level up | E-OBJ-1 |
| ⛔ **`ce`** | one-hot CE on `argmin_k d_k`. Metric-**BLIND** by construction — which is the property being controlled for | the pre-registered CONTROL |

⭐ **`softanchor` is the expected-distance form and NOT a softened CE target, and that is an
evidence call, not a taste call.** E-OBJ-1 measured metric-awareness **recovering** −0.0974 m (base)
/ −0.1670 m (XL), separated, **while SOFTENING the CE target was separated WORSE (+0.0909 m) at
EVERY tau**. Metric-awareness helps; target-softness hurts. ⇒ the softened-CE form is deliberately
**not offered**.

### 2.2 ⛔ CE CANNOT ARRIVE BY DEFAULT — MEASURED BY EXECUTION, both directions

```
--anchor-objective ce                                  -> ⛔ REFUSED, quoting +4.7502 [+3.0514, +6.3981]
--anchor-objective ce --i-know-this-is-the-control-arm -> ✅ RUNS
```

Both halves were **executed**, not asserted. The second half matters as much as the first: **a
control nobody can run is not a control**, and a comparison with no control is unattributable (C6).

### 2.3 ⛔ THE MODE COUPLING IS HARD, IN BOTH DIRECTIONS

`ANCHOR_OBJ_MODES` is data, and both mismatches produce **a number instead of an error** — this
programme's most expensive failure class:

* ⛔ **`metric` on `"onehot"`** — that head's emitted point is a **hard table lookup carrying no
  gradient at all** (by design). The loss would fall to a constant and train **nothing** while the
  log showed a metre-scale term. **MEASURED, not assumed:** the test asserts
  `out["goal_point"].requires_grad is False` **before** asserting the refusal.
* ⛔ **`softanchor` / `ce` on a snap mode** — there are no `cls_logits`; there is nothing to put a
  distribution on.

### 2.4 ⭐ THE DISCRIMINATING TEST — CE's blindness, made measurable

Two predictions **wrong in the same way categorically** (identical logits, index 1 of 3) but
differing by **~40 m versus ~0.1 m** in the metre the car is scored in:

| | `ce` | `softanchor` |
|---|---|---|
| near table (0.1 m error) | `L` | `L_near` |
| far table (40 m error) | ⛔ **exactly the same `L`** | ⭐ **> 50 × `L_near`** |

**That single assertion is E-AG2's +4.7502 explained.** And the metre-valued diagnostics
(`anchor_expected_err_m`, `anchor_argmax_err_m`) travel with the CE arm too, so a control run stays
comparable with the default on the quantity that matters — CE's own loss is in **nats**.

### 2.5 The straight-through contract, checked analytically and negative-controlled

⭐ **MEASURED ANALYTICALLY** rather than by comparison with another autograd call: for
`raw = W·g + b`, `dL/db == mean_B (emitted − target)/‖emitted − target‖` on **both** coordinates —
i.e. **the quantised axis still carries gradient**. That single fact is the difference between a
*trained* regress-then-snap and a post-hoc rounding.

⭐ **Its negative control, in the same file:** replace the straight-through with a **hard** snap and
the lateral gradient goes to **exactly 0.0** while the un-quantised axis keeps training. The
property is therefore observable failing.

### 2.6 The seam into the batch — a wiring detail that would have been a silent refusal

`plan_target` was built **only when `λ_plan` was non-zero**. The anchor objective is deliberately
runnable **with the planner loss OFF** — that is the attributable arm, because a goal that moves
must be attributable to its **own** objective and not to a WTA fan gradient arriving through the
same seam. The loader now builds the target when **either** term is in force; without that, the run
would have refused mid-training exactly when the anchor loss was the only planner term.

---

## 3. ⭐ THE AXIS WEIGHTING — raw metres, and why that IS the evidence-weighted choice

**The measurement (INHERITED, §6.4, 2 s, 881 windows):** the goal's corpus variance is **98.8 %
LONGITUDINAL** (σ_long **19.0578** vs σ_lat **2.0723** — 9.2× in σ, 84× in variance); every arm's
**residual** is longitudinal too (ridge 6.6132 / 1.0667); and the **headroom** is overwhelmingly
longitudinal — the classifier sits **1.96×** above the floor laterally (1.3310 vs 0.6802) against
**14.9×** longitudinally (13.3502 vs 0.8954).

**The decision: `--anchor-axis-w 1 1`, i.e. RAW METRES.** Three reasons, in order of weight:

1. ⭐ **A raw-metre loss is ALREADY strongly anisotropic — the evidence is in the units.** Computed
   in the test, not asserted: under the measured ridge residual it allocates
   **97.4 %** of its squared-error gradient LONGITUDINALLY, purely because that is where the metres
   are. It puts the gradient where the headroom is **without being told to**.
2. ⛔ **Whitening would UNDO exactly that.** Dividing each axis by its corpus σ drives the split to
   **< 55 %** (computed), i.e. it spends roughly half the gradient on the axis carrying **1.2 %** of
   the variance. That is **§6.4's own diagnosis of what the isotropic FPS vocabulary does wrong** —
   *"spends half its resolution on the axis carrying 1.2 % of the variance"* — repeated one level up,
   in the objective. Committing that error in the loss while the report names it in the vocabulary
   would be the 5-way-manoeuvre-softmax defect a third time.
3. **It is metric-consistent with the EVAL.** ADE and the four families are scored in metres, not in
   corpus σ. A loss in whitened units optimises a quantity nothing reports.

⚠️ **What is NOT claimed.** The weights are stamped **DECLARED**, not optimal. A whitened arm is one
flag away (`--anchor-axis-w 0.0525 0.4826`) and the knob is proved to bite (a test drives the same
`(3, 4)` residual to **5.0 / 3.0 / 4.0 m** under `(1,1) / (1,0) / (0,1)`) — *a DECLARED knob that did
nothing would be worse than no knob.* ⛔ **And a whitened DEFAULT is refused for a second reason:
the σ it would need are 2 s numbers and no 6 s counterpart exists** — hard-coding them at a 6 s plan
horizon would breach the ≤2× extrapolation rule and the instrument's own committed 6 s rule.

**So the allocation is MEASURED at run time instead of assumed.** Every step logs, per family and
**never pooled** (PI 2026-08-02, binding): `anchor_err_lon_m`, `anchor_err_lat_m`,
`anchor_lon_share_sq` (the realised split), `anchor_floor_m` (the reachable quantisation floor on
this batch), `anchor_free_err_m` (the quantisation cost, readable rather than inferred), and for the
K-way arms `anchor_top1_acc` / `anchor_chance` / `anchor_expected_err_m` / `anchor_argmax_err_m`.

---

## 4. ⛔ THE LAUNCH SURFACE — a head no command could build

**MEASURED from source:** `build_stack_from_args` set **none** of `goal_factored`,
`goal_multilabel`, `goal_cat_args`, `anchor_goal`, `n_anchors`, `n_lat_bins`, `n_agent_slots`. The
levers existed in `V6Config` and no CLI reached them. **This is the `intent_proj` defect in the
launch surface** — *a path present in the diagram, absent from the thing that runs* — and it makes
the loss unrunnable no matter how correct it is.

⭐ **Every new flag defaults to the incumbent, and §1.2 is the proof** — the identity test goes
through this function.

**MEASURED deltas, built through the CLI at the production geometry** (K = 256 unless stated):

| arm | Δ params | Δ keys | total |
|---|---|---|---|
| default | 0 | 0 | 87,893,449 |
| `--goal-cat-args` | +110,880 | +5 | 88,004,329 |
| `--anchor-goal snap_lat` (+cat) | **+111,138** | +11 | 88,004,587 |
| ⛔ `--anchor-goal onehot` (+cat) | **+143,904** | +11 | 88,037,353 |
| `--goal-factored` | +470,939 | +46 | 88,364,388 |
| *(K = 16, for scale)* `snap_lat` / `onehot` | +18,738 / +20,544 | +11 | — |

⭐ **These reproduce the sibling's independently-measured +111,138 / +143,904 / +470,939 exactly**,
through a different entry point.

### 4.1 ⛔ THE REFUSALS — every one fires in MILLISECONDS, not after a GPU-day

*(the `--gate-probes` lesson: an input read only after the training loop costs the whole budget)*

| the launch | the refusal |
|---|---|
| `--w-anchor N` with `--anchor-goal none` | an objective with no head — *how a head silently never trains* |
| ⛔ `--anchor-goal X` with no `--anchor-table` | the head refuses without one, **AND no admissible table exists**: all five banked vocabularies stop at **2.0 s** against `--plan-steps 60`. The message names the fix (`build_refc_anchors.py --horizons 5,10,…,60`) and its blocker (the TRAIN epcache) |
| ⛔ `--stage S-W` with `--anchor-goal X` | the planner is FROZEN in S-W and the head would change the `state_dict` — **which breaks the strict resume of the run training on Thor right now** |
| `--w-anchor N` in S-S | `for_stage("S-S")` zeroes it ⇒ the launch line would advertise a term that trains nothing. Only the **weight** is refused; the **geometry** must be carried forward |
| objective/mode mismatch, either direction | §2.3 |
| `--anchor-goal X` without `--goal-cat-args` | `V6Config` refuses the pairing; preflight says so first |
| ⛔ `--anchor-objective ce` without the ack | §2.2 |
| negative `--anchor-axis-w` | — |

⭐ **With its negative control:** a default `--stage S-T` command raises **no** anchor problem at
all. *A guard that refuses everything guards nothing.*

### 4.2 The anchor table, and a refusal that had to be added

`_read_anchor_table` reads the shipped `{"anchors": [K,S,2], "horizons": […]}` format and installs
it **at build time**, before the first forward — so a wrong-horizon vocabulary costs milliseconds.

⛔ **A bare `[K,S,2]` tensor carries no horizons, and the two real shipped shapes are `[5,10,15,20]`
and `[1…20]`.** Guessing between them would mislabel the whole corpus, so `horizons=None` is passed
straight through and `load_anchor_table` refuses it **by name** rather than reading `anchors[:, -1]`.
*(My first implementation invented `1..S`. It was wrong for the same reason the rule exists.)*

**⭐ THE 6 s BLOCKER, VERIFIED ON THE REAL BANKED ARTIFACTS** — `refc_anchors_full_REBUILD.pt`
(K=256, FPS, pool 200,000, canonical TRAIN corpus) and `anchors_dev256.pt`: **both are REFUSED at a
6 s plan horizon and `table_ready` stays `False`**, and — the negative control in the same test —
**both are ACCEPTED at a 2 s plan horizon**. The refusal fires on the **horizon**, not on the table.

---

## 5. ⭐ THE `"mlp"` PROBE HOLE — and it was BIGGER than the brief stated

**The brief's defect, reproduced:** with `selector="mlp"`, `fc2` is zero-init by design (so the
capacity control starts **flat** over the fan), hence `dL/dfc1 ∝ W_fc2 = 0` **exactly** and
`cand_score.fc1.{weight,bias}` are invisible to the X3 planner-surface probe.

⛔ **MEASURED, and worse than that:** `test_planner_surface_is_total` ran on `tiny_cfg()` alone,
whose `selector` default is `"none"` — so **`group_parameters("planner")` contained NO `cand_score`
parameter at all**, in either arm:

| config | `cand_score.*` in the probed planner group |
|---|---|
| default (`selector="none"`) | ⛔ **`[]` — nothing** |
| `"goal"` | `cand_bias`, `log_tau`, `goal_point.weight/bias` — **never probed** |
| `"mlp"` | `cand_bias`, `fc1.weight/bias`, `fc2.weight/bias` — **never probed**, and `fc1` invisible even when it is |

⇒ the hole was not one layer in one arm; **the entire selector surface was outside the guard's
scope** — in the arm whose whole job is to be the CONTROL that decides whether SEL-1 is mechanism or
capacity.

### 5.1 The fix, and why it does not change the trained arm

**`test_planner_surface_is_total` is now parametrised over `selector ∈ {none, goal, mlp}`, and the
probe perturbs the scorer's OUTPUT layer (`fc2` / `goal_point`) off its zero init — in the TEST, on
a stack built for the probe and discarded after it.** `MLPCandidateScorer.__init__` is **untouched**.

**Why that is the right shape, not a shortcut:** reachability is an **ARCHITECTURE** property — the
X3 probe's own philosophy — and the test already does exactly this for **two** other zero-inits (the
emission's last layer and `FiLM.to_scale_shift`). The scorer's is the **third instance of one
pattern**, so the fix joins an existing idiom instead of inventing a second one.

**The trained arm is provably unchanged:** `test_mlp_fc2_is_STILL_zero_init_after_the_probe_fix`
MEASURES that a fresh `MLPCandidateScorer` has `fc2.weight`/`fc2.bias` **exactly 0.0** and that its
score is **flat over the fan at init** (`score == cand_bias`, atol 1e-6) — so any ranking the control
acquires is still visibly something it **LEARNED**, which is the whole point of the zero init.

### 5.2 ⭐ The fix is proved to be a fix, in three measurements

| | MEASURED |
|---|---|
| the hole **EXISTS** at zero init | `fc1.weight`/`fc1.bias` **absent** from the live-edge set |
| it **CLOSES** under the perturbation | both **present** |
| ⭐ a **genuinely disconnected** `fc1` is still caught | `fc1` bypassed in `forward` (architecturally dead), `fc2` perturbed ⇒ the probe reports it **MISSING** |

The third row is the one that matters: *a probe that reported "reachable" for a disconnected layer
would be worse than no probe.*

---

## 6. ⚠️ TWO FINDINGS IN PASSING — neither is mine to fix

### 6.1 ⛔ `v6_loss_step` IS NOT REPRODUCIBLE FROM ITS OWN `generator` ARGUMENT

**MEASURED while building the loss-identity guard.** Two calls with **identical inputs, identical
weights and an identically-seeded `generator`** return different S-W totals:

```
S-W  3.9300594  vs  3.9226556      (train mode)
```

**Localised, per term:** `o1` ✅ · `o2` ✅ · `o3` ✅ · `o5` ✅ · ⛔ **`o6` 0.046874 vs 0.039470**
(an **18.7 %** swing). **Root cause read from source, not inferred:** `tanitad/models/sigreg.py:70`
draws its projection slices with `torch.randn(d, self.n_slices, …)` — **no `generator`**, so it
reads the **GLOBAL** RNG that the documented argument does not cover.

⚠️ **The non-alarmist half, verified before escalating.** `train()` calls
`torch.manual_seed(a.seed)` (`train_v6_staged.py:1699`), so a **full run is** seeded end-to-end and
reproducible. What is **not** reproducible is a **re-call** of the loss inside a process — which is
exactly what an A/B guard, a paired arm comparison or a byte-identity check does. My guard therefore
runs in `eval()` **and** re-seeds globally before each call, and says so; without that it would have
fired on noise and been "fixed" by loosening it into uselessness.
⇒ **Escalation:** either thread `generator` into `position_relaxed`/`SIGReg`, or document that O6 is
global-RNG-bound. Not my file, and O6 is *"LeJEPA's ONE validated knob — keep it fixed"*.

### 6.2 ⚠️ `--anchor-goal` without a table fails with a *good* error — 130 lines too late

Without preflight, `build_stack_from_args` builds the whole model and then dies inside
`assert_isolation` with the head's (correct, well-worded) *"no anchor table is loaded"*. At the
production geometry that is an 88 M build paid for before the refusal. **Preflight now catches it
first**; the head's refusal remains as the second line of defence. Same class as the `--gate-probes`
lesson, one function earlier.

---

## 7. ⛔ ADMISSIBILITY AND X3 — audited on the arms this turn makes launchable

| | |
|---|---|
| the head's only input | **`e_g_tac`** — from `goal_head_tac(z_tac_p, cond=e_g_str)`, vision-derived. Unchanged by this turn |
| `anchor_goal_loss`'s inputs | `head_out`, `target_xy`, `anchors`, `objective`, `axis_w` — **pinned by `inspect.signature`**. Nothing it reads enters the head |
| `target_xy` | a **LABEL** — the true ego-frame displacement at the plan horizon. *Labels may use ego; inference is vision-only* (PI 2026-08-03). It exists only in the loss |
| situation classifier | **none.** The objective's **CODE** (docstring stripped, so the scan cannot pass vacuously **or** fail on its own explanation) contains no `situation`, `detect_lane_change`, `ego`, `v0` or `nav_cmd` |
| the anchor table | a **frozen buffer**, identical for every window ⇒ zero per-window information by construction — the quantity the goal-echo null (13.5553 m against live arms at 0.79–9.49) bounds |
| the echo test | the label is `poses[t : t+h]`, the **future**; every inference input is at ≤ t. **No overlap** |
| 🟥 **`v0`** | **nothing here depends on it.** Pinned BEHAVIOURALLY: tripling and offsetting `v0` leaves `anchor_goal_point` **bit-identical**. The contradiction (`V6F_PLANNER_DESIGN.md` §1.4 ✅ vs `e_wc2_sigma_star.py:188` ⛔, worth a MEASURED **2.85×**) remains an **OPEN PI DECISION** and this turn does not pre-empt it |
| **X3** | **MEASURED on all three anchor arms** (`snap_lat` / `snap_xy` / `onehot`): `{planner_to_encoder: 0, tactical_to_below: 0, strategic_to_below: 0}`, with `n_probed > 0` on every edge — *a probe over an empty parameter set reports zero violations and has established nothing* |

⚠️ **`d_k` needs no `.detach()`** — unlike `w_select`'s `err.detach()`. It depends only on a frozen
buffer and a label, so it is a constant w.r.t. every parameter. Stated because the asymmetry with
the neighbouring term would otherwise read as an oversight.

---

## 8. ⭐ MEASURED BY EXECUTION — the arms actually run

`--dry-run`, tiny geometry, CPU, `--lambda-plan 0` (the **attributable** arm: no planner WTA
gradient), synthetic table at the plan horizon. ⛔ **These are synthetic tensors — no number here is
quotable as anything.**

| arm | step 1 → 2 | log (excerpt) |
|---|---|---|
| ⭐ `metric` / `snap_lat` | `anchor_loss` **1.5285 → 0.4680** | `anchor_err_lon_m` 0.2763 / `anchor_err_lat_m` 1.4781 · `anchor_free_err_m` 1.2578 · `anchor_lon_share_sq` 0.0426 · `anchor_floor_m` 1.8726 |
| `softanchor` / `onehot` | **6.2318 → 6.0103** (metres) | `anchor_top1_acc` 0.0 vs `anchor_chance` 0.0625 · `anchor_expected_err_m` 6.2318 |
| ⛔ `ce` / `onehot` **+ ack** | **3.2317 → 3.5887** (nats) | `anchor_expected_err_m` 6.4358 · `anchor_argmax_err_m` 7.4014 |

`terms: ["anchor", "seam", "t1"]` in every case — the anchor objective trains **with `λ_plan = 0`**,
which is the arm that makes a goal-head result attributable. The unit difference between rows 2 and
3 (metres vs nats) is exactly why the metre-valued diagnostics travel with the control.

---

## 9. Test evidence

**Baseline briefed at HEAD `70588b8`: 3235 passed / 0 failed / 17 skipped / 2 xfailed.**

`PYTHONUTF8=1 python -m pytest -q -p no:cacheprovider` from `stack/` →
⭐ **3282 passed, 17 skipped, 2 xfailed, 0 FAILED**, exit 0, 335.62 s.

**My contribution is +46** — 44 in one new file (`tests/test_v6_anchor_loss.py`) and +2 from
parametrising `test_planner_surface_is_total` over the three selector arms. 3235 + 46 = 3281, so
**1 is concurrent sibling work** landing in the same tree — the same caveat the last three reports
recorded. **0 failures, 0 errors** ⇒ nothing here regressed anything, and the skip count is
unchanged at 17, i.e. **no test of mine skipped**: both content-anchored references resolved
(`b12c190`, `efd49f5`) and both banked anchor artifacts are on this box.

**Properties pinned that would otherwise produce a NUMBER or a LABEL instead of an error:** the
architecture byte-identity through `build_stack_from_args` (negative-controlled with a discarded
`nn.Linear`); the **loss** bit-identity across all four stages (negative-controlled); the RNG-draw
count; `w_anchor` zeroed where the planner cannot move; the term being **skipped** rather than
zero-weighted; the straight-through gradient **analytically** (negative-controlled with a hard
snap); `metric`-on-`onehot` refused **after measuring** that its point carries no gradient;
`softanchor`/`ce` refused on snap modes; ⭐ **CE's metric-blindness demonstrated on a 400× metre
difference**; `softanchor`'s optimum being the nearest anchor, not a blur; the axis-weight arithmetic
computed from the measured σ; the per-family log; every preflight refusal **and** its negative
control; CE reachable **with** the ack; the 6 s refusal on **two real banked artifacts** with its 2 s
negative control; `_read_anchor_table` never inventing horizons; X3 on three anchor arms with
`n_probed > 0`; `v0`-invariance of the emitted goal; the objective's **code** carrying no
classifier/ego path; and the `fc1` hole existing, closing, and still catching a disconnected layer.

---

## 10. Escalations — requests, not notes in a README

1. ⛔ **`v6_loss_step` is not reproducible from its `generator` argument** (§6.1) — `sigreg.py:70`
   draws from the global RNG. MEASURED 18.7 % swing on O6 between two identical calls. A full run is
   fine (`train()` seeds globally); **any in-process A/B is not**. Either thread the generator
   through `position_relaxed`, or document O6 as global-RNG-bound so the next paired comparison does
   not read noise as an effect. **0 GPU.**
2. ⛔ **The 6 s anchor vocabulary is still the blocker, and it is now enforced in THREE places**
   (label deriver, head, and this turn's preflight + `_read_anchor_table`). `build_refc_anchors.py
   --horizons 5,10,…,60` over the canonical train corpus — CPU-only, ~1 CPU-hour — **needs the TRAIN
   epcache, which is not on this box.** ⚠️ And no admissible **6 s bar** exists either (§5.3 fired
   `REDERIVE`: σ(6 s) = 3.75 × σ(2 s)). **Both halves are needed before any 6 s anchor number can be
   judged.**
3. ⭐ **The gate that decides whether this objective is worth running at all is unchanged and still
   Thor-blocked:** re-run `e_ag1_anchor_floor.py` on **frozen S-W latents** (~10–25 GPU-min,
   `refc_dump_latents.py --endpoint-steps 20,60`). E-AG2 says the **surface** is the problem; this
   turn fixes the **objective**, which is necessary and not sufficient. Nothing here should be read
   as evidence that the anchor branch will work.
4. 🟥 **The `v0` adjudication still gates the goal head's input set** — worth a MEASURED **2.85×**,
   entirely longitudinal (3.16× on σ_long, +0.7 % on σ_lat). Nothing built here depends on it, by
   construction, which is precisely why it can be decided cleanly.
5. ⚠️ **`RETRACTION_LOG.md` gets one entry from this turn** (orchestrator's file; text ready to
   lift): ***a guard whose SCOPE is the default configuration cannot see the arms it was built to
   protect.*** MEASURED — `test_planner_surface_is_total` ran only on `selector="none"`, so **no
   selector parameter was ever probed**, and `MLPCandidateScorer.fc1` was additionally invisible even
   when built. Same family as C75 (a reference that is a moving pointer) and as *"an isolation check
   whose scope is the SESSION, not the SUBJECT"*: the check runs, reports green, and has measured
   nothing about the thing at risk.
6. ⚠️ **`V6F_PLANNER_DESIGN.md` should record the new launch surface** (§4's flags and MEASURED
   deltas) and the objective table, so the next design read does not re-derive them. PI/orchestrator's
   file; the tables are ready to lift.

---

## 11. Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| **This report** | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-metric-aware-anchor-loss/METRIC_AWARE_ANCHOR_LOSS.md` | **staged** |
| ⭐ **The metric-aware objective + the goal-head launch surface + 8 preflight refusals** | `stack/scripts/train_v6_staged.py` | **staged** |
| ⭐ **Its 44 tests** (both identity guards, both negative controls, the objective, the axis arithmetic, the refusals, the 6 s blocker on real artifacts, X3, the `fc1` hole) | `stack/tests/test_v6_anchor_loss.py` | **staged** |
| ⭐ **The `fc1` / selector-surface probe fix** | `stack/tests/test_v6_staged.py` | **staged** |

⛔ **I committed nothing and pushed nothing** (`AGENT_OPERATING_STANDARD` rule 1). Index blobs were
verified with `git ls-files --stage` against `git hash-object` at staging **and re-verified at the
end of the turn**.
⛔ **Thor was not contacted; 0 GPU; no stage launched.** No file owned by another live agent was
edited — `MODEL_REGISTRY.md`, `RETRACTION_LOG.md`, `V6F_PLANNER_DESIGN.md`, `CLAUDE.md`,
`scripts/v6_chain.py` and `tanitad/models/v6.py` are all **untouched** (`git status` shows exactly
the four files above).
⚠️ Scratch artifacts (synthetic anchor tables, dry-run outputs) live in the session scratchpad and
are **deliberately not banked** — they are smoke inputs, not vocabularies, and a synthetic table in
the repo is exactly the kind of file that later gets read as one.
