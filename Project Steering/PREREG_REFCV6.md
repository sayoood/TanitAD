# PRE-REGISTRATION — refcv6: keep refcv5's lateral and strategic gains, stop paying for them longitudinally

**Date:** 2026-09-10 (Europe/Berlin) · **Author:** Architecture & Inference FlyWheel ·
**Branch:** `agent/arch-inf-20260803`
**Status:** pre-registration + launch scripts + pre-launch gate **DELIVERED and STAGED**.
⛔ **NO TRAINING RUN LAUNCHED. 0 GPU spent on any arm.** There is no A40 — the pod was stopped and
is gone. Every number below is MEASURED on CPU today, or cited to a banked artifact.
**Hypothesis ids:** `E-REFCV6-A`, `E-REFCV6-B`, `E-REFCV6-C`, `E-REFCV6-D` (registered in
`Project Steering/GOALS_AND_CLAIMS.md` in the same turn as this file).

**Design source:** `Project Steering/REFCV6_ARCHITECTURE_REVIEW.md` (Master Mind, 2026-09-10;
PI-approved). This document does not re-litigate that design — it **commits its criteria before any
data exists** and ships the machinery that enforces them.

**Owner files (all pre-existing; refcv6 is a CONFIGURATION, not new model code):**
`stack/scripts/refc_v3_train.py` (all four levers) · `stack/tanitad/models/agent_slots.py`
(DiffusionDrive's Hungarian-matched detector) · `stack/tanitad/refs/refc_wp_index.py` (coupling (1)).
**New this package:** the argv single-source-of-truth, the one-variable checker, the pre-launch
gate, the supervisor, and the bar as executable code — all under
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-10-refcv6-build/code/`.

**Estimator for every interval below:** **paired episode-cluster bootstrap** over the val episodes,
`taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap`. ⛔ `overlapping_holdout_se` is never
called — it biases the **point estimate** as well as the interval (−6.67 % to +11.69 %,
**bidirectional**, over 27 dumps). `verdict_refcv6.py` **REFUSES** a panel whose estimator field
names it.

⛔⛔ **AND A SEPARATED INTERVAL IS NECESSARY, NOT SUFFICIENT.** The episode-cluster bootstrap
resamples EPISODES with the models held fixed, so it answers *"would another draw of episodes say
this?"* and never *"would another training run say this?"* (`H-ESTIM-SEED-1`). ⭐ **MEASURED on
WP-D, on this exact rig family: an arm with ZERO levers moved read "separably worse" on 5 of 9
family metrics (55.6 %) and reproduced a headline ADE effect at +0.02460 against the lever's
+0.02610.** ⇒ **every arm here carries a replicate from the start** (§3), and every bar is stated
as a **ratio to a floor measured in the SAME panel** (§5).

**Eval tier: T1** (action-closed loop, `taniteval/tools/t1_eval.py`), four families, never pooled.

---

## 0. The claims under test

> **`E-REFCV6-D`** — our `--w-u0 0.5` x0/denoising term is an **invention, not a port**
> (`D-DDV1-NO-DENOISING-LOSS`: DiffusionDrive v1 has **no ε-prediction and no denoising MSE at
> all**; its `diff_loss_weight = 20.0` is dead code, and its only trajectory supervision is
> matched-anchor L1 plus focal scoring). Removing it recovers refcv5-v2's longitudinal loss without
> giving back its lateral gains.
>
> **`E-REFCV6-A`** — DiffusionDrive's **learned agent detector** (`--agents head`: 30→100 queries
> through a transformer decoder, Hungarian-matched 3D box + class loss, **never GT at inference**)
> supplies agent content that a 108 M trunk can afford, even though the same rung **failed its
> two-seed tiny-rig gate at 17 M**.
>
> **`E-REFCV6-B`** — **waypoint-indexed cross-attention** (coupling (1), +2,208 params) into those
> sparse agent tokens improves the longitudinal family, where 92.2 % of our `os − ha` gap lives.
>
> **`E-REFCV6-C`** — training the **22-token tactical-goal head** (`--w-tac-goal > 0`) that received
> `grad_abs_sum` **exactly 0 for all 40,284 steps** of refcv5-v2 improves tactical goal-setting.

⛔ **Each is a SEPARATE ARM. Not all four at once.** refcv5-v2 moved several levers and its result
is barely attributable; the `--v2` conflation failure — **ten levers on two axes,
non-attributable** — is the precedent this rule exists for.

---

## 1. The chain of MEASURED facts this rests on

⭐ **refcv6 is a CONFIGURATION, not new code.** Verified today, not inherited:

| # | fact | class · source |
|---|---|---|
| 1 | **All four levers already exist in the trainer**, as `add_argument` declarations: `--w-tac-goal` **:4922**, `--w-u0` **:5095**, `--agents` **:5151**, `--wp-index` **:5340**. Mention counts **11 / 7 / 31 / 25** respectively. | **MEASURED** (ours) — `raw/lever_census.json`; controls `def main` = 1, `add_argument` = 110, file 5,640 lines |
| 2 | Each lever is **CONSUMED**, not merely declared — the `tac_goal_tok_head` failure mode (built, stamped, never trained) checked for explicitly. `w_u0`: `getattr(args,"w_u0")` **:4046** → `model._w_u0` → read **:2545** → added to `loss` **:2564 / :2584**. `w_tac_goal`: **:4043** → **:2751** → loss. `agents`: **:568 / :576** into `core.agents`. `wp_index`: **:653** into `core.decoder.wp_index`. | **MEASURED** (ours) — same census |
| 3 | Defaults: `U0_WEIGHT_DEFAULT = 0.0` (**:122**), `AGENT_WEIGHT_DEFAULT = 0.0` (**:121**), `AGENT_QUERIES_DEFAULT = 100` (**:157**), `--agents` `"off"`, `--wp-index` `"off"`, `--w-tac-goal` `0.0`. | **MEASURED** (ours) |
| 4 | ⛔ **The trainer REFUSES `--agents head` unless `--w-agent > 0`** (**:583** — *"builds a detector that is never supervised … the arm would read as 'agent tokens do not help' — a REFUTATION manufactured by a missing loss"*) **and unless `--agent-join` is given** (**:591** — *"has NO LABELS"*). ⇒ **there exists no argv in which `agents` moves alone.** | **MEASURED** (ours) |
| 5 | ⛔ **`--wp-index on` REFUSES `--agents off`** by design (**:5340** — *"with no agent tokens there is nothing to address and the arm would read as 'the index does not help' while never having had one"*). ⇒ **B is built on A, and B's comparison is B vs A, never B vs V0.** | **MEASURED** (ours) |
| 6 | DiffusionDrive's **exact Hungarian matching already exists**: `stack/tanitad/models/agent_slots.py` — `AgentSlotDecoder` **:256**, `hungarian` **:384**, `_match_cost` **:460**, `match_slots` **:481**, `slot_set_loss` **:523**, `track_rates_from_join` **:625**, `targets_from_join` **:690**. | **MEASURED** (ours) |
| 7 | Coupling (1) already exists: `stack/tanitad/refs/refc_wp_index.py` — `WaypointIndexConfig` **:128**, `waypoint_agent_geometry` **:192**, `waypoint_agent_relation` **:273**, `WaypointIndexBias` **:340**, `apply_radius_gate` **:384**, `build_relation` **:423**, and the three pre-registered controls `shuffle_waypoints` **:295** / `const_waypoints` **:327** / `--wp-index-detach`. | **MEASURED** (ours) |
| 8 | refcv5-v2's own argv is **65 tokens**, recoverable verbatim from its banked `config.json['argv']`. It carries `--w-u0 0.5`, `--agents off`, `--tac-goal-tok-head` **without** `--w-tac-goal` (⇒ default 0.0 ⇒ the head got zero gradient), and **no** `--wp-index`. | **MEASURED** (ours) — `C:\Users\Admin\refcv5v2_final\config.json`, 12,393 B; re-verified programmatically, verdict `MATCH`, 65/65, `first_difference: null` |
| 9 | The B1 TRAIN agent join covers **4,427 / 4,572 clips = 96.83 %**; the v7.2 TRAIN label release is **4,572 records**, md5 `0ff902130ce76886b8a925eceed9e3a5`, schema `s2-geom-v7`, vocab `v7`. | **MEASURED** — `D-B1TRAIN-JOIN-1`; label block read from refcv5-v2's own `config.json` today |
| 10 | refcv5-v2's result, T1, paired episode-cluster bootstrap, 4,823 windows / 141 episodes, confirmed at two inference seeds: **`os − ha0_ext` = +0.0205 [+0.0043, +0.0390] ⇒ FAIL**, separated the wrong way. It **lost** speed MAE (0.2919 vs 0.2540) and along-track (0.2655 vs 0.2348); it **won** heading (1.2121 vs 1.5489), yaw-rate (1.0534 vs 1.4542), cross-track (0.0994 vs 0.1226), masked curvature (0.003485 vs 0.004030 = **0.512× the straight-line floor**) and tactical lateral κ (0.8193 vs 0.7374); and it is the **only arm in the programme with a strategic output** — route accuracy **0.7708 [0.7146, 0.8254]**, κ 0.4614, **n 3,622**, against 0.3333 chance, from a **771-parameter** head (refcv4b reads **n = 0**). | **INHERITED** from `REFCV6_ARCHITECTURE_REVIEW.md` §1, which cites the run's own artifacts. ⚠️ **Not re-verified from raw eval JSON by this package** — it decides no GPU-day here, only the *direction* of the non-regression clauses. |
| 11 | `tac_goal_tok_head` is **11,286 params with `grad_abs_sum` exactly 0 for all 40,284 steps** of refcv5-v2. The trainer seam is now CLOSED at the code layer (`D-TACGOAL`), and the default path is **bit-identical** to the pre-wiring trainer over 429,563,860 floats. | **INHERITED** — `D-TACGOAL-TRAINER-SEAM-OPEN`, `D-TACGOAL-OFF-IDENTITY-1` |

⚠️ **Fact 10 is the only INHERITED row that matters, and it is marked.** Under the operating
standard a claim that decides a GPU-day must be MEASURED or PUBLISHED. Fact 10 does not decide a
GPU-day; it decides **which metrics the non-regression clauses name**. ⛔ **Before any refcv6 panel
is scored, fact 10's values must be re-read from the raw eval JSON** — they are the reference column
in §5B and a transcription error there would silently move a bar. Named as an owed item in §9.

---

## 2. ⛔ The one deliberate deviation from refcv5-v2's recipe, and its consequence

**BASE = refcv5-v2's 65-token argv, PLUS `--agent-join <join>`, carried by EVERY arm including the
control.**

The reason is the WP-D precedent verbatim (`PREREG_WPD_BEV_AUX.md` §4): *"the join must be in both
arms or the A/B also changes the dataset."* `V3Dataset` emits `agent_box/yaw/cls/valid/occ/label/ep`
per window when the join is enabled (`refc_v3_train.py:1527-1700`). If only arm **A** carried the
join, **A would differ from the control in the DATASET as well as in the lever** — invisibly, in
every log.

⛔ **CONSEQUENCE, STATED RATHER THAN HIDDEN: `V0` IS NOT A REPLICATION OF refcv5-v2.** It is a fresh
control on the refcv6 BASE. ⇒ **every refcv6 delta is paired against `V0`, never against
refcv5-v2's banked 40,284-step numbers.**

⭐ **This is not fussiness — it removes a two-factor confound that would otherwise have been
invisible.** Pairing a refcv6 arm against refcv5-v2's banked numbers would differ in **the BASE**
(the join) **and** in **the step budget** (if anything but `full` is funded) as well as in the
lever. refcv5-v2's values are the **PROVENANCE** of the clauses in §5B — they say which metrics
matter and in which direction — **not their comparison operand**.

⚠️ **Cost of the deviation, budgeted rather than discovered:** `--agent-join` loads the TRAIN join
at ~**2.1 GB RSS** (`JoinFileReader` docstring), and every arm now pays it, control included.
`--v2-lru 24` / `--workers 6` are copied from refcv5-v2's A40 and **must be re-checked on whatever
box actually runs this** — a per-run fact belongs beside its per-run divisor.

---

## 3. `one_variable` — the arms, DIFFED ON PARSED NAMESPACES

⛔ **Diffed as parsed namespaces, not as intent, and not as argv strings.** Two different strings can
parse to the same namespace (a flag at its default), and the same-looking string can parse
differently once a `dest=`, a `type=` coercion or a mutually-exclusive group is involved. *A past
sweep was invalidated because an arm silently changed the effective lambda alongside its named
lever.*

⛔ **The checker imports `refc_v3_train.build_parser()` — the REAL parser the launch will use.** A
checker with its own copy of the flags is *a check that shares the defect it checks for*, and would
go green against a trainer whose parser had moved underneath it.

### 3.1 The ten arms

| arm | lever | seed | pairs against | role |
|---|---|---|---|---|
| **V0** | — | 0 | — | the control (BASE) |
| **V0b** | — | 1 | V0 | ⭐ the training-seed **noise floor** |
| **D** | `--w-u0 0.5 → 0` | 0 | V0 | `E-REFCV6-D` — **one flag, free, run FIRST** |
| **Db** | " | 1 | D | D's own floor |
| **A** | `--agents off → head` | 0 | V0 | `E-REFCV6-A` |
| **Ab** | " | 1 | A | A's own floor |
| **B** | `--wp-index off → on` | 0 | **A** | `E-REFCV6-B` ⛔ requires A |
| **Bb** | " | 1 | B | B's own floor |
| **C** | `--w-tac-goal 0 → <PI>` | 0 | V0 | `E-REFCV6-C` ⚠️ weight is a **PI decision** |
| **Cb** | " | 1 | C | C's own floor |

### 3.2 ⭐ MEASURED TODAY — the namespace diff, every pair

`code/check_one_variable.py`, through the trainer's own `build_parser()`
(md5 `4e9c9834f20d6c50b35c163aae4ab4a8`) · `raw/one_variable_probe.json`:

```
[OK]   A vs V0  (3 keys): agents=LEVER, w_agent=CONSTITUTIVE, out=BOOKKEEPING
[OK]   B vs A   (2 keys): wp_index=LEVER, out=BOOKKEEPING
[OK]   C vs V0  (2 keys): w_tac_goal=LEVER, out=BOOKKEEPING
[OK]   D vs V0  (2 keys): w_u0=LEVER, out=BOOKKEEPING
[OK]  V0b vs V0 (2 keys): seed=BOOKKEEPING, out=BOOKKEEPING     (and Ab/Bb/Cb/Db likewise)
verdict = PASS
```

⇒ **every lever pair moves EXACTLY ONE key classified `LEVER`.**

### 3.3 ⛔ Why arm A's diff is TWO keys and not one — an ENFORCED classification, not a note

`w_agent` is **CONSTITUTIVE**: the flag without which the named lever *cannot be built at all*,
because `refc_v3_train.py:583` refuses `--agents head` at `w_agent <= 0` and
`AGENT_WEIGHT_DEFAULT = 0.0`. It is **not a second lever** — there is no argv in which `agents`
moves alone.

⛔ **This is enforced, not documented.** `arms.CONSTITUTIVE` allow-lists exactly `{"A": ("w_agent",)}`
with the guard's file:line, and the checker classifies **any** third key as `VIOLATION`. `--agent-join`
does not appear because it is in BASE for every arm (§2).

### 3.4 ⛔ `anchors` is PARITY-PINNED, not bookkeeping

Every arm points at **one** `anchors.pt`. A per-arm anchor vocabulary would be a **hidden second
lever**, so a differing `anchors` key is classified `VIOLATION_PARITY`, never tolerated as a path
difference. The gate additionally records its md5 (§6, C6).

### 3.5 ⭐ THE MUTATION PROOF — the one-variable check CAN go RED

*A check that shares the defect it checks for is green forever* — four measured instances in one
night. `code/mutation_proof.py` re-introduces six defects **this programme actually suffered** and
requires each to turn the check RED. **MEASURED: baseline GREEN, 6 / 6 KILLED**
(`raw/mutation_proof.json`):

| mutant | the real defect | killed by |
|---|---|---|
| **M1** arm D also carries `--wp-index on` | the `--v2` conflation failure | `wp_index (VIOLATION)` |
| **M2** arm D silently moves `--lr 1e-4 → 3e-4` | *"an arm silently changed the effective lambda alongside its named lever"* | `lr (VIOLATION)` |
| **M3** replicate `V0b` carries seed 0 | a noise-floor arm that measures nothing | *"replicate carries the SAME seed as its treatment"* |
| **M4** arm D points at its own `anchors_D.pt` | a hidden second lever | `anchors (VIOLATION_PARITY)` |
| **M5** ⭐ arm D sets `--w-u0 0.9` instead of `0` | **exactly ONE key differs** — a key-*counting* check passes this happily while the lever is off | *"arm is 0.9, pre-registered target is 0.0"* |
| **M6** arm B paired against V0 instead of A | `--wp-index on` refuses `--agents off` | `agents` + `w_agent (VIOLATION)` |

⭐ **M5 is the discriminating one and the reason the checker asserts VALUES, not just key counts.**
A positive-only assertion (*"exactly one key differs"*) passes M5, because the mutant really does
differ in exactly one key — it is the control wearing D's name. `EXPECTED_MOVE` states each lever's
`from`/`to` as **literals**, never as an expression over the trainer's own defaults, because an
expectation computed from the code under test measures **determinism, not correctness**.

---

## 4. The launch commands

**BASE**, verified `MATCH` against the banked `config.json['argv']` (65/65 tokens):

```
--arm hier --size base --image-hw 256 640
--v2-cache /root/data/train   --v7-labels  <…>/s2_labels_v7.2_train.jsonl.gz
--eval-cache /root/data/eval  --eval-labels <…>/s2_labels_v7.2_eval.jsonl.gz
--eval-every 500 --eval-batches 8
--batch 20 --workers 6 --prefetch-factor 1 --v2-lru 24
--lr 1e-4 --warmup 2000 --log-every 50 --save-every 500 --u8-batches
--nav-from-v7 --ego-state-inject --ego-dropout 0.5
--anchors <SHARED>/anchors.pt --n-anchors 117 --anchor-v0-conditioned
--anchor-control-units alat --sel-accel-max 2.0
--sampler ddim --w-u0 0.5 --sel-refined --sel-score-emitted
--goal-str --tac-goal-tok-head --agents off
--agent-join <join>                       # ⛔ §2 — in EVERY arm, control included
--steps <BUDGET> --seed <0|1> --out <per-arm>
```

| arm | tokens CHANGED from BASE |
|---|---|
| **V0** | *(none)* |
| **D** | `--w-u0 0` |
| **A** | `--agents head` **+** `--w-agent <W>` (constitutive) |
| **B** | `--agents head --w-agent <W>` **+** `--wp-index on` |
| **C** | `--w-tac-goal <PI>` |
| **\*b** | `--seed 1` |

⛔ **`--w-u0 0` is passed EXPLICITLY rather than dropping the flag.** `U0_WEIGHT_DEFAULT` is already
`0.0`, so omitting it would be silently equivalent — and `config.json` would then be **unable to
distinguish the operator's choice from a default**. The run record must say a human chose 0.

⚠️ Every other WP-B knob keeps its default in both arms, and `--wp-index off` refuses any non-default
knob, so a stray `--wp-index-mode shuffle` cannot ride along in a record while doing nothing.

### 4.1 ⛔ TWO VALUES ARE NOT MINE TO PICK

| flag | status |
|---|---|
| `--w-tac-goal` | ⛔ **PI DECISION — queue item 10 / `D-TACGOAL-2`.** A third tactical term at 0.05 either **raises the budget to `MANEUVER_WEIGHT` 0.15** (leaving lat/lon untouched, so the arm stays paired with the banked one) **or forces a `/3.0` re-split** (which changes lat/lon pressure and is a **separate arm, not a tweak**). Arm C is built **parameterised**; `arms.py` raises `PIDecisionRequired` rather than inventing a value. Arm C's own primary bar is already pre-registered elsewhere (`…/2026-09-06-label-wiring/PREREG_D-TACGOAL-1.md`: `TRAFFIC_LIGHT_REACT_RED` per-class recall ≥ 0.35 against a majority control reading exactly 0.0) and is **not re-litigated here**. |
| `--w-agent` | ⚠️ **`1.0` HAS PROVENANCE BUT IS NOT A PRE-REGISTERED VALUE — the PI should confirm it.** See §4.2. ⛔ **`arms.py` still refuses to invent it**: the operator passes it explicitly, so the record shows who chose it. |

### 4.2 ⭐ What `--w-agent` actually has behind it — MEASURED, and precisely scoped

The trainer's default is `0.0`, which **refuses** `--agents head`, so a value must be supplied. I
searched for a registered one rather than guessing:

* ⭐ **`1.0` is the value the TINY-RIG ARM ACTUALLY CARRIED.** `refc_v3_train.py:2593-2594`, in the
  trainer's own source: *"MEASURED on the tiny rig 2026-09-05: with `--agents head --w-agent 1.0` on
  a dataset that carries no `obstacle.offline` join…"*. It is also the value used uniformly across
  the codebase's refusal messages, `effective_weights.py:191`'s documented
  `--agents off --w-agent 1.0` family, `refc.py:3439`, `refcv5_preflight.py`, and four test fixtures.
* ⛔ **But it is NOT annotated as a pre-registered launch value, and this codebase annotates those
  when they exist.** `GOAL_POINT_WEIGHT_DEFAULT` (`:127-131`) reads *"1.0 is the **PRE-REGISTERED**
  launch value (`…/2026-09-06-goal-point/PREREG.md` §9, `--goal-point-w 1.0`)"*.
  `AGENT_WEIGHT_DEFAULT` (`:121`) carries **no such line** — only *"WP-6: the GT-supervised
  detection set loss"*. The absence is meaningful precisely because the convention exists.
* ⚠️ **And the run that source line describes FAILED** — its detector was never supervised because
  the join was missing. So `1.0` is *the weight that arm carried*, **not** a weight demonstrated to
  balance against the planner loss at 108 M.

⇒ **Recorded as `MEASURED (provenance) / NOT PRE-REGISTERED`.** `1.0` is the defensible default and
the one arm A should carry absent a ruling, but ⛔ **it is the PI's to confirm**, and if a different
value is chosen the arm is unchanged apart from that one constitutive token.

---

## 5. SUCCESS and FAILURE, committed in advance

⛔ Every bar is a **ratio to a floor measured in the same panel** (header). `Δ` is always
`arm − PAIRING[arm]`, paired, same windows, episode-cluster bootstrap. Every table prints **n
windows and n episodes**.

### 5A. ⭐⭐ THE BAR IS EXECUTABLE CODE, AND THE NON-REGRESSION CLAUSES CANNOT BE DROPPED

The clauses below are implemented in `code/verdict_refcv6.py`, which **REFUSES to emit `SUCCESS`
when a required clause's data is ABSENT**.

⭐ **This is the half most likely to be lost, and prose could not protect it.** A clause written only
in a paragraph is dropped by being *not mentioned* in the results table — which is exactly how
ADE-only reports kept going out after four families were made binding. So `MISSING_DATA` is a
**distinct verdict from `FAIL`**, and neither is `SUCCESS`:

* **absence blocks the verdict** rather than passing it, and
* absence is **not** scored as a failure either — which would invite *deleting* a family to "fix" it.

### 5B. The clauses

| # | family | clause |
|---|---|---|
| **P1** | ADE | `arm` beats **`ha0_ext`**: paired CI **separated** **AND** relative margin **≥ 0.10** |
| **P2** | ADE | `arm` beats **`ha`**: separated **AND** margin ≥ 0.10 |
| **P3** | ADE | the ADE effect is **≥ 3×** the replicate floor measured in THIS panel |
| **L1–L4** | ⭐ **LATERAL non-regression** | **heading**, **yaw-rate**, **cross-track**, **masked curvature** — none may be separably WORSE than the control by more than that metric's own replicate floor. Reference (refcv5-v2 vs refcv4b): **1.2121**/1.5489 · **1.0534**/1.4542 · **0.0994**/0.1226 · **0.003485**/0.004030. ⛔ Masked curvature must be reported **with the straight-line floor beside it** (refcv5-v2 reads **0.512×** it) or the clause is `MISSING_DATA`. |
| **S1** | ⭐ **STRATEGIC non-regression** | **route accuracy** may not be separably worse than the control, **and `n` must be > 0**. ⛔ `n = 0` is `MISSING_DATA`, never a pass — **that is refcv4b's exact state and it is precisely what this clause guards.** Reference: **0.7708** [0.7146, 0.8254], κ 0.4614, n 3,622, chance 0.3333, from **771 parameters**. |
| **G1** | LONGITUDINAL | **speed MAE and along-track REPORTED with n** — the axis refcv5-v2 lost (0.2919 vs 0.2540; 0.2655 vs 0.2348) and the axis carrying **92.2 %** of the `os − ha` gap |
| **T1** | TACTICAL | tactical metrics **reported with n** (refcv5-v2's lateral κ **0.8193** vs 0.7374) |

⛔ **SUCCESS requires P1 ∧ P2 ∧ P3 ∧ L1..L4 ∧ S1 ∧ G1 ∧ T1.** Any one failing is a **FAIL as
written**. ⛔ **Clearing the CI while missing the 0.10 margin is a FAIL**, not a partial win.

⛔ The estimator and tier are **preconditions**, not clauses: a panel naming `overlapping_holdout_se`
(or `heldout`, or `jackknife`), or stamped anything but **T1**, is `REFUSED` outright and yields no
verdict at all.

### 5C. ⭐ MEASURED TODAY — the clauses are PROVEN undroppable

`code/verdict_dropproof.py` · `raw/verdict_dropproof.json`. A deliberately **generous** reference
panel (clears ADE against both baselines with room to spare) reads `SUCCESS`; each mutant then fails
for exactly one reason — the dropped clause. **MEASURED: baseline `SUCCESS`, 13 / 13 mutants
BLOCKED.**

| mutant | verdict |
|---|---|
| omit `LATERAL.heading_err` / `yaw_rate_err` / `cross_track` / `curvature_mae_masked` (4 mutants) | `MISSING_DATA` |
| omit the **whole LATERAL family** (ADE-only reporting) | `MISSING_DATA` |
| omit the **whole STRATEGIC family** | `MISSING_DATA` |
| `route_acc` present but **`n = 0`** (refcv4b's state) | `MISSING_DATA` |
| masked curvature **without** its straight-line floor | `MISSING_DATA` |
| omit the `ha` baseline (report only `ha0_ext`) | `MISSING_DATA` |
| omit the **replicate floor** | `MISSING_DATA` |
| ADE **separated but margin 0.01 < 0.10** | ⭐ **`FAIL`** |
| estimator = `overlapping_holdout_se` | `REFUSED` |
| tier stamped **T0** instead of T1 | `REFUSED` |

### 5D. THE FAILURE TWINS — each names its next lever

⛔ Rule Zero: *"X is refuted"* is never a finished turn.

| if | then, reported as written, and the next lever |
|---|---|
| **D fails** (`--w-u0 0` does not recover longitudinal) | ⛔ **REFUTED: the x0 term is not the longitudinal cost.** Our invention is exonerated and the cost lies elsewhere. ⭐ Next lever, already implemented and aimed at the same axis: **the V2 RL stage** — `H-DDA-5`'s gain profile is **EP +5.3, DAC +1.7, NC/TTC/comfort flat**, a longitudinal-scale effect, against our **92.2 %** along-track gap. ⛔ Blocked on PI item 8 (the 487-vs-488 cold start). |
| **D succeeds** | the arm becomes the new BASE for A/B/C, and **each of those is re-diffed against it** — the one-variable check is re-run, never assumed. |
| **A fails** | ⛔ **the 17 M tiny-rig gate result REPRODUCES at 108 M** ⇒ auxiliary-detection capacity competition is **not** a small-trunk artifact. ⭐ Next lever: `--agents oracle` (WP-C's approved rung) to price the **address** separately from the **detector** — if the oracle also fails, the mechanism is refused outright and coupling (2) is the remaining DiffusionDrive lever. |
| **A succeeds, B fails** | the agent **content** helps but the waypoint **address** does not. ⭐ WP-B's own pre-registered ladder applies unchanged: `--wp-index-mode const` ≈ B ⇒ the address carries no anchor-specific information ⇒ **`--wp-index-radius-m 12`** (a hard local gate makes the address decisive rather than advisory); `--wp-index-mode shuffle` ≈ B ⇒ the gain is **capacity, not content** ⇒ drop to `--wp-index-hidden 8`. |
| **C fails** | the 22-token vocabulary does not pay at this weight. ⛔ **A weight sweep after seeing this is a NEW pre-registration, not a continuation.** |
| **any arm regresses LATERAL or STRATEGIC while winning ADE** | ⛔ **FAIL as written** — that trade is precisely what refcv6 exists to stop paying. Not tuned around. |
| **every arm sits inside its replicate floor** | ⛔ **UNDERPOWERED, not REFUTED.** ⭐ The cheapest next experiment is **SEEDS, NOT WINDOWS** — more episodes narrow an interval that was already answering the wrong question. |

### 5E. `n` and `d`, in every table

Every table prints its `n` (windows **and** episodes) and, for any probe, its `d`. ⛔ **`n ≪ d` is
UNDERPOWERED BY CONSTRUCTION, not a negative** — 2,050 features on ~700 rows once made validation
correctly pick maximal regularisation, every arm read exactly the no-information value, and the panel
nearly concluded *"the latent carries no dynamics"*.

⚠️ **And on a stochastic planner there is a THIRD variance.** refcv6 inherits `--sampler ddim` and
the emitted-fan selection, so **the same checkpoint evaluated twice does not give the same answer**.
⇒ each arm's panel must **name which question its separated CI answered** — episodes, training run,
or inference run — and the replicate arms above measure the **training** one. ⛔ An effect smaller
than the measured inference-seed floor is not an effect.

---

## 6. THE PRE-LAUNCH GATE — what it refuses on

`code/prelaunch_gate.py`. ⛔ **The verdict is the JSON artifact's `"verdict"` field, never the exit
code.** *"The admissible evidence that this gate did not run is the MISSING JSON, never the exit
code."* MEASURED 2026-09-07: a 25-minute `timeout` killed `pod_currency_audit.py` with **zero lines
of output and no `--json` written**, and its wrapper printed **`GATE_EXIT=0`** — a timed-out,
output-less **pre-launch gate** reporting clean. `launch_refcv6.sh` reads the file and refuses on
anything but `PASS`.

⛔ **`INCONCLUSIVE` IS NOT A PASS.** A check that could not run has not passed.

| # | check | refuses when | status today |
|---|---|---|---|
| **C1** | **BASE provenance** | `arms.BASE_V5V2` ≠ the banked `config.json['argv']` | ✅ **PASS** — `MATCH`, 65/65, `first_difference: null` |
| **C2** | **one variable** | any pair moves more than its declared lever, on parsed namespaces | ✅ **PASS** — 9 pairs, 0 refused; 6/6 mutants killed |
| **C3** | **import closure** | `MISSING_REMOTE > 0` **or** `DRIFT > 0` over the trainer's real import closure (`launch_closure_audit.py --verify-import`) | ⏳ needs a box |
| **C4** | **label ∩ join coverage** | coverage **< 0.90** | ✅ **PROVEN BOTH WAYS** (below) |
| **C5** | **20-step smoke** | `metrics.json` absent, non-finite, or **loss CONSTANT** | ⏳ needs a box |
| **C6** | **anchor parity** | the shared `anchors.pt` is unreadable/empty; a per-arm path is caught by C2 as `VIOLATION_PARITY` | ⏳ needs the file |

### 6.1 ⭐ C4 — the manufactured-negative guard, MEASURED both ways today

⛔ **The failure this exists for:** a join sharing only **182** clips with v7.2 is
**182 / 4,572 = 3.98 %** coverage. The arm would have trained on ~4 % of its intended supervision and
read as *"the lever does not help"* — a **MANUFACTURED NEGATIVE**, which is worse than a crash
because it looks like a result.

`raw/coverage_proof.json`, on synthetic fixtures built to the two real shapes:

| join | clips | coverage | verdict |
|---|---|---|---|
| the real B1 TRAIN shape | 4,427 / 4,572 | **0.968285** | ✅ **PASS** (and it reproduces `D-B1TRAIN-JOIN-1`'s 0.9683) |
| ⛔ the 182-clip disaster | 182 / 4,572 | **0.039808** | ⛔ **FAIL** |

⛔ **A zero here is never reported as coverage.** If either side yields an empty id set the check
returns `INCONCLUSIVE`, because *an empty intersection from a file that could not be READ is
indistinguishable from a genuine absence* — the same-breath positive control both sides must pass.

### 6.2 C3 — and the artifact that looks like a catastrophe

⚠️ `MSYS_NO_PATHCONV=1` is set by the gate. Without it, MSYS rewrites `--remote-root` and the audit
reports **120/120 `MISSING_REMOTE`** — *"a clean, plausible, catastrophic-looking finding that is
pure artifact."* ⇒ the gate additionally treats **`MISSING_REMOTE == n_rows`** as `INCONCLUSIVE`,
never as a real finding.

⛔ md5 agreement proves **transfer, not function** (C99: three green md5s on a 2.6×-stale dependency
that was never listed *because it had not been edited*), which is why `--verify-import` is required
rather than optional.

### 6.3 C5 — asserting on CONTENT

The smoke runs the **real trainer** for 20 steps and then **reads `metrics.json`**, requiring finite
loss values that **actually moved** (a constant loss is a disconnected graph, not a converged one).
⛔ Never `ls`, and never the exit code: *a decode that raises into a pre-allocated memmap leaves a
full-size file of zeros and the job can still exit 0*. A stale `metrics.json` is deleted first, so a
previous run's artifact cannot pass this gate.

---

## 7. THE SUPERVISOR — each rule from a failure that cost a run

`code/sup_refcv6.sh`, `code/assert_supervisor.sh`, `code/launch_refcv6.sh`. **MEASURED: all three
pass `bash -n`; an audit of every executable child-spawning line finds 6 of 6 carrying `200>&-`.**

1. ⛔ **The lock fd is closed on EVERY child** — the `nohup` trainer, both `python3 -` heredocs, and
   **both `sleep`s**. A child inherits every open fd, so a trainer (or a stray `sleep 180`) ends up
   holding the supervisor's lock for its entire life, and no replacement can ever start. ⭐ **`200>&-`
   on the trainer alone was shipped once and failed the same day** — the holder was the supervisor's
   own poll child. The lock-holder diagnostic (`/proc/*/fd` + cmdline) is printed **in the failure
   message itself**, because the answer has twice been a process nobody suspected.
2. ⛔ **The done-marker is written on completion and checked at startup**, so a finished run is never
   resurrected. MEASURED precedent: a run that finished 2026-08-09 was relaunched for 2 days; when
   the crashing bug was fixed a relaunch **succeeded**, resumed from a stale ckpt, and began
   overwriting `config.json` / `metrics.json` / `ckpt.pt` beside a live eval.
3. ⛔ **Kill by explicit PID** (`train.pid`, `<arm>.sup.pid`). `pkill -f <trainer>` self-matches the
   ssh command and kills your own session, returning empty output that reads exactly like *"nothing
   was running"*.
4. ⛔ **The supervisor assertion reads `/proc/<pid>/cmdline`**, never `grep -c` — which self-matches
   its own echoed command line in a PTY. It asserts the live pid is a `sup_refcv6` **for this arm**,
   because a pid file alone passes for a reused pid. ⭐ Every failure in this family **reports success
   and leaves nothing running**, so the assertion runs after every start and `launch_refcv6.sh`
   exits non-zero if it fails.
5. ⛔ **The progress marker is disjoint from anything grepped**: counts are computed pod-side and
   emitted as `ZZ<arm>-<step>-<steps>-<errs>-<launch>ZZ`; the error pattern is **assembled at
   runtime** so the command line cannot contain the literal it searches for. MEASURED three times,
   worst case a `Traceback CUDA out of memory` reported on a healthy run 3 minutes in.
6. ⛔ **Completion is asserted on `metrics.json`, never on an exit code**, and no pipe is placed
   between a command and the `$?` that is read.
7. ⚠️ **The manifest is sourced ONCE at startup.** To change a live run: edit the manifest → kill the
   **supervisor** first → kill the trainer → start a fresh supervisor. ⚠️ And **never `sed -i` a
   running supervisor** — bash reads a script lazily by byte offset.
8. ⚠️ **`refc_v3_train.py` has no `--resume`**, so a relaunch RESTARTS the arm. The supervisor says so
   in its log and `MAX_RELAUNCH=1` forbids it.

---

## 8. The GPU arm — costed, and NOT started

⛔ **Nothing has been launched. There is no A40 — the pod was stopped and is gone.** Thor and the
dev-box 4060 were used for CPU/import-level checks only; no GPU or RAM load was added to any box.

| | |
|---|---|
| arms | **V0, V0b, D, Db** first (4). A/Ab/B/Bb and C/Cb follow, gated on §4.1's two values. |
| budget | `full` = **40,284** (refcv5-v2's own, so a full-budget arm is additionally comparable to the banked numbers as a SECONDARY read) · `cut` = **12,000** (the WP-D precedent: the shortest budget past `--warmup 2000` at which the 500-step milestone gate reads a curve) |
| rate | ⚠️ **NOT MEASURED for refcv6.** refcv5-v2's own receipt is **4.0 s/step on an A40** (and it retracts its own earlier "~1.2 s/step", which came from an early window inside `--warmup`). Thor measured **4.21 s/step** aux-off on the WP-D config. ⛔ Neither travels to an unknown box; re-measure before quoting a wall-clock. |
| ⭐ cheapest first cut | **D + V0 + their replicates.** D is **one flag and free**, and the pair V0/V0b establishes the **replicate floor every other arm's bar is stated against** — so this cut is a prerequisite for the rest, not merely the cheapest. |

⚠️ **`--agent-join` adds ~2.1 GB RSS to every arm including the control** (§2), and `--v2-lru 24` /
`--workers 6` are A40 settings that must be re-checked on whatever box runs this.

---

## 9. What this pre-registration does NOT claim, and what it OWES

* ⛔ **No capability claim of any kind.** refcv6 does not exist; no arm has run.
* ⛔ **It does not pick `--w-tac-goal`** (PI item 10) **or `--w-agent`** (§4.1). Arm C and arm A are
  built parameterised and `arms.py` **refuses** rather than inventing either.
* ⚠️ **`--w-agent` is `MEASURED (provenance) / NOT PRE-REGISTERED`** (§4.2). `1.0` is the value the
  tiny-rig arm carried (`refc_v3_train.py:2593-2594`) and the codebase's uniform example, but it is
  **not** annotated as a pre-registered launch value the way `--goal-point-w 1.0` is, and the run
  that source line describes **failed for an unrelated reason** (missing labels). ⛔ **The PI
  confirms it**; `arms.py` refuses to supply it silently.
* ⚠️ **OWED — fact 10 is INHERITED.** refcv5-v2's per-family values are quoted from the architecture
  review, not re-read from raw eval JSON by this package. They are the **reference column** of §5B;
  ⛔ **re-read them from the raw JSON before scoring any refcv6 panel.**
* ⚠️ **`V0` is not a replication of refcv5-v2** (§2), and no claim here treats it as one.
* ⚠️ **The A-arm hypothesis is genuinely open.** `--agents head` FAILED its two-seed tiny-rig gate at
  17 M, and the deranged-join arm relocated the cost to auxiliary-task capacity competition. Whether
  that survives at 108 M is a **HYPOTHESIS**, and §5D commits the reading if it does not.
* ⛔ **The oracle rung prices the ADDRESS, not the system.** `--agents oracle` feeds GT boxes at
  inference and is **inadmissible as a capability claim**; `--agents head` is what the paper does.
* ⛔ **This package launched nothing and changed no model code.** Every lever it uses already existed.

---

## 10. Deliverables

| artifact | path |
|---|---|
| this pre-registration | `Project Steering/PREREG_REFCV6.md` |
| register entries (`E-REFCV6-A/B/C/D`) | `Project Steering/GOALS_AND_CLAIMS.md` |
| argv single source of truth | `…/2026-09-10-refcv6-build/code/arms.py` |
| one-variable checker (real `build_parser`) | `…/code/check_one_variable.py` |
| its mutation proof (6/6 killed) | `…/code/mutation_proof.py` |
| the bar as executable code | `…/code/verdict_refcv6.py` |
| its drop-proof (13/13 blocked) | `…/code/verdict_dropproof.py` |
| the pre-launch gate | `…/code/prelaunch_gate.py` |
| supervisor · assertion · launcher | `…/code/sup_refcv6.sh`, `assert_supervisor.sh`, `launch_refcv6.sh` |
| measured evidence | `…/raw/*.json` |
| the result write-up | `…/2026-09-10-refcv6-build/RESULT.md` |

---

## ⛔ CORRECTION 2026-09-11 — THE STRATEGIC CLAUSE'S PREMISE WAS FALSE. The clause SURVIVES; its reference value CHANGES.

⛔ **The body above is left exactly as pre-registered and is NOT rewritten.** A pre-registration
whose text is edited after the fact is no longer a pre-registration. This appendix corrects the
record; the committed criteria stand.

### What was wrong

This document states in two places that refcv5-v2 is the **"only arm"** with a strategic output,
and its drop-proof table names **`route_acc` present but `n = 0`** as *"(refcv4b's state)"*.

⛔ **Both are FALSE, and the error is mine.** The comparison table I supplied labelled a column
"refcv4b" when it was in fact the **`ha` hold-action control**. Verified directly from
`…/2026-09-07-refcv5-v2-comparison/raw/refcv5-v2_vs_refcv4b.json`, `four_families.base.strategic_computed`:

| arm | route accuracy | κ | n | 95 % CI |
|---|---|---|---|---|
| refcv5-v2 | 0.7708 | 0.4614 | **3,622** | [0.7146, 0.8254] |
| **refcv4b** | **0.7791** | **0.4864** | **3,622** | [0.7248, 0.8318] |
| `ha` (hold-action control) | — | — | **0** | — |

⇒ **refcv4b has a strategic output, at the same `n`, and scores slightly HIGHER than refcv5-v2.**
The `n = 0` belongs to the hold-action control, not to refcv4b.

### What it changes — and the change makes the bar STRICTER, not looser

⭐ **The clause itself is unaffected in structure**: *route accuracy may not be separably worse than
the control, and `n` must be > 0.* ⛔ **What changes is the reference value it is read against.**

⇒ **The strategic reference for refcv6 is `0.7791` (refcv4b), not `0.7708` (refcv5-v2)** — the
**better** of the two arms. ⚠️ This is a correction of a factual premise, **not a goalpost moved
after seeing data**: it raises the bar, and it is recorded here rather than applied silently.

⚠️ **The drop-proof case labelled *"(refcv4b's state)"* keeps its behaviour and loses its label.**
An arm reporting `route_acc` with `n = 0` must still return `MISSING_DATA` — that test is correct
and its mutant still fails. Only the parenthetical attribution was wrong.

### ⛔ What must NOT be concluded from this

⛔ **"refcv5-v2's strategic head is worthless"** does not follow. Both arms sit at ~0.77–0.78 with
**overlapping** intervals; nothing separates them. ⭐ The **771-parameter** observation stands
untouched and remains the striking one: a head that size reaching ~0.77 against a **0.3333** chance
baseline, beside a **106,067,312**-parameter core, means the strategic level is nearly free — and
that is true of **both** arms, which strengthens the claim rather than weakening it.

⭐ **And one clause gets materially STRONGER from the same correction.** On masked curvature MAE
against the straight-line floor, refcv5-v2 reads **0.5123×** — half the error of a plan that never
steers — while **refcv4b reads 1.1982×, ABOVE the floor**. refcv4b tracks curvature **worse than not
steering at all.** Against the mislabelled control that gap read 1.16×; against the real arm it is
**2.34×**. ⇒ **LATERAL is the one family where a trained arm separably beats every control**, and it
is now the strongest justification for clause **L1**.

⚠️ Also corrected, from the same source: refcv5-v2 **loses** cross-track (0.0994 vs refcv4b's
**0.0978**) and **loses** tactical lateral κ (0.8193 vs **0.8277**). Both were previously reported
as wins against the control's values. And refcv5-v2 **wins** accel MAE (**0.3596** vs 0.4346), which
had never been reported at all.

⛔ **Root-cause class: a number that is real, correctly transcribed, and attached to the WRONG ARM.**
*"A number carries its ARM and its ARTIFACT PATH, or it is not quotable."* Logged in
`Project Steering/RETRACTION_LOG.md`.

---

## ⭐ FACT 10'S REFERENCE COLUMN — RE-READ FROM RAW, 2026-09-11. The prereg's OWED item is discharged.

⛔ The values in fact 10 were **INHERITED** from `REFCV6_ARCHITECTURE_REVIEW.md` §1 and had **never
been re-read from raw eval JSON**. The prereg said they must be before any refcv6 panel is scored.
**They now have been**, from the repo's own banked artifact:
`TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-07-refcv5-v2-comparison/raw/refcv5-v2_vs_refcv4b.json`
— **T1**, 4,823 windows / 141 episodes.

| metric | refcv5-v2 | refcv4b | `ha` hold-action | `ha0_ext` extrap floor |
|---|---|---|---|---|
| **LONG** speed MAE (m/s) | 0.2919 | 0.2900 | **0.2540** | **0.2540** |
| **LONG** along-track (m) | 0.2655 | 0.2544 | 0.2348 | **0.2341** |
| **LONG** accel MAE (m/s²) | 0.3596 | 0.4346 | **0.3166** | **0.3166** |
| **LATE** heading (°) | **1.2121** | 1.2964 | 1.5489 | 1.4322 |
| **LATE** yaw-rate (°/s) | **1.0534** | 1.7368 | 1.4542 | 1.3395 |
| **LATE** cross-track (m) | 0.0994 | **0.0978** | 0.1226 | 0.1070 |
| **LATE** curvature MAE (1/m) | **0.0035** | 0.0081 | 0.0040 | 0.0037 |
| **LATE** ratio to straight-line floor | **0.5123** | 1.1982 | 0.5925 | 0.5457 |
| **STRATEGIC** route acc | 0.7708 (κ 0.4614) | **0.7791 (κ 0.4864)** | — | — |

⇒ **STRATEGIC n = 3,622 for BOTH trained arms.** ⛔ The `n = 0` belongs to the hold-action control.

### ⛔⛔ WHAT THE RE-READ MAKES VISIBLE, AND THE INHERITED TABLE DID NOT

⭐ **On LONGITUDINAL, the DO-NOTHING control beats BOTH trained arms on ALL THREE metrics** — speed
MAE **0.2540** against 0.2900/0.2919, along-track **0.2348** against 0.2544/0.2655, accel MAE
**0.3166** against 0.4346/0.3596.

⇒ **Neither trained arm has learned to hold speed better than holding the last action.** That is the
88.7 %-longitudinal-gap statement made concrete, and it is the single most important number in the
table for refcv6's bar. ⛔ An arm that "improves ADE" while still losing to `ha` longitudinally has
not fixed the thing that is actually wrong.

⭐ **LATERAL is the mirror image and the only family where training clearly pays.** refcv5-v2 beats
**every** arm and **every** control on heading, yaw-rate and curvature — and on the ratio to the
straight-line floor it reads **0.5123** against `ha`'s 0.5925 and `ha0_ext`'s 0.5457, so it is better
than doing nothing *and* better than extrapolating. ⛔ **refcv4b reads 1.1982 — worse than all three
controls, i.e. worse than a plan that never steers.**

### ⇒ BINDING FOR SCORING refcv6

1. ⛔ Score against **these** values, not against the review's prose.
2. ⭐ The **LATERAL** non-regression clause is anchored on refcv5-v2's **0.5123** ratio — the best
   figure any arm has posted, and the only family where a trained arm separably beats every control.
3. ⭐ The **STRATEGIC** clause is anchored on **refcv4b's 0.7791**, the better of the two, per the
   2026-09-11 correction appendix.
4. ⚠️ **LONGITUDINAL has no trained arm worth anchoring on.** The reference there is **`ha` at
   0.2540 / 0.2348 / 0.3166** — the bar refcv6 has to clear is *doing nothing*, and nothing has
   cleared it yet.
