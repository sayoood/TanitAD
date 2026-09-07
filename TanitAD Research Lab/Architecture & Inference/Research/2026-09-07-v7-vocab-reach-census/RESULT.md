# A COMPLETE per-token reach census for the frozen v7 vocabulary

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-v7-vocab-reach-census/`
**Date:** 2026-09-07 · **Branch:** `agent/arch-inf-20260803` · **Repo HEAD at start:** `cb84c07`
**Mandate:** PI, 2026-09-06 — *"assure that all necessary vocab are used weiter as inputs like nav commands and max speed and the rest as training signals."*

---

## 0. The answer, in five lines

| class | n | who |
|---|---:|---|
| **`TRAINING_SIGNAL`** | **23** | 7 lat actions · 7 lon actions (refc, live) · 4 strategic goals + 5 strategic actions (⚠️ **v6 trainer only**, and behind `w_s2_goal` default 0.0). The tuples are wider — the other 8 slots are `NOT_EMITTED`. |
| **`INFERENCE_INPUT`** | **3** | `NAV_FOLLOW_ROAD`, `NAV_TURN_L`, `NAV_TURN_R` |
| **`AUDIT_OR_METRIC_ONLY`** | **18** | ⛔ every tactical GOAL token that is not also an action name — **the D-TLIGHT-1 condition, still open** |
| **`UNREACHED`** | **0** | — (every emitted token reaches *something*; 18 of them reach only an audit) |
| **`NOT_EMITTED`** | **8** | `ABORT_LC`, `YIELD_MERGE`, `EXIT_{LEFT,RIGHT}_FOLLOW_ROUTE`, `LANE_CHANGE_{L,R}_FOLLOW_ROUTE`, `PREPARE_EXIT_FOLLOW_ROUTE`, `PREPARE_LANE_CHANGE_FOLLOW_ROUTE` |

52 tokens, 52 classified, none left over. Evidence class **MEASURED** throughout
(`raw/census.json`); every count is from the v7.2 blobs and every reach verdict
from executed consumer code.

> ⛔⛔ **THE HEADLINE: the tactical-goal head is BUILT AND UNWIRED IN THE ARM
> THAT IS RUNNING.** Under the **live refcv5-v2 argv**, on the real model, with
> the trainer's own total loss and one `.backward()`: `tac_goal_tok_head` →
> **`grad_none = 2/2`, abs-sum `0` — NOT WIRED**. Its **11,286 parameters** (the
> run's own `param_breakdown`) receive no gradient at all, and all 22 tactical
> goal tokens — the four traffic-light ones included — are invisible to training.
>
> ⚠️ **THIS IS A CONFIRMATION AND A MEASUREMENT, NOT A DISCOVERY, AND IT MUST BE
> READ THAT WAY.** `GOALS_AND_CLAIMS.md` already carries
> **`D-TACGOAL-TRAINER-SEAM-OPEN`** — *"the head now BUILDS, is STAMPED and is
> ROLLABLE … **supervising** it still needs the two additive edits … and the
> `MANEUVER_WEIGHT` budget decision — an owner/PI call"*. That row is correct and
> this census agrees with it. What is **new** here is (a) the first **direct
> gradient** evidence rather than a code-reading — `p.grad is None`, the exact
> discriminator `tac_goal_head.py`'s own docstring names; (b) the fact that the
> **live run is carrying the dead 11,286 parameters right now**, because it
> passes `--tac-goal-tok-head`; (c) **why both existing guards are green on it**
> (§3.2); and (d) the same question asked of **every other token** (§2).
>
> ⚠️ Two sibling documents read the other way and will mislead a hurried reader:
> the `D-TACGOAL-1` register row headlines *"THE 22-TOKEN TACTICAL GOAL SET NOW
> REACHES A SUPERVISED HEAD"* and `stack/tests/test_tactical_label_reach.py`'s
> docstring says *"⭐⭐ GAP CLOSED 2026-09-06 — D-TACGOAL-1"*. Both are true of
> the **head** and false of the **trainer** — the *true-but-wrong-for-the-reader*
> class, and the reason this census exists as a standing instrument.

---

## 1. Method — and the rule that governs it

⛔ **A CROSS-CHECK MUST BE DERIVED INDEPENDENTLY OF THE VALUE IT CHECKS**
(CLAUDE.md, four measured instances in one night — one of them a census whose
test pinned a set *against the status that set produced*, permanently blind to an
extra member).

**Nothing here asks the vocabulary module whether a token is trained.** Two
derivations, then a comparison:

**EMISSION** — read from the blobs. Every string appearing in a token-bearing
field is counted, vocabulary member or not, so an *off-vocabulary* emission (the
`REDUCE_TO_FOLLOW_ROUTE` class, 11.24 % of `a_str` on the earlier v7 sample)
would be visible rather than dropped.

**REACH** — three **independent legs**, all of which must hold:

| leg | question | how |
|---|---|---|
| **1 PROJECTION** | does the real consumer function turn this token into a valid target/input cell? | run `tactical_class_ids` / `tactical_goal_targets` / `NavEmitter` / `_goal_audit` on a **real loaded record**, then a real head + real loss + `.backward()`, and read the gradient of that token's **head row** |
| **2 CONSUMER** | does a **trainer** actually call that projection and its loss? | literal marker strings in the named trainer, each read carrying a **same-breath control marker** that must also be found |
| **3 GRADIENT** | does a gradient actually land on the built module, under the arm's own argv? | `scripts/gradreach_probe.py` — real `RefCV3Model`, real `compute_losses_v3`, the trainer's **own** `losses["loss"]`, one backward, `p.grad` per module |

The declarations (`ROLE_OF`, `NOT_YET_EXTRACTABLE`, `TACTICAL_GOAL_NEEDS_PERCEPTION`,
`TACTICAL_GOAL_UNDERPOWERED`, the two floors) are read **only at the end**, only
to be compared. Every disagreement is reported as a finding; **none can change a
class**.

⚠️ **Why leg 3 was necessary and legs 1–2 were not enough.** Leg 1 says the label
*can* become a target; leg 2 says a trainer *names* the call. Only leg 3 answers
*"does a gradient land on it"* — and that is exactly where `D-TACGOAL-1` fails.
Conversely legs 1–2 catch what leg 3 cannot: a head that is **not built at all**
has no module to inspect.

### 1.1 The consumer surfaces (executed, not asserted from memory)

| surface | kind | projection | trainer | status |
|---|---|---|---|---|
| `tac_lat_ce` | loss target | `tactical_class_ids(...)[0]` | `refc_v3_train.py` | **CONSUMED** |
| `tac_lon_ce` | loss target | `tactical_class_ids(...)[1]` | `refc_v3_train.py` | **CONSUMED** |
| `tac_goal_bce` | loss target | `tactical_goal_targets(...)` | `refc_v3_train.py` | ⛔ **NO_CONSUMER** (`tac_goal_loss`, `TacGoalEmitter` absent) |
| `str_goal_ce_v6` | loss target | `HEADS['str_goal'].index(tok)` | `train_v6_staged.py` | **CONSUMED** (behind `w_s2_goal`, default 0.0) |
| `str_action_ce_v6` | loss target | `HEADS['str_action'].index(tok)` | `train_v6_staged.py` | **CONSUMED** (same gate) |
| `str_ce_refc` | loss target | — | `refc_v3_train.py` | ⛔ **NO_CONSUMER** — refc has **no strategic token head** |
| `tac_action_ce_v6` | loss target | `V72WindowSupervision` ids | `train_v6_staged.py` | ⛔ **NO_CONSUMER** — the ids are landed, no CE reads them |
| `nav_input` | **model input** | `NavEmitter(...)` | `refc_v3_train.py` | **CONSUMED** |
| `goal_audit` | audit | `_goal_audit(g_tac)` | `tactical_label_census.py` | **CONSUMED** |

⚠️ **A store is not a read.** `v7_labels.py` merely *puts* the goal set into
`V7Label.audit['goal_flags']`; the one real reader is
`stack/scripts/tactical_label_census.py` (repo-wide scan for `goal_flags`: 5 hits
in 4 files = the store, this reader, and two tests — with a `def ` control at
**17,726** hits, so the zero elsewhere is a claim about the *content*, not the
search). `stack/tanitad/eval/constraints.py` names `g_tac.goals.SPEED_BAND` in
its **module docstring only**, to explain why it may not be fed;
`four_families.TARGET_SPEED_BANDS_MPS` is an unrelated metric-tolerance constant.
Neither reads the label. Both looked like consumers to a first regex — the
docstring false-positive is the reason the surface registry names a *reader*, not
a *mention*.

---

## 2. The census, per token


#### `STRATEGIC_GOAL_TOKENS_V7` (8 tokens)

| token | class | n train | n eval | reached by | live refcv5-v2 |
|---|---|---:|---:|---|---|
| `EXIT_LEFT_FOLLOW_ROUTE` | NOT_EMITTED | 0 | 0 | s2_goal_loss (v6) | **no** &mdash; DEAD LOGIT |
| `EXIT_RIGHT_FOLLOW_ROUTE` | NOT_EMITTED | 0 | 0 | s2_goal_loss (v6) | **no** &mdash; DEAD LOGIT |
| `FOLLOW_ROUTE` | TRAINING_SIGNAL | 2948 | 93 | s2_goal_loss (v6) | **no** |
| `LANE_CHANGE_L_FOLLOW_ROUTE` | NOT_EMITTED | 0 | 0 | s2_goal_loss (v6) | **no** &mdash; DEAD LOGIT |
| `LANE_CHANGE_R_FOLLOW_ROUTE` | NOT_EMITTED | 0 | 0 | s2_goal_loss (v6) | **no** &mdash; DEAD LOGIT |
| `STOP_AT_FOLLOW_ROUTE` | TRAINING_SIGNAL | 454 | 15 | s2_goal_loss (v6) | **no** |
| `TURN_LEFT_FOLLOW_ROUTE` | TRAINING_SIGNAL | 574 | 11 | s2_goal_loss (v6) | **no** |
| `TURN_RIGHT_FOLLOW_ROUTE` | TRAINING_SIGNAL | 596 | 28 | s2_goal_loss (v6) | **no** |

#### `STRATEGIC_ACTION_TOKENS_V7` (7 tokens)

| token | class | n train | n eval | reached by | live refcv5-v2 |
|---|---|---:|---:|---|---|
| `HOLD_MAIN_ROAD` | TRAINING_SIGNAL | 2390 | 78 | s2_goal_loss (v6) | **no** |
| `PREPARE_EXIT_FOLLOW_ROUTE` | NOT_EMITTED | 0 | 0 | s2_goal_loss (v6) | **no** &mdash; DEAD LOGIT |
| `PREPARE_LANE_CHANGE_FOLLOW_ROUTE` | NOT_EMITTED | 0 | 0 | s2_goal_loss (v6) | **no** &mdash; DEAD LOGIT |
| `PREPARE_STOP_FOLLOW_ROUTE` | TRAINING_SIGNAL | 454 | 15 | s2_goal_loss (v6) | **no** |
| `PREPARE_TURN_L_FOLLOW_ROUTE` | TRAINING_SIGNAL | 574 | 11 | s2_goal_loss (v6) | **no** |
| `PREPARE_TURN_R_FOLLOW_ROUTE` | TRAINING_SIGNAL | 596 | 28 | s2_goal_loss (v6) | **no** |
| `RESUME_CRUISE_FOLLOW_ROUTE` | TRAINING_SIGNAL | 558 | 15 | s2_goal_loss (v6) | **no** |

#### `TACTICAL_GOAL_TOKENS_V7` (22 tokens)

| token | class | n train | n eval | reached by | live refcv5-v2 |
|---|---|---:|---:|---|---|
| `CORRIDOR_OFFSET` | AUDIT_OR_METRIC_ONLY | 860 | 25 | audit['goal_flags'] (audit only) | yes |
| `EVADE_IN_CORRIDOR` | AUDIT_OR_METRIC_ONLY | 240 | 6 | audit['goal_flags'] (audit only) | yes |
| `FOLLOW_LANE` | AUDIT_OR_METRIC_ONLY | 3629 | 119 | audit['goal_flags'] (audit only) | yes |
| `GAP_TARGET` | AUDIT_OR_METRIC_ONLY | 368 | 14 | audit['goal_flags'] (audit only) | yes |
| `LANE_CHANGE_L` | TRAINING_SIGNAL | 23 | 0 | audit['goal_flags'] (audit only), tac_lat CE (refc) | yes |
| `LANE_CHANGE_R` | TRAINING_SIGNAL | 15 | 1 | audit['goal_flags'] (audit only), tac_lat CE (refc) | yes |
| `MERGE` | AUDIT_OR_METRIC_ONLY | 79 | 5 | audit['goal_flags'] (audit only) | yes |
| `OVERTAKE_VEHICLE` | AUDIT_OR_METRIC_ONLY | 20 | 1 | audit['goal_flags'] (audit only) | yes |
| `REACT_ON_ONCOMING` | AUDIT_OR_METRIC_ONLY | 333 | 11 | audit['goal_flags'] (audit only) | yes |
| `SPEED_BAND` | AUDIT_OR_METRIC_ONLY | 4572 | 147 | audit['goal_flags'] (audit only) | yes |
| `STOP_POINT` | AUDIT_OR_METRIC_ONLY | 327 | 9 | audit['goal_flags'] (audit only) | yes |
| `TAKE_EXIT_L` | AUDIT_OR_METRIC_ONLY | 21 | 0 | audit['goal_flags'] (audit only) | yes |
| `TAKE_EXIT_R` | AUDIT_OR_METRIC_ONLY | 128 | 5 | audit['goal_flags'] (audit only) | yes |
| `TRAFFIC_LIGHT_REACT` | AUDIT_OR_METRIC_ONLY | 18 | 0 | audit['goal_flags'] (audit only) | yes |
| `TRAFFIC_LIGHT_REACT_GREEN` | AUDIT_OR_METRIC_ONLY | 363 | 15 | audit['goal_flags'] (audit only) | yes |
| `TRAFFIC_LIGHT_REACT_RED` | AUDIT_OR_METRIC_ONLY | 376 | 8 | audit['goal_flags'] (audit only) | yes |
| `TRAFFIC_LIGHT_REACT_YELLOW` | AUDIT_OR_METRIC_ONLY | 22 | 3 | audit['goal_flags'] (audit only) | yes |
| `TURN_L` | TRAINING_SIGNAL | 550 | 10 | audit['goal_flags'] (audit only), tac_lat CE (refc) | yes |
| `TURN_R` | TRAINING_SIGNAL | 518 | 16 | audit['goal_flags'] (audit only), tac_lat CE (refc) | yes |
| `YIELD` | AUDIT_OR_METRIC_ONLY | 609 | 17 | audit['goal_flags'] (audit only) | yes |
| `YIELD_FOR_TURN_L` | AUDIT_OR_METRIC_ONLY | 21 | 2 | audit['goal_flags'] (audit only) | yes |
| `YIELD_FOR_TURN_R` | AUDIT_OR_METRIC_ONLY | 20 | 0 | audit['goal_flags'] (audit only) | yes |

#### `TACTICAL_LAT_ACTIONS_V7` (8 tokens)

| token | class | n train | n eval | reached by | live refcv5-v2 |
|---|---|---:|---:|---|---|
| `ABORT_LC` | NOT_EMITTED | 0 | 0 | tac_lat CE (refc) | yes &mdash; DEAD LOGIT |
| `LANE_CHANGE_L` | TRAINING_SIGNAL | 23 | 0 | audit['goal_flags'] (audit only), tac_lat CE (refc) | yes |
| `LANE_CHANGE_R` | TRAINING_SIGNAL | 15 | 1 | audit['goal_flags'] (audit only), tac_lat CE (refc) | yes |
| `LANE_KEEP` | TRAINING_SIGNAL | 2958 | 99 | tac_lat CE (refc) | yes |
| `NUDGE_L` | TRAINING_SIGNAL | 488 | 14 | tac_lat CE (refc) | yes |
| `NUDGE_R` | TRAINING_SIGNAL | 592 | 21 | tac_lat CE (refc) | yes |
| `TURN_L` | TRAINING_SIGNAL | 550 | 10 | audit['goal_flags'] (audit only), tac_lat CE (refc) | yes |
| `TURN_R` | TRAINING_SIGNAL | 518 | 16 | audit['goal_flags'] (audit only), tac_lat CE (refc) | yes |

#### `TACTICAL_LON_ACTIONS_V7` (8 tokens)

| token | class | n train | n eval | reached by | live refcv5-v2 |
|---|---|---:|---:|---|---|
| `ACCELERATE` | TRAINING_SIGNAL | 998 | 40 | tac_lon CE (refc) | yes |
| `ADAPT_SPEED_FOR_CURVE` | TRAINING_SIGNAL | 995 | 23 | tac_lon CE (refc) | yes |
| `BRAKE_TO` | TRAINING_SIGNAL | 905 | 22 | tac_lon CE (refc) | yes |
| `CREEP` | TRAINING_SIGNAL | 135 | 6 | tac_lon CE (refc) | yes |
| `CRUISE` | TRAINING_SIGNAL | 1243 | 44 | tac_lon CE (refc) | yes |
| `FOLLOW` | TRAINING_SIGNAL | 165 | 10 | tac_lon CE (refc) | yes |
| `HOLD` | TRAINING_SIGNAL | 131 | 2 | tac_lon CE (refc) | yes |
| `YIELD_MERGE` | NOT_EMITTED | 0 | 0 | tac_lon CE (refc) | yes &mdash; DEAD LOGIT |

#### `NAV_COMMAND_TOKENS` (3 tokens)

| token | class | n train | n eval | reached by | live refcv5-v2 |
|---|---|---:|---:|---|---|
| `NAV_FOLLOW_ROAD` | INFERENCE_INPUT | 2897 | 96 | NavConditioner + core one-hot | yes |
| `NAV_TURN_L` | INFERENCE_INPUT | 811 | 13 | NavConditioner + core one-hot | yes |
| `NAV_TURN_R` | INFERENCE_INPUT | 864 | 38 | NavConditioner + core one-hot | yes |

<!-- tokens covered: 52 of 52 census rows -->

> ⚠️ `TURN_L`, `TURN_R`, `LANE_CHANGE_L`, `LANE_CHANGE_R` appear in **two**
> tuples. They are `TRAINING_SIGNAL` **as lateral ACTIONS** and unsupervised **as
> tactical GOALS** — the same distinction `tactical_label_census.py` has to spell
> out in prose (*"NO (name also a tac_lat ACTION class)"*). Reading their class
> as "the goal is supervised" is the exact misreading that census warns about.

---

## 3. The findings

### 3.1 ⛔⛔ The tactical-goal seam is open: 18 tokens emitted, no loss

Registered as **`D-TACGOAL-TRAINER-SEAM-OPEN`** and still open. This
section is the **gradient measurement** of that row, not a new claim.

The `AUDIT_OR_METRIC_ONLY` set **is** the D-TLIGHT-1 condition, one table wider:

```
CORRIDOR_OFFSET  EVADE_IN_CORRIDOR  FOLLOW_LANE  GAP_TARGET  MERGE
OVERTAKE_VEHICLE  REACT_ON_ONCOMING  SPEED_BAND  STOP_POINT
TAKE_EXIT_L  TAKE_EXIT_R  TRAFFIC_LIGHT_REACT  TRAFFIC_LIGHT_REACT_GREEN
TRAFFIC_LIGHT_REACT_RED  TRAFFIC_LIGHT_REACT_YELLOW  YIELD
YIELD_FOR_TURN_L  YIELD_FOR_TURN_R
```

MEASURED, three ways:

1. **Source (positive assertion, with control).** `refc_v3_train.py` contains
   `def compute_losses_v3(` (control, present) and contains **neither**
   `tac_goal_loss` **nor** `TacGoalEmitter` **nor** `tac_goal_logits`. Repo-wide:
   `tac_goal_loss` = 11 hits in 2 files (its own definition + one test);
   `TacGoalEmitter` = 9 hits in 2 files (its own definition + one test);
   `tac_goal_logits` = 13 hits in 2 files (`refc_v3.py:1320`, which *produces*
   it, + one test). Control `cross_entropy` = 82 hits in 35 files, so the search
   read real content.
2. **Gradient, on the live argv.** `tac_goal_tok_head` -> `grad_none = 2/2`,
   `|g| = 0`, verdict **NOT WIRED**. Siblings in the same run read
   `lat_head_tac` `|g| = 0.31`, `nav_to_tac` `|g| = 115.17` — so the probe was
   powered.
3. **The run's own record.** `param_breakdown.tac_goal_tok_head = 11286`;
   `seams.tac_goal_tok_head = {requested: true, cfg: true, built: true}`.

=> the live arm carries **11,286 parameters that cannot learn**, and the two
observations that motivated the head — *"the model never brakes for a red light"*
and *"lane changes never activate"* — remain unaddressed by it.

### 3.2 ⛔ Why both existing guards pass on this — a third question was missing

* `assert_seams_are_built` asks **"is the head BUILT?"** It is. Passes.
* `effective_weights_stamp_v3` enumerates **declared loss weights** and asks
  which build a graph. Its own preamble cites the right measurement (*"42/138
  optimizer tensors took no gradient ... 52.2 % of a declared trainable budget"*)
  and its `_discriminator` is exactly `p.grad is None`. But the live record
  stamps **5 terms** (`--w-u0`, `--w-agent`, `--agent-w-project`,
  `--agent-w-ground`, `--goal-point-w`) and **the tactical-goal head is not among
  them, because it has no weight flag at all.** An instrument that enumerates
  *weights* is structurally blind to a head with *no weight*.

⭐ This is the *"a check that shares the defect it checks for is green forever"*
family with the object swapped: neither guard is wrong; the **question** each
asks is narrower than the claim being hung on it. The missing question — *does a
gradient land on every built head?* — is what `scripts/gradreach_probe.py` asks,
and it is a one-line addition to the trainer's own preflight.

### 3.3 ⛔ The 15 strategic tokens have NO training path in the refc line

`refc_v3.py:828` is `self.str_goal_head = nn.Linear(d_ctx, 3)` — a **geometric**
bearing/distance readout. `--goal-str` supervises it via
`v3.strategic_goal_loss(out["g_str"], bearing_t, dist_t, valid_t)` against a
**LAN** target, not against `g_str.token`. `refc_v3_train.py` never mentions
`HEADS["str_goal"]` or `HEADS["str_action"]`.

Their only training path in the repo is `train_v6_staged.py`'s `s2_goal_loss`
(`:4712`), sized from `stack.vocab_str.tokens` / `stack.vocab_a_str.tokens` and
pinned equal to the v7 `HEADS` tuples at `:2766`. ⚠️ It is behind
`if w.w_s2_goal:` with **default 0.0** — a *guarded* term, so at the default the
strategic heads' `p.grad` is `None`, which is the very shape
`tac_goal_head.py`'s docstring warns is indistinguishable from an unwired head.

=> **every one of the 4,572 train records carries BOTH a strategic goal and a
strategic action token — 9,144 annotations** (`g_str`: `FOLLOW_ROUTE` 2,948 +
`STOP_AT` 454 + `TURN_LEFT` 574 + `TURN_RIGHT` 596 = 4,572; `a_str`:
`HOLD_MAIN_ROAD` 2,390 + `PREPARE_STOP` 454 + `PREPARE_TURN_L` 574 +
`PREPARE_TURN_R` 596 + `RESUME_CRUISE` 558 = 4,572) — **and not one of them is
trained by the arm that is running.**

### 3.4 ⛔ The v6 line lands the factored tactical ids and applies no loss

`train_v6_staged.py:2410-2415`, the trainer's own words: *"No loss term reads
them yet — landing the tensors is this change; a tactical CE on `out["a_lat"]` /
`out["a_lon"]` is a pre-registered follow-up, not a silent addition here."*
MEASURED: `cross_entropy(out["a_lat"]` is absent (control
`V72_TACTICAL_BATCH_KEYS` present). So the tactical action tokens are trained
**only** by refc.

### 3.5 ⛔ 8 DEAD LOGITS — a live class the corpus never populates

`ABORT_LC`, `YIELD_MERGE`, `EXIT_LEFT_FOLLOW_ROUTE`, `EXIT_RIGHT_FOLLOW_ROUTE`,
`LANE_CHANGE_L_FOLLOW_ROUTE`, `LANE_CHANGE_R_FOLLOW_ROUTE`,
`PREPARE_EXIT_FOLLOW_ROUTE`, `PREPARE_LANE_CHANGE_FOLLOW_ROUTE` — `n = 0` on both
splits, and each still occupies a softmax slot.

⭐ **This set is EXACTLY `vocab_v7.NOT_YET_EXTRACTABLE`, member for member** — a
clean confirmation, derived from the blob without consulting the declaration.
`v7_labels.effective_mask` exists precisely to mask them and is the right
mitigation; the census records them so a future blob that starts emitting one is
visible.

### 3.6 ✅ No off-vocabulary emission in v7.2 — the v7 divergence did NOT survive

`refc_strategic.py`'s `OffVocabularyToken` docstring records
`REDUCE_TO_FOLLOW_ROUTE` on **90/801 records (11.24 %)** of the banked **v7**
sample. MEASURED on **v7.2**: every string in `a_str.token`, `g_str.token`,
`a_tac.lat`, `a_tac.lon`, `g_tac.goals.*` and `nav_command.token` across all
**4,719** records is in the frozen vocabulary — **0 off-vocabulary emissions**.
Control: the `a_str` counter is non-empty (5 distinct tokens, 4,572 records).
=> that refusal is a live guard with nothing to refuse on this blob; the
docstring's scope caveat (*"that measurement is on the v7 sample"*) is correct
and should now record the v7.2 result.

### 3.7 ✅ `TACTICAL_GOAL_UNDERPOWERED` agrees with the corpus, both ways

Derived independently (n_train against `GOAL_MIN_N_FOR_METRIC = 200`):
`declared_but_not_derived = []`, `derived_but_not_declared = []`. The 10 declared
members are exactly the 10 tactical goal tokens under the floor. ⭐ Worth stating
because this is the set whose *previous* five-member version was wrong in **both**
directions.

### 3.8 ⚠️ `TACTICAL_GOAL_NEEDS_PERCEPTION` disagrees with the blob on 4 tokens

`cot_backed_but_NOT_declared = ['EVADE_IN_CORRIDOR', 'LANE_CHANGE_L',
'LANE_CHANGE_R', 'YIELD']` — CoT-sourced in the data, absent from the frozen
perception set. This reproduces the divergence `v7_labels.tactical_goal_targets`
already documents and derives its negative policy around; nothing new is broken,
and the declaration is still the one that is wrong.
`declared_but_geometry_only_in_blob = []`.

---

## 4. The leak question — answered precisely

**Is any `INFERENCE_INPUT` carrying privileged state? YES — all three of them,
and it is DECLARED, GATED and STAMPED rather than hidden.**

MEASURED on the v7.2 blobs, 4,719/4,719 records: `nav_command.provenance ==
"ego-future"` — i.e. the nav command is computed from **the ego's own future
path**. The `oracle: true` boolean is present on only **4,190** of them (529
missing, 11.2 %), exactly as `v7_labels.is_oracle_nav` documents; a guard keyed
on the boolean would pass 11 % of the corpus as non-oracle.

* ⛔ **This is privileged state at inference.** The only supplier of a route on
  PhysicalAI is the ego's own future, so an arm fed this nav is an **ORACLE arm**
  and may never be compared against a vision-only arm without the label.
* ✅ **It is not an undisclosed leak.** The only code path to the field is
  `v7_labels.oracle_nav`, which checks the **manifest**, and the live run's
  `config.json` carries `v7_labels.allow_oracle_nav: true` and
  `nav_cmd_derivation: "v7.2 nav_command token (oracle, provenance ego-future;
  allow_oracle_nav=True)"`. The stamp is visible in the artifact.
* ✅ **The classifier back door is shut.** No `INFERENCE_INPUT` token is derived
  from the situation classifier: its output is not in the graph, is not a batch
  field and is not a label source
  (`goal_provenance.contains_situation_classifier_output: false`,
  `provenance_roles.situation_output: []`). Asking the admissibility question of
  the nav token — *could this have been computed from the situation classifier's
  output?* — the answer is **no**; it comes from the ego future, which is a
  **different** inadmissibility and is the one already stamped.
* ✅ **The nav command is correctly NOT a training signal.** Classing it as one
  would rebuild the flagship-v1 route echo (a bijection of its own input that
  scored 1.0000). The census pins `nav_input.kind == "model_input"` and forbids
  any loss surface whose id contains `nav`.

⚠️ **One structural oddity, low severity, worth knowing:** `NAV_COMMAND_TOKENS`
has **3** members and `refc.NAV_COMMANDS = ("follow", "left", "right",
"straight")` has **4**. Under `--nav-from-v7` the fourth row (`straight`) can
never be selected, so the core's one-hot and `NavConditioner`'s embedding each
carry one row no v7 token reaches. `assert_nav_token_alignment` pins positions
0-2 and is correct; the 4th slot is simply unreachable, not misaligned.

---

## 5. What the LIVE refcv5-v2 arm consumes

Derived from **its own recorded argv**
(`Research/2026-09-07-refcv5-v2-compose/raw/refcv5_v2_launch_config_20260906.json`,
step ~17,450 of 40,284 at the time of writing), not from what the trainer could
do.

| surface | enabled by the live argv? |
|---|---|
| `tac_lat_ce` | **yes** (`--v7-labels`) |
| `tac_lon_ce` | **yes** (`--v7-labels`) |
| `nav_input` | **yes** (`--nav-from-v7`) |
| `goal_audit` | yes (loader-side; audit only) |
| `tac_goal_bce` | flag `--tac-goal-tok-head` **is** passed and the head **is** built — but the surface has **no trainer consumer**, so nothing trains |
| `str_goal_ce_v6`, `str_action_ce_v6`, `tac_action_ce_v6` | **no** — a different trainer (`train_v6_staged.py`); no refc argv can enable them |

**15 tokens the live arm does not consume at all:**

| token | n (train+eval) | why | flag it for the next arm? |
|---|---:|---|---|
| `FOLLOW_ROUTE` | 3,041 | no strategic token head in refc | **YES** |
| `HOLD_MAIN_ROAD` | 2,468 | " | **YES** |
| `TURN_RIGHT_FOLLOW_ROUTE` | 624 | " | **YES** |
| `PREPARE_TURN_R_FOLLOW_ROUTE` | 624 | " | **YES** |
| `TURN_LEFT_FOLLOW_ROUTE` | 585 | " | **YES** |
| `PREPARE_TURN_L_FOLLOW_ROUTE` | 585 | " | **YES** |
| `RESUME_CRUISE_FOLLOW_ROUTE` | 573 | " | **YES** |
| `STOP_AT_FOLLOW_ROUTE` | 469 | " | **YES** |
| `PREPARE_STOP_FOLLOW_ROUTE` | 469 | " | **YES** |
| `EXIT_{LEFT,RIGHT}_FOLLOW_ROUTE`, `LANE_CHANGE_{L,R}_FOLLOW_ROUTE`, `PREPARE_EXIT_FOLLOW_ROUTE`, `PREPARE_LANE_CHANGE_FOLLOW_ROUTE` | 0 each | `NOT_YET_EXTRACTABLE` | no — nothing to train on |

...**plus the 18 tactical goal tokens**, which the argv *does* enable a head for
and which still reach no loss.

**The tally, and it is the one line to remember.** 52 tokens; 8 are never
emitted; **44 are emitted**. Of those 44 the live arm **trains on 14** (7 lateral
actions + 7 longitudinal actions) and **feeds 3 as inputs** (the nav commands) —
so **27 emitted tokens (18 tactical goals + 9 strategic) reach nothing at all in
the running arm**, against 17 that are used.

⛔ **The run is NOT to be restarted for this.** `--max-speed-input` landing 4 h
after launch is a timing fact, not a launch defect, and none of the above is
recoverable mid-run without changing the experiment.

### 5.1 ⛔ `--max-speed-input` is UNRUNNABLE on v7.2 — not merely "not passed"

MEASURED: `speed_max_input` is present on **0 of 4,572** train records and **0 of
147** eval records. `refc_v3_train.py` refuses the flag when *"NOT ONE of this
split's windows receives a `speed_max_input` value"*, naming the v7.2 blob's md5.
=> the E16 max-speed channel needs the **v8 label release**, not a flag flip. Any
"the next arm should carry `--max-speed-input`" recommendation that omits this is
wrong.

⚠️ And when it does run, its provenance is `ego-future` (max of the ego's OWN
realised speed over `[anchor+2 s, +6 s]`) while deployment supplies a posted
limit the driver may not reach — a **train/deploy mismatch**, which is why it
goes through the same oracle gate and stamp as the nav command.

---

## 6. Mutation proof

⛔ **A census that cannot go RED proves nothing.** Full log in
`raw/MUTATION_LOG.md`; summary:

| mutation | leg exercised | effect |
|---|---|---|
| **MUT-C0'** — attach a loss to `tac_goal_tok_head`'s logits | 3 gradient | `grad_none 2/2`, `abs-sum 0` -> `grad_none 0/2`, `abs-sum 1452` |
| **MUT-A** — delete `F.cross_entropy(out["lat_logits_tac"], lat_t)` | 2 consumer | `tac_lat_ce` -> `NO_CONSUMER`; **7 tokens change class** (`LANE_KEEP`/`NUDGE_L`/`NUDGE_R` -> `UNREACHED`). `tac_lon_ce` stays `CONSUMED` (specificity control). |
| **MUT-B** — `HEADS["tac_lat"] = TACTICAL_LON_ACTIONS_V7` | 1 projection | **14 tokens change class**; `TRAINING_SIGNAL` 23 -> 9 |
| **MUT-C1** — wire the consumer, keep the real (absent) gradient | 3 in isolation | tactical goal tokens **stay** `AUDIT_OR_METRIC_ONLY` |
| **MUT-C2** — wire the consumer **and** supply the reaching gradient | all three | all 18 flip to `TRAINING_SIGNAL` (23 -> 41, AUDIT 18 -> 0) |

⭐ MUT-C2 matters as much as the RED ones: it proves the census will **detect the
fix** rather than reporting the gap open forever.

`stack/tests/test_v7_vocab_reach_census.py` re-runs MUT-A and the fix-forward
mutation on `tmp_path` copies at every `pytest`, plus
`test_a_failed_read_is_INCONCLUSIVE_and_never_an_absence`: break only the
same-breath control and the surface must report `INCONCLUSIVE`, never
`NO_CONSUMER`. ⛔ On this mount a search tool returns *"no matches"* for a file it
could not open; without that control a missing marker and an unreadable file are
the same observation.

---

## 7. What the next arm should carry that this one does not

> **Wire `tac_goal_loss` + `TacGoalEmitter` into `refc_v3_train.py`.** Everything
> else already exists: the head is built, the targets project, the negative
> policy is derived from the blob, the `pos_weight` is computed from the split,
> and the loss function is written and tested. It is **the trainer call that is
> missing**, and it is the cheapest lever in the census — it converts 18 tokens
> (4,572 `SPEED_BAND`, 3,629 `FOLLOW_LANE`, 779 traffic-light, 609 `YIELD`, ...)
> from invisible to supervised, and it is the only wiring in this report that
> targets the PI's two standing observations directly.

Ranked follow-ups, cheapest measured effect first:

1. **`tac_goal` loss wiring** (above) — 18 tokens, 0 new modules, 11,286
   already-allocated parameters.
2. ⭐ **A gradient assertion over EVERY built head — DONE IN THIS TURN, as a
   test.** `stack/tests/test_built_heads_receive_gradient.py` (**5 tests,
   passing**) builds the arm the live argv describes, runs the trainer's own
   loss and one backward, and **fails if any built module's every parameter took
   `p.grad is None`** — except a literal `KNOWN_UNWIRED` allow-list that names
   `tac_goal_tok_head` with its reason and **must shrink**. It is two-sided: the
   day the seam is wired the "must shrink" assertion fires and names every doc to
   update. Mutation-controlled both ways — a freshly added unwired head is
   detected and named, and a module with a real ALL-ZERO gradient is NOT
   misclassified as unwired (that discriminator is why the guard survives step 0,
   where several zero-init gates legitimately read zero).
   ⛔ **This is a test, not a trainer edit** — `refc_v3_train.py` is rebuilt by
   the live supervisor on every relaunch, so it is deliberately untouched.
   ⚠️ The trainer-side refusal (a `_check_goal_point_args`-shaped two-liner)
   remains for the file's owner.

   ⭐ **THE PRECEDENT IS ALREADY IN THIS TRAINER, ONE FLAG OVER.**
   `_check_goal_point_args` refuses `--goal-point-inject` with
   `--goal-point-w <= 0` in exactly these words: *"a head that is built,
   stamped, and **supervised by nothing** … a failure that is not about the
   lever is worse than no run"*. `--tac-goal-tok-head` is that same
   configuration and has **no such refusal**, because it has no weight to
   check — which is why the guard must key on the **gradient**, not on a
   weight. The cheapest form is a two-line refusal beside the existing one,
   plus the preflight sweep for the general case.
3. **A strategic token head for refc** (or an explicit PI decision that refc's
   strategic layer stays geometric) — 9 emitted tokens, 9,144 train annotations over all 4,572 records.
4. **`--max-speed-input` — only with the v8 blob.** Zero value on v7.2, and the
   trainer will refuse it.
5. **A tactical CE in `train_v6_staged.py`** — the trainer's own pre-registered
   follow-up; only relevant if the v6 line is revived.

---

## 8. Deliverable manifest

| artifact | where it lives | in only one place? |
|---|---|---|
| `stack/scripts/v7_vocab_reach_census.py` — the reusable instrument | `repo:` **staged** (+ dev-box mirror copy) | no |
| `stack/tests/test_v7_vocab_reach_census.py` — 11 tests, 3 mutation controls | `repo:` **staged** (+ dev-box mirror copy) | no |
| `stack/tests/test_built_heads_receive_gradient.py` — 5 tests, the class-level guard | `repo:` **staged** (+ dev-box mirror copy) | no |
| `RESULT.md` (this file) | `repo:` **staged** | no |
| `raw/census.json` — the full machine-readable census | `repo:` **staged** | no |
| `raw/gradreach_live.json` / `raw/gradreach_mutated.json` | `repo:` **staged** | no |
| `raw/census_MUT_A.json` / `_MUT_B` / `_MUT_C1` / `_MUT_C2` | `repo:` **staged** | no |
| `raw/MUTATION_LOG.md` | `repo:` **staged** | no |
| `scripts/gradreach_probe.py` — the per-module gradient probe | `repo:` **staged** | no |

Nothing produced here lives on only one disk.

### 8.0 ⚠️ The trainer MOVED under this census, and every verdict was re-derived

`refc_v3_train.py` was edited by a sibling **during** this session (WP-D: two new
modules `tanitad/refs/refc_bev_aux.py` and `tanitad/data/bev_aux.py`, +12 KB in
the trainer). ⛔ A census computed against bytes that no longer exist is worse
than none, so the gradient probe and the whole census were **re-run against the
post-change worktree**:

* class counts **identical** — `TRAINING_SIGNAL 23 · INFERENCE_INPUT 3 ·
  AUDIT_OR_METRIC_ONLY 18 · UNREACHED 0 · NOT_EMITTED 8`;
* `tac_goal_tok_head` still `grad_none = 2/2`, abs-sum `0` — **NOT WIRED**;
* the current trainer still reads **0 hits** for `tac_goal_loss`,
  `TacGoalEmitter` and `tac_goal_logits`, with its control marker present.

The banked `raw/census.json` and `raw/gradreach_live.json` are the **post-change**
run.

⚠️ **A NOTE FOR THE OWNER, NOT A COMPLAINT.** At the end of this turn
`stack/tanitad/refs/refc_bev_aux.py` and `stack/tanitad/data/bev_aux.py` exist on
disk, are **imported by `refc_v3_train.py`**, and are **in neither HEAD nor the
index** (`git cat-file -e HEAD:<path>` fails; `git ls-files --cached` = 0, with a
tracked-file control = 1). Any checkout without them cannot import the trainer —
`test_tac_goal_trainer_flag.py` and the new gradient guard both die at
`ImportError` before their first assertion. ⛔ Not mine to stage; escalated here
because it is invisible from the trainer's own diff.

### 8.1 Provenance of the inputs

| input | md5 | how verified |
|---|---|---|
| `s2_labels_v7.2_train.jsonl.gz` (4,572 rec) | `0ff902130ce76886b8a925eceed9e3a5` | copied from the G: release and **md5-verified against the source** |
| `s2_labels_v7.2_eval.jsonl.gz` (147 rec) | `aa12c948f062181c3297265b51526ec5` | ⚠️ **INCONCLUSIVE against G:** — see below |
| `refcv5_v2_launch_config_20260906.json` | — | read directly from the repo |

⚠️ **One honest gap.** The **G: copy of the eval blob could not be read at all**
in this session: 7 md5 attempts over ~90 s each returned an empty string while a
**same-breath control** (the train blob, same directory) returned a 32-char hash
**every time**. Metadata resolved throughout (`ls` reports 65,787 bytes). This is
the documented per-file cloud-only non-hydration pattern, not a mount outage.
The eval blob used here is therefore identified by **five byte-identical local
copies** (`dkhead_run`, `navcomp`, `refcv5cmp`, `tanitad-refcv4b-analyze`,
`tanitad-wt/_s2build/release/v72`) all at `aa12c948...` and 65,787 bytes,
matching G:'s reported size. That is strong, but it is **not** a byte-level match
against the release copy, and it is reported as `INCONCLUSIVE` rather than
`VERIFIED`. ⛔ It affects only the `n eval` column; every `n train` figure and
every reach verdict is unaffected.

⭐ **A trap this session paid for, worth recording:** a
`cp "$src" "$dst" && a=$(md5sum "$src") && b=$(md5sum "$dst")` loop
**short-circuits on a failed `cp`** and then compares the **previous
iteration's** `$a`/`$b`, printing a false `VERIFIED`. That is the *"both operands
fail identically and the shell reports a match"* hole from `CLAUDE.md` in a new
costume — the fix is to reset the variables inside the loop **and** assert the
32-character shape before comparing, which is what caught it.

---

## 9. Cross-checks that came out right

Stated deliberately: a census nobody can check is worthless, and these are the
independent agreements that make the rest quotable.

* Traffic-light counts reproduce `CLAUDE.md` **exactly**, derived independently:
  `RED 376`, `GREEN 363`, `YELLOW 22`, colourless `18`, any `779` of 4,572.
* `SPEED_BAND` on **4,572/4,572** train records — the *"ALWAYS PRESENT"* claim in
  `vocab_v7.DEFINITIONS` holds.
* `CORRIDOR_OFFSET` 860 train / 885 total and `YIELD` 609 — matching the counts
  `v7_labels.py`'s own docstring quotes.
* `NOT_EMITTED` == `NOT_YET_EXTRACTABLE`, member for member.
* `TACTICAL_GOAL_UNDERPOWERED` == the derived below-floor set, both directions.
* The nav `oracle` boolean gap (4,190 present / 529 absent = 11.2 %) reproduces
  `is_oracle_nav`'s documented measurement.

---

## 10. Registered claims

| id | claim | evidence |
|---|---|---|
| `D-TACGOAL-TRAINER-SEAM-OPEN` | **CONFIRMED open, now by GRADIENT.** Head built + forward-run; no trainer calls `tac_goal_loss` or `TacGoalEmitter`; `p.grad is None` on both tensors under the live argv. ⭐ NEW: the LIVE refcv5-v2 arm passes `--tac-goal-tok-head`, so it is carrying 11,286 parameters that cannot learn. | `raw/gradreach_live.json`, `raw/census.json` |
| `D-TACGOAL-1` (row 9304) / `test_tactical_label_reach.py` docstring | ⚠️ **SCOPE CORRECTION.** *"now reaches a supervised head"* / *"GAP CLOSED"* are true of the HEAD and false of the TRAINER; a reader takes them as "it is supervised". The neighbouring `…SEAM-OPEN` row already says otherwise, so the register is self-consistent but its headline is not. | this file; `raw/census.json` |
| `D-VOCAB-REACH-1` | 18 of 52 v7 tokens are emitted and reach no loss; 8 are dead logits; 15 strategic tokens have no refc training path | `raw/census.json` |
| `D-VOCAB-REACH-2` | `effective_weights_stamp_v3` cannot see a head with **no weight flag**; `assert_seams_are_built` cannot see a **built head with no gradient**. Both guards are green on `D-TACGOAL-1`. | live `config.json` §`effective_weights` (5 terms), §`param_breakdown` (11,286) |
| — (confirmation) | the nav command is an ORACLE input on 4,719/4,719 records, declared and stamped; no undisclosed leak, and no situation-classifier back door | `raw/census.json`, live `config.json` |
