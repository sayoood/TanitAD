# The Alpamayo meta-action → TanitAD token mapping, with measured coverage — and the
# tokenizer freeze is blocked by ONE decision, not by a missing token

**2026-08-23 · discharging `HIERARCHY_VOCABULARY.md` §4 ("mapping table = PH1 deliverable")
and §7 item 3 ("mapping table + coverage", the open checkbox before the tokenizer freeze).
All figures `MEASURED (ours)`, re-derivable bit-exactly with `code/map_alpamayo.py`, banked
in `raw/`.**

---

## Headline, in the order that matters

1. ⭐ **The vocabulary is COMPLETE ENOUGH TO FREEZE. The corpus demands no new token type.**
   Every unmapped instance is one of three things and none of them is a missing token:
   a structural property of the emitter (`Stop` suppresses the other axes, 6.43 %), a decode
   artefact (`Reverse`, 0.13 %), or a phrase that names an action with no referent (0.27 %).
2. ⛔ **What blocks the freeze is that the LABEL SIDE AND THE MODEL SIDE ALREADY DISAGREE.**
   `tanitad/lake/tac_str_labels.py:538` defaults `vocab_version="v6.1"` and emits
   `TURN_L`/`TURN_R`; `tanitad/models/v6.py:281` defaults `tactical_lat_actions()` to **v6.0**,
   which has neither. A freeze must name **one** version. **This is a PI decision and it is
   the only thing standing between here and the freeze.**
3. ⭐ **The bigger label swing is not the one that has been argued about.** The TURN question
   is 186 clips (**3.93 %**; the cited docs round it to 3.94 %). **800 clips (16.92 %) have a
   chain-of-causation that says NUDGE
   while the `Lane` axis says `Lane Keep`** — and the label pipeline reads LAT from the
   `Lane` axis only. That is **4.3× the TURN question** and it is UNADJUDICATED.
4. ⭐ **The prior screening's "the longitudinal axis does not map at all" is RE-VERIFIED AND
   CONFIRMED — and now quantified and repaired.** The axis alone uniquely determines a LON
   token for **6.43 %** of instances (only `Stop` → `HOLD`). The **pair** (axis value, CoT
   intent) determines one for **99.87 %**. The reason was in the record all along; it is the
   chain-of-causation.
5. ⛔ **The one intent whose numeric argument the phrase actually determines is the one
   argument we may not use.** 41 phrases read a speed-limit sign's TEXT. `v6.py:188-200`
   records that sign KIND and TEXT are **FORBIDDEN** (C87) and the G1 gate is **CLOSED at
   0/31**. Taking `v_hi` from Alpamayo is the forbidden channel laundered through a teacher —
   and the same cell records that the sign channel's top false positive is a **dashboard `30`
   roundel**, i.e. the ego speedometer, which makes it an ego echo as well.

**Coverage, the number the freeze decision needs:**

| channel | instance-weighted | distinct-phrase |
|---|---:|---:|
| **`a_tac` LON** (axis × CoT intent) | **99.87 %** (4 723 / 4 729) | 6 / 7 axis values |
| **`a_tac` LAT** (from the `Lane` axis) | **93.57 %** (4 425 / 4 729) | 7 / 8 axis values |
| **factored LAT × LON pair complete** | **93.44 %** (4 419 / 4 729) | — |
| **`g_tac`/`g_str` goals, grounded referent** | **84.39 %** (3 991 / 4 729) | **94.16 %** (1 000 / 1 062) |
| …including the `FOLLOW_MAIN_ROAD` default token | **99.73 %** | **98.87 %** |
| **no goal token determinable** | **0.27 %** (13) | 12 |
| **nothing matched at all** | **0.00 %** (0) | 0 |

---

## 1. The corpus — located, hash-verified, and its parity status

| | |
|---|---|
| **artifact** | `records.parquet`, HF dataset **`Sayood/tanitad-alpamayo2-augmentation`** (public, gated=auto) |
| **sha256** | `ecae276db9969de115abb3caa1e87d97eae0535544be8f5edcc33ec45d925ed2` — **identical** to the digest banked 2026-08-06 in `…/2026-08-06-alpamayo-augmentation/a2_records_stats.json`, so the file this report reads is byte-identical to the one that report measured |
| **shape** | 23 644 rows = **4 729 unique clips × 5 tasks**; the **4 729 `task == "meta_action"` rows** are the corpus here |
| **model** | `nvidia/Alpamayo2-Super`, `NF4-backbone-4bit-UNVALIDATED` |
| **sampling** | `do_sample=True`, temperature **0.6**, seed 42, **ONE draw per clip** |

⚠️ **`HIERARCHY_VOCABULARY.md` §1 says "4,800-clip". The banked corpus is 4 729.** Use 4 729.

⚠️ **One draw at temperature 0.6 is not the model's mode, and cross-draw stability is
UNMEASURED.** Every proportion in this report is bounded above by that. This is a
`MEASURED` fact about **one sample of** Alpamayo, not about Alpamayo.

⛔ **Alpamayo is a TEACHER, not ground truth.** PhysicalAI-AV is listed as Alpamayo training
data, so these labels may be partly memorised. That makes the corpus sound for the question
asked here — *what can our vocabulary express* — and unsound for accuracy claims.

### 1.1 Parity status, carried in as instructed

Per `…/2026-08-18-alpamayo-parity-exclusion/ALPAMAYO_PARITY_EXCLUSION.md` (`MEASURED`):

- **201 of the 4 729 records are in the parity TRAIN corpus** `physicalai-train-e438721ae894`.
- Only **257** have w120 video built, so the eval split buildable today is 257 clips of which
  **201 = 78.21 % are train** — REF-A-I-JEPA scale, not a small-scale version of it.
- **6 of the 40 canonical val episodes (15.0 %)** sit inside the Alpamayo record set.
- Blast radius on published numbers: **ZERO**. No model-eval number has ever been computed here.

⇒ **Consequence for this deliverable, stated explicitly: a VOCABULARY-DESIGN measurement is
not an eval.** Nothing here scores a model, so the contamination does not invalidate a single
coverage number below. It **does** forbid two things: using Alpamayo meta-actions as
**held-out** tactical labels, and running the remaining **4 472-clip build** without calling
`parity.filter_train_clips()` first.

### 1.2 Where I looked (so "it does not exist" is never asserted from one probe)

`alpamayo` case-insensitive across the whole worktree (filenames + contents); the HF account
`Sayood` via the tree API; `stack/scripts/`; `TanitAD Research Lab/Data Engineering/`;
`Benchmarks & Eval/Research/`. **Six packages carry Alpamayo data**, and the in-repo ones are
a decoy for this question: `…/2026-08-05-alpamayo2-super/comparison/alpamayo_meta_action.jsonl.xz`
holds **39 clips**, not 4 729. The full distribution exists **only** in the HF parquet, whose
local copy lived in an expired session scratchpad. It was re-pulled read-only (26 MB, no
metered compute).

---

## 2. Two channels, and why they are mapped separately

Per clip, in **one decode**, Alpamayo emits both:

```
Slow down due to the lead vehicle braking for the stop sign      <- CHANNEL 2 (chain-of-causation)
Longitudinal: Gentle Deceleration.                                <- CHANNEL 1 (3-axis declaration)
Lateral:      Go Straight.
Lane:         Lane Keep.
```

Channel 1 grounds **`a_tac`** (the factored LAT × LON *actions*). Channel 2 grounds
**`g_tac`/`g_str`** (the *goals*). They are mapped separately because **they disagree**:

| | |
|---|---:|
| rows where both assert a LAT class | 1 642 |
| agreement | **0.4239** |
| **Cohen's κ** | **0.0109** — indistinguishable from chance |
| dominant cell | `cot=NUDGE \| axis=LANE_KEEP` = **800** |

⚠️ **One decode produced both, so this is not inter-annotator disagreement — it is internal
inconsistency of the generation.** Any supervised use must declare which channel it reads,
and a mapping that pooled them could not be audited.

---

## 3. Table A — the 3-axis meta-action → `a_tac`

### 3.1 The axis correspondence is not one-to-one, and the obvious pairing is wrong

| Alpamayo axis | maps to | why |
|---|---|---|
| **`Lane`** | our **LAT** | lane-**topological** — the same *kind* of quantity |
| **`Longitudinal`** | our **LON**, but only **jointly with the CoT** (§3.4) | a sign+severity **band**, not a **target** |
| **`Lateral`** | ⛔ **not an `a_tac` axis at all** | steering **magnitude** = the operative layer's continuous κ, which §4 leaves untokenised on purpose |

⭐ **Mapping our LAT from Alpamayo's `Lateral` axis would be a category error, and it is
measurable.** Row-normalised `Lateral` within each `Lane` value:

| `Lane` \ `Lateral` | Go Straight | Steer L | Steer R | Sharp L | Sharp R | n |
|---|---:|---:|---:|---:|---:|---:|
| Lane Keep | 59.1 % | 15.2 % | 23.8 % | 0.8 % | 1.0 % | 4 035 |
| **Turn Left** | 0.0 % | 18.8 % | 0.0 % | **81.2 %** | 0.0 % | 85 |
| **Turn Right** | 0.0 % | 0.0 % | 19.8 % | 0.0 % | **80.2 %** | 101 |
| **Left Lane Change** | **36.4 %** | 40.9 % | 22.7 % | 0.0 % | 0.0 % | 22 |
| **Right Lane Change** | **65.9 %** | 1.2 % | 23.2 % | 1.2 % | 8.5 % | 82 |
| Slightly Shift Left | 62.3 % | 7.2 % | 18.8 % | 7.2 % | 2.9 % | 69 |
| Slightly Shift Right | 48.4 % | 9.7 % | 9.7 % | 19.4 % | 9.7 % | 31 |

**A junction turn is 80–81 % `Sharp Steer *`; a lane change is ~60 % `Go Straight`. On the
`Lateral` axis a LANE CHANGE IS INDISTINGUISHABLE FROM A LANE KEEP.** `Lateral` measures
curvature magnitude; `Lane` measures lane topology.

⭐ **This retro-explains a banked number.** `…/2026-08-05-alpamayo2-super/tools/a2_parse_meta_action.py`
projected our direction classes onto the **`Lateral`** axis and scored agreement **0.4706 /
κ 0.1488** against the driven path (n=34). That low agreement was a **category error in the
projection**, not a failure of the model or of our label set. The `LAT2DIR` projection should
not be re-used.

### 3.2 LON — Alpamayo `Longitudinal` → our LON token

| Alpamayo value | n | % | our LON | args the phrase does **NOT** determine | conf |
|---|---:|---:|---|---|---|
| Gentle Deceleration | 1 594 | 33.71 | `BRAKE_TO` | `v`, `within_m` | medium |
| Maintain Speed | 1 225 | 25.90 | `CRUISE` | `v` | high |
| Gentle Acceleration | 1 151 | 24.34 | `CRUISE` | `v` | medium |
| Stop | 304 | 6.43 | `HOLD` | `within_m` | high |
| Strong Deceleration | 267 | 5.65 | `BRAKE_TO` | `v`, `within_m` | medium |
| Strong Acceleration | 182 | 3.85 | `CRUISE` | `v` | medium |
| **Reverse** | **6** | **0.13** | ⛔ **UNMAPPED** — §6.1 | — | — |

⚠️ **`gentle` vs `strong` has no slot anywhere in our LON vocabulary and is LOST.** So is the
*target*: Alpamayo's axis is rate-typed, ours is target-typed. The loss is the mirror image of
the `LAT2DIR` loss the 2026-08-05 tool found in the other direction.

### 3.3 LAT — Alpamayo `Lane` → our LAT token (**version-parameterised**)

| Alpamayo value | n | % | LAT under **v6.0** | LAT under **v6.1** | args not determined | conf |
|---|---:|---:|---|---|---|---|
| Lane Keep | 4 035 | 85.32 | `LANE_KEEP` | `LANE_KEEP` | — | high |
| Turn Right | 101 | 2.14 | `LANE_KEEP` + `g_str:TURN_RIGHT` | **`TURN_R`** | — | med / high |
| Turn Left | 85 | 1.80 | `LANE_KEEP` + `g_str:TURN_LEFT` | **`TURN_L`** | — | med / high |
| Right Lane Change | 82 | 1.73 | `LANE_CHANGE_R` | `LANE_CHANGE_R` | `within_m` | high |
| Slightly Shift Left | 69 | 1.46 | `NUDGE_L` | `NUDGE_L` | `lat_m` | high |
| Slightly Shift Right | 31 | 0.66 | `NUDGE_R` | `NUDGE_R` | `lat_m` | high |
| Left Lane Change | 22 | 0.47 | `LANE_CHANGE_L` | `LANE_CHANGE_L` | `within_m` | high |
| **(axis absent)** | **304** | **6.43** | ⛔ **UNMAPPED** — §6.2 | ⛔ same | — | — |

⭐ **Instance coverage is 93.57 % under BOTH versions.** The two tables differ on 186 rows
(3.93 %) and on **nothing else**. The choice is not about coverage; it is about whether a
junction traversal is representable *as itself* or only as `LANE_KEEP` plus a strategic goal.
See §7.2 — it is the freeze blocker.

### 3.4 ⭐ The joint LON resolution — the prior finding, confirmed and repaired

`…/2026-08-18-alpamayo-screening/ALPAMAYO_SCREENING_AND_VOCAB_REVIEW.md` §2.1 concluded the
longitudinal axis **"does not map at all"** because our LON set is **reason-typed**
(`FOLLOW`/`YIELD_MERGE`/`BRAKE_TO`) and Alpamayo's is **magnitude-typed** — *"Gentle
Deceleration" cannot say whether the ego is FOLLOWing a lead or BRAKE_TO-ing a stop line.*

**That is correct, and re-verified here:**

| | instances | % |
|---|---:|---:|
| the **axis alone** uniquely determines a LON token | 304 | **6.43 %** (only `Stop` → `HOLD`) |
| the **pair** (axis value, CoT intent) determines one | 4 723 | **99.87 %** |

Precedence: `Stop` → `HOLD`; else a reason-bearing intent → `FOLLOW`/`YIELD_MERGE`; else the
axis sign → `BRAKE_TO` (decel) / `CRUISE`. Resulting distribution: `CRUISE` 1 770 ·
`BRAKE_TO` 1 267 · `FOLLOW` 898 · `YIELD_MERGE` 484 · `HOLD` 304 · unresolved 6.

⭐ **The production label pipeline already does exactly this, independently.**
`tac_str_labels.py:129-253`: `lon_is_admissible()` uses the magnitude axis as a **prior** and
`lon_from_alpamayo()` takes the action from the CoT's referents, abstaining on ambiguity.
This report supplies the number that pipeline was built without: **6.43 % → 99.87 %**.

---

## 4. Table B — chain-of-causation → `g_tac` / `g_str` goals

1 062 distinct normalised phrases over 4 729 instances. Rules are **ordered, first match
wins**, so these are first-match counts, not concept-occurrence counts. Full rule text with
per-row rationale: `code/alpamayo_rules.py`.

| intent | n | % | goal token(s) | args the phrase **does** determine | conf |
|---|---:|---:|---|---|---|
| FOLLOW_LEAD | 959 | 20.28 | `g_tac:GAP_TARGET(agent_slot_id, time_gap_s)` | none | high |
| **CORRIDOR_DEFAULT** | **725** | **15.33** | `g_str:FOLLOW_MAIN_ROAD` · `KEEP_CORRIDOR` | none | high |
| TRAFFIC_LIGHT | 645 | 13.64 | `g_tac:TRAFFIC_LIGHT_REACT(slot, state, arc)` | **`state`** | high |
| EVADE_DYNAMIC | 626 | 13.24 | `g_tac:EVADE_IN_CORRIDOR(lat_m, slot, arc)` | none | high |
| SPEED_FOR_CURVE | 410 | 8.67 | `g_tac:SPEED_BAND` · `g_str:KEEP_CORRIDOR` | none | high |
| SPEED_FOR_FEATURE | 316 | 6.68 | `g_tac:SPEED_BAND(v_lo, v_hi)` | none | high |
| YIELD_VRU | 261 | 5.52 | `g_tac:YIELD_AT(arc, gap_slot)` | none | high |
| EVADE_STATIC_FURNITURE | 148 | 3.13 | `g_tac:EVADE_IN_CORRIDOR` | none | **medium** — §6.4 |
| YIELD_TRAFFIC | 115 | 2.43 | `g_tac:YIELD_AT` | none | high |
| YIELD_SIGN | 109 | 2.30 | `g_tac:YIELD_AT` | none | high |
| LANE_CHANGE | 67 | 1.42 | `a_tac LAT:LANE_CHANGE_L/R` · `a_str:PREPARE_LANE_CHANGE` | `dir` | high |
| JUNCTION_TURN | 56 | 1.18 | `g_str:TURN_LEFT/TURN_RIGHT/STRAIGHT_THROUGH` | direction | high |
| GAP_ACCEPT | 49 | 1.04 | `g_tac:GAP_TARGET` · `a_str:PREPARE_LANE_CHANGE` | `dir` | high |
| EVADE_OTHER | 42 | 0.89 | `g_tac:EVADE_IN_CORRIDOR` | none | medium |
| **SPEED_LIMIT** | **41** | **0.87** | `g_tac:SPEED_BAND(v_lo, v_hi)` | **`v_hi` — ⛔ INADMISSIBLE, §6.5** | high |
| RESUME_AFTER_CLEAR | 40 | 0.85 | `a_str:RESUME_CRUISE(v_target)` · `g_tac:SPEED_BAND` | none | high |
| STOP_SIGN | 40 | 0.85 | `g_tac:STOP_POINT(arc, reason=sign)` · `g_str:STOP_AT` | `reason` | high |
| FORK_SPLIT | 33 | 0.70 | `g_str:EXIT_RIGHT/EXIT_LEFT` · `a_str:PREPARE_EXIT` · `LAT:LANE_KEEP` | `dir` | medium |
| ROUNDABOUT | 17 | 0.36 | `g_tac:YIELD_AT` · `g_str:STRAIGHT_THROUGH/TURN_*` | none | medium |
| **ACTION_ONLY_NO_REFERENT** | **13** | **0.27** | ⛔ **no goal determinable** — §6.3 | — | — |
| OVERTAKE | 8 | 0.17 | `a_tac LAT:LANE_CHANGE_L/R` · `g_tac:SPEED_BAND` | `dir` | medium |
| INTERSECTION_APPROACH | 8 | 0.17 | `g_str:STRAIGHT_THROUGH` · `g_tac:SPEED_BAND` | none | medium |
| QUEUE | 1 | 0.02 | `g_tac:STOP_POINT(reason=queue)` · `GAP_TARGET` | `reason` | high |
| **WAIT_ONCOMING** | **0** | **0.00** | `g_tac:WAIT_FOR_ONCOMING` | — | §6.6 |

⚠️ **`CORRIDOR_DEFAULT` is reported as its own tier and is NEVER folded into the headline.**
It is a real token and a real mapping — the PI's stated no-route default — but it is the
**null hypothesis**, and a coverage number that absorbs 15.33 % of the corpus into
"FOLLOW_MAIN_ROAD" would be measuring the vocabulary's ability to say nothing.

### 4.1 Two honesty numbers on the rules themselves

| | value | reading |
|---|---:|---|
| `specific` hits that **also** match the default pattern | 340 (**8.52 %** of specific) | ordering sensitivity — the headline is stable to within this band |
| phrases matching **>1** specific rule | **38.21 %** | real, not a defect: *"slow down for the roundabout because of the yield sign"* genuinely is both. First-match assigns one; **a production tokenizer should emit a SET, not a class.** |

⚠️ **A rule-boundary defect was found and fixed by content audit, not by exit code.**
`\bcross\b` never matches "crossing" and `\bpedestrian\b` never matches "pedestrians"; both
were present in the first draft and silently pushed **~145 real yield instances** into the
default bucket. Every rule was then sampled and eyeballed (`code/` + the audit in the run log).

---

## 5. Which axis the corpus actually supplies goals FOR

`v6.py` factors the **goal** vocabulary too (`TACTICAL_GOAL_TOKENS_LAT/LON`, 2026-08-16).

| | instances | % |
|---|---:|---:|
| **LON goal only** | 3 019 | **63.84 %** |
| no `g_tac` token (strategic or action-only) | 894 | 18.90 % |
| **LAT goal only** | 816 | **17.26 %** |
| both | 0 | 0.00 % |

⭐ **The LAT goal axis is nearly unsupplied, and the part that is supplied is one token.**
Alpamayo grounds `EVADE_IN_CORRIDOR` and nothing else on LAT. **`ANCHOR_GOAL` — the
"+4.7 PDMS" geometric-goal lever — and `CORRIDOR_OFFSET` have NO Alpamayo source in either
channel.** They are geometry-derived by construction, so this is expected rather than
alarming, but it settles a planning question: **Alpamayo cannot supervise the LAT goal head
at all.** That head's labels must come from Engine A geometry.

---

## 6. ⛔ UNMAPPED — logged in full, never dropped

### 6.1 `Reverse` / `Reverse Left` / `Reverse Right` — 8 instances, adjudicated ARTEFACT

6 rows carry `Longitudinal: Reverse` (0.13 %) and 2 carry `Lateral: Reverse *` (0.04 %).

**They are decode artefacts, not a vocabulary gap.** All 6 `Reverse` rows carry a
**forward-driving CoT** — *"Keep lane to continue driving since the path ahead is clear"*,
*"Slow down for the right curve ahead"* — and the 2 `Reverse *` lateral rows fall on the
**same 2 clips** as a `Longitudinal: Reverse` (verified: subset relation holds). At
temperature 0.6 a 0.13 % tail is consistent with sampling noise.

⇒ **Do NOT add a REVERSE token on this evidence.** Counted here so the number is visible
rather than assumed away. *(Concurs with the 2026-08-18 screening §3.5.)*

### 6.2 ⭐ The `Stop` absorption — 304 instances (6.43 %), and it CAPS the factored coverage

**`Longitudinal: Stop` terminates the declaration.** The `Lateral` and `Lane` axes are not
emitted at all; the generation ends `Stop.<|traj_future_start|>`. The identity is **exact**:

| | |
|---|---:|
| `Stop` rows | 304 |
| of which `Lateral` **and** `Lane` both absent | **304** |
| non-`Stop` rows with any axis absent | **0** |

This is a **structural property of the emitter**, not a parse failure and not a coverage
defect in our vocabulary. But it has a hard consequence: **a factored LAT × LON head needs a
LAT token on every window and the corpus supplies none on 6.43 % of them.** LAT must either
abstain or be imputed from hindsight geometry. **This single fact caps `FACTORED_PAIR` at
93.44 % no matter how good the mapping is** — no token addition can move it.

### 6.3 13 instances (0.27 %), 12 distinct — an action with no referent

Complete list (`n · phrase`):

```
2  maintain speed due to an oncoming vehicle
1  accelerate due to the traffic controller signaling to proceed
1  maintain speed due to pedestrians staying on the sidewalk
1  resume speed since the pedestrians remain on the sidewalk
1  maintain speed due to an oncoming motorcycle
1  accelerate to match the traffic flow because the light ahead is green and the lead
   vehicle begins moving while adjacent lanes stay blocked
1  slow down due to a stopped vehicle ahead
1  resume speed because the pedestrians remain off the roadway
1  adapt speed for limited visibility in dense fog
1  slow down because the gap to the leading car is closing
1  adjust speed and keep safe distance due to the vehicle ahead
1  adapt speed for limited visibility due to dense fog
```

Each determines an `a_tac` **LON action** but names no goal referent. Three sub-classes worth
naming, all *n ≤ 2*: **a human traffic director** (1), **visibility/weather** (2), and
**negative evidence** — *"pedestrians remain **on the sidewalk**"*, i.e. a hazard explicitly
**not** taken (3). The last is interesting and has no token; at n=3 it is not a proposal.

**Zero phrases matched nothing at all.**

### 6.4 A GROUNDING gap, not a vocabulary gap — 148 instances (3.13 %)

`EVADE_STATIC_FURNITURE` — *"nudge left to increase clearance to the snowbank / guardrail /
bollards / roadside sign / vegetation / traffic island on the right"*. **The token fits
exactly.** Its **argument** does not: `EVADE_IN_CORRIDOR(lat_offset_m, obstacle_slot, past_arc_m)`
needs an `obstacle_slot`, and `obstacle.offline`'s enum over 87 481 cuboids is **10 classes,
all dynamic agents** (CLAUDE.md). **Static roadside furniture has nothing to point at.**

⇒ This is an argument-supplier gap. **Engine C (SAM) is the named candidate**
(`HIERARCHY_VOCABULARY.md` §0b: *"lane/road-surface geometry PhysicalAI's labels never had"*).
It should be counted as evidence **for** Engine C entering PH1, with a measured demand of
3.13 % of instances.

### 6.5 ⛔ An ADMISSIBILITY hazard that looks like a coverage win — 41 instances (0.87 %)

`SPEED_LIMIT` is the **only** intent whose numeric argument the phrase literally contains:
*"a 120 speed limit sign is posted ahead"*, *"a 35 mph speed limit sign sets the target speed"*.

**It is inadmissible for exactly that reason.** `stack/tanitad/models/v6.py:188-200` records
the F-14 blocker: sign **KIND and TEXT are FORBIDDEN** (`RETRACTION_LOG` C87) and the G1
sign-text gate is **CLOSED at 0/31**. A `v_hi` taken from an Alpamayo phrase is the same
forbidden fact arriving through a teacher model.

⚠️ **And it is worse than a rule violation.** The same cell records that the sign channel's
two highest-scoring **false** positives are a **dashboard `30` roundel (0.927)** and a
hoarding (0.778) — both scoring *above* true signs. **A dashboard roundel is the ego
speedometer.** A VLM reading the same pixels inherits the same failure, so a sign-derived
target speed can be an **ego echo arriving through the vision channel** — which a vision-only
admissibility audit does not watch (same family as the flagship route head's 1.0000 echo).

⇒ The 41 instances count toward coverage **as `SPEED_BAND` goals** (that token is groundable
from other evidence). **Their `v_hi` must be discarded.**

### 6.6 Tokens WE have that the corpus never exercises

| our token | Alpamayo source | reading |
|---|---|---|
| `a_tac LAT: ABORT_LC` | **none, either channel** | a single-frame declaration cannot express an abort, which is defined by a *change* of a prior decision. Not evidence to remove it — evidence that Alpamayo cannot supervise it. |
| `a_tac LON: CREEP` | **none, either channel** | produced by no (axis, intent) pair. |
| `g_tac: WAIT_FOR_ONCOMING` | **0 instances** | 142 phrases (33 distinct) mention oncoming traffic and **not one contains `wait`/`hold`/`until`/`remain stopped`** — every one is an evasion (`nudge right due to an oncoming vehicle`), an observation (`keep lane because an oncoming vehicle is approaching in the opposite lane`) or a release (`resume speed because the oncoming vehicle has passed`). **Narrow-road negotiation is absent from this corpus.** |
| `g_tac: ANCHOR_GOAL`, `CORRIDOR_OFFSET` | **none** | geometry-derived by construction (§5). |
| `g_str: LANE_TARGET` | n/a | ⛔ **NOT EMITTED** (derivation refuted 2026-08-16). Correctly absent as a mapping target here. |
| `g_str: ROUTE_TO` | phrase-adjacent only | 33 `FORK_SPLIT` phrases cite **overhead signage**, which our vocabulary can only read through the **G1-gated OCR path** — closed (§6.5). |

---

## 7. What the measurement says about the vocabulary — proposals only; the PI decides

### 7.1 No new token TYPE is proposed

I looked for one and the corpus does not demand one. Every residue is a grounding gap (§6.4),
an admissibility problem (§6.5), an emitter property (§6.2), or noise (§6.1). The nine `g_tac`,
six `a_tac` LON, eleven `g_str` and six `a_str` types **cover the empirical distribution of a
4 729-clip driving corpus as produced by a 34.3 B frontier driving VLM.**

### 7.2 ⛔ The one open decision — v6.0 vs v6.1, and it is live TODAY

| | |
|---|---|
| `models/v6.py:281` `tactical_lat_actions()` | **defaults to v6.0** (6 tokens, no TURN) — because the live v6F S-W run holds `vocab_a_lat.table.weight (6,128)` under a **tensor-strict** resume contract |
| `lake/tac_str_labels.py:538,570` · `scripts/vlm_tac_compose.py:67` | **default to v6.1** (8 tokens, `TURN_L`/`TURN_R` appended at indices 6-7) |

**The label side already emits two tokens the model side cannot consume.** That is the freeze
blocker, and it is not a question this measurement can settle.

**My recommendation: adopt v6.1 and converge the model onto the labels.** Reasons, in order of
weight: the label pipeline already ships it and reverting would force `Turn Left/Right` into
either a false `LANE_KEEP` or a permanent abstain on 186 clips; appending preserves indices
0-5, so no artifact needs re-derivation and a 6-wide head widens by padding; `v6.py:262-266`
already pins the discipline that follows (**representable, not scoreable** below n=200, which
`TURN_L`=85 and `TURN_R`=101 both are); and the change is free before the first factored
checkpoint and expensive after.

⚠️ **The honest counter-argument, stated because it is real.** Our LAT axis is
**corridor**-relative, not lane-marking-relative — `g_str:KEEP_CORRIDOR` is defined as
*"hindsight path curvature-relative follow"*. Under that definition `LANE_KEEP` inside a
junction is **not** false: the corridor is simply curving, and `g_str:TURN_LEFT` already flows
down as conditioning (§5 of the vocabulary doc). On that reading `TURN_L`/`TURN_R` are
redundant, and adding them makes `a_tac` LAT the only axis with mixed-horizon members — the
exact defect class being retired. **Both readings are defensible. This is the PI's call.**

⚠️ **The mapping table in §3.3 is version-parameterised so it is usable either way, and
coverage is 93.57 % under both.** Nothing in this report depends on the outcome.

### 7.3 ⭐ The decision that has NOT been argued about, and is 4.3× larger

`tac_str_labels.py:103-104` reads LAT from the **`Lane` axis only**; the CoT is consumed for
the longitudinal referent. **MEASURED: 800 instances (16.92 %) have a CoT that says NUDGE
while the `Lane` axis says `Lane Keep`.** All 800 are currently labelled `LANE_KEEP`.

Either the `Lane` axis fails to encode real sub-lane evasions (and 800 `NUDGE_L/R` labels are
being thrown away), or the CoT embellishes (and reading it would inject 800 false nudges).
**This is UNADJUDICATED, and at 16.92 % it is 4.3× the TURN question.** The cheapest
discriminating experiment is direct: take the ~100 clips with `cot=NUDGE, axis=LANE_KEEP` that
have w120 video built, measure the **hindsight lateral offset within the lane**, and see which
channel the geometry agrees with. Engine A disposes, as it does everywhere else.
**This is a proposal, not a launch.**

### 7.4 Smaller items

- **`gentle`/`strong` severity is lost** on every deceleration/acceleration (§3.2), 3 194
  instances. Not a missing token — an **arg** question: the existing uniform constraint slots
  (`within_m`, `by_time_s`) already express severity as a *deadline*, which is the physical
  form. Recommend deriving it from geometry rather than adding a categorical severity token.
- **A production tokenizer should emit a token SET, not a class** (§4.1, 38.21 % multi-label).
- **`EVADE_IN_CORRIDOR`'s `obstacle_slot` needs a static-object supplier** (§6.4, 3.13 %) —
  measured demand for Engine C.

---

## 8. ⭐ VERDICT ON THE TOKENIZER FREEZE

> **The measured coverage SUPPORTS freezing the token TYPE SET. It does NOT support declaring
> the tokenizer frozen today, and the reason is not coverage — it is that two parts of the
> repo are already frozen to different versions.**

**Freeze now, on this evidence:**

- `g_tac` (9), `a_tac` LON (6), `g_str` (11), `a_str` (6) — **no corpus evidence for a new
  type.** 99.87 % LON, 93.57 % LAT, 84.39 % grounded goals / 99.73 % including the default,
  and **0 phrases that match nothing.**

**Blocking the freeze — one item, PI-only:**

- ⛔ **Name ONE `a_tac` LAT version.** v6.0 (6) or v6.1 (8). `tac_str_labels.py` and `v6.py`
  currently disagree, and a freeze that leaves them disagreeing is not a freeze. My
  recommendation is **v6.1** (§7.2), with the counter-argument stated.

**Must be resolved BEFORE the frozen vocabulary is supervised (not before the freeze itself):**

1. **§7.3 — which channel supplies LAT.** 16.92 % of labels turn on it.
2. **§6.5 — `SPEED_BAND`'s `v_hi` from Alpamayo is inadmissible.** Discard the arg; keep the token.
3. **§6.2 — declare the `Stop`-row LAT policy**: abstain, or impute from geometry. 6.43 % of
   windows need an answer and the corpus has none.
4. **Cross-draw stability is UNMEASURED.** One draw at temperature 0.6 bounds every proportion
   above. A second seed on a subset is cheap and would put an interval on all of it.

**Explicitly NOT blocking:** the Alpamayo parity contamination (§1.1). It forbids using this
corpus as a *holdout*; it does not touch a vocabulary-coverage measurement, and no number here
scores a model.

---

## 9. Deliverable manifest

| artifact | where it lives | verified |
|---|---|---|
| this report | `repo:TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-23-alpamayo-mapping/ALPAMAYO_MAPPING.md` | `git ls-files --cached` |
| the mapping + coverage measurement | `repo:…/2026-08-23-alpamayo-mapping/raw/alpamayo_mapping_coverage.json` | `git ls-files --cached` |
| the raw phrase distribution (all 1 062 distinct CoT + 98 joint triples + all axis counts) | `repo:…/2026-08-23-alpamayo-mapping/raw/alpamayo_phrase_distribution.json` | `git ls-files --cached` |
| the re-derivation script | `repo:…/2026-08-23-alpamayo-mapping/code/map_alpamayo.py` | `git ls-files --cached` |
| the intent rules, with per-rule rationale | `repo:…/2026-08-23-alpamayo-mapping/code/alpamayo_rules.py` | `git ls-files --cached` |

**Nothing is stranded.** The only off-repo input is `records.parquet` (26 MB), which is
**not** banked here on purpose: it lives on HF at `Sayood/tanitad-alpamayo2-augmentation` and
its sha256 is pinned in this report and in the JSON, so the chain is closed without adding a
third copy of gated clip data to the repo.

**Re-derivation is bit-exact.** Running `code/map_alpamayo.py` from its repo path against the
HF parquet reproduces both `raw/*.json` byte-for-byte (md5 checked, both files).

## 10. Escalations

1. ⛔ **PI: name one `a_tac` LAT version (v6.0 or v6.1).** This is the tokenizer-freeze
   blocker. §7.2. Recommendation: v6.1, counter-argument stated.
2. ⛔ **UNOWNED: §7.3, the 16.92 % LAT channel question.** Larger than the item everyone has
   been arguing about, and nobody has it. Needs the ~100-clip geometric adjudication.
3. ⚠️ **`SPEED_BAND`'s F-14 blocker now has a second door.** `v6.py:188-200` names sign/OCR
   and corridor priors; **Alpamayo's CoT is a third route to the same forbidden field** and
   the blocker cell does not mention it. Whoever owns F-14 should add it.
4. ⚠️ **`HIERARCHY_VOCABULARY.md` §1 says "4,800-clip"; the corpus is 4 729.** A one-word fix
   in a doc I am not editing.
