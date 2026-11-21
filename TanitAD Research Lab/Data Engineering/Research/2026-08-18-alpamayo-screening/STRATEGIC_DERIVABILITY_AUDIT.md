# Strategic vocabulary — what the future ego path can and cannot give

**2026-08-18.** ⛔ **Written because I overstated it.** The concept's §8.4 said *"strategic labels for
all 4,723 clips are derivable without the w120 extraction"*. The PI challenged it, naming
`LANE_TARGET`, `PREPARE_LANE_CHANGE`, `HOLD_CORRIDOR`, `PREPARE_EXIT` and `ROUTE_TO`. **He is right:
a SUBSET of the vocabulary is geometry-derivable, and the tokens he named are exactly the ones that
are not.** Three of them were already formally ruled out or gated by his own prior rulings, which I
should have read before generalising from *"Engine A covers every clip"*.

All evidence below is from `stack/scripts/s2_derive.py`, `stack/scripts/ph1_fuse.py` and
`stack/tanitad/models/v6.py` — the shipped code, not a summary of it.

---

## 1. THE AUDIT — 17 tokens, token by token

**Legend:** ✅ geometry-derivable from the hindsight ego path · ⚠️ derivable but weak/proxy ·
⛔ NOT derivable — needs lane topology, a route, or an intent that geometry never observed.

### `g_str` — strategic goals (11)

| token | status | evidence |
|---|---|---|
| `TURN_LEFT` / `TURN_RIGHT` | ✅ | `_JUNCTION_ROUTE_TOKENS`; net heading over the window |
| `STRAIGHT_THROUGH` | ✅ | same route engine |
| `EXIT_LEFT` / `EXIT_RIGHT` | ⚠️ | in `_JUNCTION_ROUTE_TOKENS`, but *which lane serves the exit* is unknown (§2) |
| `STOP_AT` | ✅ | `_STOP_EVENT_TOKENS` — purely kinematic |
| `FOLLOW_MAIN_ROAD` | ✅ | the DEFAULT when no route is set (PI 2026-08-11) — a default, not a derivation |
| `KEEP_CORRIDOR` | ⚠️ | **no corridor exists** — PhysicalAI-AV ships no map or lane graph (dataset card: *"we do not include open maps data"*). Geometry can only say "lateral offset stayed small" |
| `NONE_ABSTAIN` | ✅ | abstention is always available |
| ⛔ **`LANE_TARGET`** | ⛔ **NEVER EMITTED** | *"`LANE_TARGET` leaves g_str emission ENTIRELY"* — the PI adjudicated 18 of 19 and **called 14 wrong**; he ruled the geometric gate out entirely on 2026-08-16 |
| ⛔ **`ROUTE_TO`** | ⛔ **GATED, unsupervisable** | G1 **CLOSED at 0/31**; a VLM `route_to` is remapped to geometry when junction geometry backs it and **ABSTAINED otherwise — never emitted as a guess** |

### `a_str` — strategic actions (6)

| token | status | evidence |
|---|---|---|
| `REDUCE_TO` | ✅ | kinematic gate `REDUCE_NET_DV_MS = −3.0` |
| `PREPARE_STOP` | ✅ | stop events |
| `RESUME_CRUISE` | ✅ | `_LAUNCH_EVENT_TOKENS` |
| ⛔ **`PREPARE_LANE_CHANGE`** | ⛔ **route-conditional** | admissible *"ONLY in service of route"* — it needs a ROUTE **and** to know the ego is not in the lane that serves it. Backlog #75: *"needs lane TOPOLOGY, 2 of 4 inputs missing"* |
| ⛔ **`HOLD_CORRIDOR`** | ⛔ | a corridor is a lane-geometry object; none exists in the corpus |
| ⛔ **`PREPARE_EXIT`** | ⛔ | `ph1_fuse.py`: *"an exit needs LANE CONTEXT (which lane serves the exit) … and `lane_context` is None on every clip today"* — **both axes abstain BY DECLARATION** |

**Tally: ~10 of 17 derivable, 2 weak, 5 structurally unavailable.**

## 2. ⭐ THE MEASUREMENT THAT KILLS THE OBVIOUS WORKAROUND

One might hope a lateral-offset threshold recovers lane changes from geometry. It does not:

> `s2_derive.py`, on the retired `LC_MIN_LAT_M = 3.0` gate: *"measured on aug120, **15 of the 19
> clips this gate fired on have a lateral offset FULLY EXPLAINED by constant-curvature road
> following**"*

⇒ The geometric lane-change detector was mostly detecting **road curvature**. That is why the gate
was *"SUPERSEDED AS AN EMISSION GATE"* and retained only for corroboration — *"it may never again
decide a token"*.

## 3. WHY THE VLM DOES NOT SIMPLY FILL THE GAP

The missing quantity is **lane topology** — which lane the ego is in, how many there are, which one
serves the exit. That is a *map* fact. The VLM can see affordances, but its measured strategic
behaviour is the reason it was demoted: TURN precision 100 % (19/19) yet recall **17/33 L and
2/29 R**, and **28 of its 31 `ROUTE_TO` claims sat on plain turn geometry**.

⇒ **These five tokens are a DATA gap, not a labelling-strategy gap.** No ordering of Alpamayo, VLM
and ego produces them from what the corpus contains today.

## 4. ⚠️ AND YET THE TOKENS STAY IN THE VOCABULARY — deliberately

> `s2_derive.py`: *"Both tokens REMAIN IN THE v6 VOCABULARY — the tuples size embedding tables and
> the live v6F run resumes strictly against them; **zero training support is safe, a changed shape
> is not**."*

⇒ The strategic head will carry **dead classes** — present, shaped, never supervised. That is the
correct trade (a shape change would break the strict resume), but it must be **stated in every
strategic result**, or a reader will assume 17-way competence where ~10-way was trained.

## 5. WHAT THIS DOES TO THE CONCEPT

* §8.4's claim is **narrowed**: *strategic labels for the geometry-derivable subset are available
  without video; the topology-dependent tokens are not available at all, from any tier.*
* The parallel CPU strategic build **still stands** — it just produces ~10 of 17 tokens.
* **The honest framing of the strategic layer for all three models:** it is supervised on executed
  route geometry and kinematic events, and it is **blind to lane-relative intent**. Any claim about
  strategic *planning* competence must say which tokens were trainable.
* ⭐ The lever that would change this is **lane topology** — the same missing input behind
  `PREPARE_LANE_CHANGE` (#75) and `SPEED_BAND`'s corridor priors. It is one data gap sitting under
  several stalled capabilities, which makes it a better target than any of them individually.
