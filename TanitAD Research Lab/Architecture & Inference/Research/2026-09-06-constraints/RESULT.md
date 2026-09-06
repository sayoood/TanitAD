# C-ENV-1 — the constraint channel: a target speed EXISTS and is unsupervised, a max speed DOES NOT EXIST anywhere, and the fix is to score constraints on the OUTPUT

**Date** 2026-09-06 (Europe/Berlin) · **Stream** Architecture & Inference ·
**Branch** `agent/arch-inf-20260803` · **Evidence class MEASURED unless stamped
otherwise** · **GPU-days spent: 0. No model was trained. The A40's refcv5 run was
not approached.**

**Answers PI instructions (1) constraints on goals, (8) collision/clearance, and
(10) target speed + MAX SPEED.**
**Pre-registration:** `SPEC.md` (this directory).

---

## 0. THE ANSWER, in six lines

1. **A target speed EXISTS** — `a_tac.lon_args.v_target_ms` on **4,572/4,572**
   v7.2 TRAIN records — and **no loss reads it.** Three independent structural
   mechanisms block it; it is `audit`-only.
2. **A speed ENVELOPE EXISTS** — `g_tac.goals.SPEED_BAND.{v_lo_ms, v_hi_ms}` on
   **4,572/4,572** — and **the loader drops it entirely** before it reaches even
   the audit dict.
3. ⛔ **But it is not a constraint.** It is literally `[min(v), max(v)]` of the
   **EGO'S OWN FUTURE SPEED** over 2–6 s. Median width **1.44 m/s**.
   `grounded=False` on **4,572/4,572**. Supplying it is the nav echo with *no*
   horizon guard.
4. ⛔ **A MAX SPEED exists in no layer** — raw, augmented, or lake. Every
   candidate is ego-derived, circular, or absent.
5. ⭐ **NEW, measured today: VISION CANNOT SET THE TARGET SPEED** on REF-C-base's
   frozen fan. `img` band top-1 **0.2379** against **0.4078** for the free
   baseline of repeating `v0`'s band — *separated worse* (−0.1699
   [−0.2859, −0.0614]).
6. ⇒ **THE DESIGN: constraints are SCORED ON THE OUTPUT, never SUPPLIED AT THE
   INPUT.** An input that does not exist cannot echo. Built, tested (35 tests),
   and demonstrated to fail a deliberately bad arm.

---

## 1. P1 — what exists today, PER LAYER, with coverage

⚠️ **Corpus correction up front.** These labels are joined to **B1
`physicalai-b1-w120-256x640cyl`**, *not* the `physicalai-train-e438721ae894`
parity set. Coverage fractions below are against B1.

| signal | layer | coverage | supervised? | admissible as an INPUT? |
|---|---|---|---|---|
| `a_tac.lon_args.v_target_ms` | v7.2 augmented | **4,572/4,572** | ⛔ **NO** — `audit`-only; `V72_ARGS_SUPERVISED = False` | ⛔ ego-future |
| `g_tac.goals.SPEED_BAND.v_lo_ms/v_hi_ms` | v7.2 augmented | **4,572/4,572** | ⛔ **NO** — dropped by `_goal_audit`'s keep-tuple | ⛔ ego-future, no guard |
| `a_str.args.v_target_ms` | v7.2 augmented | 558/4,572 | ⛔ NO — not surfaced | ⛔ ego-future |
| `cot_tokens.speed_limit` | v7.2 augmented | 4,572/4,572 | ⛔ NO | ⚠️ a **boolean**, not a value |
| `vtarget_guarded` | lake | 305,803/472,627 poses (64.70 %) parity-train; 637/881 windows (72.3 %) val40 | available, unused | ⛔ **INADMISSIBLE** supplied (ΔR² +0.0996) |
| `VSOURCE` `sign_limit` | lake vocab | **0** — `vsource_at` can only ever emit `curve_constrained` or `road_class_default` | — | n/a |
| posted speed limit | ⛔ **nowhere** | **0** | — | — |
| map / lane graph | ⛔ **nowhere** | **0** | — | — |

⚠️ **Temporal coverage is the number that limits everything:** one v7 record per
clip, `t0_s = 8.0`, tactical band 4.0 s wide ⇒ **18,288.0 s / 508,726.5 s =
3.59 %** of recording time (11.87 % of usable horizon). A scored frame outside
`|t_now − 8.0| ≤ 2.0` s has **no v7 tactical GT at all**.

### 1.1 ⛔ Why `SPEED_BAND` is a description, not a constraint — settled from source

`stack/scripts/s2_geom_emit_v7.py::tactical_goals`, verbatim:

```python
v = p[lo:hi + 1, 3]                 # ego speed over the 2-6 s tactical band
goals["SPEED_BAND"] = {"v_lo_ms": round(max(0.0, float(v.min())), 2),
                       "v_hi_ms": round(float(v.max()), 2), ...}
```

Census over the banked blob (`raw/speed_band_census_train.json`, md5
`0ff902130ce76886b8a925eceed9e3a5` — matches `MODEL_REGISTRY.md` independently):
n = 4,572; `v_hi` p50 **9.99** m/s; band width p50 **1.44** m/s, p95 5.79;
**729 distinct widths** (so not a stub); `grounded=False` **4,572/4,572**;
`v_target_ms` **never** above `v_hi` (0 records), inside on 2,483, below `v_lo`
on 2,089.

⇒ a 1.44 m/s median bracket around the ego's realised speed cannot be *"the max
speed the model can drive"*. And its band **[2 s, 6 s] fully contains** the
scored tactical horizon, so it has **zero** horizon disjointness — strictly worse
than `vtarget_guarded`, which at least excises the scored window.

### 1.2 ⛔ The strata route is CIRCULAR — my own first reading, retracted

I measured `strata.road_class` explaining **0.5794** of the variance in `v_hi`
against a shuffled-cell control of **0.0311**, with p95s that looked like real
limits (urban 63.4 km/h, highway 128.8 km/h), and was about to report a usable
ceiling. **`road_class` is DEFINED BY A SPEED THRESHOLD** — from the selection
design: *"highway = ≥ 20 m/s sustained (≥ 30 % of the clip), intersection = a
stop and a heading change (≥ 25° yaw span), urban = the remainder."* The 57.9 %
is very largely the definition showing through.

⚠️ **The transferable lesson: the shuffle control cannot catch this.** It
separates GROUPING from NOISE and is structurally blind to a grouping whose
LABEL was derived from the scored quantity. Same family as *"a probe that tunes
on the data it scores"*, with the tuning moved upstream into the label.

⭐ **The honest number is the one non-circular axis:** `country`, from
PhysicalAI's own `data_collection` metadata, explains **0.0605** against the
**0.0311** shuffle floor. That is not a ceiling.

---

## 2. ⭐ NEW MEASUREMENT — vision cannot set the target speed

The 2026-08-04 D-VT1 package shipped `code/vt_band_from_vision.py` — the only
instrument that asks the deployable question directly — and **it was never run**:
its `raw/` holds 10 artifacts and none is its output. It is run here **unmodified**
(the poses view was built into the shape the script already declares, rather than
the script edited to fit the data).

**MEASURED** (`raw/vt_band_from_vision.json`): REF-C-base frozen fan, val40,
**971 valid windows / 39 episodes**, 20 of 23 bands present, leave-one-episode-out,
paired episode-cluster bootstrap B=2000. **Join proof: `max|v0 − pose speed| =
0.0`, bit-exact.**

| arm | band top-1 | paired vs the FREE baseline | separated |
|---|---|---|---|
| `identity_v0_band` (free, 0 params) | **0.4078** | — | — |
| `img` (pooled, 704-d) | 0.2379 | **−0.1699 [−0.2859, −0.0614]** | **True** |
| `img_v0` | 0.2297 | −0.1782 [−0.2942, −0.0727] | True |
| `past` (causal ego) | 0.2626 | −0.1452 [−0.2884, −0.0072] | True |
| `v0` | 0.2348 | −0.1730 [−0.3155, −0.0309] | True |
| `majority` (floor) | 0.1658 | −0.2420 [−0.4027, −0.0691] | True |

⇒ **every learned arm is separated WORSE than repeating the current speed's
band.** `refc1`'s `speed_cls` is a dead parameter on this trunk — now measured on
vision, not only on ego state.

⚠️ **Two honest qualifications, both binding.**
(a) This is a **linear readout on a FROZEN trunk with no ego_dropout**. A linear
negative is a negative about *linear decodability from this representation*, not
about learnability. State the function class.
(b) ⭐ **The pooled negative is driven by the high-speed regime and the
stratification says something different** — reported because a pooled number
hides it, and flagged as **POST-HOC, not pre-registered**:

| ego speed | n | `identity` | `img` | `past` |
|---|---|---|---|---|
| 3–6 | 160 | 0.0688 | 0.0875 | 0.1938 |
| 6–10 | 208 | 0.1058 | **0.2260** | **0.3221** |
| 10–15 | 201 | **0.5771** | 0.3433 | 0.2587 |
| 15+ | 280 | **0.8214** | 0.3464 | 0.3214 |

At 15+ m/s repeating `v0` is nearly correct (0.8214) and nothing can beat it; in
the 6–10 m/s band vision **does** beat identity (0.2260 vs 0.1058). The pooled
verdict is dominated by the regime where the free baseline is almost exact.
⛔ **Not promoted to a claim — it needs its own pre-registration.**

---

## 3. P2/P3 — the design that survives, and its instrument

Full design and both committed outcomes: **`SPEC.md`**. In brief:

⭐ **Constraints are scored on the OUTPUT, never supplied at the INPUT.** Both
ceilings come from the model's own proposed trajectory and from perception it
must already produce, so **no leak is possible by construction**:

* **kinematic** `v ≤ sqrt(a_lat_max/|κ|)`, κ from the proposed path
* **clearance** `v ≤ (gap − d0)/τ`, `gap` from the **in-corridor lead**

⛔ **Coordinate discipline held:** every quantity is a function of positions at a
single step. **No finite difference of a gap appears anywhere in the module**,
because the frozen trunk's closing rate is a clean null (+0.0061) while position
is decodable (+0.4145 [+0.2018, +0.6120], constant control exactly +0.000000).

### 3.1 The gate fires in both directions — B1 EVAL, 1,951 windows

| arm | `frac_over` | `frac_under_when_allowed` |
|---|---|---|
| `gt` | 0.1229 | 0.7960 |
| ⛔ `gt_x2speed` (path fixed, speed ×2) | **0.4755** | 0.4502 |
| ⛔ `gt_creep` (path fixed, 1.0 m/s) | 0.0041 | **1.0000** |

**A one-sided metric would have scored `gt_creep` near-perfect.**
**Vacuity gate: manoeuvre_rate 0.2019 (394/1,951)**, reported beside every number.

### 3.2 ⛔ A defect the GT control caught — and it would have been a false positive generator

The first run priced the ceiling off the **min distance to ANY agent within
60 m**. GT then violated its own envelope on **71.2 %** of steps and a 1 m/s
creeping arm "over-drove" on **27.3 %**. An adjacent-lane or parked car 2 m aside
is not a longitudinal constraint. ⇒ `trajectory_clearance` (proximity) and
`lead_gap` (in-corridor, ahead) are **two different quantities** and must never
be interchanged. Pinned by a test. Second defect the same run: unclamped double
differencing gave a ceiling minimum of **0.279 m/s = a 2.3 cm turning radius** —
the vtarget jitter trap in curvature costume; clamped at κ ≤ 0.2 1/m.

### 3.3 Clearance — its `n`, its censoring, its datum

`n = 68,977` steps · **p05 1.99 m · p50 8.06 m · p95 37.26 m**. Censoring
reported per family, never dropped: **49.67 %** no kinematic ceiling (straight
path), **65.09 %** no clearance ceiling (empty corridor), **33.20 %** neither.
⛔ **Datum: rig-origin to box EDGE, ego footprint NOT subtracted** — this is
`lead_source`'s convention and **not** `alpasim-gsplat`'s bumper-to-bumper
`EGO_LEN = 4.7 m`. The two differ by 4.7 m and must never be mixed.

⚠️ **It is AGENT clearance, and the brief's premise needs correcting on two
counts — but the second correction has itself a correction, and that one matters
for P3.**

**(i)** The 11th `obstacle.offline` label is **`train_or_tram_car`** (69 boxes /
3 clips here; **2,419 measured 2026-07-27**; 275 on a disjoint 193-clip train
set), **not** `protruding_object` — which was always inside the canonical 10.

**(ii)** **10 of the 11 classes are dynamic agents**, and "world-static" is a
per-TRACK property (1,756/2,778 = **63.2 %**, parked cars), never a class
property. Parked cars are therefore a **proxy** for the roadside edge and must be
named as one.

**(iii)** ⭐ **BUT ONE CLASS DOES CARRY INFRASTRUCTURE, AND IT IS UNUSABLE HERE
FOR A REASON WORTH STATING.** `protruding_object` (3,534 boxes, **0.39 %**) is
glossed in the programme as *"overhanging signs and branches"*, and its measured
geometry fits exactly: length **0.543 m**, width **2.091 m**, bottom face
**+1.676 m** above the road (n = 176) against −0.05…−0.13 m for every
ground-standing class. So the sharper statement is: **the only class with genuine
infrastructure content is OVERHEAD, and the join drops `center_z`/`size_z`** ⇒ it
cannot enter a BEV clearance metric at all, and a sign 1.68 m up is not a lateral
obstruction in the first place. ⛔ Recovering overhead clearance would require
re-building the join to keep z. It is excluded from every `VEHICLE_CLASSES`
variant, so **no lead-gap or clearance number here moves either way**.

⇒ **P3's honest answer: there is no infrastructure-proximity signal we can score
today.** What exists is agent clearance, with world-static parked cars as the
roadside proxy.

---

## 4. ⛔ THE UNDER SIDE IS NOT SCOREABLE TODAY — and this is the dataset decision

GT is competent human driving; a criterion flagging it as under-driving on most
windows is measuring the CEILING. Sweeping both thresholds
(`raw/constraint_panel_b1eval.json::under_side_sensitivity`), the GT under-rate
**never falls below 0.51 anywhere on the grid and RISES with the threshold** —
the signature of a degenerate criterion. The LEAD/NO_LEAD split localises it:
where a real lead gap binds it is 0.29–0.77; where none does (**23,534 / 43,298 =
54.4 %** of steps) the only ceiling is kinematic (~50 m/s on a straight road) and
the criterion collapses to *"GT is not driving at 50 m/s"* — true on **99.87 %**.

⭐ **⇒ the PI's "reach max speed when the situation allows" half REQUIRES a
road-speed ceiling that exists in no layer. It is a dataset extension, not a
modelling choice.** Routes and their honest errors are costed in `SPEC.md` §6.
⛔ **I did not invent a proxy and score against it.**

---

## 5. 🔴 ESCALATIONS — to named owners

1. **→ PI (decision).** Authorise, or refuse, a **max-speed corpus extension**.
   `SPEC.md` §6 costs three routes. ⛔ OSM map-matching is **impossible** on
   PhysicalAI (`egomotion` carries no lat/lon). Without one, the under-driving
   half of instruction (10) cannot be scored — only the over-driving half can.
2. **→ v7 label owners (DataFlyWheel).** `SPEED_BAND.v_lo_ms/v_hi_ms` are minted
   on **4,572/4,572** and dropped by `_goal_audit`'s keep-tuple. Even as an
   `audit` field they would make the envelope auditable at zero training cost.
   ⚠️ Surfacing them must ship with the statement that they are **ego-future**,
   or the next reader will feed them.
3. **→ REF-C/refcv5 owners.** `stack/tanitad/eval/constraints.py` is ready to
   wire; `SPEC.md` §2 names the arms. I did **not** touch `refc.py`,
   `refc_v3_train.py`, `refcv3_arm.py`, `taniteval/ci.py`, `train_v6_staged.py`,
   `v6.py`, `predictor.py`, `goal_point.py`, `stack/tanitad/rl/`,
   `refav1_lon_cost.py`, the tactical label reader, `CLAUDE.md`, or the paper.
4. **→ agent-slot / enum owners.** `train_or_tram_car` is measured across two
   disjoint clip sets yet `refc_agents.py` asserts it does not exist, and
   **two guards would now fail if the enum were corrected**
   (`refcv5_preflight.py`, `test_refc_agents.py`). The 69 boxes currently train
   as class-unknown (`-1`) — silent, not fatal.
5. **→ eval/tools.** `goal_admissibility` still has **zero call sites** outside
   its own test (first escalated 2026-08-04, re-confirmed 2026-08-16). Every
   finding in §1 is exactly what it exists to catch.

---

## 6. What I did NOT do

* ⛔ **No model was trained and no arm was scored.** §3's arms are trajectory
  arms over banked GT; they characterise the INSTRUMENT.
* I did **not** promote the §2 stratification to a claim — it is post-hoc and
  needs its own pre-registration.
* I did **not** re-derive the `vtarget_guarded` ΔR² numbers; they are cited
  INHERITED from the 2026-08-04 package with paths.
* I did **not** invent a max-speed proxy, and did not score anything against one.
* I did **not** touch the A40, any pod, or the parity corpus.

---

## 7. Deliverable manifest

| artifact | path | state |
|---|---|---|
| **constraints module** | `stack/tanitad/eval/constraints.py` | repo, **staged** |
| its tests (**35 passed**) | `stack/tests/test_constraints.py` | repo, **staged** |
| pre-registration | `…/Research/2026-09-06-constraints/SPEC.md` | repo, **staged** |
| this document | `…/2026-09-06-constraints/RESULT.md` | repo, **staged** |
| SPEED_BAND census | `code/speed_band_census.py` · `raw/speed_band_census_train.json` | repo, **staged** |
| strata ceiling probe (+ its retraction) | `code/strata_ceiling_probe.py` · `raw/strata_ceiling_probe.json` | repo, **staged** |
| constraint panel | `code/run_constraint_panel.py` · `raw/constraint_panel_b1eval.json` | repo, **staged** |
| ⭐ vision→target-speed, first ever run | `raw/vt_band_from_vision.json` | repo, **staged** |
| poses-view builder | `code/build_poses_view.py` | repo, **staged** |

Nothing lives only on a pod or only in a worktree. The intermediate poses NPZ
(`C:/Users/Admin/tanitad-caches/constraints-20260906/val40_poses_view.npz`,
md5 `6cb0bf5859aacb1f05191c0f358f99f6`) is **regenerable** from
`code/build_poses_view.py` and is deliberately not banked.
