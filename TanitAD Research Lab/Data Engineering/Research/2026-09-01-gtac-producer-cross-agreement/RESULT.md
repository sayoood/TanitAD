<title>RESULT - the two g_tac producers are not a fork, they are near-disjoint (D-DATA-GTAC-b)</title>

# RESULT — the two `g_tac` producers agree on 1 of 7 scenarios, and "which threshold wins" is the wrong question

**Package** `E-LAB-DATA-0901` · 2026-09-01 · Data Engineering · 0 GPU (CPU only)
**Serves** `LAB_BACKLOG` ranked row **7** (P0) = register **D-DATA-GTAC-b**
**Class** `MEASURED` (ours) — analytic trajectories through both producers, `raw/gtac_cross_agreement.json`
**Tier stamp** n/a — label-producer comparison, not a model eval.

⚠️ **Scope, up front.** The agreement rate below is over **7 hand-built scenarios**, not over the
parity corpus. It is enough to establish *that* and *how* the producers differ; it is **not** a
corpus agreement rate and must never be quoted as one. What a real-pose run would add is stated
in §5.

---

## 1. The headline — the backlog row's framing needs correcting

Row 7 reads: *"two producers of the SAME label family are live with unmeasured agreement — a
silent fork in the canonical labels"*, and asks to *"settle the CORRIDOR_OFFSET 2-vs-1 contest"*.

**Measured, both producers are geometry-only and take an identical contract** (`poses [T,4]` =
(x, y, yaw, v), a key index, the same 2–6 s band), so the comparison is direct and needs no
adapter. On that contract:

### F1 — the emittable vocabularies are near-disjoint **by construction** `MEASURED (set arithmetic)`

`g_tac_geom` (**B**) carries a machine-readable `REFUSED` dict. Intersecting it with
`tactical_goals`' (**A**) declared tokens:

| axis | A's tokens | refused by B | **A's survivors** |
|---|---|---|---|
| **LAT** | ANCHOR_GOAL, CORRIDOR_OFFSET, EVADE_IN_CORRIDOR, LAT_UNCONSTRAINED | **3 of 4** | **`LAT_UNCONSTRAINED` only** |
| **LON** | SPEED_BAND, GAP_TARGET, YIELD_AT, STOP_POINT, WAIT_FOR_ONCOMING, TRAFFIC_LIGHT_REACT, LON_UNCONSTRAINED, ADAPT_SPEED_FOR_CURVE | **5 of 8** | ADAPT_SPEED_FOR_CURVE, LON_UNCONSTRAINED, STOP_POINT |

B's whole emittable set in its default arm is **{STOP_POINT, LON_UNCONSTRAINED, LAT_UNCONSTRAINED,
ABSTAIN}**.

⇒ **On the LAT axis every informative token A emits is one B refuses.** There is no
"2-vs-1 threshold contest" to settle, because B does not emit `CORRIDOR_OFFSET` at any threshold —
it **refutes the token** with a measured oracle ceiling. (The constants are also not 2-vs-1:
A uses `CORRIDOR_OFFSET_M = 1.0`, B uses `LAT_OFFSET_MIN_M = 0.75` *and* refuses.)

### F2 — behavioural agreement on the LON axis: **1 of 7 (14.3 %)** `MEASURED`

| scenario | A (lat / lon) | B (lat / lon) | LON |
|---|---|---|---|
| **K1** straight cruise *(control)* | ANCHOR_GOAL / **SPEED_BAND** | ABSTAIN / LON_UNCONSTRAINED | ✗ |
| **K2** decel to stop *(control)* | ANCHOR_GOAL / **STOP_POINT** | ABSTAIN / **STOP_POINT** | ✅ |
| S1 left arc | **CORRIDOR_OFFSET** / SPEED_BAND | ABSTAIN / LON_UNCONSTRAINED | ✗ |
| S2 sharp junction turn | ANCHOR_GOAL / SPEED_BAND | ABSTAIN / LON_UNCONSTRAINED | ✗ |
| S3 lane change | ANCHOR_GOAL / SPEED_BAND | ABSTAIN / LON_UNCONSTRAINED | ✗ |
| S4 evade & return | **CORRIDOR_OFFSET** / SPEED_BAND | ABSTAIN / LON_UNCONSTRAINED | ✗ |
| S5 stop and go | ANCHOR_GOAL / **STOP_POINT** | ABSTAIN / **ABSTAIN** | ✗ |
| *K3 creep (degenerate — recorded, **not scored**)* | LAT_UNCONSTRAINED / STOP_POINT | ABSTAIN / ABSTAIN | — |

**Controls both PASS**, which is what makes the 14.3 % interpretable:
- **K1** (no stop exists): neither producer emitted STOP_POINT ✅
- **K2** (a stop exists): both emitted STOP_POINT ✅

⭐ Without K1/K2 the number would be uninterpretable — two producers that both always abstained
would score 100 %.

⭐ **And where both DO speak, they agree tightly.** On K2 the stop arc-length reads
**A 16.4 m vs B 16.34 m** — 0.06 m apart. **The disagreement is about *what to emit*, never about
the geometry when both emit it.**

### F3 — ⛔ A emits `SPEED_BAND` from the exact substitution B names as inadmissible `MEASURED`

A emits `SPEED_BAND` on **5 of 8** scenarios, with provenance string **`geometry(held-speed)`**.
B's `REFUSED["SPEED_BAND"]` says (verbatim, `g_tac_geom.py`):

> *"F-14 BLOCKER … both named derivation inputs are unavailable on this corpus and one is
> FORBIDDEN rather than missing … **vtarget_guarded is NOT the substitute: it is hindsight ego
> behaviour, not a permitted speed.**"*

⇒ A's `geometry(held-speed)` **is** hindsight ego behaviour. This is not two thresholds
disagreeing — one producer emits a label the other has documented as underivable from the
available inputs, and derives it from the specific quantity named as disqualifying.

### F4 — ⭐⭐ A's `CORRIDOR_OFFSET` reports **21.65 m** on a plain constant-curvature arc `MEASURED`

On `S1_left_arc` (constant curvature 0.02 rad/m, 8 m/s — an ordinary sustained turn, no lane
change), A emits `CORRIDOR_OFFSET` with **`lat_offset_m = 21.65`, `arc_m = 46.2`**.

**21.65 m is 21× A's own `CORRIDOR_OFFSET_M = 1.0` threshold**, and ~6 lane widths. It is not a
corridor offset at all — it is the turn itself, measured as deviation from the ego's initial
heading ray. On `S4_evade_return` (a genuine in-lane nudge) the same code reads a sensible
**1.28 m**, so the machinery is not uniformly broken: **it fails specifically when the path
curves**, which is exactly the failure mode B's refutation predicts.

⇒ This is an **independent mechanical reproduction of B's refutation**, reached from a different
direction (a synthetic arc, not an oracle-circle residual study), and it cost seconds of CPU.

---

## 2. What this changes for TanitAD — 3 recommendations

1. ⭐ **Rewrite backlog row 7 / D-DATA-GTAC-b.** "Measure cross-agreement and settle the
   threshold contest" presumes two implementations of one label. **Measured, they are a
   near-disjoint pair**: B is a declared-refusal subset whose entire LAT axis abstains. The live
   question is not *which threshold* but **"is A's output admissible as canonical supervision?"** —
   and F3/F4 say at least `SPEED_BAND` and curved-path `CORRIDOR_OFFSET` are not.
2. ⛔ **Do not train on A's `SPEED_BAND` or on `CORRIDOR_OFFSET` where the band contains
   curvature, pending a PI call.** F3 is a documented admissibility violation (hindsight ego
   behaviour as a permitted speed — the same family as the binding "labels may use ego, inference
   is vision-only" rule being satisfied on the *label* side but the token then meaning something
   it cannot mean). F4 is a measured mislabel with a 21× margin. ⚠️ This is a **recommendation to
   the Master Mind/PI, not a unilateral fence** — the Lab does not fence a producer.
3. **Add a curvature guard as the cheap interim.** If `CORRIDOR_OFFSET` must keep shipping,
   gate it on the band's turn angle (B already computes a `theta = |kappa|*s` stratification and
   found the only admissible band is `theta < 0.05 rad`, **21.7 % of the corpus**). That single
   gate would have caught F4.

## 3. Controls and their readings

| id | control | reading | verdict |
|---|---|---|---|
| K1 | straight cruise — no stop exists | A: no STOP_POINT · B: no STOP_POINT | ✅ pass |
| K2 | decel to zero — a stop exists | A: STOP_POINT · B: STOP_POINT (16.4 vs 16.34 m) | ✅ pass |
| K3 | sub-threshold creep (0.3 m/s < `V_STOP_MS` 0.5) | A: STOP_POINT · B: ABSTAIN | recorded, **not scored** (declared degenerate before the run) |

## 4. Deliverable manifest

| artifact | where | only place? |
|---|---|---|
| `SPEC.md` | `repo:` this package | staged |
| `code/gtac_cross_agreement.py` | `repo:` this package | staged |
| `raw/gtac_cross_agreement.json` | `repo:` this package | staged |
| `raw/search_log.md` | `repo:` this package | staged |
| `RESULT.md` (this file) | `repo:` this package | staged |

## 5. Limits — what a real-pose run would add, and what it would not

- ⛔ **14.3 % is a scenario-set rate, not a corpus rate.** My 7 scenarios were chosen to span
  common manoeuvres; a corpus would weight them by real frequency. **The corpus number requires
  parity poses** (`physicalai-train-e438721ae894`), which are not on the dev box — the local
  window dumps hold only 4 waypoints, far short of the 2–6 s band.
- **F1 (vocabulary) is corpus-independent** — it is set arithmetic over source and cannot change.
- **F3 and F4 are existence results**: one counterexample establishes them, and real poses can
  only make them more common, never refute them.
- Analytic trajectories are noiseless. Real poses carry jitter that could push A's
  `CORRIDOR_OFFSET` around further, so **F4's 21.65 m is a clean-case floor, not a worst case**.
- ⚠️ Both producers were called **geometry-only**, which is their full signature (A takes only
  `poses`, `key`, `hz`, an optional `stop_reason`, `lon_vocab`). A is **not** being starved of an
  input it would normally have — checked in source before the comparison, because "the producer
  was missing its VLM leg" would have been the obvious way for this result to be an artifact.

## 6. Evidence-class ledger

| claim | class | source |
|---|---|---|
| vocabulary sets, refusal intersections | MEASURED (set arithmetic over source) | `raw/gtac_cross_agreement.json` §S1 |
| 1/7 LON agreement; per-scenario tokens; K1/K2 pass; 16.4 vs 16.34 m | MEASURED (ours) | same, §S2 + `controls` |
| `SPEED_BAND` provenance `geometry(held-speed)`; `lat_offset_m` 21.65 / 1.28 | MEASURED (ours) | same, §S2 |
| B's refutation (oracle p50 1.80 m, theta<0.05 rad = 21.7 % of corpus) | INHERITED (B's docstring, **not re-verified here**) | `stack/tanitad/data/g_tac_geom.py` module docstring |
| "A's SPEED_BAND is inadmissible" | **recommendation**, resting on F3 + B's documented blocker | not a PI decision |
