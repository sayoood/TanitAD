# V3 factorized tactical head — SPEC (2026-07-21)

*Specification only. NOTHING here is implemented or trained: it implies a REF-C retrain, which is
Sayed's call. Written so he can price it. The LABEL side (`--labels v3`) is implemented and staged;
the MODEL side is this document.*

Companion to `V3_GOAL_VOCABULARY_V1.md` (FROZEN). Nothing here changes that vocabulary — the whole
point is that the vocabulary was designed correctly and the head collapsed it.

---

## 1. The defect

```
refc.py:88-91     N_MANEUVERS = 5
                  0 lane_keep | 1 turn_left | 2 turn_right || 3 accelerate | 4 brake_stop
                  \_______ 3 LATERAL _______/               \_ 2 LONGITUDINAL _/
                                        ONE softmax
refb_labels.py    classify_maneuver / classify_maneuver_v2 resolve the collision by PRIORITY:
                  turn > brake > accel > lane_keep
```

`vocab.py:88-102` already models these as **independent orthogonal slots**: `LATMANEUVER` (9,
incl. `lane_keep`), `LONMODE` (9, incl. `stop_at_point`, `hold_stop`, `creep`, `coast`),
`TACPOINT` (5, incl. `stop_line`). In `planfan_bad_selection_good_fan_ep19` the true state is
`lane_keep` **and** `stop_at_point` **and** a stop point a few metres ahead, simultaneously. A
5-way softmax can name one of them, so it names the lateral one.

### Measured

| quantity | value | source |
|---|---|---|
| `accelerate` predicted | **0 / 881** (both arms) | `taniteval/results/planfan_clips_tactical_head_val.json` |
| `brake_stop` predicted | 7/881 base, 4/881 XL | same |
| GT longitudinal (5-way, i.e. **survivors** of the priority) | 195/881 = **22.1 %** | same |
| GT longitudinal while the head says lateral | **21.8 %** | same |
| GT longitudinal **present** before the priority collapse | **242/881 = 27.5 %** | RECONSTRUCTED from the fan's own GT waypoints, `fan_refc-base-30k.pt` |
| destroyed by `turn > brake > accel` | **63/881 = 7.2 %** | same reconstruction |
| 5-way says LATERAL while a factorized LONMODE is live (7 s horizon, incl. `coast`/`creep`/`hold_stop`) | **24.9 %** | MEASURED, `label_v3_audit.py`, 100-episode val build (see §6 caveat) |
| a LONMODE other than `free_cruise` is live | **44.1 %** | same |

`graft_maneuver=True` adds `maneuver_to_anchor(log_softmax(maneuver_logits))` **into the anchor
logits that make the selection** (`refc.py:569-571`). A ~99.5 % lateral-or-neutral prior therefore
enters the ranking directly — which is why there is no longitudinal signal anywhere in the
selection path.

---

## 2. Proposed model change

### 2.1 Two independent heads

Replace the single 5-way head

```python
N_MANEUVERS = 5
self.maneuver_head = nn.Sequential(..., nn.Linear(aux_hidden, N_MANEUVERS))
```

with two heads over the **kinematically mintable subsets** of the frozen vocab slots:

```python
# scripts/refb_labels.py is the authority for what is mintable TODAY.
N_LAT = len(refb_labels.LAT_KINEMATIC_TOKENS)   # 7  (+1 sentinel = 8 logits)
N_LON = len(refb_labels.LON_KINEMATIC_TOKENS)   # 6  (+1 sentinel = 7 logits)
self.lat_head = nn.Sequential(..., nn.Linear(aux_hidden, N_LAT + 1))
self.lon_head = nn.Sequential(..., nn.Linear(aux_hidden, N_LON + 1))
```

The `+1` is the `unknown` sentinel at index `len(table)` — the same never-clamp discipline as
`ROUTE_UNKNOWN`: an unjudgeable window must be **masked**, never folded into a real class. The
lead-referenced LONMODE tokens (`follow_lead`, `close_gap`, `open_gap`) and the context LATMANEUVER
tokens (`merge_in`, `yield_merge`) are **not in the head at all** — no lead state exists to supervise
them, and a logit no label can ever train is a dead parameter that only invites a shortcut.

Loss: two masked cross-entropies with the existing `MANEUVER_WEIGHT` split between them
(PROPOSED 0.05 / 0.05 to keep the total aux pressure identical to today's 0.1 — this is a knob, not
a finding). Targets come straight from the staged `--labels v3` batch fields `lat_idx` / `lon_idx`.

**Keep `maneuver_logits` [B,5] emitted and supervised as today** for one milestone, so the A/B is
`5-way only` vs `5-way + LATxLON`, and every published REF-C number stays reproducible. Retiring the
5-way head is a second, separate decision.

### 2.2 What `maneuver_to_anchor` should carry

Today: `conf += maneuver_to_anchor(log_softmax(maneuver_logits))`, `nn.Linear(5, N_anchors,
bias=False)`.

PROPOSED: **two grafts, summed, so the longitudinal prior can reach the ranking at all.**

```python
conf = conf + self.lat_to_anchor(log_softmax(lat_logits))    # Linear(N_LAT+1, N)
            + self.lon_to_anchor(log_softmax(lon_logits))    # Linear(N_LON+1, N)
```

Rationale, and why this is the load-bearing half of the change:

1. **The anchor vocabulary is 2-D.** Each anchor is a trajectory with both a lateral shape and a
   speed profile. A rank-5 lateral-dominated prior can only re-rank anchors along the lateral axis;
   `lon_to_anchor` is what lets "we are stopping" suppress every anchor that keeps rolling.
2. **Keep them additive and separate, not concatenated into one Linear.** Two rank-limited terms are
   interpretable — `lon_to_anchor`'s contribution can be ablated to zero and measured, which the
   single graft never allowed.
3. **Zero-init `lon_to_anchor`** (ReZero discipline, as `ctx_to_cond` already does) so the milestone
   starts byte-identical in the selection path to the current model and the graft's effect is
   attributable. `lat_to_anchor` inherits default Linear init to preserve today's behaviour.
4. **Norm parity must be monitored** (`V3_GOAL_VOCABULARY_V1.md` §Anti-shortcut, the H26 swamping bug
   class): log `||lat_to_anchor(·)||` vs `||lon_to_anchor(·)||` vs `||conf||` every log step. A graft
   that swamps `conf` is not a prior, it is a second selector.

### 2.3 Distance, which is the actual instruction

A class name is a shape; `stop_at_point in 12 m` is an instruction. `--labels v3` already mints
`route_dist_idx` (distance to the next route maneuver) and `lon.stop_dist_m` / `stop_dist_band`
(distance to the stop point), in **metres**, banded by `vocab.routedist_band`.

PROPOSED (a third small head, or a 2-slot extension of the above): an 8-way CE over
`DIST_BAND_TOKENS`. Metres, not seconds — the same junction is "5 s" at 10 m/s and "2.5 s" at
20 m/s, two tokens for one instruction; a deceleration profile is set by distance (`v² = 2ad`); and
maps/nav stacks speak metres, so a later map or VLM fills the identical slot with no unit conversion
that would need the label-time speed.

`d_none` ("looked over ≥100 m of road, nothing there") and `d_unknown` ("the window did not reach far
enough to say") are **separate tokens on purpose** — this is the v2→v2.1 lesson (a silent fallback
that conflated "cannot judge" with a real class poisoned the route prior) applied to the distance
axis. `d_unknown` must be masked out of the CE exactly like `ROUTE_UNKNOWN`.

---

## 3. Count pins (keep the table↔code check working)

`vocab.py` pins each frozen slot's `n` so a typo fails loudly. The new heads must be sized off the
same tables, never off a literal:

| symbol | value | pinned against |
|---|---|---|
| `vocab.TACTICAL_TOKENS["LATMANEUVER"]` | 9 | `_EXPECTED_N` (frozen) |
| `vocab.TACTICAL_TOKENS["LONMODE"]` | 9 | `_EXPECTED_N` (frozen) |
| `vocab.TACTICAL_TOKENS["TACPOINT"]` | 5 | `_EXPECTED_N` (frozen) |
| `refb_labels.LAT_KINEMATIC_TOKENS` ∪ `LAT_CONTEXT_TOKENS` | = the 9 LATMANEUVER tokens | `tests/test_refb_labels_v3.py` |
| `refb_labels.LON_KINEMATIC_TOKENS` ∪ `LON_LEAD_TOKENS` | = the 9 LONMODE tokens | same |
| `refb_labels.ROUTE_V3_TOKENS` | = the 9 ROUTE tokens, exactly | same |
| `refb_labels.DIST_BAND_TOKENS` == `vocab.ROUTEDIST_TOKENS` | 8 | same |
| `N_MANEUVERS` | 5 | `tests/test_refc.py` — **leave it**; it is the old head's width |

`ROUTEDIST` / `TACDIST` live in `vocab.V11_CANDIDATE_TOKENS` and are **NOT enrolled** in
`STRATEGIC_TOKENS` / `TACTICAL_TOKENS`. Enrolling them changes `GOAL_SLOTS`, `empty_goal()`,
`validate_goal()` and every embedding sized off them — that is a vocabulary version bump (v1 → v1.1)
with a migration note, and `V3_GOAL_VOCABULARY_V1.md` is FROZEN. **That enrolment is Sayed's
decision**, and it is the only vocabulary change this work implies.

---

## 4. What this costs

- One REF-C retrain at the chosen scale on the canonical corpus (parity key
  `physicalai-train-e438721ae894`, 2376 episodes). No data rebuild: the labels are derived on the fly
  by the dataset, exactly as `--labels v21` is.
- Parameter delta: two `Linear(aux_hidden, ≤8)` heads + two `Linear(≤8, N_anchors, bias=False)`
  grafts. At `N_anchors = 256` that is ≈ 2·8·256 + small ≈ **~5 k parameters**. Negligible against
  104 M (base) / 252 M (XL); this is not a capacity change, it is a **structure** change.
- The read: `frac_sel_2x_worse` (0.4109 base / 0.4540 XL) and the longitudinal share of that gap.
  The claim to test is that the ~98 %-longitudinal composition of the selection gap shrinks once a
  longitudinal prior can reach the ranking at all.

## 5. What this is NOT

- **Not a fix for `stop_line`.** Kinematics mints *where* the vehicle stops, never *why*. Pedestrian
  crossing vs stop line vs a queue behind a lead vehicle are the same ego track. `TACPOINT` stays
  `unknown` (provenance `unknown`) until the VLM/map pass fills it. `tactical_from_future_v3` returns
  the distance and refuses the name — deliberately.
- **Not a fix for `follow_lead` / `close_gap` / `open_gap`.** `lead_state` is a `None` stub. These
  need a detector/monodepth or VLM lead-state pass. They are excluded from the head.
- **Not a claim that the route label caused the ep09 selection.** At eval `nav_cmd=None → follow`
  (`refc.py:786-787`, `refc_eval.py:78`, `plan_fan.py:549`), so the strategic command is a constant
  for all 881 windows and the HUD's "strategic: route left" is the model's own `route_logits.argmax`
  (`plan_fan.py:561`), not an input. The labels are causal through **training** and through
  `graft_maneuver`, not through eval-time conditioning.

## 6. Caveat on the measured label-side numbers

The 24.9 % / 44.1 % rows in §1 are MEASURED on the **100-episode val build**
`physicalai-val-bb543bdf7836` (the only PhysicalAI val epcache on the dev box). The canonical
40-episode build `physicalai-val-0c5f7dac3b11` lives only on the eval pod. The two are **not
interchangeable** and the difference is measured, not assumed:

| | local 100-ep | canonical 40-ep |
|---|---|---|
| median `v0` | 5.62 m/s | 10.29 m/s |
| fraction `v0 < 1 m/s` | 19.0 % | 5.7 % |
| fraction \|Δheading@2s\| > 8° | 35.6 % | 22.1 % |
| 5-way `lane_keep` share | 33.3 % | 61.7 % |

The local build is substantially more urban and low-speed, so its longitudinal-mode rates are an
**upper** estimate for the canonical corpus. The canonical-substrate figures (27.5 % present /
7.2 % destroyed) come from the fan dumps and carry their own reconstruction caveat. Re-running
`scripts/label_v3_audit.py --val <canonical epcache>` on the eval pod closes this gap in minutes and
is the single highest-value follow-up.
