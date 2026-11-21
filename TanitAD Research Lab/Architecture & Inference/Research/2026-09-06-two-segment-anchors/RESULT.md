# RESULT — D-TWOSEG: the anchor vocabulary can now express a lane change

**Author:** Arch+Inference, two-segment anchor agent · **Date:** 2026-09-06
**Pre-registration:** `PREREG_TWO_SEGMENT_ANCHORS.md` (this directory), written
**before** every number below. **All seven bars are reported AS WRITTEN.**

| bar | verdict |
|---|---|
| **B1** default-OFF bit-identity | ⭐ **PASS** |
| **B2** single-segment limit | ⭐ **PASS** |
| **B3** supply (≥ 0.10 m LC, turn ≤ 0.01 m) | ⭐ **PASS** |
| **B4** deliberate regression (R1/R2 exact 0, R3 reproduces) | ⭐ **PASS** |
| **B5** 2 s structural zero | ⭐ **PASS** |
| **B6** Kamm + vacuity gate | ⭐ **PASS** |
| **B7** supervision rate ≥ 20 % | ⭐ **PASS** |

⛔ **A tiny-rig / model-free PASS is entry to a composed arm, not a published
result** (plan §2.1). ⛔ **Everything here is a SUPPLY ceiling and may never be
quoted beside an achievement** (`GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`).

---

## 0. The one-line answer

⭐ **The model can now EXPRESS a lane change: MEASURED, candidates 117-122 are the
only yaw-non-monotone paths in the vocabulary and they are all of it** (base
1,053 / 1,053 pairs monotone = 100.0000 %; extended 1,053 / 1,107 = 95.1219 %,
the 54 non-monotone pairs being exactly the 6 new candidates × 9 speeds).
⛔ **It does NOT yet CHOOSE one, and it structurally cannot**: refcv4b has 117
`anchor_logits` and cannot rank a 118th candidate. What the extension changes
**today** is the *supervision*: the geometric `a_star` lands on a two-segment
candidate on **506 of 1,297 (39.01 %)** lane-change windows, so a retrain would
receive gradient toward a lane-change shape on those windows — **no label change
required**.

---

## 1. Corpus, n, and estimator — stated once, before the tables

* **Corpus:** the **141 B1-v7.2 EVAL clips**, refcv4b T1 stride-1 dump
  (`C:/Users/Admin/tanitad-selq-20260906/out/refcv4b_t1_s1_dump`, 24,114 windows).
* **n = 18,615** windows carry a full 6.0 s recorded future and are scoreable;
  of these **1,297 strict lane-change**, **2,602 turn control**, **14,716 other**.
  `v0` spans **0.00 – 36.27 m/s**.
* ⛔ **VARIANCE, NAMED.** Every number is **model-free and deterministic** given
  the banked dump and the recorded poses: **no training draw, no inference
  sampling, no episode resampling.** The addendum's only variance was its
  5,000-window subsample; this scores **every** scoreable window, which removes
  it rather than estimating around it. ⇒ **No CI is quoted, because none of these
  is an estimate of a population** — each is the exact value on this corpus. The
  rig replicate false-positive rate (**6/42 = 14.3 %**) does not apply: no arm is
  trained here.
* ⛔ **Not `a_star` as `taniteval/tools/refcv3_arm.py` computes it** — that binds
  against `decoder.anchors` (the bank rolled once at `ref_speed_ms`) and scores
  **1.4491 m**, worse than the **0.4252 m** arm it bounds. Every binding here is
  the **per-window v0-conditioned roll**, which is the binding the TRAINER uses.

---

## 2. B1 + B2 — the OFF path is bit-identical, proved by comparison

⭐ **The integrator is demonstrably the incumbent's, not a second implementation
that happens to agree.** Rolling refcv4b's own `core.decoder.anchor_controls` at
`ref_speed_ms = 10.0` through `anchor_twoseg.roll_bank` reproduces its stored
`core.decoder.anchors` **sha256 `51f930dc…a66df`** — the value independently
recorded in `stack/scripts/restamp_refcv4b_anchors.py` from the pod-side
`anchors.pt`. Two sources, one hash. `raw/v0_repro.py`.

| check | result |
|---|---|
| checkpoint `anchors` sha256 | `51f930dc6f3564ff8f21c9070ca97b805d4e82908e42d0ef1f99be9f5a3a66df` **VERIFIED** |
| checkpoint `controls` sha256 | `b072f4c052331beb79bef117c7b233f702802b517429cc9157a841294e089664` **VERIFIED** |
| `roll_bank(controls, v = 10.0)` == stored `anchors` | **VERIFIED**, same sha |
| **B1** builder without `--two-segment`, written tensors | **BIT-IDENTICAL**, both sha256 |
| **B1** OFF artifact shape / schedule | `controls [117, 2]`, `control_schedule = constant` |
| **B2** `t_split = 6.0` vs the constant roll, 9 speeds 0–36 m/s | **BIT-IDENTICAL** (`torch.equal`, 16,848 floats) |
| **B2** same, over the full corpus | **BIT-IDENTICAL** (`34,847,280` floats) |
| **B2 MUTATION CONTROL** `t_split = 3.0` | **DIFFERS**, max │Δ│ **92.58 m** — the check can fail |

⭐ **Why it is bit-identical rather than nearly so:** the incumbent is the
`t_split_s = horizon_s` special case. The sign schedule is `+1` at every tick, so
the curvature is multiplied by exactly `1.0`. There is no separate "off" branch
to drift.

⛔ **Negative controls on the builder, all REFUSED as designed:** `--assert-parity`
together with `--two-segment`; a wrong source sha (and a *short* sha →
`INCONCLUSIVE`, never `MATCH`); extending an already-extended bank; a
`t_split` that does not land on a `dt = 0.1` tick.

---

## 3. B3 — supply: what the six candidates buy

Best-in-fan ADE over the eight slots against the recorded 6 s ego path, m.

| arm | N | ALL | lane-change | turn (CONTROL) |
|---|---|---|---|---|
| **base117** (incumbent) | 117 | 1.2943 | **1.5315** | **1.9476** |
| **ext123** (+6 two-segment) | 123 | **1.2425** | **1.3670** | **1.9412** |
| **gain** | +6 | **+0.0519** | **+0.1645** | **+0.0064** |
| R1 `t_split = 6.0` (never flips) | 123 | 1.2943 | 1.5315 | 1.9476 |
| R2 `t_split = 0.0` (flips at t₀) | 123 | 1.2943 | 1.5315 | 1.9476 |
| R3 single split 3.0 s | 119 | 1.2628 | 1.5011 | 1.9427 |
| **C** straight family (`a_lat = 0`) | 13 | 2.3702 | 1.5750 | 7.3596 |
| **C** one constant-velocity line | 1 | 3.8387 | 2.5151 | 9.0208 |
| **C** zero path (no information) | 0 | 32.6312 | 36.0329 | 23.2278 |

⭐ **B3 PASS.** LC gain **+0.1645 m ≥ 0.10 m**; the TURN control moves
**+0.0064 m ≤ 0.010 m**. **506** lane-change windows and **2,113** of all windows
improve. **10.74 %** of the lane-change supply ceiling (0.1645 / 1.5315).

⚠️ **The turn control matters more than it looks.** A constant-curvature arc is
*exactly* what a junction turn needs, so a two-segment family that "won" there
would mean the scorer, not the geometry, was doing the work. It moves by
**0.33 %** — nothing.

⚠️ **The straight-family control reads the RIGHT way round.** Restricting the fan
to its 13 `a_lat = 0` candidates costs only **0.0434 m** on lane-change windows
(1.5750 vs 1.5315) but **5.41 m** on turns (7.3596 vs 1.9476) — i.e. the
incumbent's whole lateral vocabulary is worth almost nothing on a lane change and
everything on a turn, which is exactly the gap this family fills. A control that
had *beaten* the full fan anywhere would have meant the scorer was broken.

---

## 4. B4 — the deliberate regression, and it is an EXACT zero

⛔ **If the gate cannot FAIL a knowingly-broken version, a PASS means nothing.**

| arm | LC gain | windows improved (LC / all) | expected | verdict |
|---|---|---|---|---|
| **R1** split at the horizon | **+0.0000000000** | **0 / 0** | exactly 0 | ⭐ **PASS** |
| **R2** split at t = 0 | **+0.0000000000** | **0 / 0** | exactly 0 | ⭐ **PASS** |
| **R3** single split at 3.0 s | **+0.0303943680** | 160 / 1,189 | 0.0304 ± 0.005 | ⭐ **PASS** |
| ext123 (the real family) | **+0.1645270604** | **506 / 2,113** | ≥ 0.10 | ⭐ **PASS** |

⭐ R1 and R2 are **structural** zeros, not small numbers: a split at the horizon
never flips, and a split at t₀ makes the candidate a constant arc of the opposite
sign — both already in the bank. ⭐ R3 **reproduces the independently banked
0.0304** (`.../2026-09-06-selection-quality/raw/out_p1h_two_segment.json`,
commit `56dc078`) to ten decimal places, so the gate **grades** rather than
merely firing.

---

## 5. B5 — the 2 s grid is a structural zero, and that is the SCOPE limit

Every split in the shipped family is at **t ≥ 2.0 s**, and slot 3 **is** 2.0 s.

| quantity, 2 s grid (slots 0–3) | value |
|---|---|
| max │ADE change│, base117 → ext123, over 18,615 windows | **0.0000000000e+00 m** |
| windows whose 2 s argmin is a NEW candidate | **0 / 18,615** |

⭐ **PASS** — and ⛔ **this is also the honest limitation, stated rather than
buried: a capability gained here CANNOT appear in `ade_0_2s`.** Any report that
looks for this lever in the programme's headline 2 s number will correctly find
nothing. The effect lives entirely in the 2–6 s band.

---

## 6. B6 — kinematic admissibility, with the manoeuvre rate beside it

| | new 6 candidates | incumbent 117 |
|---|---|---|
| peak total │a│ over 0–36 m/s | **0.0765 g** (0.75 m/s²) | 0.5097 g |
| (candidate, speed) pairs over μ = 0.7 | **0 / 54** | 0 |
| `kappa_cap` (0.12) reached | **never** | yes, at low speed |

⭐ **PASS.** The new candidates sit strictly inside the incumbent's envelope, and
because │a_lat│ = 0.75 ⇒ │κ│ ≤ 0.75/16 = 0.0469 they are **never clamped**, so
the candidate the file *declares* is the candidate the integrator *drives*.

⛔ **VACUITY GATE — the manoeuvre rate, beside the safety zero.** MEASURED
precedent: one zero-violation result was bought by a `turn_left` recall of exactly
0.0000. Here the family is **not** declining the manoeuvre: **506 / 1,297
(39.01 %)** of lane-change windows select a new candidate and **2,113 / 18,615**
of all windows improve. A zero-violation arm that never steers would have read
0 picks; this one reads 2,113.

⚠️ **And the clamp direction is the opposite of the intuitive one, which is why
it is asserted rather than assumed:** `κ = a_lat / max(v, 4)²` is LARGEST when the
floor binds, so the incumbent bank is clamped at LOW speed, not high. A declared
`a_lat = 3.0` realises **1.92 m/s²** at v = 4 (64 % of the column) and the full
3.0 at v = 36. Pinned in `tests/test_anchor_twoseg.py`.

---

## 7. B7 — the supervision rate: where the training signal now points

⛔ **This measures SUPERVISION, not EXECUTION**, and the two are not
interchangeable. The anchor classifier's target is
`a_star = dist.argmin(1)` over `out["anchor_bank"]` against
`refb_labels.waypoint_targets(...)` — **a pure geometric argmin of the recorded
ego path, reading no tactical label at all** (`refc_v3_train.py`, cited by code
marker; that file moved +291 lines today). ⇒ what fraction of windows would hand
the classifier a lane-change-shaped target?

| stratum | k / n | rate |
|---|---|---|
| **lane-change** | **506 / 1,297** | **39.01 %** |
| turn (control) | 57 / 2,602 | 2.19 % |
| all | 2,113 / 18,615 | 11.35 % |

⭐ **PASS** (≥ 20 %). ⭐ **And it is not a vocabulary takeover** — 11.35 % of all
windows, 2.19 % of turns. A family that had captured most windows would have been
a different failure.

**Per candidate (all windows / lane-change windows):**

| candidate | a_lat | t_split | all | lane-change |
|---|---|---|---|---|
| 117 | −0.75 | 2.0 s | 819 | **258** |
| 118 | +0.75 | 2.0 s | 570 | **245** |
| 119 | −0.75 | 3.0 s | 269 | 2 |
| 120 | +0.75 | 3.0 s | 259 | 1 |
| 121 | −0.75 | 4.0 s | 133 | 0 |
| 122 | +0.75 | 4.0 s | 63 | 0 |

⭐ **NEW FINDING — the lane-change work is done almost entirely by the EARLIEST
split.** 503 of the 506 lane-change picks are the `t_split = 2.0 s` pair; the
3.0 s and 4.0 s pairs contribute **3**. They earn their place on the *other*
14,716 windows (724 of the non-lane-change picks), which is why the all-window
gain rises from 0.0425 (2.0 s alone) to 0.0519 (all three).

⚠️ **This refines, and does not contradict, the addendum's "the split point is the
lever".** It is the *early* split point.

⭐ **Corollary that matters for the design: `t_split = 3.0 s` is the ONLY split
that produces a net-zero heading change at constant speed** (yaw ∝ κ·v·t, so the
`+` cancels the `−` iff `t_split = T/2`), and it is the one that does almost
nothing for lane changes. **A real lane change is executed on a CURVING road, so
the symmetric S-curve is the minority case.**

---

## 8. THE FOUR FAMILIES — never pooled, and there IS a trade

6 s grid, the oracle-in-vocabulary picked path vs GT. ⛔ LATERAL is read on
**curvature MAE with the straight-line floor beside it**.

**Lane-change windows (n = 1,297):**

| arm | ADE | along | cross | speed | **curv (1/m)** | heading (deg) |
|---|---|---|---|---|---|---|
| base117 | 1.5315 | 0.8285 | 1.1617 | 0.4322 | **0.002784** | 2.726 |
| **ext123** | **1.3670** | 0.8315 | **0.9533** | 0.4338 | **0.004095** | 3.082 |
| R1 (regression) | 1.5315 | 0.8285 | 1.1617 | 0.4322 | 0.002784 | 2.726 |
| **floor** straight family | 1.5750 | 0.8294 | 1.2196 | 0.4313 | **0.002726** | 2.773 |
| **floor** const-velocity | 2.5151 | 1.9152 | 1.2273 | 0.8408 | **0.002726** | 2.773 |

**All windows (n = 18,615):**

| arm | ADE | along | cross | speed | curv | heading |
|---|---|---|---|---|---|---|
| base117 | 1.2943 | 0.8922 | 0.7040 | 0.4739 | 0.005055 | 3.139 |
| **ext123** | **1.2425** | 0.8933 | **0.6365** | 0.4746 | 0.005317 | 3.164 |
| floor straight family | 2.3702 | 1.2755 | 1.7073 | 0.5772 | 0.007186 | 5.908 |
| floor const-velocity | 3.8387 | 2.6812 | 1.8613 | 1.0900 | 0.009716 | 5.908 |

**Turn control (n = 2,602):** base 1.9476 / 0.018196 curv → ext 1.9412 / 0.018399.

⛔ **THE TRADE, STATED PLAINLY: the extension buys CROSS-TRACK and pays in
CURVATURE and HEADING.** On lane-change windows cross-track improves
**−17.9 %** (1.1617 → 0.9533) and ADE **−10.7 %**, while curvature MAE worsens
**×1.47** (0.002784 → 0.004095) and heading **+0.36 deg**. LONGITUDINAL is
untouched (along +0.0030, speed +0.0016) — as it must be: `a_lon` is unchanged on
every new candidate.

⚠️ **Read the curvature number against its floor before calling it a defect of the
extension.** The straight-line floor on these windows is **0.002726**, and
base117 sits at **0.002784** — i.e. **+2.1 % over a plan that never steers.**
⛔ **The incumbent's good curvature score on lane-change windows is bought by
barely steering at all**, which is precisely the failure mode the LATERAL family
exists to expose ("an arm can win ADE while tracking the road worse than a plan
that never steers"). ext123 sits at +50 % over the floor because it actually
turns.

⚠️ **These are 6 s-grid curvature numbers and are NOT comparable to the 2 s
re-rolled arm floors `ha0` 0.006841 / `os` 0.008024.** Different grid, different
object. Do not cross-quote them.

⚠️ **NO REFINEMENT EFFECT IS ATTRIBUTED HERE.** Output smoothing α = 0.25 moved
curvature 0.008024 → 0.007391 and that credit belongs to undoing the decoder's
refinement (a sibling measures the refinement at −30.7 % ADE / ×2.03 curvature).
Nothing in this work touches the refinement path.

---

## 9. WHY the curvature is paid — the hypothesis was REFUTED, and that refutation named a strictly better lever

**Paired decomposition** (`raw/p3_curv_decomp.py`), splitting by whether the pick
changed. ⛔ **Control at a known value:** on the 791 lane-change windows whose
pick did NOT change, the two paths are the same object and every family reads
**EXACTLY** equal — asserted, and it holds.

| stratum | n | ADE b → e | curv b → e | cross b → e | head b → e |
|---|---|---|---|---|---|
| lane-change, **REPICKED** | **506** | 1.6248 → **1.2031** | 0.002588 → 0.005949 | 1.3929 → **0.8587** | 2.716 → 3.628 |
| lane-change, pick unchanged | 791 | 1.4719 → 1.4719 | 0.002909 → 0.002909 | 1.0138 → 1.0138 | 2.733 → 2.733 |
| all, REPICKED | 2,113 | 1.7589 → **1.3021** | 0.003532 → 0.005773 | 1.4520 → **0.8572** | 2.888 → 3.110 |
| all, pick unchanged | 16,502 | 1.2348 → 1.2348 | (identical) | (identical) | (identical) |

⭐ **On the windows it actually changes, the extension improves ADE by 26.0 % and
cross-track by 38.4 %.**

**THE HYPOTHESIS, pre-stated: the instantaneous sign flip lands on a slot boundary
and the polyline reads a kink. ⛔ REFUTED.** Per-segment curvature error on the
506 repicked lane-change windows:

| segment | base | ext | Δ |
|---|---|---|---|
| 0.5–1.0 s | 0.002809 | 0.005018 | +0.002209 |
| 1.0–1.5 s | 0.003190 | 0.004604 | +0.001414 |
| 1.5–2.0 s | 0.003148 | 0.004768 | +0.001621 |
| **2.0–3.0 s (contains the flip)** | 0.002614 | 0.003948 | **+0.001334 ← the SMALLEST** |
| **3.0–4.0 s** | 0.002164 | 0.008605 | **+0.006441** |
| **4.0–5.0 s** | 0.001925 | 0.007595 | **+0.005670** |
| **5.0–6.0 s** | 0.002266 | 0.007101 | **+0.004835** |

⇒ **72.04 %** of the total per-segment degradation (0.016946 of 0.023524) sits in
the **3–6 s tail**, and the segment containing the flip is the **smallest**
contributor. ⭐ **The candidate is RIGHT IN SHAPE and WRONG IN DURATION: after the
flip it keeps counter-steering for 4 s while the human has finished and gone
straight.**

---

## 10. RULE ZERO — the next lever, RUN, not merely named

### 10.1 The split-point sweep (`raw/p2_split_sweep.py`)

Each split scored **in isolation** (+2 candidates), so they rank against each
other rather than as a bundle.

| t_split | LC gain | ALL gain | LC picks | 2 s repicks |
|---|---|---|---|---|
| 1.0 s | 0.1543 | 0.0390 | 651 | **1,258** ⛔ |
| 1.5 s | 0.0240 | 0.0071 | 252 | **720** ⛔ |
| **2.0 s** | **0.1640** | 0.0425 | 506 | **0** ⭐ |
| 2.5 s | 0.1254 | 0.0455 | 365 | 0 |
| 3.0 s | 0.0304 | 0.0315 | 160 | 0 |
| 4.0 s | 0.0021 | 0.0097 | 11 | 0 |
| 5.0 s | 0.0002 | 0.0019 | 5 | 0 |

⭐ **2.0 s is the maximum, not an edge-of-grid artefact** — and the 2.0 s pair
ALONE (+2 candidates) recovers **0.1640** of the shipped family's **0.1645**.
⚠️ A family including 1.0 s reaches **0.2106** LC / **0.0611** ALL, but **gives up
the 2 s structural zero** (1,389 windows repick, 2 s ADE moves 0.005868 m). ⛔ It
is therefore **not** swapped in post-hoc: the shipped default is the
pre-registered one, and the earlier splits remain available through the existing
`--t-split` flag with this trade recorded.

### 10.2 A THIRD segment, MEASURED (`raw/p4_three_segment.py`)

The refutation in §9 says the fault is the *duration*, so the lever is a schedule
that RETURNS TO STRAIGHT: `+a_lat` on [0, t₁), `−a_lat` on [t₁, t₂), **0** after.
Each family is **2 candidates**, the same cost as the 2-segment reference.

| family | LC gain | ALL gain | Δcurv on repicks | Δcross on repicks |
|---|---|---|---|---|
| **2-seg t = 2.0 (SHIPPED)** | 0.1640 | 0.0425 | +0.003359 | −0.5320 |
| **3-seg 2.0 → 3.0** | **0.1867** | **0.0592** | **+0.000864** | **−0.7088** |
| 3-seg 2.0 → 3.5 | 0.2161 | 0.0588 | +0.001639 | −0.7217 |
| 3-seg 2.0 → 4.0 | 0.2133 | 0.0547 | +0.002344 | −0.6438 |
| **3-seg 1.5 → 3.0** | **0.2266** | 0.0523 | +0.002163 | −0.5093 |
| 3-seg 2.5 → 4.0 | 0.0957 | 0.0460 | +0.000519 | −0.4754 |
| 3-seg 3.0 → 5.0 | 0.0265 | 0.0307 | +0.000661 | −0.2950 |

⭐⭐ **`3-seg 2.0 → 3.0` DOMINATES the shipped 2-segment pair on every family at
the same candidate cost**: more lane-change gain (0.1867 vs 0.1640), 39 % more
all-window gain (0.0592 vs 0.0425), **a quarter of the curvature cost**
(+0.000864 vs +0.003359), and better cross-track (−0.7088 vs −0.5320).
`1.5 → 3.0` gives the largest lane-change gain of all (**0.2266**, +38 %).

⛔ **NOT SHIPPED THIS TURN, and deliberately so.** It needs its own schedule (a
4-column `controls`, `three_segment_alat_pulse`) and its own pre-registration.
Swapping the default after seeing these numbers would be the post-hoc selection
the pre-registration exists to prevent. **PRE-REGISTERED BAR for that arm, written
now:** a 3-segment family PASSES if it beats the shipped 2-segment family on
lane-change supply **and** costs less curvature on its own repicks, with the turn
control still ≤ 0.01 m and R1/R2-style degenerate arms still reading exact zeros.

---

## 11. Scope — what this can and cannot touch

⛔ **The dominant selection defect is LONGITUDINAL, and a lateral vocabulary fix
does not address it.** *(INHERITED, commit `a3b232b`, not re-verified here:*
**4,350 / 24,114** *windows carry the wrong longitudinal manoeuvre while the fan
held a correct candidate, and an* `a_lon` *oracle recovers* **74.2 %** *of the
regret against* `a_lat`'s **20.9 %**.*)*

**MEASURED here, the supply-side bound:** the extension moves the all-window 6 s
supply ceiling by **0.0519 of 1.2943 m = 4.01 %**, and the lane-change ceiling by
**0.1645 of 1.5315 m = 10.74 %**, on the **6.97 %** of windows (1,297 / 18,615)
that execute a lane-change shape.

⇒ **A lateral vocabulary fix can touch at most the ~20.9 % lateral share of the
regret, and within that only the lane-change sub-population.** It is a real
capability the vocabulary did not have, not a fix for the selection defect, and
it is not presented as one.

⛔ **BLOCKED, each named with what would unblock it:**

1. **Executed lane-change rate of a trained model.** refcv4b has 117 logits and
   cannot rank a 118th candidate. **Unblocked by** a retrain — a tiny-rig v7 arm
   (~17 min on Thor) or an A40 slot. ⛔ **The A40 is reserved for
   `refcv5-cap-b1-v72-40k` until 2026-09-08 07:33 UTC and was not touched.**
2. **The tactical head's three dead lane-change classes.** `a_tac.lat ==
   LANE_CHANGE_L` is 0/4,572 train and 0/147 eval clips; `tactical_actions()`'s
   reachable set is exactly {LANE_KEEP, NUDGE_L, NUDGE_R, TURN_L, TURN_R}.
   Fixing it is a **label change** and is ⛔ **forbidden by the PI this turn**
   (*"don't generate labels again"*). **Blocked on a PI decision.**
   ⚠️ It is a **different head** from the one this work feeds, and the two must
   not be conflated: the anchor head's target is geometric, the tactical head's
   is the label vocabulary.
3. **The consumer.** `refc.py::AnchoredDiffusionDecoder.roll_bank` rolls
   `anchor_controls` as a constant and `anchor_control_seq` expands to
   `[B, N, S, 2]`; a `[N, 3]` bank is **refused** by those shape checks — loudly,
   which is correct. `refc.py` is a sibling's file this turn, so
   `anchor_twoseg.roll_bank` is delivered as a **drop-in reference + test**
   (`test_two_column_path_matches_the_live_decoder_roll_bank` pins it
   `torch.equal` against the live decoder) and the decoder-side read of column 2
   is a **named hand-off**, not an omission.

---

## 12. Deliverable manifest

| artifact | where it lives |
|---|---|
| `stack/tanitad/refs/anchor_twoseg.py` | repo (NEW) — the family, the integrator, the Kamm report |
| `stack/tanitad/refs/anchor_meta.py` | repo (EDITED, additive) — `control_schedule`, `AnchorScheduleMissing/Conflict` |
| `stack/scripts/build_twoseg_anchors.py` | repo (NEW) — the builder, `--two-segment` default OFF |
| `stack/tests/test_anchor_twoseg.py` | repo (NEW) — 19 tests, all green |
| `PREREG_TWO_SEGMENT_ANCHORS.md` | this directory |
| `RESULT.md` | this directory |
| `raw/v0_repro.py`, `raw/p1…p4*.py`, `raw/out_*.json` | this directory |
| `anchors_117_parity.pt`, `anchors_123_twoseg.pt` | `C:/Users/Admin/tanitad-twoseg-20260906/raw/` (**local disk only** — anchor `.pt` files are build outputs, reproducible from the builder + the checkpoint in one command) |

**Suite:** `tests/test_anchor_twoseg.py` **19 passed**; run together with the
anchor regression set (`test_anchor_meta`, `test_anchor_flyability`,
`test_withheld_bank`, `test_anchor_tactical`, `test_anchor_prefilter`,
`test_v6_anchor_loss`) the copied `stack/tests` reads **118 passed, 17 skipped**
with the edited `anchor_meta.py`.
