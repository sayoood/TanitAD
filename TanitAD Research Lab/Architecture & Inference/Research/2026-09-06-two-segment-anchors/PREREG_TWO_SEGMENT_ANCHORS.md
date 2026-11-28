# PREREG — D-TWOSEG: a two-segment anchor family, default OFF

**Author:** Arch+Inference, two-segment anchor agent · **Date:** 2026-09-06
**Status:** PRE-REGISTERED — written BEFORE any number below existed.
**Registers as:** `D-TWOSEG-1` … `D-TWOSEG-7` in `Project Steering/GOALS_AND_CLAIMS.md`.

⛔ **This document is written before the builder runs.** Both outcomes are committed for
every bar. A bar that is missed is reported **FAILED, as written** (RULE ZERO: the failure
is not the end of the turn — the next lever is named and run).

⛔ **PI CONSTRAINT HONOURED:** *"don't generate labels again."* Nothing here rebuilds,
regenerates or re-emits any label set. The anchor vocabulary is a **candidate set**, and the
target it is scored against is **geometric** — see §1.

---

## 1. Why this lever needs no label change — the mechanism, from source

**MEASURED, read from `stack/scripts/refc_v3_train.py` (cite by marker, not line — this
trainer moved +291 lines today):** the anchor-classification target is

```
anchors = out["anchor_bank"]                       # [B, N, S, 2], the EMITTED fan
dist    = (((traj_tgt[:, None] - anchors) ** 2).sum(-1) * sv[:, None]).sum(-1)
a_star  = dist.argmin(dim=1)
loss_cls = F.cross_entropy(out["anchor_logits"], a_star)
```

`traj_tgt` is `refb_labels.waypoint_targets(pose_last, fut_ext, horizons)` — the **recorded
future ego path**. ⇒ **`a_star` is a pure geometric argmin over the emitted fan and reads no
tactical label at all.** Adding a candidate therefore changes what the classifier is
supervised toward **without touching a label set**.

⚠️ **This does NOT contradict `ADDENDUM_TWO_SEGMENT.md`'s caution — it scopes it.** Two
different heads:

| head | target | can a lane change reach it today? |
|---|---|---|
| **tactical** `a_tac.lat` | the v7.2 label vocabulary | ⛔ **NO** — `LANE_CHANGE_L/R` and `ABORT_LC` are 0/4,572 emitted (D-SELQ-LC-VOCAB / commit `56dc078`). Fixing that IS a label change and is **forbidden this turn**. |
| **anchor** `anchor_logits` | `a_star`, geometric | ⭐ **YES** — no label involved. |

⇒ the extension is admissible **and** its reach is bounded to the anchor head. Any claim that
it fixes the tactical head is false and is not made here.

⛔ **`a_star` as computed in `taniteval/tools/refcv3_arm.py` is INVALID** (it binds against
`decoder.anchors`, the bank rolled once at `ref_speed_ms`). **Nothing here is scored with
it.** Every binding below is the **per-window v0-conditioned roll**, the same one
`selstab.py` reconstructs model-free and `prove_not_astar()` demonstrates differs.

---

## 2. The extension, defined before any number

A **two-segment** candidate holds `+a_lat` for `t_split` seconds and `−a_lat` for the rest of
the 6 s horizon, with `a_lon` constant throughout. Everything else is the incumbent's:
the same integrator (`tanitad.models.kinematic.rollout_unicycle`), the same
`kappa = clamp(a_lat / max(v0, alat_v_floor)^2, ±kappa_cap)` map, the same
`alat_v_floor = 4.0`, `kappa_cap = 0.12`, `dt = 0.1`, and the same 8 slots
`[5,10,15,20,30,40,50,60]`.

**Parameterisation — a THIRD control column.** `controls` becomes `[N, 3]`:

| col | name | units |
|---|---|---|
| 0 | `a_lon_ms2` | m/s² |
| 1 | `a_lat_ms2` | m/s² |
| 2 | `t_split_s` | **s**, the time the lateral sign flips |

⭐ **The incumbent is the `t_split_s = horizon_s` special case.** With `t_split_s = 6.0` the
sign is `+1` at every tick of a 6 s roll, so the two-segment integrator **is** the constant
integrator. That is what makes the OFF path provable rather than asserted, and it is what
makes the degenerate-split regression arm exact.

**Count and split grid, justified from the MEASUREMENT, not from taste**
(`.../2026-09-06-selection-quality/raw/out_p1h_two_segment.json`, commit `56dc078`;
5,000 windows, all 1,297 strict lane-change windows kept, rest subsampled seed 0):

| family | n new | LC supply gain |
|---|---|---|
| 8 `a_lat` × 3 splits | 24 | 0.1654 |
| 4 `a_lat` × 3 splits | 12 | 0.1651 |
| **2 `a_lat` (±0.75) × 3 splits (2/3/4 s)** | **6** | **0.1645** |
| 8 `a_lat` × **1** split (3 s) | 8 | 0.0304 |
| 2 `a_lat` × 1 split | 2 | 0.0304 |

⇒ **6 candidates**: 24 buys **+0.0009 m** more for **+18 candidates**, and a single split
point loses **81.5 %** of the gain even with all eight magnitudes. **The split point is the
lever; the magnitude is not.** Bank cost **6/117 = +5.13 %**.

⭐ `a_lon = 0` on all six: the addendum's `a_lon`-varying rows are **not supersets**
(`a_lon_grid[::4]` skips 0.0) and are not read.

---

## 3. The bars — both outcomes committed

### B1 — DEFAULT-OFF BIT-IDENTITY (the parity bar). ⛔ Hard gate.
The builder with `--two-segment` absent must emit a bank whose `anchors` and `controls`
tensors are **sha256-identical** to refcv4b's live bank.
**Reference (two independent sources):** `restamp_refcv4b_anchors.py`'s recorded constants —
`anchors 51f930dc…a66df`, `controls b072f4c0…89664` — and the **live checkpoint**
`C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt` (`core.decoder.anchors`,
`core.decoder.anchor_controls`).
* **PASS** both sha256 equal, and the OFF artifact's `controls` is `[117, 2]` (not `[117, 3]`).
* **FAIL** any difference ⇒ the extension is withdrawn until it is default-inert.

### B2 — SINGLE-SEGMENT LIMIT (the integrator bar). ⛔ Hard gate.
Rolling the 117 incumbent controls through the **two-segment** integrator with
`t_split_s = horizon_s` must reproduce the incumbent roll **bit-identically** (`torch.equal`),
over ≥ 8 speeds spanning 0–36 m/s.
* **PASS** exact equality. * **FAIL** any bit differs ⇒ the two integrators are not one.

### B3 — SUPPLY (the capability bar). Pre-registered threshold.
On the **6 s** grid, best-in-fan ADE against the recorded ego path, on the 141-clip eval
corpus at stride 1:
* **PASS** the 123-candidate bank improves the **strict lane-change** supply ceiling by
  **≥ 0.10 m** AND moves the **TURN control** by **≤ 0.01 m** (either direction).
* **FAIL** gain < 0.10 m, or the turn control moves > 0.01 m (which would mean the scorer,
  not the geometry, is doing the work).

### B4 — DELIBERATE REGRESSION. ⛔ If the gate cannot FAIL these, a PASS means nothing.
Three knowingly-broken variants, each with its own committed expectation:
* **R1 `t_split_s = 6.0` (never splits)** — the six become duplicates of existing constant
  arcs. **Expect LC gain = 0.0000 EXACTLY** (a structural zero, not a small number).
* **R2 `t_split_s = 0.0` (splits at t=0)** — the six become constant arcs of the opposite
  sign, already in the bank. **Expect LC gain = 0.0000 EXACTLY.**
* **R3 single split at 3.0 s only (2 candidates)** — must reproduce the banked **0.0304**
  to ±0.005, i.e. the gate must GRADE, not just fire.
* **FAIL** if R1 or R2 reads a non-zero gain (the scorer is crediting the wrong thing), or if
  R3 does not reproduce.

### B5 — STRUCTURAL ZERO ON THE 2 s GRID. Control at a KNOWN value.
Every split is at `t ≥ 2.0 s`, so no two-segment candidate can differ from its constant
counterpart at slots 0–3. On the **2 s** grid the extension must change every one of the four
families by **EXACTLY 0.000000**.
* **PASS** exact zeros. * **FAIL** any non-zero ⇒ the roll is wrong somewhere.
⛔ **This is also the honest scope statement: a capability gained here CANNOT appear in
`ade_0_2s`.** Reported as a limitation, not buried.

### B6 — KINEMATIC ADMISSIBILITY + VACUITY GATE.
* Every new candidate's peak `|a| = sqrt(a_lon² + a_lat²)` must sit **inside** the incumbent
  bank's envelope, and its derived `|kappa|` must never reach `kappa_cap` (a clamped
  candidate is a different candidate than the one declared).
* ⛔ **The manoeuvre rate is reported beside the feasibility number** — a zero-violation
  result bought by declining the manoeuvre is not a safety result (MEASURED precedent:
  a `turn_left` recall of exactly 0.0000).
* **PASS** 0 violations of μ = 0.7 AND a non-zero lane-change supply rate. **FAIL** otherwise.

### B7 — SUPERVISION RATE (the "does it choose one" bar).
On the extended bank, the fraction of **strict lane-change windows** whose geometric
`a_star` lands on a two-segment candidate.
* **PASS** ≥ 20 % of lane-change windows, with the **all-window** rate reported beside it
  (a family that captured most windows would be a vocabulary takeover, not a fix).
* **FAIL** < 20 % ⇒ the candidates exist but the training signal rarely points at them.

⛔ **B7 measures SUPERVISION, not EXECUTION.** refcv4b has 117 logits and **cannot rank a
118th candidate**; a trained execution rate requires retraining and is a **named blocker**
(§5), never inferred from B7.

---

## 4. Estimator and evidence discipline

* Every fraction carries its **n** and its **corpus** (⚠️ a percentage alone is inadmissible).
* **Four families, never pooled, never ADE alone.** LATERAL is read on **curvature MAE with
  the straight-line floor beside it** — the re-rolled floors are `ha0` **0.006841**,
  `os` **0.008024**.
* ⛔ **The variance named.** Every number here is **model-free and deterministic** given the
  banked dump and the recorded poses: no training draw, no inference sampling, no episode
  resampling. ⇒ **the only variance is the window subsample**, and it is removed by scoring
  **every** scoreable window rather than the addendum's 5,000. **No CI is quoted**, because
  none of these are estimates of a population — they are the exact value on this corpus.
  (Rig replicate false-positive rate **6/42 = 14.3 %** does not apply: no arm is trained.)
* ⛔ **A SUPPLY ceiling may never be quoted beside an achievement**
  (`GATE_SPEC_MODEL_FREE_VS_INCLUSIVE.md`).
* ⚠️ **Attribution guard.** Output smoothing α = 0.25 improved curvature 0.008024 → 0.007391,
  and that credit belongs to **undoing the decoder's refinement**, not to selection; a sibling
  measured the refinement at **−30.7 % ADE / ×2.03 curvature**. **No refinement effect is
  attributed to these anchors**, and nothing here touches the refinement path.

---

## 5. What is BLOCKED, named in advance

1. ⛔ **Executed lane-change rate of a trained model.** Needs a retrain with the 123-candidate
   bank. **Unblocked by:** a tiny-rig v7 arm (~17 min on Thor) or an A40 slot — and the A40 is
   **reserved for `refcv5-cap-b1-v72-40k` until 2026-09-08 07:33 UTC** and must not be taken.
2. ⛔ **The tactical head's three dead lane-change classes.** Needs the v7.2 emitter fixed and
   the labels rebuilt — **forbidden by the PI this turn.** Reported as blocked on a PI decision.
3. ⛔ **The consumer.** `refc.py::AnchorDecoder.roll_bank` rolls `anchor_controls[N, 2]` as a
   constant and `anchor_control_seq` expands to `[B, N, S, 2]`; a `[N, 3]` bank is **refused**
   by the shape checks (loudly, which is correct). `refc.py` is owned by a sibling this turn,
   so the decoder-side read of column 2 is a **named hand-off**, delivered as a
   drop-in-ready reference implementation + test, not an edit.

---

## 6. Scope of the claim, stated before the numbers

⛔ **The dominant selection defect is LONGITUDINAL.** MEASURED (commit `a3b232b`):
**4,350 / 24,114** windows carry the wrong longitudinal manoeuvre while the fan held a correct
candidate, and an `a_lon` oracle recovers **74.2 %** of the regret against `a_lat`'s **20.9 %**.
⇒ **A lateral vocabulary fix can touch at most the 20.9 % lateral share**, and within that only
the lane-change sub-population. **This is not oversold as a fix for the selection defect.**
