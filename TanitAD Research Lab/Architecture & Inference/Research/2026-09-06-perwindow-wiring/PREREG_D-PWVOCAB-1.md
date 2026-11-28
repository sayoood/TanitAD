# PREREG D-PWVOCAB-1 — does a finer per-window tactical vocabulary help?

**Written 2026-09-06, BEFORE any arm exists. Status: REGISTERED, NOT RUN.**
**Blocked on the `v6.py` / `refc_v3.py` escalation in `PERWINDOW_WIRING.md` §4.2.**
Owner: per-window wiring stream. Rig: **v7-tiny ladder** (~19 M params, parity
corpus, ~17 min/arm on Thor). ⛔ Never on a full-scale run.

---

## 1. The claim, scoped honestly

⭐ **CLAIM.** Supervising the `z_tac` tactical decision heads with the **v4
per-window kinematic vocabulary (5 lateral x 6 longitudinal)** instead of the
runtime **kin3 (3x3)** improves tactical decision quality **on the same windows,
with no new supervision and no new data.**

⛔ **WHAT THIS IS NOT.** It is **not** new supervision: `loss_lat`/`loss_lon`
already train on **100 %** of windows today. It is a **vocabulary-resolution**
change on labels already computed. ⛔ **It does not address the 88.13 % v7-band
gap and must never be reported as doing so** — the v7 22/15-token sets are
**disjoint** from this kinematic line. ⛔ It does not touch the **core** aux
heads, which are structurally kin3 in `refc.py`.

**Why it is worth a rig slot at all — MEASURED, `PERWINDOW_WIRING.md` §3,
n = 68,377 windows / 400 episodes:** kin3 cannot express the v4 identity of
**16.70 %** of windows laterally and **22.64 %** longitudinally; and **92.4 %**
of what kin3 calls a "turn" is v4 `lane_keep`.

---

## 2. Arms

| arm | vocabulary on `z_tac` | purpose |
|---|---|---|
| **A0** | kin3 3x3 (today) | control |
| **A0b** | kin3 3x3, **identical flags, re-run** | ⛔ **the noise floor.** Mandatory. |
| **A1** | `kin76` 5x6 | the lever |
| **A2** ⛔ | `kin76` with **targets shuffled within each batch** | **deliberate regression** |
| **A3** | `kin76` with **all targets set to the majority token** | constant-target floor |

⛔ **A2 and A3 are gates on the INSTRUMENT, not arms of interest.** If the
protocol cannot separate A2 or A3 from A1, **a PASS on A1 means nothing and is
not reported as one.**

---

## 3. Head sizing and masking — fixed in advance

⛔ **The lateral head is FIVE units, not seven.** MEASURED: `abort_lc` **4 /
68,377**, `pull_over` **0 / 68,377**. Both are emitted as **`IGNORE_INDEX`**;
neither gets a head unit. A 7-wide head would carry two units that can never
receive a gradient, and its macro-recall denominator would be a lie.
⛔ **If the parity corpus ever becomes reachable and either token clears
n >= 200, this decision is RE-OPENED and re-registered — not silently widened.**

Longitudinal is **six** units; all six are populated (min `launch` 1,308).

---

## 4. ⛔ The bars — BOTH OUTCOMES COMMITTED IN ADVANCE

**Primary (TACTICAL family).** Macro-averaged per-class recall of the `z_tac`
lateral head, **mapped down to the kin3 partition** so A0 and A1 are scored on
**one common label set** (`lane_keep` vs `turn_left` vs `turn_right`), plus the
`kin76`-native macro recall reported for A1 alone.

* ⭐ **PASS** if A1's kin3-projected **macro recall** exceeds A0's by a margin
  that **also exceeds the A0-vs-A0b replicate spread on the same cell**, and
  A1 - A2 and A1 - A3 are both separated.
* ⛔ **FAIL** if A1 <= A0 on kin3-projected macro recall, **or** if the
  A1 - A0 difference does not exceed the A0/A0b replicate spread, **or** if A2
  or A3 is not separated from A1.
* ⛔ **A FAIL IS REPORTED AS A FAIL AND THE STREAM CONTINUES** (Rule Zero). The
  named next lever on a FAIL is: **the LON axis alone** (`coast` vs
  `stop_at_point` is the single largest collision, 15,160 windows, and 88.7 %
  of the oracle gap is longitudinal), run as its own arm before the lateral
  vocabulary is abandoned.

**Secondary, reported always, never pooled into the primary.**

## 5. ⛔ Four metric families — never pooled

| family | metric | floor beside it |
|---|---|---|
| **LONGITUDINAL** | target-speed accuracy **and** distance-keeping (headway / time-gap / TTC to the lead agent) | constant-speed floor; **state `n` and the reason per family where no lead agent is in frame** |
| **LATERAL** | ⛔ **curvature MAE** (with heading error, yaw-rate error, cross-track) | ⛔ **the straight-line floor beside it**, per the doctrine. Reference: `ha0` curvature **0.006802** vs `os` **0.008097** |
| **TACTICAL** | ⛔ **per-class recall, NEVER pooled accuracy** — selected vs executed manoeuvre, full confusion matrix | ⛔ **majority-class control, which MUST read its known value: pooled 0.8330 / macro 0.1667 = 1/6** (MEASURED on labels, §2 of the report) |
| **STRATEGIC** | `g_str` / route-token quality | ⚠️ **`g_str` and `route_target` are NOT independent** — the LAN strategic label is built on `route_from_future_v21`, so both share one kinematic source. **This must be stated wherever they are reported together.** |

⛔ **ADE is reported but is one row of four, never "the result".**
Reference floors (refcv4b): `os` **0.2975** (A40 landing roll; the Thor re-roll
0.2965 is **not** quotable) · `ha0_ext` **0.2874** · `ha` **0.2996** ·
`ha0` **0.6723**.

## 6. ⛔ The variance the interval answers — named in advance

The estimator is the **paired episode-cluster bootstrap** (`taniteval/ci.py`)
over the val episodes. It answers **"would another draw of EPISODES say this?"**
and **nothing else**.

* ⛔ It is **blind to training variance.** MEASURED on this rig: a zero-lever
  replicate produced "separated" differences on **6 of 42** family cells — a
  **14.3 % false-positive rate for `separated`**. ⇒ **A0b is mandatory and the
  A1-A0 difference is read against the A0/A0b spread, not against zero.**
* ⛔ It is blind to **inference** variance where a planner samples. This arm's
  tactical heads are argmax reads, so inference variance is not the binding
  term here — **but any ADE quoted from a sampling planner carries the
  ~0.30 m seed floor** and an effect below it is not an effect.
* ⚠️ **A tiny-rig PASS is entry to the composed refcv5 arm, NOT a published
  result.**

## 7. ⛔ Vacuity gate

A recall improvement on `lc_left`/`lc_right` means nothing if the arm never
emits a lane change. ⇒ **Report the predicted manoeuvre RATE beside every
recall**, per class. MEASURED precedent: one zero-violation result was bought
by a `turn_left` recall of exactly **0.0000**.

## 8. Reachability — the test that must exist before the arm runs

⛔ **Correctness and wiring are different claims.** `refc_v3_train.main` runs
`preflight` only under `--preflight` and otherwise calls `train()` directly, so
a check that lives in `preflight` is **not** on the training path.
⇒ The wiring lands with a test that **asserts the new path is REACHED** during
a real `train()` step (not inspected, not AST-censused) — and the test must be
shown to **FAIL when the flag is off**, per the mutation rule that an
inspection-only guard reads 0 suspects on both the fixed and the broken trainer.

## 9. Bit-identity when OFF

⛔ **Default OFF, and OFF must be bitwise identical to today.** Proven by
**comparison over the full tensor set of a fixed-seed step, not by assertion**
(precedent in this repo: 341 tensors, 0 bitwise differences). ⛔ The effective
state — flag, vocabulary version, head widths, **and `man5_active`** — is
stamped into `config.json`; the `man5` coupling in `refc_v3.py:1189` silently
drops the H19 prior for any non-kin3 vocabulary and **must not become a second
unstamped path**.

## 10. Compute

Thor or the dev-box 4060, **serialised**, `OMP_NUM_THREADS=6`.
⛔ **The A40 is reserved for the composed refcv5 arm and is not taken.**
