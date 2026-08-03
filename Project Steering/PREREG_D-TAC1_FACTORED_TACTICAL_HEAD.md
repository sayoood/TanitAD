# PRE-REGISTRATION — D-TAC1: factorising REF-C's manoeuvre head

> ## ⚠️ EXECUTION RECORD — appended 2026-08-03 AFTER the fact, WITHOUT altering anything below
>
> **§6 HAS RUN**, same day, on `tanitad-thor` (GPU idle, no training pod touched).
> Results: `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/
> 2026-08-03-dtac1-tactical-head/` (`DTAC1_RESULTS.md` + `dtac1_probe_refc-base-30k.json`).
> Substrate banked at `tanitad-thor:/home/nvidia/TanitAD/taniteval/results/
> dtac1_substrate_refc-base-30k.pt`. `refc-base` step 29999, canonical val, 39 episodes /
> **1364 windows** (denser stride than LAN_E0's 859 — NOT window-for-window comparable).
>
> | clause | outcome |
> |---|---|
> | **§6.0 `control_shuffled`** | ✅ **PASS** — `auc_lon_active` **0.4933**, macro-recall 0.3278 (chance 0.3333). The statistics discriminate. |
> | **§6.0 `control_label_source`** | ✅ **PASS** — derived vs banked `maneuvers` agreement **1.0000**. The gap §6.0 flagged is CLOSED; §2.1's arithmetic stands. |
> | **§6.1 E-A1** | `auc_lon_active` **0.7294** (per class 0.708 / 0.729 / 0.736) ⇒ **READOUT-limited** (threshold ≥ 0.65, fixed in advance). |
> | **§6.3 my registered prediction (INPUT-limited)** | ⛔ **REFUTED.** I was wrong. `RETRACTION_LOG.md` R-2026-08-03-dtac1. |
> | **§6.2 E-A2** | `pooled` 0.3833 → `pooled+v0` **0.4346** macro-recall (+0.051), every class's AUC up. F1 is real but MODEST, not the missing ingredient. |
> | **§6.3 "READOUT-limited ⇒ F2+F3 sufficient"** | ⚠️ **TOO OPTIMISTIC, and the τ frontier is what shows it.** F2 alone (τ=0) gives brake recall **0.072** / accel **0.045** — essentially nothing. F3 lifts `brake_stop` to **0.503** at τ=0.5 with NO retrain, but **`accelerate` never exceeds 0.153 at ANY τ** because the rarest class (`brake_stop`) crowds it out as τ rises. |
> | **NEW, not anticipated here** | **132 / 1364 = 9.68 %** of windows have a live longitudinal manoeuvre destroyed into a turn by the 5-way LABEL. No decode rule can recover those — this is the irreducible part that justifies the retrain. |
>
> **Revised lever ordering (was F1 > F2 > F3 in §3; MEASURED ordering is F3 > F2 > F1):**
> F3 is free and available today for `brake_stop`; F2 is what the retrain actually buys (the 9.68 %,
> `accelerate`, and the `lon_to_anchor` selection graft); F1 rides along for +0.051 macro-recall.
>
> ⛔ **Nothing in §§0–9 below has been edited.** The pre-registered text stands as written so the
> outcomes stay falsifiable — including the prediction that was refuted.

**Date:** 2026-08-03 (Europe/Berlin) · **Author:** STREAM A (REF-C tactical-head agent) ·
**Status:** code + instruments **DELIVERED and STAGED**, **no training launched**, **0 GPU spent**.
**Owner file:** `stack/tanitad/refs/refc.py` (no other stream touches it).

**Estimator for every interval below:** episode-cluster bootstrap over the val episodes,
`taniteval/taniteval/ci.py::episode_cluster_bootstrap`; **paired** for two arms on the same windows.
`overlapping_holdout_se` is never called — it biases the point estimate (−6.67 %…+11.69 %,
bidirectional) as well as the interval.

**This document is written BEFORE any D-TAC1 number exists. Both outcomes of the discriminating
experiment are committed in §6, with the decision thresholds fixed here, in advance.**

---

## 0. The claim under test

> REF-C's 5-way manoeuvre head never emits a longitudinal class. The documented mechanism is that a
> single softmax MIXES lateral and longitudinal classes, so the lateral classes — which dominate the
> corpus — win every argmax.

**I re-derived this from source. The first half is confirmed and made exact. The second half, as
stated, is REFUTED by the programme's own measurement**, and the correction changes which fix is
load-bearing. §1–§3.

---

## 1. The defect, as a chain of MEASURED facts

Every link carries a primary source. None is INHERITED.

| # | fact | class · source |
|---|---|---|
| 1 | REF-C emits **one** 5-way softmax: `N_MANEUVERS = 5` over `(lane_keep, turn_left, turn_right, accelerate, brake_stop)` — 3 lateral + 2 longitudinal classes in one mutually-exclusive simplex. | **MEASURED** — `stack/tanitad/refs/refc.py:100`, head built at `refc.py` `RefCModel.__init__` (`maneuver_head`, `Linear(aux_hidden, N_MANEUVERS)`) |
| 2 | The label is minted by a **PRIORITY collapse** of two orthogonal axes, `turn > brake > accel > lane_keep`. | **MEASURED** — `stack/scripts/refb_labels.py:100-109` (`classify_maneuver`, v1) and `:339-347` (`classify_maneuver_v2`) |
| 3 | The REF-C trainer supervises that head with the **v1** labeler, unweighted CE, weight 0.1. | **MEASURED** — `stack/scripts/refc_train.py:345-347` (`window_maneuver_labels` → `F.cross_entropy`), `MANEUVER_WEIGHT = 0.1` at `:79` |
| 4 | The head's logits are added **into the anchor confidences that make the trajectory selection**: `conf += maneuver_to_anchor(log_softmax(maneuver_logits))`, `Linear(5, n_anchors, bias=False)`. | **MEASURED** — `refc.py`, `AnchoredDiffusionDecoder.forward`, the H19 block |
| 5 | On the canonical val at **n = 859 windows / 39 episodes** the head predicts `accelerate` **0 / 93** and `brake_stop` **7 / 78**; `never_predicted: ["accelerate"]`. | **MEASURED** — `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-lan-refc-e0/LAN_E0_RESULTS.md` §5 TACTICAL |
| 6 | It is **INVARIANT to `nav_cmd`**: all four nav arms give a bit-identical manoeuvre histogram, because `maneuver_head` reads `pooled` while `nav_cmd` enters only via `measurement` → decoder. | **MEASURED** — same artifact, §5 |
| 7 | **The manoeuvre head never sees the ego speed.** `man_logits = self.maneuver_head(pooled)` — the image embedding alone. `v0` is scaled and ego-dropped at `refc.py` `forward` and consumed **only** by `self.measurement` → the decoder condition. | **MEASURED** — `stack/tanitad/refs/refc.py`, `RefCModel.forward` (the `v` / `m` block precedes the aux-head block) |
| 8 | The same file's `refc1` target-speed head **does** concatenate the measurement (`speed_cls(cat([pooled, m]))`), so "the head should see more than the image" is a pattern REF-C already uses elsewhere. | **MEASURED** — `refc.py`, `RefCModel.forward`, `refc1` block |

---

## 2. The mechanism, in algebra, from source

Because the 5-way label is a **deterministic function of a (lat, lon) pair** (fact 2), a factorised
posterior `p_lat × p_lon` induces the 5-way posterior exactly:

```
P5(turn_left)  = P_lat(turn_left)
P5(turn_right) = P_lat(turn_right)
P5(lane_keep)  = P_lat(lane_keep) · P_lon(steady)
P5(accelerate) = P_lat(lane_keep) · P_lon(accelerate)     <-- a PRODUCT
P5(brake_stop) = P_lat(lane_keep) · P_lon(brake_stop)     <-- a PRODUCT
```

(implemented and pinned as an exact round-trip: `refc_tactical.derive_man5_logprobs` /
`invert_man5`, `tests/test_refc_tactical.py::test_derive_is_a_distribution_and_invert_round_trips`).

So the shipped head emits `accelerate` only when

```
P_lat(lk)·P_lon(acc)  >  P_lat(lk)·P_lon(steady)     <-- the longitudinal question
                      >  P_lat(turn_left)            <-- a LATERAL comparison
                      >  P_lat(turn_right)           <-- a LATERAL comparison
```

Lines 2–3 have nothing to do with the longitudinal question and are the mixing: **any lateral
uncertainty raises the bar a longitudinal class must clear.** That is the mechanism, exactly.

### 2.1 …but "the lateral classes win every argmax" is REFUTED

The measured histogram (fact 5) does not say that:

| class | n true | n predicted | recall |
|---|---:|---:|---:|
| lane_keep | 510 | **675** | 0.9745 |
| turn_left | 110 | **106** | 0.8182 |
| turn_right | 68 | **71** | 0.8529 |
| accelerate | 93 | **0** | 0.0000 |
| brake_stop | 78 | **7** | 0.0385 |

**The turns are emitted at very nearly their true rate** (106 vs 110, 71 vs 68) — `P_lat` is well
learned and the turns are *not* stealing the argmax. And the accounting closes almost exactly:
`675 − 510 = 165` excess `lane_keep` against `93 + 78 − 7 = 164` missing longitudinal. **The
longitudinal mass lands entirely in `lane_keep`.**

⇒ The binding failure is the **within-`lane_keep`** comparison `P_lon(steady)` vs `P_lon(brake)` vs
`P_lon(accel)`, which is degenerate. Restricted to that branch the longitudinal marginal is
**steady 510 / brake 78 / accel 93 = 74.9 % / 11.5 % / 13.7 %** (MEASURED, derived from the table
above). A 3-way softmax under an unweighted CE with a 74.9 % majority class and weak features can
argmax `steady` everywhere *even after factorisation*.

**This matters: it means factorising the head, on its own, is not guaranteed to fix anything.**

---

## 3. Three separable causes — and the honest statement of which is which

| lever | claim | status |
|---|---|---|
| **F1 INPUT** | The head is asked a question about speed (`dv = v(t+2s) − v(t)`) while blind to speed (fact 7). | **HYPOTHESIS** that this is the binding constraint. Tested by E-A2. |
| **F2 STRUCTURE** | The single softmax multiplies the longitudinal decision by the lateral one (§2) and cannot represent "lane_keep AND braking" at all. | **MEASURED** as a property of the code. Its *sufficiency* is **HYPOTHESIS**, tested by E-A1. |
| **F3 DECISION** | Even factorised, an argmax over a 74.9 %-majority softmax need not emit a minority class; the matching fix is prior-corrected decoding (logit adjustment). | **MEASURED** as arithmetic (§2.1). Its *necessity* is tested by E-A1's adjusted decode. |

⚠️ **The programme's existing spec** (`TanitAD Research Hub/Architecture & Inference/
V3_FACTORIZED_TACTICAL_HEAD_SPEC.md`, 2026-07-21) proposes **F2 only**. If E-A2 lands as I predict,
F2 alone would have produced a null result and cost a full REF-C retrain to find out.

---

## 4. The architecture choice, argued against the alternatives

### 4.1 What is implemented

1. **Factorise into LAT(3) × LON(3)** — `lane_keep / turn_left / turn_right` ×
   `brake_stop / steady / accelerate` — with the **exact collapse table** back to the shipped 5-way,
   so `maneuver_logits [B,5]` is still emitted (derived) and every downstream reader keeps working.
2. **One shared trunk, two linear readouts.** MEASURED: 104,191,577 → **104,192,474 params, +897
   (+0.00086 %)**. Two independent MLPs would have cost **+272,001 (+0.261 %)** and let anyone
   attribute an A/B win to capacity — the test caught that, and it also falsified the old spec's
   "≈ 5 k parameters" ESTIMATE.
3. **The tactical head reads the ego speed** (`tactical_speed_input`) — the **speed channel only**,
   never the full measurement, because `nav_cmd` is a CONSTANT at eval (`nav_cmd=None → follow`) and
   training a head on a signal that vanishes at eval is the C6 confound.
4. **Two summed anchor grafts** replacing the one rank-5 graft: `lat_to_anchor` (default init,
   inherits today's LIVE H19 role) + `lon_to_anchor` (**zero-init**), so the step-0 selection path is
   today's lateral-only prior and every later change is attributable to the longitudinal seam.
5. **Prior-corrected decode** (`man_prior_tau`) over class log-prior buffers that travel with the
   checkpoint, **uniform at init** so an un-updated prior is an exact identity at any τ.

### 4.2 Why NOT the alternatives

| alternative | why it is rejected |
|---|---|
| **A. Keep the 5-way head, fix only the decision rule** (class-balanced / focal CE, or post-hoc logit adjustment). Genuinely attractive: post-hoc adjustment needs **no retrain at all**. | **Necessary but not sufficient.** A 5-way argmax cannot return a *pair*: on a window that is `turn_left` AND `braking`, no threshold recovers the braking, because the label itself destroyed it (fact 2) and the head was never trained on it. It also leaves the H19 graft rank-5 and lateral-dominated. ⇒ **adopted as F3, inside the factorisation, not instead of it.** |
| **B. Cartesian-product head** — one 9-way softmax over (lat, lon) pairs. | Removes the priority collapse but keeps ONE softmax, so it re-imports §2.1's marginal collapse in a worse form: the `(lane_keep, steady)` cell is ~56 % of windows and the rare cells are rarer still. It also cannot express lateral and longitudinal *independence*, which is the property the whole vocabulary (`vocab.py`'s orthogonal `LATMANEUVER` × `LONMODE` slots) is built on. Strictly more parameters, strictly less structure. |
| **C. Drop classification, regress (target speed, curvature).** | Loses the discrete decision the hierarchy thesis is about — "tactical decision quality" is a binding metric family and a scalar cannot be confused-matrixed. REF-C already has a gated target-speed head (`refc1`) that is OFF by default; that is a complement, not a replacement. |
| **D. The 2026-07-21 spec's 8×7 vocabulary heads** (`LAT_KINEMATIC_TOKENS` + sentinel × `LON_KINEMATIC_TOKENS` + sentinel). | Richer, and the right *destination*. But it couples the architecture change to a **label-pipeline migration** (`--labels v3` fields `lat_idx`/`lon_idx`), so a null result would be non-attributable between "the factorisation didn't help" and "the new labels are worse". The 3×3 factorisation is derivable from the **identical kinematics the trainer already reads**, its projection is byte-identical to today's label (asserted every step, and fuzz-tested), and the head width is a config knob so the 9-token vocabulary drops in later without redoing this work. **Minimal attributable change first.** |

---

## 5. What is implemented and staged (no training launched)

| artifact | repo path | what it is |
|---|---|---|
| vocabulary + labels + collapse/inverse | `stack/tanitad/refs/refc_tactical.py` | LAT/LON classes, the collapse table, v1- and v2-faithful factored labelers, `derive_man5_logprobs` / `invert_man5`, `logit_adjust`, `prior_centered_logprobs` |
| model seam | `stack/tanitad/refs/refc.py` | `factored_maneuver`, `tactical_speed_input`, `man_prior_tau`, `graft_prior_center`; `lat_head`/`lon_head` off a shared trunk; `lat_to_anchor` + zero-init `lon_to_anchor`; `update_tactical_prior`; `refc_factored_config()` |
| trainer | `stack/scripts/refc_train.py` | `--factored-maneuver` / `--tactical-speed-input` / `--man-prior-tau` / `--no-graft-prior-center`; two masked CEs at **LAT 0.05 + LON 0.05 = MANEUVER_WEIGHT 0.10 exactly**; per-step `lon_active_pred` and graft-norm parity logging; a **fail-loud drift guard** that refuses to train if the factored labels stop collapsing to the shipped 5-way label |
| probe | `stack/scripts/refc_tactical_probe.py` | E-A1 + E-A2 + both negative controls, one command, JSON out |
| tests | `stack/tests/test_refc_tactical.py` | 24 tests: mirrored-constant pins, the collapse self-consistency control, derive/invert round-trip, logit-adjust identity + its negative control, gated-flag byte-identity, zero-init + gradient reach, the **speed-input negative control**, graft ablatability, trainer weight conservation, drift-guard, capacity pin, probe end-to-end |

**Gated-flag discipline holds:** with `factored_maneuver=False` the state_dict and every forward
output are byte-identical to today, so **every published REF-C number stays reproducible**.

---

## 6. ⭐ THE CHEAPEST DISCRIMINATING EXPERIMENT — run BEFORE any GPU-day

Two experiments, **one forward pass over val on an EXISTING checkpoint, no training**. Command:

```
OMP_NUM_THREADS=6 PYTHONPATH=<repo>/stack python scripts/refc_tactical_probe.py \
  --ckpt /home/nvidia/models/refc-base/ckpt.pt \
  --val-dir /home/nvidia/valdata/physicalai-val-0c5f7dac3b11 \
  --preset base --out /tmp/dtac1_probe_refc-base-30k.json
```

Both the checkpoint and the **canonical 40-episode** val cache are present on `tanitad-thor`
(MEASURED 2026-08-03 by direct `find`). Thor is not training. Cost: minutes, no training pod touched.

### 6.0 Negative controls, run FIRST and reported alongside (non-negotiable)

* **`control_shuffled`** — permute the window↔logits pairing. Every accuracy and AUC must fall to
  chance. If a statistic still "separates" under this, it is reading the class prior and is not
  quotable.
* **`control_label_source`** — the derived 5-way label must reproduce the epcache's banked
  `maneuvers`. ⚠️ The §2.1 table's `n true` column comes from the **banked** field; my probe derives
  the label with the **v1** rule the trainer uses. If they disagree, **the §2.1 counts are re-read
  against the derived label and this pre-registration's §2.1 arithmetic is re-derived before
  anything is concluded.** This gap is named here rather than discovered later.

### 6.1 E-A1 — counterfactual factored decode (is the information THERE?)

Invert the collapse on the existing 5-way posterior and ask what the longitudinal decision would
have been unmixed. The **threshold-free** statistic is `auc_lon_active` = ROC-AUC of
`1 − P_lon(steady)` against "a longitudinal manoeuvre is happening".

### 6.2 E-A2 — linear probe on the head's OWN input (is the information REACHABLE?)

Episode-disjoint 2-fold multinomial logistic regression onto the longitudinal label from
(a) `pooled` alone — literally what the head is given today; (b) `v0` alone; (c) `pooled + v0`.
Read: **macro-recall (balanced accuracy, chance = 0.3333)** and one-vs-rest AUC.

### 6.3 ⛔ BOTH OUTCOMES, COMMITTED IN ADVANCE — thresholds fixed HERE

| branch | trigger (fixed now) | what it means | what happens next |
|---|---|---|---|
| **READOUT-limited** | `auc_lon_active ≥ 0.65` | The 5-way head HAS longitudinal information; the mixed argmax and the prior hide it. | **F2 + F3 are sufficient.** A large part is recoverable **with no retrain**: publish the prior-corrected factored decode as a REF-C read-out patch, re-score the TACTICAL family, and make the retrain OPTIONAL (it then buys only the graft). **F1 is demoted to a nice-to-have.** |
| **INPUT-limited** | `auc_lon_active ≤ 0.55` | The head carries no longitudinal information at all. | **F1 is REQUIRED and F2 alone would have been a null result.** The retrain is justified, and it must carry `--tactical-speed-input`. The 2026-07-21 spec's F2-only proposal is **superseded**, and that supersession is logged in `RETRACTION_LOG.md`. |
| **INDETERMINATE** | `0.55 < auc_lon_active < 0.65` | Tie-break by E-A2: if `pooled_only` macro-recall < 0.40 while `pooled_plus_v0` ≥ 0.45, read as **INPUT-limited**; otherwise as **READOUT-limited**. | as the branch it resolves to |

**My prediction, registered before running: INPUT-limited.** Grounds: fact 7 (the head is
structurally blind to `v0`), plus the programme's MEASURED speed-input result on REF-A
(3.73 → 0.83 m fwd-ADE, speed-R² 0.61 → 0.965 when `v0` was added as an input channel). **If
`auc_lon_active ≥ 0.65` I am wrong, and I will say so in the results file rather than reframe it.**

⚠️ **A null on BOTH** (`auc_lon_active` ≈ 0.5 *and* `pooled_plus_v0` macro-recall ≈ chance) refutes
the whole D-TAC1 premise: the 2 s longitudinal label would then be unpredictable from this substrate
at all, and the right response is to question the **label horizon**, not to build a bigger head.
That outcome is committed here too.

---

## 7. IF a retrain is warranted — the arm, and how it is judged

Not launched by this stream. Pre-committed so it cannot be re-scoped after seeing a number.

**Arms** (identical data, parity key `physicalai-train-e438721ae894`, skip-hash `f09e44db`; identical
optimizer/schedule/anchors/seed; the ONLY differences are the named flags):

| arm | flags | purpose |
|---|---|---|
| `refc-base-30k` | *(existing)* | control — already trained, no GPU |
| `dtac1-full` | `--factored-maneuver --tactical-speed-input --man-prior-tau 1.0` | the fix |
| `dtac1-f2only` | `--factored-maneuver` | isolates STRUCTURE (this is the old spec's proposal) |
| `dtac1-nolon-graft` | `--factored-maneuver --tactical-speed-input` + `lon_to_anchor` frozen at zero | isolates whether the longitudinal prior reaching **selection** is what moves the trajectory, vs. merely reporting a better class |

**Primary read — the four metric families, per family, never pooled**, each with the *paired*
episode-cluster bootstrap on the same windows (CLAUDE.md, binding):

* **TACTICAL** (the family this change targets): per-class recall + confusion over the LON axis;
  `lon_active` emission rate vs the label rate; macro-recall. **Pre-committed success:**
  `accelerate` and `brake_stop` recall both **> 0.20** with macro-recall separated above the control.
  **Pre-committed failure:** `never_predicted` still contains a longitudinal class.
* **LONGITUDINAL**: target-speed accuracy and distance-keeping (headway / time-gap / TTC).
  Pre-committed direction: better or not-separated. **A separated regression here fails the arm even
  if TACTICAL improves.**
* **LATERAL**: heading, curvature, yaw-rate, cross-track. Pre-committed: **not separated** — the
  factorisation must not buy longitudinal emission with lateral quality. This is the guard-rail.
* **STRATEGIC**: route/goal-setting quality, unchanged surface (this arm does not touch `route_head`).
  Reported so the absence of an effect is visible rather than assumed.
* **ADE** stays, at all horizons, as **one row of five** — never as "the result".

Any family that genuinely cannot be computed is reported **per family with its reason and n**, never
silently dropped.

---

## 8. What this is NOT

* **Not** a fix for `stop_line` / `TACPOINT` naming — kinematics mints *where* a vehicle stops, never
  *why*. That needs the VLM/map pass.
* **Not** a fix for `follow_lead` / `close_gap` / `open_gap` — `lead_state` is a `None` stub, so those
  tokens have no supervision and are deliberately absent from the head.
* **Not** a route/LAN fix. LAN E0 already MEASURED that supplying the oracle route makes cross-track
  *separated worse*, and that both aux heads read `pooled` so LAN cannot reach them. D-TAC1 changes
  what the tactical head reads and how it decides; it does not change the route pathway.
* **Not** a capacity experiment: **+897 parameters, MEASURED**.
* **Not** a claim that the 3×3 vocabulary is the final one. It is the minimal attributable step
  toward the frozen 9-token `LATMANEUVER` × `LONMODE` slots.

---

## 9. Deliverable manifest

All in the working tree of `agent/benchmarks-eval-20260802`, **staged, not committed, not pushed.**

| artifact | path |
|---|---|
| this pre-registration | `Project Steering/PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md` |
| factorised tactical vocabulary + labels + collapse/inverse | `stack/tanitad/refs/refc_tactical.py` |
| model seam | `stack/tanitad/refs/refc.py` |
| trainer wiring | `stack/scripts/refc_train.py` |
| probe (E-A1 + E-A2 + controls) | `stack/scripts/refc_tactical_probe.py` |
| tests (24) | `stack/tests/test_refc_tactical.py` |

**Nothing lives only on a pod or only in an agent's context.** The probe is designed to run on
`tanitad-thor` (checkpoint + canonical val cache both verified present there 2026-08-03) and to write
its JSON to a path the operator names; that JSON is the artifact §6.3 is adjudicated from.
