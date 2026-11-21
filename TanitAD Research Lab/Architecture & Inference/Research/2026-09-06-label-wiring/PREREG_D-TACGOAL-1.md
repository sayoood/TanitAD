# PRE-REGISTRATION — D-TACGOAL-1: supervising the 22-token tactical goal set

**Registered 2026-09-06, BEFORE any arm is trained.** Both outcomes are
committed in advance. ⛔ No goalpost may move after the data is seen; a post-hoc
finding needs its own pre-registration.

---

## 1. The claim under test

> **H-TACGOAL-1.** Supervising the v7 tactical goal SET — 22 multi-label tokens
> that are currently minted on 4,572 clips and trained by nothing — makes the
> traffic-light state and the lane-change intent **predictable from vision at the
> tactical latent**, at a per-class recall that beats the majority-class control.

⚠️ **This is a SUPERVISION claim, not a driving claim.** It is deliberately not
"the model brakes for red lights". Closed-loop braking needs AlpaSim or a real
vehicle (binding ruling: a planner feeding its own predictor is still open loop),
and the label is clip-level, so no arm here can support a braking claim.

## 2. Arms

| arm | flags | what it isolates |
|---|---|---|
| **A0 baseline** | the paired banked refc_v3 config, `--v7-labels`, **no** `--w-tac-goal` | the recipe every banked number was produced under |
| **A0b replicate** | A0's flags, A0's seed, run again, **zero levers moved** | ⛔ the rig's own run-to-run noise floor |
| **A1 goal-head** | A0 + `--w-tac-goal 0.05 --tac-goal-negatives measured` | the head |
| **A2 naive-negatives** | A0 + `--w-tac-goal 0.05 --tac-goal-negatives all` | whether the honest ignore policy is worth its cost |
| **C-major** | no training — the majority-class predictor, scored by the same function on the same cells | ⭐ must read recall **exactly 0.0** / **exactly 1.0** |
| **C-frames-blind** | A1 with `--ablate-frames` | ⛔ the deliberate regression: a gate that has never FAILED an image-blind arm certifies nothing |

⛔ **A0b is not optional.** A separated CI from a one-seed arm is necessary and
**not sufficient**: on the v7-tiny rig a replicate with zero levers moved
produced "separated" differences on **6 of 42 family cells — a 14.3 %
false-positive rate**. Any A1-vs-A0 difference is read against A0b's spread, not
against zero.

## 3. Primary criteria, committed now

**PASS** requires **all** of:

1. **`TRAFFIC_LIGHT_REACT_RED` per-class recall ≥ 0.35** on the eval split, with
   its precision reported beside it — against a majority control that reads
   **exactly 0.0** (RED's majority is absent). The bar is set at roughly twice
   chance for a balanced 376/403 problem and below any level that would imply
   deployment.
2. **The margin over A0b's replicate spread on the same cell is larger than that
   spread**, i.e. the effect exceeds the rig's run-to-run noise floor.
3. **A1 does not regress the trajectory metric** beyond A0b's spread — reported
   in **all four families** (longitudinal, lateral, tactical, strategic), never
   ADE alone, never pooled, each with a **paired episode-cluster bootstrap** CI
   (⛔ never `overlapping_holdout_se`).
4. **The six-link wiring assertions and the mutation proof pass on the trained
   checkpoint**, not only on the tiny rig.

**FAIL** is any of: RED recall < 0.35; the margin inside A0b's spread; a
trajectory regression beyond that spread; or the frames-blind arm scoring within
A0b's spread of A1 (which would mean the head is reading something other than the
image).

⛔ **A FAIL is reported as a FAIL and the next lever is executed in the same
run.** The pre-declared next levers, in order of measured effect:

* **L1 — regenerate the labels at K=9 anchors** (0.5 min compute, 18 MB, → 100 %
  band coverage). Largest measured lever available; it does not help the TL
  tokens directly but removes the 12 % ceiling under everything else.
* **L2 — the two-stage traffic-light formulation**: stage A *"is a light
  claimed?"* (779 pos vs the 131 probed-and-none), stage B a **3-way softmax over
  RED/GREEN/YELLOW conditional on stage A** (376/363/22, n=761 labelled). Better
  posed than a flat sigmoid triple and free of new data.
* **L3 — mine the 62.1 % of CoT texts carrying an explicit `"<n> s"`** for a
  timing signal, with a control that must show the timestamps are not random with
  respect to the token.

## 4. Metrics, per family — ⛔ never ADE alone, never pooled

| family | reported |
|---|---|
| **LONGITUDINAL** | target-speed accuracy, headway / time-gap / TTC to the lead agent |
| **LATERAL** | heading, **curvature**, **yaw-rate**, cross-track |
| **TACTICAL** | ⭐ **per-class** recall/precision/n over all 22 goal tokens + the `tac_lat`/`tac_lon` confusion; selected-vs-executed manoeuvre |
| **STRATEGIC** | reported as **NOT COMPUTED, with the reason**: the strategic layer is being removed from the next experiments on PI instruction |

Where a family cannot be computed, it is stated **per family with the reason and
the n** — never silently dropped.

## 5. Instruments that are INVALID here and must not appear

⛔ `oracle_sel`, `anchor_acc`, `sel_agrees_oracle` (invalid on refcv4b).
⛔ The `|dyaw| > 0.15` gate. ⛔ `overlapping_holdout_se` in any form — it biases
the point estimate (−6.67 % to +11.69 %, bidirectional) as well as the interval.
⛔ Any T0 number quoted as driving performance; every number carries its T-tier.

## 6. Known confounds, declared in advance

* **`--v7-labels` pins `tac_vocab_version` to v7.0**, and under v7.0 the tactical
  brain stops driving the H19 anchor prior (`man5 = None`). The baseline must
  carry the same pin or the comparison is confounded.
* **The aux-weight budget.** `MANEUVER_WEIGHT = 0.10` is split `/2` across
  lat+lon. A1 adds 0.05 **on top**, so the tactical budget becomes 0.15 and
  `lat`/`lon` pressure is unchanged. The alternative — re-splitting to `/3` — is
  a **separate arm**, not a tweak, because it changes `lat`/`lon` and breaks
  comparability with every banked arm.
* **The label is clip-level and untimed** (`t_nominal_s` constant 4.00). The head
  is supervised on ~12 % of frames and learns a clip-level claim. No result here
  can be read as per-frame reaction.
* **Information disjointness.** The goal SET is an OUTPUT only. It is never fed
  to the forward — asserted by L6, which checks the signature exposes no
  goal-target channel, so the echo is impossible by construction rather than by
  convention.

## 7. Compute

Dev-box RTX 4060 or Thor. ⛔ **The A40 runs refcv5 until ≈2026-09-08 07:33 UTC
and must not be touched.** No new corpus, no new GPU-hours beyond one training
arm plus its replicate; the label regeneration (L1) is ~1 minute, CPU.

## 8. Escalation

Two seams live in `stack/scripts/refc_v3_train.py`, which a sibling owns: the
batch attach and the loss term. They are written out verbatim in
`refc_v3_train.tac_goal.patch.md` in this package. ⛔ **This pre-registration
does not authorise editing that file**; the arm cannot launch until its owner
applies the two additive edits or hands the file over.
