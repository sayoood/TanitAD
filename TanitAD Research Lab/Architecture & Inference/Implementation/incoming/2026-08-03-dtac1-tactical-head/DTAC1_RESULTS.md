# D-TAC1 — REF-C's manoeuvre head: diagnosis re-derived, fix implemented, probe RUN

**Date:** 2026-08-03 (Europe/Berlin) · **Stream A** · **0 GPU-days of training spent.**
**Pre-registration:** `Project Steering/PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md`
(written and its thresholds fixed **before** the probe ran).
**Raw artifact:** `dtac1_probe_refc-base-30k.json` (this directory).
**Banked substrate (0-GPU re-analysis):**
`tanitad-thor:/home/nvidia/TanitAD/taniteval/results/dtac1_substrate_refc-base-30k.pt`.

**Run:** `refc-base` `ckpt_step 29999`, 104.192 M params, `sd_missing 0 / sd_unexpected 0`,
canonical val `physicalai-val-0c5f7dac3b11`, **39 episodes / 1364 windows**, stride 5,
decoder steps 2, on `tanitad-thor` (GPU idle, no training pod touched).
**Estimator:** `taniteval.ci.episode_cluster_bootstrap`, unit = episode. `overlapping_holdout_se`
never called.

⚠️ **n = 1364, not the 859 of `LAN_E0_RESULTS.md`** — same 39 episodes, denser stride. The two runs
are NOT window-for-window comparable; every count below is on the 1364-window grid.

---

## 0. Headline

1. **The documented mechanism is half right.** The 5-way softmax does mix the axes, and the algebra
   of that mixing is now exact. But **"the lateral classes win every argmax" is REFUTED** — the turns
   are emitted at almost exactly their true rate. The longitudinal mass lands entirely in
   `lane_keep`.
2. **A third defect, previously unrecorded, is in the source: the manoeuvre head never sees the ego
   speed.** It reads `pooled` (image only) while its own label is `dv = v(t+2s) − v(t)`.
3. **⛔ I pre-registered the wrong prediction and the probe refuted it.** I predicted INPUT-limited;
   the measurement says **READOUT-limited** (`auc_lon_active` **0.7294**, shuffled control 0.4933).
   The information IS in the existing head.
4. **But the pre-registered "READOUT-limited ⇒ F2+F3 are sufficient" is ALSO too optimistic**, and
   the τ frontier is what shows it. Factorisation alone recovers essentially nothing; prior-corrected
   decoding recovers `brake_stop` and **cannot recover `accelerate` at any τ**.
5. Net: **a zero-GPU read-out patch is available today for `brake_stop`. The retrain is still
   justified — for `accelerate`, for the 9.68 % of windows the label itself destroys, and for the
   selection graft — but NOT for brake reporting.** That is a different and much better-scoped ask
   than "retrain REF-C because the head is broken".

---

## 1. The diagnosis, re-derived from source

| # | fact | source |
|---|---|---|
| 1 | ONE 5-way softmax over 3 lateral + 2 longitudinal classes | `stack/tanitad/refs/refc.py:100`, `RefCModel.__init__` |
| 2 | label = PRIORITY collapse `turn > brake > accel > lane_keep` | `stack/scripts/refb_labels.py:100-109` (v1), `:339-347` (v2) |
| 3 | trainer supervises with the **v1** labeler, unweighted CE, weight 0.1 | `stack/scripts/refc_train.py:345-347`, `:79` |
| 4 | head logits enter the **anchor confidences that make the selection** | `refc.py`, `AnchoredDiffusionDecoder.forward`, H19 block |
| 5 | ⭐ **the head reads `pooled` — the IMAGE embedding alone.** `v0` is scaled/ego-dropped and consumed only by `self.measurement` → the decoder | `refc.py`, `RefCModel.forward` |
| 6 | the same file's `refc1` speed head **does** concatenate the measurement | `refc.py`, `refc1` block |

### 1.1 The mixing, as algebra

The 5-way label is a deterministic function of a `(lat, lon)` pair, so

```
P5(turn_left)  = P_lat(turn_left)
P5(lane_keep)  = P_lat(lane_keep) · P_lon(steady)
P5(accelerate) = P_lat(lane_keep) · P_lon(accelerate)      <-- a PRODUCT
P5(brake_stop) = P_lat(lane_keep) · P_lon(brake_stop)      <-- a PRODUCT
```

so `accelerate` wins the argmax only if it beats `steady` (the real question) **and** beats
`P_lat(turn_left)` and `P_lat(turn_right)` (a lateral comparison that has nothing to do with it).
Implemented and pinned as an exact round-trip (`refc_tactical.derive_man5_logprobs` / `invert_man5`).

### 1.2 REFUTED: "the lateral classes win every argmax"

MEASURED, this run, the shipped 5-way decode:

| class | n true | n predicted | recall |
|---|---:|---:|---:|
| lane_keep | 818 | **1078** | 0.9743 |
| turn_left | 174 | 165 | 0.8218 |
| turn_right | 109 | 114 | 0.8349 |
| **accelerate** | **146** | **0** | **0.0000** |
| **brake_stop** | **117** | **7** | **0.0256** |

Accuracy 0.7581, macro-recall 0.5313, `never_predicted: ["accelerate"]`.
**The turns are calibrated** (165 vs 174, 114 vs 109) and `818 → 1078` is `+260` against
`146 + 117 − 7 = 256` missing longitudinal. The lateral readout in isolation is fine:
**macro-recall 0.8290, accuracy 0.9348**. The failure is the **within-`lane_keep`** longitudinal
comparison, whose marginal is `steady 0.7104 / accelerate 0.1774 / brake_stop 0.1122`.

### 1.3 The label destroys 9.68 % of the longitudinal decisions outright

**MEASURED: 132 / 1364 = 9.68 %** of windows carry a live longitudinal manoeuvre AND are labelled a
turn. The 5-way *target* cannot represent them, so **no decode rule can ever recover them** — this is
the irreducible part that requires the factorised LABEL, i.e. a retrain.

---

## 2. The probe (E-A1 / E-A2) — controls first

| control | result | reading |
|---|---|---|
| **`control_label_source`** — derived 5-way vs the epcache's banked `maneuvers` | **agreement 1.0000** over 1364 windows | the two label mints have NOT drifted; §1.2's counts sit on one label surface. (The pre-registration flagged this as an open gap — it is now closed.) |
| **`control_shuffled`** — window↔logits pairing permuted | `auc_lon` 0.515 / 0.493 / 0.482, `auc_lon_active` **0.4933**, factored macro-recall **0.3278** (chance 0.3333) | the statistics discriminate; they are not reading the class prior |

### 2.1 E-A1 — the information IS there (my prediction refuted)

`auc_lon_active` (ROC-AUC of `1 − P_lon(steady)` against "a longitudinal manoeuvre is happening",
recovered from the EXISTING 5-way head by inverting the collapse):

| | value |
|---|---:|
| **`auc_lon_active`** | **0.7294** |
| per class: brake_stop / steady / accelerate | 0.7082 / 0.7294 / **0.7362** |
| shuffled control | **0.4933** |
| pre-registered threshold for READOUT-limited | ≥ 0.65 |

⇒ **READOUT-limited.** ⛔ **I predicted INPUT-limited and I was wrong.** Logged in
`RETRACTION_LOG.md` (R-2026-08-03-dtac1) with its root-cause class.

### 2.2 E-A2 — the speed input helps, but it is not the missing ingredient

Episode-disjoint 2-fold multinomial logistic regression onto the longitudinal label
(chance macro-recall 0.3333):

| feature set | macro-recall | AUC brake / steady / accel |
|---|---:|---|
| `pooled` only (what the head is given TODAY) | 0.3833 | 0.5378 / 0.4839 / 0.7135 |
| `v0` only (1 feature, the channel it is NOT given) | 0.3416 | 0.3626 / 0.6988 / 0.7104 |
| **`pooled + v0` (the F1 proposal)** | **0.4346** | **0.5769 / 0.5612 / 0.7684** |

Adding `v0` improves **every** class's AUC and macro-recall by **+0.051**. Real, and modest.
⚠️ The trained head reaches 0.708–0.736 on the same features where a LINEAR probe reaches
0.484–0.714 — so this probe **understates** what `pooled` carries and must not be read as
"`pooled` is uninformative".

### 2.3 ⭐ The τ frontier — the finding that reorders the whole fix

`decode_factored_*` on the longitudinal axis (true counts: brake 153 / steady 969 / accel 242):

| τ | accuracy | macro-recall | recall brake / steady / accel | n_pred brake / steady / accel |
|---:|---:|---:|---|---|
| **0.00** *(= F2 alone)* | 0.7045 | 0.3621 | **0.072** / 0.969 / **0.045** | 47 / 1299 / 18 |
| 0.25 | 0.6782 | 0.4125 | 0.255 / 0.892 / 0.091 | 147 / 1159 / 58 |
| **0.50** | 0.5953 | 0.4588 | **0.503** / 0.720 / **0.153** | 398 / 868 / 98 |
| 0.75 | 0.5132 | 0.4709 | 0.739 / 0.583 / 0.091 | 652 / 661 / 51 |
| 1.00 | 0.4267 | **0.4761** | 0.902 / 0.435 / 0.091 | **865** / 462 / 37 |
| 1.25 | 0.3644 | 0.4666 | 0.987 / 0.339 / 0.074 | 984 / 339 / 41 |
| 2.00 | 0.1510 | 0.3624 | 0.987 / 0.042 / 0.058 | 1247 / 41 / 76 |

Three things fall out, and none of them was in the plan:

1. **Factorisation ALONE (τ = 0) recovers essentially nothing** — brake recall 0.072, accel 0.045,
   macro-recall 0.3621 against a 0.3333 chance floor. The 2026-07-21 spec's F2-only proposal would
   have cost a full REF-C retrain to land here.
2. **`brake_stop` IS recoverable by decode alone**: 0.026 → **0.503** at τ = 0.5, for −0.109
   accuracy. No retrain, no new parameters, a post-hoc transform of logits the model already emits.
3. ⛔ **`accelerate` is NOT recoverable at ANY τ.** It peaks at **0.153** (τ = 0.5) and *falls* after.
   The reason is the same disease one level down: as τ rises, `brake_stop` (the rarest class, prior
   0.112) takes the biggest boost and **crowds out `accelerate`, not `steady`** — 865 brake
   predictions against 153 true at τ = 1. A prior-corrected **argmax over 3** is still an argmax.

⚠️ Reading τ off this table is **fitting on val**. It is a FRONTIER REPORT. A deployed τ must be
chosen on train/dev data (the model's own EMA prior buffers) and only then confirmed here.

---

## 3. What each lever actually buys — MEASURED, replacing the pre-registered ordering

| lever | what it buys | evidence | needs a retrain? |
|---|---|---|---|
| **F3** decode (prior correction) | `brake_stop` recall 0.026 → **0.503** at τ = 0.5 | §2.3 | **NO** |
| **F2** structure (factorised head + label) | the *only* route to the **9.68 %** of windows whose longitudinal class the 5-way label destroys; and the only surface a per-axis decode rule can act on at all | §1.3, §2.3 τ = 0 row | **YES** |
| **F2** `lon_to_anchor` graft | the only way a longitudinal prior reaches **anchor SELECTION** (today's graft is rank-5 and lateral-dominated) | architecture; untested — this is what the retrain measures | **YES** |
| **F1** speed input | +0.051 macro-recall, +0.04/+0.05 AUC on brake/accel in a linear probe | §2.2 | **YES** |
| **new** per-class threshold calibration on the LON axis | the only candidate that can lift `accelerate` without brake crowding it out | §2.3 point 3 — **not yet implemented** | NO (post-hoc) |

**Revised priority: F3 (free, today) → F2 (the retrain's real justification) → F1 (a modest rider on
the same retrain).** The pre-registration had this ordering backwards, and the probe cost minutes.

---

## 4. What is implemented and staged

Gated-flag discipline holds: `factored_maneuver=False` ⇒ state_dict and every forward output are
byte-identical to today, so **every published REF-C number stays reproducible**.

| artifact | path |
|---|---|
| factorised vocabulary, labels, collapse + exact inverse, logit adjustment | `stack/tanitad/refs/refc_tactical.py` |
| model seam: `factored_maneuver` / `tactical_speed_input` / `man_prior_tau` / `graft_prior_center`, shared trunk + `lat_head`/`lon_head`, `lat_to_anchor` + **zero-init** `lon_to_anchor`, `update_tactical_prior`, `refc_factored_config()` | `stack/tanitad/refs/refc.py` |
| trainer: 4 CLI levers, two CEs at **0.05 + 0.05 = MANEUVER_WEIGHT 0.10 exactly**, `lon_active_pred` + graft-norm-parity logging, **fail-loud labeler-drift guard** | `stack/scripts/refc_train.py` |
| probe (E-A1, E-A2, τ frontier, both controls, substrate dump) | `stack/scripts/refc_tactical_probe.py` |
| 24 tests | `stack/tests/test_refc_tactical.py` |
| pre-registration | `Project Steering/PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md` |

**Capacity, MEASURED not estimated:** 104,191,577 → **104,192,474 = +897 params (+0.00086 %)**.
The first implementation here built two independent MLPs and cost **+272,001 (+0.261 %)**; the
capacity test caught it and the head was rebuilt on a shared trunk. The 2026-07-21 spec's
"≈ 5 k parameters" was an ESTIMATE and is superseded by this measurement.

`pytest -q` on `stack/`: **1808 passed, 12 skipped, 2 xfailed**.

---

## 5. Recommended next actions (in priority order)

1. **0 GPU, today — the read-out patch.** Choose τ on a TRAIN split, then re-score REF-C's TACTICAL
   family with the prior-corrected factored decode and publish it beside the raw one. The banked
   substrate makes every τ experiment free. Expected: `brake_stop` recall 0.026 → ~0.5.
2. **0 GPU — per-class threshold calibration on the LON axis**, to test whether `accelerate` can be
   lifted above 0.153 without `brake_stop` crowding it out. Named in §2.3; not implemented.
3. **The retrain** (`dtac1-full` + the three ablation arms in the pre-registration §7), justified by
   the 9.68 % label-destroyed windows, `accelerate`, and the selection graft — **not** by brake
   reporting, which is now free. Four metric families with the paired episode-cluster bootstrap;
   LATERAL is the guard-rail (pre-committed: must be **not separated**).
4. **Re-read `LAN_E0_RESULTS.md` §5's tactical table** knowing the lateral axis is fine and the
   defect is one-sided — the "TOP DEFECT" framing is correct but its mechanism was mis-attributed.
