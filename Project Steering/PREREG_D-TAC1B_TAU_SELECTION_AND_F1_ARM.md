# PRE-REGISTRATION — D-TAC1b: an HONEST τ, and the missing F1-only arm

**Date:** 2026-08-03 (Europe/Berlin) · **Stream:** REF-C tactical head, follow-up ·
**Status when written:** code for T2 staged and green; **T1 has NOT been run.** 0 GPU-days.
**Parent:** `Project Steering/PREREG_D-TAC1_FACTORED_TACTICAL_HEAD.md`
**Parent results:** `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/`
`2026-08-03-dtac1-tactical-head/DTAC1_RESULTS.md` (+ its `adversarial-verification/`).

**Estimator for every interval below:** `taniteval/taniteval/ci.py`
`::paired_episode_cluster_bootstrap`, unit = val episode, callable reducers for macro-recall /
macro-F1. ⛔ `overlapping_holdout_se` is never called.

---

## 0. The two defects this closes

Both were flagged by the D-TAC1 stream **against its own work**, and neither was closed there.

| # | defect | primary source |
|---|---|---|
| **T1** | The prior-corrected decoding result (`brake_stop` 0.026 → 0.503) **read τ off the val set.** So did the class prior it adjusts by. | `DTAC1_RESULTS.md` §2.3 ⚠️ note; adversarial `ADVERSARIAL_RECORD.md` **R6** (*"uses the VAL LABEL MARGINAL as its prior"*) |
| **T2** | `tactical_speed_input` (F1) was **coupled** to `factored_maneuver` (F2), so no arm isolated INPUT. F1's only estimate would have been `dtac1-full − dtac1-f2only`, where the two arms **also differ in the head itself**. | `stack/tanitad/refs/refc.py` (the `ValueError` at the aux-head block, pre-2026-08-03); parent §7 arm table |

---

## 1. T1 — what "honest" can and cannot mean here, stated BEFORE the run

### 1.1 The ideal protocol, and why it is not available

The ideal is: select `(π, τ)` on TRAIN windows, confirm on val. That needs REF-C's 5-way
posteriors on train windows, i.e. a forward pass over a **train** episode cache.

**MEASURED absence, 2026-08-03, five probes at four paths** (`tanitad-thor`, the only non-training
box available; the two live pods are off-limits by brief and `tanitad-pod`/`tanitad-pod3` both
answer `Connection refused`):

1. `ls -d /home/nvidia/*data* /home/nvidia/*train* /home/nvidia/*cache*` → only `valdata`
2. `find / -xdev -maxdepth 6 -name "physicalai-train*"` → nothing
3. `find /home/nvidia /mnt /data -maxdepth 4 -name "ep_*.pt"` → only
   `/home/nvidia/valdata/physicalai-val-0c5f7dac3b11`
4. the banked substrate itself (`dtac1_substrate_refc-base-30k.pt`) carries **val windows only**
5. `refc-base`'s checkpoint predates `factored_maneuver`, so it carries **no** `lat_log_prior` /
   `lon_log_prior` EMA buffer — the second route named in the brief does not exist for this arm

⇒ **A train-selected τ is NOT computable from any reachable artifact today.** This is recorded as a
result, not routed around. §1.4 states exactly what would make it computable.

### 1.2 What IS available, and why it is a real fix rather than a re-label

**Leave-one-episode-out (LOEO) selection inside val.** For each of the 39 val episodes `e`:

```
π_e  = empirical LON class log-prior of the windows NOT in e
τ_e  = argmax over the τ grid of CRITERION(windows not in e ; π_e)
pred = logit_adjust(log P_lon , π_e , τ_e).argmax()   evaluated ONLY on e's windows
```

Every reported window is decoded by a rule fitted **without it**, and **without its episode** — the
correct unit, since windows inside one clip are strongly dependent. Pooling the 39 held-out folds
gives an out-of-fold estimate over the same 1364 windows.

⚠️ **This is NOT a train-selected τ and will not be called one.** It removes *"τ was read off the
number being reported"*. It does **not** remove *"selection and reporting share a distribution"* —
val-vs-train shift is untested and untestable here. Both statements travel with every number.

⚠️ **I am not blind to the val frontier** — `DTAC1_RESULTS.md` §2.3 publishes all 8 rows, so the
val-optimal τ is already known to me. That is precisely why the guard is **mechanical** (a fold
cannot see itself) rather than a promise about my own state of knowledge. `ADVERSARIAL_RECORD.md`
**R11** is the lesson: "thresholds fixed in advance" is unverifiable self-report; an out-of-fold
protocol is checkable from the code.

### 1.3 ⛔ FIXED IN ADVANCE — criteria, grid, denominators, thresholds

**τ grid:** `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00` — the parent's published grid, unchanged.

**Two co-primary criteria, both fixed here, both reported, neither chosen after the fact:**

* **PRIMARY-A — LON macro-recall** (balanced accuracy, chance 1/3). Chosen because it is the
  statistic `DTAC1_RESULTS.md` §2.3 published, so the honest-cost comparison is like-for-like.
* **PRIMARY-B — LON macro-F1.** Chosen because **R3** showed precision is never computed anywhere in
  the parent's artifacts, and recall alone is *maximised by the very mechanism τ uses* (firing more
  often on the rare class). A criterion a decision rule can game is not a criterion.

**Denominators — both, always, never mixed** (**R2 / R10**):

* **ALL** — 1364 windows.
* **REPRESENTABLE** — the 1232 windows whose longitudinal class the 5-way *label* does not destroy
  into a turn. R2 measured that at τ=0 **all 11 correct `accelerate` predictions fell on the 132
  destroyed windows**, so the ALL-denominator accelerate number is not a claim about recoverability.

**Every per-class row carries recall AND precision AND F1** (**R3**).

**Pre-committed readings:**

| outcome | trigger, fixed now | what it means |
|---|---|---|
| **the patch survives honesty** | OOF `brake_stop` recall ≥ **0.35** on ALL windows **and** the paired CI on Δmacro-recall vs τ=0 **excludes zero** | the free read-out patch is real; publish it as a REF-C read-out, τ selected out-of-fold |
| **the patch is a val artifact** | OOF Δmacro-recall CI **includes zero**, or OOF `brake_stop` recall < 0.15 | the 0.503 was τ fitted to the eval set; retract the headline and say so |
| **partial** | anything between | report the honest number and the cost, claim nothing beyond it |

**Pre-committed regardless of outcome:** the **cost of honesty** =
`(val-optimal-τ score) − (out-of-fold score)`, on both criteria, both denominators. If it is large,
that is the finding.

### 1.4 What a genuinely train-selected τ would need (so this is actionable, not a shrug)

1. A **train** episode cache reachable from a non-training box — either an existing
   `physicalai-train-e438721ae894` epcache relayed to `tanitad-thor`, or ~40 train episodes rebuilt
   there (parity key + skip-hash `f09e44db` must be preserved or the arm is not comparable).
2. One forward pass of `refc-base` over it with `stack/scripts/refc_tactical_probe.py --dump`
   (the same script, unmodified — it already banks the substrate). Minutes on an idle GPU.
3. Re-run `stack/scripts/refc_tactical_tau_select.py --train-substrate <that .pt>`, which **already
   implements** the train-selected path and falls back to LOEO only when it is absent.

⚠️ The train-selected τ will be **biased low**: the model fits its own training windows, so its LON
posterior is sharper there and needs less prior correction. That bias is *not* a reason to prefer
LOEO — it is a reason to report both and say which is which.

---

## 2. T2 — the F1-only arm

**What is implemented** (staged, green, no training launched): `tactical_speed_input` is decoupled
from `factored_maneuver`, so the **shipped 5-way head** can read the ego speed.
`refc_f1only_config()` is the arm; the flag still **defaults OFF**, so a disabled flag is a
byte-identical state_dict and byte-identical outputs and every published REF-C number stays
reproducible.

**MEASURED capacity** (`param_breakdown`, meta device, pinned EXACTLY by
`tests/test_refc_tactical.py::test_f1only_is_not_a_capacity_change`):

| arm | total params | Δ vs base |
|---|---:|---:|
| `refc-base` (shipped) | 104,191,577 | — |
| **`refc-f1only` (new)** | **104,191,961** | **+384 (+0.000369 %)** |
| `refc-factored` | 104,192,474 | +897 (+0.000861 %) |

+384 = exactly `aux_hidden` = one extra input column into `maneuver_head.0`. The decoder is
**bit-identical** (8,634,505 params both arms): the rank-5 graft is untouched.

### 2.1 ⛔ SUCCESS AND FAILURE, FIXED IN ADVANCE (this arm needs a retrain — NOT launched here)

**The lower bound, and why it is only a lower bound.** E-A2 MEASURED `pooled` 0.3833 →
`pooled+v0` **0.4346** macro-recall (+0.051) with a **linear** probe. The parent's own §2.2 warns
that probe understates what `pooled` carries (the trained head reaches AUC 0.708–0.736 where the
linear probe reaches 0.484–0.714), and **R5** further showed those AUCs are fold-pooling artifacts
(the +0.051 delta itself is seed-stable at 0.0513 across 5 seeds). So:

* ⛔ **+0.051 is NOT a prediction and must not be quoted as one.** It is a linear-readout lower
  bound on how much *linearly decodable* longitudinal information `v0` adds to `pooled`.

| outcome | trigger, fixed now | consequence |
|---|---|---|
| **F1 is load-bearing** | `dtac1-f1only` LON macro-recall separated above `refc-base` (paired episode-cluster bootstrap, CI excludes 0) **and** ≥ +0.03 | F1 rides on every subsequent REF-C arm; the parent's demotion of F1 to "nice-to-have" (§6.3 READOUT branch) is **withdrawn** |
| **F1 is inert** | CI includes zero | the input was never the constraint; F1 is dropped from the arm set and `refc_f1only_config` is kept only as a documented negative. `ego_dropout` is then the FIRST thing to check before concluding — the head is trained without the channel on half the samples |
| **F1 hurts** | separated *below* base on LATERAL or LONGITUDINAL | revert; report as a regression, do not re-scope |

**Guard-rails, per the binding four-family rule:** LATERAL must be **not separated** (the input must
not buy longitudinal emission with lateral quality); LONGITUDINAL better-or-not-separated;
STRATEGIC reported so absence of effect is visible; ADE at all horizons as one row of five.

### 2.2 Integration risks named here rather than discovered later

* **Archived one-off scripts call `model.maneuver_head(pooled)` directly** — 5 of them under
  `TanitAD Research Hub/.../incoming/` (`recovery_aug_ft.py`, `encoder_canary.py`, `e2a_localize.py`,
  `gate1_finetune.py`, `refc_floor_driver.py`, `proto_gate1_finetune.py`). They are correct against
  every SHIPPED checkpoint (`tactical_speed_input=False` ⇒ input width unchanged) and would raise a
  shape error against an F1 checkpoint. That is fail-loud, not silent — but it is named.
* **R13 stands and is not fixed by this work:** `maneuver_decision` still collapses the longitudinal
  class on turns, by construction. Downstream readers wanting the longitudinal decision must read
  `lon_decision`, not `maneuver_decision`.

---

## 3. Four metric families — what this work can and cannot report

The τ patch (T1) is a **decode-only** transform. That is now pinned mechanically, not argued:
`tests/test_refc_tactical.py::test_man_prior_tau_cannot_move_the_trajectory` asserts
`traj / anchor_logits / anchor_traj / offset / sel_idx / maneuver_logits / lat_logits / lon_logits /
route_logits / pooled / measurement / ctx` are **bit-identical** across τ = 0 vs τ = 2 at fixed
weights and ONE fixed input, with a vacuity control proving τ is live under that prior.

| family | T1 (τ patch) | why |
|---|---|---|
| **TACTICAL** | **measured, full panel** | the family the change targets |
| **LONGITUDINAL** | **Δ ≡ 0**, and target-speed / headway / time-gap / TTC **not computable from the banked substrate** (it holds logits, pooled, v0, labels, eid — no predicted speed, no lead-agent state), n available = 0 | decode-only + substrate contents |
| **LATERAL** | **Δ ≡ 0**; heading / curvature / yaw-rate / cross-track not computable (no predicted trajectory banked), n = 0. The lateral *classification* readout IS reported and is labelled classification, never kinematics (**R8**) | decode-only + substrate contents |
| **STRATEGIC** | **Δ ≡ 0**; `route_logits` not banked, n = 0 | decode-only + substrate contents |
| **ADE** | **Δ ≡ 0** at every horizon | decode-only |

For T2 (the F1 arm) all five families are **required in full** and are only obtainable after the
retrain — which is why T2 ships as code + a pre-registration, not as a result.

---

## 4. Negative controls — run FIRST, reported before any headline (non-negotiable)

| control | must show |
|---|---|
| **self-consistency, component vs family** | `collapse(lat, lon) == man5` elementwise = 1.0000, and `man5 == man_banked` = 1.0000. A drift here voids every count. |
| **frontier re-derivation** | the parent's published 8-row τ frontier re-derived from the substrate, matching to 4 dp. The headline is re-computed from the artifact, never copied from the table (*quote a run directory, not a number*). |
| **shuffled** | the FULL out-of-fold pipeline on permuted logits must land at chance. A selection procedure that still "wins" under a shuffle is selecting on the class prior. |
| **uniform prior is inert** | with a uniform π, every τ must decode bit-identically to τ=0. |
| **τ = 0 is the identity** | the τ=0 decode must equal the raw argmax. |
| **fold disjointness** | assert, per fold, that no episode of the scoring fold appears in the selection set. The whole claim rests on this, so it is asserted, not assumed. |

---

## 5. Deliverable manifest (all staged, never committed, never pushed)

| artifact | path |
|---|---|
| this pre-registration | `Project Steering/PREREG_D-TAC1B_TAU_SELECTION_AND_F1_ARM.md` |
| F1-only seam + preset | `stack/tanitad/refs/refc.py` (`refc_f1only_config`) |
| trainer CLI decoupling | `stack/scripts/refc_train.py` |
| τ selection + controls | `stack/scripts/refc_tactical_tau_select.py` |
| tests | `stack/tests/test_refc_tactical.py` |
| results + raw JSON | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-dtac1-tactical-head/` |
