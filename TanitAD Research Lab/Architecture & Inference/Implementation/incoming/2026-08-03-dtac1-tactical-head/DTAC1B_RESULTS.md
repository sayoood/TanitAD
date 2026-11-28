# D-TAC1b — the τ was fitted on val. Here is what it is worth when it is not.

**Date:** 2026-08-03 (Europe/Berlin) · **0 GPU-days.** No training launched, no pod touched.
**Pre-registration:** `Project Steering/PREREG_D-TAC1B_TAU_SELECTION_AND_F1_ARM.md`
(sha256 `c12f14a54af8c1a0cafb7093e912fe5549f20fcc285ef1ca8c5c650e6adf9524`, written and hashed
**before** the selection ran).

**⭐ RUN DIRECTORY — quote this, not a number:**
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-dtac1-tactical-head/`
· selection output `dtac1_tau_selection_refc-base-30k.json`
· substrate `dtac1_substrate_refc-base-30k.pt` (**now banked in the repo**, md5
`e7793439a4f390e18daaef07a6d905ac`, identical to the Thor copy — it was previously on one disk).

**Substrate:** `refc-base` step 29999, canonical val `physicalai-val-0c5f7dac3b11`,
**39 episodes / 1364 windows**, stride 5, decoder steps 2. Everything below is a pure re-analysis
of already-emitted logits: CPU, seconds, no GPU.

**Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, unit = episode, B = 4000, callable
reducers for macro-recall / macro-F1. ⛔ `overlapping_holdout_se` never called.

⚠️ **Denominator note, carried in every table.** All longitudinal numbers here are on the **3-way
LON label**. The parent's headline `0.026` was on the **5-way** label — a different denominator
(`ADVERSARIAL_RECORD.md` **R10**). The matched τ=0 baseline is **0.0719**, not 0.026.

---

## 0. Headline

1. **The τ-on-val defect is REAL but it was NOT load-bearing.** Selected leave-one-**episode**-out —
   every window decoded by a rule fitted without it *or any window from its clip* — the modal τ under
   macro-F1 is **0.50 in 36 / 39 folds (92.3 %)**: exactly the τ the parent read off val. The cost of
   honesty is **1.49 – 3.16 % relative**, and the paired OOF-vs-val-optimal CI **includes zero in all
   8 comparisons**. The parent flagged the right defect and it turned out to be cheap.
2. **The pre-registered "survives honesty" trigger is MET on both criteria.** OOF `brake_stop` recall
   ≥ 0.35 on ALL windows and Δmacro-recall vs τ=0 separated: `+0.1069 [+0.0381, +0.1749]`
   (macro-recall rule) and `+0.0719 [+0.0200, +0.1286]` (macro-F1 rule).
3. ⛔ **But the headline `0.026 → 0.503` does not survive, for a reason that is NOT τ-fitting.**
   Honest OOF brake recall is **0.4248** (ALL) / **0.3333** (label-representable), at precision
   **0.1711 / 0.1287** — 380 fires against 153 true. Precision was never computed anywhere in the
   parent's artifacts (**R3**); it is the whole story.
4. ⭐ **NOT ANTICIPATED — on the 1232 windows the 5-way label CAN represent, the free read-out patch
   is statistically indistinguishable from doing nothing.** Δmacro-F1 `+0.0107 [−0.0418, +0.0665]`,
   Δmacro-recall `+0.0213 [−0.0256, +0.0726]` — **neither separated**. The apparent gain lives on the
   132 windows the label destroys, which is the same set the parent already called irrecoverable.
5. ⭐ **NOT ANTICIPATED — selecting on macro-recall is self-defeating.** It buys Δmacro-recall
   `+0.1069` (separated) for Δmacro-F1 **`−0.0006 [−0.0922, +0.0788]`** and a *separated* accuracy
   loss of **−0.2757 [−0.3851, −0.1745]**. Recall bought with an equal-and-opposite precision loss is
   not an improvement, and only PRIMARY-B could see that.
6. **The instrument's empirical null is 0.3678, not 1/3.** The full pipeline on *shuffled* logits
   scores macro-recall **0.3678** / macro-F1 0.3307. Every macro-recall here must be read against
   0.3678 (same class as **R4**).
7. **T2 done:** the ego-speed input is decoupled from the factorisation, so the **shipped 5-way head**
   can read it. `refc_f1only_config()` = **+384 params (+0.000369 %)**, MEASURED, decoder bit-identical.

---

## 1. Controls — run and reported FIRST

| control | result | reading |
|---|---|---|
| **self-consistency, component vs family** — `collapse(lat, lon) == man5` | **1.0000** over 1364 | the factorised labels and the 5-way label are one surface |
| **label source** — derived vs the epcache's banked `maneuvers` | **1.0000** | the two mints have not drifted |
| **frontier re-derivation** — the parent's published 8-row τ frontier recomputed FROM the substrate | **max abs deviation 0.0001, matches to 4 dp** | the substrate and the published table are the same run. The headline is recomputed, never copied. |
| **uniform prior is inert** | identical decode at every τ in the grid | a run that never updates its prior can never silently change a decode |
| **τ = 0 is the raw argmax** | identical | the baseline is the shipped behaviour, exactly |
| **fold disjointness** | asserted per fold inside `loeo()` | the entire out-of-fold claim rests on it, so it is asserted, not assumed |
| ⭐ **shuffled — the FULL out-of-fold pipeline on permuted logits** | macro-recall **0.3678**, macro-F1 0.3307, accuracy 0.5139, modal τ 0.5 | **the empirical null is 0.3678, not the nominal 0.3333.** The selection procedure itself extracts ~0.034 of macro-recall from the class prior alone. |

Confirmed en route: the val LON marginal is `brake 0.1122 / steady 0.7104 / accel 0.1774` — i.e.
**R6 is confirmed**: the parent's "prior-corrected" decode divided by the marginal of the very set it
reported on. Across the 39 LOEO folds that prior is stable (brake 0.1008–0.1151), which is why the
cost of honesty is small.

---

## 2. The honest τ — protocol, and what it selects

**Protocol.** Leave-one-episode-out inside val. For episode `e`: fit `π_e` and `τ_e` on the windows
**not** in `e`, decode only `e`'s windows. Pooled over 39 folds → an out-of-fold estimate on the same
1364 windows.

⛔ **This is NOT a train-selected τ and is never called one.** A train-selected τ needs REF-C's
posteriors on TRAIN windows. **MEASURED absent, 5 probes at 4 paths** (§4). What LOEO removes is
*"τ was read off the number being reported"*. What it does not remove is *"selection and reporting
share a distribution"*. Both statements travel with every number below.

| criterion (fixed in advance) | modal OOF τ | fold agreement | τ range | π range (brake) |
|---|---:|---:|---|---|
| **PRIMARY-A macro-recall** | **1.00** | 29/39 = 74.4 % | {0.75, 1.00, 1.25} | 0.1008 – 0.1151 |
| **PRIMARY-B macro-F1** | **0.50** | **36/39 = 92.3 %** | {0.25, 0.50} | 0.1008 – 0.1151 |

PRIMARY-B's modal τ is **the parent's published τ = 0.5**. The τ was not badly overfit.

---

## 3. TACTICAL family — the full panel, both denominators, recall AND precision AND F1

**ALL windows (n = 1364).** LON axis; `n true` = brake 153 / steady 969 / accelerate 242.

| rule | acc | macro-R | macro-F1 | brake rec / prec / F1 | steady rec / prec | accel rec / prec / F1 |
|---|---:|---:|---:|---|---|---|
| **τ = 0 (shipped)** | 0.7045 | 0.3621 | 0.3409 | 0.0719 / 0.2340 / 0.110 | 0.9690 / 0.7229 | 0.0455 / 0.6111 / 0.085 |
| **OOF, macro-F1 rule (τ*=0.5)** | 0.5916 | 0.4341 | **0.4064** | **0.4248** / 0.1711 / 0.244 | 0.7286 / 0.7879 | 0.1488 / 0.4091 / 0.218 |
| **OOF, macro-R rule (τ*=1.0)** | 0.4289 | **0.4690** | 0.3402 | 0.8693 / **0.1597** / 0.270 | 0.4427 / 0.8720 | 0.0950 / 0.5897 / 0.164 |
| *val-optimal oracle (τ=0.5)* | 0.5953 | 0.4588 | 0.4190 | 0.5033 / 0.1935 / 0.280 | 0.7203 / 0.8041 | 0.1529 / 0.3776 / 0.218 |

**REPRESENTABLE windows only (n = 1232 — the 132 label-destroyed removed, R2/R10).**
`n true` = brake 117 / steady 969 / accelerate 146.

| rule | acc | macro-R | macro-F1 | brake rec / prec | accel rec / prec |
|---|---:|---:|---:|---|---|
| **τ = 0 (shipped)** | 0.7695 | 0.3487 | 0.3298 | 0.0769 / 0.2812 | **0.0000 / 0.0000** |
| **OOF, macro-F1 rule** | 0.6104 | 0.3700 | 0.3406 | 0.3333 / 0.1287 | 0.0479 / 0.1228 |
| **OOF, macro-R rule** | 0.4367 | 0.4519 | 0.3170 | 0.8376 / 0.1367 | 0.0753 / 0.4074 |

⭐ **R2 is reproduced exactly:** at τ = 0 on the representable subset, `accelerate` recall is
**0.0000** — every correct accelerate prediction in the ALL table sits on a label-destroyed window.

### 3.1 Paired episode-cluster bootstrap, B = 4000, 39 episodes — Δ vs τ = 0

| selected by | denominator | Δ macro-recall | Δ macro-F1 | Δ accuracy |
|---|---|---|---|---|
| **macro-F1 (τ*=0.5)** | ALL | **+0.0719 [+0.0200, +0.1286]** ✅ sep | +0.0655 [−0.0029, +0.1312] ✗ | −0.1129 [−0.1861, −0.0471] ⛔ sep worse |
| **macro-F1 (τ*=0.5)** | REPRESENTABLE | +0.0213 [−0.0256, +0.0726] ✗ | **+0.0107 [−0.0418, +0.0665]** ✗ | −0.1591 [−0.2385, −0.0878] ⛔ sep worse |
| **macro-recall (τ*=1.0)** | ALL | **+0.1069 [+0.0381, +0.1749]** ✅ sep | **−0.0006 [−0.0922, +0.0788]** ✗ | −0.2757 [−0.3851, −0.1745] ⛔ sep worse |
| **macro-recall (τ*=1.0)** | REPRESENTABLE | +0.1032 [+0.0289, +0.1754] ✅ sep | −0.0128 [−0.0931, +0.0625] ✗ | −0.3328 [−0.4432, −0.2290] ⛔ sep worse |

### 3.2 The cost of honesty — Δ vs the val-optimal oracle

| criterion | denominator | val-optimal | out-of-fold | cost | cost % | paired CI |
|---|---|---:|---:|---:|---:|---|
| macro-recall | ALL | 0.4761 | 0.4690 | 0.0071 | 1.49 % | −0.0071 [−0.0235, +0.0085] ✗ |
| macro-recall | REPRESENTABLE | 0.4654 | 0.4519 | 0.0135 | 2.90 % | −0.0136 [−0.0302, +0.0025] ✗ |
| macro-F1 | ALL | 0.4190 | 0.4064 | 0.0126 | 3.01 % | −0.0126 [−0.0426, +0.0130] ✗ |
| macro-F1 | REPRESENTABLE | 0.3517 | 0.3406 | 0.0111 | 3.16 % | −0.0112 [−0.0329, +0.0029] ✗ |

**No comparison separates.** ⇒ **Fitting τ on val cost between 1.5 % and 3.2 % relative, and that
cost is inside the noise.** Reported as pre-committed regardless of direction; the direction is
"cheap", and that is the answer to Task 1's *"how much does the honest τ cost?"*.

### 3.3 Adjudication against the pre-committed table

| pre-committed branch | trigger | outcome |
|---|---|---|
| **the patch survives honesty** | OOF brake recall ≥ 0.35 on ALL **and** Δmacro-recall CI excludes 0 | ✅ **MET on both criteria** (0.4248 and 0.8693; both Δmacro-recall separated) |
| the patch is a val artifact | Δ CI includes 0, or brake recall < 0.15 | not triggered |
| **cost of honesty** (pre-committed as reported regardless) | — | **1.49 – 3.16 % relative, not separated** |

⚠️ **NOT anticipated by the pre-registration, and stated as such rather than folded in:** the
REPRESENTABLE-denominator null (§0.4) and the macro-recall self-defeat (§0.5). Neither branch of the
pre-committed table covers them; they are new findings, and they are why the honest recommendation
below is weaker than "the patch survives" alone would suggest.

---

## 4. ⛔ Why this is NOT a train-selected τ — the absence, measured

Five probes, four paths, two hosts. `tanitad-thor` is the only non-training box available
(`tanitad-new` runs v5f and `tanitad-pod4` runs v1arch — off-limits by brief; `tanitad-pod` and
`tanitad-pod3` both answer `Connection refused`).

1. `ls -d /home/nvidia/*data* /home/nvidia/*train* /home/nvidia/*cache*` → only `valdata`
2. `find / -xdev -maxdepth 6 -name "physicalai-train*"` → nothing
3. `find /home/nvidia /mnt /data -maxdepth 4 -name "ep_*.pt"` → only `valdata/physicalai-val-0c5f7dac3b11`
4. the banked substrate itself carries **val windows only** (`val_dir` field)
5. `refc-base` predates `factored_maneuver` ⇒ **no `lat_log_prior` / `lon_log_prior` EMA buffer** in
   the checkpoint — the second route the brief named does not exist for this arm

**What would make it computable** (the path is implemented, only the data is missing):

1. a train epcache reachable from a non-training box — relay `physicalai-train-e438721ae894` to Thor,
   or rebuild ~40 train episodes there **preserving parity key + skip-hash `f09e44db`**;
2. one forward pass with the **unmodified** `stack/scripts/refc_tactical_probe.py --dump`;
3. `refc_tactical_tau_select.py --train-substrate <that .pt>` — the train-selected branch is already
   in the script and LOEO is only its fallback.

⚠️ A train-selected τ will be **biased low** (the model fits its own training windows, so its LON
posterior is sharper there and needs less correction). That is a reason to report both, not to
prefer LOEO.

---

## 5. FOUR METRIC FAMILIES — per family, with reason and n

The τ patch is a **post-hoc argmax on already-emitted logits**. That is now pinned mechanically:
`tests/test_refc_tactical.py::test_man_prior_tau_cannot_move_the_trajectory` asserts
`traj / anchor_logits / anchor_traj / offset / sel_idx / maneuver_logits / lat_logits / lon_logits /
route_logits / pooled / measurement / ctx` are **bit-identical** across τ = 0 vs τ = 2 at fixed
weights and **one** fixed input — with a vacuity control proving τ is live under that prior. (An
earlier adversarial pass saw a 0.0039 "leak" here; it was confounded by drawing `v0` twice.)

| family | Δ under the τ patch | level, and why it is or is not computable | n |
|---|---|---|---|
| **TACTICAL** | **MEASURED in full**, §3 | recall + precision + F1 + confusion, both denominators, paired bootstrap | 1364 / 1232 |
| **LONGITUDINAL** | **EXACTLY ZERO** (pinned by the test above) | target-speed accuracy and headway / time-gap / TTC need a **predicted speed** and **lead-agent state**; the substrate banks logits, pooled, v0, labels, eid only. Not a shortfall of this analysis — the trajectory is provably unchanged. | 0 |
| **LATERAL** | **EXACTLY ZERO** | heading / curvature / yaw-rate / cross-track need a predicted trajectory, not banked. The lateral **classification** readout is reported below and is **not** a substitute (**R8**). | 0 |
| **STRATEGIC** | **EXACTLY ZERO** | `route_logits` not banked; REF-C also evaluates with `nav_cmd=None` | 0 |
| **ADE** | **EXACTLY ZERO** at every horizon | decode-only | 0 |

**LATERAL readout — CLASSIFICATION, explicitly not kinematics:** accuracy 0.9348, macro-recall
0.8290, macro-F1 0.8665; `lane_keep` 0.9806 rec / 0.9431 prec, `turn_left` **0.7816 rec / 0.8889
prec** (153 predicted vs 174 true), `turn_right` **0.7248 / 0.9080** (87 vs 109).
⇒ **R8 is reproduced**: under the lateral readout the turns are **under-predicted by 12.1 % and
20.2 %**, not "calibrated". The τ patch does not touch this axis.

---

## 6. T2 — the ego-speed input, decoupled (the second defect)

**The defect as found:** `tactical_speed_input` existed but **raised `ValueError` unless
`factored_maneuver` was also on**. So the shipped 5-way head could never read the ego speed, and the
pre-registered arm set (`dtac1-full` = F1+F2+F3, `dtac1-f2only` = F2) contained **no arm isolating
F1**. F1's only estimate would have been `full − f2only`, in which the two arms *also* differ in the
head itself (a shared trunk with two 3-way readouts vs a 2-layer MLP with one 5-way readout).

**What is implemented:** the flag is now independent. `refc_f1only_config()` is the INPUT-only arm —
same head, same 5-way label, same CE, same rank-5 graft, same decode; **one extra input column.**

**MEASURED capacity** (`param_breakdown`, meta device):

| arm | total | Δ vs base | aux | decoder |
|---|---:|---:|---:|---:|
| `refc-base` (shipped) | 104,191,577 | — | 274,760 | 8,634,505 |
| **`refc-f1only` (new)** | **104,191,961** | **+384 (+0.000369 %)** | 275,144 | **8,634,505** (bit-identical) |
| `refc-factored` | 104,192,474 | +897 (+0.000861 %) | 275,529 | 8,634,633 |

+384 = exactly `aux_hidden`. **Less than half the factored arm's +897 and ~1/700 of the +272,001 the
first two-MLP attempt cost.** Pinned EXACTLY (not as a band) by
`test_f1only_is_not_a_capacity_change`.

**Gated-flag discipline preserved:** default OFF ⇒ same state_dict keys, one shape difference only
when ON, byte-identical outputs when OFF. Verified by loading, not by exit code.

**Tests added (24 → 28 in this module):**
`test_speed_input_on_the_SHIPPED_5way_head_is_live_and_gated` (negative control **first**: with the
flag off, `maneuver_logits` is **bit-identical** across v0 = 0 → 25 m/s — that *is* the defect; with
it on they move, and `anchor_logits` moves too, so the speed reaches SELECTION);
`test_f1only_is_the_shipped_head_widened_by_exactly_one_column` (same key set both ways, exactly one
shape difference, speed column receives gradient); `test_f1only_is_not_a_capacity_change`;
`test_man_prior_tau_cannot_move_the_trajectory`; `test_trainer_end_to_end_f1only`;
`test_cli_rejects_the_orphan_DECODE_lever_only`.

⚠️ **`--man-prior-tau` still requires `--factored-maneuver`** — it adjusts the per-axis priors, which
only the factored seam registers. Only the INPUT lever was freed.

⚠️ **Pre-registered, NOT measured — E-A2's +0.051 is a LOWER BOUND, not a prediction.** It is a
*linear* probe on `pooled` vs `pooled+v0`; the parent's own §2.2 warns the probe understates what
`pooled` carries, and **R5** showed its per-class AUCs are fold-pooling artifacts (only the +0.051
delta is seed-stable, 0.0513 across 5 seeds). Success/failure thresholds for the F1 arm are fixed in
`PREREG_D-TAC1B` §2.1 and require a retrain — **not launched by this stream**.

**Integration risk, named here rather than discovered later:** six archived one-off scripts under
`TanitAD Research Hub/.../incoming/` call `model.maneuver_head(pooled)` directly. They remain correct
against every shipped checkpoint (flag off ⇒ input width unchanged) and would raise a **shape error**
— fail-loud, not silent — against an F1 checkpoint.

---

## 7. Recommendation

1. **Do NOT publish the prior-corrected decode as a default.** It is not free: at the honest τ = 0.5
   it costs a *separated* accuracy loss of −0.1129 [−0.1861, −0.0471] and brake precision 0.1711, and
   on the label-representable windows it is **not separated from doing nothing**. Ship it as an
   **optional reporting mode** with its precision and accuracy cost attached.
2. **If it is used, use τ = 0.5 selected by macro-F1** (36/39 folds), never τ = 1.0 by macro-recall —
   that rule is separated *worse* on accuracy and gains nothing in F1.
3. **The retrain justification is UNCHANGED and is not weakened by any of this.** It never rested on
   brake reporting: it rests on the 132 label-destroyed windows (9.68 %), on `accelerate`, and on the
   `lon_to_anchor` selection graft. §3's representable-denominator null *strengthens* that case — the
   decode rule cannot reach what the label destroyed.
4. **Add `dtac1-f1only` to the arm set** (`PREREG_D-TAC1B` §2.1). It is +384 params and it is the only
   way F1 is attributable.
5. **A train-selected τ remains open** and is now a data-movement task, not a code task (§4).

---

## 8. Deliverable manifest — every artifact and where it lives

| artifact | where | state |
|---|---|---|
| pre-registration (hashed before the run) | `Project Steering/PREREG_D-TAC1B_TAU_SELECTION_AND_F1_ARM.md` | repo, staged |
| this results file | `…/incoming/2026-08-03-dtac1-tactical-head/DTAC1B_RESULTS.md` | repo, staged |
| raw selection output | `…/incoming/2026-08-03-dtac1-tactical-head/dtac1_tau_selection_refc-base-30k.json` | repo, staged |
| **banked substrate (was Thor-only)** | `…/incoming/2026-08-03-dtac1-tactical-head/dtac1_substrate_refc-base-30k.pt` | repo, staged — md5 `e7793439a4f390e18daaef07a6d905ac`, verified identical to `tanitad-thor:/home/nvidia/TanitAD/taniteval/results/dtac1_substrate_refc-base-30k.pt` |
| τ-selection instrument + controls | `stack/scripts/refc_tactical_tau_select.py` | repo, staged |
| F1-only seam + `refc_f1only_config` | `stack/tanitad/refs/refc.py` | repo, staged |
| trainer CLI decoupling | `stack/scripts/refc_train.py` | repo, staged |
| tests (28 in module) | `stack/tests/test_refc_tactical.py` | repo, staged |

**Nothing lives only on a pod, only on Thor, or only in an agent's context.** The substrate that was
previously banked on Thor alone is now in the repo with a verified md5.
