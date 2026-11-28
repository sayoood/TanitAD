# E-GOAL-3 — PRE-REGISTRATION

**Written BEFORE any number in `EGOAL_3.md` was computed.** Amendments are appended in §10 with a
reason; nothing above §10 is edited or deleted after the first measurement lands.

**Stream:** `2026-07-28-egoal-3-trained-head` · **Wall-clock date:** 2026-07-27
*(the folder carries the program's narrative date, which runs ahead of wall-clock — flagged, not
silently absorbed).*
**Hosts:** `tanitad-pod2` (A40 idle; 8 CPU shards of a wide-FOV build are running — CPU capped and
`nice`d, GPU untouched, no memory pressure added) + dev box CPU where noted.
⛔ **pod1 is TRAINING `flagship-v2corpus-30k` and is NEVER contacted.**
**Estimator:** paired episode-cluster bootstrap, `taniteval/taniteval/ci.py`, **B = 2000**,
unit = **the val episode**. `overlapping_holdout_se` is **never called**.

---

## 1. The question, stated so that it can come back NO

E-GOAL-1 and E-GOAL-2 both measured a **resampled residual**: the true 2 s along-track goal plus a
residual drawn from a head's out-of-fold error pool. Both refused to license it in the same words:

> *"+25.4 % is a RESAMPLED RESIDUAL, not a trained head."*

A resampler assumes the head's error structure **transfers**. It cannot represent the one thing that
matters most: **a real head's error correlates with which window is hard.** A trained head can do
**better** than the resampler (it can learn structure the resampler cannot represent) or **worse**
(its errors may concentrate on exactly the windows where the pick matters).

**E-GOAL-3 trains the head and feeds its ACTUAL per-window predictions through the ACTUAL REF-C
selection rule.** No residual is resampled on the along axis.

---

## 2. The deployment, and the background it is measured against

### 2.1 Deployment (fixed, inherited, re-derived before use)

| | value | class |
|---|---:|---|
| val build | `physicalai-val-0c5f7dac3b11`, **600 episodes** | fixed |
| fan | `fan_refc-xl-30k_600ep.pt` (E-GOAL-2, committed `refc_rerank.dump()` unmodified) | fixed |
| windows / clusters | **13 198 / 600** | `INHERITED` → gate **F-0** |
| as-trained `a0` | **0.5015** | `INHERITED` → gate **F-0** |
| true-2 s-goal `R_goal2s` | **0.1933** | `INHERITED` → gate **F-0** |
| `oracle_in_fan` | **0.1547** | `INHERITED` → gate **F-0** |
| **headroom** = `a0 − R_goal2s` | **0.3082** | the recovery denominator |

Source: `…/2026-07-28-egoal-2-power/raw/e2_place_n600_parent_resampled.json` — **the raw JSON, never
the doc table.**

### 2.2 ⛔ C30 — THE CROSS-TRACK BACKGROUND IS NAMED IN ADVANCE AND HELD FIXED

E-GOAL-2 measured a **15.9-recovery-point swing and a separation flip** at fixed `n` and fixed
along-track error, purely on which learned cross-track sits in the background.

> **PRIMARY BACKGROUND: `parent_resampled`.** E-GOAL-2's registered conservative carrier — the parent
> head's own 881 cross-track residuals resampled i.i.d. onto the true cross-track, **drawn once per
> seed from `np.random.default_rng(5000 + s)`, `s ∈ [0,16)`, and IDENTICAL ACROSS EVERY ARM within a
> seed.** A background that differed per arm would be a treatment, not a background.
> **Every headline number in `EGOAL_3.md` is quoted under this background and says so.**

Secondary backgrounds, reported but never headline: **`sel`** (the selector's own 2 s endpoint, zero
fitting) and **`reduced`** (ridge on the fan-only blocks). All three are run on every arm.

---

## 3. The treatment — the trained head

**Target.** `y_long` = the 2 s **along-track** displacement in the ego frame at the window's last
observed frame = `gt[:, -1, 0]` of the fan dump.

**Features — PAST ONLY, derived from `poses[T,4] = (x, y, yaw, speed)` at 10 Hz.** Ten columns,
mirroring `lead_state_gate.EGO_COLS` (the family E-GOAL-1/2 identified), with `L` = the window's last
observed index and `DT = 0.1 s`:

| column | index expression | past? |
|---|---|---|
| `v` | `s[L]` | ✅ |
| `ax` | `(s[L] − s[L−1]) / DT` | ✅ |
| `yawrate` | `wrap(yaw[L] − yaw[L−2]) / (2·DT)` | ✅ |
| `curv` | `yawrate / max(v, 0.5)` | ✅ |
| `abs_curv` | `|curv|` | ✅ |
| `ay` | `v · yawrate` | ✅ |
| `dv_0p5` | `s[L] − s[L−5]` | ✅ |
| `dv_1p0` | `s[L] − s[L−10]` | ✅ |
| `v_lag_0p5` | `s[L−5]` | ✅ |
| `v_lag_1p0` | `s[L−10]` | ✅ |

⛔ **Every index is CLAMPED at 0.** A negative Python index wraps to the END of the array — i.e. it
would silently read the FUTURE. This is the exact hazard the brief names; it fires on the first
window of every episode (`L = 7`, `L−10 = −3`). The clamp count is reported.

**Regressor.** `HistGradientBoostingRegressor`, **hyper-parameters imported verbatim** from
`eg_fit.fit_predict` (`max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
l2_regularization=1.0, early_stopping=False, random_state=0`), **identical for every arm**, so extra
columns can only help an arm through held-out generalisation.

### 3.1 Arms — every one fitted with the identical regressor and folds

| arm | features | role |
|---|---|---|
| ⭐ **`H_ego`** | all 10 | **THE TREATMENT** |
| `H_nohist` | 6 (history dropped) | the honest reference the noise arm must land on |
| ⛔ **`H_noise_hist`** | 10, the 4 history columns **replaced by Gaussian noise matched to each column's own mean and SD** | **capacity fixed, information removed** |
| `H_v0` | `v` only | the parent head's ego content |
| `CV_head` | none — `ŷ = v · 2.0` | constant velocity, no fit |
| ⛔ **`N_SHUF`** | all 10, **permuted across episodes** | **deliberately failing input** |
| **`P_ORACLE_TRUE`** | — `ŷ = y_long` | **positive control** |

### 3.2 Training configuration — two, both reported

| tag | fit on | predicts | leakage surface |
|---|---|---|---|
| ⭐ **T-OOF** *(the brief's literal ask)* | 5 **episode-disjoint** folds over the **600 val episodes** (`eg_common.clip_folds`, seed 0) | all 13 198 windows, out-of-fold | none by construction |
| ⭐ **T-TRAIN** *(the deployable configuration)* | the **parity train corpus** `physicalai-train-e438721ae894` (2376 episodes, dense grid) | all 13 198 val windows | **zero val episodes in training** — verified by content fingerprint (§5) |

**Which is the headline:** `H_ego | T-OOF | parent_resampled`. T-TRAIN is reported beside it as the
deployable configuration and as an independent replication.

---

## 4. ⭐ THE PRE-REGISTERED DECISION RULE, and the input that makes each one FIRE

Read on **`H_ego | T-OOF | parent_resampled`**, both resamplers (`iid` and `by_speed`).

| verdict | condition | what input makes it fire — **and is it reachable?** |
|---|---|---|
| ⭐ **CONFIRM** | separated-BETTER on **both** resamplers **AND** recovery ≥ **+19.1 %** (= 0.75 × E-GOAL-2's +25.4 %) | reachable: E-GOAL-2's `P_ORACLE` returns +54.6 % on this exact deployment and background, so the axis can produce a large separated positive. |
| **PARTIAL** | separated-better on both resamplers but recovery **< +19.1 %** | ⭐ reachable: **`E1_nohist` returns +9.1 % separated-better on this exact deployment and background.** A trained head no better than a no-history head lands here. |
| ⛔ **REFUTE** | **not** separated-better on both resamplers (CI touches 0, or separated-worse) | ⭐ reachable: `E0_v0` (−54.3 %) and `CV` (−44.8 %) are **separated-WORSE** on this exact deployment and background. If a trained head's errors concentrate on the windows that matter, `H_ego` lands here. **REFUTE is live and will be stated plainly, without re-scoping.** |

**If REFUTE fires**, `EGOAL_3.md` states that the resampled estimate **did not transfer** and that the
program's only positive selection lever is retired. No scope reduction, no substitute claim.

## 4.1 ⛔ C31 — THE NEGATIVE CONTROL IS RE-RUN AT n = 600, AND THE MECHANISM RESTS ON A DIRECT CONTRAST

E-GOAL-2 found that at n = 600 **its own pre-registered separation predicate stops discriminating** —
an information-free arm separates at +9.1 %. A control validated at n = 40 is not validated at n = 600.

Therefore, registered in advance, and each measured at **my actual n = 600**:

| contrast (paired, same windows, same background) | required |
|---|---|
| ⭐ **`H_ego` vs `H_noise_hist`** — history vs **noise in the same columns**, capacity fixed | separated with `H_ego` better, if speed history is the lever |
| ⛔ **`H_nohist` vs `H_noise_hist`** | **MUST BE NULL** — real absence and fake presence must be indistinguishable |
| `H_ego` vs `H_v0` | separated-better |
| ⛔ **`N_SHUF` vs as-trained** | **separated-WORSE at n = 600** — a deliberately failing input that must still fail at my n |
| `H_noise_hist` vs as-trained | **reported as the C31 diagnostic**: whether the separation predicate itself still discriminates at my n |

**The mechanism claim (`speed history is the lever`) is made ONLY on the direct contrast, never on
"`H_ego` is separated".**

---

## 5. ⛔ THE LEAK / OVERLAP CHECK — BY CONTENT, ON THE PATH I ACTUALLY READ

Priority 2 (after the future-content audit). **`episode_id` is not a usable key** (the parity train
corpus has 2376 episodes and 2342 unique ids) and **filenames collide completely** across caches.

**Method:** sha256 over the raw `poses[T,4]` float32 bytes, `torch.load(mmap=True)`.

| check | paths | required |
|---|---|---|
| **A** — my T-TRAIN training set × my scoring set | `_epcache/physicalai-train-e438721ae894` (2376) × `/root/valdata/physicalai-val-0c5f7dac3b11` (600) | **overlap = 0** |
| **B** — the path identity | the val path my **feature builder** reads must be the **same path the fan dump reads** (`refc_rerank.VAL`) | identical string, and fingerprints identical |
| **C** — filename contrast | the same comparison by filename | **reported** — E-GOAL-2 measured 600/600 filename overlap against 0/600 real overlap |

**The overlap count is reported even if it is zero, together with the exact path checked.** This
matters more here than in E-GOAL-2 because E-GOAL-3 **actually trains on the train corpus.**

⛔ **Parity is sacred.** `_epcache` is never written; no episode is re-selected; the parity corpus is
read **only** to extract poses and to fingerprint it.

---

## 6. ⛔ C23 — THE FUTURE-CONTENT AUDIT, BY DEFINITION AND EMPIRICALLY. **PRIORITY 1.**

`head_deg` in the fan dumps is the **FUTURE net heading change** and sits beside `v0`
(`driving_diagnostic.net_heading_change_deg` reads `poses[last + horizon, 2]`). **A trained head is
far more exposed to this than a resampler was.** Three instruments, all run **before any fit**:

1. **By definition** — the index table in §3, from the producing expression, not the column name.
2. ⭐ **The `future_blind` test** — recompute every feature from a pose array whose rows **> L** have
   been overwritten with garbage, and require the features to be **bit-identical**
   (`max |Δ| = 0.0`). A feature that reads the future changes.
   **Its failing half:** the *target* `y_long` must **CHANGE** under the same corruption — otherwise
   the instrument has no power and proves nothing.
3. **Negative-index guard** — `assert min(index) >= 0` for every feature index, plus the count of
   windows that hit the clamp.

⚠️ Nothing derived from `head_deg`, `a_gt`, `v_target`, `vt_*` or any `gt`-derived quantity is fed to
any arm; `eg_common.assert_no_oracle` is called on every feature name list.

---

## 7. FIDELITY GATES — assume my scoring code has a bug until one of these says otherwise

> *"Three streams in two days found bugs in their own scoring code producing stable, plausible, wrong
> numbers."*

| gate | what it proves | abort condition |
|---|---|---|
| **F-0** | the deployment quantities re-derive from the fan | any of `a0` / `R_goal2s` / `oracle_in_fan` / headroom off by ≥ 5×10⁻⁴ |
| ⭐ **F-1** | **my placement engine reproduces E-GOAL-2's n = 600 result**: E-GOAL-2's frozen `E1_ego` resampled pool, pushed through MY code under `parent_resampled`, must return **−0.0784 [−0.0960, −0.0606], +25.4 %** | deviation ≥ **0.010 recovery points** |
| ⭐ **F-2** | **per-row identity of the (episode, L) join**: my `v` recomputed from poses must equal the dump's `v0` **exactly**, and my `y_long` must equal `gt[:,-1,0]` exactly | `max |Δ| > 0` on either |
| ⛔ **F-3** | **F-2 has power**: the same comparison with L shifted by +1 must FAIL | F-3 passes |
| **F-4** | per-episode window counts reconstructed from `T` match the dump's `eid` run-lengths | any mismatch |

`realise` / `pick_nearest_to` / `goal_reference` / `ade` are **IMPORTED** from
`…/2026-07-27-egoal-1-lead-vehicle/code/eg_place.py`, never re-implemented.

---

## 8. ⛔ C24 — THE FAMILY IS MEASURED ON THE ESTIMATOR, NOT ASSUMED

The measured heads are **near-unbiased (α ≈ 0.996 — not `SHRINK`)** and **heavy-tailed
(RMS/MAE 1.867 — not `ISO`)**. At n = 600 the family-matched σ₀ is **1.2195 m**; a head at 0.9305 m
**clears it by 1.31× while FAILING the inherited `ISO` 0.813 bar by 1.14×** — same number, opposite
verdicts.

So: **RMS/MAE ratio, shrinkage α and bias are measured on the trained head's own residuals and the
family is NAMED.** The requirement curve is rebuilt from **the trained head's own residual pool**,
scaled and run through the real rule.
⚠️ **The curve is a companion, not the verdict.** With a real head the realised recovery is measured
directly, so **no verdict here is taken from a curve read.** Both the family-matched σ₀ and the
inherited `ISO` 0.813 m are reported side by side.

---

## 9. WHAT I WILL NOT DO

- Not re-select episodes. Parity (`physicalai-train-e438721ae894`, skip-hash `f09e44db`) is sacred;
  anything that re-selects is **REFUSED**.
- Not quote a recovery number without its cross-track background.
- Not read a verdict off a requirement curve.
- Not call `overlapping_holdout_se`, and not quote a `heldout` split-mean.
- Not touch pod1. Not add GPU or memory pressure to a busy pod.
- Not put a clip UUID or raw PhysicalAI content into any artifact.
- Not commit and not push. `git add` only.

---

## 10. AMENDMENTS (appended after the fact, with reasons; nothing above is edited)

### A1 — the speed-differencing ladder (added AFTER the primary placement landed)

**Reason.** The registered primary returned `H_ego` vs `H_nohist` = **−0.0029 [−0.0052, −0.0006]**,
i.e. **≈ 1 recovery point of 46.3**, against E-GOAL-2's *"64 % of the recovery is speed history"*.
Inspecting my own feature table showed why the two are not measuring the same thing: **`H_nohist`
still contains `ax = (v[L] − v[L−1]) / DT`, which IS 0.1 s of speed history.** So my registered
ablation asks *"what do the 0.5–1.0 s lags add?"*, **not** *"what does speed history add?"* — and
E-GOAL-2's claim turns on the second question.

**What is added** — two arms, same regressor, same folds, same hyper-parameters, no re-tuning:

| arm | features | isolates |
|---|---|---|
| `H_inst` | `v, ay, curv, abs_curv, yawrate` | **instantaneous state with NO speed differencing of any length** |
| `H_v0_ax` | `v, ax` | **0.1 s of speed history, alone** |

giving the ladder `H_v0` → `H_v0_ax` → `H_inst` → `H_nohist` → `H_ego`, plus the contrast
`H_nohist` vs `H_inst` (**speed differencing vs none**).

**⛔ This amendment does not touch the registered verdict rule of §4**, which is read on
`H_ego | T-OOF | parent_resampled` and was already decided by the primary run. It refines the
**mechanism** question of §4.1 only. Both possible outcomes are committed in advance:
- if `H_inst` ≈ `H_v0` (poor), **speed differencing is the lever** and E-GOAL-2's mechanism claim
  survives in kind while its *size* (64 %) is wrong;
- if `H_inst` ≈ `H_nohist` (good), **speed history is NOT the lever at all** and E-GOAL-2's mechanism
  claim is refuted outright, not merely resized.

### A2 — ⚠️ the "both resamplers" clause of §4 is INAPPLICABLE, and is replaced, not quietly dropped

**Reason.** §4 requires separation *"on both resamplers (`iid` and `by_speed`)"*. That clause was
inherited from E-GOAL-1/2, where the treatment **was** a resampled along-track residual and `iid` vs
`by_speed` were two ways of **drawing** it. **E-GOAL-3's treatment is a trained head: the along-track
prediction is deterministic per window and there is nothing to resample.** Inspecting
`e2_place.place()` confirms the distinction lives *entirely* on the along axis — the
`parent_resampled` **cross**-track draw is i.i.d. in both modes — so running "`by_speed`" here would
produce a bit-identical number, not a second reading.

**What replaces it, and it is strictly stronger:** separation is required on **all three cross-track
backgrounds (`parent_resampled`, `sel`, `reduced`) × both training configurations (T-OOF, T-TRAIN)**
= **6 cells**, with the C30 background span reported. A background swing was measured by E-GOAL-2 at
**15.9 recovery points with a separation flip**; the two resamplers moved E-GOAL-2's own primary by
**0.8 points**. The substitution therefore tests the axis that actually moves the answer.

**This is recorded as a substitution, not a pass.** The §4 rule as literally written cannot be
evaluated; the verdict in `EGOAL_3.md` says so explicitly.
