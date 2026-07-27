# E-GOAL-3 — the TRAINED goal head, through the real rule, at n = 600

**Stream:** `2026-07-28-egoal-3-trained-head` · **Wall-clock date:** 2026-07-27
*(folder named for the program's narrative date, which runs ahead of wall-clock — flagged, not
silently absorbed).*
**Hosts:** `tanitad-pod2` (parity-cache pose extraction only — **GPU never touched**, CPU `nice`d
beside a running wide-FOV build, disk checked with a real 500 MB `dd` at 340 MB/s, never `df`) +
dev box CPU (fit + placement).
⛔ **pod1 was NEVER contacted** — it is training `flagship-v2corpus-30k`. pod3 and the eval pod were
probed read-only once each.
**Pre-registration:** `PRE_REGISTRATION.md`, written **before any number below was computed**;
two amendments appended in its §10, nothing above them edited.
**Estimator:** paired episode-cluster bootstrap, `taniteval/taniteval/ci.py`, **B = 2000**,
unit = **the val episode**. `overlapping_holdout_se` is **never called**.
**Total compute:** ~4 min pod2 CPU (no GPU, zero GPU-hours) + ~9 min dev-box CPU.

---

## 0. HEADLINE

> ### ⭐⭐ **CONFIRM. A trained ego-kinematics goal head, fed through the ACTUAL REF-C rule with its ACTUAL per-window predictions, recovers `+46.3 %` of the fan's headroom at n = 600 under the CONSERVATIVE `parent_resampled` background — `−0.1426 [−0.1573, −0.1273]`, separated. That is `1.82×` E-GOAL-2's resampled `+25.4 %`, which THIS ENGINE reproduces to `0.003` recovery points. The resampled estimate transferred, and it UNDER-stated the lever.**

> ### ⭐ **In the DEPLOYABLE configuration — fitted on the parity train corpus, 2376 episodes, zero val episodes (0/600 by pose content) — it is `+50.7 %`, `−0.1564 [−0.1719, −0.1408]`, separated-better and separated-better than the OOF head itself.**

> ### ⛔⛔ **AND THE MECHANISM CLAIM IT INHERITED IS REFUTED. The lever is NOT "one second of speed history" — it is ONE 0.1 s SPEED DIFFERENCE. `v` alone is `−19.4 %` (separated-WORSE); `v + ax` is `+46.3 %`, a TIGHT NULL against the full ten-column head (T-OOF: `+0.0002 [−0.0023, +0.0027]`; at T-TRAIN the extra eight columns are separated but worth `2.9 %`). The `dv_*`/`v_lag_*` block E-GOAL-2 credited with `64 %` of the recovery is worth `0.9` of `46.3` points — `2.0 %`.**

> ### ⭐⭐ **AND THE ROOT CAUSE IS FOUND AND REPLICATED ON E-GOAL-2's OWN CORPUS WITH E-GOAL-2's OWN CODE: `egomotion`'s NATIVE `ax` channel is a poor derivative of the speed the target integrates. Replacing it with a 0.1 s finite difference gives `E1_v_axfd = 0.9270 m` — a NULL against the whole 10-column `E1_ego` at `0.9305 m` — while the native pair `E1_v_ax` is `1.1808 m`. E-GOAL-2's "speed history" was a PROXY for a derivative its own `ax` column failed to supply.**

**⇒ v5 carries a goal input. Size it against `+46.3 %` under a named background, and the feature is
`v` + `dv/dt` over ONE 0.1 s tick — two scalars from IMU/CAN, no history buffer, no new sensor.**

---

## 1. ⛔ C23 — THE FUTURE-CONTENT AUDIT. Priority 1, and it is CLEAN by DEFINITION and EMPIRICALLY

`raw/e3_features_val.json` · `raw/e3_features_train.json` · `code/e3_features.py`

A trained head is far more exposed to leakage than a resampler was, and the fan dumps carry a live
trap: **`head_deg` is the FUTURE net heading change** and sits beside `v0`
(`driving_diagnostic.net_heading_change_deg` reads `poses[last + horizon, 2]`). **Nothing derived
from the fan dump reaches any arm** — every feature is built from `poses[T,4]` by an expression
audited three ways.

### 1.1 By definition — every fed column, with its index expression

`L` = the window's **last observed frame** (`start + WINDOW − 1`, `WINDOW = 8`), `DT = 0.1 s`.

| column | index expression | offsets read | past? |
|---|---|---|---|
| `v` | `s[L]` | 0 | ✅ |
| `ax` | `(s[L] − s[L−1]) / DT` | 0, −1 | ✅ |
| `yawrate` | `wrap(yaw[L] − yaw[L−2]) / (2·DT)` | 0, −2 | ✅ |
| `curv` | `yawrate / max(v, 0.5)` | — | ✅ |
| `abs_curv` | `\|curv\|` | — | ✅ |
| `ay` | `v · yawrate` | — | ✅ |
| `dv_0p5` | `s[L] − s[L−5]` | 0, −5 | ✅ |
| `dv_1p0` | `s[L] − s[L−10]` | 0, −10 | ✅ |
| `v_lag_0p5` | `s[L−5]` | −5 | ✅ |
| `v_lag_1p0` | `s[L−10]` | −10 | ✅ |
| **`y_long`** | `(p[L+20] − p[L]) · ê_yaw(L)` | **+20** | ⛔ **THE TARGET** |

**The complete set of offsets any feature reads is `{0, −1, −2, −5, −10}` — max = 0.** Asserted at
runtime (`features()` raises on any positive offset), not merely documented.

### 1.2 ⭐ EMPIRICALLY — the `future_blind` test, on EVERY window, not a sample

> **Method.** Per window, overwrite **every pose row after `L`** with large random garbage
> (`N(10⁴, 10⁴)`) and recompute the features. A feature that reads the future **changes**.

| corpus | windows audited | **max \|Δ feature\|** | **max \|Δ target\|** | target moved on |
|---|---:|---:|---:|---|
| ⭐ **val600** (the scoring set) | **13 198 / 13 198 — all of them** | **`0.0`** ✅ | 53 106.84 m | **13 198 / 13 198** ✅ |
| **train2376** (the T-TRAIN training set) | 6 830 (40 episodes) | **`0.0`** ✅ | 47 609.83 m | **6 830 / 6 830** ✅ |

> ⭐ **`max |Δ| = 0.0` on every fed feature of every scored window** — the features are provably a
> function of `poses[:L+1]` alone.
> ⭐ **The instrument's power is demonstrated, not asserted:** the **target** — future by definition
> — moves on **100 % of windows** under the identical corruption. A blind test that could not fail
> would prove nothing; this one fails exactly where it must.

### 1.3 ⚠️ The negative-index trap the brief names — it FIRES, and it is handled

**A negative Python index wraps to the END of the array**, so `v_lag_1p0` at `L = 7` would read
`poses[−3]` — **the third-from-last frame of the episode, deep in the future.**

| | value |
|---|---:|
| windows where some index would have gone negative | **600 of 13 198 (4.55 %)** |
| …i.e. | **exactly the first window of every episode** |
| handling | every index is `max(L + off, 0)` — a causal "as much history as exists" degradation |

**Not hypothetical.** Un-clamped, 4.55 % of the scored windows would have been fed a future-derived
speed lag — and §1.2's blind test is the instrument that would have caught it.

---

## 2. ⛔ THE LEAK / OVERLAP CHECK — BY CONTENT, ON THE PATHS I ACTUALLY READ

`raw/e3_leak.json` · `code/e3_leak.py`

**This matters more here than in E-GOAL-2, because E-GOAL-3 actually TRAINS on the parity train
corpus.** Method: **sha256 over the raw `poses[T,4]` float32 bytes**, computed by `e3_features.py`
**on the same in-memory tensor the features are derived from** — the fingerprint is of the bytes
actually used, not of a second read of a nominally identical copy. *(E-GOAL-2's own reported gap was
that a sibling audited `_epcache` while the dump read `/root/valdata` on the same pod: an audit of a
different copy is not an audit of this one.)*

| check | required | measured |
|---|---|---|
| **A — train2376 (my training set) × val600 (my scoring set), BY CONTENT** | overlap = 0 | ⭐ **`overlap_n` = 0 / 600 · 0.0000 %** ✅ |
| **B — the same comparison BY FILENAME** | — | ⚠️ **600 / 600 = 100 %.** A name check would have called this a total leak. **The fingerprint is load-bearing, not decorative.** |
| **C — internal collisions** | 0 | val600 **600 unique of 600**; train2376 **2376 unique of 2376** ✅ |
| **E — path identity** | must match | my builder reads `/root/valdata/physicalai-val-0c5f7dac3b11`; **`refc_rerank.VAL` is the same string** ✅ |
| ⭐ **D — independent cross-check** | — | **600 / 600 sha256 AGREE with E-GOAL-2's own `e2_leak.py`**, a different script on a different run; the aggregate (600/600 unique, 2376/2376 unique, overlap 0) is **identical** ✅ |

> ⭐ **0 of 600 val episodes overlap the parity train corpus by pose content, on the exact path the
> fan dump reads — reported because it was run, not because it came back clean.**
> ⛔ **Parity untouched:** `_epcache` never written, no episode re-selected; the parity corpus was
> read only to extract poses and to fingerprint it.

---

## 3. FIDELITY — five gates, and one of them is a per-row identity

> *"Three streams in two days found bugs in their own scoring code producing stable, plausible,
> wrong numbers. Assume yours has one until a fidelity control says otherwise."*

The fan dump records no time index, so the `(episode, L)` mapping is **reconstructed** from
`refc_rerank.dump()`'s own window rule — and then **proved**, not assumed.

| gate | what it proves | measured |
|---|---|---|
| **F-0** | the deployment re-derives from the fan | `a0` **0.5015**, `R_goal2s` **0.1933**, `oracle_in_fan` **0.1547**, headroom **0.3082** — **all four exact to 4 dp** vs E-GOAL-2's raw JSON ✅ |
| **F-4** | window/episode alignment | my reconstructed episode index sequence **== the dump's `eid`, all 13 198 rows** ✅ |
| ⭐ **F-2** | the join, per row | ⭐ **`max \|v_mine − v0_dump\| = 0.0` over 13 198 rows — an EXACT PER-ROW IDENTITY.** A wrong join cannot pass this. |
| **F-2b** | the target and the frame convention | `max \|y_long − gt[:,−1,0]\| = 9.05 × 10⁻⁶ m` (pure float32 round-trip) ✅ |
| ⛔ **F-3** | **F-2 has power** | the identical comparison with rows shifted by one **fails hard: 35.40 m/s and 74.68 m** ✅ |
| ⭐⭐ **F-1** | **my placement engine == E-GOAL-2's** | E-GOAL-2's frozen `E1_ego` pool through THIS engine returns **+25.42 % / +26.20 %** and **`−0.0784 [−0.0960, −0.0606]` — identical to four decimals**, max deviation **0.003 recovery points** ✅ |

> ⭐ **F-1 is what makes the headline interpretable.** +46.3 % vs +25.4 % is a **treatment**
> difference, not an engine difference: the same code, the same fan, the same background, the same
> seeds, returns E-GOAL-2's number when fed E-GOAL-2's input.

`realise` / `goal_reference` / `pick_nearest_to` / `ade` are **IMPORTED** from E-GOAL-1's
`eg_place.py`; the cross-track backgrounds come from E-GOAL-2's `e2_place.fan_blocks` /
`reduced_head`, **imported**. Neither repo module was edited.

---

## 4. ⭐ THE TRAINED HEAD — RMS and MAE on the canonical 600-episode windows

`raw/e3_head.json` · `code/e3_fit.py`

Target `y_long` = the 2 s **along-track** displacement. Regressor: `HistGradientBoostingRegressor`
with **E-GOAL-1's hyper-parameters imported verbatim** (`eg_fit.fit_predict`), **identical for every
arm**, so extra columns can only help through held-out generalisation.
**13 198 windows / 600 episodes — the fan dump's window count, to the row.**

| arm | features | **T-OOF** RMS / MAE (m) | **T-TRAIN** RMS / MAE (m) |
|---|---|---:|---:|
| ⭐ **`H_ego`** | all 10 | **0.7449 / 0.4819** | ⭐ **0.7121 / 0.4545** |
| ⭐ **`H_v0_ax`** | **`v`, `ax` only** | **0.7519 / 0.4888** | **0.7387 / 0.4705** |
| `H_nohist` | history dropped (6) | 0.7628 / 0.4960 | 0.7331 / 0.4680 |
| ⛔ `H_noise_hist` | history → matched Gaussian noise | 0.7570 / 0.4943 | 0.7315 / 0.4677 |
| `H_inst` | `v, ay, curv, abs_curv, yawrate` — **no speed differencing** | **1.4821 / 1.0177** | **1.4404 / 0.9763** |
| `H_v0` | `v` only | 1.4500 / 0.9818 | 1.4468 / 0.9712 |
| `CV_head` | no fit, `ŷ = v·2 s` | 1.4490 / 0.9753 | *(same)* |
| ⛔ **`N_SHUF`** | all 10, permuted across episodes | **21.0171 / 17.2658** | **20.0456 / 16.6760** |
| `P_ORACLE_TRUE` | the true goal *(bound, never a capability)* | 0 / 0 | 0 / 0 |

- **T-OOF** = 5 **episode-disjoint** folds over the 600 val episodes (`eg_common.clip_folds`, seed 0).
- **T-TRAIN** = fitted on the **parity train corpus** (`physicalai-train-e438721ae894`, **2376
  episodes / 406 099 windows**), predicting all 13 198 val windows with **zero** val episodes in
  training (§2). The **deployable** configuration, and what `EGOAL_1.md §6.3` asked for.

> ⭐ **The trained head reaches 0.7449 m (T-OOF) / 0.7121 m (T-TRAIN). It clears the E-GOAL-2
> family-matched σ₀ (1.2195 m) by 1.64× AND — for the first time in this program — the inherited
> `ISO` 0.813 m bar as well** (1.09× / 1.14×). **The `BAR-INHERITED-FROM-THE-WRONG-FAMILY` conflict
> does not arise here: the head is good enough that both bars agree.**

**Error family, MEASURED on the estimator (C24):** near-unbiased (**α = 0.9983 / 0.9988**, bias
+0.003 / +0.012 m ⇒ **NOT `SHRINK`**) and heavy-tailed (**RMS/MAE 1.546 / 1.567** vs a Gaussian's
1.2533 ⇒ **NOT `ISO`**) ⇒ the family is **`EMPIRICAL`** — but **lighter-tailed than the dev-corpus
head's 1.867**, i.e. a *different* empirical family, which is exactly why the bar had to be re-run.

⚠️ **These RMS values are NOT directly comparable to E-GOAL-1's 0.9305 m** — a different window set
(the parity val600 canonical windows vs a 10 Hz dev-corpus grid keyed `14231cd29c74`). Normalised by
the same-corpus `CV` baseline the comparison is admissible and still favourable: **head/CV = 0.514
(T-OOF) / 0.491 (T-TRAIN)** against E-GOAL-1's **0.673**. §6 removes most of this threat by
re-running the decisive contrast **on E-GOAL-1's own corpus.**

---

## 5. ⭐⭐ THE PLACEMENT — the head's REAL predictions through the REAL rule

`raw/e3_place_n600_{parent_resampled,sel,reduced}.json` · `code/e3_place.py`

Deployment: **13 198 windows / 600 episode clusters**, `a0` **0.5015**, `R_goal2s` **0.1933**,
headroom **0.3082**.

### 5.1 ⛔ C30 — THE BACKGROUND IS NAMED, AND THE HEADLINE IS QUOTED UNDER THE CONSERVATIVE ONE

**PRIMARY: `parent_resampled`** — E-GOAL-2's registered conservative carrier (the parent head's own
881 cross-track residuals resampled onto the true cross-track, cross MAE **0.4004 m**, drawn once per
seed from `default_rng(5000 + s)` and **identical across every arm within a seed**).
Secondaries: `sel` (cross MAE 0.2347 m) and `reduced` (0.2281 m).

| arm | along RMS | ⭐ **`parent_resampled`** (PRIMARY) | `sel` | `reduced` |
|---|---:|---|---:|---:|
| ⭐ **`T_OOF\|H_ego`** | 0.7449 | ⭐ **+46.3 % · −0.1426 [−0.1573, −0.1273] ✅ BETTER** | +61.5 % ✅ | +62.1 % ✅ |
| ⭐ **`T_TRAIN\|H_ego`** | 0.7121 | ⭐ **+50.7 % · −0.1564 [−0.1719, −0.1408] ✅ BETTER** | +66.4 % ✅ | +67.1 % ✅ |
| ⭐ **`T_OOF\|H_v0_ax`** | 0.7519 | **+46.3 % · −0.1428 [−0.1577, −0.1273] ✅ BETTER** | +61.5 % ✅ | +61.9 % ✅ |
| `T_OOF\|H_nohist` | 0.7628 | +45.3 % · −0.1397 ✅ | +60.5 % ✅ | +61.0 % ✅ |
| ⛔ `T_OOF\|H_noise_hist` | 0.7570 | **+45.4 % ✅ — an information-free-history arm SEPARATES** | +60.7 % ✅ | +61.2 % ✅ |
| `T_OOF\|H_inst` | 1.4821 | **−26.6 % · +0.0821 [+0.0702, +0.0927] ✅ WORSE** | −14.5 % | −14.4 % |
| `T_OOF\|H_v0` | 1.4500 | **−19.4 % · +0.0596 [+0.0479, +0.0703] ✅ WORSE** | −7.9 % | −7.5 % |
| `CV_head` | 1.4490 | **−18.6 % · +0.0572 [+0.0454, +0.0680] ✅ WORSE** | −6.1 % | −6.0 % |
| ⛔ **`N_SHUF`** | 21.0171 | **−2993.4 % · +9.2260 [+8.8197, +9.6260] ✅ WORSE** | −2991.3 % | −2990.7 % |
| **`P_ORACLE_TRUE`** *(bound)* | 0 | **+77.4 % · −0.2386 [−0.2548, −0.2218] ✅ BETTER** | +94.8 % | +95.3 % |

Realised `ade_0_2s` with its own interval: `T_OOF|H_ego` **0.3589 [0.3487, 0.3701]**;
`T_TRAIN|H_ego` **0.3451 [0.3348, 0.3565]** (RMS-family companion 0.4620 / 0.4477 — **same ordering,
same conclusions**; no claim rests on a disagreement between the two).

> ⭐ **Every control behaves. The run is not VOID.** The positive bound is separated-better, and
> **four** arms are separated-**WORSE** — including the deliberately failing `N_SHUF` **at the actual
> n = 600**, which is the C31 requirement.
>
> ⚠️ **C30 replicates almost exactly: the background span is 15.8 recovery points at T-OOF (+46.3 →
> +62.1) and 16.4 at T-TRAIN**, against E-GOAL-2's measured 15.9. ⭐ **But separation does NOT flip:
> all 6 primary cells (3 backgrounds × 2 training configurations) are separated-better.**
> **The headline is the conservative one.**
>
> ⚠️ **`P_ORACLE_TRUE` is +77.4 %, not +100 %, under the primary background** — the missing 22.6
> points are the *background's own* cross-track error (it reaches +94.8 % / +95.3 % under `sel` /
> `reduced`). This is the price of the conservative carrier, made explicit rather than hidden.

### 5.2 ⭐ Why +46.3 % and not +25.4 % — decomposed, not asserted

The requirement curve scales **the trained head's own residual about the truth**, preserving its
correlation with window difficulty. At **k = 1.25 the head's residual has RMS 0.9311 m** — matched to
E-GOAL-2's `E1_ego` pool at **0.9305 m**:

| | along RMS | recovery |
|---|---:|---:|
| E-GOAL-2, **decorrelated resampled residual** | 0.9305 m | **+25.4 %** |
| ⭐ this stream, **real head, scaled to the same RMS** | 0.9311 m | ⭐ **+29.3 %** |
| ⭐ this stream, **real head at its actual accuracy** | 0.7449 m | ⭐ **+46.3 %** |

> ⭐⭐ **At matched RMS the real, CORRELATED head is +3.9 recovery points BETTER than the
> decorrelated resampler.** E-GOAL-1's claim that *"decorrelation makes the outcome worse, so the
> construction is conservative"* — measured there only on the *parent's* residual — is now
> **confirmed on a real trained head.** The remaining **+17.0 points** is simply the head being more
> accurate (0.7449 vs 0.9305 m).
> ⚠️ **Honest limit:** the +3.9 mixes *correlation* and *tail shape* (RMS/MAE 1.546 vs 1.867). This
> design cannot separate them, and I do not claim it does.

### 5.3 The requirement curve — a COMPANION, not the verdict

Family: the trained head's own residual, scaled, run through the real rule.

| k | 0.0 | 0.25 | 0.5 | 0.75 | **1.0** | 1.25 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| along RMS (m) | 0 | 0.186 | 0.372 | 0.559 | **0.745** | 0.931 | 1.117 | 1.490 | 2.235 | 3.724 |
| recovery | +77.4 % | **+78.5 %** | +72.6 % | +61.2 % | **+46.3 %** | +29.3 % | +11.6 % | −27.5 % | −110.7 % | −287.6 % |

| | σ₀ break-even | σ₅₀ half prize |
|---|---:|---:|
| ⭐ **this stream, n = 600, `parent_resampled`, real head** | **1.2276 m** | **0.6982 m** |
| E-GOAL-2, n = 600, same background, resampled residual | 1.2195 m | 0.5851 m |
| inherited **`ISO`** bar | 0.813 m | 0.439 m |

> ⭐ **σ₀ = 1.2276 m against E-GOAL-2's 1.2195 m — 0.7 % apart. An independent replication of the
> family-matched break-even on a DIFFERENT error family (RMS/MAE 1.546 vs 1.867) and a REAL
> correlated head.** The bar is a property of the deployment and the rule, not of the estimator that
> happened to measure it.
> ⚠️ **The curve is NOT monotone at the top: k = 0.25 (+78.5 %) beats k = 0 (+77.4 %).** A
> *slightly* noisy along-track goal beats the exactly-true one, because the cross-track background is
> wrong — the perfect along coordinate is not the optimal pointer when its partner is not. Small
> (1.1 points) and reported rather than smoothed away.
> ⚠️ It still goes hard negative — **−287.6 % at 3.72 m**. **A goal channel must be gated on measured
> accuracy before it is wired in. A confidently wrong goal is destructive, not neutral.**

### 5.4 T-OOF vs T-TRAIN — more data helps, and the deployable arm is the better one

`H_ego`: **T-OOF vs T-TRAIN = +0.0138 [+0.0106, +0.0169], separated** ⇒ **fitting on the 2376-episode
parity corpus is separated-BETTER than 5-fold OOF inside the 600 val episodes**, replicated on
`H_nohist` (+0.0133) and `H_noise_hist` (+0.0129). ⭐ **The deployable configuration is not a
compromise — it is the stronger one.**

---

## 6. ⛔⛔ C31 — THE NEGATIVE CONTROL AT n = 600, AND WHAT IT DESTROYS

### 6.1 The predicate is confirmed NON-DISCRIMINATING at n = 600, in MY data too

⛔ **`H_noise_hist` — an arm whose four history columns are pure Gaussian noise — separates at
`+45.4 %`.** E-GOAL-2's `PREDICATE-STOPS-DISCRIMINATING-AT-HIGH-n` fires again, on a different
treatment. ⇒ **"`H_ego` is separated" cannot support any mechanism claim.**
⭐ But the predicate is **not dead**: the deliberately failing `N_SHUF` is separated-**WORSE** at the
actual n (−2993 %), as are `H_v0`, `H_inst` and `CV_head`. It discriminates *"carries ego
information"* from *"does not"* — it simply cannot discriminate *"carries speed history"*.

### 6.2 ⭐ The direct paired contrasts — where the mechanism claim actually lives

Primary background, paired on the same windows.

| contrast | T-OOF | T-TRAIN |
|---|---|---|
| ⭐ **`H_v0_ax` vs `H_v0`** — **ONE 0.1 s speed difference** | ⭐ **−0.2025 [−0.2176, −0.1864] SEP** | ⭐ **−0.2075 [−0.2236, −0.1906] SEP** |
| ⭐ **`H_ego` vs `H_v0_ax`** — everything beyond two columns | ⭐ **+0.0002 [−0.0023, +0.0027] NULL** | −0.0045 [−0.0077, −0.0017] SEP |
| `H_ego` vs `H_nohist` — the 0.5–1.0 s lags | −0.0029 [−0.0052, −0.0006] SEP | −0.0034 [−0.0054, −0.0013] SEP |
| `H_ego` vs `H_noise_hist` — history vs noise, capacity fixed | −0.0027 [−0.0052, −0.0005] SEP | −0.0036 [−0.0056, −0.0017] SEP |
| ⛔ **`H_nohist` vs `H_noise_hist`** — **MUST BE NULL** | ⭐ **+0.0002 [−0.0019, +0.0020] null** ✅ | ⭐ **−0.0002 [−0.0008, +0.0004] null** ✅ |
| `H_inst` vs `H_v0` — rotation state, no speed differencing | **+0.0225 [+0.0179, +0.0269] SEP — WORSE** | +0.0016 [−0.0040, +0.0061] null |
| `H_nohist` vs `H_inst` — speed differencing vs none | **−0.2218 [−0.2366, −0.2060] SEP** | **−0.2102 [−0.2262, −0.1934] SEP** |
| `H_ego` vs `P_ORACLE_TRUE` — distance to the bound | +0.0960 [+0.0895, +0.1035] SEP | +0.0821 [+0.0752, +0.0897] SEP |

### 6.3 ⭐⭐ The recovery ladder, and it refutes the inherited mechanism

| rung | features | T-OOF recovery | T-TRAIN |
|---|---|---:|---:|
| `H_v0` | `v` | **−19.4 %** | −18.0 % |
| `H_inst` | `v, ay, curv, abs_curv, yawrate` (**no speed differencing**) | **−26.6 %** | −18.6 % |
| ⭐ **`H_v0_ax`** | **`v, ax`** | ⭐ **+46.3 %** | ⭐ **+49.3 %** |
| `H_nohist` | + `ay, curv, abs_curv, yawrate` | +45.3 % | +49.7 % |
| ⭐ **`H_ego`** | + `dv_0p5, dv_1p0, v_lag_0p5, v_lag_1p0` | **+46.3 %** | **+50.7 %** |

> ⭐⭐ **ONE 0.1 s SPEED DIFFERENCE IS WORTH +65.7 RECOVERY POINTS (T-OOF) / +67.3 (T-TRAIN)** — it
> flips the arm from separated-WORSE to separated-BETTER by itself.
> ⛔ **The `dv_*` / `v_lag_*` block E-GOAL-2 credited with 64 % of the recovery is worth +0.9 points
> of 46.3 (2.0 %) / +1.1 of 50.7 (2.1 %).**
> ⭐ **Physically it is exactly right.** The target is a 2 s displacement ≈ ∫v dt. Constant velocity
> gives `2v` (`CV_head`, −18.6 %); **acceleration is the first-order correction, `2v + 2a`** — and it
> is the dominant one. `H_inst` shows rotation state without it buys **nothing** (it is *worse* than
> `v` alone at T-OOF, a pure fitting cost).

### 6.4 ⭐⭐ WHY THE TWO STREAMS DISAGREED — the discriminating experiment, on E-GOAL-2's OWN corpus

`raw/e3_axprobe.json` · `code/e3_axprobe.py`

E-GOAL-1/2's `ax` is `egomotion`'s **NATIVE** acceleration channel interpolated at t
(`lead_state_gate.ego_frame`); mine is `(v[L] − v[L−1]) / 0.1`, a finite difference of the cache's
own speed channel. **The two candidate causes — corpus vs `ax` quality — were separated by re-running
on E-GOAL-1's own parquet with E-GOAL-1's own fitter, folds and seed.**

| arm (E-GOAL-1's dev corpus, 99 935 windows / 612 clips) | along RMS |
|---|---:|
| `E1_ego` — the full 10 columns | **0.9305** ✅ *reproduces E-GOAL-1 exactly (fidelity)* |
| `E1_nohist` — history dropped, **native** `ax` | **1.0733** ✅ *reproduces exactly (fidelity)* |
| `E1_v` — `v` alone | **1.5246** ✅ *reproduces exactly* |
| `E1_v_ax` — `v` + **native** `ax` | **1.1808** |
| ⭐ **`E1_v_axfd` — `v` + a 0.1 s FINITE DIFFERENCE of `v`** | ⭐ **0.9270** |
| `E1_nohist_axfd` — `E1_nohist` with `ax` replaced by the finite difference | **0.9659** |

| contrast | ΔRMS | paired MAE Δ | |
|---|---:|---|---|
| ⭐ **`E1_ego` vs `E1_v_axfd`** | **+0.0035** | **−0.0065 [−0.0259, +0.0095]** | ⭐ **NULL** |
| ⭐ **`E1_v_axfd` vs `E1_v_ax`** — finite difference vs the native channel | **−0.2539** | −0.1704 [−0.2002, −0.1430] | ✅ SEP |
| `E1_nohist_axfd` vs `E1_nohist` | **−0.1074** | −0.0890 [−0.1080, −0.0713] | ✅ SEP |
| `E1_ego` vs `E1_nohist` — E-GOAL-2's history block | −0.1428 | −0.1035 [−0.1273, −0.0838] | ✅ SEP |

> ⭐⭐ **ON E-GOAL-2's OWN CORPUS, `v` + ONE 0.1 s SPEED DIFFERENCE (0.9270 m) IS STATISTICALLY
> INDISTINGUISHABLE FROM THE WHOLE TEN-COLUMN HEAD INCLUDING ITS 1 s OF HISTORY (0.9305 m).**
> ⭐ **The finite difference alone recovers 0.1074 of E-GOAL-2's 0.1428 m "history" gap — 75 % of it
> — and is worth 0.2539 m more than the NATIVE `ax` channel on identical windows** (the two correlate
> at only **0.759** despite near-equal SDs, 0.766 vs 0.744).
> ⛔ **⇒ E-GOAL-2's *"64 % of the recovery is speed history"* is a FEATURE-ENGINEERING ARTEFACT.
> The `dv_*` / `v_lag_*` block was serving as a PROXY for the derivative of speed that the native
> `ax` column failed to supply.** The *direction* of E-GOAL-2's claim survives — **change in speed is
> the lever** — but the *required history length* does not: **0.1 s, not 1 s, and 0.1 s is better.**
> ⭐ **The two streams are not in conflict once the column is fixed. They agree.**

### 6.5 ⭐ Each verdict's FAILING value was DEMONSTRATED, not asserted

| rule | fires on | **demonstrated in this run's own data?** |
|---|---|---|
| ⛔ **REFUTE** | not separated-better | ⭐ **YES, four ways** at the actual n on the primary background: `CV_head` **+0.0572 [+0.0454, +0.0680]**, `H_v0` **+0.0596**, `H_inst` **+0.0821**, `N_SHUF` **+9.2260** — all separated-**WORSE**. |
| **PARTIAL** (separated-better, < +19.1 %) | | ⭐ **YES** — the requirement curve at k = 1.5 (RMS 1.1173 m) returns **+11.6 %, separated-better**. |
| ⭐ **CONFIRM** | | fired, at +46.3 %. |
| a genuine **NULL** at n = 600 | | ⭐ **YES, three of them**, CI widths **0.0012–0.0050** — tight nulls, not unpowered ones. |

---

## 7. VERDICT, AND WHAT IT LICENSES

### 7.1 ⭐⭐ CONFIRM — **and it is the strong form**

> **The trained head's realised recovery under the NAMED conservative background `parent_resampled`
> is `+46.3 %` (T-OOF, `−0.1426 [−0.1573, −0.1273]`) and `+50.7 %` (T-TRAIN, the deployable
> configuration, `−0.1564 [−0.1719, −0.1408]`) — separated-better on all three backgrounds and both
> training configurations, 6 of 6 cells.**
> **The pre-registered CONFIRM threshold was ≥ +19.1 %. It is exceeded by 2.4×.**
> ⭐ **The resampled estimate TRANSFERRED, and it was CONSERVATIVE: a real head with correlated
> errors beats the decorrelated resampler by +3.9 recovery points at matched RMS, and by +20.9
> points at its actual accuracy.**

⚠️ **One clause of the registered rule could not be evaluated, and it is recorded rather than quietly
passed** (`PRE_REGISTRATION §10 A2`): §4 asked for separation *"on both resamplers"*, but **a trained
head has nothing to resample** — the `iid`/`by_speed` distinction lives entirely on the along axis,
which E-GOAL-3 replaces. It is substituted by **3 backgrounds × 2 training configurations**, the axis
E-GOAL-2 measured as moving the answer by 15.9 points rather than 0.8.

### 7.2 ⛔ REFUTED IN PASSING: the mechanism, not the lever

**E-GOAL-2's *"64 % of the recovery is speed history"* does not survive**, on three independent
readings: the recovery ladder (§6.3, the block is worth 2.0 %), the direct contrast (§6.2,
`H_ego` vs `H_v0_ax` is a null), and the root-cause probe **on E-GOAL-2's own corpus with its own
code** (§6.4). **Stated plainly, not re-scoped:** the block was a proxy for a derivative that
`egomotion`'s native `ax` column failed to supply.

### 7.3 ⭐ Does v5 carry a goal input? **YES — and the feature list changes**

| | E-GOAL-2 said | ⭐ **E-GOAL-3 measures** |
|---|---|---|
| feature | `dv_0p5, dv_1p0, v_lag_0p5, v_lag_1p0` (1 s of history) | ⭐ **`v` and `ax = Δv over one 0.1 s tick`** |
| size to plan with | +25.4 % | ⭐ **+46.3 % (T-OOF) / +50.7 % (T-TRAIN)** under `parent_resampled` |
| evidence | resampled residual | ⭐ **a trained head's actual per-window predictions** |
| cost | IMU/CAN, no new sensor | **the same, minus the history buffer** |

**Plan v5 with +46 %, quoted under `parent_resampled`, and gate the goal channel on measured accuracy
(§5.3) before wiring it in.**

### 7.4 What I refuse to conclude

- **NOT that the v5 selector will realise +46 %.** This measures a goal injected into a **frozen**
  REF-C-XL fan through a **fixed** proximity rule. A jointly trained selector is a different object,
  and nothing here says it inherits the number.
- **NOT that the cross-track axis is solved.** The primary background is still a **resampled**
  cross-track residual — only the ALONG axis is a real head. `eh2_cache.pt` cannot be rebuilt at 600.
  Mitigation: `sel` and `reduced` are *real, un-resampled learned* cross-tracks and both give a
  **larger** recovery, so the primary remains the conservative reading.
- **NOT that 0.7449 m beats 0.9305 m as heads.** Different window sets. §6.4 is the admissible
  comparison, and it is on E-GOAL-1's corpus.
- **NOT anything past 2 s**, and every number is a displacement/ADE number — **blind to collision.**
- **NOT that `ax` is safe unaudited.** It is one index away from a leak; §1.2 is why it is quotable.
- **NOT that the head-fit axis is powered on E-GOAL-1's corpus.** `SEPARATION-CLAIMED-ON-AN-
  UNPOWERED-AXIS` still applies there; §6.4 reports the paired **MAE** family for that reason.

---

## 8. ESCALATIONS — these must not sit in a file

1. 🔴 **`V5_PLAN.md §8` and `Gates/flagship-v5-retrain.PREP.md` item 6 must be updated by their
   owner, and BOTH halves change.**
   - the size: *"Size it against +25.4 %"* → ⭐ **+46.3 % (T-OOF) / +50.7 % (T-TRAIN), under
     `parent_resampled`, from a TRAINED head.** The tier goes from *"CONFIRMED as a resampled
     residual, NOT deployable"* to **CONFIRMED with an actual head, out-of-fold, episode-disjoint,
     0/600 leak, at n = 600.**
   - the feature: *"`dv_0p5`, `dv_1p0`, `v_lag_0p5`, `v_lag_1p0` enter the v5 selector"* →
     ⛔ **wrong columns. `v` and `ax` (one 0.1 s Δv) carry 97 % of it; the four lag columns carry
     2 %.** A v5 built on the published list would ship a history buffer for 2 % of the effect.
2. 🔴 **E-GOAL-2's mechanism headline must be withdrawn or re-scoped by its owner.** *"64 % of the
   recovery is speed history"* is refuted **on E-GOAL-2's own corpus, with E-GOAL-2's own fitter,
   folds and seed** (§6.4), while both of its fidelity anchors (0.9305 / 1.0733) reproduce exactly.
   Its *statistical* result stands; its *causal attribution* does not.
3. 🔴 **`stack/scripts/lead_state_gate.py::ego_frame` exports a weak `ax`.** On identical windows a
   0.1 s finite difference of its own `v` is worth **0.2539 m of along-track RMS** more than the
   native channel it emits (correlation only **0.759**). **Every consumer of `EGO_COLS` inherits
   this.** The fix is one line — emit `ax_fd` beside `ax` — and it belongs in the repo reader, not in
   this folder. *(An orthogonality instrument sat unmerged for 10 days on exactly this failure mode.)*
4. 🟠 **For `RETRACTION_LOG.md` — root-cause classes:**
   - ⭐ **`ABLATION-CREDITED-TO-THE-WRONG-COLUMN`** — an ablation that removes block **B** while
     leaving a **degraded proxy for B's information** inside block **A** credits **B** with what the
     proxy failed to carry. Measured here at **64 % → 2 %**, with the correct attribution
     (`ax` quality) confirmed on the original corpus. **The check is: repair the proxy and re-run the
     ablation** — not to re-fit until the answer changes, but because *"what does removing B cost?"*
     and *"what information does B carry?"* are different questions whenever A is imperfect.
     *(Sibling of `CONTROL-WEAK-BY-MODEL-CLASS`: there the control was mis-shaped for the regressor;
     here the baseline was mis-shaped for the target.)*
   - **`RESAMPLED-RESIDUAL-UNDER-STATES-A-TRAINED-HEAD`** — a decorrelating injection is
     **conservative**, now measured on a real head: **+3.9 recovery points at matched RMS.** E-GOAL-1
     asserted the direction from the parent's residual alone; this quantifies it. **A resampled
     placement is a LOWER bound on a trained head of the same accuracy, not an estimate of it.**
5. 🟠 **`MODEL_REGISTRY §1.2a`** — E-GOAL-2's escalation 3 stands unchanged (REF-C-XL's `a0`
   **0.4712 → 0.5015** going 40 → 600, against the corpus getting easier for v1 and the CV floor).
   Independently re-derived here as gate F-0, **exact to 4 dp**.

---

## 9. THREATS TO VALIDITY I COULD NOT REMOVE

| threat | direction | status |
|---|---|---|
| ⛔ **The cross-track background is still a RESAMPLED residual** — only the along axis is a real head. `eh2_cache.pt` is 881-window-only. | conservative | the two *real* learned cross-tracks (`sel`, `reduced`) both give **larger** recovery (+61.5 / +62.1 %), so the primary is the low end. **Building `eh2` at 600 is the clean fix and is NOT done here.** |
| ⚠️ **The fan is frozen and the rule is fixed.** A jointly trained v5 selector is a different object. | unknown, possibly large | §7.4. This is the next experiment, and it can fail. |
| ⚠️ **RMS values are on a different window set than E-GOAL-1's dev corpus.** | inflates the apparent head improvement | normalised by same-corpus CV (0.514 vs 0.673) it survives; §6.4 re-runs the decisive contrast **on E-GOAL-1's corpus** and it replicates. |
| ⚠️ `H_ego` vs `H_v0_ax` is a **null at T-OOF but separated (−0.0045) at T-TRAIN.** | the "two columns are enough" claim is n-dependent | reported both ways. Even at T-TRAIN the extra 8 columns are **2.9 %** of the effect. |
| ⚠️ **The `+3.9` at matched RMS mixes correlation and tail shape** (1.546 vs 1.867). | unknown sign on the split | stated; this design cannot separate them. |
| ⚠️ **The requirement curve is non-monotone at k = 0.25** (+78.5 % > +77.4 %). | small | 1.1 points, reported. It is a property of the *background*, not of the head. |
| **The committed `fan_refc-xl-30k.pt` is not bit-reproducible on a current pod** (1/881 picks, Δ`a0` −0.0002, not separated). | negligible here | `INHERITED` from E-GOAL-2 §2.3. This stream uses **only** the pod2-built 600-episode fan, so no cross-host mixing occurs. |
| 2 s, displacement/ADE only | unknown, possibly large | §7.4. |
| The dev box holds a corpus keyed `14231cd29c74`, not parity | — | **no parity-dependent step ran there.** Pose extraction from the parity caches ran on pod2; the dev box received only derived feature/target arrays and the pod-built fan (md5-verified identical on both hosts). |

**Evidence classes.** §§1–6 are `MEASURED (ours)` with artifact paths and are `DECISION-GRADE`
(n = 600 episode clusters, paired episode-cluster bootstrap, B = 2000, out-of-fold and
episode-disjoint, leak-checked by content). E-GOAL-2's +25.4 % / +26.2 %, the deployment quantities
and σ₀ = 1.2195 m are `INHERITED` — **and every one of them is re-derived here and reproduced**
(F-0 exact to 4 dp; F-1 to 0.003 recovery points; E-GOAL-1's 0.9305 / 1.0733 / 1.5246 reproduced
exactly by `e3_axprobe.py`). The inherited `ISO` 0.813 m is `PUBLISHED (cited)`. §7.4's *"a jointly
trained selector"* is `HYPOTHESIS` with a named test.

> **TIER: CONFIRMED (decision-grade)** — for *"a TRAINED ego-kinematics goal head, scored with its
> actual per-window predictions through the actual REF-C rule, recovers a separated +46.3 % of the
> fan's headroom at n = 600 under the conservative `parent_resampled` background, and the lever is
> `v` plus one 0.1 s speed difference."*
> **NOT LICENSED** — *"a v5 selector trained with this goal input will realise +46 %."*

---

## 10. DELIVERABLE MANIFEST

**Everything below is staged in the repo working tree (`git add`). Nothing was committed or pushed.**
⚠️ **Exactly two artifacts exist outside the repo, and both are named here rather than left to an
audit.**

| artifact | where | what |
|---|---|---|
| `EGOAL_3.md` | `repo:…/incoming/2026-07-28-egoal-3-trained-head/` | this document |
| `PRE_REGISTRATION.md` | same | bars, arms, failing-value proofs, decision rule — written before any fit; §10 carries amendments **A1** (the speed-differencing ladder) and **A2** (the inapplicable resampler clause), nothing above them edited |
| `code/e3_features.py` | same | ⭐ pose → PAST-ONLY features + target, the **`future_blind`** audit, the negative-index guard, the in-line pose fingerprint |
| `code/e3_leak.py` | same | S0 — content overlap, filename contrast, path identity, cross-check vs E-GOAL-2 |
| `code/e3_fit.py` | same | the trained head; **imports** `eg_fit.fit_predict` and `eg_common.clip_folds` |
| `code/e3_place.py` | same | ⭐ the placement; **imports** `eg_place.realise` and `e2_place.fan_blocks`/`reduced_head`; gates F-0/F-1; the C31 contrasts; the family curve |
| `code/e3_axprobe.py` | same | ⭐ the discriminating experiment on E-GOAL-1's own corpus |
| `raw/e3_features_val.json` · `raw/e3_features_train.json` | same | the C23 audit + all 600 + 2376 pose fingerprints |
| `raw/e3_leak.json` | same | ⭐ 0 / 600 by content, 600 / 600 by filename |
| `raw/e3_head.json` | same | every arm's RMS/MAE/family, both training configurations |
| `raw/e3_place_n600_{parent_resampled,sel,reduced}.json` | same | ⭐ the placements, all three backgrounds, with F-0/F-1 and every contrast |
| `raw/e3_place_n600_*_realised.npz` | same | per-window realised `ade_0_2s` for every arm — **every interval recomputable without the fan** |
| `raw/e3_axprobe.json` | same | the `ax`-quality root cause |
| ⚠️ `e3_val600_windows.npz` (0.8 MB) · `e3_train2376_windows.npz` (22.9 MB) | **`pod2:/workspace/_egoal3/` + dev-box scratchpad** | the feature/target matrices |
| ⚠️ `fan_refc-xl-30k_600ep.pt` (123 MB) | **`pod2:/workspace/_egoal2/` + dev-box scratchpad** | E-GOAL-2's fan, md5 `42ea6b09570bd84b5380b5715f81a453` **verified identical on both hosts** |

### ⚠️ The artifacts outside the repo — stated, not discovered later

**Neither the feature matrices nor the fan is staged, and nothing is stranded:**
- **The producers are staged.** `e3_features.py` regenerates both matrices deterministically in
  **~4 min of pod2 CPU** (`OMP_NUM_THREADS=6 nice -n 10 python3 e3_features.py --which val|train`);
  the fan regenerates in 19 min of A40 via E-GOAL-2's staged `e2_dump600.py`.
- **Both live in two places** (pod2 + dev box), so neither is single-disk.
- **Every number in this document is recomputable from the staged `raw/*.json` + `*.npz` +
  `taniteval/ci.py`.**
- ⚠️ **`.gitignore` bans `*.parquet` and large binaries repo-wide** — a per-window dump derived from
  a gated-confidential corpus is exactly what that rule protects. **I did not `git add -f`.**
  **Decision needed from the owner:** if a corpus-derived feature matrix should be shareable, the
  right home is HF under `Sayood/` (gated), not an override of a privacy-shaped policy.
  **Flagged rather than actioned.**

**Inputs consumed** (read-only): `pod2:/root/valdata/physicalai-val-0c5f7dac3b11` (600 eps),
`pod2:/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894` (2376 eps),
`pod2:/workspace/_egoal2/fan_refc-xl-30k_600ep.pt`,
`…/2026-07-28-egoal-2-power/raw/{e2_place_n600_parent_resampled.json, e2_leak.json}`,
`…/2026-07-27-egoal-1-lead-vehicle/raw/{eg_oof_pred_gbm.npz, eg_windows.parquet, eg_place.json}`,
`…/2026-07-27-goal-input/raw/gi_head_preds.npz`.

⛔ **Parity untouched:** `_epcache` never written, no episode re-selected; the parity train corpus was
read only to extract poses and to fingerprint it. **Nothing that re-selects episodes was run.**
🔒 No clip UUID or raw PhysicalAI content reaches any artifact (`episode_id` appears only as the
sorted-file index the fan dump itself uses).

**Suite green:** `cd stack && pytest -q` → **1264 passed, 12 skipped** in 83 s (2026-07-27, run at the
end of this stream). `git status --short stack/ taniteval/` is **empty** — this stream added **no
files** to either; all new code lives in this hub folder and **imports** the repo harness rather than
modifying it.
