# E-GOAL-4 — does the +46.3 % survive JOINT TRAINING?

**Stream:** `2026-07-28-egoal-4-joint` · **Wall-clock date:** 2026-07-27
*(folder named for the program's narrative date, which runs ahead of wall-clock — flagged, not
silently absorbed).*
**Host:** dev box CPU only. ⛔ **pod1 (training `flagship-v2corpus-30k`) and pod2 (the 120° cache
build) were NEVER contacted; pod3 and eval were not used.** Total compute: **~68 min of dev-box CPU,
zero GPU-hours, zero pod load.**
**Pre-registration:** `PRE_REGISTRATION.md`, written **before any number below was computed**;
three amendments in its §9 (**A1** before any arm was fitted, **A2** after the S0 audit, **A3** after
a 4-arm core run), nothing above them edited.
**Estimator:** paired episode-cluster bootstrap, `taniteval/taniteval/ci.py`, **B = 2000**,
unit = **the val episode**, **n = 600**. `overlapping_holdout_se` is **NEVER** called.

---

## 0. HEADLINE

> ### ⭐⭐ **CONFIRM. The recovery SURVIVES joint training and is LARGER: a trained selector fed the goal reaches `+62.09 %` of the fan's headroom out-of-fold under the conservative `parent_resampled` background and `+64.08 %` under the FUTURE-BLIND `sel` background — against the fixed rule's `+46.34 %` / `+61.46 %`. It BEATS the fixed rule with the SAME goal on both (`−0.0479 [−0.0530, −0.0431]` and `−0.0081 [−0.0112, −0.0051]`, separated). None of the three registered failure modes occurred.**

> ### ⛔ **BUT +46.3 % IS NOT WHAT THE GOAL IS WORTH TO A TRAINED SELECTOR, AND SIZING v5 AGAINST IT WOULD OVER-CREDIT THE GOAL BY 1.76×. A trained selector with NO goal already recovers `+35.62 %`. The goal's own marginal, capacity-matched against a real goal from the WRONG episode, is `+26.31` recovery points — `−0.0811`, and IDENTICAL on both backgrounds (`[−0.0904, −0.0720]` and `[−0.0888, −0.0732]`). E-GOAL-3's number is measured against the AS-TRAINED selector and CONFLATES the decision rule with the information.**

> ### ⭐⭐⭐ **AND THE GOAL HEAD'S ACCURACY IS WORTH ALMOST NOTHING END-TO-END. A naive `2·v0` goal — along-track RMS `1.4490 m`, which the fixed rule turns into `−18.55 %`, separated-WORSE — delivers `+62.07 %` through the trained selector. `S_goal` vs `S_goal_cv` is `+2.01 [0.52, 3.47]` recovery points (`sel`) / `+3.89 [2.27, 5.55]` (`parent_resampled`). The SAME upgrade through the fixed rule is worth `+67.57` / `+64.89` points. ⇒ a `16.7×–33.6×` COLLAPSE in the value of goal accuracy.**

> ### ⛔ **THEREFORE E-GOAL-3's REQUIREMENT CURVE DOES NOT TRANSFER. "σ₀ = 1.2276 m break-even; a confidently wrong goal is destructive, not neutral" is a property of the FIXED RULE. On the identical degradation ladder the fixed rule goes destructive at `1.128 m` and reaches `−111.78 %` at `2.256 m`, while the TRAINED selector is still `+16.73 %`, separated-BETTER, at the same `2.256 m`. It never crosses zero anywhere in the grid.**

**⇒ v5's selector carries a goal input — but size it at `+26.3` recovery points, not `+46.3`, and
do NOT fund goal-head accuracy: what the slot supplies is a REFERENCE FRAME, not a prediction.**

---

## 1. ⛔ C23 — THE FUTURE-CONTENT AUDIT. Priority 1, run BEFORE anything was fitted

`raw/e4_audit.json` · `code/e4_audit.py`, `code/e4_common.py`

A jointly trained model is *more* exposed than a head: a fixed projection can only use a column one
way; **a learner can invert it.** So every fed column is declared with its expression, a runtime
guard refuses the dump's future fields, and the whole feature set is re-derived from a corrupted
future.

### 1.1 By definition — every fed column, traced to the COMMITTED code that produced it

`L` = the window's last **observed** frame. Provenance verified in
`taniteval/taniteval/refc_rerank.py::dump` and `stack/scripts/driving_diagnostic.py`, not from names.

| block | columns | what it reads | past? |
|---|---|---|---|
| **`F_ans`** (11) | `logit`, `softmax`, `logit_rank`, `end_along/cross`, `mid_along/cross`, `pathlen`, `mean_speed`, `abs_head_end`, `shape_dev` | `model(fw, nav_cmd, v0, steps)` where `fw = fr[t : t+WINDOW]` — **frames up to `L`** — and `nav_mode="follow_constant"` ⇒ `nav_cmd = None` (`refc_eval.resolve_nav`, verified: the `oracle` branch is the only one that touches future poses and it is not taken) | ✅ |
| **`F_ctx`** (7) | `v0`, `ax_fd`, `cv_end_along`, `sel_end_along/cross`, `logit_max`, `logit_ent` | `v0 = poses[L,3]`; `cv = baseline_waypoints(poses, L)` reads **`poses[L]` and `poses[L−1]` only** (verified in source); `ax_fd = (s[L]−s[L−1])/0.1` | ✅ |
| **`F_goal`** (5) | `g_along`, `g_cross`, `d_along`, `d_cross`, `d_rule` | `g_along` = E-GOAL-3's `T_OOF|H_v0_ax`, a function of `v` and `ax_fd`; `g_cross` = **the background — see §1.3** | see §1.3 |
| ⛔ **refused** | `gt`, `a_gt`, `head_deg`, `v_target`, `vt_valid`, `vt_lookahead`, `speed` | `head_deg = wrap(poses[L+20,2] − poses[L,2])` — **offset +20**; `a_gt` reads `poses[L+20,3]` | ⛔ future |

⭐ **The label is `ade(fan[w,c], gt[w])` — future BY DESIGN.** It is the supervision target and never
a feature; `assert_no_future` is a runtime guard, not a comment.

### 1.2 ⭐ EMPIRICALLY, over ALL 3 378 688 rows — and the instrument is shown to DISCRIMINATE

> **Method.** Overwrite every future-supplying field with `N(10⁴, 10⁴)` and re-derive **every** fed
> column. A column that reads the future **changes**.

| what | max \|Δ\| | reading |
|---|---:|---|
| ⭐ **every `F_ans` + `F_ctx` column, all 13 198 × 256 rows** | **`0.0`** | ✅ provably future-blind |
| **`F_goal` under `sel`** | **`0.0`** | ✅ **the future-blind background** |
| ⛔ **`F_goal` under `parent_resampled`** | **`5.2008 × 10⁴`** | ⛔ **FIRES — and it should** |
| ⭐ **POWER: the LABEL** | **`3.611 × 10⁴`**, moved on **13 198 / 13 198 windows (100.0 %)** | ✅ |

> ⭐⭐ **A blind test that could not fail proves nothing. This one returns `0.0` where it must, fires
> where it must, and its power is demonstrated on the label.**
> ⛔ **And what it caught is the design's biggest threat, pre-registered in §4 of the
> pre-registration BEFORE it was measured:** `parent_resampled`'s cross coordinate is
> `true_cross + resampled_residual` — **future-derived BY CONSTRUCTION**, inherited from E-GOAL-2/3
> where it is the *simulated* cross-track head. **That is why every result below is reported on BOTH
> backgrounds, and why `sel` — where the goal is provably future-blind — is the co-primary.**

**The negative-index trap**, inherited and re-reported: **600 of 13 198 windows (4.55 %)** — exactly
the first of every episode — would read `poses[−3]`, i.e. the end of the episode. E-GOAL-3's clamp
(`max(L+off, 0)`) is what stops it. Its pose-level `future_blind` result is `INHERITED`, re-read from
`e3_features_val.json`: **feature max \|Δ\| `0.0`, target moved on 13 198 / 13 198.**

⚠️ **What I could NOT re-run:** the pose-level corruption itself — the poses live on pod2, which is
running the 120° build and was not touched. It covers `v` and `ax_fd`. **The FAN- and GOAL-derived
columns E-GOAL-4 ADDS are audited above, by this stream, over every row.**

---

## 2. ⛔ THE LEAK CHECK — BY CONTENT, WITH THE PATH AND THE COUNT

`raw/e4_audit.json` → `LEAK` · `code/e4_audit.py`

**This stream trains INSIDE the 600 val episodes, so the surface that binds is BETWEEN ITS OWN
FOLDS** — not the parity corpus. Method: **sha256 over the raw `poses[T,4]` float32 bytes**, from
E-GOAL-3's staged per-episode fingerprints.
**Path fingerprinted:** `pod2:/root/valdata/physicalai-val-0c5f7dac3b11` — the same string
`refc_rerank.VAL` holds, i.e. the path the fan dump actually reads.
**Fingerprint source:** `…/2026-07-28-egoal-3-trained-head/raw/e3_features_val.json`.

| check | required | measured |
|---|---|---|
| ⭐ **L-1 — fold disjointness BY POSE CONTENT**, all **10** fold pairs | 0 shared | ⭐ **`0` shared sha256 in every pair**; 120 / 120 / 120 / 120 / 120 fingerprints per fold, **600 unique across folds** ✅ |
| **L-3 — internal collisions** | 0 | **600 unique of 600** ✅ |
| **L-4 / G-2 — the goal head's folds == the selector's** | bit-identical | ✅ `clip_folds(epi.astype(str), 5, 0)`, **2639/2640/2640/2641/2638 windows, 120 episodes each**; the fan-`eid` convention was checked and gives the **same** partition |
| ⚠️ **L-2 — the same surface BY FILENAME** | — | ⚠️ **600 / 600 = 100 %** vs **0 / 600 by content** (`INHERITED`, `e3_leak.json`). **The fingerprint is load-bearing, not decorative.** |
| **L-5 — val600 × parity train `e438721ae894`** | 0 | **0 / 600 by content** (`INHERITED`, re-read from `e3_leak.json`; this stream does not train on the parity corpus) |

⛔ **Parity untouched.** No episode was re-selected, `_epcache` was never written, no corpus was
built. The parity corpus was not read at all by this stream.
🔒 No clip UUID or raw PhysicalAI content reaches any artifact (`episode_id` appears only as the
sorted-file index the fan dump itself uses).

---

## 3. FIDELITY — six gates, and two of them are per-row identities with demonstrated power

| gate | what it proves | measured |
|---|---|---|
| **G-0** | the deployment re-derives from the fan | `a0` **0.5015**, `R_goal2s` **0.1933**, `oracle_in_fan` **0.1547**, headroom **0.3082** — **all four exact to 4 dp** vs E-GOAL-2/3 raw JSON ✅ |
| ⭐⭐ **G-1** | **the fixed rule reproduces E-GOAL-3** | ⭐ **max deviation `0.004` recovery points** over 8 cells (4 arms × 2 backgrounds). `FIXED_goal` returns **+46.34 %** vs E-GOAL-3's **+46.34 %**, `FIXED_oracle` **+77.41 %** vs **+77.41 %**, `FIXED_cv` **−18.55 %** vs **−18.55 %**, and under `sel` **+61.46 / +94.84 / −6.11 %** all to 0.004 ✅ |
| ⭐ **G-3** | **the label IS the metric, per row** | `label(w, sel[w])` vs the per-window `a0`: **max \|Δ\| = `0.0`**; `min_c label` vs `oracle_in_fan`: **`0.0`** ✅ |
| ⛔ **G-5** | **G-3 has power** | the identical comparison with rows shifted by one **fails hard: 5.8926** ✅ |
| **F-2** | the (episode, `L`) join | fan `v0` vs pose-derived `v`: **max \|Δ\| = `0.0` over 13 198 rows**; shifted by one: **35.4032** ✅ |
| **G-4** | 1 background seed ≈ 16 | max deviation **0.357** recovery points (tolerance 1.5) ✅ |
| ⭐ **G-6** | my goal head IS E-GOAL-3's | my per-fold refit of `GBM(v, ax_fd)` reproduces `T_OOF|H_v0_ax`: **max \|Δ\| = `0.0`** ✅ |

> ⭐ **G-1 is what makes every comparison below interpretable.** The trained arms and the fixed rule
> run through the *same* engine, the *same* fan, the *same* background draw and the *same* `ade`,
> and that engine returns E-GOAL-3's published cells to **four decimal places**.
> ⚠️ **Determinism:** `S_nogoal` returned **0.3917** in **three independent runs** (a 4-arm core run
> and both full runs) — bit-identical, as it must be, since it never touches the background.

---

## 4. ⭐⭐ THE RESULT — both backgrounds, every arm, n = 600

`raw/e4_select_{parent_resampled,sel}.json` · `raw/e4_summary.json` · `code/e4_select.py`

**The coupling, named:** *a per-candidate learned selector over the frozen 256-anchor REF-C-XL fan,
trained on each candidate's realised `ade_0_2s` (3 378 688 rows), with the goal head's prediction as
an input feature, evaluated end-to-end by the pick it actually makes.* 5 episode-disjoint folds,
out-of-fold, identical hyper-parameters for every arm.
⭐ **`d_rule` — the fixed rule's own statistic — is one of the fed columns**, so the learner *can*
express `argmin d_rule` exactly. `CONTROL-WEAK-BY-MODEL-CLASS` is closed by construction, and §5.3
shows it is closed empirically too.

| arm | ⭐ **`parent_resampled`** (conservative) | ⭐ **`sel`** (FUTURE-BLIND) |
|---|---|---|
| `A0` — the as-trained REF-C-XL selector | 0.5015 · 0 % | 0.5015 · 0 % |
| ⛔ `FIXED_shuf` | 12.8286 · **−3999.6 % WORSE** | 12.8221 · **−3997.5 % WORSE** |
| ⛔ `FIXED_cv` — a `2·v0` goal, fixed rule | 0.5598 · **−18.91 % WORSE** | 0.5204 · **−6.11 % WORSE** |
| `S_nogoal_z` | 0.4272 · +24.13 % | 0.4272 · +24.13 % |
| ⭐ **`S_nogoal`** — **trained, NO goal** | **0.3917 · +35.62 % BETTER** | **0.3917 · +35.62 % BETTER** |
| ⛔ `S_goal_shuf` — a REAL goal, WRONG episode | 0.3912 · +35.79 % | 0.3851 · +37.76 % |
| `S_goal_crossonly` *(A3)* | 0.3727 · +41.80 % | 0.3789 · +39.77 % |
| `FIXED_goal` — **E-GOAL-3's rule** | **0.3580 · +46.56 %** *(16-seed: +46.34 %)* | **0.3121 · +61.46 %** |
| `S_goalonly` — `F_goal` only | 0.3357 · +53.81 % | 0.3101 · +62.10 % |
| ⭐ **`S_goal_cv`** — a `2·v0` goal, **TRAINED** | **0.3221 · +58.22 %** | ⭐ **0.3102 · +62.07 %** |
| `S_goal_alongonly` *(A3)* | 0.3175 · +59.72 % | 0.3189 · +59.24 % |
| ⭐⭐ **`S_goal`** — **THE TREATMENT** | ⭐ **0.3102 [0.3006, 0.3205] · +62.09 %** | ⭐ **0.3040 [0.2930, 0.3157] · +64.08 %** |
| `S_goal_ego` — the 10-column head | 0.3089 · +62.51 % | 0.3048 · +63.84 % |
| `S_goal_coadapt` | 0.3078 · +62.85 % | 0.3029 · +64.45 % |
| ⛔ `S_LEAK` — **fed the future `head_deg`** | 0.3084 · +62.66 % | 0.3036 · +64.22 % |
| `S_goal_INSAMPLE` | 0.3028 · +64.47 % | 0.2964 · +66.57 % |
| `S_goal_oracle` *(bound)* | 0.1962 · +99.06 % | 0.1923 · +100.33 % |
| `S_goal_oracle2d` *(bound)* | 0.1758 · +105.68 % | 0.1758 · +105.68 % |

⚠️ **`S_nogoal`'s 0.3917 vs the "in-sample re-scoring ceiling 0.4907" is a CROSS-DEPLOYMENT
comparison and is NOT quoted as clearing a bar.** 0.4907 was measured on the **881-window /
40-episode** deployment where `a0` = 0.4714; this one has `a0` = 0.5015
(`MODEL_REGISTRY` rule 1: never mix deployments). What can be said within this deployment is the
paired contrast: `S_nogoal` is separated-better than `A0` by **−0.1098 [−0.1239, −0.0960]**.

---

## 5. ⭐⭐ WHAT THE NUMBER ACTUALLY MEANS — the decomposition the headline must carry

### 5.1 ⛔ +46.3 % is not the goal's marginal value. Against a trained selector it is +26.3 points.

The fixed rule's +46.3 % is measured against **`A0`, the AS-TRAINED selector**. The counterfactual
for *"does v5's selector need a goal INPUT"* is **a trained selector without one**.

| contrast | `parent_resampled` | `sel` | recovery points |
|---|---|---|---|
| ⭐⭐ **`S_goal` vs `S_nogoal`** | **−0.0816 [−0.0911, −0.0721] SEP** | **−0.0877 [−0.0957, −0.0792] SEP** | **+26.47 [23.39, 29.56]** / **+28.46 [25.70, 31.05]** |
| ⭐⭐ **`S_goal` vs `S_goal_shuf`** — capacity-matched | **−0.0811 [−0.0904, −0.0720] SEP** | **−0.0811 [−0.0888, −0.0732] SEP** | ⭐ **+26.31 both** |

> ⭐⭐ **The capacity-matched marginal is `+26.31` recovery points and it is IDENTICAL to four
> decimals on two backgrounds whose totals differ by 2 points and whose *fixed-rule* totals differ by
> 15.** That is the C30-clean number: **background-invariant, capacity-matched, out-of-fold.**
> ⇒ **E-GOAL-3's +46.34 % over-credits the goal by `1.76×`** relative to what a trained selector
> gains from it.

### 5.2 ⭐ A3 — which axis carries it, and how big the oracle-cross contamination actually is

| axis (vs `S_nogoal`) | `parent_resampled` | `sel` |
|---|---|---|
| ⭐ **along alone** (`S_goal_alongonly`) | **−0.0743 [−0.0820, −0.0665] · +24.11 pts** | **−0.0728 [−0.0806, −0.0648] · +23.62 pts** |
| **cross alone** (`S_goal_crossonly`) | −0.0190 [−0.0256, −0.0132] · +6.16 pts | −0.0128 [−0.0190, −0.0069] · +4.15 pts |
| ⭐ **cross ON TOP OF along** | ⭐ **−0.0073 [−0.0123, −0.0025] · +2.37 pts** | −0.0149 [−0.0187, −0.0109] · +4.83 pts |
| **along ON TOP OF cross** | −0.0625 · +20.28 pts | −0.0749 · +24.30 pts |

> ⭐⭐ **The along axis carries 83–91 % of the marginal, and — the point of A3 — the
> oracle-contaminated cross of `parent_resampled` is worth `+2.37` recovery points on top of along.
> The C23 threat is therefore BOUNDED AND SMALL, measured rather than argued.**
> ⭐ It also **replicates the program's own result** that the lever is *how far the car will travel*,
> not *where the road goes* — now end-to-end through a trained selector.

### 5.3 ⭐⭐⭐ THE MECHANISM — the fixed rule OBEYS the goal; the trained selector treats it as EVIDENCE

`raw/e4_infoprobe.json` · `code/e4_infoprobe.py`. **The pick read-out:** every arm's chosen candidate
has an along-track endpoint, so `RMS(end_along[pick] − y_true)` is the along-track goal the arm
**behaved as if it had** — including arms never given one.

| arm | effective along-track goal RMS (m) | the goal it was GIVEN |
|---|---:|---|
| `A0` — the as-trained selector | **1.4164** | — |
| ⭐ **`S_nogoal`** | ⭐ **0.9785** | **none** |
| `S_goal_shuf` | 0.9628 | a real goal from the wrong episode |
| ⭐⭐ **`S_goal_cv`** | ⭐⭐ **0.8060** | ⛔ **1.4490** (`2·v0`) |
| ⭐ **`S_goal`** | **0.7995** | 0.7519 (`v + ax_fd`) |
| `S_goal_oracle` | 0.2935 | 0.0 (the truth) |
| *best-in-fan by ADE* | *0.2882* | — |

> ⭐⭐⭐ **`S_goal_cv` is handed a goal with `1.4490 m` of along-track RMS and makes picks that target
> to `0.8060 m` — nearly TWICE as accurate as the goal it was given. The fixed rule cannot do that:
> it projects onto the goal, so its pick INHERITS the goal's error (`FIXED_cv` = −18.55 %,
> separated-WORSE).**
> ⭐ **And `S_nogoal` targets to `0.9785 m` with no goal at all** — it learns its own, from the ADE
> label. **"No goal" is not "no goal"; it is "an implicit one".** That is why the marginal is +26.3
> and not +46.3.

**Consequently the goal head's ACCURACY is worth almost nothing end-to-end:**

| contrast | `parent_resampled` | `sel` |
|---|---|---|
| ⭐⭐ **`S_goal` vs `S_goal_cv`** — learned `v+ax_fd` vs naive `2·v0` | **−0.0120 [−0.0171, −0.0070] · +3.89 [2.27, 5.55] pts** | **−0.0062 [−0.0107, −0.0016] · +2.01 [0.52, 3.47] pts** |
| the SAME upgrade through the **FIXED rule** (E-GOAL-3, raw JSON) | −18.55 % → +46.34 % = **+64.89 pts** | −6.11 % → +61.46 % = **+67.57 pts** |
| ⭐ **`S_goal_cv` vs `FIXED_cv`** — the SAME goal, two rules | **−0.2377 [−0.2518, −0.2234] · +77.13 pts** | **−0.2102 [−0.2253, −0.1952] · +68.20 pts** |

> ⭐⭐ **A `16.7×` (`parent_resampled`) to `33.6×` (`sel`) COLLAPSE in the value of goal accuracy** —
> and the effect is still *separated*, so this is a measured shrinkage, not an unpowered null.
> ⭐ **And `S_goal_cv` vs `S_goal_shuf` is `+22.42` / `+24.30` points, separated — a naive `2·v0`
> goal carries 85–92 % of the real head's entire marginal.** What the slot supplies is a **geometric
> reference frame**, not a prediction.

### 5.4 ⛔ THEREFORE E-GOAL-3's REQUIREMENT CURVE DOES NOT TRANSFER — the same ladder, two rules

Goal = `y_true + k·e` where `e` is the real head's residual. **Identical goals, identical background,
identical windows; only the decision rule differs.**

| k | along RMS (m) | ⭐ **TRAINED** | **FIXED** |
|---|---:|---|---|
| 0.0 | 0.000 | **+99.06 %** ✅ | +77.60 % ✅ |
| 0.5 | 0.376 | **+91.42 %** ✅ | +72.68 % ✅ |
| **1.0** | **0.752** | ⭐ **+62.09 %** ✅ | +46.56 % ✅ |
| 1.5 | 1.128 | **+35.85 %** ✅ | +11.48 % ✅ |
| 2.0 | 1.504 | **+22.30 %** ✅ **BETTER** | ⛔ **−28.54 % WORSE** |
| 3.0 | 2.256 | ⭐ **+16.73 %** ✅ **BETTER** | ⛔ **−111.78 % WORSE** |

> ⛔⛔ **E-GOAL-3's "gate the goal channel on measured accuracy — a confidently wrong goal is
> destructive, not neutral" is a PROPERTY OF THE FIXED RULE and it does NOT survive joint training.**
> The fixed rule crosses zero between 1.128 and 1.504 m; **the trained selector does not cross zero
> anywhere in the grid, and is still separated-BETTER at 2.256 m — 1.84× beyond E-GOAL-3's own
> σ₀ = 1.2276 m break-even.**
> ⇒ **the break-even σ₀ is not a property of the goal or the deployment. It is a property of the
> CONSUMER.**

### 5.5 The trained selector's advantage over the fixed rule GROWS with goal noise

| | `parent_resampled` (cross MAE 0.4023) | `sel` (cross MAE 0.2347) |
|---|---|---|
| `S_goal` vs `FIXED_goal` | **−0.0479 [−0.0530, −0.0431] · +15.54 pts** | **−0.0081 [−0.0112, −0.0051] · +2.63 pts** |
| `S_goalonly` vs `FIXED_goal` — the learner given **only the rule's own inputs** | −0.0224 · +7.27 pts | ⭐ **−0.0020 [−0.0038, −0.0001] · +0.65 pts** |

> ⭐ **`S_goalonly` reproduces the fixed rule to `0.65` recovery points when the goal is clean** —
> so the learner CAN express the rule, empirically as well as by construction, and
> `CONTROL-WEAK-BY-MODEL-CLASS` is closed. **The gain is in HOW the goal is consumed, and it scales
> with how wrong the goal is** (+0.65 → +7.27 points as cross MAE goes 0.23 → 0.40 m). **That is
> hedging under uncertainty, measured.**

---

## 6. ⛔ THE THREE REGISTERED FAILURE MODES — none occurred, and each was measured, not assumed

| failure mode | measurement | result |
|---|---|---|
| **the selector learns to ignore the goal** | `S_goal` vs `S_nogoal` | ⭐ **separated-better on BOTH backgrounds** (−0.0816 / −0.0877). **Did not occur.** |
| **goal head and selector co-adapt / overfit** | ⭐ `S_goal_coadapt` — the training rows carry the head's **IN-SAMPLE** predictions | ⭐ **−0.0023 [−0.0047, +0.0002]** and **−0.0011 [−0.0028, +0.0006]** — **NULL on both.** **Did not occur.** |
| **in-sample holds, out-of-fold collapses** | `S_goal_INSAMPLE` vs `S_goal` | **−0.0073 [−0.0094, −0.0053]** / **−0.0077 [−0.0109, −0.0051]** = **2.4 / 2.5 recovery points.** `S_nogoal`'s gap is **3.80**. **A 2–4-point generalisation gap. Did not occur.** |
| *(the brief's third: "helps but far less than +46 %")* | `S_goal` vs `FIXED_goal` | ⭐ the trained selector is **separated-BETTER than the fixed rule on both backgrounds.** **The opposite of the anticipated failure.** |

---

## 7. ⛔⛔ C31 — THE NEGATIVE CONTROL RE-RUN AT MY n, AND IT PARTLY FAILS

### 7.1 The separation predicate is confirmed NON-DISCRIMINATING at n = 600, a third time

⛔ **`S_goal_shuf` — a real goal from the WRONG episode — separates against `A0` at `+35.79 %` /
`+37.76 %`.** So does `S_nogoal`, which has no goal at all. ⇒ **"`S_goal` is separated" supports
nothing.** The verdict rests on the **direct paired contrasts** of §5, exactly as pre-registered.
*(C31 fires for the third consecutive stream.)*

### 7.2 ⛔ AND MY OWN "MUST BE NULL" CONTROL FIRES ON ONE BACKGROUND — reported, bounded, subtracted

| `S_nogoal` vs `S_goal_shuf` | required | measured |
|---|---|---|
| `parent_resampled` | NULL | ⭐ **+0.0005 [−0.0028, +0.0040] — null** ✅ |
| ⛔ **`sel`** | NULL | ⛔ **+0.0066 [+0.0035, +0.0097] — SEPARATED** |

⛔ **My negative control is not information-free, and I can say exactly why.** With a shuffled goal,
`d_along` is still a monotone transform of `end_along` *within* a window, and **`d_rule` is still a
genuine geometric statistic of the candidate** — its distance to a straight path at a plausible-but-
wrong scale. So `S_goal_shuf` is **capacity- and geometry-matched, not information-free.**

> **Magnitude: `2.14` recovery points, against a treatment of `28.46`** — **7.5 %.**
> ⇒ **the headline marginal is quoted as `S_goal` vs `S_goal_shuf` (`+26.31`, both backgrounds),
> which SUBTRACTS this component**, rather than the `S_nogoal` contrast which does not.
> **Stated as a defect in my own registered control, not worked around.**

### 7.3 ⛔ AND MY REGISTERED C23 POWER CONTROL IS WEAK — said plainly

`S_LEAK` — the trained selector fed **`head_deg`, a genuinely future field** (`poses[L+20,2]`) —
returns **−0.0017 [−0.0037, +0.0002]** and **−0.0004 [−0.0019, +0.0011]: NULL on both.**
⛔ **The registered end-to-end leak detector did not fire.** I do not claim power I did not
demonstrate.

⭐ **The power demonstration that DID work, and it is stronger:** `S_goal_oracle` — a goal column
that IS future-derived — moves the answer by **+0.1139 [+0.1071, +0.1210]** and **+0.1117
[+0.1048, +0.1188]**, i.e. **36.9 / 36.2 recovery points, separated.** Together with §1.2's
`0.0`-vs-`5.2 × 10⁴` discrimination and the label moving on 100 % of windows, **the pipeline is
demonstrably able to detect future content in a fed column.**

⭐ **And `S_LEAK`'s null is itself a finding:** a *future heading* field is worth ≈ 0 on this metric —
consistent with the program's measured *"even a perfect cross-track buys 2.9 %"* and with §5.2's
cross-axis result. **Not a broken instrument; a small effect.**

### 7.4 Every verdict branch was REACHABLE in this run's own data

| rule | fires on | ⭐ **demonstrated here?** |
|---|---|---|
| ⛔ **REFUTE** | a null vs `S_nogoal` | ⭐ **YES** — `S_nogoal` vs `S_goal_shuf` is **+0.0005 [−0.0028, +0.0040]**, a tight null at the actual n. **REFUTE was live.** |
| separated-**WORSE** than `A0` | a goal that harms the pick | ⭐ **YES, three ways**: `FIXED_cv` **+0.0583 [+0.0460, +0.0692]**, `FIXED_shuf` **+12.3271**, `FIXED_goal_k2.0` **+0.0880 [+0.0718, +0.1052]** |
| **PARTIAL** (separated-better but < +37.0 %) | a degraded goal | ⭐ **YES** — the ladder at **k = 1.5 returns +35.85 %, separated-better and BELOW the CONFIRM threshold**; k = 2.0 returns +22.30 % |
| ⭐ **CONFIRM** | a usable goal | fired, at +62.09 % / +64.08 % |
| a genuine **NULL** at n = 600 | two arms carrying the same information | ⭐ **YES, four of them** — `S_goal_coadapt`, `S_LEAK`, `S_goal_ego` (×2), CI widths **0.0015–0.0062** — **tight nulls, not unpowered ones** |

---

## 8. ⭐ TWO INHERITED CLAIMS RE-TESTED END-TO-END — one replicates, one is closed

- ⭐ **E-GOAL-3's C32 correction REPLICATES through a trained selector.** `S_goal_ego` (the full
  10-column head) vs `S_goal` (`v + ax_fd`) is **−0.0013 [−0.0044, +0.0018]** and **+0.0007
  [−0.0022, +0.0035] — NULL on both backgrounds.** ⇒ **`v` + one 0.1 s speed difference remains the
  whole feature list, now confirmed end-to-end and not only on head-fit RMS.**
- ⭐ **The "wrong loss" objection (A1.1) is closed.** The within-window-centred target `S_goal_z` is
  **separated-WORSE** than the registered raw-ADE loss (+0.0126 / +0.0042), and its marginal is
  **larger, not smaller** (`S_goal_z` vs `S_nogoal_z` = −0.1044 / −0.1189 = **+33.9 / +38.6 points**).
  **No conclusion depends on the loss.**
- ⭐ **The "ties" objection (A1.2) is closed by measurement.** Mean candidates tied at the score
  minimum: **1.005–1.15**, p95 **≤ 2**, ties present on **≤ 13.7 %** of windows and on **0.9–1.6 %**
  for `S_goal`. **The piecewise-constant score does not blur the ordering.**

---

## 9. VERDICT

### 9.1 ⭐⭐ CONFIRM, on the pre-registered rule — but the number to plan with is the MARGINAL

> **`S_goal` is separated-better than `A0` (−0.1914 / −0.1975) AND separated-better than `S_nogoal`
> (−0.0816 / −0.0877), with recovery `+62.09 %` / `+64.08 %` against a pre-registered CONFIRM
> threshold of `+37.0 %`. All three conditions hold on BOTH backgrounds, including the FUTURE-BLIND
> one. ⇒ CONFIRM.**
>
> ⭐ **The +46 % survives joint training. It is not a property of the fixed rule — the fixed rule was
> if anything an UNFAVOURABLE consumer, and the trained selector beats it with the same goal.**
>
> ⛔ **BUT the goal's own marginal contribution is `+26.31` recovery points (capacity-matched,
> identical on both backgrounds), not +46.3. Plan v5 with +26, not +46.**

### 9.2 ⭐⭐⭐ The finding that changes the v5 design, stated plainly

**The goal input is a REPRESENTATION, not an information channel — and this is measured, not argued.**

- ⭐ **P-1: `g_along` regressed back onto `v` and `ax_fd` gives `R² = 0.999894`**, and **G-6 proves
  the identity exactly** (`max |Δ| = 0.0`).
- ⛔ **`S_nogoal` is fed `v0` and `ax_fd`.** Under the future-blind `sel` background `g_cross` is
  `sel_end_cross`, **also already a fed column.**
- ⇒ ⛔ **Under `sel`, EVERY `F_goal` column is a deterministic function of columns the selector
  ALREADY HAS. The goal input contains ZERO information `S_nogoal` lacks — and it is still worth
  `+26.3` separated recovery points.**

⇒ **What the goal head buys is inductive bias: a task-shaped, low-dimensional summary that a
23-column gradient-boosted selector does not construct for itself.** That is a real and cheap win —
and it also means **the "a strategic brain must SUPPLY the goal" framing is not what makes this
work.** A goal becomes an information channel only if its supplier sees something the selector does
not; at the current feature list it does not.

### 9.3 What I refuse to conclude

- ⛔ **NOT that v5's selector will realise +62 %.** This is a **frozen fan** and a **frozen goal
  head**; only the *selector* is trained. Training REF-C's proposal generator jointly is a different
  object again, needs a GPU, and is not tested here.
- ⛔ **NOT that the goal input is worthless.** +26.3 points, separated, on both backgrounds, from two
  scalars already on the CAN bus, is a large and cheap win. What is refuted is its **size** (+46 →
  +26) and the **reason** it works (representation, not information).
- ⛔ **NOT that goal ACCURACY never matters.** It is worth 2–4 points *for this consumer, at this
  accuracy range, on this metric*. §5.4 shows the trained selector degrades gracefully rather than
  not at all — at k = 3 it is +16.7 %, well below +62 %.
- ⛔ **NOT that `S_nogoal`'s +35.6 % is a deployable selector.** It is a re-ranker of a frozen fan
  trained on `ade_0_2s` labels, out-of-fold within 600 val episodes. **It has never been trained on
  the parity train corpus and is not a v5 candidate as it stands.**
- ⛔ **NOT anything past 2 s**, and every number is a displacement/ADE number — **blind to collision.**
- **NOT that the cross-track axis is settled.** Both backgrounds are *given*; no real cross-track
  head was trained here.

---

## 10. ESCALATIONS — these must not sit in a file

1. 🔴 **`V5_PLAN.md §8` and `Gates/flagship-v5-retrain.PREP.md` item 6 must be updated by their
   owner. THE SIZE CHANGES AND THE REASON CHANGES.**
   - *"Size it against +46.3 % / +50.7 %"* → ⭐ **the goal's marginal against a TRAINED selector is
     `+26.3` recovery points** (`−0.0811`, separated, identical on `parent_resampled` and `sel`).
     **+46.3 % is the fixed rule's total against the AS-TRAINED selector and conflates the decision
     rule with the information.**
   - ⛔ **DO NOT FUND GOAL-HEAD ACCURACY.** A naive `2·v0` goal delivers **+62.07 %** through a
     trained selector; the learned `v + ax_fd` head buys **+2.01 / +3.89** points on top. Through the
     fixed rule the same upgrade was worth **+64.89**.
   - ✅ **The feature list is unchanged and now confirmed end-to-end:** `S_goal_ego` vs `S_goal` is a
     **null on both backgrounds**. `v` + `ax_fd` remains the whole list.
2. 🔴 **E-GOAL-3's requirement curve must be re-scoped by its owner — it is a property of the
   CONSUMER, not of the goal.** *"σ₀ = 1.2276 m break-even; gate the goal channel on measured
   accuracy; a confidently wrong goal is destructive, not neutral"* holds **only for the fixed rule.**
   On the identical ladder the trained selector is **separated-BETTER at 2.256 m**, 1.84× beyond that
   σ₀, and never crosses zero. **A break-even RMS needs its family, its deployment, its background —
   and now its DECISION RULE.**
3. 🟠 **`MODEL_REGISTRY.md`'s 0.4907 annotation should gain a deployment tag.** It is the in-sample
   re-scoring ceiling on the **881-window / 40-episode** deployment (`a0` 0.4714), and it is now being
   quoted against 600-episode numbers (`a0` 0.5015). This stream deliberately did **not** quote it as
   a bar; the next stream may not be as careful.
4. 🟠 **For `RETRACTION_LOG.md` — root-cause classes:**
   - ⭐ **`C34 — A LEVER MEASURED AGAINST THE WRONG COUNTERFACTUAL`** ⇒ **an improvement measured
     against the DEPLOYED baseline credits the treatment with everything that changed, including the
     decision rule it arrived with.** MEASURED: the goal's recovery is **+46.34 %** against the
     as-trained selector and **+26.31 points** against a trained selector without a goal — **1.76×**.
     **The check is: build the strongest baseline that does NOT have the treatment, and quote the
     marginal.** *(Sibling of `MARGINAL-MISTAKEN-FOR-CONDITIONAL`; here a conditional was quoted as
     if it were a marginal.)*
   - ⭐⭐ **`C35 — A REQUIREMENT CURVE IS A PROPERTY OF THE CONSUMER`** ⇒ **a break-even accuracy
     derived by running a signal through ONE decision rule does not transfer to another rule, and can
     invert.** MEASURED: at 2.256 m of along-track goal error the fixed rule is **−111.78 %,
     separated-WORSE**; the trained selector is **+16.73 %, separated-BETTER**, on the identical
     goals and windows. ⚠️ **Consequence had it stood: v5 would have gated its goal channel on an
     accuracy bar that does not apply to it.**
   - ⭐ **`C36 — AN INPUT CAN BE WORTH POINTS WHILE CARRYING NO INFORMATION`** ⇒ **a feature that is a
     deterministic function of features the model already has can still be worth a large, separated
     effect — as inductive bias.** MEASURED: `R² = 0.999894` and an exact refit identity, yet
     **+26.3** separated recovery points. **The check is: before attributing a gain to a new
     information channel, regress the feature onto the ones already fed.** ⚠️ **This inverts the
     usual reading — the finding is not that the feature is useless, but that funding its SUPPLIER is
     the wrong lever.**
   - **`C31` fires a third time**, and this stream's own registered "MUST BE NULL" control
     **separated on one background** (§7.2) because a shuffled goal still yields a real geometric
     feature. **A shuffle control is capacity-matched, not necessarily information-free.**
   - **`C23` power control failed** (§7.3): `S_LEAK`, fed a real future field, is a null. **Register
     more than one power demonstration; a single one can be a small effect rather than a proof.**

---

## 11. THREATS TO VALIDITY I COULD NOT REMOVE

| threat | direction | status |
|---|---|---|
| ⛔ **The fan is FROZEN and the goal head is FROZEN. Only the SELECTOR is trained.** | unknown | **The scope is stated in `PRE_REGISTRATION §1`, in advance.** A jointly trained *proposal generator* is a further experiment and needs a GPU. |
| ⛔ **`parent_resampled`'s cross is FUTURE-DERIVED by construction** (§1.2) | inflates the primary | **quantified: +2.37 recovery points on top of along** (§5.2), and every result is replicated on the future-blind `sel` background where it is **larger**, not smaller. |
| ⛔ **My "MUST BE NULL" control separates on `sel`** (§7.2) | inflates `S_goal` vs `S_nogoal` | **bounded at 2.14 of 28.46 points (7.5 %)**; the headline uses the capacity-matched contrast which subtracts it. |
| ⛔ **My registered C23 power control is a NULL** (§7.3) | weakens the leak demonstration | **stated**; power rests on `S_goal_oracle` (36 recovery points) and on §1.2's discrimination instead. |
| ⚠️ **1 background seed for the trained arms, 16 for the fixed rule** | small | **G-4: max deviation 0.357 recovery points** against a registered 1.5 tolerance. |
| ⚠️ **One model class (HistGradientBoosting).** A different selector architecture may have a different implicit-goal capacity, which moves `S_nogoal` and therefore the marginal. | **unknown, and it is the main limit on the +26.3** | The marginal is a statement about *this* selector. `S_goal_z` (a different objective) gives a **larger** marginal, which is weak evidence the direction is robust. **A neural selector is not tested.** |
| ⚠️ **`S_nogoal` is fed `v0` and `ax_fd`** — the goal head's entire input | makes the marginal a lower bound on what a goal from a *richer* supplier would buy | **stated, and it is the point of §9.2.** |
| ⚠️ **Trained inside 600 val episodes**, 5-fold OOF; not fitted on the parity train corpus | narrows what "deployable" means | E-GOAL-3 measured `T_TRAIN` > `T_OOF` for the head; not re-run here for the selector. |
| The dev box holds a corpus keyed `14231cd29c74`, not parity | — | **no parity-dependent step ran.** This stream read only the pod-built fan, E-GOAL-3's derived feature matrix and its staged JSON. |
| 2 s, displacement/ADE only | unknown, possibly large | §9.3 |

**Evidence classes.** §§1–8 are `MEASURED (ours)` with artifact paths and are **DECISION-GRADE**
(n = 600 episode clusters, paired episode-cluster bootstrap B = 2000, out-of-fold, episode-disjoint,
fold-disjointness verified by pose content). E-GOAL-3's +46.34 / +61.46 / −18.55 / +77.41 %, the
deployment quantities and `T_OOF|H_v0_ax` are `INHERITED` — **and every one of them is re-derived
here and reproduced** (G-0 exact to 4 dp; G-1 to **0.004** recovery points; G-6 to **0.0**). The
0.4907 in-sample ceiling and the `ISO` bars are `PUBLISHED (cited)` and are **not** used as bars
(§4). §9.3's "a jointly trained proposal generator" is `HYPOTHESIS`.

> **TIER: CONFIRMED (decision-grade)** — for *"a jointly trained selector fed an ego-kinematics goal
> retains a separated recovery out-of-fold on two backgrounds, exceeds the fixed rule, and the goal's
> capacity-matched marginal is +26.3 recovery points."*
> **ALSO CONFIRMED (decision-grade)** — for *"the value of goal ACCURACY collapses 16–34× under joint
> training, and E-GOAL-3's break-even σ₀ does not transfer."*
> **NOT LICENSED** — *"a v5 whose PROPOSAL GENERATOR is also trained jointly will realise +62 %"*, and
> *"the +26.3 points would survive a different selector architecture."*

---

## 12. DELIVERABLE MANIFEST

**Everything below is staged in the repo working tree (`git add`). I committed nothing and pushed
nothing.**

⚠️🔴 **BUT FOUR OF MY FILES WERE COMMITTED BY SOMEONE ELSE, MID-RUN — the CLAUDE.md hazard, live.**
I staged the S0 deliverables as soon as they were clean (banking incrementally, as briefed). A
sibling then committed **`5b53a1a` "v2 parity hole CLOSED…"**, which — because `git commit` takes the
**ENTIRE INDEX** — swept in `PRE_REGISTRATION.md`, `code/e4_audit.py`, `code/e4_common.py` and
`raw/e4_audit.json` **under a commit message about something else.** ✅ **Verified harmless in
content:** `git diff HEAD` on all four is **empty**, so the committed versions are the current ones,
and `PRE_REGISTRATION.md` is staged again with amendment **A3** on top. **Reported, not discovered in
an audit later** — this is the third recorded instance of that failure mode, and the first where the
sweeper and the swept were different agents running concurrently.

⚠️ **Exactly two inputs live outside the repo, and both are named here rather than left to an audit.**

| artifact | where | what |
|---|---|---|
| `EGOAL_4.md` | `repo:…/incoming/2026-07-28-egoal-4-joint/` | this document |
| `PRE_REGISTRATION.md` | same | bars, arms, failing-value proofs, decision rule — written before any fit; §9 carries **A1** (loss / ties / G-6), **A2** (the audit's finding), **A3** (the axis decomposition + the marginal), nothing above them edited |
| `code/e4_common.py` | same | ⭐ the feature builder, the column table that IS the C23 by-definition audit, the runtime future-field guard, the two named backgrounds |
| `code/e4_audit.py` | same | ⭐ S0 — C23 over all 3.4 M rows, the leak check by content, gates G-0/G-2/G-3/G-5/F-2 |
| `code/e4_select.py` | same | ⭐⭐ the jointly trained selector, all arms, gates G-1/G-4/G-6, the C31 contrasts, the degradation ladder |
| `code/e4_infoprobe.py` | same | ⭐ P-1 the information probe, P-2 the pick read-out, P-3 the cross axis |
| `code/e4_summary.py` | same | the two backgrounds side by side, the extra paired contrasts, the mechanical verdict |
| `raw/e4_audit.json` | same | ⭐ the C23 audit + the leak check (0 shared across all 10 fold pairs) |
| `raw/e4_select_parent_resampled.json` · `raw/e4_select_sel.json` | same | ⭐⭐ every arm, every contrast, every gate, both backgrounds |
| `raw/e4_select_*_realised.npz` | same | **per-window realised `ade_0_2s`, the pick index and the tie count for every arm** — every interval recomputable without the fan |
| `raw/e4_infoprobe.json` | same | ⭐ `R² = 0.999894`, the pick read-out for all 16 arms |
| `raw/e4_summary.json` | same | the cross-background table, the extra contrasts, the ladder |
| ⚠️ `fan_refc-xl-30k_600ep.pt` (123 MB) | **`pod2:/workspace/_egoal2/` + dev-box scratchpad** | E-GOAL-2's fan, md5 **`42ea6b09570bd84b5380b5715f81a453`** — verified on this host |
| ⚠️ `e3_val600_windows.npz` (0.8 MB) · `e3_head_preds.npz` (1.6 MB) | **`pod2:/workspace/_egoal3/` + dev-box scratchpad** | E-GOAL-3's feature matrix and head predictions |
| logs | `dev-box scratchpad:egoal4/runA.log`, `runB.log` | the two full runs (2164.8 s / 1700.2 s) |

### ⚠️ The inputs outside the repo — stated, not discovered later

**Nothing this stream PRODUCED is single-disk.** All three external files are *inputs*, all three are
E-GOAL-2/3's, all three already live in two places (pod2 + dev box), and all three have staged
producers (`e2_dump600.py`, `e3_features.py`, `e3_fit.py`). ⚠️ `.gitignore` bans large binaries
repo-wide and the fan is a gated-confidential-corpus derivative — **I did not `git add -f`.**
**Every number in this document is recomputable from the staged `raw/*.json` + `raw/*.npz` +
`taniteval/ci.py` without the fan.**

**Nothing left running.** No background process, no pod job, no PID. Total dev-box compute
**~68 min** (audit 3.9 s · run A 2164.8 s · run B 1700.2 s · probe 2.1 s · summary 22 s), **zero GPU**.

**Suite:** `cd stack && pytest -q` → **1320 passed, 12 skipped, 3 FAILED** in 101 s (2026-07-27).
⚠️ **All 3 failures are a CONCURRENT SIBLING's in-flight work, not this stream's.** They are in
`stack/tests/test_v5_trainer_v2_val.py`, which is **UNTRACKED** and was written at **12:01 — while
run A was still executing** — alongside that sibling's unstaged edits to
`stack/scripts/train_flagship_v4.py` and `stack/tanitad/data/parity.py` (the parity-enforcement
stream). ⭐ **With that one in-flight file ignored: `1298 passed, 12 skipped, 0 failed` in 99 s.**
**This stream added NO files to `stack/` or `taniteval/`** — `git status --short stack/ taniteval/`
shows only the sibling's three entries — and all new code lives in this hub folder and **imports**
the repo harness (`eg_place.ade`/`realise`/`goal_reference`/`pick_nearest_to`,
`eg_common.clip_folds`/`ci_paired`, `eg_fit.fit_predict`, `taniteval/ci.py`) rather than modifying it.
🔴 **Escalated to that sibling's owner rather than touched: those three tests are RED right now.**

⚠️ **A sibling agent's file (`…/incoming/2026-07-27-geometry-configurable/GEOMETRY_CONFIGURABLE.md`)
was already in the index when this stream started.** It is not mine and I did not commit — flagged so
that whoever commits knows the index is shared.
