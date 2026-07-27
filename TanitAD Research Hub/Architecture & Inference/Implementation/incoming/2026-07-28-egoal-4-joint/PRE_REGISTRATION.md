# E-GOAL-4 — PRE-REGISTRATION

**Written 2026-07-27 (wall-clock) BEFORE any number in `EGOAL_4.md` was computed.**
*(Folder named `2026-07-28-…` for the program's narrative clock, which runs ahead of wall-clock —
flagged, not silently absorbed.)*
**Stream:** `2026-07-28-egoal-4-joint` · **Host:** dev box CPU only.
⛔ **pod1 (training `flagship-v2corpus-30k`) and pod2 (120° cache build) are NOT contacted.**
**Estimator:** paired episode-cluster bootstrap, `taniteval/taniteval/ci.py`, **B = 2000**,
unit = **the val episode**, n = **600**. `overlapping_holdout_se` is **NEVER** called.

Amendments, if any, are appended in **§9** and nothing above them is edited.

---

## 1. THE QUESTION, AND THE SENTENCE THAT DEFINES IT

E-GOAL-3 closed with:

> ⛔ **NOT LICENSED: that a jointly trained v5 selector inherits +46 %. This is a goal injected into
> a *frozen* fan through a *fixed* rule — that is the next experiment, and it can fail.**

**E-GOAL-4 measures whether the recovery survives when the fixed rule is replaced by a TRAINED
SELECTOR that consumes the goal as an input feature.**

⚠️ **What is NOT in scope, stated so the scope cannot drift later.** The REF-C-XL fan stays frozen
(no GPU is available to this stream, and re-training REF-C is not a one-day experiment). The
"joint training" tested here is **joint training of the SELECTOR with the goal**, not of the
proposal generator. That is the coupling the v5 design question actually turns on — v5's selector is
the object that would carry a goal input — and it is named as such throughout.

---

## 2. THE COUPLING, NAMED

**`S_goal` — a per-candidate learned selector over the frozen 256-anchor REF-C-XL fan, trained
end-to-end on each candidate's realised ADE, with the goal head's prediction as an input feature,
and evaluated by the pick it actually makes.**

| | |
|---|---|
| rows | one per **(window, candidate)** — 13 198 × 256 = **3 378 688** |
| label | `ade(fan[w, c], gt[w])` — the mean-over-4-waypoint L2 that `ade_0_2s` is defined as, `eg_place.ade` **IMPORTED** |
| pick | `argmin_c score(w, c)` |
| scored | `ade(fan[w, argmin], gt[w])` — the **identical** function every prior stage used |
| fit | `HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)` — **IDENTICAL for every arm**, so extra columns can only help through held-out generalisation |
| folds | **5 episode-disjoint folds**, `eg_common.clip_folds(epi.astype(str), k=5, seed=0)` — **the same function, the same input, the same seed as the goal head's own folds** (gate G-2) |

**Feature blocks.**

- `F_ans` — per-candidate, **fan-only**: logit, softmax weight, logit rank, endpoint along/cross,
  1 s along/cross, path length, mean speed, |heading at 2 s|, curvature proxy.
- `F_ctx` — per-window, replicated over candidates: `v0`, `ax_fd`, CV endpoint along, the
  as-trained selector's own endpoint along/cross, max logit, logit entropy.
- ⭐ `F_goal` — **THE TREATMENT**: `g_along`, `g_cross`, `d_along = end_along − g_along`,
  `d_cross = end_cross − g_cross`, and **`d_rule`** = the fixed rule's own statistic
  (`mean-L2 of the candidate to goal_reference(goal)`).

⭐ **`d_rule` is included on purpose.** With it in the feature set the learner **can express the
fixed rule exactly** (`argmin d_rule` is one split on one column). If `S_goal` still under-performs
the fixed rule, that cannot be blamed on model class — retraction class
`CONTROL-WEAK-BY-MODEL-CLASS` is pre-emptively closed.

**Goal supplier:** E-GOAL-3's `T_OOF|H_v0_ax` — the **`v` + `ax_fd`** head the brief names, out-of-fold
on the same 5 folds. `T_OOF|H_ego` is a registered robustness check.

---

## 3. ARMS — registered in advance, all scored at n = 600

| arm | features | role |
|---|---|---|
| `A0` | — | the as-trained REF-C-XL selector (`sel` from the dump). **The reference every paired CI is against.** |
| ⭐ `S_nogoal` | `F_ans` + `F_ctx` | **THE CONTROL THE VERDICT RESTS ON** — a trained selector with no goal |
| ⭐⭐ `S_goal` | `F_ans` + `F_ctx` + `F_goal` | **THE TREATMENT** |
| `S_goalonly` | `F_goal` only | can the learner express the rule at all? |
| ⛔ `S_goal_shuf` | as `S_goal`, goal **permuted ACROSS EPISODES** | **C31 negative control, at MY n.** Must not help. |
| `S_goal_cv` | as `S_goal`, goal = `2·v0` (constant velocity) | an information-poor goal, trained |
| `S_goal_oracle` | as `S_goal`, goal = the TRUE 2 s endpoint | ⚠️ **bound, never a capability** (`BOUND-QUOTED-AS-CAPABILITY`) |
| ⛔ `S_LEAK` | as `S_goal` + **`head_deg`** (the future net heading change) | **the C23 POWER demonstration** — a known future field, fed deliberately |
| `FIXED_goal` | no fit | E-GOAL-3's rule with the **same goal, same background** — the comparator |
| `FIXED_oracle` | no fit | the rule with the true goal (E-GOAL-3's `P_ORACLE_TRUE`) |
| `FIXED_cv` | no fit | the rule with a `2·v0` goal (E-GOAL-3's `CV_head`) |
| `S_goal_INSAMPLE` | as `S_goal`, fit and scored on the **same** rows | the **in-sample vs out-of-fold** gap |
| `S_goal_coadapt` | as `S_goal`, but training rows carry the goal head's **IN-SAMPLE** prediction | ⭐ **the co-adaptation failure mode, measured directly** |

---

## 4. ⛔ C30 — THE BACKGROUND, NAMED IN ADVANCE AND HELD FIXED

Recovery spans **+13.3 % … +29.2 %** purely on the cross-track background (E-GOAL-2, 15.9 points),
replicated at 15.8 by E-GOAL-3, **and separation can flip inside it.**

- ⭐ **PRIMARY: `parent_resampled`** — E-GOAL-2's registered conservative carrier and **E-GOAL-3's
  headline background**, so the +46.3 % this stream is testing is quoted on the same carrier.
  ⚠️ **Drawn ONCE, from `default_rng(5000 + 0)`, and used BIT-IDENTICALLY by every arm — trained and
  fixed alike.** (E-GOAL-3 averaged 16 draws; a trained arm would need 16 refits per fold. The
  16-seed fixed-rule value is reported beside the 1-seed one so the single draw is shown to be
  representative — **registered as a gate, G-4, with a ±1.5 recovery-point tolerance**.)
- ⭐ **CO-PRIMARY: `sel`** — the REF-C selector's own 2 s endpoint. Zero fitting, and ⭐ **FUTURE-BLIND**
  (see §5). Every arm is re-run here.

⚠️ **Registered in advance, because it is the single biggest threat in this design:**
`parent_resampled`'s cross column is `true_cross + resampled_residual` — **future-derived BY
CONSTRUCTION**. A fixed projection can only use it one way; **a trained selector can learn to invert
it.** ⇒ **If the two backgrounds disagree, the `sel` cell is the one that survives the C23 audit and
the verdict is read there.** Stated now, not after seeing the numbers.

---

## 5. ⛔ C23 — THE FUTURE-CONTENT AUDIT (Priority 1, runs BEFORE anything is fitted)

1. **By definition** — every fed column enumerated with its exact expression and the offsets it
   reads. A runtime assertion refuses any field in
   `{gt, a_gt, head_deg, v_target, vt_valid, vt_lookahead, speed, y_long, y_lat}` reaching an arm's
   `X`, **except** the two arms explicitly labelled as bounds/power-controls (`S_goal_oracle`,
   `S_LEAK`).
2. **Empirically, over ALL 13 198 windows** — E-GOAL-3's `future_blind` corruption (overwrite every
   pose row after `L` with `N(10⁴, 10⁴)` and recompute). Required: `max |Δ|` **= 0.0** on every
   pose-derived fed column.
3. ⭐ **POWER, demonstrated three ways, not asserted:**
   - the **target** must move on 100 % of windows under the identical corruption;
   - the same instrument, same code, must **FIRE** on `parent_resampled`'s cross column and return
     **exactly 0.0** on `sel`'s — characterising the background *and* proving the test discriminates;
   - ⛔ `S_LEAK`, fed the future `head_deg`, must be **separated-better than `S_goal`**, or the
     end-to-end pipeline cannot detect a real leak and no clean result from it is quotable.

---

## 6. THE LEAK CHECK — BY CONTENT, WITH THE PATH AND THE COUNT

This stream **trains inside the 600 val episodes**, so the leak surface is *between its own folds*.

| check | required |
|---|---|
| **L-1 — fold disjointness by CONTENT** (sha256 of raw `poses[T,4]` float32 bytes, from E-GOAL-3's staged per-episode fingerprints) | every fold pair shares **0** episodes **by fingerprint** |
| **L-2 — the same comparison BY FILENAME** | reported as the contrast (prior streams: 600/600 by name, 0/600 by content) |
| **L-3 — internal collisions** | 600 unique sha256 of 600 |
| **L-4 — the goal head's folds == the selector's folds** (gate G-2) | bit-identical assignment, else fold *f*'s goal came from a head that saw fold *f* |
| **L-5 — val600 × parity train `physicalai-train-e438721ae894`** | 0 overlap by content (inherited surface; re-derived from the staged fingerprints, path reported) |

---

## 7. THE DECISION RULE — and the FAILING VALUE of each branch

`rec(X) = (mean(A0) − mean(X)) / headroom`, `headroom = a0 − R_goal2s`.
All out-of-fold. All under **`parent_resampled`**, with **`sel`** reported beside it.

| verdict | condition |
|---|---|
| ⭐ **CONFIRM** | `S_goal` separated-**better** than `A0` **AND** separated-**better** than `S_nogoal` **AND** `rec(S_goal) ≥ +37.0 %` *(= 80 % of the fixed rule's +46.3 %)* |
| **PARTIAL** | both separations hold **AND** `rec(S_goal) < +37.0 %` |
| ⛔ **REFUTE** | `S_goal` **NOT** separated-better than `S_nogoal` (the selector learned to ignore the goal), **OR** `S_goal` **NOT** separated-better than `A0` |

**⭐ The verdict rests on the DIRECT CONTRAST `S_goal` vs `S_nogoal`, not on "does it separate".**
C31: both prior stages found their separation predicate non-discriminating at n = 600 (an
information-free arm separated at +9.1 % / +45.4 %). **The same is assumed true here until measured.**

### 7.1 What would make each branch FAIL, and how it is demonstrated **in this run's own data**

| branch | the input that makes it fire | demonstration registered in advance |
|---|---|---|
| ⛔ **REFUTE** | a null on `X vs S_nogoal` | ⭐ **`S_goal_shuf` vs `S_nogoal` MUST be a NULL** — an arm with a real goal from the wrong episode. If that null appears, REFUTE was reachable. |
| separated-**WORSE** than `A0` | a goal that harms the pick | ⭐ **`FIXED_cv`** (E-GOAL-3 measured −18.6 %) and **`S_goal_shuf`** |
| **PARTIAL** | a goal good enough to separate but not to reach +37 % | ⭐ **the goal-degradation ladder** — refit `S_goal` on `y + k·e` for `k ∈ {0, 1, 1.5, 2, 3}`; some rung must land separated-better and below +37 % |
| ⭐ **CONFIRM** | a real, usable goal | ⭐ **`S_goal_oracle`** — the true goal through the trained selector |

**If any of the four is NOT reachable in this run's data, the run is reported as structurally unable
to return that verdict**, and the headline says so.

### 7.2 The three named failure modes, each with its own measurement

| failure mode | measured by | fires when |
|---|---|---|
| **the selector learns to ignore the goal** | `S_goal` vs `S_nogoal` | null, while `FIXED_goal` is far better than `A0` |
| **goal head and selector co-adapt / overfit** | `S_goal_INSAMPLE` vs `S_goal`; `S_goal_coadapt` vs `S_goal` | in-sample recovery holds while OOF collapses |
| **helps, but far less than +46 %** | `S_goal` vs `FIXED_goal`, paired | `FIXED_goal` separated-better than `S_goal` |

⛔ **If PARTIAL or REFUTE fires it is reported plainly and NOT re-scoped.** The brief states this
would be the most valuable single result of the week, and this pre-registration commits to it in
advance: **a REFUTE means the +46 % is a property of the fixed rule, not of the information.**

---

## 8. FIDELITY GATES — the run is VOID if any fails

| gate | what it proves | required |
|---|---|---|
| **G-0** | the deployment re-derives from the fan | `a0` 0.5015, `R_goal2s` 0.1933, `oracle_in_fan` 0.1547, headroom 0.3082 — all to 4 dp vs E-GOAL-2/3 raw JSON |
| **G-1** | the fixed rule reproduces E-GOAL-3 | `FIXED_goal` (16 seeds, `T_OOF|H_v0_ax`) returns **+46.3 %** to within 1.0 recovery point of `e3_place_n600_parent_resampled.json` |
| **G-2** | the folds are identical to the goal head's | `clip_folds` assignment bit-identical (§6 L-4) |
| **G-3** | the label is the metric | `min_c label(w,c)` == `oracle_in_fan` per window, `max |Δ| < 1e-6`; and `label(w, sel[w])` == the per-window `a0`, `max |Δ| < 1e-6` |
| **G-4** | 1 background seed ≈ 16 | `FIXED_goal` at S=1 within **±1.5 recovery points** of S=16 |
| **G-5** | ⛔ **G-3 has power** | the same identity with rows shifted by one **must fail hard** |

---

## 9. AMENDMENTS

*(appended after the fact; nothing above is edited)*
