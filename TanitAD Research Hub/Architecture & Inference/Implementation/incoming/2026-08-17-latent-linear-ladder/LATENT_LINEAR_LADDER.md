# ⛔ THE v6 OPERATIVE LATENT CANNOT LINEARLY REPORT THE EGO'S OWN SPEED — and the banked lead-gap signal is mostly an EGO-SPEED PROXY

**Date:** 2026-08-17 · **Branch:** `agent/arch-inf-20260803` · **Agent:** latent-linear-ladder
**Cites, and does not touch:** `…/incoming/2026-08-17-probe-positive-control/PROBE_POSITIVE_CONTROL.md`,
`…/incoming/2026-08-17-slot-probe-parity/SLOT_PROBE_PARITY.md`.
**Eval tier:** ⛔ **T0-DIAGNOSTIC.** A frozen-latent readout is a world-model diagnostic and is
**never** driving performance. No number here is an ADE, a closed-loop result, or a claim about how
the car drives.

---

## ⭐ THE ANSWER, IN ONE PARAGRAPH

**The sanity anchor FAILS, and the apparatus PASSES its positive control — so for once the negative
is a fact about the latent.** A ridge on `v6F-SW-30k@11250`'s operative cells recovers the **ego's
own speed** at **r = +0.247**, MAE **5.526 m/s [4.860, 6.232]** against a constant's **4.097**
(**K1 +1.429 [+0.636, +2.202], separated — it LOSES to a constant**), i.e. it explains **6.1 % of
the variance of the quantity the car most certainly knows about itself.** ⭐ **The calibration that
makes that number mean something:** the *identical* ridge, on the *identical* windows, split, seeds
and estimator, recovers a **synthetic DISTRIBUTED encoding of that same `v0`** at **r = +1.000**
(0.10× noise), **+0.994** (1×), **+0.968** (3×) and still **+0.724** at **10× noise** — so
⛔ **THE REAL v6 LATENT CARRIES LESS LINEARLY-READABLE EGO SPEED THAN A RANDOM PROJECTION OF IT
BURIED UNDER TEN TIMES ITS OWN MAGNITUDE IN NOISE.** The readout is not the limit; the latent is.
⭐ **And this is the full operative state, not a lossy view of it:** `V6Stack.cells` is
`z_op.reshape(..., n_cells, d_readout)` (`stack/tanitad/models/v6.py:3729`) — **a pure reshape,
bijective and lossless**, so flattening the cells *is* `z_op`.

⭐⭐ **AND THE THREAD THIS RUN WAS OPENED TO FOLLOW COLLAPSES UNDER ITS OWN TRIVIAL-PROXY CHECK.**
The precedent's sharpest new fact was *"under a linear readout the v6 latents beat the random-latent
null by ~1.8 m on lead gap, r +0.159 vs −0.018"*. **MEASURED HERE: a ridge on the EGO-SPEED SCALAR
ALONE — one feature — recovers the lead gap at 3.571 m [3.151, 4.040], r +0.683, R² +0.467,
K1 −1.562 [−2.023, −1.136] SEPARATED, PASS.** That is **1.9× better than the whole 2 048-dimension
latent** (6.713 m, r +0.159, K1 **+1.580 FAIL**) and it is **the only non-oracle arm in F-18 or its
successors ever to pass K1 on lead gap.** ⇒ ⛔ **Lead gap is largely a restatement of ego speed (the
time-gap law), and the latent's share of it is mostly that proxy: partialling `v0` out drops the
latent's lead-gap correlation from +0.159 to +0.052.** The ~1.8 m gap over the null is real and
reproduces exactly — it is simply **not evidence that the latent encodes agents.**

⭐ **THE LADDER IS NOT FLAT, AND ITS SHAPE IS THE ACTIONABLE PART.** Against a null that sits at
zero on every rung (|r| ≤ 0.030, r² ≤ 0.0009), the latent's linearly-decodable content is:
**coarse scene scale and longitudinal ego state at 2–8 % of variance** (`n_agents_all` r² 0.076,
`ego_v0` 0.061, `nearest_any` 0.048, `ego_accel` 0.035, `lead_gap` 0.025, `n_agents_grid` 0.020,
`lead_present` 0.009) — and **NOTHING AT ALL, indistinguishable from the null, on everything
ROTATIONAL or RELATIVE-MOTION**: `ego_yawrate` r² 0.004, `ego_curv` 0.0001, `lead_closing` 0.0000,
`lead_inv_ttc` 0.0001. ⇒ **The latent has a weak "how fast and how cluttered is this scene" signal
and no measurable yaw, curvature, or closing-speed content whatsoever.** That is a sharp statement
about the architecture and it lands directly on the LATERAL and LONGITUDINAL metric families.

⛔ **AND I FOUND AN INSTRUMENT DEFECT IN THE READOUT ITSELF — pc6 PENALISES ITS OWN INTERCEPT.**
`pc6_linear_readout.ridge_fit` builds `X.T @ X + alpha * np.eye(d)` on a design matrix whose last
column **is the bias**, so as `alpha` grows the prediction collapses toward **ZERO, not toward the
mean**. ⇒ **The readout is structurally unable to express the very constant K1 scores it against**,
the alpha sweep is silently truncated, and a no-signal arm is driven to an arbitrarily bad score
rather than to a tie. MEASURED on a synthetic target of mean 7.0 with no signal, MAE across
alpha 1e-2→1e8: **0.0001 → 1.146 → 6.631 → 7.000**. **The repair is run on the arm, the null AND
both controls (§7), and the headline survives it** — but every pre-existing `K1 fails` from this
readout, including the precedent's §2.3 table, carries this caveat.

⚠️ **WHAT THIS DOES NOT SAY.** It is a **LINEAR** readout: a quantity absent here could still be
present non-linearly, and this is not a claim that the world model is untrained. It is an
**EARLY-READ at 41.7 %** of training (11 250 of 30 000). ⚠️ And on **`lead_closing` / `lead_inv_ttc`
the positive control is WEAK by construction** — the oracle memory encodes positions only, no rate
features (`pc1_oracle_cache.py`: `[cx/60, cy/16, sin yaw, cos yaw, l/10, w/5, 1]`) — so those two
nulls are **"not linearly present"**, not **"the apparatus could have found it"** (§2.4).

---

## 0. ⛔ THE STAMPS THAT BIND EVERY NUMBER BELOW

1. **`v6F-SW-30k@11250`** unless another step is named. The checkpoint is part of the arm. The
   checkpoint ladder (§6) additionally reads `@2000 / @9000 / @9250 / @10000` from banked caches.
2. ⚠️ **EVERY v6 POINT IS AN EARLY-READ at 11 250 / 30 000 = 41.7 %.** 30 000 remains the primary
   read.
3. ⚠️ **LEAD-ENRICHED, NOT PARITY** — inherited unchanged from the parity run and the positive
   control (`parity: False` in every cache meta). 130 clips, 60 probe-train / **70 eval**, clip-
   disjoint.
4. **Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2000, clustered on the 70
   eval episodes.** ⛔ `overlapping_holdout_se` is never imported.
5. ⭐ **THE READOUT IS pc6's, AND EQUIVALENCE IS PROVED RATHER THAN ASSERTED.** `ll1_ladder.py`
   IMPORTS `ridge_fit` from `pc6_linear_readout` (no second implementation of the solve) and its
   `--gate-pc6` asserts that target `lead_gap` at seed 0 reproduces the banked
   `pc6_ridge_s11250.json`. **MEASURED, and it is exact:**

   | gate check | banked | ours |
   |---|---|---|
   | `ridge_err_m` | 6.7128 | **6.7128** |
   | `c_const_err_m` | 5.1329 | **5.1329** |
   | `K1_delta` | +1.5799 | **+1.5799** |
   | `corr_pred_gt` | +0.1592 | **+0.1592** |
   | `alpha_chosen` | 10.0 | **10.0** |
   | n train / eval windows | 2231 / 2721 | **2231 / 2721** |

   ⭐ **And the second, independent reproduction:** running the banked `ORC-DIRECT` cache through
   this ladder gives `lead_gap` **1.0165 m, r +0.979** against the precedent's **1.016 m, r 0.979**.
6. **GPU: none. This is a CPU ridge.** ⛔ **Thor was never used for compute** — no checkpoint was
   pulled and nothing was run there. Only banked local artifacts were read. Thor's live 30k run
   (PID **25477**) was **ALIVE at step 12 500** at the end of this run (§11).

---

## 1. ⭐ THE POSITIVE CONTROL, RUN FIRST — because a negative without one taught this programme nothing

The D1 withdrawal is the standing lesson: **five negative controls proved the slot probe was not
cheating and not one proved it could measure.** So the anchor's negative is inadmissible until the
readout is shown able to find ego speed when it is there.

`ll2_ego_oracle.py` takes the REAL @11250 cache and replaces **only** `cells`, keeping every window,
target, class, rate, `clip_id`, `episode_uid` and the declared split — so the paired bootstrap's
clusters are literally the same objects. Cell *k* = `P_k @ [v0/10, 1]` with an **independent random
`[128, 2]` projection per cell** (seed 20260817), rescaled to the real cells' global std
(**0.030898** — identical to `pc1`'s banked value) and noised at `noise_rel ×` that std.

⭐ **It is deliberately the DISTRIBUTED control, not the easy one.** The banked `ORC-DIRECT` puts its
answer at a **fixed address** (cell 0), which does not test whether a ridge with 2 049 features and
~2 200 training windows can find a signal **smeared across every dimension** — which is how a
learned latent would carry it. Here no feature is a copy of the label.

| `ego_v0`, n 3023 / 70 | MAE (m/s) | **K1** vs C-CONST 4.097 | **r** | r within-episode | R² |
|---|---|---|---|---|---|
| **EGO-ORACLE 0.10× noise** | **0.046** | **−4.051 PASS ✅** | **+1.000** | +1.000 | +1.000 |
| **EGO-ORACLE 1.0× noise** | **0.484** | **−3.613 PASS ✅** | **+0.994** | +0.973 | +0.988 |
| **EGO-ORACLE 3.0× noise** | **1.148** | **−2.949 PASS ✅** | **+0.968** | +0.871 | +0.933 |
| **EGO-ORACLE 10× noise** | 3.471 | −0.626 [−1.475, +0.130] **ns** | **+0.724** | +0.438 | +0.387 |
| ⛔ **`v6F-SW-30k@11250` (the real arm)** | **5.526** | **+1.429 [+0.636, +2.202] sep FAIL** | **+0.247** | +0.149 | **−0.547** |
| RANDOM-LATENT NULL | 5.215 | +1.118 sep FAIL | **−0.025** | −0.010 | −0.739 |

⇒ ⛔ **THE ONE-LINE STATEMENT OF THIS RUN:**

> **On a memory built from a random projection of the ego's own speed with TEN TIMES that signal's
> magnitude added as noise, this ridge recovers ego speed at r = +0.724. On the trained v6 operative
> latent it recovers it at r = +0.247.**

⇒ The pre-registered third outcome — *"nothing recoverable, not even ego speed ⇒ the cache or the
readout is broken, report that and stop"* — is **REFUTED**. The cache is sound (the pose grid binds
exactly, §0/§8), the readout is sound (above), and the equivalence gate passes. **The anchor's
failure is a property of the latent.**

⚠️ **A limit stated rather than hidden:** the control establishes the readout can find a *linear*
distributed encoding. It cannot establish that the readout would find an encoding that is present
but **non-linear**. Everything below is a statement about **linear decodability**.

---

## 2. THE LADDER

### 2.1 Correlation with the truth, every rung with its null and its controls

`r_wep` = the correlation after **both** prediction and truth are demeaned by their own eval episode
— the **ANTI-EPISODE-IDENTITY** statistic (within-episode gap SD is only ~4 m, so a readout that
merely identifies the episode scores well globally and ~0 here). `r_pv0` = the **partial**
correlation with **ego speed partialled out** — the TRIVIAL-PROXY test.

| rung | target | n win/clust | **v6 r** | **NULL r** | **C-V0 r** | ORACLE r | v6 `r_wep` | v6 `r_pv0` |
|---|---|---|---|---|---|---|---|---|
| EGO ⭐anchor | `ego_v0` | 3023/70 | **+0.247** | −0.025 | +1.000 | +0.548 | +0.149 | — |
| EGO | `ego_accel` | 3023/70 | **+0.187** | −0.011 | −0.112 | +0.021 | +0.115 | **+0.169** |
| EGO | `ego_yawrate` | 3023/70 | **+0.060** | −0.008 | −0.020 | +0.031 | +0.099 | +0.063 |
| EGO | `ego_curv` | 2221/70 | **−0.009** | −0.022 | −0.085 | −0.095 | +0.033 | +0.002 |
| SCENE | `n_agents_grid` | 3023/70 | **+0.141** | −0.015 | +0.322 | +0.874 | +0.042 | +0.094 |
| SCENE | `n_agents_all` | 3023/70 | **+0.275** | +0.027 | +0.335 | +0.660 | +0.101 | +0.194 |
| OBJECT | `lead_present` | 3023/70 | **+0.093** | −0.012 | +0.152 | +0.414 | +0.090 | +0.076 |
| OBJECT | `nearest_any` | 3019/70 | **+0.219** | −0.024 | +0.408 | +0.856 | **+0.013** | +0.179 |
| OBJECT | `lead_gap` | 2721/70 | **+0.159** | −0.018 | **+0.683** | +0.979 | +0.166 | **+0.052** |
| OBJ-DYN | `lead_closing` | 2721/70 | **−0.006** | +0.005 | +0.017 | +0.432 | +0.067 | −0.004 |
| OBJ-DYN | `lead_inv_ttc` | 2721/70 | **−0.008** | +0.030 | −0.036 | +0.118 | +0.032 | −0.002 |

⭐ **THE NULL IS CLEAN ON EVERY ROW** (|r| ≤ 0.030), which is what makes the small positives
readable at all. **Seven rungs are above the null and four are at it.**

### 2.2 ⭐ The same ladder as VARIANCE EXPLAINED — the fairest single number per rung

⚠️ **Why `r²` and not `R²`.** The fit is **over-dispersed**: it emits close to full variance at low
correlation (for `ego_v0`, `pred_sd` 5.420 against `gt_sd` 5.583 at r = 0.247), so its MAE loses to
a constant while `r` is positive and its `R²` is **negative**. `r2_ceiling = r²` is the variance a
**perfectly rescaled version of the same readout** would explain. Quoting both stops "K1 fails"
from being misread as "r is zero", and stops "r is positive" from being oversold.

| rung | target | **v6 r²** | NULL r² | C-V0 r² | ORACLE r² |
|---|---|---|---|---|---|
| SCENE | `n_agents_all` | **0.0758** | 0.0007 | 0.1124 | 0.4356 |
| EGO ⭐ | `ego_v0` | **0.0609** | 0.0006 | 1.0000 | 0.3000 |
| OBJECT | `nearest_any` | **0.0480** | 0.0006 | 0.1667 | 0.7325 |
| EGO | `ego_accel` | **0.0350** | 0.0001 | 0.0125 | 0.0004 |
| OBJECT | `lead_gap` | **0.0254** | 0.0003 | **0.4672** | 0.9583 |
| SCENE | `n_agents_grid` | **0.0200** | 0.0002 | 0.1038 | 0.7643 |
| OBJECT | `lead_present` | **0.0087** | 0.0001 | 0.0231 | 0.1713 |
| EGO | `ego_yawrate` | **0.0036** | 0.0001 | 0.0004 | 0.0010 |
| OBJ-DYN | `lead_inv_ttc` | **0.0001** | 0.0009 | 0.0013 | 0.0139 |
| EGO | `ego_curv` | **0.0001** | 0.0005 | 0.0071 | 0.0089 |
| OBJ-DYN | `lead_closing` | **0.0000** | 0.0000 | 0.0003 | 0.1865 |

⇒ ⭐ **THE SHAPE, STATED PLAINLY.** There is a **cliff, not a slope**. Everything the latent carries
is **coarse scene scale and longitudinal ego state, at 2–8 % of variance**. Everything **rotational**
(`ego_yawrate`, `ego_curv`) and everything **relative-motion** (`lead_closing`, `lead_inv_ttc`) is
**at the null**. The programme's LATERAL family (curvature error, yaw-rate error) and the
longitudinal family's *distance-keeping* half (TTC, closing) are asking the planner to use
information **that is not linearly in the state it is planning from**.

### 2.3 ⛔ Every rung in error and K1 — no row without its null

Positive K1 = the arm is **worse** than the constant. C-CONST is the **probe-train median**.

| target | unit | C-CONST | **v6 err / K1** | NULL err / K1 | C-V0 err / K1 | ORACLE err / K1 |
|---|---|---|---|---|---|---|
| `ego_v0` | m/s | 4.097 | **5.526 / +1.429 FAIL** | 5.215 / +1.118 FAIL | 0.000 / **−4.097 PASS** | 3.753 / −0.344 ns |
| `ego_accel` | m/s² | 0.5095 | 0.5049 / −0.005 ns | 0.5051 / −0.004 ✅* | 0.5029 / −0.007 ✅* | 0.5126 / +0.003 ns |
| `ego_yawrate` | rad/s | 0.0285 | 0.0332 / +0.005 FAIL | 0.0288 / +0.000 FAIL | 0.0285 / +0.000 FAIL | 0.0326 / +0.004 FAIL |
| `ego_curv` | 1/m | 0.0069 | 0.0083 / +0.001 FAIL | 0.0070 / +0.000 FAIL | 0.0069 / −0.000 ns | 0.0081 / +0.001 FAIL |
| `n_agents_grid` | ct | 9.798 | 10.381 / +0.583 ns | 10.994 / +1.195 FAIL | 9.777 / −0.021 ns | 3.088 / **−6.710 PASS** |
| `n_agents_all` | ct | 37.936 | 39.742 / +1.806 ns | 43.851 / +5.915 FAIL | 37.932 / −0.004 ns | 26.737 / **−11.198 PASS** |
| `lead_present` | prob | 0.0999 | 0.339 / +0.239 FAIL | 0.426 / +0.326 FAIL | 0.205 / +0.105 FAIL | 0.229 / +0.129 FAIL |
| `nearest_any` | m | 5.484 | 6.538 / +1.054 FAIL | 6.563 / +1.079 FAIL | 5.405 / −0.079 ns | 3.032 / **−2.452 PASS** |
| `lead_gap` | m | 5.133 | **6.713 / +1.580 FAIL** | 8.534 / +3.401 FAIL | **3.571 / −1.562 PASS** | 1.017 / **−4.116 PASS** |
| `lead_closing` | m/s | 1.112 | 1.141 / +0.029 FAIL | 1.105 / −0.007 ns | 1.112 / −0.000 ns | 1.095 / −0.017 ns |
| `lead_inv_ttc` | 1/s | 0.0802 | 0.0813 / +0.001 ns | 0.0795 / −0.001 ns | 0.0798 / −0.000 ✅* | 0.0783 / −0.002 ns |

\* ⚠️ **These three "PASS" marks are DEGENERATE and must not be quoted as skill.** They are
separated-and-negative by **0.004–0.007 units** on targets where every arm ties the constant to
three decimals, and the **random-latent null earns two of them**. A criterion that a noise arm
passes is not reporting skill — the same class as the precedent's `K3`. They are printed rather
than filtered because filtering them would be choosing which controls to believe.

⛔ **`v6` NEVER PASSES K1 ON ANY RUNG.** The only non-degenerate passes in the table belong to the
GT oracle and to **one scalar of ego speed**.

### 2.4 ⚠️ WHERE THE POSITIVE CONTROL DOES NOT REACH — stated, not glossed

The `ORACLE` column is `ORC-DIRECT`, whose cells encode **positions and extents only**:
`[cx/60, cy/16, sin yaw, cos yaw, l/10, w/5, presence]` (`pc1_oracle_cache.py`). **There are no rate
features in it.** So on `lead_closing` (oracle r +0.432) and `lead_inv_ttc` (oracle r **+0.118**,
r² 0.014) the oracle is a **weak** upper bound — it only reaches those quantities through their
correlation with geometry. ⇒ **For the two OBJECT-DYNAMICS rungs the honest verdict is "not
linearly present in the latent", NOT "an apparatus that could have found it did not".** The
ROTATIONAL rungs are worse served still: `ego_yawrate` and `ego_curv` have **no positive control at
all** in this run (the oracle scores +0.031 / −0.095 on them, i.e. it has no ego content either), so
their nulls are the weakest rows in the ladder. **Building an EGO-ORACLE for yaw-rate is one line of
`ll2_ego_oracle.py` and is the first item in §10.**

---

## 3. ⭐⭐ THE TRIVIAL-PROXY FINDING — the reason the precedent's new fact does not survive

The brief's own warning: *"if a quantity is recoverable, ask whether it is recoverable from
something TRIVIAL — a positive that is really a proxy is worse than a null."* `C-V0` is the
identical ridge with the 2 048-dimension latent replaced by **the single scalar `v0`**.

| `lead_gap`, n 2721 / 70 clusters | MAE (m) | **K1** vs C-CONST 5.133 | **r** | **r²** | `r_wep` |
|---|---|---|---|---|---|
| `ORC-DIRECT` (GT oracle) | 1.017 [0.989, 1.044] | −4.116 **PASS ✅** | +0.979 | 0.958 | +0.953 |
| ⭐ **`C-V0` — ONE FEATURE, ego speed** | **3.571 [3.151, 4.040]** | **−1.562 [−2.023, −1.136] sep PASS ✅** | **+0.683** | **0.467** | +0.711 |
| ⛔ **`v6F-SW-30k@11250` — 2 048 features** | **6.713 [6.010, 7.482]** | **+1.580 [+0.765, +2.455] sep FAIL** | **+0.159** | 0.025 | +0.166 |
| RANDOM-LATENT NULL | 8.534 [8.148, 8.936] | +3.401 sep FAIL | −0.018 | 0.000 | +0.004 |

⇒ ⛔ **Three things follow, and the third is the one that matters.**

1. **The precedent's ~1.8 m gap over the null REPRODUCES EXACTLY** (8.534 − 6.713 = **1.821 m**) and
   is not disputed. What is disputed is what it licenses.
2. ⭐ **A single scalar of ego speed beats the entire latent by 3.14 m and passes K1 where the
   latent fails by 1.58 m.** Drivers hold headway roughly proportional to speed; the lead gap is
   therefore **46.7 % predictable from `v0` alone** with no perception whatsoever.
3. ⛔ **The latent's lead-gap signal is mostly that proxy.** Partialling `v0` out collapses it from
   **r +0.159 → +0.052** — about **⅔ of an already tiny correlation is ego speed re-expressed.**
   ⇒ **"The latent carries some linearly-decodable lead information" is not supportable as a claim
   about AGENTS.** The residual after `v0` is r 0.052, r² **0.0027**.

⭐ **The same pattern on presence, by AUC** (the natural metric for a binary rung, reported because
MAE on a 90 %-positive label is uninformative): `lead_present` AUC — **ORACLE 0.796 · C-V0 0.673 ·
v6 0.586 · NULL 0.484.** **Ego speed alone detects the presence of a lead better than the world
model's own state does.**

---

## 4. EPISODE IDENTITY — the leakage the brief warned a rich latent is MORE prone to

`C-EPMEAN` (leave-one-out mean of the window's own eval episode) is carried on every fit as **K5**,
and `r_wep` is reported on every row. **v6 fails K5 on every rung** — it never beats the
episode-identity oracle. The sharper statistic is what survives episode demeaning:

| target | v6 r | **v6 `r_wep`** | reading |
|---|---|---|---|
| `nearest_any` | +0.219 | **+0.013** | ⛔ **essentially ALL episode identity** |
| `n_agents_grid` | +0.141 | **+0.042** | ⛔ mostly episode identity |
| `n_agents_all` | +0.275 | **+0.101** | ⚠️ ~⅔ episode identity |
| `ego_v0` | +0.247 | **+0.149** | ✅ genuine within-episode content |
| `ego_accel` | +0.187 | **+0.115** | ✅ genuine within-episode content |
| `lead_gap` | +0.159 | **+0.166** | ✅ not episode identity (but see §3 — it is `v0`) |

⇒ ⭐ **The SCENE-density rungs are largely the readout recognising WHICH CLIP it is in**, which is
exactly the failure mode the brief flagged. **The EGO rungs are not** — their (small) signal is real
within-episode variation. **So the honest ranking of what the latent actually encodes is narrower
than §2.2's raw `r²` column suggests: longitudinal ego state, weakly, and little else.**

---

## 5. SEED SPREAD — and an honest statement that ≥3 seeds is a WEAK variance notion here

The brief requires ≥3 seeds and a between-condition vs between-seed comparison, because the parity
run measured **3.096 m of K1 spread across three seeds** on one frozen cache. **Three seeds were run
on every arm and every rung.** ⚠️ **But a ridge is a CLOSED-FORM SOLVE: there is no optimiser seed.**
The only seed-dependent step is the episode-disjoint inner split that chooses `alpha`.

| target | v6 err seed-range | v6 K1 seed-range | |v6 − NULL| err gap |
|---|---|---|---|
| `ego_v0` | **0.0000** | **0.0000** | 0.311 |
| `lead_gap` | **0.0000** | **0.0000** | **1.821** |
| `n_agents_all` | 1.1145 | 1.1145 | 4.109 |
| `nearest_any` | 0.8300 | 0.8300 | 0.025 |
| *the other 7 rungs* | **0.0000** | **0.0000** | — |

⇒ **8 of 11 rungs have EXACTLY ZERO seed spread** (the alpha search lands on the same value), and
the largest is 1.11 units. **Between-condition ≫ between-seed on the headline rows** — unlike the
slot probe, where seed spread exceeded the checkpoint spread. ⚠️ **This is not a strength of the
result, it is a property of the estimator**, and it means **the ≥3-seed rule does not supply
uncertainty here**. The uncertainty that counts is the **episode-cluster bootstrap CI**, which is
quoted on every headline number — and per the standing rule it resamples **eval episodes, not
fits**. ⚠️ **`nearest_any` is the one rung where the seed range (0.830) EXCEEDS the arm-vs-null gap
(0.025); no ordering is claimed there.**

---

## 6. THE CHECKPOINT TRAJECTORY — 9 250 steps of training changed nothing measurable

Five banked caches, identical windows, split, seeds and estimator; `r` per rung.

| target | @2000 | @9000 | @9250 | @10000 | @11250 |
|---|---|---|---|---|---|
| `ego_v0` | **+0.263** | +0.214 | +0.222 | +0.267 | +0.247 |
| `ego_accel` | +0.141 | +0.175 | +0.188 | **+0.189** | +0.187 |
| `ego_yawrate` | +0.022 | +0.054 | +0.058 | **+0.066** | +0.060 |
| `ego_curv` | −0.006 | −0.004 | −0.004 | +0.001 | −0.009 |
| `n_agents_grid` | **+0.174** | +0.147 | +0.136 | +0.105 | +0.141 |
| `n_agents_all` | +0.274 | +0.178 | +0.190 | +0.220 | **+0.275** |
| `lead_present` | +0.065 | +0.091 | +0.065 | +0.060 | **+0.093** |
| `nearest_any` | **+0.274** | +0.210 | +0.179 | +0.181 | +0.219 |
| `lead_gap` | +0.137 | +0.108 | +0.060 | +0.151 | **+0.159** |
| `lead_closing` | −0.003 | +0.014 | −0.008 | **+0.030** | −0.006 |
| `lead_inv_ttc` | −0.011 | −0.011 | −0.010 | −0.006 | −0.008 |

⇒ ⭐ **The linear content of the latent is essentially SET BY STEP 2 000 and does not grow.** Four
rungs are at their **best at @2000** (`ego_v0`, `n_agents_grid`, `nearest_any`, and `n_agents_all`
within 0.001). The only rungs that improve monotonically-ish are `ego_accel` (+0.141 → +0.187) and
`ego_yawrate` (+0.022 → +0.060), both still tiny. ⚠️ **No trend is claimed as significant** — these
are point `r` values without per-checkpoint CIs, and `nearest_any`'s own seed range (0.830 in MAE)
shows how little a single rung's motion means. **What IS claimed is the absence of a large effect:
nothing here moves by more than ~0.1 in `r` across 9 250 steps**, which is the relevant fact for
whether waiting for 30 000 will change the verdict. ⚠️ It might; this is an early read.

---

## 7. ⛔ THE INSTRUMENT DEFECT AND ITS REPAIR — pc6 penalises its own bias term

### 7.1 The defect

`pc6_linear_readout.ridge_fit` is

```python
A = X.T @ X + alpha * np.eye(d)          # d INCLUDES the appended ones-column
```

and `prep()` appends a bias column to `X`. **The intercept is therefore shrunk like any other
coefficient**, so `pred → 0` as `alpha → ∞`, not `pred → mean(y)`.

**MEASURED** (`raw/log_*`; reproduced standalone), a synthetic target of mean **7.0** with
essentially no signal, MAE by alpha:

| alpha | 1e-2 | 1 | 1e2 | 1e4 | 1e6 | 1e8 |
|---|---|---|---|---|---|---|
| **pc6 `ridge_fit`** | 0.0001 | 0.0140 | 1.1455 | 6.6312 | 6.9962 | **7.0001** |
| **intercept unpenalised** | 0.0001 | 0.0001 | 0.0014 | 0.0073 | 0.0077 | **0.0077** |

⇒ **Two consequences, both material.** (a) **The alpha sweep is silently truncated** — the banked
`alpha_inner_mae` curves turn *upward* past 100 for exactly this reason, so the chosen alpha is not
the shrinkage optimum. (b) ⛔ **The readout cannot express the null hypothesis it is scored against.**
A no-signal arm is driven toward predicting zero and scored the target's own magnitude, so
`K1 fails` conflates *"no signal"* with *"the instrument cannot fall back to a constant"*.

### 7.2 The repair, run on the arm, the null AND both controls

`--fit-mode centred` centres `y` and leaves the intercept unpenalised; the alpha grid is extended to
1e7. Same caches, same split, same estimator, same seeds.

| target | v6 K1 (pc6) | **v6 K1 (repaired)** | **NULL K1 (repaired)** | ORACLE K1 (repaired) | v6 r (repaired) |
|---|---|---|---|---|---|
| `ego_v0` | +1.429 FAIL | **+0.427 ns** | +0.194 ns | **−0.825 PASS** | **+0.322** |
| `ego_accel` | −0.005 ns | +0.016 FAIL | +0.018 FAIL | +0.018 FAIL | +0.127 |
| `ego_yawrate` | +0.005 FAIL | +0.000 FAIL | +0.000 FAIL | +0.000 ns | +0.030 |
| `ego_curv` | +0.001 FAIL | +0.000 FAIL | +0.000 FAIL | +0.000 FAIL | +0.002 |
| `n_agents_grid` | +0.583 ns | +0.576 ns | −0.228 ns | **−6.916 PASS** | +0.141 |
| `n_agents_all` | +1.806 ns | **−5.003 PASS** | ⚠️ **−1.884 PASS** | **−12.720 PASS** | **+0.390** |
| `lead_present` | +0.239 FAIL | +0.129 FAIL | +0.115 FAIL | +0.073 FAIL | +0.109 |
| `nearest_any` | +1.054 FAIL | **+0.071 ns** | +0.533 FAIL | **−2.723 PASS** | **+0.310** |
| `lead_gap` | +1.580 FAIL | +0.736 FAIL | +0.043 ns | **−4.553 PASS** | **+0.073** |
| `lead_closing` | +0.029 FAIL | +0.093 FAIL | +0.093 FAIL | +0.064 ns | −0.036 |
| `lead_inv_ttc` | +0.001 ns | +0.004 FAIL | +0.004 FAIL | +0.002 ns | −0.029 |

⇒ **Four readings, and the fourth is the verdict.**

1. ✅ **The repair does what it should:** it converts structural losses into ties. `ego_v0`
   +1.429 FAIL → **+0.427 not separated**; `nearest_any` +1.054 FAIL → +0.071 ns. The oracle gains
   an extra K1 PASS (`ego_v0` −0.825), which is the correct direction for a fix.
2. ✅ **It raises the arm's correlations** where there was signal: `ego_v0` +0.247 → **+0.322**,
   `nearest_any` +0.219 → **+0.310**, `n_agents_all` +0.275 → **+0.390**.
3. ⛔ **It is NOT a free ride, and the null caught the one place it would have fooled us.** On
   `n_agents_all` the repaired v6 arm posts a **K1 −5.003 PASS** — and **the RANDOM-LATENT NULL also
   passes, −1.884.** That rung's C-CONST is the train **median** of a heavy right-tailed count, so
   shrinking toward the **mean** beats it with no information at all. **The pass is a
   baseline-choice artefact, not skill.** Reported because the null is what makes it visible.
4. ⛔ **THE VERDICT: under the repaired readout there is no rung where the v6 latent earns a K1 PASS
   that the null does not also earn.** And `lead_gap` gets **worse** under the repair
   (r +0.159 → **+0.073**, and the repaired null at +0.043 ns is *better* than the arm at
   +0.736 FAIL). **The headline survives the repair; the precedent's lead-gap fact does not.**

---

## 8. WHERE IN THE STACK IS IT LOST? — a partial answer, and its limits

The chain is `pixels → encoder tokens [640, 768] → readout → z_op [2048] → cells [16, 128]`.

⭐ **`cells` is NOT a lossy view.** `V6Stack.cells` (`stack/tanitad/models/v6.py:3729`) is
`z_op.reshape(*z_op.shape[:-1], n_cells, d_readout)` — **a pure reshape**. Flattening the 16×128
cells reproduces `z_op` exactly, so **§2's ladder probes the full operative state**, and the only
learned, potentially-lossy module between the encoder's tokens and it is `self.readout`.

`cache_tok11250` banks the encoder's patch tokens on a **subset** of the grid. Same ridge, same
split, same estimator; features are the tokens **mean-pooled over the 16×40 grid → 768**, with a
**matched-random null** built from the same features' per-dimension mean/sd:

| `ego_v0`, MATCHED windows n 1507 / 70 | MAE (m/s) | K1 | **r** | `r_wep` |
|---|---|---|---|---|
| encoder **TOKENS**, mean-pooled | 4.142 | +0.047 ns | **+0.285** | +0.118 |
| operative **CELLS**, same windows | 5.703 | +1.608 FAIL | **+0.241** | +0.135 |
| TOKENS **matched-random NULL** | 4.816 | +0.721 FAIL | **−0.053** | −0.005 |

⇒ **The encoder's pooled tokens carry ego speed at r +0.285 against their own null's −0.053 — real,
and only marginally more than the operative state's +0.241 on the same windows.** ⚠️ **This is a
WEAK localisation and must not be over-read in either direction.** Mean-pooling is **lossy**, so
`+0.285` is a **LOWER bound** on what the tokens hold: this run **cannot** establish that ego motion
is absent from the encoder, only that **pooling the encoder's output does not recover much more than
the operative state already has.** ⇒ **It does not indict `readout`, and it does not clear it.**
The decisive version — an unpooled or spatially-structured token probe — is §10.

---

## 9. THE FOUR METRIC FAMILIES

Per the binding rule, every family is addressed, with the reason and the `n` where it does not apply.
⛔ **ADE is not reported and is not applicable:** this is a frozen-latent state readout, not a
trajectory eval.

| family | what this run reports | verdict |
|---|---|---|
| **LONGITUDINAL** | target-speed: `ego_v0` (r **+0.247**, r² 0.061) and `ego_accel` (+0.187, 0.035). Distance-keeping: `lead_gap` (+0.159, 0.025 — **and +0.052 once `v0` is partialled out**), `lead_closing` (**−0.006, AT THE NULL**), inverse-TTC (**−0.008, AT THE NULL**). | ⛔ The family's *own control variable* is at 6 % of variance and its **time-gap / TTC half is at zero.** |
| **LATERAL** | heading-rate: `ego_yawrate` (r **+0.060**, r² 0.0036). Curvature: `ego_curv` (**−0.009**, 0.0001), n 2221 (windows below the 2.0 m/s floor are dropped and counted). Cross-track is **not applicable** — there is no predicted trajectory in a frozen-latent readout. | ⛔ **Nothing. Both rotational rungs are indistinguishable from the null** — and both lack a positive control (§2.4), so this is the weakest-evidenced row. |
| **TACTICAL** | ⚠️ **NOT MEASURED, n = 0.** This instrument has no manoeuvre decode and no selected-vs-executed comparison; the cache banks no manoeuvre label. `lead_present` (AUC **0.586** vs C-V0's 0.673) is a *precursor*, not a manoeuvre-decision metric. | ⚠️ Absent by instrument scope, not by choice. A work item, not an excuse — §10. |
| **STRATEGIC** | ⚠️ **NOT MEASURED, n = 0.** No route or goal label exists in this 130-clip lead-enriched pool, and per `CLAUDE.md` PhysicalAI-AV carries no map, lane graph or route signal at all. | ⚠️ Not computable on this corpus with this cache. |

---

## 10. ⭐ WHAT I WOULD RUN NEXT, in cost order (all CPU or ≤1 GPU-hour, no trunk compute)

1. ⭐ **An EGO-ORACLE for YAW-RATE and CURVATURE** — one line of `ll2_ego_oracle.py` (swap the
   projected scalar). §2.4 is explicit that the two ROTATIONAL rungs currently have **no positive
   control**, which makes them the weakest rows in the ladder despite being the LATERAL family's
   whole basis. **Until this runs, "the latent has no yaw content" is an unverified negative.**
2. **Re-run the ladder at 30 000** when Thor finishes. Every v6 number here is a 41.7 % early read.
   The chain is one command per arm and costs no GPU.
3. **An unpooled token probe** (§8) — spatially-structured or patch-subset features rather than a
   mean-pool, so the encoder-vs-`readout` localisation becomes decisive rather than suggestive.
4. **Report the `C-V0` column on every future latent readout in this programme.** §3 shows a single
   privileged scalar beating a 2 048-dimension latent on the metric the programme had been treating
   as evidence about agents. **Any latent-readout result without it is not interpretable.**
5. **A non-linear readout (small MLP) with the same caches, splits, controls and positive control.**
   Everything here is a statement about **linear** decodability only; the same ladder under an MLP
   is the cheapest test of whether the information is present but entangled.

---

## 11. THOR — untouched and healthy

⛔ **No compute of any kind was run on Thor**; no checkpoint was pulled and no snapshot was made.
Everything here ran on the dev box's CPU from banked local caches. **MEASURED at the end of this
run** (opaque-marker probe, per the polling-monitor trap):

| | |
|---|---|
| trainer PID | **25477 — ALIVE**, uptime **1-22:43:29** |
| run | `v6F-SW-30k`, `train_v6_staged.py --stage S-W --steps 30000` |
| last `metrics.json` record | **step 12 500**, `loss` 1.43387 |

⇒ **Advancing and healthy** (step 12 500 against ~12 500 expected at handover).

## 11a. SUITES — ⚠️ ONE FAILURE, AND IT IS A CONCURRENT-EDIT RACE, NOT THIS RUN

| suite | result | baseline | verdict |
|---|---|---|---|
| `taniteval` | **1092 passed**, 0 failed, 281 s | 1092 / 0 | ✅ **matches** |
| `stack` run 1 | ⚠️ **3802 passed, 1 FAILED**, 7 skipped, 2 xfailed, 692 s | 3803 / 0 / 7 / 2 | ⚠️ **one short** |
| `stack` run 2 | `raw/suite_stack.txt` | 3803 / 0 / 7 / 2 | re-run after the edit settled |

⛔ **THE FAILURE IS NOT MINE, AND I AM NOT ASSERTING THE SUITE IS GREEN ON MY OWN SAY-SO.** The
failing test is
`tests/test_v6_anchor_loss.py::test_the_objective_reads_no_situation_classifier_no_ego_no_v0`.
**Four facts, all MEASURED:**

1. ⭐ **I modified nothing under `stack/` or `taniteval/`.** `git diff --name-only HEAD` shows
   `stack/scripts/train_v6_staged.py` and `stack/scripts/v6_chain.py` modified — **by another agent,
   concurrently**; the repo was clean at my session start. Every artifact of mine is a NEW file
   under this incoming directory.
2. **The test reads that exact file.** It imports `anchor_goal_loss` **from `train_v6_staged`**
   (`tests/test_v6_anchor_loss.py:64`) and asserts on `inspect.getsource(anchor_goal_loss)` — so its
   result is a function of that file's *live* contents.
3. ⭐ **The file was written DURING my suite run.** `train_v6_staged.py` mtime
   **2026-08-17 23:47:10**; the run spanned roughly **23:40 → 23:52** (692 s). `v6_chain.py` was
   written at **23:51:39**, i.e. the tree was still moving as the suite finished.
4. ✅ **The test PASSES in isolation** once the edit settled: `1 passed in 1.96s`.

⇒ **Root-cause class: a source-inspecting test run against a working tree another agent is editing.**
Same family as the standing warning that *"with several agents live, the index moves under you"* —
here it is the **worktree**, not the index, and the victim is a test rather than a `git add`.
⚠️ **`stack` run 2's number in `raw/suite_stack.txt` is the one to read**, and it carries the same
caveat: it can only certify the tree as it stood at that moment.

---

## 12. DELIVERABLE MANIFEST

**All paths relative to the repo root** `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\`, under
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-17-latent-linear-ladder/`.
⛔ **Staged into the working tree, NOT committed and NOT pushed.**

| artifact | path | what it is |
|---|---|---|
| this report | `LATENT_LINEAR_LADDER.md` | the findings |
| rendered tables | `raw/RENDER_TABLES.md` | every table above, generated from the banked JSONs |
| assembled summary | `raw/SUMMARY.json` | all 19 arm blobs in one object, no recomputation |
| **the ladder** | `code/ll1_ladder.py` | pc6's ridge (imported `ridge_fit`) over a graded target ladder; carries the pose-grid binding proof, the pc6 equivalence gate, `--fit-mode centred` repair, `--features tokens_mean`, `--randomise-features` null |
| **anchor positive control** | `code/ll2_ego_oracle.py` | builds the EGO-ORACLE caches (distributed random projection of `v0`, 4 noise levels) |
| renderer | `code/ll3_summarise.py` | reads `raw/ll_*.json` → tables; computes nothing |
| chain | `code/chain_ladder.sh` | one invocation per arm: `ctrl / main / ckpt / egoorc / tokens / repair` |
| per-arm results (19) | `raw/ll_*.json` | one JSON per arm, all 11 rungs × 3 seeds, with CIs, alphas, ns |
| per-arm logs (19) | `raw/log_*.txt` | stdout of every run |
| ego-oracle meta | `raw/ll2_meta_egoorc_n0.1.json` | the synthetic control's construction record |
| suites | `raw/suite_taniteval.txt`, `raw/suite_stack_run1.txt`, `raw/suite_stack.txt` | pytest output; run 1 carries the concurrent-edit failure documented in §11a |

**Caches read (NOT copied — they are large and already banked by the precedent runs):**
`…/scratchpad/sp2/cache_{s02000,s09000,s09250,s10000,s11250,tok11250,nullmatched}/latents.pt`,
`…/scratchpad/pc/cache_orcdir/latents.pt`, `…/scratchpad/sp2/p3_selection.json`,
`…/scratchpad/sp2/lead130_agents.jsonl`, and the 130-clip episode cache
`…/scratchpad/sp2/cache/slotprobe-lead130-w120-256x640cyl/`.
⚠️ **CREATED IN SCRATCH, NOT BANKED (they are regenerable in ~30 s by `ll2_ego_oracle.py`):**
`…/scratchpad/ll/cache_egoorc_n{0.1,1,3,10}/latents.pt` (~33 MB each).

---

## 13. ⛔ ESCALATIONS — these need a decision, and they are not filed in a README

1. ⛔⛔ **`pc6_linear_readout.ridge_fit` PENALISES ITS OWN INTERCEPT (§7).** Every `K1 fails` this
   readout has ever emitted — including the precedent's §2.3 table, which is currently the
   programme's only positive statement about the v6 latent — conflates *"no signal"* with *"the
   instrument cannot express a constant"*. **The fix is two lines** and is implemented and validated
   in `ll1_ladder.py --fit-mode centred`. **Decision needed: adopt it in `pc6_linear_readout.py` and
   re-render the precedent's §2.3, or record why not.**
2. ⛔ **THE PRECEDENT'S §2.3 SENTENCE NEEDS AMENDING.** *"The latent carries some linearly-decodable
   lead information"* is **not supportable**: the same signal is recovered better by the ego-speed
   scalar alone (K1 **PASS** at r +0.683 vs the latent's **FAIL** at +0.159), and partialling `v0`
   out leaves r **+0.052**. ⚠️ **I did not edit that document** — it is another agent's incoming
   directory. **This escalation is the request.**
3. ⚠️ **THE POSE-GRID OFF-BY-TWO IS A LATENT TRAP FOR ANY FUTURE EGO PROBE (§0/§8).** Reading the
   source gives the WRONG index: `window_frame` returns `t + window - 1` and `_contract.py` sets
   `pose_last = ep.poses[t + window - 1]`, but the episode `poses` array is on the **raw** frame grid
   while the dataset's frames are **3-stacks** (`n_stack = 3`), so the correct index is
   **`frame_idx + 2`**. My first version assumed 0 and was caught **only** because `v0` is banked
   per row (max mismatch 0.667 m/s on 4 869 of 5 617 rows). **`ll1_ladder.bind_pose_grid` now
   DISCOVERS the offset and accepts it only on EXACT equality — that helper should be reused, not
   re-derived.**
4. ⚠️ **THE ROTATIONAL RUNGS HAVE NO POSITIVE CONTROL (§2.4, §10.1).** The LATERAL family's entire
   basis in this ladder rests on two nulls whose apparatus has never been shown able to measure
   them. **One line of `ll2_ego_oracle.py` closes it; until then those two rows are the weakest in
   the report and are marked as such.**
5. ⚠️ **`C-V0` SHOULD BECOME A STANDING COLUMN** on every latent-readout result in this programme
   (§10.4). A privileged scalar out-performing a 2 048-dimension latent is exactly the confound
   class that produced the nav-echo and C6 findings, and it costs one extra fit.
