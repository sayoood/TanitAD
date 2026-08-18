# ⛔ E-R1-0 — THE POOL IS **NOT** THE BINDING CONSTRAINT, AND THE CORPUS IS NOT EITHER. **A frozen DINOv2 reads relative motion out of OUR OWN FRAMES, THROUGH THE DEPLOYED 40:1 POOL, where our trained encoder reads zero with no pool at all**

**Date:** 2026-08-18 · **Branch:** `agent/arch-inf-20260803` · **Agent:** pooling-ladder-ER10
**Executes:** `…/Research/2026-08-18-pooling-bottleneck-R1R2/POOLING_BOTTLENECK_R1R2.md` §7.1 (E-R1-0)
and §8.1 (the corpus-narrowness discriminator).
**Eval tier:** ⛔ **T0-DIAGNOSTIC.** A frozen-latent linear readout is a **world-model diagnostic**
and is **never** driving performance. No number here is an ADE, a closed-loop result, or a claim
about how the car drives.
**Compute:** dev-box RTX 4060 only. ⛔ **ZERO training. Thor was never touched** — no checkpoint
pulled, nothing run there. ⛔ **Zero pod.**

> **Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` · `INHERITED` (another
> agent/doc, NOT re-verified by me) · `ESTIMATED` · `HYPOTHESIS`.

---

## ⭐ THE ANSWER, IN ONE PARAGRAPH

**`MEASURED`** (`raw/er10_main.json`). On the frozen `v6F-SW-30k@11250` trunk, over the banked
130-clip probe windows (**1 302 probe-train / 1 507 eval windows in 70 episode clusters**), the
identical linear readout was run at **four spatial-averaging ratios that differ in nothing but the
pooling kernel** over the same 16 × 40 token grid — **40:1 (the deployed `AvgPool2d((4,10))`), 10:1,
4:1 and 1:1 (no pooling at all)** — with every arm forced to **exactly 2 048 features by a fixed
Gaussian random projection**, over **5 projection seeds**. ⛔ **On `lead_closing`, `lead_inv_ttc`,
`ego_curv` and `ego_yawrate` — the four rungs that read exactly 0.0000 at the deployed ratio, and on
which the pooling hypothesis makes its sharpest prediction — REMOVING THE POOL ENTIRELY MOVES r² BY
|Δ| ≤ 0.0002, with the paired episode-cluster-bootstrap CI containing zero on ALL FIVE SEEDS**
(`lead_closing` Δr²(1:1 − 40:1) = **+0.00001 [−0.00597, +0.00504]**).

⭐⭐ **AND THE INSTRUMENT IS PROVED ABLE TO SEE EXACTLY THAT EFFECT, ON EXACTLY THOSE RUNGS.** The
`PC-2OBJ` positive control plants the answer in **two tokens inside ONE deployed 4 × 10 cell with
opposite sign**, so the cell mean annihilates it *exactly* and no finer arm's cell contains both.
**MEASURED, it is a clean step:** `lead_closing` r² **0.00000 at 40:1 → 0.99980 / 0.99980 / 0.99970
at 10:1 / 4:1 / 1:1**; `ego_yawrate` **0.00027 → 0.99977 / 0.99973 / 0.99963**. ⇒ **when a signal IS
destroyed by the deployed pool, this ladder reports a step from 0 to 1. The real latent produces no
step at all.** By §7.1's own pre-registration that is the **`⛔ R1 IS DROPPED`** branch, verbatim:
*"the information is not in the tokens either. The pool is NOT the binding constraint, and no loss
that merely reads the tokens can add what the encoder never encoded."*

⚠️ **What DID move, and why it is not a rescue.** Two rungs rise materially as the pool is removed —
`lead_gap` r² **0.0050 → 0.0750 (15×)** and `ego_v0` **0.0524 → 0.1240 (2.4×)** — but **neither meets
the pre-registered criterion** (0 / 5 and 2 / 5 seeds with a CI excluding zero), and `lead_gap` is
**C92's ego-speed proxy**: with `v0` partialled out it collapses to **0.0060 → 0.0087**, i.e. the 15×
becomes **1.4×**. ⇒ **the largest ladder effect in the experiment is mostly a restatement of ego
speed**, which is the third time that control has changed a headline in this programme.

⭐⭐⭐ **AND JOB 2 — THE DISCRIMINATOR — RETURNS THE CELL THAT KILLS THE REMAINING EXCUSE (§8).**
A **frozen `facebook/dinov2-base`**, which has never seen a driving scene, run over the **SAME 1 507
windows** and pushed through the **SAME deployed `AvgPool2d((4,10))`**, recovers `lead_closing` at
r² **0.01713** (partial-`v0` r **+0.129**), `lead_gap` at **0.44997** (partial r² **0.120**) and the
ego's own speed at **0.71733** — against our encoder's **0.00000 / 0.00496 / 0.05240** at the same
ratio, and against **0.00002 for `lead_closing` with the pool removed and all 491 520 features fitted
exactly** (§7). ⭐ **DINOv2-B/14 is 86 M parameters against our encoder's 87.3 M, so this is not a
capacity gap.** ⇒ ⛔ **the information IS linearly present in our images, it SURVIVES the deployed
mean, and neither the pool nor the corpus is the binding constraint. What is left is the ENCODER and
the objective that trained it.**

⛔ **AND THE ARGUMENT THE SPEC REASONS FROM IS QUANTITATIVELY TOO STRONG.** §1/§7 reason that *"a
40:1 unweighted mean must be destroying individuation"*. **MEASURED: a LOCALISED plant — the answer
written into 4 of the 640 tokens, all inside ONE deployed cell — is recovered at r² = 1.0000 at
40:1.** The mean attenuates a localised signal by `k/K` and its noise by `1/√K`, so SNR falls only as
**`√(k/K)` = √(4/40) ≈ 0.32**, i.e. **~3×, not ~40×**. A mean is a far weaker destroyer of a *linear*
local code than the ratio suggests; what it genuinely cannot do is **separate two objects inside one
cell** — which is what PC-2OBJ isolates, and which the real latent gives no evidence of needing.

---

## 0. ⛔ THE STAMPS THAT BIND EVERY NUMBER BELOW

1. **Checkpoint:** `v6F-SW-30k@11250` — the checkpoint IS part of the arm. ⚠️ **An EARLY-READ at
   11 250 / 30 000 = 37.5 %** of the live S-W run. 30 000 remains the primary read.
2. ⚠️ **LEAD-ENRICHED, NOT PARITY** — `parity: False` in the cache meta, inherited unchanged from
   the precedent (`INHERITED`; `…/2026-08-17-latent-linear-ladder/`). 130 clips, 60 probe-train /
   **70 eval**, clip-disjoint. ⛔ **NO EPISODE IS SELECTED, ADDED, REMOVED, REORDERED OR RE-HASHED
   by this work** — the banked window set is read verbatim.
3. **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, `n_boot` 2 000, clustered on the
   **70 eval episodes**. ⛔ `overlapping_holdout_se` is never imported.
4. ⛔ **C92 — every fit passes `intercept_col=-1`.** The argument is written into every emitted
   `config`/JSON and is **asserted at startup from source**, not remembered: `MEASURED`, at
   `alpha = 1e9` on a synthetic no-signal target of mean 4.834342, the penalised solve predicts
   **0.000000** and the repaired solve predicts **4.834342** (`raw/er10_main.json`
   → `ridge_intercept_gate`). A run that could not demonstrate that refuses to start.
5. ⛔ **C97 — `pred_sd / gt_sd` is emitted on EVERY row** and any K1 PASS below `0.10` is stamped
   **`K1_DEGENERATE`** in the artifact and in the tables below. It fires: `n_agents_grid` PASSes at
   10:1 / 4:1 / 1:1 / cells with ratios **0.055 / 0.051 / 0.048 / 0.054** and is therefore **not
   quotable as a pass**.
6. ⭐ **The primary quantity is `r²_ceiling = corr(pred, truth)²`, not MAE-skill.** Both C92 and C97
   act on the *dispersion* of the fit (a shrunk fit predicts the mean; an unshrunk one over-disperses),
   and both therefore move K1 without moving the correlation. `r²_ceiling` is the quantity those two
   defects cannot bias, so it is what the pre-registered Δ is computed on. **K1 and `pred_sd/gt_sd`
   are reported alongside it on every row, never in place of it.**
7. **Solve:** `pc6_linear_readout.ridge_fit` is **IMPORTED**, not re-implemented. The targets, pose
   binding and rung definitions are **IMPORTED** from `ll1_ladder`. There is no second copy.

---

## 1. ⛔ THE THREE GATES, RUN BEFORE ANY NUMBER WAS READ

C94's root-cause class is *a fixture that models the CONSUMER'S EXPECTATION instead of the
PRODUCER'S OUTPUT*. All three gates below assert against a **producer's committed output**, never
against a literal written by this file's author.

| gate | what it asserts | result |
|---|---|---|
| **J5 — reproduction** (`raw/er10_gate.json`) | this harness, run on the **incumbent biased solve**, reproduces the banked `ll_cells_tokwin.json` on **all 11 rungs** — `err`, `c_const_err`, `K1_delta`, `corr`, `alpha_chosen`, `n_eval` | ✅ **PASS**, 66 checks |
| **J2 — the token grid** (`raw/er10_j2_gridcheck.json`) | (a) a unit impulse at token `(r,c)` lands in pooled unit `(r//kh, c//kw)` for **all four kernels**; (b) `proj(AvgPool2d((4,10))(banked tokens))` reproduces the **banked `cells`** using the **checkpoint's own `readout.proj` weights** | ✅ **PASS** — (a) exact on all 4 kernels; (b) corr **0.99932**, worst relative mean error **0.0358** (the producer ran under bf16 autocast and stored fp16, so the residual is rounding; it is **reported, not asserted to zero**) |
| **intercept** (in every run's JSON) | `ridge_fit` exposes `intercept_col` **and** the repaired solve falls back to the train **mean** at large α | ✅ **PASS** |

⭐ **J2(b) is the one that matters most, and it upgrades an INHERITED claim to MEASURED.** The spec
derives *"`AvgPool2d((4,10))`, 40 tokens per cell, 64 × 160 px per cell"* from source. This measures
the same fact from the **banked artifact**: the impulse pin gives **40 tokens and 64 × 160 px per
deployed unit**, and the operator check shows that pooling the banked tokens with that kernel and
applying the checkpoint's own projection **reproduces the banked latent**. ⇒ **the ladder's arms
really are the deployed operator with only the kernel varied.**

⚠️ **One correction to the spec's §1.1/§1.7 table, MEASURED.** The live run's encoder is
**`d_model = 768`, not the 384** in `V6Config()`'s defaults (`cache_tok11250/sp1_meta.json`:
`d_model_tokens: 768`). **The pooling geometry is unaffected** (16 × 40 grid, `AvgPool2d((4,10))`,
40 tokens/cell — all re-measured above), but the spec's parameter table (encoder 15 327 360 = 17.4 %,
`predictor_op` 68.5 %) is computed from **`V6Config()` defaults and therefore does not describe the
live checkpoint.** The raw feature counts in §7.1 double accordingly: **12 288 / 49 152 / 122 880 /
491 520**, not 6 144 / 24 576 / 61 440 / 245 760. This does not change any conclusion — every arm is
projected to 2 048 — but the numbers in the spec should be re-derived from the live config before
they are quoted again.

---

## 2. ⭐ THE LADDER — `r²_ceiling`, mean ± sd over 5 projection seeds

**`MEASURED`** · `raw/er10_main.json` · T0-DIAGNOSTIC · `intercept_col=-1` · 2 000 boot · 70 clusters.

| rung | family | **40:1** *(deployed kernel)* | **10:1** | **4:1** | **1:1** *(no pooling)* | **DEPLOYED CELLS** *(40:1 + the learned `Linear(768→128)`)* |
|---|---|---|---|---|---|---|
| `ego_v0` (n=1507) | LONGITUDINAL | 0.0524 ±0.0008 | 0.0711 ±0.0214 | 0.0680 ±0.0067 | **0.1240 ±0.0078** | 0.1026 |
| `ego_accel` (n=1507) | LONGITUDINAL | 0.0122 ±0.0002 | 0.0075 | 0.0090 | 0.0037 | 0.0309 |
| ⭐ `ego_yawrate` (n=1507) | LATERAL | **0.0003** | 0.0002 | 0.0002 | **0.0001** | 0.0009 |
| ⭐ `ego_curv` (n=1103) | LATERAL | **0.0000** | 0.0000 | 0.0000 | **0.0000** | 0.0000 |
| `n_agents_grid` (n=1507) | SCENE | 0.0286 ±0.0205 | 0.0567 | 0.0580 | 0.0570 | 0.0876 |
| `n_agents_all` (n=1507) | SCENE | 0.1017 ±0.0030 | 0.1075 | 0.1099 | 0.1161 ±0.0040 | **0.1671** |
| `lead_present` (n=1507) | OBJECT | 0.0051 | 0.0052 | 0.0094 | 0.0109 | 0.0165 |
| `nearest_any` (n=1506) | OBJECT | 0.0763 | 0.0755 | 0.0764 | 0.0712 | 0.1083 |
| `lead_gap` (n=1362) | LONGITUDINAL | 0.0050 | 0.0548 | 0.0738 | **0.0750 ±0.0040** | 0.0038 |
| ⭐ `lead_closing` (n=1362) | LONGITUDINAL | **0.0000** | 0.0001 | 0.0000 | **0.0000** | 0.0005 |
| ⭐ `lead_inv_ttc` (n=1362) | LONGITUDINAL | **0.0012** | 0.0015 | 0.0013 | **0.0013** | 0.0007 |

⭐ **THE SHAPE, stated plainly.** The four ⭐ rungs — the relative-motion and rotational quantities —
are **flat at the null across a 40× change in spatial averaging**. `lead_gap` and `ego_v0` rise;
`ego_accel` and `nearest_any` *fall* slightly. **Nothing in the profile has the form the pooling
hypothesis predicts**, which is a monotone rise concentrated on individuation and relative motion.

### 2.1 The same table with `v0` PARTIALLED OUT — C92's trivial-proxy control (seed 0)

| rung | 40:1 | 10:1 | 4:1 | 1:1 | cells |
|---|---|---|---|---|---|
| `ego_accel` | 0.0075 | 0.0033 | 0.0043 | 0.0010 | 0.0256 |
| `ego_yawrate` | 0.0005 | 0.0003 | 0.0003 | 0.0003 | 0.0013 |
| `ego_curv` | 0.0006 | 0.0005 | 0.0005 | 0.0005 | 0.0009 |
| `n_agents_all` | 0.0572 | 0.0687 | 0.0658 | 0.0659 | 0.1202 |
| `lead_present` | 0.0037 | 0.0037 | 0.0075 | 0.0115 | 0.0108 |
| `nearest_any` | 0.0424 | 0.0446 | 0.0437 | 0.0393 | 0.0646 |
| ⭐ `lead_gap` | 0.0060 | 0.0016 | 0.0021 | **0.0087** | 0.0092 |
| `lead_closing` | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 0.0005 |
| `lead_inv_ttc` | 0.0006 | 0.0008 | 0.0007 | 0.0007 | 0.0003 |

⛔ **`lead_gap`'s 15× ladder effect is 91 % ego speed.** 0.0050 → 0.0750 raw becomes
**0.0060 → 0.0087** once `v0` is removed, and the 40:1 arm's raw correlation is **+0.070 while its
partial correlation is −0.077** — a sign flip. **This is C92 recurring for the third time on this
target and it must travel with any future quotation of a `lead_gap` result.**

### 2.2 ⭐ A finding that is NOT about pooling, and is worth more than most of this table

The **DEPLOYED CELLS** column is the same 40:1 pooling followed by the checkpoint's **learned**
`Linear(768 → 128)`; the **40:1** column is the same pooling followed by a **random** projection to
the same 2 048 features. **`MEASURED`: the learned readout beats the random one on 8 of 11 rungs**
(1 tie — `ego_curv`, both 0.0000 — and 2 losses), by up to **3.4×** (`n_agents_grid` 0.0876 vs
0.0286, `n_agents_all` 0.1671 vs 0.1017, `ego_v0` 0.1026 vs 0.0524) — and the paired Δ on
`n_agents_all` is the **only contrast in the entire experiment that is separated on all five seeds
and survives partialling `v0`** (Δr² **+0.0693**, Δ MAE **−1.349 [−2.619, −0.181]** separated).
⇒ **the 49 280-parameter "geometry firewall" is doing real work, and it is not a bottleneck being
squeezed — it is the best-performing 2 048-dim view of the pooled tokens in the table.** It loses on
exactly two rungs: `lead_gap` (0.0038 vs 0.0050 — the ego-speed proxy) and `lead_inv_ttc` (0.0007 vs
0.0012 — both at the random null).

---

## 3. ⛔ THE PRE-REGISTERED CONTRAST: Δ r²_ceiling (arm − 40:1), paired episode-cluster bootstrap

§7.1 commits to: *R1 PROCEEDS iff r² rises monotonically as the pooling ratio falls, **AND** the
paired CI on Δr²(1:1 − 40:1) excludes 0, **AND** that Δ survives partialling out `v0`, **AND** the
oracle positive control passes.* Every clause is evaluated below, on **all five seeds** (a single
lucky seed may not carry a verdict).

| rung | Δr²(1:1 − 40:1), mean of 5 seeds | seeds with CI excluding 0 | after partialling `v0` | verdict |
|---|---|---|---|---|
| ⭐ `lead_closing` | **+0.00000** — per-seed +0.00001 / +0.00001 / −0.00001 / +0.00002 / −0.00002, CIs ≈ [−0.006, +0.005] | **0 / 5** | 0 / 5 | ⛔ **DROP** |
| ⭐ `lead_inv_ttc` | **+0.00010** | **0 / 5** | 0 / 5 | ⛔ **DROP** |
| ⭐ `ego_curv` | **+0.00001** | **0 / 5** | 0 / 5 | ⛔ **DROP** |
| ⭐ `ego_yawrate` | **−0.00014** *(the wrong sign)* | **0 / 5** | 0 / 5 | ⛔ **DROP** |
| `lead_gap` | +0.07008 | **0 / 5** | 0 / 5 | ⛔ fails the criterion; and §2.1 shows it is `v0` |
| `ego_v0` | +0.07157 | **2 / 5** | 0 / 5 | ⛔ fails the criterion (not all seeds) |
| `n_agents_all` | +0.01437 | **0 / 5** | 0 / 5 | ⛔ fails the criterion |
| `lead_present` | +0.00582 | 0 / 5 | 0 / 5 | ⛔ |
| `n_agents_grid` | +0.02844 | 0 / 5 | 0 / 5 | ⛔ |
| `nearest_any` | −0.00507 | 0 / 5 | 0 / 5 | ⛔ |
| `ego_accel` | −0.00856 | 0 / 5 | 0 / 5 | ⛔ |

⛔ **NOT ONE RUNG SATISFIES THE PRE-REGISTERED `R1 PROCEEDS` CONDITION.** The four rungs the
experiment was designed around fail it by three orders of magnitude.

---

## 4. ⭐ THE POSITIVE CONTROLS — three plants, and the one that settles it

C79 is the standing lesson: **D1 was withdrawn because a probe failed its positive control.** All
three plants are added to the **REAL tokens**, so the background covariance is the real one; the
planted vector carries `ego_v0/10`, `lead_gap/15`, `lead_closing/2`, `ego_yawrate/0.1` through a
fixed random `[768, 4]` map, at **1.0 × the measured token sd (0.929428)**.

| control | where the answer is written | 40:1 | 10:1 | 4:1 | 1:1 | reads |
|---|---|---|---|---|---|---|
| **PC-DIST** | ALL 640 tokens (a global code) | **1.0000** | 1.0000 | 1.0000 | 1.0000 | ✅ the harness can read every rung at every arm's n/p |
| **PC-LOCAL** | 4 tokens, one 2 × 2 block inside ONE deployed cell | **1.0000** | 1.0000 | 1.0000 | 1.0000 | ⚠️ a *localised* linear code SURVIVES the deployed 40:1 mean intact at this SNR |
| ⭐⭐ **PC-2OBJ** | 2 tokens inside the SAME deployed cell, **opposite sign** | **0.0000** | **0.9998** | **0.9998** | **0.9997** | ⭐ a **step exactly at the deployed ratio** — the ladder has FULL POWER |

*(`lead_closing` row shown; `ego_yawrate` 0.00027 → 0.99977 / 0.99973 / 0.99963, `ego_v0` 0.05207 →
0.99957 / 0.99953 / 0.99943, `lead_gap` 0.00490 → 0.99920 / 0.99900 / 0.99877 — `raw/er10_pc_*.json`.)*

⭐ **PC-2OBJ carries a free internal consistency check that I did not design and should be recorded.**
At 40:1 the two plants cancel *exactly*, so that arm sees only the real tokens — and it reproduces the
real ladder's 40:1 values to 4 decimal places (`ego_v0` 0.05207 vs **0.0524**, `lead_gap` 0.00490 vs
**0.0050**, `lead_closing` 0.00000 vs **0.0000**, `ego_yawrate` 0.00027 vs **0.0003**). ⇒ **the
injection does exactly what it claims, proved by an arithmetic identity rather than by inspection.**

⛔ **THE CONSEQUENCE FOR THE VERDICT.** PC-2OBJ shows the ladder reporting **0 → 1** when the deployed
mean genuinely destroys a signal. The real latent's four ⭐ rungs report **0 → 0**. **A negative from
an instrument that has just demonstrated a 1.0 dynamic range on the same rungs, the same windows and
the same seeds is a fact about the latent.**

⚠️ **AND PC-LOCAL IS ITSELF A RESULT, not just a control.** The spec's central intuition — 40 tokens
averaged against a 2.4-patch object — predicts heavy loss. **Measured, there is none at this
amplitude.** §5 sweeps the amplitude down to find the band where pooling does cost something.

---

## 5. ⭐⭐ THE PC-LOCAL AMPLITUDE SWEEP — POOLING **HELPS** A LOCALISED SIGNAL, MONOTONICALLY

**`MEASURED`** · `raw/er10_pcloc_a*.json` · target `lead_closing`, 2 projection seeds each, everything
else identical. The plant is 4 of 640 tokens, all inside ONE deployed cell; the amplitude is in units
of the measured token sd (0.929428).

| planted amplitude | **40:1** *(deployed)* | **10:1** | **4:1** | **1:1** *(no pooling)* |
|---|---|---|---|---|
| 1.0 × | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| 0.3 × | **0.99985** | 0.99970 | 0.99975 | 0.99830 |
| 0.1 × | **0.99870** | 0.99775 | 0.99760 | 0.98390 |
| 0.03 × | **0.98430** | 0.97170 | 0.96985 | 0.77915 |
| ⭐ 0.01 × | **0.85945** | 0.74155 | 0.67265 | ⛔ **0.00000** |
| 0.003 × | 0.00000 | 0.00010 | 0.00000 | 0.00000 |

⭐ **THE SWEEP BRACKETS THE INSTRUMENT'S FULL DYNAMIC RANGE, so the negative can be stated as a
BOUND rather than an absence.** The deployed arm's detection threshold for a 4-token localised linear
plant lies **between 0.003 × and 0.01 × the token sd** (0.00000 → 0.85945 across that decade).
⇒ **`lead_closing` reads 0.00000 on the real latent at that same arm, so whatever `lead_closing` code
this encoder writes is either (a) smaller than ~0.3 % of the token magnitude, (b) not spatially
localised, or (c) not linear.** That is a quantitative statement, and none of the three readings is
repaired by moving a loss from post-pool to pre-pool.

⛔⛔ **THE ORDERING IS STRICT AND MONOTONE IN THE OPPOSITE DIRECTION TO THE HYPOTHESIS: the DEPLOYED
40:1 arm is the BEST reader of a localised planted signal at EVERY amplitude, and the UN-POOLED 1:1
arm is the WORST — collapsing to exactly 0.00000 at 0.01 × while the deployed arm still reads
0.859.**

**Why, and it is not mysterious — two effects, both pointing the same way:**
1. **Averaging suppresses the un-planted background faster than it attenuates the plant.** Signal
   scales `k/K`, noise `1/√K`, so SNR ∝ `√(k/K)` — but the *competing* background inside the readout
   also shrinks, and with a fixed feature budget that dominates.
2. ⚠️ **The random projection keeps a smaller FRACTION of a larger source space.** 2 048 of 12 288
   (40:1) is 16.7 %; 2 048 of 491 520 (1:1) is **0.42 %**. **This is the RP handicap the method
   section declared, now quantified: it is real, it is large at low SNR, and it runs AGAINST the fine
   arms.**

⭐ **AND THIS IS EXACTLY WHY §4's PC-2OBJ IS THE CONTROL THAT SETTLES THE VERDICT, NOT THIS ONE.**
Read alone, the sweep would leave the objection open — *"maybe the fine arms were handicapped into a
null"*. **PC-2OBJ closes it:** when the information is genuinely available ONLY to a finer arm (the
two opposing plants cancel exactly in the deployed cell), the fine arms report **0.9998** and the
deployed arm reports **0.0000** — **the handicap does not stop the ladder detecting a
pooling-destroyed signal; it reverses the ordering completely.** ⇒ **the two controls bracket the
instrument in both directions, and the real latent's four ⭐ rungs sit flat in the middle of that
bracket.** §7's no-projection supplement removes effect (2) entirely as a final check.


---

## 6. ⛔ THE NEGATIVE CONTROLS — and the level they put the four ⭐ rungs at

**`MEASURED`** · `raw/er10_null.json` (matched-random features per arm: `N(mu, sd)` with the arm's
own per-feature train mean/sd — the identical construction as `pA_null_matched.py`), 3 seeds.

⭐ **THE NULL'S OWN FLOOR IN `r²_ceiling` IS ≈ 0.0002 – 0.0020 ON EVERY RUNG AND EVERY ARM.** Set
against §2's table that is the sharpest way to state the result:

| rung | real latent, **1:1** (no pooling, 491 520 raw features) | matched-random null, same arm | reads |
|---|---|---|---|
| `lead_closing` | **0.00002** | 0.00067 ±0.00037 | ⛔ **BELOW the null** |
| `ego_curv` | **0.00000** | 0.00080 ±0.00033 | ⛔ **BELOW the null** |
| `ego_yawrate` | **0.00014** | 0.00073 ±0.00061 | ⛔ **BELOW the null** |
| `lead_inv_ttc` | **0.00134** | 0.00020 ±0.00016 | ⚠️ at the null |
| *(for contrast)* `lead_gap` | 0.07504 | 0.00093 ±0.00125 | ✅ far above the null |
| *(for contrast)* `n_agents_all` | 0.11608 | 0.00030 ±0.00016 | ✅ far above the null |

⇒ **Three of the four are not merely "small" — with the pool removed entirely they are BELOW a
matched-random feature set drawn to the same per-feature moments.** The null is not a floor these
rungs are approaching; it is a level they are already under.

⭐ **AND THE C97 GUARD REPRODUCED ITS OWN DOCUMENTED FAILURE, which is the only way to know it
works.** C97 records a pure-noise arm "PASSing" `n_agents_all` at K1 **−1.884** with `pred_sd`
0.715 against `gt_sd` 46.459. **MEASURED here: the matched-random null PASSes `n_agents_all` at
K1 −1.9087 with `pred_sd/gt_sd` = 0.091 — and the guard stamps it `K1_DEGENERATE`.** The real
latent's `n_agents_all` pass sits at K1 **−3.33 … −3.80** with ratios **0.19 – 0.26**, i.e. above
the floor and 2× the null's margin, so it survives. ⇒ **the guard separates the two cases it was
built to separate, on this experiment's own data.**

⚠️ **`n_agents_grid`'s PASS at 10:1 / 4:1 / 1:1 / cells is `K1_DEGENERATE` (ratios 0.048 – 0.055)
and is NOT quotable as a pass.** It is reported so it cannot be quietly re-quoted later.

### 6.0 ⭐ INTEGRATION — the SIBLING agent's formal C97 guard landed, and I ran it over EVERY row

The brief said a formal degeneracy guard was being built concurrently and that I should **use it if
it lands**. **It landed** — `taniteval/taniteval/degeneracy.py` (`k1_guard`, `screen_banked_k1`,
`SD_RATIO_FLAT_FLOOR`, and an exact `K1 = K1B + K1C` decomposition with a *proved* bound
`|K1B| ≤ pred_sd`). ⇒ **`code/er10_apply_k1_guard.py` IMPORTS it and re-screens all 674 banked K1
rows** (`raw/er10_k1_guard_screen.json`); **no second guard was written.**

| module screen verdict | rows |
|---|---|
| `not screened out (layer 2 still required)` | 333 |
| `SUSPECT — constant-offset component proven` | 292 |
| `SUSPECT — readout is a flat line` | 49 |

⛔ **AND IT SURFACED A RECONCILIATION THAT MATTERS AND THAT I AM ESCALATING (§11.9).** The two floors
differ — my inline stamp used **0.10**, the module's `SD_RATIO_FLAT_FLOOR` is **0.05** — and they
disagree on exactly the row C97 was written about:

⚠️ **The matched-random null's `n_agents_all` K1 PASS at seed 0** (`K1 −1.9087`, `pred_sd/gt_sd`
**0.091**) is **flagged `K1_DEGENERATE` by this deliverable's 0.10 floor** and **NOT screened out by
the module's layer 1 + 3** — its ratio clears 0.05, and `|K1| = 1.91 < pred_sd = 4.23` so the
`k1_exceeds_own_spread` theorem does not fire either. *(At seeds 1 and 2 the same arm collapses to
ratio 0.0001 and IS proven constant-offset.)* ⇒ ⚠️ **a pure-noise arm can still emerge from
layer 1 + 3 with an un-screened `n_agents_all` PASS. The module says so itself — "passing this screen
is NOT evidence the verdict is latent-attributable; that needs layer 2".**

⇒ ⛔ **CONSEQUENCE FOR THIS REPORT, applied rather than noted:** `n_agents_all` is the ONE rung where
our latent passes K1 (§2, §7), **and the matched-random null passes it too**. **Until `k1_guard`'s
layer 2 is run on the predictions, our `n_agents_all` PASS is NOT quotable as latent-attributable** —
only the **margin** is (ours **−3.33 … −4.68**, the null's **−1.89**, and ours is non-degenerate at
ratio 0.19 – 0.26 against the null's 0.0001 – 0.091). **The r²_ceiling comparison in §2 is unaffected
— that is why it, not K1, is the primary quantity (§0.6).**

### 6.1 ⛔⛔ THE TRIVIAL-PROXY CONTROL — ONE SCALAR BEATS ALL FOUR ARMS ON FIVE OF ELEVEN RUNGS

**`MEASURED`** · `raw/er10_proxyv0.json` — the identical ridge with the latent replaced by the
**single scalar `v0`**, the ego's own speed, which the model is **given as an input**.

| rung | **`v0` ALONE (1 feature)** | best pooling arm (of 4) | DEPLOYED cells | reads |
|---|---|---|---|---|
| ⛔ `lead_gap` | **0.4673** — K1 **−1.5581 PASS**, `pred_sd/gt_sd` 0.578 | 0.0750 (1:1) | 0.0038 | ⛔ **6.2× the best arm, 123× the deployed latent** |
| ⛔ `nearest_any` | **0.1689** | 0.0764 (4:1) | 0.1083 | ⛔ **2.2×** |
| ⛔ `n_agents_grid` | **0.1036** — PASS | 0.0580 (4:1) | 0.0876 | ⛔ **1.8×** |
| `n_agents_all` | 0.1120 — K1 **−4.2552 PASS**, ratio 0.176 | 0.1161 (1:1) | **0.1671** | ⚠️ ties the best pooling arm; only the learned readout beats it |
| `ego_curv` | 0.0074 | 0.0000 | 0.0000 | ⛔ v0 alone beats every arm (both negligible) |
| `ego_accel` | 0.0130 | 0.0122 (40:1) | 0.0309 | ⚠️ ties |
| `lead_closing` | **0.0000** | 0.0001 | 0.0005 | — nothing recovers it, including `v0` |
| `lead_inv_ttc` | 0.0025 | 0.0015 | 0.0007 | — all at the null |

⭐⭐ **THIS IS THE SHARPEST STATEMENT AVAILABLE FROM THE EXPERIMENT.** On `lead_gap`, `nearest_any`
and `n_agents_grid`, **a single number the model already receives as an input outperforms every one
of the four pooling arms — including the un-pooled 640-token arm with 491 520 raw features.** ⇒ **the
pooling ratio cannot be the binding constraint on rungs where removing the pool entirely still loses
to one scalar.** This is C92 generalised from `lead_gap` to the whole ladder, and it is why §2.1's
`v0`-partialled table is not an optional column.

⚠️ **Note the one place the latent genuinely wins:** `n_agents_all` — the DEPLOYED cells (0.1671)
beat `v0` alone (0.1120) and every random-projection arm. **Coarse scene density is the one thing
this latent linearly carries that ego speed does not supply.**

---

## 7. ⭐ THE SUPPLEMENT WITH **NO RANDOM PROJECTION AT ALL** — exact ridge on every feature

**`MEASURED`** · `raw/er10_full.json` · `code/er10_full_ridge.py`.

The RP is the mandated dimension-matched control, and §5 measured that **it handicaps the fine
arms**. A `DROP` verdict must not rest on that, so the same fit was repeated with **no projection**:
the ridge's **DUAL (kernel) form** is algebraically identical to the primal and costs `O(n²D + n³)`,
so **all 491 520 features can be fitted exactly** at n_train ≈ 1 302.

⛔ **EQUIVALENCE IS PROVED, NOT ASSERTED.** `MEASURED`: on the 40:1 arm, at the same α and the same
windows, `pc6_linear_readout.ridge_fit(…, intercept_col=-1)` (primal, 12 289 features dense) and this
file's dual agree to **max |Δpred| = 2.205e-06 = 1.41e-06 of the prediction sd** → **PASS**.

| rung | **40:1** (D=12 288) | **10:1** (D=49 152) | **4:1** (D=122 880) | **1:1** (D=**491 520**) |
|---|---|---|---|---|
| `ego_v0` | 0.08681 | 0.13656 | 0.14090 | **0.15643** |
| ⭐ `ego_yawrate` | **0.00054** | 0.00054 | 0.00041 | **0.00021** |
| ⭐ `ego_curv` | **0.00000** | 0.00001 | 0.00001 | **0.00003** |
| `lead_gap` | 0.01148 *(r·v0-partial **−0.120**)* | 0.03937 *(−0.010)* | 0.05171 *(−0.001)* | **0.05131** *(**+0.009**)* |
| ⭐⭐ `lead_closing` | **0.00058** | **0.00045** | **0.00022** | ⛔ **0.00002** |
| ⭐ `lead_inv_ttc` | **0.00167** | **0.00181** | **0.00147** | **0.00096** |
| `n_agents_all` | ⭐ **0.16043** | 0.13980 | 0.11430 | 0.13552 |

⛔⛔ **THE RANDOM PROJECTION WAS NOT HIDING ANYTHING, AND THIS IS THE ROW THAT CLOSES THE ONLY REAL
OBJECTION TO THE VERDICT.** With **every one of 491 520 features fitted exactly**, no dimension
reduction whatsoever and no pooling whatsoever, **`lead_closing` reads r² = 0.00002.** Across the
whole ladder it goes **0.00058 → 0.00045 → 0.00022 → 0.00002** — it **FALLS** as the pool is removed.
⚠️ On those rungs the chosen α sits at the **grid edge (1e9) with `pred_sd/gt_sd` ≈ 0.001–0.009**,
i.e. the fit is **fully shrunk to the train mean**: the readout's own α selection concludes there is
nothing to fit. That is stamped `alpha_at_grid_edge` in the artifact rather than hidden.

⛔ **AND `lead_gap` IS FINALLY SETTLED.** Without any projection, its partial correlation after
removing `v0` runs **−0.120 → −0.010 → −0.001 → +0.009** across the ladder. **r²_partial at the
un-pooled endpoint is 0.00008.** ⇒ **the entire `lead_gap` signal, at every pooling ratio, with every
feature, is ego speed.**

⭐ **AND IT SHARPENS §2.2's SIDE-FINDING.** Without the RP, `n_agents_all` is **best at the DEPLOYED
40:1 ratio (0.16043)** and does not improve at any finer ratio (0.13980 / 0.11430 / 0.13552). **The
one quantity this latent genuinely carries is carried at least as well after the 40:1 mean as before
it.**

⛔ **AND THE PAIRED Δ AGREES WITH §3 ON THE UN-PROJECTED FIT TOO** — `MEASURED`, Δr²(1:1 − 40:1):
`ego_curv` **+0.00003 [−0.00178, +0.00411]**, `ego_yawrate` **−0.00032 [−0.00323, +0.00341]**,
`lead_gap` **+0.03983 [−0.00707, +0.08585]** — **not separated**, exactly as under the projection.
⇒ **removing the RP changes no verdict.**

---

## 8. ⭐ JOB 2 — THE CORPUS-NARROWNESS DISCRIMINATOR (spec §8.1)

### 8.1 ⛔ DINOv3 IS GATED AND OUR TOKEN IS REFUSED — settled by THREE probes, not one

**`MEASURED`** · `raw/dino_availability.json` + this run's own load. Per the absence rule, more than
one location and more than one name were probed:

| probe | method | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
|---|---|---|
| 1 | `HfApi.model_info` | ✅ **OK** — metadata readable, `gated: manual`, 6 files incl. `model.safetensors` |
| 2 | `HfApi.list_repo_files` (a different endpoint) | ✅ **OK** — file list readable |
| ⭐ 3 | an actual **config/weight FETCH** | ⛔ **403** — *"You are trying to access a gated repo"* |

⚠️ **This is exactly the shape that makes a one-probe availability claim wrong in EITHER direction:
the metadata is public and the weights are not.** All four `facebook/dinov3-*` repos probed read
`gated: manual`; `facebook/dinov2-base`, `facebook/dinov2-with-registers-base` and
`timm/vit_base_patch16_224.dino` read `gated: False`.
⛔ **Accepting the licence is a human action on huggingface.co and is NOT mine to take, and I did NOT
look for an unofficial mirror** — that would be licence circumvention. Escalated at §11.2.

### 8.2 ⛔ THE SUBSTITUTION, AND EXACTLY WHAT IT COSTS — declared, not silent

**Substitute: `facebook/dinov2-base` (ViT-B/14), ungated, at 224 × 560.**

⭐ **Why the input size is not arbitrary: it is derived from the patch size so the foreign grid EQUALS
ours.** Our encoder tiles 256 × 640 px at patch 16 into **16 × 40 = 640 tokens**; DINOv2 at patch 14
tiles 224 × 560 into **16 × 40 = 640 tokens**. `MEASURED`, and asserted at runtime rather than
assumed: the aspect ratio is **0.400000 in both cases** — the code **refuses to run** on an
anisotropic resize (`er10_dino_cache.py`) — so the resize is a pure isotropic 0.875× downscale and
**the 40:1 / 10:1 / 4:1 / 1:1 kernels are the IDENTICAL operator on an IDENTICAL grid.**

| axis | cost of the substitution |
|---|---|
| **object scale in patches** | ⚠️ negligible: the median lead's 37.8 px becomes 33.1 px, i.e. **2.36 patches instead of 2.40 (−2 %)**. The quantity the pooling argument is about is preserved |
| **corpus** | ⚠️ **LVD-142M instead of LVD-1689M** — 142 M curated images instead of 1.689 B. ⭐ Still ~60 000× our 2 376 driving episodes and still multi-domain, so it **serves** the diversity argument — but it is a **weaker instance**, so a NEGATIVE here is weaker evidence than DINOv3 would give |
| **architecture** | no RoPE, no register tokens, patch 14 not 16 |
| ⇒ **what may be quoted** | the §8.1 right-column verdict may be stated only as *"not linearly present **to DINOv2-B/14**"*, never as *"not present at all"* |

⚠️ **And the spec's own declared limit survives the substitution:** DINOv2 is **image**-trained, so
per-token relative motion is available to a **linear** readout only through the 3-sub-frame
concatenation (`[640, 3 × 768] = [640, 2304]`, the construction §8.1 specifies). **A negative on the
relative-motion rungs is therefore weaker evidence than a positive one would be.**

⛔ **WINDOW IDENTITY IS PINNED, NOT ASSUMED.** The row set is the banked v6 cache's rows with **only
`tokens` replaced** — same clips, same frame indices, same order, same targets, same split. `cells`
is **deliberately absent** from the foreign cache so an accidental `--arms cells` raises instead of
silently reading v6 numbers. **`MEASURED`: 2 809 rows, grid 16 × 40, `d_model` 2 304, built in 295 s
on the dev-box 4060.** No episode is selected.

### 8.3 ⭐⭐⭐ THE RESULT — THE INFORMATION **IS** IN OUR IMAGES, AND IT **SURVIVES THE DEPLOYED POOL**

**`MEASURED`** · `raw/er10_dino.json` · same 1 507 eval windows, same 70 clusters, same
`intercept_col=-1`, same estimator, 3 projection seeds. ⭐ **Both columns below are the DEPLOYED
40:1 `AvgPool2d((4,10))` — the identical operator on the identical 16 × 40 grid. Only the ENCODER
differs.**

| rung | **DINOv2-B/14, 40:1 cells** | **our v6 encoder, 40:1 cells** | ratio |
|---|---|---|---|
| `ego_v0` | **0.71733** (r **+0.853**) | 0.05240 | ⭐ **13.7×** |
| `lead_gap` | **0.44997** *(`v0`-partial r **+0.346** ⇒ r² **0.120**)* | 0.00496 *(partial **−0.077**)* | ⭐ **91×** |
| `nearest_any` | **0.36460** *(partial +0.489)* | 0.07626 | **4.8×** |
| `n_agents_all` | **0.33630** *(partial +0.541)* | 0.10170 | **3.3×** |
| `n_agents_grid` | **0.32770** *(partial +0.496)* | 0.02856 | **11.5×** |
| `lead_present` | **0.06857** | 0.00512 | **13.4×** |
| ⭐⭐ `lead_closing` | **0.01713** *(partial **+0.129** — it SURVIVES the `v0` control)* | **0.00000** | ⭐ **non-zero vs exactly zero** |
| ⭐ `lead_inv_ttc` | **0.01160** *(partial +0.110)* | 0.00122 | **9.5×** |
| ⭐ `ego_yawrate` | **0.00323** | 0.00028 | **11.5×** |
| `ego_curv` | 0.00040 | 0.00000 | — (both negligible) |
| `ego_accel` | 0.01020 | 0.01224 | 0.83× — the ONE rung where ours wins |

⛔⛔ **THIS RESOLVES §8.1's 2 × 2, AND IT LANDS IN THE CELL THAT KILLS BOTH REMAINING EXCUSES.**
The external encoder's **40:1 CELLS KEEP** relative motion and lead geometry. Per §8.1's own committed
table that is the row *"DINOv3 tokens carry relative motion / DINOv3 40:1 cells KEEP it"* ⇒
**"the pool is survivable with a good enough encoder ⇒ THE ENCODER IS THE LEVER, NOT THE LOSS
PLACEMENT."**

⇒ **Three statements now hold together, and they are mutually reinforcing:**
1. ⛔ **The corpus excuse is dead.** *"The information is not linearly present in our images"* is
   **refuted**: a frozen encoder that has **never seen a driving scene** reads `lead_closing`,
   `lead_gap`, agent counts and ego speed out of **our own frames**.
2. ⛔ **The pooling excuse is dead** (E-R1-0 §2/§3/§7, and again here): the same 40:1 mean that
   allegedly destroys individuation **passes all of it through** for DINOv2.
3. ⭐ **What is left is the ENCODER and the objective that trained it.** Our S-W trunk's encoder,
   with the pool removed entirely and **all 491 520 features** fitted exactly (§7), reads
   `lead_closing` at **0.00002**. DINOv2, through the deployed 40:1 mean and 2 048 features, reads it
   at **0.01713**.

### 8.4 ⭐⭐ AND THE POOLING LEVER'S TRUE MAGNITUDE, MEASURED ON AN ENCODER THAT ACTUALLY CARRIES THE SIGNAL

The v6 ladder is flat because there is nothing to lose. **DINOv2 lets the same ladder be run on an
encoder that demonstrably HAS the information — so it measures what removing the pool is actually
worth, in the best case available to us.**

| rung | DINOv2 **40:1** | DINOv2 **1:1** | **what removing the pool buys** | our v6 **40:1 → 1:1** |
|---|---|---|---|---|
| `lead_gap` | 0.44997 | **0.51510** *(partial-`v0` r +0.480 ⇒ r² **0.230**)* | **+14 %** | 0.00496 → 0.07504 *(partial 0.0060 → 0.0087)* |
| ⭐ `lead_closing` | 0.01713 | **0.02223** *(partial r +0.148)* | **+30 %** | 0.00000 → 0.00002 |
| `lead_inv_ttc` | 0.01160 | 0.01207 | +4 % | 0.00122 → 0.00134 |
| `lead_present` | 0.06857 | **0.14170** | **+107 %** | 0.00512 → 0.01094 |
| `ego_v0` | **0.71733** | 0.67837 | **−5 %** | 0.05240 → 0.12398 |
| `n_agents_all` | **0.33630** | 0.26107 | **−22 %** | 0.10170 → 0.11608 |
| `nearest_any` | **0.36460** | 0.32313 | **−11 %** | 0.07626 → 0.07120 |
| `ego_yawrate` | **0.00323** | 0.00280 | −13 % | 0.00028 → 0.00014 |

⭐⭐ **THE MAGNITUDE THAT SHOULD DECIDE THE SCHEDULE, IN ONE LINE: on an encoder that genuinely
carries lead geometry, removing the deployed 40:1 pool is worth +14 % on `lead_gap` and +30 % on
`lead_closing` — and it COSTS 5–22 % on the ego and scene-density rungs. The ENCODER gap on the same
rungs is 91× and ∞.** ⇒ **even in the best case R1 was ever going to buy a fraction of a fraction of
what the encoder is worth.** That is the strongest available argument for §9.2b's reprioritisation,
and it is `MEASURED`, not argued.

### 8.5 ⭐⭐⭐ THE POSITIVE CONTROL FOR THE **CRITERION ITSELF** — and it is the single most important row in this deliverable

`MEASURED`, `raw/er10_dino.json` → `deltas_vs_p40`. Running §7.1's **pre-registered** contrast on the
foreign encoder:

| encoder | rung | Δr²(1:1 − 40:1) | ALL seeds' CI excludes 0 **and** Δ>0 | survives partialling `v0` | §7.1 verdict |
|---|---|---|---|---|---|
| ⭐ **DINOv2-B/14** | `lead_present` | **+0.07309** | ✅ **YES** | ✅ **YES** | ✅ **`R1 PROCEEDS`** |
| our v6 trunk | *(best of 11 rungs)* | +0.07157 (`ego_v0`) | ⛔ **no** (2 / 5 seeds) | ⛔ no | ⛔ **`R1 IS DROPPED`** |

⛔⛔ **THE PRE-REGISTERED CRITERION IS NOT IMPOSSIBLY STRICT — IT IS SATISFIABLE, AND SOMETHING
SATISFIES IT ON THE SAME 1 507 WINDOWS, THE SAME SPLIT, THE SAME BOOTSTRAP AND THE SAME PROJECTION
BUDGET.** A frozen DINOv2's `lead_present` **passes every clause**: monotone rise as the pool is
removed, paired episode-cluster-bootstrap CI excluding zero on **every** seed, and survival of the
`v0` trivial-proxy control. **Our trunk passes it on zero of eleven rungs.**

⇒ ⭐ **This is the positive control the whole experiment needed and that §7.1 did not think to
specify: not "can the instrument read a planted answer" (PC-DIST/PC-2OBJ answered that), but "can the
DECISION RULE ever return PROCEED".** It can. It just does not, for us.

⚠️ **Note the sign structure, which is the same one §5's sweep predicted:** the pool HELPS the
aggregate/global rungs (`ego_v0`, `n_agents_all`, `nearest_any`) and HURTS the per-object rungs
(`lead_present`, `lead_closing`, `lead_gap`). ⭐ **That is the pooling hypothesis's mechanism appearing
exactly where it should — in an encoder that has object structure to lose.** Our trunk shows no such
sign structure because it has none to lose.

⚠️ **THE ASYMMETRIES, DECLARED — this is not a like-for-like encoder comparison and must not be
quoted as one:**
* **Input budget.** DINOv2 encodes the three sub-frames **separately** and concatenates
  (`3 × 768 = 2 304` per token); our encoder sees the same three frames as **9 stacked channels in
  ONE forward** and emits 768. DINOv2 therefore gets explicit per-frame separation for free. ⚠️ Both
  are projected to the same 2 048 fit features, and both see **exactly the same pixels**.
* **Objective.** Ours is trained to **predict**; a predictive objective is *entitled* to discard what
  its loss does not need. ⭐ **That is the finding, not a defence of it** — §9.2's conclusion is
  precisely that the objective, not the pool, is what to change.
* **Scale.** DINOv2-B/14 is 86 M parameters vs our encoder's **87.3 M** (`MEASURED`, §11.3) — ⭐ **so
  this is NOT a capacity gap.** It is a training-objective and training-corpus gap.
* **Substitution.** Per §8.2 this is DINOv2 (LVD-142M), not the spec's DINOv3 (LVD-1689M). ⭐ For
  **this** direction of result the substitution is *conservative*: a **weaker** external encoder
  already refutes the corpus excuse, so DINOv3 could only strengthen it.

---

## 9. ⛔ THE VERDICT AGAINST THE PRE-REGISTRATION — and what it re-prioritises

### 9.1 The four branches §7.1 committed to, and which one fired

| §7.1 branch | condition | fired? |
|---|---|---|
| ✅ **R1 PROCEEDS** | monotone rise **AND** Δr²(1:1−40:1) CI excludes 0 **AND** survives `v0` **AND** oracle passes | ⛔ **NO** — 0 / 11 rungs |
| 🔸 **DOWNGRADE** | only the **aggregate** rungs improve | ⛔ **NO** — `n_agents_all`'s Δ is +0.014, CI contains 0 on 5/5 seeds |
| ⚠️ **VOID** | the positive control fails | ⛔ **NO** — PC-DIST 1.0000 at every arm, PC-2OBJ a clean 0 → 0.9998 step |
| ⛔ **R1 IS DROPPED** | the Δ CI **contains 0** on the relative-motion / individuation rungs | ✅ ⭐ **YES, on all four, on all five seeds, by three orders of magnitude** |

⭐ **AND THE DECISION RULE ITSELF CARRIES A POSITIVE CONTROL (§8.5), which is the thing that makes a
`DROP` admissible rather than merely convenient:** the same criterion, on the same windows, split,
bootstrap and projection budget, returns **`R1 PROCEEDS` for a frozen DINOv2's `lead_present`**
(Δ +0.07309, every seed separated, survives `v0`). ⇒ **the rule is satisfiable; our trunk does not
satisfy it.**

⛔ **E-R1-0 RETURNS `R1 IS DROPPED`.** The spec's own committed interpretation applies verbatim:
*"the information is not in the tokens either. The pool is NOT the binding constraint, and no loss
that merely reads the tokens can add what the encoder never encoded."* **A fresh S-W run to
introduce a pre-pool token loss is NOT warranted by this evidence** — which is exactly what the
pre-gate existed to decide, given §5.1 (there is no cheap late entry; only a new S-W run can
introduce R1 at all).

### 9.2 ⭐ WHAT IT RE-PRIORITISES — the 2 × 2's PLACEMENT axis is now MEASURED INERT

§2 of the spec frames R1 and R2 as a 2 × 2 whose **placement** main effect is `O3 → R1` and whose
**target** main effect is `O3 → R2-cells`. **E-R1-0 measures the placement axis at T0 and finds it
carries nothing on the rungs of interest.** ⇒ the remaining untested lever is the **TARGET** axis,
and its arm is **R2-cells** — which is also, by the spec's own §6.4, **the cheapest training arm in
the entire design: 16 899 parameters, no attention, none of R1's O(n²) step-time risk.**

⇒ **RECOMMENDATION (escalated, §11): re-order the programme so E-R2-0 (the zero-training target
validation, §7.3) is the next pre-gate, and drop E-R1-1 from the schedule.** This also retires
open item §12.1.3 (R1's unmeasured step-time cost) — it no longer gates anything.

### 9.2b ⭐⭐ AND THE DISCRIMINATOR ADDS A SECOND, LARGER RE-PRIORITISATION

§8's result is not merely "R1's failure is interpretable". It is a **positive** result about where
the lever is, and it is a cheaper experiment than either R1 or R2:

⭐ **A frozen general-purpose encoder, at 86 M parameters against our 87.3 M, extracts from OUR OWN
FRAMES — through OUR OWN deployed pool — what our trained trunk does not carry at all.** The gap is
not capacity, not pooling, not the corpus. ⇒ **the highest-value next experiments are ENCODER
experiments, and two of them cost no training at all:**

1. ⭐ **A frozen-external-encoder READOUT arm (zero training on the trunk).** DINOv2's 40:1 cells
   already beat our trunk's on 10 of 11 rungs. **How much of the operative planner's job can a frozen
   external encoder + a small learned readout do?** This is the DINO-WM recipe the programme already
   named as the v3 direction (`MEMORY: tanitad-v3-direction`), now with a measured reason.
2. ⭐ **A DISTILLATION target.** If DINOv2's tokens carry what our objective discards, an auxiliary
   that regresses our encoder's tokens onto a frozen DINOv2's tokens is a **label-free `aux` loss**
   in exactly the family `ISOLATION_MATRIX` already permits — and it is a *target*-axis lever, i.e.
   the same axis §9.2 identified as the only one still untested.

⛔ **NEITHER IS PROPOSED AS A DESIGN HERE** — naming them as the consequence of a specific measured
outcome is what stops them becoming scope creep argued after the fact (the spec's own R3 discipline).
**They are escalated to the PI in §11.1 as the reprioritisation this pre-gate bought.**

### 9.3 ⚠️ WHAT THIS DOES **NOT** SAY — three limits I am declaring rather than burying

1. ⛔ **It does not refute R3 (attentive pooling), and it must not be quoted as doing so.** PC-2OBJ
   shows the mean *does* annihilate **signed, opposing contributions inside one cell**; PC-LOCAL
   shows it does *not* damage a single localised code. R3's premise — *"the mean is the destroyer"* —
   is **narrowed, not refuted**: the mean is destructive only for superposed opposing structure, and
   the latent gives no evidence of carrying any. R3 would still change what the trunk sees; it simply
   can no longer be justified by "the pool destroys individuation".
2. ⚠️ **It is a LINEAR readout at ONE checkpoint.** A quantity absent here could be present
   non-linearly, and every number is an **EARLY-READ at 11 250 / 30 000 = 37.5 %**. ⭐ **The cheap
   fix is named and costed: re-running this entire panel on the 30 k checkpoint is ~25 minutes of
   dev-box GPU + CPU and ZERO Thor**, because the harness, the gates and the controls are all
   committed here — only a new `sp1` token cache is needed.
3. ⚠️ **The window set is LEAD-ENRICHED, not parity** (`parity: False`), inherited unchanged from the
   precedent. It selects nothing new, but it is not the canonical 2 376-episode corpus and no number
   here may be compared to one that is.

---

## 10. THE FOUR METRIC FAMILIES — per family, with its tier, its emitter, and its n

⛔ Per the 2026-08-02 binding rule, ADE alone is never "the result"; and per C95 every quantity names
its **tier** *and* its **emitter**.

| family | quantity | tier | emitter | result |
|---|---|---|---|---|
| **LONGITUDINAL** — target speed | `ego_v0`, `ego_accel` | **T0** | this ladder (`er10_pool_ladder.py`, `ridge_fit(intercept_col=-1)`) | `ego_v0` r² 0.0524 → 0.1240 across the ladder (2 / 5 seeds separated ⇒ **fails the criterion**); `ego_accel` 0.0122 → 0.0037 (**falls**) |
| **LONGITUDINAL** — distance keeping | `lead_gap`, `lead_closing`, `lead_inv_ttc` | **T0** | same | `lead_gap` 0.0050 → 0.0750 raw but **0.0060 → 0.0087** with `v0` partialled; ⛔ `lead_closing` **0.0000 at every ratio**; `lead_inv_ttc` **0.0012 → 0.0013**, both at the random null |
| **LATERAL** — heading / curvature / yaw-rate | `ego_yawrate`, `ego_curv` | **T0** | same | ⛔ **0.0003 → 0.0001** and **0.0000 → 0.0000**; both at or below the matched-random null at every ratio |
| **LATERAL** — cross-track | — | — | — | ⛔ **NOT APPLICABLE at T0, with the reason: a frozen-latent linear readout emits no trajectory, so there is no track to be cross of.** Available only at T1 (`taniteval/tools/t1_eval.py`). n = 0 by construction |
| **TACTICAL** — manoeuvre decision / goal setting | — | — | — | ⛔ **NOT COMPUTABLE from a frozen S-W trunk** — it needs `layer_tac` + `planner`, which S-W does not train. ⚠️ C95: `sel_gap` exists at two scopes and the trainer's T0 log key is **not** what the S-T gate reads. The spec's §12.2 work item (a manoeuvre-class **classification** rung on this ladder) remains open and is **named, not excused** — see §11 |
| **STRATEGIC** — route / goal quality | — | — | — | ⛔ **NOT COMPUTABLE before S-S exists.** Declared with the reason |

⛔ **No experiment in this deliverable produces a T1 number, and none may be reported as "the model
drives better" or "worse".**

---

## 11. ⛔ ESCALATIONS AND OPEN ITEMS — raised here, not parked in a README

**These need a decision or an owner. Rule 3 of the operating standard: an orthogonality instrument
sat unmerged for 10 days because the request lived in a README nobody re-read.**

1. ⛔ **E-R1-1 SHOULD BE REMOVED FROM THE SCHEDULE, AND E-R2-0 PROMOTED TO NEXT PRE-GATE.**
   E-R1-0 returned `R1 IS DROPPED` on its own pre-registration (§9.1). The spec's 2 × 2 has one
   remaining untested lever — the **target** axis — and its arm is **R2-cells** (16 899 params, no
   attention). ⭐ **This is a decision about a whole S-W training run and it is mine to raise, not to
   take.**
2. ⛔ **`facebook/dinov3-*` IS GATED (`gated: manual`) AND OUR TOKEN IS REFUSED (403).** A human must
   accept the licence at `huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m` for the spec's
   preferred §8.1 arm. ⛔ **Accepting a licence is not an action I may take, and I did NOT look for an
   unofficial mirror** — that would be licence circumvention. §8 below runs the strongest *ungated*
   substitute and declares exactly what the substitution costs.
3. ⛔ **THE SPEC'S §1.7 FRAMING NUMBER IS A SCOPE ERROR, AND I RE-DERIVED IT FROM THE LIVE
   CHECKPOINT.** §1.7 computes its table from **`V6Config()` DEFAULTS** and presents it as the shape
   of the problem. **`MEASURED` from the banked `v6F_sw_step011250.fp16.pt` state_dict — 573 tensors,
   verified to contain NO buffers and NO shared storages, so the sum is a clean parameter count:**

   | group | spec §1.7 (`V6Config()` defaults) | **LIVE `@11250` (MEASURED)** |
   |---|---|---|
   | `predictor_op` | 60 193 539 — **68.5 %** | **187 985 408 — 55.9 %** |
   | `encoder` | 15 327 360 — **17.4 %** | **87 284 736 — 25.9 %** |
   | `readout` | 49 280 — 0.06 % | **98 432 — 0.029 %** |
   | `masked_cells` (the O3 head) | 1 649 792 — 1.9 % | 1 649 792 — 0.49 % |
   | `predictor_tac` / `predictor_str` | (as `layer_*`) 5 765 165 / 4 152 993 | 27 033 344 / 26 377 728 |
   | **total** | 87 893 449 | **336 559 305** |

   Live config: `enc_dim 768, enc_depth 12, enc_heads 12, vit5_encoder True`;
   `pred_dim 1024, pred_depth 12, pred_heads 16`; `readout_grid 4, readout_dim 128`.
   ⇒ ⛔ **the spec's headline — *"68.5 % of the stack sits downstream of a 40:1 mean and the
   perception path that feeds it holds 17.4 %"* — is 55.9 % / 25.9 % on the live model, and the total
   is understated 3.8×.** The rhetorical gap between the two halves is **2.2×, not 3.9×**. Same
   root-cause class as `CLAUDE.md`'s Thor "per-worker cost" retraction: **a number MEASURED in one
   scope, quoted in another.**
4. ⚠️ **AND A SEPARATE FACT THAT FALLS OUT OF IT, VERIFIED BEFORE ALARMING.** 336 559 305 params
   exceeds `PARAM_BUDGET = 300_000_000` (`v6.py:142`) — but this is **NOT a silent invariant breach**:
   `MEASURED`, the run was launched with **`param_budget: 350000000`** in its own args, so
   `assert_param_budget()` (`train_v6_staged.py:2255`) passed by construction. ⚠️ **It is, however, a
   contradiction with the programme's own one-line identity — `CLAUDE.md` opens "Sub-300M
   hierarchical 4-brain latent world model" — and with the spec's §1.7 "headroom 212 106 551".
   The live S-W run is 336.6 M, 12.2 % over "sub-300M", by an explicit launch-time override.**
   ⇒ **for the PI:** either the identity line is stale or the override was not intended to persist.
   Not mine to decide; raised because two documents now disagree with the artifact.
5. ⚠️ **`pc6_linear_readout.py` — the instrument this and at least four other deliverables depend on —
   still lives ONLY in `…/incoming/2026-08-17-probe-positive-control/code/`**, and
   `taniteval/tests/test_ridge_intercept_penalty.py` reaches across the repo **by path** to test it.
   The spec raised this (§12.1.2); **it is still true, and this deliverable adds a fifth dependant.**
   Operating-standard rule 3 territory. Not mine to move mid-flight.
6. ⚠️ **C97's outstanding backlog is untouched by this work and grows by nothing here:** 24 files,
   214 verdict rows, **170 still standing on the biased solve, 90 of them separated FAILs**. Every
   number in THIS deliverable is on the repaired solve, so it needs no re-read — but the ladder it
   descends from (`…/2026-08-17-latent-linear-ladder/`) does.
7. ⭐ **A cheap, named follow-on rather than an excuse (rule 3): the 30 k re-read.** Everything here —
   harness, three gates, four controls, the pre-registered Δ — is committed and re-runnable. When the
   live S-W run reaches 30 000, re-running this panel needs **one new `sp1` token cache and ~25 min of
   dev-box time, ZERO Thor**. The verdict in §9 is stamped `@11250` and should be re-stamped then.
8. ⚠️ **The T0 TACTICAL rung the spec named (§12.2) is still absent** — the ladder is regression-only,
   so manoeuvre-class decodability from the frozen latent cannot be reported. It is cheap (a
   multinomial ridge on these same banked features) and it would make the TACTICAL family reportable
   at T0 for the first time. **Named as a work item, not used as an excuse** (§10 reports the family
   as NOT COMPUTABLE with its reason).

---

## 12. ⚠️ WHAT I DID NOT DO — stated plainly rather than scoped away

* ⛔ **I changed NOTHING in `stack/`, `taniteval/`, or any training path.** No module was added, no
  trainer touched, no checkpoint written. The live Thor S-W run was never contacted.
* ⚠️ **I DID change the shared dev-box venv, and I re-ran both suites BECAUSE of it.** The §8
  discriminator needs a model loader, so `transformers 5.15.0`, `tokenizers 0.22.2`, `safetensors
  0.8.0` and `regex` were installed **with `--no-deps`** precisely so nothing could drag torch
  forward (`CLAUDE.md`'s `uv pip install` trap). **`MEASURED` after the install, by the rule that
  `import torch` is not proof: a real `Conv2d` on CUDA succeeds and `torch` is unchanged at
  `2.11.0+cu128` / CUDA 12.8.** Suites re-run on the changed environment — **`stack` 3842 passed,
  0 failed, 7 skipped, 2 xfailed (821 s); `taniteval` 1136 passed, 0 failed (291 s)** —
  banked at `raw/suite_stack.txt` and `raw/suite_taniteval.txt`. ⚠️ Both counts are **higher** than
  the brief's inherited 3816 / 1107 because sibling agents landed tests during this session; the
  relevant fact is **zero failures**.
* **I did not re-measure the 37.77 px lead width** (the spec's §1.4) — it needs the obstacle join.
  It stays `INHERITED`, and nothing here depends on it: the pooling geometry is measured directly
  (§1, J2).
* **I did not run E-R2-0.** It is the spec's other zero-training pre-gate and, on this evidence, the
  one that should run next (§11.1) — but it needs an optical-flow estimator decision that is an ops
  call I did not make (spec §12.1.4).
* **I did not resolve the `ISOLATION_MATRIX` ruling on R2** (spec §5.4). Unchanged and still a
  prerequisite for E-R2-1.
* **I did not accept the DINOv3 licence** (§11.2), and did not use an unofficial mirror.

---

## 13. DELIVERABLE MANIFEST

⛔ **Nothing produced here lives on a pod, in a worktree, or only in my context.** Every artifact is
in the repo and **STAGED** (`git add`, never committed, never pushed). Nothing in `stack/`,
`taniteval/`, or any training path was created or modified.

**Base:** `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-18-pooling-ladder-ER10/`

| artifact | what it is | in only ONE place? |
|---|---|---|
| `POOLING_LADDER_ER10.md` | this report | no — **staged** |
| `code/er10_pool_ladder.py` | the ladder: pool → fixed random projection → repaired ridge → paired episode-cluster bootstrap; the three planted positive controls; the C92 and C97 guards; the banked-reproduction gate | no — **staged** |
| `code/er10_full_ridge.py` | the no-projection supplement: **exact DUAL ridge on all 491 520 features**, gated against the primal `pc6` solve | no — **staged** |
| `code/er10_j2_gridcheck.py` | J2 — the impulse pin on the token grid + `proj(pool(tokens))` vs the banked `cells` using the checkpoint's own `readout.proj` | no — **staged** |
| `code/er10_dino_probe.py` | the §8.1 weight-availability probe (7 candidates × 2 API paths) | no — **staged** |
| `code/er10_dino_cache.py` | the frozen external encoder over the SAME banked windows, with the aspect-ratio and grid asserts | no — **staged** |
| `code/er10_summarise.py` | renders every table in this report **from the banked JSONs only** — it computes nothing | no — **staged** |
| ⭐ `code/er10_apply_k1_guard.py` | **INTEGRATION** — imports the sibling agent's `taniteval.degeneracy.screen_banked_k1` and re-screens all 674 banked K1 rows. **Not a second guard** | no — **staged** |
| `raw/er10_k1_guard_screen.json` | that screen's output, per row, with both floors recorded | no — **staged** |
| `raw/suite_stack.txt`, `raw/suite_taniteval.txt` | the two suites re-run **after** the venv change (3842 / 1136 passed, 0 failed) | no — **staged** |
| `raw/dino_meta.json` | the external-encoder build provenance (patch, input size, aspect assert, window-identity pin, wall 295.4 s) | no — **staged** |
| `code/chain_er10.sh` | the runbook: `gate → pc → main → null → proxy → pcsweep → full → dino` | no — **staged** |
| `raw/er10_gate.json` | J5 reproduction gate vs `ll_cells_tokwin.json` (66 checks, PASS) | no — **staged** |
| `raw/er10_j2_gridcheck.json` | J2 result | no — **staged** |
| `raw/er10_main.json` | ⭐ **the ladder** — 5 arms × 11 rungs × 5 projection seeds + the pre-registered paired Δ | no — **staged** |
| `raw/er10_null.json` | matched-random null, per arm | no — **staged** |
| `raw/er10_proxyv0.json` | ⭐ the C-V0 trivial-proxy arm | no — **staged** |
| `raw/er10_pc_dist.json`, `raw/er10_pc_local.json`, `raw/er10_pc_local2.json` | the three planted positive controls | no — **staged** |
| `raw/er10_pcloc_a*.json` | the PC-LOCAL amplitude sweep | no — **staged** |
| `raw/er10_full.json` | the exact full-feature dual-ridge supplement | no — **staged** |
| `raw/er10_dino.json`, `raw/dino_meta.json` | ⭐ the §8.1 corpus-narrowness discriminator | no — **staged** |
| `raw/dino_availability.json` | the DINOv3 gating measurement (3 probes) | no — **staged** |
| `raw/log_er10_*.txt` | the per-run stdout of every stage | no — **staged** |

**Intermediates deliberately NOT staged** (regenerable, and large): the banked token caches and the
external-encoder cache live in the session scratchpad
(`…/scratchpad/sp2/cache_tok11250/`, `…/scratchpad/er10/`). ⚠️ **The v6 token cache
(`cache_tok11250`, 2.78 GB) is a session-scratchpad artifact produced by the PRECEDENT agent and is
NOT in git.** It is regenerable from `sp1_cache_latents.py --want-tokens` + the checkpoint, both of
which are in the repo, but **if the scratchpad is cleared it costs a ~9-minute GPU rebuild.**
Flagged, not assumed.

---

## 14. ⛔ TWO LATE ESCALATIONS (numbered to follow §11)

9. ⛔ **TO THE OWNER OF `taniteval/taniteval/degeneracy.py` — `SD_RATIO_FLAT_FLOOR = 0.05` does NOT
   flag the case C97 was written about.** `MEASURED` (§6.0, `raw/er10_k1_guard_screen.json`): a
   **matched-random null** PASSes `n_agents_all` at **K1 −1.9087** with `pred_sd/gt_sd` **0.091** —
   above the 0.05 floor — and `|K1| = 1.91 < pred_sd = 4.23`, so the `k1_exceeds_own_spread` theorem
   does not fire either. **Layer 1 + 3 therefore lets a pure-noise PASS through on the module's own
   headline example.** This deliverable's stricter **0.10** floor does flag it. ⇒ **either the floor
   should move, or layer 2 must be mandatory before any `n_agents_all` PASS is quoted.** Not mine to
   change in a sibling's module mid-flight; raised because I now depend on it.
10. ⚠️ **THE INDEX CONTAINS OTHER AGENTS' STAGED WORK — I did not commit and did not `--amend`.**
   `MEASURED` at end of turn, foreign staged entries: `Project Steering/RETRACTION_LOG.md`,
   `…/incoming/2026-08-17-latent-linear-ladder/LATENT_LINEAR_LADDER.md`,
   `…/incoming/2026-08-18-ladder-corrected/raw/suite_stack.txt`,
   `…/incoming/2026-08-18-thor-closure-audit/THOR_CLOSURE_AUDIT.md`,
   `stack/scripts/launch_closure_audit.py`, `stack/tests/test_launch_closure_audit.py`.
   ⛔ **Whoever commits must read `CLAUDE.md`'s git-hygiene rule first** — `git commit` takes the
   WHOLE index, and this index is shared.
