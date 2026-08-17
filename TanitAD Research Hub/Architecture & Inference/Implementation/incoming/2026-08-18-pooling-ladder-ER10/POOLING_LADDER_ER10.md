# ⛔ E-R1-0 — THE POOL IS **NOT** THE BINDING CONSTRAINT. Removing it entirely changes NOTHING on the four rungs the pooling hypothesis was built to explain

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
the same 2 048 features. **`MEASURED`: the learned readout beats the random one on 9 of 11 rungs**,
by up to **1.6×** (`n_agents_all` 0.1671 vs 0.1017, `nearest_any` 0.1083 vs 0.0763, `ego_v0` 0.1026
vs 0.0524) — and the paired Δ on `n_agents_all` is the **only contrast in the entire experiment that
is separated on all five seeds and survives partialling `v0`** (Δr² **+0.0693**, Δ MAE **−1.349
[−2.619, −0.181]** separated). ⇒ **the 49 280-parameter "geometry firewall" is doing real work.** It
loses on exactly one rung — `lead_gap` (0.0038 vs 0.0050) — which is the ego-speed proxy.

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

