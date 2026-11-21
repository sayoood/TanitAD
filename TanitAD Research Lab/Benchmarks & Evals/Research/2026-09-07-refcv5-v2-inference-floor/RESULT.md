# refcv5-v2 INFERENCE-RUN NOISE FLOOR --- and a nav watch item found on the way

**Status: PARTIAL.** Seed 0 complete; seeds 1 and 2 rolling on the dev-box RTX 4060.
⛔ **No conclusion about the noise floor may be drawn from this file until all three land.**

## 0. What this measures, and what it explicitly does NOT

**Question:** refcv5-v2 runs `--sampler ddim`, which draws `torch.randn_like` **at eval**
(`refc.py:1926`). refcv4b does not --- MEASURED: seeds 0 and 7 bit-identical, 4/4 episode dumps
by md5. ⇒ **inference noise enters the landing's paired delta from ONE SIDE ONLY**, and nobody
has measured how much. `CLAUDE.md` binds: *for any arm whose planner samples, a replicate must
vary the INFERENCE seed, and an effect smaller than that floor is not an effect.*

**Design.** Three rolls of the arm's OWN `ckpt_15000.pt` (md5 `f2853e722afcd9eba60df2a3205517ab`,
pulled from the pod; **0 A40 GPU** --- the training run was untouched) at
`--infer-seed 0 / 1 / 2`. Same checkpoint, same flags, same corpus, same bootstrap seed; **one
variable.** This is `A0b_replicate`'s design moved one level down ⇒ **any separated cell between
two of these rolls is a FALSE POSITIVE BY CONSTRUCTION.**

⛔ **THE CHECKPOINT IS STEP 15,000 OF 40,284 --- MID-TRAINING.** No LEVEL in this file is a
capability number and none may be quoted as one. The deliverable is the **SPREAD**, and even that
is an **ESTIMATE** for the 40k arm: a sharper model may have a tighter or wider sampling
distribution. ⛔ The landing still runs its own replicate; this does not replace it.

**Two controls, both fixed BEFORE any data existed:**
1. ⭐ **Known-value control, free and already in the harness.** `refcv3_arm.py`'s own source says
   the model-free arms `ha` / `ha0` / `ha0_ext` **read no frames**. They are deterministic by
   construction and **MUST be identical across inference seeds.** If they move, the EXPERIMENT is
   broken --- not the arm.
2. ⭐ **Discriminator for the null result.** `--infer-seed` provably reaches the draw
   (`refcv3_arm.py:1778` `torch.manual_seed` + `torch.cuda.manual_seed_all`). ⇒ three IDENTICAL
   rolls would mean **the DDIM branch is not reached at eval**, NOT that seeding is broken. Fixing
   this in advance is what makes a null interpretable instead of ambiguous.

**Evidence class: MEASURED (ours).** Tier **T1** (self-action open loop). Estimator
`episode_cluster_bootstrap`, n = 4,823 windows / 141 episodes, n_boot 2,000.
⛔ `overlapping_holdout_se` is not used anywhere here.

## 1. Seed 0 --- arm levels (⛔ mid-training; recorded to compare against seeds 1/2, nothing else)

| arm | ade_dense_m | fde_last_m | LON_speed_mae | LAT_cross_mae |
|---|---|---|---|---|
| `os` | 0.4007 | 0.8439 | 0.3388 | 0.1797 |
| `ha` ⭐ model-free | 0.2996 | 0.6588 | 0.2540 | 0.1226 |
| `ha0` ⭐ model-free | 0.6723 | 1.4029 | 0.4880 | 0.3132 |
| `ha0_ext` ⭐ model-free | 0.2874 | 0.6323 | 0.2540 | 0.1070 |
| `os_navshuf` | 0.4007 | 0.8452 | 0.3391 | 0.1779 |
| `os_navzero` | 0.4005 | 0.8419 | 0.3413 | 0.1719 |
| `oracle_sel` (T0) | 0.2938 | 0.5811 | 0.2251 | 0.1592 |

⭐ The three model-free rows are the ones to watch across seeds. **They must not move at all.**

## 2. Seed 0 --- paired deltas (same windows, paired episode-cluster bootstrap)

| comparison | ade delta | 95 % CI | separated |
|---|---|---|---|
| `os - ha0` | **-0.27162** | [-0.32507, -0.21958] | yes |
| `os - ha` | **+0.10117** | [+0.08231, +0.12206] | yes |
| `os - ha0_ext` | **+0.11342** | [+0.09429, +0.13493] | yes |
| `oracle_sel - os` | **-0.10702** | [-0.12829, -0.08805] | yes |
| `os - os_navshuf` | **-0.00001** | [-0.00466, +0.00486] | **no** |
| `os - os_navzero` | **+0.00025** | [-0.00531, +0.00602] | **no** |
| `os_navzero - ha0_ext` | **+0.11317** | [+0.09446, +0.13422] | yes |

⛔ **At step 15,000 the arm would FAIL both bars** --- separated WORSE than `ha0_ext` (+0.1134)
and than `ha` (+0.1012). ⚠️ **This is not a prediction about the landing.** It is 15,000 of 40,284
steps, and the arm does already beat constant-velocity `ha0` by a separated -0.2716.

⭐ **`oracle_sel - os` = -0.1070 separated** prices the SELECTION headroom: a perfect selector
over the same fan is worth 0.107 m at this checkpoint. That is the `E-READOUT`-adjacent finding
restated on the real arm --- the binding constraint is the selection step.

## 3. ⚠️ THE WATCH ITEM --- nav moves essentially nothing at this checkpoint

`os - os_navshuf` = **-0.00001** and `os - os_navzero` = **+0.00025**, neither separated, on a
paired CI of about **±0.005**. Restated where it bites: **zeroing the nav input changes the arm's
distance from its own acceptance bar by 0.00025 m** (+0.11342 → +0.11317).

⛔ **WHAT THIS IS NOT.** It is **not** "refcv5-v2 ignores nav":
* step **15,000 of 40,284** --- the arm may not have learned to use it yet;
* **ONE inference seed**. The paired bootstrap answers *would another draw of EPISODES say this?*
  and is structurally blind to the sampler. ⇒ **if the seed-to-seed spread of `os` exceeds ~0.005,
  this bound is not even resolvable**, and that is exactly what seeds 1 and 2 decide. Pairing
  removes noise common to both arms, so the bound is plausibly real --- but "plausibly" is not a
  measurement.

⭐ **Why it is recorded now anyway:** the PI's standing mandate is *"assure that all necessary
vocab are used weiter as inputs like nav commands and max speed."* An arm whose nav input is
worth 0.00025 m is the exact condition that mandate exists to catch, and it is **cheap to re-read
at the landing** --- `os_navshuf` and `os_navzero` are already in every roll. ⇒ **carried as a
WATCH ITEM on the landing read, not as a finding.**

⚠️ Related and already binding: all 4,719 v7.2 nav records are `ego-future`. Per the PI ruling of
2026-09-04 that is a first-class ROUTE INPUT, not a leak --- but it is noiseless and perfectly
timed where a real router is coarse, so no arm's nav may be read as a production command.

## 4. RESULT --- all three seeds in

**Status: COMPLETE.** `--infer-seed 0 / 1 / 2`, same checkpoint, same flags, one variable.
Raw: `raw/floor_3seed.json`, `raw/floor_table_3seed.json`.

### 4.1 ✅ The known-value control read its known value --- THREE TIMES, EXACTLY

| arm | s0 | s1 | s2 | range |
|---|---|---|---|---|
| `ha` | 0.2996 | 0.2996 | 0.2996 | **0.00000** |
| `ha0` | 0.6723 | 0.6723 | 0.6723 | **0.00000** |
| `ha0_ext` | 0.2874 | 0.2874 | 0.2874 | **0.00000** |
| `paired_ha_minus_ha0` | −0.37279 | −0.37279 | −0.37279 | **0.00000** |

⭐ These arms **read no frames**, so they are deterministic BY CONSTRUCTION and this is what a
passing control looks like: not "close", **exactly zero**, on all six metrics, three times.
Meanwhile every MODEL arm moved ⇒ **the DDIM branch IS reached at eval**, which is the
discriminator fixed in §0 before any data existed. A null here would have been ambiguous without
it; it is not ambiguous.

### 4.2 ⭐ THE FLOOR IS METRIC-SPECIFIC, AND IT SCALES WITH HOW MANY STOCHASTIC ARMS THE STATISTIC DIFFERENCES

This is the transferable result, and it is a clean three-tier ladder:

| statistic | stochastic arms differenced | seed range |
|---|---|---|
| `ha − ha0` | **0** | **0.00000** |
| `os − ha0_ext`, `os − ha`, `os − ha0` | **1** | **0.00073** |
| `os − navshuf` | **2** | **0.00223** |
| `os − navzero` | **2** | **0.00364** |

⇒ **A delta against a DETERMINISTIC control costs ~0.0007; a delta between two SAMPLING arms
costs 3–5× that.** ⛔ So "the inference floor of this arm" is not one number — **quote the floor
for the statistic you are claiming on.** *(Same family as the estimator rules: `H-ESTIM-SEED-1`
says a separated CI answers the EPISODE question only; this adds that even the inference question
has a different answer per statistic.)*

### 4.3 ✅ THE LANDING BAR IS SAFE --- 155× the floor

`os − ha0_ext` = **+0.11342 / +0.11288 / +0.11269**, mean **+0.11300**, seed range **0.00073**,
**separated at all three seeds**. ⇒ **ratio 155.1×.** Sampler noise does **not** threaten
`BAR-REFCV5V2-1`. The worry that motivated this measurement is resolved, and resolved favourably.
`os − ha` behaves identically (range 0.00073, separated ×3), so `BAR-REFCV5V2-2` is equally safe.
⭐ `oracle_sel − os` = −0.10702 / −0.10605 / −0.10481 (range 0.00221, separated ×3, **48× the
floor**) ⇒ the **selection headroom is real** on the arm itself, not only on a tiny rig.

### 4.4 ⛔ THE NAV EFFECT IS SMALLER THAN ITS OWN NOISE — `NOT MEASURED`, not "nav does nothing"

| | s0 | s1 | s2 | range | mean | effect / floor |
|---|---|---|---|---|---|---|
| `os − navshuf` | −0.00001 | −0.00224 | −0.00201 | 0.00223 | −0.00142 | **0.64×** |
| `os − navzero` | **+0.00025** | −0.00101 | **−0.00339** | 0.00364 | −0.00138 | 0.38× |

⛔ **The effect is 0.64× its own inference floor, and `os − navzero` FLIPS SIGN across seeds.**
None of the six cells is separated. ⇒ the correct verdict is **`NOT MEASURED` at this
checkpoint** — the instrument cannot resolve an effect this small on a sampling planner.

⛔ **THIS IS NOT "the arm ignores nav", AND §3'S WATCH ITEM IS NOW SUPERSEDED BY IT.** §3 recorded
a seed-0 point estimate of **+0.00025** and asked whether it would survive. It did not: it is
smaller than the floor, and its sign is not stable. ⭐ **Had this been banked at one seed it would
have published a sign that does not exist** — which is exactly what a replicate arm is for, one
level down from `H-ESTIM-SEED-1`.
✅ What IS admissible is a **BOUND**: the paired CI at every seed straddles zero with half-width
≈ 0.005 m, so **|nav effect on ADE| < ~0.005 m at step 15,000**, against an arm sitting 0.113 m
from its bar. ⚠️ Whether nav matters at **40,284** steps is untested and remains a landing
watch item — `os_navshuf` / `os_navzero` are in every roll, so re-reading it is FREE.

### 4.5 Scope

⛔ **Step 15,000 of 40,284.** No LEVEL here is a capability number and none may be quoted as one:
at this checkpoint the arm would FAIL both bars (separated WORSE than `ha0_ext` by +0.113 and than
`ha` by +0.100), while beating constant-velocity `ha0` by −0.272. That is training progress, not
a verdict. The **SPREAD** is the deliverable, and it is an **ESTIMATE** for the 40k arm — a
sharper model may sample differently. ⛔ **The landing still runs its own replicate**
(`compare.sh --infer-seed 1`); this measurement tells us what to expect, it does not replace it.
