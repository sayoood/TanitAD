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

## 4. Seeds 1 and 2 --- PENDING

When they land: (a) assert `ha` / `ha0` / `ha0_ext` are **identical** to seed 0 --- if not, stop and
fix the experiment; (b) report the spread of `os` and, more importantly, of **`os - ha0_ext`**,
the bar's OWN statistic; (c) state whether the ±0.005 nav bound in §3 survives the sampler.

*Instrument: `C:\Users\Admin\refcv5cmp\noisefloor.sh`, `floor_extract.py`. Bar and prereg:
`Project Steering/PREREG_REFCV5_V2_LANDING.md`. Register row: GOALS_AND_CLAIMS "That owed
pre-registration is now WRITTEN".*
