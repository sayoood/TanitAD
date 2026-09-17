# RL Stage A — §13.2 **SUCCESS**, and the lever clears the rig's OWN floor on 9 of 10 metrics. ⛔ `sel_ade_m` does not and is withdrawn; the primary endpoint clears its floor by only 1.4×.

**Date:** 2026-09-17 · **Evidence class: MEASURED** · **Tier: T0** (deployed sampler, recorded
future, proxy reward) · **Estimator:** `taniteval.ci.paired_episode_cluster_bootstrap`, n_boot 2000,
seed 0 · **2 of 3 Stage A arms complete:** `L1-RL-s0` ✅ · `L1-NORL-s0` ✅ · `L1-RL-s1` ⛔ **FAILED
TWICE (CUDA OOM)**.

## ⛔⛔ Read this before any number below

Nine of twelve paired metrics are **separated** on the programme's own estimator. ⛔ **That is
NECESSARY, NOT SUFFICIENT, and this rig is the reason the rule exists.** A separated interval from
**one seed per arm** answers *"would another draw of episodes say this?"* — never *"would another
training run say this?"*. MEASURED elsewhere in this programme: a replicate at the **same seed**
cleared "separated" on **14.3 %** of cells on the v7-tiny rig and **55.6 %** on WP-D. The replicate
is what bounds that floor **and it has not completed**.

⇒ **No statement below is a claim about the RL lever.** They are absolute, controlled, paired
differences between two arms that differ in one variable, awaiting their floor.

## ⭐⭐⭐ THE VERDICT — §13.2 **SUCCESS**, and the replicate REHABILITATES most of this package

`L1-RL-s1` ran **600/600** unchanged on a quiet box (`grad_norm_max` **203.58**, clipped
**2/600 = 0.33 %**, wall 1,968.0 s, 1.30 GB checkpoint), verified by **artifact** — 600 rows and a
checkpoint — never by exit code.

### §13.1 F1 — all three arms PASS

| arm | Δspread vs BASE | 95 % CI | mean | retained | **F1** |
|---|---|---|---|---|---|
| L1-RL-s0 | −8.9872 | [−11.3541, −6.5142] | 28.53 m | 76.0 % | **PASS** |
| L1-NORL-s0 | −2.1583 | [−3.9559, −0.3553] | 35.36 m | 94.2 % | **PASS** |
| **L1-RL-s1** | **−9.8677** | [−12.1547, −7.4317] | **27.65 m** | **73.7 %** | **PASS** |

### §13.2 PRIMARY ENDPOINT — both seeds separate positive

`Δfan(s0)` **+0.0363** [+0.0243, +0.0475] · `Δfan(s1)` **+0.0616** [+0.0493, +0.0735].
Both lower bounds > 0, all three F1 PASS ⇒ **§13.2 VERDICT: SUCCESS**, on the pre-registered rule.

### ⭐⭐ AND THE THING THE WHOLE PACKAGE WAS WAITING FOR: THE L1 RIG'S **OWN** FLOOR IS TIGHT

`L1-RL-s1` vs `L1-RL-s0` — same flags, same lever, zero levers moved — separates on **4 of 10**,
against the release rig's **10 of 10**:

| metric | RELEASE-form floor | **L1 floor** | tighter by |
|---|---|---|---|
| `fan_minade_m` | −2.1319 | −0.0051 | **418.0×** |
| `fan_pdms_best` | +0.1113 | +0.0004 | **278.2×** |
| `fan_nc_fail_frac` | −0.1126 | −0.0024 | 46.9× |
| `sel_ttc` | +0.1866 | +0.0041 | 45.5× |
| `fan_endpoint_spread_m` | −17.2200 | −0.8805 | 19.6× |
| `sel_pdms` | +0.2431 | +0.0145 | 16.8× |
| `fan_pdms_mean` | +0.2474 | +0.0253 | 9.8× |
| `sel_nc` | +0.1298 | +0.0152 | 8.5× |
| `sel_ep` | +0.1288 | +0.0298 | 4.3× |
| `sel_ade_m` | −3.2637 | +1.0341 | 3.2× |

⭐⭐ **THIS IS A RESULT IN ITS OWN RIGHT AND IT WAS NOT PRE-REGISTERED: the matched-anchor IL form
plus Amendment A-1 did not merely save the fan — they STABILISED THE RIG by roughly an order of
magnitude.** ⇒ **the catastrophic floor belongs to the RELEASE configuration, not to L1**, and the
"every cross-arm number is inside the floor" reading above is **superseded for this package** while
remaining true of the sibling rig it was measured on.

### ⛔ The lever against its own floor, tiered honestly

| tier | metrics | floor ÷ lever |
|---|---|---|
| **ROBUST** | `fan_minade_m` 0.06× · `sel_ttc` 0.06× · `fan_nc_fail_frac` 0.08× · `fan_endpoint_spread_m` 0.13× · `sel_pdms` 0.32× | floor < ⅓ of the lever |
| ⚠️ **MARGINAL** | `fan_pdms_best` 0.36× · `sel_nc` 0.65× · **`fan_pdms_mean` 0.70×** · `sel_ep` 0.99× | floor is ⅓–1× the lever |
| ⛔ **INSIDE THE FLOOR** | **`sel_ade_m` 3.70×** | the floor **exceeds** the lever |

⛔⛔ **`sel_ade_m` +0.2798 [+0.1129, +0.4634] IS WITHDRAWN AS A LEVER CLAIM.** Its run-to-run floor
is **+1.0341**, **3.70×** the effect. The separated interval was real and answered the wrong
question. *(It remains a correct statement about these two particular arms; it is not a statement
about the RL term.)*

⚠️⚠️ **AND THE PRIMARY ENDPOINT ITSELF IS MARGINAL — I am not going to bank this SUCCESS quietly.**
`fan_pdms_mean` clears its own floor by only **1.4×** (lever +0.0363, floor +0.0253). ⭐ The two
seeds' `Δfan` are **+0.0363** and **+0.0616** — they **agree in sign**, which is the part that
matters, and they differ by **0.0253**, which is *exactly* the measured floor. ⛔ **§13.2's rule
requires both CIs' lower bounds > 0 and NO margin over the run-to-run floor**, so a lever 1.4× its
floor passes it. That is a **defect in the criterion**, not in the arms, and it is recorded here
rather than discovered later: a future endpoint on this rig should require a **stated multiple** of
a **measured** floor, the way D3's T-B bar already does (**≥ 3× the seed floor**).

### The collision finding, now on a FIFTH checkpoint

| | BASE | L1-NORL-s0 | L1-RL-s0 | **L1-RL-s1** |
|---|---|---|---|---|
| selected plan collides | 28 / 493 | 25 / 493 | 37 / 493 | **29 / 493** |
| …fan still held a collision-free candidate | 28 / 28 | 25 / 25 | 37 / 37 | **29 / 29** |
| picks the fan's best exactly | 48.7 % | 52.3 % | 35.3 % | **68.8 %** |
| pooled normalised skill | 0.7499 | 0.7652 | 0.5818 | **0.5954** |

⭐ **Five checkpoints, four training states, and in every collision window a collision-free plan was
in the fan — 119 for 119.** ⚠️ Note `L1-RL-s1` picks the fan's best on **68.8 %** of windows against
`L1-RL-s0`'s 35.3 % while their pooled skill is nearly the same (0.5954 vs 0.5818): the exact-pick
*rate* moves a lot between seeds, the *pooled* loss does not — another reason to read a rate against
its own floor before calling it an effect.

Banked: `raw/stageA_VERDICT.json`.

## ⛔⛔⛔ THE FLOOR, MEASURED FROM DUMPS THAT WERE ALREADY BANKED — READ THIS FIRST

The caveat at the top of this document was written as a rule. It is now a **number**, and it is
worse than the rule implied. Six held-out dumps — `heldout_base`, `heldout_base_repeat`, the two
2026-09-15 release-form arms, and this package's two — were verified to be on the **IDENTICAL 493
windows / 40 episode clusters** (checked on `(sha12, t0)`, all six sets equal), so every pairing
below is admissible on the paired estimator. **Zero GPU.**

### 1. The repeat read is bit-identical — but ⛔ IT IS A REPRODUCIBILITY CHECK, NOT AN INFERENCE FLOOR

The **same checkpoint scored twice** (`heldout_base` vs `heldout_base_repeat`) is **bit-identical on
493 of 493 windows**; all ten metrics read **+0.0000 with CI [0.0000, 0.0000]**.

⛔⛔ **MY FIRST READING OF THIS WAS WRONG AND IS WITHDRAWN.** I wrote that the DDIM path is
"deterministic end to end" and that `CLAUDE.md`'s third-variance caveat therefore "does not bind on
this rig". **It is bit-identical because the reader PINS THE SEED**, and the source says so in the
line itself:

```
inp, out = ctx.capture([it], seed=500_000 + k)      # PAIRED inference noise per window
```

⇒ the repeat answers *"is this pipeline reproducible?"* — **never** *"would another inference draw
say this?"*. **The third variance on this rig remains UNMEASURED**, and measuring it needs a
different sampler seed, which no banked dump has. ⇒ `CLAUDE.md`'s caveat — raised on refav1, whose
iCEM planner genuinely **samples**, with a measured seed floor of **≈0.30 m** ADE — **stands here
untouched**, neither confirmed nor narrowed.

⭐⭐ **AND IT IS WORSE THAN "UNMEASURED": IT HAS BEEN MEASURED ON THIS MODEL FAMILY, AND IT IS NOT
ZERO.** The **D3** package — same model family, same night — opens with *"§2. THE INFERENCE-SEED
FLOOR — measured first, because everything is read against it"* and registers a **minimum-time-gap
floor of 0.0054 s** [−0.0007, +0.0117] by **varying the inference seed**, with one quantity moving
**64 % of its own point estimate** across seeds. ⇒ the ddv2_rl held-out reader's pinned
`seed = 500_000 + k` does not make that variance vanish; it **hides** it. ⛔ So every **absolute**
number in this package carries a real, non-zero, unquantified inference-draw variance, and only the
**paired** deltas are protected — by the pinning, which is what it is for. ⭐ D3 also shows the
correct instrument already exists in the programme: vary the seed and measure the floor first. ⚠️ Same defect class as the oracle-gap retraction
logged the same night: **a measurement read as answering a question it does not answer.**

⭐ **What IS true, and it is more useful than what I claimed.** The seed is `500_000 + k` — a
function of the **window index only**, so *every arm sees the SAME inference noise on the SAME
window*. That is a deliberate pairing device, and it means **inference variance cancels inside every
paired cross-arm delta in this package by construction**. ⇒ the deltas in §2 and below are clean of
it; what they are **not** clean of is training variance, which is precisely §2's subject. ⛔ Any
**absolute** number here (`sel_pdms` = 0.9161, say) still carries an unmeasured inference-draw
variance, so it is a within-rig quantity and not a portable one.

### 2. ⛔⛔ The TRAINING floor is CATASTROPHIC — 10 of 10 metrics separated on a RUN-TO-RUN replicate

`heldout_arm-rl-s0` vs `heldout_arm-rl-s1` from 2026-09-15: **same flags, same rig, zero levers
moved** — differing in seed *and* in launch. ⚠️ **Read the section above before calling this a SEED
floor: launch alone changes the regime on this trainer, so this is a RUN-TO-RUN floor.**

| metric | **this package's LEVER** (RL − NORL) | **SEED ALONE** | floor ÷ lever |
|---|---|---|---|
| `sel_pdms` | −0.0449 | **+0.2431** | **5.41×** |
| `sel_ade_m` | +0.2798 | **−3.2637** | **11.66×** |
| `sel_ttc` | −0.0669 | +0.1866 | 2.79× |
| `sel_nc` | −0.0233 | +0.1298 | 5.57× |
| `sel_ep` | −0.0301 | +0.1288 | 4.28× |
| `fan_endpoint_spread_m` | −6.8289 | **−17.2200** | 2.52× |
| `fan_minade_m` | −0.0877 | −2.1319 | 24.31× |
| `fan_pdms_best` | +0.0011 | +0.1113 | **101.18×** |

**Every seed-only interval excludes zero. 10 of 10.**

⇒ ⛔ **EVERY CROSS-ARM NUMBER IN THIS PACKAGE IS SMALLER THAN THE SEED-ONLY EFFECT ON THE SIBLING
RIG, BY 2.5× TO 101×.** The nine "separated" rows are downgraded from *separated WORSE* to **inside
the only training-variance floor this rig family has measured**, pending `L1-RL-s1`.

⚠️ **Scope it honestly, because the scope is the only thing that keeps this from being fatal.** The
2026-09-15 arms ran the **RELEASE-form** lever (`--il-form release --grad-clip 0`) — the
configuration MEASURED to collapse the fan by **93 %**, and the reason L1 and Amendment A-1 exist at
all. A wildly unstable arm having a wild seed variance is **expected**, so this is an **UPPER bound
on instability**, not an estimate of L1's own floor. ⛔ But it is the **only** training-variance
measurement this rig family has, and "the stable configuration is probably much tighter" is a
**hypothesis**, not a floor. ⇒ **`L1-RL-s1` is not bookkeeping; it is the load-bearing measurement
of this entire package**, which is why it was relaunched unchanged rather than shrunk to fit.

⭐ **Nothing here is a retraction.** This document said from its first version that a separated
one-seed interval is *necessary, not sufficient*, and that no statement in it was a lever claim.
That hedge was correct; what has changed is that it now carries a number instead of a rule.

### 3. What SURVIVES the floor, and it is the more interesting half

⭐ The floor is a statement about **differences between two trained arms**. It does not touch a
reading of **one arm against a control computed on its own windows from its own fan** — and those
are exactly the findings below. Better: they now **reproduce across four checkpoints in three
different training states**:

| | `base` (untouched) | `base_repeat` | `L1-NORL-s0` | `L1-RL-s0` |
|---|---|---|---|---|
| windows where the **selected** plan collides | **28** / 493 | 28 / 493 | **25** / 493 | **37** / 493 |
| …of which the fan **still held a collision-free candidate** | **28 / 28** | 28 / 28 | **25 / 25** | **37 / 37** |
| picks the fan's best candidate exactly | 240 = **48.7 %** | 240 = 48.7 % | 258 = **52.3 %** | 174 = 35.3 % |
| pooled normalised skill (0 = random, 1 = oracle) | **0.7499** | 0.7499 | **0.7652** | 0.5818 |

⛔⛔ **THE COLLISION DEFECT IS NOT L1'S DOING — IT IS ALREADY IN THE UNTOUCHED CHECKPOINT.** 28 of
493 windows on a model that received **no RL, no L1, no extra training at all**, and in **28 of 28**
a collision-free candidate was sitting in the fan. ⇒ this is a **refcv5-v2 property**, i.e.
**programme-level**, and it is not attributable to the lever this package was testing.

### 4. And two more things the base control settles

* **`L1-NORL-s0` ≈ the base.** Only **4 of 10** metrics separated and **`sel_pdms` is NOT**
  (+0.0084 [−0.0059, +0.0255]). ⇒ 600 steps of matched-anchor IL barely move the model off its cold
  start, which is what a *control* arm should do and is worth having verified rather than assumed.
* **`L1-RL-s0` is separably worse than the base on 10 of 10** — `sel_pdms` **−0.0365**
  [−0.0615, −0.0161], `sel_ttc` −0.0487, `sel_nc` −0.0183. ⚠️ Read against §2's floor this is
  **suggestive, not established**; but it is anchored to a checkpoint that required **no training at
  all**, which is a cleaner anchor than an arm-to-arm difference.

Banked: `raw/noise_floors_and_base_control.json`.

## ⭐ The one-variable proof — MEASURED, not argued

A flag in `argv` is a claim about the **launch**. These are the trainer's own step-0 rows, which is a
claim about the **run**. Of **44** scalar fields logged at step 0, **35 are BIT-IDENTICAL** across
the two arms and **9 differ** — and every one of the 9 is either wall-clock or downstream of the RL
term:

| differing field | `L1-RL-s0` | `L1-NORL-s0` | why |
|---|---|---|---|
| `rl_coef_abs_sum` | 0.438634 | **0.0** | the lever itself |
| `frac_nonzero_adv_used` | 0.236111 | **0.0** | the lever itself |
| `rl_part` | +0.18136519 | **−2.2726e−10** | the lever itself (float noise in the control) |
| `loss` | 0.24668123 | 0.06531604 | IL + `rl_part` |
| `grad_norm` · `grad_norm_clipped` | 29.0603 | 3.0445 | downstream of `loss` |
| `param_delta_norm` | 0.010238 | 0.010108 | downstream of `grad_norm` |
| `fetch_s` · `wall_s` | 0.905 · 3.5 | 1.488 · 6.1 | wall-clock, not state |

⭐ **The decomposition closes to the last bit.** `norl_loss + rl_part_RL = 0.2466812345720652`
against `rl_loss = 0.24668123479932547` — a residual of **2.2726e−10**, which is *exactly* the
control's own `rl_part`. So `rl_loss − norl_loss = rl_part_RL − rl_part_NORL` identically: the RL
term is precisely the added component and nothing else moved.

⭐ **And everything upstream is bit-identical** — `il_matched_anchor_m` 0.6531603753566741,
`chain_endpoint_spread_m` 32.89187240600586, `reward_mean` 0.5934856534004211,
`frac_positive_after_bar` 0.01655982993543148, `human_pdms_mean` 1.0 and the four `cand_*_mean` —
which is what proves same seed, same data order, same init. This is the admissibility evidence for
the comparison; without it "one variable" would be an assertion about intent.

## The two arms that ran

Both 600/600 steps, `check_arm_l1.py` exit **0**, `problems: []` — I1/I3/I7–I10 all hold.
`train --arm rl --seed 0 --steps 600 --batch 4 --il-form matched --grad-clip 100` and the
same line with `--arm norl`, on the RTX 4060.

| instrument | **L1-RL-s0** | **L1-NORL-s0** (length-matched control) |
|---|---|---|
| rows · form · clip · release | 600 · `matched_anchor` · 100.0 · `is_release: false` | 600 · `matched_anchor` · 100.0 · `is_release: false` |
| reward first → last | 0.5935 → **0.8884** | 0.5935 → 0.8411 *(first identical — same seed)* |
| IL distance first → last | 0.6532 → **0.4476** m | 0.6532 → **0.4220** m |
| chain spread first → last | 32.89 → **24.24** m | 32.89 → **30.59** m |
| `grad_norm_max` · `steps_clipped` | **216.78** · **3 / 600** | 79.71 · **0 / 600** |
| `human_nc_eq_1_frac_pooled` | 1.0 | 1.0 |
| wall | 1,883 s (31.4 min) | 2,061 s |

⭐ **The clip behaved as Amendment A-1 predicted.** At 100 it bound **3 of 600** on the RL arm and
**0 of 600** on the control — a spike guard, not a rescale. A-1 was made because the original 1.0
would have bound **600/600**, an every-step 17–62× rescale rather than a divergence guard. These are
the first arms run since that amendment and both are consistent with the reason for it.

## Held-out T0 read — 493 windows, stride 10, the SAME windows for both arms

| metric | `L1-RL-s0` | `L1-NORL-s0` |
|---|---|---|
| `sel_pdms` — what selection picks | **0.8712** | **0.9161** |
| `human_pdms` | 0.9860 | 0.9860 |
| ⭐ `fan_pdms_best` — best candidate **present in the fan** | **0.9949** | **0.9938** |
| `fan_pdms_mean` | 0.6992 | 0.6629 |
| `sel_ade_m` | **2.5399** | **2.2601** |
| `sel_fde_m` | 7.5608 | 7.1233 |
| ⭐ `fan_minade_m` | **0.8175** | **0.9051** |
| `fan_endpoint_spread_m` | 28.53 | 35.36 |
| `fan_nc_fail_frac` | 0.1221 | 0.1506 |
| `sel_nc` · `sel_ttc` · `sel_ep` · `sel_comfort` | 0.9260 · 0.8418 · 0.8842 · 0.9980 | 0.9493 · 0.9087 · 0.9142 · 0.9980 |
| `traj_matches_fan_sel` FALSE | **0 of 493** | **0 of 493** |
| selection gap `fan_pdms_best − sel_pdms` | 0.1237 | 0.0777 |
| selection ratio `sel_ade / fan_minade` | **3.107×** | **2.497×** |

## Paired held-out deltas — ALL TWELVE metrics, none dropped

⛔ Reported complete rather than filtered: a table showing only the separated rows is a table chosen
after seeing the data.

| metric | RL − NORL | 95 % CI | |
|---|---|---|---|
| `sel_pdms` | **−0.0449** | [−0.0712, −0.0228] | **separated WORSE** |
| `sel_ade_m` | **+0.2798** | [+0.1129, +0.4634] | **separated WORSE** |
| `sel_fde_m` | +0.4374 | [−0.1667, +1.0849] | **not separated** |
| `sel_ttc` | −0.0669 | [−0.1041, −0.0331] | separated worse |
| `sel_nc` | −0.0233 | [−0.0450, −0.0064] | separated worse |
| `sel_ep` | −0.0301 | [−0.0509, −0.0134] | separated worse |
| `sel_comfort` | **0.0000** | [−0.0060, +0.0059] | **not separated** |
| **`fan_endpoint_spread_m`** | **−6.8289** | [−7.5891, −6.0245] | **separated NARROWER** |
| `fan_minade_m` | −0.0877 | [−0.1414, −0.0336] | separated *better* |
| `fan_pdms_mean` | +0.0363 | [+0.0243, +0.0475] | separated better |
| `fan_nc_fail_frac` | −0.0285 | [−0.0408, −0.0159] | separated better |
| ⭐ `fan_pdms_best` | **+0.0011** | [−0.0005, +0.0034] | **not separated** |

## ⭐ The shape of it, which is what matters

**The fan narrows and the selected plan gets worse, while the best candidate in the fan is
unchanged.** L1 is *mode-preserving* IL — preserving the fan is the entire point of the lever — and
on this arm the fan is **6.83 m narrower** than its control's.

⚠️ **Read the "better" rows honestly: they are all narrowing in disguise.** `fan_minade_m`,
`fan_pdms_mean` and `fan_nc_fail_frac` improve exactly as a fan concentrated toward the mode would:
pull the candidates in and the *average* candidate gets closer and safer while the *best* one does
not (`fan_pdms_best` **not separated**). Every metric that reads better is a mean over a tightened
distribution; every metric that reads worse is what the car would actually execute.

⭐ Three independent readings now point at **SELECTION**, not the generator:

1. **This arm, read against itself:** `fan_pdms_best` **0.9949** against `sel_pdms` **0.8712** — a
   gap of **0.1237** — and `sel_ade` **2.54 m** against `fan_minade` **0.82 m**, a **3.107×** ratio.
   ⭐ This needs no cross-arm comparison and no control: it is one arm read against itself. ⚠️ It
   holds in the control too (**0.0777**, **2.497×**), so it is a property of the architecture, not
   of the lever. ⛔ **But see the next section: "the selector does not pick the good plan" is the
   WRONG reading of this gap, and I wrote it before measuring the random-pick control.**
2. **D3, same model family, same night:** the planner **attends** to the lead (T-G 1.92×, separated
   at all four layers) yet greying the lead out moves the time gap by **0.09× the noise floor**. The
   read is there and unused.
3. **Here:** the RL term narrows the fan **without improving its best member**. It is removing
   options, not making better ones.

## ⛔⛔ CORRECTION — the selector is NOT failing to pick the good plan. The regret is a TAIL.

⚠️ **This section corrects the first version of this document, which read the
`fan_pdms_best` − `sel_pdms` gap as "the fan contains a near-human plan and the selector does not
pick it".** That sentence is an inference from a gap, with **no control**. The control was already
sitting in the banked rows and costs zero GPU: `fan_pdms_mean` is the expected score of a candidate
drawn **uniformly** from the fan, i.e. exactly what a selector that reads **nothing** would score.
MEASURED, paired episode-cluster bootstrap, n_boot 2000:

| | `L1-RL-s0` | `L1-NORL-s0` |
|---|---|---|
| pick-at-random control (`fan_pdms_mean`) | 0.6992 | 0.6629 |
| **the deployed selector** (`sel_pdms`) | **0.8712** | **0.9161** |
| oracle ceiling (`fan_pdms_best`) | 0.9949 | 0.9938 |
| selector **−** random | **+0.1720** [+0.1308, +0.2136] | **+0.2532** [+0.2119, +0.2964] |
| selector **−** oracle | −0.1237 [−0.1709, −0.0835] | −0.0777 [−0.1140, −0.0454] |
| normalised skill (0 = random, 1 = oracle) — mean · median | 0.6680 · **0.9475** | **0.8156** · **1.0000** |
| picks the fan's best candidate **exactly** | 174 / 493 = **35.3 %** | 258 / 493 = **52.3 %** |
| … within 0.01 of it | 229 / 493 = 46.5 % | 329 / 493 = **66.7 %** |
| picks **worse than chance** over its own fan | 63 / 493 = **12.8 %** | 34 / 493 = **6.9 %** |
| share of all regret carried by the worst 10 % of windows | **68.6 %** | **86.5 %** |

⭐ **On the control arm the selector picks the single best candidate in the fan on the MEDIAN
window** (normalised skill median **1.0000**, exact pick on **52.3 %**), and it beats the
random-pick control by **+0.2532** with the interval clear of zero. A selector that "does not pick
the good plan" cannot do that. ⇒ **the earlier sentence is withdrawn.**

⭐⭐ **What is true instead, and it points somewhere much cheaper: the loss is a TAIL.** On the
control, **86.5 % of all selection regret is carried by the worst 10 % of windows** — 49 of 493 —
while the other 444 are at or near the ceiling. That is not "rebuild selection"; that is **find what
those windows have in common**, which is a stratification question answerable on the banked rows at
zero GPU. ⚠️ **I then tested whether they ARE the same 49 across checkpoints, and they are only
partly — see "Is the tail a stable set of windows?" below, which corrects this sentence rather
than repeating it.**

⭐⭐ **And it sharpens the RL attribution to something the fan-width story could not say.** The RL
arm hands its selector a **strictly easier problem** — its random-pick control is **higher**
(0.6992 vs 0.6629, because the fan is tighter) and its oracle ceiling is **no lower** (0.9949 vs
0.9938) — and the selector still scores **worse** (0.8712 vs 0.9161). Exact picks fall by **84
windows (−17.0 pp)** and windows selected **worse than chance nearly double (34 → 63)**. ⇒ **the RL
term's damage is localised to SELECTION, not to the generator**, which no reading of
`fan_endpoint_spread_m` alone could establish.

⚠️ **Scope, honestly.** The *within-arm* rows (selector vs its own random control, exact-pick
counts, the regret tail) are absolute readings of one arm against a control computed on the **same
windows from the same fan**, so the one-seed caveat at the top does not touch them. The
*cross-arm* rows (−17.0 pp, 34 → 63) are RL-minus-NORL differences and are **subject to it in
full** — they await `L1-RL-s1` like every other cross-arm number here.

Banked: `raw/selection_skill_vs_random.json`.

## ⭐⭐ THE NEXT LEVER, MEASURED AND PRICED: 5 % of windows carry 62 % of the oracle gap

The tail above is not a mood, it is a **named sub-population**, and it is small. Stratifying the
worst 10 % against the other 444 windows — **reproduced independently on both arms**, which is the
only reason it is quoted at one seed:

| | tail (49) | rest (444) | ratio |
|---|---|---|---|
| `fan_nc_fail_frac` — share of the FAN that collides | **0.3851** | **0.1247** | **3.09×** |
| `fan_endpoint_spread_m` | 31.50 | 35.79 | 0.88× |
| `fan_pdms_best` — was a good plan present? | **0.9728** | 0.9961 | 0.98× |
| `human_pdms` | 0.9405 | 0.9910 | 0.95× |
| `t0` — position in the episode | 73.65 | 74.37 | 0.99× |

*(`L1-RL-s0` reads 0.3377 / 0.0983 = **3.44×**, spread 27.03 / 28.70, `fan_pdms_best` 0.9867,
`t0` 75.98 / 74.11 — same shape.)*

⭐ **It is the CROWDED population, and the fan is NARROWER there, not wider.** `fan_nc_fail_frac`
is a property of the **fan**, not of the selection, so unlike the sub-scores it is not definitional.
And `fan_pdms_best` stays at **0.9728** — a good plan was present even in the tail.

### Which sub-score discriminates — and which is structurally incapable of it

⛔ The tail is defined by `fan_pdms_best − sel_pdms`, and `sel_pdms` is a function of the four PDM
sub-scores, so *"the tail has low sub-scores"* is **partly definitional**. What is **not**
definitional is where each sub-score's failures **live**:

| sub-score | windows imperfect (of 493) | of which in the tail | |
|---|---|---|---|
| `sel_nc` | **25** | **25** | **100.0 %** |
| `sel_ttc` | **45** | **43** | **95.6 %** |
| `sel_ep` | **216** | 34 | **15.7 %** |
| `sel_comfort` | **1** | 1 | — |

⚠️ **`sel_comfort` is imperfect on ONE window in the whole corpus** (sd 0.045). Its absence from the
tail is **structural** — a sub-score that never varies cannot carry regret — so "the selector is not
trading safety for comfort" would have been a wrong reading, and is not made. ⭐ The real
discriminator is the contrast between `sel_ep` and `sel_nc`/`sel_ttc`: **ego-progress failures are
common (216/493 = 43.8 %) and mostly benign**, while **collision and TTC failures are rare and
almost entirely inside the tail**.

### ⛔⛔ And in EVERY collision window, a collision-free plan was sitting in the fan

| | `L1-NORL-s0` | `L1-RL-s0` |
|---|---|---|
| windows whose **selected** plan collides | **25** | **37** |
| …of which the fan still held a **collision-free** candidate | **25 / 25 = 100 %** | **37 / 37 = 100 %** |
| mean share of the fan that was collision-free there | **55.3 %** | **60.6 %** |
| mean `fan_pdms_best` available there | 0.9570 | 0.9824 |

**The selector picks a colliding plan out of a fan that is MAJORITY collision-free, in 100 % of the
cases where it collides.** That is not a hard-scene problem and not a generator problem.

### ⚠️ IS THE TAIL A STABLE SET OF WINDOWS? Enriched, but NOT a memorisable 49 — and I over-promised

Above I wrote that the next lever is *"find what those windows have in common"*. That presumes the
49 are **the same 49**. Tested across all six banked checkpoints, against the right control —
**chance overlap of two random 49-of-493 subsets is 4.87 windows**, not zero:

| | |
|---|---|
| pairwise tail overlap | **median 3.49× chance** (min 1.85×, max 6.78×) |
| windows in the tail of **all six** checkpoints | **1** (6-way chance ≈ 0.0005) |
| windows in **any** tail | **134 / 493 = 27.2 %** |
| windows where the selected plan **collides on all six** | **9** |
| …collides on **at least one** | **127 / 493 = 25.8 %** |

⚠️ **So the honest reading is in between, and my sentence was too strong.** A **3.49×** enrichment is
far above chance, so there **is** a real scene component — the tail is not noise. But **27.2 % of the
corpus enters some tail** and only **one** window is hard on every checkpoint. ⇒ *"find what those
windows have in common"* is **not** the well-posed target I implied; the tail is largely
**run-dependent**, which is exactly what the launch-nondeterminism section predicts.

⭐⭐ **AND THAT MAKES THE COLLISION-GATE ARGUMENT STRONGER, NOT WEAKER.** If the same 25 windows
failed every time, a memorised hard-case list could paper over it. They do not: a **different**
25–37 fail each run, drawn from a pool of **127**. ⇒ **the gate has to work generally**, which is
precisely what a perception channel buys and a lookup table does not.

⭐ **What IS a well-posed inspection target: the 9 windows that collide on ALL SIX checkpoints** —
stably hard across three training states and two lever families, and small enough to look at frame
by frame. That is the bounded next action, and it is the one I name instead.

Banked: `raw/regret_tail_stability.json`.

### ⛔⛔ AND THE DECISIVE QUALIFIER, WHICH I CHECKED BEFORE LETTING THE ABOVE PROPAGATE

**These arms have NO AGENT INPUT AT ALL.** Verified from the run record by three independent
signals, not asserted:

1. `argv` carries **`--agents off`**;
2. the base run's own directory is **`refcv5-v2-noagents-b1-v72-40k`**;
3. the config's `agent_join`, `agent_join_digest` and `agent_join_stats` are all **`None`**.

⛔ **And the REWARD does see them.** The proxy scores collision and TTC against agent tracks from
`b1eval_agents.jsonl.xz`, with `ttc_offsets [0, 3, 6, 9]`, and **172 windows were dropped for
missing agent data** — a drop that is only possible because the reward reads that file.

⇒ **The objective sees the other agents; the model does not.** So the correct reading is **NOT**
*"the selector ignores information it has"* — it is *"the selector is graded on a constraint it has
no structured channel for"*. ⚠️ It is not blind: the trunk sees the image, so other vehicles are in
the **pixels**. What is missing is their conversion into a collision-relevant representation — which
is precisely what **D3** measured from the other side the same night, and why the two readings agree.

⭐⭐ **THIS IS A NEW AND INDEPENDENT ARGUMENT FOR RE-OPENING THE AGENT SEAM, ON A LEDGER THE OLD
DECISION NEVER TOUCHED.** The agent channel was gated off because the **auxiliary agent task** cost
accuracy at two seeds on the tiny rig. That is a finding about a *training task*. This is a finding
about a *selection constraint*: the channel is worth a measured **62.4 % of the oracle gap**, and
`refc_agents.slot_features` — continuous metric range and bearing — **already exists** behind the
`--agents off` gate. ⛔ Re-opening it is still an ARM, not a recovery: its tiny-rig exclusion may or
may not generalise, and nothing here says the auxiliary task became free.

### The repair, priced before any GPU is spent on it

An **oracle-repair ceiling**: lift one named sub-population to the best candidate the fan actually
held, and ask what the corpus mean becomes. ⛔ A ceiling, never a result — no selector reaches an
oracle — and it is equally a basis for **refusing** a lever whose price is too small to matter.

| repair | n | Δ `sel_pdms` | 95 % CI | share of the full oracle gap |
|---|---|---|---|---|
| **collision windows only** | **25** | **+0.0485** | [+0.0228, +0.0801] | **62.4 %** |
| the whole worst-10 % tail | 49 | +0.0672 | [+0.0347, +0.1040] | 86.5 % |
| every window (full oracle) | 493 | +0.0777 | [+0.0454, +0.1140] | 100 % |

*(`L1-RL-s0`: **+0.0733** on 37 windows = **59.3 %** of its larger gap.)*

⭐⭐ **Five per cent of windows carry sixty-two per cent of the recoverable PDMS**, and the
intervention is a **hard constraint, not a learned weight**: never select a colliding candidate
while a collision-free one is in the fan.

⛔⛔ **THE CATCH, AND IT IS THE WHOLE ENGINEERING PROBLEM.** `sel_nc` is scored against the
**RECORDED future** — this is T0. So the table above says what a **perfect collision checker** would
have bought, **not** what the deployed policy can see at inference. A selection-time gate needs a
**predicted** occupancy, and under the vision-only rule it may not read the recorded future. ⇒ this
is not a free re-ranking; it is a **requirement on perception**.

⭐ **Which makes it a direct, quantified argument for the PI's refcv6 perception directive** — and, independently, for the agent seam above. The
BEV map head being wired right now is exactly the organ a collision gate would read, and this prices
what it is worth on the selection side: **62 % of the oracle gap**, concentrated in 5 % of windows,
on a rig where the fan already contains the right answer. ⚠️ And it lines up with D3 from the same
night — the planner **attends** to the lead and the read changes nothing — because the tail is
precisely the population where the lead is what matters.

Banked: `raw/selection_regret_strata.json`, `raw/selection_regret_subscores.json`,
`raw/selection_repair_ceiling.json`.

⚠️ **It is still T0** — deployed sampler, **recorded future**, **proxy reward** — so it is a
diagnostic and **never a driving number**. And `fan_pdms_best` is a **max over the fan**, which is
optimistic by construction: an oracle that picks the best of N is not a selector.

## ⛔ Why the replicate failed, twice, and what was NOT done about it

**CUDA out of memory** in `ctx.capture` — the rollout forward — at step 186, then 157, on an 8 GB
card, while another job held **24.85 GB RSS** (pinned-host allocation is where this surfaces). Arms
1 and 2 ran when the box was quiet.

⛔ **The batch was NOT reduced to get it through.** Batch is held constant across arms, so a
smaller-batch arm 3 would not be a replicate of arm 1 — it would be a third condition wearing a
replicate's name, and its "floor" would measure the batch change. It re-runs unchanged when the box
is quiet.

⚠️ **The first failure reported `exit 0`** because the command chain ended in `tail`, whose status
won — the `$?`-after-a-pipeline trap, hit again despite a standing memory about it. **The artifact
check caught what the exit code hid** (`rows 186 != 600`, no checkpoint). The re-run captured the
status directly and reported `ARM3_RC=1` honestly.

## ⛔⛔ What the failed arm established — and the half of it that is now REFUTED

`grad_norm_max` **11,511** with **44** steps over 100 in only 157 rows, and the log shows
`g 147.304->clip`. A-1 replaced a clip of 1.0 — which would have bound **600/600** — with 100,
predicting it would bind rarely on stable arms and often on a diverging one. MEASURED: **3/600** on
arm 1, **0/600** on arm 2, **44/157** on that failed launch. ⭐ **That prediction holds and A-1 is
validated**: the clip does discriminate a stable arm from a diverging one.

⛔⛔ **BUT I ALSO WROTE "SEED 1 IS THE DIVERGING SEED", AND THAT IS REFUTED BY THE RELAUNCH.** The
same command — same seed, same flags, same batch — relaunched on a quiet box does **not** diverge:
over the same first 157 steps it peaks at **203.6** against the failed launch's **11,511** (**56.5×**)
and binds the clip **2/157 = 1.3 %** against **44/157 = 28.0 %** (**22.0×**).

⭐⭐ **THE MECHANISM, AND IT IS A PROGRAMME-LEVEL FACT ABOUT THIS TRAINER: THE RUN IS NOT
REPRODUCIBLE ACROSS LAUNCHES, AND THE NONDETERMINISM AMPLIFIES.** Step 0 is identical on **42 of 44**
scalar fields (the two that differ are `fetch_s` and `wall_s` — wall-clock, not state). The two
launches first part company at **step 2**, by **2.86e−06** absolute and **3.03e−07** relative, and
from there:

| step | launch A `grad_norm` | launch B `grad_norm` | relative divergence |
|---|---|---|---|
| 0 · 1 | 40.887421 · 21.098879 | 40.887421 · 21.098879 | **0** (bit-identical) |
| 2 | 9.455194 | 9.455192 | 3.03e−07 |
| 10 | 15.317718 | 15.317820 | 6.66e−06 |
| 40 | 6.361910 | 6.348524 | 2.10e−03 |
| 60 | 6.412241 | 4.751979 | 2.59e−01 |
| 80 | 4.470911 | 20.014410 | **3.48** |
| 120 | **11511.416016** | 14.693929 | 9.99e−01 |

**Relative divergence grows from 3e−07 to O(1) in eighty steps** — about one order of magnitude
every ten steps — so a float-level reduction-order difference decides whether the arm diverges.

⇒ ⛔ **A SAME-SEED RELAUNCH ON THIS RIG IS A FRESH DRAW, NOT A REPRODUCTION.** Three consequences,
and they touch this whole package:

1. **The arm 3 now running is not "the diverging seed, retried".** It is a second draw that landed
   in the stable regime. It is a perfectly good training-variance replicate — which is what §13.2
   asks for — but it reproduces nothing.
2. ⚠️ **The floor in §2 is mis-attributed and I am correcting it here.** I called it a "SEED-ONLY"
   floor. The two 2026-09-15 arms differ in seed **and** in launch, and this shows **launch alone
   changes the regime**. It is a **RUN-TO-RUN** floor. The magnitude is unchanged; the cause is not
   the seed.
3. ⛔ **This is the third time tonight that a measurement was read as answering a question it does
   not answer** — after the oracle gap and the pinned inference seed. All three were caught, each
   one faster than the last, and this one before it left this document.

### ⛔ The CAUSE is NOT measured, and I am not going to assert one

⚠️ It would be easy to write *"TF32 / cuDNN autotune, and `strict_numerics()` is the fix"* — the
repo even invites it. `stack/tanitad/instruments/numerics.py` diagnosed **this exact mechanism** in
July 2026 (*"TF32/cuDNN selecting different kernels (different reduction orders, ~1e-3
precision)"*), ships `strict_numerics()`, and states the doctrine: *"every probe fit, every gate
evaluation … run inside `strict_numerics()`. **Training keeps fast kernels.**"* MEASURED today:
that context manager is used in **22** files — **every one a measurement path** — and **0** times in
`ddv2_rl_refcv5.py` (same-breath control: the file's 17 imports read fine). And across the whole
stack, `use_deterministic_algorithms` / `cudnn.deterministic` / `CUBLAS_WORKSPACE_CONFIG` appear
**0** times, against **1,089** `manual_seed` sites — *this programme seeds heavily and has never
made a training run reproducible.*

⛔ **But TF32 is deterministic-but-imprecise, not nondeterministic**, so it cannot by itself explain
two launches differing at step 2. The standard culprit is **non-deterministic atomic reductions in
backward**, which `strict_numerics()` does **not** address and
`torch.use_deterministic_algorithms(True)` does. ⇒ **naming `strict_numerics()` as the fix would be
the same error class this document has already retracted three times tonight: asserting a mechanism
I have not measured.**

⭐ **So the experiment is written instead of the claim.** `code/determinism_probe.sh` +
`code/det_wrap.py`, ~10 min of GPU, queued behind the replicate:

1. **Name the op** — one 3-step run under `use_deterministic_algorithms(True, warn_only=False)`, so
   the first offending kernel **raises and names itself**. That converts *"something amplifies"* into
   *"this op is the source"* in a single short run.
2. **Two launches as-is** and **two under determinism**, 20 steps each. ⭐ The readout is
   **bit-identity of `grad_norm`**, never a metric: under one condition two launches either agree to
   the last bit or they do not.

⛔ `det_wrap.py` sets the flags in a **parent process** via `runpy` — the trainer is read, never
written — because it was the load-bearing arm still running when this was designed, and this repo's
rule is *never edit a running script*. ⚠️ It also **refuses** unless `CUBLAS_WORKSPACE_CONFIG` is
already in the environment, since that must precede CUDA initialisation and a silent miss would make
the determinism arm quietly fail open.

⚠️ **And whatever it finds is an ARM, not a patch.** Changing the trainer's numerics changes every
future arm's comparability with the banked ones, and determinism costs throughput. That is a
decision, not a fix to apply on my own authority.

⚠️ **What is NOT measured:** launch A died at step 156 with no checkpoint, so the **held-out**
consequence of its divergence is unknown. The 56.5× and 22.0× are training-trajectory quantities,
not metric quantities. Banked: `raw/launch_nondeterminism.json`.

## Budget

≈1.82 GPU-h spent (2 arms + 2 held-out reads + 2 failed attempts). A completed arm 3 plus its
held-out read adds ≈0.73 h → **≈2.55 h**, above the pre-registered **≤ 2.45 h** condition for
Stage B. ⇒ **Stage B is recorded NOT RUN**, never "unnecessary".

## Where the artifacts live

⚠️ **ONE PLACE:** `devbox:C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917/` —

| | |
|---|---|
| `l1-rl-s0/` | `ckpt.pt` **1.3 GB**, `metrics.jsonl` (600 rows), `config.json`, `run.json` |
| `l1-norl-s0/` | `ckpt.pt` **1.3 GB**, `metrics.jsonl` (600 rows), `config.json`, `run.json` |
| `heldout_l1-rl-s0.json` | 493 rows, 211 KB (+ its `.log`) |
| `heldout_l1-norl-s0.json` | 493 rows, 210 KB (+ its `.log`) |
| `l1-rl-s1/` | ⛔ no checkpoint — both attempts died in the rollout forward; `l1-rl-s1.log` kept |

The two run directories in full are
`devbox:C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917/l1-rl-s0/` and
`devbox:C:/Users/Admin/tanitad-caches/ddv2rl-l1-20260917/l1-norl-s0/`.

Too large for the repo; `raw/` here carries every number that has been quoted, and the held-out rows
are keyed by **`sha12`**, never by clip id.

## Next

⭐ **The named next lever is above and needs no GPU to specify: a collision gate on selection, worth a measured 62.4 % of the oracle gap, blocked only on a predicted occupancy the perception wiring is being built for.**

`L1-RL-s1` re-runs **unchanged** — same batch, same flags, same seed 1 — once the box is quiet; it
is the only thing between this package and a lever claim. **Stage B is NOT RUN** under the
pre-registered ≤ 2.45 h condition, and that is recorded as a budget outcome, never as
"unnecessary".
