# The rate is retired by MEASUREMENT, a statistic that CAN see distance-keeping exists, and refcv4b's fan bank is closed at FINAL

**date:** 2026-09-06 · **stream:** RL gate (Arch + Inference FlyWheel) ·
**branch:** `agent/arch-inf-20260803` · **pre-registrations:**
`PREREG_STATISTIC_RANKING.md` (commit `d9fb892`) and `PREREG_H-RL-GATE-STAT-1.md`
(same commit), `PREREG_H-RL-GATE-STAT-2.md` (commit `63e2609`) — **all committed BEFORE
the numbers they judge.**

---

## 0. The answer in one paragraph

The Master Mind retired `G-REWARD`'s RATE on **instrument** grounds and bound every
replacement to a mutation test run **before** any scoring. ⭐ **The mutation test was run on
all nine candidates first, and it does not merely rank them — it convicts the rate twice.**
Nothing was bit-identical, so the strict criterion disqualified nobody; but the
pre-registered **response-to-noise ratio** sorts the field **exactly by family** — MASS
1.13–1.21, ORDER 0.52–0.96, RANK 0.50, SIGN 0.23–0.27 — and puts the retired rate **last at
0.23**, i.e. **its response to a real 5 m move of the lead is under a quarter of its own
episode noise**. ⛔ **And it responds BACKWARDS**: moving the lead away, which can only
relax distance-keeping, pays the constant-velocity path **more** by every mass statistic
(`S6_wmr` **+0.0739**) while the rate moves **−0.0120** and Cliff's delta **−0.0240**. The
survivor `S6_wmr` — the share of the total gap mass paid to the trivial path, on the rate's
own [0, 1] scale — was pre-registered with `G-REWARD`'s own **0.30** transposed unchanged,
and scored on the held-out half of the episodes. ⛔ **`H-RL-GATE-STAT-1` FAILED: 0.301183,
CI [0.201040, 0.433903].** It misses by **0.0012**. Per Rule Zero I did not stop there: the
next lever was pre-registered and run in the same turn — `H-RL-GATE-STAT-2`, the same
statistic and the same bar on **the reward the code actually ships**. Separately: ⭐ **the
refcv4b FINAL checkpoint is on the box, md5-verified on both ends, and it cost the training
A40 nothing measurable** (3.9375 s/step before, **3.9220 during**, 3.9450 after) — **B6 is
closed at step 40,284, not 9,500.**

⛔ **`G-REWARD` REMAINS FAILED ON ITS COMMITTED STATISTIC (THE RATE).** Nothing here
re-scores it, and neither new hypothesis may be presented as `G-REWARD` repaired.

---

## 1. ⭐⭐ THE MUTATION RANKING — run FIRST, on every candidate, before any was scored

### 1.1 The mutation, and why it is literal rather than analogous

`PREREG_STATISTIC_RANKING.md` fixed it before any number existed:

```
PERTURB(d):  lead_track[i, 0] += d   for i in {1, 2, 3}
             lead_track[0] and lead_track[4] held EXACTLY fixed
```

The panel's lead track is exactly `[5, 2]` — the reward prefix `GRID_S = (0.0, 0.5, 1.0,
1.5, 2.0) s` — so **"three of five lead samples move by 5 m" is the same perturbation the
unit test used to disqualify `_headway`'s `amin`**, applied at panel scale rather than on a
fixture. `_headway` scores steps 1–4, so it moves three of the four scored steps and pins
the fourth: precisely the configuration in which a minimum can be blind.

### 1.2 ⛔ CONTROLS — every one reads its known value

MEASURED, SELECT split, 3,672 windows / 40 episodes:

| control | reading | verdict |
|---|---|---|
| **NULL** — the clone-and-add path with `d = 0.0` must reproduce the untouched cell | max\|Δcomponent\| **0.000e+00** | ⭐ PASS |
| **CAP INERTNESS** — with `cap="lead"` (so `progress` really does read the lead), the pinned endpoint must leave `progress` bit-identical | max\|Δprogress\| **0.000e+00** | ⭐ PASS |
| ⭐ **CAP DISCRIMINATING** — the same edit **plus the endpoint** must MOVE `progress`, or the line above is vacuous | max\|Δprogress\| **0.496361**, moved on **35.78 %** of windows | ⭐ PASS |
| **TERM BLINDNESS**, `headway_reduce="min"` | `headway` **bit-identical on 66.91 %** of `hold_v0` windows and **65.66 %** of `human` windows; composed gap bit-identical on **53.81 %** | ⭐ the unit test **GENERALISES** |
| the same under `q0.25` | 30.04 % / 29.71 %, composed gap 27.78 % | the quantile more than halves the blindness |
| **CHANNEL ISOLATION** — which components move at all | `headway` **0.6996**; `progress`, `collision`, `comfort`, `feasibility` **all exactly 0.0000** | ⭐ a clean single-channel intervention |

⭐ **The fourth row is the ruling's premise, measured rather than inherited:** on real
windows `amin` is blind to a 5 m move of the lead **two thirds of the time**.

### 1.3 ⛔ THE RANKING — the disqualification criterion, and what it actually caught

MEASURED, SELECT split (3,672 windows / 40 episodes), `headway_reduce="q0.25"` so each
candidate is judged on **its own** blindness and not the term's, episode-cluster bootstrap
`n_boot` 4,000:

| rank | candidate | family | value | SD_boot | ΔS(5 m) | ΔS(15 m) | **RNR** | verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | `S9_skewsplit` | MIXED | 0.007055 | 0.003089 | −0.004835 | −0.007500 | **1.57** | survives — ⛔ a **DIAGNOSTIC** by the candidate table, not selectable |
| **2** | ⭐ **`S6_wmr`** | **MASS** | 0.359618 | 0.061102 | **+0.073936** | **+0.104582** | **1.21** | ⭐ **top PREFERENCE statistic** |
| 3 | `S8_logmassratio` | MASS | −0.577022 | 0.268665 | +0.309657 | +0.433576 | 1.15 | survives |
| 4 | `S5_mean` | MASS | −0.007880 | 0.003589 | +0.004045 | +0.005745 | 1.13 | survives |
| 5 | `S4_q75` | ORDER | 0.005475 | 0.001676 | +0.001604 | +0.002396 | 0.96 | UNDERPOWERED |
| 6 | `S3_median` | ORDER | −0.000825 | 0.001506 | −0.000790 | −0.001755 | 0.52 | UNDERPOWERED |
| 7 | `S7_signedrank` | RANK | 0.402786 | 0.046742 | +0.023276 | +0.024077 | 0.50 | UNDERPOWERED |
| 8 | `S2_cliffs` | SIGN | −0.105664 | 0.090258 | −0.023965 | −0.058279 | 0.27 | UNDERPOWERED |
| 9 | ⛔ **`S1_rate`** — the retired statistic | SIGN | 0.482843 | 0.051525 | **−0.011983** | −0.029139 | **0.23** | **UNDERPOWERED and WRONG-SIGNED** |

**NULL passes and monotonicity holds for all nine** (`|ΔS(15)| ≥ |ΔS(5)|`, 9/9).

⭐ **Three readings, and the second is new:**

1. ⛔ **The strict criterion disqualified nobody.** No candidate was bit-identical at 5 m.
   Reported as it came out: the ruling's bit-identity test is *necessary* and, at panel
   scale, **not selective** — it is the pre-registered **RNR** that separates the field.
2. ⭐⭐ **The ordering is BY FAMILY, which is the mechanism showing through:** MASS
   (1.13–1.21) > ORDER (0.52–0.96) > RANK (0.50) > SIGN (0.23–0.27). A statistic that
   discards magnitude cannot see a magnitude change — and the arithmetic says so without
   being told.
3. ⛔⛔ **THE RATE'S RESPONSE HAS THE WRONG SIGN.** Moving the lead **away** relaxes
   distance-keeping, so the constant-velocity path must be paid **more**. Every MASS
   statistic agrees (`S6_wmr` +0.0739, `S5_mean` +0.0040, `S8` +0.3097). ⛔ `S1_rate`
   moves **−0.0120** and `S2_cliffs` **−0.0240**: they say `hold_v0` wins **less often**
   when the reward is paying it **more**. ⇒ **This is the rate/mean divergence produced by
   a CONTROLLED INTERVENTION rather than observed in a population** — the strongest form
   of the finding this package has reached, and it is independent of any score.

### 1.4 ⭐ A fourth, independent instrument reading — the conflict ladder runs OPPOSITE ways

On the same SELECT rows, unperturbed:

| rung | all | ≤5.0 s | ≤4.0 s | ≤3.0 s | ≤2.5 s | ≤2.0 s | ≤1.5 s |
|---|---|---|---|---|---|---|---|
| `S1_rate` | 0.4828 | 0.4845 | 0.5073 | 0.5229 | 0.5287 | **0.5990** | 0.5971 |
| ⭐ `S6_wmr` | 0.3596 | 0.4288 | 0.4314 | 0.3592 | 0.2993 | **0.2208** | 0.1670 |

⇒ **As the scene tightens the rate RISES and the win-magnitude ratio FALLS.** The two read
the same rows in opposite directions along the axis the gate exists to measure. ⛔ The
`<=2.0 s` cell is quoted here as an INSTRUMENT observation and **is not, and may not become,
a rung the bar is moved to** — `PREREG_H-RL-GATE-STAT-1.md` §3 forbids it in advance.

---

## 2. ⛔ `H-RL-GATE-STAT-1` — pre-registered, scored on the held-out half, and FAILED

### 2.1 What was committed before the split was read

`S6_wmr <= 0.30` at the **`all`** rung on cell **`nopert_min`**, whole-interval. The bar is
`G-REWARD`'s own committed 0.30 **transposed** — same [0, 1] scale, same 0.5-neutral
reading, same direction, no tuning. The rung is `G-REWARD`'s own headline rung. The cell is
the **bit-identical pre-repair reward**, so exactly one thing differs from `G-REWARD`: the
statistic. ⭐ **The primary cell had not been computed on either split when the bar was
written** — the ranking ran on the `q0.25` cell.

### 2.2 ⛔ CONTROLS

| control | reading | verdict |
|---|---|---|
| ⭐ **CHANNEL — the cell must BE the cell `G-REWARD` scored.** `S1_rate` on `nopert_min` over all 6,089 windows | **0.441780259484316** vs banked **0.441780259484316**, err **0.000e+00** | ⭐ PASS, exactly |
| **SPLIT DISJOINTNESS** | SELECT 3,672 / SCORE 2,417, **overlap 0**, union == panel | ⭐ PASS |
| **FROZEN** stays far below the human on the scored split | **0.003724** | PASS |

⚠️ **The banked composed MEAN (`−0.011239379601720929`) is a STALE baseline and is
deliberately NOT used as a control** — `rewards._collision` changed after the 2026-09-05
bank (13 of 18,267 evaluations, 3 on `hold_v0`, `3/6089 = 4.927e-04` exactly). The RATE is
unaffected and is the control. ⛔ Any register or report row quoting that mean as a
*current* value carries this correction.

### 2.3 MEASURED — SCORE split, 2,417 windows / 33 episodes

| rung | n win | n eps | `S6_wmr` | episode-cluster CI |
|---|---|---|---|---|
| **all** (pre-registered) | 2,417 | 33 | **0.301183** | **[0.201040, 0.433903]** |
| ≤5.0 s | 1,996 | 27 | 0.351737 | [0.228517, 0.525667] |
| ≤4.0 s | 1,701 | 26 | 0.389164 | [0.238442, 0.601778] |
| ≤3.0 s | 1,178 | 23 | 0.325982 | [0.172073, 0.613398] |
| ≤2.5 s | 1,054 | 16 | 0.292101 | [0.138439, 0.658354] |
| ≤2.0 s | 758 | 10 | 0.272110 | [0.102518, 0.763938] |
| ≤1.5 s | 348 | 8 | 0.224002 | [0.074838, 0.821589] |
| ≤1.0 s | 84 | 3 | 0.103509 | [0.000000, 0.977474] |

> ⛔ **VERDICT: `H-RL-GATE-STAT-1` FAILS.** `0.301183` against `<= 0.30` — on the point
> estimate **and** on the whole interval. **Reported as written.**

⚠️ **It misses by 0.001183 — 0.4 % of the bar** — where the retired rate reads **0.4418**
on the same rows. That is not a pass and is not offered as one; it is the size of the gap
the reward still has to close. ⚠️ The sub-rungs fall below 0.30 but their intervals widen
to uselessness (10 episodes at ≤2.0 s), so **none of them clears anything**, and none was
selectable.

---

## 3. ⭐ THE NEXT LEVER, PRE-REGISTERED AND RUN IN THE SAME TURN — `H-RL-GATE-STAT-2`

### 3.1 Why it is the next lever and not a goalpost move

`H-RL-GATE-STAT-1` deliberately held the REWARD at the pre-repair configuration so that
exactly one thing differed from `G-REWARD`. The question it could not ask is the one the
programme needs: **does the reward the code SHIPS today clear the same bar?** Since
2026-09-06 the shipped default is `progress_lead_cap = "achievable"` (`_progress_cap_mode`
maps `True` to `"achievable"`) with `headway_reduce = "min"` — a combination **never
computed in this package**.

⛔ **The bar, the rung and the statistic are all unchanged.** `PREREG_H-RL-GATE-STAT-2.md`
(commit `63e2609`, blob `74ce9c05a…` verified in `HEAD`, md5
`85f49f8bfd67a50145ce9c7bc56fde68`, written 14:37:40 local) states in its own text that it
was written **expecting to fail** — the SELECT split reads 0.3596 on the `q0.25` cell and
the held-out pre-repair cell read 0.3012, both above 0.30.

### 3.2 ⛔ CONTROLS

| control | reading | verdict |
|---|---|---|
| ⭐ **RANKING REPLICATION** — the whole mutation ranking recomputed from scratch on the enlarged pass | **every value identical**: `S1_rate` RNR **0.23**, `S6_wmr` **1.21**, all nine ΔS and SD_boot to the digit | ⭐ PASS |
| ⭐ **CHANNEL**, pinned to `nopert_min` | **0.441780259484316** vs banked, err **0.000e+00** | ⭐ PASS |
| **CAP BINDS** — `ship` must differ from `nopert_min` | `progress` differs on **38.18 %** of `hold_v0` windows and **73.28 %** of `human` windows, max abs Δ **0.915774** | ⭐ PASS — and the asymmetry is the finding |
| **SPLIT DISJOINTNESS** | 3,672 / 2,417, overlap **0**, union == panel | ⭐ PASS |
| **FROZEN** | 0.112536 under `ship` (0.003724 uncapped) — see §3.5 | reported |

⚠️ **A DEFECT IN MY OWN CONTROL, FOUND BY RUNNING IT AND FIXED.** The first version
anchored the CHANNEL control to *the scored cell*, so on `ship` it compared a different
reward's rate against the banked one and read `err 1.322e-01 pass=False` on a perfectly
healthy panel. The banked 0.441780 was measured on the **pre-repair** cell, so the control
must be **pinned there** whatever is being scored. Fixed, re-run, and the scored cell's own
retired rate is now reported beside it as a diagnostic. ⭐ **That diagnostic is itself a
cross-check that lands: `S1_rate` on `ship` reads 0.573985876170143, replicating the
predecessor's independently-computed `achievable` all-window rate of 0.5740** — different
tool, different rows file, same windows.

### 3.3 MEASURED — SCORE split, 2,417 windows / 33 episodes

| rung | n win | n eps | `S6_wmr` | episode-cluster CI |
|---|---|---|---|---|
| **all** (pre-registered) | 2,417 | 33 | ⛔ **0.456155** | ⛔ **[0.309354, 0.630646]** |
| ≤5.0 s | 1,996 | 27 | 0.474631 | [0.298503, 0.714519] |
| ≤4.0 s | 1,701 | 26 | 0.460198 | [0.276545, 0.710059] |
| ≤3.0 s | 1,178 | 23 | 0.360732 | [0.170054, 0.653922] |
| ≤2.5 s | 1,054 | 16 | 0.309131 | [0.114038, 0.674906] |
| ≤2.0 s | 758 | 10 | 0.241782 | [0.057964, 0.704675] |
| ≤1.5 s | 348 | 8 | 0.105805 | [0.027952, 0.507479] |
| ≤1.0 s | 84 | 3 | 0.000545 | [0.000000, 0.241740] |

> ⛔ **VERDICT: `H-RL-GATE-STAT-2` FAILS, and not narrowly.** `0.456155` against `<= 0.30`,
> with the **whole interval above the bar** — the only cell in this package of which that
> is true. **Reported as written.**

### 3.4 ⭐⭐ AND THE FAILURE LOCALISES THE DEFECT — the SHIPPED cap is the worst cell

All five reward configurations, same held-out windows, same statistic:

| cell | reward configuration | `S6_wmr` | CI | retired `S1_rate` | mean gap | `frozen>=human` |
|---|---|---|---|---|---|---|
| `nopert_min` | pre-repair (**`H-RL-GATE-STAT-1`**) | 0.301183 | [0.201040, 0.433903] | 0.431940 | −0.013220 | 0.003724 |
| ⛔ **`ship`** | **shipped default** (`achievable` + `min`) | ⛔ **0.456155** | ⛔ **[0.309354, 0.630646]** | 0.545718 | **−0.002561** | 0.112536 |
| `lead_min` | `lead` cap + `min` | 0.302476 | [0.202475, 0.430071] | 0.396359 | −0.014723 | 0.112536 |
| `nopert` | `hq25` (off + `q0.25`) | **0.295613** | [0.188218, 0.432859] | 0.437733 | −0.013316 | 0.003724 |
| `both_lq` | `lead` + `q0.25` | 0.296900 | [0.195242, 0.424024] | 0.398428 | −0.014819 | 0.107158 |

**Paired episode-cluster bootstrap on the same windows, `n_boot` 4,000:**

| comparison | Δ `S6_wmr` | CI | separated? |
|---|---|---|---|
| ⛔ **`ship` − `lead_min`** | **+0.153679** | **[+0.084053, +0.254735]** | ⛔ **YES** |
| ⛔ **`ship` − `nopert_min`** | **+0.154972** | **[+0.072454, +0.270546]** | ⛔ **YES** |
| `nopert` − `nopert_min` (the headway quantile alone) | −0.005570 | [−0.042898, +0.030698] | no |

⭐ **Per-term attribution, and it is unambiguous.** Weighted mean gap `hold_v0 − human`:

| cell | `progress` | `headway` | `collision` |
|---|---|---|---|
| `nopert_min` | **−0.009769** | −0.000141 | −0.003310 |
| ⛔ `ship` | ⛔ **+0.000890** | −0.000141 | −0.003310 |
| `lead_min` | **−0.011272** | −0.000141 | −0.003310 |

`headway` and `collision` are **identical to the digit** across all three. ⇒ ⛔⛔ **The
entire +0.1547 is `progress`, and the `achievable` cap FLIPS it from paying the human to
paying the constant-velocity path.**

⚠️ **Estimator honesty, and here it is unusually clean.** These paired intervals answer
*"would another draw of EPISODES say this?"* — and on this panel that is the **only**
question there is. `H-ESTIM-SEED-1`'s training variance and the inference-seed variance do
not enter, because **no arm is trained and no planner samples**: the five cells are five
deterministic arithmetic functions of one fixed geometry. ⛔ Tier **T0**, non-parity
RL-fit windows, **not a driving number**.

### 3.5 ⭐⭐ THE MECHANISM — proved on a fixture, in six lines

⭐ The predecessor **self-disclosed** this in their own pre-registration: *"the
pre-registered `min(ref_free, ref_lead)` carries a SECOND cap at the free reference that
was not the stated intent"*. **This turn prices it, and pins why it is asymmetric.**

`ref_free = max(v0 · H, min_ref)`. The second cap therefore truncates exactly the
candidates that travel **further than `v0 · H`** — and **`hold_v0` sits at exactly `v0` by
construction, so it is never truncated**, while the human, whenever she accelerates, is.
MEASURED on a fixture with the lead 400 m away (`test_rl_gate_statistic.py`, 5/5 green):

* a faster-than-`v0` candidate: `lead` cap **bit-identical to off**, `achievable`
  **strictly lower**;
* a candidate at exactly `v0`: `achievable` **bit-identical to off**;
* ⭐ discriminating control: bring the lead to 30 m and the `lead` cap **does** bite, so
  its inertness above is a property of the distance and not of the mode.

⇒ ⛔ **The shipped cap removes the HUMAN's headroom and none of the trivial path's.** That
is why `progress` differs from the uncapped term on **73.28 %** of `human` windows against
**38.18 %** of `hold_v0` windows, and it is the whole of the +0.1547.

⚠️ **A second caution, and it attaches to BOTH cap variants — I nearly mis-assigned it.**
`frozen >= human` rises 0.003724 to **0.112536** under `achievable` *and* under `lead`
(0.107158 with the quantile). ⇒ this is a property of capping `progress` **at all**, not of
the second cap: a plan that never moves beats the human on ~11 % of windows under any cap,
against 0.4 % uncapped. Pinned by its own test.

### 3.6 ⛔ WHAT MAY NOT BE CONCLUDED FROM §3.4

⛔ **`nopert` (0.295613) is NOT a pass and may never be quoted as one.** Its point estimate
is below 0.30 but **its interval straddles the bar** (whole-interval discipline), and it was
**not the pre-registered cell** of either hypothesis. Reported as SECONDARY, exactly as
`PREREG_H-RL-GATE-STAT-1.md` §5.4 requires, so a reader can see what a cell choice would
have bought — and see that it was not taken.
⛔ **A third hypothesis picking the best cell after seeing this table is inadmissible**, and
`PREREG_H-RL-GATE-STAT-2.md` §3 named that in advance so it could not be attempted quietly.

---

## 4. ⭐ B6 CLOSED AT refcv4b FINAL — and the training A40 paid nothing

### 4.1 ⛔ The read-only pull, with its harm proof

⛔ The A40 is running refcv5's ~44 h training (`D-REFCV5-LAUNCH-1`, ETA ≈ 2026-09-08 06:20
UTC). A checkpoint copy is disk + network, not GPU — but *"prove you did no harm"*:

| | steps | n windows | mean s/step | range |
|---|---|---|---|---|
| **BEFORE the copy** | 50 → 1,650 | 29 | **3.9375** | [3.9240, 3.9560] |
| ⭐ **DURING** (the window containing 12:05:51–12:07:36 UTC) | 1,650 → 1,700 | 1 | **3.9220** | — |
| **AFTER** | 1,700 → 2,250 | 10 | **3.9450** | [3.9300, 3.9580] |

⇒ ⭐ **The window containing the copy is the FASTEST of the entire run** — 0.0155 s/step
below the pre-copy mean and below its minimum — and the post-copy mean sits inside the
pre-copy range (+0.19 %). **No measurable degradation.**

⚠️ **Method, because the naive reading is wrong here.** This trainer logs no `step_s` and
`elapsed_s` is CUMULATIVE, so rates are differenced between logged rows. The eval at every
500th step is charged into the window **after** the boundary (500→550 reads 5.196, 1000→1050
5.024, 1500→1550 5.182), so **those three windows are excluded** from all three means; a
rate computed over an eval-spanning window reads 4.2495 and looks like a 7.9 % regression
that never happened.

**The transfer, verified on both ends:**

| | |
|---|---|
| source | `tanitad-a40:/workspace/experiments/refcv4b-b1-v72-40k/ckpt_40284_FINAL.pt` |
| **pod md5** | **`99b573e8277d94a5e3bfbf630cb4d751`** — and it matches `MODEL_REGISTRY.md` §4.6 |
| **local md5** | **`99b573e8277d94a5e3bfbf630cb4d751`** — **MATCH** |
| size, both ends | **1,285,301,425 B** |
| wall clock | 105 s (~12.2 MB/s), read-only, nothing written to any pod path |
| the run's `config.json` | md5 `58ad809aaf241729c1695bec2ab30d8e`, **byte-identical** to the one the step-9,500 bank used |

⛔ Nothing was written to the pod. No `git fetch`. No process killed. Every command used
`ssh -n`.

### 4.2 MEASURED — the FINAL fan bank, against step 9,500

`fan_bank_refcv4b_FINAL_240w.npz` — **240 windows × 117 candidates × 5 slots, 103 episodes,
69 with a lead**, `dt` 0.5 s, 71.1 s on the dev-box RTX 4060. ⭐ **The same 240 windows and
the same 103 episodes as the 9,500 bank**, so this is a matched-window comparison in which
only the checkpoint moved.

| metric (`--quality composed`, T0/fan-level) | **@9,500** | **@FINAL (40,284)** | Δ | `rand_best@k` @FINAL |
|---|---|---|---|---|
| `fan_floor@1` | 0.589504 | 0.587214 | −0.002289 | 0.462461 |
| `fan_floor@5` | 0.571106 | **0.576436** | **+0.005330** | 0.556808 |
| `fan_floor@8` | 0.564321 | **0.571251** | **+0.006930** | 0.566320 |
| `fan_floor@10` | 0.560872 | **0.567809** | **+0.006936** | 0.569466 |
| `fan_floor@32` | 0.524690 | **0.543104** | **+0.018415** | 0.580723 |
| `fan_floor@64` | 0.450229 | **0.487402** | **+0.037174** | 0.584776 |
| `fan_floor@117` | 0.200698 | 0.192015 | −0.008683 | 0.587214 |
| ⭐ `fan_diversity` | 2.565058 | **2.086324** | **−0.478734 (−18.7 %)** | — |
| `fan_endpoint_std` | 3.733775 | **3.000200** | −0.733575 (−19.6 %) | — |

⭐ **Three things changed between 23.6 % and 100 % of training, and they are one story:**

1. **The fan TIGHTENED** — diversity −18.7 %, endpoint spread −19.6 %.
2. **The floor ROSE at every interior k** (5 through 64), most at large k (+0.0372 at
   k = 64). The model did not just concentrate; **its worst candidates got better.**
3. ⭐ **The `rand_best@k` crossing REPLICATES IN THE SAME INTERVAL.** At 9,500 the floor
   crosses the best-of-N control between **k = 8** (0.5643 > 0.5583) and **k = 10** (0.5609
   < 0.5625); at FINAL between **k = 8** (0.5713 > 0.5663) and **k = 10** (0.5678 < 0.5695).
   ⇒ the crossing the instrument was built around is a property of the readout, not of one
   checkpoint — now replicated on the **same model at a different training step**, having
   already replicated across models.

⚠️ `fan_floor@1` and `@117` fall slightly. `@117` is the whole-fan minimum and follows the
tightening tail; neither is a headline, and both are reported so the table is not read as
uniform improvement.

### 4.3 ⭐⭐ P3 RE-PROVED BY MUTATION on the FINAL bank — third independent replication

The deliberate 90 % collapse, run on this new bank:

| | 9,500 | **FINAL** |
|---|---|---|
| `d_diversity` | −2.308552 (rel −0.9000) | **−1.877692 (rel −0.9000)** |
| `d_floor@117` under `geom_feasibility` | +13.602342 | ⭐ **+17.568424** |
| both halves fire? | ⭐ yes | ⭐ **yes** |
| `floor_verdict` | COLLAPSE-SUSPECT | **COLLAPSE-SUSPECT** |

⇒ ⭐ **A destroyed fan again reports a LARGE POSITIVE floor gain** — +9.614186, +13.602342,
now **+17.568424** on three different banks — which is exactly why `fan_floor@k` may never
be read without `fan_diversity` beside it. `floor_verdict` still classifies the collapsed
pair rather than passing it.

⚠️ **And the PARTIAL limit reproduces too**, correctly: under `--quality composed` the floor
half is **INERT** (`d_floor` exactly 0.000000), because collapsing the GEOMETRY cannot move
a **precomputed** per-candidate reward column. The instrument says so itself and names the
re-run that exercises both halves. ⛔ That is the control being honest about its own scope,
and it is why the geometry quality is the one that carries the proof.

---

## 5. ⭐ What was PROVEN EARLIER AND RE-PROVEN HERE, by mutation and not by inspection

| proof | banked | **re-measured this turn** |
|---|---|---|
| control-space `logp` carries a policy gradient | 71 / 71 trainable tensors | ⭐ **71 / 71**, twice |
| ⭐ explored fan `envelope_violation`, control space | **exactly 0.000000** | ⭐ **exactly 0.000000**, twice — **bit-identical across three independent draws** |
| metre-space `envelope_violation` on the same model and batch | 36.096268 | **26.158453**, then **29.912878** |
| `|grad|` sum, control space | 4.070923e+05 | 3.810021e+05, 6.016779e+05 |
| all 16 arms build and validate their `PostTrainConfig` | 16 | **16** |
| verdict | PASS | ⭐ **PASS** |

⭐⭐ **This is a STRONGER statement than the banked one, and the strength is in the
asymmetry.** The metre-space violation is **draw-dependent** — 36.10, 26.16, 29.91 across
three inference draws — while the control-space zero is **bit-identical every time**. ⇒
*"flyable by construction"* is now shown to be **construction and not a lucky draw**: a
structural identity that no draw disturbs, against a regression whose magnitude wanders but
never approaches zero.

⛔ **The preflight is still NOT an arm launch** — no optimizer, no `step()`, no checkpoint,
no result.

---

## 6. Suite state — a CONTROLLED comparison, and the one failure is the MOUNT

* ⭐ **RL suite: 289 passed, 1 failed** (284 plus the 5 new `test_rl_gate_statistic.py`) across every `stack/tests/test_rl_*.py` (17 files),
  up from the predecessor's 254 because 31 tests were added since and none regressed.
* ⛔ **The single failure is `test_rl_veto_explicit.py::test_driver_arm_table_declares_the_veto`,
  and it is a MOUNT failure, not a code failure.** Its traceback ends at
  `OSError: [Errno 22] Invalid argument` reading
  `G:\…\stack\tanitad\eval\echo_gate.py` — the documented G:-mount hard-failure mode.
  ⚠️ **Reported INCONCLUSIVE, not passing:** ten consecutive retries on the SAME file over
  ~70 s all failed with the same `OSError`, so I cannot show it green and I do not claim it.
  It touches no file I wrote.
* ⭐ **WHICH TREE RAN, established positively rather than assumed.** The venv's editable
  install registers a `MetaPathFinder`
  (`__editable___tanitad_0_0_1_finder.py`, `MAPPING = {'tanitad': 'G:\…\stack\tanitad'}`),
  which **takes precedence over `PYTHONPATH`**. ⇒ every job in this turn imported `tanitad`
  from the **repo**, i.e. from HEAD — not from my working clone. That is the correct code,
  and my clone's copies were md5-verified file-by-file against it, so the two are the same
  bytes.
* ⚠️ **A whole-`stack/tests` run is not comparable** and is not quoted: 6 sibling files fail
  **collection** on this clone (`test_accel_probe`, `test_accum_effective_batch`,
  `test_closedloop_floor`, `test_frame_align`, `test_refav1_lead_block`,
  `test_xodr_junction_probe`), four of them with the same `OSError: [Errno 22]`.

---

## 7. Escalations — raised here, not written into a doc for someone to find

1. ⛔⛔ **THE SHIPPED `progress_lead_cap` DEFAULT COSTS +0.153679 [+0.084053, +0.254735],
   SEPARATED, AND THE ONE-LINE ALTERNATIVE IS ALREADY IMPLEMENTED AND TESTED.**
   `_progress_cap_mode` maps `True` to `"achievable"`; mapping it to `"lead"` recovers the
   whole difference (`lead_min` 0.302476 vs `ship` 0.456155) and restores `progress`'s gap
   to −0.011272, *better* than pre-repair's −0.009769. ⛔ **I have NOT changed it.** It is a
   reward-design decision that changes every downstream RL evaluation, and adopting it as
   an *improvement* would need its own pre-registration scored on a grid neither split has
   seen — this panel cannot supply that any more. **Owner: PI / Master Mind. 0 GPU.**
   The measurement is pinned by `stack/tests/test_rl_gate_statistic.py` either way, so the
   decision cannot be lost.
2. ⛔⛔ **`G-REWARD`'s REPLACEMENT STATISTIC.** `S6_wmr` **passes the instrument test that
   the rate fails** (RNR 1.21 vs 0.23; right-signed vs wrong-signed) and **fails the
   transposed 0.30 bar on both cells** (0.301183 and 0.456155). ⇒ **There is now a
   statistic that can measure distance-keeping; the reward does not yet clear a bar under
   it.** Whether `S6_wmr <= 0.30` becomes the adopted gate — and at which rung — is the
   decision the standing ruling reserves. **Owner: PI / Master Mind. 0 GPU.**
3. ⚠️ **THE STALE BANKED MEAN IS STILL QUOTED IN THE REGISTER.**
   `GOALS_AND_CLAIMS.md`'s `G-REWARD (ruling)` row carries **−0.011239 [−0.017039,
   −0.005306]** as an all-window mean gap. That value predates a change in
   `rewards._collision` (13 of 18,267 evaluations; `3/6089 = 4.927e-04` exactly). The
   **RATE is unaffected** and reproduces to **0.000e+00**. ⛔ The mean needs the correction
   attached wherever it is quoted as current. **Owner: whoever owns that row.**
4. ⚠️ **THE MUTATION CRITERION AS RULED IS NOT SELECTIVE AT PANEL SCALE.** Bit-identity
   disqualified **0 of 9** candidates. What separates the field is the pre-registered
   **RNR**, and the rate's conviction rests on **0.23** plus a **wrong-signed** response,
   not on bit-identity. ⇒ If the criterion is to bind future candidates it should be
   restated as *"must move by at least its own bootstrap SD, in the right direction"*.
   **Owner: PI / Master Mind** (it is a change to the ruling's own wording).
5. ⚠️ **ONE RL TEST CANNOT BE SHOWN GREEN AND I DO NOT CLAIM IT.**
   `test_rl_veto_explicit.py::test_driver_arm_table_declares_the_veto` dies at
   `OSError: [Errno 22]` reading `stack/tanitad/eval/echo_gate.py` **on the G: mount** —
   the mount, not the code, and ten same-file retries all failed. **INCONCLUSIVE, not
   passing.** It touches no file I wrote.

## 8. Deliverable manifest

| artifact | where it lives |
|---|---|
| pre-registration 3 — the mutation ranking | `…/2026-09-06-refcv4b-rl-repair/PREREG_STATISTIC_RANKING.md` (repo, commit `d9fb892`) |
| pre-registration 4 — `H-RL-GATE-STAT-1` + its bar | `…/PREREG_H-RL-GATE-STAT-1.md` (repo, commit `d9fb892`) |
| pre-registration 5 — `H-RL-GATE-STAT-2` + its bar | `…/PREREG_H-RL-GATE-STAT-2.md` (repo, commit `63e2609`, blob-verified) |
| this report | `…/2026-09-06-refcv4b-rl-repair/RESULT_GATE_STATISTIC.md` (repo) |
| ⭐ the mutation-ranking tool (`--split select` only) | `stack/scripts/rl_statistic_mutation_rank.py` (repo) |
| ⭐ the scoring tool (REFUSES a bar not in a pre-registration) | `stack/scripts/rl_statistic_score.py` (repo) |
| ⭐ mutation proofs of the statistic and the cap, 5/5 | `stack/tests/test_rl_gate_statistic.py` (repo) |
| the ranking panel + its controls | `…/raw/statistic_mutation_rank.json`, `…_v2.json` (repo) |
| the two held-out scores | `…/raw/score_PREREG_H-RL-GATE-STAT-1.json`, `…-2.json` (repo) |
| the five-cell held-out panel + paired deltas | `…/raw/cell_panel_heldout.json` (repo) |
| ⭐ **the refcv4b FINAL fan bank (B6)** + its report | `…/raw/fan_bank_refcv4b_FINAL_240w.npz` (+`.report.json`) (repo) |
| the FINAL primary readout + both collapse controls | `…/raw/fanfloor_refcv4b_FINAL_composed.json`, `…_geom_feasibility.json`, `…_composed_selftest.json`, `…_geom_feasibility_selftest.json` (repo) |
| the control-space preflight re-proof | `…/raw/cs_preflight_reproof.json` (repo) |
| the A40 harm proof (rates before / during / after, both md5s) | `…/raw/a40_copy_harm_proof.txt` (repo) |
| ⚠️ **ONE PLACE ONLY** — the 29 MB per-window rows (12 cells × 3 sides × 5 components × 6,089) | `devbox:C:\Users\Admin\rlgate\statrank_rows_v2.json` — re-derivable in ~11 min from the banked tool and the fit120 corpus, so deliberately not banked |
| ⚠️ **ONE PLACE ONLY** — the 1.29 GB refcv4b FINAL checkpoint | `devbox:C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt` (md5 `99b573e8277d94a5e3bfbf630cb4d751`) and `tanitad-a40:/workspace/experiments/refcv4b-b1-v72-40k/ckpt_40284_FINAL.pt` — **two copies, both md5-verified** |
