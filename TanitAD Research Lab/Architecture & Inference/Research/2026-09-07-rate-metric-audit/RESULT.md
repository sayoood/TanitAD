# Rate-metric audit — is the curvature arithmetic wrong?

**Agent:** Architecture & Inference · **Date:** 2026-09-07 · **Branch:** `agent/arch-inf-20260803`
**Register:** answers `D-CURV-84X-UNREPRODUCED` (`Project Steering/PI_VIDEO_REVIEW_2026-09-06.md` §ADDENDUM)
**Compute:** ZERO GPU. CPU/numpy only. Nothing pulled from HuggingFace. The A40 was not approached.

---

## ⛔ TIER DECLARATION — AND WHY THE FOUR FAMILIES DO NOT APPLY HERE

⛔ **No model produced a trajectory in this audit, so the four metric families and the T-tier ladder
do not apply to the arithmetic findings, and stamping one would be a category error.** §1–§4 are
claims about a FUNCTION, decided against closed-form targets.

⚠️ **§5 re-derives two arms' curvature, and THAT carries a stamp:**
**MEASURED (ours) — post-hoc re-scoring of a banked plan fan; no model executed.**
Corpus `taniteval/results/fan_refc-base-30k.pt` and `fan_refc-xl-30k.pt`, **n = 881 windows /
40 episodes**, ckpt `refc-base-30k` / `refc-xl-30k` **step 29999**, anchors 128 / 256.
⭐ **Horizon and grid, stamped:** `wp_steps = [5, 10, 15, 20]` → `ts0 = [0, 0.5, 1.0, 1.5, 2.0] s`
— **2.0 s, and UNIFORM at 0.5 s.** Tier inherits the dump's own provenance (T1 self-action open
loop). Intervals are paired episode-cluster bootstrap, 10,000 resamples, 40 clusters.

---

## 1. ⛔ THE HEADLINE — the named mechanism is REFUTED, and a different one is REAL

| question | answer |
|---|---|
| Is the rate arithmetic wrong, decided by the analytic circle? | ⛔ **NO for curvature.** `_seq_geometry` recovers `1/R` on the REAL non-uniform v3 horizon to **0.67 % worst case** |
| Was the 84× caused by the non-uniform v3 horizon set? | ⛔ **NO — the v3 set was never involved.** The dump's own contract is `wp_steps = [5,10,15,20]`, **uniform** |
| Does the 84× reproduce? | ⭐ **YES, BIT-EXACTLY** — `2.30973482131958`, ratio **84.40×** |
| Is the 84× a true statement about smoothness? | ⛔ **NO.** It is manufactured by **43 stopped-ego windows (4.9 %)** where curvature is undefined |
| Does the verdict move? | ⭐ **YES — the SIGN FLIPS, both separated.** The model is **BETTER** than a plan that never steers |
| Does it cancel in a paired contrast? | ⭐ **It cancels arm-vs-arm; it does NOT cancel against a floor.** An INTERACTION, not a shared bias |

⛔ **One rate metric IS wrong on a mis-declared grid, and it is not curvature: `yaw_rate`.**

---

## 2. P0 — THE ARITHMETIC, FROM SOURCE

### 2.1 `_seq_geometry` — what it divides by, quoted

`taniteval/taniteval/four_families.py:149` (`def _seq_geometry(wp, dt=DT_S)`), body at **:166–:184**:

```python
min_ds = MIN_DS_MPS * dt                            # :167  0.5 m/s * dt
p  = torch.cat([zero, wp], dim=1)                   # :170  PREPENDS THE EGO ORIGIN
d  = p[:, 1:] - p[:, :-1]                           # :171
ds = torch.linalg.norm(d, dim=-1)                   # :172
speed = ds / dt                                     # :173  <-- /dt
valid = ds > min_ds                                 # :175
heading = torch.atan2(d[..., 1], d[..., 0])         # :176  <-- NO division
dh = heading[:, 1:] - heading[:, :-1]               # :179
yaw_rate = dh / dt                                  # :181  <-- /dt
ds_mid = 0.5 * (ds[:, 1:] + ds[:, :-1])             # :185
curvature = dh / (ds_mid + _EPS)                    # :186  <-- / ARC LENGTH, NOT dt
accel = (speed[:, 1:] - speed[:, :-1]) / dt         # :189  <-- /dt
```

⭐ **The load-bearing line is `:186`.** Curvature divides heading change by **arc length**, so `dt`
never enters it. The module says so at `:183-184` — *"dt-INVARIANT by construction (both dh and ds
are geometric), which is why curvature and heading were the only two rate-like metrics the old
hard-coded dt did NOT corrupt."* ⇒ the sibling's premise — *"curvature, yaw-rate and heading-rate
are all rate metrics"* — is **false for two of the three**: heading is an angle (no division) and
curvature is a per-METRE rate (÷ arc length). Only `yaw_rate` (and `speed`, `accel`) are per-TIME.

⛔ **And `infer_dt` (`:187`) already refuses a non-uniform grid** rather than guessing:
`"NON-UNIFORM wp_steps {steps} — cannot derive a single dt"`. The suspected silent divide-by-one-dt
path is **guarded at the source**.

### 2.2 The horizon set, from source, in STEPS

`stack/tanitad/refs/refc_v3.py:121`:

```python
#: 6.0 s @ 10 Hz — BINDING (PLAN_STEPS=60, DT=0.1). 0.5 s stride through the
#: operative band (0, 2], 1 s stride through the tactical band (2, 6].
V3_HORIZONS: tuple[int, ...] = (5, 10, 15, 20, 30, 40, 50, 60)
```

⭐ **Settled: the values are STEPS on a uniform 0.1 s tick.** Slot times are
`0.5 / 1.0 / 1.5 / 2.0 / 3.0 / 4.0 / 5.0 / 6.0 s`, so the inter-slot spacing really is
**0.5 s ×4 then 1.0 s ×4 — genuinely NON-UNIFORM.** The sibling read the set correctly.
⛔ **What does not follow is that curvature is therefore wrong** — §3 settles that by construction.

---

## 3. ⭐ DECIDED BY CONSTRUCTION — the analytic circle

A circular arc of radius `R` has curvature **exactly `1/R`** everywhere; a straight line **exactly 0**.
Both fed through the REAL `_seq_geometry` on the REAL horizon set.
Code `raw/circle_test.py`, output `raw/circle_out.json`.

### 3.1 Curvature recovers `1/R` on the NON-UNIFORM v3 horizon

| grid | R = 20 m | R = 50 m | R = 200 m | R = 1000 m |
|---|---|---|---|---|
| **A — v3 NON-UNIFORM** `[0.5,1,1.5,2,3,4,5,6] s` | **1.0067×** | 1.0011× | 1.0001× | 1.0000× |
| **B — UNIFORM 0.5 s** (discriminating control) | **1.0026×** | 1.0004× | 1.0000× | 1.0000× |
| C — UNIFORM, same 6 s span | 1.0059× | 1.0009× | 1.0001× | 1.0000× |
| D — UNIFORM 2 s prefix (the gate arm's grid) | 1.0026× | 1.0004× | 1.0000× | 1.0000× |
| E — DENSE 0.1 s | 1.0001× | 1.0000× | 1.0000× | 1.0000× |

⭐ **THE DISCRIMINATING CONTROL SETTLES THE MECHANISM.** If non-uniformity were the defect, row A
would fail where row B succeeds. Both recover `1/R`; the worst-case residual is **0.67 % at R = 20 m**
(a 20 m radius is a tight junction turn), and **the UNIFORM control carries 0.26 % of it**. The extra
cost of the non-uniform spacing is therefore **≈ 0.4 %** — chord discretisation, `O(φ²)`, not a
grid error. Identical in float32 (production dtype): A = 1.001073×.

**NULL — straight line, true curvature exactly 0:** `max|κ| = 0.000e+00` on **all five grids**,
including the non-uniform one. Exact, not approximate.

**dt-invariance, byte-level:** `_seq_geometry(wp, 0.1)["curvature"]`, `(wp, 0.5)` and `(wp, 1.0)`
are `torch.equal`. The `dt` argument cannot move curvature at all.

### 3.2 The metric that IS wrong on a mis-declared grid

Same arc, true yaw rate `v/R`:

| grid | dt passed | yaw-rate ratio |
|---|---|---|
| v3 NON-UNIFORM | 0.1 | **7.50×** |
| UNIFORM 0.5 s | 0.1 | **5.00×** |
| UNIFORM 0.5 s | 0.5 | **1.00×** ✓ |
| dense 0.1 s | 0.1 | **1.00×** ✓ |

⚠️ The 7.50× is exactly `mean((Δt_i + Δt_{i+1})/2) / 0.1 = 0.75/0.1` — an arithmetic identity, not
an approximation. ⛔ **But for a pred-vs-gt comparison both arms pass through the same `1/dt`, so a
yaw-rate MAE is uniformly scaled and no sign or separation verdict can move from this** — only the
physical units of the reported magnitude are wrong. This is the already-registered 2026-08-03 defect
(`test_four_families_dt.py`), and `infer_dt` + the carried `dt_s` stamp already guard it.

---

## 4. ⛔ WHERE THE 84× ACTUALLY COMES FROM

### 4.1 It was never `_seq_geometry`, and never the v3 grid

The figure is produced by a **local reimplementation**, not the canonical metric:
`…/Research/2026-09-06-p4-p13-p14-validation/raw/p14_banked_fan.py:38-47`

```python
def curvature(p, ts):
    d1 = np.gradient(p, ts, axis=-2)          # ts = the FULL time vector
    d2 = np.gradient(d1, ts, axis=-2)
    num = np.abs(d1[...,0]*d2[...,1] - d1[...,1]*d2[...,0])
    den = np.power(d1[...,0]**2 + d1[...,1]**2, 1.5)
    return num / np.maximum(den, 1e-6)        # <-- floored denominator, NO validity mask
```

⛔ **Two facts kill the named mechanism outright:**

1. It passes **`ts0`, the full sample-time vector**, to `np.gradient`, which handles non-uniform
   spacing correctly. It does **not** divide by one `dt`.
2. ⭐ **The dump it read is not on the v3 grid at all.** `fan_refc-base-30k.pt` carries
   `wp_steps = [5, 10, 15, 20]` and `gt` of shape `(881, 4, 2)` — **four slots, uniform 0.5 s,
   2.0 s horizon.** Confirmed three ways: the tensor shape, the dump's `wp_steps` key, and the
   banked run log `p14_fan_base30k.txt:5` — *"waypoint frames [5, 10, 15, 20] → times [0.5, 1.0,
   1.5, 2.0] s (horizon 2.0 s)"*.

⇒ ⛔ **The 84× and the gate arm's 1.0× were BOTH measured on the same uniform 2 s grid.**
The grid cannot be the explanation for their disagreement, and the register's premise —
*"the gate agent reported every rate metric on the uniform 2 s prefix only, deliberately"* — is
true of **both** arms.

### 4.2 The 84× reproduces BIT-EXACTLY, then decomposes

`raw/rederive_84x.py` → `raw/rederive_out.json`. Control first:

| control | re-derived | published | match |
|---|---|---|---|
| `rank_shipped` κMAE | `2.30973482131958` | `2.30973482131958` | ⭐ EXACT |
| `straight_floor` κMAE | `0.027366388589143753` | `0.027366388589143753` | ⭐ EXACT |
| ratio | **84.400** | "~84×" | ⭐ |
| paired `shipped − oracle` | `−0.01141 [−0.03256, +0.00050]` | `−0.0114 [−0.0327, +0.0005]` | ⭐ EXACT |

⭐ **The 84× is REAL as an arithmetic fact.** It did not fail to reproduce. What it is not, is a
statement about smoothness:

| statistic (base-30k, n = 881) | `rank_shipped` | `straight_floor` |
|---|---|---|
| **MEAN** κMAE (the published number) | **2.30973** | 0.02737 |
| **MEDIAN** κMAE | **0.00067** | 0.00074 |
| p90 | **0.02307** | 0.03402 |
| p99 | 68.77 | 0.4087 |
| max | 127.2 | 3.0 |
| windows with κMAE > 10 | **32** | **0** |

⛔ **On the median AND the p90 window the model is BETTER than the plan that never steers.**
The entire 84× lives in the tail.

### 4.3 ⭐ THE MECHANISM, AND A DELIBERATE REGRESSION THAT PROVES IT

`curvature = |d1 × d2| / max(|d1|³, 1e-6)` is **singular as `|d1| → 0`**. The ten worst windows:

| κMAE | min step (m) | `v0` (m/s) |
|---|---|---|
| 127.24 | 0.01453 | 0.382 |
| 119.63 | 0.00210 | **0.000** |
| 116.76 | 0.00724 | **0.000** |
| 84.28 | 0.00036 | **0.000** |
| … 6 more, all `v0 ≤ 0.027` | | |

⛔ **THE EGO IS PARKED.** Sub-millimetre inter-waypoint steps, divided by their own cube.

⭐ **The deliberate regression — split by ego motion, same estimator, nothing else changed:**

| subset | n | shipped | straight_floor | ratio |
|---|---|---|---|---|
| ALL windows (**as published**) | 881 | 2.30973 | 0.02737 | **84.40×** |
| **MOVING only** (`v0 > 0.5 m/s`) | 838 | 0.00522 | 0.00926 | ⭐ **0.56×** |
| **STOPPED only** (`v0 ≤ 0.5 m/s`) | **43** | 47.22099 | 0.38016 | **124.21×** |

⇒ **43 windows — 4.9 % of the corpus — carry the whole finding.** Remove them and the model is
**1.8× BETTER** than the floor. Put them back and the 84× returns. The guard can fail in both
directions, so the mechanism is demonstrated, not asserted.

⭐ **WHY THE COMPARISON WAS NEVER FAIR:** a straight-line plan has `κ ≡ 0` **exactly, at any speed**
(§3.1 null, verified at v = 10, 1, 0.1, 0.01, 0.0 m/s). It is **structurally immune** to the
singularity the model is exposed to. `|κ_floor − κ_gt|` is bounded by the GT's own curvature;
`|κ_model − κ_gt|` is unbounded. ⓘ The tell was visible in the published table all along:
`cv_floor` and `straight_floor` have **different geometry** (cross-track 0.5259 vs 0.4662) yet
**byte-identical** κMAE `0.02737` — because both are straight lines, so both reduce to `mean|κ_gt|`.

### 4.4 The verdict under the canonical metric

Same dump, same grid, same windows, scored with `_seq_geometry` **and its `pair_valid` mask**:

| arm | p14 UNMASKED | **canonical MASKED** |
|---|---|---|
| `rank_shipped` | 2.30973 | **0.007711** |
| `straight_floor` | 0.02737 | **0.012113** |
| **ratio** | **84.40×** | ⭐ **0.637× — the model is 1.57× BETTER** |

**2×2, isolating the cause** (`raw/decompose.py`):

| | no mask | masked |
|---|---|---|
| **p14 estimator** | **84.40×** *(published)* | **0.49×** |
| **`_seq_geometry`** | 3.63× | **0.64×** |

⇒ ⭐ **THE MISSING VALIDITY MASK IS THE CAUSE, NOT THE ESTIMATOR CHOICE.** Both estimators flip to
"model better" once stopped steps are gated; neither survives without it. ⛔ `_seq_geometry`'s
`MIN_DS_MPS = 0.5` gate — `0.25 m` on this 0.5 s grid — masks **5.3 %** of pairs, and that is
exactly the population that manufactured the 84×.

---

## 5. ⭐ SIZE, SIGN, AND WHETHER IT CANCELS — the load-bearing question

`raw/paired_cancel.py`. Paired episode-cluster bootstrap, 10,000 resamples, 40 clusters.
Contrast = `A − B`; **negative = the first arm is better**.

| contrast | UNMASKED (published style) | **MASKED (canonical)** | verdict moves? |
|---|---|---|---|
| **base** `shipped − oracle` *(anchor vs anchor)* | −0.01141 [−0.03256, +0.00050] **not sep** | −0.00134 [−0.00501, +0.00112] **not sep** | ⭐ **NO** |
| **base** `shipped − straight_floor` *(anchor vs floor)* | **+2.28237 [+0.51780, +4.57674] SEP WORSE** | **−0.00390 [−0.00708, −0.00100] SEP BETTER** | ⛔ **YES — SIGN FLIPS** |
| **xl** `shipped − oracle` | +0.15240 [−0.14854, +0.48794] **not sep** | −0.00016 [−0.00145, +0.00092] **not sep** | ⭐ **NO** |
| **xl** `shipped − straight_floor` | **+1.55583 [+0.35017, +3.24901] SEP WORSE** | **−0.00558 [−0.00903, −0.00258] SEP BETTER** | ⛔ **YES — SIGN FLIPS** |

⭐⭐ **THE ANSWER, AND IT IS NOT "IT CANCELS":**

* **SIGN:** the defect **INFLATES** curvature error, without bound.
* **SIZE:** up to **84.40×** (base) / **57.85×** (XL) at the aggregate; **124×** on the stopped
  sub-population; **1.00×** on the moving one.
* ⭐ **IT CANCELS in an ARM-vs-ARM paired contrast** — both arms draw from the same fan and are
  equally exposed, so the shared bias subtracts out. **Both `shipped − oracle` verdicts are
  unchanged (not separated, before and after).**
* ⛔ **IT DOES NOT CANCEL against a FLOOR** — a straight-line floor is structurally immune, so the
  bias is an **INTERACTION**, and both floor-referenced verdicts **reverse sign while remaining
  separated**. That is the strongest form of "the verdict moves": not "uncertain", but
  **confidently backwards**.

⇒ ⛔ **Rule for the programme: any curvature contrast referenced to a straight-line or
constant-velocity floor is invalid unless the estimator is masked. Arm-vs-arm curvature contrasts
stand.**

---

## 6. ⛔ ENUMERATED — the affected published numbers, by artifact path

⚠️ Enumerated by a **filesystem walk with per-file retries** (`raw/enumerate_sites.py`), **37,742
files**, because `git grep`/`rg` under-report on the G: mount while exiting 0 — during this audit a
same-breath control read 0 **twice** and both were flaps, not absences. 12 files unreadable, all in
stale `.claude/worktrees/` copies. Same-breath control (`2.30973` → 10 live files) non-zero.

### 6.1 ⛔ AFFECTED — produced by the unmasked estimator, floor-referenced ⇒ **verdict reverses**

| artifact | lines | number |
|---|---|---|
| `…/2026-09-06-p4-p13-p14-validation/raw/p14_fan_base30k.json` | 22, 30 | `2.30973`, `2.32114` |
| `…/2026-09-06-p4-p13-p14-validation/raw/p14_fan_base30k.txt` | 16, 17, 20, 21 | the κMAE column |
| `…/2026-09-06-p4-p13-p14-validation/raw/p14_fan_xl30k.json` / `.txt` | 17, 20, 21 | `1.58320` |
| `…/2026-09-06-p4-p13-p14-validation/RESULT.md` | 81, 82, 85, 86, **100, 101, 103**, 106 | §1.5 table + §1.6(b) |
| `Project Steering/REFCV5_MISSING_PIECES_PLAN.md` | **251**, 450–455 | §8.2 + §9 caveat |
| `Project Steering/PI_VIDEO_REVIEW_2026-09-06.md` | **37** (row #9), 61–81 | + the ADDENDUM |
| `Project Steering/GOALS_AND_CLAIMS.md` | **10088–10094**, 11312 | the D-row + the P1 cross-ref |
| `…/2026-09-07-refcv5-v2-compose/README.md` | **474** (+329) | row #9 restatement |
| `…/2026-09-07-p1-agent-gate/PREREG.md` | 144 | quotes the 84× |
| `…/2026-09-07-p1-agent-gate/RESULT.md` | 122, 123, 125, 127, 133 | the "did not reproduce" claim |
| `…/2026-09-06-refcv4b-landing/raw/EDDA4_BANK_PRIOR.json` | 6187 | banked prior |

**The 12 κMAE values in the two p14 tables** (6 arms × 2 checkpoints) are all unmasked and all
inflated. ⛔ The two **floor** rows (`cv_floor`, `straight_floor` = `0.02737`) are the exception:
they are `mean|κ_gt|` and are **correct as computed** — they are simply not comparable to the others.

### 6.2 ⭐ NOT AFFECTED — verified, not assumed

* **Every `shipped − oracle` κ contrast in p14** (base `−0.0114`, XL `+0.1524`, both *not separated*).
  Arm-vs-arm ⇒ cancels ⇒ **verdict unchanged** (§5). These stand as published.
* **Every curvature number computed through `_seq_geometry`** — proven `1/R`-correct on the
  non-uniform v3 horizon and dt-invariant at byte level (§3.1).
* ⭐ **The P1 gate arm's LATERAL curvature** (`GOALS_AND_CLAIMS.md:11312`: arms `0.00973–0.00981`
  vs straight-line floor `0.00950`, *not separated*). These sit in the same `~0.01` band as this
  audit's own masked read (shipped `0.0077`, floor `0.0121`), i.e. the **masked** regime.
  ⇒ **the gate arm's 1.0× was the sound number all along.**
* **Every yaw-rate / speed / accel MAE** — uniformly scaled by `1/dt` for both arms of any
  comparison; units may be wrong, no sign or separation verdict moves (§3.2).

---

## 7. ⭐ THE FIX — shipped WITH the analytic pin

⛔ **`_seq_geometry` needs no change: it is correct.** What was missing is an **absolute anchor** —
`test_four_families_dt.py` pins curvature only by *invariance* and *ratios*, and a metric that is
uniformly wrong passes every one of those. That is precisely why this suspicion could not be
settled by reading the tests.

**Landed:** `taniteval/tests/test_four_families_curvature_analytic.py` — **14 tests, all passing**,
pinning:

1. curvature recovers **`1/R`** on the **REAL non-uniform v3 horizon**, R ∈ {20, 50, 200, 1000} m;
2. the straight-line **null is exactly 0** on every grid;
3. the **discriminating control** — the non-uniform grid costs < 0.5 % over the uniform one;
4. curvature is **byte-identical** across `dt ∈ {0.1, 0.5, 1.0}` (catches any reintroduced `/dt`);
5. yaw-rate **does** need the grid and reads `5×` when mis-declared (keeps the two straight);
6. ⛔ curvature is **unbounded without `pair_valid`** on a stopped path, and `pair_valid` excludes
   **every** pair of one;
7. the gate **can fail** — a moving path keeps every pair and still reads `1/R`;
8. a straight floor is **immune at any speed**, so unmasked floor-referenced contrasts are invalid;
9. the two independent estimators **agree on `1/R`**.

⛔ **MUTATION-TESTED — a guard that has not been made to fail is not a guard.** Two defects
reintroduced into a local mirror (repo untouched, restored byte-identical, blob
`1ef8e6e480d28893e8ccf3ea36f80c106411db89`):

| mutation | result |
|---|---|
| `curvature = dh / dt` — **the mechanism the sibling alleged** | ⭐ **9 of 14 FAIL** |
| `min_ds = 0.05` — the pre-2026-08-03 fixed gate | ⭐ **the stopped-path guard FAILS** |

⛔ **No live-sibling file was touched.** `refc_v3.py`, `refc.py`, `refc_v3_train.py` were **read
only**. The banked p14 artifacts are **not rewritten** — history stays as committed; this document
supersedes.

---

## 8. ⭐ WHAT THE 1.0× vs 84× DISAGREEMENT ACTUALLY WAS

⛔ **Not a grid error, and not a failure to reproduce.** Both arms measured on a **uniform 2 s**
grid. The disagreement was a **SCOPE ERROR compounded by an ESTIMATOR difference**:

| | the 84× (p14) | the 1.0× (P1 gate) |
|---|---|---|
| corpus | banked fan, 881 win / 40 ep | B1 EVAL split, 779 win / 28 ep |
| checkpoint | refc-base-30k @ **29999** | v7-tiny @ **500** |
| grid | `[5,10,15,20]` = 2.0 s **uniform** | `[5,10,15,20]` = 2.0 s **uniform** — ⭐ **the same** |
| estimator | local `\|d1×d2\|/\|d1\|³`, **unmasked** | masked, `~0.01` band |
| stopped windows | **included** (43, 4.9 %) | effectively gated |

⇒ ⭐ **The two numbers were never in conflict: the gate arm's 1.0× is what the masked metric reads,
and this audit reproduces it (0.64×) on the 84×'s OWN corpus and checkpoint.** The 84× is what the
same corpus reads when the singularity is left in. ⛔ **Nobody should re-open this on grounds of
corpus or checkpoint — the mask is the whole difference.**

---

## 9. ⛔ DOES ANY PROGRAMME VERDICT MOVE?

⭐ **YES — one, and it reverses.**

> ⛔ **RETIRED: "REF-C is ~84× worse on curvature than a plan that never steers."**
> On its own corpus, checkpoint and grid, with the singular windows masked, **REF-C is
> `0.64×` the straight-line floor — i.e. BETTER, separated** (base `−0.00390 [−0.00708, −0.00100]`;
> XL `−0.00558 [−0.00903, −0.00258]`, paired episode-cluster bootstrap, 10,000 resamples).

⚠️ **Consequences, scoped honestly:**

* **PI video-review row #9 ("trajectories are not smooth") loses its supporting evidence.** ⛔ It
  does **not** become "trajectories are smooth" — it returns to **OPEN and UNMEASURED**. The honest
  statement is that the programme has **no valid floor-referenced smoothness number**, in either
  direction.
* ⛔ **`REFCV5_MISSING_PIECES_PLAN.md` §8.2's "honest ceiling" loses one of its two pillars.** The
  vocabulary-resolution gap (§6, 88.13 % of the horizon unlabelled) is untouched and still binds.
* ⭐ **P14's own verdict is UNCHANGED** — it was always "a LONGITUDINAL lever, curvature not
  separated", and the arm-vs-arm contrast that says so cancels the defect (§5). **No P14 number moves.**
* ⭐ **The P1 agent-gate verdict is UNCHANGED and was right** — its 1.0× was the sound reading.
* ⛔ **`D-CURV-84X-UNREPRODUCED` is answered and should be CLOSED**, with its named mechanism
  recorded as **refuted** so it is not re-derived: the non-uniform v3 horizon does **not** corrupt
  curvature, and no other curvature number in the programme is touched by it.

---

## 10. Evidence class on every claim

| claim | class |
|---|---|
| `_seq_geometry:186` divides by arc length, not `dt` | **MEASURED (source read, quoted with line numbers)** |
| `V3_HORIZONS` is in STEPS, spacing 0.5 s then 1.0 s | **MEASURED (`refc_v3.py:121` + its comment)** |
| curvature recovers `1/R` on the non-uniform grid to 0.67 % | **MEASURED (ours) — analytic target, `raw/circle_test.py`** |
| straight line reads exactly 0 on every grid | **MEASURED (ours) — exact, `0.000e+00`** |
| the 84× reproduces bit-exactly | **MEASURED (ours) — `raw/rederive_84x.py`, control PASS** |
| 43 stopped windows carry it; moving-only = 0.56× | **MEASURED (ours) — deliberate regression, both directions** |
| masked verdict is `0.64×`, sign-flipped and separated | **MEASURED (ours) — paired episode-cluster bootstrap, B = 10,000** |
| it cancels arm-vs-arm, not arm-vs-floor | **MEASURED (ours) — 2×2 + both contrast shapes** |
| the enumeration in §6 | **MEASURED (ours) — 37,742-file walk, retried, control non-zero** |
| the sibling's report | ⛔ **INHERITED — verified from source and REFUTED in its named mechanism** |

⛔ **What this audit does NOT establish:** whether REF-C's trajectories are smooth *in absolute
terms*. It establishes only that the number claiming they are not is invalid, and that against the
floor it was compared to, on a masked metric, the model is better. A real smoothness verdict needs
a floor that is not structurally immune to the estimator's singularity — that is a new question,
not a resolved one.
