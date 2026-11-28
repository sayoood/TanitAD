# PRE-REGISTRATION — does finer angular resolution buy anything? (v5: 256×640 vs 384×960)

**Written 2026-07-27 (local, Europe/Berlin), BEFORE `res_eval.py` was run for the first time and
before any score, AP, R² or interval existed.** The only numbers below are deterministic geometry
(px/deg, token counts, frame parameters), which are inputs, not outcomes.
**Stream:** `resolution-gain`. **Host:** dev box (RTX 4060).
⛔ **No pod is touched.** pod1 is training; pod2 is finishing the 120° build with an armed val
waiter; a sibling owns the rig clean fix.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`. **Tiers:** `PROVISIONAL` / `CONFIRMED` /
`DECISION-GRADE`.
**Estimator, everywhere:** paired episode/clip-cluster bootstrap (`taniteval/taniteval/ci.py`,
B = 2000, seed 0), unit = **clip**. ⛔ **`overlapping_holdout_se` appears nowhere in this stream.**
🔒 PhysicalAI-AV is gated-confidential: counts only, never clip UUIDs.

---

## 1. The question, and the design

The PI chose **256×640** as v5's first step and asked *"which gain do we have in using the higher
resolution."* Both candidates deliver the **same 120° field**; they differ only in **angular
resolution**:

| candidate | frame | `f_ref` | **px/deg** (uniform, cylindrical) | tokens | PNG-lossless storage |
|---|---|---:|---:|---:|---:|
| **256×640 cylindrical** (chosen) | 256×640 @120.00° | 305.5775 | **5.3333** | **640** | 112.9 GB |
| 384×960 cylindrical | 384×960 @120.00° | 458.3662 | **8.0000** | **1440** | 221.9 GB |

*(px/deg and `f_ref` MEASURED by construction, `artifacts/geometry_ledger.json`; storage INHERITED
from `flagship-v5-retrain.PREP.md` §3.7.2.)* **Storage does not decide this — 221.9 GB is still
under today's 349 GB. Compute does: 1440 tokens is ~2.25× the encoder FLOPs of 640** *(the exact
ratio for a ViT at fixed `d_model`/`depth` is `(12Nd² + 2N²d)` — 2.44× at N = 640 → 1440, and
2.25× on the linear term alone; either way it turns a GPU-week into several)*.

### 1.1 ⭐ The design: run the ladder DOWNWARD on frames we can already build

**We do not build a 384×960 corpus to answer this.** We take the frame v5 will actually train on,
**remove** angular resolution from it in controlled steps, and measure what the loss costs.

> **The inference, stated as the rule it is: if the model is insensitive to LOSING angular
> resolution across the range straddling 5.333 px/deg, it will almost certainly not benefit from
> GAINING it.**

### 1.2 ⚠️ And the asymmetry, stated honestly rather than hidden

**Insensitivity to loss is strong evidence, not proof, of insensitivity to gain.** A model can sit
at a floor in one direction and not the other: a representation can be saturated with respect to
*removing* detail (the task's discriminative content survives blurring) while still being able to
*exploit* finer detail if it were trained on it. **This experiment BOUNDS the upside; it does not
settle it.**

**What it would take to settle it** (costed in §6 of the report, not launched here): a short
**matched training ladder** — the same episodes, the same steps, the same optimizer, 2–3 angular
resolutions — reading the **slope**, not an absolute. That is hours-to-days of GPU, not the several
GPU-weeks a full 384×960 v5 would cost, and it is only justified if this ladder returns **GAIN**.

### 1.3 ⭐ The free calibration point — and a correction to the brief's figure

**The brief states that 256×640 @120° is "essentially the same on-axis figure (4.686)" as today's
deployed 4.643 px/deg. The 4.686 is a stale number: it belongs to the FOV audit's `256×640 @ 100°
pinhole letterbox` (`f_eff` 268.5), not to the 120° cylindrical frame v5 is actually built at.**
MEASURED, deterministic:

| frame | projection | px/deg on axis | px/deg at the edge | vs today's on-axis |
|---|---|---:|---:|---:|
| today, 256×256 @51.394° | pinhole, `f_ref` 266 | **4.6426** | 5.7176 (`1/cos²θ`) | 1.000× |
| **v5, 256×640 @120.00°** | **cylindrical**, `f_ref` 305.5775 | **5.3333** | **5.3333** (uniform) | **1.1488×** |
| 384×960 @120.00° | cylindrical, `f_ref` 458.3662 | 8.0000 | 8.0000 | 1.7231× |

**The brief's qualitative claim survives and its number does not:** v5 is **1.149×** today's on-axis
density and **0.933×** today's edge density — genuinely comparable, but not equal. ⇒ **the ladder
carries an explicit rung at `k = 1.1488` (`D_today`) that lands the wide frame exactly on today's
deployed on-axis angular resolution.** That rung is the calibration point.

---

## 2. The probe, and why this one

**Primary probe: the situation classifier's image-only arm**, re-fit per rung.
Frozen deployed-v1 encoder + readout (`v1_trunk.pt`, STRICT-loaded; `shape_shim.verify_identity()`
is **bit-identical** at the deployed shape), PCA rank 16 fitted on TRAIN rows only, closed-form
ridge with λ chosen by out-of-fold AP over chunk-grouped folds. Metric: **AP**, `ties="collapse"`
(`taniteval.rank_metrics`), against the audited chance comparator.

**Why this probe and not the alternatives:**

| candidate | verdict | why |
|---|---|---|
| ⛔ `ade_0_2s` | **REFUSED as primary** | Two independent lines say it is the wrong target: the ADE-optimal pick collides **4.7×** more often than the rule-optimal one (separated), and published L2/ADE vs closed-loop Driving Score is **ρ = −0.36, p = 0.43** while Ego Progress is **ρ = 0.83**. (INHERITED, PREP card §0.) |
| **situation classifier** | ✅ **PRIMARY** | Built, with real held-out clusters (**153** lane-change / **264** intersection on the parity corpus; **44** / **175** in this local universe), and its camera-only arm clears chance at **2.17× / 2.60×** (INHERITED, `2026-07-26-situation-classifier` §5). It is **semantic and camera-only**, which is exactly the axis angular resolution should move. |
| **kinematic linear probe** | ✅ **SECONDARY, and the sensitivity instrument** | Image-only ridge regression of ego **speed** and **|yaw rate|** from the same frozen state. It is *continuous* (so far higher power per clip than a rare-event AP) and it reads the 3-frame stack's apparent motion, which is precisely what a low-pass degrades. |
| pseudo-simulation composite | ⛔ not used | Its measured resolution is poor (paired half-width **±0.0028** at T3's n; DAC missing and comfort a **literal constant** over 1,708,288 candidates) and it needs a retrained policy per rung — the expensive thing this design exists to avoid. |
| IDM channels | ⛔ not used | They are a lead-vehicle construct; **41.65 %** of windows have no vehicle ahead within 50 m (INHERITED, E-GOAL-1), so the probe is undefined on 4 windows in 10. |

### 2.1 ⛔ C13 — the probe must DEMONSTRATE it can fail, before any null is readable

**`S-DEMO` (registered gate, per probe, per situation).** The extreme rung **`D_6`** — a **6×**
angular-resolution loss taking the wide frame to **0.889 px/deg**, i.e. the information content of a
**107-px-wide** image across 120° — must be detected as **worse than `V5_640` with a paired CI that
excludes 0 downward.**

- **If `S-DEMO` fails for a probe/situation, that probe/situation returns `INSTRUMENT-BLIND` and NO
  verdict is emitted from it about any milder rung.** A guard that cannot fail is worse than none.
- `S-DEMO` is what makes the milder nulls readable, and it is stated before the numbers so it cannot
  be produced afterwards.

### 2.2 The registered MDE, expressed in the instrument's own demonstrated range

A null is only admissible if the instrument's demonstrated dynamic range is large against its own
resolution. **Registered:** for every reported null, compute

```
DR = |Δ(D_6)|  /  halfwidth( CI[ Δ(D_1p5) ] )
```

**If `DR < 2`, the null returns `UNPOWERED`, not `NO GAIN`.** *(This is the C30 discipline — a
recovery or null number without its background is inadmissible — applied to a null.)*

---

## 3. The arms — capacity matched by construction (C34)

**Every rung of the primary ladder is rendered onto the SAME 256×640 raster, encoded by the SAME
frozen trunk, with the SAME 640 tokens and the SAME resampled positional embedding. Only the pixel
content's angular bandwidth changes.** Nothing else can be the cause of a rung-vs-rung difference.

### Ladder A — PRIMARY. 256×640 cylindrical @ 120.00°, `f_ref` 305.5775, **640 tokens**

| arm | `k` | **px/deg** | equivalent width @120° | role |
|---|---:|---:|---:|---|
| **`V5_640`** | 1.0000 | **5.3333** | 640 | **the chosen v5 frame — the BASELINE** |
| `D_today` | 1.1488 | **4.6426** | 557 | ⭐ **calibration: today's deployed on-axis density** |
| **`D_1p5`** | 1.5000 | **3.5556** | 427 | ⭐ **the exact MIRROR of the 384×960 step — the PRIMARY CONTRAST** |
| `D_2` | 2.0000 | 2.6667 | 320 | |
| `D_3` | 3.0000 | 1.7778 | 213 | |
| **`D_6`** | 6.0000 | **0.8889** | 107 | **the `S-DEMO` sensitivity demonstration** |
| `A_1p5_alias` | 1.5, **no low-pass** | — | 427 | **the aliasing control** |

### Ladder B — REPLICATION, and it carries NO handicap. 256×256 @51.394°, `f_ref` 266, **256 tokens**

| arm | `k` | px/deg | role |
|---|---:|---:|---|
| `B_today` | 1.0 | 4.6426 | ⭐ **TODAY'S DEPLOYED INPUT — the trunk's NATIVE shape** |
| `B_D2` | 2.0 | 2.3213 | |
| `B_D6` | 6.0 | 0.7738 | ladder B's `S-DEMO` |

⭐ **Why ladder B matters and is not padding.** The frozen v1 trunk was trained at 256×256 / 51.4°,
so ladder A runs under a positional-embedding resample — a train/test shape shift. **That shift is
IDENTICAL across ladder A's rungs and therefore cannot bias a rung-vs-rung contrast**, but it can be
argued to compress the trunk's dynamic range. **Ladder B has ZERO shape shift** — `B_today` *is* the
deployed input. **If ladder B reproduces ladder A's slope, "the handicap explains the null" is
refuted rather than argued about.**

### The direct upward arm — SECONDARY, declared weak-if-null

| arm | frame | px/deg | tokens | handicap |
|---|---|---:|---:|---|
| `U_960` | 384×960 cyl @120.00°, `f_ref` 458.3662 | **8.0000** | **1440** | ⚠️ **one-directional, against it** |

⚠️ **Declared before the numbers (this is the FOV audit's §1.7 discipline, adopted verbatim).**
`U_960` needs the 16×16 learned positional grid resampled to 24×60, a shape shift v1 never saw.
**The bias runs one way: against the higher-resolution arm.** Therefore

- `U_960` **WINNING** under that handicap is **strong** evidence of gain and **overrides** a ladder
  null;
- `U_960` **tying or losing** is **weak** evidence and may be reading the handicap. **It does not by
  itself license `NO GAIN`.**

### 3.1 The degradation, and why it is a resolution change rather than an aliasing artifact

Each rung resamples the rendered frame to `1/k` with **`antialias=True`** (a true pre-filter) and
back to the common raster. **Downsampling without a low-pass folds high frequencies back into the
band — that measures aliasing, not resolution.** `A_1p5_alias` is built deliberately the wrong way
so the difference is **demonstrated rather than asserted**. The cylindrical observed mask is
re-applied after degradation so blur cannot bleed the honest-black unobserved periphery inward.

**Registered spectral ledger (`A-SPEC`):** the radially-averaged power spectrum per rung must show a
cutoff that moves as `1/k`, measured on a central observed crop. If it does not, the degradation is
not what it claims and no rung is quotable.

---

## 4. OUTCOME RULES — fixed now, executed in code by `res_eval.verdict()`, every branch reachable

Let `Δ(X) = metric(X) − metric(V5_640)`, paired clip-cluster bootstrap, B = 2000, higher = better.
Adjudicated **per probe and per situation. Never pooled.**

| # | verdict | condition | what it means |
|---|---|---|---|
| 0 | ⚠️ **`UNPOWERED`** | < **40** held-out positive clip clusters for that situation, **or** `DR < 2` (§2.2) | no verdict; **never reported as "no effect"** |
| 1 | ⛔ **`INSTRUMENT-BLIND`** | `Δ(D_6)` CI does **not** exclude 0 downward | the probe cannot detect a **6×** resolution loss ⇒ it cannot rank resolutions; no verdict |
| 2 | ⭐ **`GAIN`** | `Δ(D_1p5)` CI excludes 0 **downward** (the 1.5× step straddling v5 costs materially) — **or** `Δ(U_960)` CI excludes 0 **upward** | there is headroom on this axis ⇒ **recommend the upward test and cost it** |
| 3 | ✅ **`NO GAIN`** | `S-DEMO` passed, `DR ≥ 2`, **and both** `Δ(D_1p5)` **and** `Δ(D_2)` CIs contain 0 | the model is insensitive to a **1.5–2× angular-resolution LOSS** ⇒ **256×640 is the right stopping point and 384×960 is not worth several GPU-weeks** |
| 4 | ⚠️ **`PARTIAL`** | rules 2 and 3 disagree **across situations**, or `Δ(D_1p5)` contains 0 while `Δ(D_2)` is separated-worse | **report per situation**; if only `Δ(D_2)` fires, the knee is located *below* the v5 frame (between 2.67 and 3.56 px/deg), which is still `NO GAIN` for the upward question **with the knee stated** |

**Stated before measurement — what makes each rule return a FAILING value, and that it can:**

- **`GAIN` can fail** — it fails whenever `Δ(D_1p5)`'s CI contains 0. That is the outcome the PI's
  own instinct argues against, and it is registered as fully admissible and reportable at equal
  prominence. *(The program has a live precedent: the FOV audit's lane-change result came back
  **separated in the direction opposite** to the widening case, 0.759 [0.528, 0.997].)*
- **`NO GAIN` can fail** — it fails whenever `D_6` is not detected (→ `INSTRUMENT-BLIND`) or `DR < 2`
  (→ `UNPOWERED`). It cannot be reached by an instrument that simply sees nothing.
- **`INSTRUMENT-BLIND` can fire** — it is the same guard that returned `REFUSED` on all three
  situations in the sibling FOV sweep four days ago, i.e. it demonstrably fires in this codebase.
- **`UNPOWERED` can fire** — roundabout has **34** held-out positive clusters against the 40 bar and
  is expected to return it. Lane change has **44** and clears by 4.

**Secondary registrations, reported but NOT gating:**

- **`A-CTRL`** — `A_1p5_alias` is expected to be **no better** than `D_1p5`. If the two are
  indistinguishable, the probe cannot tell aliasing from a true resolution change and the ladder's
  reading is weakened — reported, not hidden.
- **`D_today`** — expected to lie between `V5_640` and `D_1p5`. It is the calibration rung; a
  non-monotone value there is a warning about the whole ladder.
- **`B`-ladder slope vs `A`-ladder slope** — agreement refutes the handicap objection (§3).

---

## 5. Bi-directional validation (both required, both able to fail)

- **`V-FID-A` (the wide frame is the frame v5 is being built at).** The per-clip **observed**
  fraction of the 120° cylindrical render must reproduce the independent **n = 3,000** rig census in
  `…/2026-07-28-wide-fov-build/WIDE_FOV_BUILD.md` §5 — rig A **0.0017 %** masked, rig B **8.897 %**
  (8.10–10.52 %, sd 0.63 %). A frame built with the wrong `f_ref`, the wrong projection or a
  geometric-centre principal point will not land there.
- **`V-FID-B` (the deployed frame is the deployed frame).** Ladder B's baseline is
  `calib.ftheta_crop_resize` **by call, not by re-implementation**, and the per-clip crop box comes
  from `calib.ftheta_crop_box` itself.
- **`C-NEG` (a deliberately failing input).** Column-shuffled features fitted and scored through the
  identical pipeline must land at chance. If `C-NEG` separates, the harness leaks and nothing is
  quotable. The chance comparator itself is audited by
  `taniteval.rank_metrics.assert_chance_comparator`, which is unwaivable.

---

## 6. Declared limitations, before the numbers

1. ⛔ **This is a FROZEN-ENCODER probe. It answers *"is the information there and does a trained
   representation use it?"*, not *"would a model trained at 384×960 be better?"*** Only a training
   ladder answers the second, and §1.2 says so and costs it.
2. ⛔ **The dev box does NOT hold the parity episode cache** (local key `14231cd29c74`, not
   `e438721ae894`). Every arm is rebuilt from raw mp4s, so this is a **self-contained paired
   experiment**; its absolute APs are never comparable to the situation classifier's pod3 numbers
   and are not quoted as such. **Parity is untouched: nothing here re-selects episodes.**
3. ⚠️ **The universe is the 500 locally decodable R0 clips**, not 2,376. Power is stated per
   situation and rule 0 protects the conclusion.
4. ⚠️ **The intersection label is the TURN half only** (the cross-traffic half needs
   `obstacle.offline` per clip). INHERITED licence: `2026-07-26-situation-classifier` §4 V4.
5. ⚠️ **No timing or throughput claim is made anywhere in this stream**, so GPU contention on a
   shared desktop costs wall-clock only and cannot invalidate a number. *(This is deliberate: it is
   why the experiment is safe to run here at all, and it is the hazard that forced the sibling
   encoder stream to REFUSE its whole throughput table.)*
6. ⚠️ **`U_960` is a shim, not a retrained model.** Its handicap is declared in §3 and its null is
   declared weak in advance.

---

## 6b. ⚠️ AMENDMENT A1 — logged BEFORE any full-n number existed

**Added 2026-07-27, after the §1–§6 text above was staged (`sha256`
`c7ba5cf10deaf5c17fb2800cce52def5529f9c75c4754996d7e354829d47232d`) and after a **code smoke-run at
n = 40 clips whose every branch returned `UNPOWERED`** by the registered guard. **No number from that
smoke run is used, quoted, or was capable of informing this amendment** — the only thing it showed is
that the pipeline executes and that the guard fires.

**What is added: `P3` — the P2 heads, re-sliced PER SITUATION.**
The exact same fitted kinematic heads from P2, evaluated on the held-out rows **inside each
situation's own event window** (`ongoing_{situation}`). It introduces **no new fit, no new
hyper-parameter and no new target** — it re-slices the held-out side.

**Why, stated as a foreseeable-power argument and not a result-driven one.** The brief requires the
answer **per situation, never pooled**. P1 supplies that but is a *rare-event AP* on a training side
of only 157 clips, and the sibling FOV sweep — the same probe, the same universe — returned
**`REFUSED` on all three situations**. If P1 goes blind, the stream would have no per-situation read
at all. P3 is the cheap insurance: intersection alone carries ~271 events × ~30 frames of *ongoing*
rows, an order of magnitude more than its anticipation positives.

**Same rules, unchanged.** P3 is adjudicated by the identical `verdict()` function, with the same
`S-DEMO` gate, the same `DR ≥ 2` requirement and the same 40-cluster power bar. **Nothing in §4 is
relaxed.** P1 remains the PRIMARY probe; P3 cannot promote a verdict P1's guard refuses — it is
reported alongside, per situation.

---

## 7. What this does NOT decide

- **Which FOV** — owned by `2026-07-27-fov-crop-audit` (settled: ~100–120°, wide not square).
- **Which projection / the plumbing** — owned by `2026-07-27-geometry-configurable`.
- **The encoder architecture and tokenization** — owned by `2026-07-27-encoder-tokenization`
  (settled: keep `d_model` 768 / `depth` 12, adopt no tokenization trick in v5).
- **Whether v5 runs wide at all** — decided by the PI on 2026-07-27; this stream only asks how
  finely.
