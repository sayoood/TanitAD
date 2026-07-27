# DOES FINER ANGULAR RESOLUTION BUY ANYTHING? — 256×640 vs 384×960 for flagship v5

**Date:** 2026-07-27 (local, Europe/Berlin). **Stream:** `resolution-gain`.
**Question, verbatim from the PI:** *"we need to investigate in a smart cheap way which gain do we
have in using the higher resolution."*
**Host:** dev box (`C:/Users/Admin/venvs/tanitad/Scripts/python.exe`, RTX 4060).
⛔ **No pod was touched.** At briefing: pod1 training, pod2 finishing the 120° build behind an
armed val waiter (PID 576967), a sibling on the rig clean fix. *(That build and the clean fix both
landed while this ran — `284c591`; reconciled in §4.7.)*
⛔ **No file under `stack/` or `taniteval/` was modified.** Everything in this stream lives in this
folder.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`. **Tiers:** `PROVISIONAL` / `CONFIRMED` /
`DECISION-GRADE`.
**Estimator, everywhere:** paired episode/clip-cluster bootstrap (`taniteval/taniteval/ci.py`,
B = 2000, seed 0), unit = **clip**. ⛔ **`overlapping_holdout_se` appears nowhere in this stream and
is not importable from any script in it.**
🔒 PhysicalAI-AV is gated-confidential: counts only, no clip UUID and no raw content in this folder.
**Pre-registration:** `PRE_REGISTRATION.md`, staged **before `res_eval.py` was ever run at full n**
(sha256 `c7ba5cf1…` at first staging, `3e558c81…` after amendment A1).
⭐ **And the audit trail is unusually strong, by accident: a sibling's whole-index commit
`d5d5afb` swept my staged `PRE_REGISTRATION.md` and scripts into git history — in a commit that
contains ZERO result artifacts.** The rules are therefore in the commit graph, provably before
`res_result.json` existed. *(I did not commit anything. This is the `git commit` whole-index
hazard `CLAUDE.md` warns about, firing again today — benign here, and reported in §8.)*

---

<!-- HEADLINE -->
# 0. HEADLINE

**Tier: CONFIRMED. Class: MEASURED (ours, `artifacts/res_result.json`; 500 clips, 16,962 rows,
343 held-out clips, B = 2000).**

> ## ✅ **`NO GAIN`. STOP AT 256×640. 384×960 IS NOT WORTH THE SPEND.**
>
> **Three independent probes returned `NO GAIN` — the pre-registered verdict — and not one returned
> `GAIN`.** Every remaining branch is a *refusal* (`INSTRUMENT-BLIND` / `UNPOWERED`), never a
> silent null.
>
> **The primary contrast is `D_1p5`: the exact MIRROR of the 384×960 step, a 1.5× angular-resolution
> change straddling the v5 frame.** It costs **nothing measurable**:
>
> | probe | clusters | **Δ(`D_1p5`) vs the v5 frame** | separated? |
> |---|---:|---|---|
> | situation-classifier AP, **intersection** | 166 | **−0.00150 AP [−0.00402, +0.00139]** | **no** |
> | image-only speed R² (pooled) | 343 | **−0.00080 R² [−0.00353, +0.00186]** | **no** |
> | image-only speed R², **intersection** | 175 | **−0.00824 R² [−0.02044, +0.00464]** | **no** |
>
> **And the probes are demonstrably NOT blind — that is what makes the null readable.** The same
> instruments detect the extreme rung with a hard separation: a **6×** loss costs **−0.01544 AP
> [−0.02479, −0.00624]** (intersection) and **−0.04025 R² [−0.05130, −0.03020]** (speed). Registered
> dynamic ranges: **DR = 5.71 · 14.93 · 4.28**, all far above the pre-registered `DR ≥ 2` bar.
>
> ⭐ **THE KNEE IS 2.3–3.0× BELOW THE FRAME v5 WILL TRAIN AT.** The first separated rung is `D_3` at
> **1.778 px/deg** on *both* powered probes (−0.00623 AP [−0.01071, −0.00141]; −0.00886 R²
> [−0.01447, −0.00369]), and the independent, **handicap-free** ladder B separates first at
> **2.321 px/deg** (−0.01610 R² [−0.02138, −0.01111]). **v5 runs at 5.333 px/deg. There is ~2.5× of
> unused headroom BELOW it before anything measurable happens — which is the same statement as
> "there is nothing above it to buy."**
>
> ⭐ **AND THE DIRECT UPWARD ARM DID NOT WIN ANYWHERE.** `U_960` — a real 384×960 / **1440-token**
> render, the registered override that could have flipped this to `GAIN` — is **not separated** on
> intersection AP (−0.00405 [−0.01348, +0.00575]) and **separated-WORSE** on speed R² (−0.00847
> [−0.01680, −0.00036]), yaw R² (−0.04513) and lane-change AP (−0.00485). Under its declared
> one-directional handicap a *loss* is weak evidence — **but it is certainly not evidence of gain,
> and the override never fired.**
>
> ### The bound, stated as a number rather than a mood
>
> **Mirroring `D_1p5`'s interval, the most that 384×960 could plausibly buy is +0.0014 AP — 1.7 % of
> the intersection probe's entire above-chance margin — and +0.0019 R², 0.27 % of the speed head's
> R².** The price is **2.587× encoder FLOPs** (an ESTIMATED 1.71–2.19× training step), **221.9 GB**
> instead of 112.9 GB, and a full corpus rebuild.
>
> ⚠️ **This BOUNDS the upside; it does not prove it is zero (§5).** It is a frozen-encoder result,
> and a downward ladder is structurally unable to observe a gain above its own baseline. **What it
> does prove is that nothing in our current tasks or representations is resolution-starved at
> 5.333 px/deg**, and that is the question that was blocking the spend.

### Per situation, never pooled — and two of the three return a REFUSAL, not a null

| situation | held-out clusters | P1 (situation AP) | P3 (speed R², same situation) |
|---|---:|---|---|
| **intersection** | 166 / 175 | ✅ **`NO GAIN`** (DR 5.71) | ✅ **`NO GAIN`** (DR 4.28) |
| **lane change** | 43 / 44 | ⛔ `INSTRUMENT-BLIND` | ⛔ `INSTRUMENT-BLIND` |
| roundabout | 30 / 34 | ⚠️ `UNPOWERED` | ⚠️ `UNPOWERED` |

**The one situation the FOV audit identified as the widening lever — intersections — is exactly the
one that is powered here, and it says `NO GAIN` on two independent probes.** Lane change returns
`INSTRUMENT-BLIND` because a **6×** resolution loss produced only −0.00151 AP there; **no verdict is
emitted for it in either direction**, which is the honest outcome and not a null. *(Descriptively —
and it is descriptive, not a verdict — that is consistent with the FOV audit's own powered finding
that lane changes are preceded by **less** peripheral content, 0.759 [0.528, 0.997].)*

⭐ **The rig-clean-fix that landed mid-run (`284c591`, a 176×624 / 429-token centred SLICE of
this very frame) does NOT stale this result:** a centred slice cannot move `f_ref`, so the clean
frame is **also 5.3333 px/deg** — the ladder's axis is unchanged and the verdict transfers
verbatim. It also hands back **33 % of the tokens**, which is compute better spent on field or
capacity than on acuity. Full reconciliation: §4.7.

### ⭐ One thing this run settled that it was not asked to, and it supports the widening

The FOV audit's model-side sweep returned `REFUSED` on power four days ago, leaving *"does a wider
input actually help a model?"* open. **On the same probe with 9.4× the clips, it is now answered in
the widening's favour** (paired CIs, `artifacts/cross_ladder.json`):

| contrast | intersection AP | lane-change AP | speed R² | **yaw-rate R²** |
|---|---|---|---|---|
| `V5_640 − B_today` | **+0.04246 [+0.01521, +0.07367]** sep | **+0.00992 [+0.00196, +0.02572]** sep | **+0.02774 [+0.00739, +0.04757]** sep | ⛔ **−0.03546 [−0.04559, −0.02492]** sep |
| **at MATCHED px/deg** `D_today − B_today` | **+0.04167 [+0.01396, +0.07368]** sep | +0.00972 [+0.00203, +0.02477] sep | +0.02785 [+0.00770, +0.04742] sep | −0.03557 [−0.04554, −0.02519] sep |
| `V5_640 − D_today` (resolution alone) | **+0.00079 [−0.00093, +0.00231]** | +0.00020 [−0.00131, +0.00181] | −0.00011 [−0.00186, +0.00173] | +0.00011 [−0.00093, +0.00122] |

⛔ **C34, stated rather than glossed:** `V5_640` and `B_today` differ in **field, projection AND
token count** — three things — so **this does not attribute the gain to field.**
⭐ **But the NEGATIVE attribution is exact, and it is the point: `V5_640 − D_today` — the 1.149×
resolution step alone — is a NULL on all four probes, and 98.1 % of the wide frame's intersection
advantage survives with angular resolution held equal.** ⇒ **whatever the wide frame buys, it is
not angular resolution.**
⚠️ **And the counter-case, at equal prominence: on YAW RATE the direction REVERSES** — today's
narrow frame beats the wide one by **+0.03546 R²**, separated, and that too is unchanged at matched
resolution. Reported because it is inconvenient, not despite it.
<!-- /HEADLINE -->

---

# 1. THE QUESTION, MADE PRECISE — and a correction to the figure it was posed with

Both candidates deliver the **same 120° field**. They differ only in angular resolution.
**MEASURED, deterministic** (`scripts/res_geometry.py` → `artifacts/geometry_ledger.json`):

| candidate | frame | projection | `f_ref` | **px/deg** | tokens | storage (PNG lossless) |
|---|---|---|---:|---:|---:|---:|
| today (deployed v1) | 256×256 @ 51.394° | pinhole | 266.0 | **4.6426** on axis · 5.7176 at edge | 256 | 44.8 GB |
| **256×640 (chosen)** | 256×640 @ **120.00°** | **cylindrical** | 305.5775 | **5.3333** (uniform) | **640** | **112.9 GB** |
| 384×960 | 384×960 @ **120.00°** | cylindrical | 458.3662 | **8.0000** (uniform) | **1440** | 221.9 GB |

*(px/deg and `f_ref` MEASURED here; storage INHERITED, `flagship-v5-retrain.PREP.md` §3.7.2.)*

## 1.1 ⚠️ The brief's px/deg figure is stale — and fixing it CREATES the calibration point

The brief states that *"256×640 @ 120° has essentially the same on-axis figure (4.686)"* as today's
4.643. **4.686 belongs to a different frame.** MEASURED: it is the FOV audit's
`256×640 @ 100° **pinhole** letterbox` (`f_eff` 268.5 ⇒ 4.686416 px/deg). The frame v5 is actually
being built at is **120° cylindrical**, `f_ref` **305.5775**, i.e. **5.3333 px/deg uniformly**.

> **The qualitative claim survives; the number does not.** v5 is **1.1488×** today's *on-axis*
> density and **0.9328×** today's *edge* density — genuinely comparable, but not equal.
> ⇒ ⭐ **the ladder therefore carries an explicit rung at exactly `k = 1.148788` (`D_today`) that
> lands the wide frame on today's deployed on-axis angular resolution. That is the free calibration
> point the brief asked for, and it only exists because the figure was recomputed rather than
> inherited.**

## 1.2 ⛔ Storage does not decide this. Compute does — and it is smaller than the brief says

**MEASURED, deterministic ViT arithmetic** (`scripts/res_cost.py` → `artifacts/cost_model.json`;
per layer `12Nd² + 2N²d`, `d = 768`, `depth = 12`, patch 16, `in_channels` 9):

| frame | tokens | encoder GFLOPs/frame | **× today's 256** | **× the chosen 640** | attention share | encoder params |
|---|---:|---:|---:|---:|---:|---:|
| today 256×256 | 256 | 23.40 | 1.000 | 0.371 | 5.16 % | 87,022,848 |
| **v5 256×640** | **640** | 63.04 | **2.693** | **1.000** | 11.98 % | 87,317,760 |
| *alt 320×800* | *1000* | *105.14* | *4.492* | *1.668* | *17.53 %* | *87,594,240* |
| **384×960** | **1440** | **163.07** | **6.968** | **2.587** | **23.44 %** | **87,932,160** |

*(The attention shares run slightly below the encoder stream's 5.3 % / 12.2 % because this table's
denominator includes the patch-embedding convolution. The conclusion is unchanged and is theirs:
**attention is still a minority term even at 1440 tokens**, so nothing here argues for windowed or
linear attention.)*

⇒ **Two corrections to the framing, both in the direction of making 384×960 cheaper than stated:**

1. **The brief's "8–9× the encoder cost of today's 256" is OVERSTATED — measured 6.97×.**
2. **And 8–9× is the wrong comparator anyway.** The decision is against the **chosen** 640-token
   frame, not against today's 256. That ratio is **2.587×** encoder FLOPs ⇒ an **ESTIMATED
   1.71–2.19× training step** (encoder share of a step bracketed at 0.45/0.60/0.75; the 60 % figure
   is **INHERITED and not re-verified** — it is the encoder stream's own open escalation E4).
3. **Parameters are a non-issue in both directions:** +909,312 over today (the positional embedding
   alone), leaving the deployable model far inside sub-300M.

**So the real price of 384×960 is ~2× a GPU-week plus 221.9 GB and a full corpus rebuild — not
"several GPU-weeks", but not cheap either. That is the number the answer below has to beat.**

---

# 2. THE DESIGN — run the ladder DOWNWARD on frames we already have

**We did not build a 384×960 corpus to answer this.** We take the frame v5 will actually train on,
**remove** angular resolution from it in controlled steps, and measure what the loss costs.

> **The rule, pre-registered:** if the model is insensitive to **LOSING** angular resolution across
> the range straddling 5.333 px/deg, it will almost certainly not benefit from **GAINING** it.

## 2.1 ⛔ C34 — capacity is matched by construction, not argued

Every rung of the primary ladder is rendered onto **the same 256×640 raster**, encoded by **the same
frozen deployed-v1 trunk** with **the same 640 tokens** and **the same resampled positional
embedding**. **Only the pixel content's angular bandwidth changes.** Nothing else can be the cause of
a rung-vs-rung difference. *(`shape_shim.verify_identity()` → `max_abs_diff = 0.0`, bit-identical at
the deployed shape, so this is the real v1 trunk and not an approximation of it.)*

## 2.2 The rungs (MEASURED geometry, `artifacts/geometry_ledger.json`)

**Ladder A — PRIMARY.** 256×640 cylindrical @120.00°, `f_ref` 305.5775, **640 tokens**, all rungs.

| arm | `k` | **px/deg** | equivalent width @120° | × today's on-axis | role |
|---|---:|---:|---:|---:|---|
| **`V5_640`** | 1.0000 | **5.3333** | 640.0 | 1.1488 | **the chosen v5 frame — the BASELINE** |
| `D_today` | 1.1488 | **4.6426** | 557.1 | **1.0000** | ⭐ **calibration: today's deployed density** |
| **`D_1p5`** | 1.5000 | **3.5556** | 426.7 | 0.7659 | ⭐ **the exact MIRROR of the 384×960 step** |
| `D_2` | 2.0000 | 2.6667 | 320.0 | 0.5744 | |
| `D_3` | 3.0000 | 1.7778 | 213.3 | 0.3829 | |
| **`D_6`** | 6.0000 | **0.8889** | 106.7 | 0.1915 | **the `S-DEMO` sensitivity demonstration** |
| `A_1p5_alias` | 1.5, **no low-pass** | — | 426.7 | — | **the aliasing control** |

**Ladder B — REPLICATION, and it carries NO handicap.** 256×256 @51.394°, `f_ref` 266, 256 tokens:
`B_today` (4.6426 px/deg — **today's deployed input, produced by calling `calib.ftheta_crop_resize`
itself**), `B_D2` (2.3213), `B_D6` (0.7738).

**Upward arm — SECONDARY, declared weak-if-null.** `U_960`: 384×960 cylindrical @120°, **8.0000
px/deg, 1440 tokens.**

## 2.3 ⚠️ The one-directional handicaps, declared before the numbers (they are in the pre-registration)

The frozen v1 trunk was trained at 256×256 / 51.4°, so any other shape runs under a
positional-embedding resample — a train/test shape shift.

- **Within ladder A the shift is IDENTICAL across rungs**, so it cannot bias a rung-vs-rung contrast.
- **`U_960` has its own, third shape** (24×60 token grid). **The bias runs one way: against it.**
  A `U_960` **win** under that handicap is **strong** evidence; a tie or loss is **weak** evidence and
  does not by itself license a null.
- ⭐ **Ladder B has ZERO shape shift** — `B_today` *is* the deployed input. **If ladder B reproduces
  ladder A's slope, "the handicap explains the null" is refuted rather than argued about.**

---

# 3. VALIDATION — what fired, what passed, and one registered sub-prediction that FAILED

## 3.1 ✅ `V-FID-A` — the wide frame is the frame v5 is being built at, checked against an n = 3,000 census

The 120° cylindrical render masks the periphery where the ray leaves the sensor. That masked
fraction is a **fingerprint of the frame**: it depends on `f_ref`, the projection, and the per-clip
principal point. `WIDE_FOV_BUILD.md` §5 measured it independently over the **whole 3,000-clip
selection**. MEASURED here on the ladder's own renders:

| | **this stream, ALL 500 clips** (`artifacts/vfid_rig_census.json`) | `WIDE_FOV_BUILD` §5 census (n = 3,000) | its real-decode counterpart (n = 24) |
|---|---:|---:|---:|
| rig A masked periphery | **0.0038 %** (sd 0.0152 %, **max 0.0635 %**) | 0.0017 % (median 0, **max 0.064 %**) | 0.0056 % (max 0.064 %) |
| **rig B masked periphery** | **8.9344 %** (**8.0994–10.1233 %**, sd 0.5333 %) | **8.897 %** (**8.10–10.52 %**, sd 0.63 %) | 9.507 % (8.10–11.60 %) |

⭐ **This is a near-exact independent reproduction, from a different code path on a different host:
rig B's mean differs by 0.037 percentage points, and rig A's MAXIMUM agrees to three decimals
(0.0635 % vs 0.064 %).** A frame built with the wrong `f_ref`, the wrong projection, or a
geometric-centre principal point instead of the per-clip one would not land there.
*(Rig mix differs — 23.2 % / 76.8 % A/B here vs 27.07 % / 72.93 % in the parity selection — because
this universe is the 500 locally decodable clips, not the canonical 3,000. The per-rig masked
fractions are unaffected: every projection uses that clip's own principal point.)*
⚠️ **It also re-confirms class C26 at 120°: the mask is essentially a pure rig-B effect, so a
wide-FOV v5 still sees a rig-correlated periphery.** → E3.

## 3.2 ✅ `V-FID-B` — ladder B's baseline IS the deployed crop

`B_today` is produced by **calling `calib.ftheta_crop_resize` and `calib.ftheta_crop_box`**, not by
re-implementing them. That is a stronger guarantee than the numerical C-FID the sibling FOV audit
needed (it had its own crop implementation to check). The per-clip crop box is recorded because it is
rig-dependent and is the quantity a wrong principal point would move.

## 3.3 `A-SPEC` — the low-pass, MEASURED. One leg passes, one registered leg FAILS

`scripts/res_spectral.py` → `artifacts/spectral_ledger.json`; 10 clips × 6 frames, radially averaged
power spectrum on a central 192×448 crop chosen to sit inside the observed mask (the mask edge is a
step discontinuity with power at every frequency and would contaminate the measurement).

| arm | `k` | low-pass? | px/deg | `f95` (cyc/px) | `f_cut` −40 dB (cyc/px) | predicted `0.5/k` |
|---|---:|---|---:|---:|---:|---:|
| `V5_640` | 1.000 | — | 5.3333 | **0.20313** | 0.21641 | 0.5000 |
| `D_today` | 1.149 | ✅ | 4.6426 | 0.08281 | 0.15625 | 0.4352 |
| `A_1p5_alias` | 1.500 | ⛔ **no** | 3.5556 | **0.08516** | **0.16016** | 0.3333 |
| `D_1p5` | 1.500 | ✅ | 3.5556 | **0.06875** | **0.14141** | 0.3333 |
| `D_2` | 2.000 | ✅ | 2.6667 | 0.05078 | 0.11953 | 0.2500 |
| `D_3` | 3.000 | ✅ | 1.7778 | 0.03906 | 0.09844 | 0.1667 |
| `D_6` | 6.000 | ✅ | 0.8889 | 0.02578 | 0.06641 | 0.0833 |

**✅ Leg 1 — monotone information removal: PASSES.** Both `f95` and `f_cut` fall strictly
monotonically with `k`, over a **7.9×** range in `f95`.

**✅ Leg 2 — `A-CTRL`, the aliasing control: PASSES, and it is the one that matters.** At the
**identical nominal factor 1.5**, the arm built **without** the low-pass carries **1.239× more**
high-frequency power (`f95` 0.08516 vs 0.06875; `f_cut` 0.16016 vs 0.14141). ⇒ **"downsampling
without a low-pass measures aliasing, not resolution" is demonstrated on our own frames, not
asserted — and the ladder uses the low-passed operation.**

**⛔ Leg 3 — the registered prediction `f_cut ≈ 0.5/k` FAILS, and I report it as a failure rather
than redefining the metric.** Measured `f_cut/prediction` runs 0.433 → 0.797, i.e. every rung's
−40 dB point sits **well below** its own Nyquist — *including the undegraded baseline* (0.216 vs
0.500). **The reason is measured, not guessed: the corpus's own spectrum is already 40 dB down at
0.43 of Nyquist, so the −40 dB point is SPECTRUM-limited, not filter-limited, at every rung.** The
prediction was wrong about what `f_cut` measures. *(Kept and reported because a registered
sub-prediction that is quietly re-scoped after the fact is exactly the forking-paths failure
`GATE_PROTOCOL` §0.3 forbids. Legs 1 and 2 are what the ladder actually needs, and both pass.)*

## 3.4 `C-NEG` and the chance-comparator audit

Column-shuffled features, fitted and scored through the identical pipeline, must land at chance; the
chance comparator itself is audited by `taniteval.rank_metrics.assert_chance_comparator`, which is
unwaivable and exists because this program once shipped a "chance" baseline scoring 1.726× chance.
Results in §4.

<!-- RESULTS -->
---

# 4. RESULTS

**All numbers from `artifacts/res_result.json`; every table below is machine-generated into
`artifacts/tables.md` by `scripts/res_tables.py` — nothing is transcribed by hand.**
500 clips · **16,962 rows** · 157 TRAIN / 343 HELD-OUT clips, chunk-grouped · B = 2000 · unit = clip.

## 4.0 Power, and the guard doing its job

| situation | held-out rows | held-out positives | **positive clip clusters** | base rate | C-POW (bar = 40) |
|---|---:|---:|---:|---:|---|
| intersection | 8,705 | 862 | **166** | 0.09902 | ✅ OK |
| lane change | 9,609 | 203 | **43** | 0.02113 | ✅ OK (by 3) |
| roundabout | 9,585 | 146 | **30** | 0.01523 | ⚠️ **UNDERPOWERED — no verdict** |

## 4.1 ✅ `C-NEG` and the chance audit — the harness does not leak

| situation | C-NEG shuffled AP | base rate | AP/base | above chance? |
|---|---:|---:|---:|---|
| lane change | 0.02395 | 0.02113 | 1.133 | ✅ **at chance** |
| intersection | 0.09323 | 0.09902 | 0.942 | ✅ **at chance** |
| roundabout | 0.01529 | 0.01523 | 1.004 | ✅ **at chance** |

Regression C-NEG: shuffled R² = **−0.31307** (speed) / **−0.30106** (yaw) — far *worse* than
predicting the mean. `assert_chance_comparator` passed on the constant-score comparator in all three
situations (unwaivable; it exists because this program once shipped a "chance" baseline scoring
1.726× chance).

**And the baseline arms see the target.** `V5_640` clears chance at **1.834×** (intersection),
**2.040×** (lane change); `B_today` — literally the deployed frame — at **1.405×** / **1.571×**.
*(Calibration, DIRECTIONAL only: the situation classifier reports **2.60× / 2.17×** camera-only on
the parity corpus. Lane change lands almost on top of it; intersection lands lower, as expected from
the TURN-half-only label and a 157-clip train side. **Absolute APs are not comparable across
universes and are not claimed to be — limitation 3.**)*

## 4.2 ⭐ LADDER A — the primary. Same raster, same trunk, same 640 tokens; only bandwidth changes

**P1 — situation-classifier AP, INTERSECTION (166 clusters).** `V5_640` AP = 0.18161, chance 0.09902.

| arm | px/deg | equiv. width | AP | **Δ vs `V5_640` [95 % CI]** | separated? |
|---|---:|---:|---:|---|---|
| **`V5_640`** | **5.3333** | 640 | **0.18161** | — | — |
| `D_today` ⭐ | 4.6426 | 557 | 0.18082 | −0.00079 [−0.00231, +0.00093] | no |
| **`D_1p5`** ⭐ | **3.5556** | 427 | 0.18010 | **−0.00150 [−0.00402, +0.00139]** | **no** |
| `D_2` | 2.6667 | 320 | 0.17867 | −0.00294 [−0.00622, +0.00074] | no |
| **`D_3`** | **1.7778** | 213 | 0.17538 | **−0.00623 [−0.01071, −0.00141]** | ⭐ **YES** |
| **`D_6`** | **0.8889** | 107 | 0.16617 | **−0.01544 [−0.02479, −0.00624]** | ✅ **YES — `S-DEMO`** |
| `A_1p5_alias` | 3.5556 | 427 | 0.18094 | −0.00067 [−0.00313, +0.00211] | no |
| `U_960` | **8.0000** | 960 (1440 tok) | 0.17755 | −0.00405 [−0.01348, +0.00575] | no |

**P2 — image-only speed R² (pooled, 343 held-out clips).** `V5_640` R² = 0.69980.

| arm | px/deg | R² | **ΔR² vs `V5_640` [95 % CI]** | separated? |
|---|---:|---:|---|---|
| **`V5_640`** | **5.3333** | **0.69980** | — | — |
| `D_today` ⭐ | 4.6426 | 0.69991 | +0.00011 [−0.00173, +0.00186] | no |
| **`D_1p5`** ⭐ | **3.5556** | 0.69900 | **−0.00080 [−0.00353, +0.00186]** | **no** |
| `D_2` | 2.6667 | 0.69633 | −0.00347 [−0.00756, +0.00031] | no |
| **`D_3`** | **1.7778** | 0.69094 | **−0.00886 [−0.01447, −0.00369]** | ⭐ **YES** |
| **`D_6`** | **0.8889** | 0.65955 | **−0.04025 [−0.05130, −0.03020]** | ✅ **YES — `S-DEMO`** |
| `A_1p5_alias` | 3.5556 | 0.69855 | −0.00124 [−0.00372, +0.00122] | no |
| `U_960` | **8.0000** | 0.69132 | **−0.00847 [−0.01680, −0.00036]** | ⛔ **YES — WORSE** |

> ### The shape both probes agree on
> **Flat to `D_2`, then it breaks.** `D_1p5` accounts for only **9.7 %** (AP) and **2.0 %** (R²) of
> the effect `D_6` produces, with a CI containing zero in both. The response is **strongly convex**:
> nothing happens until you go **2–3× below the v5 frame**, and then it happens fast.
> ⇒ **v5 sits well inside the flat region. That is what "no gain above it" looks like from below.**

## 4.3 ⭐ LADDER B — the replication that kills the handicap objection

`B_today` is **today's deployed 256×256 / 51.4° input at the trunk's NATIVE shape — no positional
embedding was resampled, so there is no train/test shape shift at all.**

| arm | px/deg | **ΔR² (speed) vs `B_today`** | **ΔAP (intersection) vs `B_today`** |
|---|---:|---|---|
| `B_today` | 4.6426 | — | — |
| `B_D2` | **2.3213** | **−0.01610 [−0.02138, −0.01111]** ⭐ **sep** | **−0.00263 [−0.00521, −0.00069]** ⭐ **sep** |
| `B_D6` | 0.7738 | **−0.07009 [−0.08350, −0.05694]** sep | **−0.01077 [−0.01754, −0.00570]** sep |

> **Ladder B reproduces ladder A's shape on a different frame, a different projection, a different
> token count and with ZERO shape-shift handicap — and it locates the knee in the same place.**
> Ladder A: not separated at 2.667 px/deg, separated at 1.778. Ladder B: separated at 2.321.
> **⇒ the knee is ~1.8–2.3 px/deg on two independent instruments. "The handicap explains the null"
> is refuted, not argued about.**

## 4.4 The aliasing control, and the calibration rung

- **`A_1p5_alias` vs `D_1p5`** — −0.00068 vs −0.00150 AP; −0.00124 vs −0.00080 R². **Indistinguishable,
  both null.** ⇒ at a 1.5× factor neither operation costs anything measurable, so **the ladder's
  reading does not depend on which resampling was used at that rung.** *(The two ARE different
  operations — §3.3 measured the aliased arm carrying 1.239× more high-frequency power. They are
  simply both free at this factor. `A-CTRL` was registered as reported-not-gating, and this is the
  report.)*
- **`D_today`, the calibration rung** — the wide frame degraded to **exactly** today's deployed
  on-axis density (4.6426 px/deg) — is a **null on every probe**: −0.00079 AP, +0.00011 R².
  ⇒ ⭐ **the v5 frame's 15 % finer angular resolution over today's deployment buys nothing
  measurable, in the direction we can measure.**

## 4.5 The verdicts, as returned by `verdict()` — 3 × `NO GAIN`, 0 × `GAIN`, 5 refusals

| probe | key | held-out clusters | **verdict** | DR |
|---|---|---:|---|---:|
| **P1** situation AP | **intersection** | 166 | ✅ **`NO GAIN`** | **5.708** |
| P1 situation AP | lane change | 43 | ⛔ `INSTRUMENT-BLIND` | — |
| P1 situation AP | roundabout | 30 | ⚠️ `UNPOWERED` | — |
| **P2** speed R² | pooled | 343 | ✅ **`NO GAIN`** | **14.934** |
| P2 yaw-rate R² | pooled | 343 | ⛔ `INSTRUMENT-BLIND` | — |
| **P3** speed R² | **intersection** | 175 | ✅ **`NO GAIN`** | **4.284** |
| P3 speed R² | lane change | 44 | ⛔ `INSTRUMENT-BLIND` | — |
| P3 yaw-rate R² | intersection | 175 | ⛔ `INSTRUMENT-BLIND` | — |
| P3 yaw-rate R² | lane change | 44 | ⛔ `INSTRUMENT-BLIND` | — |
| P3 speed / yaw R² | roundabout | 34 | ⚠️ `UNPOWERED` | — |

**Every non-verdict is a refusal the rules produced, not a null I chose to read as one.**
⚠️ **The yaw-rate probe is `INSTRUMENT-BLIND` everywhere** (`D_6` = −0.00544 R² [−0.01190, +0.00032]
pooled — it *just* misses). **Ego yaw rate is apparently readable from these features almost
independently of angular resolution, so that probe carries no bits on this question.** Reported
because it is one third of the P2/P3 surface and hiding it would overstate the evidence.

⚠️ **A candid note on lane change**, since the rule's output and the descriptive picture differ and
both should be visible. The lane-change probe **is** capable of separating effects of this size — it
separated `U_960` (−0.00485 [−0.01256, −0.00069]) and `B_today` (−0.00992 [−0.02572, −0.00196]) — yet a **6×** resolution loss produced only
**−0.00151 [−0.00800, +0.00662]**. The registered rule calls that `INSTRUMENT-BLIND` and **I honour the
rule**; the descriptive reading is that resolution matters *even less* at lane changes than at
intersections. **No verdict is claimed for lane change in either direction.**

## 4.6 The cross-ladder contrasts, with paired intervals

`artifacts/cross_ladder.json` (`scripts/res_cross.py`, which imports `res_eval` rather than
re-implementing it). These are quoted in §0 and are the only contrasts in this document that cross
between the two ladders, so they get their own paired bootstrap rather than a difference of two
point estimates.

| contrast | intersection AP | lane-change AP | speed R² | yaw-rate R² |
|---|---|---|---|---|
| `V5_640 − B_today` | **+0.04246 [+0.01521, +0.07367]** | **+0.00992 [+0.00196, +0.02572]** | **+0.02774 [+0.00739, +0.04757]** | ⛔ **−0.03546 [−0.04559, −0.02492]** |
| `D_today − B_today` *(matched px/deg)* | **+0.04167 [+0.01396, +0.07368]** | +0.00972 [+0.00203, +0.02477] | +0.02785 [+0.00770, +0.04742] | −0.03557 [−0.04554, −0.02519] |
| **`V5_640 − D_today`** *(resolution ALONE)* | **+0.00079 [−0.00093, +0.00231]** | +0.00020 [−0.00131, +0.00181] | −0.00011 [−0.00186, +0.00173] | +0.00011 [−0.00093, +0.00122] |

**Read the third row.** It isolates the *only* thing that separates the top two rows — a 1.149×
angular-resolution step — and it is a **null on all four probes**. **98.1 % of the wide frame's
intersection advantage, and 100.4 % of its speed advantage, survive with resolution held equal.**
⇒ the wide frame's advantage (and its yaw-rate deficit) are properties of field / projection /
token count, **not of px/deg** — which is the same conclusion the downward ladder reaches from the
other direction.

## 4.7 ⭐ RECONCILIATION with the rig-clean-fix that landed WHILE this ran (`284c591`)

A sibling stream committed a **clean frame** mid-run: a **centred sub-rectangle** of exactly the
256×640 / 120° cylindrical frames this ladder renders from — **176×624, 429 tokens
(−33.0 %), 117.000° × 32.131°**, proven **6/6 bit-identical** to a slice of the built cache
(1,206 frames, `max_abs_diff` 0). It also reports that **8.67 % of clips (260 of 3,000) could never
supply 120° horizontally** (pooled min 118.958°).

**Does that stale this result? No — and the reason is exactly the axis this stream measures.**

| | |
|---|---|
| the clean frame's `f_ref` | **305.5775 — UNCHANGED** (a centred slice cannot move the boresight or the focal) |
| ⇒ its angular resolution | **5.3333 px/deg — IDENTICAL to `V5_640`, this ladder's baseline** |
| what changes | the **extent** (rows/cols) and hence the **token count**, 640 → 429 |
| what this ladder measured | **px/deg**, holding raster and capacity fixed |

⇒ **The verdict transfers verbatim.** *"Is 5.3333 px/deg enough?"* is the same question on a
176×624 clean frame as on a 256×640 one, and a 1.5×-finer clean frame would cost the same ~2.25–
2.6× in tokens for the same bounded-near-zero return.
⭐ **It also makes the decision cheaper in the same direction:** the clean frame already gives back
**33 % of the tokens**, so the compute the 384×960 upgrade would have consumed is now available for
**field or capacity — which is where §4.6 says the measurable effect actually lives.**
⚠️ **One genuine consequence for this document:** my renders include the honest-black masked
periphery that the clean frame removes. That can only have *handicapped* `V5_640` relative to a
clean-framed model — i.e. it runs **against** the wide arm, not in its favour — and it is orthogonal
to the resolution axis, which is matched across every rung by construction.

## 4.8 ⚠️ The one place a per-situation slice is awkward, declared

P3's within-slice R² is computed against that slice's own mean, and the heads were fitted on all
rows — so the *level* can be negative (e.g. `speed|lane_change` R² ≈ −0.92: lane changes occupy a
narrow speed band, and a head fitted globally does worse there than that slice's own mean). **The
paired Δ between two arms on identical rows is still exactly the quantity the ladder needs; the
absolute level is not interpretable and is not interpreted.**
<!-- /RESULTS -->

---

# 5. ⚠️ WHAT THIS BOUNDS AND WHAT IT DOES NOT SETTLE — stated as plainly as the result

**This experiment is structurally incapable of observing a gain above its own baseline.** That is
not a caveat added afterwards; it is a property of a downward ladder, and it is the C13/C14 hazard
applied to this design: *before recording a limit, ask whether the instrument could have reported a
larger value.* **The downward rungs could not.** The only arm here that can report an upward effect
is `U_960`, and it carries a declared one-directional handicap.

**So the honest logical form is:**

| | |
|---|---|
| **what is MEASURED** | the model's sensitivity to angular resolution *below* 5.333 px/deg, on two probes, per situation, with a demonstrated failing direction |
| **what is INFERRED** | that a representation which does not lose anything when detail is removed is unlikely to gain from more of it |
| ⚠️ **why the inference is strong but not a proof** | a model can be at a floor in one direction and not the other. A *frozen* v1 encoder was trained at 4.64 px/deg and can only report what it already learned to use; an encoder **trained** at 8.0 px/deg could in principle learn to exploit detail this one never had a reason to encode. **This bounds; it does not settle.** |
| **what would settle it** | a **matched short training ladder** — §6 |

⚠️ **And one more limit, stated because it cuts against the convenient reading:** the probes here are
a semantic situation classifier and a kinematic regression. **Neither is a fine-grained detection
task.** If finer resolution buys anything, the likeliest place is small/distant object detection —
which this program does not currently instrument and which is *not* what the v5 objective optimises.
**The result in §4 is a statement about the tasks we actually train and gate on.**

---

# 6. THE UPWARD TEST — designed and COSTED, and DELIBERATELY NOT LAUNCHED

> ⛔ **§4 returned `NO GAIN`, so on the pre-registered rules this test is NOT warranted and it was
> NOT launched.** It is written out in full anyway, for two reasons: the brief asked for the design
> and its cost either way, and **a PI who wants the upward answer regardless should not have to pay
> a design round-trip for it.** §6.5 fixes its decision rule now, before it could ever be run.

## 6.1 The cheapest honest design

**Two arms, identical in everything but input angular resolution, measuring the SLOPE.**

| arm | frame | px/deg | tokens | isolates |
|---|---|---:|---:|---|
| `R_640` | 256×640 cyl @120° | 5.3333 | 640 | the chosen v5 frame |
| `R_960` | 384×960 cyl @120° | 8.0000 | 1440 | **the candidate** |

Identical episodes (a **declared subset** of the parity train split — see the parity note below),
identical seed, step count, optimizer, schedule and loss weights. **`state_dim` stays 2048** — the
`SpatialGridReadout` is the geometry firewall, so the predictor, both policies, imagination and every
grounding head are untouched (INHERITED + verified end-to-end by
`2026-07-27-encoder-tokenization` §2.1). **Only `--frame-h/--frame-w/--frame-hfov/--projection` and
the encoder's `image_width` change.**

A third rung (`320×800`, 1000 tokens, 6.667 px/deg — costed in §1.2 at 1.668× the 640 frame) is
**only** worth building if the two-arm slope is positive and the knee then needs locating.

## 6.2 ⛔ The read, and the trap in it

**Do NOT read this on `ade_0_2s`** (PREP card §0: the ADE-optimal pick collides 4.7× more often;
published L2/ADE vs closed-loop Driving Score is ρ = −0.36, p = 0.43). The registered primary is the
**map-free composite**, with `wm_canary_ade_2s ≤ 0.55` as the kill secondary.

⚠️ **But the composite may not be able to resolve this, and that must be faced BEFORE the spend.**
INHERITED, PREP card §3.1 and the encoder stream §5.4: the composite's paired half-width at T3's n
is **±0.0028**; **DAC is missing** and **comfort is a literal constant** (100.0000 % violation over
1,708,288 candidates), so two of its terms carry no information; and the three trained arms occupy
**0.6096–0.6100 — 0.2 % of the distance to random (0.3968)**. **A short 3k-step ladder read on that
instrument has a real chance of returning `UNPOWERED`, which would buy nothing for 1–2 GPU-days.**

> ⭐ **Recommendation, and it is the cheapest part of this whole plan: read the short ladder on the
> probe THIS STREAM just built as well as on the composite.** It is the only instrument in the
> program with a *demonstrated* resolution-failing direction (`S-DEMO`, §4), it costs minutes on the
> two resulting checkpoints, and it removes the `UNPOWERED` risk from the critical path.

## 6.3 The cost, itemised — and the one input that is NOT measurable from here

| item | quantity | class |
|---|---|---|
| 384×960 PNG bytes/clip | ~90 MB (2.25× the **MEASURED** 40 MB/clip at 256×640) | **ESTIMATED** |
| full 2,400-clip 384×960 train cache | **~214 GB** — ⛔ does **not** fit beside the existing ~95 GB cache inside pod2's proven 124.55 GB headroom | **ESTIMATED** on a MEASURED per-clip rate |
| ⇒ **the validation runs on a declared 600-clip subset** | ~54 GB (+ ~13.5 GB for a 150-clip val split) | design |
| build wall-clock | **MEASURED** 660 clips/h at 256×640 on 8 shards, and the build is **download-bound** (`WIDE_FOV_BUILD` §8), so ~450–660 clips/h ⇒ **≈ 1.1–1.7 h** for 600 + 150 clips | **ESTIMATED** on a MEASURED rate |
| HF egress | ~40 chunks × ~1.9 GB ≈ **76 GB** | ESTIMATED |
| the 256×640 arm's corpus | **already built on pod2** — free | MEASURED |
| step-time ratio 960 : 640 | **1.71–2.19×** | ESTIMATED (FLOPs MEASURED; encoder share INHERITED) |
| anchor for the absolute | the v4 30k run at 256 tokens took **212,544.6 s = 59.0 GPU-h** | INHERITED (`RETRACTION_LOG.md`, C10 entry, quoting the run's own artifacts) |
| ⇒ a **3k-step** matched pair | `R_640` ≈ 9–14 GPU-h, `R_960` ≈ 15–30 GPU-h ⇒ **≈ 24–43 GPU-h ≈ 1–2 GPU-days on one A40** | **ESTIMATED** |
| **against** a full 384×960 v5 | **~2× a GPU-week** (1.71–2.19× the step, 221.9 GB, full rebuild) | ESTIMATED |

🔴 **The one number that cannot be produced from here: the encoder's real share of a training step.**
The dev box is contended (any timing from it is inadmissible — the sibling encoder stream had to
REFUSE its entire throughput table for exactly this reason) and both idle pods are committed. **This
is the encoder stream's own open escalation E4 and it must be measured on the training host before
the ladder is sized.** It is the difference between a 1-day and a 2-day validation.

## 6.4 ⛔ Parity — the constraint this design must not break

The canonical corpus is `physicalai-train-e438721ae894` (2,376 episodes, skip-hash `f09e44db`).
**A 600-clip validation subset is NOT a re-selection of the parity corpus and must never be
registered as one.** It is a declared, matched subset used for a *slope*, both arms drawing the
identical clip-id list, and **no number from it may be quoted against `MODEL_REGISTRY.md`**. If that
cannot be guaranteed by the tooling, build the full 2,400-clip 384×960 cache on a host with the room
— or do not run it.

## 6.5 Pre-registered decision rule for the upward test (fixed now, before it is run)

- **PAY** — `Δcomposite(R_960 − R_640)` CI excludes 0 **upward** on the 600-episode deployment read,
  **and** `wm_canary_ade_2s ≤ 0.55` on `R_960` ⇒ **v5 pays 2.587× encoder FLOPs and 221.9 GB.**
- **DON'T PAY** — the CI contains 0 or excludes 0 downward ⇒ **v5 ships at 256×640.**
- **UNPOWERED** — the composite's MDE (**|Δ| ≳ 0.006**, INHERITED) is not cleared ⇒ report
  `UNPOWERED`, **never "no difference"**, and fall back to the §6.2 probe.

<!-- RECOMMENDATION -->
---

# 7. RECOMMENDATION — one plain paragraph, and it is a decision to NOT spend

> ## **Train v5 at 256×640 / 120° cylindrical and do not build 384×960.**
>
> **The measured answer to the PI's question — *"which gain do we have in using the higher
> resolution"* — is: none that we can detect, on the tasks we train and gate on.** A **1.5×**
> angular-resolution change straddling the v5 frame — the exact mirror of the 384×960 step — costs
> **−0.00150 AP [−0.00402, +0.00139]** on intersection anticipation and **−0.00080 R²
> [−0.00353, +0.00186]** on image-only speed, neither separated, on probes that detect a 6× change
> at **DR 5.71 / 14.93 / 4.28**. Mirrored, that bounds the plausible upside of 384×960 at
> **+0.0014 AP (1.7 % of the probe's whole above-chance margin)** and **+0.0019 R² (0.27 %)** —
> against **2.587× encoder FLOPs**, an ESTIMATED **1.71–2.19× training step**, **+109 GB** of
> corpus and a full rebuild. **A direct 384×960 / 1440-token arm was built and it did not win
> anywhere; on speed and yaw it was separated-worse.** **The `GAIN` branch was reachable, its
> override was reachable, and neither fired.**
>
> **Do not launch the upward training ladder.** It is designed and costed in §6 (**≈ 1–2 GPU-days
> plus a ~54 GB rebuild**) and it is ready if the PI wants it anyway — but on this evidence it would
> be spent to detect an effect bounded near zero, on an instrument (the map-free composite) that has
> **never demonstrated a resolution-failing direction** and whose own MDE is **|Δ| ≳ 0.006**.
>
> ⭐ **Spend the pixels on FIELD, not on acuity.** At **matched** angular resolution the wide
> 640-token frame is **+0.04167 AP [+0.01396, +0.07368]** over today's deployed frame, while the
> resolution step alone (`V5_640 − D_today`) is a **null on all four probes**; and degrading acuity
> by 1.5× costs nothing. **The two facts together say the lever is field/capacity, and the ladder
> says it is not resolution.** If more compute is ever available for the input, the evidence points
> at the FOV
> audit's `320×640` vertical-preserving option (true VFOV 49.25° vs the letterbox's 39.35°) or at
> the H2 second camera — whose `CROSS/OFF_FRONT` lift is **10.127× [1.962, 24.561]**, the largest
> effect in that document — **not at more pixels per degree.**

## 7.1 What the evidence does NOT support, stated as plainly

- ⛔ **It does not support "384×960 is definitely worthless."** It bounds the upside; §5 says why
  that is not the same thing, and a frozen encoder cannot report what a *trained* one might learn to
  use.
- ⛔ **It says nothing about lane changes or roundabouts.** Both returned refusals
  (`INSTRUMENT-BLIND` / `UNPOWERED`). **No verdict is claimed for either, in either direction.**
- ⛔ **It says nothing about small/distant object detection**, which is where finer resolution would
  most plausibly pay and which this program does not currently instrument (§5).
- ⚠️ **It does not attribute the wide frame's advantage to field.** `V5_640` vs `B_today` differs in
  field, projection and token count — **C34** — and this run cannot decompose it. What it *can* say,
  cleanly, is that the advantage is **not** angular resolution: the resolution step alone
  (`V5_640 − D_today`) is a null on all four probes, and 98.1 % of the advantage survives at matched
  px/deg.
- ⚠️ **On yaw rate the deployed narrow frame is separated-BETTER** (+0.03546 R²). If ego-rotation
  perception matters for v5, that is a cost of the wide frame worth watching — and it is *not*
  fixable with more pixels per degree, since the yaw probe is `INSTRUMENT-BLIND` to resolution.

## 7.2 If the PI wants the upward test anyway — the two changes that would make it worth running

1. **Read it on a probe with a demonstrated failing direction.** This stream's ladder has one
   (`S-DEMO`, DR up to 14.9); the map-free composite does not, and two of its terms carry no
   information on our fan. Running P1/P2/P3 on the two short-trained encoders costs **minutes** and
   removes the `UNPOWERED` risk from the critical path.
2. **Measure the encoder's share of a training step first** (E2). Without it the ladder cannot be
   sized, and the difference is a factor of two in the spend.
<!-- /RECOMMENDATION -->

---

# 8. DELIVERABLE MANIFEST

**Everything below is `git add`ed into the repo working tree. Nothing was committed, nothing was
pushed, no branch was switched. `stack/` and `taniteval/` are untouched by this stream.**
Path prefix: `repo: TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-28-resolution-gain/`

| artifact | what it is | where it lives | only one copy? |
|---|---|---|---|
| `RESOLUTION_GAIN.md` | this report | **repo** | no |
| ⭐ `PRE_REGISTRATION.md` | the rules, staged **before** `res_eval.py` ran at full n (sha256 `c7ba5cf1…`, then `3e558c81…` after amendment A1) | **repo** | no |
| ⭐ `artifacts/res_result.json` | **every number in §4** — per-arm AP/R², every paired contrast with its CI, both controls, and the machine-rendered verdicts | **repo** | no |
| `artifacts/tables.md` | §4's tables, generated from the JSON (nothing transcribed by hand) | **repo** | no |
| ⭐ `artifacts/geometry_ledger.json` | the px/deg ledger and the arithmetic of the §1.1 correction | **repo** | no |
| ⭐ `artifacts/spectral_ledger.json` | `A-SPEC`: the low-pass proven, the aliasing control, **and the failed leg** | **repo** | no |
| `artifacts/cost_model.json` | §1.2: tokens, FLOPs, attention share, params, storage, step-time bracket | **repo** | no |
| ⭐ `artifacts/vfid_rig_census.json` | `V-FID-A` over **all 500 clips** — the per-rig masked-periphery census that reproduces a sibling stream's n = 3,000 measurement | **repo** | no |
| ⭐ `artifacts/cross_ladder.json` | the cross-ladder contrasts with paired CIs (§0, §4.6) — wide vs deployed, and the same pair at **matched px/deg** | **repo** | no |
| `artifacts/trunk_provenance.json` | ⭐ how to rebuild the frozen v1 trunk **from a permanent artifact** (see E5) | **repo** | no |
| `scripts/res_extract.py` | the ladder: render → degrade → encode, with `V-FID-A/B` and the spectral hook | **repo** | no |
| `scripts/res_eval.py` | P1/P2/P3, the paired bootstrap, both controls, and `verdict()` | **repo** | no |
| `scripts/res_cross.py` | the cross-ladder contrasts (re-fits only the 3 arms involved; imports `res_eval`, re-implements nothing) | **repo** | no |
| `scripts/res_geometry.py` · `res_cost.py` · `res_spectral.py` · `res_tables.py` | the deterministic ledgers and the renderer | **repo** | no |
| per-clip frozen features `feats/clip_*.npz` (11 arms × 500 clips) | dev-box scratch | **dev box only** | ⚠️ **yes — deliberately.** Derived from a gated corpus; regenerated by `res_extract.py` (resumable, skips existing npz) in ~75 min on this host |
| `labels/fov_labels.npz`, `labels/_LOCAL_ONLY_k2clip.json` | situation labels + the 🔒 clip-UUID map | **dev box only** | 🔒 the UUID map may never enter the repo; the labels regenerate in ~3 min from `2026-07-27-fov-crop-audit/scripts/fov_labels.py` (reproduced its `labels_summary.json` **exactly** — 500 clips, 30 chunks, 157/343, identical event counts) |

**Source artifacts read, not modified:** `stack/tanitad/data/{calib.py, physicalai.py}` ·
`stack/tanitad/models/{encoder.py, readout.py}` · `taniteval/taniteval/{ci.py, rank_metrics.py}` ·
`2026-07-27-fov-crop-audit/scripts/{fov_geom.py, shape_shim.py, fov_labels.py}` ·
`2026-07-26-situation-classifier/scripts/{sc_situations.py, sc_extract_trunk.py}`.

**Test status.** Run by me at HEAD `2b0f166`: `stack/` → **1,385 passed, 12 skipped, 0 failed**;
`taniteval/` → **565 passed**. *(The brief quoted 1,379/12 for `stack/`; the tree had already
advanced. **Skips unchanged at 12 — zero new skips.**)*
⚠️ **The tree advanced again WHILE this stream ran** — `d5d5afb` then `284c591` landed, and a
concurrent sibling has `stack/{scripts/train_flagship_v4.py, tanitad/data/parity.py,
tanitad/data/v2_dataset.py}` modified in the working tree right now. **Those commits report
`stack/` 1,415 passed / 12 skipped and `taniteval/` 606 passed, zero new skips.** I am **not**
re-quoting a suite run over a sibling's in-flight edits as my own evidence; **this stream touched
neither suite**, and `git status` confirms every file I changed is inside this folder.

⚠️ **A sibling committed my staged work.** `d5d5afb` ("the closed-loop metric was BLIND TO
OVER-TRAVEL...") swept `PRE_REGISTRATION.md` and five scripts from the shared index into its commit.
**I did not commit and did not push.** It is benign here — it even strengthens the pre-registration's
audit trail (that commit contains no result artifact) — but it is the whole-index hazard `CLAUDE.md`
documents, firing again, and it is worth the orchestrator knowing it happened today.

---

# 8b. 🔴 ESCALATIONS — raised here, in the report, not buried in a README

| # | escalation | who | why it cannot wait |
|---|---|---|---|
| **E1** | ⭐ **The v5 frame's angular resolution has been quoted wrong. It is 5.3333 px/deg (120° cylindrical, `f_ref` 305.5775), not 4.686** — that figure belongs to the FOV audit's *100° pinhole* letterbox. And **"8–9× the encoder cost of today's 256" is 6.97×**, while the decision-relevant ratio — against the **chosen** 640-token frame — is **2.587×**. | v5 launch-card owner | Both numbers are inputs to the spend decision, and the second makes 384×960 look ~3× more expensive than it is. |
| **E2** | 🔴 **The encoder's share of a training step is STILL unmeasured** and it sizes both the upward test (§6.3) and the v5 run itself. This is the encoder stream's open **E4**; this stream could not close it (the dev box is contended ⇒ any timing from it is inadmissible, and both idle pods are committed). | v5 launch owner | It is the only unmeasured input in the whole geometry plan, and it is the difference between a 1-day and a 2-day validation. |
| **E3** | ⚠️ **Class C26 is re-confirmed at 120°, independently.** MEASURED here on the v5 frame's own renders: rig A **0.0000 %** masked periphery, rig B **9.219 %** — landing inside `WIDE_FOV_BUILD` §5's n = 3,000 census (0.0017 % / 8.897 %). **Widening does NOT remove the rig-correlated asymmetry; it converts fabricated pixels into honestly masked ones and keeps the asymmetry.** | the sibling doing the rig clean fix | A wide v5 will still train on a rig-correlated mask, and this model demonstrably eats shortcuts (zeroing `v0` moves the imagined decode ×93.7). |
| **E4** | ⭐🔴 **NEW, and it is a measured COST of the geometry v5 is already committed to: the wide 120° frame is separated-WORSE at reading EGO YAW RATE from pixels** — `V5_640 − B_today = −0.03546 R² [−0.04559, −0.02492]`, and it is **not** a resolution effect (unchanged at matched px/deg: −0.03557). Meanwhile the same frame is separated-**better** on intersection anticipation (+0.04246) and on speed (+0.02774). ⚠️ **This lands on a live item:** the PREP card's §2 calls *"teach the encoder to perceive ego-motion"* **"the real item"**, and the latent-ablation verdict found the imagined metric decode is *"very nearly a PURE INTEGRATOR OF THE HANDED SCALAR"*. **v5 will train on a frame that is measurably worse at the perception this program has already identified as its weak point.** ⛔ **Do NOT read this as an argument against widening** — the same table shows widening winning on two of four probes, and this is a frozen-shim measurement under a declared handicap. It is a **flag to instrument**, not a verdict. | v5 launch owner + whoever owns C3 | It is cheap to watch (the `g_*` grounding keys and the speed/yaw probes already exist) and expensive to discover after a GPU-week. |
| **E5** | ⚠️ **`v1_trunk.pt` — the frozen probe trunk THREE streams rest on (situation classifier, FOV crop audit, this one) — lives in a session scratchpad outside git, and its own `src` field points at ANOTHER session's temp directory.** ✅ **I checked rather than alarmed: it IS reproducible** — `sc_extract_trunk.py` run on the permanent `C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt` (step 29999) yields a **bitwise identical** trunk on all 151 tensors. **What is owed is not a rescue but a recorded recipe**, which is now `artifacts/trunk_provenance.json`. It should live with the geometry layer, not in one incoming/ folder — the same shape as the `crux.py` escalation the FOV audit raised four days ago and which is still open. | geometry-layer owner | Three streams' headline probes are currently one `rm -rf %TEMP%` from being unreproducible-without-rework, and nobody had written the recipe down. |

---

# 9. LIMITATIONS, STATED PLAINLY

1. ⛔ **Frozen-encoder probe.** It answers *"is the information there, and does a trained
   representation use it?"*, not *"would a model trained at 384×960 be better?"* §5 and §6 say so and
   cost the difference.
2. ⛔ **A downward ladder cannot observe a gain above its baseline** (§5). Only `U_960` can, and it is
   handicapped.
3. ⛔ **The dev box does not hold the parity episode cache** (local key `14231cd29c74`, not
   `e438721ae894`). Every arm is rebuilt from raw mp4s, so this is a **self-contained paired
   experiment**; its absolute APs and R²s are **not** comparable to the situation classifier's pod3
   numbers and are never quoted as such. **Nothing here re-selects episodes; parity is untouched.**
4. ⚠️ **The universe is the 500 locally decodable R0 clips**, not 2,376, and the TRAIN side is only
   157 of them (chunk-grouped). The registered 40-cluster power guard is what protects the
   conclusion.
5. ⚠️ **The intersection label is the TURN half only** (the cross-traffic half needs
   `obstacle.offline` per clip). INHERITED licence: `2026-07-26-situation-classifier` §4 V4
   (perpendicular cross traffic is 2.415× [1.057, 7.931] more common on a tight turn).
6. ⚠️ **Neither probe is a fine-grained detection task** (§5). If resolution buys anything it most
   likely buys small/distant object detection, which this program does not instrument.
7. ⚠️ **No timing or throughput claim is made anywhere in this stream.** GPU contention on a shared
   desktop cost wall-clock only and cannot invalidate a number here — which is precisely why this
   experiment was safe to run on this host at all.
8. ⚠️ **`U_960` is a shim, not a retrained model.** Any adopted shape would be retrained, not shimmed.
