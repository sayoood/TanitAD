# WP-A ORACLE RECHECK — the readout-ceiling ladder under the CORRECTED azimuth address

**Date** 2026-09-08 · **Agent** Architecture & Inference FlyWheel · **Compute** dev-box RTX 4060
only, **0 A40, 0 Thor** · **Evidence class** `MEASURED (ours)` · **Tier** NOT APPLICABLE —
representation probe, no trajectory produced.

**Closes the one item `R-2026-09-08-wpa-mirror` left open:** *"the ORACLE ladder has NOT been
re-read under the corrected address … until then the ladder's absolute AP values are NOT
QUOTABLE."*

---

## 0. The answer, in three lines

1. **YES — the oracle path carries the same mirrored address.** `s6_oracle.py:81-82` computes
   `az = arctan2(Y_m, X_m)` then `colf = (F_REF*az + W_PX/2)/PATCH - 0.5`, which is
   `s5_indexed.py:77-78`'s formula character for character. It is the **opposite sense** to
   `stack/tanitad/data/bev_aux.py:196` and `stack/tanitad/data/psg_targets.py:72`.
2. **And it does not matter — MEASURED, not argued.** The ladder is address-sense **INVARIANT**:
   every rung agrees between the two senses far inside the rig's own training-seed noise floor.
3. **The ladder is therefore restored as a valid readout-ceiling measurement**, with two repairs
   that are improvements on what was banked: the AP is now **tie-group summed**, so the all-zero
   control reads its base rate **exactly**, and the numbers are quoted against a **replicate**
   floor rather than an episode interval alone.

---

## 1. Does the oracle path carry the mirrored address? — YES, and here is the file:line

| module | formula | +59.99° (hard LEFT) | −59.99° (hard RIGHT) |
|---|---|---|---|
| `stack/tanitad/data/bev_aux.py:196` | `col = floor((hfov/2 − az)/daz)` | **column 0** | column 39 |
| `stack/tanitad/data/psg_targets.py:72` | `col = int((hfov/2 − az)//(hfov/n))` | **column 0** | column 39 |
| `…/wpa-readout/code/s5_indexed.py:78` | `u = f_ref*az + W/2` | **col 39.50** | col −0.50 |
| **`…/wpa-readout/code/s6_oracle.py:82`** | **`colf = (f_ref*az + W/2)/patch − 0.5`** | **col 39.50** | **col −0.50** |

⭐ Run three ways rather than read once: the table above is **executed output**
(`raw/address_xcheck.txt`), importing the repo's own `bev_aux.azimuth_column` (md5
`954c7e6e2f98ee4ee225f2f3cf12ec20`, byte-identical copy) and the snapshot's
`psg_targets.azimuth_column`, against the two `colf` senses evaluated numerically.

The two senses are an **exact** mirror, asserted at runtime rather than assumed:

```
colf_prog + colf_wpa = (W/2 − f·az)/P + (W/2 + f·az)/P − 1 = W/P − 1 = TW − 1 = 39
max |colf_prog + colf_wpa − 39|  =  0.000e+00      (raw/o1_oracle_mirror.log)
```

⇒ the oracle path is **not** exempt from the defect by construction. Whether it is exempt in
*effect* is the measurement below.

---

## 2. The prediction, stated before any number was seen

Written into `code/o1_oracle_mirror.py`'s docstring and committed to the working tree **before the
run started** (2026-09-08 19:47Z), and carried into the output JSON as
`prediction_stated_before_measuring`:

> **INVARIANT.** The oracle map is **BUILT with the same address the head READS with**, so a sign
> flip permutes the intermediate map and un-permutes it at the lookup. Every column-axis operation
> in the path is equivariant under that mirror: the bilinear column lookup is exactly
> mirror-symmetric (the floor/ceil weights swap); every width pooling kernel used (1, 2, 5, 10, 20,
> 40) **divides TW = 40**, so the pooling partition maps onto itself under `c → 39 − c`, and so does
> the nearest-upsample. The only asymmetry is the learned 3×3 `mix` conv — a **relabelling** the
> head can learn equally well either way.
> ⇒ both senses must match **within the replicate floor** (banked: `orc_16x40` spans 0.4651–0.4773
> = **0.0122 AP** over train seeds 0/1/2), and the ladder must stay **monotone DOWN** under BOTH.

⭐ **This is the OPPOSITE prediction from the real-trunk arms**, where the mirror cost 1.57–2.10×
and turned the ladder **monotone UP**. That is what makes it falsifiable rather than a restatement:
one script, one variable moved, and the two panels must come out differently.

⭐ **And an invariance result is worthless on its own** — it is also what you would see if the head
simply ignored azimuth. So the panel carries a **mutation control**: `xwire`, the map **built under
one sense and read under the other**, i.e. the real historical defect deliberately reintroduced. It
**must** collapse toward `pos_only`. Committed in advance: *if `xwire` does not go RED, the
invariance claim proves nothing and must be discarded.*

**VERDICT: the prediction HELD, on both halves.** §3 and §4.

---

## 3. Controls first — the all-zero arm reads its base rate EXACTLY

⛔ `R-2026-09-07-ap-ties`: a naive per-sample AP breaks ties by **array order**, so a constant
score — which cannot rank at all — reads **above** its own base rate. `s6_oracle.py:182-188` is the
naive form. Both forms were run on the same control, so the bias is measured rather than asserted:

| control | AP | vs base rate |
|---|---|---|
| test base rate (**literal integer ratio**) | `337008 / 20806104` = **0.016197554333** | — |
| tie-group AP, constant score 0.37 | **0.016197554333** | **EXACTLY EQUAL** ✅ |
| tie-group AP, all-zero score | **0.016197554333** | **EXACTLY EQUAL** ✅ |
| **naive AP (s6's own form), all-zero** | 0.016340207902 | **+0.881 %** ⛔ |

⭐ The naive figure reproduces **WP-A's banked `0.016340211`** to eight decimals, which is what
makes this a diagnosis of that panel rather than a different measurement. **The tie defect is real
and it is now repaired**; it was also never large enough to matter against a ladder running
0.476 → 0.091, which is why every §5 conclusion of `RESULT.md` survived it.

⛔ **No headline number here is an accuracy** — an all-zero predictor scores **98.3802 %** accuracy
on this panel.

**n and d, printed as required:** rows fit/val/**test** = 7842 / 2407 / **2974**;
**d = 6996** addressable BEV cells (identical to WP-A's banked 6996 — restricting to cells valid
under **both** senses removed nothing, so the panels are directly comparable);
scored cells n = **20,806,104**.

---

## 4. The ladder, both senses — MEASURED

### 4.1 The ladder (train seed 0, tie-group AP, paired episode-cluster bootstrap, 2000 draws)

| rung | az cols | rows | **CORRECTED** (`prog`) | MIRRORED (`wpa`) | Δ prog−wpa | paired CI95 | separated |
|---|---|---|---|---|---|---|---|
| **16×40** full grid | 40 | 16 | **0.4762** | 0.4773 | −0.0011 | [−0.0041, +0.0016] | no |
| 8×20 — refcv5's map | 20 | 8 | **0.3341** | 0.3367 | −0.0025 | [−0.0053, +0.0003] | no |
| 4×40 — rows pooled only | 40 | 4 | **0.3113** | 0.3084 | +0.0029 | [−0.0002, +0.0057] | no |
| 4×8 — v7-tiny deployed | 8 | 4 | **0.2067** | 0.2106 | −0.0038 | [−0.0065, −0.0006] | ⚠ yes |
| 16×4 — cols pooled only | 4 | 16 | **0.2060** | 0.2028 | +0.0032 | [+0.0002, +0.0062] | ⚠ yes |
| 4×4 — v1/v6 flagship | 4 | 4 | **0.1584** | 0.1594 | −0.0010 | [−0.0022, +0.0002] | no |
| 4×2 | 2 | 4 | **0.1267** | 0.1251 | +0.0015 | [+0.0006, +0.0025] | ⚠ yes |
| 1×1 — global pool | 1 | 1 | **0.0914** | 0.0909 | +0.0005 | [+0.0000, +0.0011] | ⚠ yes |
| `pos_only` — marginal control | 40 | 16 | 0.0316 | 0.0316 | +0.0000 | — | — |
| all-zero — control | — | — | **0.016197554** | **0.016197554** | 0 | — | **= base rate exactly** |

⭐ **The MIRRORED column reproduces WP-A's banked panel to four decimals at every rung**
(0.4773 / 0.3367 / 0.2028 / 0.3084 / 0.2106 / 0.1594 / 0.1251 / 0.0909 against banked
0.4773 / 0.3367 / 0.2028 / 0.3083 / 0.2106 / 0.1594 / 0.1251 / 0.0909; seed 1 likewise reproduces
`0.4651` exactly). That is what makes this a **re-read of the same panel**, not a second experiment
that happens to agree — the same property that made WP-D's real-trunk recheck a diagnosis.

**Both ladders are strictly monotone decreasing.** Max |sense effect| across all eight rungs =
**0.0038**.

### 4.2 ⚠️ FOUR OF EIGHT SENSE CONTRASTS READ `separated` — AND ALL FOUR ARE NOISE

⛔ **Read this before quoting any `separated` above.** The largest of those "separated" effects is
**0.0038 AP**. The rig's own **replicate** — same flags, same sense, same split, **only the training
seed changed** — is *larger* and *also* separated:

| contrast | Δ AP | CI95 | separated |
|---|---|---|---|
| **REPLICATE** s0 − s1 @16×40, both `prog` | **+0.0115** | [+0.0065, +0.0164] | **YES** |
| sense prog − wpa @16×40 | −0.0011 | [−0.0041, +0.0016] | no |

⇒ **changing nothing but the training seed moves the ladder ~10× more than changing the address
sense**, and does so with a separated interval. This is `H-ESTIM-SEED-1` exactly as written in
`CLAUDE.md`: the episode-cluster bootstrap answers *"would another draw of EPISODES say this?"* and
is **structurally blind** to training variance. ⇒ **every `separated` in §4.1 is a false positive at
the level of the claim being made**, and the honest verdict is read against the replicate floor:

| rung | prog s0 | prog s1 | replicate \|Δ\| | wpa s0 | wpa s1 | replicate \|Δ\| | **sense \|Δ\|** |
|---|---|---|---|---|---|---|---|
| 16×40 | 0.4762 | 0.4646 | 0.0116 | 0.4773 | 0.4651 | **0.0122** | **0.0011** |
| 8×20 | 0.3341 | 0.3336 | 0.0005 | 0.3367 | 0.3337 | 0.0030 | **0.0026** |
| 4×4 | 0.1584 | 0.1571 | 0.0013 | 0.1594 | 0.1564 | 0.0030 | **0.0010** |
| 1×1 | 0.0914 | 0.0918 | 0.0004 | 0.0909 | 0.0941 | 0.0033 | **0.0005** |

**The sense effect is below the replicate floor at every rung where both were measured**, and
below it by 11× at the headline rung. ⇒ **INVARIANT. The prediction held.**

### 4.3 ⭐⭐ The mutation control — and it goes RED, hard

The map **built under one sense and read under the other** — the real historical defect,
deliberately reintroduced into the arm that is supposed to be immune:

| arm | build | read | AP | vs its matched full arm | paired CI95 |
|---|---|---|---|---|---|
| **`xwire_16x40`** | `prog` | `wpa` | **0.0750** | **−0.3996 (6.3× collapse)** | [−0.4420, −0.3586] **sep** |
| **`xwire_16x40_r`** | `wpa` | `prog` | **0.0737** | **−0.4019** | [−0.4459, −0.3606] **sep** |
| `xwire_8x20` | `prog` | `wpa` | 0.0695 | −0.2646 | — |
| `pos_only` (no feature at all) | — | — | 0.0316 | — | — |
| all-zero (base rate) | — | — | 0.0162 | — | — |

⭐ **The cross-wire costs 105× the largest sense effect** (0.3996 vs 0.0038) and is separated in both
directions. ⇒ the head uses azimuth **heavily**; the invariance in §4.1 is emphatically **not**
"the head ignores the address".

⭐⭐ **And the lateral family shows the mirrored feature is worse than NO feature** — `xwire`'s
lateral MAE at 15–30 m is **4.956 m against `pos_only`'s 4.479 m**. A mirrored map does not merely
fail to help; it **actively misleads**, which is precisely the mechanism WP-D measured on the real
trunk. *(`xwire` still beats `pos_only` on AP, +0.0433 — it retains row/range structure and coarse
presence, which is where that residue comes from.)*

### 4.4 Against the full 16×40 grid — the design table, re-measured

| pooled to | AP | **AP retained** | AP cost | banked (mirrored, naive-AP) |
|---|---|---|---|---|
| 8×20 (refcv5) | 0.3341 | **70.2 %** | **1.43×** | 71.6 % / 1.40× |
| 4×8 (v7-tiny deployed) | 0.2067 | **43.4 %** | **2.30×** | 44.1 % / 2.27× |
| 4×4 (v1/v6 flagship) | 0.1584 | **33.3 %** | **3.01×** | 33.6 % / 2.98× |

### 4.5 Axis attribution — re-measured

* azimuth only, 40 → 4 (`16×4`): 0.4762 → 0.2060 = **−56.7 %** *(banked −57.0 %)*
* elevation only, 16 → 4 (`4×40`): 0.4762 → 0.3113 = **−34.6 %** *(banked −34.6 %)*
* ⇒ ratio **1.64×** *(banked 1.65×)* — **azimuth remains the more expensive axis to cut to a given
  resolution**, with the same stated confound (40→4 is a 10× cut, 16→4 is 4×; it is not a per-factor
  attribution).
* matched-factor datum: both axes 2× (`8×20`) = **−29.8 %** *(banked −28.4 %)* against −34.6 % for
  elevation alone 4×.

### 4.6 LATERAL family — re-measured at the corrected address

Mean |error| between the true occupancy centroid's lateral coordinate and the probability-weighted
prediction, per forward-range band (`s6_oracle.py`'s own `lateral_mae`, recomputed post-hoc from the
banked corrected-address test scores). **n per band** (test rows with any true occupancy):
0–15 m **1560**, 15–30 m **1795**, 30–45 m **1582**, 45–60 m **1404**.

| rung | 0–15 m | 15–30 m | 30–45 m | 45–60 m | banked 15–30 m (mirrored) |
|---|---|---|---|---|---|
| **16×40** | **1.185 m** | **1.346 m** | 1.711 m | 2.205 m | 1.372 m |
| 8×20 | 1.302 m | 1.805 m | 2.559 m | 2.916 m | 1.814 m |
| 4×40 | 1.948 m | 1.983 m | 2.700 m | 2.804 m | 1.975 m |
| 4×8 | 2.000 m | 2.110 m | 2.954 m | 3.087 m | 2.133 m |
| 16×4 | 1.626 m | 2.258 m | 2.871 m | 2.948 m | 2.246 m |
| **4×4** | 2.121 m | **2.665 m** | 3.433 m | 3.298 m | 2.669 m |
| 4×2 | 2.926 m | 3.276 m | 3.548 m | 3.330 m | 3.283 m |
| 1×1 | 5.129 m | 4.545 m | 4.190 m | 4.638 m | 4.550 m |
| `pos_only` control | 5.078 m | 4.479 m | 4.175 m | 4.648 m | — |
| ⚠ `xwire_16x40` mutation | **5.488 m** | **4.956 m** | **4.655 m** | **5.067 m** | — |

⇒ **4×4 costs 1.98× the lateral error of 16×40 at 15–30 m (2.665 m vs 1.346 m)** *(banked: 1.95×,
2.669 vs 1.372)*. The column is restored essentially unchanged.

---

## 5. Why the oracle survived a defect that destroyed the real-trunk column

⭐⭐ **The two panels differ in exactly the property that decides it, and the `xwire` control
measures the difference INSIDE this panel rather than arguing it.**

* In the **ORACLE** arm the feature map is **scattered into the token grid by the same address
  function the head looks it up with**. A sign flip permutes the intermediate map and un-permutes
  it at the lookup, so the composition is unchanged. The address sense is a **private naming
  convention** of a closed loop, and a consistent renaming costs nothing.
* In the **REAL TRUNK** arm the feature map comes from an **encoder**, whose azimuth content is
  fixed by the physical camera. There the address sense is a **claim about the world**, and getting
  it backwards makes the head read agents on the left out of the columns that show the right. That
  is why the mirror cost the real trunk 1.57–2.10× and inverted its ladder.

⛔ **This is not a reason the recheck was unnecessary — it is the reason it had to be run.** The
sentence above is precisely the "tempting argument" the retraction warned against, and an argument
of exactly that shape is what put `R-2026-09-08-wpa-mirror` in the log. What converts it into a
result is the **`xwire` mutation control**: the oracle map built under one sense and read under the
other — the real historical defect, deliberately reintroduced into the arm that is supposed to be
immune. It goes **RED**, and by roughly the margin the real trunk suffered. ⇒ the invariance is
**not** "the head ignores azimuth"; the head uses azimuth heavily, and a genuine build/read mirror
destroys it. The sense is free **only** when both ends move together.

---

## 6. What moves in `E-READOUT-CEILING-1`, and what does not

### ✅ UN-FLAGGED — quotable again, at the corrected address

| claim | status |
|---|---|
| The **ORACLE ladder's absolute AP values** | **RESTORED.** Re-measured at the corrected address with tie-group AP; every rung agrees with the banked value inside the replicate floor. |
| The **AP-retained / AP-cost table** (8×20 71.6 %, 4×8 44.1 %, 4×4 33.6 %; costs 1.40× / 2.27× / 2.98×) | **RESTORED.** Re-measured at the corrected address: **70.2 % / 43.4 % / 33.3 %**, costs **1.43× / 2.30× / 3.01×** (§4.4). ⚠️ The small differences are **seed**, not address: the banked figures are a 3-seed mean over a 0.0122 floor and these are seed 0. **Quote either; do not mix them in one table.** |
| The **axis attribution** (azimuth-only −57.0 % vs elevation-only −34.6 %, ratio 1.65×) and its stated confound | **RESTORED** — re-measured **−56.7 % / −34.6 %, ratio 1.64×** (§4.5). The confound statement is unchanged and still required. |
| The **lateral-MAE column** (1.372 m at 15–30 m for 16×40, 2.669 m for 4×4) | **RESTORED** — re-measured **1.346 m** and **2.665 m**, so **4×4 costs 1.98× the lateral error of 16×40** (banked 1.95×). §4.6, with n per band. |
| The **replicate floor** and the two-variances framing (`H-ESTIM-SEED-1`) | **RESTORED and extended** — now measured under both senses. |
| ⭐ The **all-zero control** | **UPGRADED, not merely restored.** Under tie-group AP it reads its base rate **EXACTLY** (`337008/20806104`). `R-2026-09-07-ap-ties` retracted the `=` and the ✓ on this row; with the correct estimator they are **earned back**, and the naive form's +0.881 % is now a measured property of the *estimator*, not of the panel. |

### ⛔ STILL RETRACTED — nothing here moves

| claim | status |
|---|---|
| *"the map does not yet contain agents"* / *"none of it transfers"* | **STILL FALSE.** This recheck says nothing about the real-trunk column; WP-D's corrected re-read is what governs it, and `D0 − pos_only` is separated on both geometries. |
| *"WP-D is the prerequisite, not WP-B"* | **STILL VOID.** **WP-B remains the next lever**, 0 new training arms to decide. |
| The **real-trunk AP column** (0.027–0.034 as published) | **STILL mirrored-address numbers.** §5 explains why the oracle's immunity does **not** extend to them — the mechanism that spares the oracle is exactly the one the real trunk lacks. |

### ✅ UNAFFECTED — already standing, and untouched by this work

Quantisation geometry (median **4** BEV cells per token cell, max 313; **242 of 640** token cells
receiving any ground-plane cell — reproduced verbatim by this run under **both** senses);
the cylindrical **120.000°** FOV and the `fov_census`; *"refcv5 has NO `SpatialGridReadout`"*.

---

## 7. ⛔ Scope — what this measurement is NOT

* ⛔ **It is an ORACLE.** The feature map is **BUILT FROM THE TARGET**. It prices a **readout
  geometry** under a perfect front-end. **No arm here is a perception result**, and no number here
  may be quoted as one.
* **Tier: NOT APPLICABLE** — a representation probe. No trajectory is produced, so T0/T1 do not
  apply.
* The **paired episode-cluster bootstrap answers *"would another draw of EPISODES say this?"***.
  The **replicate arms** answer *"would another TRAINING RUN say this?"* (`H-ESTIM-SEED-1`). Both
  are reported, separately, and the sense effect is read against the **replicate** floor — which is
  the one that binds here.
* Four-families note: this probe has no trajectory, so the longitudinal / tactical / strategic
  families are **not computable by construction** (n = 0 — there is no predicted trajectory, no
  manoeuvre decision and no route). The **lateral** family IS computable and is reported in §4.6
  rather than dropped.

---

## 8. ⛔ ESCALATION — a retracted claim is still driving an open PI decision

`Project Steering/PI_DECISION_QUEUE.md:156-159` still states, as **MEASURED**:

> *"on the real trunk none of it transfers — test AP 0.027–0.034 against a 0.0325 marginal control.
> ⇒ **the map does not yet contain agents.** An agent seam re-enabled today would be feeding a
> representation that cannot localise them, so the cheap order is to give the map spatial structure
> first."*

⛔ **Every sentence of that is retracted by `R-2026-09-08-wpa-mirror`** — the AP column is
mirrored-address, *"the map does not yet contain agents"* is FALSE, and the ordering conclusion it
supports is the VOID one. It is the input to an **open** P1 decision whose stated default is *"P1
stays out"*. ⇒ **This needs sweeping by the Master Mind before the PI reads that queue**; it is not
in this agent's scope to edit the PI's decision queue, and per the operating standard it is
escalated here rather than written into a README nobody re-reads.

⭐ **Conversely, four documents that inherit the ORACLE ladder need NO correction and can simply be
un-flagged**, because the ladder did not move:
`Project Steering/PREREG_WPB_WAYPOINT_INDEX.md:70`, `Project Steering/PI_DECISION_QUEUE.md:157`
(the `0.4713` half only), `…/2026-09-07-bev-from-front-camera/REFCV5C_DESIGN.md:52,63`,
`…/2026-09-08-wpb-waypoint-index/RESULT.md:43`. All four quote `AP 0.4713` as a perfect-front-end
cap; **that cap stands.**

---

## 9. Deliverable manifest

| artifact | where it lives | only one place? |
|---|---|---|
| `ORACLE_RECHECK.md` (this file) | `repo:…/2026-09-07-wpa-readout-localisation/ORACLE_RECHECK.md` | no — also `C:\Users\Admin\wpa-readout\oracle_recheck\` |
| `code/o1_oracle_mirror.py` — the panel (both senses + xwire + bootstrap) | `repo:…/code/o1_oracle_mirror.py` | no |
| `code/o2_tables.py` — table renderer (read-only) | `repo:…/code/o2_tables.py` | no |
| `code/o3_lateral.py` — LATERAL family, post-hoc | `repo:…/code/o3_lateral.py` | no |
| `raw/oracle_mirror.json` — every arm, every control, every paired CI | `repo:…/raw/oracle_mirror.json` | no |
| `raw/o1_oracle_mirror.log` — the run's own stdout incl. the runtime assertions | `repo:…/raw/o1_oracle_mirror.log` | no |
| `raw/address_xcheck.txt` — the three-implementation address table, executed | `repo:…/raw/address_xcheck.txt` | no |
| `raw/lateral.json` + `raw/o3_lateral.log` — lateral MAE per arm per range band | `repo:…/raw/` | no |
| `raw/test_scores.npz` — per-arm test-set probabilities (≈0.9 GB) | ⚠️ **`C:\Users\Admin\wpa-readout\oracle_recheck\raw\test_scores.npz` ONLY** | ⚠️ **YES** — too large to bank; regenerable in ~19 GPU-min from `o1_oracle_mirror.py` |
| WP-A's feature bank `k8r1` (input, not produced here) | `C:\Users\Admin\wpa-readout\bank\k8r1` (pre-existing) | unchanged by this work |

**Steering files touched (surgical, single-clause, staged):**
`Project Steering/GOALS_AND_CLAIMS.md` — `E-READOUT-CEILING-1`'s ⏳ NOT-QUOTABLE clause;
`Project Steering/RETRACTION_LOG.md` — `R-2026-09-08-wpa-mirror`'s ⏳ OPEN-and-owed clause.
⛔ Staged, **not committed, not pushed**.
