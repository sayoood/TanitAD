# TanitAD — program report, 2026-07-27 17:57 Berlin (15:57 UTC)

**Previous report: `2026-07-26-1257`.** The 2026-07-26 17:57 slot was missed; this report covers the
interval since 2026-07-26 12:57.

⚠️ **Scope honesty, stated first.** `LOOP_STATE.md` is dated **2026-07-26 ~17:0x UTC** and is a day
stale. Everything below marked **MEASURED (today)** was verified by me in this iteration against raw
artifacts or a live pod probe. Streams I did **not** re-verify today are marked
**INHERITED (LOOP_STATE, not re-verified)** and must not be quoted as current.

⛔ **LOOP_STATE's own headline is now superseded.** It reads *"NO ADMISSIBLE GATE HORIZON IS A
MEASUREMENT, INCLUDING THE K=20 WE ALREADY USE"* (out-of-envelope K=20 **12.26 %**, K=185 **90.24 %**).
**Pseudo-simulation closed that wound today: 0.00 % out-of-envelope, verdict `MEASUREMENT`, on all 10
arm processes.** The EXTRAPOLATION stamp comes off the closed-loop co-primary.

---

## 1. Fleet — MEASURED (today, live `ssh` probe, native OpenSSH)

| host | GPU | state |
|---|---|---|
| `tanitad-pod` (pod1) | **100 % / 15,348 MiB** | 🟢 **`flagship-v2corpus-30k`, step 16,750 / 30,000 (55.8 %)**, elapsed **2 d 02:31**, PID 699286. Ckpts at 5000/15000/latest. |
| `tanitad-pod2` | **100 % / 25,899 MiB** | 🟢 **FOV model-side sweep**, 4 × `fov_extract.py` shards, elapsed 1 h 04 m. |
| `tanitad-pod3` | **0 % / 0 MiB** | 🔴 **WAS IDLE → refilled this iteration.** Repo drifted at `0f93b98`. |
| `tanitad-eval` | **0 % / 0 MiB** | 🔴 **IDLE with 5 zombie processes** (PIDs 1487279/1487782/1487785/1487926/1487931, alpasim, **0.0 % CPU for 4 days** — the documented futex deadlock). Reap queued. |

⚠️ **`step_s 538.8` on pod1 is ACCUMULATED over `--log-every 50` → 10.78 s/step.** Not a pathology —
the documented false-alarm trap. 336 log lines × 50 = 16,800, consistent. **~40 h to 30k.**

**Idle capacity was found and refilled in the same turn**, per the standing rule that a report is not
a launch: pod3 → IDM `steer` retrain on the 160-clip corpus; pod2 → wide-FOV cache build **queued
behind** the sweep; eval → reap then free.

---

## 2. What landed since the last report — 8 streams, every estimator named

**Estimator for every interval below: paired episode-cluster bootstrap, `taniteval/ci.py`, B = 2000,
unit = episode. `overlapping_holdout_se` was used nowhere.**

### 2.1 🔴 Closed loop — a zero-parameter baseline ranks FIRST (MEASURED, today)
`…/incoming/2026-07-27-pseudosim-arm-panel/`. 10 arms, identical 15,981 rows, row identity asserted;
**all five pre-registered gates passed**; **0.00 % out-of-envelope ⇒ `MEASUREMENT`.**

`cv_holdv0` **0.5705** > `v4_oracle` 0.5622 > `refc_xl` 0.5499 > `v1` 0.5471 > `refc_small`/`base`
0.5444/0.5439 > `nospeed` 0.5394 ≫ `v4_blind` 0.3749. Standing-still adversary: `composite()`
**refuses to emit** (`VacuousMetric`) — the anti-gaming guard held.

**CV is separated above REF-C at all three scales (+0.0203…+0.0255) and NOT separated from either
flagship arm.** ⭐ **Mechanism found: `TacticalPolicy.forward(states, ctx)` HAS NO ACTION INPUT** —
`nospeed − v1 = −0.0055, n.s.`, against a **6.5×** open-loop speed ablation. The planner cannot see
the channel we proved matters.

### 2.2 ⛔ The 2 s selection surface is EXHAUSTED — four refutations, one mechanism
- **Fan conditioning REFUTE** (`…/2026-07-27-fan-conditioning/`, replicated on 4 fans). **100.0 % of
  windows already contain a candidate within 0.5 m/s of the speed taken** (mean gap 0.0525 m/s);
  restricting the oracle to speed-matched candidates moves **+0.0000 m [0.0000, 0.0000]**.
- ⭐ **An in-sample anchor set with a PERFECT ceiling (0.0000) still realises only 0.4167 ⇒ 100 % of
  the residual is SELECTION.**
⇒ **Do not fund CoverNet anchors, longitudinal admissibility filtering, or further re-scoring of the
frozen 2 s fan.**

### 2.3 ⛔ The 88 % goal result is an ORACLE ARTIFACT — and the redirect is worth more
`…/2026-07-27-goal-input/`. **Tautology test is a NULL, not underpowered:** selector **1.0061
[0.827, 1.193]** vs learned head **1.0028 [0.846, 1.169]**, paired **−0.0034 [−0.1169, +0.1232]**,
needing **≈ 5 × 10⁴ episodes** to separate. **Predicting the 2-D endpoint IS the picking problem.**
Break-even σ₀ **0.955 m**; achieved **1.330 m ⇒ −10.4 %**; the latent-only head is **separated-WORSE
(+0.0464 [+0.0164, +0.0792])** on all three fans.

⭐ **Decomposed: oracle along-track +83.7 % (separated), oracle cross-track +2.9 % (NOT separated).**
⇒ **A map / route / lane graph supplies the 2.9 % axis. "We need AlpaSim or an external mapped
corpus" was aimed at the wrong axis.**

### 2.4 ⭐ E-GOAL-1 — the longitudinal axis IS realisable, from EGO SPEED HISTORY
`…/2026-07-27-egoal-1-lead-vehicle/`. ⛔ **Lead-vehicle state REFUTES (CONFIRMED tier)** — **41.65 %
of windows have no vehicle ahead within 50 m**; RMS 0.9305 → 0.8983 (**+0.0322 [−0.0040, +0.0969],
not separated**), worth **+2.3 points**.

⭐⭐ **But run through the ACTUAL rule, a head with the measured error structure recovers +23.6 % of
the fan's headroom FROM EGO KINEMATICS ALONE (−0.0638 [−0.1271, −0.0008], separated), +25.9 % with
lead — while the parent's head through the IDENTICAL construction is separated-WORSE at −33.1 %.
`dv_*`/`v_lag_*` alone are worth +0.1428 m [+0.0686, +0.2516] — 4.4× the entire lead block. The
parent's head had `v0` and NO HISTORY.** ⇒ the earlier −10.4 % was a **missing-feature** result.
**Tier: PROVISIONAL** — the conservative `by_speed` resampler gives the same sign and size
(+20.3 %/+22.9 %) but is **not separated**.

### 2.5 ⭐ The one DEPLOYABLE win — C2 ungated
`…/2026-07-27-canary-proxy/`. **C2 with v1's world model, UNGATED: 0.8563 → 0.5196,
−0.3366 [−0.4507, −0.2310], separated, out-of-fold. A 39 % cut, zero training, no oracle anywhere.**
The canary proxy **exists** (R² **0.5526** / Pearson 0.7518 vs **0.070** for `v0`) but is **dominated
by its own prerequisite** — it needs 2-WM ensemble features, and with a second WM you score ungated
for 2.1× more. `v0` as a gate is **separated-WORSE than doing nothing (+0.0411 [+0.0133, +0.0752])**.

### 2.6 ⭐ IDM v3 — the worst channel was a BROKEN LABEL, not a model failure
`…/2026-07-27-idm-v3/`. comma2k19 heading is `arctan2` of ENU velocity, **undefined at standstill**:
**26.27 % of frames below 0.5 m/s are physically impossible, 0.000 % above it** (PhysicalAI: zero in
every bin). Repairing it moves the **already-deployed head, nothing retrained**, from pooled
`yaw_rate` **R² 0.105 → 0.811**; retrained **0.841**.

**Shipped:** speed **0.907** · yaw **0.841** (**−0.0060 [−0.0090, −0.0029] separated, −22 % MAE** vs
deployed) · steer **0.408** 🔴 **regressed from 0.742** (data budget, 68 vs 160 clips — **do not
replace the deployed steer**). `long_accel` **closed and removed from the contract** — HL-Gauss ran as
*declared additional* arms, all fail at best R² −0.25, and it is decisive because the symexp variant's
**discretisation ceiling is 0.9999** and it still reads −0.339.

**HF: `Sayood/tanitad-idm-head-v3`, private, weights-only.** Gating is **UI-only — needs Sayed.**

**Extrinsics: REFUTED**, four routes, three controls. **Camera height is per-clip, 1.245–1.607 m,
29 % spread; all three circulating constants (1.5/1.43/1.22) are wrong — 1.22 is below the observed
minimum. Rig identity is NOT a proxy** (rig medians 1.5 % apart, within-rig spread 29 %).

### 2.7 ⭐ Geometry — the PI approved 100–120°, and storage INVERTED
`…/2026-07-27-geometry-configurable/` + `…/2026-07-27-fov-crop-audit/` + `…/2026-07-27-encoder-tokenization/`.

| geometry | raw | **PNG lossless** |
|---|---:|---:|
| deployed 256² @ 51.4° | 349 GB | **44.8 GB** |
| ⭐ **120° 256×640 cylindrical** | 873 GB | **112.9 GB** |
| 120° 384×960 cylindrical | 1965 GB | **221.9 GB** |

⭐⭐ **120° lossless is 112.9 GB — under a third of today's 349 GB at 51.4°. Tripling the field of
view SHRINKS the corpus 3×, bit-exactly.** Provisioning request **withdrawn**. PNG **decodes faster
than JPEG** (1.60 vs 2.96 ms/frame); JPEG max abs error **68/255 even at q95**.

- ⛔ **100° is IMPOSSIBLE at 256×256 — it silently delivers 67.1°** (crop needs 1595 px, clamps at
  1080). Widening **requires** more columns. Now a permanent test.
- ✅ **Selection-vs-cache MEASURED:** a re-crop invalidates **only the cache** — proved by reproducing
  the real on-disk key `14231cd29c74`. ⚠️ But `parity.corpus_key_of()` matches on **directory name**,
  so a re-cropped cache reads NON-PARITY and the trainer **refuses** it.
  **Runbook: rebuild → `register_geometry_sibling()` → commit manifest → train.**
- **The crop IS costing us, at intersections only:** decision-relevant cross traffic **6.192×
  [1.738, 13.155]** in the cropped-away band; **93.6 % of it missed today**; ≥100° is on the right
  side of a knee (**70° recovers 3.4 %, 100° 31.3 %**). ⚠️ **Honest against the case: lane changes
  show LESS peripheral content (0.759 [0.528, 0.997]).**
- **Encoder is 33.0 % of 263.4 M; widening 256 → 640 tokens costs +294,912 params (+0.34 %)** —
  sub-300M is **not** the constraint. **Cylindrical reaches 96.5–110.3° at 448–512 tokens** vs pinhole
  **640** for 100.5°.
- **Recommendation: 256×640 cylindrical (the full 120°), `f_eff` 266, patch 16, PNG. Adopt NO
  tokenization trick.**

### 2.8 Latent action models — NOT ADOPTED
`…/Research/2026-07-27-latent-action-models/`. 2 of 4 pre-registered refutation criteria fired.
**The sign is inverted for us:** UniVLA / MVP-LAM / "Why Latent Actions Fail" gain by **suppressing**
ego motion (88.7 vs 56.5; LIBERO-Long 79.4 vs **0.2**) — in driving **ego motion IS the action**.
LFG, Vista, VPT and DriveVA exploit our exact label asymmetry and **none** uses a latent-action
bottleneck.

### 2.9 Streams NOT re-verified today — INHERITED (LOOP_STATE), do not quote as current
**E1c / closed-loop CL-SFT · 4-brain dominance Gates A/B/C · H2 sensor attention · Orin/Thor ·
AlpaSim · YouTube D-B · the v2 50 h corpus build · TanitDataSet HF push.** No fresh probe was run on
these in this iteration. ⚠️ **AlpaSim's only visible footprint on `tanitad-eval` today is 5 processes
at 0.0 % CPU for 4 days** — i.e. its worker pool is dead, whatever its last state was.

---

## 3. Retractions — 14 new root-cause classes in one day (`RETRACTION_LOG.md`)

An unusually high count, and it is the boost programme working rather than a regression: **three
separate streams caught bugs in their own scoring code**, and **two pure-noise gates would have been
written up as separated wins** without the firing-rate column their pre-registration demanded.

| class | one line |
|---|---|
| **C16** fabricating intermediary | a PDF fetch **invented a verbatim quote, a section name and 3 numbers** — and it was the most load-bearing claim in the brief. `PUBLISHED` now requires a stated fetch depth. |
| **C17** marginal mistaken for conditional | two streams converged on the wrong lever; **both measured a marginal**, convergence read as confirmation. |
| **C18** correlation without slope | **r = −0.974** sat on a slope of **−0.129** against a required **+1.0**. |
| **C19** a stratum win is not a deployable win | the 53 % stratum result is **−0.0852 deployed**; quoted bare it overstates **4.4×**. |
| **C20** optimise the objective you are paid for | aiming at the canary rather than utility costs **3.6×**. |
| **C21** a docstring is not a measurement | **mine** — "the flagship trains on a comma+PhysicalAI mix" came from a docstring. It trains on **PhysicalAI alone, 100 %**. |
| **C22** bound quoted as capability | the 88 % oracle; the achievable version is separated-**worse**. |
| **C23** oracle shaped as ego state | `head_deg` is **future** heading change, sitting beside `v0`. **Caught pre-fit.** |
| **C24** RMS placed on a noise curve | over-predicted damage **5.7×**. |
| **C25** an unpaired ladder quoted as a measured effect | **"vision enters at rank ≈ 16" STRUCK from VALIDATED.** |
| **C26** a rig-correlated fabrication in the deployed input | pads **0.00 % rig A / 11.21 % rig B**, in **every number since D-016 R1**. |
| **C27** real-vs-shuffled measures harm avoided | needs a **NONE** arm or it reads as "geometry works, p<0.05". |
| **C28** a constant where the quantity is per-clip | three `cam_h` constants, all wrong. |
| **C29** the model was right and the label was wrong | R² 0.105 → 0.811 with **nothing retrained**. |

---

## 4. Decisions owed by Sayed

1. 🔴 **v5 input geometry — final shape.** Measurement says **256×640 cylindrical = full 120°,
   640 tokens, ~2.8× compute, 112.9 GB.** Storage would also allow **384×960** (120° *and* ~1.5×
   angular resolution, 221.9 GB — still under today's footprint) but that is **~1440 tokens and
   multiple GPU-weeks.** *Recommendation: 256×640 unless the resolution is worth the training time.*
2. 🔴 **The rig-padding confound (C26).** Cylindrical removes the fabrication but leaves a
   **rig-correlated mask** — still a rig-correlated signal. The clean fix is a vertical field both
   rigs fully observe; **made expressible, deliberately not chosen**, because choosing it costs a
   measurement nobody has run.
3. 🟠 **Enable HF gating on `Sayood/tanitad-idm-head-v3`** — UI-only, needs his machine.
4. 🟠 **Re-issue every published comma `yaw_rate`** — the deployed head's is **0.105 → 0.811**, so
   anything quoting it is stale.
5. **Carried from LOOP_STATE, not re-verified today:** v4 restart lever & budget · P1/renderer
   re-validation on the yaw arm · corridor **1.391** as a 2nd grid row · AV2 ~147 MiB pull ·
   **ZOD access application** (the only commercially-usable candidate) · wheelbase **B** ·
   +17 scenes for HP-4 · ~103 scenes strategic · **nuScenes Terms + HF cleanup — both need his
   computer, BLOCKED indefinitely.**

**NOT AUTHORIZED and not done:** the 30-pod-day X2 verdict run · the wheelbase fix · any deletion ·
HF publishing beyond what is already decided.

---

## 5. Blocked, and on what

| item | blocked on |
|---|---|
| v5 training run | **the small validation** (PI's own condition) → which is blocked on the wide cache → **queued on pod2 behind the FOV sweep (~1 h)** |
| `LakeRecord.image_size` | a **scalar that cannot express a non-square frame at all** — schema bump owed before the default flip |
| FOV model-side sweep verdict | running; returned **`REFUSED`** at n=20 clusters against a bar of 40 and **the bar was not lowered**. Now at **512 held-out clips / 80 intersection events**. |
| nuScenes, HF cleanup | Sayed's machine — indefinite |
| `fanc_goal_decomp.json` | **no producing script staged anywhere** (probed twice) — the original "−31.4 %/−5.8 %/+88.0 %" is **unreproducible from the repo**. Substance independently re-derived; the artifact must be reproduced or withdrawn. |

---

## 6. What I would do next if uninterrupted — priority order

1. **Build the wide cache on pod2** the moment `fov_extract.py` exits, then **register the geometry
   sibling key**. Nothing about v5 can proceed without it. *(Launched, gated on the sweep.)*
2. **Run the PI's small validation** — matched short runs, old geometry vs 256×640 cylindrical, on
   identical episodes, **primary = the map-free composite, NOT `ade_0_2s`**, with its bar, n **and
   MDE** pre-registered. ⚠️ A validation whose MDE exceeds the effect it must detect is a guard that
   cannot fail (C13), and we have shipped several.
3. **E-GOAL-2: re-run the speed-history head at n = 600.** The +23.6 % is **PROVISIONAL**; the effect
   separates at **≈ 222 episodes** and **the 600-episode parity-preserving val build already exists.**
   This is the cheapest upgrade from PROVISIONAL to CONFIRMED in the programme.
4. **Wire `C2` ungated into the v5 selector** — the only deployable win on the board (**−0.3366**,
   zero training) and it is independent of every open geometry question.
5. **Give `TacticalPolicy` an action input.** It is the mechanism behind §2.1's null and it is a
   single, well-localised change with a measured motivation.
6. **Finish the FOV model-side sweep** and hold the 40-cluster bar.
7. **Reap `tanitad-eval`** and put the small-validation runs on it.
8. **Update `LOOP_STATE.md`** — its headline is superseded (§0) and its fleet table is a day wrong.

**Restart budget: v4 stays 0 / 2 — unspent, not forfeited.**
