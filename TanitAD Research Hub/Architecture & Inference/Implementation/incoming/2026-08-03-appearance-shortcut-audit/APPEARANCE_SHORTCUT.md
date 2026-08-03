# D-APPEAR — ⛔ THE APPEARANCE SHORTCUT DOES **NOT** SURVIVE OFF-HIGHWAY. The programme-scale claim is WITHDRAWN.

**Date** 2026-08-03 · **0 pod GPU-h** — `tanitad-new` (v5f) and `tanitad-pod4` (v1arch) untouched.
All compute on the dev box (RTX 4060 for matmuls, CPU for the eigendecompositions),
`OMP_NUM_THREADS` set on every run.
**Pre-registration** `Project Steering/PREREG_APPEARANCE_SHORTCUT.md` — outcomes **S / C / P / VOID**
and every threshold fixed **before** any PhysicalAI number existed.
**Extends** `…/incoming/2026-08-03-latent-bottleneck/LATENT_BOTTLENECK.md` §0.0 (RANK 1, escalation 0).

| artifact | what it is |
|---|---|
| `results_p1_physicalai.json` | ⭐ **the pre-registered verdict** — 15 arms + the null on PhysicalAI-AV |
| `results_p1_physicalai.preds.npz` | ⭐ per-window predictions + gt + eid + speed + manoeuvre — re-analysable at 0 GPU, no refit |
| `results_p1b_mechanism.json` | ⭐⭐ **the mechanism** — the within-clip / across-clip ladder on BOTH corpora |
| `results_p2_rig.json` | the cross-rig contrast (G1 horizon row, G2/G4 transfer, G3 shift sweep) |
| `results_p3_sitclf.json` | the scenario-classifier probe |
| `results_p4_screen.json` | the promoted latent screen run on 4 substrates + an oracle |
| `raw/summary_tables.txt` + `summarize.py` | ⭐ every table below emitted FROM the JSON — nothing hand-transcribed |
| `raw/run_log*.txt` | full logs, including the crashed first P1 pass (kept, not hidden) |

---

## 0. THE ANSWER, in the order the brief asked for it

### 0.1 ⛔ P1 — **OUTCOME C (CORPUS-SPECIFIC).** The still frame reads speed at the NULL on PhysicalAI.

The pre-registered primary statistic is
`RATIO = R²_speed(pix32_centre_rbf) / R²_speed(v1_window)` on the same held-out windows,
**encoder-matched** — both corpora probed with the identical frozen `v1_speedjerk_ckpt.pt` step 29999.

| | comma2k19 highway | **PhysicalAI-AV** |
|---|---:|---:|
| `pix32_centre_rbf` — one 32×32 grey **still frame** | **+0.6642** | **−0.0025 = the empirical null** |
| `v1_window` — 18,432 features, 800 ms latent | +0.7145 | **+0.6752** |
| **RATIO** | **0.9296** | **−0.0037** |
| still-frame arm separates from its own shuffled control? | yes | **no — it does not separate at all** |

*(comma2k19 values MEASURED by the sibling stream,
`…/2026-08-03-latent-bottleneck/results_temporal_falsifier.json`, and **re-measured here through
this run's own code path** — `results_p1b_mechanism.json` reproduces `+0.6642` and `+0.7145`
bit-for-bit, so the cross-corpus contrast is one script, two corpora.)*

The run is **admissible**, not VOID: `v1_window` separates on `speed`
(Δ vs its own shuffled control **+0.6777 [+0.6328, +0.7235]**, paired episode-cluster bootstrap,
B = 2000, 80 held-out episodes), and the `NULL_train_mean` arm reproduces the floor at **−0.0025**.

⇒ **Pre-registered consequence, which I now execute: the programme-scale appearance-shortcut claim
is WITHDRAWN.** `LATENT_BOTTLENECK.md` §0.0's *"Appearance dominates motion by ~1.75× for reading
speed"* is **true of comma2k19 highway and false of PhysicalAI-AV**, and §7's warning that its
magnitude elsewhere is UNKNOWN was the correct reading.

### 0.2 ⭐⭐ AND THE INVERSION IS THE REAL FINDING — off-highway, **MOTION is the only pixel route**

The same panel, same split, same recipe:

| pixel arm | features | comma2k19 | **PhysicalAI** |
|---|---:|---:|---:|
| `pix32_centre_rbf` **still frame** | 1,024 | **+0.6642** | **−0.0025 null** |
| `stk32_centre_rbf` the 3 sub-frames of ONE index (300 ms) | 1,024 | — | −0.0025 null |
| `pix1_window_rbf` whole-frame mean intensity | 9 | −0.0052 null | −0.0025 null |
| `pix8_tdiff_rbf` adjacent-frame **difference** | 512 | +0.3778 | **+0.2492 SEP** |
| `mot8_centre_rbf` **motion energy, ONE instant, 64 features** | 64 | — | **+0.3707 SEP** |
| `mot8_window_rbf` motion energy, 800 ms | 576 | +0.5633 | **+0.3922 SEP** |
| `mot16_window_rbf` motion energy, 800 ms | 2,304 | +0.5200 | **+0.4124 SEP** |

On comma2k19 appearance beat the best motion arm by **1.75×**. On PhysicalAI appearance is at
**exactly the null in every form tried** — 1,024-feature, 9-feature, linear, rbf, single-instant,
within-stack — while **64 features of motion energy separate**. The ordering does not shrink;
**it reverses.**

### 0.3 ⭐⭐ THE MECHANISM, MEASURED — it is a **scene-memorisation** map, not a physical cue

The pre-registered ladder (`results_p1b_mechanism.json`) separates *"the probe is broken"* from
*"the map exists but does not transfer"*:

| corpus | arm | `within_clip` (random WINDOW split, **LEAKY by construction**) | `across_clip` (episode-disjoint — the real number) | retained |
|---|---|---:|---:|---:|
| comma2k19 highway | `pix32_centre_rbf` | **+0.9825** | **+0.6642** | **68 %** |
| comma2k19 highway | `pix32_centre` (linear) | +0.8745 | −0.0588 | 0 % |
| comma2k19 highway | `v1_window` | +0.8841 | +0.7145 | 81 % |
| **PhysicalAI** | `pix32_centre_rbf` | **+0.8023** | **−0.0025** | **0 %** |
| **PhysicalAI** | `pix32_centre` (linear) | +0.2546 | −0.0025 | 0 % |
| **PhysicalAI** | `v1_window` | +0.7235 | **+0.6752** | 93 % |

⚠️ `within_clip` is **not** a generalisation number and is never quoted as one — adjacent windows
overlap and share frames. Its only job is the one it does here: **the PhysicalAI pixel substrate is
not degenerate.** A still frame reads speed at **+0.8023** inside a clip and at **the null** across
clips.

⇒ **The appearance→speed map is real everywhere and TRANSFERS ONLY ON comma2k19.** The reason is a
property of that corpus, not of vision: comma2k19 val is **one driver, one vehicle, one camera, one
road class**, so an episode-disjoint split is *not* a domain-disjoint split — a map fitted on 33 of
its episodes still applies to the other 17. PhysicalAI-AV is 500 distinct clips across cities,
vehicles and **two camera rigs**; the same map transfers to nothing.
**The learned latent, by contrast, retains 93 % across PhysicalAI clips** — it is reading something
that generalises, and a still frame is not it.

⚠️ **This also qualifies the ORIGINAL comma2k19 finding, and that is the more useful half.** The
93 % figure was never wrong; what it measures is *"a still frame is 93 % as good as the latent
**when train and test share a rig, a vehicle and a road class**"*. Read that way it stops being a
claim about vision and becomes a claim about **comma2k19's episode-disjointness**.

### 0.4 P2 — the cross-rig collapse is **NOT re-explained by the appearance shortcut**, and it does not reproduce here

Three separate things, kept separate:

| # | measured | value |
|---|---|---|
| G1 | **horizon row of the cached frames**, rig A vs rig B (MEASURED from the pixels, not read from a build config) | argmax **129 vs 137** (offset **+8 rows** in 256-space); centroid **117.6 vs 109.0** (offset **−8.6**) |
| — | what a LEGACY geometric-centre crop would cost | rig B is ~215 px off in 1920×1080 ⇒ in a ~533 px crop resized to 256 that is **~100 output ROWS** |
| G2/G4 | `v1_window` speed R²: **A→A +0.7052 · B→B +0.7194 · B→A +0.7127 · A→B +0.7011** | **no cross-rig drop at all** |
| G2/G4 | `mot8_window_rbf`: A→A +0.4572 · B→B +0.3811 · **B→A +0.3991** · A→B +0.3843 | motion transfers |
| G2/G4 | `pix32_centre_rbf`: −0.0106 / −0.0465 / −0.0248 / −0.0177 | **at the null in every cell** — nothing to collapse |
| G3 | synthetic vertical shift, `mot8_window_rbf` fitted unshifted on rig A | +0.4571 → +0.4399 (4 rows) → +0.4145 (8) → **+0.3334 (16)** → +0.2226 (24) → +0.0745 (32) → **−0.1418 (48)** |

**What this establishes.** In *this* cache the two rigs' horizons agree to **8 of 256 rows**, i.e.
the ~100-row misalignment the collapse was attributed to **is not present here** — and with the
geometry aligned, the frozen v1 latent shows **no cross-rig speed drop whatsoever**.

**What the prior number actually is.** `…/2026-07-22-idm-proof/results.json`, read from source:
`experiments/rigA_to_rigB/val/in_rig_heldout_rigA/r2/speed` = **+0.7863** and
`…/cross_rig_rigB/r2/speed` = **−2.4654**. ⛔ **The widely-quoted "+0.930 → −2.465" pairs two
DIFFERENT experiments** — the +0.9297 is
`experiments/physicalai_to_comma2k19/val/in_corpus_heldout_paival/r2/speed`. The same experiment's
own in-rig baseline is **+0.7863**, so the collapse is **+0.7863 → −2.4654**, not +0.930 → −2.465.
*(Logged in `RETRACTION_LOG.md`, class C4/C6.)*

**⚠️ WHAT I CANNOT SEPARATE, stated plainly.** Two things differ between that run and mine — the
**cache geometry** and the **head** (theirs: a 2.9 M-param MLP, 10 epochs of SGD, which can
extrapolate to R² −2.47 off-domain; mine: an exact ridge with a train-mean fallback, which
structurally cannot). I did not re-run their head on their cache, so I **cannot attribute the
−2.4654 between those two causes**. Per the pre-registration that is the registered answer:
**CANNOT SEPARATE geometry from head-extrapolation.** What I *can* say, and it is what the brief
asked: **the appearance shortcut is not a live third explanation** — it is at the null in all four
rig cells, so there is no shortcut here to transfer or fail to transfer.

**G3 is informative in the direction that matters.** A pure geometric shift *does* destroy a motion
arm — 16 rows costs 27 % of the baseline and 48 rows drives it negative — so geometry is a
**sufficient** mechanism for a large cross-rig drop, and the legacy crop's ~100-row offset is well
past the top of that sweep. The G3 cell for `pix32_centre_rbf` is reported **VOID**: an arm at the
null has no skill to lose, so no degradation can be measured on it.

### 0.5 P3 — the scenario classifier is **NOT THREATENED** by the appearance→speed route

⭐ **Independently corroborated.** The sitclf stream's own still-frame control (same encoder, last
RGB frame replicated 3×, motion deleted) costs **~70 % of the skill on `intersection`**
(recovery 0.297 / 0.303 / 0.316, all separated). This audit reaches the same conclusion from a
completely different direction — the *speed* route rather than the *motion* route — on the same
banked substrate.

The shortcut's **first hop does not exist on this corpus** (clip-disjoint, 33,215 held-out frames,
167 clips): still 32×32 frame → speed **R² +0.0102**; frozen v1 latent → speed **R² +0.6900**.
With the first hop dead the second cannot carry anything, and the arms confirm it:

| situation | n_ho | pos | base | `img_latent` | `img_still32` | `ego_speed_true` | `speed_from_appearance` | `speed+img` | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lane_change | 27,293 | 630 | 0.0231 | **2.0261** | 1.0475 | 1.6853 | **1.0855** | 2.5104 | **NOT THREATENED** |
| roundabout | 27,231 | 355 | 0.0130 | **5.5192** | 4.4787 | 0.9952 | **1.0226** | 5.5332 | **NOT THREATENED** |
| intersection | 25,158 | 2,147 | 0.0853 | **2.3491** | 1.1314 | 1.2089 | **1.1691** | 2.4637 | **NOT THREATENED** |

*(AP-lift = AP / base rate, so **chance = 1.0**, and the columns are lifts not deltas.)*

| situation | shortcut share (**excess** lift) | shortcut share (raw lift, as pre-registered) | vision's incremental lift **over the true speed channel** |
|---|---:|---:|---:|
| lane_change | **+0.083** | +0.536 | **+0.825** (80 % of `img_latent`'s excess) |
| roundabout | **+0.005** | +0.185 | **+4.538** (100 %) |
| intersection | **+0.125** | +0.498 | **+1.255** (93 %) |

⚠️ **A DEFECT IN MY OWN PRE-REGISTRATION, reported rather than quietly fixed.** §3 wrote
*"≥ 50 % of `img_latent`'s AP-lift"* without stating that AP-lift's chance value is **1.0, not 0**.
On the raw form an arm sitting exactly at chance scores 0.50 of an arm at 2.00 while contributing
nothing, and `lane_change` would have been read **THREATENED** at 0.536. Both forms are in the JSON;
the verdict is taken on the **excess** form and the registered form is shown beside it.

⚠️ **Scope.** These are **single-frame** diagnostic arms; the deployed `head_img` consumes an
8-frame window through a PCA block. They are comparable **to each other**, not to the banked sitclf
AP. ⛔ I edited nothing under the sitclf tree and overturn no sitclf verdict.

⚠️ **The sitclf stream's caveat applies to me too and I adopt it:** the frozen trunk never saw a
degenerate stack, so any still-frame drop is an **UPPER BOUND** on motion's contribution, not a
point estimate. My `pix32_*` arms sidestep that (they are raw pixels, never fed through the trunk),
but `img_still32` above is a trunk-free 32×32 probe and is therefore a **lower** bound on what a
trunk could extract from a still frame. Neither is the point estimate.

### 0.6 P4 — the screen is now a repo rail, with tests, and its calibration went from 1 substrate to 5

`stack/tanitad/eval/latent_screen.py` + `stack/tests/test_latent_screen.py` (**12 tests, all
passing**). The load-bearing test is `test_collapsed_latent_fails_on_jitter`: a synthetic latent
built with no 100 ms resolution **must** be rejected, and rejected specifically on the jitter ratio.
A second test pins that a collapsed latent can still read **scene** speed well — which is precisely
why a speed-R² number can never be the gate, and why `dynenc-branchB` survived 40 k steps.

⚠️ **The σ gate travels with its estimator, enforced by a test.**
`test_sigma_estimator_travels_with_every_sigma` fails if any dict reporting a σ omits
`SIGMA_ESTIMATOR`, which records that **σ ≤ 0.28 m/s is the 9-point Savitzky-Golay form** and
*σ ≲ 0.1 m/s (~47×)* is the **same requirement under a 2-point centred difference**.

⚠️ One deliberate change from the reference script: `fit_stride` is now **adaptive**
(`max_fit_rows`, default 6,000) instead of hard-coded to 3. Pooling the window positions multiplies
rows by W, and a 40,000-row pooled dual is a ~2.5e12-flop `eigh` — tens of minutes, which would
defeat a minutes-scale pre-flight screen. The stride actually used is returned and recorded, so two
screens are never silently compared across different strides.

---

## 1. THE PANEL — PhysicalAI-AV, every arm against its OWN shuffled control

**Substrate.** 240 episodes of the PhysicalAI **r0 500-clip build**, `k = 4` (9-position,
NON-CAUSAL, 800 ms) windows at stride 6, episode-disjoint `i % 3` split →
**160 train / 80 held out = 4,787 / 2,390 windows**. That is deliberately matched to the comma
panel's 4,554 / 2,346 so a difference between corpora is not a difference in n.
Rigs present: **181 rig B / 59 rig A**.

⛔ **PARITY.** The corpus is `physicalai-train-14231cd29c74` + `physicalai-val-bb543bdf7836`
(the r0 500-clip probe build) — **NOT** the canonical training corpus
`physicalai-train-e438721ae894` (2,376 eps, skip-hash `f09e44db`). **Nothing here re-selects
episodes for any training arm**; this is a read-only probe over a banked cache and the distinction
travels with every number.

**Alignment is verified, not assumed** — this is the run's foundation, because the latent bank and
the pixel bank were produced by two different streams:

| # | check | result |
|---|---|---|
| **A1** | the latent bank's own ego speed vs the episode cache's `poses[:, 3]`, row for row, all 240 episodes | **max abs Δ = 9.54e-07 m/s** (float32 rounding) |
| **A2** | episode index → `clip_id`, reconstructed from `discover_r0_clips` + `split_clips`, checked against each cache's `episode_id` hash | asserted per episode, 0 failures |
| **A3** | per-clip `cy` from the local intrinsics parquets vs the independent `2026-07-22-idm-proof/rig_table.json` | **809 clips agree, 0 disagree** |

| arm | feat | kernel | speed R² | Δ vs its own shuffled control | sep |
|---|---:|---|---:|---|---|
| `NULL_train_mean` | 0 | — | **−0.0025** | (the empirical null) | |
| `v1_window` | 18,432 | linear | **+0.6752** | +0.6777 [+0.6328, +0.7235] | **SEP** |
| `v1_window_shufframes` | 18,432 | linear | **+0.6812** | +0.6837 [+0.6396, +0.7301] | **SEP** |
| `v1_centre` | 2,048 | linear | +0.6391 | +0.6416 [+0.5919, +0.6905] | **SEP** |
| `v1_tdiff` | 16,384 | linear | −0.0025 | +0.0000 | — |
| `pix32_centre_rbf` | 1,024 | rbf | **−0.0025** | +0.0000 | — |
| `pix32_centre` | 1,024 | linear | −0.0025 | +0.0000 | — |
| `stk32_centre` | 1,024 | linear | −0.0025 | +0.0000 | — |
| `stk32_centre_rbf` | 1,024 | rbf | −0.0025 | +0.0000 | — |
| `pix1_window_rbf` | 9 | rbf | −0.0025 | +0.0000 | — |
| `pix8_tdiff_rbf` | 512 | rbf | **+0.2492** | +0.2517 [+0.1351, +0.3450] | **SEP** |
| `mot8_centre_rbf` | 64 | rbf | **+0.3707** | +0.3732 [+0.2810, +0.4545] | **SEP** |
| `mot8_window_rbf` | 576 | rbf | **+0.3922** | +0.3947 [+0.2906, +0.4869] | **SEP** |
| `mot16_window_rbf` | 2,304 | rbf | **+0.4124** | +0.4149 [+0.3155, +0.5005] | **SEP** |

*(`Δ = +0.0000` with no interval is the degeneracy guard, not a measurement: the arm and its control
both hit the 0.01 skill gate and both emit the train mean, so their predictions are bit-identical
and "separated" is forced to False. See `summarize.py`/the JSON for the flag.)*

### 1a. ⭐ P1b's frame-order-shuffle control — the speed read uses **no 800 ms order at all**

Pre-registered rule: shuffle the 9 window positions per row; `|ΔR²| ≤ 0.05` ⇒ order-free.

`v1_window` **+0.6752** → `v1_window_shufframes` **+0.6812**. **Δ = +0.0060 ⇒ ORDER-FREE.**

⚠️ **Read this precisely, because the obvious reading is wrong.** It does **not** say the latent's
speed read is motionless. Each of the 9 window positions is itself computed from a **D-015 3-frame
stack spanning ~300 ms** (`in_channels = 9`), so shuffling the positions destroys the 800 ms
ordering while leaving every position's **internal 300 ms motion** intact. The finding is:
**the speed read lives inside the 300 ms stack, and the 800 ms ordering above it contributes
nothing.** That is consistent with `mot8_centre_rbf` — 64 features of single-instant motion energy —
separating, and with `stk32_centre` (the raw 3 sub-frames, no nonlinearity on their difference)
sitting at the null.

---

## 2. THE FOUR METRIC FAMILIES — per family, never pooled. ⛔ ADE is ONE ROW of four.

Reported for `v1_window`, the still-frame arm and the best motion arm, on the same held-out windows.
Full numbers in `raw/summary_tables.txt`; the JSON carries every field.

| family | what is reported | availability on this corpus |
|---|---|---|
| **LONGITUDINAL** | target-speed MAE + bias (scalar **and** trajectory-derived), along-track MAE + bias, accel MAE | ✅ **available**. ⛔ **distance-keeping / headway / TTC is UNAVAILABLE**: this substrate carries no lead-agent channel. **This is a WORK ITEM, not a pass** — PhysicalAI-AV *does* ship `obstacle.offline` on 97.44 % of clips, so the labels exist and the ingest is what is missing (`physicalai_r0.py` reads 4 of 36 features). n = 2,390 windows, 0 with a lead agent resolved. |
| **LATERAL** | heading MAE, **curvature MAE**, **yaw-rate MAE**, cross-track MAE + bias + final | ✅ **available** on all 2,390 windows |
| **TACTICAL** | manoeuvre-decision quality split **lateral / longitudinal / mixed** (the 5-way softmax that mixes them is the programme's largest known defect, so it is never reported as one number) + goal/anchor selection | ✅ **available** |
| **STRATEGIC** | route/goal-setting quality | ⛔ **UNAVAILABLE** — this probe emits no route or goal head and the substrate carries no route label. n = 2,390. **WORK ITEM.** The reason it cannot be computed here is structural, not a data gap: an IDM-style scalar+trajectory probe has no strategic output to score. |

⚠️ **The families do not change the verdict and were not used to reach it** — the pre-registered
statistic is a `speed` R² ratio. They are reported because an eval that reports one family is
incomplete, and because the LONGITUDINAL row is where the still-frame arm's failure is visible as a
*speed bias*, not only as a lower R².

---

## 3. STRATIFICATION — pre-registered, because a pooled number hides the regime

Reported per speed bin and per `refb.MANEUVER_CLASSES` manoeuvre, with `n` and `n_episodes`; bins
below **n = 100 windows or 5 episodes** are marked **UNPOWERED** and are not evidence.
Full table in `raw/summary_tables.txt` (`P1 — STRATIFIED`).

⚠️ **How to read it.** Within-stratum R² uses the **stratum's own** variance as its denominator, so
a narrow speed bin has a shrunken denominator by construction and its R² is **not** comparable to
the pooled value. **Read the ordering across arms within a row, never the level.**

The ordering is the same in every powered stratum as it is pooled: **`v1_window` > motion energy >
still frame at the null.** There is no speed bin and no manoeuvre class in which the still frame
recovers — which is the answer to the brief's stratification question. The corpus stream's
speed-dependent tactical lossy rate (38.2 % at 1–3 m/s → 1.8 % at 10–15 m/s) therefore does **not**
hide a regime where the shortcut lives.

---

## 4. WHAT THIS DOES **NOT** ESTABLISH

* ⛔ **v5f was NOT screened and NOT probed.** No banked v5f latent exists for these clips; producing
  one needs the v5f checkpoint and a GPU pass over the wide-FOV cylindrical cache. **This is the
  single largest gap in the audit** and it is the arm the programme is actually deciding about.
* ⛔ **REF-C's ResNet trunk was NOT screened.** Same reason.
* ⛔ **The cross-rig −2.4654 is NOT attributed.** Geometry and head-extrapolation both remain live;
  separating them needs their head re-run on this cache, which I did not do.
* ⛔ **No claim about `long_accel` is reopened.** OUTCOME V is pre-registered elsewhere and this run
  does not bear on it.
* ⚠️ **One encoder.** Every latent here is `v1_speedjerk_ckpt.pt` step 29999. P4 widens the screen's
  **corpus** calibration from 1 to 5 substrates; the **encoder** calibration is still n = 1.
* ⚠️ **The PhysicalAI corpus is the r0 500-clip probe build**, not the canonical 2,376-episode
  training corpus. The contrast is corpus-level and does not transfer to a training arm without
  re-running on the canonical cache.
* ⚠️ **The still-frame arms are raw 32×32 pixels**, not a degenerate stack through the frozen trunk.
  Those two controls bound the same quantity from opposite sides and neither is a point estimate.

---

## 5. ESCALATIONS — these need someone else to act

0. ⭐⭐ **`LATENT_BOTTLENECK.md` §0.0 and its RANK-1 framing must be amended.** The sentence
   *"Appearance dominates motion by ~1.75× for reading speed"* is **corpus-specific**; off-highway
   the ordering **reverses** and the still frame is at the null. Its own §7 already warned this was
   possible — the amendment is to make the measured answer visible where the claim lives, not to
   blame the framing. **Owner: the latent stream.**
1. ⭐⭐ **`Project Steering/MODEL_REGISTRY.md` and every doc quoting "+0.930 → −2.465" must be
   corrected to "+0.7863 → −2.4654".** The +0.9297 belongs to a *different* experiment
   (`physicalai_to_comma2k19/in_corpus_heldout_paival`). Logged in `RETRACTION_LOG.md`.
2. ⭐ **`Project Steering/GATE_PROTOCOL.md`: adopt `tanitad.eval.latent_screen` as a pre-flight gate**
   for any encoder-training authorisation. It is now a repo instrument with 12 contract tests, so
   this is a protocol edit, not an implementation task. *(Second stream to raise this; the first was
   the latent stream. It has now been asked for twice and is not mine to make.)*
3. ⭐ **`BACKLOG.md` B5 (frozen V-JEPA 2)**: re-scope to run the screen FIRST. This audit adds a
   second reason — its acceptance criterion should be **motion-energy-beating speed transfer across
   clips**, which is now a measured, corpus-robust target (`mot16_window_rbf` +0.4124 on PhysicalAI)
   rather than the comma-specific appearance number.
4. ⛔ **LONGITUDINAL distance-keeping / headway / TTC is a WORK ITEM with a known fix.**
   PhysicalAI-AV ships `obstacle.offline` on **97.44 %** of clips; `physicalai_r0.py` reads **4 of
   36** features and no lead-agent channel reaches the episode cache. Until it does, every
   longitudinal eval on this corpus reports one of its two halves. **Owner: the data/ingest stream.**
5. ⚠️ **The sitclf stream should note the corroboration**: their still-frame control (motion deleted,
   ~70 % of `intersection` skill lost) and this audit's speed-route probe (shortcut share 0.005–0.125
   excess, vision's incremental lift over the true speed channel +0.83 to +4.54) agree. **Neither is
   a point estimate** — theirs is an upper bound on motion's contribution, mine a lower bound on what
   a trunk could take from a still frame.
6. ⚠️ **`PREREG_APPEARANCE_SHORTCUT.md` §3's decision rule is defective as written** (AP-lift's
   chance value is 1.0, not 0). Corrected in the JSON and in §0.5; the pre-registration file is left
   **unedited on purpose** so the registered rule stays auditable.

---

## 6. REPRODUCTION

```bash
cd "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-03-appearance-shortcut-audit"
OMP_NUM_THREADS=4 python build_pai_substrate.py --n-episodes 240 --stride 6           # ~130 s
OMP_NUM_THREADS=2 python build_pai_substrate.py --rig a --n-episodes 120 --stride 6 \
    --out C:/Users/Admin/tanitad-data/eval/dappear_rigA.pt
OMP_NUM_THREADS=2 python build_pai_substrate.py --rig b --n-episodes 120 --stride 6 \
    --out C:/Users/Admin/tanitad-data/eval/dappear_rigB.pt
OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 python run_p1_offhighway.py  --n-boot 2000
OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 python run_p1b_mechanism.py
OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 python run_p2_rig.py        --n-boot 2000
OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 python run_p3_sitclf.py     --n-boot 2000
OMP_NUM_THREADS=6 PYTHONIOENCODING=utf-8 python run_p4_screen.py
python summarize.py > raw/summary_tables.txt
cd ../../../../../stack && pytest -q
```

⚠️ **`PYTHONIOENCODING=utf-8` is not decoration.** On this Windows box a redirected stdout defaults
to cp1252 and a `⭐` in a log line raises `UnicodeEncodeError` **after** the compute is finished —
it killed the first P1 pass at the final summary line and cost a 22-minute re-run. The crashed log
is kept at `raw/run_log_p1_crashed_unicode.txt`. The runners now log ASCII only; the env var is
belt-and-braces.
