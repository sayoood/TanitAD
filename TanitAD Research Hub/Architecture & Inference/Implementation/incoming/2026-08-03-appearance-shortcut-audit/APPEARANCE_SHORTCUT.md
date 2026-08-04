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

**RATIO CI95 (episode-cluster bootstrap, both arms resampled jointly over the 80 held-out episodes,
B = 2000, 2000/2000 finite draws): [−0.0498, −0.0000].** The whole interval is below the
OUTCOME-C threshold of 0.40, so the verdict does not depend on the point estimate.
**Paired Δ (still − latent): −0.6777 [−0.7235, −0.6328], SEPARATED.**

The run is **admissible**, not VOID: `v1_window` separates on `speed`
(Δ vs its own shuffled control **+0.6777 [+0.6328, +0.7235]**, paired episode-cluster bootstrap,
B = 2000, 80 held-out episodes), and the `NULL_train_mean` arm reproduces the floor at **−0.0025**.

⇒ **Pre-registered consequence, which I now execute: the programme-scale appearance-shortcut claim
is WITHDRAWN.** `LATENT_BOTTLENECK.md` §0.0's *"Appearance dominates motion by ~1.75× for reading
speed"* is **true of comma2k19 highway and false of PhysicalAI-AV**, and §7's warning that its
magnitude elsewhere is UNKNOWN was the correct reading.

### 0.2 ⭐⭐ AND THE INVERSION IS THE REAL FINDING — off-highway, **MOTION is the only pixel route**

The same panel, same split, same recipe:

All values below are **held-out speed R²** read from the two runs' JSON by `summarize.py`
(`raw/summary_tables.txt`, *CROSS-CORPUS CONTRAST*) — the same arms, the same encoder, two corpora:

| pixel arm | features | comma2k19 | **PhysicalAI** |
|---|---:|---:|---:|
| `pix32_centre_rbf` **still frame** | 1,024 | **+0.6642** | **−0.0025 null** |
| `pix32_centre` still frame, linear | 1,024 | −0.0588 | −0.0025 null |
| `stk32_centre` the 3 sub-frames of ONE index (300 ms) | 1,024 | −0.1016 | −0.0025 null |
| `pix1_window_rbf` whole-frame mean intensity | 9 | −0.0052 null | −0.0025 null |
| `pix8_tdiff_rbf` adjacent-frame **difference** | 512 | +0.3778 | **+0.2492 SEP** |
| `mot8_centre_rbf` **motion energy, ONE instant, 64 features** | 64 | — | **+0.3707 SEP** |
| `mot8_window_rbf` motion energy, 800 ms | 576 | +0.5582 | **+0.3922 SEP** |
| `mot16_window_rbf` motion energy, 800 ms | 2,304 | +0.5148 | **+0.4124 SEP** |

⚠️ **A transcription trap this table already caught.** `LATENT_BOTTLENECK.md` §0.0 lists
`mot8_window_rbf +0.5633` and `mot16_window_rbf +0.5200` — those are the arms' **paired Δ vs their
shuffled control**, not their **R²** (+0.5582 / +0.5148). The two differ by the control's own score
and are one column apart in the JSON. Every number in this document is emitted by `summarize.py`
straight from the JSON for exactly this reason.

On comma2k19 the still frame beat the best motion arm by **1.19×** on R² (+0.6642 vs +0.5582; the
"~1.75×" in `LATENT_BOTTLENECK.md` compares against `pix8_tdiff_rbf` **+0.3778**, the best
*difference*-basis arm, not against motion energy — both comparisons are in the JSON and they say
different things). On PhysicalAI appearance is at **exactly the null in every form tried** —
1,024-feature, 9,216-feature, 9-feature, linear, rbf, single-instant, within-stack — while
**64 features of motion energy separate**. The ordering does not shrink; **it reverses.**

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

### ⭐ Two rival explanations, both KILLED by measurement rather than by argument

**(a) "the PhysicalAI pixel substrate is broken/degenerate."** No — measured
(`P1b — substrate integrity`): `pix32` centre-frame mean **0.3059**, std **0.1746**, range
**[0.000, 1.000]**, **0 constant features**, median per-feature std **0.14452** — *higher* dynamic
range than comma2k19's (mean 0.2502, std 0.1673, median feature std 0.08453). And the same block
reads speed at **+0.8023** within-clip. The pixels are fine.

**(b) "PhysicalAI's speed distribution is too narrow for a shortcut to be visible."** No — measured
(`P1b — speed distributions`): the **coefficient of variation is essentially identical**,
comma2k19 **0.589** vs PhysicalAI **0.621**. The *levels* differ enormously (mean **19.56 m/s**,
p50 23.30, highway vs mean **6.02 m/s**, p50 6.31, urban) and PhysicalAI actually spends **more**
time near standstill (frac < 1 m/s: **0.152** vs 0.097), but the relative spread a regressor has to
explain is the same. The null is not a range artifact.

⇒ What is left is the one thing that does differ: **how many distinct scenes, vehicles and cameras
the held-out set contains.** That is the mechanism.

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

**The paired intervals** (cross-rig minus within-rig, **same held-out rows**, episode-cluster
bootstrap B = 2000; 116 rig-A / 120 rig-B episodes, `cy` means **542.58** and **753.81**):

| arm | on rig A (n_ep 39) | on rig B (n_ep 40) |
|---|---|---|
| `v1_window` | **+0.0075 [−0.0318, +0.0502]** not separated | **−0.0183 [−0.0555, +0.0120]** not separated |
| `mot8_window_rbf` | −0.0580 [−0.1400, +0.0228] not separated | +0.0031 [−0.0810, +0.0808] not separated |
| `pix32_centre_rbf` | −0.0142 [−0.0678, +0.0380] not separated | +0.0288 [+0.0014, +0.0593] *separated* |

⚠️ **The one "separated" cell is a separation between two NULLS** (−0.0465 vs −0.0177, both below
zero) and carries no information about speed. It is shown because suppressing an inconvenient
`separated` flag is worse than explaining it.

**What this establishes.** In *this* cache the two rigs' horizons agree to **8 of 256 rows**, i.e.
the ~100-row misalignment the collapse was attributed to **is not present here** — and with the
geometry aligned, the frozen v1 latent shows **no cross-rig speed drop whatsoever**, with a paired
interval that excludes anything larger than ±0.06 in either direction.

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

| situation | shortcut share (**excess** lift) | shortcut share (raw lift, as pre-registered) | vision's incremental lift **over the true speed channel** | paired `img − speed_from_appearance` | paired `speed+img − speed` |
|---|---:|---:|---:|---|---|
| lane_change | **+0.083** | +0.536 | **+0.825** (80 % of `img_latent`'s excess) | +0.941 [−0.199, +2.789] **not sep** | +0.825 [−0.120, +2.196] **not sep** |
| roundabout | **+0.005** | +0.185 | **+4.538** (100 %) | +4.497 [−0.171, +14.863] **not sep** | +4.538 [+0.228, +15.061] **SEP** |
| intersection | **+0.125** | +0.498 | **+1.255** (93 %) | **+1.180 [+0.678, +1.794] SEP** | **+1.255 [+0.848, +1.990] SEP** |

⚠️ **Only `intersection` is properly powered, and I will not dress the other two up.** It is the
only situation where **both** paired intervals separate (2,147 positives, base rate 0.0853).
`lane_change` (630 pos) and `roundabout` (355 pos) give the same point estimates in the same
direction but their intervals include zero, so on those two the verdict rests on point estimates.
The registered outcome is NOT THREATENED on all three; the **evidence strength is
intersection ≫ lane_change ≈ roundabout**.

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

**✅ It reproduces its own reference through the promoted module** (`results_p4_screen.json`,
`reproduction_check`), which is the test of a promotion rather than a fork:

| quantity | reference (`…/latent-bottleneck/results_mechanism.json`) | promoted module |
|---|---:|---:|
| jitter ratio | 51.0 | **50.9644** |
| derivative corr | +0.0891 | **+0.08907** |
| derivative corr, per-position | +0.0061 | **+0.00624** |
| cos adjacent 100 ms | 0.98825 | **0.98825** (exact) |
| derived accel R² | −0.3773 | −0.40866 |

*(The accel R² differs because the module uses the **fixed** 9-point SG stencil while the reference
**swept** the stencil on the held-out set. The module reports the swept value too, labelled
`upper_bound_best_stencil` and marked CHEATING BY CONSTRUCTION. Difference reported, not tuned away.)*

**The fleet pass — 4 substrates + an oracle. The oracle PASSES, so the pass is admissible.**

| latent | jitter (≤2) | dcorr (>0.50) | accel R² (>+0.50) | σ m/s (≤0.28) | cos 100 ms | verdict |
|---|---:|---:|---:|---:|---:|---|
| frozen v1 · comma2k19 | **50.96** | +0.0891 | −0.4087 | 5.943 | 0.9882 | **FAIL** ×4 |
| frozen v1 · PhysicalAI mixed | **7.42** | +0.4019 | +0.1677 | 2.162 | 0.9904 | **FAIL** ×4 |
| frozen v1 · PhysicalAI rig A | 8.19 | +0.3659 | +0.1391 | 2.028 | 0.9920 | **FAIL** ×4 |
| frozen v1 · PhysicalAI rig B | 10.49 | +0.3168 | +0.0636 | 1.985 | 0.9895 | **FAIL** ×4 |
| ORACLE true speed window | 1.00 | +1.0000 | +1.0000 | 0.000 | 0.9103 | **PASS** |

⭐ **A new calibration fact the screen would not have without this run: the SAME encoder is 6.9×
less jittery and 4.5× better correlated on PhysicalAI than on comma2k19** (7.42 vs 50.96;
+0.4019 vs +0.0891). The screen's numbers are **corpus-dependent**, so a threshold calibrated on one
corpus is not portable — which is the same lesson as §0.3, arriving from a different instrument.
It still FAILS every gate on both, so the reject verdict is robust; the *distance* to the gate is not.

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

| family / metric | `v1_window` | `mot16_window_rbf` **motion** | `pix32_centre_rbf` **still** | `NULL_train_mean` |
|---|---:|---:|---:|---:|
| *(ADE@2s, m — **one row of four**)* | 2.2808 [2.0939, 2.4744] | 2.9256 [2.6663, 3.2135] | **4.0855** | **4.0855** |
| **LONG** target-speed MAE (m/s) | **1.6668** | 2.2406 | **3.1457** | **3.1457** |
| **LONG** target-speed bias (m/s) | −0.1656 | −0.2229 | −0.1898 | −0.1898 |
| **LONG** along-track MAE / bias (m) | 2.2226 / −0.2147 | 2.7954 / −0.2654 | 3.9086 / −0.2342 | 3.9086 / −0.2342 |
| **LONG** accel MAE (m/s²) | 0.7078 | 0.7635 | 0.7513 | 0.7513 |
| **LONG** distance-keeping / headway / TTC | ⛔ UNAVAILABLE, n = 2,390 | ⛔ | ⛔ | ⛔ |
| **LAT** heading MAE (rad), n | **0.08177**, 8,299 | 0.09553, 8,358 | 0.10779, 8,381 | 0.10779 |
| **LAT** curvature MAE (1/m), n | **0.017730**, 6,154 | 0.024360, 6,208 | 0.023980, 6,219 | 0.023980 |
| **LAT** yaw-rate MAE (rad/s) | **0.06272** | 0.10540 | 0.10894 | 0.10894 |
| **LAT** cross-track MAE (m) | **0.2680** | 0.4427 | 0.5076 | 0.5076 |
| **TAC** manoeuvre bal-acc — **lateral** (chance 0.3333) | **0.8229** | 0.5137 | **0.3333** | 0.3333 |
| **TAC** manoeuvre bal-acc — **longitudinal** (chance 0.3333) | **0.3886** | 0.3390 | **0.3333** | 0.3333 |
| **TAC** manoeuvre bal-acc — **mixed 5-way** (chance 0.2000) | 0.5195 | 0.3104 | **0.2000** | 0.2000 |
| **TAC** goal/anchor selection | PARTIAL — see below | PARTIAL | PARTIAL | PARTIAL |
| **STRAT** route/goal | ⛔ UNAVAILABLE, n = 2,390 | ⛔ | ⛔ | ⛔ |

⭐⭐ **The still-frame arm's four-family row is BIT-IDENTICAL to the train-mean null in every
family.** Not "worse" — *the same object*. It is the strongest available form of the P1 verdict and
it is visible only because the families were computed.

⭐ **A corroboration the audit did not go looking for: TACTICAL/longitudinal is at chance for EVERY
arm** — 0.3886, 0.3390, 0.3333 against a chance of 0.3333 — while TACTICAL/lateral reaches **0.8229**
for the latent. That is the programme's known longitudinal defect (the 5-way softmax that mixes
lateral and longitudinal) reproducing on an independent substrate and an independent probe.
**A scalar ADE cannot see it**, which is the whole reason the four-family rule exists.

**Availability, per family, with reason and n — a missing metric is a WORK ITEM, not a pass:**

| family | status | reason |
|---|---|---|
| LONGITUDINAL | ✅ available except distance-keeping | **no lead-agent track on this substrate.** comma2k19 ships no object annotation at all; PhysicalAI-AV **does** ship `obstacle.offline` on **97.44 %** of clips, but the episode ingest does not read it (`physicalai_r0.py` reads 4 of 36 features). **n = 2,390 windows, 0 with a lead agent resolved.** |
| LATERAL | ✅ available | heading n = 8,299–8,381 waypoint pairs, curvature n = 6,154–6,219 (curvature needs three consecutive waypoints, hence the smaller n) |
| TACTICAL | ✅ available; goal-setting **PARTIAL** | selected-vs-executed manoeuvre **is** reported, split lateral/longitudinal/mixed. Anchor/goal selection is not: the IDM probe emits a **single regressed trajectory** and has no anchor set to select from. |
| STRATEGIC | ⛔ UNAVAILABLE, **n = 2,390** | no route/goal label on either substrate. PhysicalAI-AV is settled at five independent probes as carrying **no map, lane graph, junction annotation or route signal**, and its egomotion is clip-local metres with **no GNSS**, so map-matching is impossible. A strategic read needs AlpaSim/NuRec `map.xodr` or an external corpus. |

⚠️ **The families did not decide the verdict and were not used to reach it** — the pre-registered
statistic is a `speed` R² ratio, fixed in advance. They are reported because an eval that reports one
family is incomplete, and because two of the findings above (the null-identity and the longitudinal
chance-level tactical read) exist **only** in the family rows.

---

## 3. STRATIFICATION — pre-registered, because a pooled number hides the regime

### 3a. BY MANOEUVRE — the informative cut, and it is unanimous

`refb.MANEUVER_CLASSES`, held-out speed R², 80 held-out episodes:

| manoeuvre | n | ⌀ speed | `v1_window` | `pix32_centre_rbf` **still** | `mot8_window_rbf` **motion** |
|---|---:|---:|---:|---:|---:|
| `lane_keep` | 775 | 5.83 | **+0.7773** | −0.0008 | +0.5881 |
| `turn_left` | 411 | 5.88 | **+0.4074** | −0.0006 | +0.1412 |
| `turn_right` | 396 | 5.79 | **+0.5563** | −0.0031 | −0.0952 |
| `accelerate` | 403 | 5.70 | **+0.7287** | −0.0053 | +0.4434 |
| `brake_stop` | 405 | 7.82 | **+0.4518** | −0.3366 | +0.0216 |

⇒ **The still frame is at or below zero in every manoeuvre class.** There is no manoeuvre in which
it recovers, and the ordering `v1_window` > motion > still holds in all five.

### 3b. BY SPEED BIN — reported as MAE, because within-bin R² is not usable here

⚠️ **Within-stratum R² uses the STRATUM's OWN variance as its denominator.** A bin 1 m/s wide has
almost no variance to explain, so every arm scores hugely negative (e.g. `speed_0_1`: `v1_window`
**−65.0**, still frame **−533.8**) and the numbers say nothing about the arms. Those rows are in the
JSON for completeness and are **not evidence**. The usable within-bin statistics are **MAE** and the
prediction↔truth correlation:

| speed bin (m/s) | n | ⌀ v | σ v | `v1_window` MAE / r | still MAE / r | motion MAE / r |
|---|---:|---:|---:|---|---|---|
| 0–1 | 355 | 0.13 | 0.25 | **1.570** / +0.466 | 5.824 / — | 1.969 / +0.626 |
| 1–3 | 195 | 2.01 | 0.58 | **2.093** / +0.135 | 3.942 / — | 3.273 / +0.233 |
| 3–6 | 535 | 4.66 | 0.83 | 1.367 / +0.293 | **1.301** / — | 1.704 / +0.153 |
| 6–10 | 939 | 7.94 | 1.12 | **1.308** / +0.413 | 1.982 / — | 1.550 / +0.136 |
| 10–15 | 345 | 11.48 | 1.26 | **2.683** / +0.040 | 5.520 / — | 4.489 / −0.027 |
| 15+ | 21 | 16.44 | — | — | — | — **UNPOWERED** (n = 21, 4 episodes) |

*(`r` is blank for the still-frame arm because it emits a CONSTANT — the train mean — so its
correlation is undefined. That is the finding, not a missing number.)*

⚠️ **The one row where the still frame "wins" is an artifact and must not be quoted.** In the
3–6 m/s bin its constant prediction (the global train mean, ≈ 6.2 m/s) happens to sit inside the
bin, so a predictor with **zero** information scores the lowest MAE there. This is exactly why a
constant-vs-model comparison needs the shuffled control, and the shuffled control says
**not separated**.

⇒ The corpus stream's speed-dependent tactical lossy rate (**38.2 % at 1–3 m/s → 1.8 % at
10–15 m/s**) does **not** hide a regime where the shortcut lives: there is no speed bin in which the
still frame carries speed information at all.

Bins below **n = 100 windows or 5 episodes** are marked **UNPOWERED** and are not evidence.
Full tables in `raw/summary_tables.txt` (`P1 — STRATIFIED`).

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

⚠️ **`--merge` exists because this run lost two passes to the box killing a long job.**
`run_p1_offhighway.py --merge` reloads the result JSON *and* the per-arm prediction cache
(`results_p1_physicalai.preds.npz`, written after **every** arm) and refits only what is missing;
`run_p1b_mechanism.py --merge --only <corpus>` does the same per corpus. A 22-minute panel that
restarts from zero every time something is killed is a panel that never finishes. **Predictions are
banked per arm, not at the end** — the same discipline the programme's stranded-artifact rule
demands, applied inside a single run.

## 7. VERIFICATION

* `cd stack && pytest -q` → **2023 passed, 12 skipped, 2 xfailed** (630.9 s). The brief's baseline
  was 1900 passed; the suite grew today across several streams, and **12 of the new tests are this
  stream's** (`stack/tests/test_latent_screen.py`).
* Every number in this document is emitted by `summarize.py` from the result JSONs into
  `raw/summary_tables.txt`. **If a number here is not in that file, it is a transcription error** —
  and §0.2 documents one such error found in a *sibling* document by exactly this mechanism.
* Staging verified with `git ls-files --cached`, never with an `add` exit code.
* ⛔ Nothing committed, nothing pushed, no branch switched, no pod touched.
