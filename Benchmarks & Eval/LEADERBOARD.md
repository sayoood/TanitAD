# TanitAD Leaderboard

*Rewritten 2026-07-21 from `Project Steering/MODEL_REGISTRY.md` §6 + the raw eval artifacts under
`taniteval/results/`. This closes registry gap **R5** ("LEADERBOARD.md is stale and in the wrong units
— newest row is camera-frame ADE@1s @27 k"): every internal row below is now **metric-BEV `ade_0_2s`
in metres** on the canonical 881 windows, and the old camera-frame gate ladder has been moved to §8
under its own unit label. Maintained by the Benchmarks & Eval agent.*

**Regenerate the driving tables — CPU-only, no GPU, no pod, seconds:**
```
python -m taniteval.runner driving-all          # recompute every arm with a windows_*.pt
python -m taniteval.driving --leaderboard       # emit §2's markdown table
```

---

## 0. How to read this page — units, estimator, floors (binding)

**UNITS.** Internal rows (§1–§6) are **metric-BEV ego-frame `ade_0_2s`, metres**, averaged over the
waypoints at 0.5/1/1.5/2 s. They are **not** the camera-frame `ADE@1s` of the 2026-07-12 D1 gate
(§8) and the two must never be compared. Speeds are m/s, headings degrees, latency milliseconds.

**CORPUS.** `physicalai-val-0c5f7dac3b11` — **881 windows / 40 episodes**, window 8, stride 8,
K = 20 @ 10 Hz, `nav=follow`, operative step intent-free. Identical windows for every arm, so every
cross-arm delta below is **paired**. §7 and §8 use *different corpora* and are labelled as such.

**ESTIMATOR.** The decision-grade interval is the **episode-cluster bootstrap** over the 40 val
episodes (`taniteval/ci.py`, B = 2000); for two arms or an arm-vs-floor on the same windows it is the
**paired** form. The legacy `heldout ± ci95` — historically mislabelled "8-split episode-disjoint
jackknife" — is `overlapping_holdout_se` and is measured **1.28–2.06× too narrow** across 10 arms
(MEASURED, `Project Steering/CI_RECOMPUTE_2026-07-20.json`). It appears in §1 **only** in a column
explicitly marked deprecated, so published figures stay traceable; `taniteval/driving.py` *refuses*
to emit it.

**win / tie / LOST is three-way on purpose.** A paired interval that excludes zero while favouring the
**floor** means the trivial baseline beat the model. Six arms are CI-separated *against themselves*
on speed MAE; a sep/tie rendering would have printed those as wins.

**FLOORS** on the same 881 windows (MEASURED, `taniteval/results/driving_flagship-30k.json`):

| floor | ADE@2s m | FDE@2s m | miss@2m | speed MAE m/s | \|along\| m | \|cross\| m | heading° | κ-sign |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **constant velocity (CV)** | **0.8377** | 1.7406 | 0.3042 | 0.4678 | 1.0955 | 1.0089 | 6.623 | 0.6103 |
| **hold-v0** (straight at entry speed) | **0.7876** | 1.6521 | 0.2917 | 0.4818 | 1.1040 | 0.9137 | 6.344 | 0.5119 |
| best-of-3 kinematic (CV/CTRV/go-straight) | *0.5005* | — | — | — | — | — | — | — |
| CTRV oracle | *0.523* | — | — | — | — | — | — | — |
| no-vision ego-status ceiling (AD-MLP repro) | *0.5735* | — | — | — | — | — | — | — |

*hold-v0 is the strongest trivial **longitudinal** floor and the one VTARGET provably loses to at 2 s
(MAE 1.65 vs 0.475, MODEL_REGISTRY §4.1). CV and hold-v0 are both straight lines, so they are also the
honest "no lateral skill" bar. Rows 3–5 are from MODEL_REGISTRY §0.3/§6; rows 1–2 are recomputed here.*

> ⚠ **Open-loop ⊥ closed-loop (standing footnote, G-B1).** arXiv 2605.00066 (Apr-2026, 15 methods):
> ADE/FDE have **no reliable correlation** with closed-loop Driving Score. Our own evidence: flagship v1
> open-loop **0.4522 → closed-loop 1.685**, divergence >5 m on **22.2 %** of windows (MODEL_REGISTRY
> §1.2). **An external-simulator data point exists (§5.5) but is reconstruction-OOD confounded:** REF-C's
> open-loop ADE on the AlpaSim NuRec reconstructions is **1.52 (3.21× its real-footage 0.4728)**, so its
> closed-loop failures measure model × reconstruction-fidelity — **not** a clean open-loop⊥closed-loop
> demonstration (RETRACTION_LOG C6). **A 2026-07-23 real-footage low-OOD harness (§5.5, both arms 1.02–1.20×
> OOD, n = 40 paired) now confirms the REF-C > flagship-v1 closed-loop ordering OFF-reconstruction** — but it
> measures lane-keeping/drift, not off-road/collision. **Never rank a TanitAD checkpoint on an open-loop number alone.** Every
> internal row carries `claim_strength: open-loop / weak`.

---

## 1. Internal leaderboard — open-loop ADE, metric-BEV `ade_0_2s` (m)

*Ranks and the deprecated column from MODEL_REGISTRY §6; the full-set point estimate and the
episode-cluster interval are recomputed from the committed window dumps and **match §6 exactly** where
§6 quotes them (flagship 0.4271 [0.3675, 0.4871]; REF-C-XL 0.4714 [0.3896, 0.5556]; REF-C-base 0.4728
[0.3835, 0.5699]). Latency is this arm's own `taniteval/results/eff_<key>.json`, fp32, batch 1, A40.*

| Rank | Arm | key | Step | Params | **ADE@2s m, full-set [ep-cluster CI95]** | FDE@2s m, full-set | miss@2m, full-set | beats CV, paired Δ m | tick p50 fp32 | 10 Hz @p99 | *ADE@2s heldout ± ci95 (DEPRECATED)* |
|---:|---|---|---:|---:|---|---:|---:|:--:|---:|:--:|---|
| **1=** | Flagship v1 (speed+jerk) FINAL | `flagship-30k` | 29 999 | 263.4 M | **0.4271** [0.3675, 0.4871] | 0.9075 | 0.045 | ✅ +0.4106 | 97.3 ms | ❌ | *0.4522 ± 0.0312* |
| **1=** | REF-C-XL (anchored diffusion) FINAL | `refc-xl-30k` | 29 999 | 251.9 M | **0.4714** [0.3896, 0.5556] | 1.0061 | 0.142 | ✅ +0.3663 | 44.1 ms | ✅ | *0.458 ± 0.057* |
| **1=** | REF-C-base (anchored diffusion) FINAL | `refc-base-30k` | 29 999 | **104.2 M** | **0.4728** [0.3835, 0.5699] | 1.0031 | 0.142 | ✅ +0.3649 | **21.8 ms** | ✅ | *0.4523 ± 0.0497* |
| — ‡ | Flagship v1.6 (LP-FT, `ab` head) | `flagship-v16-ab-ft` | 5 999 | ~263 M | **0.4375** [0.3423, 0.5501] | 0.9297 | 0.106 | ✅ +0.4003 | — | — | *0.4886 ± 0.0800* ⚠ |
| — ‡ | REF-C-**small** (anchored diffusion) FINAL | `refc-small-30k` | 29 999 | **54.7 M** | **0.5261** [0.4295, 0.6262] | 1.1115 | 0.171 | ✅ +0.3116 | **11.5 ms** | ✅ | *0.5007 ± 0.0671* |
| — | *hold-v0 (trivial floor)* | — | — | 0 | *0.7876* | 1.6521 | 0.292 | — | — | — | — |
| — | *constant velocity (trivial floor)* | — | — | 0 | *0.8377* | 1.7406 | 0.304 | — | — | — | *0.8248* |
| 3 | REF-B v2 (arch-v2) FINAL | `refb-v2-30k` | 29 999 | 271.6 M | 0.5913 [0.4766, 0.7131] | 1.2434 | 0.207 | ✅ +0.2464 | — | — | *0.5921 ± 0.0685* |
| — ‡ | REF-C-XL snapshot | `refc-xl` | ~28 000 | 251.9 M | 0.6048 [0.5170, 0.7009] | 1.1873 | 0.167 | ✅ +0.2329 | 44.0 ms | ✅ | — |
| 4 | Flagship v1, 19 k relay | `flagship-speed` | 19 000 | 263.4 M | 0.6152 [0.5422, 0.6951] | 1.3168 | 0.167 | ✅ +0.2225 | 99.6 ms | ❌ | *0.6277 ± 0.0551* |
| 5 | REF-B v2 @20 k milestone | `refb-v2-20k` | 20 000 | 271.6 M | 0.6435 [0.5410, 0.7516] | 1.3218 | 0.216 | ✅ +0.1942 | — | — | *0.6462 ± 0.0548* |
| 6 | REF-B speed | `refb-10k` | 10 000 | 262.8 M | 0.8372 [0.6753, 1.0218] | 1.6964 | 0.268 | ✗ +0.0005 | 60.5 ms | ✅ | *0.8255 ± 0.0992* |
| 7 | REF-B v1 | `refb` | 6 000 | 262.5 M | 0.8629 [0.6928, 1.0385] | 1.7351 | 0.318 | ✗ −0.0252 | 59.8 ms | ✅ | *0.8682 ± 0.0817* |
| 8 | P2 CEM planner over frozen v1 | `planner_p2` | (n/a) | 0 trained | — 🟥 no window dump | — | — | ✗ | — | — | *0.893 ± 0.114* |
| 9 | REF-A DINOv2 4B | `refa-dinov2` | 29 999 | 156.6 M† | 2.1675 [1.9081, 2.4212] | 3.2803 | 0.613 | ✗ −1.3298 | 88.6 ms | ❌ | *2.1322 ± 0.1821* |
| 10 | Flagship **no-speed** (ablation control) | `flagship-nospeed` | ~22 000 | 263.4 M | 3.0175 [2.5450, 3.5444] | 5.0282 | 0.742 | ✗ | 101.6 ms | ❌ | *2.9176 ± 0.3558* |
| 11 | REF-A dyn-in 4B | `refa-dynin-30k` | 29 999 | 156.6 M† | 3.0471 [2.4984, 3.6878] | 4.7642 | 0.741 | ✗ | 84.5 ms | ❌ | *2.9196 ± 0.3937* |
| 12 | Flagship v2 (killed) | `flagship-v2-6k` | 6 000 | 272.9 M | 5.9396 [4.3273, 7.6249] | 12.4011 | 0.852 | ✗ | — | — | *6.179 ± 1.2845* |
| — | Flagship v3enc | — | running | 272.9 M | 🟥 not evaluated | — | — | — | — | — | — |

**Rank numbers are MODEL_REGISTRY §6's, unchanged.** ‡ marks three arms that have a window dump and a
tier-0 row but **no §6 rank** — `flagship-v16-ab-ft` (an ADE tie with v1, so it cannot be ordered),
`refc-small-30k` (FINAL, but a SEPARATED third rung ~0.053 m below the base≈XL tie — the ladder's first
separation, registry §4.2) and `refc-xl` (a pre-final snapshot). They are shown in ADE order but never
renumbered; the registry is the source of truth for ranks. `beats CV` is the **paired** episode-cluster delta (CV − model, m).

† REF-A's params **exclude the external frozen DINOv2/I-JEPA encoder**, and so does its latency —
never compare a features-in row to a pixels-in row unadjusted.
⚠ v1.6's heldout comes from a **different eid family** (`eval_flagship_v16.py` clusters on real
`episode_id`, `bench.py` on file indices 0–39) — the two `heldout` means are **not comparable**.
The full-set and episode-cluster columns are unaffected. v1.6 vs v1 paired: **Δ +0.0104
[−0.0888, +0.1147], NOT separated** (MODEL_REGISTRY §1.4b) → an ADE **tie**, which is why it carries
no rank.

**Ranks 1= are a three-way ADE tie no paired test can order** (§6: ADE Δ +0.0013 [−0.0281, +0.0316]
base-vs-XL; +0.0443 [−0.0544, +0.1465] flagship-vs-XL). **Latency is the only separator among them —
and §2 now adds a second one.**

---

## 2. Driving capability — TanitEval v2 tier-0 · **the standard read**

*MEASURED 2026-07-21, `python -m taniteval.runner driving-all`, CPU-only over the committed
`windows_<key>.pt`. Every interval is an **episode-cluster bootstrap** (B = 2000, 40 episodes); every
win/tie/LOST is a **paired** episode-cluster test against a trivial floor. Spec:
`TanitAD Research Hub/Benchmarks & Eval/TANITEVAL_V2_METRIC_SUITE.md`. Artifacts:
`taniteval/results/driving_<key>.json`, and inline in every `results/<key>.json`.*

**ADE is one column, not the verdict.** The same 0.43-vs-0.47 that reads as a tie in §1 decomposes
into three different competencies here, and the arms rank differently on each.

| arm | ADE@2s m [ep-cluster boot CI95] | along / cross @2s m (vs CV) | speed MAE m/s: model vs hold-v0 (vs CV) | cruise Δ m/s vs hold-v0 | heading on straights ° | κ-sign | tick p50 | where the win lives |
|---|---|---|---|---|---|---|---|---|
| flagship-30k | **0.4271** [0.3675, 0.4871] | 0.841 **tie** / 0.237 win | 0.471 vs 0.482 **tie** | −0.212 **LOST** | 7.98 vs CV 1.399 | 0.954 | 97.3 ms (fp32) | **lateral only** |
| refc-xl-30k | **0.4714** [0.3896, 0.5556] | 0.878 win / 0.280 win | 0.455 vs 0.482 **tie** | −0.069 **LOST** | 3.863 vs CV 1.399 | 0.919 | 44.1 ms (fp32) | both axes |
| refc-base-30k | **0.4728** [0.3835, 0.5699] | 0.866 win / 0.292 win | 0.446 vs 0.482 **tie** | −0.054 **LOST** | 5.834 vs CV 1.399 | 0.916 | 21.8 ms (fp32) | both axes |
| flagship-v16-ab-ft | **0.4375** [0.3423, 0.5501] | 0.683 win / 0.423 win | **0.389 vs 0.482 win** | −0.058 **LOST** | 7.687 vs CV 1.399 | 0.865 | — | both axes |
| refc-small-30k | **0.5261** [0.4295, 0.6262] | 0.970 win / 0.314 win | 0.506 vs 0.482 **tie** | −0.091 **LOST** | 3.531 vs CV 1.399 | 0.921 | 11.5 ms (fp32) | both axes |
| refb-v2-30k | **0.5913** [0.4766, 0.7131] | 1.029 **tie** / 0.408 win | 0.530 vs 0.482 **LOST** | −0.097 **LOST** | 5.75 vs CV 1.399 | 0.907 | — | lateral only |
| refc-xl | **0.6048** [0.5170, 0.7009] | 1.033 **tie** / 0.340 win | 0.572 vs 0.482 **LOST** | −0.186 **LOST** | 8.974 vs CV 1.399 | 0.869 | 44.0 ms (fp32) | lateral only |
| flagship-speed | **0.6152** [0.5422, 0.6951] | 1.178 **tie** / 0.407 win | 0.663 vs 0.482 **LOST** | −0.406 **LOST** | 8.992 vs CV 1.399 | 0.927 | 99.6 ms (fp32) | lateral only |
| refb-v2-20k | **0.6435** [0.5410, 0.7516] | 1.109 **tie** / 0.434 win | 0.579 vs 0.482 **LOST** | −0.128 **LOST** | 6.222 vs CV 1.399 | 0.889 | — | lateral only |
| refb-10k | **0.8372** [0.6753, 1.0218] | 1.157 **tie** / 0.894 win | 0.546 vs 0.482 **LOST** | −0.115 **LOST** | 2.181 vs CV 1.399 | 0.803 | 60.5 ms (fp32) | lateral only |
| refb | **0.8629** [0.6928, 1.0385] | 1.153 **tie** / 0.960 **tie** | 0.541 vs 0.482 **LOST** | −0.106 **LOST** | 1.854 vs CV 1.399 | 0.781 | 59.8 ms (fp32) | neither axis separated |
| refa-dinov2 | **2.1675** [1.9081, 2.4212] | 3.099 **LOST** / 0.578 win | 1.775 vs 0.482 **LOST** | −1.472 **LOST** | 1.925 vs CV 1.399 | 0.866 | 88.6 ms (fp32) | lateral only |
| flagship-nospeed | **3.0175** [2.5450, 3.5444] | 4.968 **LOST** / 0.448 win | 2.521 vs 0.482 **LOST** | −2.250 **LOST** | 5.986 vs CV 1.399 | 0.948 | 101.6 ms (fp32) | lateral only |
| refa-dynin-30k | **3.0471** [2.4984, 3.6878] | 4.536 **LOST** / 0.782 **tie** | 2.379 vs 0.482 **LOST** | −1.963 **LOST** | 4.781 vs CV 1.399 | 0.838 | 84.5 ms (fp32) | neither axis separated |
| flagship-v2-6k | **5.9396** [4.3273, 7.6249] | 11.659 **LOST** / 3.066 **LOST** | 6.353 vs 0.482 **LOST** | −7.282 **LOST** | 16.867 vs CV 1.399 | 0.751 | — | neither axis separated |

**Column definitions.** `along/cross` = the Frenet split of the 2 s residual on the GT path tangent
(orthonormal, so `along² + cross² = ‖err‖²` exactly); the tag is the paired test **vs CV**.
`speed MAE` compares the planned speed profile to the realised one, floor = **hold-v0**; the tag is
the paired test **vs CV**. `cruise Δ` = **L1 CRUISE-QUALITY**, speed MAE on the 639 longitudinally
steady windows, paired vs hold-v0. `heading on straights` = **T3** on the 634 windows with
\|net heading\| < 5°. `κ-sign` = **T4** curvature *sign* agreement (the curvature *magnitude* is
refused at this resolution — MEASURED 24× the signal). `tick p50` = panel 04b, fp32, batch 1, A40.

### What the split changes — five readings that a single ADE column hid

1. **The rank-1= three-way tie is not a tie on driving.** All three beat CV on ADE, but only the two
   REF-C arms beat CV **along-track** (XL +0.2170 [+0.0584, +0.3783]; base +0.2300 [+0.0773, +0.3816],
   both separated). **flagship v1's along-track win is +0.2543 [−0.0278, +0.5304] — not separated.**
   The program's flagship is the *only* member of its own rank tier with no CI-separated longitudinal
   competency. Its entire separated advantage is lateral: cross-track +0.7720 [+0.4166, +1.1914].
2. **flagship-v1.6 is an ADE tie and a longitudinal win.** It is the **only arm in the program** whose
   speed MAE beats CV with a separated interval (+0.0785 [+0.0066, +0.1516]); its along-track error is
   the best measured anywhere (0.683 m), its progress error the best (0.697 m vs v1's 0.837), and its
   longitudinal share of squared error drops from v1's **0.8933 to 0.5638**. It pays for it laterally:
   cross-track 0.423 vs v1's 0.237, path geometry 0.204 vs 0.111, κ-sign 0.865 vs 0.954. The registry's
   "unfreezing changed nothing measurable" is exactly right **on ADE** — and on the split it is wrong in
   both directions at once: unfreezing **traded lateral geometry for longitudinal tracking.**
3. **No arm in the program can hold a steady speed as well as doing nothing.** Every one of the 14 rows
   is CI-separated **against** hold-v0 on the 639 steady windows — from −0.054 (refc-base) to −7.28
   (flagship v2). This is a program-level finding, not a flagship quirk (§3).
4. **On going straight, the ADE ranking inverts.** CV scores 1.399° mean heading error on the 634
   straight windows. The best arms there are the two *worst* on ADE among the trained set —
   `refb` 1.854° and `refb-10k` 2.181° — while flagship v1 scores 7.98° and REF-C-XL 3.863°. On sharp
   curves it flips back: flagship v1 3.811° vs `refb` 26.559° (§4). Neither ordering is visible in ADE.
5. **A catastrophic ADE can hide an intact competency.** `flagship-nospeed` (3.0175 m, the no-speed
   ablation control) still beats CV on cross-track (+0.5611 [+0.1934, +0.9886], separated) and posts
   κ-sign 0.948; **98.97 %** of its squared error is along-track. Its failure is *purely* longitudinal —
   the cleanest confirmation of the speed-channel result we have, and invisible in the scalar.

---

## 3. Longitudinal regime — L1 CRUISE-QUALITY vs L2 TRANSIENT-RESPONSE

*Speed MAE (m/s) by realised longitudinal regime (\|mean accel over the window\| vs ±0.5 m/s², a
PROPOSED threshold). Floor = hold-v0. Paired episode-cluster, orientation floor − model, so **positive
= the model wins**. `steady` n=639, `brake` n=95, `accel` n=147.*

| arm | steady: model / hold-v0 | paired Δ [CI95] | brake Δ | accel Δ |
|---|---|---|---|---|
| flagship-30k | 0.4231 / 0.2109 | **−0.2122** [−0.2778, −0.1443] **LOST** | +0.6433 win | +0.5716 win |
| refc-xl-30k | 0.2796 / 0.2109 | **−0.0687** [−0.1205, −0.0227] **LOST** | +0.0958 **tie** | +0.3998 win |
| refc-base-30k | 0.2646 / 0.2109 | **−0.0537** [−0.0981, −0.0119] **LOST** | +0.0810 **tie** | +0.3955 win |
| flagship-v16-ab-ft | 0.2684 / 0.2109 | **−0.0575** [−0.0954, −0.0228] **LOST** | +0.6404 win | +0.3906 win |
| refc-small-30k | 0.3016 / 0.2109 | **−0.0908** [−0.1544, −0.0344] **LOST** | −0.1304 **LOST** | +0.3316 win |
| refb-v2-30k | 0.3080 / 0.2109 | **−0.0971** [−0.1554, −0.0464] **LOST** | −0.0921 **tie** | +0.1949 win |
| refc-xl | 0.3972 / 0.2109 | **−0.1863** [−0.2306, −0.1451] **LOST** | −0.0621 **tie** | +0.3090 win |
| flagship-speed | 0.6167 / 0.2109 | **−0.4059** [−0.4791, −0.3266] **LOST** | +0.4780 win | +0.3690 win |
| refb-v2-20k | 0.3385 / 0.2109 | **−0.1277** [−0.1794, −0.0810] **LOST** | −0.1234 **LOST** | +0.0542 **tie** |
| refb-10k | 0.3262 / 0.2109 | **−0.1154** [−0.1768, −0.0627] **LOST** | −0.1361 **LOST** | +0.2026 win |
| refb | 0.3166 / 0.2109 | **−0.1058** [−0.1529, −0.0609] **LOST** | −0.1972 **LOST** | +0.2312 win |
| refa-dinov2 | 1.6831 / 0.2109 | **−1.4722** [−1.6692, −1.2879] **LOST** | −1.0228 **LOST** | −0.6922 **LOST** |
| flagship-nospeed | 2.4611 / 0.2109 | **−2.2503** [−2.6697, −1.8783] **LOST** | −1.5438 **LOST** | −1.4437 **LOST** |
| refa-dynin-30k | 2.1740 / 0.2109 | **−1.9632** [−2.4065, −1.6136] **LOST** | −1.9921 **LOST** | −1.5498 **LOST** |
| flagship-v2-6k | 7.4926 / 0.2109 | **−7.2817** [−9.5500, −5.0368] **LOST** | −1.9020 **LOST** | −2.3065 **LOST** |

**Cruise quality and transient response point in opposite directions for the same checkpoint, and
ADE averages them away.** flagship v1 is **2.0× worse than hold-v0** on the 639 steady windows while
winning brake (+0.6433) and accel (+0.5716) decisively — a model that disturbs a speed that needed no
disturbing. **72.5 % of the corpus is steady**, so the failure dominates the corpus and is still
invisible in the scalar because the *geometry* carries ADE there (flagship steady ADE 0.3834 vs
hold-v0's 0.5430).

`flagship-v16-ab-ft` is the interesting row: it cuts the cruise loss **3.7×** versus v1 (−0.0575 vs
−0.2122) *while keeping* v1's braking response (+0.6404 vs +0.6433) — the only arm that does both.
The REF-C arms have the best cruise of any trained arm but are only tied on braking.

---

## 4. Heading and curvature by GT curvature bucket (T3 / T4)

*Mean heading error @2 s, degrees. Buckets: straight <5° (n=634) / gentle 5–15° (n=103) / sharp ≥15°
(n=144) on \|net heading change\|. **R5 applies:** heading is heavy-tailed, so the median is shown
beside the mean and is the honest reducer — flagship v1's corpus mean of 6.61° carries a bootstrap CI
of [2.34, 12.02].*

| arm | straight: mean / median | gentle | sharp | κ-sign straight / gentle / sharp |
|---|---|---|---|---|
| flagship-30k | 7.980 / **1.105** | 2.060 | 3.811 | 0.947 / 0.932 / 0.998 |
| refc-xl-30k | 3.863 / **0.520** | 4.901 | 7.704 | 0.925 / 0.861 / 0.935 |
| refc-base-30k | 5.834 / **0.509** | 6.136 | 8.022 | 0.918 / 0.858 / 0.947 |
| flagship-v16-ab-ft | 7.687 / **0.757** | 7.504 | 9.653 | 0.876 / 0.786 / 0.875 |
| refc-small-30k | 3.531 / **0.534** | 4.293 | 8.319 | 0.927 / 0.877 / 0.926 |
| refb-v2-30k | 5.750 / **0.608** | 5.843 | 10.855 | 0.908 / 0.845 / 0.944 |
| refc-xl | 8.974 / **1.682** | 6.798 | 8.891 | 0.868 / 0.806 / 0.919 |
| flagship-speed | 8.992 / **1.470** | 2.100 | 4.487 | 0.914 / 0.916 / 0.991 |
| refb-v2-20k | 6.222 / **0.678** | 8.328 | 11.298 | 0.888 / 0.819 / 0.942 |
| refb-10k | 2.181 / **0.933** | 6.923 | 21.666 | 0.850 / 0.647 / 0.708 |
| refb | 1.854 / **0.700** | 7.372 | 26.559 | 0.862 / 0.537 / 0.597 |
| refa-dinov2 | 1.925 / **0.895** | 6.168 | 17.915 | 0.891 / 0.754 / 0.840 |
| flagship-nospeed | 5.986 / **1.184** | 21.305 | 41.720 | 0.952 / 0.922 / 0.951 |
| refa-dynin-30k | 4.781 / **0.976** | 6.658 | 15.319 | 0.860 / 0.712 / 0.836 |
| flagship-v2-6k | 16.867 / **5.177** | 27.059 | 52.596 | 0.839 / 0.534 / 0.521 |
| ***CV floor*** | *1.399 / **0.451*** | *7.852* | *28.743* | *0.764 / 0.233 / 0.204* |

**The mean/median gap is the point.** On the mean, flagship v1 is **5.7×** worse than a straight line
at going straight (7.980 vs 1.399). On the **median** — the reducer R5 actually mandates — it is
**2.45×** (1.105 vs 0.451). Both are real; the mean says a tail of windows is badly wrong, the median
says the typical straight window is only moderately wrong. **Quote the median as the headline and the
mean as the tail evidence; never quote one alone.**

Every trained arm beats CV decisively on **gentle and sharp** curves and on **curvature sign** (CV
scores 0.233 / 0.204 sign agreement there — a straight line has no sign to agree with). That is where
the vision is doing work.

---

## 5. Deployment axis — inference efficiency (panel 04b)

*MEASURED on one A40, batch 1, ≥200 warmed iterations, per-iteration CUDA events,
`torch.cuda.synchronize()` bracketed, precision applied identically to every arm and recorded. Source:
`taniteval/results/eff_<key>.json` (canonical files only; quarantined runs excluded).*

| arm | p50 fp32 | p99 fp32 | p50 tf32 | p50 amp16 | params | meets 10 Hz @p99 |
|---|---:|---:|---:|---:|---:|:--:|
| refc-small-30k | **11.50** | 11.56 | — | — | 54.7 M | ✅ (fp32; tf32/amp16 not measured) |
| refc-base-30k | **21.78** | 22.33 | 15.81 | 15.88 | 104.2 M | ✅ all three |
| refc-xl-30k | 44.06 | 44.44 | 27.78 | 21.00 | 251.9 M | ✅ all three |
| refb / refb-10k | 59.80 / 60.47 | 60.31 / 61.12 | — | — | 262.8 M | ✅ |
| refa-dynin-30k † | 84.52 | 128.36 | — | — | 156.6 M | ❌ |
| refa-dinov2 † | 88.58 | 107.67 | — | — | 156.6 M | ❌ |
| flagship-30k | 97.32 | 122.77 | 97.70 | 123.83 | 263.4 M | ❌ all three |
| flagship-nospeed | 101.58 | 127.91 | — | — | 263.4 M | ❌ |

† excludes the external frozen encoder — not comparable to a pixels-in arm.

**Admissibility is a gate, ranking is a frontier, and there is no scalar composite.** An arm is
admissible only if it (a) meets the 10 Hz budget at **p99** in its declared deploy precision **and**
(b) beats every trivial floor on the headline capability metric with a CI-separated paired bootstrap.
Any single number trading metres against milliseconds embeds an exchange rate nobody has measured —
and our arms rank *oppositely* on the two axes (REF-C wins latency 2.2–4.6×, the flagship wins batched
throughput 34.8 vs 29.9 windows/s @ batch 32). Report the Pareto frontier; do not collapse it.
R12 (closed 2026-07-21): the composed inference levers put flagship v1 at **18.75 ms p50 / 18.76 p99 =
53.3 Hz**, which *does* clear (a).

> ⚠ **Conflict on record, reported not resolved.** MODEL_REGISTRY §6 reading 3 quotes the flagship
> tick as **103.42 / 93.76 / 104.49 ms** (fp32/tf32/amp16) and REF-C-XL amp16 as **26.12 ms**. The
> committed artifacts say **97.32 / 97.70 / 123.83** and **21.00**. The fp32 and tf32 REF-C figures
> agree (44.28≈44.06, 27.84≈27.78); the flagship's do not, and the flagship's own repeatability
> record (`eff_repeatability.json`, 5 clean reps) is **99.03–100.05 ms p50**, which brackets neither.
> The two sets were evidently measured in different sessions. **This page quotes the committed
> artifact**; the conclusion is unchanged in every version (REF-C is multiple-× faster; the flagship
> misses 10 Hz at p99 in all three precisions). Needs one reconciliation pass by the eval-pod owner.

### 5.1 Deployment path — ONNX + TensorRT-FP16 + CUDA-graph (Orin / Thor) — MEASURED on an A40 proxy

*The deploy object is flagship v1's **planning tick** (`encode 1 new 9-ch frame → slide 8-state window →
20 sequential operative steps → SE(2) accumulate`). MEASURED 2026-07-22 on an **A40 (SM 8.6) proxy**; raw
under `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-22-orin-thor-deployment/artifacts/`.
Full staged plan + evidence classes: that folder's `DEPLOYMENT_PLAN.md`.*

| measurement | value | source JSON |
|---|---|---|
| composed planning tick (L1+L2+L3+L7) | **18.75 ms p50 / 18.76 p99 = 53.3 Hz** | `eff_levers_flagship-30k.json` (07-21; registry §1.2, R12) |
| CUDA-graph rollout, K=20 (predictor-only proxy) | eager **96.40 → graph 27.87 ms** p50 (**3.46×**) | `bench_latency_report.json` |
| predictor 1-call, fp32 / fp16 | 4.96 / 4.12 ms p50 | `bench_latency_report.json` |
| TensorRT-FP16 engine (A40 proxy) | encoder **1.205 ms** · predictor **0.666 ms** p50; **MHA fuses** (no standalone softmax) | `trt_fp16_report.json` |
| static-shape ONNX export | encoder + predictor build clean, torch-vs-ORT parity ≤ **1.9e-6** | `export_report.json` |

**Per-chip precision map (PUBLISHED, vendor specs — a plan, not a measured tick):** Orin (Ampere SM 8.7)
→ **FP16 baseline; INT8 only behind a per-layer benchmark; NO FP8, NO FP4**. Thor (Blackwell) →
**FP16/FP8 + NVFP4** (the 4× weight-traffic win, Thor-only). INT8 on an Orin ViT can run **~2.7× slower
than FP16** on non-optimal kernels — INT8 is a per-layer hypothesis to disprove, never a default.

**⚠️ Hardware-blocked, stated honestly.** Every latency above is an **A40** number; the A40 TRT engine is a
**proxy — TRT engines are NOT portable across GPU architectures** (Orin SM 8.7, Thor Blackwell). Real
Orin/Thor throughput, the on-device engine build, and any NVFP4 number **need the target silicon** (not on
hand) and are **not fabricated** here. What the A40 build establishes and *does* transfer: the ONNX→TRT
path builds with no plugin, MHA fusion is achievable for our ViT (retires the NVIDIA #4537 risk on SM 8.6),
CUDA-graph capture is exact, and the 20-step rollout is the binding term.

---

## 5.5 CLOSED-LOOP — AlpaSim NuRec reconstructions (n = 12) · ⚠️ RECONSTRUCTION-OOD CONFOUNDED

*MEASURED 2026-07-22 on the AlpaSim closed-loop harness (NuRec photoreal reconstructions, **480×854**,
20 s rollouts) — the program's **first external-simulator** closed-loop numbers (the imagination-in-the-loop
harness behind the G-B1 footnote / MODEL_REGISTRY §1.2 was self-referential). Raw
(`…/incoming/2026-07-22-alpasim-closedloop-evalpod/`): `REFC_suite_results.json`
(+ `REFC_suite_{base,xl}_results.json`), open-loop control `REFC_openloop_diagnostic.json`, flagship
`Flagship_v1_results-summary.json`. A **"pass" = no at-fault collision AND no off-road**; `mean score`
folds in progress-to-GT (`score_criteria`). **A DIFFERENT AXIS from §1–§5 — never mixed with open-loop
ADE.***

> ⚠️⚠️ **HEADLINE — these numbers are ENV-CONFOUNDED, not a clean model result (`RETRACTION_LOG.md` C6,
> 07-22).** The open-loop control settles it: **REF-C's open-loop ADE *on the AlpaSim reconstructions* is
> 1.52 m (de@2s 2.58), 3.21× its taniteval real-footage 0.4728** — consistent across **4 scenes / 288
> predictions** (per-scene 1.40–1.77 m; `REFC_openloop_diagnostic.json`). REF-C is fed NuRec input **~3×
> off its training distribution**, so the at-fault / pass numbers below measure **model ×
> reconstruction-fidelity, NOT the model.** The base-vs-XL *ordering* survives (the same OOD hits both);
> **"REF-C collides closed-loop" does NOT survive as a model indictment.**

| arm | params | **at-fault collision** | off-road | **pass rate** | **mean score** | **dist-to-GT (m)** | progress-rel |
|---|---:|:--:|:--:|:--:|:--:|---:|---:|
| **REF-C-base** | 104.2 M | **33.3 % (4/12)** | 16.7 % (2/12) | **6/12** | **0.345** | **1.642** | 0.877 |
| **REF-C-XL** | 251.9 M | **33.3 % (4/12)** | 25.0 % (3/12) | **5/12** | **0.246** | 1.973 | 0.885 |

**⚠️ n = 12 — one scene = 8.3 pp, and the raw JSON's own caveat is "wide binomial CIs at n = 12".**
Further caveats carried verbatim from `REFC_suite_results.json`: **n = 12 subset** of the 916-scene public
suite (not the full set); **480×854** render (the earlier single-scene runs were 1080×1920); **NuRec
reconstructions, not real-world**. Both arms' collisions are entirely *at-fault*
(`collision_any == collision_at_fault` = 0.333); XL's extra failures over base are off-road.

**base ≥ XL ORDERING holds under the shared OOD.** The open-loop "anchor-width is the lever, encoder scale
is not" result (§1 / registry §4.3: base ties XL on ADE at 2.4× fewer params) **carries into closed loop** —
base **mean score 0.345 > XL 0.246**, **passes 6/12 vs 5/12**, and is **closer to GT (1.64 vs 1.97 m)** — at
the same 33 % at-fault rate. Both arms eat the same reconstruction-OOD, so the *ordering* is readable even
though the *levels* are not a clean model result. **Scale bought no closed-loop advantage.**

**Flagship v1 DOES drive closed-loop (via its `tactical_policy` head) — but a PAIRED n = 12 suite REVERSES
the n = 1 "beats REF-C" read.** MEASURED 2026-07-23, same 12 scenes, both models fed the identical NuRec
renders, f-theta verified live (flag f_eff 265.7 / refc 265.6): **REF-C base statistically beats flagship
v1** — pass **8/12 vs 2/12**, mean score **0.496 vs 0.066**, paired Δ **−0.430 [−0.646, −0.215]** (scene-cluster
boot95 excludes 0), score sign-test **8-0** (p = 0.008), pass-McNemar **6-0** (p = 0.031); **at-fault collisions
TIED** (1-1, p = 1.0). Mechanism (MEASURED, `flagship_vs_refc_suite_results.json` / `…NOTE.md`): flagship's
tactical head is a **high-deviation planner** — plan_dev **1.12 vs REF-C 0.34** (3.3× wider) — so its failure
mode is **off-road, not collision** (8/12 offroad). The lone n = 1 pass on wide-highway `01d503d4` (score
0.699, rollout `71f9740c`) was a **lucky scene** where the wide swerve happened to dodge the collision; it
does **not** generalize. **This still corrects the older "v1 can't drive closed-loop" claim** (pure v1 does
drive from observations via its tactical policy) — but the "v1 **beats** REF-C" headline is retracted
(`RETRACTION_LOG.md` C5, 07-23). Same within-sim / ~3.2× OOD caveat. **Resolution confound RESOLVED (2026-07-23):**
a native-1080×1920 paired re-run **holds the delta** (**−0.295 [−0.494, −0.117]**, sign-test 7-0) → the **model is
the dominant axis, resolution is second-order** (flagship's deficit shrinks ~30% at native but stays significant;
collisions tied at both res). The only axis still open is **sim2real reconstruction-OOD** (~3.2×), which needs a
real-footage harness, not another sim run. Raw: `flagship_vs_refc_native1080_{results.json,NOTE.md}`.

**⭐ sim2real reconstruction-OOD axis CLOSED — REF-C base still beats flagship v1 on a REAL-FOOTAGE low-OOD
harness (n = 40, 2026-07-23).** The one axis left open above (~3.2× NuRec reconstruction-OOD) is settled by a
*different* instrument that needs no renderer: **real-footage log-replay** (drive the recorded frames, integrate
the ego kinematically, arc-length re-index + homography-warp for on-policy deviation — both arms held at
**1.02–1.20× OOD**, ≪ NuRec's 3.75×; `…/incoming/2026-07-23-lowood-lanekeeping-refc/lowood_lanekeep_40ep.json`,
episode-cluster bootstrap, paired). Because a map/agent-free source **cannot** emit off-road/collision (that axis
still needs a lower-OOD renderer), it carries a new low-OOD metric **`corridor_departure_rate`** (on-policy
|XTE| > 1.75 m lane-half-width).

| n = 40 / 881 win, paired | flagship v1 | REF-C base | Δ (flag − refc) | separated |
|---|:--:|:--:|:--:|:--:|
| closed-loop ADE@2s (m) | 1.488 [1.329, 1.647] | **0.564** [0.452, 0.676] | +0.924 [+0.781, +1.065] | **yes** |
| `corridor_departure_rate`@1.75m | 0.0318 | **0.0134** | +0.0184 [+0.0077, +0.0328] | **yes** |
| peak XTE (m) | 0.764 | **0.442** | +0.321 [+0.193, +0.495] | yes |

**REF-C base wins in EVERY stratum → the C7 ordering (REF-C > flagship closed-loop) is now TRIPLE-confirmed**
across three independent instruments (n = 1 scene-dependent → n = 12 NuRec AlpaSim → n = 40 real-footage), so it
is **not a reconstruction artifact**. The metric also **decomposes flagship's deficit**: in longitudinal scenes
both arms keep the lane near-perfectly (departure 0.4 % / 0.04 %) yet flagship's ADE is 4× REF-C's (1.455 vs
0.354) → flagship's gap is **longitudinal, not lane-keeping** (its 89 %-longitudinal signature); in junctions
flagship departs 2.3× more (14.6 % vs 6.4 %, peak XTE 2.37 vs 1.46 m) → its tactical head is **high-deviation**,
independently reconfirming the mechanism above. ⚠️ **Bound:** lane-keeping / on-policy drift only, NOT
off-road/collision; within-source relative; deployed-decoder vs deployed-decoder.

> ⚠️ **Retractions on record (`RETRACTION_LOG.md`, 07-22/07-23).** **C5** — the n=1 *"REF-C collides
> at-fault"* over-read the worst-case scene `01d503d4`. **C6** — the n=12 *"REF-C fails ~half closed-loop"*
> is **reconstruction-OOD confounded** (open-loop-on-reconstructions control 3.21×): run the
> open-loop-vs-known control **before** attributing a closed-loop failure to the model. **C7 (07-23)** — the
> n=1 *"flagship v1 **beats** REF-C closed-loop"* is **reversed** by the paired n=12 suite (REF-C base wins
> 8/12 vs 2/12, sign-test 8-0): a closed-loop win from n=1 is scene-dependent — never headline it until n ≥ ~12.

---

## 6. What TanitEval v2 deliberately does **not** measure

Refusals are part of the contract and are recorded in every `driving_<key>.json`. Each is a data
limitation, not an oversight:

| refused | why |
|---|---|
| headway / distance-keeping / **TTC** | no lead-agent state exists anywhere in the stored data (`lead_state` is a shape-fixed `None` stub); no boxes, no tracks, no depth |
| any **VTARGET**-referenced target-speed metric at 2 s | refuted with numbers — it sits +1.42 m/s above v0 and loses to holding v0 (MAE 1.65 vs 0.475); it is the right quantity at the wrong timescale |
| **intersection / roundabout / merge capability** | the events are 5–20 s, the horizon is 2 s, the clips are ~20 s. A 2 s window inside a roundabout is kinematically indistinguishable from a constant-radius curve. The S1 strata are **kinematic signatures** (`launch_from_stop`, `stop_approach`, `sustained_turn`) and must never be renamed |
| **lane-centre deviation / lane-keeping** | no lane geometry exists; the only lane number in the codebase is a hard-coded `LANE_HALF_M = 1.75` proxy |
| naive **curvature MAE** at the persisted resolution | MEASURED 1.2015 vs a signal of 0.0495 1/m — 24× the signal. It measures knot jitter. Sign agreement survives and discriminates |
| **collision rate / drivable-area / NAVSIM PDMS / nuPlan CLS / CARLA DS** | need agent boxes, an HD map or a simulator |
| a scalar **capability × efficiency** composite | embeds an unmeasured exchange rate; the arms rank oppositely on the two axes |

---

## 7. Different-corpus measurements — do not mix with §1–§5

*These are **not** on the 881 PhysicalAI val windows and are not comparable to the tables above.*

### 7.1 Trivial-baseline floor, comma2k19 + Cosmos-DD (2026-07-15) — camera/BEV mixed, 26 132 anchors

CTRV wins 55–58 % of anchors, CV 20–30 %. **Report `skill_score` = model_ADE ÷ best-of-3 floor.**

| stratum (comma-hwy, v≈25 m/s) | n | best-of-3 floor ADE@1s | CV@1s | CTRV@1s | floor@2s |
|---|---:|---:|---:|---:|---:|
| straight | 18 785 | **0.056 m** | 0.088 | 0.062 | 0.206 |
| gentle | 3 008 | **0.059 m** | 0.275 | 0.060 | 0.228 |
| sharp (speed-gated ≥2 m/s) | 212 | **0.164 m** | 0.404 | 0.167 | 0.608 |

Baselines use privileged GT ego-state → a *denominator*, not a competitor. Curvature strata are
speed-gated (κ = yaw_rate/v is singular at v→0; 12.4 % of comma anchors are near-standstill).
Source: `Implementation/incoming/2026-07-15-baseline-floor/`.

### 7.2 Ego-status shortcut ceiling, comma-hwy (2026-07-17) — metric-BEV, 7 920 val anchors

| predictor | L2@1s | L2@2s | L2@3s | avg |
|---|---:|---:|---:|---:|
| stop (null) | 24.88 | 49.85 | 74.89 | 49.87 |
| best-of-3 kinematic floor | 0.122 | 0.479 | 1.102 | **0.571** |
| **ego-status shortcut** (no vision, learned, held-out) | 0.144 | 0.552 | 1.256 | **0.658** |

comma highway is **73.9 % straight** — identical to nuScenes → our open-loop val inherits the same
ego-status-shortcut pathology (AD-MLP, arXiv 2312.03031). Convention note: `pointwise` = UniAD,
`cumulative` = ST-P3/VAD; they differ ~2×. The **in-corpus** version of this ceiling — the one that
belongs beside §1 — is the **0.5735 m** ego-status ridge on our own 881 windows.

> ⚠ **The warning this page must carry.** The set of metrics computable from ego logs alone is
> *precisely* the set the critique literature showed is gameable by an ego-status MLP with no
> perception. **A map-free suite cannot, on its own, discriminate perception quality.** The two
> antidotes are first-class members of the suite, not extras: the **ego-status ceiling** (0.5735 —
> flagship v1's 0.4271 clears it) and the **vision-ablation on high-divergence windows** (vision effect
> **+1.325 m, CI [+1.04, +1.64]**, CI-separated).

### 7.3 Supervised-IDM cross-domain probe (2026-07-22) — a FINDING, not a leaderboard model

*A ~2.9 M supervised inverse-dynamics head (latent window → speed / yaw-rate / steer / accel), the
pre-registered gate for a YouTube-scale IDM data pipeline. Raw:
`TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-07-22-idm-proof/results.json`.
Gate: cross-domain speed R² > 0.9 AND yaw R² > 0.9 AND ADE@2s < 1.5× the in-domain held-out ADE.*

| split | in-distribution | cross-domain | verdict |
|---|---|---|---|
| PhysicalAI → **comma2k19** (primary go/no-go) | held-out speed R² **0.930**, yaw R² 0.924, ADE@2s 2.73 | comma speed R² **0.657**, yaw R² **0.000**, ADE@2s 6.56 | **FAIL** (ADE ratio 2.40) |
| rig-A → **rig-B** (same corpus, other camera rig) | held-out speed R² 0.786, ADE@2s 4.36 | rig-B speed R² **−2.465**, yaw R² −0.109, ADE@2s 17.47 | **FAIL** (ADE ratio 4.01) |

**The supervised-IDM paradigm works in-distribution and does NOT transfer.** It fails even the
*same-corpus, other-rig* split (rig-B speed R² −2.465 — worse than predicting the mean), so the failure is
**domain shift, not dataset**. The YouTube-IDM data line is **gated on the re-gate** and does not proceed on
these numbers. Recorded as a finding; **no model row.**

---

## 8. Historical — camera-frame gate ladder (SUPERSEDED, different unit)

*Retained for traceability only. **Unit: camera-frame `ADE@1s`**, not metric-BEV `ade_0_2s`. This block
was the newest content on this page until 2026-07-21 and was the substance of registry gap R5.*

### FLAGSHIP — step 27 000, route-resampled protocol, exact training val (comma+pai), 2026-07-12

| Gate | Verdict | Value | Note |
|---|---|---|---|
| **D1** (probe ADE@1s) | FAIL | **6.44 ± 0.55 m** (8 route splits; range 4.96–7.41) | **camera-frame unit** — superseded by the metric-BEV harness in §1 |
| **D2** (imagination ranking) | ✅ PASS | dir-acc 0.864, P4 fwd-dyn 0.971, fit-R² 0.98 | the world-model-usable-for-selection claim holds |
| **D3** (imagined vs oracle @2s) | FAIL, K-step-improved | imagined 1.97 m vs oracle 1.52 m, ratio 1.30 | K-step closed the ratio from ~4× |

> **D1/D3 statistical-power footnote (G-B1).** Those gates were reported from a **single fixed seed=0**
> split at 4–9 val episodes; a measured power audit shows the ADE@1s estimator swings **5–7 m across
> split seeds** on the *same* checkpoint (95 % CI half-width ±4.5 m at n=4). Single-seed D1 values are
> descriptive, not decision-grade. The step-14k→21k "regression" (5.18→11.52 m) is inside that noise
> band. Superseded by §1's 881-window, 40-episode, episode-cluster protocol.

### Live scenario metrics — SC-01 Work-Zone Phantom (2026-07-08, scripted policies, single seed)

| Policy (scripted, NOT our checkpoint) | OKRI ↓ | LOPS ↑ | TMS ↑ | CNCE ↑ | LAL-v1 |
|---|---|---|---|---|---|
| reactive (E2E-like) | 32.37 | 0.00 | 0.006 | 8.68e5 | −0.7 |
| world_model (anticipatory) | **12.83** | **0.834** | 0.023 | 1.06e6 | −0.7 |

Weak rows: scripted archetypes, single seed, and **LAL-v1 is non-discriminative here** (its −1.5 m/s³
trigger never fires on a comfort-bounded ease-off) — superseded by **LAL-v2**, which returns +0.3…+3.1 s
anticipation lead vs −0.3 s reactive. LOPS's 0.0 is structural (a no-estimate policy scores 0 by
definition), so it proves latent-track *presence, not quality*. Not an edge claim.

---

## 9. External context — published numbers, **not comparable to §1–§5**

*Different benchmarks, different sensors, different corpora. Kept as targets and orientation only.*

### Open-loop (NAVSIM v2 EPDMS, navtest/navhard)

| System | EPDMS | Source / date | Note |
|---|---|---|---|
| SOTA claim (survey) | 89.3 | arXiv 2606.19641, 2026-06 | navtest |
| HAD | 88.6 | arXiv 2604.03581, 2026-04 | diffusion + metric-decoupled RL (navtest) |
| Drive-JEPA | 93.3 (PDMS, NAVSIM **v1**) | arXiv 2601.22032, 2026-01 | v1 metric — not comparable to EPDMS |
| DrivoR (test-time opt) | **56.3** (EPDMS, **navhard**) | arXiv 2606.07170, 2026-06 | navhard #1 |
| DriveFuture | **55.5** (EPDMS, **navhard**) | arXiv 2605.09701, 2026-04 | navhard #1 *learned*; future-aware latent WM |
| PDM-Closed (baseline) | 51.3 (EPDMS, **navhard**) | arXiv 2506.04218 / leaderboard 2026-03 | pseudo-sim (3DGS aug), R²≈0.8 vs CL |
| **TanitAD** | — | — | **structurally not computable**: EPDMS needs agent boxes, drivable-area polygons and a route centerline. We adopt only its **Extended Comfort** idea and the human-log filter; a partial PDMS is never published |

### Closed-loop (Bench2Drive, CARLA) — the arbiter block

| System | Driving Score | Success Rate | Source / date |
|---|---|---|---|
| TF++ (VLAAD-MIL) | 86.97 | 71.97 % | arXiv 2603.25946, 2026 |
| ADT | 77.90 | 55.0 % | Bench2Drive leaderboard, 2026 |
| **TanitAD** | — | — | Phase 1; MetaDrive closed-loop first (G0.5), then CARLA/Bench2Drive |

CARLA seed variance ≈ 5 DS same-model → our closed-loop rows will report mean ± CI over ≥3 seeds; a
"beats baseline" claim requires separated CIs.

### Competitor parameter envelope (W-05 / CNCE)

| System | Params | Deployment class | Source |
|---|---|---|---|
| NVIDIA Alpamayo-2 | **32 B** | on-car VLA policy | Opponent profiles, 2026-07 |
| Wayve GAIA-3 | **15 B** | offline generative world model | Opponent profiles, 2026-07 |
| **TanitAD flagship v1** | **263.4 M** | on-car hierarchical latent WM + tactical | MODEL_REGISTRY §1.2 |
| **TanitAD REF-C-base** | **104.2 M** | on-car anchored-diffusion planner, 21.8 ms fp32 | MODEL_REGISTRY §4.3 |

*Not an apples-to-apples score* — a parameter/compute-envelope comparison only. The efficiency wedge is
credible only *at matched safe-progress*.

---

## 10. Provenance, regeneration, and what is still missing

**Every number in §1–§6 traces to** `Project Steering/MODEL_REGISTRY.md` §6/§1.x or to a committed
artifact under `taniteval/results/`: `driving_<key>.json` (§2–§4, §6), `eff_<key>.json` (§5),
`windows_<key>.pt` (the substrate for all of them). Nothing here is transcribed from a summary,
changelog or weekly report. **The 2026-07-22 additions trace to their staged raw JSONs:** §5.5 closed-loop
→ `…/incoming/2026-07-22-alpasim-closedloop-evalpod/{REFC_suite_results,REFC_openloop_diagnostic,Flagship_v1_results-summary}.json`;
§5.1 deployment →
`…/incoming/2026-07-22-orin-thor-deployment/artifacts/{export,bench_latency,trt_fp16}_report.json`;
the `refc-small-30k` rows (§1–§5) → `…/incoming/2026-07-22-refc-small-30k/refc-small-30k.json`; §7.3 IDM →
`…/incoming/2026-07-22-idm-proof/results.json`.

**Regenerate:** `python -m taniteval.runner driving-all` then
`python -m taniteval.driving --leaderboard`. CPU-only, offline, ~1 minute for all 24 dumps.

**Known gaps, marked UNVERIFIED:**
- `planner_p2` and flagship **v3enc** have **no window dump**, so they have no §2 row. P2's ADE
  (0.893 ± 0.114) survives only under the deprecated estimator.
- `flagship-v16-ab-ft`, `refb-v2-*` and `flagship-v2-6k` have **no `eff_<key>.json`** → no latency
  column. Nothing blocks it but a run on an idle GPU.
- **Tier-1 is blocked on one line.** `rollout.collect` computes the dense 20-step path and discards 16
  of 20 steps at `rollout.py:94`. Persisting it unlocks jerk, the adopted nuPlan/NAVSIM comfort bounds,
  the curvature *profile*, decel-onset lead time (already implemented and unmerged since 2026-07-09)
  and plan-stability / Extended Comfort. ~1 MB per arm.
- **Regime thresholds are PROPOSED** (±0.5 m/s², \|κ\| < 1e-3, the 5°/15° curvature split). The
  measured effects are large and unlikely to be threshold-driven, but a sensitivity sweep is owed.
- The 4-waypoint speed is a **0.5 s box-average**, not an instantaneous speed, so the §3 cruise
  numbers are **conservative** — the dense path will make them stricter, not looser.
- `driving.py` buckets curvature at 5°/15°; `driving_diagnostic.curvature_bucket` (used by
  `bench.by_curvature`) buckets at 5°/20°. Both are recorded in every block; the panels are not
  interchangeable until reconciled.
