<title>Place identity survives a change of trunk stage — ID-REPLICATES at +0.2045 against +0.2088, and path geometry gets worse, not better</title>

# `E-DEP-VGEO-6` (DP18-1): **`ID-REPLICATES`** — the s16 stage reproduces the s32 place-identity verdict almost exactly (**+0.2045 [+0.1289, +0.2766]** vs **+0.2088 [+0.1329, +0.2782]**), while metric path geometry moves **off** the VGEO-5 branch table in the **worse** direction

**2026-09-20 · Research Lab (LAB-RUN-017) · Deployment & Optimization · serves INJECTED row I-1 (top of lane), 6th consecutive pass, via LR15-1 / DP18-1**
⛔ **Pre-registered:** `SPEC.md` + `code/vgeo6_s16.py` + `code/make_vgeo6.py` pinned in `raw/prereg_pin.json` at 06:45:19 UTC, before the run computed anything. **0 GPU**, CPU only, 152 s. ⛔ **Tier: none** — a representation diagnostic on frozen features; no driving claim.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **`ID-REPLICATES`.** On the s16 stage, `latent_3520|raw − pixel_1440|raw` P2 = **+0.2045**, CI **[+0.1289, +0.2766]**, n = 29 clips. The s32 verdict was **+0.2088 [+0.1329, +0.2782]**. The two points differ by **0.0043** — about **1/17th** of either CI's half-width. The pre-committed magnitude clause (< +0.1044 ⇒ `WEAK`) is **not** triggered. ⇒ place identity is a property of the **trunk across both its stages**, not of the s32 readout. | MEASURED `raw/vgeo6_step4.json` |
| **F2** | ⭐⭐ **The harness control passed in full, and passed in the exact shape it was predicted to.** Every rng-free quantity reproduced the landed VGEO-5 value **exactly** — split 134/94/40, 6,704 rows, the same 40 scored ordinals, the screen's **195 kept / 182 dropped / 17 surviving**, 29 triplet clips with the same 40 per-clip counts, `C_const` 0.0000/0.5000, and `pixel_1440|raw` **P1r 0.1116 · P2 0.5329 · P3 0.5923**. Every quantity SPEC §3-H2 predicted would drift (the `|diag` arms, all bootstrap CIs) drifted, and **nothing else did**. | MEASURED |
| **F3** | ⛔⭐ **The secondary lands OFF the §4b table, in the direction that makes the VGEO-5 decision stronger, not weaker.** `latent_3520|diag − pixel_1440|diag` P1r-s = **−0.1441**, CI **[−0.2763, −0.0161]** — the CI lies **entirely below zero**. SPEC §4b named only "contains 0" and "lower > 0", so per the VGEO-5 convention this is a **`WORSE-BRANCH`** and is stated as one: on s16, **pixels beat the trunk** at metric path geometry. | MEASURED |
| **F4** | ⛔ **But F3 is NOT a clean stage contrast, and SPEC §3-H2 is why.** The `|diag` arms are precisely the ones pre-registered as non-comparable across runs: their weights come from an `nnls` fit on an rng-drawn subsample, and the shared rng stream diverges. It demonstrably did — the **same** `pixel_1440|diag` arm reads **P2 0.7335** here against **0.7494** there, on identical inputs. ⇒ *Within* this run, pixels beat s16 latents on path geometry. **"s16 is worse than s32 at path geometry" is NOT established**, and no number in this package supports it. | MEASURED + the pre-registered caveat firing |
| **F5** | ⭐ **The replication is not confined to the gridded arm.** The globally pooled s16 arm also clears zero: `latent_352|raw − pixel_1440|raw` P2 = **+0.1748 [+0.0957, +0.2487]**. Place identity survives collapsing the readout from 2×5 cells to **one** — which is what a *place* signal, as opposed to a *layout* signal, should do. | MEASURED |

---

## 1 · Gates, in the pinned order

| gate | bar | read | |
|---|---|---|---|
| **H1** harness | §3 table reproduces exactly | **all 13 quantities exact** (see F2 and `raw/h1_control.txt`) | ✅ |
| **G2** null | `C_presid` P1r \|point\| ≤ 0.03 | **−0.0038** | ✅ |
| **G3** harness | `C_const` exactly 0.0000; `C_mut` \|point\| ≤ 0.05 | **0.0000** · **−0.0141** | ✅ |
| **G4** n | surviving scored ≥ 15; P2 clips ≥ 20 | **17** · **29** | ✅ |
| **G1** power | `C_oracle` P1r ≥ 0.99 | **0.9978** [0.9968, 0.9986] | ✅ |
| **P2 own controls** | `C_const` P2 = 0.5000; `pixel_1440|raw` P2 ∈ [0.45, 0.55] | **0.5000** · **0.5329** | ✅ |

⭐ **Six passes into this line, this is the first run where every gate cleared on the first attempt.** VGEO-3 failed its null, VGEO-4 fired `VOID-POWER`, VGEO-5 cleared its gates but had no cross-run control at all.

## 2 · The panel

`latent_*|raw − pixel_1440|raw` (paired clip bootstrap, 2,000 resamples, 95 % percentile):

| arm | P2 (place identity) | P1r-s (path geometry) | P3 (global path length) |
|---|---|---|---|
| **s16 `latent_3520`** ⭐ | **+0.2045 [+0.1289, +0.2766]** | −0.3668 [−0.5794, −0.1638] | −0.1513 [−0.2297, −0.0763] |
| **s16 `latent_352`** | +0.1748 [+0.0957, +0.2487] | −0.3743 [−0.6051, −0.1560] | −0.2633 [−0.3450, −0.1888] |
| *s32 `latent_7040` (landed, for reference)* | *+0.2088 [+0.1329, +0.2782]* | *−0.4009 [−0.6102, −0.2029]* | *−0.2050 [−0.2739, −0.1384]* |

⚠️ **The `|raw` row is the only one comparable across the two runs** (no `nnls`, no rng; the pixel arm reproduced exactly). The `|diag` rows are within-run reads only — F4.

⭐ **Read the P1r-s `|raw` column as a control on F1, not as a second finding.** Both stages are strongly *negative* there and near-identical (−0.3668 vs −0.4009). A trunk that beat pixels on *everything* would suggest the pixel baseline was simply weak; instead the same pixel baseline **beats** the trunk on metric geometry and **loses** to it on place, on the same clips, in the same run. **P2's positive result is therefore signal, not a baseline artifact** — and that discrimination is the single most load-bearing thing in this package.

## 3 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | Injected row **I-1** (top of lane, 6th consecutive pass). VGEO-5 produced the line's first verdict on a single stage; LR15-1 pre-committed this replication precisely because a one-readout verdict is not a design basis. |
| **CONSEQUENCE** | **LR15-2 is now supported by a replication, not by a single read.** Two stages of the frozen trunk agree, to within 1/17th of a CI half-width, that the trunk beats pixels at *recognising a place* and loses to them at *measuring a path*. ⇒ If a value head takes the frozen trunk, its target should be **place / progress-through-place**, not metric distance. ⛔ This is a **Master Mind decision**, not the Lab's; the Lab has now supplied both halves of the evidence LR15-2 asked for. |
| **COMBINATION** | Lands on the same spot as two independent frontier findings: **DeepSight `2605.10564`** and **World Tokens `2608.09730`** (2026-09-19 scan, finding 2) both measured that the world model's *target* matters more than its pixel fidelity. Here the *frozen representation* shows the matching asymmetry — it is better than pixels at semantic identity and worse at metric geometry. It also sits directly on the **v6 readout-geometry ceiling** (4 azimuth bins over 120°): the ceiling is a *localisation* ceiling, and this run says localisation is the thing this trunk is worst at relative to raw pixels. |
| **CHANCES / RISKS** | **Chance:** a value head keyed on place identity can use the frozen trunk with no retraining and no metric supervision. **Risks:** (a) **two stages of ONE checkpoint is not a trunk family** — SPEC §4c says so and it still binds; (b) the P2 distractor triplets are matched **in pixel distance**, so "place identity" means *beats a pixel-distance-matched distractor*, which is a weaker claim than "identifies the place"; (c) n = 29 clips, one corpus, one checkpoint; the bootstrap answers a draw of **clips** and nothing else (`H-ESTIM-SEED-1`); (d) the `|diag` non-comparability (F4) removes the cross-stage geometry question from this package entirely. |
| **EXPERIMENT** | ⭐ **`E-DEP-VGEO-7` (proposed DP20-1, 0 GPU):** re-run this identical panel on a **second checkpoint** of a different training run, and make the `|diag` arms comparable by seeding `fit_diag`'s subsample from a **dedicated** `default_rng(SEED + dims)` instead of the shared stream — the one change that would have let today's secondary be read. **Committed:** P2 clears zero on the second checkpoint ⇒ "trunk-family property" becomes *established* rather than *admissible*, and LR15-2's re-target is ready for the PI; P2 fails ⇒ the property is **run-specific** and the frozen-trunk value line closes. ⛔ Needs one banked non-refcv5 checkpoint with a token bank at both stages. |

## 4 · Stopping condition (Rule Zero)

**(3a).** The pre-committed bar was cleared on the primary and the run's own harness control passed. **Row I-1 STAYS OPEN**: today answered *which target a frozen-trunk value head should take*, and the **quality half — does a value head plan better than fan-scoring — remains untouched after six passes.** That is stated here, again, rather than allowed to look served.

`Deliverables: SPEC.md · RESULT.md · code/{vgeo6_s16.py, make_vgeo6.py} · raw/{prereg_pin.json, vgeo6_step4.json, vgeo6_step4.log, vgeo6_vs_vgeo5.diff, h1_control.txt, search_log.md}`
