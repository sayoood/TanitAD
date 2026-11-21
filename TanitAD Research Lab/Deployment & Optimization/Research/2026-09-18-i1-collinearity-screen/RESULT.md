<title>I-1 collinearity screen — every gate passes; path geometry is a powered TIE, place identity is a verdict</title>

# `E-DEP-VGEO-5`: the screen restores power (C-oracle **0.9978**), all four gates pass, and the line gets its first two verdicts — **P1r-s-TIE** and **ID-TRUNK**

**2026-09-18 · Research Lab (LAB-RUN-015) · Deployment & Optimization · serves INJECTED row I-1 (top of lane) via LR14-1 / DP17-1**
⛔ **Tier: none.** A representation diagnostic on frozen features — no rollout, no driving, no four-family eval. Nothing here is a driving claim.
⛔ **Pre-registered:** `SPEC.md` sha256 `ef8c62d8…`, `code/vgeo5_screened.py` sha256 `fe36e892…`, pinned in `raw/prereg_pin.json` **before** the run computed anything. The only code change from VGEO-4 is the screen (SPEC §2).

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **The pre-registered screen restores the positive control: C-oracle 0.9978 [0.9969, 0.9986] vs the committed ≥ 0.99** (VGEO-4 unscreened: 0.8983). The screen dropped **182 of 377** (clip, Δ) cells (48.3 %) and kept **195**. **17 scored clips survive** (bar ≥ 15). | MEASURED `raw/vgeo5_step4.json` (`fit_meta.screen`) |
| **F2** | ✅ **The null panel still holds on the screened subsample:** C-presid **+0.0180 [−0.0126, +0.0465]** (bar \|point\| ≤ 0.03), C-mut **−0.0075**, C-const **exactly 0.0000**. 0 degenerate partial cells (VGEO-4: 262). | MEASURED |
| **F3** | ⭐⭐⭐ **VERDICT `P1r-s-TIE`: `latent_7040\|diag − pixel_1440\|diag` = −0.0707 [−0.2630, +0.1444]**, 17 clips. This is the fifth read of this contrast leaning to pixels, and the **first one taken with a null that reads zero AND a positive control that reads ~1**. ⚠️ **"Powered" is bounded, not absolute:** the oracle shows the statistic can see a *perfect* signal; the CI width (0.41) means a trunk advantage up to **+0.144** is not excluded. What the run excludes is a trunk advantage **larger than +0.144**. | MEASURED |
| **F4** | ⭐⭐⭐ **VERDICT `ID-TRUNK` — the first verdict this line has produced on place identity.** Gated by its **own** controls (C-const P2 **exactly 0.5000**; pixel floor P2 **0.5329 [0.5108, 0.5555]**, inside [0.45, 0.55]): `latent_7040\|raw − pixel_1440\|raw` P2 = **+0.2088 [+0.1329, +0.2782]**, 29 clips. Four reads now agree (+0.2468 / +0.2088 / +0.2474 / +0.2088). The frozen trunk tells *which place* apart from a pixel-matched distractor clearly better than pixels do. | MEASURED |
| **F5** | ⚠️ **Descriptive, off the committed table (WORSE-BRANCH note):** under the screen the **raw (unweighted) latent distance ANTI-orders path length** once position is partialled out — `latent_7040\|raw` P1r-s **−0.2893 [−0.4039, −0.1702]** (unscreened VGEO-4: −0.1495). The fitted diagonal metric removes it (+0.0357). ⇒ the unweighted latent distance mostly measures something other than path length, and **the only geometry the trunk offers here comes from a learned metric**. | MEASURED, descriptive |

**Verdicts against the committed branches (SPEC §3): G1–G4 PASS ⇒ `P1r-s-TIE`; P2 gate PASS ⇒ `ID-TRUNK`.**

## 1 · The gates, in committed order

| gate | bar | read | |
|---|---|---|---|
| G1 C-oracle (P1r-s only) | ≥ 0.99 | **0.9978** [0.9969, 0.9986] | ✅ |
| G2 C-presid | \|point\| ≤ 0.03 | **+0.0180** [−0.0126, +0.0465] | ✅ |
| G3 C-const / C-mut | 0.0000 / \|point\| ≤ 0.05 | **0.0000** / **−0.0075** | ✅ |
| G4 surviving clips | ≥ 15 | **17** | ✅ (margin 2) |
| P2 C-const / pixel floor | 0.5000 / [0.45, 0.55] | **0.5000** / **0.5329** | ✅ |

## 2 · What was run

Bank, speed join and split identical to VGEO-3/4 (134/139 clips joined, 411 frames interpolated; fit 94 / scored 40, seed `20260913`); STEP 4, 6,704 rows, **116.8 s CPU**, below-normal priority. ⛔ **GPU not used**: `nvidia-smi` listed two `python.exe` compute contexts, so the charter's precondition did not hold. ⚠️ **Attempt 1 died at `import numpy`** (the system Python, not `venvs/tanitad`) **before computing anything**. Its log is kept as `raw/vgeo5_step4_attempt1_wrong_interpreter.log`, and the pin predates both attempts. Estimator: clip-cluster bootstrap, 2,000 resamples. Per `H-ESTIM-SEED-1` it answers *"another draw of CLIPS"* only: one checkpoint, one trunk.

## 3 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | Closes the question I-1's distance half has carried for four passes; feeds **AI13-2** (trunk freeze) and **L-3**. |
| **CONSEQUENCE** | Per the committed TIE branch: **a cost-to-go value head that reads path length does not get to claim the frozen trunk**. That is an MM decision, and it now rests on a valid instrument. ⭐ The same run gives the trunk a positive, replicated property: **place identity** (F4). ⇒ The value line should **move to what the trunk measurably carries**: a *place/progress-along-route* value, not a metric-distance one. |
| **COMBINATION** | Fits FROST-Drive (09-17): the frozen trunk is strong on *semantic identity*, and the width/readout is where geometry is lost (LR14-4). Fits F5: geometry appears only under a learned metric, which points at the **readout**, not the encoder. Fits Delta-JEPA (today, A2): displacement geometry must be *trained in*, it does not fall out of a frozen JEPA-style trunk. |
| **CHANCES / RISKS** | **Upside:** I-1 has a usable, powered instrument for the first time, and the screen is reusable on any trunk. **Risks:** (a) 17 clips, margin 2 over the bar; (b) the screened subsample is biased toward clips with variable speed (SPEC §2, renamed estimand); (c) one trunk; (d) a small trunk advantage (≤ +0.144) is not excluded. |
| **EXPERIMENT** | **`E-DEP-VGEO-6` (proposed DP18-1, 0 GPU):** run the *identical* screened panel on a **second trunk** (the s16 tokens in the same bank, `tokens_s16_fp16.npy`) and on the **readout-widened** 7040-d pooling variants, with **P2 as the primary**. **Committed:** ID-TRUNK replicates on s16 ⇒ place identity is a trunk-family property and the value line re-targets to it; it fails on s16 ⇒ the property is checkpoint-specific and not a design basis. |

## 4 · Stopping condition (Rule Zero)

**(3a) for the instrument; (3b) for the value head.** The pre-registered bar was **cleared** — every gate passed on the first screened run. The *scientific* answer is a TIE on path geometry, and it is reported as written. ⭐ **This pass did not stop at it:** the same run produced the positive verdict (F4) that tells the value line **where to go next**, and DP18-1 is the next arm (0 GPU, unblocked). The decision to re-target the value head off metric distance is the **Master Mind's** and is escalated rather than taken here.

`Deliverables: SPEC.md · RESULT.md · code/vgeo5_screened.py · raw/{prereg_pin.json, vgeo5_step4.json, vgeo5_step4.log, vgeo5_step4_attempt1_wrong_interpreter.log, search_log.md}`
