<title>I-1 position-matched null — the null is FIXED, the power control FAILED, and the failure has a measured cause</title>

# `E-DEP-VGEO-4`: the DP15-1 repair WORKS (null 0.0044 vs 0.0942), but `VOID-POWER` fires on C-oracle 0.8983 < 0.90 — and the deficit is collinearity, not sample size

**2026-09-17 · Research Lab (LAB-RUN-014) · Deployment & Optimization · serves INJECTED row I-1 (top of lane) via DP15-1**
⛔ **Tier: none.** A representation diagnostic on frozen features — no rollout, no driving, no four-family eval. Nothing here is a driving claim.
⛔ **Pre-registered:** `SPEC.md` sha256 `e1a68f73…`, `code/vgeo4_position_null.py` sha256 `645f14a0…`, both pinned in `raw/prereg_pin.json` **before** the run computed anything.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **The DP15-1 repair works. The position-matched null reads zero where the cross-clip null could not.** C-presid `latent_7040\|diag` **+0.0044 [−0.0163, +0.0259]** against a bar of \|point\| ≤ 0.03 ✅. VGEO-3's C-vswap — the control that voided the previous pass — read **+0.0942 [−0.0447, +0.2275]**. Replicated at STEP 2: **+0.0076 [−0.0035, +0.0179]**. | MEASURED `raw/vgeo4_step4.json`, `raw/vgeo4_step2.json` |
| **F2** | ⛔⭐⭐ **`VOID-POWER` FIRES. C-oracle reads 0.8983 [0.8128, 0.9681] against a committed bar of ≥ +0.90** — a miss of **0.0017**. Reported as written. Replicated at STEP 2: **0.8930 [0.8193, 0.9569]**. Per SPEC §5 the primaries are therefore **not read as verdicts**. | MEASURED |
| **F3** | ⭐⭐⭐ **The oracle deficit is ATTENUATION UNDER COLLINEARITY, not low power — and that is a measured statement, not a guess.** Per-cell, ρ(oracle, \|r_lp\|) = **−0.7116** while ρ(oracle, n_pairs) = **−0.0504**. Banded: \|r_lp\| < 0.5 → oracle **0.9989** (0 % of cells below 0.9); 0.5–0.8 → **0.9972** (0 %); 0.8–0.95 → **0.9383** (11.9 %); **\|r_lp\| ≥ 0.95 → 0.7644, 48.8 % below 0.9**. Cell **median 0.9979**, mean 0.9288. ⇒ In **86 of 348 scored cells (24.7 %)** path length is near-determined by frame position, and partialling position out there removes the signal *by construction*. ⭐ **Doubling the pairs did not help** (STEP 2 = 13,396 rows vs 6,704, oracle 0.8930) — which is exactly what the n-independence predicts. | MEASURED `raw/vgeo4_oracle_diag.json` |
| **F4** | ⚠️ **Descriptive only (no verdict): the path-geometry contrast is a tie leaning to pixels, for the fourth time.** `latent_7040\|diag − pixel_1440\|diag` P1r **−0.0936 [−0.2198, +0.0411]** (STEP 2: **−0.0696 [−0.1833, +0.0536]**). Same direction as VGEO-1, VGEO-2 and VGEO-3. | MEASURED, descriptive |
| **F5** | ⭐⭐ **Descriptive only, but it now replicates three times across two passes and two step sizes: the trunk wins on PLACE IDENTITY.** `latent_7040\|raw − pixel_1440\|raw` on pixel-matched distractors: **+0.2088 [+0.1324, +0.2790]** (STEP 2 **+0.2474 [+0.1825, +0.3155]**; VGEO-3b **+0.2468**). Its own controls pass: C-const exactly **0.5000**, pixel floor **0.5329 [0.5107, 0.5553]** (STEP 2 **0.5203**), both inside the committed [0.45, 0.55]. | MEASURED, descriptive |
| **F6** | ⛔⭐ **A defect in my own SPEC, found by its own gate: a BLANKET gate over heterogeneous primaries is itself a fault.** C-oracle is defined on the **P1r** statistic (an oracle *distance* vs path length, position partialled). It says nothing about **P2**, a 2-alternative-forced-choice on matched distractors whose own controls (C-const exactly 0.5, pixel floor in band) both passed. SPEC §5 nonetheless wrote *"the primaries are not read at all if either fails"*, so F5 is voided by a control that cannot speak to it. ⚠️ **I am reporting this rather than re-scoping it after the fact** — the correction is pre-registered for the next pass, not applied to this one. | MEASURED (the SPEC text) |

**Verdict against the committed branches: `VOID-POWER` (SPEC §5).** No verdict fires on either primary. **I-1 stays unblocked for a metric head and still blocked for trunk credit** — unchanged from 09-15.

---

## 1 · The control panel, in the pre-committed gate order

| control | committed bar | STEP 4 | STEP 2 | |
|---|---|---|---|---|
| **C-presid** (the DP15-1 repair) | \|point\| ≤ 0.03 | **+0.0044** [−0.0163, +0.0259] | **+0.0076** [−0.0035, +0.0179] | ✅ PASS |
| **C-oracle** (power) | point ≥ +0.90 | **0.8983** [0.8128, 0.9681] | **0.8930** [0.8193, 0.9569] | ⛔ **FAIL** |
| **C-const** | exactly 0 / exactly 0.5 | **0.0000** / **0.5000** | **0.0000** / **0.5000** | ✅ PASS |
| **C-mut** | \|point\| ≤ 0.05 | **−0.0053** [−0.0267, +0.0148] | **−0.0096** [−0.0329, +0.0121] | ✅ PASS |
| **pixel floor** (P2, raw) | reported | **0.5329** [0.5107, 0.5553] | **0.5203** [0.5038, 0.5369] | ✅ in band |

⭐ **C-oracle earned its place on its first outing.** VGEO-3's panel was null-only, so it could not tell *"no effect"* from *"no power"*. Adding one positive control that must read a known value converted an apparent fourth tie into a **measured statement about the instrument** — and F3 then localised the cause in 0.4 s of CPU.

## 2 · What was run

* **Bank:** refcv5-v2 s32 tokens, `C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens/`. **Speed join** from the A&I LiDAR-BEV GT (`…/bev_gt/<sha12>.bevgt.npz`, `ego_v_ms`, `deskew_source == egomotion_alpamayo`). **134 / 139 clips joined**, 5 excluded (ids in the JSON), 411 invalid frames interpolated. Fit 94 / scored 40 clips, seeded `20260913`, disjoint, asserted — **the same split as VGEO-3**, so the passes are comparable.
* **P1r is defined on 29 of the 40 scored clips** (the rest have < 5 % path-length variation at every gap); P2 on 29 (STEP 4) / 30 (STEP 2) triplet clips. STEP 4 = 6,704 rows / 133 s; STEP 2 = 13,396 rows / 243 s. 262 (STEP 4) and 244 (STEP 2) degenerate partial cells dropped at the 1e-6 denominator guard and counted in the JSON.
* **Dev-box state:** ⛔ **GPU was NOT used and was not available** — `nvidia-smi` showed four `python.exe` processes in its compute-apps list at 100 % util / 7.7 GB, so the charter's *"4060 only if no python compute"* condition did not hold. CPU only, below-normal priority.
* **Estimator:** clip-cluster bootstrap over scored clips, 2,000 resamples, 95 % percentile, paired where a difference is reported. ⚠️ Per `H-ESTIM-SEED-1` it answers **"would another draw of CLIPS say this?"** only — one checkpoint, one trunk; blind to training and inference variance. No lever effect is claimed from it.

## 3 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | Gates injected **I-1** and **L-3**; feeds the trunk-freeze decision **AI13-2**; F3/F6 are instrument-doctrine and touch every probe panel in the programme. |
| **CONSEQUENCE** | No claim is licensed. **But the block is now located:** it is no longer *"the null won't read zero"* (fixed, F1) but *"a quarter of our cells cannot answer the question at all"* (F3). That is a corpus property, not a method failure, and it is the first time this line has produced a reason rather than another void. |
| **COMBINATION** | ⭐⭐ **F3 is the same shape as the `overlapping_holdout_se` family in `CLAUDE.md`, one level down**: an estimator whose *question* is narrower than the claim hung on it. Here, in cells with \|r_lp\| ≥ 0.95, *"does the latent order path length beyond position?"* has no answer, because path length **is** position there. ⭐ **F6 is the `df` / `step_s` / cylindrical-FOV family exactly** — a rule that is true somewhere (C-oracle governs P1r) applied where it does not hold (P2). ⭐ **F5 now has three consistent reads**, and combines with the A&I BEV-head tie and VGEO-1/2/3: *the trunk beats pixels on WHICH PLACE and never on HOW FAR* — the split proposed on 09-15 survives a fourth read. |
| **CHANCES / RISKS** | **Upside:** the collinearity screen in §5 is cheap, pre-registerable, derived from a by-construction-known quantity (the oracle) rather than from the arms, and would make the next run's tie interpretable for the first time. **Risks:** (a) screening to \|r_lp\| < 0.8 discards ~44 % of cells and may leave too few clips — the screen must therefore report its surviving n **before** the contrast; (b) the surviving cells are a biased subsample (clips with variable speed), so the estimand narrows and must be renamed; (c) one trunk, one checkpoint, 29 clips. |
| **EXPERIMENT** | **`E-DEP-VGEO-5` (proposed DP17-1, 0 GPU, ~5 min):** re-run with a **pre-registered collinearity screen** — drop every (clip, Δ) cell with \|r_lp\| ≥ 0.80 **before** any statistic is computed. **Committed in advance, in this order:** (1) C-oracle on the retained cells must read **≥ 0.99** (F3 measures 0.9972–0.9989 there, so this is a real bar the run can miss, not a rubber stamp); (2) C-presid must still read \|point\| ≤ 0.03; (3) surviving clips must be **≥ 15** or there is no verdict and we say so; only then (4) read `7040\|diag − pixel\|diag`: CI lower > 0 ⇒ trunk carries path geometry; CI contains 0 ⇒ **tie confirmed with both a valid null and a powered statistic**, and the cost-to-go value line moves off the frozen trunk (MM decision). ⭐ **And scope C-oracle to P1r explicitly** so the P2 place-identity primary is gated by its own controls (F6). |

## 4 · What this changes (≤3)

1. ⛔ **Record `E-DEP-VGEO-4` as VOID-POWER in the claims register** — the third consecutive VOID on this line, and the first with a *diagnosed cause*. Do not cite F4 or F5 as results.
2. ⭐⭐ **Adopt the positive-control rule programme-wide: any probe panel that can only be failed by a null must also carry an arm that must PASS.** A null-only panel cannot separate *no effect* from *no power*, and this line spent two passes learning that. Proposed as a `CLAUDE.md` addition, not applied unilaterally.
3. ⭐ **A gate must be scoped to the primary its control speaks to** (F6). Blanket gates over heterogeneous primaries void evidence the control never examined.

## 5 · Stopping condition (Rule Zero)

**(3b) — the next lever is named, costed and unblocked, and this pass did not stop at the refutation.** When the committed bar failed I did not report and halt: I ran the STEP-2 sensitivity (confirming the failure is not sample size) and then the per-cell diagnostic (locating the cause), both in the same run, for 4 minutes of CPU. `E-DEP-VGEO-5` above needs **0 GPU and ~5 minutes** and is blocked on nothing. ⚠️ It is deliberately **not** run in this pass: SPEC §5 committed *"no post-hoc repair"*, and applying a screen chosen after seeing the arms would be exactly the goalpost move the rule exists to prevent. It is pre-registered here and belongs to the next pass.

## 6 · Manifest

| artifact | location |
|---|---|
| SPEC / pre-registration | `repo:TanitAD Research Lab/Deployment & Optimization/Research/2026-09-17-i1-position-matched-null/SPEC.md` |
| RESULT | `repo:…/RESULT.md` |
| Code | `repo:…/code/vgeo4_position_null.py`, `repo:…/code/vgeo4_oracle_diag.py` |
| Raw | `repo:…/raw/vgeo4_step4.{json,log}`, `…/raw/vgeo4_step2.{json,log}`, `…/raw/vgeo4_oracle_diag.json`, `…/raw/prereg_pin.json`, `…/raw/search_log.md` |
| Inputs (not copied: 63 GB token bank + LiDAR GT, MM/A&I-owned) | `devbox:C:/Users/Admin/tanitad-caches/bevhead-20260913/{tokens,bev_gt}/` ⚠️ **live only on the dev box** |
