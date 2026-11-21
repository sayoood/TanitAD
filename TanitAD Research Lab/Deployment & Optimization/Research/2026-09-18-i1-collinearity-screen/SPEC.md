<title>SPEC E-DEP-VGEO-5 — the collinearity screen, pre-registered before anything is computed</title>

# `E-DEP-VGEO-5` — pre-registration

`Research Lab · 2026-09-18 (LAB-RUN-015) · Deployment & Optimization · serves INJECTED row I-1 (top of lane) via LR14-1 / DP17-1`
⛔ **Pinned before the code was run.** sha256 of this file and of `code/vgeo5_screened.py` are recorded in `raw/prereg_pin.json`, written before the run computed anything.

---

## 1 · Why this run exists

`E-DEP-VGEO-4` (2026-09-17) fixed the null (C-presid **+0.0044**) but fired `VOID-POWER`: the
positive control C-oracle read **0.8983 < 0.90**. Its diagnostic located the cause: the oracle's
partial statistic attenuates where path length is near-collinear with frame position
(ρ(oracle, |r_lp|) = **−0.7116**; ρ(oracle, n_pairs) = **−0.0504**; |r_lp| < 0.8 bands read
**0.9989 / 0.9972**). VGEO-4's SPEC forbade a post-hoc repair, so the screen was pre-registered
for this pass (`…/2026-09-17-i1-position-matched-null/RESULT.md` §3 EXPERIMENT; LAB_BACKLOG LR14-1).

## 2 · The one change

`code/vgeo5_screened.py` is `vgeo4_position_null.py` (sha256 `645f14a0…`) with **exactly one
addition**: in P1r, every (clip, Δ) cell with `|Spearman(pathlen, start frame)| ≥ 0.80` (computed
on the TRUE path length, before any distance) is dropped, identically in every mode (real,
C-presid, C-mut, C-const, C-oracle). Bank, split (seed `20260913`), STEP 4, estimator: unchanged.

⚠️ **The estimand narrows and is renamed `P1r-s`:** *path-geometry ordering on cells where path
length is not determined by position* — a biased subsample (clips whose speed varies within the
gap). A verdict on P1r-s is not a verdict on all of P1r.

⛔ **Tier: none.** A representation diagnostic on frozen features. No driving claim.

## 3 · Committed outcomes — gate order is binding

**P1r-s gate (C-oracle is scoped to P1r-s ONLY — correcting VGEO-4's F6 blanket-gate defect):**

| step | condition | if it fails |
|---|---|---|
| G1 | C-oracle P1r-s point **≥ 0.99** | `VOID-POWER` again ⇒ the screen did not restore power; report, do not read P1r-s |
| G2 | C-presid P1r-s \|point\| **≤ 0.03** | `VOID-NULL` |
| G3 | C-const P1r-s exactly **0.0000**; C-mut \|point\| **≤ 0.05** | `VOID-HARNESS` |
| G4 | surviving scored clips (≥ 1 retained cell) **≥ 15** | `NO-VERDICT-N` — say so |

Only then read `latent_7040|diag − pixel_1440|diag` P1r-s (paired clip bootstrap):

| branch | condition | verdict |
|---|---|---|
| **P1r-s-TRUNK** | CI lower **> 0** | trunk carries path geometry beyond pixels where the question is answerable ⇒ I-1's distance half UNBLOCKED on this trunk |
| **P1r-s-TIE** | CI contains 0 | tie **with a valid null AND a powered statistic** ⇒ the cost-to-go value line moves off the frozen trunk (MM decision) |
| **P1r-s-PIXELS** | CI upper **< 0** | pixels beat the trunk; same consequence as TIE, stronger |
| ⚠️ WORSE-BRANCH | any arm off this table | stated explicitly |

**P2 (place identity) gate — its OWN controls, independent of C-oracle:** C-const P2 exactly
**0.5000** and `pixel_1440|raw` P2 within **[0.45, 0.55]**. Then `latent_7040|raw − pixel_1440|raw`
P2: CI lower > 0 ⇒ **ID-TRUNK** (a verdict, first time); CI contains 0 ⇒ **ID-TIE**. P2 is
untouched by the screen (it is not a per-cell partial statistic), so this is the first pass where
the three prior descriptive reads (+0.2088 / +0.2474 / +0.2468) can become a verdict.

⛔ **No post-hoc repair this pass.** If G1 fails, the next lever is named, not applied.

## 4 · Estimator and its scope

Clip-cluster bootstrap over scored clips, 2,000 resamples, 95 % percentile, paired for differences.
Per `H-ESTIM-SEED-1` it answers **"would another draw of CLIPS say this?"** only — one checkpoint,
one trunk (refcv5-v2 s32), blind to training and inference variance.
