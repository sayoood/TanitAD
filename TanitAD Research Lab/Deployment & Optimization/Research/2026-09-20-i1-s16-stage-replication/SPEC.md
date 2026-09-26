<title>SPEC E-DEP-VGEO-6 — does the place-identity verdict survive a change of trunk STAGE? Pre-registered before anything is computed</title>

# `E-DEP-VGEO-6` (DP18-1) — pre-registration

`Research Lab · 2026-09-20 (LAB-RUN-017) · Deployment & Optimization · serves INJECTED row I-1 (top of lane) via LR15-1 / DP18-1`
⛔ **Pinned before the code was run.** sha256 of this file, of `code/vgeo6_s16.py` and of `code/make_vgeo6.py` are recorded in `raw/prereg_pin.json`, written before the run computed anything.

---

## 1 · Why this run exists

`E-DEP-VGEO-5` (2026-09-18) cleared every gate it had pre-registered and returned the I-1 line's
**first verdict**: `ID-TRUNK` — the frozen refcv5-v2 trunk beats a matched pixel baseline at
**recognising the same place** (`latent_7040|raw − pixel_1440|raw` P2 = **+0.2088**, CI
**[+0.1329, +0.2782]**), while metric **path geometry** was a *powered* **TIE** (−0.0707,
[−0.263, +0.144]) ⇒ a metric-distance value head does not get the frozen trunk.

⭐ **A verdict on one readout is not a verdict on the trunk.** `ID-TRUNK` was measured on a single
stage — **s32** (704 ch, 8×20) — pooled to a 2×5 grid. If the same panel on a **different stage of
the same checkpoint** does not reproduce it, then "the trunk carries place identity" is really
"*this readout* carries place identity", and it is not a design basis for the I-1 value line.
LR15-1 pre-committed this run for exactly that reason, with **P2 as the primary**.

## 2 · The one change

`code/vgeo6_s16.py` is **generated** from `vgeo5_screened.py` (sha256
`990745cb1cfcca23398a6bba7570e1d38ef7eda9ad676dd57096ff9db7f21659`) by `code/make_vgeo6.py`
through **11 enumerated substitutions**, each asserted to fire exactly once. The derivation is the
audit trail; `diff -u` against the landed file is in `raw/vgeo6_vs_vgeo5.diff`.

The change is the encoder **stage** the two latent arms read:

| | VGEO-5 (landed) | VGEO-6 (this run) |
|---|---|---|
| token file | `tokens_s32_fp16.npy` | `tokens_s16_fp16.npy` |
| token tensor | (27664, **704**, **8**, **20**) | (27664, **352**, **16**, **40**) |
| global arm | `latent_704` = mean over 8×20 | `latent_352` = mean over 16×40 |
| gridded arm | `latent_7040` = 704 × **2×5** (4×4 blocks) | `latent_3520` = 352 × **2×5** (**8×8** blocks) |

⭐ **The readout GEOMETRY is held fixed at 2×5** and only the stage changes — so a difference is
attributable to the stage, not to how many cells the head sees. Everything else is byte-identical:
the bank, the speed source, the split seed (`20260913`), the collinearity screen (`|r_lp| ≥ 0.80`),
`DELTAS`, `MIN_PAIRS`, `P2TOL`, `P2MIN`, the estimator, **the pixel control** and **the P2 distractor
triplets** (built from pixels alone, before any latent is touched).

⛔ **Tier: none.** A representation diagnostic on frozen features. No driving claim.

## 3 · ⭐ The harness control — what MUST reproduce, and what MUST NOT

Because the pixel arm and the triplets are untouched, this run carries a control that the previous
four passes did not have: **parts of it must reproduce the landed numbers exactly.** An absence of
reproduction here is a broken harness, not a finding.

**H1 — must reproduce EXACTLY (fail ⇒ `VOID-HARNESS`, nothing is read):**

| quantity | landed value |
|---|---|
| `clips_kept` / `fit_clips` / `scored_clips` / `rows` | 134 / 94 / 40 / 6704 |
| `scored_clip_ordinals` | the same 40 ordinals |
| screen `cells_kept` / `cells_dropped` / `surviving_scored_clips` | 195 / 182 / 17 |
| `n_triplet_clips` and `p2_kept_per_clip_of_200` | 29 and the same 40 counts |
| `C_const` P1r / P2 **point** | 0.0000 / 0.5000 |
| `pixel_1440|raw` **point** P1r / P2 / P3 | +0.1116 / 0.5329 / 0.5923 |

⭐ **H2 — PREDICTED TO DIFFER, and predicted here so the difference cannot later be sold as a
finding.** `fit_diag` draws `rng.choice(len(L), cap)` with `cap = 40000000 // dims`, from the
**module-level** rng. The latent arms have different widths (352/3520 vs 704/7040 ⇒ caps
113636/11363 vs 56818/5681), so the shared rng stream **diverges before `pixel_1440`'s `fit_diag`
runs**. Therefore:

- every **`|diag`** arm (including `pixel_1440|diag`) may differ — it is a different nnls subsample;
- every **bootstrap CI** may differ — `boot()` draws from the same stream.

⇒ **The exact-reproduction claim of H1 is confined to rng-free point estimates.** `C_oracle` and
`C_presid` use a *fresh* `default_rng(SEED+7)` inside `p1r`, so their points should also be near-identical
(**not** gated as exact: they consume the latent-derived `diag` weights through `dist_fn`).

## 4 · Committed gates — order is binding

| step | condition | if it fails |
|---|---|---|
| **H1** | the table in §3 reproduces exactly | `VOID-HARNESS` — report, read nothing |
| **G2** | `C_presid` P1r \|point\| ≤ **0.03** | `VOID-NULL` |
| **G3** | `C_const` P1r exactly **0.0000**; `C_mut` P1r \|point\| ≤ **0.05** | `VOID-HARNESS` |
| **G4** | surviving scored clips ≥ **15**; P2 clips ≥ **20** | `NO-VERDICT-N` — say so |
| **G1** | `C_oracle` P1r point ≥ **0.99** (scopes the **secondary** P1r-s read ONLY, not P2) | `VOID-POWER` on P1r-s; **P2 is unaffected** and is still read |

### 4a · PRIMARY — P2, place identity

Read `latent_3520|raw − pixel_1440|raw` P2 (paired clip bootstrap, 2,000 resamples, 95 % percentile).
P2's own controls: `C_const` P2 exactly **0.5000** and `pixel_1440|raw` P2 within **[0.45, 0.55]**.

| branch | condition | verdict |
|---|---|---|
| **ID-REPLICATES** | CI lower **> 0** | place identity survives a stage change ⇒ it is a **trunk-family** property, and the I-1 value line re-targets from metric distance to **place / progress** (MM decision) |
| **ID-STAGE-SPECIFIC** | CI contains 0 | the s32 `ID-TRUNK` verdict is a property of **that stage's readout**, not of the trunk ⇒ **not a design basis**; I-1's representation half returns to open |
| **ID-PIXELS** | CI upper **< 0** | pixels beat s16 on place; same consequence as above, stronger |

⭐ **Pre-committed magnitude clause:** a replication that clears zero but whose point is **< half**
the s32 point (**< +0.1044**) is reported as **ID-REPLICATES-WEAK** and does **not** license the
re-target on its own — LR15-1's consequence is a design decision, and a verdict that only just
clears its own null is not a basis for one.

### 4b · SECONDARY — P1r-s, path geometry (reported, NOT a verdict)

VGEO-5 read a **powered TIE** on s32 (−0.0707, [−0.263, +0.144]). Committed in advance:

- s16 CI **contains 0** ⇒ the TIE is stage-independent; the committed VGEO-5 branch (no metric-distance
  value head on the frozen trunk) **stands unchanged**.
- s16 CI **lower > 0** ⇒ ⭐ a **STAGE FINDING**: path geometry is present at s16 and absent at s32.
  This **re-opens the distance half** — but it is reported only, and **acted on only under its own
  pre-registration**, because it would be a post-hoc-selected stage.

### 4c · What this run cannot say

The panel scores **one checkpoint** (refcv5-v2, `ckpt_step` 40284) on **two of its stages**. Per
`H-ESTIM-SEED-1` the bootstrap answers *"would another draw of CLIPS say this?"* and nothing else:
it is blind to training seed and to inference variance. **Two stages of one checkpoint is not a
trunk family**; a replication here makes "trunk-family property" *admissible*, not *established* —
the next lever for that is a second checkpoint, and it is named, not run.

⛔ **No post-hoc repair this pass.** If a gate fails, the next lever is named, not applied.
