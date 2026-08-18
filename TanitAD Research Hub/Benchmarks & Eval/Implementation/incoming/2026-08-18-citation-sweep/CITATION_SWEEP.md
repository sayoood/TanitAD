# ⛔ THE LADDER / POOLING / O2-O4 / K1 CITATION SWEEP — what survives, what died, and the two "replacement" numbers that are themselves single-seed

**Date** 2026-08-18 · **Branch** `agent/arch-inf-20260803` · **Agent** citation-sweep
**GPU** ⛔ **none.** Documentation integrity only. Thor was not touched; no trainer, pod or cache was read.
**Eval tier** every number below is **T0-DIAGNOSTIC** — a frozen-latent linear readout is a world-model
diagnostic and is **never** driving performance. No ADE, no closed loop, nothing about how the car drives.
**Estimator** `taniteval.ci.paired_episode_cluster_bootstrap`, `n_boot 2000`, 70 episode clusters.
⛔ `overlapping_holdout_se` is never quoted, and no number re-quoted here is a `heldout` split-mean —
every value was re-derived by **opening the banked per-seed JSON**, not by copying a summary (C91).

---

## ⭐ THE HEADLINE, BEFORE ANY CORRECTION IS READ

⚠️ **A reader who takes "the ladder was corrected again" as "the ladder was wrong" will discard a real
result.** State the split every time:

| | |
|---|---|
| ✅ **SURVIVES AND STRENGTHENED** | **The v6 operative latent cannot linearly report ego speed**, and **the banked lead-gap signal is mostly an ego-speed proxy.** Both are now measured at **three seeds on both repair routes** and neither moved in direction. The ego-speed **scalar** the model is handed reads `lead_gap` at **MAE 3.5712, r +0.6835, r² 0.4672, K1 −1.5618 [−2.0229, −1.1363] separated PASS, guard OK — identical on all three seeds and both routes** — against the 2 048-dimension latent's **r² 0.0069** and a K1 that never passes. |
| ⛔ **DIED — not "changed", WITHDRAWN** | five claims, §2. |
| ⚠️ **MOVED, and the replacement moved again** | three numbers, §1 — and **two of the replacements in circulation are SINGLE-SEED values**, §1.2. |

---

## 1. THE CANONICAL RE-QUOTE TABLE — quote from here, and carry the provenance line with it

**PROVENANCE LINE, to be carried verbatim by any document quoting these:**

> `MEASURED` · arm **`v6F-SW-30k@11250`** ⚠️ **EARLY READ, 11 250 / 30 000 = 37.5 %** · **T0-DIAGNOSTIC** ·
> **130-clip lead-enriched probe pool**, 60 probe-train / **70 eval clips, clip-disjoint** ⚠️ **NOT the
> 40-episode val set** · ridge `intercept_col=-1` (C92) + C97 degeneracy guard · **3 inner-split seeds
> {0, 1, 2}** · estimator **paired episode-cluster bootstrap**, `n_boot 2000`, 70 clusters ·
> **route A (`unpen`)** unless route B is named — ⛔ **the two routes are NEVER pooled.**
> Artifacts: `…/incoming/2026-08-18-ladder-3seed/raw/reread_unpen/ll3_*.json` (route A),
> `…/raw/reread_centred/ll3_*.json` (route B), aggregated in `…/raw/reread3_table.json`.
> Re-derivation banked here: `raw/canonical_requote_table.json`.

### 1.1 Rung profile — `r²`, route A, `v6F-SW-30k@11250`

| rung | seed 0 | seed 1 | seed 2 | ⭐ **3-seed mean** | matched-random null (3-seed mean) | ⛔ **stale value still in circulation** |
|---|---|---|---|---|---|---|
| `n_agents_all` | 0.1519 | 0.1573 | 0.1745 | **0.1613** | 0.0002 | **0.076** |
| `nearest_any` | 0.0964 | 0.1004 | 0.0964 | **0.0977** | 0.0007 | 0.048 |
| `ego_v0` | 0.1032 | 0.0913 | 0.0756 | **0.0900** | 0.0007 | 0.061 |
| `n_agents_grid` | 0.0200 | 0.0880 | 0.0305 | **0.0462** | 0.0003 | 0.020 |
| `ego_accel` | 0.0161 | 0.0350 | 0.0051 | **0.0187** | 0.0001 | 0.035 |
| `lead_present` | 0.0118 | 0.0091 | 0.0053 | **0.0088** | 0.0000 | 0.009 |
| `lead_gap` | 0.0053 | 0.0097 | 0.0057 | **0.0069** | 0.0001 | 0.025 |
| `ego_yawrate` | 0.0009 | 0.0009 | 0.0015 | **0.0011** | 0.0001 | 0.004 |
| `lead_closing` | 0.0013 | 0.0000 | 0.0013 | **0.0009** | 0.0000 | **0.0000** |
| `lead_inv_ttc` | 0.0008 | 0.0008 | 0.0008 | **0.0008** | **0.0009** | 0.0001 |
| `ego_curv` | 0.0000 | 0.0000 | 0.0000 | **0.0000** (5e-06) | **0.0005** | **0.0001** |

⭐ **A ROUTE FACT MEASURED HERE AND NOT PREVIOUSLY STATED, AND IT DOES *NOT* LICENSE POOLING.**
`MEASURED` by differencing the two banked route files row by row: on **every rung above the `r²` values
are bit-identical between route A and route B**, except `ego_v0` **seed 0** (A **0.1031** / B **0.1034**).
⚠️ **This is a fact about `r²` on these rows only.** The **K1/K1B** numbers still differ by route exactly
as C100/C103 measured — `ego_v0`'s K1 differs by **0.3957** (A **+0.0317** / B **+0.4274**) and its K1B by
a factor of **8**. ⛔ **The pooling ban stands unchanged**; what this adds is that an `r²` citation does
not need to choose a route, while every K1 citation does — and must say which.

### 1.2 ⚠️⚠️ TWO NUMBERS THE RETRACTIONS OFFER AS REPLACEMENTS ARE THEMSELVES SEED-0 VALUES

This is the C103/C107 hazard one turn further on: the correction was itself taken at one seed, and
`LADDER_3SEED.md` §6a's own rule — *"any citation quoting an individual `r²` from this ladder must quote
the 3-seed mean and its seed spread, not a single seed's value"* — was not applied to them.

| quantity | quoted as the replacement (C103 / C107 / the sweep brief) | ⭐ **what it actually is** | ⭐ **the 3-seed form** |
|---|---|---|---|
| `lead_gap` `r_pv0` (partial-`v0` correlation) | **−0.107** | **route A/B seed 0** (−0.1065) | **−0.0884**, per seed **−0.1065 / −0.0665 / −0.0922**, ⚠️ **seed SPREAD [−0.1065, −0.0665] — a DISPERSION, NOT a confidence interval** |
| `lead_gap` latent-vs-null MAE margin | **"0.694 m WORSE than the random-latent null"** | **seed 0** (5.8694 vs 5.1754) | ⛔ **+0.283 m worse on the 3-seed mean, and the SIGN FLIPS on seed 1** (Δ **+0.694 / −0.017 / +0.173** m; on seed 1 the latent is 0.017 m *better* than the null) |

⇒ **The direction is unchanged and both still say "no latent lead-gap signal"** — but *"0.694 m worse"*
must not be quoted as a fixed effect, and **−0.107 must be written as "−0.0884 (3-seed mean; seed 0
−0.107)"** wherever it appears. ⚠️ **`n_agents_all` 0.1613, `lead_closing` 0.0009 and `ego_curv` 0.0000
ARE already 3-seed means** — those three replacements are sound as issued.

### 1.3 ⭐ AND ONE "STALE" PAIR IS ONLY HALF STALE — do not over-correct it

The pair **"K1 −1.562 PASS vs +1.580 FAIL"** appears in five documents. `MEASURED`:

| half | status |
|---|---|
| **`−1.562`** — ego speed ALONE (`C-V0`) on `lead_gap` | ✅ **NOT STALE. CONFIRMED AND SEED-STABLE:** K1 **−1.5618 [−2.0229, −1.1363]**, separated **PASS**, guard **OK**, **identical on all three seeds and on both routes**; MAE 3.5712, r +0.6835, r² 0.4672. |
| **`+1.580`** — the 2 048-dim latent on `lead_gap` | ⛔ **STALE — a PRE-C92 value from the biased floor.** Repaired, route A = route B: **+0.7364 [+0.1297, +1.4425] separated (guard `CONSTANT-OFFSET-ONLY`) / +0.0253 [−0.1124, +0.1632] not separated / +0.2155 [−0.1468, +0.6341] not separated.** ⇒ quote *"the latent's `lead_gap` K1 is +0.736 / +0.025 / +0.216 across seeds and never passes"*. |

⇒ **Fix the second number only.** The trivial-proxy finding — the thing the pair exists to say — is the
half that survived.

### 1.4 The `ego_v0` headline, stated at three seeds

`MEASURED`, route A: K1 **+0.0317 [−0.5318, +0.5081]** not separated (s0) · **+0.1643 [−0.1521, +0.4618]**
not separated (s1) · **+0.9870 [+0.2424, +1.6573] SEPARATED — the latent is separated WORSE than a
constant**, guard `CONSTANT-OFFSET-ONLY` (s2). Route B seed 0 is **+0.4274 [−0.1841, +0.9986]**.

⇒ ⚠️ **C103's *"the latent's ego-speed readout TIES a constant (K1 +0.032 [−0.532, +0.508])"* is a
route-A seed-0 statement and the tie is not seed-stable.** The 3-seed-honest form is stronger, not
weaker: ⭐ **the latent ties or loses to a constant on ego speed and never beats it**, while the
**EGO-ORACLE at 10× noise earns a guarded PASS on all three seeds and both routes** (K1 **−1.6037
[−2.4058, −0.8844]**, r **+0.8280**, guard OK). The comparison is **PASS-vs-(TIE-or-WORSE)**.

---

## 2. ⛔ WITHDRAWN — these did not move, they died

| claim | status | authority | what replaces it |
|---|---|---|---|
| *"the v6 latent reads scene density"* | ⛔ **WITHDRAWN** | C100 → C103 → C107 | It is **~80 % `v0`.** The latent's only 3-seed-stable guarded PASSes are `n_agents_all` at four checkpoints and **the single ego-speed scalar wins all four on the 3-seed mean** (margins +0.262 / +0.243 / +0.217 / +0.211 K1B, scalar-favouring), with **seed 0 the outlier on every one**. |
| *"the latent beats a random null by 1.6–1.8 m"* | ⛔ **WITHDRAWN — it INVERTS** | C92 → C97 → C103 | At the eval-optimal alpha (cheating in the arm's favour) the true margin is **~0.02–0.07 m — a 25–90× overstatement — and no alpha anywhere reaches a PASS.** On the repaired 3-seed read the latent is **+0.283 m WORSE** than the matched-random null (§1.2). The arm wins on the **inner split** and loses on **held-out episodes** ⇒ **episode-level overfitting, not agent geometry.** |
| *"the 40:1 pooling bottleneck explains D1"* | ⛔ **REFUTED** | C104 (E-R1-0, pre-registered) | Removing the pool entirely moves r² by **\|Δ\| ≤ 0.0002 with the CI containing zero on all five seeds**. **`R1 IS DROPPED` by its own pre-registered criterion.** The constraint is the **encoder/objective**: through the *same* deployed pool on the *same* windows, `facebook/dinov2-base` (**86 M vs our 87.3 M — not a capacity gap**) reads `lead_gap` **0.44997 vs 0.00496**, `ego_v0` **0.71733 vs 0.05240**, `lead_closing` **0.01713 vs 0.00000**. |
| *"random init beats trained 3.6× on both rungs"* | ⛔ **HALF WITHDRAWN; the RATIO withdrawn outright** | C109 | `ego_v0` **survives and gains a real estimator**: paired episode-cluster bootstrap Δr²c **+0.150 [+0.055, +0.226], p(Δ>0)=1.000**, positive in **27/27** cells. `lead_gap` **dies** — **0 of 27 cells CI-separated**, p(Δ>0) 0.71–0.76, **sign flips in 9/27**. ⛔ **The "3.6×" itself must go**: it compares a near-constant predictor (`pred_sd/gt_sd` 0.014) to a live one (0.89) and moves to **2.8× / 2.0×** when the ridge inner split is re-drawn. ⇒ **the admissible claim is SIGNAL-vs-NO-SIGNAL on one rung, not a ratio.** |
| *"seed spread is exactly zero on 8 of 11 rungs, so ≥3 seeds supply no uncertainty here"* | ⛔ **WITHDRAWN — measured under a defect** | C103 → C107 | The C92 intercept defect had **frozen the alpha sweep**. Measured on all 165 rows: the **defective** instrument picks the same alpha on all 3 seeds for **132 of 165** rows, the **repaired** one for **42** — a **3.1× drop in seed-stability caused by a repair**. ⚠️ Report the counter-column too: max K1 seed spread is *larger* on the incumbent (**4.239 vs 2.812**) ⇒ the honest claim is *"the repair unfroze the majority"*, **not** *"the incumbent had no variance"*. |

⚠️ **AND ONE CONTROL DIED UNDER A SURVIVING CONCLUSION** (C109): **`PC-2OBJ` — the positive control both
C104 and C106 cite — is INERT AT THE DEPLOYED POOLING RATIO BY CONSTRUCTION** (two *opposing* plants
inside one cell cancel; run at p40 it reproduced the un-planted arm to **5e-05**). ⇒ ⛔ **C104's
sentence *"the instrument had full power to see a pooling-destroyed signal"* must cite `PC-LOCAL` /
`PC-DIST`** (our own trained tokens through the deployed pool, **0.0596 → 1.0000, K1 9/9**), **not
`PC-2OBJ`**. ⭐ **C104's conclusion is NOT overturned** — PC-LOCAL/PC-DIST do fire and the 40:1→1:1 null
stands — but the headline control was the wrong one, which is the D1/C79 shape.

---

## 3. ⭐ HOW TO CITE INTO THESE DOCUMENTS — the anchor rule

⚠️ **Every document corrected here was corrected IN PLACE, so every line-number citation into it is now
invalid** — and this sweep created the next round of them (C103's hazard, C90 inverted).

⛔ **RULE, adopted by this sweep and applied to every correction block it wrote:**
**cite by SECTION HEADING and QUOTED PHRASE, never by line number.** A heading survives an in-place
rewrite; `:178-194` does not. Where a line-number citation was found, it was replaced by the heading it
pointed at, and the heading text was left unchanged so the anchor stays stable.

Correction blocks written by this sweep are marked `⛔ CORRECTION 2026-08-18 (citation sweep)` and are
**additive** — no prior text was deleted from a live-stream document, so the superseded claim stays
visible beside its replacement.

---

## 4. THE SWEEP ITSELF — how the site list was built, and why it is not a hand-list

⛔ **A hand-listed set is what C99 and C105 punished, twice.** The site list was derived by **content
pattern over the whole tracked tree**, three disjoint families, then intersected with the numbers the
retractions moved:

```
git grep -l -I -E "n_agents_all|n_agents_grid|lead_closing|ego_v0|ego_yawrate|\bK1B\b|LATENT_LINEAR_LADDER|pc6_linear_readout" -- '*.md'
git grep -l -I -E "40:1|AvgPool2d|pooling bottleneck|POOLING_BOTTLENECK|37\.8 px|40 ViT tokens"        -- '*.md'
git grep -n -I -E "\+1\.580|1\.562|\+0\.159|\+0\.052|n_agents_all[^0-9]{0,40}0\.076|1\.6.{0,4}1\.8 m"  -- '*.md'
```

⚠️ **`git grep` was used rather than a shell loop precisely because of C107's finding**: this repo's paths
contain spaces (`TanitAD Research Hub`, `Architecture & Inference`), and an unquoted/word-split
verification degenerates into comparing empty-to-empty and reports success — *"360 files, 0 mismatches"*.
**Every path handled in this sweep was passed `-z`-safe or fully quoted, and the staging check is a blob
comparison, not an exit code.**

### 4.1 The site inventory

| document | stale content found | action taken |
|---|---|---|
| `…/Research/2026-08-18-pooling-bottleneck-R1R2/POOLING_BOTTLENECK_R1R2.md` | ⛔ **no C104 banner at all** — the whole R1 premise is refuted and the document still reads as a live pre-registration; §1.5 quotes `n_agents_all` **0.076** / `ego_curv` **0.0001** / `lead_closing` **0.0000** / `r_pv0` **+0.052, r² 0.0027**, with **five line-number citations** into a twice-rewritten file; §1.7's `predictor_op` **68.5 %** is C104's scope error | ⭐ **top banner + §1.5 correction block + §1.7 note** (additive) |
| `…/Research/2026-08-17-O234-DESIGN-RESEARCH.md` | banner is current through C104 but predates C106/C107/C109; §1 item 4, §3.4's table, **§3.4a's r² row**, §5, §8 item 2 and the **E-PROBE-A / E-PROBE-C rows** all carry the pre-repair numbers; banner quotes `336,559,305` (C106: **336,542,025**) and `PC-2OBJ` (C109: inert) | ⭐ **banner extension + §3.4a correction block + E-PROBE row note** |
| `…/incoming/2026-08-18-pooling-ladder-ER10/POOLING_LADDER_ER10.md` | C104's own artifact; cites `PC-2OBJ` as the positive control that establishes power | ⭐ **C109 correction block** |
| `…/Research/2026-08-18-encoder-experiments/PREREG_ENCODER_EXPERIMENTS.md` | carries C106 but predates C109 — *"3.6× on both rungs, on all three seeds"* | ⭐ **C109 correction block** |
| `…/incoming/2026-08-18-k1-degeneracy-guard/K1_DEGENERACY_GUARD.md` | C100's artifact; no C103/C107 banner; its one substantive survivor is a seed artefact | ⭐ **supersession banner** |
| `…/incoming/2026-08-18-o2-live-and-ridge-reread/O2_LIVE_AND_RIDGE_REREAD.md` | *"partialling `v0` out drops r from +0.159 to +0.052"* | ⭐ **inline correction** |
| `Project Steering/Reports/2026-08-17-2319-program-report.md` | §3 EFFICIENCY: *"~1.8 m better than the random null, r +0.159"* — the exact sentence C92 retracts | ⭐ **correction note** (a dated report; the original sentence is preserved) |
| `…/incoming/2026-08-17-latent-linear-ladder/LATENT_LINEAR_LADDER.md` | current through C107 — but §5.1/§12 quote `r_pv0` **−0.107** and *"0.694 m worse"* as if 3-seed, and its pooling-thesis section predates C104 | ⭐ **seed-provenance correction + C104 note** |

### 4.2 ⚠️ NOT TOUCHED, and why — stated so the absence is visible

| document | why not |
|---|---|
| `Project Steering/RETRACTION_LOG.md` | **append-only and serialised**; several agents are live. This sweep's findings (§1.2, §1.3) are **proposed as an entry in §6 below, deliberately not appended.** |
| `…/incoming/2026-08-17-probe-positive-control/PROBE_POSITIVE_CONTROL.md` and its `raw/RENDER_TABLES.md` | the **primary record of a withdrawn result** (D1/C79). Its `+1.580` / `+0.159` are the *pre-repair* values it measured; rewriting them would destroy the record the withdrawal rests on. ⚠️ It is already headed *"D1 IS WITHDRAWN"*. Its 37.77 px lead-width figure is unaffected. |
| `…/incoming/2026-08-17-latent-linear-ladder/raw/RENDER_TABLES.md` | a **rendered raw artifact**, not prose. Regenerated output must match its run; editing it would break the correspondence to `raw/ll_*.json`. |
| `Project Steering/MODEL_REGISTRY.md` | **re-verified this sweep**: `git grep` for every ladder rung name returns **no ladder number in it**. C103's check holds. |
| `Paper/TANITAD_PAPER.md` | ⭐ **VERIFIED: zero ladder, pooling-ratio, O2–O4 or K1 numbers** (grep over all eleven rung names, `K1B`, `AvgPool2d`, `40:1`, `pc6_linear_readout` → **0 hits**). Its two `pooling` mentions are (a) an architecture description and (b) the **published DINO-WM claim** *"pooling is where geometry goes to die"* used as the rationale for the **LF0** future-work lever. ⛔ **Not a stale citation — but LF0 has effectively been EXECUTED and returned negative.** Escalated, §6.3. |
| `Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md` | same shape: it presents **RC1 "pooling bottleneck"** explicitly as one of **four root-cause HYPOTHESES**, quoting DINO-WM, not our measurement. ⛔ **RC1 is now measured false on our stack** — flagged in §6.3, not edited: it is a **dated report**, and rewriting a dated report's hypothesis list would destroy the record of what was believed when. |
| `PROJECT_STATE.md`, `Project Steering/BACKLOG.md`, `Project Steering/LOOP_STATE.md`, `PROGRAM_OVERVIEW.md`, `Project Steering/MODEL_REGISTRY.md` | ⭐ **VERIFIED by the same eleven-rung grep — 0 hits in each.** No ladder, pooling, O2–O4 or K1 number in any of them. C103's registry check holds and is now re-confirmed against a wider pattern. |
| the 2026-07-28 `artifacts/tables.md` family, `EGOPROGRESS_*`, `BOUNDED_TERMS_*`, `PSS_*`, `RECOVERY_*`, `CLOSEDLOOP_*` | matched only on `lead_gap` used as a **driving/headway metric name in a different instrument**, not the ladder rung. **False positives, verified by opening them.** |

---

## 5. THE FOUR METRIC FAMILIES

Per the binding rule, each family is addressed with its reason and `n` where it does not apply.
⛔ **ADE is not reported and is not applicable** — this is a documentation-integrity sweep over a
frozen-latent state readout, not a trajectory eval.

| family | what this sweep touched | verdict |
|---|---|---|
| **LONGITUDINAL** | the target-speed rung (`ego_v0`) and the distance-keeping rungs (`lead_gap`, `lead_closing`, `lead_inv_ttc`) are the rungs whose citations were stale | ⛔ **Nothing quotable becomes quotable.** The corrected numbers say the family's own control variable is read better by a single scalar the model is handed, and its time-gap/TTC half sits at or below its own null. |
| **LATERAL** | `ego_yawrate` (3-seed mean r² **0.0011**, one "survivor" reproduced by a `torch.randn` cache), `ego_curv` (**0.0000**, **below** its own null 0.0005) | ⛔ **Nothing quotable**, and ⚠️ **both rungs still lack a positive control** — an unverified negative, carried forward unchanged. |
| **TACTICAL** | ⚠️ **NOT MEASURED, n = 0.** The ladder is regression-only; no manoeuvre label is banked in these caches. | ⚠️ Absent by instrument scope. **A work item, not an excuse** — carried forward as `LATENT_LINEAR_LADDER` §14 item 6. |
| **STRATEGIC** | ⚠️ **NOT MEASURED, n = 0.** No route or goal label exists in the 130-clip lead-enriched pool, and per `CLAUDE.md` PhysicalAI-AV carries no map, lane graph or route signal at all. | ⚠️ Not computable on this corpus with this cache. |

---

## 6. ⛔ ESCALATIONS — decisions, not README notes

1. ⭐ **PROPOSED `RETRACTION_LOG.md` ENTRY — text ready, DELIBERATELY NOT APPENDED**, because the log is
   serialised and several agents are live. ⚠️ **Number it at write time.**
   > **C1xx — THE CORRECTION'S OWN REPLACEMENT NUMBERS WERE SINGLE-SEED, AND ONE "STALE" PAIR WAS ONLY
   > HALF STALE (2026-08-18, citation sweep).** The sweep mandated by C103/C107 re-derived every moved
   > number from the banked 3-seed JSON rather than from the retractions
   > (`…/incoming/2026-08-18-citation-sweep/raw/canonical_requote_table.json`). **Two of the replacements
   > in circulation are seed-0 values**: `lead_gap` `r_pv0` **−0.107** is seed 0 — the 3-seed mean is
   > **−0.0884**, per seed −0.1065 / −0.0665 / −0.0922, and that bracket is a **SEED SPREAD, NOT A
   > CONFIDENCE INTERVAL**; and *"the latent is **0.694 m** worse than the random-latent null"* is seed 0
   > — the 3-seed mean is **+0.283 m worse and the SIGN FLIPS on seed 1**. ⚠️ **C103's *"the latent TIES
   > a constant on `ego_v0`, K1 +0.032 [−0.532, +0.508]"* is likewise route-A seed-0, and the tie is not
   > seed-stable** — on seed 2 the latent is **separated WORSE** (+0.987 [+0.242, +1.657]). ⭐ **The
   > direction of all three strengthens; only their form was wrong.** ⛔ **And the opposite error was
   > found too: the pair *"K1 −1.562 PASS vs +1.580 FAIL"* is HALF stale — `−1.562` (ego speed alone) is
   > CONFIRMED, seed-stable on all three seeds and both routes; only `+1.580` is the pre-C92 value.**
   > ⇒ **ROOT-CAUSE CLASS: A CITATION SWEEP THAT COPIES THE RETRACTION INSTEAD OF THE ARTIFACT
   > PROPAGATES THE RETRACTION'S OWN PROVENANCE ERRORS.** `LADDER_3SEED.md` §6a had already written the
   > rule — *"quote the 3-seed mean and its seed spread, not a single seed's value"* — and the retraction
   > summarising it did not apply the rule to itself. **A sweep must re-open the JSON, and must check
   > each half of a compound claim separately: a pair can be half-refuted, and correcting the surviving
   > half is as damaging as leaving the dead one.**
2. ⛔ **DECISION-GRADE — `POOLING_BOTTLENECK_R1R2.md` IS A LIVE PRE-REGISTRATION FOR AN EXPERIMENT THAT
   HAS ALREADY BEEN REFUTED.** It carried **no C104 banner at all**. A banner was added, but ⏳ **the
   document's status (`DESIGN + PRE-REGISTRATION`) is the PI's to retire or rescope** — this sweep does
   not have the authority to mark another stream's pre-registration dead, and R2 (the *target* axis) was
   **promoted**, not dropped, so the file is not simply obsolete.
3. ⭐⭐ **DECISION-GRADE, AND IT IS NOT EDITORIAL — `LF0` / `RC1` HAVE ALREADY BEEN ANSWERED, NEGATIVELY,
   AND NEITHER DOCUMENT KNOWS IT.**
   * `Paper/TANITAD_PAPER.md` future-work: **LF0 — *"locate first: probe the pre-pool spatial tokens"***,
     justified by the PUBLISHED DINO-WM line *"pooling is where geometry goes to die"*.
   * `Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md`: **RC1 "pooling
     bottleneck"**, listed as one of four label-free-testable root-cause hypotheses for the missing
     lead-distance variable.
   ⛔ **`E-R1-0`'s 1:1 arm IS the pre-pool probe LF0 specifies**, and it returned **`|Δr²| ≤ 0.0002`
   with the CI containing zero on all five seeds** ⇒ **RC1 is measured FALSE on our stack, and LF0 as a
   *diagnostic* is spent.** ⭐ **This is a saving, not a loss**: LF0 need not be run, and its budget
   should move to the encoder arms the evidence indicts. ⚠️ **The remaining three hypotheses (RC2
   rare-event / RC3 no spatial masking pressure / RC4 horizon) are untouched by C104 and stay open** —
   ⛔ **do not let "RC1 is dead" be read as "the RC list is dead".**
   ⛔ **Escalated, not edited.** The paper is another stream's file; the addendum is a **dated report**
   and rewriting its hypothesis list would destroy the record of what was believed when.
   ⏳ **The decision is the paper stream's: retire LF0, or restate it as an executed negative.**
4. ⚠️ **THE ALPHA GRID IS STILL BINDING ON THE LADDER ROWS THIS SWEEP RE-QUOTES.** `INHERITED` from
   `LADDER_3SEED.md` §2.2: the chosen alpha sits at a **grid edge** on **78 / 94 / 82 of 176** rows.
   ⭐ C109 partially closes it — widening α to **1e13** changed the encoder rows by ≤ 0.0008 — but that
   was measured on the **C106 adversarial rows**, not on the 176 ladder rows. ⛔ **Do not inherit C109's
   reassurance onto the ladder; it is a different row set.** *(That is the scope error this programme
   keeps paying for.)*

---

## 6a. SUITES, STAGING, AND THE INDEX I DID NOT PUT THINGS IN

**Suites — NOT RUN, and the reason is scope, not convenience.** `MEASURED`:
`git diff --cached --name-only -- stack taniteval` **from this agent's changes is EMPTY.** This sweep
modified **eleven files, none of them under `stack/` or `taniteval/`** — eight hub/steering `.md`
documents plus three new files in this incoming directory. Briefed baselines (`stack` **3893 / 0**,
`taniteval` **1136 / 0**) are therefore **untouched by construction**. ⚠️ C107's operational note also
applies: a suite run as a gate while a multi-process CPU job is live returns FAILs that are about the
concurrency, not the code.

⚠️ **BUT THE SHARED INDEX IS NOT EMPTY OF `stack/`, AND IT IS NOT MINE.** `MEASURED` at the end of this
turn, **7 `stack/` paths sit staged by sibling streams**: `scripts/launch_closure_audit.py`,
`scripts/pod_git_drift.py`, `scripts/scoped_commit.py`, `tanitad/models/v6.py`,
`tests/test_launch_closure_audit.py`, `tests/test_pod_git_drift.py`, `tests/test_v6_frozen_external.py`.
⛔ **I did not write, edit, or stage any of them.** Recorded here because `CLAUDE.md`'s git-hygiene rule
is explicit that a pathspec-free commit sweeps the **whole index** — so whoever commits next must know
these are present and say so in the message.

**Staging verified by BLOB COMPARISON, not by an exit code.** `git add` exit codes are not evidence,
and `--cached` is insufficient for a **modified tracked** file — it answers *"is this path in the
index?"*, which is **yes** even when the index still holds the pre-edit blob. Each of the 11
deliverables was checked as `git ls-files --stage` (index blob) **against** `git hash-object`
(worktree blob): ⭐ **11 files checked, 0 blob mismatches.**
⭐ **AND THE CHECK WAS FALSIFIED, because a check that cannot fail is not a check (C107).** Run against
a control set it correctly reported **2 failures** — `CLAUDE.md` as `BLOB DIFFERS` (it is genuinely
modified-and-unstaged by another stream) and a non-existent path as `NOT IN INDEX`. ⚠️ **Paths were
fed NUL-separated**, never word-split: every path in this repo contains spaces, and the split form
compares empty-to-empty and reports success.

---

## 7. DELIVERABLE MANIFEST

**All paths relative to the repo root** `G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\`.
⛔ **Nothing produced by this sweep lives in only one place** — every artifact is in the repo and staged.

| artifact | path | what it is |
|---|---|---|
| **this sweep record** | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-18-citation-sweep/CITATION_SWEEP.md` | the findings, the canonical table, the not-touched list |
| ⭐ **the re-derivation** | `repo:…/2026-08-18-citation-sweep/raw/canonical_requote_table.json` | every number in §1, opened from the banked per-seed JSON (not copied from a retraction) |
| **the extractor** | `repo:…/2026-08-18-citation-sweep/code/build_requote_table.py` | reads `…/2026-08-18-ladder-3seed/raw/reread_{unpen,centred}/ll3_*.json`; computes nothing about the model |
| correction block | `repo:…/Architecture & Inference/Research/2026-08-18-pooling-bottleneck-R1R2/POOLING_BOTTLENECK_R1R2.md` | top banner + §1.5 + §1.7 |
| correction block | `repo:…/Architecture & Inference/Research/2026-08-17-O234-DESIGN-RESEARCH.md` | banner extension + §3.4a + E-PROBE rows |
| correction block | `repo:…/Architecture & Inference/Research/2026-08-18-encoder-experiments/PREREG_ENCODER_EXPERIMENTS.md` | C109 |
| correction block | `repo:…/Architecture & Inference/Implementation/incoming/2026-08-18-pooling-ladder-ER10/POOLING_LADDER_ER10.md` | C109 / `PC-2OBJ` |
| correction block | `repo:…/Architecture & Inference/Implementation/incoming/2026-08-18-k1-degeneracy-guard/K1_DEGENERACY_GUARD.md` | C103/C107 supersession |
| inline correction | `repo:…/Architecture & Inference/Implementation/incoming/2026-08-18-o2-live-and-ridge-reread/O2_LIVE_AND_RIDGE_REREAD.md` | `r_pv0` |
| inline correction | `repo:…/Architecture & Inference/Implementation/incoming/2026-08-17-latent-linear-ladder/LATENT_LINEAR_LADDER.md` | seed provenance of `−0.107` / `0.694 m`; C104 note |
| correction note | `repo:Project Steering/Reports/2026-08-17-2319-program-report.md` | §3 EFFICIENCY |

**Inputs read and NOT copied:** `…/incoming/2026-08-18-ladder-3seed/raw/**` (the 3-seed route-A and
route-B JSONs and `reread3_table.json`) and `…/incoming/2026-08-18-c106-adversarial/raw/**` — both
already banked in-repo by their producing streams.

⛔ **STAGED, NEVER PUSHED.** No `git commit` and no `git push` was run by this agent.
