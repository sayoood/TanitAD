# ⚠️ v2corpus was trained on ~half of v1's validation set

**MEASURED 2026-07-29 04:2x UTC, with the v2corpus arm still at step 25,900 / 30,000 — i.e. BEFORE
the checkpoint exists and BEFORE any comparison number was computed.** This is the void check
mandated by `Project Steering/PREREG_v2corpus_vs_v1.md`, run as its own first step exactly so a
contaminated number never gets published.

Artifact: `v2bal_val40_leak_check.json`.

## The finding

| quantity | value |
|---|---|
| v2bal selection | **9,000 clips** → 8,391 distinct `episode_id` |
| canonical val40 surface | **40 episode_ids** |
| ⚠️ **intersection (val episodes inside v2corpus TRAIN)** | **21** |
| leak-free val episodes remaining | **19** (harness bar is ≥ 8) |

**⇒ 52.5 % of the 40-episode canonical validation surface is inside v2corpus's training corpus.**

## Why this is REAL and not a collision artifact

`episode_id = int.from_bytes(clip_id[:4])` collides (9,000 clips → 8,391 distinct ids), so an
intersection *can* raise false positives. The base rate settles it:

- v2bal selected **9,000 of the 18,731-clip pool = 48.0 %**.
- Observed overlap: **21 / 40 = 52.5 %**.

**These agree.** That is precisely the overlap expected when the validation episodes live in the same
PhysicalAI pool and roughly half that pool is selected. A collision artifact would produce a small
overlap on top of a near-zero true rate, not one that lands on the selection fraction.

⇒ **The leak is structural, not incidental: v2bal re-selected from the whole pool without excluding
v1's validation episodes.**

## What this does and does not void

⛔ **The v2corpus ↔ v1 contrast CANNOT be run on the full 40-episode surface.** Scoring v2corpus
there would be scoring it on its own training data for half the episodes, which inflates it — and the
inflation is one-sided, so it would *manufacture* a "more data helps" result.

✅ **The comparison is still possible on the 19 leak-free episodes**, which clears the harness's
≥ 8 bar. But:
- ⚠️ **19 episode clusters is a much weaker interval** than 40. The paired episode-cluster bootstrap
  resamples episodes, so the CI will widen substantially. **A tie on 19 clusters is far weaker
  evidence than a tie on 40** and must not be reported as if equivalent.
- ⚠️ The 19 survivors are **not a random subsample** — they are the episodes v2bal happened not to
  select, and the selection was manoeuvre-balanced. **The retained set may be biased toward
  lane-keeping**, i.e. away from the turns the v2 corpus was built to add. That cuts against the v2
  arm and must travel with any result.
- ⛔ **v1's published 0.4271 is a 40-episode number and is NOT the comparator on this surface.**
  v1 must be **re-scored on the same 19 episodes** or the contrast is between two different surfaces.

## Required next steps, in order

1. **Confirm at clip_id level.** This check is at `episode_id` granularity. Resolve the actual val
   clip_ids (via `discover_r0_clips → sorted → split_clips(val_frac=0.2, seed=0)`) and re-intersect.
   The base-rate argument above makes a reversal unlikely, but the number that voids an experiment
   should be exact.
2. **Re-score v1 on the 19 leak-free episodes** so both arms share a surface.
3. Run the contrast per the prereg, with **`leak_free_n = 19` in the headline**, not a footnote.
4. ⭐ **Consider whether the honest answer is to rebuild a clean val for the v2 line.** The v2 corpus
   was always intended as the base for the *next* flagship generation; it needs a validation split
   disjoint from its own training selection. Comparing it to v1 on v1's surface may simply be the
   wrong experiment.

## Root-cause class

**A corpus re-selection that did not inherit the previous generation's val exclusion.** The v2
selection pipeline optimised manoeuvre balance over the full pool; nothing in it knew that 600
specific episodes were reserved as validation for the parity line. Not a coding error — a **missing
constraint** in the selection spec.

⇒ **RULE: any corpus re-selection must take the incumbent validation episode list as an explicit
exclusion input, and must emit the intersection count as a build artifact.** Had the v2 build printed
`val_overlap = 21`, this would have been visible in July, not on the eve of the comparison.

## Evidence class

| claim | class |
|---|---|
| 21/40 intersection at episode_id granularity | **MEASURED (ours)** — `v2bal_val40_leak_check.json` |
| 9,000/18,731 = 48.0 % selection fraction | **MEASURED** — `V2_CORPUS_DESIGN.md` |
| "the leak is structural, not a collision artifact" | **INFERRED** from base-rate agreement (52.5 % vs 48.0 %) — strong, but clip_id confirmation is step 1 |
| "the 19 survivors may be biased toward lane-keeping" | **HYPOTHESIS** — follows from balanced selection, not measured |
