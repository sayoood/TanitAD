<title>SPEC E-DEP-VGEO-4 — the position-matched null, pre-registered before anything is computed</title>

# `E-DEP-VGEO-4` — pre-registration

`Research Lab · 2026-09-17 (LAB-RUN-014) · Deployment & Optimization · serves INJECTED row I-1 (top of lane) via DP15-1`
⛔ **Pinned before the code was run.** sha256 of this file and of `code/vgeo4_position_null.py` are recorded in `raw/prereg_pin.json`, written before the run computed anything.

---

## 1 · Why this run exists

`E-DEP-VGEO-3` was **VOID twice** (2026-09-15). Run 2's only surviving failure was the control
`C-vswap` (path lengths taken from a *different* clip) reading **+0.0942 [−0.0447, +0.2275]**
against a bar of |point| ≤ 0.05 — a cross-clip null that could not read zero because a
**population-level trend** (speed rises with position inside a clip, and the latent drifts with
position too) survives any swap of whole speed series.

DP15-1 pre-committed the repair: **a within-clip, position-matched null.** This SPEC executes it.

## 2 · Data (identical bank to VGEO-3; no new inputs)

* Bank `C:/Users/Admin/tanitad-caches/bevhead-20260913/tokens` — refcv5-v2 s32 tokens (`tokens_s32_fp16.npy`), `pix64_u8.npy`, `index.npz`.
* Speed `…/bev_gt/<sha12>.bevgt.npz`, field `ego_v_ms`, `deskew_source == "egomotion_alpamayo"`, invalid frames interpolated (count reported).
* `STEP = 4` primary, `STEP = 2` sensitivity. Fit/score clip split seeded `20260913`, disjoint, asserted — **the same split as VGEO-3**, so the runs are comparable.
* ⛔ **Tier: none.** A representation diagnostic on frozen features. No rollout, no driving, no four-family eval. Nothing here is a driving claim.

## 3 · The estimand — what changes from VGEO-3

VGEO-3's P1 read `Spearman(dist, pathlen)` within clip × frame-gap Δ. Both sides carry the
within-clip position trend, so the statistic cannot separate *"the latent orders path length"*
from *"both grow with position"*.

**P1r (this run's primary) is the PARTIAL Spearman controlling for the pair's start frame:**

```
r_partial = (r_dl − r_dp · r_lp) / sqrt((1 − r_dp²)(1 − r_lp²))
```
with `r_··` Spearman correlations over the pairs of one clip × one Δ; `d` = feature distance,
`l` = path length, `p` = start frame. Averaged over Δ (nanmean) per clip, then bootstrapped over
clips. The position trend is removed **from both sides by construction**, which is what VGEO-3's
swap-based null tried and failed to do.

## 4 · Controls — every one must read a KNOWN value, and they run in BOTH directions

⛔ Per `CLAUDE.md`: a control that can only pass is not a control. Two of these must read zero and
one must read ≈ +1, so an instrument that is merely inert cannot pass the panel.

| id | construction | committed bar | what its failure means |
|---|---|---|---|
| **C-presid** ⭐ | the DP15-1 null: within clip × Δ, regress `l` on start frame (rank space), **permute the residuals**, rebuild `l_null = fit + permuted residual`, recompute P1r | **\|point\| ≤ 0.03** on `latent_7040\|diag` | the position trend still leaks ⇒ **VOID**, and the partial correlation is not the right estimand either |
| **C-const** | `dist ≡ 0` | **exactly 0.0000** for P1r, **exactly 0.5000** for P2 | the harness is scoring something other than the distance |
| **C-mut** | permute row identities within clip | \|point\| ≤ 0.05 | the pairing is not what carries the signal |
| **C-oracle** ⭐ | `dist := pathlen + N(0, 0.01·σ)` — an oracle distance that *is* the target | **point ≥ +0.90** on P1r | ⛔ **the discriminating control.** If a by-construction-perfect distance does not read ≈ +1, the statistic is too weak to detect the effect and a null result is UNINFORMATIVE, not evidence of a tie |
| **pixel floor** | `pixel_1440` run through every arm | reported, not barred | a latent that does not beat raw pixels has added nothing |

⭐ **C-oracle is new and is the point.** VGEO-3's failures were all "a null did not read zero". A
panel of null-only controls cannot distinguish *"no effect"* from *"no power"*. C-oracle is the
positive control that makes a tie interpretable.

## 5 · Committed outcomes — written before the data was touched

**Gate order is binding: C-presid and C-oracle are read FIRST, and the primaries are not read at
all if either fails.**

| branch | condition | verdict |
|---|---|---|
| **VOID-NULL** | C-presid \|point\| > 0.03 | the null still leaks; report it, do NOT read the primaries, and the value line stays where 09-15 left it |
| **VOID-POWER** | C-oracle point < +0.90 | underpowered statistic; a tie would be uninterpretable. Report and stop |
| **P1r-TRUNK** | controls pass **and** `latent_7040\|diag − pixel_1440\|diag` P1r CI lower bound **> 0** | ⭐ the frozen trunk carries path geometry beyond pixels. I-1's distance half is UNBLOCKED on this trunk |
| **P1r-TIE** | controls pass **and** that CI contains 0 | ⛔ tie confirmed **with a valid null** — the three prior reads become four, and a cost-to-go value head does NOT get to claim this trunk. Escalate to the MM as a decision |
| **P1r-PIXELS** | controls pass **and** that CI upper bound **< 0** | pixels beat the trunk on path geometry; same consequence as TIE, stated more strongly |
| ⚠️ **WORSE-BRANCH** | any arm behaves off this table (per `MM-2`) | say so explicitly rather than force-fitting the nearest row |

**Second primary (F5 promoted from footnote to verdict, per DP15-1):**

| branch | condition | verdict |
|---|---|---|
| **ID-TRUNK** | `latent_7040\|raw − pixel_1440\|raw` **P2** CI lower > 0 **and** C-const reads exactly 0.5 **and** the pixel arm's own P2 is within [0.45, 0.55] | ⭐ the trunk identifies **which place** beyond a pixel-matched distractor — a verdict, not a footnote |
| **ID-TIE** | that CI contains 0 | place identity is a tie too, and the trunk has no measured read-out advantage of either kind |

⛔ **No post-hoc repair this pass.** VGEO-3 spent its one repair and VGEO-3b was VOID anyway. If a
control fails, the branch above fires and the pass reports it.

## 6 · Estimator and its scope

Clip-cluster bootstrap over scored clips, 2,000 resamples, 95 % percentile, paired where a
difference is reported. ⚠️ Per `H-ESTIM-SEED-1` it answers **"would another draw of CLIPS say
this?"** only. One checkpoint, one trunk: it is blind to training variance and to inference
variance. No lever effect is claimed from it.
