# RESULT — H-REFAV1-MOTION: **REFUTED for this injection form** (the shuffle control decided it)

**2026-09-02 · T0 mechanism probe · pre-registered criteria applied verbatim, no re-reading**
Raw: `raw/result.json` (600 steps × 3 arms) · `raw/result_shakeout30.json` (the 30-step
shakeout, readout untrained, never evidence) · `raw/run600.log` · episodes `raw/eplist.txt`.

## The read, against the SPEC's committed criteria

Held-out feature MSE (frozen `std(DINOv3)` space — fixed, cross-arm comparable), mean over
the 5 episode-disjoint val episodes, real DINOv3-L features, single seed:

```
     arm     k1      k2-5    k6-15   k16-30      [copy-last floor]
    base   0.4608   0.4845   0.5143   0.5460    [0.2311 0.3763 0.5138 0.6122]
  motion   0.4130   0.4407   0.4751   0.5032
 shuffle   0.4210   0.4438   0.4678   0.4780    <- the control
```

* ✅ criterion half 1: motion beats base at k∈[6,30] — **5/5 val episodes**, per-episode
  Δ +0.0349…+0.0458, no sign flips.
* ⛔ **criterion half 2 FAILS: the SHUFFLED-diff arm shows the same gain** — and at
  k16–30 it is *better* than the true difference (0.4780 vs 0.5032). The temporally
  WRONG difference helping equally is precisely the SPEC's committed failure branch:

> *"or the shuffle matches the true diff — then the channel is regularisation, not
> motion, and H-REFAV1-MOTION is refuted for this injection form."*

## What was learned (and what was NOT)

1. ⭐ **The extra input channel helps — as augmentation, not as motion.** Both diff arms
   beat base everywhere; the benefit does not require temporal adjacency. A cheap,
   honest gain is available (any second-frame channel), but it cannot be SOLD as motion.
2. **The control was load-bearing, again.** Without the shuffle arm this would have
   shipped as "cross-channel motion injection works (+8 %, 5/5 episodes)" — a
   plausible, wrong, pre-registered-looking claim.
3. ⚠️ Scope: 19/5 episodes, single seed, 600 steps, no-hierarchy arms, frozen target
   space. The refutation is of THIS injection form (initial-state difference) at probe
   scale — not of temporal information in general. V-JEPA-style *video* pretraining
   (Drive-JEPA's route) and multi-frame predictor inputs remain untested alternatives.
4. **Absolute levels, stated honestly:** at k1 every arm is still ~2× ABOVE the
   copy-last floor (600 steps is early; the frozen-space readout converges slowly),
   while at k16–30 every arm BEATS the floor. Arm *comparison* is matched-everything
   and stands; absolute quality claims do not leave this file.

## Consequence for refav1

`motion_inject` stays in the code, **default OFF**, with this refutation cited at the
definition site. It must not appear in a launch line without a new pre-registration
that separates augmentation from motion (e.g. a noise-channel third control).
