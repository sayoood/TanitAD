# Pure imagination decays SUPER-LINEARLY — 2× the horizon costs 3.75× the error

**Pre-registered `PREREG_imagination_horizon.md` (`eb27a36`) BEFORE the runs.** MEASURED 2026-07-29
on pod3 (v5 and v2corpus untouched). Arm: **v1 `flagship4b-speedjerk-30k`**, 881 windows / 40 episodes.

## 1. What "camera not fed" already means here

The canary encodes an **8-frame context** and then rolls the operative predictor forward **in latent
space with no further frames**, grounding to SE(2). That IS the camera being cut off. v1's headline
**0.4271** is a 2 s pure-imagination number — the mode was never new, only unmeasured at other horizons.

## 2. Result — two horizons measured

| horizon | imagined steps K | **ADE** | n | ratio vs 1 s |
|---|---|---|---|---|
| **1 s** | 10 | **0.11246** | 881 | 1.00× |
| **2 s** | 20 | **0.42148** | 881 | **3.75×** |
| 4 s | 40 | — | — | **NOT MEASURED** |
| 8 s | 80 | — | — | **NOT MEASURED** (CUDA device-side assert) |

✅ **The 2 s value reproduces v1's canonical 0.42148 exactly**, so the sweep is wired to the same
rollout/grounding/SE(2) mechanics the registry anchors against — the 1 s point is trustworthy for
the same reason.

⭐ **Doubling the horizon multiplies error by 3.75×** — a growth exponent of **log₂(3.75) = 1.91**,
i.e. **near-quadratic**. Error does **not** accumulate like integration noise (which would give
≈√t, exponent 0.5) or even linearly. On these two points the decay is **super-linear**, which is the
**OUTCOME B** direction of the pre-registration: imagination degrades faster than the horizon grows.

## 3. ⛔ What is NOT established, and I am not going to imply it

1. **Two points do not fix an exponent.** 1.91 is the slope through exactly two measurements. It is
   consistent with quadratic growth; it does not establish it, and it must not be extrapolated —
   the programme's own rule forbids projecting a fit beyond 2× its fitted range, and this *is* the
   2× range.
2. ✅ **The 4 s / 8 s failures are now DIAGNOSED — and my hypothesis was only half right.**
   The assert is
   `indexSelectSmallIndex: Assertion 'srcIndex < srcSelectDimSize' failed` (`Indexing.cu:1478`)
   — an **`index_select` out of bounds**, i.e. an EMBEDDING LOOKUP past the end of a table, not a
   missing ground-truth label. ⇒ **the 20-step bound lives in the MODEL, not only in the data.**
   ⛔ **This overturns the fix I first proposed.** Building a val cache with longer futures would
   **not** unlock 4 s on its own — the predictor cannot be *asked* for a horizon index it has no
   embedding row for. Extending imagination requires a **model change (a larger horizon table) and
   a retrain**, which is a materially bigger commitment than rebuilding a cache.
3. ⚠️ Therefore **"2 s is the usable ceiling" is NOT shown.** What is shown is that error grows
   super-linearly across the only range currently measurable, and that the range is *currently*
   capped by the data, not by an observed failure of the model.

## 4. Why this matters even at two points

If the 1.91 exponent holds even approximately, the cost of imagining further is severe: a 4 s plan
would carry ~4× the 2 s error before any planner mistake is added. That bears directly on the
hierarchy — **a strategic brain planning over long horizons cannot lean on this world model's
imagination without re-perception**, and the frequency of that re-perception is now a measurable
quantity rather than an assumption.

## 5. Next — REVISED once the assert was read

⛔ **NOT** "just build a longer val cache" (my first answer). The bound is an embedding table in the
predictor, so a longer cache alone changes nothing.

The horizon question now splits in two, and only one half is cheap:
1. **Cheap, and worth doing first:** sweep *within* the trained range — K = 4, 8, 12, 16, 20
   (0.4–2.0 s). That yields **five** points instead of two and turns 1.91 from a two-point slope
   into a real fit, with **no model change at all**. If the exponent holds across five points inside
   the range, it is a property of this world model rather than an artifact of two samples.
2. **Expensive, and now a PI decision:** to measure beyond 2 s at all, the predictor needs a larger
   horizon embedding and a retrain. That is not a measurement — it is a new arm.

⇒ **Do (1) next. Do not spend a retrain on (2) until the in-range fit says the extrapolation is
worth buying.**

## 6. Provenance

`code/imag_sweep.sh` (reuses the harness's own `--canary-only` path; only K varies) ·
`raw_imag_K10.json` · `raw_imag_K20.json` · pre-registration `eb27a36`.
