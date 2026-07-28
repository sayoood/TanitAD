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
2. **The 4 s and 8 s failures are UNDIAGNOSED.** The leading hypothesis is a **dataset bound, not
   model divergence**: the cached episodes declare `max_horizon: 20` (= 2 s of future poses), so
   K=40 asks for ground-truth futures that do not exist. ⚠️ **I did not confirm this** — the log did
   not carry the assert's origin, and asserting a cause I have not read would be exactly the error
   this programme keeps logging. It is a hypothesis with an obvious test (build a val cache with
   longer futures), not a finding.
3. ⚠️ Therefore **"2 s is the usable ceiling" is NOT shown.** What is shown is that error grows
   super-linearly across the only range currently measurable, and that the range is *currently*
   capped by the data, not by an observed failure of the model.

## 4. Why this matters even at two points

If the 1.91 exponent holds even approximately, the cost of imagining further is severe: a 4 s plan
would carry ~4× the 2 s error before any planner mistake is added. That bears directly on the
hierarchy — **a strategic brain planning over long horizons cannot lean on this world model's
imagination without re-perception**, and the frequency of that re-perception is now a measurable
quantity rather than an assumption.

## 5. Next

Build a val cache with `max_horizon` ≥ 80 (8 s of future poses) and re-run the identical sweep. That
converts two points into four and turns the exponent from a slope into a fit. Until then the 1.91 is
a **direction**, not a law.

## 6. Provenance

`code/imag_sweep.sh` (reuses the harness's own `--canary-only` path; only K varies) ·
`raw_imag_K10.json` · `raw_imag_K20.json` · pre-registration `eb27a36`.
