<title>SPEC — what does the 6 s horizon cost at inference? (2026-08-31)</title>

# SPEC — E-LAB-DEPLOY-0831

`TanitAD Research Lab · Deployment & Optimization · 2026-08-31`
`Seeds: Thor inference/latency, the deploy pipeline. Cross-seed: V7_LAUNCH_GATE.md P4.`
`Class: MEASURED microbenchmark on the dev-box RTX 4060 + arithmetic. ⛔ Thor NOT touched.`

---

## 1. What & why

`V7_LAUNCH_GATE.md` P4 requires the recipe to reach a **6 s** horizon, and eventually the
**strategic 8–30 s** band. The gate discusses this as an *accuracy* question. **It is also a
latency question, and nobody has priced it.**

The structural fact: **a rollout step consumes the output of the step before it, so a K-step
rollout is K sequential forward passes.** No accelerator parallelises that away — batching
helps across *candidates*, never along the *chain*. At `dt = 0.1`:

| band | horizon | flat K |
|---|---|---|
| operative | 0–2 s | 20 |
| tactical | 2–6 s | **60** (the live `k60p30k` arm) |
| strategic | 8–30 s | **300** |

⇒ the question this package answers: **what per-step latency does a flat rollout need in order
to fit a 10 Hz control loop, and is that achievable?**

⛔ **Anti-duplication preflight.** The deployment line already covers quantisation thoroughly —
`2026-08-28-vit-int8-fp8-thor`, `2026-08-29-quant-gate-spec`, `2026-08-30-b1-gate-strongly-typed`
— and `2026-07-18-predictor-cudagraph-and-numerics-sweep` already measured **one** predictor
pass (6.08 → 2.36 ms under manual CUDA-graph capture, 2.57×, on this same 4060). **None of them
prices SEQUENTIAL DEPTH.** That is the gap this fills, and the 2026-07-18 numbers are used as
context, not re-measured.

## 2. Hypotheses, with criteria committed IN ADVANCE

**H-LAB-LAT-1.** *Single-stream rollout latency is linear in sequential depth K.*
- **SUPPORTED if** `p50(K)/K` is approximately constant across K ∈ [1, 120].
- **REFUTED if** it falls materially with K ⇒ ⛔ **that would most likely mean my timer never
  synchronised**, not that the GPU parallelised the chain. This is why C1 exists.

**H-LAB-LAT-2.** *A flat 60-step rollout fits a 10 Hz tick.*
- **SUPPORTED if** measured K=60 latency leaves room for the encoder and the rest of the tick.
- **REFUTED if** it does not ⇒ **temporal abstraction becomes a deployment REQUIREMENT rather
  than an accuracy preference**, which is the outcome that would change how P4 is argued.
- ⚠️ Both outcomes committed before the run.

**H-LAB-LAT-3.** *The strategic band is reachable by flat rollout.*
- Evaluated at K=300 by extrapolation from the measured linear fit. ⚠️ K=300 is **not measured**;
  K=120 is, and the extrapolation is labelled as such wherever it appears.

## 3. Controls — mandatory, and each must read a known value

| id | control | what it must read | why it exists |
|---|---|---|---|
| **C0** | K iterations of a Python no-op | ≈ the timer floor, and **flat in K** | if a no-op chain grew with K, the harness would be measuring itself |
| **C0b** | K iterations of the cheapest real kernel (`z + 1`) | a small, **linear-in-K** cost | isolates the per-kernel launch floor from per-step compute |
| **C1** | `p50(K)/K` across K ∈ [1, 120] | ≈ constant | ⛔ **the async-timing trap.** A chain that looks sublinear in K has almost certainly not been synchronised |
| **C2** | K=60 run twice | agreement, with the spread **reported not hidden** | the dev box also drives a display; run-to-run noise must be visible, not averaged away |

## 4. ⛔ Scope limits, stated before the numbers

1. **This is a PROXY predictor, not ours.** Two transformer-block stacks (7.1 M and 37.8 M
   params) at batch 1. It measures the **structure** — linearity, per-step floor — so that the
   break-even arithmetic rests on a measured curve. ⛔ **No number here is a TanitAD model
   latency and none may be quoted as one.**
2. **This is an RTX 4060, not Thor.** Absolute values will move on the target. ⭐ **Linearity in
   K will not** — it follows from the data dependence, not from the device.
3. **Not CUDA-graphed.** The 2026-07-18 measurement (2.57× on one pass) would shift the curve
   down; it cannot change its slope.
4. ⚠️ **The box was not idle**: `nvidia-smi` read 2,255 MiB used / 26 % util before the run.
   Reported, not hidden — it is the likely source of C2's spread.
