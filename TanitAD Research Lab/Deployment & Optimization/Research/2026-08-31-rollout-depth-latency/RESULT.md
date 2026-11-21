<title>RESULT — rollout latency is LINEAR in depth, and that makes a flat strategic rollout undeployable</title>

# RESULT — E-LAB-DEPLOY-0831

`TanitAD Research Lab · Deployment & Optimization · 2026-08-31`
`Class: MEASURED microbenchmark (dev-box RTX 4060, torch 2.11.0+cu128, fp16, batch 1, 60 reps / 15 warmup) + arithmetic.`
`⛔ Thor NOT touched — k60p30k was mid-run throughout.`

⚠️ **AUTHORSHIP, stated because it affects how this is read.** The SPEC, the benchmark
code and the measurement are the Lab agent's. It died on an API connection error at
*"let me bank the artifacts"*, twice. **This write-up was completed by the Master Mind
from the agent's own banked `raw/rollout_depth_4060.json`** — no number here was
re-measured or re-derived, and the pre-committed criterion in SPEC §2 is used exactly as
it was written **before** the data existed.

---

## 1. ⭐ H-LAB-LAT-1 — SUPPORTED. Latency is linear in sequential depth.

SPEC §2 committed: *supported if `p50(K)/K` is approximately constant across K ∈ [1,120].*

| K | tiny (7.1 M) p50 | ms/step | mid (37.8 M) p50 | ms/step |
|---|---|---|---|---|
| 1 | 1.078 ms | 1.0784 | 3.334 ms | 3.3344 |
| 8 | 8.763 | 1.0954 | 26.401 | 3.3001 |
| 30 | 32.941 | 1.0980 | 89.555 | 2.9852 |
| **60** | **70.101** | **1.1684** | **203.074** | **3.3846** |
| 120 | 136.530 | 1.1378 | 362.267 | 3.0189 |

**ms/step across K=1→120: 1.0784 → 1.1378 (1.06×) and 3.3344 → 3.0189 (0.91×).** Flat.
⇒ **a K-step rollout costs K × one pass, with no per-step penalty and no amortisation.**
Nothing parallelises the chain: batching helps across *candidates*, never along it.

**Controls read their known values:** `C0_nowork` is flat in K (0.0029 → 0.0033 ms — an
empty loop must not scale, and does not, so the harness adds nothing); `C0b_add1` does
scale (0.0227 → 0.5157 ms), so the loop *is* measurable and is ~0.5 ms at K=60 — three
orders below the signal.

## 2. ⛔⛔ THE CONSEQUENCE: A FLAT STRATEGIC ROLLOUT DOES NOT FIT THE CONTROL LOOP

At `dt = 0.1` the bands are flat-K 20 / 60 / 300. Against a **10 Hz budget (100 ms)**:

| band | K | tiny 7.1 M | mid 37.8 M |
|---|---|---|---|
| operative 0–2 s | 20 | ~21.6 ms ✅ | ~66.7 ms ✅ |
| **tactical 2–6 s** | **60** | **70.1 ms ✅ (tight)** | **203 ms ⛔ 2× over** |
| **strategic 8–30 s** | **300** | **~350 ms ⛔** | **~1,015 ms ⛔** |

⭐ **The tactical 6 s horizon — the one `k60p30k` is training right now — IS deployable, but
only at tiny scale**, and it consumes 70 % of the budget before the planner, the encoder or
the readout have run.

⛔ **The strategic band is unreachable by a flat rollout at every scale measured.** And the
obvious rescue does not close it: this same 4060 measured **6.08 → 2.36 ms (2.57×)** for one
predictor pass under manual CUDA-graph capture (`2026-07-18-predictor-cudagraph-and-numerics-sweep`).
Applying that best case to tiny/K=300 gives **~136 ms — still over budget**, before anything
else in the loop.

## 3. ⭐⭐⭐ WHY THIS MATTERS BEYOND DEPLOYMENT

`V7_LAUNCH_GATE.md` P4 argues for **temporal abstraction** — each level rolling at its own
`dt`, ratios **1 : ~3 : ~15** read off the label bands — rather than one long flat rollout.
That argument was made from **training** economics (MM-E15: `o5_k 80` is 10× rollout compute
on a predictor that collapses past one tick).

⇒ **This package reaches the same conclusion from the opposite end, and independently: a flat
strategic rollout is not merely expensive to train, it is UNDEPLOYABLE.** A strategic level
stepping at ~1 s reaches 30 s in ~30 steps — **~35 ms at tiny scale, comfortably inside the
budget.** Temporal abstraction stops being an efficiency preference and becomes a
**deployment requirement**.

## 4. ⚠️ Limits — this is structure, not our number

* **PROXY predictor, not TanitAD's.** The JSON says so in its own `what_this_is` field.
  `tiny_d4_dim384` (7.1 M, dim 384 / depth 4) brackets our `predictor_op` (6.5 M) in
  parameters but not in shape (ours is dim 256 / depth 3). ⇒ the **linearity** transfers; the
  **milliseconds** do not.
* ⛔ **RTX 4060 ≠ Thor.** Nothing here is a Thor latency claim. The deployment target must be
  re-measured on Thor before any number is quoted as ours.
* **batch 1, fp16, no CUDA-graph capture** in this benchmark (the 2.57× is cited from a prior
  package, not re-measured here).
* The 10 Hz budget is the **whole** loop; these figures are the rollout **alone**.

## 5. Deliverable manifest

| artifact | where |
|---|---|
| `SPEC.md` (hypothesis + criterion, pre-committed) | this package — Lab agent |
| `code/rollout_depth_bench.py` | this package — Lab agent |
| `raw/rollout_depth_4060.json` | this package — Lab agent, with `meta` + `controls` |
| `RESULT.md` (this file) | Master Mind, from the banked JSON after the agent's second API failure |

⇒ **Recommended next, and it is cheap:** re-run `code/rollout_depth_bench.py` **on Thor** once
`k60p30k` releases the GPU, and re-run it with CUDA-graph capture enabled. Both are the same
script with different hosts/flags, and together they convert this from a structural result
into a deployment number.
