<title>SPEC - terminal value vs fan-scoring at equal rollout budget (I-1)</title>

# SPEC - E-LAB-DEPLOY-0901: at equal rollout budget, is BREADTH cheaper than DEPTH?

**Trigger.** `LAB_BACKLOG` **INJECTED lane, row I-1** (MM, 2026-08-29, state `open`) — the top
open injected row. The injection lane's standing guarantee: whenever it is non-empty, the daily
Lab run dedicates **at least one package** to the top injected row. This is that package.

**I-1 verbatim.** *"Latent model-predictive planning with a LEARNED terminal value over composed
h=1 rollouts — does a cheap value head beat fan-scoring at equal rollout budget?"*

## ⚠️ Scope, stated before anything else

I-1 contains two questions and this package answers **only the first**:

| half | question | answered here? |
|---|---|---|
| **COST** | at equal budget `B = N x K`, what does each way of spending B cost in wall-clock? | ✅ **MEASURED** |
| **QUALITY** | does the value head produce *better plans* than fan-scoring? | ⛔ **NO** — needs a trained value head, a task and a metric |

⛔ **Nothing in `RESULT.md` may be read as "the value head beats fan-scoring".** What is measured
is what "equal budget" *costs*, which is the precondition for the quality comparison being fair.

## Hypotheses, both outcomes committed IN ADVANCE

⭐ The pre-commitment is verifiable, not asserted: it was written into the docstring of
`code/breadth_vs_depth_bench.py` and saved to disk **before the script was executed**
(see its "THE CLAIM UNDER TEST" paragraph, retained verbatim in the shipped file).

| id | hypothesis | outcome A (supported) | outcome B (refuted) |
|---|---|---|---|
| H-BVD-1 | **Breadth is sub-linear** in N while depth is linear in K | at fixed B the (large N, small K) corner is strictly cheaper; a terminal value becomes a **latency** lever | breadth grows ~linearly (~960x from N=1 to N=960) ⇒ **I-1's premise is wrong** and the row should be re-scoped or closed |
| H-BVD-2 | The value head's own cost is negligible against the rollout it removes | "cheap" in I-1 is literal | the head is not cheap and I-1 must price it |

**Pre-committed criteria.**
- H-BVD-1 supported iff `p50(N=960)/p50(N=1)` at fixed K is **< 100x** (vs 960x for linear).
- H-BVD-2 supported iff value-head p50 is **< 5 %** of the rollout p50 it is compared against.

## Controls (a probe without controls that read known values manufactures results — CLAUDE.md)

| id | control | must read |
|---|---|---|
| C0 | no-work loop | orders below the real block |
| C1 | depth sweep at N=1 | ~flat ms/step **AND agreement with the 2026-08-31 package on the identical config** — disagreement invalidates both |
| C2 | breadth sweep at fixed K | the load-bearing measurement |
| C3 | identity (same arm twice) | small run-to-run spread |
| C4 | value-head cost | H-BVD-2 |

**Method.** dev-box RTX 4060, fp16, `torch 2.11.0+cu128`, PROXY predictor
(dim 384 / depth 4 / heads 6 / tokens 16 — **identical to the 2026-08-31 package** so C1 is a
true reproduction). ⛔ Thor not touched; no pod touched. `nvidia-smi` showed no python compute
before launch.

**Known limit, in advance.** This is a proxy, not `predictor_op`; the *structure* transfers, the
*milliseconds* do not, and none of it is a Thor number.
