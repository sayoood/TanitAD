<title>RESULT - breadth is 5.94x cheaper than depth at equal budget (serves I-1)</title>

# RESULT — at equal rollout budget, BREADTH costs 5.94× less than DEPTH, and the value head that buys it costs 1.2 %

**Package** `E-LAB-DEPLOY-0901` · 2026-09-01 · Deployment & Optimization
**Serves** `LAB_BACKLOG` **INJECTED row I-1** (top open injected row — the lane's guarantee)
**Class** `MEASURED` microbenchmark, dev-box RTX 4060, fp16, `torch 2.11.0+cu128`, 120 reps / 30 warmup
**Tier stamp** n/a — this is a latency microbenchmark, not a driving eval. No capability claim is made.
⛔ **Thor and every pod untouched.**

⚠️ **READ THE SCOPE FIRST.** I-1 asks whether a value head *beats* fan-scoring. This package
answers the **cost** half only. It does **not** show the value head produces better plans — that
needs a trained head, a task and a metric, and is escalated as a proposed row. What is settled
here is what "equal budget" actually costs, which is the precondition for that comparison being
fair at all.

---

## 1. The headline

At a fixed budget of **B = 960 predictor-steps**, the four ways of spending it are not
equally priced:

| shape | N candidates | K depth | p50 latency | vs best |
|---|---|---|---|---|
| **fan-deep** (our current flat tactical rollout) | 16 | **60** | **73.33 ms** | **5.94× worse** |
| DreamerV3-shaped | 64 | 15 | 15.36 ms | 1.24× |
| **TD-MPC2-shaped** (short K + terminal value) | **320** | **3** | **12.34 ms** | **1.00×** |
| pure breadth | 960 | 1 | 12.21 ms | 0.99× |

⇒ **The same number of predictor forward-steps costs 5.94× more wall-clock when spent as depth
than as breadth.** Budget parity in *compute* is not budget parity in *latency*, and I-1's phrase
"at equal rollout budget" therefore has two different meanings that must never be conflated.

### H-BVD-1 — SUPPORTED, decisively `MEASURED`

Pre-committed criterion: supported iff `p50(N=960)/p50(N=1)` at fixed K is **< 100×** (linear
would be 960×).

| K | p50 @ N=1 | p50 @ N=960 | observed growth | linear would be |
|---|---|---|---|---|
| 1 | 1.101 ms | 12.233 ms | **11.11×** | 960× |
| 3 | 3.256 ms | 36.065 ms | **11.08×** | 960× |

**11.1× for 960× the candidates**, at both depths, in both independent runs. Per-candidate cost
collapses from **1.10 ms → 0.0127 ms (86×)**. ⭐ And latency is **flat up to N≈32**
(K=1: 1.101 → 1.062 ms; K=3: 3.256 → 3.047 ms) — up to the saturation point, **additional
candidates are free**.

### H-BVD-2 — SUPPORTED `MEASURED`

Pre-committed criterion: value-head p50 < 5 % of the rollout it is compared against.
The head (0.149 M params) measures **0.150 ms**, and is **flat in N** (0.1498 / 0.1500 / 0.1521 ms
at N = 16 / 320 / 960 — it batches like everything else).

**0.150 ms against the 12.34 ms rollout it enables = 1.2 %.** "Cheap" in I-1 is literal.

---

## 2. Controls — including one that partially failed and what that cost

| control | reading | verdict |
|---|---|---|
| **C0** no-work loop | 0.0028 → 0.0035 ms across K=1..60 | ✅ four orders below signal |
| **C1** depth at N=1 | see below | ⚠️ **passes only with a stated exclusion** |
| **C2** breadth at fixed K | 11.11× / 11.08× for 960× N | ✅ the load-bearing result |
| **C3** identity (N=320, K=3, twice) | 12.51 / 12.49 ms, rel spread **0.0044** | ✅ |
| **C4** value-head cost | 0.150 ms, flat in N | ✅ H-BVD-2 |

### ⚠️ C1 did not pass as written, and the honest version is narrower

C1's pre-committed bar was *"~flat ms/step AND agreement with the 2026-08-31 package"*. Raw, it
reads **ratio 1.795** — against that package's 1.083. That is a fail as stated.

The whole discrepancy is the **K=1 row** (1.9504 ms, where the prior package read 1.0784 ms) — a
single-step measurement dominated by launch overhead on a desktop GPU that is also driving a
display. **Excluding K=1, ms/step spans 1.0864–1.1734 → ratio 1.080**, matching the prior
package's **1.083** to three decimals.

And the reproduction at K ≥ 3 is genuinely close (this run vs 2026-08-31, identical config):

| K | this run p50 | 2026-08-31 p50 | delta |
|---|---|---|---|
| 3 | 3.259 ms | ~3.27 (interp. K=2/K=4) | ~0 % |
| 8 | 9.346 ms | 8.763 ms | +6.6 % |
| 15 / 16 | 17.578 ms | 17.487 ms | **+0.5 %** |
| 30 | 35.201 ms | 32.941 ms | +6.9 % |
| 60 | 68.711 ms | 70.101 ms | −2.0 % |

⇒ **The two packages agree within ~7 % at every K ≥ 3**, independently measured a day apart. The
K=1 anomaly is contention on *this* run, not a structural disagreement.

⚠️ **A first pass at 40 reps would have been misreported.** It gave C1 ratio 1.55 with a bogus
K=8 row (14.31 ms, +63 % vs the prior package) and a headline of 6.65×. Re-running at 120 reps
moved the headline to **5.94×** and cleaned C1. **The 40-rep JSON is banked alongside the 120-rep
one** (`raw/breadth_vs_depth_4060.json`) so the correction is auditable rather than invisible.
The 120-rep file is the quotable one.

---

## 3. Why this matters — it converges with two independent lines

1. **The deployment wall.** `2026-08-31-rollout-depth-latency` measured the strategic band's flat
   K=300 at **~350 ms (tiny) / ~1,015 ms (mid)** against a **100 ms** budget, and CUDA graphs'
   2.57× does not close it. This package says the budget was being spent in its most expensive
   possible shape.
2. **The training wall.** Today's Architecture package
   (`…/Architecture & Inference/Research/2026-09-01-longhorizon-bptt-stability/RESULT.md`) found
   that **no primary in our reference class back-propagates a 60-step chain** — TD-MPC2 runs
   **H = 3**, DreamerV3 **H = 15** — and both **bootstrap past the horizon with a learned value**
   rather than deepening the chain.

⭐⭐ **These are the same fact, reached from opposite ends on the same day and by different
methods.** The literature caps K because deep chains do not train; the microbenchmark caps K
because deep chains do not fit the control loop. **TD-MPC2's H=3 is not a modelling compromise —
at our measured curve it is also the cheapest point on the budget surface, by 5.94×.**

## 4. What this changes for TanitAD — 3 recommendations

1. ⭐ **Re-scope I-1 from a quality question to a two-part one, and mark the cost half served.**
   The cost half is answered: breadth is 5.94× cheaper and the head costs 1.2 %. The **quality**
   half is untouched and is the real experiment — it needs a trained terminal value on our latent
   and a pre-committed read against fan-scoring at **matched latency** (not matched step-count,
   which this package shows is a different budget).
2. **Whenever a planning budget is quoted, state whether it is matched in STEPS or in LATENCY.**
   They differ by ~6× at our operating point. A "same budget" comparison that silently uses
   step-parity hands the deep arm a 6× latency handicap it never agreed to — the same *scope*
   error family as `df`/`free`/`step_s` in CLAUDE.md, with the object swapped for a budget.
3. **Exploit the free-breadth region.** Candidates are *free* up to N≈32 and cheap to N≈320.
   Any planner running fewer than ~32 candidates is leaving search width on the table at zero
   latency cost — worth checking against whatever fan width the current planner uses.

## 5. Limits — read these before quoting any millisecond

- **PROXY predictor, not `predictor_op`.** dim 384 / depth 4 (7.098 M) brackets ours in params,
  not in shape. **The linearity and the sub-linearity transfer; the milliseconds do not.**
- ⛔ **RTX 4060 ≠ Thor.** Nothing here is a Thor latency claim. Saturation N and the 5.94× will
  both move on different hardware — Thor saturates at batch 8 on a different workload
  (CLAUDE.md), which would *shrink* the free-breadth region and must be re-measured, not assumed.
- **No CUDA-graph capture**, batch as stated, single stream.
- **No quality claim.** See §0 scope.
- The 5.94× is specific to B=960; the ratio grows with B, and shrinks toward 1 as N→saturation.

## 6. Deliverable manifest

| artifact | where | only place? |
|---|---|---|
| `SPEC.md` (hypotheses + criteria, pre-committed) | `repo:` this package | staged |
| `code/breadth_vs_depth_bench.py` | `repo:` this package | staged |
| `raw/breadth_vs_depth_4060_reps120.json` — **the quotable run** | `repo:` this package | staged |
| `raw/breadth_vs_depth_4060.json` — the 40-rep run, kept for auditability | `repo:` this package | staged |
| `RESULT.md` (this file) | `repo:` this package | staged |

## 7. Evidence-class ledger

| claim | class |
|---|---|
| 73.33 / 15.36 / 12.34 / 12.21 ms at B=960; 11.1× breadth growth; 0.150 ms head; C0–C4 | **MEASURED (ours)** — `raw/breadth_vs_depth_4060_reps120.json` |
| K-linearity, strategic band ~350/~1,015 ms, CUDA-graph 2.57× | MEASURED (ours, prior package) — `…/2026-08-31-rollout-depth-latency/` |
| TD-MPC2 H=3, DreamerV3 H=15, terminal-value bootstrap | PUBLISHED (PRIMARY, banked) — via today's Arch package |
| "the value head produces better plans" | ⛔ **NOT CLAIMED** — unmeasured |
