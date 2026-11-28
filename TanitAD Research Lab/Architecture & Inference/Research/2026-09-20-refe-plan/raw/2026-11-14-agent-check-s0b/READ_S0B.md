# S0-B read (agent check, CPU gate): S0B-READOUT. The footprint veto on TRUE future agents recovers 0.43 of the oracle; a single-frame raster used as a veto HURTS.

Master Mind, 2026-11-14.
- **Freeze:** `FREEZE_S0B.md`, landed `2314a462` before the run.
- **Run:** `run/` (`S0B_RESULT.md`, `s0b_result.json`, `s0b_flags.npz`, `cid/cid_result.json`, logs).
- **Evidence class:** MEASURED. Dev-box CPU, 1,510 s for the read, plus C-ID 5 min.
- **Tier:** a navtrain offline readout against NAVSIM's own sub-scores of P6's banked proposals. It is not a driving number, and nothing was keyed.
- **Estimator:** paired log-cluster bootstrap (10,000 draws, seed 20260927). It answers "would another draw of logs say this" only. **The held-out column has 13 logs**, so its intervals are wide.

## Gates before the read
| gate | result |
|---|---|
| pins | the 6 frozen sha256 values match |
| tests | `test_s0b.py` OK; `--mutations` ALL RED (7) |
| **C-ID** (64 covered tokens, dev box) | **PASS.** Cell agreement min **0.9989** (bar 0.99), mean IoU 0.977; one-cell tolerance 0.992 / 0.9998. **NAVSIM `score_plans` on the banked proposals reproduced the fit pack's nc and ttc on 64 of 64 tokens, exactly.** |
| C-REG-zero / C-REG-one | gain **exactly 0.0** in both columns |
| C-SHUF | gain −3.11 (held-out) / −2.53 (all): below 0, capture far below 0.10 |

## Result (deciding column: held-out covered sets, n = 1,343, 13 logs)

**Oracle O_NC+TTC: +1.811 PDMS** (NC-only +0.306, TTC-only +1.740).

| arm | gain (PDMS × 100) [95 % CI] | capture of O [95 % CI] |
|---|---|---|
| **R-true** (true agents at the 10 Hz future) | **+0.782** [+0.228, +1.679] | **0.432** [0.182, 0.629] |
| R-true-2Hz | +0.845 [+0.257, +1.767] | 0.467 [0.203, 0.704] |
| R-dyn (dynamic agents only) | +0.783 | 0.432 |
| **R-t0** (agents frozen at t0: the single-frame ceiling) | **−2.680** [−4.174, −1.271] | −1.480 |
| R-cv (t0 pose + true t0 velocity) | −0.867 [−1.697, +0.160] | −0.479 |

The all-covered column (3,958 sets, 47 logs; decides nothing) gives the same row: R-true capture 0.459 [0.278, 0.588], R-t0 −1.157, R-cv −0.520.

**Outcome (committed table): S0B-READOUT** in both columns. c(R-true) < 0.50. No control failed.

## What it says (MEASURED, held-out)
- **The readout itself flags well, but not where it matters.**
  - R-true's flag against truth over all proposals: precision **0.941**, recall **0.811**.
  - **On P6's top 5: precision 0.793, recall 0.655.** A third of the failing top-5 proposals are not flagged, because NAVSIM's at-fault and tracker logic counts things that a pose-overlap test does not.
- **A static or constant-velocity world is far worse than useless as a hard veto.**
  - R-t0 top-5 precision is **0.079** and R-cv's is 0.211.
  - Vetoing on them removes good proposals, so the pick falls to worse ones.
  - So a single-frame agent raster cannot be read through a hard veto. Any agent check needs either the future (motion) or a learned, soft combination with P6's own scores.
- The NC-only and TTC-only terms behave alike (R-true 0.37 / 0.45; R-t0 −8.2 / −1.5).

## Next (the card's S0B-READOUT row; Rule Zero, same turn)
1. **Learned fusion `q = f(flags, P6 logits)`**, fit out-of-fold on fit-extra and read on held-out. It is frozen in `FREEZE_S0C.md` before it runs.
   - It includes **F0, a fusion of P6's own six logits with no perception input.** F0 is the cheapest form of the card's "direct top-5 re-ranker".
2. **R-sim** (the agent LIST through NAVSIM's `score_plans`).
   - With TRUE lists it is an identity: C-ID has just shown `score_plans` reproduces the truth on 64 of 64 tokens, so c = 1.0 by construction.
   - Its information is in **R-sim-t0 / R-sim-cv**: single-frame and constant-velocity lists through the scorer's own at-fault logic.
   - It is built next, at ~0.5–0.7 CPU-s per set per arm (INHERITED from `FEASIBILITY_AGENTS.md` 2.3).
