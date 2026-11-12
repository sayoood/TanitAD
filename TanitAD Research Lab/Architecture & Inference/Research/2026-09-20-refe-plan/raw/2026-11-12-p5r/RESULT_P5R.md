# P5-R result: PASS on both ranks. The regularised inversion is smoother than Tc20s and tracks better

2026-11-12, Master Mind, from `raw/verdict.json`, `raw/lambda_star.json` and `raw/controls.json` (chain end `ZZP5R_CHAIN_DONE`, 08:33Z).

- **Pre-registration:** `PREREG_P5R.md`, sha256 `cfd23e0651d66e497238a0708c717ba00acbcb2189295efcfb15dad657922a42`, frozen 2026-11-12T08:03:43Z, before any code. No addendum.
- **MM rulings:** `MM_RULINGS_P5R.md` (sha256 `c183b215...`), written 08:13Z, before this verdict existed.
- **Evidence class:** MEASURED, pre-registered, read on the EVAL split, log-disjoint from TUNING.
- **Tier:** a label-construction diagnostic with the tracker in the loop (NAVSIM v1.1 PDM components, vendored), on navtrain P7-bank rows whose log has a local nuPlan DB, CPU. **Not navtest and not PDMS.**
- **Estimator:** paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, per rank over that rank's EVAL logs. It answers "another draw of logs". The labels, tracker and scorer are deterministic.

## Controls (card §8): all 13 gates PASS
- λ = 1e12 gives a result within 1e-6 of `traj`, and the λ = ∞ path is bit-exact.
- Descent holds on all rows, with the median strictly lower.
- The identity known-value check passes; the C-R3 and C-R4 deliberate regressions make B4 FAIL.
- The λ = 0 zero control gives J_reg exactly 0.
- S1–S5 hold (finite, yaw branch, re-simulation max deviation **0.0**, fallbacks ≤ 1 %, λ reaches the objective).

## λ* (card §4, TUNING split only)
**λ\* = 3.0**, status OK. It is the smallest λ in the grid whose four TB-6 ratios (measured and projected, acc and heading rate) are all ≤ 1.80 on both ranks.

## Verdict on EVAL, per rank (rank 0 decides)

| | rank 0 (1,200 rows, 294 logs) | rank 1 (1,200 rows, 275 logs) |
|---|---|---|
| **TB-6 heading-rate ratio, P5-R** | **1.564** [1.409, 1.660]; projected 1.628 | **1.344** [1.276, 1.388]; projected 1.340 |
| TB-6 heading-rate ratio, Tc20s, same rows | 1.950; projected **2.080** (fails the 2× bar) | 1.664; projected 1.662 |
| **TB-6 acc ratio, P5-R** | **1.158**; projected 1.153 | **1.148**; projected 1.146 |
| TB-6 acc ratio, Tc20s, same rows | 1.736; projected 1.712 | 1.654; projected 1.651 |
| B2: sim8 failure, P5-R − Tc20s (bar: upper CI ≤ +1.0 pp) | **−0.08 pp [−0.27, +0.00]** | **−0.08 pp [−0.27, +0.00]** |
| B3: sim8 failure, P5-R − raw (bar: upper CI < 0) | **−4.92 pp [−6.50, −3.48]** | **−4.50 pp [−5.98, −3.19]** |
| B4: 8-knot max-xy error, P5-R − Tc20s (bar: upper CI ≤ +0.10 m) | **−0.0133 m [−0.0150, −0.0117]** | **−0.0211 m [−0.0231, −0.0190]** |
| errors | 0 | 0 |
| **verdict** | **PASS** (B1, B1p, B2, B3, B4; S1–S5) | **PASS** |

## What it means
- **P5-R dominates Tc20s on the same rows.**
  - Its labels are far smoother. Rank-0 heading-rate ratio is 1.56 against 1.95, and acceleration 1.16 against 1.74.
  - It tracks at least as well: sim8 failures are no worse (−0.08 pp [−0.27, 0.00]).
  - Its 8-knot error is smaller: −1.3 cm at rank 0, −2.1 cm at rank 1, both CIs excluding 0.
- The raw-trajectory repair is kept: −4.9 pp sim8 failures at rank 0.
- **This independently confirms the TB-6 projection.** On these EVAL rows, Tc20s's projected full-bank rank-0 heading-rate ratio is 2.080, against the MM's earlier reweighting estimate of 2.091 (`p5c/gates/preview_63k`). Tc20s is expected to fail TB-6 at rank 0 on the pod's full table.
- **Which label P5-C uses** follows MM ruling R4, committed before this verdict:
  - if the pod's full-table TB-6 on Tc20s fails at rank 0 (projected), P5-R is the label, after R2 (the PI is told, under the 11-11 tracker-aware ruling) and R3 (a PREREG_P5C amendment for P5-R's TB-0: ≤ 1e-6 plus identical sim8 verdicts);
  - if Tc20s passes after all, the R4 order keeps Tc20s.
  - Given the dominance shown here, the MM will **recommend to the PI to switch to P5-R regardless**. That is a PI decision, not a post-hoc change of R4.
- **Cost of labelling the full bank** (ESTIMATED from this run): rank-0 EVAL took 629 s for 1,200 rows on one process, about 0.52 s/row. 150,603 rows ≈ 22 CPU-h, about 20 min on 64 of the pod's 128 cores, after P7.

## Not shown here
- Anything about driving. This is about the labels only.
- Whether a student trained on P5-R labels beats one trained on raw `traj`: that is P5-C's read (C1 vs C0), with its own bar.
- STRATEGIC family: PAUSED (PI 2026-09-19). The four-family reads belong to P5-C.

## Artifacts
- `raw/`: `verdict.json`, `lambda_star.json`, `controls.json`, `prep.json`, `budget.json`, `tune_r{0,1}.jsonl`, `eval_r{0,1}.jsonl`.
- `PREREG_P5R.md` + `.sha256`, `p5r_labels.py`, `p5r_run.py`, `test_p5r_labels.py`, `HANDOFF.md`, `MM_RULINGS_P5R.md`.
- Run scratch: `F:/tanitad_scratch/p5r_agent/run/`. It is deleted after the `raw/` copies are verified against it.
