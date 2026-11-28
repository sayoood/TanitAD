# Pre-registered check: is the step-5,000 G0 seed spread a fluke, or seed-structured?

**Master Mind, 2026-10-04 02:43 Berlin (00:43Z). Written BEFORE any step-50,400 or step-30,000 G0 number exists** (the step-50,400
chain is waiting on the GPU lock; its sha256 is recorded in `PREREG_SEED_GROUP_CHECK_50400.sha256`).

## Why

A5 attributes the step-5,000 G0 failure to an 8-seed under-estimate of the seed spread. The data behind A5 do not
fully support the "sampling accident" reading (MEASURED, `raw/step5000/g0.json`, `raw/g0diag_step5000/diag.json`):

| seeds | n | row-mean sd of `eval_traj` | same code path? |
|---|---|---|---|
| 0–7 (G0, one process) | 8 | 0.00212 | `g0_refcv7.run_eval`, micro 2,2,3,3,3,3 |
| 8–23 (diag, one process) | 16 | 0.01308 | the same function, the same micro-batching |

- F test of var(8–23) / var(0–7): F = 38.1, one-sided p = 3.1e-5 under normality. Levene (robust): p = 0.018.
- Per-batch spread is similar in both groups (0.017–0.033 for seeds 0–7, 0.019–0.041 for seeds 8–23). The difference
  sits entirely in the cross-batch covariance: off-diagonal / trace = −0.94 for seeds 0–7. With n = 8 and p = 8 this
  ratio restates the small row-sum variance; it is not independent evidence.
- Seed 0 replays bit-for-bit in the fresh diag process (1,097 of 1,098 keys).

Two readings remain: (A) a genuine tail event in seeds 0–7, or (B) something tied to the seed values or to running
order in one process, which would make an 8-seed G0 a structurally miscalibrated instrument rather than an unlucky
one. A5 (24 seeds) is the right estimator under both readings. The ROOT-CAUSE CLASS differs, and it decides
whether the battery's other STOCHASTIC gates (and the refcv6 battery's) need re-reading.

## The check (zero extra GPU: the step-50,400 G0 runs seeds 0–23 anyway)

On the step-50,400 G0 (`raw/step50400/g0.json`), for `eval_traj`:
- compute the row-mean sd of seeds 0–7 and of seeds 8–23, and the one-sided F test var(8–23)/var(0–7);
- **(A) supported** if p ≥ 0.05: the step-5,000 pattern does not recur, and "an 8-sample sd taken as stable" is the class;
- **(B) supported** if p < 0.01 in the same direction: the pattern recurs at a different checkpoint, so it is
  seed- or harness-structured. Then the next lever is a fresh-process replay of seeds 1–7 (bit-exact against the G0
  values), and every 8-seed STOCHASTIC verdict in the refcv6 and refcv7 batteries gets re-read;
- 0.01 ≤ p < 0.05: INCONCLUSIVE, reported as such; run the fresh-process replay.

This read is a diagnostic of the instrument. It never changes the G0-A5 verdict, which is computed as A5 registers it.
