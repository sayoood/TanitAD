
- ⛔ **TWO SEEDED GENERATORS WITH THE SAME FORMULA AND THE SAME SEED ARE ONE GENERATOR.** MEASURED 2026-10-28 (DAG-ORC r1, `PREREG_V8F_DAG_ORC`
  Amendment 2).
  * `DaggerSwap` and `EgoPerturb` both seeded `(seed × 1,000,003 + step) mod 2^62`. The registered argv set both seeds to s, and both read their
    selection uniform from row 0 of `torch.rand`, so the two rows were identical.
  * The composed perturbation therefore fired only where the swap did not: P(perturbed | not swapped) = 0.222 for a sample eligible for both,
    not 0.3. A s0 read 0.2492, and K-REACH refused it after 3,000 steps on box 2.
  * The composition tests passed, because their fixture never combined equal seeds with overlapping eligibility.
  * ⇒ **Two stochastic levers in one loop get domain-separated streams, and the test reproduces the registered argv's seeds together with
    overlapping eligibility.** A count check on the real run (K-REACH) is what caught it; keep one on every composed lever.
