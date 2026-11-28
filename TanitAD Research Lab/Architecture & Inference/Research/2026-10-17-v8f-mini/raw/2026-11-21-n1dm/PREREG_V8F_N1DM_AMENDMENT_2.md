# PREREG_V8F_N1DM, Amendment 2: the suite and the arms run beside RHK's seed-1 bank step (FROZEN by the MM at 2026-11-21T19:48:47Z, before any N1DM arm trains, any G-SUITE result or any number)

Card `PREREG_V8F_N1DM.md` (958d143d) and Amendment 1 (96847be8) stand. Code of record stays **573ad839**. This amendment changes WHEN and on which lock the chain's steps run. It moves no lever, argv, cell, bar, floor, estimator, tier or outcome.

## 1. The PI's go
PI, first-hand, 2026-11-21: "Dont let the pod idle, use it effectively". Then, asked "Run N1DM's suite and arms now, alongside RHK's bank step, on a private lock?": **"Yes, overlap"**.

## 2. Why
RHK's seed-1 bank step (`rh_k_chain.sh`, from 19:05:04Z) holds `/root/gpu_queue.lock` for about 1.9 h. MEASURED at 19:44Z, it ran 4 CPU-bound shards at 103 % CPU each, using 2,185 of 20,475 MiB of GPU at 0 to 4 % utilisation. The container had 22.3 GB rss against its 71.0 GB limit (the rest of the usage was reclaimable page cache) and a cfs quota of 15.3 cores.

## 3. What runs where
* **`/root/strat1/n1dm/n1dm_chain_early.sh`** (started 19:47:46Z; log `n1dm_chain_early.log`) is `n1dm_chain.sh` with four changes.
  * A private lock (`n1dm/early.lock`) replaces the shared one.
  * It starts only if RHK_s0 shows `ARM_DONE` and RHK's latest step line is a bank step.
  * Before each arm it checks that RHK is still in a bank step; otherwise it stops with `ZZ N1DM EARLY STOP ZZ` and leaves that arm to the main chain.
  * It has no dumps step.
  * Everything else is the same functions and done-markers: tree, inputs, G-SUITE (`n1dm_gsuite.sh`, unchanged: pinned cores, at most 3 shards, OMP/MKL 2), pairing, and arms s0 then s1 (`n1dm_arm.sh`, unchanged).
* **Two trainings never overlap.** `n1dm_arm.sh` and RHK's `rhk_arm.sh` both wait for the GPU to read below 3,500 MiB before they start, for up to 1 h each. The bank alone reads 2,185 MiB.
* **A watchdog** inside the early chain kills only THIS chain's trainer (processes whose command line holds `runs/N1DM_sK/train`) if the GPU passes 19,000 MiB, so the bank shards keep their memory. A killed arm fails the early chain, and the main chain re-runs that arm.
* **The dumps run only in the main chain, under the shared lock, after RHK ends.** No eval runs on a pod where an arm may train.
* **The main chain** `n1dm_chain.sh` is launched by `n1dm_after_rhk2.sh` (started 19:47:43Z; it took over from `n1dm_after_rhk.sh`) once RHK's log ends with `ZZ ROLLHIST-K ...` AND no early chain runs. It skips every step with a done-marker and runs the rest: any arm left, then the dumps. `/root/N1DW_READY` stays set until the main chain holds the shared lock.

## 4. What it can change, said plainly
* Wall-clock times, and so the cost reading: G-SUITE and the arms share the pod with the bank shards. Timing-sensitive tests may run slower. A G-SUITE regression caused by contention would fail the step, and it is re-run, not waived.
* The RHK bank's timing, through CPU contention. Its records do not change: the generator is deterministic in its inputs (K-B1 replay rows are compared bit for bit).
* Nothing in any N1DM arm's argv, data, init or seed.
