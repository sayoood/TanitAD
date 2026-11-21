# MM rulings on PREREG_V8F_WEATHERHEAD (2026-11-16 ~22:45Z, at the freeze)

The card was drafted by an Opus architecture agent before any build, fit or score. These rulings answer its section 0 and add three items found in review. They bind the builder and the read exactly as the card's own text does.

## Answers to section 0

1. **Form (a) is covered; Q4 does not go to the PI as a blocking question.**
   * No driving parameter moves, no pod or box-2 time is used, and the planner outputs are bit-identical by test (T-ISO a-d).
   * The frozen STRAT1 card prescribes this arm ("If the probe detects, the next arm is that head"), and the PI said "Build the arm" for STRAT1.
   * The PI is told in the same report that the fit runs on the dev-box GPU, about 3-6 h ESTIMATED, and can stop it.
   * Shipping `ctx_pool_head` as a model output in a release is a separate step, reported with the read.
2. **The gate token is required.** `wh_check.py` mints `wh-gate-token-v1` (section 4.6). Every G-check has its deliberate-regression arm that must read FAIL. No fit starts without the token.
3. **Fit cache: the dev-box train cache** `F:/TanitAD-scratch/speedread/cache/train`.
   * The head is fitted and read on caches of one encode family: this train cache and the eval cache of record. S1's adapter was trained on the pod's encode, and that shift is common to every arm of the card (register row `D-EVALCACHE-POD-DEVBOX`, cause MEASURED 2026-11-16).
   * P-CACHE also prints the train cache's encoder provenance (`encoder_module_sha256` from the cache's own record where one exists, else "not recorded") beside the content_md5 count. No bar depends on either.
4. **All five variables, one fit, output map by the section 7 rules: accepted.**
5. **Base commit `a4357e79`: accepted.** The flag defaults off, and off equals `a4357e79` bit for bit (snapshot test).

## Added in review (binding)

6. **Memory: the fit may not hold 5.9 GB of features in one process unless the commit headroom allows it.**
   * The dev box is commit-limited: memory rule `devbox-commit-limit-oom`, gate new jobs at 5.5 GB or more of commit free.
   * The first STRAT1 read lost three dumps to `ArrayMemoryError` (2026-11-16 18:02Z).
   * Precondition **P-COMMIT**: before loading features, the fit checks free commit against the feature size plus 5.5 GB. When it is short, the fit streams batches from the clip files (memory-mapped) instead of holding them.
   * A test pins the two paths to the same bytes: on two batches, streamed and in-RAM features are byte-equal. Its regression arm must read RED: an offset of one window.
7. **Order on the dev box.** The CPU parts (T0 and T1 dumps, reads) start after the STRAT1 seeds read and the LON-PLAN readout have ended, or under the same commit gate. The GPU fits take the shared lock `C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock` (below 3,500 MiB used) and never force it.
8. **What the read is compared with.** `H - probe` (weather) is read only against the probe's dev-box-cache rows (`wprobe_out.npz` on the eval cache of record), never against its pod-cache rows.

## Priority for the builder (a stopped chain still yields value)

* P1-P2 (H_s0, H_s1) with K0, K2-K6, K-ID, T-ISO, T-VIS and reachability.
* Then P3-P4 (the random adapter, which fixes the wording), then P5-P6 (R1, the fit replicate), then P7-P9.
