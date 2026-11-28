# AMENDMENT_1 to PREREG_RECOVER_FT_FROZEN.md (sha256 57caba3b…, frozen 2026-11-02T19:40:09+0100): the rejoin target's continuation

**Written by the Master Mind, PRE-DATA:** no training step, no real-data table build and no rollout of this card has run; no number of
any arm exists. This amendment changes one operator definition before its first use and is frozen below (sha256 + wall clock in
`FROZEN_SHA.txt`).

**Defect found by the build agent.** The frozen `rejoin` target blends the first K = 10 knots from the STRAIGHT continuation of the
deviated heading `(L_i, 0)` onto the re-expressed logged future `g'_i`. At zero deviation `g' = gt`, but the target still differs from
`gt` on any curved path (it starts straight and bends onto the logged path), so it would teach O to under-turn at the start of every
curve in the perturbed share of the batch. That is a curvature / heading harm built into the target, unrelated to the lever.

**Amended definition (the only change).** The continuation is the logged manoeuvre attached to the deviated pose, i.e. the logged future
`gt_i` in the NOW frame read as if driven from the deviated car's own pose:

    target_i = (1 - w_i) * gt_i + w_i * g'_i,   w_i = smoothstep((i + 1) / K) for i < K,   target_i = g'_i for i >= K,
    g'_i = Rot(-dpsi) (gt_i - (0, dy))  (unchanged).

**Known answer (binding test):** zero deviation (dy = 0, dpsi = 0) gives `target == gt` EXACTLY (max abs 0.0), for every K and every
path, curved or straight. A speed-only deviation leaves the NOW pose unchanged, so its target is `gt` (the human's speed profile from
the first knot), as in ORC. `--recover-target logged` (pure frame change, `g'`) is unchanged.

**Unchanged:** arms, argv, deviation law, p, K, seeds, the read, every threshold, the frame-freeze rule, the predictions and the
committed next arms.
