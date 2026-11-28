# GAPSCALE: a yardstick for the stage-1 render gap (2026-11-07 04:48 dev-box clock; descriptive, decides nothing)

* **Statistic:** `t2a_v8f_encoder.cosine`, the flattened, uncentered cosine of the pool4x10 [40, 1024] tokens. It is the one stage 1 used for the render gap.
* **Data:** real frames of the eval-141 cache (`E:/Projects/TanitAD-artifacts/sfresh_inputs/cache/eval/clips`: 141 clips, 14,237 frames). Seeded with numpy default_rng 0, 2,000 pairs per row. Output: `GAPSCALE.json`, from `gapscale.py`.

| pair | median [p10, p90] |
|---|---|
| render at the logged pose vs its real frame (stage 1, a few ticks per scene) | 0.837 (scene 00040136), 0.740 (scene 7c72937c) |
| real frame vs the same clip 1 tick (0.1 s) later | 0.975 [0.949, 0.990] |
| ... 10 ticks (1 s) later | 0.900 [0.830, 0.963] |
| ... 30 ticks (3 s) later | 0.856 [0.767, 0.931] |
| ... 50 ticks (5 s) later | 0.834 [0.752, 0.919] |
| real frame vs a frame of a different clip | 0.640 [0.558, 0.721] |
| real frame vs the eval-141 mean (the BLIND tensor's source) | 0.806 [0.748, 0.848] |

* **Reading [E], beside:** a render sits about as far from its own real frame as a real frame does from the same clip 5 s later (scene 00040136), or further (7c72937c). The eval mean sits at about that distance from any real frame. So a render's tokens may carry little more about the real scene than the mean token does, which fits BLIND ≈ C0 at T2 (`D-V8F-T2-CRAWL-READ-1`).
* **Limits:** two scenes, a few ticks each, and render tokens were not cached, so the render-to-mean cosine is not measured. This decides nothing. The test is `theory/PREREG_T1_BLIND.md` (frozen `9619414a`).
