# RESULT_GAP48: the NuRec render gap on the read scenes (descriptive; decides nothing)

**Status: OK.** Scenes read 48 of 48, used 48. Condition `objects`, every 10th tick. Evidence class: MEASURED (ours): GAP48_RESULT.json and tokens/*.pt of the pull; recomputed here.

## 1. Render vs the real frame at the same tick (flattened uncentered cosine of the [40,1024] tokens)

| pooled over | n | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| all (scene, tick) pairs | 960 | 0.8401 | 0.7871 | 0.8859 | 0.6643 | 0.9410 |
| scene medians | 48 | 0.8428 | 0.8017 | 0.8818 | 0.7437 | 0.9294 |

Median of the pairs, 95 % scene-cluster bootstrap interval: [0.8307, 0.8503] (2000 draws, seed 0). Scenes with a median below 0.8: 4 of 48.
Relative L2 (pairs): median 0.5572, p10 0.4737, p90 0.6432.

## 2. Distance to the pinned eval-141 mean (paired per tick)

| quantity | n | median | p10 | p90 |
|---|---|---|---|---|
| cosine(render tokens, mean) | 960 | 0.7098 | 0.6780 | 0.7419 |
| cosine(real tokens, mean) | 960 | 0.7427 | 0.7009 | 0.7810 |
| paired difference, render minus real | 960 | -0.0331 | -0.0646 | -0.0077 |

Median paired difference, 95 % scene-cluster interval: [-0.0388, -0.0267]. The render is closer to the mean than its real frame in 4.4 % of pairs and in 0 of 48 scenes (scene median). Mean tensor sha256 `cd6a7ab5...a2fcb9` (asserted).

## 3. Yardstick, beside (gapscale/GAPSCALE.json: real frames of the eval-141 cache, same statistic)

| pair | median [p10, p90] |
|---|---|
| **GAP48: render vs its real frame** | **0.8401 [0.7871, 0.8859]** |
| stage 1 render vs real, two scenes (no actors attached) | 00040136 0.8366, 7c72937c 0.7404 |
| real vs the same clip, lag k1 | 0.9753 [0.9486, 0.9896] |
| real vs the same clip, lag k10 | 0.9000 [0.8297, 0.9634] |
| real vs the same clip, lag k30 | 0.8557 [0.7665, 0.9311] |
| real vs the same clip, lag k50 | 0.8338 [0.7519, 0.9191] |
| real vs a frame of a different clip | 0.6397 [0.5582, 0.7206] |
| real vs the eval-141 mean | 0.8057 [0.7477, 0.8477] |

The GAP48 median (0.8401) sits nearest the row "same clip, lag k50" (0.8338). Descriptive placement only; no threshold is attached to it.

## 4. Per scene

| scene | root | pairs | cosine median [p10, p90] | rel-L2 median | render-to-mean | real-to-mean | identity |
|---|---|---|---|---|---|---|---|
| e848c843 | scenes2 | 20 | 0.8378 [0.8170, 0.8587] | 0.5612 | 0.6964 | 0.7260 |  |
| 9e17ce05 | scenes2 | 20 | 0.8415 [0.8273, 0.8546] | 0.5555 | 0.6855 | 0.7013 |  |
| a2a4322c | scenes2 | 20 | 0.8394 [0.8058, 0.8549] | 0.5604 | 0.7131 | 0.7208 |  |
| e34fcbdc | scenes2 | 20 | 0.8305 [0.8049, 0.8496] | 0.5718 | 0.7131 | 0.7677 |  |
| 3e455540 | scenes2 | 20 | 0.8712 [0.8527, 0.8889] | 0.5058 | 0.7596 | 0.7783 |  |
| 1b024ad1 | scenes2 | 20 | 0.8870 [0.8487, 0.9086] | 0.4711 | 0.6875 | 0.7162 |  |
| fc41d9e9 | scenes2 | 20 | 0.8743 [0.8531, 0.8921] | 0.4964 | 0.7443 | 0.7726 |  |
| d3267951 | scenes2 | 20 | 0.8243 [0.8026, 0.8362] | 0.5859 | 0.7144 | 0.7629 |  |
| 5da7524e | scenes2 | 20 | 0.8812 [0.8392, 0.8920] | 0.4843 | 0.7150 | 0.7286 |  |
| 0caa5e02 | scenes2 | 20 | 0.8646 [0.8355, 0.8849] | 0.5114 | 0.6953 | 0.7209 |  |
| 5ab1d9cb | scenes2 | 20 | 0.8111 [0.7935, 0.8231] | 0.6159 | 0.6992 | 0.7200 |  |
| ec41061e | scenes2 | 20 | 0.8895 [0.8766, 0.9046] | 0.4658 | 0.6705 | 0.6895 |  |
| 6bffbe09 | scenes2 | 20 | 0.8050 [0.7880, 0.8536] | 0.6167 | 0.7223 | 0.7648 |  |
| 15eaddae | scenes2 | 20 | 0.8373 [0.7954, 0.8560] | 0.5590 | 0.7299 | 0.7909 |  |
| cda8a096 | scenes2 | 20 | 0.8590 [0.8308, 0.8823] | 0.5219 | 0.6923 | 0.7099 |  |
| c8c798be | scenes2 | 20 | 0.8499 [0.8023, 0.8675] | 0.5373 | 0.7271 | 0.7549 |  |
| fa035232 | scenes2 | 20 | 0.7784 [0.7536, 0.7970] | 0.6545 | 0.6959 | 0.7368 |  |
| 77d5887d | scenes2 | 20 | 0.8551 [0.8124, 0.8941] | 0.5331 | 0.7229 | 0.7605 |  |
| 7c72937c | kit | 20 | 0.8247 [0.7928, 0.8378] | 0.5812 | 0.7153 | 0.7456 |  |
| a686c7d0 | scenes2 | 20 | 0.8608 [0.8399, 0.8765] | 0.5226 | 0.7409 | 0.7655 |  |
| d078258b | scenes2 | 20 | 0.8848 [0.7889, 0.9046] | 0.4778 | 0.6976 | 0.7054 |  |
| 5894f7ff | scenes2 | 20 | 0.8087 [0.7707, 0.8224] | 0.6075 | 0.7053 | 0.7710 |  |
| 76f7e025 | scenes2 | 20 | 0.8468 [0.8274, 0.8549] | 0.5451 | 0.6992 | 0.7345 |  |
| b12fa656 | scenes2 | 20 | 0.8167 [0.7957, 0.8328] | 0.5962 | 0.7416 | 0.7801 |  |
| 9e792fc4 | scenes2 | 20 | 0.8559 [0.8345, 0.8897] | 0.5310 | 0.7195 | 0.7553 |  |
| 5b94e1f5 | scenes2 | 20 | 0.8442 [0.7841, 0.8633] | 0.5509 | 0.7465 | 0.7786 |  |
| 3d6a22f7 | scenes2 | 20 | 0.8247 [0.7755, 0.8459] | 0.5785 | 0.7197 | 0.7858 |  |
| 951ce3ff | scenes2 | 20 | 0.9294 [0.8837, 0.9392] | 0.3755 | 0.6877 | 0.6989 |  |
| 769e6724 | scenes2 | 20 | 0.8783 [0.8567, 0.8917] | 0.4924 | 0.7073 | 0.7265 |  |
| 9d73b364 | scenes2 | 20 | 0.8016 [0.7762, 0.8389] | 0.6150 | 0.7087 | 0.7216 |  |
| 7fd4d554 | scenes2 | 20 | 0.8494 [0.8356, 0.8767] | 0.5479 | 0.6932 | 0.7185 |  |
| 26b07382 | scenes2 | 20 | 0.8724 [0.8478, 0.8967] | 0.5040 | 0.7096 | 0.7369 |  |
| 10eef229 | scenes2 | 20 | 0.8630 [0.8073, 0.8828] | 0.5145 | 0.7157 | 0.7355 |  |
| 132ae6cb | scenes2 | 20 | 0.8495 [0.7937, 0.8718] | 0.5458 | 0.7213 | 0.7274 |  |
| 904af7cf | scenes2 | 20 | 0.8737 [0.8518, 0.8835] | 0.4964 | 0.7163 | 0.7554 |  |
| 5cdc4c2f | scenes3 | 20 | 0.7964 [0.7386, 0.8139] | 0.6292 | 0.6836 | 0.7321 |  |
| 4611c801 | scenes3 | 20 | 0.8216 [0.7890, 0.8626] | 0.5905 | 0.7214 | 0.7691 |  |
| a4353805 | scenes3 | 20 | 0.8654 [0.8393, 0.8840] | 0.5132 | 0.7213 | 0.7664 |  |
| 9e6753c2 | scenes3 | 20 | 0.8018 [0.7807, 0.8113] | 0.6140 | 0.7097 | 0.7735 |  |
| dd9fd9d1 | scenes3 | 20 | 0.7724 [0.7353, 0.8002] | 0.6553 | 0.6901 | 0.7698 |  |
| 764c8a4a | scenes3 | 20 | 0.7437 [0.7314, 0.7537] | 0.6971 | 0.6828 | 0.7160 |  |
| f58c6fa3 | scenes3 | 20 | 0.8287 [0.8019, 0.8477] | 0.5696 | 0.7208 | 0.7750 |  |
| 6140cba2 | scenes3 | 20 | 0.8145 [0.7781, 0.8364] | 0.5980 | 0.6976 | 0.7366 |  |
| 90ab8c24 | scenes3 | 20 | 0.8032 [0.7729, 0.8578] | 0.6122 | 0.6887 | 0.7449 |  |
| 41dadaf2 | scenes3 | 20 | 0.8350 [0.8159, 0.8576] | 0.5646 | 0.7184 | 0.7708 |  |
| 6eaeeb45 | scenes3 | 20 | 0.8556 [0.8393, 0.8720] | 0.5309 | 0.6943 | 0.7251 |  |
| bfb44da0 | scenes3 | 20 | 0.8339 [0.8188, 0.8508] | 0.5613 | 0.6874 | 0.7282 |  |
| b303ebe3 | scenes3 | 20 | 0.8831 [0.8649, 0.9077] | 0.4803 | 0.7403 | 0.7567 |  |

Box-2 scene wall time (sum over scenes): 19.6 min, per scene median 27.9 s.

## Estimator and limits

pooled quantiles are over every (scene, tick) pair; the interval is a scene-cluster bootstrap (would another draw of SCENES say this). No training or inference sampling in the loop.

One renderer build per scene (deterministic, stage-1 T5 bit-equal), the frozen in-loop DINOv3-L encoder, no policy in the loop: this measures the render, not an arm. It decides nothing by itself; the placement against the yardstick is a description.

