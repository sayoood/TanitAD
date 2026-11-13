# SLOWWM read, step 4: IA-0 for all seven arms (BESIDE, no bar)

**Tier T0.** IA-0 is the world model's own multi-step prediction skill in z1, scored against that arm's own Copy-Last. It is **BESIDE** the pre-registered planner read and carries **no bar**. **MEASURED** from `results/ia0_<arm>.json` and `IA0_SLOWWM.json`, all written by `ia0_slowwm.py` (blob on the tip). The runs took place 2026-11-13: SLOW_s0 at 00:24, the other six at 01:40-02:07. The log is `ia0_slowwm.log`.

The interval is the episode-cluster bootstrap over **141 clips**. It answers "would another draw of episodes say this?". It does not answer "would another training run say this?"; the seed pairs below speak to that question.

## Skill vs own Copy-Last at the band end (point [lo, hi])

| arm | basis | L1 at 2 s | L2 at 6 s | L3 at 12 s |
|---|---|---|---|---|
| SLOW_s0 | slow | 0.134 [0.103, 0.164] | 0.431 [0.371, 0.480] | 0.537 [0.456, 0.601] |
| SLOW_s1 | slow | 0.144 [0.112, 0.175] | 0.432 [0.374, 0.479] | 0.549 [0.476, 0.612] |
| SLOW_rep_s2 | slow | 0.129 [0.092, 0.163] | 0.450 [0.389, 0.504] | 0.581 [0.501, 0.643] |
| PCA-B_s0 | pca | 0.184 [0.149, 0.217] | 0.466 [0.417, 0.506] | 0.595 [0.526, 0.651] |
| PCA-B_s1 | pca | 0.185 [0.148, 0.221] | 0.450 [0.398, 0.493] | 0.540 [0.460, 0.608] |
| R35H_s0 | adapter | 0.259 [0.198, 0.316] | 0.546 [0.493, 0.588] | 0.672 [0.618, 0.713] |
| R35H_s1 | adapter | 0.267 [0.207, 0.321] | 0.555 [0.501, 0.599] | 0.687 [0.639, 0.728] |

Every cell is readable, healthy and above its own Copy-Last.

## With a one-frame context (the planner's c1 form), L1 at 2 s

| arm | skill [lo, hi] | vs own Copy-Last |
|---|---|---|
| SLOW_s0 | -0.198 [-0.260, -0.141] | below |
| SLOW_s1 | -0.116 [-0.170, -0.067] | below |
| SLOW_rep_s2 | -0.122 [-0.181, -0.068] | below |
| PCA-B_s0 | -0.072 [-0.132, -0.017] | below |
| PCA-B_s1 | -0.090 [-0.154, -0.032] | below |
| R35H_s0 | -0.016 [-0.112, +0.066] | not separated |
| R35H_s1 | -0.020 [-0.112, +0.060] | not separated |

At L2 and L3 every arm is above its own Copy-Last with one frame of context as well (`IA0_SLOWWM.json`, `band_end_context1`).

## How to read it

* **The order holds across seeds:** at L1, SLOW (3 seeds, 0.129-0.144) < PCA-B (2 seeds, 0.184-0.185) < R35H (2 seeds, 0.259-0.267), and the seed pairs sit inside each other's intervals. Each arm is still scored in its **own** z1 space against its **own** Copy-Last, so this is a descriptive pattern, not a ranking of world models.
* **HYPOTHESIS:** a slow basis makes Copy-Last stronger by construction, because slow features change little over 2 s. A lower skill-over-Copy-Last for SLOW is then expected and is not by itself evidence of a worse model. The pre-registered test of whether the slow basis helps **planning** is the planner read (README step 3). That read is running.
* `results/FAILED_ia0_*.txt` in the working folder are the refusal records of the 00:24 attempt (free RAM 5.51 GB under the 6.0 GB guard). The 01:40 run then wrote a result for each of those arms. They are not banked here.
