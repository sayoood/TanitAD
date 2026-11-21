"""What can the episode-cluster bootstrap say about the M1 crawl at the n the two on-hand NuRec scenes give?  CPU, arithmetic only.

PART 1 (MEASURED inputs: m1_ticks_in_t2a.json, the banked T2-A box-2 rollouts of 2026-10-17; refcv7 and P10 are NOT v8F -- they stand in
         for 'a model that crawls' and 'a floor' only to size the interval).
    Paired cluster bootstrap over the 9 rollout starts of scene 7c72937c (B=2000, seed 0) on the per-start M1-tick share (M1 ticks / 50).
PART 2 (ESTIMATED): scenes needed for a T1-kin-sized effect, from the T1-kin interval of the composite-vs-P10 M1 row
    (+0.0549 [+0.0356, +0.0775], 141 episodes, LF read RESULTS.md line 81, s0): SE = (hi-lo)/(2*1.96), per-episode SD = SE*sqrt(141).
Run: python -I m1_power.py > m1_power.json
"""
import json
import math
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
d = json.load(open("m1_ticks_in_t2a.json", encoding="utf-8"))
sc = d["scenes"]["7c72937c"]["arms"]
starts = [str(s) for s in (0, 17, 34, 51, 68, 85, 102, 119, 136)]


def per_start_share(arm):
    return np.array([sc[arm]["m1_per_start"][s] / 50.0 for s in starts])


rng = np.random.default_rng(0)


def paired_boot(a, b, B=2000):
    diff = a - b
    n = len(diff)
    idx = rng.integers(0, n, size=(B, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "n_clusters": n,
            "n_clusters_nonzero_diff": int((diff != 0).sum()), "frac_resamples_exactly_zero": float((means == 0).mean())}


res = {"part1_scene_7c72937c_banked_refcv7_vs_P10": {}}
for a, b in (("refcv7_vmax_off_s0", "cl_p10"), ("refcv7_vmax_off_s1", "cl_p10"), ("refcv7_vmax_off_s1", "refcv7_vmax_off_s0")):
    res["part1_scene_7c72937c_banked_refcv7_vs_P10"][f"{a}_minus_{b}"] = paired_boot(per_start_share(a), per_start_share(b))
# the starts overlap: a window is 5 s, starts are 1.7 s apart
res["part1_note"] = ("starts 1.7 s apart with 5 s windows overlap ~66 %; the clusters are NOT independent episodes. "
                     "Only 3 of 9 starts carry any M1 tick (starts 34, 51, 102), i.e. one launch event in one clip.")

# PART 2
lo, hi, n_ep, delta = 0.0356, 0.0775, 141, 0.0549
se = (hi - lo) / (2 * 1.96)
sd = se * math.sqrt(n_ep)
z = 1.959964 + 0.841621   # 80 % power, two-sided 5 %
res["part2_scenes_needed_ESTIMATED"] = {
    "source": "LF read RESULTS.md s0 row 'M1 crawl share' +0.0549 [+0.0356,+0.0775], 141 episodes",
    "se_t1kin": se, "per_episode_sd": sd,
    "k_for_80pct_power_at_delta_0.055": math.ceil((z * sd / 0.055) ** 2),
    "k_for_80pct_power_at_half_effect_0.0275": math.ceil((z * sd / 0.0275) ** 2),
    "k_for_ci_halfwidth_le_0.0275": math.ceil((1.959964 * sd / 0.0275) ** 2),
    "caveat": "a scene-pre-screened M1 set has a larger mean effect and a different SD than a random eval-141 episode; order of magnitude only",
}
json.dump(res, sys.stdout, indent=1)
