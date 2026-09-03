"""CONTROL — the probe's own bootstrap against the PROGRAMME's own estimator.

The brief binds the interval to `taniteval/ci.py`'s paired episode-cluster bootstrap.
`cost_surface_probe._boot_mean` re-implements that shape (per-window indicator, episodes
resampled with replacement, full-set point estimate, percentile bounds) because the
statistic here is a fraction rather than a displacement. A re-implementation is only
admissible if it reads the same value as the instrument it replaces, so this runs BOTH on
the SAME per-window arrays out of the banked JSON and prints the difference.

⛔ `overlapping_holdout_se` is never used and is not imported.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\taniteval")
from taniteval.ci import (episode_cluster_bootstrap,  # noqa: E402
                          paired_episode_cluster_bootstrap)

sys.path.insert(0, r"C:\Users\Admin\tanitad-wt\taniteval\tools")
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "csp", r"C:\Users\Admin\tanitad-wt\taniteval\tools\cost_surface_probe.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

HERE = os.path.dirname(os.path.abspath(__file__))
N_BOOT = 10000
out = {}
for tag, f in (("incumbent", "cost_surface_incumbent.json"),
               ("ep2", "cost_surface_ep2.json")):
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        continue
    with open(p, encoding="utf-8") as fh:
        j = json.load(fh)
    rows = j["rows"]
    eid = np.array([r["ep_file"] for r in rows])
    blk = {}
    for arm in P.ARMS:
        v = np.array([1.0 if r[f"{arm}_turn"] else 0.0 for r in rows])
        mine = P._boot_mean(v, eid, N_BOOT, 0)
        theirs = episode_cluster_bootstrap(v, eid, reduce="mean", n_boot=N_BOOT,
                                           seed=0, dp=6)
        blk[arm] = {"mine": [mine["point"], mine["lo"], mine["hi"]],
                    "taniteval_ci": [theirs["mean"], theirs["lo"], theirs["hi"]],
                    "d_point": abs(mine["point"] - theirs["mean"]),
                    "d_lo": abs(mine["lo"] - theirs["lo"]),
                    "d_hi": abs(mine["hi"] - theirs["hi"]),
                    "estimator_theirs": theirs["estimator"],
                    "n_episodes": theirs["n_episodes"]}
    # the PAIRED form, on the share that decides the boundary factor
    a = np.array([1.0 if r["B_full_turn"] else 0.0 for r in rows])
    b = np.array([1.0 if r["A_full_turn"] else 0.0 for r in rows])
    mine = P._boot_mean(a - b, eid, N_BOOT, 0)
    theirs = paired_episode_cluster_bootstrap(a, b, eid, n_boot=N_BOOT, seed=0)
    blk["share_ii_boundary_PAIRED"] = {
        "mine": [mine["point"], mine["lo"], mine["hi"]],
        "taniteval_ci": [theirs.get("delta", theirs.get("mean")),
                         theirs["lo"], theirs["hi"]],
        "separated_taniteval": theirs.get("separated"),
        "degenerate": theirs.get("degenerate", False),
        "degenerate_note": theirs.get("degenerate_note"),
        "p_delta_gt0": theirs.get("p_delta_gt0"),
        "estimator": theirs.get("estimator")}
    out[tag] = blk
    worst = max(max(v["d_point"], v["d_lo"], v["d_hi"])
                for k, v in blk.items() if "d_point" in v)
    print(f"== {tag}: worst |our bound - taniteval.ci bound| over "
          f"{len(P.ARMS)} arms = {worst:.6f}  (n_episodes "
          f"{blk[P.ARMS[0]]['n_episodes']}, n_boot {N_BOOT})")
    for arm in P.ARMS:
        m, t = blk[arm]["mine"], blk[arm]["taniteval_ci"]
        print(f"   {arm:14s} ours {m[0]:.4f} [{m[1]:.4f}, {m[2]:.4f}]   "
              f"taniteval {t[0]:.4f} [{t[1]:.4f}, {t[2]:.4f}]")
    pb = blk["share_ii_boundary_PAIRED"]
    print(f"   PAIRED share(ii): ours {pb['mine']}  taniteval {pb['taniteval_ci']}"
          f"  separated={pb['separated_taniteval']}")

with open(os.path.join(HERE, "ci_crosscheck.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, default=str)
print("\nwritten: ci_crosscheck.json")
