"""Is the KEPT-vs-REFUSED difference real, and WHICH refusal reason drives it?

The first pass showed refused windows carry MORE and CLOSER near-forward targets. That is the
direction that flatters the bar, so it needs an interval before it is asserted -- and a
decomposition, because `future` (route horizon) and `agents` (missing join record) are different
mechanisms with different implications.

⛔ Episode-clustered, because windows are not independent (CLASS-2026-09-20-REPLICATION-UNIT).
⛔ Targets only. No model pass, no GPU.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import statistics as st
import sys

import numpy as np

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
PER_GROUP = 260
B = 4000
SEED = 20260920


def stats(item):
    box = item["agent_box"].float().numpy()
    val = item["agent_valid"].numpy().astype(bool)
    b = box[val]
    if b.shape[0] == 0:
        return None
    gx, gy = b[:, 0], b[:, 1]
    near = (gx >= 0) & (gx <= 60) & (np.abs(gy) <= 16)
    return {"n_near": float(near.sum()), "range_m": float(np.hypot(gx, gy).mean())}


def cluster_ci(by_ep, rng):
    keys = list(by_ep)
    if len(keys) < 2:
        return None
    ms = sorted(st.mean(x for k in rng.choices(keys, k=len(keys)) for x in by_ep[k])
                for _ in range(B))
    return [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B)], 3)]


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    groups = collections.defaultdict(list)
    for w in range(len(corp.ds)):
        e = corp.eligibility(w)
        groups["KEPT" if e is None else e].append(w)
    print({k: len(v) for k, v in groups.items()})

    rng_np = np.random.default_rng(SEED)
    rng = random.Random(SEED)
    out = {}
    for name, pool in groups.items():
        pick = rng_np.choice(len(pool), size=min(PER_GROUP, len(pool)), replace=False)
        near_by, rng_by = collections.defaultdict(list), collections.defaultdict(list)
        for i in pick:
            w = int(pool[int(i)])
            try:
                s = stats(corp.ds[w])
            except Exception:
                s = None
            if s:
                ep = corp.ds.index[w][0]
                near_by[ep].append(s["n_near"])
                rng_by[ep].append(s["range_m"])
        n = sum(len(v) for v in near_by.values())
        if not n:
            continue
        out[name] = {
            "n_windows": n, "n_episodes": len(near_by),
            "mean_n_near": round(st.mean(x for v in near_by.values() for x in v), 3),
            "CI_n_near": cluster_ci(near_by, rng),
            "mean_range_m": round(st.mean(x for v in rng_by.values() for x in v), 2),
            "CI_range_m": cluster_ci(rng_by, rng),
        }

    print("\nBY REFUSAL REASON, episode-clustered CI95 (B=%d):" % B)
    for k, v in out.items():
        print(f"  {k:<8} n={v['n_windows']:<4} eps={v['n_episodes']:<3} "
              f"n_near {v['mean_n_near']:<6} {v['CI_n_near']}   "
              f"range {v['mean_range_m']:<6} {v['CI_range_m']}")

    kept = out.get("KEPT")
    if kept:
        print("\nOVERLAP WITH KEPT (the test that matters):")
        for k, v in out.items():
            if k == "KEPT":
                continue
            sep_n = v["CI_n_near"][0] > kept["CI_n_near"][1] or v["CI_n_near"][1] < kept["CI_n_near"][0]
            sep_r = v["CI_range_m"][0] > kept["CI_range_m"][1] or v["CI_range_m"][1] < kept["CI_range_m"][0]
            print(f"  {k:<8} n_near separated from KEPT: {sep_n}   range separated: {sep_r}")
    pathlib.Path("bias2.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
