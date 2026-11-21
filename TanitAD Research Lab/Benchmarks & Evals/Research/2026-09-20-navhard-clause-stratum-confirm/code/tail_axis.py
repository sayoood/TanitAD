"""Close the last open axis of the eligibility-bias question.

TrainingFlyWheel proved arithmetically that `future` is a fixed POSITIONAL TAIL -- the last 39
windows of every episode -- so it can only differ from KEPT on position-in-episode. It named that
axis as the remaining candidate and did not test it.

TWO CHECKS:
  A  VERIFY THE TAIL CLAIM DIRECTLY, by index. If `future` is the tail, then in every episode the
     refused-for-future windows are exactly the highest 39 indices. That is an identity and it is
     checkable without trusting the arithmetic.
  B  TEST THE NAMED AXIS. If late windows carry systematically different TARGET DYNAMICS -- faster
     or more numerous moving agents -- then `future` is biased on the one axis position could
     drive, and the bar's population would be affected after all. Target rates, not predictions:
     no model pass, no GPU.
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


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)

    by_ep = collections.defaultdict(list)          # ep -> [(local_idx, reason)]
    for w in range(len(corp.ds)):
        ep, loc = corp.ds.index[w]
        by_ep[ep].append((loc, corp.eligibility(w), w))

    # ---- A: is `future` exactly the top-39 by index, in every episode? ----
    viol, fut_counts, tails = [], [], 0
    for ep, rows in by_ep.items():
        rows.sort()
        fut = [r[0] for r in rows if r[1] == "future"]
        fut_counts.append(len(fut))
        if not fut:
            continue
        top = [r[0] for r in rows[-len(fut):]]
        if sorted(fut) == sorted(top):
            tails += 1
        else:
            viol.append((ep, len(fut), min(fut), max(fut), rows[-1][0]))
    print(f"A) episodes: {len(by_ep)}   `future` count per episode: "
          f"min {min(fut_counts)} max {max(fut_counts)} distinct {sorted(set(fut_counts))}")
    print(f"   `future` == the top-N indices in {tails}/{len(by_ep)} episodes; "
          f"violations {len(viol)}")
    if viol:
        print("   first violations:", viol[:3])

    # ---- B: do the tail windows carry different TARGET DYNAMICS? ----
    groups = collections.defaultdict(list)
    for ep, rows in by_ep.items():
        for loc, reason, w in rows:
            groups["KEPT" if reason is None else reason].append((ep, w))

    rng_np = np.random.default_rng(SEED)
    rng = random.Random(SEED)
    out = {}
    for name in ("KEPT", "future"):
        pool = groups[name]
        pick = rng_np.choice(len(pool), size=min(PER_GROUP, len(pool)), replace=False)
        spd_by, mov_by = collections.defaultdict(list), collections.defaultdict(list)
        for i in pick:
            ep, w = pool[int(i)]
            try:
                item = corp.ds[w]
                val = item["agent_valid"].numpy().astype(bool)
                rates = item["agent_rates"].float().numpy()
                mask = item["agent_rates_mask"].numpy().astype(bool)
                sel = val & mask[:, 0] if mask.ndim > 1 else val & mask
                if sel.sum() == 0:
                    continue
                spd = np.hypot(rates[sel, 0], rates[sel, 1])
            except Exception:
                continue
            spd_by[ep].append(float(spd.mean()))
            mov_by[ep].append(float((spd > 0.5).mean()))
        n = sum(len(v) for v in spd_by.values())
        if not n:
            out[name] = {"n": 0}
            continue

        def ci(d):
            keys = list(d)
            ms = sorted(st.mean(x for k in rng.choices(keys, k=len(keys)) for x in d[k])
                        for _ in range(B))
            return [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B)], 3)]
        out[name] = {"n_windows": n, "n_episodes": len(spd_by),
                     "mean_target_speed_mps": round(
                         st.mean(x for v in spd_by.values() for x in v), 3),
                     "CI_speed": ci(spd_by),
                     "frac_targets_moving": round(
                         st.mean(x for v in mov_by.values() for x in v), 3),
                     "CI_moving": ci(mov_by)}

    print("\nB) TARGET DYNAMICS, episode-clustered CI95 (B=%d):" % B)
    for k, v in out.items():
        print(f"   {k:<7} {v}")
    k, f = out.get("KEPT"), out.get("future")
    if k and f and k.get("n_windows") and f.get("n_windows"):
        sep_s = f["CI_speed"][0] > k["CI_speed"][1] or f["CI_speed"][1] < k["CI_speed"][0]
        sep_m = f["CI_moving"][0] > k["CI_moving"][1] or f["CI_moving"][1] < k["CI_moving"][0]
        print(f"\n   separated from KEPT -- speed: {sep_s}   moving-fraction: {sep_m}")
        print("   => " + ("BIASED on the position axis" if (sep_s or sep_m)
                          else "NOT biased on the one axis position could drive"))
    pathlib.Path("tail_axis.json").write_text(
        json.dumps({"future_is_tail_in": tails, "episodes": len(by_ep),
                    "future_per_episode": sorted(set(fut_counts)), "dynamics": out},
                   indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
