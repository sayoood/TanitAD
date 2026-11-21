"""Independently census `eligibility`, AND answer the question that matters more than the ceiling:
are the REFUSED windows systematically different from the KEPT ones?

If they are, the perception bar (5.9539 m) is measured on a biased subset of the corpus, and that
is a bigger problem than the effective-n ceiling.

⛔ VERIFIED, NOT INHERITED. The 27.2 % figure I was given was a denominator error and I repeated
it downstream, so the replacement census is re-derived here from `eligibility`'s own return value
over the FULL grid, not sampled and not taken on trust.

⛔ NO MODEL PASS. Bias is tested on the TARGETS, which need no forward pass and no GPU.
"""
from __future__ import annotations

import collections
import json
import pathlib
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
SAMPLE = 150          # windows per group for the target-geometry comparison


def near_stats(item):
    """Near-forward TARGET geometry for one window. Targets only -- no prediction involved."""
    box = item["agent_box"].float().numpy()
    val = item["agent_valid"].numpy().astype(bool)
    b = box[val]
    if b.shape[0] == 0:
        return None
    gx, gy = b[:, 0], b[:, 1]
    near = (gx >= 0) & (gx <= 60) & (np.abs(gy) <= 16)
    return {"n_targets": int(val.sum()), "n_near": int(near.sum()),
            "mean_range": float(np.hypot(gx, gy).mean()),
            "mean_near_x": float(gx[near].mean()) if near.any() else None}


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    n_grid = len(corp.ds)
    print(f"grid windows: {n_grid}")

    reasons, per_ep = collections.Counter(), collections.defaultdict(lambda: [0, 0])
    kept, refused = [], []
    for w in range(n_grid):
        e = corp.eligibility(w)
        reasons[e if e is not None else "ELIGIBLE"] += 1
        ep = corp.ds.index[w][0]
        per_ep[ep][0 if e is None else 1] += 1
        (kept if e is None else refused).append(w)

    tot = sum(reasons.values())
    assert tot == n_grid, f"control: census {tot} != grid {n_grid}"
    print(f"\nFULL-GRID CENSUS (denominator = {n_grid}, stated because the last one was wrong):")
    for k, v in reasons.most_common():
        print(f"  {k:<12} {v:>6}  {v/n_grid:7.2%}")

    zero_ep = [e for e, (ok, _) in per_ep.items() if ok == 0]
    fracs = sorted(ok / (ok + bad) for ok, bad in per_ep.values())
    print(f"\nepisodes: {len(per_ep)}   with ZERO eligible windows: {len(zero_ep)}")
    print(f"  => effective-n CEILING = {len(per_ep) - len(zero_ep)}")
    print(f"  per-episode eligible fraction: min {fracs[0]:.3f}  p10 {fracs[len(fracs)//10]:.3f}"
          f"  median {fracs[len(fracs)//2]:.3f}  max {fracs[-1]:.3f}")

    # ---- THE BIAS TEST: do the refused windows differ in their TARGET geometry? ----
    rng = np.random.default_rng(20260920)
    out = {}
    for name, pool in (("KEPT", kept), ("REFUSED", refused)):
        pick = rng.choice(len(pool), size=min(SAMPLE, len(pool)), replace=False)
        rows = []
        for i in pick:
            try:
                s = near_stats(corp.ds[int(pool[int(i)])])
            except Exception:
                s = None
            if s:
                rows.append(s)
        if not rows:
            out[name] = {"n": 0}
            continue
        out[name] = {
            "n_windows": len(rows),
            "mean_n_targets": round(st.mean(r["n_targets"] for r in rows), 2),
            "mean_n_near": round(st.mean(r["n_near"] for r in rows), 2),
            "mean_range_m": round(st.mean(r["mean_range"] for r in rows), 2),
            "frac_with_any_near": round(
                sum(1 for r in rows if r["n_near"] > 0) / len(rows), 3),
        }
    print("\nBIAS TEST -- target geometry, KEPT vs REFUSED (no model, no GPU):")
    for k, v in out.items():
        print(f"  {k:<8} {v}")

    rec = {"_what": "full-grid eligibility census + kept-vs-refused target-geometry bias test",
           "_evidence_class": "MEASURED (ours), CPU only, no model pass",
           "_denominator": n_grid, "census": dict(reasons),
           "census_frac": {k: round(v / n_grid, 4) for k, v in reasons.items()},
           "n_episodes": len(per_ep), "episodes_with_zero_eligible": len(zero_ep),
           "effective_n_ceiling": len(per_ep) - len(zero_ep),
           "per_episode_eligible_frac": {"min": round(fracs[0], 3),
                                         "median": round(fracs[len(fracs) // 2], 3),
                                         "max": round(fracs[-1], 3)},
           "bias_test": out}
    pathlib.Path("elig_bias.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
