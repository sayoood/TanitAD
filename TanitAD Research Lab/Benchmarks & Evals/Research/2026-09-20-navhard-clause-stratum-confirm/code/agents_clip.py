"""Verify the retraction: is the `agents` refusal a CLIP-level defect, not a speed-dependent one?

This corrects a claim I LANDED (5547a76 characterises the channel as removing "windows 2.08x
faster than kept ones"), so it is re-derived here rather than inherited — the programme rule, and
doubly so because the source is retracting its own earlier number.

TWO CHECKS, both independent of the agent join (speed comes from POSES):
  A  CONCENTRATION — are the scattered refusals confined to a few clips?
  B  DOSE-RESPONSE — does the refusal rate rise MONOTONICALLY with ego speed? A two-group
     contrast cannot tell a dose-response from a confound; deciles can.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
W = 8


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)

    per_ep = collections.defaultdict(lambda: collections.Counter())
    rows = []
    for w in range(len(corp.ds)):
        e_i, t = corp.ds.index[w]
        r = corp.eligibility(w)
        per_ep[e_i][r if r is not None else "KEPT"] += 1
        pa = corp.poses(e_i)
        t0 = t + W - 1
        if 1 <= t0 < pa.shape[0]:
            rows.append((float(torch.linalg.norm(pa[t0][:2] - pa[t0 - 1][:2])),
                         e_i, r == "agents"))

    # ---- A: concentration of `agents` refusals across clips ----
    dead = [e for e, c in per_ep.items() if c["KEPT"] == 0]
    scat = {e: c["agents"] for e, c in per_ep.items()
            if c["agents"] > 0 and c["KEPT"] > 0}
    tot_scat = sum(scat.values())
    top = sorted(scat.items(), key=lambda kv: -kv[1])
    print(f"A) episodes {len(per_ep)}  wholly dead {len(dead)} {sorted(dead)}")
    print(f"   clips with SCATTERED agents-refusals: {len(scat)} of "
          f"{sum(1 for c in per_ep.values() if c['KEPT']>0)} live   total {tot_scat}")
    for e, n in top[:6]:
        print(f"     ep {e:>3}: agents {n:>4}  kept {per_ep[e]['KEPT']:>4}  "
              f"({n/tot_scat:.1%} of all scattered)")
    if top:
        print(f"   top-1 share {top[0][1]/tot_scat:.1%}   "
              f"top-2 share {sum(n for _, n in top[:2])/tot_scat:.1%}")

    # ---- B: dose-response over speed deciles (LIVE clips only) ----
    live = [r for r in rows if per_ep[r[1]]["KEPT"] > 0]
    live.sort()
    n = len(live)
    print(f"\nB) DOSE-RESPONSE over {n} live windows, deciles of ego speed:")
    print("   dec  speed range          n_win  n_eps  frac_agents_refused")
    fracs = []
    for d in range(10):
        lo, hi = d * n // 10, (d + 1) * n // 10
        chunk = live[lo:hi]
        f = sum(1 for r in chunk if r[2]) / len(chunk)
        fracs.append(f)
        print(f"   {d+1:>3}  {chunk[0][0]:.3f}-{chunk[-1][0]:.3f}  "
              f"{len(chunk):>6} {len({r[1] for r in chunk}):>6}  {f:.4f}")
    zeros = sum(1 for f in fracs if f == 0.0)
    mono = all(fracs[i] <= fracs[i + 1] for i in range(9))
    print(f"\n   deciles with EXACTLY zero: {zeros}/10")
    print(f"   monotone non-decreasing in speed: {mono}")
    print(f"   top decile {fracs[9]:.4f} vs 9th {fracs[8]:.4f}  -> "
          + ("top is LOWER; a speed-dependent join is REFUTED" if fracs[9] < fracs[8]
             else "top is highest; consistent with speed dependence"))
    pathlib.Path("agents_clip.json").write_text(json.dumps(
        {"_what": "is the agents refusal clip-level or speed-level?",
         "wholly_dead_eps": sorted(dead), "scattered_clips": len(scat),
         "scattered_total": tot_scat,
         "top_clips": [{"ep": e, "agents": n, "kept": per_ep[e]["KEPT"]} for e, n in top],
         "decile_frac": [round(f, 4) for f in fracs],
         "deciles_zero": zeros, "monotone": mono}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
