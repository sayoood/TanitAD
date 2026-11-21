"""Does near-forward box centre error DEPEND ON EGO SPEED?

This is the link nobody has measured, and without it the sign of the `agents` refusal's effect on
the bar is unknown. TrainingFlyWheel measured that `agents`-refused windows are 2.08x FASTER than
kept ones (separated, but on only 5 episodes). If error does not depend on speed, that difference
cannot move the bar whatever its size. If error RISES with speed, removing fast windows FLATTERS
the bar and 5.9539 m is optimistic.

DESIGN -- stratified, because a correlation over an unstratified sample wastes the model passes:
take the TOP and BOTTOM speed quartiles of the ELIGIBLE windows (speed is free -- poses only) and
score equal numbers from each. A contrast between quartiles has far more power per forward pass
than a correlation over the middle.

⛔ Uses `box_quality.match_pairs` and `population_mask`, the canonical mutation-proven functions --
NOT a reimplementation. The lesson of `RETR-2026-09-20-ABSENCE-AT-ONE-LOCATION-BOX-SCORER`.
⛔ CPU only. The GPU is held for the PI and the panel is paused on his HOLD.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import statistics as st
import sys

import numpy as np
import torch

REPO = pathlib.Path("D:/Projects/TanitAD")
sys.path.insert(0, str(REPO / "taniteval" / "tools"))
sys.path.insert(0, str(REPO / "stack"))
import s1_pass as SP                                      # noqa: E402
from box_quality import match_pairs                       # noqa: E402

A8 = pathlib.Path("C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run")
EP = "D:/Projects/TanitAD-artifacts/v2ep-eval124clean-416x1024cyl-halfB"
LAB = ("C:/Users/Admin/tanitad-caches/a7-imagenet-knockout-20260919/inputs/"
       "s2_labels_v8_eval.jsonl.gz")
AG = "D:/Projects/TanitAD-artifacts/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz"
W = 8                    # window length; t0 = t + W - 1. Pinned by the refusal arithmetic:
                         # 199 poses, ROUTE_TICKS 60 => refuse t0+60 > 198 => t = 132..170 = 39.
PER_Q = 55               # windows scored per quartile
B = 4000
SEED = 20260920


def ego_speed(corp, e_i, t):
    """metres per tick at t0, from POSES -- a channel the agent join does not touch."""
    pa = corp.poses(e_i)
    t0 = t + W - 1
    if t0 < 1 or t0 >= pa.shape[0]:
        return None
    return float(torch.linalg.norm(pa[t0][:2] - pa[t0 - 1][:2]))


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)

    # --- speed for every ELIGIBLE window: poses only, no model ---
    rows = []
    for w in range(len(corp.ds)):
        if corp.eligibility(w) is not None:
            continue
        e_i, t = corp.ds.index[w]
        s = ego_speed(corp, e_i, t)
        if s is not None:
            rows.append((s, w, e_i))
    rows.sort()
    n = len(rows)
    q1, q3 = rows[: n // 4], rows[3 * n // 4:]
    print(f"eligible with speed: {n}   Q1 median {q1[len(q1)//2][0]:.3f}"
          f"   Q4 median {q3[len(q3)//2][0]:.3f} m/tick")

    rng_np = np.random.default_rng(SEED)
    rng = random.Random(SEED)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    out = {}
    for name, pool in (("SLOW_q1", q1), ("FAST_q4", q3)):
        pick = rng_np.choice(len(pool), size=min(PER_Q, len(pool)), replace=False)
        by_ep, speeds, npairs = collections.defaultdict(list), [], 0
        for i in pick:
            s, w, e_i = pool[int(i)]
            item = corp.ds[w]
            if not bool(item.get("agent_label", False)):
                continue
            o = mp.forward(item, str(corp.clip_ids[e_i]))
            dec = {k: v[0].float().cpu() for k, v in o["perception"]["box_slots"].items()
                   if torch.is_tensor(v) and v.dim() >= 2}
            r = match_pairs({k: v[None] for k, v in dec.items()},
                            {"box": item["agent_box"][None].float(),
                             "valid": item["agent_valid"][None], "cls": item["agent_cls"][None],
                             "rates": item["agent_rates"][None].float(),
                             "rates_mask": item["agent_rates_mask"][None].bool()})
            gx, gy = np.asarray(r["gx"]), np.asarray(r["gy"])
            near = (gx >= 0) & (gx <= 60) & (np.abs(gy) <= 16)
            if not near.any():
                continue
            l1 = (np.asarray(r["dx"])[near] + np.asarray(r["dy"])[near])
            by_ep[e_i].append(float(l1.mean()))
            speeds.append(s)
            npairs += int(near.sum())
        keys = list(by_ep)
        ms = sorted(st.mean(x for k in rng.choices(keys, k=len(keys)) for x in by_ep[k])
                    for _ in range(B)) if len(keys) > 1 else None
        out[name] = {
            "n_windows": sum(len(v) for v in by_ep.values()), "n_episodes": len(keys),
            "n_near_pairs": npairs,
            "mean_speed_m_per_tick": round(st.mean(speeds), 3) if speeds else None,
            "mean_near_L1_m": round(st.mean(x for v in by_ep.values() for x in v), 3),
            "CI95": [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B)], 3)] if ms else None,
        }
        print(f"  {name}: {out[name]}", flush=True)

    a, b = out["SLOW_q1"], out["FAST_q4"]
    sep = (a["CI95"] and b["CI95"] and
           (b["CI95"][0] > a["CI95"][1] or b["CI95"][1] < a["CI95"][0]))
    out["_separated"] = bool(sep)
    out["_speed_ratio"] = round(b["mean_speed_m_per_tick"] / a["mean_speed_m_per_tick"], 2)
    out["_VERDICT"] = (
        "ERROR DEPENDS ON SPEED -- the agents refusal CAN move the bar; sign follows the direction"
        if sep else
        "NO SPEED DEPENDENCE DETECTED -- the agents speed difference cannot move the bar via this "
        "channel at this n")
    print(f"\nspeed ratio FAST/SLOW = {out['_speed_ratio']}x")
    print(out["_VERDICT"])
    pathlib.Path("speed_vs_error.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
