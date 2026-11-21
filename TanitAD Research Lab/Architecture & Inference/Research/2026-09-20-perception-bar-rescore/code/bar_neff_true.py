"""The perception bar's effective cluster count, from the ACTUAL matched pairs — no proxy.

The TrainingFlyWheel computed n_eff 12.60 from RAW gate targets (139) as a proxy for scored pairs,
flagged that the proxy overcounted the banked matched n (129) by 7.2 %, and has since explained the
gap exactly: the trainer pads to the nearest 32 over ALL targets and applies the near-forward
filter LATER, so near-behind targets consume slots.

⇒ The proxy is now understood, which means it no longer has to be used. `box_quality.match_pairs`
returns the pairs that were actually SCORED, so the cluster sizes can be counted from them
directly. This replaces a corrected proxy with a measurement.

⛔ Same window selection as the bar itself: trainer_windows(ds, 1000), eligibility filter, [:24].
⛔ CPU only.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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


def n_eff(sizes):
    tot = sum(sizes)
    return (tot * tot) / sum(s * s for s in sizes) if tot else 0.0


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    wis = [w for w in SP.trainer_windows(corp.ds, 1000)
           if corp.eligibility(w) is None][:24]

    per_ep = collections.Counter()
    per_win = []
    total = 0
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        o = mp.forward(item, str(corp.clip_ids[e_i]))
        dec = {k: v[0].float().cpu() for k, v in o["perception"]["box_slots"].items()
               if torch.is_tensor(v) and v.dim() >= 2}
        r = match_pairs({k: v[None] for k, v in dec.items()},
                        {"box": item["agent_box"][None].float(),
                         "valid": item["agent_valid"][None], "cls": item["agent_cls"][None],
                         "rates": item["agent_rates"][None].float(),
                         "rates_mask": item["agent_rates_mask"][None].bool()})
        gx, gy = np.asarray(r["gx"]), np.asarray(r["gy"])
        k = int(((gx >= 0) & (gx <= 60) & (np.abs(gy) <= 16)).sum())
        per_ep[e_i] += k
        per_win.append(k)
        total += k

    sizes = sorted(v for v in per_ep.values() if v > 0)
    e = n_eff(sizes)
    out = {"_what": "bar n_eff from ACTUAL matched near-forward pairs, not a raw-target proxy",
           "_basis": "box_quality.match_pairs on A8 ckpt_5000, 24 windows, CPU",
           "windows": len(per_win), "episodes_touched": len(per_ep),
           "non_empty_clusters": len(sizes), "total_near_pairs": total,
           "cluster_sizes": sizes, "n_eff": round(e, 2),
           "pct_of_non_empty": round(100 * e / len(sizes), 1) if sizes else None,
           "banked_n_for_cross_check": 129}
    print(json.dumps(out, indent=1))
    print(f"\ntotal near-forward pairs {total}  (banked 129)  "
          + ("✔ MATCHES the banked n" if total == 129 else "⛔ DIFFERS from the banked n"))
    print(f"non-empty clusters {len(sizes)}  ->  n_eff {e:.2f} "
          f"({100*e/len(sizes):.1f}% of them)")
    pathlib.Path("bar_neff_true.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
