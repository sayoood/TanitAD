"""Is the box head's centre error LONGITUDINAL (x) or ISOTROPIC? They are different levers.

`box3d_centre` is a SUMMED L1 over both axes and the trainer logs no split, so this is measured on
HELD-OUT predictions. ⛔ It uses the TRAINING matcher (`agent_slots.match_slots`, Hungarian over
EVERY valid target) rather than the AP's 2 m greedy matcher: at a ~10 m error the 2 m matcher has
almost no pairs to average, and pairing only the lucky hits would flatter the head. The training
matcher is also what `box3d_centre` itself is computed through, so the split is comparable to it.

Reports |dx| and |dy| per matched pair, and the same split for the NEAR-FORWARD subset the
collision gate actually cares about (x in [0, 60], |y| <= 16).
Usage: python probe_box_axis.py <out.json> [--n 24]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "taniteval" / "tools"))
import s1_pass as SP                                      # noqa: E402
from tanitad.models.agent_slots import match_slots        # noqa: E402

A8 = r"C:\Users\Admin\tanitad-caches\a8-occupancy-5k-20260919\run"


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    out_path = a[0]
    n_win = int(a[2]) if len(a) > 2 and a[1] == "--n" else 24
    corp = SP.Corpus(A8 + r"\config.json",
                     r"D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB",
                     r"C:\Users\Admin\tanitad-caches\a7-imagenet-knockout-20260919\inputs\s2_labels_v8_eval.jsonl.gz",
                     r"D:\Projects\TanitAD-artifacts\b1-agent-join-3d-20260917\b1eval_agents_3d.jsonl.xz",
                     None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, A8 + r"\ckpt_5000.pt", "cpu")
    wis = [w for w in SP.trainer_windows(corp.ds, 1000) if corp.eligibility(w) is None][:n_win]
    dx, dy, gx, gy = [], [], [], []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        dec = {k: v[0].float().cpu() for k, v in out["perception"]["box_slots"].items()
               if torch.is_tensor(v) and v.dim() >= 2}
        pred = {k: v[None] for k, v in dec.items()}
        tgt = {"box": item["agent_box"][None].float(), "valid": item["agent_valid"][None],
               "cls": item["agent_cls"][None]}
        m = match_slots(pred, tgt)
        r, c = m["rows"][0].tolist(), m["cols"][0].tolist()
        for i, j in zip(r, c):
            p = dec["box"][i, :2]
            g = item["agent_box"][j, :2].float()
            dx.append(abs(float(p[0] - g[0])))
            dy.append(abs(float(p[1] - g[1])))
            gx.append(float(g[0]))
            gy.append(float(g[1]))
    dx, dy, gx, gy = map(np.asarray, (dx, dy, gx, gy))
    near = (gx >= 0) & (gx <= 60) & (np.abs(gy) <= 16)

    def blk(mask, tag):
        if not mask.any():
            return {"n": 0, "note": "no pairs"}
        return {"n": int(mask.sum()), "abs_dx_m": round(float(dx[mask].mean()), 3),
                "abs_dy_m": round(float(dy[mask].mean()), 3),
                "l1_sum_m": round(float((dx[mask] + dy[mask]).mean()), 3),
                "x_share_of_l1": round(float(dx[mask].sum() / (dx[mask].sum() + dy[mask].sum())), 4),
                "median_abs_dx_m": round(float(np.median(dx[mask])), 3),
                "median_abs_dy_m": round(float(np.median(dy[mask])), 3), "_what": tag}
    rec = {"_what": "held-out box centre error, split by axis, through the TRAINING matcher",
           "_evidence_class": "MEASURED (ours)", "_tier": "held-out (halfB), A8 ckpt_5000",
           "windows": len(wis), "matcher": "agent_slots.match_slots (Hungarian, every target)",
           "all_pairs": blk(np.ones_like(near, dtype=bool), "every matched target"),
           "near_forward": blk(near, "x in [0,60], |y|<=16 — the gate's own population"),
           "far_or_behind": blk(~near, "everything else")}
    Path(out_path).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    for k in ("all_pairs", "near_forward", "far_or_behind"):
        v = rec[k]
        print("%-14s n=%-5s |dx| %s m  |dy| %s m  L1 %s m  x-share %s" % (
            k, v.get("n"), v.get("abs_dx_m"), v.get("abs_dy_m"), v.get("l1_sum_m"),
            v.get("x_share_of_l1")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
