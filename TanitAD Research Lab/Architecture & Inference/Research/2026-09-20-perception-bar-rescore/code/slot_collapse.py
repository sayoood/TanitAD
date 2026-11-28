"""Has the SLOT DECODER collapsed — do all 100 queries emit essentially the same box?

`286e0d3` showed presence is near-constant ACROSS SLOTS (within-window IQR 0.054). That raises a
question about a LANDED number: if the slots are degenerate, then the near-forward centre error of
**5.9539 m** (`a9e75c6`, `cb274b0`) is not a localisation measurement at all — it is the distance
from a single blob to whatever targets happen to be near it.

⭐ THE TEST, with its control built in. Measure the dispersion of PREDICTED box centres across the
100 slots, and compare it to the dispersion of the TARGET centres in the same window:

    ratio = spread(pred centres) / spread(target centres)

  ratio ~ 1  -> the decoder spreads its boxes like the scene does; only presence collapsed
  ratio ~ 0  -> SLOT COLLAPSE: every query emits the same box and the bar means something else

⛔ The target spread is the control and it is not optional: a scene whose agents are all clustered
would make a low predicted spread look like collapse. Dividing by the scene's own spread removes
that.
⛔ Matcher-free, CPU only.
"""
from __future__ import annotations

import json
import pathlib
import statistics as st
import sys

import numpy as np
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
N_WIN = 40


def spread(xy: np.ndarray) -> float:
    """mean distance of points from their centroid — scale-free, robust to n."""
    if xy.shape[0] < 2:
        return 0.0
    c = xy.mean(axis=0)
    return float(np.hypot(xy[:, 0] - c[0], xy[:, 1] - c[1]).mean())


def main() -> int:
    corp = SP.Corpus(str(A8 / "config.json"), EP, LAB, AG, None)
    SP.attach_agent_gt(corp, corp.targs)
    mp = SP.ModelPass(corp, str(A8 / "ckpt_5000.pt"), "cpu")
    wis = [w for w in SP.trainer_windows(corp.ds, 1000)
           if corp.eligibility(w) is None][:N_WIN]

    rows = []
    for wi in wis:
        item = corp.ds[wi]
        if not bool(item.get("agent_label", False)):
            continue
        e_i, _ = corp.ds.index[wi]
        out = mp.forward(item, str(corp.clip_ids[e_i]))
        pb = out["perception"]["box_slots"]["box"][0].float().cpu().numpy()[:, :2]
        tb = item["agent_box"].float().numpy()
        tv = item["agent_valid"].numpy().astype(bool)
        t = tb[tv][:, :2]
        if t.shape[0] < 2:
            continue
        sp_p, sp_t = spread(pb), spread(t)
        rows.append({"pred_spread": sp_p, "tgt_spread": sp_t,
                     "ratio": sp_p / sp_t if sp_t else float("nan"),
                     "n_slots": int(pb.shape[0]), "n_tgt": int(t.shape[0]),
                     "pred_uniq": int(np.unique(np.round(pb, 2), axis=0).shape[0])})
        if len(rows) % 10 == 0:
            print(f"  {len(rows)} windows", flush=True)

    assert rows, "control: zero windows"
    ratios = sorted(r["ratio"] for r in rows)
    res = {"_what": "has the slot decoder collapsed? predicted vs target centre dispersion",
           "_evidence_class": "MEASURED (ours), CPU, A8 ckpt_5000, matcher-free",
           "_control": "divide by the SCENE's own target spread, so a clustered scene cannot "
                       "masquerade as collapse",
           "n_windows": len(rows), "n_slots": rows[0]["n_slots"],
           "mean_pred_spread_m": round(st.mean(r["pred_spread"] for r in rows), 3),
           "mean_target_spread_m": round(st.mean(r["tgt_spread"] for r in rows), 3),
           "ratio_mean": round(st.mean(ratios), 4),
           "ratio_median": round(ratios[len(ratios) // 2], 4),
           "ratio_min": round(ratios[0], 4), "ratio_max": round(ratios[-1], 4),
           "mean_distinct_pred_boxes_at_1cm": round(
               st.mean(r["pred_uniq"] for r in rows), 1)}
    res["_VERDICT"] = (
        "⛔ SLOT COLLAPSE — the decoder emits near-identical boxes; the 5.9539 m bar is NOT a "
        "localisation measurement" if res["ratio_median"] < 0.15 else
        "the decoder spreads its boxes comparably to the scene ⇒ only PRESENCE collapsed and the "
        "centre-error bar stands as a localisation measurement"
        if res["ratio_median"] > 0.5 else
        "⚠️ INTERMEDIATE — the decoder is under-dispersed but not degenerate; the bar needs this "
        "ratio quoted beside it")
    print(json.dumps(res, indent=1))
    print("\n" + res["_VERDICT"])
    pathlib.Path("slot_collapse.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
