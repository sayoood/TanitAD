"""Make `D-DAC-ZEROES-THE-HUMAN` actionable: WHERE does the human's DAC fail?

MEASURED already: with the SAM3 map live, `pdm_proxy.dac_from_drivable` scores the RECORDED
human non-compliant on 44.6 % of held-out windows. That number alone does not say what to do.
The rule fails a window if ANY of 4 ego-box corners at ANY of 41 ticks lands on a SEEN cell with
drivable fraction < 0.5, so there are three candidate causes and they imply different repairs:

  (a) FAR RANGE  — the offending tick is late / far ahead, where the SAM3 map is least reliable
                   ⇒ the repair is a RANGE CAP on the DAC horizon, not a new map;
  (b) THRESHOLD  — offending cells sit just under 0.5 drivable ⇒ the repair is the threshold;
  (c) EVERYWHERE — failures start at t0 next to the ego ⇒ the map or the frame is wrong, and
                   neither (a) nor (b) would help.

This prints, per window: the FIRST offending tick, its distance from the ego, the offending
cell's drivable fraction, and the DAC the human would score if the horizon stopped at 1/2/3/4 s.
No model and no GPU: the human's states come from the recorded poses, exactly as item 19 builds
them. Usage: python probe_dac_horizon.py <out.json> [--n 400]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "taniteval" / "tools"))
import s1_pass as SP                                      # noqa: E402
from tanitad.rl import pdm_proxy as P                     # noqa: E402

A8 = r"C:\Users\Admin\tanitad-caches\a8-occupancy-5k-20260919\run"


def first_offender(states, fr, sn, cfg=P.PROXY):
    """The rule of ``dac_from_drivable``, opened up: the first (tick, corner) that fails."""
    corners = P._ego_boxes(states[None], cfg)[0]           # [T, 4, 2]
    ix = torch.floor(corners[..., 0] / 0.5).long()
    iy = torch.floor((corners[..., 1] + 16.0) / 0.5).long()
    h, w = fr.shape
    inside = (ix >= 0) & (ix < h) & (iy >= 0) & (iy < w)
    frac = fr[ix.clamp(0, h - 1), iy.clamp(0, w - 1)]
    seen = sn[ix.clamp(0, h - 1), iy.clamp(0, w - 1)]
    off = inside & seen & (frac < 0.5)                     # [T, 4]
    if not bool(off.any()):
        return None
    t = int(off.any(dim=1).float().argmax())
    c = int(off[t].float().argmax())
    return {"tick": t, "t_s": round(t * cfg.dt, 2),
            "range_m": round(float(corners[t, c].norm()), 2),
            "x_m": round(float(corners[t, c, 0]), 2), "y_m": round(float(corners[t, c, 1]), 2),
            "drivable_frac": round(float(frac[t, c]), 3)}


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    out_path, n = a[0], (int(a[2]) if len(a) > 2 and a[1] == "--n" else 400)
    corp = SP.Corpus(A8 + r"\config.json",
                     r"D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB",
                     r"C:\Users\Admin\tanitad-caches\a7-imagenet-knockout-20260919\inputs\s2_labels_v8_eval.jsonl.gz",
                     r"D:\Projects\TanitAD-artifacts\b1-agent-join-3d-20260917\b1eval_agents_3d.jsonl.xz",
                     r"D:\Projects\TanitAD-artifacts\sam3-maps-eval")
    wis = [w for w in SP.trainer_windows(corp.ds, 1000) if corp.eligibility(w) is None][:n]
    rows = []
    for wi in wis:
        it = corp.light_item(wi)
        if it["map_drivable"] is None:
            continue
        st, fr, sn = it["human"], it["map_drivable"], it["map_seen"]
        dac_h = {}
        for k, tag in ((10, "1s"), (20, "2s"), (30, "3s"), (40, "4s")):
            dac_h[tag] = float(P.dac_from_drivable(st[None, :k + 1], fr, sn)[0])
        rows.append({"sha12": it["sha12"], "t0": it["t0"], "dac": dac_h,
                     "first_offender": first_offender(st, fr, sn),
                     "seen_frac": round(float(sn.float().mean()), 4),
                     "drivable_frac_mean_seen": round(float(fr[sn].mean()) if bool(sn.any())
                                                      else float("nan"), 4)})
    fails = [r for r in rows if r["first_offender"]]
    off_t = np.array([r["first_offender"]["t_s"] for r in fails]) if fails else np.array([])
    off_r = np.array([r["first_offender"]["range_m"] for r in fails]) if fails else np.array([])
    off_f = np.array([r["first_offender"]["drivable_frac"] for r in fails]) if fails else np.array([])
    rec = {"_what": "D-DAC-ZEROES-THE-HUMAN, opened up: where does the human's DAC fail?",
           "_evidence_class": "MEASURED (ours)", "_tier": "T0 (recorded human, no model)",
           "n_windows": len(rows),
           "human_dac_by_horizon": {k: round(float(np.mean([r["dac"][k] for r in rows])), 4)
                                    for k in ("1s", "2s", "3s", "4s")},
           "fail_share_4s": round(len(fails) / max(len(rows), 1), 4),
           "first_offender_t_s": {"p10": float(np.quantile(off_t, 0.1)) if fails else None,
                                  "median": float(np.median(off_t)) if fails else None,
                                  "p90": float(np.quantile(off_t, 0.9)) if fails else None},
           "first_offender_range_m": {"p10": float(np.quantile(off_r, 0.1)) if fails else None,
                                      "median": float(np.median(off_r)) if fails else None,
                                      "p90": float(np.quantile(off_r, 0.9)) if fails else None},
           "offending_drivable_frac": {"median": float(np.median(off_f)) if fails else None,
                                       "share_above_0.4": float((off_f > 0.4).mean()) if fails else None},
           "map_seen_frac_mean": round(float(np.mean([r["seen_frac"] for r in rows])), 4),
           "rows": rows}
    Path(out_path).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("human DAC by horizon:", rec["human_dac_by_horizon"])
    print("fail share @4s %.4f | first offender t_s %s | range_m %s | offending frac %s" % (
        rec["fail_share_4s"], rec["first_offender_t_s"], rec["first_offender_range_m"],
        rec["offending_drivable_frac"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
