"""Does the box read score the head against targets it is never TRAINED on? (Master Mind, 2026-09-20)

The absence-vs-defect question: a head scored on targets it cannot represent looks exactly like a
head that cannot localise. Two halves, and only the first needs compute:

 * SOURCE (free, read not argued): the dataset emits agent targets RAW — "Targets are emitted RAW
   (no visibility filter here). The filter lives in refc_agents.agent_losses" (refc_v3_train.py
   ~2676) — and the box3d loss matches through `agent_slots.match_slots`, which Hungarian-matches
   EVERY valid target, dropping only the FARTHEST when targets exceed queries (here 32 targets vs
   100 queries, so it never drops). No range mask, no clip, in training OR in the read.
 * MEASURED here: the share of GT boxes inside the head's decode extent (x in [0, 60], |y| <= 16),
   plus the range distribution — the number that says which world we are in.

No model, no GPU: the targets come from the trainer's own dataset fields.
Usage: python probe_gt_extent.py <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "taniteval" / "tools"))
import s1_pass as SP                                      # noqa: E402

A8 = r"C:\Users\Admin\tanitad-caches\a8-occupancy-5k-20260919\run"
X_FWD, Y_HALF = 60.0, 16.0                                 # SlotDecodeRanges on this checkpoint


def main(argv=None) -> int:
    out_path = (sys.argv[1:] if argv is None else argv)[0]
    corp = SP.Corpus(A8 + r"\config.json",
                     r"D:\Projects\TanitAD-artifacts\v2ep-eval124clean-416x1024cyl-halfB",
                     r"C:\Users\Admin\tanitad-caches\a7-imagenet-knockout-20260919\inputs\s2_labels_v8_eval.jsonl.gz",
                     r"D:\Projects\TanitAD-artifacts\b1-agent-join-3d-20260917\b1eval_agents_3d.jsonl.xz",
                     None)
    SP.attach_agent_gt(corp, corp.targs)
    wis = [w for w in SP.trainer_windows(corp.ds, 1000) if corp.eligibility(w) is None]
    xs, ys, per_window = [], [], []
    n_lab = 0
    for wi in wis:
        it = corp.ds[wi]
        if not bool(it.get("agent_label", False)):
            continue
        n_lab += 1
        v = it["agent_valid"].bool()
        b = it["agent_box"][v][:, :2].numpy()
        if len(b):
            xs.append(b[:, 0])
            ys.append(b[:, 1])
        inside = ((b[:, 0] >= 0) & (b[:, 0] <= X_FWD) & (np.abs(b[:, 1]) <= Y_HALF)).sum() if len(b) else 0
        per_window.append({"n_gt": int(len(b)), "n_inside": int(inside)})
    x, y = np.concatenate(xs), np.concatenate(ys)
    rng = np.hypot(x, y)
    inside = (x >= 0) & (x <= X_FWD) & (np.abs(y) <= Y_HALF)
    rec = {"_what": "share of the box read's GT inside the head's decode extent",
           "_evidence_class": "MEASURED (ours)", "_tier": "T0 (labels only, no model)",
           "_source_finding": ("training and the read use the SAME target set: the dataset emits "
                               "targets RAW (refc_v3_train.py ~2676) and agent_slots.match_slots "
                               "matches EVERY valid target, dropping only the farthest when "
                               "targets exceed queries (32 vs 100 here, so never). No range mask "
                               "or clip on either side."),
           "labelled_windows": n_lab, "n_gt": int(len(x)),
           "decode_extent": {"x_fwd_m": X_FWD, "y_half_m": Y_HALF},
           "inside_share": round(float(inside.mean()), 4),
           "inside_n": int(inside.sum()),
           "x_m": {"p05": float(np.quantile(x, .05)), "median": float(np.median(x)),
                   "p95": float(np.quantile(x, .95)), "min": float(x.min()), "max": float(x.max())},
           "y_m": {"p05": float(np.quantile(y, .05)), "median": float(np.median(y)),
                   "p95": float(np.quantile(y, .95)), "min": float(y.min()), "max": float(y.max())},
           "range_m": {"median": float(np.median(rng)), "p90": float(np.quantile(rng, .9)),
                       "share_within_60m": round(float((rng <= 60).mean()), 4),
                       "share_behind_ego": round(float((x < 0).mean()), 4)}}
    Path(out_path).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("GT boxes %d over %d labelled windows | INSIDE the decode extent: %.4f (%d)"
          % (rec["n_gt"], n_lab, rec["inside_share"], rec["inside_n"]))
    print("x median %.1f p95 %.1f | y median %.1f p95 %.1f | range median %.1f p90 %.1f | "
          "behind ego %.4f" % (rec["x_m"]["median"], rec["x_m"]["p95"], rec["y_m"]["median"],
                               rec["y_m"]["p95"], rec["range_m"]["median"],
                               rec["range_m"]["p90"], rec["range_m"]["share_behind_ego"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
