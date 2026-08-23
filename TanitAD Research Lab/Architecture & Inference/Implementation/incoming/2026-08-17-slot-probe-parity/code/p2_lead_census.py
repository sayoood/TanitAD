"""P2 — per-episode LEAD-BEARING census over the full train-corpus obstacle join.

⭐ WHY: the 2026-08-16 run selected eval clips by PROVIDER ORDER (convenience) and
got only 13 lead-carrying eval episodes. The episode-cluster bootstrap clusters on
EPISODES, so its power is set by that 13, not by the 1,669 windows. This census
ranks all 2,308 joined episodes by the number of frames that actually carry a GT
in-corridor lead, using the EXACT criterion sp2_probe.gt_lead_gap applies
(cx > 0, |cy| <= 1.75 m, cx <= 30 m) so the selection predicate and the scoring
predicate are the same function of the label.

Evidence class: MEASURED (ours; artifact = this JSON + the md5-verified join).
"""
import json, lzma, sys, time
from collections import defaultdict
from pathlib import Path

CORRIDOR_M, LEAD_MAX_M = 1.75, 30.0
GRID_X, GRID_Y = 60.0, 16.0          # SlotDecodeRanges / bev_raster.GRID_DEFAULT

src = Path(sys.argv[1]); out = Path(sys.argv[2])
n_frames = defaultdict(int); n_lead = defaultdict(int)
n_boxes = defaultdict(int); n_ingrid = defaultdict(int)
gap_sum = defaultdict(float)
ingrid_hist = defaultdict(int)        # per-frame in-grid agent counts (for n_slot_queries)
t0 = time.time(); nl = 0
with lzma.open(str(src), "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        cid = r["clip_id"]; ag = r["agents"]
        n_frames[cid] += 1; n_boxes[cid] += len(ag)
        best = None; ing = 0
        for d in ag:
            cx = d["cx"]; cy = d["cy"]
            if 0.0 < cx <= GRID_X and abs(cy) <= GRID_Y:
                ing += 1
            if cx > 0.0 and abs(cy) <= CORRIDOR_M and cx <= LEAD_MAX_M:
                if best is None or cx < best:
                    best = cx
        n_ingrid[cid] += ing
        ingrid_hist[ing] += 1
        if best is not None:
            n_lead[cid] += 1; gap_sum[cid] += best
        nl += 1
        if nl % 50000 == 0:
            print(f"[p2] {nl} records  {time.time()-t0:.0f}s", flush=True)

per = []
for cid in sorted(n_frames):
    nf, nld = n_frames[cid], n_lead[cid]
    per.append({"clip_id": cid, "n_labelled_frames": nf, "n_lead_frames": nld,
                "lead_frac": round(nld / max(nf, 1), 4),
                "mean_lead_gap_m": round(gap_sum[cid] / nld, 3) if nld else None,
                "n_boxes": n_boxes[cid],
                "mean_ingrid_per_frame": round(n_ingrid[cid] / max(nf, 1), 3)})
per.sort(key=lambda d: (-d["n_lead_frames"], d["clip_id"]))

# in-grid per-frame distribution -> the n_slot_queries fit (prereg §2)
tot = sum(ingrid_hist.values())
cum = 0; pct = {}
for k in sorted(ingrid_hist):
    cum += ingrid_hist[k]
    for q in (50, 90, 95, 99, 99.5):
        if q not in pct and cum / tot >= q / 100.0:
            pct[q] = k
rec = {"_evidence_class": "MEASURED (ours; full train-corpus join)",
       "join": str(src), "n_records": nl, "n_clips": len(n_frames),
       "criterion": {"lead": "cx>0 and |cy|<=1.75 and cx<=30.0 "
                             "(sp2_probe.gt_lead_gap, identical function)",
                     "in_grid": "0<cx<=60 and |cy|<=16 (SlotDecodeRanges)"},
       "totals": {"n_lead_frames": sum(n_lead.values()),
                  "n_clips_with_any_lead": sum(1 for c in n_frames if n_lead[c] > 0),
                  "n_boxes": sum(n_boxes.values())},
       "ingrid_per_frame_percentiles": pct,
       "ingrid_max": max(ingrid_hist),
       "ingrid_mean": round(sum(k*v for k, v in ingrid_hist.items())/tot, 3),
       "wall_s": round(time.time()-t0, 1),
       "per_clip": per}
out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in rec.items() if k != "per_clip"}, indent=1), flush=True)
print("[p2] TOP-15:", json.dumps(per[:15], indent=1), flush=True)
