"""INFORMATIVE (zero GPU, banked records only): how much of a G-MAP-OVERFIT verdict is the
constant-lr READ-OUT. For each early MAIN arm, per class: the gated step-1,000 reading, and over
the last five readings (steps 600-1,000) the median, min and max; plus the train loss at each of
those steps. Same-seed replicates at step 1,000 (the NEW-2 prereg MAIN vs MAIN_long's first
1,000 steps) give the run-to-run spread with NOTHING changed.
Usage: readout_noise.py <raw/gmo_early dir> [<out json>]"""
import json
import statistics as st
import sys
from pathlib import Path

D = Path(sys.argv[1])
ARMS = [("NEW-2 (prereg)", "g_map_overfit.EARLY_NONBINDING.json"),
        ("NEW-2 replicate (MAIN_long to 1,000)", "g_map_overfit_MAIN_long.INFORMATIVE.json"),
        ("A12 (near lift)", "g_map_overfit_A12.EARLY_NONBINDING.json"),
        ("A15 (+ near refine block)", "g_map_overfit_A15.EARLY_NONBINDING.json"),
        ("A16 (+ mf weights; partial record)", "g_map_overfit_A16.PARTIAL_NONBINDING.json"),
        ("A17.1 (A15 + lr decay 900-1,000)", "g_map_overfit_A171.EARLY_NONBINDING.json")]
CLS = ["nocls", "drivable", "sidewalk", "lane", "crosswalk", "arrow", "edge", "hatched"]
out = {}
for name, fn in ARMS:
    p = D / fn
    if not p.is_file():
        out[name] = {"missing": fn}
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    curve = (r["results"]["healthy"]["curve"] if "results" in r else
             r["arms"]["healthy"]["curve"] if "arms" in r else r["result"]["curve"])
    last5 = [c for c in curve if 600 <= c["step"] <= 1000]
    assert [c["step"] for c in last5] == [600, 700, 800, 900, 1000], (name, [c["step"] for c in last5])
    row = {"train_loss_600_1000": [round(c["train_loss"], 4) for c in last5], "classes": {}}
    for k in CLS:
        xs = [c["iou"][k] for c in last5]
        row["classes"][k] = {"at_1000": round(xs[-1], 4), "median_600_1000": round(st.median(xs), 4),
                             "min": round(min(xs), 4), "max": round(max(xs), 4),
                             "at_1000_is_max": xs[-1] == max(xs)}
    out[name] = row
print(f"{'arm':40s} " + " ".join(f"{k:>17s}" for k in ("lane", "edge", "nocls")))
for name, row in out.items():
    if "missing" in row:
        print(f"{name:40s} (record not banked yet)")
        continue
    cells = []
    for k in ("lane", "edge", "nocls"):
        c = row["classes"][k]
        cells.append(f"{c['at_1000']:.3f} (med {c['median_600_1000']:.3f})")
    print(f"{name:40s} " + " ".join(f"{x:>17s}" for x in cells))
    print(f"{'':40s} loss 600-1000: {row['train_loss_600_1000']}")
if len(sys.argv) > 2:
    Path(sys.argv[2]).write_bytes((json.dumps(out, indent=1) + "\n").encode("utf-8"))
