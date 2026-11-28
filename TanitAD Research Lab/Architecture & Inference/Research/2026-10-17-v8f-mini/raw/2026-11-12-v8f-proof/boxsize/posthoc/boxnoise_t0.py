"""boxnoise_t0.py -- post-hoc read beside card ADD124 (label tracks only, names nothing; run after the names were opened). CPU only.
How fine does the size of the lead's box have to be read for the map G to beat the gap head? Noise is put on the LABEL sizes of the 3,841 evaluation frames and the gap is read through G.
(a) relative noise on width and height (per cent of size); (b) noise in pixels on each box edge for a camera with 16 or 64 pixels per degree.
Usage: python boxnoise_t0.py <read_boxsize.py> <eval join> <box dir> <train list> <out.json>"""
import argparse
import importlib.util
import json
import sys

import numpy as np

reader, join, box, trl, out = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("RB", reader)
RB = importlib.util.module_from_spec(spec)
sys.modules["RB"] = RB
spec.loader.exec_module(RB)
a = argparse.Namespace(dump=[], out_root="/workspace/v8proof/boxsize/out", box_dir=box)
J = RB.load_npz(join)
L, preds, lab = RB.load_arrays(a, J)
t3 = RB.t3_rows(L)
labg = L["D_0"]["lab_gap"]
G, info = RB.build_map(box, RB.ids_of(trl))
m = t3 & lab["has"]
w, h, y = lab["w"][m], lab["h"][m], labg[m]
BINS = ((0, 10), (10, 20), (20, 40), (40, 60))


def read(wn, hn):
    g = G.predict(np.maximum(wn, 0.05), np.maximum(hn, 0.05))
    e = g - y
    r = {"E": float(np.abs(e).mean())}
    for lo, hi in BINS:
        b = RB.gap_bin_mask(y, lo, hi)
        r["%d-%d" % (lo, hi)] = {"E": float(np.abs(e[b]).mean()), "signed": float(e[b].mean())}
    return r


def mean_of(fn, draws=3):
    rs = [fn(np.random.RandomState(1240 + d)) for d in range(draws)]
    o = {"E": float(np.mean([r["E"] for r in rs]))}
    for lo, hi in BINS:
        k = "%d-%d" % (lo, hi)
        o[k] = {"E": float(np.mean([r[k]["E"] for r in rs])), "signed": float(np.mean([r[k]["signed"] for r in rs]))}
    return o


res = {"note": "post-hoc, label tracks only, names nothing", "n": int(m.sum()), "exact": read(w, h), "relative": {}, "pixels": {}}
print("n", res["n"], "exact sizes E %.3f" % res["exact"]["E"])
for pc in (2, 5, 8, 10, 15, 20):
    s = pc / 100.0
    r = mean_of(lambda rs: read(w * np.exp(rs.normal(0, s, len(w))), h * np.exp(rs.normal(0, s, len(h)))))
    res["relative"][str(pc)] = r
    print("size noise %2d %%: E %.3f | " % (pc, r["E"]) + " ".join("%s E %.2f s %+.2f" % (k, r[k]["E"], r[k]["signed"]) for k in ("0-10", "10-20", "20-40", "40-60")))
for ppd in (16, 64):
    for px in (0.5, 1.0, 2.0, 4.0):
        sd = px * np.sqrt(2.0) / ppd
        r = mean_of(lambda rs: read(w + rs.normal(0, sd, len(w)), h + rs.normal(0, sd, len(h))))
        res["pixels"]["%d_%.1f" % (ppd, px)] = r
        print("%2d px/deg, %.1f px per edge: E %.3f | " % (ppd, px, r["E"]) + " ".join("%s E %.2f s %+.2f" % (k, r[k]["E"], r[k]["signed"]) for k in ("0-10", "10-20", "20-40", "40-60")))
json.dump(res, open(out, "w", encoding="utf-8"), indent=1)
