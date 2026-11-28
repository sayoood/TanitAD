"""How far PAST the mapped drivable edge does an off-road corner actually sit?

The renders let a human judge a sample; this measures every one of them. For each corner the
rule flags on an explicitly off-road cell (sidewalk / kerb / hatching), it reports the distance
to the nearest cell the map calls drivable — so "the car left the road" and "the corner touched
the far side of the boundary cell" stop being the same sentence.

⛔ A distance of ~0.5 m is ONE CELL: at 0.5 m resolution the map cannot place a kerb more finely
than that, and a reading inside one cell is a boundary-resolution case, not a measurement of
off-road driving.

    python dac_kerb_depth.py --raw <raw dir>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dac_anatomy import CELL_M, DEFAULTS, THR, Y_HALF_M, _import_harness

ONE_CELL_M = CELL_M


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k, default=v)
    ap.add_argument("--raw", required=True)
    a = ap.parse_args()
    raw = Path(a.raw)
    S1, P, SMG = _import_harness()
    ch = list(SMG.CHANNELS)
    i_dr = ch.index("drivable")

    vs = [json.loads(l) for l in (raw / "dac_violating_samples.jsonl").read_text(encoding="utf-8").splitlines()]
    off = defaultdict(list)
    for s in vs:
        if s["explicit_off"] >= THR:
            off[(s["sha12"], s["t0"])].append(s)

    corp = S1.Corpus(a.config, a.cache, a.labels, a.agents, a.maps, lru=2)
    rows = []
    for wi in S1.trainer_windows(corp.ds, 1000):
        if corp.eligibility(wi) is not None:
            continue
        e_i, t = corp.ds.index[wi]
        cid = str(corp.clip_ids[e_i])
        key = (S1.sha12(cid), int(t + corp.W - 1))
        if key not in off:
            continue
        it = corp.light_item(wi)
        g = corp.shim.map_store.get(cid)
        mf = g.read(np.asarray([it["t0"] + corp.raw_off]))
        cart, seen = np.asarray(mf.cart[0], np.float32), np.asarray(mf.seen[0], bool)
        drv = (cart[i_dr] >= THR) & seen                      # the mapped drivable corridor
        gi, gj = np.nonzero(drv)
        gx, gy = (gi + 0.5) * CELL_M, (gj + 0.5) * CELL_M - Y_HALF_M
        for s in off[key]:
            d = np.hypot(gx - s["x_m"], gy - s["y_m"])
            j = int(np.argmin(d))
            col = drv[int(s["x_m"] / CELL_M)] if 0 <= s["x_m"] < 60 else np.zeros(drv.shape[1], bool)
            rows.append({"sha12": key[0], "t0": key[1], "tick": s["tick"], "corner": s["corner"],
                         "x_m": s["x_m"], "y_m": s["y_m"],
                         "distance_to_mapped_drivable_m": round(float(d[j]), 3),
                         "corridor_cells_at_this_x": int(col.sum()),
                         "corridor_width_m": round(float(col.sum()) * CELL_M, 2)})
    per_w = defaultdict(list)
    for r in rows:
        per_w[(r["sha12"], r["t0"])].append(r["distance_to_mapped_drivable_m"])
    dists = [r["distance_to_mapped_drivable_m"] for r in rows]
    within = sum(1 for k, v in per_w.items() if max(v) <= ONE_CELL_M)
    rep = {
        "_what": "depth past the mapped drivable edge, for every explicitly off-road corner",
        "_evidence_class": "MEASURED (ours; CPU, read-only)",
        "windows": len(per_w), "off_road_corner_samples": len(rows),
        "distance_to_mapped_drivable_m": {
            "median": round(float(np.median(dists)), 3),
            "p90": round(float(np.percentile(dists, 90)), 3),
            "max": round(float(np.max(dists)), 3),
            "<= one cell (0.5 m)": int(sum(1 for d in dists if d <= ONE_CELL_M)),
            "<= two cells (1.0 m)": int(sum(1 for d in dists if d <= 2 * ONE_CELL_M)),
            "> 1.0 m": int(sum(1 for d in dists if d > 2 * ONE_CELL_M)),
        },
        "windows_whose_WORST_off_road_corner_is_within_one_cell": within,
        "corridor_width_m_at_those_samples": {
            "median": round(float(np.median([r["corridor_width_m"] for r in rows])), 2),
            "min": round(float(np.min([r["corridor_width_m"] for r in rows])), 2),
        },
        "ego_half_width_m": round(P.PROXY.ego_width / 2, 4),
    }
    (raw / "kerb_depth.json").write_text(json.dumps(rep, indent=1) + "\n",
                                         encoding="utf-8", newline="\n")
    with (raw / "kerb_depth_samples.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
