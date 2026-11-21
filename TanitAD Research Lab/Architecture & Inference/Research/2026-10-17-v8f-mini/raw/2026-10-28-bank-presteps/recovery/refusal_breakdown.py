"""Where do the V1 refusals concentrate? Re-runs the V1 labeller over ALL tick-2 off_lat / off_head records of bank_s0 and bins the outcome by own speed,
heading error and lateral error. Beside-only (no card statistic). Writes ../raw/refusal_breakdown_s0.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import common as C  # noqa: E402
import labeller as L  # noqa: E402

tab = C.attach_logged(C.load_table("s0"), C.load_logged())
lg = C.load_logged()
G, V, offs, Ts = lg["G"], lg["V"], lg["offs"], lg["Ts"]
sel = np.flatnonzero(((tab["reason"] == C.OFF_LAT) | (tab["reason"] == C.OFF_HEAD)) & (tab["tick"] == 2))
res = []
for rec in sel:
    c = int(tab["cidx"][rec])
    off, T = int(offs[c]), int(Ts[c])
    r = int(tab["r"][rec])
    out = L.recovery_label(G[off:off + T], V[off:off + T], r, (float(tab["ox"][rec]), float(tab["oy"][rec]), float(tab["oyaw"][rec])), float(tab["v_own"][rec]))
    res.append(out["status"])
res = np.array(res)
v = tab["v_own"][sel]
dpsi = tab["dpsi"][sel]
elat = tab["e_lat"][sel]
vgt = tab["v_gt_r"][sel]


def table(name, edges, x):
    rows = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() == 0:
            continue
        rows[f"{lo}-{hi}"] = {"n": int(m.sum()), **{s: round(float((res[m] == s).mean()), 4) for s in ("emitted", "refused_infeasible", "no_logged_future", "degenerate", "path_exhausted")}}
    return rows


out = {"n_tick2_cut_records": int(len(sel)), "overall": {s: round(float((res == s).mean()), 4) for s in sorted(set(res.tolist()))},
       "by_own_speed_mps": table("v", [0, 1, 3, 6, 10, 100], v), "by_heading_error_deg": table("dpsi", [0, 30, 60, 90, 181], dpsi),
       "by_lateral_error_m": table("elat", [0, 3, 5, 8, 15, 1e9], elat),
       "stationary_human_share_among_degenerate": float((vgt[res == "degenerate"] < 0.5).mean()) if (res == "degenerate").any() else None,
       "note": "outcome shares within each bin; no_logged_future is the end-edge condition hidden by the matcher's cut order"}
Path(HERE.parent / "raw" / "refusal_breakdown_s0.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
